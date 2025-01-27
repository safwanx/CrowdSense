from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import io
import numpy as np
from torchvision import transforms
import logging
from typing import Dict, Any, Optional, List
import math
import copy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Crowd Management API",
             description="API for crowd counting using MAN model",
             version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model components from original MAN implementation
class LearnableGlobalLocalMultiheadAttention(nn.Module):
    NUM_WEIGHTS = 9
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = nn.Parameter(torch.Tensor(self.NUM_WEIGHTS * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.Tensor(self.NUM_WEIGHTS * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, shape, value):
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        
        scaling = float(self.head_dim) ** -0.5
        
        q = F.linear(query, self.in_proj_weight[:self.embed_dim], self.in_proj_bias[:self.embed_dim])
        k = F.linear(key, self.in_proj_weight[self.embed_dim:2*self.embed_dim], 
                    self.in_proj_bias[self.embed_dim:2*self.embed_dim])
        v = F.linear(value, self.in_proj_weight[2*self.embed_dim:3*self.embed_dim], 
                    self.in_proj_bias[2*self.embed_dim:3*self.embed_dim])

        q = q * scaling
        
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_output_weights

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = LearnableGlobalLocalMultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu
        self.normalize_before = normalize_before

    def forward(self, src, shape, src_mask=None, src_key_padding_mask=None, pos=None):
        src2, weights = self.self_attn(src, src, shape, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src, weights

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, shape, mask=None, src_key_padding_mask=None, pos=None):
        output = src
        weights_list = []

        for layer in self.layers:
            output, weights = layer(output, shape, src_mask=mask,
                                  src_key_padding_mask=src_key_padding_mask, pos=pos)
            weights_list.append(weights)

        if self.norm is not None:
            output = self.norm(output)

        return output, weights_list

class VGG_Trans(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features

        d_model = 512
        nhead = 2
        num_layers = 4
        dim_feedforward = 2048
        dropout = 0.1
        
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)

        self.reg_layer_0 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        rh = int(h) // 16
        rw = int(w) // 16
        
        x = self.features(x)
        
        bs, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        x, _ = self.encoder(x, (h, w))
        x = x.permute(1, 2, 0).view(bs, c, h, w)
        
        x = F.interpolate(x, size=(rh, rw), mode='bilinear', align_corners=False)
        x = self.reg_layer_0(x)
        
        return torch.relu(x)

def make_layers():
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 
           512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# Initialize model and load weights
try:
    model = VGG_Trans(make_layers())
    model.load_state_dict(torch.load("D:/Documents/Programming/SP/CrowdSense/Crowd site/CrowdSense/backend/model/model_1.pth", map_location='cpu'))
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

# Image preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def process_image(image_bytes: bytes) -> torch.Tensor:
    """Process uploaded image for model input"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Calculate new dimensions while maintaining aspect ratio
        min_size = 512
        max_size = 2048
        w, h = image.size
        
        scale = min_size / min(h, w)
        if h * scale > max_size or w * scale > max_size:
            scale = max_size / max(h, w)
        
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        image = image.resize((new_w, new_h), Image.BICUBIC)
        img_tensor = transform(image).unsqueeze(0)
        
        return img_tensor
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image format")

def get_prediction(image_tensor: torch.Tensor) -> Dict[str, Any]:
    """Get crowd count prediction"""
    try:
        with torch.no_grad():
            b, c, h, w = image_tensor.shape
            if h >= 3584 or w >= 3584:
                h_stride = math.ceil(h / 3584)
                w_stride = math.ceil(w / 3584)
                h_step = h // h_stride
                w_step = w // w_stride
                
                total_count = 0
                for i in range(h_stride):
                    for j in range(w_stride):
                        h_start = i * h_step
                        h_end = h_start + h_step if i != h_stride-1 else h
                        w_start = j * w_step
                        w_end = w_start + w_step if j != w_stride-1 else w
                        
                        patch = image_tensor[:, :, h_start:h_end, w_start:w_end]
                        density_map = model(patch)
                        total_count += density_map.sum().item()
            else:
                density_map = model(image_tensor)
                total_count = density_map.sum().item()
            
            return {
                "count": round(total_count),
                "status": "success"
            }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

# API Endpoints
@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": "Welcome to the Crowd Management API",
        "status": "active"
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Predict crowd count from image"""
    if not file.content_type.startswith("D:/Documents/Programming/SP/CrowdSense/Crowd site/CrowdSense/image"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_bytes = await file.read()
        image_tensor = process_image(image_bytes)
        result = get_prediction(image_tensor)
        return result
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}