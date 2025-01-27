from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
import io
from torchvision import transforms
import logging
import math
import sys
import os
from typing import Dict, Any

# Add the model directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.vgg_c import vgg19_trans

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Crowd Management API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
try:
    model = vgg19_trans()
    model.load_state_dict(torch.load("C:/Users/srian/Downloads/model_1.pth", map_location='cpu'))
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
                        density_map = model(patch)[0]  # Note: model returns tuple of (output, features)
                        total_count += density_map.sum().item()
            else:
                density_map = model(image_tensor)[0]  # Note: model returns tuple of (output, features)
                total_count = density_map.sum().item()
            
            return {
                "count": round(total_count),
                "status": "success"
            }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

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