from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
import numpy as np
import cv2
from PIL import Image
import io
from torchvision import transforms
import logging
import math
import sys
import os
from typing import Dict, Any
import base64

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

# Create static folder if it doesn't exist
os.makedirs('static', exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize model
try:
    model = vgg19_trans()
    model.load_state_dict(torch.load("/model_1.pth", map_location='cpu'))
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

def process_image(image_bytes: bytes) -> tuple[torch.Tensor, np.ndarray]:
    """Process uploaded image for model input"""
    try:
        # Read image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        original_size = image.size
        
        # Calculate new dimensions while maintaining aspect ratio
        min_size = 512
        max_size = 2048
        w, h = image.size
        
        scale = min_size / min(h, w)
        if h * scale > max_size or w * scale > max_size:
            scale = max_size / max(h, w)
        
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Resize image
        image_resized = image.resize((new_w, new_h), Image.BICUBIC)
        
        # Convert to tensor
        img_tensor = transform(image_resized).unsqueeze(0)
        
        # Convert to numpy array for overlay
        original_np = np.array(image)
        
        return img_tensor, original_np
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image format")

def generate_heatmap(density_map: np.ndarray, original_image: np.ndarray) -> np.ndarray:
    """Generate heatmap overlay"""
    try:
        # Resize density map to match original image size
        density_map = cv2.resize(density_map, (original_image.shape[1], original_image.shape[0]))
        
        # Normalize density map
        density_map = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-8)
        
        # Create heatmap
        heatmap = cv2.applyColorMap((density_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Create overlay
        overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
        
        return overlay
    except Exception as e:
        logger.error(f"Error generating heatmap: {str(e)}")
        raise

def get_prediction(image_tensor: torch.Tensor, original_image: np.ndarray) -> Dict[str, Any]:
    """Get crowd count prediction and heatmap"""
    try:
        with torch.no_grad():
            # Get density map prediction
            density_map = model(image_tensor)[0]
            
            # Calculate count
            count = density_map.sum().item()
            
            # Convert density map to numpy
            density_map_np = density_map.squeeze().cpu().numpy()
            
            # Generate heatmap overlay
            overlay = generate_heatmap(density_map_np, original_image)
            
            # Save overlay image
            filename = f"heatmap_{int(count)}.jpg"
            cv2.imwrite(os.path.join("static", filename), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            
            return {
                "count": round(count),
                "heatmap_url": f"/static/{filename}",
                "status": "success"
            }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.get("/")
def read_root():
    """Root endpoint"""
    return {"message": "Welcome to the Crowd Management API"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Predict crowd count from image"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        image_bytes = await file.read()
        
        # Process image
        image_tensor, original_image = process_image(image_bytes)
        
        # Get prediction and heatmap
        result = get_prediction(image_tensor, original_image)
        
        return result
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))