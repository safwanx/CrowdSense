from fastapi import FastAPI, File, UploadFile
import torch
from PIL import Image
import io

# Initialize the FastAPI app
app = FastAPI()

# Load your PyTorch model
class CrowdModel(torch.nn.Module):
    def __init__(self):
        super(CrowdModel, self).__init__()
        # Define model architecture here if needed
        # Example: self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        # Define forward pass here
        return x  # Replace with actual forward logic

# Load the saved model
model = CrowdModel()
model.load_state_dict(torch.load("model_1.pth", map_location=torch.device("cpu")))
model.eval()  # Set the model to evaluation mode

# Preprocess uploaded image
def preprocess_image(image) -> torch.Tensor:
    img = Image.open(io.BytesIO(image)).convert("RGB")
    img = img.resize((256, 256))  # Adjust size to match model input
    img_tensor = torch.tensor([list(img.getdata())]).float().reshape(1, 3, 256, 256) / 255.0
    return img_tensor

# Root route (already added in Step 2)
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI Backend"}

# Add the /predict/ endpoint here
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image
    image = await file.read()

    # Preprocess the image
    input_tensor = preprocess_image(image)

    # Perform prediction
    with torch.no_grad():
        output = model(input_tensor)

    # Return predictions as JSON
    return {"predictions": output.tolist()}