# CrowdSense

CrowdSense is a prototype crowd monitoring system centered on a MAN-based crowd counting model with a FastAPI inference API. The repo also includes lightweight front-end mockups for visualization.

## Backend (FastAPI)

**Key Files**
- `backend/main.py` API entrypoint, image preprocessing, model inference, heatmap overlay generation, and static file serving.
- `backend/models/vgg_c.py` VGG19 backbone with transformer encoder + regression head.
- `backend/models/transformer_cosine.py` Transformer encoder layers using custom attention.
- `backend/models/module_consistency.py` Learnable global/local multi-head attention implementation.

**API Endpoints**
- `GET /` Health message.
- `POST /predict/` Upload an image under form field `file`. Returns JSON with `count`, `heatmap_url`, and `status`.

**How Inference Works**
- The image is resized to a reasonable scale, normalized, and sent through the model.
- The output density map is summed to produce a crowd count.
- A heatmap overlay is generated using OpenCV and saved under `static/`.

## Model (MAN)

This project uses the MAN (Multifaceted Attention Network) for crowd counting.
Paper: `https://arxiv.org/abs/2203.02636`

**Architecture In This Repo**
- VGG19 backbone (ImageNet-pretrained) for feature extraction.
- Transformer-style encoder with learnable global/local attention to refine spatial features.
- Regression head that outputs a density map; summing the map yields the crowd count.

**Input Preprocessing**
- Images are resized to a practical scale (min dimension 512, max dimension 2048).
- Inputs are normalized with ImageNet mean and standard deviation.

**Checkpoint**
- `backend/main.py` loads a local `.pth` checkpoint path. Replace it with your own checkpoint compatible with `vgg19_trans()`.


**Expected Dataset Layout**
- These tools assume images live in an `images/` folder and annotations in a sibling `annotations/` folder with matching base names.

## Setup and Run (Local)

**Backend**
1. Install dependencies: `fastapi`, `uvicorn`, `torch`, `torchvision`, `numpy`, `opencv-python`, `pillow`, `python-multipart`.
1. Optional for annotation tools: `scipy`, `matplotlib`.
1. Update the model checkpoint path in `backend/main.py` to a valid `.pth` file.
1. Start the API from the `backend/` directory:
   ```bash
   python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

**Test the API**
```bash
curl -X POST -F "file=@path/to/image.jpg" http://localhost:8000/predict/
```

**Front-End**
- Open `frontend.html` or `HTML & CSS/index.html` in a browser.
- The upload demo expects the API to be running at `http://localhost:8000`.
