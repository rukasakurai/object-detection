from fastapi import FastAPI, UploadFile, File
from typing import List
from azureml.core import Workspace, Model
import torch
from torchvision.transforms import functional as F
from PIL import Image

# Initialize FastAPI app
app = FastAPI()

# Load the model from Azure ML Model Registry
def load_model():
    ws = Workspace.from_config()  # Assumes a config file in the working directory
    model_path = Model(ws, name="custom_object_detection_model").download(target_dir="./models")
    
    # Load the PyTorch model weights
    num_classes = 2  # Adjust based on your dataset
    model = torch.hub.load("pytorch/vision:v0.10.0", "fasterrcnn_resnet50_fpn", pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

# Load the model once on server startup
model = load_model()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image file and returns object detection predictions.
    """
    # Read the image
    image = Image.open(file.file).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(image_tensor)

    # Process predictions
    predictions = []
    for box, label, score in zip(
        outputs[0]["boxes"], outputs[0]["labels"], outputs[0]["scores"]
    ):
        if score > 0.5:  # Confidence threshold
            predictions.append({
                "box": box.tolist(),
                "label": int(label),
                "score": float(score)
            })

    return {"predictions": predictions}
