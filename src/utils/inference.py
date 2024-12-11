"""
Provides a function to perform inference using a trained model on a single image.
"""

import torch
from torchvision import transforms
from PIL import Image

from src.models import get_model
from src.utils import load_checkpoint

def inference_single_image(image_path, checkpoint_path, device='cpu'):
    """
    Perform inference using a trained model on a single image.

    Args:
        image_path (str): Path to the image file.
        checkpoint_path (str): Path to the model checkpoint file.
        device (str): Device to use for inference (default: 'cpu').

    Returns:
        torch.Tensor: Predicted class probabilities.
    """
    # Load the image
    image = Image.open(image_path)

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)

    # Load the model
    model = get_model()
    load_checkpoint(model, checkpoint_path, device=device)

    # Perform inference
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        probabilities = torch.softmax(output, dim=1)

    return probabilities