"""
This module creates a FastAPI web application for uploading an image
and displaying its classification result.
"""

import os
from typing import Annotated

from fastapi import FastAPI, File, UploadFile, HTTPException, Request 
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms

app = FastAPI()

# Configure upload folder
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

# Mount the uploads folder as a static files directory
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")

templates = Jinja2Templates(directory="templates")

# Load the classes
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

def allowed_file(filename: str) -> bool:
    """
    Checks if the uploaded file has an allowed extension.

    Args:
        filename (str): The name of the uploaded file.

    Returns:
        bool: True if the file has an allowed extension, False otherwise.
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path: str, model: torch.nn.Module) -> str:
    """
    Predicts the class of an image using the provided model.

    Args:
        image_path (str): The path to the image file.
        model (torch.nn.Module): The pre-trained CNN model.

    Returns:
        str: The predicted class name.
    """
    # Define transformations for the image
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Make the prediction
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        output = model(image)
        _, predicted_idx = torch.max(output, 1)
        predicted_class = classes[predicted_idx[0]]

    return predicted_class

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Renders the index page.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_file(request: Request, file: UploadFile = File(...)):
    """
    Handles file uploads and displays the classification result.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    if file and allowed_file(file.filename):
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        try:
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
        except Exception:
            raise HTTPException(status_code=500, detail="Error saving file")

        # Load the model (you might want to load this only once at app startup)
        from classifier_model import ImageClassifier

        classifier = ImageClassifier()
        classifier.load_model()  # Load the saved model

        # Predict the image class
        predicted_class = predict_image(file_path, classifier.model)

        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "image_path": file.filename,  # Pass only the file name
                "predicted_class": predicted_class,
            }
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid file type")