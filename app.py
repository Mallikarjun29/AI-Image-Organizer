"""
This module creates a FastAPI web application for uploading an image
and displaying its classification result.
"""

import os
import shutil
from typing import Annotated

from fastapi import FastAPI, File, UploadFile, HTTPException, Request 
from fastapi.responses import HTMLResponse, FileResponse
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

# Configure the folder for organized images
ORGANIZED_FOLDER = "organized_images"
Path(ORGANIZED_FOLDER).mkdir(parents=True, exist_ok=True)

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
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
async def upload_files(request: Request, files: list[UploadFile] = File(...)):
    """
    Handles multiple file uploads, classifies them, organizes them into folders,
    and creates a zip file for download.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    results = []  # To store the results for each file

    for file in files:
        if not file.filename:
            continue  # Skip files with no name
        if file and allowed_file(file.filename):
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            try:
                with open(file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
            except Exception:
                raise HTTPException(status_code=500, detail=f"Error saving file: {file.filename}")

            # Load the model (you might want to load this only once at app startup)
            from classifier_model import ImageClassifier

            classifier = ImageClassifier()
            classifier.load_model()  # Load the saved model

            # Predict the image class
            predicted_class = predict_image(file_path, classifier.model)

            # Create a folder for the predicted class if it doesn't exist
            class_folder = os.path.join(ORGANIZED_FOLDER, predicted_class)
            Path(class_folder).mkdir(parents=True, exist_ok=True)

            # Move the file to the class folder
            organized_file_path = os.path.join(class_folder, file.filename)
            os.rename(file_path, organized_file_path)

            # Append the result for this file
            results.append({"filename": file.filename, "predicted_class": predicted_class})
        else:
            raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}")

    # Create a zip file of the organized_images folder
    zip_file_path = "organized_images.zip"
    shutil.make_archive("organized_images", "zip", ORGANIZED_FOLDER)

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "results": results,  # Pass the list of results to the template
            "zip_file_path": zip_file_path,  # Pass the zip file path to the template
        }
    )

@app.get("/download-zip/")
async def download_zip():
    """
    Endpoint to download the zip file of organized images.
    """
    zip_file_path = "organized_images.zip"
    if not os.path.exists(zip_file_path):
        raise HTTPException(status_code=404, detail="Zip file not found")
    return FileResponse(zip_file_path, media_type="application/zip", filename="organized_images.zip")