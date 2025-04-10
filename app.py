"""
This module creates a FastAPI web application for uploading an image
and displaying its classification result.
"""

import os
import shutil
from typing import Annotated
import uuid
from datetime import datetime
from pymongo import MongoClient
import re
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException, Request 
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms

from prometheus_client import Counter, make_asgi_app # Gauge

from ml_pipeline.classifier_model import ImageClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure upload folder
UPLOAD_FOLDER = Path("uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

# Configure the folder for organized images
ORGANIZED_FOLDER = Path("organized_images")
ORGANIZED_FOLDER.mkdir(parents=True, exist_ok=True)

# Mount the uploads folder as a static files directory
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")

# Configure Jinja2 templates
templates = Jinja2Templates(directory="templates")

# --- MongoDB Configuration ---
MONGO_URI = os.getenv('MONGO_URI', "mongodb://localhost:27017/")
DB_NAME = os.getenv('MONGO_DB', "image_organizer")
COLLECTION_NAME = os.getenv('MONGO_COLLECTION', "image_metadata")

print(f"Connecting to MongoDB at {MONGO_URI}...")
try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    # Test connection
    client.admin.command('ping')
    print("MongoDB connection successful")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    raise

# --- Model Loading ---
try:
    print("Initializing classifier...")
    classifier = ImageClassifier()
    
    # Find the latest model version
    models_dir = Path("models")
    if not models_dir.exists():
        raise FileNotFoundError("Models directory not found")
    
    model_files = list(models_dir.glob("model_v*.pth"))
    if not model_files:
        # Try loading default model
        default_model = models_dir / "model.pth"
        if not default_model.exists():
            raise FileNotFoundError("No model files found")
        model_path = str(default_model)
    else:
        # Get latest version
        versions = [int(f.stem.split('_v')[1]) for f in model_files]
        latest_version = max(versions)
        model_path = str(models_dir / f"model_v{latest_version}.pth")
    
    print(f"Loading model from: {model_path}")
    classifier.load_model(model_path)
    print("Model loaded successfully")

except Exception as e:
    print(f"Error loading model: {str(e)}")
    import traceback
    traceback.print_exc()
    classifier = None

# Load the classes
classes = (
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

# --- Helper Functions ---
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
    try:
        # Define transformations for the image
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Move image tensor to the same device as model
        device = next(model.parameters()).device
        image_tensor = image_tensor.to(device)

        # Make the prediction
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            predicted_class = classes[predicted_idx[0]]

        return predicted_class
    except Exception as e:
        print(f"Error predicting image {image_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Error predicting image: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Renders the index page.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_files(request: Request, files: list[UploadFile] = File(...)):
    """
    Handles multiple file uploads, classifies them, and stores results.
    """
    if classifier is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Instead of cleaning the directory, we'll just clean its contents
    org_folder = Path(ORGANIZED_FOLDER)
    try:
        # Clean contents while preserving the directory structure
        if org_folder.exists():
            # Remove files first
            for f in org_folder.glob("**/*"):
                if f.is_file():
                    try:
                        f.unlink(missing_ok=True)
                    except Exception as e:
                        logger.warning(f"Could not remove file {f}: {e}")
    except Exception as e:
        logger.error(f"Error while cleaning directory contents: {e}")
        # Continue even if cleaning fails
    
    results = []
    pattern = r"img_\d+_origlabel_(\d+)_idx_\d+\.png"

    for file in files:
        if not file.filename:
            continue  # Skip files with no name
        
        # Generate a random 16-character alphanumeric name for the image
        random_name = str(uuid.uuid4().hex[:16]) + ".png"

        if file and allowed_file(file.filename):
            file_path = os.path.join(UPLOAD_FOLDER, random_name)
            try:
                with open(file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error saving file: {file.filename} - {str(e)}")

            # Extract actual class from filename
            actual_class = None
            match = re.match(pattern, file.filename)
            if match:
                class_idx = int(match.group(1))
                if 0 <= class_idx < len(classes):
                    actual_class = classes[class_idx]

            try:
                # Predict the image class
                predicted_class = predict_image(file_path, classifier.model)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error predicting class: {str(e)}")

            # Create a folder for the predicted class if it doesn't exist
            class_folder = os.path.join(ORGANIZED_FOLDER, predicted_class)
            Path(class_folder).mkdir(parents=True, exist_ok=True)

            # Save the renamed file in the class folder
            organized_file_path = os.path.join(class_folder, file.filename)
            try:
                shutil.copy2(file_path, organized_file_path)
            except Exception as e:
                print(f"Warning: Could not copy to organized folder: {e}")

            # Save metadata in MongoDB
            metadata = {
                "original_name": file.filename,
                "random_name": random_name,
                "actual_class": actual_class,
                "predicted_class": predicted_class,
                "path": file_path,
                "timestamp": datetime.utcnow()
            }
            try:
                collection.insert_one(metadata)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error saving to database: {str(e)}")

            # Append the result for this file
            results.append({
                "filename": file.filename, 
                "predicted_class": predicted_class,
                "actual_class": actual_class,
                "random_name": random_name
            })
        else:
            raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}")

    # Create a zip file of the organized_images folder
    try:
        zip_file_path = "organized_images.zip"
        shutil.make_archive("organized_images", "zip", ORGANIZED_FOLDER)
    except Exception as e:
        print(f"Warning: Could not create zip file: {e}")
        zip_file_path = None

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
