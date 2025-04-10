import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import yaml
import shutil
from pathlib import Path
from pymongo import MongoClient
import json
import re
import time

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.labels = []
        self._load_samples()

    def _load_samples(self):
        # Connect to MongoDB
        client = MongoClient("mongodb://localhost:27017/")
        db = client["image_organizer"]
        collection = db["image_metadata"]

        for img_path in self.data_dir.glob('*.png'):
            # Get the random name of the image
            random_name = img_path.name
            # Find corresponding metadata in MongoDB
            metadata = collection.find_one({"random_name": random_name})
            if metadata and "class" in metadata:
                self.samples.append(str(img_path))
                # Convert class name to index
                classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
                try:
                    class_idx = classes.index(metadata["class"])
                    self.labels.append(class_idx)
                except ValueError:
                    print(f"Warning: Unknown class {metadata['class']} for {random_name}")
                    continue

        client.close()
        print(f"Loaded {len(self.samples)} images with labels")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

def prepare_data():
    """Prepare data from uploads directory for training"""
    # Setup paths
    project_root = Path(__file__).parent.parent
    uploads_dir = project_root / "uploads"
    retraining_images_dir = project_root / "data/retraining/images"
    dataset_info_path = project_root / "data/retraining/dataset_info.json"
    retraining_images_dir.mkdir(parents=True, exist_ok=True)

    # CIFAR-10 classes (in order matching the numeric labels)
    classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    # Connect to MongoDB with retry logic
    max_retries = 5
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            # Use the container name 'mongodb' instead of localhost
            mongo_uri = os.getenv('MONGO_URI', 'mongodb://mongodb:27017/')
            client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            # Test connection
            client.admin.command('ping')
            print("Successfully connected to MongoDB")
            db = client["image_organizer"]
            collection = db["image_metadata"]
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Failed to connect to MongoDB (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Failed to connect to MongoDB after all retries")
                raise

    print(f"Copying files from {uploads_dir} to {retraining_images_dir}")
    copied_count = 0
    dataset_info = {
        "samples": [],
        "labels": []
    }

    for img_path in uploads_dir.glob('*.png'):
        try:
            metadata = collection.find_one({"random_name": img_path.name})
            if metadata and metadata.get("actual_class") is not None:
                shutil.copy2(img_path, retraining_images_dir)
                try:
                    class_idx = classes.index(metadata["actual_class"])
                    dataset_info["samples"].append(str(retraining_images_dir / img_path.name))
                    dataset_info["labels"].append(class_idx)
                    copied_count += 1
                except ValueError:
                    print(f"Warning: Unknown class {metadata['actual_class']} for {img_path.name}")
                    continue
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue

    print(f"Copied {copied_count} files successfully.")
    
    if copied_count == 0:
        print("Warning: No files found with valid actual_class!")
        return

    # Save dataset info
    with open(dataset_info_path, 'w') as f:
        json.dump(dataset_info, f)
    
    print(f"Saved dataset info with {len(dataset_info['samples'])} samples")

    client.close()

if __name__ == '__main__':
    prepare_data()