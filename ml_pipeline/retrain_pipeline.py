import os
import sys
from pathlib import Path
import torch
import torchvision
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timedelta, timezone
import dvc.api
from PIL import Image
import yaml
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MIN_SAMPLES_FOR_RETRAINING = 100
RETRAINING_WINDOW_DAYS = 7
MONGO_URI = "mongodb://localhost:27017/"
MODEL_DIR = "models"
METRICS_FILE = "metrics.json"

class RetrainingPipeline:
    def __init__(self):
        self.client = MongoClient(MONGO_URI)
        self.db = self.client["image_organizer"]
        self.collection = self.db["image_metadata"]
        self.last_training_file = Path("last_training_time.txt")
        
    def get_last_training_time(self):
        """Get the timestamp of last training"""
        if self.last_training_file.exists():
            with open(self.last_training_file, 'r') as f:
                timestamp_str = f.read().strip()
                return datetime.fromisoformat(timestamp_str)
        return None
        
    def save_last_training_time(self):
        """Save the current time as last training time"""
        with open(self.last_training_file, 'w') as f:
            f.write(datetime.now(timezone.utc).isoformat())
    
    def get_drift_samples(self):
        """Get samples from MongoDB since last training time"""
        last_training = self.get_last_training_time()
        if not last_training:
            # If no last training time, use samples from the last hour
            last_training = datetime.now(timezone.utc) - timedelta(hours=1)
        
        samples = list(self.collection.find({
            "timestamp": {"$gte": last_training},
            "drift_detected": True
        }))
        
        return samples
    
    def load_images(self, sample_paths):
        """Load images from paths"""
        images = []
        valid_paths = []
        
        for path in sample_paths:
            try:
                img = Image.open(path)
                images.append(img)
                valid_paths.append(path)
            except Exception as e:
                logger.error(f"Error loading image {path}: {e}")
                
        return images, valid_paths
    
    def supplement_with_cifar(self, n_samples_needed):
        """Get additional samples from CIFAR-10 if needed"""
        cifar_train = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True
        )
        
        indices = np.random.choice(
            len(cifar_train), 
            n_samples_needed, 
            replace=False
        )
        
        return [cifar_train[i][0] for i in indices]
    
    def prepare_training_data(self):
        """Prepare training data from drift samples and CIFAR"""
        samples = self.get_drift_samples()
        sample_paths = [s['path'] for s in samples]
        
        images, valid_paths = self.load_images(sample_paths)
        logger.info(f"Loaded {len(images)} drift detected samples")
        
        if len(images) < MIN_SAMPLES_FOR_RETRAINING:
            needed = MIN_SAMPLES_FOR_RETRAINING - len(images)
            logger.info(f"Getting {needed} additional samples from CIFAR-10")
            cifar_images = self.supplement_with_cifar(needed)
            images.extend(cifar_images)
        
        return images
    
    def mark_samples_as_trained(self, sample_paths):
        """Mark samples as used in training"""
        self.collection.update_many(
            {"path": {"$in": sample_paths}},
            {"$set": {"used_in_training": True}}
        )
    
    def retrain_model(self):
        """Retrain the model using DVC"""
        try:
            # Prepare data
            images = self.prepare_training_data()
            if not images:
                logger.error("No training data available")
                return False
            
            # Save training data
            data_dir = Path("data/retraining")
            data_dir.mkdir(parents=True, exist_ok=True)
            
            sample_paths = []
            for idx, img in enumerate(images):
                path = data_dir / f"sample_{idx}.png"
                img.save(path)
                sample_paths.append(str(path))
            
            # Update DVC pipeline
            os.system("dvc repro")
            
            # Push changes
            os.system("dvc push")
            
            # Mark samples as used in training
            self.mark_samples_as_trained(sample_paths)
            
            # Save current time as last training time
            self.save_last_training_time()
            
            # Log metrics
            with open(METRICS_FILE, 'r') as f:
                metrics = yaml.safe_load(f)
            
            logger.info(f"Retraining completed. Metrics: {metrics}")
            return True
            
        except Exception as e:
            logger.error(f"Error in retraining: {e}")
            return False

if __name__ == "__main__":
    pipeline = RetrainingPipeline()
    pipeline.retrain_model()