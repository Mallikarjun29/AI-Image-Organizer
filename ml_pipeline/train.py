import os
import torch
import torch.nn as nn
import json
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import mlflow
from classifier_model import ImageClassifier
import tempfile
import shutil

class RetrainingDataset(Dataset):
    def __init__(self, dataset_info_path, transform=None):
        with open(dataset_info_path, 'r') as f:
            data = json.load(f)
        self.samples = data['samples']
        self.labels = data['labels']
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

def train_model():
    """Train the model using data prepared in the previous stage"""
    print("Starting model training...")
    
    # Setup paths
    data_dir = Path("/app/data/retraining")
    models_dir = Path("/app/models")
    dataset_info_path = data_dir / "dataset_info.json"

    if not dataset_info_path.exists():
        raise FileNotFoundError(f"Dataset info not found at {dataset_info_path}")

    # Load dataset info
    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)

    # Initialize model
    classifier = ImageClassifier()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.model.to(device)

    # Setup data loading
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create dataset and dataloader
    dataset = RetrainingDataset(dataset_info_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Training settings
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.model.parameters(), lr=0.001)
    epochs = 10

    # Create temporary directory for model saving
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_model_path = Path(temp_dir) / "model_v1.pth"
        
        # Training loop
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(dataloader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = classifier.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            print(f'Epoch {epoch + 1}, Loss: {running_loss/len(dataloader)}')

        # Save model to temporary location first
        torch.save(classifier.model.state_dict(), str(temp_model_path))
        print(f"Model saved temporarily to {temp_model_path}")
        
        # Create models directory if it doesn't exist
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Move the file from temp location to final destination
        final_model_path = models_dir / "model_v1.pth"
        shutil.copy2(temp_model_path, final_model_path)
        print(f"Model moved to final location: {final_model_path}")

    print("Training completed successfully")

if __name__ == '__main__':
    train_model()