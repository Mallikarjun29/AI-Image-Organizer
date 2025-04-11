import os
import json
import torch
import torch.nn as nn
import yaml
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import tempfile
import shutil
from classifier_model import ImageClassifier

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
    print("Starting model training...")
    
    # Setup paths
    data_dir = Path("/app/data/retraining")
    dataset_info_path = data_dir / "dataset_info.json"
    models_dir = Path("/app/models")
    final_model_path = models_dir / "model_v1.pth"  # Fixed path for DVC

    if not dataset_info_path.exists():
        raise FileNotFoundError(f"Dataset info not found at {dataset_info_path}")

    # Load hyperparameters
    with open('hyperparameters.yaml', 'r') as f:
        params = yaml.safe_load(f)

    # Initialize model and training
    classifier = ImageClassifier()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.model.to(device)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load dataset
    dataset = RetrainingDataset(dataset_info_path, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=params.get('batch_size', 32), 
        shuffle=True
    )

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        classifier.model.parameters(), 
        lr=params.get('learning_rate', 0.001)
    )
    epochs = params.get('epochs', 10)

    print(f"Starting training with {len(dataset)} samples...")
    metrics = {
        "epochs": epochs,
        "dataset_size": len(dataset),
        "model_version": params.get('model_version', 1)
    }

    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = classifier.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss/len(dataloader)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')
        metrics[f"epoch_{epoch+1}_loss"] = avg_loss

    # Create models directory if it doesn't exist
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save model directly to the final path
    torch.save(classifier.model.state_dict(), str(final_model_path))
    print(f"Model saved as {final_model_path}")

    # Save metrics
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)

if __name__ == '__main__':
    train_model()