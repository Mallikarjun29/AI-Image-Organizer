import torch
import torch.nn as nn
import torchvision
import json
from pathlib import Path

def get_latest_model_path():
    """Get the path to the latest model version"""
    models_dir = Path(__file__).parent.parent / "models"
    model_versions = [int(f.stem.split('_v')[-1]) 
                     for f in models_dir.glob('model_v*.pth')]
    if not model_versions:
        return models_dir / "model.pth"  # fallback to default name
    latest_version = max(model_versions)
    return models_dir / f"model_v{latest_version}.pth"

def evaluate():
    # Load test dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    # Load latest model version
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
    model.fc = nn.Linear(512, 10)
    model_path = get_latest_model_path()
    print(f"Evaluating model: {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Evaluate
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            all_predictions.extend(predicted.tolist())
            all_targets.extend(targets.tolist())

    accuracy = 100 * correct / total

    # Save evaluation metrics
    metrics = {
        'accuracy': accuracy,
        'total_samples': total
    }

    with open('evaluation.json', 'w') as f:
        json.dump(metrics, f)

    print(f'Accuracy on test set: {accuracy:.2f}%')

if __name__ == '__main__':
    evaluate()