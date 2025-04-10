"""
This module handles loading a pre-trained CNN model from PyTorch Hub
and adapting it for image classification tasks.
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
import mlflow
import mlflow.pytorch
from pathlib import Path

class ImageClassifier:
    """
    A class for loading a pre-trained CNN model and adapting it
    for image classification tasks.
    """

    def __init__(self, model_name="resnet18", num_classes=10):
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = self._load_pretrained_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model_dir = Path(__file__).parent.parent / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _load_pretrained_model(self):
        """
        Loads a pre-trained model from PyTorch Hub and adapts it for the specified number of classes.
        """
        if self.model_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            model = models.resnet18(weights=weights)
        else:
            raise ValueError(f"Model name '{self.model_name}' is not supported.")

        # Get the number of input features for the last fully connected layer
        num_features = model.fc.in_features
        # Replace the last fully connected layer
        model.fc = nn.Linear(num_features, self.num_classes)
        return model

    def save_model(self, model_path):
        """
        Saves the model to the specified path.
        Args:
            model_path (str): The path where the model should be saved.
        """
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path):
        """
        Loads the model from the specified path.
        Args:
            model_path (str): The path from where the model should be loaded.
        """
        try:
            state_dict = torch.load(model_path)
            self.model.load_state_dict(state_dict)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            raise