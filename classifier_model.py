"""
This module handles loading a pre-trained CNN model from PyTorch Hub
and adapting it for image classification tasks.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import mlflow
import mlflow.pytorch
from cifar10_data_loader import Cifar10DataLoader

class ImageClassifier:
    """
    A class for loading a pre-trained CNN model and adapting it
    for image classification tasks.
    """

    def __init__(self, model_name="resnet18", num_classes=10):
        """
        Initializes the ImageClassifier.

        Args:
            model_name (str): The name of the pre-trained model to load.
            num_classes (int): The number of classes for the classification task.
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = self._load_pretrained_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Initialize device here
        self.model.to(self.device)  # Move model to GPU

    def _load_pretrained_model(self):
        """
        Loads a pre-trained CNN model from PyTorch Hub and
        adapts it for the specified classification task.

        Returns:
            torch.nn.Module: The adapted CNN model.
        """
        # Load pre-trained model
        if self.model_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            model = models.resnet18(weights=weights)
        else:
            raise ValueError(f"Model name '{self.model_name}' is not supported.")

        # Freeze all the layers
        # for param in model.parameters():
        #     param.requires_grad = False

        # Get the number of input features for the last fully connected layer
        num_features = model.fc.in_features

        # Replace the last fully connected layer with a new one
        # with the number of output features equal to the number of classes
        model.fc = nn.Linear(num_features, self.num_classes)

        return model

    def save_model(self, model_path="image_classifier_model.pth"):
        """
        Saves the model to the specified path.

        Args:
            model_path (str): The path where the model should be saved.
        """
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path="image_classifier_model.pth"):
        """
        Loads the model from the specified path.

        Args:
            model_path (str): The path from where the model should be loaded.
        """
        self.model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")

    def log_model_info(self):
        """
        Logs model information to MLflow.
        """
        mlflow.log_param("model_name", self.model_name)
        mlflow.log_param("num_classes", self.num_classes)
        # You can log other relevant information here

    def train(self, train_loader, test_loader, epochs=8, learning_rate=0.001):
        """
        Trains the model on the CIFAR-10 dataset.
        Args:
            train_loader (torch.utils.data.DataLoader): The training data loader.
            test_loader (torch.utils.data.DataLoader): The test data loader.
            epochs (int): The number of epochs to train for.
            learning_rate (float): The learning rate for training.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.NAdam(self.model.parameters(), lr=learning_rate)  # Only train the last layer

        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)

        for epoch in range(epochs):
            self.model.train()  # Set model to training mode
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)  # Move data to GPU
                optimizer.zero_grad()  # Zero the gradients
                outputs = self.model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Calculate the loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update the weights
                running_loss += loss.item()

            print(f"Epoch {epoch + 1} Loss: {running_loss / len(train_loader)}")
            mlflow.log_metric("training_loss", running_loss / len(train_loader), epoch)

            # Evaluate the model on the test set
            self.model.eval()  # Set model to evaluation mode
            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data[0].to(self.device), data[1].to(self.device)
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f"Epoch {epoch + 1} Accuracy: {accuracy}%")
            mlflow.log_metric("test_accuracy", accuracy, epoch)

if __name__ == "__main__":
    # Example usage:
    mlflow.start_run()

    classifier = ImageClassifier()  # Uses default resnet18 and 10 classes

    # Log model info to MLflow
    classifier.log_model_info()

    # Load data
    data_loader = Cifar10DataLoader()
    train_loader = data_loader.get_train_loader()
    test_loader = data_loader.get_test_loader()

    # Train the model
    classifier.train(train_loader, test_loader)

    # Save the model
    classifier.save_model()
    mlflow.pytorch.log_model(
        classifier.model,
        "image_classifier_model",
        input_example=torch.randn(1, 3, 224, 224).to(classifier.device)  # Add input_example
    )

    mlflow.end_run()

    print("Pre-trained ResNet18 model loaded, adapted, and trained for CIFAR-10.")