"""
This module handles the loading and preprocessing of the CIFAR-10 dataset
using PyTorch's torchvision library.

It defines a Cifar10DataLoader class to encapsulate the data loading logic
and provides methods for retrieving the training and test data loaders.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class Cifar10DataLoader:
    """
    A class for loading and preprocessing the CIFAR-10 dataset.
    """

    def __init__(self, data_dir="./data", batch_size_train=128, batch_size_test=100, num_workers=2):
        """
        Initializes the Cifar10DataLoader.

        Args:
            data_dir (str): The directory where the CIFAR-10 dataset will be stored.
            batch_size_train (int): The batch size for the training data loader.
            batch_size_test (int): The batch size for the test data loader.
            num_workers (int): The number of workers for data loading.
        """
        self.data_dir = data_dir
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.num_workers = num_workers
        self.transform_train = self._get_train_transform()
        self.transform_test = self._get_test_transform()
        self.classes = (
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

    def _get_train_transform(self):
        """
        Defines the data transformations for the training set.

        Returns:
            torchvision.transforms.Compose: The composed transformations for training data.
        """
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def _get_test_transform(self):
        """
        Defines the data transformations for the test set.

        Returns:
            torchvision.transforms.Compose: The composed transformations for test data.
        """
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def get_train_loader(self):
        """
        Returns the training data loader.

        Returns:
            torch.utils.data.DataLoader: The training data loader.
        """
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=self.transform_train
        )
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size_train,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def get_test_loader(self):
        """
        Returns the test data loader.

        Returns:
            torch.utils.data.DataLoader: The test data loader.
        """
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=True, transform=self.transform_test
        )
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size_test,
            shuffle=False,
            num_workers=self.num_workers,
        )

if __name__ == "__main__":
    # Example usage:
    data_loader = Cifar10DataLoader()
    train_loader = data_loader.get_train_loader()
    test_loader = data_loader.get_test_loader()

    # You can now use train_loader and test_loader in your training/evaluation loop
    # For example, to iterate through the training data:
    # for batch_idx, (data, target) in enumerate(train_loader):
    #     print(f"Batch {batch_idx}: Data shape = {data.shape}, Target shape = {target.shape}")
    print("Data loaders created successfully!")