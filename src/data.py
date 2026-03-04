import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


def get_binary_mnist_dataloaders(data_dir: str = 'data', batch_size: int = 32, val_split: float = 0.2):
    """
    Loads MNIST, filters for classes 0 and 1, and splits into train, val, and test sets.

    Args:
        data_dir (str): Directory for dataset.
        batch_size (int): Batch size.
        val_split (float): Fraction of training data to use for validation.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load datasets
    train_val_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    # Filter indices for classes 0 and 1
    train_val_indices = [i for i, target in enumerate(train_val_dataset.targets) if target in [0, 1]]
    test_indices = [i for i, target in enumerate(test_dataset.targets) if target in [0, 1]]

    # Split train_val into train and validation
    num_train_val = len(train_val_indices)
    split = int(np.floor(val_split * num_train_val))
    
    # Shuffle indices manually for the split
    np.random.seed(42)
    np.random.shuffle(train_val_indices)
    
    val_indices = train_val_indices[:split]
    train_indices = train_val_indices[split:]

    # Create Subsets
    train_subset = Subset(train_val_dataset, train_indices)
    val_subset = Subset(train_val_dataset, val_indices)
    test_subset = Subset(test_dataset, test_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
