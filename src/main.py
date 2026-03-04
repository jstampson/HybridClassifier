import os
from src.data import get_binary_mnist_dataloaders


def main():
    """
    Entry point for testing the binary MNIST data loader.
    """
    print("Setting up data loaders...")
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)

    train_loader, test_loader = get_binary_mnist_dataloaders(
        data_dir='data',
        batch_size=32
    )

    # Fetch a single batch from the training loader
    images, labels = next(iter(train_loader))
    
    print(f"Train DataLoader created. Number of batches: {len(train_loader)}")
    print(f"Test DataLoader created. Number of batches: {len(test_loader)}")
    print(f"Example batch image shape: {images.shape}")
    print(f"Example batch label shape: {labels.shape}")
    print(f"Unique labels in this batch: {labels.unique().tolist()}")


if __name__ == "__main__":
    main()
