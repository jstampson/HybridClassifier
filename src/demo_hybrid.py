import torch
from src.data import get_binary_mnist_dataloaders
from src.models import HybridModel

def demo_hybrid():
    print("--- Hybrid Model Demonstration ---")
    
    # 1. Setup DataLoader
    train_loader, _, _ = get_binary_mnist_dataloaders(data_dir='data', batch_size=4)
    images, labels = next(iter(train_loader))
    
    # 2. Initialize Model
    model = HybridModel(n_layers=2)
    model.eval()
    
    # 3. Forward Pass
    with torch.no_grad():
        outputs = model(images)
        
    print(f"Batch Input Shape:  {images.shape}")
    print(f"Classical Target:   {labels.tolist()}")
    print("-" * 30)
    print(f"Quantum Model Out:  \n{outputs}")
    print(f"Output Shape:       {outputs.shape}")

if __name__ == "__main__":
    demo_hybrid()
