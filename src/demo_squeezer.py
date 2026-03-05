import torch
from src.data import get_binary_mnist_dataloaders
from src.models import ClassicalSqueezer
import math

def demo_squeezer():
    # 1. Setup Data Loader
    print("--- Squeezer Demonstration ---")
    train_loader, _, _ = get_binary_mnist_dataloaders(data_dir='data', batch_size=1)
    
    # 2. Extract exactly one sample
    image, label = next(iter(train_loader))
    
    # 3. Initialize Squeezer
    model = ClassicalSqueezer()
    model.eval() # Set to evaluation mode
    
    # 4. Pass image through squeezer
    with torch.no_grad():
        squeezed_output = model(image)
        
    # 5. Print results
    print(f"Original image shape:  {image.shape} (batch, channel, height, width)")
    print(f"Flattened input size:  {image.numel()} pixels")
    print(f"Target label:          {label.item()}")
    print("-" * 30)
    print(f"Squeezed output shape: {squeezed_output.shape}")
    print(f"Quantum Angles (rad):  {squeezed_output.squeeze().tolist()}")
    
    # Show value range check
    print(f"Within [-pi, pi] range: {torch.all(squeezed_output.abs() <= math.pi).item()}")

if __name__ == "__main__":
    demo_squeezer()
