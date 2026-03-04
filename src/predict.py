import torch
from src.data import get_binary_mnist_dataloaders
from src.models import HybridModel
import matplotlib.pyplot as plt
import os

def predict_and_visualize(num_samples=5):
    """
    Demonstrates how to use the trained HybridModel to classify new, unseen samples from the MNIST test set.
    """
    print("--- Portfolio Inference Demo ---")
    
    # 1. Load Data (specifically the test set)
    _, _, test_loader = get_binary_mnist_dataloaders(data_dir='data', batch_size=num_samples)
    images, labels = next(iter(test_loader))
    
    # 2. Initialize Model
    # In a production scenario, you would load weights here:
    # model.load_state_dict(torch.load('model_weights.pth'))
    model = HybridModel(n_layers=3)
    model.eval()
    
    # 3. Perform Inference
    print(f"Classifying {num_samples} unseen samples from the test set...")
    with torch.no_grad():
        outputs = model(images)
        # Map expectation values [-1, 1] to [0, 1] probabilities
        probs = (outputs + 1) / 2.0
        # Binary prediction (threshold 0.5)
        predictions = (probs > 0.5).int().view(-1)
        confidence = probs.view(-1)
    
    # 4. Visualize Results
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        img = images[i].squeeze().numpy()
        plt.imshow(img, cmap='gray')
        
        pred_label = predictions[i].item()
        true_label = labels[i].item()
        conf_score = confidence[i].item() if pred_label == 1 else (1 - confidence[i].item())
        
        color = 'green' if pred_label == true_label else 'red'
        plt.title(f"Pred: {pred_label}\nTrue: {true_label}\nConf: {conf_score:.2%}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('inference_demo.png')
    print("Inference visualization saved to 'inference_demo.png'.")
    print("-" * 30)
    for i in range(num_samples):
        print(f"Sample {i+1}: True={labels[i].item()} | Predicted={predictions[i].item()} (Confidence: {confidence[i].item():.4f})")

if __name__ == "__main__":
    predict_and_visualize()
