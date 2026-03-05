import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import os
import time

from src.data import get_binary_mnist_dataloaders
from src.models import HybridModel, PureClassicalModel

def train_one_model(model, name, train_loader, val_loader, test_loader, epochs=8, lr=0.005):
    print(f"\n--- Training {name} ---")
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    train_losses = []
    val_accuracies = []
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data)
            probs = (outputs + 1) / 2.0
            loss = criterion(probs, target.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"[{name}] Ep {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for data, target in val_loader:
                outputs = model(data)
                probs = (outputs + 1) / 2.0
                preds = (probs > 0.5).int()
                val_preds.extend(preds.view(-1).tolist())
                val_targets.extend(target.tolist())
        
        val_acc = accuracy_score(val_targets, val_preds)
        val_accuracies.append(val_acc)
        print(f"==> [{name}] Epoch {epoch+1} | Avg Loss: {avg_train_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
    
    duration = time.time() - start_time
    print(f"[{name}] Finished in {duration/60:.2f} min")
    
    # Final Test
    model.eval()
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            probs = (outputs + 1) / 2.0
            preds = (probs > 0.5).int()
            test_preds.extend(preds.view(-1).tolist())
            test_targets.extend(target.tolist())
            
    test_acc = accuracy_score(test_targets, test_preds)
    test_conf = confusion_matrix(test_targets, test_preds)
    
    return {
        "train_losses": train_losses,
        "val_accuracies": val_accuracies,
        "test_acc": test_acc,
        "test_conf": test_conf
    }

def main():
    # Setup
    BATCH_SIZE = 16
    EPOCHS = 5 # Reduced for quicker comparison
    LR = 0.005
    
    train_loader, val_loader, test_loader = get_binary_mnist_dataloaders(
        data_dir="data", batch_size=BATCH_SIZE
    )
    
    # Train Hybrid
    hybrid_model = HybridModel(n_layers=3)
    hybrid_results = train_one_model(hybrid_model, "Hybrid (Quantum)", train_loader, val_loader, test_loader, epochs=EPOCHS, lr=LR)
    
    # Train Classical
    classical_model = PureClassicalModel()
    classical_results = train_one_model(classical_model, "Pure Classical", train_loader, val_loader, test_loader, epochs=EPOCHS, lr=LR)
    
    # Print Comparison Summary
    print("\n" + "="*40)
    print("FINAL COMPARISON")
    print("="*40)
    print(f"{'Metric':<20} | {'Hybrid':<15} | {'Classical':<15}")
    print("-" * 56)
    print(f"{'Final Test Acc':<20} | {hybrid_results['test_acc']*100:>14.2f}% | {classical_results['test_acc']*100:>14.2f}%")
    print("="*40)

    # Plot Comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(hybrid_results['train_losses'], label='Hybrid')
    plt.plot(classical_results['train_losses'], label='Classical')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(hybrid_results['val_accuracies'], label='Hybrid')
    plt.plot(classical_results['val_accuracies'], label='Classical')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('comparison_results.png')
    print("\nComparison plots saved to 'comparison_results.png'.")

if __name__ == "__main__":
    main()
