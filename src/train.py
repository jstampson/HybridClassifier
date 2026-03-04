import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import os
import time

from src.data import get_binary_mnist_dataloaders
from src.models import HybridModel

def train_model():
    # 1. Hyperparameters
    BATCH_SIZE = 16
    EPOCHS = 8
    LEARNING_RATE = 0.005
    DATA_DIR = 'data'
    
    print("--- Starting Hybrid Quantum-Classical Training ---")
    print(f"Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    
    # 2. Setup Data
    train_loader, test_loader = get_binary_mnist_dataloaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE)
    
    # 3. Initialize Model, Loss, and Optimizer
    model = HybridModel(n_layers=3)
    criterion = nn.BCELoss()
    # Using weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # Scheduler to decrease LR over time
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    # Training logs
    train_losses = []
    
    # 4. Training Loop
    model.train()
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            
            # HybridModel returns expectation values in range [-1, 1]
            # Convert to probabilities [0, 1] for BCELoss
            probs = (outputs + 1) / 2.0
            
            # Ensure target is float for BCELoss
            loss = criterion(probs, target.float().unsqueeze(1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 40 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"==> Epoch {epoch+1} Complete | Average Loss: {avg_loss:.4f}")

    total_time = time.time() - start_time
    print(f"Finished Training in {total_time/60:.2f} minutes")

    # 5. Evaluation
    print("\n--- Evaluating on Test Set ---")
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            # Map back to [0, 1] probabilities
            probs = (outputs + 1) / 2.0
            # Predict class based on 0.5 threshold
            preds = (probs > 0.5).int()
            
            all_preds.extend(preds.view(-1).tolist())
            all_targets.extend(target.tolist())

    # Metrics
    accuracy = accuracy_score(all_targets, all_preds)
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(conf_matrix)

    # 6. Visualization
    # Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.title('Hybrid Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig('loss_curve.png')
    print("Training loss curve saved to 'loss_curve.png'.")

    # Confusion Matrix Plot
    plt.figure(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix (Acc: {accuracy*100:.1f}%)')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to 'confusion_matrix.png'.")

if __name__ == "__main__":
    train_model()
