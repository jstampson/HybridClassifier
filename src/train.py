"""
train.py – Training script for the Hybrid Quantum-Classical Classifier.

Trains the HybridModel on the binary (0 vs 1) MNIST dataset, evaluates it on
the held-out test set, and saves the model weights to disk so that predict.py
can load them for inference.

Usage (from the project root):
    # Windows PowerShell
    $env:PYTHONPATH = "."
    python src/train.py

    # Linux / macOS
    PYTHONPATH=. python src/train.py

Outputs:
    checkpoints/hybrid_model.pth   – Saved model weights
    loss_curve.png                 – Per-epoch training loss curve
    confusion_matrix.png           – Test-set confusion matrix
"""

import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

from src.data import get_binary_mnist_dataloaders
from src.models import HybridModel

# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters  (edit these to experiment)
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 0.005
LR_GAMMA = 0.9          # ExponentialLR decay factor per epoch
WEIGHT_DECAY = 1e-4     # L2 regularisation
N_LAYERS = 3            # Strongly entangling layers inside the quantum circuit
DATA_DIR = "data"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "hybrid_model.pth")


def train_model():
    """
    Full training + evaluation pipeline for the Hybrid Quantum-Classical model.

    Steps:
        1. Load & split MNIST (classes 0 & 1) into train / val / test.
        2. Initialise HybridModel, BCELoss, Adam optimiser, LR scheduler.
        3. Train for `EPOCHS` epochs, logging loss every 40 batches.
        4. Validate after each epoch and log validation accuracy.
        5. Evaluate on the held-out test set and report accuracy.
        6. Save model weights to ``checkpoints/hybrid_model.pth``.
        7. Persist a loss curve and confusion matrix as PNG files.
    """
    print("=" * 60)
    print("  Hybrid Quantum-Classical Training")
    print("=" * 60)
    print(f"  Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LEARNING_RATE}")
    print(f"  Quantum layers: {N_LAYERS}  |  Checkpoint: {CHECKPOINT_PATH}")
    print("=" * 60)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_binary_mnist_dataloaders(
        data_dir=DATA_DIR, batch_size=BATCH_SIZE
    )
    print(f"[✓] Train batches: {len(train_loader)}  |  "
          f"Val batches: {len(val_loader)}  |  "
          f"Test batches: {len(test_loader)}")

    # ── 2. Model / Loss / Optimiser ───────────────────────────────────────────
    model = HybridModel(n_layers=N_LAYERS)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_GAMMA)

    train_losses: list[float] = []
    val_accuracies: list[float] = []

    # ── 3 & 4. Training loop ──────────────────────────────────────────────────
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = model(data)
            # Quantum circuit outputs ⟨Z⟩ ∈ [-1, 1]; map to probability ∈ [0, 1]
            probs = (outputs + 1.0) / 2.0
            loss = criterion(probs, target.float().unsqueeze(1))

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if batch_idx % 40 == 0:
                print(
                    f"  Epoch {epoch+1}/{EPOCHS} | "
                    f"Batch {batch_idx:>4}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for data, target in val_loader:
                probs = (model(data) + 1.0) / 2.0
                preds = (probs > 0.5).int().view(-1)
                val_preds.extend(preds.tolist())
                val_targets.extend(target.tolist())

        val_acc = accuracy_score(val_targets, val_preds)
        val_accuracies.append(val_acc)
        print(
            f"  ── Epoch {epoch+1} complete | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Val Acc: {val_acc*100:.2f}%"
        )

    total_time = time.time() - start_time
    print(f"\n[✓] Training finished in {total_time / 60:.2f} minutes")

    # ── 5. Test evaluation ────────────────────────────────────────────────────
    print("\n--- Test Set Evaluation ---")
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for data, target in test_loader:
            probs = (model(data) + 1.0) / 2.0
            preds = (probs > 0.5).int().view(-1)
            all_preds.extend(preds.tolist())
            all_targets.extend(target.tolist())

    test_accuracy = accuracy_score(all_targets, all_preds)
    conf_matrix = confusion_matrix(all_targets, all_preds)

    print(f"  Test Accuracy : {test_accuracy * 100:.2f}%")
    print(f"  Confusion Matrix:\n{conf_matrix}")

    # ── 6. Save checkpoint ────────────────────────────────────────────────────
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"\n[✓] Model weights saved → {CHECKPOINT_PATH}")

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    # Training loss curve
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, EPOCHS + 1), train_losses, marker="o", label="Train Loss")
    plt.title("Hybrid QNN – Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.xticks(range(1, EPOCHS + 1))
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    print("[✓] Loss curve saved → loss_curve.png")

    # Confusion matrix
    plt.figure(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix  (Acc: {test_accuracy*100:.1f}%)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("[✓] Confusion matrix saved → confusion_matrix.png")

    print("\n" + "=" * 60)
    print(f"  Final test accuracy: {test_accuracy*100:.2f}%")
    print("  Run predict.py to classify new samples.")
    print("=" * 60)


if __name__ == "__main__":
    train_model()
