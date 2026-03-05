"""
predict.py – Inference script for the Hybrid Quantum-Classical Classifier.

Classifies unseen handwritten digit images (0 or 1) from the MNIST test set
using a trained HybridModel checkpoint.

Usage (from the project root):
    # Windows PowerShell
    $env:PYTHONPATH = "."
    python src/predict.py

    # Linux / macOS
    PYTHONPATH=. python src/predict.py

    # Optional: point at a custom weights file
    python src/predict.py --weights checkpoints/hybrid_model.pth --samples 8

The script expects a saved checkpoint produced by train.py. If no checkpoint is
found it will raise a clear error rather than silently running random weights.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch

from src.data import get_binary_mnist_dataloaders
from src.models import HybridModel


# ─────────────────────────────────────────────────────────────────────────────
# Default paths
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = os.path.join("checkpoints", "hybrid_model.pth")
DEFAULT_SAMPLES = 8


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, n_layers: int = 3) -> HybridModel:
    """
    Instantiate a HybridModel and load pre-trained weights from *checkpoint_path*.

    Args:
        checkpoint_path: Path to a .pth file produced by train.py.
        n_layers:        Number of StronglyEntanglingLayers – must match training.

    Returns:
        The model in evaluation mode with weights loaded.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"\n[ERROR] Checkpoint not found: '{checkpoint_path}'\n"
            "  ➤  Train the model first:\n"
            "       $env:PYTHONPATH=\".\"; python src/train.py\n"
            "  ➤  Or point --weights at an existing .pth file."
        )

    model = HybridModel(n_layers=n_layers)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[✓] Loaded weights from: {checkpoint_path}")
    return model


def classify(model: HybridModel, images: torch.Tensor):
    """
    Run inference and convert raw ⟨Z⟩ expectation values to class predictions.

    The quantum circuit returns expectation values in [-1, 1].
    We map them linearly to probabilities p ∈ [0, 1] via p = (v + 1) / 2,
    then threshold at 0.5 to get a hard binary prediction.

    Args:
        model:  A trained HybridModel in eval mode.
        images: Tensor of shape (N, 1, 28, 28) – normalised MNIST images.

    Returns:
        predictions (Tensor[int], shape N): 0 or 1
        probabilities (Tensor[float], shape N): probability of class 1
    """
    with torch.no_grad():
        raw = model(images)                 # shape (N, 1), values in [-1, 1]
        probs = (raw + 1.0) / 2.0          # map to [0, 1]
        predictions = (probs > 0.5).int().view(-1)
        probabilities = probs.view(-1)
    return predictions, probabilities


def confidence_for_prediction(pred: int, prob_class1: float) -> float:
    """
    Return the model's confidence in its own prediction.

    - If the prediction is class 1, confidence = prob_class1.
    - If the prediction is class 0, confidence = 1 - prob_class1.
    """
    return prob_class1 if pred == 1 else 1.0 - prob_class1


# ─────────────────────────────────────────────────────────────────────────────
# Main inference routine
# ─────────────────────────────────────────────────────────────────────────────

def predict_and_visualize(
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    num_samples: int = DEFAULT_SAMPLES,
    data_dir: str = "data",
    output_path: str = "inference_demo.png",
):
    """
    End-to-end inference demo:
      1. Load the trained model from disk.
      2. Sample `num_samples` images from the MNIST test set.
      3. Classify each image and compute confidence.
      4. Plot results with colour-coded titles (green = correct, red = wrong).
      5. Save the figure and print a summary table to stdout.

    Args:
        checkpoint_path: Path to the saved .pth weights file.
        num_samples:     Number of test images to classify.
        data_dir:        Directory where MNIST data is stored / downloaded.
        output_path:     File path for the saved visualisation.
    """
    print("=" * 60)
    print("  Hybrid Quantum-Classical Classifier – Inference Demo")
    print("=" * 60)

    # ── 1. Load model ────────────────────────────────────────────────────────
    model = load_model(checkpoint_path)

    # ── 2. Load test data ────────────────────────────────────────────────────
    _, _, test_loader = get_binary_mnist_dataloaders(
        data_dir=data_dir, batch_size=num_samples
    )
    images, true_labels = next(iter(test_loader))
    print(f"[✓] Loaded {num_samples} unseen test samples  "
          f"(label distribution: {true_labels.tolist()})")

    # ── 3. Classify ──────────────────────────────────────────────────────────
    predictions, probabilities = classify(model, images)

    # ── 4. Visualise ─────────────────────────────────────────────────────────
    cols = min(num_samples, 8)
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.8))
    axes = [axes] if rows == 1 and cols == 1 else list(
        axes.flat if hasattr(axes, "flat") else axes
    )

    for i in range(num_samples):
        ax = axes[i]
        img = images[i].squeeze().numpy()
        ax.imshow(img, cmap="gray")
        ax.axis("off")

        pred = predictions[i].item()
        true = true_labels[i].item()
        conf = confidence_for_prediction(pred, probabilities[i].item())
        colour = "green" if pred == true else "red"

        ax.set_title(
            f"Pred: {pred}  |  True: {true}\nConf: {conf:.1%}",
            color=colour,
            fontsize=8,
            pad=4,
        )

    # Hide any unused subplot axes
    for j in range(num_samples, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Hybrid QNN – Binary MNIST Inference\n(green = correct, red = incorrect)",
        fontsize=11,
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"[✓] Visualisation saved → {output_path}")

    # ── 5. Console summary ───────────────────────────────────────────────────
    print()
    print(f"{'#':<5} {'True':>6} {'Pred':>6} {'Conf':>8} {'Status':>8}")
    print("-" * 38)
    correct = 0
    for i in range(num_samples):
        pred = predictions[i].item()
        true = true_labels[i].item()
        conf = confidence_for_prediction(pred, probabilities[i].item())
        status = "✓" if pred == true else "✗"
        if pred == true:
            correct += 1
        print(f"{i+1:<5} {true:>6} {pred:>6} {conf:>7.1%} {status:>8}")

    print("-" * 38)
    print(f"Accuracy on this batch: {correct}/{num_samples}  "
          f"({correct/num_samples:.1%})")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with a trained HybridModel on MNIST test data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        default=DEFAULT_CHECKPOINT,
        metavar="PATH",
        help="Path to the saved model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        metavar="N",
        help="Number of test images to classify.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        metavar="DIR",
        help="Directory for MNIST data.",
    )
    parser.add_argument(
        "--output",
        default="inference_demo.png",
        metavar="FILE",
        help="Output file path for the inference visualisation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    predict_and_visualize(
        checkpoint_path=args.weights,
        num_samples=args.samples,
        data_dir=args.data_dir,
        output_path=args.output,
    )
