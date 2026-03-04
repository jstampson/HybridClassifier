# Hybrid Quantum-Classical Image Classifier (MNIST)

[![Quantum Computing](https://img.shields.io/badge/Quantum-PennyLane-blueviolet)](https://pennylane.ai/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-PyTorch-ee4c2c)](https://pytorch.org/)

A high-performance Hybrid Quantum-Classical Neural Network (HQNN) designed for binary classification of MNIST handwritten digits. This project demonstrates the integration of Variational Quantum Circuits (VQC) into a standard PyTorch deep learning pipeline.

## 🚀 Key Features

- **Hybrid Architecture**: Combines a 4-layered Classical Deep CNN/Linear "Squeezer" with a 4-qubit Variational Quantum Circuit.
- **Quantum Layer**: Implements **Angle Embedding** and **Strongly Entangling Layers** using PennyLane.
- **Inference Ready**: Includes scripts to demonstrate classification on unseen test data with confidence scores.
- **Comparative Analysis**: benchmarks HQNN against a purely classical architecture of equivalent width/depth.
- **Modular Design**: PEP8 compliant code structured for scalability and professional review.

## 🏗️ Architecture

The model follows a two-stage pipeline:

1.  **Classical Squeezer**: A Deep Neural Network (ReLU, BatchNorm, Kaiming Init) that reduces the 784-dimensional MNIST input to 4 latent features. The output is scaled by $\pi$ to serve as rotation angles.
2.  **Quantum Classifier**: A 4-qubit circuit where the features are encoded into $R_y$ rotations. A variational ansatz (Strongly Entangling Layers) processes the state, and the expectation value of $\langle Z \rangle$ on the first qubit provides the binary classification signal.

## 📊 Results Summary

| Model | Test Accuracy | Parameters | Expressivity |
| :--- | :--- | :--- | :--- |
| **Hybrid HQNN** | **~99.95%** | Mixed | High (Entangled) |
| **Pure Classical** | **~99.81%** | Fixed | Linear Head |

*Note: The choice of 4 qubits was made to optimize simulation efficiency while maintaining enough Hilbert space dimensionality for the binary classification task.*

## 🛠️ Installation & Setup

1.  **Clone the Repo**:
    ```bash
    git clone https://github.com/jstampson/HybridClassifier.git
    cd HybridClassifier
    ```
2.  **Setup Environment**:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # Windows
    # source venv/bin/activate # Linux/macOS
    pip install -r requirements.txt
    ```

## 💻 Usage

### 1. Run Comparison
Train both the Hybrid and Classical models to see the performance delta:
```bash
$env:PYTHONPATH="."; python src/compare_models.py
```

### 2. Classify Unseen Samples
Demonstrate the model on new, unseen test samples from MNIST:
```bash
$env:PYTHONPATH="."; python src/predict.py
```
This will generate an `inference_demo.png` showing the images, predictions, and confidence levels.

## 📝 Background for Interviewers

This project explores the **Near-term Intermediate-Scale Quantum (NISQ)** paradigm. By using a classical "squeezer," we mitigate the high cost of data re-uploading and large qubit counts, focusing the quantum circuit's power on the non-linear decision boundary in low-dimensional space.

The implementation uses **Backpropagation through Quantum Circuits** (via PennyLane's Torch interface), allowing for simultaneous end-to-end training of both classical and quantum weights.
