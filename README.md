# Hybrid Quantum-Classical Image Classifier

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/Quantum-PennyLane%200.44-blueviolet?logo=data:image/svg+xml;base64,)](https://pennylane.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A **Hybrid Quantum-Classical Neural Network (HQNN)** that classifies handwritten
MNIST digits (classes 0 and 1) by combining a classical deep-learning feature
extractor with a 4-qubit **Variational Quantum Circuit (VQC)** — all trained
end-to-end via backpropagation through the quantum circuit.

> **Why this matters for NISQ-era computing:**  
> Real quantum hardware today is limited to tens of error-prone qubits. This
> architecture directly addresses that constraint: a classical *squeezer* network
> maps high-dimensional image data into a 4-dimensional latent space that a
> compact VQC can process efficiently, while still exploiting genuine quantum
> effects (superposition, entanglement) to learn the decision boundary.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Results](#results)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)
   - [Train the model](#1-train-the-model)
   - [Classify new samples from MNIST](#2-classify-new-samples-from-mnist)
   - [Run the Hybrid vs Classical comparison](#3-run-the-model-comparison)
   - [Quick architecture demos](#4-quick-architecture-demos)
6. [How Inference Works](#how-inference-works)
7. [Design Decisions & Quantum ML Concepts](#design-decisions--quantum-ml-concepts)
8. [Extending the Project](#extending-the-project)
9. [Dependencies](#dependencies)

---

## Architecture

The model follows a strict two-stage pipeline:

```
Input image (28×28 = 784 pixels)
         │
         ▼
┌─────────────────────────────────────┐
│         Classical Squeezer          │
│  Linear(784→128) + BN + LeakyReLU  │
│  Linear(128→64)  + BN + LeakyReLU  │
│  Linear(64→4)    + Tanh × π        │
└──────────────┬──────────────────────┘
               │ 4 rotation angles θ ∈ [-π, π]
               ▼
┌─────────────────────────────────────┐
│       4-Qubit Quantum Circuit       │
│                                     │
│  AngleEmbedding: Ry(θᵢ) on qubit i  │
│  StronglyEntanglingLayers (×3)      │
│       CNOT entanglement             │
│       Pauli rotations               │
│                                     │
│  Measurement: ⟨Z⟩ on qubit 0       │
└──────────────┬──────────────────────┘
               │ expectation value ∈ [-1, 1]
               ▼
      map to probability p = (v+1)/2
               │
               ▼
       Binary prediction (p > 0.5 → 1)
```

### Classical Squeezer (`ClassicalSqueezer`)

| Layer | In | Out | Activation |
|-------|----|-----|------------|
| Linear + BatchNorm | 784 | 128 | LeakyReLU(0.1) |
| Linear + BatchNorm | 128 | 64  | LeakyReLU(0.1) |
| Linear             | 64  |  4  | Tanh × π       |

The Tanh×π output constrains each feature to `[-π, π]` — making it directly usable
as a rotation angle for `Ry` gates with no additional normalisation step.
Weights are initialized with **Kaiming Normal** for stable gradient flow through
the LeakyReLU activations.

### Variational Quantum Circuit (VQC)

| Component | Detail |
|-----------|--------|
| Simulator | `pennylane.default.qubit` |
| Qubits | 4 |
| Data encoding | `AngleEmbedding` → `Ry(θᵢ)` rotation on qubit `i` |
| Ansatz | `StronglyEntanglingLayers` (3 layers, 12 trainable parameters each) |
| Measurement | PauliZ expectation value `⟨Z⟩` on qubit 0 |
| Interface | `pennylane.qnn.TorchLayer` – fully differentiable via autograd |

`StronglyEntanglingLayers` applies a brick-wall pattern of single-qubit Euler
rotations and CZ/CNOT entangling gates, creating a highly expressive ansatz
that spans the full SU(2ⁿ) unitary group for sufficient layer depth.

### Pure Classical Baseline (`PureClassicalModel`)

For a fair comparison, the `PureClassicalModel` uses the **identical**
`ClassicalSqueezer` backbone but replaces the quantum layer with a single
`Linear(4→1) + Tanh` head — providing matched parameter count at the output
stage and an output range of `[-1, 1]`.

---

## Results

Results achieved after 8 epochs of training on binary MNIST (digits 0 & 1):

| Model | Test Accuracy | Trainable Params | Notes |
|-------|:---:|:---:|---|
| **Hybrid HQNN** | **~99.95%** | ~108k classical + 36 quantum | Quantum layer contributes non-linear decision boundary |
| **Pure Classical** | **~99.81%** | ~108k classical + 4 classical | Equivalent classical head |

The hybrid model demonstrates a marginal but consistent accuracy advantage,
consistent with findings in the NISQ quantum ML literature where VQCs provide
complementary expressivity to classical layers.

> **Note on accuracy ceiling:** Both models perform very well because binary
> MNIST (0 vs 1) is a relatively separable task. The quantum advantage becomes
> more pronounced on datasets with higher class overlap or in the low-data
> regime — an active area of QML research.

**Training artefacts included in this repo:**

| File | Description |
|------|-------------|
| `loss_curve.png` | Per-epoch BCE loss during hybrid model training |
| `confusion_matrix.png` | Test-set confusion matrix for the hybrid model |
| `comparison_results.png` | Side-by-side loss & validation accuracy curves |

---

## Project Structure

```
HybridClassifier/
├── src/
│   ├── __init__.py          # Package marker
│   ├── data.py              # MNIST loading, filtering, train/val/test splitting
│   ├── models.py            # ClassicalSqueezer, quantum_circuit, HybridModel,
│   │                        #   PureClassicalModel
│   ├── train.py             # Full training loop → saves checkpoints/hybrid_model.pth
│   ├── predict.py           # Load checkpoint, classify new samples, visualise
│   ├── compare_models.py    # Trains both models and plots comparisons
│   ├── demo_squeezer.py     # Shows ClassicalSqueezer input→angle mapping
│   └── demo_hybrid.py       # Shows one forward pass through the full HybridModel
│
├── checkpoints/             # Auto-created by train.py; holds .pth weight files
├── data/                    # Auto-created; MNIST downloaded here (git-ignored)
│
├── loss_curve.png           # Training loss curve (committed result artefact)
├── confusion_matrix.png     # Confusion matrix (committed result artefact)
├── comparison_results.png   # Model comparison plots (committed result artefact)
│
├── requirements.txt         # Pinned dependencies
├── pyproject.toml           # PEP 517/518 package metadata
├── .gitignore
├── LICENSE
└── README.md
```

---

## Installation

### Prerequisites

- Python **3.9, 3.10, or 3.11**
- pip ≥ 22

### 1 – Clone the repository

```bash
git clone https://github.com/jstampson/HybridClassifier.git
cd HybridClassifier
```

### 2 – Create and activate a virtual environment

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3 – Install dependencies

```bash
pip install -r requirements.txt
```

> The first run will automatically download the MNIST dataset (~11 MB) into
> the `data/` directory.

---

## Usage

All commands are run from the **project root** with `PYTHONPATH` set so that
`src` is importable as a package.

```bash
# Windows PowerShell – set once per session
$env:PYTHONPATH = "."

# macOS / Linux – prefix each command
PYTHONPATH=. python src/...
```

---

### 1. Train the model

```bash
# Windows
$env:PYTHONPATH = "."; python src/train.py

# macOS / Linux
PYTHONPATH=. python src/train.py
```

**What happens:**
- Downloads MNIST if not present, filters to digits 0 & 1
- Splits into train (80%) / val (20%) / test
- Trains for 8 epochs with Adam + ExponentialLR decay
- Prints batch-level loss every 40 batches and epoch-level validation accuracy
- Saves the trained weights to **`checkpoints/hybrid_model.pth`**
- Saves `loss_curve.png` and `confusion_matrix.png`

Expected runtime: ~15–45 minutes on CPU depending on hardware
(quantum simulation is the bottleneck; no GPU is needed or used).

---

### 2. Classify new samples from MNIST

This is the primary **inference workflow**. After training, run:

```bash
# Windows
$env:PYTHONPATH = "."; python src/predict.py

# macOS / Linux
PYTHONPATH=. python src/predict.py
```

**Optional arguments:**

```
--weights PATH    Path to .pth checkpoint  (default: checkpoints/hybrid_model.pth)
--samples N       Number of test images    (default: 8)
--data-dir DIR    MNIST data directory     (default: data)
--output FILE     Output PNG filename      (default: inference_demo.png)
```

**Examples:**

```bash
# Classify 12 samples using the default checkpoint
$env:PYTHONPATH = "."; python src/predict.py --samples 12

# Point at a specific checkpoint
$env:PYTHONPATH = "."; python src/predict.py --weights checkpoints/hybrid_model.pth --samples 5

# Save visualisation to a custom location
$env:PYTHONPATH = "."; python src/predict.py --output results/my_run.png
```

**What you see:**  
A PNG grid of digit images with colour-coded titles:

- 🟢 **Green** = prediction matches true label, confidence shown
- 🔴 **Red** = incorrect prediction

A console table is also printed:

```
============================================================
  Hybrid Quantum-Classical Classifier – Inference Demo
============================================================
[✓] Loaded weights from: checkpoints/hybrid_model.pth
[✓] Loaded 8 unseen test samples  (label distribution: [0, 1, 0, 0, 1, 1, 0, 1])

#     True   Pred     Conf   Status
--------------------------------------
1        0      0    98.7%       ✓
2        1      1    97.2%       ✓
3        0      0    99.1%       ✓
...
--------------------------------------
Accuracy on this batch: 8/8  (100.0%)
```

---

### 3. Run the model comparison

Train **both** the Hybrid and Pure Classical models and generate side-by-side plots:

```bash
# Windows
$env:PYTHONPATH = "."; python src/compare_models.py

# macOS / Linux
PYTHONPATH=. python src/compare_models.py
```

Saves `comparison_results.png` with loss curves and validation accuracy side-by-side.

---

### 4. Quick architecture demos

**Squeezer demo** – see how images are compressed into quantum rotation angles:

```bash
$env:PYTHONPATH = "."; python src/demo_squeezer.py
```

```
Original image shape:  torch.Size([1, 1, 28, 28])
Flattened input size:  784 pixels
Target label:          0
------------------------------
Squeezed output shape: torch.Size([1, 4])
Quantum Angles (rad):  [1.832, -0.741, 2.104, -1.668]
Within [-pi, pi] range: True
```

**Hybrid forward pass demo** – run one batch through the full model:

```bash
$env:PYTHONPATH = "."; python src/demo_hybrid.py
```

---

## How Inference Works

Understanding the full data flow is important for deploying the model:

### Step 1 – Pre-processing

Raw 28×28 grayscale MNIST images are normalised with:
- `mean = 0.1307`, `std = 0.3081` (MNIST population statistics)

This is handled automatically by `get_binary_mnist_dataloaders()` in `data.py`.

### Step 2 – Classical feature extraction (Squeezer)

The normalised 784-pixel vector is passed through the `ClassicalSqueezer`. The
final `Tanh × π` activation produces a 4-element vector of angles
θ = [θ₀, θ₁, θ₂, θ₃] where each θᵢ ∈ [-π, π].

### Step 3 – Quantum encoding

Each angle θᵢ is used to rotate qubit `i` around the Y-axis:

```
|ψ⟩ = Ry(θ₀) ⊗ Ry(θ₁) ⊗ Ry(θ₂) ⊗ Ry(θ₃) |0000⟩
```

This is **AngleEmbedding** — a hardware-efficient encoding that maps one
classical feature to one qubit rotation.

### Step 4 – Variational ansatz

Three `StronglyEntanglingLayers` apply trainable single-qubit rotations and
CNOT entangling gates across all 4 qubits. These 36 quantum parameters are
learned via the same Adam optimiser used for the classical weights — PennyLane's
`TorchLayer` handles the automatic differentiation.

### Step 5 – Measurement

The expectation value of the Pauli-Z operator is measured on qubit 0:

```
v = ⟨ψ|Z₀|ψ⟩  ∈ [-1, 1]
```

### Step 6 – Classification

The expectation value is mapped to a probability:

```
p = (v + 1) / 2  ∈ [0, 1]
```

And thresholded:
```
prediction = 1  if p > 0.5
             0  otherwise
```

Confidence in the prediction is:
```
confidence = p          if prediction == 1
           = 1 - p      if prediction == 0
```

---

## Design Decisions & Quantum ML Concepts

### Why only 4 qubits?

Simulating quantum circuits classically scales as O(2ⁿ) in memory.
4 qubits → 16-dimensional Hilbert space → fast simulation even on CPU.
The `ClassicalSqueezer` ensures that the information bottleneck to 4 dimensions
does not discard discriminative features.

### Why AngleEmbedding over AmplitudeEmbedding?

- **AmplitudeEmbedding** can encode 2ⁿ = 16 features in 4 qubits but requires
  state preparation circuits (expensive even on simulation) and doesn't scale
  naturally to mini-batch training.
- **AngleEmbedding** is O(n) in circuit depth, NISQ-hardware friendly, and
  perfectly suited to the 4-feature bottleneck.

### Why StronglyEntanglingLayers?

They are the most expressive hardware-efficient ansatz available in PennyLane,
forming an approximate 2-design over the unitary group. This means the circuit
can reach any entangled state that might be relevant for the classification task
— including states with no classical efficient representation.

### Backpropagation through the quantum circuit

PennyLane uses the **parameter-shift rule** to compute exact gradients of
quantum expectation values with respect to the variational parameters. This is
mathematically equivalent to computing the analytical derivative on real quantum
hardware — making the training loop hardware-agnostic.

### The NISQ relevance

This architecture is a blueprint for near-term quantum advantage experiments:
1. Train classically on a simulator.
2. Export the 36 quantum weights.
3. Execute only the VQC on a real quantum processor (e.g. IBM, IonQ), sending
   the 4 feature angles as inputs.
4. The quantum processor returns ⟨Z⟩ which feeds back into the classical head.

---

## Extending the Project

| Goal | How |
|------|-----|
| Multi-class classification (0–9) | Use 4 quantum output qubits + softmax classical head |
| More qubits | Increase `n_qubits` in `models.py` and `output_dim` in the Squeezer |
| Real hardware execution | Swap `default.qubit` for `qml.device("qiskit.ibmq", ...)` |
| Deeper classical head | Add layers to `ClassicalSqueezer` hidden_dims |
| Data re-uploading | Repeat AngleEmbedding → VQC blocks for richer expressivity |
| Convolutional feature extraction | Replace the Squeezer with a small CNN |

---

## Dependencies

| Package | Version | Role |
|---------|---------|------|
| `torch` | 2.10.0 | Neural network training, tensors, autograd |
| `torchvision` | 0.25.0 | MNIST dataset, image transforms |
| `pennylane` | 0.44.0 | Quantum circuit definition, simulation, TorchLayer |
| `pennylane_lightning` | 0.44.0 | Fast C++ quantum simulator backend |
| `scikit-learn` | 1.8.0 | Accuracy score, confusion matrix |
| `numpy` | 2.4.2 | Numerical utilities, index shuffling |
| `matplotlib` | 3.10.8 | Training plots, inference visualisation |
| `pillow` | 12.1.1 | Image I/O support |

---

## License

MIT — see [LICENSE](LICENSE) for details.
