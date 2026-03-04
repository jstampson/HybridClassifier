import torch
import torch.nn as nn
import math
import pennylane as qml
import numpy as np
import random

class ClassicalSqueezer(nn.Module):
    """
    A classical neural network component that compresses MNIST images into a low-dimensional
    feature space suitable for quantum circuit embedding.

    This module takes a flattened 28x28 image (784 features) and reduces it to 4 features.
    The choice of 4 dimensions is specifically optimized for 4-qubit quantum simulations,
    allowing each feature to be mapped directly to a rotation angle on a unique qubit
    without exceeding the computational complexity typical for local quantum simulators.

    The output is constrained within [-pi, pi] using a Tanh activation scaled by pi,
    making the features immediately usable as rotation angles (e.g., for Ry or Rx gates).
    """
    def __init__(self, input_dim: int = 784, hidden_dims: list = [128, 64], output_dim: int = 4):
        super(ClassicalSqueezer, self).__init__()
        
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(0.1))
            curr_dim = h_dim
            
        layers.append(nn.Linear(curr_dim, output_dim))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
        
        # Proper initialization for stable training
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the squeezer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 784).
            
        Returns:
            torch.Tensor: Encoded features of shape (batch_size, 4), scaled by pi.
        """
        # Ensure input is flattened if not already
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        x = self.network(x)
        return x * math.pi

# Define the number of qubits for the quantum layer
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    """
    Quantum node that embeds classical features and applies a variational circuit.
    
    Args:
        inputs: Classical features (batch_size, 4).
        weights: Trainable parameters for the StronglyEntanglingLayers.
    """
    # 1. Encode the 4 classical features into R_y rotations
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    
    # 2. Apply heavily entangled parameterized layers
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    
    # 3. Return the expectation value of PauliZ on the first qubit
    return qml.expval(qml.PauliZ(0))


class HybridModel(nn.Module):
    """
    A hybrid quantum-classical neural network for binary image classification.
    
    It combines a ClassicalSqueezer to reduce the image to 4 features, and a
    PennyLane Quantum Neural Network layer to process these features.
    """
    def __init__(self, n_layers: int = 2):
        super(HybridModel, self).__init__()
        
        # Classical feature extractor with optimized architecture
        self.squeezer = ClassicalSqueezer(input_dim=784, hidden_dims=[128, 64], output_dim=4)
        
        # Quantum layer wrapped as a standard PyTorch module
        # StronglyEntanglingLayers requires weight shape (n_layers, n_qubits, 3)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the hybrid model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28) or (batch_size, 784).
            
        Returns:
            torch.Tensor: Expectation values of shape (batch_size, 1) to be used for classification.
        """
        # Classical feature extraction
        features = self.squeezer(x)
        
        # Quantum processing
        out = self.qlayer(features)
        
        # TorchLayer might output (batch_size) instead of (batch_size, 1), depending on PyTorch batching
        # Ensure the output is (batch_size, 1)
        if out.dim() == 1:
            out = out.unsqueeze(1)
            
        return out


class PureClassicalModel(nn.Module):
    """
    A purely classical mirror of the HybridModel for comparison.
    Uses the same ClassicalSqueezer but replaces the Quantum Layer
    with a single Linear layer that maps 4 features to 1 output.
    """
    def __init__(self):
        super(PureClassicalModel, self).__init__()
        # Use the exact same Squeezer architecture
        self.squeezer = ClassicalSqueezer(input_dim=784, hidden_dims=[128, 64], output_dim=4)
        
        # Mirror the quantum layer's function: 4 inputs -> 1 scalar output
        # Using Tanh to match the HybridModel's expectation value range [-1, 1]
        self.classical_head = nn.Sequential(
            nn.Linear(4, 1),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.squeezer(x)
        # We scale features back down to [0, 1] relative range if needed, 
        # but here we just pass the squeezed features (scaled by pi) through.
        return self.classical_head(features)
