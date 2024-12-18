import os
import torch
import torch.nn as nn
from sae import *
from dataset import *

# Example usage
input_dim = 768  # Your activation dimension
hidden_dim = 1024  # Desired hidden layer dimension
k = 128  # Number of top activations to keep

model = train_sparse_autoencoder(
    train_dir="path/to/train/activations",
    val_dir="path/to/val/activations",
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    k=k,
    batch_size=32,
    num_epochs=100,
    learning_rate=1e-3,
    sparsity_factor=10.0,
    patience=7
)