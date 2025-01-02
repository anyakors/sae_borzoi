import os
import torch
import torch.nn as nn
from sae import *
from dataset import *
import json
import numpy as np


config_file = "config.json"
with open(config_file) as config_open:
    configs = json.load(config_open)

# Example usage
input_dim = configs["input_channels"]  # Your activation dimension
hidden_dim = configs["expansion_factor"]*configs["input_channels"]  # Desired hidden layer dimension
k = int(configs["topk_pct"]*configs["input_channels"]) # Number of top activations to keep
learning_rate = configs["learning_rate"]
sparsity_target = configs["sparsity_target"]

#load global max if it exists
if os.path.exists(configs["global_max_save_path"]):
    global_max = np.load(configs["global_max_save_path"])
    print("Using global max from file:", global_max)
else:
    global_max = None

if not os.path.exists(configs["model_save_path"]):
    os.makedirs(configs["model_save_path"])

model = train_sparse_autoencoder(
    train_dir=configs["activations_path"],
    val_dir=configs["activations_path_val"],
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    k=k,
    batch_size=2,
    num_epochs=100,
    learning_rate=learning_rate,
    sparsity_factor=10.0,
    sparsity_target=sparsity_target,
    patience=7,
    checkpoint_dir=configs["model_save_path"],
    global_max=global_max
)