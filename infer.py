import json
import h5py
import torch
import pandas as pd
import numpy as np
import os
from dataset import NormalizeActivations
from sae import infer_sparse_autoencoder

# Load configuration
with open('config.json', 'r') as f:
    configs = json.load(f)

activations_path = configs['activations_path']
train_seqs_path = f"{activations_path}/train_seqs.bed"

pad = (524288-196608)//2

#load global max if it exists
if os.path.exists(configs["global_max_save_path"]):
    with open(configs["global_max_save_path"]) as file_open:
        global_max = json.load(file_open)
    print("Using global max from file")
else:
    global_max = None

# Save results to CSV
results_df = infer_sparse_autoencoder(
    configs['model_save_path'],
    activations_path,
    configs["input_channels"],
    configs["expansion_factor"]*configs["input_channels"],
    int(configs["topk_pct"]*configs["input_channels"]),
    transform=NormalizeActivations(global_max=global_max), 
    resolution=524288//configs["seq_len"], 
    pad=pad, 
    top_chunk_num=32)

results_df.to_csv(os.path.join(configs['model_save_path'], 'strongest_activations.csv'))