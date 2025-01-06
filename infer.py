import json
import h5py
import torch
import pandas as pd
import numpy as np

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

activations_path = config['activations_path']
train_seqs_path = f"{activations_path}/train_seqs.bed"

model_save_path = config['model_save_path']
# Load the best SAE model
model = torch.load(model_save_path)
model.eval()

# Load sequence coordinates
seq_coords = []
with open(train_seqs_path, 'r') as f:
    for line in f:
        seq_coords.append(line.strip().split())

# Initialize results list
results = []

# find the number of activations per file
activation_file = f"{activations_path}/activations_0.h5"
with h5py.File(activation_file, 'r') as f:
    activations = f['activations'][:]
    num_activations = activations.shape[0]

# input seq len = 524288
# output seq len = 196608
# pad = (524288-196608)/2 = 163840

pad = (524288-196608)//2

top_values_n = 100

# Iterate through activation files
for i in range(len(seq_coords) // num_activations):  # Assuming each file corresponds to 16 entries
    activation_file = f"{activations_path}/activations_{i}.h5"
    with h5py.File(activation_file, 'r') as f:
        activations = f['activations'][:]
        
        # Convert activations to tensor and run forward pass
        activations_tensor = torch.tensor(activations, dtype=torch.float32)
        with torch.no_grad():
            output = model(activations_tensor)
        
        # Flatten the output and find the top 100 activations
        output_flat = output.view(-1)
        top_values, top_indices = torch.topk(output_flat, top_values_n)
        
        # Map index to sequence coordinates
        seq_index = i * num_activations + max_index // activations.shape[1]
        sae_node_id = max_index % activations.shape[1]
        seq_coord = seq_coords[seq_index]

        # Map indices to SAE node numbers
        for value, index in zip(top_values, top_indices):
            sae_node_id = index.item() % output.shape[1]
            results.append({
                'SAE_Node_ID': sae_node_id,
                'Activation_Value': value.item()
            })

        # Append result
        results.append({
            'SAE_Node_ID': sae_node_id,
            'Chromosome': seq_coord[0],
            'Start': seq_coord[1],
            'End': seq_coord[2],
            'Max_Activation': max_activation
        })

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('strongest_activations.csv', index=False)