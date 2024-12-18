import os
from dataset import *
import json
import numpy as np


config_file = "config.json"
with open(config_file) as config_open:
    configs = json.load(config_open)

# Find global maximum across all files
activation_pattern = configs["activations_path"]
global_max = find_global_max(activation_pattern)
print(f"Global maximum value: {global_max}")

# save to npy file
np.save(configs["global_max_save_path"], global_max)