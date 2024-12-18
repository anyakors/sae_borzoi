import os
from dataset import *


# Find global maximum across all files
activation_pattern = "v2_activations/conv1d_1/activations_*.h5"
global_max = find_global_max(activation_pattern)
print(f"Global maximum value: {global_max}")
