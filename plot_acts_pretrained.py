import pandas as pd
import json
import shutil
import subprocess
import os
import glob
import numpy as np
import h5py
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

config_file = "config.json"
with open(config_file) as config_open:
    configs = json.load(config_open)

activation_path_pattern = configs['activations_path']
seqs = pd.read_csv(activation_path_pattern+'/train_seqs.bed', sep='\t', header=0)

# save new bed with paded coordinates

pad = (524288-196608)//2
seqs['start'] = seqs['1'] - pad
seqs['end'] = seqs['2'] + pad
# clip start to 0
seqs['start'] = seqs['start'].clip(lower=0)
# retain only seqs with length 524288
seqs = seqs[seqs['end']-seqs['start']==524288]

data_folder = 'data/'
if not os.path.isdir(data_folder):
    os.mkdir(data_folder)

seqs[['0', 'start', 'end', 'chunk_ind', 'inner_ind']].to_csv(data_folder+'train_seqs_padded.bed', sep='\t', header=False, index=False)

temp_folder = 'data/temp'
if not os.path.isdir(temp_folder):
    os.mkdir(temp_folder)

proc = subprocess.Popen(["which", "shuffleBed"], stdout=subprocess.PIPE)
out = proc.stdout.read().decode("utf-8")
bedtools_exec = "/".join(out.strip("\n").split("/")[:-1])
print("bedtools executable path to be used:", bedtools_exec)

f = open(f"{temp_folder}/seqs_intersect.bed", "w")
subprocess.call(
    [
        f"{bedtools_exec}/intersectBed",
        "-b",
        'data/GRCh38_cCREs_screen_exons_tiger.bed',
        "-a",
        data_folder+'train_seqs_padded.bed',
        "-wa", "-wb",
    ],
    stdout=f,
)

layer_seq_len = configs["seq_len"]

resolution = int(524288/layer_seq_len)
print("Resolution:", resolution)

ints = pd.read_csv(f"{temp_folder}/seqs_intersect.bed", sep="\t", header=None)
ints.columns = ["chr", "start", "end", "chunk_ind", "inner_ind", "chr2", "start2", "end2", "x1", "x2", "element"]

file_paths_ = os.listdir(activation_path_pattern)
file_paths = sorted(glob.glob(activation_path_pattern+'/*.h5'))

chunk_ids_present = [int(f.split('.')[0].split('_')[-1]) for f in file_paths_ if '.h5' in f]

els = []
vals = []

for el in tqdm(ints['element'].unique()):
    ints_el = ints[ints['element']==el]
    feature_count = 0
    for i, row in ints_el.iterrows():
        file_idx = row['chunk_ind']
        seq_idx = row['inner_ind']
        rel_start = max(0, int(np.floor((row['start2'] - row['start'])/resolution)))
        rel_end = min(int(np.ceil((row['end2'] - row['start'])/resolution)), layer_seq_len)
        if file_idx in chunk_ids_present and rel_end-rel_start!=0 and feature_count<10000:
            with h5py.File(f"{activation_path_pattern}/activations_{file_idx}.h5", 'r') as f:
                first_key = list(f.keys())[0]  # Get the first dataset key
                seq_len = f[first_key].shape[1]
                activations = f[first_key][seq_idx,rel_start:rel_end,:]
            try:
                vals.append(np.max(activations))
                els.append(el)
                feature_count += 1
            except ValueError:
                print(f'Error with {el} in {file_idx} at {seq_idx} from {rel_start} to {rel_end}')
                print(activations.shape)

df_act = pd.DataFrame({'element': els, 'activation': vals})

df_ca = df_act[df_act['element'].str.contains('CA')]
df_tf = df_act[df_act['element'].str.contains('TF')]
df_ls = df_act[df_act['element'].str.contains('LS')]

layer_name = configs['layer_name']
if not os.path.isdir(f"data/{layer_name}"):
    os.mkdir(f"data/{layer_name}")

df_act.to_csv(f'data/{layer_name}/activations.csv', index=False)

plt.figure(figsize=(3, 2))
ax = sns.histplot(data=df_ca, x='activation', hue='element', bins=50, stat='density', common_norm=False, multiple='layer')
plt.xlabel('Max activation')
plt.ylabel('Density')

# put legend outside of the plot
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

plt.savefig(f'data/{layer_name}/activations_CA.png', dpi=300, bbox_inches='tight')
plt.savefig(f'data/{layer_name}/activations_CA.pdf', bbox_inches='tight')

plt.close()
plt.figure(figsize=(3, 2))
ax = sns.histplot(data=df_tf, x='activation', hue='element', bins=50, stat='density', common_norm=False, multiple='layer')
plt.xlabel('Max activation')
plt.ylabel('Density')

# put legend outside of the plot
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

plt.savefig(f'data/{layer_name}/activations_TF.png', dpi=300, bbox_inches='tight')
plt.savefig(f'data/{layer_name}/activations_TF.pdf', bbox_inches='tight')

plt.close()
plt.figure(figsize=(3, 2))
ax = sns.histplot(data=df_ls, x='activation', hue='element', bins=50, stat='density', common_norm=False, multiple='layer')
plt.xlabel('Max activation')
plt.ylabel('Density')

# put legend outside of the plot
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

plt.savefig(f'data/{layer_name}/activations_pLS.png', dpi=300, bbox_inches='tight')
plt.savefig(f'data/{layer_name}/activations_pLS.pdf', bbox_inches='tight')

exon_elems = [x for x in df_act['element'].unique() if 'exon' in x]

for exon_el in exon_elems:
    df_ex = df_act[df_act['element']==exon_el]

    plt.close()
    plt.figure(figsize=(3, 2))
    ax = sns.histplot(data=df_ex, x='activation', hue='element', bins=50, stat='density', common_norm=False, multiple='layer')
    plt.xlabel('Max activation')
    plt.ylabel('Density')

    # put legend outside of the plot
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    plt.savefig(f'data/{layer_name}/activations_{exon_el}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'data/{layer_name}/activations_{exon_el}.pdf', bbox_inches='tight')