import os
import subprocess


model_folders = ["conv1d_2_noabs_16_topk0.05_lr0.0001",
"conv1d_2_noabs_16_topk0.05_lr1e-05",
"conv1d_2_noabs_16_topk0.05_lr1e-06",
"conv1d_2_noabs_16_topk0.1_lr0.0001",
"conv1d_2_noabs_16_topk0.1_lr1e-05",
"conv1d_2_noabs_16_topk0.1_lr1e-06",
"conv1d_2_noabs_16_topk0.2_lr0.0001",
"conv1d_2_noabs_16_topk0.2_lr1e-05",
"conv1d_2_noabs_16_topk0.2_lr1e-06",
"conv1d_2_noabs_8_topk0.05_lr0.0001",
"conv1d_2_noabs_8_topk0.05_lr1e-05",
"conv1d_2_noabs_8_topk0.05_lr1e-06",
"conv1d_2_noabs_8_topk0.1_lr0.0001",
"conv1d_2_noabs_8_topk0.1_lr1e-05",
"conv1d_2_noabs_8_topk0.1_lr1e-06",
"conv1d_2_noabs_8_topk0.2_lr0.0001",
"conv1d_2_noabs_8_topk0.2_lr1e-05",
"conv1d_2_noabs_8_topk0.2_lr1e-06"]

if not os.path.exists('temp'):
    os.makedirs('temp')

for im,model_folder in enumerate(model_folders):
    model_path = f"/home/anya/code/sae_borzoi/models/{model_folder}"

    slurm_string = '#!/bin/bash \n \n'
    slurm_string += '#SBATCH -p cpu \n \n'
    slurm_string += '#SBATCH -n 1 \n#SBATCH -c 2 \n#SBATCH -J meme \n'
    slurm_string += f"#SBATCH -o /home/anya/code/sae_borzoi/models/{model_folder}/job0_infer.out \n"
    slurm_string += f"#SBATCH -e /home/anya/code/sae_borzoi/models/{model_folder}/job0_infer.err \n"
    slurm_string += '#SBATCH --mem 22000 \n#SBATCH --time 2-0:0:0 \n'
    slurm_string += f'source /home/anya/.bashrc; echo $HOSTNAME; python meme-analysis.sh /home/anya/code/sae_borzoi/models/{model_folder}'

    with open(f'temp/job_{im}.sb', 'w') as f:
        f.write(slurm_string)

    # run with subprocess
    subprocess.run(['sbatch', f'temp/job_{im}.sb'])
    