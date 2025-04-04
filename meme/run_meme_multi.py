import os
import subprocess


cwd = os.getcwd()

model_folders = os.listdir(os.path.join(cwd, 'models'))
model_folders = [folder for folder in model_folders if 'noabs' in folder]
model_folders = [folder for folder in model_folders if '.csv' not in folder]

if not os.path.exists('meme/temp'):
    os.makedirs('meme/temp')

for im,model_folder in enumerate(model_folders):
    model_path = os.path.join(cwd, 'models', model_folder)

    slurm_string = '#!/bin/bash \n \n'
    slurm_string += '#SBATCH -p standard \n \n'
    slurm_string += '#SBATCH -n 1 \n#SBATCH -c 2 \n#SBATCH -J meme \n'
    slurm_string += f"#SBATCH -o temp/job_{im}.out \n"
    slurm_string += f"#SBATCH -e temp/job_{im}.err \n"
    slurm_string += '#SBATCH --mem 22000 \n#SBATCH --time 2-0:0:0 \n'
    bash_script = os.path.join(cwd, 'meme/meme-analysis.sh')
    slurm_string += f"source /home/anya/.bashrc; echo $HOSTNAME; bash {bash_script} {model_path}"

    with open(os.path.join(cwd, f'meme/temp/job_{im}.sb'), 'w') as f:
        f.write(slurm_string)

    # run with subprocess
    subprocess.run(['sbatch', f'meme/temp/job_{im}.sb'])
    