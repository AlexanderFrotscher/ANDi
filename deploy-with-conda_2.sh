#!/bin/bash
#SBATCH --job-name=BraTS-eval
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=16         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-01:30            # Runtime in D-HH:MM
#SBATCH --partition=a100 # Partition to submit to
#SBATCH --gres=gpu:1              # optionally type and number of gpus
#SBATCH --mem=32GB                 # Memory pool for all cores (see also --mem-per-cpu)

# print info about current job
echo "---------- JOB INFOS ------------"
scontrol show job $SLURM_JOB_ID
echo -e "---------------------------------\n"

# Due to a potential bug, we need to manually load our bash configurations first
source $HOME/.bashrc

# Next activate the conda environment
conda activate myenv

# Run our code
echo "-------- PYTHON OUTPUT ----------"
srun -n 1 -N 1 -c 16 --mem=32GB --partition=a100 --time=0-01:30 --gres=gpu:1 accelerate launch eval_3d.py
echo "---------------------------------"

# Deactivate environment again
conda deactivate
