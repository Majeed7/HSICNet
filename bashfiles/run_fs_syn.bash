#!/bin/bash

# SLURM directives (optional, for cluster usage)
#SBATCH --time=72:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=defq
#SBATCH --gres=gpu:1

# Change directory to the home folder
cd ~

# Load CUDA 12.1
module unload all
module load cuda12.1/toolkit/12.1

# Activate the Python environment
source /var/scratch/mmi454/envs/hsic_env/bin/activate

# Print system info
echo "Python version:"
python --version
echo "CUDA version:"
nvcc --version

# Run the Python script
echo "Running Python script..."
python ~/HSIC/HSICNet/main_synthetic_fs.py

# Print CUDA availability again after the script runs
echo "Rechecking CUDA availability in PyTorch:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Deactivate the environment
deactivate