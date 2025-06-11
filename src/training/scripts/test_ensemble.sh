#!/bin/bash --login
#SBATCH --job-name=test_ensemble
#SBATCH --output=test_ensemble_%j.out
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:1

cd /home3/$USER/plant-disease-classification

eval "$(conda shell.bash hook)"
conda activate plant-disease

srun python -u -m src.training.test_ensemble