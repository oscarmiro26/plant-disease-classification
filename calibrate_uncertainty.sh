#!/bin/bash --login
#SBATCH --job-name=calibrate_resnet
#SBATCH --output=calibrate_resnet_%j.out
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

srun python -u -m src.postprocessing.uncertainty_calibration.py --model-path outputs/models/resnet50_9897.pth