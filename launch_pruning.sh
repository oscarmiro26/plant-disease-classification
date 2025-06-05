#!/bin/bash
#SBATCH --job-name=run_pruning
#SBATCH --output=habrok_logs/output_pruning.log          
#SBATCH --error=habrok_logs/error_pruning.log           
#SBATCH --time=12:00:00               
#SBATCH --nodes=1     
#SBATCH --gpus-per-node=v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G                  

module load Python/3.10.4-GCCcore-11.3.0
source __ # Change to environment direction

python -m src.training.resnet_filter_pruner \
    --pruner fpgm \
    --experiment_mode \
    --experiment_name fpgm_pruning \
    --num_workers 16 \
    --batch_size 64 \
    --fine_tune \
    --epochs 50
    
echo "Job completed successfully."
deactivate