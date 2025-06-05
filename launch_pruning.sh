#!/bin/bash
#SBATCH --job-name=run_pruning
#SBATCH --output=habrok_logs/output_pruning.log          
#SBATCH --error=habrok_logs/error_pruning.log           
#SBATCH --time=00:30:00               
#SBATCH --nodes=1     
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G                  

module load Python/3.10.4-GCCcore-11.3.0
source $HOME/venvs/my_venv/bin/activate
python -m src.training.resnet_filter_pruner \
    --pruner norm \
    --norm l1 \
    --num_workers 16 \
    --batch_size 64 \
    --sparsity 0.5 \
    --fine_tune \
    --epochs 1
    
echo "Job completed successfully."
deactivate