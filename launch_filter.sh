#!/bin/bash
#SBATCH --job-name=run_pruning_l1
#SBATCH --output=habrok_logs/output_l1.log          
#SBATCH --error=habrok_logs/error_l1.log           
#SBATCH --time=01:00:00               
#SBATCH --nodes=1     
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=8G                  

module load Python/3.10.4-GCCcore-11.3.0
source $HOME/venvs/my_venv/bin/activate
python -m src.training.resnet_filter_norm \
    --save_model \
    --num_workers 32 \
    --batch_size 64 \
    --epochs 50 \
    --norm l1 \
    --sparsity 0.5 
echo "Job completed successfully."
deactivate