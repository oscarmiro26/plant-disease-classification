#!/bin/bash --login
#SBATCH --job-name=all_models_train
#SBATCH --output=all_models_train_%j.out
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:1

cd /home3/$USER/plant-disease-classification

# load conda
eval "$(conda shell.bash hook)"
conda activate plant-disease

# list of training modules to run
MODELS=(
  train_resnet18
  train_resnet34
  train_resnet50
  train_resnet101
  train_resnet152
  train_efficientnet_b0
)

for m in "${MODELS[@]}"; do
  echo "=== Starting $m ==="
  srun python -u -m src.training.$m
  if [ $? -ne 0 ]; then
    echo "!!! $m failed, exiting job."
    exit 1
  fi
  echo "=== Finished $m ==="
done

echo "All trainings completed successfully."
