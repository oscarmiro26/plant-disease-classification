import os
import argparse
import torch
import torch_pruning as tp
from torch_pruning.pruner.importance import MagnitudeImportance
from torch_pruning.utils.op_counter import count_ops_and_params
import numpy as np
import pandas as pd
from time import time

from . import config
from ..utils.logger import setup_logger
from ..utils.resnet_utils import (
    load_model, 
    get_scheduler,
    get_optimizer,
    get_criterion,
    load_data, 
    train, 
    validate, 
    evaluate
)


# Pruning Hyperparameters
FILTER_NORM_RANGE = [0.10, 0.90, 0.10]
RANDOM_SEED = 42

# Learning rates (1/10 of original)
LR_CLASSIFIER     = 1e-4
LR_LAYER4         = 1e-5
LR_LAYER3         = 1e-6
# Unfreeze schedule (epochs to unfreeze)
UNFREEZE_L4_AT    = 5
UNFREEZE_L3_AT    = 10
# LR scheduler
STEP_SIZE         = 7
GAMMA             = 0.1
# Mixing sampler
MIX_ALPHA         = 0.7
# Early stopping patience
PATIENCE          = 10

# Set training config
MODEL = load_model()
OPTIM = get_optimizer(MODEL, lr=LR_CLASSIFIER)
SCHED = get_scheduler(OPTIM, step_size=STEP_SIZE, gamma=GAMMA)
CRITERION = get_criterion()

# Logging setup
logger, RUN_DIR = setup_logger(
    model_name="resnet50_pruning_filter_norm",
    base_log_dir=config.LOG_DIR
)

# Set seed
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

def main(args):
    logger.info("Starting ResNet50 Filter Norm Pruning and Fine-Tuning")
    logger.info(f"Using model: {args.model_name}")
    if args.experiment_mode:
        logger.info("Running in experiment mode with pruning ratios [0.1, 0.9].")
    else:
        logger.info(f"Running in normal mode with pruning ratio {args.sparsity:.2f}.")
    logger.info(f"Using norm type: {args.norm}")
    logger.info(f"Number of epochs for fine-tuning: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}, Num workers: {args.num_workers}")
    logger.info(f"Experiment name: {args.experiment_name}")
    logger.info(f"Run directory: {RUN_DIR}")
    # Load data
    logger.info("Loading data...")
    train_loader, val_loader, test_loader = load_data(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mix_alpha=MIX_ALPHA)
    dummy_inputs = torch.randn(8, 3, 224, 224).to(config.DEVICE)
    metrics = {
        'sparsity': [],
        'accuracy': [],
        'flops': [],
        'params': [],
        'latency': [],
        'accuracy_drop': [],
        'compression_ratio': [],
        'flops_reduction': [],
        'latency_reduction': []
    }
    # Sweep pruning ratios
    if args.experiment_mode:
        logger.info("Running in experiment mode with predefined pruning ratios.")
        ratios = FILTER_NORM_RANGE
    else:
        logger.info("Running in normal mode with a single pruning ratio.")
        ratios = [args.sparsity]
    metrics['sparsity'] = ratios
    for ratio in ratios:
        logger.info(f"Pruning ratio {ratio:.2f} start")
        model = load_model(args.model_name).to(config.DEVICE)

        # Compute pre-pruning metrics
        acc_before = evaluate(model, test_loader, get_acc=True, report=False)
        flops_before, params_before = count_ops_and_params(model, dummy_inputs)
        
        start = time.time()
        for _ in range(32):
            _ = model(dummy_inputs)
        end = time.time()
        latency_before = end - start

        logger.info(f"Pre-pruning test accuracy: {acc_before:.2f}")
        logger.info(f"Pre-pruning FLOPs: {flops_before / 1e9:.2f} GFLOPs, Params: {params_before / 1e6:.2f} M")
        logger.info(f"Pre-pruning latency: {latency_after:.4f} seconds for 32 inferences")

        # Prune model
        p = 1 # Default to L1 norm
        if args.norm == 'l1':
            logger.info("Using L1 norm for pruning.")
            p = 1
        elif args.norm == 'l2':
            logger.info("Using L2 norm for pruning.")
            p = 2
        logger.info(f"Pruning model with ratio {ratio:.2f} and L{p} Norm")
        pruner = tp.pruner.MagnitudePruner(
            model,
            dummy_inputs,
            importance=MagnitudeImportance(p=p),
            global_pruning=True,
            pruning_ratio=ratio,
            ignored_layers=[model.fc]
        )
        pruner.step()

        # freeze except classifier
        for _, p in model.named_parameters(): p.requires_grad = False
        for p in model.fc.parameters(): p.requires_grad = True

        optim = get_optimizer(model, lr=LR_CLASSIFIER)
        sched = get_scheduler(optim, step_size=STEP_SIZE, gamma=GAMMA)
        criterion = get_criterion()

        # fine-tune
        best_val = float('inf')
        logger.info(f"Fine-tuning pruned model with pruning ratio {ratio:.2f} and L{p} Norm")
        for epoch in range(args.epochs):
            train(model, train_loader, optim, criterion)
            epoch_val_loss = validate(model, val_loader, criterion)
            sched.step()

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    logger.info(f"Early stopping triggered at epoch {epoch} (no improvement for {PATIENCE} epochs).")
                    break
            logger.info(f"Epoch {epoch+1}/{args.epochs}, Val Loss: {epoch_val_loss:.4f}")

        logger.info(f"Pruning {ratio:.2f} → best val loss {best_val:.4f}")
        # Compute post-pruning metrics

        report = False if args.experiment_mode else True
        acc_after = evaluate(model, test_loader, get_acc=True, report=report)
        flops_after, params_after = count_ops_and_params(model, dummy_inputs)
        start = time.time()
        for _ in range(32):
            _ = model(dummy_inputs)
        end = time.time()
        latency_after = end - start
        logger.info(f"Post-pruning test accuracy: {acc_after:.2f}")
        logger.info(f"Post-pruning FLOPs: {flops_after / 1e9:.2f} GFLOPs, Params: {params_after / 1e6:.2f} M")
        logger.info(f"Post-pruning latency: {latency_after:.4f} seconds for 32 inferences")

        metrics['accuracy'].append(acc_after)
        metrics['flops'].append(flops_after)
        metrics['params'].append(params_after)
        metrics['latency'].append(latency_after)

        # Final metrics
        accuracy_diff = (acc_before - acc_after) / acc_before * 100
        compression_ratio = params_before / params_after
        flops_reduction = (flops_before - flops_after) / flops_before * 100
        latency_diff = (latency_before - latency_after) / latency_before * 100
        logger.info(f"Accuracy drop: {accuracy_diff:.2f}%")
        logger.info(f"Compression ratio: {compression_ratio:.2f}x")
        logger.info(f"FLOPs reduction: {flops_reduction:.2f}%")
        logger.info(f"Latency reduction: {latency_diff:.2f}%")
        metrics['accuracy_drop'].append(accuracy_diff)
        metrics['compression_ratio'].append(compression_ratio)
        metrics['flops_reduction'].append(flops_reduction)
        metrics['latency_reduction'].append(latency_diff)

        if args.save_model:
            logger.info(f"Saving model with pruning ratio {ratio:.2f} and L{p} Norm")
            save_path = os.path.join(config.MODELS_DIR, f'pruned_resnet50_{ratio}_l{p}.pth')
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved best model at ratio {ratio:.2f} to {save_path}")
    
    if args.experiment_mode:
        data = pd.DataFrame(metrics)
        exp_metrics_path = os.path.join(RUN_DIR, f'{args.experiment_name}_metrics.csv')
        data.to_csv((exp_metrics_path), index=False)
        logger.info(f"Experiment completed. Pruning metrics saved to {exp_metrics_path}")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNet50 L1 Filter Pruning and Fine-Tuning")
    # Model and data loading
    parser.add_argument('--model_name', type=str, default='resnet50_9897.pth', help='Name of the pre-trained model (default: "resnet50_9897.pth")')
    parser.add_argument('--save_model', action='store_true',
                        help='Whether to save the final pruned model')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers (default: 4)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and validation (default: 64)')
    
    # Experiment
    parser.add_argument('--experiment_mode', action='store_true',
                        help='Run in experiment mode. Configured in launch_experiment.sh')
    parser.add_argument('--experiment_name', type=str, default='resnet50_fpgm_experiment',
                        help='Name of the experiment (default: "resnet50_fpgm_experiment")')

    # Config
    parser.add_argument('--norm', type=str, default='l1', choices=['l1', 'l2'], help='Norm type for pruning (default: "l1")')
    parser.add_argument('--sparsity', type=float, default=0.5, help='Pruning sparsity level (default: 0.5)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for fine-tuning (default: 50)')

    args = parser.parse_args()

    main(args)