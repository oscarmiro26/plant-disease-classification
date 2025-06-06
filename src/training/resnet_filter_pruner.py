import os
import io
import argparse
import torch
import torch_pruning as tp
from torch.ao.pruning._experimental.pruner import FPGMPruner
import numpy as np
import pandas as pd

from ..training import config
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
FILTER_RANGE = list(np.arange(0.1, 1.0, 0.1))
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
    model_name="resnet50_pruning",
    base_log_dir=config.LOG_DIR
)

# Set seed
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

def prune_model(model, dummy_inputs, ratio, pruner_type, norm):
    """
    Prune `model` with given `ratio`, `pruner_type` ('norm' or 'fpgm') and `norm` ('l1' or 'l2').
    Returns pruned model and chosen p‐norm.
    """
    if pruner_type == 'norm':
        p = 1 if norm == 'l1' else 2
        pruner = tp.pruner.MagnitudePruner(
            model,
            dummy_inputs,
            importance=tp.pruner.importance.MagnitudeImportance(p=p),
            global_pruning=True,
            pruning_ratio=ratio,
            ignored_layers=[model.fc]
        )
        pruner.step()
        return model
    else:
        pruning_config = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                pruning_config.append({
                    "tensor_fqn": f"{name}.weight",
                    "sparsity_level": ratio
                })
        pruner = FPGMPruner(sparsity_level=ratio)
        pruner.prepare(model, pruning_config)
        pruner.enable_mask_update = True
        pruner.step()
        model = pruner.prune()
        return model

def fine_tune_model(model, train_loader, val_loader, epochs):
    """
    Freeze all layers except classifier, then fine‐tune for `epochs` with early stopping.
    Returns fine‐tuned model.
    """
    # freeze except fc
    for _, param in model.named_parameters(): param.requires_grad = False
    for param in model.fc.parameters(): param.requires_grad = True

    optimizer = get_optimizer(model, lr=LR_CLASSIFIER)
    sched = get_scheduler(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    criterion = get_criterion()

    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        # Gradual unfreezing
        if epoch == UNFREEZE_L4_AT:
            for param in model.layer4.parameters():
                param.requires_grad = True
            optimizer.add_param_group({'params': model.layer4.parameters(), 'lr': LR_LAYER4})
            logger.info(f"Unfroze layer4 at epoch {epoch} with lr={LR_LAYER4}.")
        if epoch == UNFREEZE_L3_AT:
            for param in model.layer3.parameters():
                param.requires_grad = True
            optimizer.add_param_group({'params': model.layer3.parameters(), 'lr': LR_LAYER3})
            logger.info(f"Unfroze layer3 at epoch {epoch} with lr={LR_LAYER3}.")
        train(model, train_loader, optimizer, criterion)
        val_loss = validate(model, val_loader, criterion)
        sched.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        logger.info(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}")
    return model

def main(args):
    logger.info("Starting ResNet50 Filter Norm Pruning and Fine-Tuning")
    logger.info(f"Pruner type: {args.pruner.upper()}")
    if args.pruner == 'norm':
        logger.info(f"Using norm type: {args.norm.upper()}")
    logger.info(f"Using model: {args.model_name}")
    if args.experiment_mode:
        logger.info("Running in experiment mode with pruning ratios [0.1, 0.9].")
    else:
        logger.info(f"Running in normal mode with pruning ratio {args.sparsity:.2f}.")
    logger.info(f"Number of epochs for fine-tuning: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}, Num workers: {args.num_workers}")
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
        'accuracy_org': [],
        'flops_org': [],
        'params_org': [],
        'size_org': [],
        'latency_org': [],
        'accuracy_after': [],
        'flops_after': [],
        'params_after': [],
        'size_after': [],
        'latency_after': [],
        'accuracy_drop': [],
        'compression_ratio': [],
        'flops_reduction': [],
        'size_reduction': [],
        'latency_reduction': [],
    }
    # Sweep pruning ratios
    if args.experiment_mode:
        logger.info(f"Running in experiment mode with predefined pruning ratios {FILTER_RANGE}.")
        ratios = FILTER_RANGE
    else:
        logger.info("Running in normal mode with a single pruning ratio.")
        ratios = [args.sparsity]
    metrics['sparsity'] = ratios
    model = load_model(args.model_name)
    model_size = os.path.getsize(os.path.join(config.MODELS_DIR, args.model_name))
    # Compute pre-pruning metrics
    logger.info("Evaluating pre-pruning model...")
    latency_before, acc_before = evaluate(model, test_loader, run_dir=RUN_DIR, logger=logger, get_acc=True, report=False, return_latency=True)
    flops_before, params_before = tp.utils.count_ops_and_params(model, dummy_inputs)
    logger.info(f"Evaluation completed.")
    logger.info(f"Pre-pruning test accuracy: {acc_before:.4f}")
    logger.info(f"Pre-pruning FLOPs: {flops_before / 1e9:.2f} GFLOPs")
    logger.info(f"Params: {params_before / 1e6:.2f} M")
    logger.info(f"Pre-pruning model size: {model_size / 1e6:.2f} MB")
    logger.info(f"Pre-pruning latency: {latency_before:.4f} seconds (average over test set)")
    for ratio in ratios:
        logger.info(f"Pruning ratio {ratio:.2f}...")
        logger.info("Pruning model...")
        model = prune_model(model, dummy_inputs, ratio, args.pruner, args.norm)
        logger.info("Model pruning completed.")
        
        # Get pruned model size
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        pruned_model_size = buffer.getbuffer().nbytes

        # fine-tune
        logger.info("Fine-tuning pruned model...")
        if args.fine_tune:
            model = fine_tune_model(model, train_loader, val_loader, args.epochs)
        logger.info("Fine-tuning completed.")

        # Compute post-pruning metrics
        latency_after, acc_after = evaluate(model, test_loader, run_dir=RUN_DIR, logger=logger, get_acc=True, report=False, return_latency=True)
        flops_after, params_after = tp.utils.count_ops_and_params(model, dummy_inputs)
        logger.info(f"Post-pruning test accuracy: {acc_after:.4f}")
        logger.info(f"Post-pruning FLOPs: {flops_after / 1e9:.2f} GFLOPs")
        logger.info(f"Params: {params_after / 1e6:.2f} M")
        logger.info(f"Pruned model size: {pruned_model_size / 1e6:.2f} MB")
        logger.info(f"Post-pruning latency: {latency_after:.4f} seconds (average over test set)")

        # Comparison metrics
        accuracy_diff = (acc_before - acc_after) / acc_before * 100
        compression_ratio = params_before / params_after * 100
        flops_reduction = (flops_before - flops_after) / flops_before * 100
        size_diff = (model_size - pruned_model_size) / model_size * 100
        latency_diff = (latency_before - latency_after) / latency_before * 100

        # Metrics
        metrics['accuracy_org'].append(acc_before)
        metrics['flops_org'].append(flops_before)  
        metrics['params_org'].append(params_before) 
        metrics['size_org'].append(model_size)  
        metrics['latency_org'].append(latency_before)
        metrics['accuracy_after'].append(acc_after)
        metrics['flops_after'].append(flops_after)  
        metrics['params_after'].append(params_after)  
        metrics['size_after'].append(pruned_model_size)  
        metrics['latency_after'].append(latency_after)
        metrics['accuracy_drop'].append(accuracy_diff) 
        metrics['compression_ratio'].append(compression_ratio)
        metrics['flops_reduction'].append(flops_reduction)
        metrics['size_reduction'].append(size_diff)
        metrics['latency_reduction'].append(latency_diff)

        if args.save_model:
            if args.pruner == 'norm':
                p = 1 if args.norm == 'l1' else 2
                logger.info(f"Saving model with pruning ratio {ratio:.2f} and L{p} Norm")
                save_path = os.path.join(config.MODELS_DIR, f'norm_pruned_resnet50_{ratio}_l{p}.pth')
            else:
                logger.info(f"Saving FPGM pruned model with ratio {ratio:.2f}")
                save_path = os.path.join(config.MODELS_DIR, f'fpgm_pruned_resnet50_{ratio}.pth')
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model saved at {save_path}")
        # Reload model for each ratio
        model = load_model(args.model_name)  

    # Save metrics to CSV
    data = pd.DataFrame(metrics)
    pruning_metrics_path = os.path.join(RUN_DIR, f'pruning_metrics.csv')
    data.to_csv((pruning_metrics_path), index=False)
    logger.info(f"Pruning metrics saved to {pruning_metrics_path}")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNet50 L1 Filter Pruning and Fine-Tuning")
    # Model and data
    parser.add_argument('--model_name', type=str, default='resnet50_9897.pth', help='Name of the pre-trained model (default: "resnet50_9897.pth")')
    parser.add_argument('--data_dir', type=str, default=config.RAW_DATA_DIR, help='Directory for dataset (default: config.RAW_DATA_DIR)')
    parser.add_argument('--save_model', action='store_true',
                        help='Whether to save the final pruned model')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers (default: 4)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and validation (default: 64)')
    
    # Experiment
    parser.add_argument('--experiment_mode', action='store_true',
                        help='Run in experiment mode. Configured in launch_experiment.sh')

    # Config
    parser.add_argument('--pruner', type=str, default='fpgm', choices=['fpgm', 'norm'], help='Pruner type (default: "fpgm")')
    parser.add_argument('--norm', type=str, default='l1', choices=['l1', 'l2'], help='Norm type for norm pruning (default: "l1")')
    parser.add_argument('--sparsity', type=float, default=0.5, help='Pruning sparsity level (default: 0.5)')
    parser.add_argument('--fine_tune', action='store_true',
                        help='Whether to fine-tune the model after pruning')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for fine-tuning (default: 50)')

    args = parser.parse_args()
    main(args)