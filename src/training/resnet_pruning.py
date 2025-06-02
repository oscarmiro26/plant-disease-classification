import os
import torch
import torch.nn as nn
import torch_pruning as tp
from torch_pruning.importance import MagnitudeImportance
import optuna
import pandas as pd
import numpy as np
from torchvision.models import resnet50
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from balanced_loss import Loss 
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    precision_recall_fscore_support
)
from datetime import datetime

from ..training import config
from ..data.sampler import create_sampler
from ..data.splitting import create_splits
from ..data.datasets import PlantDiseaseDataset
from ..utils.logger import setup_logger


# Pruning Hyperparameters
TRIAL_EPOCHS = 10
PRUNING_EPOCHS = 50
BASE_LR = 0.001
LR_DECAY = 0.1
LR_DECAY_EPOCHS = [3, 6, 9]
N_TRIALS = 20
RANDOM_SEED = 42
NUM_WORKERS = 16 
# Filter Norm Pruning Range
FILTER_NORM_RANGE = [0.10, 0.30, 0.05]  # min, max, step
# FPGM Pruning Range 
FPGM_RANGE = []

## Fine-Tuning Hyperparameters
BATCH_SIZE        = 64
NUM_EPOCHS        = 50
NUM_WORKERS       = 16
# Learning rates
LR_CLASSIFIER     = 1e-3
LR_LAYER4         = 1e-4
LR_LAYER3         = 1e-5
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

def load_model():
    # Load model
    base = resnet50(pretrained=False)
    num_classes = len(config.INV_LABEL_MAP)
    num_ftrs = base.fc.in_features
    base.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    resnet_model_path = os.path.join(config.MODELS_DIR, "resnet50_9897.pth")
    state = torch.load(resnet_model_path, map_location="cpu")
    base.load_state_dict(state)
    model = base.eval()
    model.to(config.DEVICE)
    logger.info(f"Model loaded fro  m {resnet_model_path}")
    return model

def split_data():
    # Create train/validation/test splits
    logger.info("Splitting data...")
    train_df, val_df, test_df = create_splits(
        data_dir=config.DATA_DIR,
        label_map=config.LABEL_MAP,
        test_size=config.TEST_SPLIT_SIZE,
        val_size=config.VALIDATION_SPLIT_SIZE,
        random_seed=RANDOM_SEED
    )
    return train_df, val_df, test_df

def samples_per_class():
    """Calculate number of samples per class in the training set."""
    train_df, _, _ = split_data()
    return [train_df['label'].tolist().count(label) for label in config.LABEL_MAP.keys()]

def load_data():
    """Load data and create DataLoaders for training, validation, and testing."""
    # Split data
    train_df, val_df, test_df = split_data()

    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_ds = PlantDiseaseDataset(train_df, transform=train_transform)
    val_ds   = PlantDiseaseDataset(val_df, transform=val_transform)
    test_ds   = PlantDiseaseDataset(test_df, transform=val_transform)

    # Create sampler
    train_sampler = create_sampler(
        train_df, 
        config.LABEL_MAP,
        use_mixing_sampler=True,
        alpha=MIX_ALPHA
    )

    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader, test_loader

def train(model, train_loader, optimizer, criterion):
    """Train the model during one epoch."""
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_train_loss = running_loss / len(train_loader.dataset)
    return epoch_train_loss

def validate(model, val_loader, criterion):
    """Validate the model during one epoch."""
    model.eval()
    running_val = 0.0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_val += loss.item() * imgs.size(0)
    epoch_val_loss = running_val / len(val_loader.dataset)
    return epoch_val_loss

def evaluate(model, test_loader):
    """Evaluate the model on the test set."""
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(config.DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds,
                                   target_names=[config.INV_LABEL_MAP[i] for i in range(config.NUM_CLASSES)], digits=4)
    logger.info("Confusion Matrix:\n%s", cm)
    logger.info("Classification Report:\n%s", report)

    # Save metrics CSV
    prec, rec, f1, sup = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(config.NUM_CLASSES)), zero_division=0)
    metrics_df = pd.DataFrame({
        'class': [config.INV_LABEL_MAP[i] for i in range(config.NUM_CLASSES)],
        'precision': prec, 'recall': rec, 'f1-score': f1, 'support': sup
    })
    metrics_df.to_csv(os.path.join(RUN_DIR, 'metrics.csv'), index=False)

# ==================== Optuna Objective Function ====================

def objective(trial):
    """Optuna objective function for structured pruning"""
    logger.info(f"Starting trial {trial.number}")
    model = load_model()
    train_loader, val_loader, _ = load_data()
    optimizer = torch.optim.AdamW([
        {'params': model.fc.parameters(), 'lr': LR_CLASSIFIER},
    ], weight_decay=1e-4)
    samples_per_class = samples_per_class()
    criterion = Loss(
        loss_type="focal_loss",
        samples_per_class=samples_per_class,
        class_balanced=True
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    # Pre-pruning validation loss
    logger.info("Computing pre-pruning validation loss...")
    original_val_loss = validate(model, val_loader, criterion)

    # Calculate pre-pruning metrics
    example_inputs = torch.randn(1, 3, 224, 224).to(config.DEVICE)
    params_before = sum(p.numel() for p in model.parameters())
    flops_before = tp.utils.count_ops_and_params(model, example_inputs)[0]
    
    # Pruning hyperparameters
    prune_ratio = trial.suggest_float('prune_ratio', FILTER_NORM_RANGE[0], FILTER_NORM_RANGE[1], step=FILTER_NORM_RANGE[2])
    norm_degree = trial.suggest_categorical('norm_degree', [1, 2])
        
    # Pruning
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=MagnitudeImportance(p=norm_degree),
        global_pruning=True,
        prune_ratio=prune_ratio,
        ignored_layers=[model.fc]  # Don't prune final classifier
    )
    pruner.step()
    
    # Post-pruning fine-tuning (freezing layers)
    for _, param in model.named_parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    model = model.to(config.DEVICE)
    best_val_loss = float('inf')

    # Best loss tracking
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    # Train and validate for TRIAL_EPOCHS
    logger.info(f"Starting training for {TRIAL_EPOCHS} epochs with pruning ratio {prune_ratio:.2f} and L{norm_degree} norm.")
    for epoch in range(TRIAL_EPOCHS):
        # Gradual unfreezing (only for L4 since L3 happens at the end)
        if epoch == UNFREEZE_L4_AT:
            for param in model.layer4.parameters():
                param.requires_grad = True
            optimizer.add_param_group({'params': model.layer4.parameters(), 'lr': LR_LAYER4})
            logger.info(f"Unfroze layer4 at epoch {epoch} with lr={LR_LAYER4}.")

        # Train and validate
        epoch_train_loss = train(model, train_loader, optimizer, criterion)
        epoch_val_loss = validate(model, val_loader)
        # Log losses
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        # Step scheduler
        scheduler.step()
        logger.info(f"Epoch {epoch:02d} | Train: {epoch_train_loss:.4f} | Val: {epoch_val_loss:.4f}")

        # Record best validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
        
        # Report to Optuna
        trial.report(epoch_val_loss, epoch)
        
        # Check for pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Calculate post-pruning metrics
    params_after = sum(p.numel() for p in model.parameters())
    flops_after = tp.utils.count_ops_and_params(model, example_inputs)[0]

    # Calculate reductions
    compression_ratio = params_before / params_after
    flops_reduction = (flops_before - flops_after) / flops_before
    
    # Store important metrics
    trial.set_user_attr('original_val_loss', original_val_loss)
    trial.set_user_attr('best_val_loss', best_val_loss)
    trial.set_user_attr('compression_ratio', compression_ratio)
    trial.set_user_attr('flops_reduction', flops_reduction)
    
    return best_val_loss

# ==================== Main Execution ==================== #

if __name__ == "__main__":

    # ===================== Optuna ==================== >

    # Create study
    study_name = f"resnet_pruning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=2,
            interval_steps=1
        )
    )

    optuna.logging.set_verbosity(optuna.logging.INFO)
    # Run optimization
    print(f"\nStarting Optuna optimization with {N_TRIALS} trials...")
    print(f"Using device: {config.DEVICE}")
    
    study.optimize(objective, n_trials=N_TRIALS)
    
    # ==================== Results Summary ==================== >
    
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE")
    print("="*50)
    
    # Best trial results
    trial = study.best_trial
    print(f"\nBest trial: {trial.number}")
    print(f"Best validation loss: {trial.value:.4f}")
    print(f"Parameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
    
    print(f"\nMetrics:")
    print(f"  Original Validation Loss: {trial.user_attrs['original_val_loss']:.4f}")
    print(f"  Best Validation Loss: {trial.user_attrs['best_val_loss']:.4f}")
    print(f"  Compression Ratio: {trial.user_attrs['compression_ratio']:.1%}")
    print(f"  FLOPs Reduction: {trial.user_attrs['flops_reduction']:.1%}")
    
    # ==================== Save Best Model ==================== >
    
    print("\n" + "="*50)
    print("CREATING AND SAVING BEST MODEL")
    print("="*50)
    
    # Recreate best model
    best_model = load_model()
    train_loader, val_loader, test_loader = load_data()
    optimizer = torch.optim.AdamW([
        {'params': best_model.fc.parameters(), 'lr': LR_CLASSIFIER},
    ], weight_decay=1e-4)
    samples_per_class = samples_per_class()
    criterion = Loss(
        loss_type="focal_loss",
        samples_per_class=samples_per_class,
        class_balanced=True
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    example_inputs = torch.randn(1, 3, 224, 224).to(config.DEVICE)
    
    # Apply best pruning configuration
    best_prune_ratio = trial.params['prune_ratio']
    best_norm_degree = trial.params['norm_degree']
    
    if best_norm_degree == 1:
        importance = MagnitudeImportance(p=1)
    else:
        importance = MagnitudeImportance(p=2)
    
    pruner = tp.pruner.MagnitudePruner(
        best_model,
        example_inputs=example_inputs,
        importance=importance,
        global_pruning=True,
        prune_ratio=best_prune_ratio,
        ignored_layers=[best_model.fc]
    )
    pruner.step()

    # ==================== Fine-Tune and Save ==================== >    

    logger.info(f"Applying best pruning ratio {best_prune_ratio:.2f} and L{best_norm_degree}.")
    for _, param in best_model.named_parameters():
        param.requires_grad = False
    for param in best_model.fc.parameters():
        param.requires_grad = True

    # Best loss tracking
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    # Save initial model state
    checkpoint_path = os.path.join(config.MODELS_DIR, f'pruned_resnet50_initial.pth')

    # Train and validate for PRUNING_EPOCHS
    logger.info(f"Starting post-pruning fine-tuning for {PRUNING_EPOCHS} epochs with pruning ratio {best_prune_ratio:.2f} and L{best_norm_degree} norm.")
    for epoch in range(PRUNING_EPOCHS):
        # Gradual unfreezing
        if epoch == UNFREEZE_L4_AT:
            for param in best_model.layer4.parameters():
                param.requires_grad = True
            optimizer.add_param_group({'params': best_model.layer4.parameters(), 'lr': LR_LAYER4})
            logger.info(f"Unfroze layer4 at epoch {epoch} with lr={LR_LAYER4}.")
        if epoch == UNFREEZE_L3_AT:
            for param in best_model.layer3.parameters():
                param.requires_grad = True
            optimizer.add_param_group({'params': best_model.layer3.parameters(), 'lr': LR_LAYER3})
            logger.info(f"Unfroze layer3 at epoch {epoch} with lr={LR_LAYER3}.")

        # Train and validate
        epoch_train_loss = train(best_model, train_loader, optimizer, criterion)
        epoch_val_loss = validate(best_model, val_loader)
        # Log losses
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        # Step scheduler
        scheduler.step()
        logger.info(f"Epoch {epoch:02d} | Train: {epoch_train_loss:.4f} | Val: {epoch_val_loss:.4f}")

        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping triggered at epoch {epoch} (no improvement for {PATIENCE} epochs).")
                break
    
    # Report final validation loss
    logger.info(f"Final validation loss after pruning and fine-tuning: {best_val_loss:.4f}")

    # Evaluate the final model
    evaluate(best_model, test_loader)
    
    # Save final model
    final_model_path = os.path.join(config.MODELS_DIR, 'pruned_resnet50_final.pth')
    torch.save(best_model.state_dict(), final_model_path)
    logger.info(f"Final pruned model saved to {final_model_path}")
    