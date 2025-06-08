import os
import sys

# ─── Make relative imports work when running this file directly ─────────────
if __name__ == "__main__" and __package__ is None:
    # Assume this script is in the 'training' package. Adjust if needed.
    __package__ = "training"

import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from balanced_loss import Loss  # see: https://github.com/fcakyon/balanced-loss

from ..data.splitting import create_splits
from ..data.datasets import PlantDiseaseDataset
from ..training import config
from ..training.losses import FocalLoss
from ..data.sampler import calculate_class_weights, create_sampler
from ..utils.logger import setup_logger

# ─── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE        = 64
NUM_EPOCHS        = 50
NUM_WORKERS       = 4

# Learning rates
LR_CLASSIFIER     = 1e-3
LR_LAYER4         = 1e-4
LR_LAYER3         = 1e-5

# “Unfreeze” schedule
UNFREEZE_L4_AT    = 5
UNFREEZE_L3_AT    = 10

# LR scheduler
STEP_SIZE         = 7
GAMMA             = 0.1

# Mixing sampler
MIX_ALPHA         = 0.7
PATIENCE          = 10

VERBOSE_LOGGING   = True

# ─── Logging setup ─────────────────────────────────────────────────────────────
logger, RUN_DIR = setup_logger(
    model_name="resnet50",
    base_log_dir=config.LOG_DIR
)

# ─── Data transforms ───────────────────────────────────────────────────────────
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
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def train_and_evaluate():
    logger.info("Starting training run.")

    if VERBOSE_LOGGING:
        # Copy this script into the run directory for reproducibility
        script_src = os.path.abspath(__file__)
        script_dst = os.path.join(RUN_DIR, os.path.basename(__file__))
        try:
            import shutil
            shutil.copy(script_src, script_dst)
            logger.info(f"Copied main script to {script_dst}")
        except Exception as e:
            logger.warning(f"Failed to copy script into RUN_DIR: {e}")

    # ─── 1. Create train/val/test splits ────────────────────────────────────────
    train_df, val_df, test_df = create_splits(
        data_dir   = config.RAW_DATA_DIR,
        label_map  = config.LABEL_MAP,
        test_size  = config.TEST_SPLIT_SIZE,
        val_size   = config.VALIDATION_SPLIT_SIZE,
    )

    # ─── 2. Compute class weights & sampler ─────────────────────────────────────
    class_weights = calculate_class_weights(train_df, config.LABEL_MAP).to(config.DEVICE)
    train_sampler = create_sampler(
        train_df,
        config.LABEL_MAP,
        use_mixing_sampler=True,
        alpha=MIX_ALPHA
    )

    # ─── 3. Datasets & DataLoaders ─────────────────────────────────────────────
    train_ds  = PlantDiseaseDataset(train_df,  transform=train_transform)
    val_ds    = PlantDiseaseDataset(val_df,    transform=val_transform)
    test_ds   = PlantDiseaseDataset(test_df,   transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size = BATCH_SIZE,
        sampler    = train_sampler,
        num_workers= NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds,
        batch_size = BATCH_SIZE,
        shuffle    = False,
        num_workers= NUM_WORKERS
    )
    test_loader = DataLoader(
        test_ds,
        batch_size = BATCH_SIZE,
        shuffle    = False,
        num_workers= NUM_WORKERS
    )

    # ─── 4. Build ResNet50 & replace head ───────────────────────────────────────
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, config.NUM_CLASSES)
    )
    # Only the head’s weights are trainable to start
    for param in model.fc.parameters():
        param.requires_grad = True

    model = model.to(config.DEVICE)

    # ─── 5. Loss, optimizer, scheduler ──────────────────────────────────────────
    # Balanced focal loss
    samples_per_class = [train_df['label'].tolist().count(lbl) for lbl in config.LABEL_MAP.keys()]
    criterion = Loss(
        loss_type="focal_loss",
        samples_per_class=samples_per_class,
        class_balanced=True
    ).to(config.DEVICE)

    # Start optimizer with only the head parameters
    optimizer = torch.optim.AdamW(
        [{'params': model.fc.parameters(), 'lr': LR_CLASSIFIER}],
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    best_val_loss    = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        # ─── 5.1 Gradual unfreezing ─────────────────────────────────────────────
        if epoch == UNFREEZE_L4_AT:
            # Unfreeze layer4
            for param in model.layer4.parameters():
                param.requires_grad = True
            optimizer.add_param_group({
                'params': model.layer4.parameters(),
                'lr':      LR_LAYER4
            })
            logger.info(f"Unfroze layer4 at epoch {epoch} with lr={LR_LAYER4}")

        if epoch == UNFREEZE_L3_AT:
            # Unfreeze layer3
            for param in model.layer3.parameters():
                param.requires_grad = True
            optimizer.add_param_group({
                'params': model.layer3.parameters(),
                'lr':      LR_LAYER3
            })
            logger.info(f"Unfroze layer3 at epoch {epoch} with lr={LR_LAYER3}")

        # ─── 6. Training epoch ─────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs   = imgs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # ─── 7. Validation ──────────────────────────────────────────────────────
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(config.DEVICE)
                labels = labels.to(config.DEVICE)
                outputs = model(imgs)
                loss    = criterion(outputs, labels)
                running_val += loss.item() * imgs.size(0)

        epoch_val_loss = running_val / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        scheduler.step()
        logger.info(f"Epoch {epoch:02d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        # ─── 8. Early stopping ─────────────────────────────────────────────────
        if epoch_val_loss < best_val_loss:
            best_val_loss    = epoch_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs).")
                break

    # ─── 9. Save final model + loss curve ─────────────────────────────────
    model_path = os.path.join(RUN_DIR, "resnet50.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model weights saved to {model_path}")

    # Plot and save the loss curves
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train")
    plt.plot(range(1, len(val_losses)   + 1), val_losses,   label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    loss_curve_path = os.path.join(RUN_DIR, "loss_curve.png")
    plt.savefig(loss_curve_path)
    plt.close()
    logger.info(f"Loss curve saved to {loss_curve_path}")

    # ─── 10. Final evaluation on test set ────────────────────────────────────
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(config.DEVICE)
            outputs = model(imgs)
            preds   = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels,
        all_preds,
        target_names=[config.INV_LABEL_MAP[i] for i in range(config.NUM_CLASSES)],
        digits=4
    )
    logger.info("Test Confusion Matrix:\n%s", cm)
    logger.info("Test Classification Report:\n%s", report)

    # Save per‐class metrics to CSV
    prec, rec, f1, sup = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=list(range(config.NUM_CLASSES)),
        zero_division=0
    )
    metrics_df = pd.DataFrame({
        "class":      [config.INV_LABEL_MAP[i] for i in range(config.NUM_CLASSES)],
        "precision":  prec,
        "recall":     rec,
        "f1-score":   f1,
        "support":    sup
    })
    metrics_csv = os.path.join(RUN_DIR, "metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    logger.info(f"Per-class metrics saved to {metrics_csv}")


if __name__ == "__main__":
    train_and_evaluate()
