# train_resnet18.py

import os
import sys

if __name__ == "__main__" and __package__ is None:
    __package__ = "training"

import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from balanced_loss import Loss

from ..data.splitting import create_splits
from ..data.datasets import PlantDiseaseDataset
from ..training import config
from ..training.losses import FocalLoss
from ..data.sampler import calculate_class_weights, create_sampler
from ..utils.logger import setup_logger

BATCH_SIZE      = 64
NUM_EPOCHS      = 50
NUM_WORKERS     = 4
LR_CLASSIFIER   = 1e-3
LR_LAYER4       = 1e-4
LR_LAYER3       = 1e-5
UNFREEZE_L4_AT  = 5
UNFREEZE_L3_AT  = 10
STEP_SIZE       = 7
GAMMA           = 0.1
MIX_ALPHA       = 0.7
PATIENCE        = 10
VERBOSE_LOGGING = True

logger, RUN_DIR = setup_logger(
    model_name="resnet18",
    base_log_dir=config.LOG_DIR
)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def train_and_evaluate():
    logger.info("Starting training run.")
    if VERBOSE_LOGGING:
        script_src = os.path.abspath(__file__)
        script_dst = os.path.join(RUN_DIR, os.path.basename(__file__))
        try:
            import shutil; shutil.copy(script_src, script_dst)
            logger.info(f"Copied main script to {script_dst}")
        except Exception as e:
            logger.warning(f"Failed to copy script: {e}")

    train_df, val_df, test_df = create_splits(
        data_dir   = config.RAW_DATA_DIR,
        label_map  = config.LABEL_MAP,
        test_size  = config.TEST_SPLIT_SIZE,
        val_size   = config.VALIDATION_SPLIT_SIZE,
    )

    class_weights = calculate_class_weights(train_df, config.LABEL_MAP).to(config.DEVICE)
    train_sampler = create_sampler(train_df, config.LABEL_MAP, use_mixing_sampler=True, alpha=MIX_ALPHA)

    train_ds = PlantDiseaseDataset(train_df, transform=train_transform)
    val_ds   = PlantDiseaseDataset(val_df,   transform=val_transform)
    test_ds  = PlantDiseaseDataset(test_df,  transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    for p in model.parameters(): p.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, config.NUM_CLASSES)
    )
    for p in model.fc.parameters(): p.requires_grad = True

    model = model.to(config.DEVICE)

    samples_per_class = [train_df['label'].tolist().count(lbl) for lbl in config.LABEL_MAP]
    criterion = Loss("focal_loss", samples_per_class, class_balanced=False).to(config.DEVICE)

    optimizer = torch.optim.AdamW(
        [{'params': model.fc.parameters(), 'lr': LR_CLASSIFIER}],
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, STEP_SIZE, GAMMA)

    best_val_loss, patience_counter = float('inf'), 0
    train_losses, val_losses = [], []

    for epoch in range(1, NUM_EPOCHS+1):
        if epoch == UNFREEZE_L4_AT:
            for p in model.layer4.parameters(): p.requires_grad = True
            optimizer.add_param_group({'params': model.layer4.parameters(), 'lr': LR_LAYER4})
            logger.info(f"Unfroze layer4 at epoch {epoch}")
        if epoch == UNFREEZE_L3_AT:
            for p in model.layer3.parameters(): p.requires_grad = True
            optimizer.add_param_group({'params': model.layer3.parameters(), 'lr': LR_LAYER3})
            logger.info(f"Unfroze layer3 at epoch {epoch}")

        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward(); optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
                running_val += criterion(model(imgs), labels).item() * imgs.size(0)
        val_loss = running_val / len(val_loader.dataset)
        val_losses.append(val_loss)

        scheduler.step()
        logger.info(f"Epoch {epoch}: Train {train_loss:.4f} | Val {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss, patience_counter = val_loss, 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    torch.save(model.state_dict(), os.path.join(RUN_DIR, "resnet18.pth"))
    plt.figure()
    plt.plot(range(1,len(train_losses)+1), train_losses, label="Train")
    plt.plot(range(1,len(val_losses)+1),   val_losses,   label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "loss_curve.png")); plt.close()

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(config.DEVICE)
            preds = model(imgs).argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds,
                                   target_names=[config.INV_LABEL_MAP[i] for i in range(config.NUM_CLASSES)],
                                   digits=4)
    logger.info("Test Confusion Matrix:\n%s", cm)
    logger.info("Test Classification Report:\n%s", report)

    prec, rec, f1, sup = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(config.NUM_CLASSES)), zero_division=0
    )
    pd.DataFrame({
        "class":     [config.INV_LABEL_MAP[i] for i in range(config.NUM_CLASSES)],
        "precision": prec, "recall": rec, "f1-score": f1, "support": sup
    }).to_csv(os.path.join(RUN_DIR, "metrics.csv"), index=False)

if __name__ == "__main__":
    train_and_evaluate()
