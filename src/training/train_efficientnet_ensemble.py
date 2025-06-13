import os
import sys
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)

from ..data.splitting import create_splits
from ..data.datasets import PlantDiseaseDataset
from ..training import config
from ..data.sampler import calculate_class_weights, create_sampler
from ..utils.logger import setup_logger

BATCH_SIZE = 32
NUM_EPOCHS = 100
NUM_WORKERS = 4
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4

PATIENCE = 15

VERBOSE_LOGGING = True

logger, RUN_DIR = setup_logger(
    model_name="effnet_b0_ensemble_replication", base_log_dir=config.LOG_DIR
)

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
val_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class AdaptiveEnsemble(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.effnet1 = models.efficientnet_b0(weights=weights)
        self.effnet2 = models.efficientnet_b0(weights=weights)

        num_ftrs = self.effnet1.classifier[1].in_features
        self.effnet1.classifier = nn.Identity()
        self.effnet2.classifier = nn.Identity()

        self.ensemble_classifier = nn.Sequential(
            nn.Linear(num_ftrs * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features1 = self.effnet1(x)
        features2 = self.effnet2(x)
        concatenated_features = torch.cat((features1, features2), dim=1)
        output = self.ensemble_classifier(concatenated_features)
        return output


def train_and_evaluate():
    logger.info(
        "Starting REPLICATION training run for EfficientNet_B0 Adaptive Ensemble."
    )

    if VERBOSE_LOGGING:
        script_src = os.path.abspath(__file__)
        script_dst = os.path.join(RUN_DIR, os.path.basename(__file__))
        try:
            shutil.copy(script_src, script_dst)
            logger.info(f"Copied main script to {script_dst}")
        except Exception as e:
            logger.warning(f"Failed to copy script into RUN_DIR: {e}")

    train_df, val_df, test_df = create_splits(
        data_dir=config.RAW_DATA_DIR,
        label_map=config.LABEL_MAP,
        test_size=config.TEST_SPLIT_SIZE,
        val_size=config.VALIDATION_SPLIT_SIZE,
    )

    train_ds = PlantDiseaseDataset(train_df, transform=train_transform)
    val_ds = PlantDiseaseDataset(val_df, transform=val_transform)
    test_ds = PlantDiseaseDataset(test_df, transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    model = AdaptiveEnsemble(num_classes=config.NUM_CLASSES, pretrained=True)
    model = model.to(config.DEVICE)

    criterion = nn.CrossEntropyLoss().to(config.DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-7
    )

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    logger.info("Starting end-to-end fine-tuning with all layers trainable.")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * imgs.size(0)
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch:02d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | LR: {current_lr:.1e}"
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            model_path = os.path.join(RUN_DIR, "effnet_b0_ensemble_best.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(f"Validation loss improved. Saved best model to {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(
                    f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)."
                )
                break

    model_path_final = os.path.join(RUN_DIR, "effnet_b0_ensemble_final.pth")
    torch.save(model.state_dict(), model_path_final)
    logger.info(f"Final model weights saved to {model_path_final}")

    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")
    plt.tight_layout()
    loss_curve_path = os.path.join(RUN_DIR, "loss_curve.png")
    plt.savefig(loss_curve_path)
    plt.close()
    logger.info(f"Loss curve saved to {loss_curve_path}")

    best_model_path = os.path.join(RUN_DIR, "effnet_b0_ensemble_best.pth")
    if os.path.exists(best_model_path):
        logger.info(
            f"Loading best model from {best_model_path} for final test evaluation."
        )
        model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
    else:
        logger.warning("No best model found. Evaluating with the final model state.")

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
    report = classification_report(
        all_labels,
        all_preds,
        target_names=[config.INV_LABEL_MAP[i] for i in range(config.NUM_CLASSES)],
        digits=4,
    )
    logger.info("Test Confusion Matrix:\n%s", cm)
    logger.info("Test Classification Report (on best model):\n%s", report)

    prec, rec, f1, sup = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(config.NUM_CLASSES)), zero_division=0
    )
    metrics_df = pd.DataFrame(
        {
            "class": [config.INV_LABEL_MAP[i] for i in range(config.NUM_CLASSES)],
            "precision": prec,
            "recall": rec,
            "f1-score": f1,
            "support": sup,
        }
    )
    metrics_csv = os.path.join(RUN_DIR, "test_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    logger.info(f"Per-class test metrics saved to {metrics_csv}")


if __name__ == "__main__":
    train_and_evaluate()
