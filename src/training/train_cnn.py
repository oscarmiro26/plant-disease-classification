import os
import sys
import logging

import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

from ..data.splitting import create_splits
from ..data.datasets import PlantDiseaseDataset
from ..data.preprocessing import train_transforms, val_test_transforms
from ..data.sampler import calculate_class_weights, create_sampler
from ..models.cnn import CNN
from ..training import config
from ..training.losses import FocalLoss


BATCH_SIZE     = 128
NUM_EPOCHS     = 50
NUM_WORKERS    = 4
LEARNING_RATE  = 1e-3
STEP_SIZE      = 10
GAMMA          = 0.8
MIX_ALPHA      = 0.7
PATIENCE       = 10


# Set up logging
LOG_BASE = os.path.join(config.LOG_DIR, "cnn")
os.makedirs(LOG_BASE, exist_ok=True)
existing = [d for d in os.listdir(LOG_BASE) if os.path.isdir(os.path.join(LOG_BASE, d))]
run_idx = len(existing)
run_name = f"CNN_{run_idx}"
RUN_DIR = os.path.join(LOG_BASE, run_name)
os.makedirs(RUN_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(RUN_DIR, "train.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('train')
print(f"Logging to {RUN_DIR}")

def train_and_evaluate():
    logger.info(f"Starting run {run_name}.")
    # Creating data splits
    train_df, val_df, test_df = create_splits(
        data_dir    = config.RAW_DATA_DIR,
        label_map   = config.LABEL_MAP,
        test_size   = config.TEST_SPLIT_SIZE,
        val_size    = config.VALIDATION_SPLIT_SIZE,
    )

    # Calculating class weights
    class_weights = calculate_class_weights(train_df, config.LABEL_MAP).to(config.DEVICE)
    train_sampler = create_sampler(
        train_df, 
        config.LABEL_MAP,
        use_mixing_sampler=True,
        alpha=MIX_ALPHA
    )

    # Creating datasets
    train_ds = PlantDiseaseDataset(train_df, transform=train_transforms)
    val_ds   = PlantDiseaseDataset(val_df,   transform=val_test_transforms)
    test_ds  = PlantDiseaseDataset(test_df,  transform=val_test_transforms)

    # Creating data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # Creating model, criterion, optimizer, scheduler
    model = CNN(
        num_classes=config.NUM_CLASSES,
        base_channels=64,
        dropout=0.5
    ).to(config.DEVICE)

    criterion = FocalLoss(alpha=class_weights, gamma=2.0, reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    train_losses, val_losses = [], []
    for epoch in range(1, NUM_EPOCHS + 1):
        # Training
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        epoch_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation for hyperparameter tuning
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        scheduler.step()
        logger.info(f"Epoch {epoch:02d}  Train Loss: {epoch_train_loss:.4f}  Val Loss: {epoch_val_loss:.4f}")

        # Early stopiing
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping triggered at epoch {epoch} (no improvement for {PATIENCE} epochs).")
                break


    # Saving the model weights
    model_path = os.path.join(RUN_DIR, 'model.pth')
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model weights at {model_path}")

    # Plot losses
    plt.figure()
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Train')
    plt.plot(range(1, NUM_EPOCHS + 1), val_losses,   label='Validation')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    loss_plot = os.path.join(RUN_DIR, 'loss.png')
    plt.savefig(loss_plot)
    plt.close()
    logger.info(f"Saved loss plot to {loss_plot}")

    # Evaluation on the test set
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(config.DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    # Confusion matrix and classification report
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=[config.INV_LABEL_MAP[i] for i in range(config.NUM_CLASSES)], 
        digits=4
    )

    # Log metrics so they are human readable
    logger.info("Confusion Matrix:")
    logger.info(f"\n{cm}")
    logger.info("Classification Report:")
    logger.info(f"\n{report}")

    # Save per-class metrics to CSV
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=list(range(config.NUM_CLASSES)),
        zero_division=0
    )
    metrics_df = pd.DataFrame({
        'class': [config.INV_LABEL_MAP[i] for i in range(config.NUM_CLASSES)],
        'precision': precision,
        'recall': recall,
        'f1-score': f1,
        'support': support
    })
    metrics_csv = os.path.join(RUN_DIR, 'classification_metrics.csv')
    metrics_df.to_csv(metrics_csv, index=False)
    logger.info(f"Saved classification metrics CSV to {metrics_csv}")

    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(
        cm,
        index=[config.INV_LABEL_MAP[i] for i in range(config.NUM_CLASSES)],
        columns=[config.INV_LABEL_MAP[i] for i in range(config.NUM_CLASSES)]
    )
    cm_csv = os.path.join(RUN_DIR, 'confusion_matrix.csv')
    cm_df.to_csv(cm_csv)
    logger.info(f"Saved confusion matrix CSV to {cm_csv}")


if __name__ == "__main__":
    train_and_evaluate()
