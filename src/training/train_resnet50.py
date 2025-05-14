import os
import logging

import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
# Use the new weights enum for pretrained weights
from torchvision.models import ResNet50_Weights
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

from ..data.splitting import create_splits
from ..data.datasets import PlantDiseaseDataset
from ..training import config

# Hyperparameters
BATCH_SIZE        = 64
NUM_EPOCHS        = 30
NUM_WORKERS       = 4
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

# Logging setup
LOG_BASE = os.path.join(config.LOG_DIR, "resnet50")
os.makedirs(LOG_BASE, exist_ok=True)
existing = [d for d in os.listdir(LOG_BASE) if os.path.isdir(os.path.join(LOG_BASE, d))]
run_idx  = len(existing)
run_name = f"ResNet50_gradual_{run_idx}"
RUN_DIR  = os.path.join(LOG_BASE, run_name)
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

# Transforms for 256x256 images
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
    logger.info(f"Starting run {run_name}.")

    # Data splits
    train_df, val_df, test_df = create_splits(
        data_dir  = config.RAW_DATA_DIR,
        label_map = config.LABEL_MAP,
        test_size = config.TEST_SPLIT_SIZE,
        val_size  = config.VALIDATION_SPLIT_SIZE,
    )

    # Datasets + loaders
    train_ds  = PlantDiseaseDataset(train_df, transform=train_transform)
    val_ds    = PlantDiseaseDataset(val_df,   transform=val_transform)
    test_ds   = PlantDiseaseDataset(test_df,  transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Model: ResNet50 with pretrained weights
    # Use the default pretrained ImageNet weights
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    # Freeze all layers except head
    for name, param in model.named_parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config.NUM_CLASSES)
    for param in model.fc.parameters():
        param.requires_grad = True
    model = model.to(config.DEVICE)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {'params': model.fc.parameters(), 'lr': LR_CLASSIFIER},
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    train_losses, val_losses = [], []
    for epoch in range(1, NUM_EPOCHS + 1):
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

        # Training
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

        # Validation
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                running_val += loss.item() * imgs.size(0)
        epoch_val_loss = running_val / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        scheduler.step()
        logger.info(f"Epoch {epoch:02d} | Train: {epoch_train_loss:.4f} | Val: {epoch_val_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), os.path.join(RUN_DIR, 'resnet50_gradual.pth'))
    # Plot losses
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train')
    plt.plot(range(1, len(val_losses)+1),   val_losses,   label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, 'loss_curve.png')); plt.close()

    # Test evaluation
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

if __name__ == "__main__":
    train_and_evaluate()
