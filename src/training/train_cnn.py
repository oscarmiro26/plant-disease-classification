import os
import sys
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from ..data.splitting import create_splits
from ..data.datasets import PlantDiseaseDataset
from ..data.preprocessing import train_transforms, val_test_transforms
from ..data.sampler import calculate_class_weights, create_sampler
from ..models.cnn import CNN
from ..training import config

BATCH_SIZE     = 128
NUM_EPOCHS     = 50
NUM_WORKERS    = 4
LEARNING_RATE  = 1e-3
STEP_SIZE      = 10
GAMMA          = 0.1

def train_and_evaluate():
    # Creating data splits
    train_df, val_df, test_df = create_splits(
        data_dir    = config.RAW_DATA_DIR,
        label_map   = config.LABEL_MAP,
        test_size   = config.TEST_SPLIT_SIZE,
        val_size    = config.VALIDATION_SPLIT_SIZE,
        random_seed = 42
    )

    # Calculating class weights
    class_weights = calculate_class_weights(train_df, config.LABEL_MAP).to(config.DEVICE)
    train_sampler = create_sampler(train_df, config.LABEL_MAP)

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
        dropout_rate=0.5
    ).to(config.DEVICE)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    # Training loop
    train_losses, val_losses = [], []
    for epoch in range(1, NUM_EPOCHS + 1):
        # Training
        model.train()
        running_train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * imgs.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation for hyperparameter tuning
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
        print(f"Epoch {epoch:02d}  Train Loss: {epoch_train_loss:.4f}  Val Loss: {epoch_val_loss:.4f}")

    # Plot losses
    plt.figure()
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses)
    plt.plot(range(1, NUM_EPOCHS + 1), val_losses)
    plt.title("Training vs. Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train", "Validation"])
    plt.tight_layout()
    plt.show()

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
    print("Confusion Matrix (rows=true, cols=predicted):")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=[config.INV_LABEL_MAP[i] for i in range(config.NUM_CLASSES)],
        digits=4
    ))

if __name__ == "__main__":
    train_and_evaluate()
