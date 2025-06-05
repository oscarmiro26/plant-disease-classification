import os
from time import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    precision_recall_fscore_support,
    accuracy_score
)
import pandas as pd
from balanced_loss import Loss
from typing import Tuple, List, Optional
from logging import Logger
from ..training import config
from ..data.splitting import create_splits
from ..data.datasets import PlantDiseaseDataset
from ..data.sampler import create_sampler


def load_model(model_name: str = "resnet50_9897.pth") -> nn.Module:
    """Load a pretrained ResNet50 model.

    Args:
        model_name (str): Name of the model checkpoint file to load.

    Returns:
        nn.Module: The loaded ResNet50 model moved to DEVICE.
    """
    base = resnet50(weights=None)
    num_classes = len(config.INV_LABEL_MAP)
    num_ftrs = base.fc.in_features
    base.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    resnet_model_path = os.path.join(config.MODELS_DIR, model_name)
    state = torch.load(resnet_model_path, map_location="cpu")
    base.load_state_dict(state)
    model = base.to(config.DEVICE)
    return model


def split_data(random_seed: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split raw data into train, validation, and test DataFrames.

    Args:
        random_seed (Optional[int]): Seed for reproducible splits.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrames for train, val, and test.
    """
    return create_splits(
        data_dir=config.RAW_DATA_DIR,
        label_map=config.LABEL_MAP,
        test_size=config.TEST_SPLIT_SIZE,
        val_size=config.VALIDATION_SPLIT_SIZE,
        random_seed=random_seed
    )

def get_samples_per_class() -> List[int]:
    """Return list of sample counts per class in training set.

    Returns:
        List[int]: Sample counts for each class in the train split.
    """
    train_df, _, _ = split_data(random_seed=42)
    return [train_df['label'].tolist().count(label) for label in config.LABEL_MAP.keys()]

def get_scheduler(
    optimizer: torch.optim.Optimizer,
    step_size: int = 10,
    gamma: float = 0.1
) -> torch.optim.lr_scheduler._LRScheduler:
    """Return a StepLR scheduler for the optimizer.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to schedule.
        step_size (int): Interval for stepping the LR.
        gamma (float): Multiplicative LR decay factor.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: Configured StepLR scheduler.
    """
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

def get_optimizer(
    model: nn.Module,
    lr: float = 0.001,
    weight_decay: float = 1e-4
) -> torch.optim.Optimizer:
    """Return an AdamW optimizer for the model.

    Args:
        model (nn.Module): Model whose parameters will be optimized.
        lr (float): Learning rate.
        weight_decay (float): Weight decay factor.

    Returns:
        torch.optim.Optimizer: Configured AdamW optimizer.
    """
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

def get_criterion() -> Loss:
    """Return a Focal Loss criterion.

    Returns:
        Loss: Focal loss configured with class balance.
    """
    return Loss(
        loss_type="focal_loss",
        samples_per_class=get_samples_per_class(),
        class_balanced=True
    )

def load_data(
    batch_size: int,
    num_workers: int,
    mix_alpha: float
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size for all loaders.
        num_workers (int): Workers count for parallel data loading.
        mix_alpha (float): MixUp alpha hyperparameter.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, val, and test loaders.
    """
    train_df, val_df, test_df = split_data(random_seed=42)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_ds = PlantDiseaseDataset(train_df, transform=train_transform)
    val_ds = PlantDiseaseDataset(val_df, transform=val_transform)
    test_ds = PlantDiseaseDataset(test_df, transform=val_transform)
    train_sampler = create_sampler(
        train_df,
        config.LABEL_MAP,
        use_mixing_sampler=True,
        alpha=mix_alpha
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=torch.cuda.is_available())
    return train_loader, val_loader, test_loader


def train(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Loss
) -> float:
    """Train the model for one epoch.

    Args:
        model (nn.Module): Model to train.
        data_loader (DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        criterion (Loss): Loss function.

    Returns:
        float: Average training loss.
    """
    model.train()
    running_loss = 0.0
    for images, labels in data_loader:
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(data_loader.dataset)


def validate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: Loss
) -> float:
    """Validate the model for one epoch.

    Args:
        model (nn.Module): Model to validate.
        data_loader (DataLoader): Validation data loader.
        criterion (Loss): Loss function.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
    return running_loss / len(data_loader.dataset)


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    run_dir: str,
    logger: Optional[Logger] = None,
    get_acc: bool = False,
    report: bool = True,
    return_latency: bool = False,
) -> None:
    """Evaluates model performance on test set and saves metrics.

    Args:
        model (nn.Module): Model to evaluate.
        data_loader (DataLoader): Test data loader.
        run_dir (str): Directory to save evaluation metrics.
        logger (Optional[Logger]): Logger for metric output.
        get_acc (bool): Whether to compute and return accuracy.
        report (bool): Whether to generate classification report.
        return_latency (bool): Whether to return latencies.

    Returns:
        Optional[List[float]]: List containing average latency and accuracy if requested.
    """
    return_objects = []
    all_preds, all_labels = [], []
    model.eval()
    if return_latency:
        start = time()
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(config.DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    if return_latency:
        end = time()
        latency = (end - start) / len(data_loader.dataset)
        return_objects.append(latency)
        if logger:
            logger.info(f"Average inference latency: {latency:.4f} seconds per sample")

    if report:
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(
            all_labels, all_preds,
            target_names=[config.INV_LABEL_MAP[i] for i in range(config.NUM_CLASSES)],
            digits=4
        )

        # Optional logging
        if logger:
            logger.info("Confusion Matrix:\n" + str(cm))
            logger.info("Classification Report:\n" + report)
        # Save metrics to CSV
        prec, rec, f1, sup = precision_recall_fscore_support(
            all_labels, all_preds, labels=list(range(config.NUM_CLASSES)), zero_division=0)
        metrics_df = pd.DataFrame({
            'class': [config.INV_LABEL_MAP[i] for i in range(config.NUM_CLASSES)],
            'precision': prec, 'recall': rec, 'f1-score': f1, 'support': sup
        })
        metrics_df.to_csv(os.path.join(run_dir, 'metrics.csv'), index=False)

    if get_acc:
        accuracy = accuracy_score(all_labels, all_preds) if get_acc else None
        return_objects.append(accuracy)

    return return_objects if return_objects else None

