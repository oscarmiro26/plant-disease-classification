import os
import sys

if __name__ == "__main__" and __package__ is None:
    __package__ = "training"

import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

from ..data.splitting import create_splits
from ..data.datasets import PlantDiseaseDataset
from . import config
from ..utils.logger import setup_logger
from ..models.resnet_ensemble import ResNetEnsemble

PROJECT_ROOT = os.path.abspath(
    os.path.join(__file__, "..", "..", "..")
)

ENSEMBLE_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "models",
    "resnet_ensemble"
)

MODEL_PATHS = [
    os.path.join(ENSEMBLE_DIR, f"resnet{n}.pth")
    for n in (18, 34, 50, 101, 152)
]

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def main():
    logger, RUN_DIR = setup_logger(
        model_name="ensemble_test",
        base_log_dir=config.LOG_DIR
    )
    logger.info("Starting ensemble evaluation.")

    _, _, test_df = create_splits(
        data_dir  = config.RAW_DATA_DIR,
        label_map = config.LABEL_MAP,
        test_size = config.TEST_SPLIT_SIZE,
        val_size  = config.VALIDATION_SPLIT_SIZE,
    )

    test_ds = PlantDiseaseDataset(test_df, transform=test_transform)
    test_loader = DataLoader(
        test_ds,
        batch_size   = 64,
        shuffle      = False,
        num_workers  = 4
    )

    ensemble = ResNetEnsemble(
        num_classes  = config.NUM_CLASSES,
        model_names  = [f"resnet{n}" for n in (18, 34, 50, 101, 152)],
        model_paths  = MODEL_PATHS,
        device       = config.DEVICE
    )
    ensemble.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(config.DEVICE)
            preds = ensemble(imgs).argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels,
        all_preds,
        target_names=[config.INV_LABEL_MAP[i] for i in range(config.NUM_CLASSES)],
        digits=4
    )
    logger.info("Ensemble Test Confusion Matrix:\n%s", cm)
    logger.info("Ensemble Test Classification Report:\n%s", report)

    prec, rec, f1, sup = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=list(range(config.NUM_CLASSES)),
        zero_division=0
    )
    metrics_df = pd.DataFrame({
        "class":     [config.INV_LABEL_MAP[i] for i in range(config.NUM_CLASSES)],
        "precision": prec,
        "recall":    rec,
        "f1-score":  f1,
        "support":   sup
    })
    metrics_path = os.path.join(RUN_DIR, "ensemble_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Per-class metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
