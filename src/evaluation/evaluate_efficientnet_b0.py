import os
import sys
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from ..data.splitting import create_splits
from ..data.datasets import PlantDiseaseDataset
from ..training import config

MODEL_SUB_PATH = "logs/effnet_b0_ensemble_replication/effnet_b0_ensemble_replication_1/effnet_b0_ensemble_best.pth"
BATCH_SIZE = 32
NUM_WORKERS = 4

class AdaptiveEnsemble(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        weights = None
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
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features1 = self.effnet1(x)
        features2 = self.effnet2(x)
        concatenated_features = torch.cat((features1, features2), dim=1)
        output = self.ensemble_classifier(concatenated_features)
        return output


def evaluate_agent(model_path):
    print(f"Starting evaluation for model: {model_path}")

    print("Loading and preparing test data...")
    _, _, test_df = create_splits(
        data_dir=config.RAW_DATA_DIR,
        label_map=config.LABEL_MAP,
        test_size=config.TEST_SPLIT_SIZE,
        val_size=config.VALIDATION_SPLIT_SIZE,
    )

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_ds = PlantDiseaseDataset(test_df, transform=eval_transform)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    print(f"Test dataset loaded with {len(test_ds)} images.")

    print("Loading model architecture and weights...")
    model = AdaptiveEnsemble(num_classes=config.NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model = model.to(config.DEVICE)
    model.eval() # Set model to evaluation mode

    print("Running inference on the test set...")
    all_preds, all_labels = [], []
    total_inference_time = 0
    num_images = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(config.DEVICE)
            
            if config.DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            outputs = model(imgs)
            
            if config.DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()

            total_inference_time += (end_time - start_time)
            num_images += imgs.size(0)

            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)

    avg_inference_time_ms = (total_inference_time / num_images) * 1000
    print(f"\nTotal inference time: {total_inference_time:.4f} seconds")
    print(f"Total images evaluated: {num_images}")
    print(f"Average inference time per image: {avg_inference_time_ms:.4f} ms")

    report = classification_report(
        all_labels,
        all_preds,
        target_names=[config.INV_LABEL_MAP[i] for i in range(config.NUM_CLASSES)],
        digits=4
    )
    print("\nClassification Report:")
    print(report)

    print("\nGenerating Confusion Matrix plot...")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[config.INV_LABEL_MAP[i] for i in range(config.NUM_CLASSES)],
                yticklabels=[config.INV_LABEL_MAP[i] for i in range(config.NUM_CLASSES)])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    plot_path = os.path.join(os.path.dirname(model_path), "confusion_matrix.png")
    plt.savefig(plot_path)
    print(f"Confusion Matrix plot saved to: {plot_path}")


if __name__ == "__main__":
    full_model_path = os.path.join(config.PROJECT_ROOT, MODEL_SUB_PATH)
    evaluate_agent(model_path=full_model_path)
