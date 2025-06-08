#!/usr/bin/env python
import os
import sys
import argparse

# ─── Make relative imports work when running this file directly ─────────────
if __name__ == "__main__" and __package__ is None:
    __package__ = "training"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

from ..data.splitting import create_splits
from ..data.datasets import PlantDiseaseDataset
from ..training import config


class ModelWithTemperature(nn.Module):
    """
    A thin decorator that adds a temperature parameter to a pretrained model;
    temperature scaling is performed on the logits.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        # Initialize temperature > 1 so scaling can only reduce confidence
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        logits = self.model(x)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        # Expand temperature to match logits shape
        return logits / self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))

    @torch.no_grad()
    def compute_ece(self, logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
        """
        Computes Expected Calibration Error (ECE)
        """
        softmaxes = torch.softmax(logits, dim=1)
        confidences, predictions = softmaxes.max(dim=1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=logits.device)

        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)
        for start, end in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            # find indices where confidence in (start, end]
            in_bin = confidences.gt(start.item()) * confidences.le(end.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece.item()

    def set_temperature(self, valid_loader: DataLoader, device: torch.device) -> float:
        """
        Tune the temperature on the validation set using NLL loss.
        Returns the optimized temperature.
        """
        self.to(device)
        self.model.to(device)
        # Gather all logits and labels
        logits_list  = []
        labels_list  = []
        with torch.no_grad():
            for imgs, labels in valid_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                logits = self.model(imgs)
                logits_list.append(logits)
                labels_list.append(labels)

        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        # Before calibration
        nll_criterion = nn.CrossEntropyLoss().to(device)
        before_nll = nll_criterion(logits, labels).item()
        before_ece = self.compute_ece(logits, labels)

        # Optimize temperature with LBFGS
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def _eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(_eval)

        # After calibration
        after_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_ece = self.compute_ece(self.temperature_scale(logits), labels)

        print(f"Optimal temperature: {self.temperature.item():.4f}")
        print(f"NLL  before / after: {before_nll:.4f} → {after_nll:.4f}")
        print(f"ECE  before / after: {before_ece:.4f} → {after_ece:.4f}")

        return self.temperature.item()


def plot_reliability_diagram(logits: torch.Tensor,
                             labels: torch.Tensor,
                             n_bins: int,
                             filename: str):
    """
    Plots a reliability diagram and saves to disk.
    """
    softmaxes = torch.softmax(logits, dim=1)
    confidences, predictions = softmaxes.max(dim=1)
    accuracies = predictions.eq(labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    accuracies_bin = np.zeros(n_bins)
    confidences_bin = np.zeros(n_bins)
    prop_bin = np.zeros(n_bins)

    for i, (lower, upper) in enumerate(zip(bin_lowers, bin_uppers)):
        in_bin = (confidences.detach().cpu().numpy() > lower) & (confidences.detach().cpu().numpy() <= upper)
        prop_in_bin = in_bin.mean()
        prop_bin[i] = prop_in_bin
        if prop_in_bin > 0:
            accuracies_bin[i]  = accuracies.detach().cpu().numpy()[in_bin].mean()
            confidences_bin[i] = confidences.detach().cpu().numpy()[in_bin].mean()

    # plot
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.bar(bin_lowers, accuracies_bin - confidences_bin,
            width=1.0/n_bins, edgecolor='black', align='edge', alpha=0.7)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy – Confidence")
    plt.title("Reliability Diagram (gap)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main(args):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    # ─── 1. Recreate splits ────────────────────────────────────────────────────
    train_df, val_df, test_df = create_splits(
        data_dir  = config.RAW_DATA_DIR,
        label_map = config.LABEL_MAP,
        test_size = config.TEST_SPLIT_SIZE,
        val_size  = config.VALIDATION_SPLIT_SIZE,
    )

    # ─── 2. Datasets & Loaders ─────────────────────────────────────────────────
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    test_transform = val_transform  # same for calibration & evaluation

    val_ds  = PlantDiseaseDataset(val_df,  transform=val_transform)
    test_ds = PlantDiseaseDataset(test_df, transform=test_transform)

    val_loader  = DataLoader(val_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # ─── 3. Load your trained model ────────────────────────────────────────────
    model = models.resnet50(weights=None)
    # rebuild the same head
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, config.NUM_CLASSES)
    )
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # ─── 4. Calibrate temperature ──────────────────────────────────────────────
    model_ts = ModelWithTemperature(model)
    T = model_ts.set_temperature(val_loader, device)

    # ─── 5. Evaluate on test set ──────────────────────────────────────────────
    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            all_logits.append(logits)
            all_labels.append(labels.to(device))
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    # Pre‐ and post‐calibration ECE
    ece_before = model_ts.compute_ece(logits, labels)
    ece_after  = model_ts.compute_ece(model_ts.temperature_scale(logits), labels)
    print(f"Test ECE before: {ece_before:.4f}")
    print(f"Test ECE after : {ece_after:.4f}")

    # ─── 6. Plot reliability diagrams ─────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    plot_reliability_diagram(logits, labels, args.n_bins,
                             os.path.join(args.output_dir, "reliability_before.png"))
    plot_reliability_diagram(model_ts.temperature_scale(logits), labels, args.n_bins,
                             os.path.join(args.output_dir, "reliability_after.png"))
    print(f"Reliability diagrams saved to {args.output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Uncertainty calibration via temperature scaling")
    p.add_argument("--model-path",  type=str, required=True,
                   help="Path to .pth model weights (resnet50.pth)")
    p.add_argument("--output-dir",  type=str, default="calibration",
                   help="Where to save plots")
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--n-bins",      type=int, default=15,
                   help="Number of bins for ECE / reliability diagram")
    args = p.parse_args()
    main(args)

"""
usage:

python calibrate_uncertainty.py \
  --model-path runs/2025-06-01_12-00-00/resnet50.pth \
  --output-dir runs/2025-06-01_12-00-00/calibration \
  --batch-size 64
"""