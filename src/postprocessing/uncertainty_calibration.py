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
    Decorator that adds a single log-temperature parameter to a pretrained model,
    allowing T = exp(log_T) to be learned (so T can go < 1 or > 1).
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        # learnable log-temperature, initialized to 0 => T = exp(0) = 1
        self.log_T = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        logits = self.model(x)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        T = torch.exp(self.log_T).to(logits.device)
        # divide logits by scalar T
        return logits / T.unsqueeze(1).expand(logits.size(0), logits.size(1))

    @torch.no_grad()
    def compute_ece(self, logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
        """
        Expected Calibration Error (ECE), binning the confidence range [0,1].
        """
        softmaxes = torch.softmax(logits, dim=1)
        confidences, predictions = softmaxes.max(dim=1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=logits.device)

        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)
        for start, end in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            in_bin = confidences.gt(start) & confidences.le(end)
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                acc_in_bin = accuracies[in_bin].float().mean()
                conf_in_bin = confidences[in_bin].mean()
                ece += torch.abs(conf_in_bin - acc_in_bin) * prop_in_bin

        return ece.item()

    def set_temperature(self, valid_loader: DataLoader, device: torch.device) -> float:
        """
        Fit log_T on validation set by minimizing NLL via LBFGS.
        Returns learned T = exp(log_T).
        """
        self.to(device)
        self.model.to(device)
        logits_list, labels_list = [], []
        with torch.no_grad():
            for imgs, labels in valid_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                logits = self.model(imgs)
                logits_list.append(logits)
                labels_list.append(labels)

        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        # baseline
        nll = nn.CrossEntropyLoss().to(device)
        before_nll = nll(logits, labels).item()
        before_ece = self.compute_ece(logits, labels)

        # optimize log_T
        optimizer = torch.optim.LBFGS([self.log_T], lr=0.01, max_iter=50)

        def _eval():
            optimizer.zero_grad()
            loss = nll(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(_eval)

        after_nll = nll(self.temperature_scale(logits), labels).item()
        after_ece = self.compute_ece(self.temperature_scale(logits), labels)
        T_opt = torch.exp(self.log_T).item()

        print(f"Optimal T: {T_opt:.4f}")
        print(f"NLL before / after: {before_nll:.4f} → {after_nll:.4f}")
        print(f"ECE before / after: {before_ece:.4f} → {after_ece:.4f}")

        return T_opt


def plot_reliability_diagram(logits: torch.Tensor,
                             labels: torch.Tensor,
                             n_bins: int,
                             filename: str):
    """
    Plots accuracy – confidence gap bars and saves to `filename`.
    """
    softmaxes = torch.softmax(logits, dim=1)
    confidences = softmaxes.max(dim=1)[0].detach().cpu().numpy()
    predictions = softmaxes.max(dim=1)[1].detach().cpu().numpy()
    labels_np   = labels.detach().cpu().numpy()
    accuracies  = (predictions == labels_np).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    lows, ups = bin_edges[:-1], bin_edges[1:]
    gaps = np.zeros(n_bins)

    for i, (l, u) in enumerate(zip(lows, ups)):
        mask = (confidences > l) & (confidences <= u)
        if mask.sum() > 0:
            acc_bin = accuracies[mask].mean()
            conf_bin = confidences[mask].mean()
            gaps[i] = acc_bin

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.bar(lows, gaps, width=1.0/n_bins, edgecolor="black", align="edge", alpha=0.7)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram (gap)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main(args):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    # 1) recreate splits
    train_df, val_df, test_df = create_splits(
        data_dir  = config.RAW_DATA_DIR,
        label_map = config.LABEL_MAP,
        test_size = config.TEST_SPLIT_SIZE,
        val_size  = config.VALIDATION_SPLIT_SIZE,
    )

    # 2) data loaders
    norm_mean = [0.485, 0.456, 0.406]
    norm_std  = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    val_ds  = PlantDiseaseDataset(val_df,  transform=transform)
    test_ds = PlantDiseaseDataset(test_df, transform=transform)
    val_loader  = DataLoader(val_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 3) load model
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, config.NUM_CLASSES),
    )
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 4) calibrate
    model_ts = ModelWithTemperature(model)
    T_opt = model_ts.set_temperature(val_loader, device)

    # 5) evaluate on test
    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            all_logits.append(model(imgs))
            all_labels.append(labels.to(device))
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    ece_b = model_ts.compute_ece(logits, labels)
    ece_a = model_ts.compute_ece(model_ts.temperature_scale(logits), labels)
    print(f"Test ECE before: {ece_b:.4f}")
    print(f"Test ECE after : {ece_a:.4f}")

    # 6) plots
    os.makedirs(args.output_dir, exist_ok=True)
    plot_reliability_diagram(logits, labels,   args.n_bins,
                             os.path.join(args.output_dir, "reliability_before.png"))
    plot_reliability_diagram(model_ts.temperature_scale(logits),
                             labels, args.n_bins,
                             os.path.join(args.output_dir, "reliability_after.png"))
    print(f"Reliability diagrams saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Uncertainty calibration via log‐temperature scaling")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to your trained resnet50.pth")
    parser.add_argument("--output-dir", type=str, default="calibration",
                        help="Directory to save the plots")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-bins", type=int, default=15,
                        help="Number of bins for ECE / reliability diagram")
    args = parser.parse_args()
    main(args)