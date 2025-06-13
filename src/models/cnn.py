import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    A simple CNN with 4 conv blocks, doubling channels each time, plus a two-layer head.
    """

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        base_channels: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()

        # Iteratively build the CNN blocks
        layers = []
        in_c = input_channels
        for mult in [1, 2, 4, 8]:
            out_c = base_channels * mult
            layers += [
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_c = out_c
        self.features = nn.Sequential(*layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 8, base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
