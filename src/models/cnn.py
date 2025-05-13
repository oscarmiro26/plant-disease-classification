import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    A CNN: three convolutional blocks with increased channel widths.
    """
    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        base_channels: int = 64,
        dropout_rate: float = 0.5
    ):
        super().__init__()

        # Feature extractor: 3 conv blocks, each doubling channels, with pooling
        self.features = nn.Sequential(
            # Block 1: in -> base
            nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/2

            # Block 2: base -> base*2
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/4

            # Block 3: base*2 -> base*4
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)   # H/8
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(base_channels * 4, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
