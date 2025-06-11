import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
)


class ResNetEnsemble(nn.Module):
    """
    Ensemble of multiple ResNet models (18, 34, 50, 101, 152).
    Forward pass averages the logits from each member model.

    Example usage:

    ensemble = ResNetEnsemble(
        num_classes=38,
        pretrained=False,
        model_paths=[
            "resnet18.pth",
            "resnet34.pth",
            "resnet50.pth",
            "resnet101.pth",
            "resnet152.pth",
        ],
        device="cuda"
    )
    """
    def __init__(self, num_classes, pretrained=True, model_paths=None, device=None):
        super(ResNetEnsemble, self).__init__()
        # Determine device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Build individual ResNet instances
        self.models = nn.ModuleList([
            self._make_resnet(models.resnet18, ResNet18_Weights, num_classes, pretrained),
            self._make_resnet(models.resnet34, ResNet34_Weights, num_classes, pretrained),
            self._make_resnet(models.resnet50, ResNet50_Weights, num_classes, pretrained),
            self._make_resnet(models.resnet101, ResNet101_Weights, num_classes, pretrained),
            self._make_resnet(models.resnet152, ResNet152_Weights, num_classes, pretrained),
        ])

        # Optionally load custom checkpoints for each model
        if model_paths:
            assert len(model_paths) == len(self.models), \
                "model_paths must have five entries (for ResNet18,34,50,101,152)"
            for model, path in zip(self.models, model_paths):
                state_dict = torch.load(path, map_location=self.device)
                model.load_state_dict(state_dict)

        # Move ensemble to the target device
        self.to(self.device)

    def _make_resnet(self, constructor, weights_enum, num_classes, pretrained):
        """
        Instantiate a single ResNet variant and replace its head.
        """
        weights = weights_enum.DEFAULT if pretrained else None
        model = constructor(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    def forward(self, x):
        """
        Forward pass: run input through each ResNet and average logits.

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, 3, H, W).

        Returns:
            torch.Tensor: averaged logits of shape (batch_size, num_classes).
        """
        x = x.to(self.device)
        outputs = [model(x) for model in self.models]
        stacked = torch.stack(outputs, dim=0)  # shape: (ensemble_size, batch, num_classes)
        return torch.mean(stacked, dim=0)
