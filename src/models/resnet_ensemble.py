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

MODEL_REGISTRY = {
    "resnet18": (models.resnet18, ResNet18_Weights.DEFAULT),
    "resnet34": (models.resnet34, ResNet34_Weights.DEFAULT),
    "resnet50": (models.resnet50, ResNet50_Weights.DEFAULT),
    "resnet101": (models.resnet101, ResNet101_Weights.DEFAULT),
    "resnet152": (models.resnet152, ResNet152_Weights.DEFAULT),
}


class ResNetEnsemble(nn.Module):
    def __init__(self, model_paths, model_names, num_classes, device="cuda"):
        super().__init__()
        self.models = nn.ModuleList()

        for path, name in zip(model_paths, model_names):
            if name not in MODEL_REGISTRY:
                raise ValueError(f"Unsupported model name: {name}")

            base_fn, _ = MODEL_REGISTRY[name]
            model = base_fn(weights=None)

            # Replace the head with the same structure used during training
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes),
            )

            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            self.models.append(model)

    def forward(self, x):
        outputs = [m(x) for m in self.models]
        return torch.stack(outputs).mean(dim=0)
