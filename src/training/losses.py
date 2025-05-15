import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss for multi-class classification.
    See https://arxiv.org/abs/1708.02002.
    """
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        if alpha is not None:
            # Wrote this to fix warning
            if isinstance(alpha, torch.Tensor):
                self.alpha = alpha.detach().clone().float()
            else:
                self.alpha = torch.as_tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None
        
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.alpha is not None:
            self.alpha = self.alpha.to(logits.device)

        ce_loss = F.cross_entropy(
            logits, 
            labels, 
            weight=self.alpha,
            reduction="none"
        )

        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            focal = focal.mean()
        elif self.reduction == "sum":
            focal = focal.sum()
        return focal
