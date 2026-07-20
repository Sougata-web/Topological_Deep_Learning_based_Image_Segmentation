from __future__ import annotations

import torch
from torch import nn


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        dimensions = tuple(range(1, probabilities.ndim))

        intersection = torch.sum(
            probabilities * target,
            dim=dimensions,
        )
        denominator = torch.sum(
            probabilities + target,
            dim=dimensions,
        )

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ) -> None:
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        bce = self.bce(logits, target)
        dice = self.dice(logits, target)
        total = self.bce_weight * bce + self.dice_weight * dice

        return total, {
            "bce": bce.detach(),
            "dice_loss": dice.detach(),
        }
