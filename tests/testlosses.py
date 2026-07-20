from __future__ import annotations

import torch

from src.losses.segmentation import SegmentationLoss, SoftDiceLoss
import pytest
import torch
from src.losses.phloss import PersistentHomologyLoss


def test_soft_dice_loss_is_small_for_correct_logits() -> None:
    target = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]]]]
    )
    logits = torch.where(
        target > 0.5,
        torch.tensor(20.0),
        torch.tensor(-20.0),
    )

    loss = SoftDiceLoss()(logits, target)

    assert float(loss) < 1e-6


def test_segmentation_loss_returns_components() -> None:
    logits = torch.randn(
        2,
        1,
        16,
        16,
        requires_grad=True,
    )
    target = (torch.rand(2, 1, 16, 16) > 0.5).float()

    loss_module = SegmentationLoss(
        bce_weight=0.5,
        dice_weight=0.5,
    )
    total, components = loss_module(logits, target)
    total.backward()

    assert total.ndim == 0
    assert set(components) == {"bce", "diceloss"}
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
def test_ph_loss_is_finite_and_backpropagates() -> None:
    loss_module = PersistentHomologyLoss(
        patch_size=16,
        maximum_patches=2,
        dimensions=(0, 1),
    )
    logits = torch.randn(
        2,
        1,
        16,
        16,
        requires_grad=True,
    )
    target = torch.zeros(2, 1, 16, 16)
    target[:, :, 4:12, 4:12] = 1.0
    target[:, :, 7:9, 7:9] = 0.0
    loss = loss_module(logits, target)
    loss.backward()
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_ph_loss_runs_on_cuda() -> None:
    device = torch.device("cuda:0")
    loss_module = PersistentHomologyLoss(
        patch_size=16,
        maximum_patches=1,
        dimensions=(0, 1),
    ).to(device)
    logits = torch.randn(
        1,
        1,
        16,
        16,
        device=device,
        requires_grad=True,
    )
    target = torch.zeros_like(logits)
    target[:, :, 3:13, 3:13] = 1.0
    loss = loss_module(logits, target)
    assert loss.device.type == "cuda"
    loss.backward()
    assert logits.grad is not None