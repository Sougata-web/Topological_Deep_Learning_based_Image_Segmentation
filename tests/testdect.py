from __future__ import annotations

import torch

from src.losses.dectloss import (
    DECTTopologyLoss,
    DifferentiableECT2D,
    extract_center_patches,
)


def test_extract_center_patches_shape() -> None:
    tensor = torch.randn(2, 1, 16, 16)
    patches = extract_center_patches(
        tensor,
        patch_size=8,
        maximum_patches=5,
    )

    assert patches.shape == (5, 1, 8, 8)


def test_dect_shape_and_finiteness() -> None:
    transform = DifferentiableECT2D(
        number_of_directions=4,
        number_of_heights=8,
        sigmoid_steepness=10.0,
    )
    probability = torch.rand(3, 1, 12, 12)

    result = transform(probability)

    assert result.shape == (3, 4, 8)
    assert torch.isfinite(result).all()


def test_dect_loss_backpropagates() -> None:
    loss_module = DECTTopologyLoss(
        number_of_directions=4,
        number_of_heights=8,
        sigmoid_steepness=10.0,
        patch_size=8,
        maximum_patches=2,
        include_background=True,
    )

    logits = torch.randn(
        2,
        1,
        8,
        8,
        requires_grad=True,
    )
    target = (torch.rand(2, 1, 8, 8) > 0.5).float()

    loss = loss_module(logits, target)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_identical_binary_logits_have_small_loss() -> None:
    loss_module = DECTTopologyLoss(
        number_of_directions=4,
        number_of_heights=8,
        sigmoid_steepness=15.0,
        patch_size=8,
        maximum_patches=1,
        include_background=True,
    )

    target = torch.zeros(1, 1, 8, 8)
    target[:, :, 2:6, 2:6] = 1.0
    logits = torch.where(
        target > 0.5,
        torch.tensor(20.0),
        torch.tensor(-20.0),
    )

    loss = loss_module(logits, target)

    assert float(loss) < 1e-8
