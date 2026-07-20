from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from skimage.measure import euler_number, label


def betti_numbers_2d(
    binary_image: np.ndarray,
    connectivity: int = 1,
) -> tuple[int, int]:
    binary_image = binary_image.astype(bool)

    beta_0 = int(label(binary_image, connectivity=connectivity).max())
    chi = int(euler_number(binary_image, connectivity=connectivity))
    beta_1 = int(beta_0 - chi)

    return beta_0, beta_1


def iter_patches(
    image: np.ndarray,
    patch_size: int,
    stride: int,
) -> Iterator[np.ndarray]:
    height, width = image.shape

    if patch_size > min(height, width):
        yield image
        return

    y_positions = list(range(0, height - patch_size + 1, stride))
    x_positions = list(range(0, width - patch_size + 1, stride))

    # Include edge-aligned patches when dimensions are not divisible by stride.
    if y_positions[-1] != height - patch_size:
        y_positions.append(height - patch_size)
    if x_positions[-1] != width - patch_size:
        x_positions.append(width - patch_size)

    for y in y_positions:
        for x in x_positions:
            yield image[y:y + patch_size, x:x + patch_size]


def topology_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    patch_size: int = 64,
    stride: int = 64,
) -> dict[str, float]:
    prediction_patches = list(iter_patches(prediction, patch_size, stride))
    target_patches = list(iter_patches(target, patch_size, stride))

    if len(prediction_patches) != len(target_patches):
        raise RuntimeError("Prediction and target patch counts differ.")
    if not prediction_patches:
        raise ValueError("No topology patches were produced.")

    beta_0_errors: list[float] = []
    beta_1_errors: list[float] = []
    euler_errors: list[float] = []

    for predicted_patch, target_patch in zip(
        prediction_patches,
        target_patches,
        strict=True,
    ):
        predicted_beta_0, predicted_beta_1 = betti_numbers_2d(
            predicted_patch
        )
        target_beta_0, target_beta_1 = betti_numbers_2d(target_patch)

        beta_0_errors.append(abs(predicted_beta_0 - target_beta_0))
        beta_1_errors.append(abs(predicted_beta_1 - target_beta_1))
        euler_errors.append(
            abs(
                (predicted_beta_0 - predicted_beta_1)
                - (target_beta_0 - target_beta_1)
            )
        )

    beta_0_array = np.asarray(beta_0_errors)
    beta_1_array = np.asarray(beta_1_errors)

    return {
        "betti0_error": float(beta_0_array.mean()),
        "betti1_error": float(beta_1_array.mean()),
        "betti_error": float((beta_0_array + beta_1_array).mean()),
        "euler_error": float(np.mean(euler_errors)),
    }
