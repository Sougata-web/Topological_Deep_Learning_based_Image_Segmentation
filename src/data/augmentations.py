from __future__ import annotations

import numpy as np


def augment_pair(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        image = np.flip(image, axis=0)
        mask = np.flip(mask, axis=0)

    if rng.random() < 0.5:
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=1)

    rotations = int(rng.integers(0, 4))
    image = np.rot90(image, rotations)
    mask = np.rot90(mask, rotations)

    if rng.random() < 0.5:
        gamma = float(rng.uniform(0.8, 1.2))
        image = np.power(np.clip(image, 0.0, 1.0), gamma)

    if rng.random() < 0.3:
        noise = rng.normal(0.0, 0.025, size=image.shape)
        image = np.clip(image + noise, 0.0, 1.0)

    return np.ascontiguousarray(image), np.ascontiguousarray(mask)
