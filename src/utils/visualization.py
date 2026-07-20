from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_prediction_figure(
    image: np.ndarray,
    target: np.ndarray,
    probability: np.ndarray,
    output_path: str | Path,
    threshold: float = 0.5,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prediction = probability >= threshold
    error = np.zeros((*target.shape, 3), dtype=np.float32)

    true_positive = np.logical_and(prediction, target > 0.5)
    false_positive = np.logical_and(prediction, target <= 0.5)
    false_negative = np.logical_and(~prediction, target > 0.5)

    error[true_positive] = [1.0, 1.0, 1.0]
    error[false_positive] = [1.0, 0.0, 0.0]
    error[false_negative] = [0.0, 1.0, 0.0]

    figure, axes = plt.subplots(1, 5, figsize=(18, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input")
    axes[1].imshow(target, cmap="gray")
    axes[1].set_title("Ground truth")
    axes[2].imshow(probability, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[2].set_title("Probability")
    axes[3].imshow(prediction, cmap="gray")
    axes[3].set_title("Prediction")
    axes[4].imshow(error)
    axes[4].set_title("Overlay")

    for axis in axes:
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
