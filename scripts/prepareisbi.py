from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile

from src.config import load_config
from src.utils.seed import seed_everything


def normalize_stack(stack: np.ndarray) -> np.ndarray:
    stack = stack.astype(np.float32)
    normalized = np.empty_like(stack, dtype=np.float32)

    for index, image in enumerate(stack):
        low, high = np.percentile(image, [1.0, 99.0])
        image = np.clip(image, low, high)
        normalized[index] = (image - low) / max(high - low, 1e-8)

    return normalized


def main(config_path: str) -> None:
    config = load_config(config_path)
    seed = int(config["project"]["seed"])
    seed_everything(seed)

    image_path = Path(config["data"]["image_tif"])
    label_path = Path(config["data"]["label_tif"])
    split_path = Path(config["data"]["split_file"])
    processed_dir = Path(config["data"]["processed_dir"])

    if not image_path.exists():
        raise FileNotFoundError(f"Image TIFF not found: {image_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label TIFF not found: {label_path}")

    images = tifffile.imread(image_path)
    labels = tifffile.imread(label_path)

    if images.ndim != 3 or labels.ndim != 3:
        raise ValueError(
            "Expected TIFF stacks shaped [slices, height, width]."
        )
    if images.shape != labels.shape:
        raise ValueError(
            f"Image and label shapes differ: {images.shape} vs {labels.shape}."
        )

    images = normalize_stack(images)
    labels = (labels > 0).astype(np.float32)

    processed_dir.mkdir(parents=True, exist_ok=True)
    images_output = processed_dir / "images.npy"
    labels_output = processed_dir / "labels.npy"

    np.save(images_output, images)
    np.save(labels_output, labels)

    number_of_slices = images.shape[0]
    validation_fraction = float(config["data"]["validation_fraction"])
    test_fraction = float(config["data"]["test_fraction"])

    if validation_fraction < 0.0 or test_fraction < 0.0:
        raise ValueError("Split fractions cannot be negative.")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError(
            "validationfraction + testfraction must be less than 1."
        )
    if number_of_slices < 3:
        raise ValueError("At least three slices are required for all splits.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(number_of_slices)

    validation_count = max(
        1,
        int(round(number_of_slices * validation_fraction)),
    )
    test_count = max(
        1,
        int(round(number_of_slices * test_fraction)),
    )

    if validation_count + test_count >= number_of_slices:
        excess = validation_count + test_count - number_of_slices + 1
        if test_count > 1:
            reduction = min(excess, test_count - 1)
            test_count -= reduction
            excess -= reduction
        if excess > 0 and validation_count > 1:
            validation_count -= min(excess, validation_count - 1)

    test_indices = indices[:test_count]
    validation_indices = indices[
        test_count:test_count + validation_count
    ]
    training_indices = indices[test_count + validation_count:]

    if len(training_indices) == 0:
        raise RuntimeError("The computed training split is empty.")

    splits = {
        "train": sorted(int(index) for index in training_indices),
        "validation": sorted(
            int(index) for index in validation_indices
        ),
        "test": sorted(int(index) for index in test_indices),
    }

    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", encoding="utf-8") as handle:
        json.dump(splits, handle, indent=2)

    print(f"Images: {images_output}")
    print(f"Labels: {labels_output}")
    print(f"Splits: {split_path}")
    print(
        "Slice counts: "
        f"train={len(splits['train'])}, "
        f"validation={len(splits['validation'])}, "
        f"test={len(splits['test'])}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/isbi.yaml",
        help="Path to the YAML configuration file.",
    )
    arguments = parser.parse_args()
    main(arguments.config)
