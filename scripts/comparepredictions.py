from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METHODS = ("baseline", "ph", "dect")


def load_test_indices(split_file: Path) -> list[int]:
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with split_file.open("r", encoding="utf-8") as handle:
        splits = json.load(handle)

    if "test" not in splits:
        raise KeyError(f"No 'test' split found in: {split_file}")

    return [int(index) for index in splits["test"]]


def find_prediction_path(
    output_root: Path,
    method: str,
    dataset_index: int,
    test_position: int,
) -> Path:
    prediction_directory = (
        output_root / method / "evaluation" / "predictions"
    )

    if not prediction_directory.exists():
        raise FileNotFoundError(
            f"Prediction directory not found: {prediction_directory}\n"
            f"Evaluate {method} first."
        )

    # Evaluation may save using the original dataset index.
    indexed_path = prediction_directory / f"slice{dataset_index:03d}.npy"

    # Alternatively, evaluation may number test samples sequentially.
    sequential_path = prediction_directory / f"slice{test_position:03d}.npy"

    if indexed_path.exists():
        return indexed_path

    if sequential_path.exists():
        return sequential_path

    available_files = sorted(prediction_directory.glob("slice*.npy"))
    available_names = ", ".join(path.name for path in available_files[:10])

    if len(available_files) > 10:
        available_names += ", ..."

    raise FileNotFoundError(
        f"No prediction found for {method}.\n"
        f"Original-index filename tried: {indexed_path}\n"
        f"Sequential filename tried: {sequential_path}\n"
        f"Available files: {available_names or 'none'}\n"
        f"Re-run evaluation if the directory is empty."
    )


def normalize_prediction(prediction: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction).squeeze()

    if prediction.ndim != 2:
        raise ValueError(
            "Expected a 2D prediction after removing singleton dimensions, "
            f"but received shape {prediction.shape}."
        )

    return prediction


def main(
    output_root: str,
    processed_directory: str,
    split_file: str,
    threshold: float,
) -> None:
    root = Path(output_root)
    processed = Path(processed_directory)
    comparison_directory = root / "comparisons" / "predictions"
    comparison_directory.mkdir(parents=True, exist_ok=True)

    images_path = processed / "images.npy"
    labels_path = processed / "labels.npy"

    if not images_path.exists():
        raise FileNotFoundError(f"Images not found: {images_path}")

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    images = np.load(images_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")
    test_indices = load_test_indices(Path(split_file))

    if not test_indices:
        raise ValueError("The test split is empty.")

    for test_position, dataset_index in enumerate(test_indices):
        if dataset_index < 0 or dataset_index >= len(images):
            raise IndexError(
                f"Dataset index {dataset_index} is outside the image array "
                f"with length {len(images)}."
            )

        predictions: dict[str, np.ndarray] = {}

        for method in METHODS:
            prediction_path = find_prediction_path(
                output_root=root,
                method=method,
                dataset_index=dataset_index,
                test_position=test_position,
            )

            predictions[method] = normalize_prediction(
                np.load(prediction_path)
            )

        image = np.asarray(images[dataset_index]).squeeze()
        label = np.asarray(labels[dataset_index]).squeeze()
        target = label >= 0.5

        if image.ndim != 2 or label.ndim != 2:
            raise ValueError(
                f"Expected 2D image and label for dataset index "
                f"{dataset_index}, but received image shape {image.shape} "
                f"and label shape {label.shape}."
            )

        for method, prediction in predictions.items():
            if prediction.shape != target.shape:
                raise ValueError(
                    f"{method} prediction shape {prediction.shape} does not "
                    f"match label shape {target.shape} for dataset index "
                    f"{dataset_index}."
                )

        figure, axes = plt.subplots(2, 5, figsize=(18, 8))

        axes[0, 0].imshow(image, cmap="gray")
        axes[0, 0].set_title("Input")

        axes[0, 1].imshow(label, cmap="gray")
        axes[0, 1].set_title("Ground truth")

        for column, method in enumerate(METHODS, start=2):
            axes[0, column].imshow(
                predictions[method],
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
            )
            axes[0, column].set_title(
                f"{method.upper()} probability"
            )

        axes[1, 0].axis("off")
        axes[1, 1].axis("off")

        for column, method in enumerate(METHODS, start=2):
            binary = predictions[method] >= threshold

            overlay = np.zeros(
                (*target.shape, 3),
                dtype=np.float32,
            )

            true_positive = binary & target
            false_positive = binary & ~target
            false_negative = ~binary & target

            overlay[true_positive] = [1.0, 1.0, 1.0]
            overlay[false_positive] = [1.0, 0.0, 0.0]
            overlay[false_negative] = [0.0, 1.0, 0.0]

            axes[1, column].imshow(overlay)
            axes[1, column].set_title(
                f"{method.upper()} errors"
            )

        for axis in axes.ravel():
            axis.axis("off")

        figure.suptitle(
            f"Test slice {dataset_index} "
            f"(test position {test_position})"
        )
        figure.tight_layout()
        figure.savefig(
            comparison_directory
            / f"slice{dataset_index:03d}_comparison.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(figure)

    print(
        f"Saved {len(test_indices)} prediction comparisons to: "
        f"{comparison_directory.resolve()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument(
        "--processed-directory",
        default="data/processed/isbi",
    )
    parser.add_argument(
        "--split-file",
        default="data/splits/isbi_split.json",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    arguments = parser.parse_args()

    main(
        arguments.output_root,
        arguments.processed_directory,
        arguments.split_file,
        arguments.threshold,
    )
