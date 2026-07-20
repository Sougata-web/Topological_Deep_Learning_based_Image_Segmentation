from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ("baseline", "ph", "dect")
COLORS = {
    "baseline": "black",
    "ph": "tab:blue",
    "dect": "tab:orange",
}


def find_column(frame: pd.DataFrame, *names: str) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def load_history(output_root: Path, method: str) -> pd.DataFrame:
    path = output_root / method / "history.csv"

    if not path.exists():
        raise FileNotFoundError(f"Training history not found: {path}")

    return pd.read_csv(path)


def load_metrics(output_root: Path, method: str) -> dict[str, float]:
    path = output_root / method / "evaluation" / "metrics.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation metrics not found: {path}\n"
            f"Run: python -m scripts.evaluate "
            f"--config configs/isbi.yaml --method {method} "
            f"--device cuda:0"
        )

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def plot_histories(
    histories: dict[str, pd.DataFrame],
    output_directory: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))

    for method, frame in histories.items():
        epoch_column = find_column(frame, "epoch")
        training_column = find_column(
            frame,
            "segmentation_loss",
            "segmentationloss",
        )
        validation_column = find_column(
            frame,
            "validation_loss",
            "validationloss",
        )
        topology_column = find_column(
            frame,
            "topology_loss",
            "topologyloss",
        )

        if epoch_column is None:
            raise KeyError(f"No epoch column found for {method}.")

        epochs = frame[epoch_column]

        if training_column is not None:
            axes[0].plot(
                epochs,
                frame[training_column],
                label=method.upper(),
                color=COLORS[method],
                linewidth=2,
            )

        if validation_column is not None:
            axes[1].plot(
                epochs,
                frame[validation_column],
                label=method.upper(),
                color=COLORS[method],
                linewidth=2,
            )

        if topology_column is not None and method != "baseline":
            axes[2].plot(
                epochs,
                frame[topology_column],
                label=method.upper(),
                color=COLORS[method],
                linewidth=2,
            )

    axes[0].set_title("Training segmentation loss")
    axes[1].set_title("Validation loss")
    axes[2].set_title("Topology loss")

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.grid(alpha=0.25)
        axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_directory / "training_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_metric_comparison(
    metrics: dict[str, dict[str, float]],
    output_directory: Path,
) -> None:
    preferred_metrics = [
        "dice",
        "iou",
        "bettierror",
        "eulererror",
        "assd",
        "inferencesecondsperslice",
    ]

    available_metrics = [
        metric
        for metric in preferred_metrics
        if all(metric in metrics[method] for method in METHODS)
    ]

    if not available_metrics:
        raise KeyError("No common comparison metrics were found.")

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(16, 9),
    )
    axes = axes.ravel()

    for index, metric in enumerate(available_metrics):
        values = [metrics[method][metric] for method in METHODS]

        axes[index].bar(
            [method.upper() for method in METHODS],
            values,
            color=[COLORS[method] for method in METHODS],
        )
        axes[index].set_title(metric)
        axes[index].grid(axis="y", alpha=0.25)

        for position, value in enumerate(values):
            axes[index].text(
                position,
                value,
                f"{value:.4g}",
                ha="center",
                va="bottom",
            )

    for index in range(len(available_metrics), len(axes)):
        axes[index].axis("off")

    figure.suptitle(
        "Baseline vs PH vs DECT",
        fontsize=16,
    )
    figure.tight_layout()
    figure.savefig(
        output_directory / "metric_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_metric_table(
    metrics: dict[str, dict[str, float]],
    output_directory: Path,
) -> None:
    frame = pd.DataFrame(metrics).T
    frame.index.name = "method"
    frame.to_csv(output_directory / "metrics_comparison.csv")


def main(output_root: str) -> None:
    root = Path(output_root)
    comparison_directory = root / "comparisons"
    comparison_directory.mkdir(parents=True, exist_ok=True)

    histories = {
        method: load_history(root, method)
        for method in METHODS
    }
    metrics = {
        method: load_metrics(root, method)
        for method in METHODS
    }

    plot_histories(histories, comparison_directory)
    plot_metric_comparison(metrics, comparison_directory)
    save_metric_table(metrics, comparison_directory)

    print(f"Saved comparisons to: {comparison_directory.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="outputs",
    )
    arguments = parser.parse_args()
    main(arguments.output_root)
