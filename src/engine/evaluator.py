from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.metrics.segmentation import (
    average_symmetric_surface_distance,
    binary_segmentation_metrics,
    partition_metrics,
)
from src.metrics.topology import topology_metrics
from src.utils.timing import Timer
from src.utils.visualization import save_prediction_figure


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]

    if not finite:
        return 0.0

    return float(np.mean(finite))


# @torch.no_grad()
@torch.inference_mode()

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_directory: str | Path,
    threshold: float,
    topology_patch_size: int,
    topology_stride: int,
    save_predictions: bool = True,
) -> dict[str, float]:
    model.eval()
    output_directory = Path(output_directory)
    prediction_directory = output_directory / "predictions"
    prediction_directory.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict[str, float]] = []
    inference_times: list[float] = []

    for batch in loader:
        image = batch["image"].to(device,non_blocking=device.type=="cuda")
        target = batch["mask"].cpu().numpy()[0, 0]
        slice_index = int(batch["slice_index"].item())

        with Timer(device) as timer:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(image)
                probability = torch.sigmoid(logits)

        inference_times.append(timer.elapsed_seconds)


        probability_array = (
            probability.float().cpu().numpy()[0, 0]
        )

        binary_prediction = probability_array >= threshold
        binary_target = target >= 0.5

        metrics = binary_segmentation_metrics(
            probability_array,
            target,
            threshold,
        )
        metrics.update(partition_metrics(binary_prediction, binary_target))
        metrics.update(
            topology_metrics(
                binary_prediction,
                binary_target,
                patch_size=topology_patch_size,
                stride=topology_stride,
            )
        )
        metrics["assd"] = average_symmetric_surface_distance(
            binary_prediction,
            binary_target,
        )
        all_metrics.append(metrics)

        if save_predictions:
            np.save(
                prediction_directory / f"slice_{slice_index:03d}.npy",
                probability_array,
            )
            save_prediction_figure(
                image.cpu().numpy()[0, 0],
                target,
                probability_array,
                prediction_directory / f"slice_{slice_index:03d}.png",
                threshold,
            )

    if not all_metrics:
        raise ValueError("Evaluation loader produced no samples.")

    summary: dict[str, float] = {}
    for key in all_metrics[0]:
        values = [metrics[key] for metrics in all_metrics]
        summary[key] = finite_mean(values)

    summary["inference_seconds_per_slice"] = float(
        np.mean(inference_times)
    )
    summary["inference_seconds_std"] = float(np.std(inference_times))
    summary["evaluated_slices"] = float(len(all_metrics))
    summary["assd_nonfinite_slices"] = float(
        sum(
            not math.isfinite(metrics["assd"])
            for metrics in all_metrics
        )
    )

    with (output_directory / "metrics.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)

    return summary
