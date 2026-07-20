from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from src.config import loadconfig
from src.losses.dectloss import DECTTopologyLoss
from src.losses.phloss import (
    GudhiCubicalPHBackend,
    PersistentHomologyLoss,
)
from src.utils.seed import seed_everything
from src.utils.timing import Timer, synchronize


def build_loss(method: str, config: dict) -> nn.Module:
    if method == "dect":
        dect = config["dect"]
        return DECTTopologyLoss(
            number_of_directions=int(dect["numdirections"]),
            number_of_heights=int(dect["numheights"]),
            sigmoid_steepness=float(dect["sigmoidsteepness"]),
            patch_size=int(dect["patchsize"]),
            maximum_patches=int(dect["maxpatchesperbatch"]),
            include_background=bool(dect["includebackground"]),
        )

    if method == "ph":
        ph = config["ph"]
        dimensions = tuple(int(value) for value in ph["dimensions"])
        return PersistentHomologyLoss(
            backend_factory=lambda: GudhiCubicalPHBackend(dimensions),
            patch_size=int(ph["patchsize"]),
            maximum_patches=int(ph["maxpatchesperbatch"]),
        )

    raise ValueError(f"Unknown topology method: {method}")


def benchmark(
    operation: Callable[[], None],
    warmup_iterations: int,
    iterations: int,
) -> list[float]:
    for _ in range(warmup_iterations):
        operation()

    synchronize()
    timings: list[float] = []

    for _ in range(iterations):
        with Timer() as timer:
            operation()
        timings.append(timer.elapsed_seconds)

    return timings


def main(
    config_path: str,
    method: str,
    device_name: str | None,
) -> None:
    config = loadconfig(config_path)
    seed_everything(int(config["project"]["seed"]))

    if device_name is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device_name)
        
    


    loss_module = build_loss(method, config).to(device)
    patch_size = int(config[method]["patchsize"])
    batch_size = int(config["training"]["batchsize"])

    base_logits = torch.randn(
        batch_size,
        1,
        patch_size,
        patch_size,
        device=device,
    )
    target = (
        torch.rand(
            batch_size,
            1,
            patch_size,
            patch_size,
            device=device,
        )
        > 0.5
    ).float()

    def operation() -> None:
        logits = base_logits.detach().clone().requires_grad_(True)
        loss = loss_module(logits, target)
        loss.backward()

    timings = benchmark(
        operation,
        warmup_iterations=int(config["benchmark"]["warmupiterations"]),
        iterations=int(config["benchmark"]["iterations"]),
    )

    summary = {
        "method": method,
        "device": str(device),
        "batchsize": batch_size,
        "patchsize": patch_size,
        "iterations": len(timings),
        "meanseconds": float(np.mean(timings)),
        "stdseconds": float(np.std(timings)),
        "medianseconds": float(np.median(timings)),
        "minimumseconds": float(np.min(timings)),
        "maximumseconds": float(np.max(timings)),
    }
    if device.type == "cuda":
        summary["peak_memory_mib"] = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
        )
        summary["peak_reserved_memory_mib"] = (
            torch.cuda.max_memory_reserved(device) / (1024**2)
        )
    output_directory = (
        Path(config["project"]["outputdir"]) / "comparisons"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{method}_topology_benchmark.json"

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/isbi.yaml")
    parser.add_argument(
        "--method",
        choices=("ph", "dect"),
        required=True,
    )
    parser.add_argument("--device", default=None)
    arguments = parser.parse_args()
    main(arguments.config, arguments.method, arguments.device)
