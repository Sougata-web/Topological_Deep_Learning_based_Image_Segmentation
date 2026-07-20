from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.isbidataset import ISBIPatchDataset
from src.engine.trainer import Trainer
from src.losses.dectloss import DECTTopologyLoss
from src.losses.phloss import PersistentHomologyLoss
from src.losses.segmentation import SegmentationLoss
from src.models.unet import UNet
from src.utils.device import (
    configure_cuda,
    get_device,
    print_device_summary,
)
from src.utils.seed import seed_everything


def build_topology_loss(
    method: str,
    config: dict[str, Any],
) -> torch.nn.Module | None:
    if method == "baseline":
        return None

    if method == "dect":
        dect = config["dect"]

        return DECTTopologyLoss(
            number_of_directions=int(dect["num_directions"]),
            number_of_heights=int(dect["num_heights"]),
            direction_chunk_size=int(dect["direction_chunk_size"]),
            sigmoid_steepness=float(dect["sigmoid_steepness"]),
            patch_size=int(dect["patch_size"]),
            maximum_patches=int(dect["max_patches_per_batch"]),
            include_background=bool(dect["include_background"]),
        )

    if method == "ph":
        ph = config["ph"]
        dimensions = tuple(
            int(value)
            for value in ph["dimensions"]
        )

        return PersistentHomologyLoss(
            patch_size=int(ph["patch_size"]),
            maximum_patches=int(ph["max_patches_per_batch"]),
            dimensions=dimensions,
        )

    raise ValueError(f"Unknown training method: {method}")


def make_loader(
    dataset: ISBIPatchDataset,
    batch_size: int,
    number_of_workers: int,
    device: torch.device,
    shuffle: bool,
) -> DataLoader:
    arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": number_of_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": number_of_workers > 0,
        "drop_last": False,
    }

    if number_of_workers > 0:
        arguments["prefetch_factor"] = 2

    return DataLoader(**arguments)


def main(
    config_path: str,
    method: str,
    device_name: str | None,
) -> None:
    config = load_config(config_path)
    seed = int(config["project"]["seed"])

    seed_everything(
        seed,
        deterministic=bool(
            config["training"].get("deterministic", False)
        ),
    )

    device = get_device(device_name)
    configure_cuda(device)
    print_device_summary(device)

    processed_directory = Path(config["data"]["processed_dir"])
    images_path = processed_directory / "images.npy"
    labels_path = processed_directory / "labels.npy"
    split_file = Path(config["data"]["split_file"])

    for path in (images_path, labels_path, split_file):
        if not path.exists():
            raise FileNotFoundError(
                f"Required prepared-data file not found: {path}. "
                "Run scripts/prepareisbi.py first."
            )

    patch_size = int(config["data"]["patch_size"])

    training_dataset = ISBIPatchDataset(
        images_path=images_path,
        labels_path=labels_path,
        split_file=split_file,
        split="train",
        patch_size=patch_size,
        patches_per_epoch=int(
            config["data"]["train_patches_per_epoch"]
        ),
        seed=seed,
        augment=True,
        foreground_sampling_probability=float(
            config["data"]["foreground_sampling_probability"]
        ),
    )

    validation_dataset = ISBIPatchDataset(
        images_path=images_path,
        labels_path=labels_path,
        split_file=split_file,
        split="validation",
        patch_size=patch_size,
        patches_per_epoch=int(
            config["data"]["validation_patches_per_epoch"]
        ),
        seed=seed + 1_000_000,
        augment=False,
        foreground_sampling_probability=0.5,
    )

    batch_size = int(config["training"]["batch_size"])
    number_of_workers = int(config["data"]["num_workers"])

    training_loader = make_loader(
        training_dataset,
        batch_size,
        number_of_workers,
        device,
        shuffle=True,
    )
    validation_loader = make_loader(
        validation_dataset,
        batch_size,
        number_of_workers,
        device,
        shuffle=False,
    )

    model = UNet(
        input_channels=int(config["model"]["in_channels"]),
        output_channels=int(config["model"]["out_channels"]),
        base_channels=int(config["model"]["base_channels"]),
    ).to(device)

    segmentation_loss = SegmentationLoss(
        bce_weight=float(config["loss"]["bce_weight"]),
        dice_weight=float(config["loss"]["dice_weight"]),
    ).to(device)

    topology_loss = build_topology_loss(method, config)

    if topology_loss is not None:
        topology_loss = topology_loss.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(config["training"]["epochs"])),
    )

    trainer = Trainer(
        model=model,
        segmentation_loss=segmentation_loss,
        topology_loss=topology_loss,
        topology_weight=float(config["loss"]["topology_weight"]),
        topology_warmup_epochs=int(
            config["training"]["topology_warmup_epochs"]
        ),
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_amp=bool(config["training"]["amp"]),
        gradient_accumulation_steps=int(
            config["training"]["gradient_accumulation_steps"]
        ),
        output_directory=(
            Path(config["project"]["output_dir"]) / method
        ),
    )

    print(f"Training method: {method}")
    print(f"Device: {device}")

    trainer.fit(
        training_loader=training_loader,
        validation_loader=validation_loader,
        epochs=int(config["training"]["epochs"]),
        patience=int(
            config["training"]["early_stopping_patience"]
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/isbi.yaml",
    )
    parser.add_argument(
        "--method",
        choices=("baseline", "ph", "dect"),
        required=True,
    )
    parser.add_argument("--device", default=None)
    arguments = parser.parse_args()

    main(
        arguments.config,
        arguments.method,
        arguments.device,
    )
