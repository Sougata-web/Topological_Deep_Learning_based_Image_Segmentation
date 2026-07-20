from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.isbidataset import ISBISliceDataset
from src.engine.evaluator import evaluate_model
from src.models.unet import UNet
from src.utils.seed import seed_everything

from src.utils.device import (
    configure_cuda,
    get_device,
    print_device_summary,
)



def main(config_path: str, method: str, device_name: str | None) -> None:
    config = load_config(config_path)
    seed_everything(int(config["project"]["seed"]))

    if device_name is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device_name)
        
    device = get_device(device_name)
    configure_cuda(device)
    print_device_summary(device)


    processed_dir = Path(config["data"]["processed_dir"])
    dataset = ISBISliceDataset(
        images_path=processed_dir / "images.npy",
        labels_path=processed_dir / "labels.npy",
        split_file=config["data"]["split_file"],
        split="test",
    )

    number_of_workers = int(config["data"]["num_workers"])

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=number_of_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=number_of_workers > 0,
    )


    model = UNet(
        input_channels=int(config["model"]["in_channels"]),
        output_channels=int(config["model"]["out_channels"]),
        base_channels=int(config["model"]["base_channels"]),
    ).to(device)

    output_directory = (
        Path(config["project"]["output_dir"]) / method
    )
    checkpoint_path = output_directory / "best.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state"])
    state = checkpoint.get("model_state", checkpoint.get("modelstate"))

    if state is None:
        raise KeyError("Checkpoint contains no model state.")

    model.load_state_dict(state)


    summary = evaluate_model(
        model=model,
        loader=loader,
        device=device,
        output_directory=output_directory / "evaluation",
        threshold=float(config["training"]["threshold"]),
        topology_patch_size=int(
            config["evaluation"]["topology_patch_size"]
        ),
        topology_stride=int(config["evaluation"]["topology_stride"]),
        save_predictions=bool(
            config["evaluation"]["save_predictions"]
        ),
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/isbi.yaml")
    parser.add_argument(
        "--method",
        choices=("baseline", "ph", "dect"),
        required=True,
    )
    parser.add_argument("--device", default=None)
    arguments = parser.parse_args()
    main(arguments.config, arguments.method, arguments.device)
