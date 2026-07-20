from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.augmentations import augment_pair


class ISBIPatchDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        images_path: str | Path,
        labels_path: str | Path,
        split_file: str | Path,
        split: str,
        patch_size: int,
        patches_per_epoch: int,
        seed: int,
        augment: bool,
        foreground_sampling_probability: float = 0.7,
    ) -> None:
        self.images = np.load(images_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")

        if self.images.shape != self.labels.shape:
            raise ValueError("Image and label arrays must have identical shapes.")

        with Path(split_file).open("r", encoding="utf-8") as handle:
            splits = json.load(handle)

        if split not in splits:
            raise KeyError(f"Unknown split: {split}")

        self.indices = np.asarray(splits[split], dtype=np.int64)
        if len(self.indices) == 0:
            raise ValueError(f"Split '{split}' contains no slices.")

        self.patch_size = int(patch_size)
        self.patches_per_epoch = int(patches_per_epoch)
        self.seed = int(seed)
        self.augment = bool(augment)
        self.foreground_sampling_probability = float(
            foreground_sampling_probability
        )

        height, width = self.images.shape[-2:]
        if self.patch_size > min(height, width):
            raise ValueError(
                f"Patch size {self.patch_size} exceeds image shape "
                f"{height}x{width}."
            )

    def __len__(self) -> int:
        return self.patches_per_epoch

    def sample_coordinates(
        self,
        mask: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[int, int]:
        height, width = mask.shape
        size = self.patch_size

        use_foreground = (
            rng.random() < self.foreground_sampling_probability
            and np.any(mask > 0)
        )

        if use_foreground:
            foreground = np.argwhere(mask > 0)
            center_y, center_x = foreground[
                int(rng.integers(0, len(foreground)))
            ]
            y = int(np.clip(center_y - size // 2, 0, height - size))
            x = int(np.clip(center_x - size // 2, 0, width - size))
        else:
            y = int(rng.integers(0, height - size + 1))
            x = int(rng.integers(0, width - size + 1))

        return y, x

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        # Include worker seed so workers do not reproduce identical streams.
        worker = torch.utils.data.get_worker_info()
        worker_seed = 0 if worker is None else worker.seed
        rng = np.random.default_rng(self.seed + item + worker_seed)

        slice_index = int(
            self.indices[int(rng.integers(0, len(self.indices)))]
        )
        image = np.asarray(self.images[slice_index], dtype=np.float32)
        mask = np.asarray(self.labels[slice_index], dtype=np.float32)

        y, x = self.sample_coordinates(mask, rng)
        size = self.patch_size

        image_patch = image[y:y + size, x:x + size]
        mask_patch = mask[y:y + size, x:x + size]

        if self.augment:
            image_patch, mask_patch = augment_pair(
                image_patch, mask_patch, rng
            )

        return {
            "image": torch.from_numpy(image_patch[None]).float(),
            "mask": torch.from_numpy(mask_patch[None]).float(),
            "slice_index": torch.tensor(slice_index, dtype=torch.long),
        }


class ISBISliceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        images_path: str | Path,
        labels_path: str | Path,
        split_file: str | Path,
        split: str,
    ) -> None:
        self.images = np.load(images_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")

        with Path(split_file).open("r", encoding="utf-8") as handle:
            splits = json.load(handle)

        if split not in splits:
            raise KeyError(f"Unknown split: {split}")

        self.indices = list(splits[split])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        image = np.asarray(self.images[index], dtype=np.float32)
        mask = np.asarray(self.labels[index], dtype=np.float32)

        return {
            "image": torch.from_numpy(image[None]).float(),
            "mask": torch.from_numpy(mask[None]).float(),
            "slice_index": torch.tensor(index, dtype=torch.long),
        }
