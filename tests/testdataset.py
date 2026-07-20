from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.data.isbidataset import ISBIPatchDataset, ISBISliceDataset


@pytest.fixture()
def dataset_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    images = np.linspace(
        0.0,
        1.0,
        num=4 * 32 * 32,
        dtype=np.float32,
    ).reshape(4, 32, 32)

    labels = np.zeros_like(images)
    labels[:, 10:22, 12:24] = 1.0

    images_path = tmp_path / "images.npy"
    labels_path = tmp_path / "labels.npy"
    split_path = tmp_path / "split.json"

    np.save(images_path, images)
    np.save(labels_path, labels)

    with split_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "train": [0, 1],
                "validation": [2],
                "test": [3],
            },
            handle,
        )

    return images_path, labels_path, split_path


def test_patch_dataset_shapes(
    dataset_files: tuple[Path, Path, Path],
) -> None:
    images_path, labels_path, split_path = dataset_files

    dataset = ISBIPatchDataset(
        images_path=images_path,
        labels_path=labels_path,
        split_file=split_path,
        split="train",
        patch_size=16,
        patches_per_epoch=10,
        seed=42,
        augment=False,
        foreground_sampling_probability=1.0,
    )

    sample = dataset[0]

    assert len(dataset) == 10
    assert sample["image"].shape == (1, 16, 16)
    assert sample["mask"].shape == (1, 16, 16)
    assert sample["image"].dtype.is_floating_point
    assert sample["mask"].dtype.is_floating_point
    assert int(sample["sliceindex"]) in {0, 1}
    assert sample["mask"].sum() > 0


def test_patch_sampling_is_deterministic(
    dataset_files: tuple[Path, Path, Path],
) -> None:
    images_path, labels_path, split_path = dataset_files

    dataset = ISBIPatchDataset(
        images_path=images_path,
        labels_path=labels_path,
        split_file=split_path,
        split="train",
        patch_size=16,
        patches_per_epoch=5,
        seed=7,
        augment=True,
    )

    first = dataset[3]
    second = dataset[3]

    assert np.array_equal(first["image"].numpy(), second["image"].numpy())
    assert np.array_equal(first["mask"].numpy(), second["mask"].numpy())


def test_slice_dataset(
    dataset_files: tuple[Path, Path, Path],
) -> None:
    images_path, labels_path, split_path = dataset_files

    dataset = ISBISliceDataset(
        images_path=images_path,
        labels_path=labels_path,
        split_file=split_path,
        split="test",
    )

    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["image"].shape == (1, 32, 32)
    assert sample["mask"].shape == (1, 32, 32)
    assert int(sample["sliceindex"]) == 3


def test_oversized_patch_is_rejected(
    dataset_files: tuple[Path, Path, Path],
) -> None:
    images_path, labels_path, split_path = dataset_files

    with pytest.raises(ValueError, match="exceeds image shape"):
        ISBIPatchDataset(
            images_path=images_path,
            labels_path=labels_path,
            split_file=split_path,
            split="train",
            patch_size=64,
            patches_per_epoch=5,
            seed=1,
            augment=False,
        )
