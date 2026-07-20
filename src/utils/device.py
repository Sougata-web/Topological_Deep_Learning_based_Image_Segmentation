from __future__ import annotations

import torch


def get_device(device_name: str | None = None) -> torch.device:
    if device_name is not None:
        if device_name == "cuda":
            device_name = "cuda:0"

        device = torch.device(device_name)

        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but this PyTorch installation cannot "
                "access CUDA. Install a CUDA-enabled PyTorch build."
            )

        return device

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA device is available. Check the NVIDIA driver and "
            "install a CUDA-enabled PyTorch build."
        )

    return torch.device("cuda:0")


def configure_cuda(device: torch.device) -> None:
    if device.type != "cuda":
        return
    if device.index is None:
        device=torch.device("cuda:0")

    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def print_device_summary(device: torch.device) -> None:
    if device.type != "cuda":
        print(f"Device: {device}")
        return

    properties = torch.cuda.get_device_properties(device)
    memory_gib = properties.total_memory / (1024**3)

    print(f"Device: {device}")
    print(f"GPU: {properties.name}")
    print(f"GPU memory: {memory_gib:.2f} GiB")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
