from __future__ import annotations

import time
from contextlib import ContextDecorator
from types import TracebackType
from typing import Self

import torch


def synchronize(device: torch.device | None = None) -> None:
    if not torch.cuda.is_available():
        return

    if device is None:
        torch.cuda.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


class Timer(ContextDecorator):
    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device
        self.elapsed_seconds = 0.0
        self.start = 0.0

    def __enter__(self) -> Self:
        synchronize(self.device)
        self.start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        synchronize(self.device)
        self.elapsed_seconds = time.perf_counter() - self.start
