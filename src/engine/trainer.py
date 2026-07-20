from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.timing import Timer


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        segmentation_loss: nn.Module,
        topology_loss: nn.Module | None,
        topology_weight: float,
        topology_warmup_epochs: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        device: torch.device,
        use_amp: bool,
        output_directory: str | Path,
        gradient_accumulation_steps: int,

    ) -> None:
        self.model = model
        self.segmentation_loss = segmentation_loss
        self.topology_loss = topology_loss
        self.topology_weight = topology_weight
        self.topology_warmup_epochs = topology_warmup_epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp and device.type == "cuda"
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp,
        )
        self.gradient_accumulation_steps = max(
            1,
            int(gradient_accumulation_steps),
        )


    def topology_scale(self, epoch: int) -> float:
        if self.topology_loss is None:
            return 0.0
        if self.topology_warmup_epochs <= 0:
            return self.topology_weight

        fraction = min(1.0, epoch / self.topology_warmup_epochs)
        return self.topology_weight * fraction

    def train_epoch(
        self,
        loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        self.model.train()
        if hasattr(loader.dataset, "set_epoch"):
            loader.dataset.set_epoch(epoch)
        self.optimizer.zero_grad(set_to_none=True)

        totals = {
            "loss": 0.0,
            "segmentation_loss": 0.0,
            "topology_loss": 0.0,
            "topology_seconds": 0.0,
        }
        number_of_batches = 0
        topology_scale = self.topology_scale(epoch)

        progress = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        topology_scale = self.topology_scale(epoch)
        self.optimizer.zero_grad(set_to_none=True)

        for batch_index, batch in enumerate(progress):
            image = batch["image"].to(
                self.device,
                non_blocking=self.device.type == "cuda",
            )
            target = batch["mask"].to(
                self.device,
                non_blocking=self.device.type == "cuda",
            )

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_amp,
            ):
                logits = self.model(image)
                segmentation_value, _ = self.segmentation_loss(
                    logits,
                    target,
                )

            topology_value = logits.sum() * 0.0

            if self.topology_loss is not None and topology_scale > 0.0:
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=False,
                ):
                    with Timer(self.device) as topology_timer:
                        topology_value = self.topology_loss(
                            logits.float(),
                            target.float(),
                        )

                if not topology_value.requires_grad:
                    raise RuntimeError(
                        "Topology loss is detached from the prediction graph."
                    )

                if not torch.isfinite(topology_value):
                    raise FloatingPointError(
                        f"Non-finite topology loss: {topology_value.item()}."
                    )

                totals["topology_seconds"] += (
                    topology_timer.elapsed_seconds
                )

            loss = (
                segmentation_value
                + topology_scale * topology_value
            )

            scaled_loss = loss / self.gradient_accumulation_steps
            self.scaler.scale(scaled_loss).backward()

            totals["loss"] += float(loss.detach().item())
            totals["segmentation_loss"] += float(
                segmentation_value.detach().item()
            )
            totals["topology_loss"] += float(
                topology_value.detach().item()
            )
            number_of_batches += 1

            should_step = (
                (batch_index + 1) % self.gradient_accumulation_steps == 0
                or (batch_index + 1) == len(loader)
            )

            if should_step:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                if self.scheduler is not None:
                    self.scheduler.step()

        # averages = {
        #     key: value / max(number_of_batches, 1)
        #     for key, value in totals.items()
        # }
        # # Preserve total topology time for efficiency reporting.
        # averages["topology_seconds_total"] = totals["topology_seconds"]
        # return averages
        
        averages = {
            "loss": totals["loss"] / max(number_of_batches, 1),
            "segmentation_loss": (
                totals["segmentation_loss"]
                / max(number_of_batches, 1)
            ),
            "topology_loss": (
                totals["topology_loss"]
                / max(number_of_batches, 1)
            ),
            "topology_seconds": (
                totals["topology_seconds"]
                / max(number_of_batches, 1)
            ),
            "topology_seconds_total": totals["topology_seconds"],
        }
        return averages


    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()

        total_loss = 0.0
        number_of_batches = 0

        for batch in loader:
            image = batch["image"].to(self.device, non_blocking=True)
            target = batch["mask"].to(self.device, non_blocking=True)

            logits = self.model(image)
            loss, _ = self.segmentation_loss(logits, target)

            total_loss += float(loss)
            number_of_batches += 1

        return {
            "validation_loss": total_loss / max(number_of_batches, 1)
        }

    def fit(
        self,
        training_loader: DataLoader,
        validation_loader: DataLoader,
        epochs: int,
        patience: int,
    ) -> None:
        best_validation = float("inf")
        stale_epochs = 0
        log_path = self.output_directory / "history.csv"

        fieldnames = [
            "epoch",
            "loss",
            "segmentation_loss",
            "topology_loss",
            "topology_seconds",
            "topology_seconds_total",
            "validation_loss",
            "epoch_seconds",
        ]

        with log_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()

            for epoch in range(1, epochs + 1):
                with Timer() as epoch_timer:
                    training = self.train_epoch(training_loader, epoch)
                    validation = self.validate(validation_loader)

                row = {
                    "epoch": epoch,
                    **training,
                    **validation,
                    "epoch_seconds": epoch_timer.elapsed_seconds,
                }
                writer.writerow(row)
                handle.flush()

                checkpoint = {
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "validation_loss": validation["validation_loss"],
                    "epoch_seconds": epoch_timer.elapsed_seconds,
                }
                torch.save(
                    checkpoint,
                    self.output_directory / "last.pt",
                )

                if validation["validation_loss"] < best_validation:
                    best_validation = validation["validation_loss"]
                    stale_epochs = 0
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state": self.model.state_dict(),
                            "validation_loss": best_validation,
                        },
                        self.output_directory / "best.pt",
                    )
                else:
                    stale_epochs += 1

                print(
                    f"Epoch {epoch:03d}: "
                    f"train={training['loss']:.5f}, "
                    f"validation={validation['validation_loss']:.5f}, "
                    f"time={epoch_timer.elapsed_seconds:.2f}s"
                )

                if stale_epochs >= patience:
                    print("Early stopping.")
                    break
