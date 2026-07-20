from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

try:
    from torch_topological.nn import CubicalComplex
except ImportError as error:
    raise ImportError(
        "The PH model requires torch-topological. Install it with "
        "'python -m pip install torch-topological'."
    ) from error

from src.losses.dectloss import extract_center_patches


class TorchTopologicalCubicalPHBackend(nn.Module):
    """
    Differentiable cubical persistent-homology surrogate.

    Finite persistence points are compared independently in each requested
    homology dimension. Points are sorted by persistence and paired in that
    order. Unmatched points are penalized by squared distance to the diagonal.

    This is persistence-sorted matching, not optimal Wasserstein matching and
    not the exact critical-point matching procedure of Hu et al.
    """

    def __init__(
        self,
        dimensions: tuple[int, ...] = (0, 1),
    ) -> None:
        super().__init__()

        if not dimensions:
            raise ValueError("At least one homology dimension is required.")

        if any(dimension < 0 for dimension in dimensions):
            raise ValueError("Homology dimensions must be non-negative.")

        self.dimensions = tuple(sorted(set(int(d) for d in dimensions)))
        self.ph_layer = CubicalComplex(dim=max(self.dimensions))

    @staticmethod
    def _flatten_output(output: Any) -> list[Any]:
        """Flatten nested torch-topological outputs without flattening tensors."""
        if hasattr(output, "diagram"):
            return [output]

        if isinstance(output, torch.Tensor):
            return [output]

        if isinstance(output, Sequence) and not isinstance(
            output,
            (str, bytes),
        ):
            flattened: list[Any] = []

            for item in output:
                flattened.extend(
                    TorchTopologicalCubicalPHBackend._flatten_output(item)
                )

            return flattened

        raise TypeError(
            "Unexpected CubicalComplex output type: "
            f"{type(output).__name__}."
        )

    @staticmethod
    def _validate_diagram(diagram: torch.Tensor) -> torch.Tensor:
        if not isinstance(diagram, torch.Tensor):
            raise TypeError("Persistence diagrams must be torch.Tensor values.")

        if diagram.numel() == 0:
            return diagram.reshape(0, 2)

        if diagram.ndim != 2 or diagram.shape[-1] != 2:
            raise ValueError(
                "Expected a persistence diagram shaped [N, 2], got "
                f"{tuple(diagram.shape)}."
            )

        return diagram

    @classmethod
    def _extract_diagrams(
        cls,
        output: Any,
    ) -> dict[int, torch.Tensor]:
        """
        Extract diagrams grouped by homology dimension.

        torch-topological versions may represent dimension as:
        - one Python integer for the complete diagram;
        - one scalar tensor;
        - a tensor containing one dimension value per diagram point.
        """
        grouped: dict[int, list[torch.Tensor]] = {}
        tensor_fallback_dimension = 0

        for item in cls._flatten_output(output):
            if isinstance(item, torch.Tensor):
                diagram = cls._validate_diagram(item)
                grouped.setdefault(
                    tensor_fallback_dimension,
                    [],
                ).append(diagram)
                tensor_fallback_dimension += 1
                continue

            if not hasattr(item, "diagram"):
                raise TypeError(
                    "PersistenceInformation output has no 'diagram' field."
                )

            diagram = cls._validate_diagram(item.diagram)

            if not hasattr(item, "dimension"):
                raise TypeError(
                    "PersistenceInformation output has no 'dimension' field."
                )

            dimension = item.dimension

            if isinstance(dimension, torch.Tensor):
                dimension = dimension.reshape(-1)

                if dimension.numel() == 0:
                    continue

                if dimension.numel() == 1:
                    homology_dimension = int(dimension.item())
                    grouped.setdefault(
                        homology_dimension,
                        [],
                    ).append(diagram)
                    continue

                if dimension.numel() != diagram.shape[0]:
                    raise ValueError(
                        "The number of dimension labels does not match the "
                        "number of persistence points: "
                        f"{dimension.numel()} vs {diagram.shape[0]}."
                    )

                for homology_dimension_tensor in torch.unique(dimension):
                    homology_dimension = int(
                        homology_dimension_tensor.item()
                    )
                    mask = dimension == homology_dimension_tensor

                    grouped.setdefault(
                        homology_dimension,
                        [],
                    ).append(diagram[mask])

                continue

            homology_dimension = int(dimension)
            grouped.setdefault(homology_dimension, []).append(diagram)

        merged: dict[int, torch.Tensor] = {}

        for dimension, diagrams in grouped.items():
            if len(diagrams) == 1:
                merged[dimension] = diagrams[0]
            else:
                merged[dimension] = torch.cat(diagrams, dim=0)

        return merged

    @staticmethod
    def _finite_points(diagram: torch.Tensor) -> torch.Tensor:
        diagram = TorchTopologicalCubicalPHBackend._validate_diagram(
            diagram
        )

        if diagram.numel() == 0:
            return diagram

        return diagram[torch.isfinite(diagram).all(dim=1)]

    @staticmethod
    def _distance_to_diagonal_squared(
        diagram: torch.Tensor,
    ) -> torch.Tensor:
        if diagram.numel() == 0:
            return diagram.sum()

        persistence = diagram[:, 1] - diagram[:, 0]

        # Squared Euclidean distance from (birth, death) to y = x.
        return 0.5 * persistence.square().sum()

    @classmethod
    def diagram_distance(
        cls,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        prediction = cls._finite_points(prediction)
        target = cls._finite_points(target)

        if prediction.numel() == 0 and target.numel() == 0:
            return prediction.sum()

        if prediction.numel() == 0:
            # This term cannot produce a prediction gradient because there
            # are no finite predicted features to move.
            return cls._distance_to_diagonal_squared(target)

        if target.numel() == 0:
            return cls._distance_to_diagonal_squared(prediction)

        prediction_persistence = (
            prediction[:, 1] - prediction[:, 0]
        )
        target_persistence = target[:, 1] - target[:, 0]

        prediction_order = torch.argsort(
            prediction_persistence,
            descending=True,
        )
        target_order = torch.argsort(
            target_persistence,
            descending=True,
        )

        prediction = prediction[prediction_order]
        target = target[target_order]

        matched_count = min(
            prediction.shape[0],
            target.shape[0],
        )

        matched_loss = F.mse_loss(
            prediction[:matched_count],
            target[:matched_count],
            reduction="sum",
        )

        unmatched_prediction_loss = (
            cls._distance_to_diagonal_squared(
                prediction[matched_count:]
            )
        )
        unmatched_target_loss = cls._distance_to_diagonal_squared(
            target[matched_count:]
        )

        normalizer = float(
            max(
                prediction.shape[0],
                target.shape[0],
                1,
            )
        )

        return (
            matched_loss
            + unmatched_prediction_loss
            + unmatched_target_loss
        ) / normalizer

    def _sample_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        prediction_output = self.ph_layer(prediction)

        # Target topology is constant and needs no backward graph.
        with torch.no_grad():
            target_output = self.ph_layer(target)

        prediction_diagrams = self._extract_diagrams(
            prediction_output
        )
        target_diagrams = self._extract_diagrams(target_output)

        # This guarantees a graph-connected zero if no finite diagrams exist.
        total = prediction.sum() * 0.0

        for dimension in self.dimensions:
            prediction_diagram = prediction_diagrams.get(dimension)

            if prediction_diagram is None:
                prediction_diagram = prediction.new_empty((0, 2))

            target_diagram = target_diagrams.get(dimension)

            if target_diagram is None:
                target_diagram = target.new_empty((0, 2))

            total = total + self.diagram_distance(
                prediction_diagram,
                target_diagram,
            )

        return total / float(len(self.dimensions))

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                "Prediction and target filtrations must have identical "
                f"shapes, got {tuple(prediction.shape)} and "
                f"{tuple(target.shape)}."
            )

        if prediction.ndim == 4:
            if prediction.shape[1] != 1:
                raise ValueError(
                    "Expected a single channel in [B, 1, H, W]."
                )

            prediction = prediction[:, 0]
            target = target[:, 0]

        elif prediction.ndim != 3:
            raise ValueError(
                "Expected [B, H, W] or [B, 1, H, W] filtrations."
            )

        if prediction.shape[0] == 0:
            return prediction.sum() * 0.0

        sample_losses = [
            self._sample_loss(
                prediction[index],
                target[index],
            )
            for index in range(prediction.shape[0])
        ]

        return torch.stack(sample_losses).mean()


class PersistentHomologyLoss(nn.Module):
    def __init__(
        self,
        patch_size: int,
        maximum_patches: int,
        dimensions: tuple[int, ...] = (0, 1),
    ) -> None:
        super().__init__()

        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")

        if maximum_patches <= 0:
            raise ValueError("maximum_patches must be positive.")

        self.backend = TorchTopologicalCubicalPHBackend(
            dimensions=dimensions,
        )
        self.patch_size = int(patch_size)
        self.maximum_patches = int(maximum_patches)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if logits.shape != target.shape:
            raise ValueError(
                "Logits and target must have identical shapes, got "
                f"{tuple(logits.shape)} and {tuple(target.shape)}."
            )

        prediction = torch.sigmoid(logits)

        prediction_patches = extract_center_patches(
            prediction,
            self.patch_size,
            self.maximum_patches,
        )
        target_patches = extract_center_patches(
            target,
            self.patch_size,
            self.maximum_patches,
        )

        if prediction_patches.shape != target_patches.shape:
            raise RuntimeError(
                "Prediction and target topology patch shapes differ."
            )

        # Foreground superlevel filtration represented as a lower-star
        # sublevel filtration.
        prediction_filtration = 1.0 - prediction_patches
        target_filtration = 1.0 - target_patches

        loss = self.backend(
            prediction_filtration,
            target_filtration,
        )

        if loss.ndim != 0:
            raise RuntimeError(
                f"PH backend returned shape {tuple(loss.shape)}, "
                "but a scalar loss is required."
            )

        if not torch.isfinite(loss):
            raise FloatingPointError("PH loss is non-finite.")

        return loss
