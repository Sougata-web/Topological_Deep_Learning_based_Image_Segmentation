from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def extract_center_patches(
    tensor: torch.Tensor,
    patch_size: int,
    maximum_patches: int,
) -> torch.Tensor:
    _, _, height, width = tensor.shape

    if patch_size > min(height, width):
        raise ValueError(
            f"Topology patch size {patch_size} exceeds tensor shape "
            f"{height}x{width}."
        )
    if maximum_patches <= 0:
        raise ValueError("maximum_patches must be positive.")

    patches = tensor.unfold(2, patch_size, patch_size)
    patches = patches.unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5)
    patches = patches.reshape(-1, tensor.size(1), patch_size, patch_size)

    # Unfold always yields at least one patch after the size check.
    return patches[:maximum_patches]


class DifferentiableECT2D(nn.Module):
    def __init__(
        self,
        number_of_directions: int = 16,
        number_of_heights: int = 32,
        sigmoid_steepness: float = 25.0,
        direction_chunk_size: int = 4,
        learnable_directions: bool = False,
    ) -> None:
        super().__init__()

        angles = torch.linspace(
            0.0,
            2.0 * math.pi,
            steps=number_of_directions + 1,
        )[:-1]

        directions = torch.stack(
            [torch.cos(angles), torch.sin(angles)],
            dim=1,
        )

        if learnable_directions:
            self.directions = nn.Parameter(directions)
        else:
            self.register_buffer("directions", directions)

        self.register_buffer(
            "heights",
            torch.linspace(-1.5, 1.5, number_of_heights),
        )

        self.sigmoid_steepness = float(sigmoid_steepness)
        self.direction_chunk_size = max(
            1,
            int(direction_chunk_size),
        )


    @staticmethod
    def extend_to_cells(
        vertex_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        horizontal_edges = torch.maximum(
            vertex_values[:, :, :, :-1],
            vertex_values[:, :, :, 1:],
        )
        vertical_edges = torch.maximum(
            vertex_values[:, :, :-1, :],
            vertex_values[:, :, 1:, :],
        )
        squares = torch.maximum(
            torch.maximum(
                vertex_values[:, :, :-1, :-1],
                vertex_values[:, :, 1:, :-1],
            ),
            torch.maximum(
                vertex_values[:, :, :-1, 1:],
                vertex_values[:, :, 1:, 1:],
            ),
        )

        return vertex_values, horizontal_edges, vertical_edges, squares

    def smooth_count(self, values: torch.Tensor) -> torch.Tensor:
        heights = self.heights.view(1, 1, -1, 1, 1)
        values = values.unsqueeze(2)

        membership = torch.sigmoid(
            self.sigmoid_steepness * (heights - values)
        )
        return membership.sum(dim=(-1, -2))


    def forward(self, probability_map: torch.Tensor) -> torch.Tensor:
        if probability_map.ndim != 4 or probability_map.size(1) != 1:
            raise ValueError(
                "Expected probability maps shaped [B, 1, H, W]."
            )

        _, _, height, width = probability_map.shape
        device = probability_map.device
        dtype = probability_map.dtype

        y = torch.linspace(
            -1.0,
            1.0,
            height,
            device=device,
            dtype=dtype,
        )
        x = torch.linspace(
            -1.0,
            1.0,
            width,
            device=device,
            dtype=dtype,
        )

        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack([grid_x, grid_y], dim=-1)

        directions = F.normalize(
            self.directions.to(device=device, dtype=dtype),
            dim=1,
        )

        chunks: list[torch.Tensor] = []

        for start in range(
            0,
            directions.size(0),
            self.direction_chunk_size,
        ):
            direction_chunk = directions[
                start:start + self.direction_chunk_size
            ]

            directional_height = torch.einsum(
                "hwc,dc->dhw",
                coordinates,
                direction_chunk,
            )

            filtration_vertices = (
                directional_height.unsqueeze(0)
                + 0.5 * (1.0 - probability_map)
            )

            vertices, horizontal, vertical, squares = (
                self.extend_to_cells(filtration_vertices)
            )

            chunk_ect = (
                self.smooth_count(vertices)
                - self.smooth_count(horizontal)
                - self.smooth_count(vertical)
                + self.smooth_count(squares)
            )
            chunks.append(chunk_ect)

        return torch.cat(chunks, dim=1)



class DECTTopologyLoss(nn.Module):
    def __init__(
        self,
        number_of_directions: int,
        number_of_heights: int,
        sigmoid_steepness: float,
        direction_chunk_size: int,
        patch_size: int,
        maximum_patches: int,
        include_background: bool = True,
    ) -> None:
        super().__init__()

        self.transform = DifferentiableECT2D(
            number_of_directions=number_of_directions,
            number_of_heights=number_of_heights,
            sigmoid_steepness=sigmoid_steepness,
            direction_chunk_size=direction_chunk_size,
        )
        self.patch_size = int(patch_size)
        self.maximum_patches = int(maximum_patches)
        self.include_background = bool(include_background)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
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

        predicted_ect = self.transform(prediction_patches)
        target_ect = self.transform(target_patches)
        loss = F.mse_loss(predicted_ect, target_ect)

        if self.include_background:
            predicted_background_ect = self.transform(
                1.0 - prediction_patches
            )
            target_background_ect = self.transform(
                1.0 - target_patches
            )
            loss = 0.5 * (
                    loss
                    + F.mse_loss(
                        predicted_background_ect,
                        target_background_ect,
                    )
                )

        normalizer = float(self.patch_size * self.patch_size)
        return loss / (normalizer * normalizer)
