from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(input_channels, output_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            input_channels,
            input_channels // 2,
            kernel_size=2,
            stride=2,
        )
        self.conv = DoubleConv(input_channels, output_channels)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(x)

        difference_y = skip.size(2) - x.size(2)
        difference_x = skip.size(3) - x.size(3)

        x = F.pad(
            x,
            [
                difference_x // 2,
                difference_x - difference_x // 2,
                difference_y // 2,
                difference_y - difference_y // 2,
            ],
        )

        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()

        channels = base_channels
        self.input_block = DoubleConv(input_channels, channels)
        self.down1 = Down(channels, channels * 2)
        self.down2 = Down(channels * 2, channels * 4)
        self.down3 = Down(channels * 4, channels * 8)
        self.down4 = Down(channels * 8, channels * 16)

        self.up1 = Up(channels * 16, channels * 8)
        self.up2 = Up(channels * 8, channels * 4)
        self.up3 = Up(channels * 4, channels * 2)
        self.up4 = Up(channels * 2, channels)

        self.output = nn.Conv2d(channels, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.input_block(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.output(x)
