"""U-Net generator for phase prediction from dual-camera images."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Double convolution block: Conv-BN-ReLU-Conv-BN-ReLU."""

    def __init__(self, in_ch: int, out_ch: int, mid_ch: int | None = None):
        super().__init__()
        mid = mid_ch or out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetGenerator(nn.Module):
    """U-Net generator for phase prediction from dual-camera images.

    Input: 2-channel image (Daheng + MiiCam) or 1-channel (single camera)
    Output: 1-channel phase map (normalized 0-1, same resolution as input)
    """

    def __init__(
        self,
        in_channels: int = 2,
        features: list[int] | None = None,
        output_mode: Literal["phase", "coeffs"] = "phase",
        n_coeffs: int = 55,
    ):
        """Initialize U-Net generator.

        Args:
            in_channels: Input image channels (1 for single cam, 2 for dual cam).
            features: Feature channels for U-Net encoder.
            output_mode: Output mode - "phase" for 2D map, "coeffs" for 1D vector.
            n_coeffs: Number of Zernike coefficients (only used if output_mode="coeffs").
        """
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512, 1024]

        self.features = features
        self.output_mode = output_mode
        self.n_coeffs = n_coeffs

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (downsampling)
        for i in range(len(features)):
            in_ch = in_channels if i == 0 else features[i - 1]
            out_ch = features[i]
            self.downs.append(DoubleConv(in_ch, out_ch))

        # Decoder (upsampling)
        self.ups.append(
            nn.ConvTranspose2d(features[-1] * 2, features[-1], kernel_size=2, stride=2)
        )
        self.ups.append(DoubleConv(features[-1] * 2, features[-1]))
        for i in reversed(range(len(features) - 1)):
            in_ch = features[i + 1]
            out_ch = features[i]
            self.ups.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(out_ch * 2, out_ch))

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Final output layer
        self.final_conv = nn.Conv2d(features[0], 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # Optional: coefficient output head
        if output_mode == "coeffs":
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.coeff_head = nn.Linear(features[-1] * 2 + features[0], n_coeffs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        bottleneck_feat: torch.Tensor = x

        # Decoder
        skip_connections = skip_connections[::-1]
        up_idx = 0
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skip_connections[up_idx]

            if x.shape != skip.shape:
                x = nn.functional.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=True
                )

            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)
            up_idx += 1

        # Output based on mode
        if self.output_mode == "coeffs":
            pooled = self.global_pool(x).flatten(1)
            bottleneck_pooled = self.global_pool(bottleneck_feat).flatten(1)
            combined = torch.cat([pooled, bottleneck_pooled], dim=1)
            return self.coeff_head(combined)
        else:
            return self.sigmoid(self.final_conv(x))


def build_unet(
    in_channels: int = 2,
    features: list[int] | None = None,
    device: str = "cpu",
) -> UNetGenerator:
    """Build U-Net generator model.

    Args:
        in_channels: Input image channels.
        features: Feature channels for U-Net encoder.
        device: Target device.

    Returns:
        UNetGenerator instance.
    """
    model = UNetGenerator(in_channels=in_channels, features=features)
    return model.to(device)
