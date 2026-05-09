"""PatchGAN discriminator for phase prediction."""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator for phase prediction.

    Input: concatenated (image, phase) or just phase
    Output: patch-based real/fake predictions
    """

    def __init__(
        self,
        in_channels: int = 3,  # 2 (dual cam) + 1 (phase) or 1+1
        num_features: list[int] | None = None,
    ):
        super().__init__()
        if num_features is None:
            num_features = [64, 128, 256, 512]

        layers = nn.ModuleList()

        # First layer: no batch norm
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, num_features[0], 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )
        )

        # Middle layers
        for i in range(1, len(num_features)):
            in_f = num_features[i - 1]
            out_f = num_features[i]
            stride = 2 if i < len(num_features) - 1 else 1
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_f, out_f, 4, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(out_f),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )

        # Output layer
        layers.append(nn.Conv2d(num_features[-1], 1, 4, stride=1, padding=1))

        self.layers = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def build_discriminator(
    in_channels: int = 3,
    num_features: list[int] | None = None,
    device: str = "cpu",
) -> PatchGANDiscriminator:
    """Build PatchGAN discriminator model.

    Args:
        in_channels: Input channels.
        num_features: Feature channels.
        device: Target device.

    Returns:
        PatchGANDiscriminator instance.
    """
    model = PatchGANDiscriminator(in_channels=in_channels, num_features=num_features)
    return model.to(device)
