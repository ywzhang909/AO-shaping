"""Phase prediction models: U-Net generator + PatchGAN discriminator."""

from __future__ import annotations

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
    ):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512, 1024]

        self.features = features
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (downsampling)
        for i in range(len(features)):
            in_ch = in_channels if i == 0 else features[i - 1]
            out_ch = features[i]
            self.downs.append(DoubleConv(in_ch, out_ch))

        # Decoder (upsampling)
        # The skip connections from encoder are: [features[0], features[1], ..., features[-1]]
        # After reversal, decoder processes from deepest to shallowest.
        # Bottleneck output: features[-1]*2 channels
        # First up-conv: features[-1]*2 -> features[-1], concat with skip (features[-1])
        # Then: features[-1]*2 -> features[-1] via DoubleConv
        # Subsequent: features[i+1] -> features[i], concat with skip (features[i])
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        up_idx = 0
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skip_connections[up_idx]

            # Handle size mismatch
            if x.shape != skip.shape:
                x = nn.functional.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=True
                )

            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)
            up_idx += 1

        return self.sigmoid(self.final_conv(x))


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


def build_model(
    in_channels: int = 2,
    unet_features: list[int] | None = None,
    disc_features: list[int] | None = None,
    device: str = "cpu",
) -> tuple[UNetGenerator, PatchGANDiscriminator]:
    """Build generator and discriminator models.

    Args:
        in_channels: Input image channels (1 for single cam, 2 for dual cam).
        unet_features: Feature channels for U-Net encoder.
        disc_features: Feature channels for discriminator.
        device: Target device.

    Returns:
        Tuple of (generator, discriminator).
    """
    generator = UNetGenerator(in_channels=in_channels, features=unet_features).to(
        device
    )
    discriminator = PatchGANDiscriminator(
        in_channels=in_channels + 1, num_features=disc_features
    ).to(device)
    return generator, discriminator
