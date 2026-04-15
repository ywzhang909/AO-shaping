"""Phase prediction models: U-Net generator + PatchGAN discriminator + Zernike regression."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights


# =============================================================================
# Abstract Base Class
# =============================================================================


class BasePhasePredictor(ABC, nn.Module):
    """Abstract base class for phase predictors.

    Provides common interface for all phase prediction models:
    - Input: Image tensor (B, C, H, W)
    - Output: Zernike coefficients (B, n_coeffs) or phase map (B, 1, H, W)
    """

    def __init__(self, in_channels: int = 1, n_zernike_terms: int = 55):
        """Initialize base predictor.

        Args:
            in_channels: Number of input channels (1 for single, 2 for dual camera).
            n_zernike_terms: Number of Zernike terms to predict (default 55 for n_max=10).
        """
        super().__init__()
        self.in_channels = in_channels
        self.n_zernike_terms = n_zernike_terms

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor - coefficients (B, n_coeffs) or phase (B, 1, H, W).
        """
        pass

    def get_output_dim(self) -> int:
        """Get output dimension (n_zernike_terms for regression, None for phase)."""
        return self.n_zernike_terms


# =============================================================================
# Original U-Net Generator (Legacy - Phase Map Output)
# =============================================================================


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

    Note: Legacy model for 2D phase map prediction. For Zernike coefficient
    prediction, use ResNetRegression or SimpleCNNRegression.
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
            # Global average pooling to get feature vector
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            # Feature dimension: features[-1] * 2 (bottleneck) + features[0] (skip)
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
        if self.output_mode == "coeffs":
            bottleneck_feat = x  # Store for coefficient head

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

        # Output based on mode
        if self.output_mode == "coeffs":
            # Global pooling of decoder output
            pooled = self.global_pool(x).flatten(1)
            # Combine with bottleneck features
            bottleneck_pooled = self.global_pool(bottleneck_feat).flatten(1)
            combined = torch.cat([pooled, bottleneck_pooled], dim=1)
            return self.coeff_head(combined)
        else:
            return self.sigmoid(self.final_conv(x))


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator for phase prediction.

    Input: concatenated (image, phase) or just phase
    Output: patch-based real/fake predictions

    Note: Legacy model for GAN training.
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


# =============================================================================
# ResNet-based Regression Model (Zernike Coefficients)
# =============================================================================


class ResNetRegression(BasePhasePredictor):
    """ResNet backbone + FC head for Zernike coefficient prediction.

    Uses pretrained ResNet18/34 as encoder (removes final FC layer).
    Adds adaptive pooling to handle different input sizes.
    FC head outputs n_zernike_terms coefficients.

    Architecture:
        Input (B, C, H, W) -> ResNet Encoder -> Global Avg Pool ->
        FC(512, n_coeffs) -> Output (B, n_coeffs)

    Args:
        in_channels: Number of input channels (1 for single, 2 for dual camera).
        n_zernike_terms: Number of Zernike terms to predict (default 55 for n_max=10).
        backbone: ResNet backbone - "resnet18" or "resnet34".
        pretrained: Use pretrained ImageNet weights.
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_zernike_terms: int = 55,
        backbone: Literal["resnet18", "resnet34"] = "resnet18",
        pretrained: bool = True,
    ):
        """Initialize ResNet regression model."""
        super().__init__(in_channels=in_channels, n_zernike_terms=n_zernike_terms)

        self.backbone_name = backbone

        # Load ResNet backbone
        if backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            resnet = resnet18(weights=weights)
            self.feature_dim = 512
        else:  # resnet34
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            resnet = resnet34(weights=weights)
            self.feature_dim = 512

        # Replace first conv layer to match in_channels
        if in_channels != 3:
            original_conv = resnet.conv1
            resnet.conv1 = nn.Conv2d(
                in_channels,
                original_conv.out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            # Initialize with kaiming init for new channel count
            nn.init.kaiming_normal_(
                resnet.conv1.weight, mode="fan_out", nonlinearity="relu"
            )
            # If in_channels < 3, replicate weights; if > 3, zero init extra channels
            if in_channels < 3:
                resnet.conv1.weight = nn.Parameter(
                    resnet.conv1.weight[:, :in_channels, :, :].clone()
                )

        # Remove final FC layer - keep everything else (avgpool included)
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        # Adaptive pooling handles different input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # FC head for regression
        self.fc = nn.Linear(self.feature_dim, n_zernike_terms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Zernike coefficients of shape (B, n_zernike_terms).
        """
        # ResNet encoder
        x = self.encoder(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # FC head
        return self.fc(x)


# =============================================================================
# Simple CNN Regression Model (Zernike Coefficients)
# =============================================================================


class SimpleCNNRegression(BasePhasePredictor):
    """Lightweight CNN + FC head for Zernike coefficient prediction.

    Simple conv blocks: Conv2d -> BatchNorm -> ReLU -> MaxPool
    4 layers with channel sizes [32, 64, 128, 256]
    Global average pooling -> FC(256, n_coeffs)

    Architecture:
        Input (B, C, H, W) ->
        ConvBlock(32) -> ConvBlock(64) -> ConvBlock(128) -> ConvBlock(256) ->
        GlobalAvgPool -> FC(256, n_coeffs) ->
        Output (B, n_coeffs)

    Args:
        in_channels: Number of input channels (1 for single, 2 for dual camera).
        n_zernike_terms: Number of Zernike terms to predict (default 55 for n_max=10).
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_zernike_terms: int = 55,
    ):
        """Initialize simple CNN regression model."""
        super().__init__(in_channels=in_channels, n_zernike_terms=n_zernike_terms)

        # Channel progression
        channels = [32, 64, 128, 256]

        # Build CNN blocks
        self.conv_blocks = nn.ModuleList()

        in_ch = in_channels
        for out_ch in channels:
            self.conv_blocks.append(self._make_conv_block(in_ch, out_ch))
            in_ch = out_ch

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # FC head
        self.fc = nn.Linear(channels[-1], n_zernike_terms)

    def _make_conv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """Create a conv block: Conv2d -> BN -> ReLU -> MaxPool."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Zernike coefficients of shape (B, n_zernike_terms).
        """
        for block in self.conv_blocks:
            x = block(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        return self.fc(x)


# =============================================================================
# Model Registry
# =============================================================================


MODEL_REGISTRY: dict[str, type[BasePhasePredictor]] = {
    "unet": UNetGenerator,
    "resnet18": ResNetRegression,
    "resnet34": ResNetRegression,
    "simple_cnn": SimpleCNNRegression,
}


def build_model(
    model_type: str,
    in_channels: int = 1,
    n_coeffs: int = 55,
    device: str = "cpu",
    **kwargs,
) -> BasePhasePredictor:
    """Build a phase prediction model.

    Args:
        model_type: Model type - "unet", "resnet18", "resnet34", "simple_cnn".
        in_channels: Number of input channels (1 for single, 2 for dual camera).
        n_coeffs: Number of Zernike coefficients to predict.
        device: Target device ("cpu" or "cuda").
        **kwargs: Additional model-specific arguments.

    Returns:
        Model instance.

    Raises:
        ValueError: If model_type is not recognized.

    Examples:
        >>> # ResNet18 for 55 Zernike terms
        >>> model = build_model("resnet18", in_channels=1, n_coeffs=55)
        >>> # Simple CNN for 28 Zernike terms (n_max=7)
        >>> model = build_model("simple_cnn", in_channels=2, n_coeffs=28)
        >>> # U-Net for phase map (legacy)
        >>> model = build_model("unet", in_channels=2, n_coeffs=55)
    """
    if model_type not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model_type: '{model_type}'. Available: {available}")

    model_cls = MODEL_REGISTRY[model_type]

    # Special handling for ResNet which has backbone arg
    if model_type in ("resnet18", "resnet34"):
        backbone = model_type  # e.g., "resnet18" or "resnet34"
        model = model_cls(
            in_channels=in_channels,
            n_zernike_terms=n_coeffs,
            backbone=backbone,
            **kwargs,
        )
    elif model_type == "unet":
        # Legacy UNet with optional coeff output
        output_mode = kwargs.get("output_mode", "phase")
        model = model_cls(
            in_channels=in_channels,
            output_mode=output_mode,
            n_coeffs=n_coeffs,
        )
    else:
        model = model_cls(
            in_channels=in_channels,
            n_zernike_terms=n_coeffs,
            **kwargs,
        )

    return model.to(device)


# =============================================================================
# Legacy build_model for GAN training
# =============================================================================


def build_gan_models(
    in_channels: int = 2,
    unet_features: list[int] | None = None,
    disc_features: list[int] | None = None,
    device: str = "cpu",
) -> tuple[UNetGenerator, PatchGANDiscriminator]:
    """Build generator and discriminator models for GAN training.

    Note: Legacy function for phase map prediction. For Zernike coefficient
    prediction, use build_model() instead.

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
