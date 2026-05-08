"""Zernike coefficient prediction models - ResNet and Simple CNN regression."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights


class BasePhasePredictor(ABC, nn.Module):
    """Abstract base class for phase predictors.

    Provides common interface for all phase prediction models:
    - Input: Image tensor (B, C, H, W)
    - Output: Zernike coefficients (B, n_coeffs) or phase map (B, 1, H, W)
    """

    def __init__(self, in_channels: int = 1, n_zernike_terms: int = 55):
        super().__init__()
        self.in_channels = in_channels
        self.n_zernike_terms = n_zernike_terms

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def get_output_dim(self) -> int:
        return self.n_zernike_terms


class ResNetRegression(BasePhasePredictor):
    """ResNet backbone + FC head for Zernike coefficient prediction.

    Uses pretrained ResNet18/34 as encoder (removes final FC layer).
    Adds adaptive pooling to handle different input sizes.
    FC head outputs n_zernike_terms coefficients.
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_zernike_terms: int = 55,
        backbone: Literal["resnet18", "resnet34"] = "resnet18",
        pretrained: bool = True,
    ):
        super().__init__(in_channels=in_channels, n_zernike_terms=n_zernike_terms)

        self.backbone_name = backbone

        if backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            resnet = resnet18(weights=weights)
            self.feature_dim = 512
        else:
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
            nn.init.kaiming_normal_(resnet.conv1.weight, mode="fan_out", nonlinearity="relu")
            if in_channels < 3:
                resnet.conv1.weight = nn.Parameter(
                    resnet.conv1.weight[:, :in_channels, :, :].clone()
                )

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

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.feature_dim, n_zernike_terms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class SimpleCNNRegression(BasePhasePredictor):
    """Lightweight CNN + FC head for Zernike coefficient prediction.

    Simple conv blocks: Conv2d -> BatchNorm -> ReLU -> MaxPool
    4 layers with channel sizes [32, 64, 128, 256]
    Global average pooling -> FC(256, n_coeffs)
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_zernike_terms: int = 55,
    ):
        super().__init__(in_channels=in_channels, n_zernike_terms=n_zernike_terms)

        channels = [32, 64, 128, 256]

        self.conv_blocks = nn.ModuleList()

        in_ch = in_channels
        for out_ch in channels:
            self.conv_blocks.append(self._make_conv_block(in_ch, out_ch))
            in_ch = out_ch

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], n_zernike_terms)

    def _make_conv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.conv_blocks:
            x = block(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        return self.fc(x)


MODEL_REGISTRY: dict[str, type[BasePhasePredictor]] = {
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
    """Build a phase prediction model for Zernike coefficient prediction.

    Args:
        model_type: Model type - "resnet18", "resnet34", "simple_cnn".
        in_channels: Number of input channels (1 for single, 2 for dual camera).
        n_coeffs: Number of Zernike coefficients to predict.
        device: Target device ("cpu" or "cuda").
        **kwargs: Additional model-specific arguments.

    Returns:
        Model instance.
    """
    if model_type not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model_type: '{model_type}'. Available: {available}")

    model_cls = MODEL_REGISTRY[model_type]

    if model_type in ("resnet18", "resnet34"):
        backbone = model_type
        model = model_cls(
            in_channels=in_channels,
            n_zernike_terms=n_coeffs,
            backbone=backbone,
            **kwargs,
        )
    else:
        model = model_cls(
            in_channels=in_channels,
            n_zernike_terms=n_coeffs,
            **kwargs,
        )

    return model.to(device)


def build_zernike_model(
    model_type: str = "resnet18",
    in_channels: int = 2,
    n_zernike_terms: int = 55,
    device: str = "cpu",
    **kwargs,
) -> BasePhasePredictor:
    """Build Zernike coefficient prediction model.

    Convenience function: equivalent to build_model().

    Args:
        model_type: Model type - "resnet18", "resnet34", "simple_cnn".
        in_channels: Number of input channels.
        n_zernike_terms: Number of Zernike terms.
        device: Target device.
        **kwargs: Additional arguments.

    Returns:
        Model instance.
    """
    return build_model(model_type, in_channels, n_zernike_terms, device, **kwargs)
