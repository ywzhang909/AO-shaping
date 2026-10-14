"""Phasor U-Net model for inverse beam shaping.

This module carries the *continuous* representation of the SLM phase through the
network as a two-channel phasor ``(cos, sin)`` rather than as a single wrapped
angle. The reasons are physical, not cosmetic:

* The hardware phase is stored wrapped in ``[0, 2*pi)`` (see the driver
  contract: the only ``mod 2*pi`` wrap lives in ``Santec.create_phase_from_array``).
  A wrapped target has an artificial branch cut at ``+/-pi``: two nearly identical
  phases that straddle the cut look maximally different in value. Recovering the
  angle from a network output therefore needs a two-argument inverse trigonometric
  function, whose output is itself discontinuous at that cut, so the gradient is
  ill-defined exactly where the data is not.
* A phasor is smooth and unambiguous: ``(cos, sin)`` has no cut, and the network
  can move continuously between any two phases. The final mapping back to an
  angle (for download) happens off-network, once, where the discontinuity is a
  representation artefact rather than a training signal.

The model is a thin subclass of :class:`ml.phase.unet.UNetGenerator`: it reuses
the entire U-Net body and only swaps the single-channel output head for a
two-channel one. We do **not** fork the encoder/decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger

from ml.phase.unet import UNetGenerator
from ml.shaping.config import ShapingConfig

__all__ = ["PhasorUNet", "build_phasor_unet", "unit_phasor"]


def unit_phasor(raw: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Project a raw two-channel tensor onto the unit circle.

    Interprets the first channel as the real part and the second as the
    imaginary part of a phasor and normalises each spatial location to unit
    magnitude, so the output is a valid ``(cos, sin)`` pair everywhere.

    Args:
        raw: Tensor of shape ``(B, 2, H, W)`` (or any ``(..., 2, ...)`` with the
            quadrature axis at dim 1). Channel 0 is the real part, channel 1 the
            imaginary part.
        eps: Magnitude threshold below which a location is considered "dark"
            (no defined phase) and parked at ``(1, 0)``.

    Returns:
        Tensor of the same shape as ``raw`` whose two channels always satisfy
        ``cos**2 + sin**2 == 1`` to within floating-point tolerance.

    Raises:
        ValueError: If ``raw`` does not have the quadrature axis of length 2 at
            dimension 1.

    Notes:
        Locations whose raw magnitude is ``<= eps`` (including an all-zero input,
        and the dark background of a far-field image) carry no phase information.
        The normalisation ``raw / (magnitude + eps)`` would only asymptotically
        approach unit length there and is exactly zero for a zero input, which is
        not a valid phasor. We instead clamp the denominator away from zero and
        explicitly park sub-threshold cells at ``(1, 0)``. The operation stays
        differentiable and gradient flows through for all informative cells.
    """
    if raw.ndim < 2 or raw.shape[1] != 2:
        raise ValueError(
            "unit_phasor expects the quadrature axis to be dim 1 of length 2 "
            f"(e.g. (B, 2, H, W)); got shape {tuple(raw.shape)}."
        )

    cos_raw = raw[:, 0:1]
    sin_raw = raw[:, 1:2]
    magnitude = torch.hypot(cos_raw, sin_raw)
    divisor = torch.clamp(magnitude, min=eps)
    cos = cos_raw / divisor
    sin = sin_raw / divisor

    dark = magnitude <= eps
    cos = torch.where(dark, torch.ones_like(cos), cos)
    sin = torch.where(dark, torch.zeros_like(sin), sin)
    return torch.cat([cos, sin], dim=1)


class PhasorUNet(UNetGenerator):
    """U-Net whose head predicts a continuous ``(cos, sin)`` phasor field.

    The full encoder/decoder of :class:`ml.phase.unet.UNetGenerator` is reused;
    only the output convolution is widened from one channel to two and the
    bounded activation is disabled (``output_mode="image"``), so ``forward``
    returns the raw two-channel field. Callers apply :func:`unit_phasor` to
    obtain a valid phasor.

    Input convention is ``(B, in_channels, grid, grid)`` where the default two
    channels are ``[far-field image, standardised log-exposure broadcast]``. The
    exposure is a real covariate (nine values spanning 0.1-80 ms), and the target
    image intentionally keeps its absolute brightness rather than being
    peak-normalised, so the network is told how bright the frame should be.

    Args:
        in_channels: Number of input channels (must be >= 1).
        features: Encoder channel widths. ``None`` uses the U-Net default
            ``[64, 128, 256, 512, 1024]``. Must contain at least two entries.
        grid: Spatial size the network is intended to run on, stored for
            bookkeeping (e.g. 64). Passed through to consumers.

    Raises:
        ValueError: If ``in_channels < 1`` or ``features`` has fewer than two
            entries.
    """

    def __init__(
        self,
        in_channels: int = 2,
        features: list[int] | None = None,
        grid: int | None = None,
    ) -> None:
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}.")
        if features is not None and len(features) < 2:
            raise ValueError(
                f"features must contain at least two encoder widths, got {features}."
            )

        resolved = [int(f) for f in features] if features is not None else None
        super().__init__(
            in_channels=int(in_channels),
            features=resolved,
            output_mode="image",
        )

        # Widen the single-channel head to a two-channel phasor head. The U-Net
        # body (encoder, decoder, bottleneck) is untouched.
        self.final_conv = nn.Conv2d(self.features[0], 2, kernel_size=1)
        self.grid = grid


def build_phasor_unet(
    in_channels: int = 2,
    features: list[int] | None = None,
    device: str = "cpu",
    *,
    config: ShapingConfig | None = None,
) -> PhasorUNet:
    """Build a :class:`PhasorUNet`, optionally from a :class:`ShapingConfig`.

    Args:
        in_channels: Input channels; overridden by ``config`` when given.
        features: Encoder widths; overridden by ``config`` when given.
        device: Torch device string the model is moved to.
        config: Optional configuration. When provided, ``config.in_channels``,
        ``config.features`` and ``config.grid`` take precedence over the
            corresponding arguments.

    Returns:
        A :class:`PhasorUNet` placed on ``device``.
    """
    grid: int | None = None
    if config is not None:
        in_channels = config.in_channels
        features = list(config.features)
        grid = config.grid

    model = PhasorUNet(in_channels=in_channels, features=features, grid=grid)
    logger.debug(
        "built PhasorUNet in_channels={} features={} grid={} device={}",
        in_channels,
        features,
        grid,
        device,
    )
    return model.to(device)
