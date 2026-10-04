"""Learned Zernike-coefficient -> CCD far-field forward model.

The **model half** of a supervised forward map on the 2f Fourier bench. The
target physics is a *deterministic* chain:

.. code-block:: text

    coefficients c (Noll order, raw radians)
        -> pupil phase      sum_j c_j Z_j
        -> pupil field      exp(i * phase)
        -> centre-pad + FFT fftshift(fft2(ifftshift(field)))
        -> |F|^2            INTENSITY, because a CCD integrates intensity and
                           does not report field amplitude
        -> per-frame peak normalisation

:func:`peak_normalize` is the canonical output contract, and it is
**numerically identical in intent** to the ``"peak"`` branch of
:meth:`ml.zernike.models.ZernikeAmpModel._normalize`::

    scale = x.amax(dim=(-2, -1), keepdim=True)
    return x / torch.clamp(scale, min=_EPS)

Both models in this module emit a **RAW, unbounded** single-channel image and
the caller applies :func:`peak_normalize`. See the output-head comment on
:class:`ZernikeCoeffConvNet` for why a bounded head would be wrong.

Why a coefficient-projection + conv decoder rather than flatten-to-vector
-----------------------------------------------------------------------
The map is deterministic, smooth, and low-dimensional-to-image, and its output
has **genuine 2-D locality**: a shifted/scaled Airy-like spot whose position
follows the tip/tilt components and whose radial structure follows the
higher-order modes. A flatten-to-``grid*grid`` MLP throws that geometry away and
has to re-learn translation equivariance from a single weight matrix.
:class:`ZernikeCoeffConvNet` therefore *creates* spatial structure at the
bridge (``coeff_proj`` reshapes a bare vector into a constant feature map) and
then decodes it with translation-equivariant convolutions.
:class:`ZernikeCoeffMLP` is the deliberate baseline arm for exactly that
comparison.

Why this is **not** ``ml.phase.unet.UNetGenerator`` (and must not be "simplified"
into it)
--------------------------------------------------------------------------
``UNetGenerator`` offers exactly two output modes, and neither fits
(``src/ml/phase/unet.py``):

* ``output_mode="phase"`` returns ``self.sigmoid(self.final_conv(x))``
  (line 129) -- a hard-wired ``nn.Sigmoid()``. It cannot represent values
  outside ``(0, 1)`` and cannot reach exactly 0 or 1. Our output is an image
  whose absolute scale is set *afterwards* by per-frame peak normalisation, so a
  bounded head would silently saturate the dynamic range: every value pushed
  beyond the rail would be squashed to the same number and the model could no
  longer tell "very bright core" from "bright core".
* ``output_mode="coeffs"`` returns a raw ``Linear`` **vector**, not an image.

There is no output mode giving a raw unbounded single-channel image, and
``build_unet`` does not even expose ``output_mode``/``n_coeffs``. Hence a new
module; ``test_forward_model.py::TestAntiSigmoidHead`` is the regression guard
that makes a future "let's just reuse the U-Net" shortcut fail loudly.

Why the norm layer is configurable and defaults to ``group``
-------------------------------------------------------
``unet.DoubleConv`` hardcodes ``nn.BatchNorm2d``. That is the wrong choice for
this task and the reason this module has its own block:

* The target is **peak-normalised per frame**, so its per-sample scale is
  already normalised away. ``BatchNorm`` additionally carries running
  statistics that are updated every training step and consumed in ``eval()``,
  which *couples* what the network does at train time to the order and history
  of the batches it has seen. On a 64x64 regression target with few hundred
  records that coupling is pure variance.
* ``GroupNorm`` normalises per-sample over channel groups, so train and eval
  behave identically and no state is carried across epochs.
* ``batch`` and ``none`` remain selectable so the choice is measurable rather
  than asserted.

Input-scale expectation (deliberately **not** handled here)
------------------------------------------------------------
The real corpus coefficients are **tiny** -- measured ``max|c| = 0.0617 rad``,
``rms = 0.0142`` -- and are zero-padded from observed lengths 15/36/78 up to
:data:`DEFAULT_N_COEFFS`, so 58-121 of the input dimensions are structurally
always zero. No input-side normalisation is applied in ``forward``: that belongs
to the Dataset, which owns the input contract. The ``Linear`` bridge must
nevertheless learn those small scales, which is why
``nn.init.xavier_uniform_`` (not a zero or tiny init) is used.

Metrics: judge this model on R^2 / correlation / spot metrics, never MSE
----------------------------------------------------------------------
Because the target is peak-normalised per frame, MSE/PSNR/SSIM are degenerate
selectors. Measured on a sibling task: a *total-energy* normalisation gave
PSNR 72 dB and SSIM 0.9996 while R^2 was **worse**. Per-frame peak
normalisation pins ``max == 1`` on both prediction and target, so those three
metrics mostly measure the normalisation rather than the fit.

Implementation note: these classes are plain ``nn.Module`` subclasses and are
**not** ``@dataclass(frozen=True)``. A frozen dataclass injects a ``__setattr__``
that raises ``FrozenInstanceError`` on *every* attribute assignment, so an
``nn.Module`` built that way cannot even run ``self.coeff_proj = nn.Linear(...)``
(verified on Python 3.14: ``FrozenInstanceError: cannot assign to field
'config'``), and a non-frozen dataclass with no declared fields makes two models
with different configs compare equal. The *config* object, where immutability
is genuinely wanted, is frozen; the models are not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from loguru import logger

from ao_shaping.utils.wavefront.zernike_calc import calc_n_zernike_terms

__all__ = [
    "DEFAULT_N_COEFFS",
    "ZernikeCoeffConfig",
    "ZernikeCoeffConvNet",
    "ZernikeCoeffMLP",
    "build_forward_model",
    "count_parameters",
    "peak_normalize",
]

Architecture = Literal["conv", "mlp"]
NormLayer = Literal["group", "batch", "none"]

#: Accepted ``architecture`` values, for validation and error messages.
_ARCHITECTURES: tuple[str, ...] = ("conv", "mlp")
#: Accepted ``norm_layer`` values, for validation and error messages.
_NORM_LAYERS: tuple[str, ...] = ("group", "batch", "none")

#: Radial order the padded input width is sized for. ``n_max = 15`` is the
#: default order used by the sibling physics model and the ML training entry
#: point, so ``DEFAULT_N_COEFFS`` is derived from it rather than hardcoded:
#: ``calc_n_zernike_terms(15) == sum_{n=0..15} (n + 1) == 136``.
_DEFAULT_N_MAX: int = 15

#: Padded Noll-ordered input width, i.e. ``calc_n_zernike_terms(15)``.
DEFAULT_N_COEFFS: int = int(calc_n_zernike_terms(_DEFAULT_N_MAX))

#: Floor for the peak-normalisation divisor. Matches ``_EPS`` in
#: ``ml.zernike.models`` so the two normalisers cannot drift apart.
_EPS = 1e-12


def peak_normalize(x: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Per-sample peak normalisation -- THE canonical output contract.

    Deliberately identical in intent to the ``"peak"`` branch of
    :meth:`ml.zernike.models.ZernikeAmpModel._normalize`: divide by the
    per-sample maximum over the two spatial axes, with the divisor floored at
    ``eps`` so an all-zero (or all-negative) frame yields finite values instead
    of NaN/Inf.

    Args:
        x: Tensor of shape ``(..., H, W)``. Both 3-D ``(B, H, W)`` and 4-D
            ``(B, C, H, W)`` inputs are supported; the reduction always runs over
            the last two dims only.
        eps: Positive lower bound for the divisor.

    Returns:
        A tensor of the same shape and dtype where each sample's maximum is
        exactly ``1.0``. Negative values are **not** clamped away and the
        divisor is **not** shared across the batch -- only per sample.
    """
    scale = x.amax(dim=(-2, -1), keepdim=True)
    return x / torch.clamp(scale, min=eps)


@dataclass(frozen=True)
class ZernikeCoeffConfig:
    """Geometry and capacity contract for the coefficient forward models.

    Attributes:
        n_coeffs: Width of the Noll-ordered coefficient input vector.
            Defaults to :data:`DEFAULT_N_COEFFS` (``calc_n_zernike_terms(15)``).
            Shorter observed vectors are zero-padded to this width.
        grid: Side length of the square far-field image. Defaults to 64, the
            grid the dataset materialises.
        architecture: ``"conv"`` (default) selects
            :class:`ZernikeCoeffConvNet`, ``"mlp"`` selects
            :class:`ZernikeCoeffMLP`.
        bottleneck: Spatial side length of the constant feature map that
            ``coeff_proj`` creates. Must satisfy
            ``bottleneck * 2 ** len(features) == grid`` for the conv model --
            that identity is checked in :class:`ZernikeCoeffConvNet.__init__`.

            The default 4 is *derived*, not chosen: it is
            ``grid // 2 ** len(features) == 64 // 16`` for the defaults below.
            ``grid`` is the immovable side of that identity (it is the grid the
            dataset materialises and the ``(B, 1, grid, grid)`` output
            contract), so ``bottleneck`` is the field that has to give.
        features: Decoder channel widths, coarse -> fine. Every entry halves the
            resolution doubling stage count by one, so ``len(features)`` fixes
            the number of upsamples.
        latent_channels: Channel count of the projected feature map.
        hidden: First hidden width of the MLP arm. The MLP expands to
            ``2 * hidden`` and then projects straight to ``grid * grid``.
        norm_layer: ``"group"`` (default), ``"batch"`` or ``"none"``. See the
            module docstring for why ``batch`` is not the default here.
        norm_groups: Requested GroupNorm group count. Reduced per-width to
            ``gcd(width, norm_groups)`` when a width is not divisible.
    """

    n_coeffs: int = DEFAULT_N_COEFFS
    grid: int = 64
    architecture: Architecture = "conv"
    # grid == bottleneck * 2 ** len(features)  ->  64 == 4 * 2**4.
    bottleneck: int = 4
    features: tuple[int, ...] = (256, 128, 64, 32)
    latent_channels: int = 256
    hidden: int = 512
    norm_layer: NormLayer = "group"
    norm_groups: int = 8


class _ConvBlock(nn.Module):
    """``Conv -> norm -> GELU -> Conv -> norm`` with a configurable norm.

    Mirrors ``ml.phase.unet.DoubleConv``'s shape (two 3x3 convolutions with
    padding 1) but replaces the hardcoded ``nn.BatchNorm2d`` with a selectable
    norm -- see the module docstring. ``GELU`` replaces ``ReLU`` to match the
    rest of this package.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: NormLayer,
        norm_groups: int,
    ) -> None:
        """Build the block.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            norm_layer: ``"group"``, ``"batch"`` or ``"none"``.
            norm_groups: Requested GroupNorm groups; reduced to
                ``gcd(out_channels, norm_groups)`` when not divisible.

        Raises:
            ValueError: If ``norm_layer`` is unknown, or the resulting group
                count is below 1.
        """
        super().__init__()
        if norm_layer not in _NORM_LAYERS:
            raise ValueError(f"norm_layer must be one of {_NORM_LAYERS}, got {norm_layer!r}")

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = _make_norm(norm_layer, out_channels, norm_groups)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = _make_norm(norm_layer, out_channels, norm_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block to a ``(B, C, H, W)`` tensor."""
        return self.norm2(self.conv2(self.act(self.norm1(self.conv1(x)))))


def _make_norm(norm_layer: NormLayer, channels: int, norm_groups: int) -> nn.Module:
    """Return the normalisation layer for ``channels`` channels.

    When ``norm_layer == "group"`` the group count is reduced to the greatest
    common divisor of ``channels`` and ``norm_groups`` if the requested count
    does not divide the width -- picking an invalid count would raise deep
    inside ``GroupNorm`` with an unhelpful message, and silently rounding would
    hide the fact that the requested value was unusable. A debug message records
    the reduction.

    Args:
        norm_layer: ``"group"``, ``"batch"`` or ``"none"``.
        channels: Number of channels the norm must handle.
        norm_groups: Requested GroupNorm group count.

    Returns:
        An ``nn.Module``: ``nn.GroupNorm``, ``nn.BatchNorm2d`` or ``nn.Identity``.

    Raises:
        ValueError: If ``norm_layer`` is unknown or the group count is below 1.
    """
    if norm_layer == "none":
        return nn.Identity()
    if norm_layer == "batch":
        return nn.BatchNorm2d(channels)
    if norm_layer != "group":  # pragma: no cover - guarded by caller validation
        raise ValueError(f"norm_layer must be one of {_NORM_LAYERS}, got {norm_layer!r}")

    if norm_groups < 1:
        raise ValueError(f"norm_groups must be >= 1, got {norm_groups}")
    groups = norm_groups if channels % norm_groups == 0 else math.gcd(channels, norm_groups)
    if groups != norm_groups:
        logger.debug(
            "GroupNorm groups {} does not divide {} channels; using gcd = {}",
            norm_groups,
            channels,
            groups,
        )
    if groups < 1:  # pragma: no cover - gcd of two positives is >= 1
        raise ValueError(f"resolved GroupNorm groups must be >= 1, got {groups}")
    return nn.GroupNorm(groups, channels)


class ZernikeCoeffConvNet(nn.Module):
    """Coefficient-projection + conv decoder: ``(B, n_coeffs) -> (B, 1, g, g)``.

    The bridge (``coeff_proj``) is where spatial structure is *created*: a bare
    coefficient vector is projected to ``latent_channels * bottleneck ** 2`` and
    reshaped into a constant feature map. Everything after that is
    translation-equivariant upsampling, which is what lets the decoder express
    "the spot moved" without a fully connected layer having to memorise every
    translation.

    Not a ``@dataclass``: see the implementation note at the end of the module
    docstring.
    """

    def __init__(self, config: ZernikeCoeffConfig) -> None:
        """Validate the configuration and build the network.

        Args:
            config: Geometry/capacity contract. ``grid`` must be exactly
                reachable, i.e. ``bottleneck * 2 ** len(features) == grid``.

        Raises:
            ValueError: On any invalid field, including an unreachable ``grid``
                or a ``norm_layer`` outside ``{"group", "batch", "none"}``.
        """
        super().__init__()
        _validate_common(config)

        self.config = config
        reachable = config.bottleneck * 2 ** len(config.features)
        if reachable != config.grid:
            raise ValueError(
                f"grid={config.grid} is unreachable: bottleneck={config.bottleneck} "
                f"with {len(config.features)} upsampling stage(s) of scale 2 "
                f"produces {config.bottleneck} * 2**{len(config.features)} = {reachable} "
                f"pixels; require bottleneck * 2**len(features) == grid, so either set "
                f"bottleneck={config.grid // 2 ** len(config.features)} "
                f"(with len(features)={len(config.features)}) or adjust "
                f"len(features) to match bottleneck={config.bottleneck}"
            )
        if config.bottleneck < 1:
            raise ValueError(f"bottleneck must be >= 1, got {config.bottleneck}")

        self.coeff_proj = nn.Linear(
            config.n_coeffs, config.latent_channels * config.bottleneck * config.bottleneck
        )

        blocks: list[nn.Module] = []
        channels = config.latent_channels
        for width in config.features:
            # Upsample first, then refine: nearest-neighbour upsampling introduces
            # no new values to average over, so the subsequent 3x3 sees a real
            # neighbourhood at the finer scale.
            blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    _ConvBlock(channels, width, config.norm_layer, config.norm_groups),
                )
            )
            channels = width
        self.decoder = nn.Sequential(*blocks)

        # Output head: 1x1 conv, RAW and UNBOUNDED on purpose.
        #
        # NO activation here. `UNetGenerator`'s image head is literally
        # `self.sigmoid(self.final_conv(x))` (src/ml/phase/unet.py line 129),
        # which cannot leave (0, 1) and would saturate this model's dynamic
        # range. The per-frame scale is imposed afterwards by
        # `peak_normalize`, so the head must be free to emit signed values.
        self.head = nn.Conv2d(channels, 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming-normal conv weights, Xavier-uniform linear weights, zero bias."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Map coefficients to a raw far-field image.

        Args:
            coeffs: ``(B, n_coeffs)`` coefficient tensor in raw radians. No
                input-side normalisation is applied (see the module docstring);
                zero-padded dimensions are expected and tolerated.

        Returns:
            ``(B, 1, grid, grid)`` float32 image, **RAW / UNBOUNDED**. The caller
            applies :func:`peak_normalize`.

        Raises:
            ValueError: If ``coeffs`` is not a 2-D tensor with ``n_coeffs`` columns.
        """
        _validate_coeffs(coeffs, self.config.n_coeffs)
        projected = self.coeff_proj(coeffs)
        batch = coeffs.shape[0]
        bottleneck = self.config.bottleneck
        features = projected.view(batch, self.config.latent_channels, bottleneck, bottleneck)
        return self.head(self.decoder(features))

    def denormalized(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper returning the peak-normalised prediction.

        Convenience only -- the loss/metric harness should keep the raw output so
        it can choose its own normalisation.

        Note that the ``argmax`` of the returned tensor is **exactly 1.0 by
        construction**. No absolute-brightness question (how much light arrived
        at the sensor, whether the laser drifted) is answerable from this
        model's output; only relative structure is.

        Args:
            coeffs: ``(B, n_coeffs)`` coefficient tensor in raw radians.

        Returns:
            ``(B, 1, grid, grid)`` tensor whose per-sample maximum is 1.0.
        """
        return peak_normalize(self.forward(coeffs))


class ZernikeCoeffMLP(nn.Module):
    """Flatten-to-vector baseline: ``(B, n_coeffs) -> (B, 1, g, g)``.

    The deliberate control arm for :class:`ZernikeCoeffConvNet`. It keeps the
    output contract identical (raw, unbounded, ``(B, 1, g, g)``) so the two are
    directly comparable, but it discards 2-D locality by emitting all
    ``grid * grid`` pixels from one ``Linear`` -- which is precisely the
    hypothesis the comparison tests.

    Not a ``@dataclass``: see the implementation note at the end of the module
    docstring.
    """

    def __init__(self, config: ZernikeCoeffConfig) -> None:
        """Validate the configuration and build the baseline network.

        Args:
            config: Geometry/capacity contract. Only ``n_coeffs``, ``grid`` and
                ``hidden`` are consumed; the conv-only fields (bottleneck,
                features, latent_channels) are irrelevant here, so the
                ``bottleneck * 2 ** len(features) == grid`` constraint is *not*
                enforced.

        Raises:
            ValueError: On any invalid field shared with the conv arm.
        """
        super().__init__()
        _validate_common(config)
        if config.hidden < 1:
            raise ValueError(f"hidden must be >= 1, got {config.hidden}")

        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.n_coeffs, config.hidden),
            nn.GELU(),
            nn.Linear(config.hidden, config.hidden * 2),
            nn.GELU(),
            nn.Linear(config.hidden * 2, config.grid * config.grid),
        )
        # No final activation -- same RAW contract as the conv arm.
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-uniform linear weights, zero bias (no conv layers here)."""
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Map coefficients to a raw far-field image.

        Args:
            coeffs: ``(B, n_coeffs)`` coefficient tensor in raw radians.

        Returns:
            ``(B, 1, grid, grid)`` float32 image, **RAW / UNBOUNDED**.

        Raises:
            ValueError: If ``coeffs`` is not a 2-D tensor with ``n_coeffs`` columns.
        """
        _validate_coeffs(coeffs, self.config.n_coeffs)
        flat = self.net(coeffs)
        return flat.view(coeffs.shape[0], 1, self.config.grid, self.config.grid)

    def denormalized(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Peak-normalised prediction; see :meth:`ZernikeCoeffConvNet.denormalized`.

        Args:
            coeffs: ``(B, n_coeffs)`` coefficient tensor in raw radians.

        Returns:
            ``(B, 1, grid, grid)`` tensor whose per-sample maximum is 1.0.
        """
        return peak_normalize(self.forward(coeffs))


def build_forward_model(config: ZernikeCoeffConfig) -> nn.Module:
    """Instantiate the model named by ``config.architecture``.

    Args:
        config: Geometry/capacity contract; ``architecture`` selects the arm.

    Returns:
        A :class:`ZernikeCoeffConvNet` or :class:`ZernikeCoeffMLP`.

    Raises:
        ValueError: If ``architecture`` is not one of ``{"conv", "mlp"}``.
    """
    if config.architecture == "conv":
        return ZernikeCoeffConvNet(config)
    if config.architecture == "mlp":
        return ZernikeCoeffMLP(config)
    raise ValueError(
        f"architecture must be one of {_ARCHITECTURES}, got {config.architecture!r}"
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters.

    Args:
        model: Any module, typically one of the forward models above.

    Returns:
        Total number of elements across all parameters with ``requires_grad``.
    """
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _validate_common(config: ZernikeCoeffConfig) -> None:
    """Validate the config fields both arms share.

    Args:
        config: The configuration to check.

    Raises:
        ValueError: If any shared field is invalid.
    """
    if config.n_coeffs < 1:
        raise ValueError(f"n_coeffs must be >= 1, got {config.n_coeffs}")
    if config.grid < 1:
        raise ValueError(f"grid must be >= 1, got {config.grid}")
    if config.latent_channels < 1:
        raise ValueError(f"latent_channels must be >= 1, got {config.latent_channels}")
    if not config.features:
        raise ValueError("features must contain at least one decoder width")
    for width in config.features:
        if width < 1:
            raise ValueError(f"every entry of features must be >= 1, got {width}")
    if config.norm_layer not in _NORM_LAYERS:
        raise ValueError(
            f"norm_layer must be one of {_NORM_LAYERS}, got {config.norm_layer!r}"
        )


def _validate_coeffs(coeffs: torch.Tensor, n_coeffs: int) -> None:
    """Validate the coefficient input shape.

    Args:
        coeffs: Candidate input tensor.
        n_coeffs: Expected number of columns.

    Raises:
        TypeError: If ``coeffs`` is not a ``torch.Tensor``.
        ValueError: If it is not 2-D, is not floating point, or has the wrong
            number of columns.
    """
    if not isinstance(coeffs, torch.Tensor):
        raise TypeError(f"coeffs must be a torch.Tensor, got {type(coeffs).__name__}")
    if coeffs.dim() != 2:
        raise ValueError(
            f"coeffs must be (B, n_coeffs), got shape {tuple(coeffs.shape)}"
        )
    if not coeffs.is_floating_point():
        raise ValueError(f"coeffs must be floating point, got {coeffs.dtype}")
    if coeffs.shape[1] != n_coeffs:
        raise ValueError(
            f"coeffs has {coeffs.shape[1]} column(s), expected n_coeffs={n_coeffs}"
        )