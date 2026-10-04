"""Physics-based Zernike forward model: measured SLM phasor -> far-field pattern.

The model optimises **one global Zernike coefficient vector** (a single
correction for the whole optical bench) and evaluates it through a real
Fraunhofer forward model:

.. code-block:: text

    measured phasor (phase_cos, phase_sin)          [B, 1, g, g]
        -> complex pupil field = cos + i*sin
        -> pupil *= exp(i * sum_k c_k * Z_k)       (coherent addition, NOT atan2)
        -> centre-pad to far-field grid
        -> fftshift(fft2(ifftshift(field)))         (canonical bench convention)
        -> |F|^2 (intensity) or |F| (amplitude)
        -> peak / sum normalisation

Why the input is a *phasor* and not an angle
---------------------------------------------
``ml.hwdataset`` deliberately stores the measured phase as ``(cos, sin)``
because ``arctan2`` has a branch cut at +/-pi: an angle representation puts a
discontinuity on the input surface. This model consumes the phasor directly
and stays in the complex domain end to end -- radians are never reconstructed
via ``atan2``.

Units
-----
Zernike coefficients are **raw unwrapped radians**, matching the hardware path
(:func:`ml.gsnet_debug.offline.reconstruct_pupil_phase_rad`).
No ``um_to_waves()`` or ``* 2pi`` conversion is applied; adding one is the unit
bug documented in ``AGENTS.md``.

Piston
------
Noll 1 ``(0, 0)`` is a **far-field no-op**: a constant phase across the pupil
only shifts the global phase reference, leaving the intensity unchanged. It is
therefore excluded from the learnable basis, giving
``K = calc_n_zernike_terms(n_max) - 1``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from ao_shaping.utils.wavefront.zernike_calc import (
    ZernikeGenerator,
    calc_n_zernike_terms,
)

__all__ = [
    "ZernikeAmpConfig",
    "ZernikeAmpFitConfig",
    "ZernikeAmpHybrid",
    "ZernikeAmpModel",
    "ZernikeAmpResult",
    "ZernikeBasis",
]

Observable = Literal["amplitude", "intensity"]
Normalization = Literal["peak", "sum", "none"]
OptimizerName = Literal["adam", "adamw", "sgd"]

#: Physical quantity compared against the CCD frame.
_OBSERVABLES: tuple[str, ...] = ("amplitude", "intensity")
#: Output rescaling applied after the observable is formed.
_NORMALIZATIONS: tuple[str, ...] = ("peak", "sum", "none")
_OPTIMIZERS: tuple[str, ...] = ("adam", "adamw", "sgd")

#: 0-based slot of Noll 1 == (0, 0) inside a Noll-ordered coefficient vector.
_PISTON_SLOT = 0

_EPS = 1e-12


def _centre_crop(field: torch.Tensor, grid: int) -> torch.Tensor:
    """Crop a ``(..., m, m)`` field back to its centre ``grid x grid``.

    A 2-D Fourier transform puts the optical axis at index ``0``; ``fftshift``
    moves it to the centre, so the physically comparable region is the centre
    window. For an even ``m`` the crop is exactly centred.

    Args:
        field: Complex or real tensor whose last two dims are ``(m, m)``.
        grid: Side length to keep.

    Returns:
        The centre ``grid x grid`` crop, or ``field`` unchanged when it is already
        no larger than ``grid``.
    """
    side = field.shape[-1]
    if side <= grid:
        return field
    start = (side - grid) // 2
    return field[..., start : start + grid, start : start + grid]


@dataclass
class ZernikeAmpConfig:
    """Geometry and observable contract for :class:`ZernikeAmpModel`.

    Attributes:
        n_max: Maximum radial order. ``K = calc_n_zernike_terms(n_max) - 1``
            non-piston modes are learned. Must be >= 1.
        grid: Side length ``g`` of the square pupil/CCD grid. Must be >= 2.
        radius: Aperture radius in pixels. ``None`` uses the inscribed circle
            (``grid / 2``), matching ``ZernikeGenerator``'s default.
        observable: ``"intensity"`` (default) or ``"amplitude"`` (``|F|``).
            Defaults to ``"intensity"`` because **the measurements say so**:
            on real ``slm_zernike_shaping`` data, ``n_max=15``,
            ``far_field_padding=10``, 1010 training records --

            ==================  ======  ===========
            observable          R^2     PSNR
            ==================  ======  ===========
            ``"intensity"``     +0.797  27.0 dB
            ``"amplitude"``     +0.675  23.7 dB
            ==================  ======  ===========

            and the same ordering held at ``n_max=4`` (+0.691 vs +0.634). This
            is also the physically correct choice: a CCD integrates intensity,
            it does not report field amplitude, so regressing ``|F|^2`` against a
            measured frame matches what the sensor actually recorded. The original
            spec said ``amp``, which is why ``"amplitude"`` was the first default;
            it is kept as a supported value, but it is measurably worse.
        normalization: ``"peak"`` (default), ``"sum"``, or ``"none"``.

            **``"sum"`` is a trap and must not be selected on MSE alone.**
            Dividing both prediction and target by their own total energy makes
            them agree almost trivially -- measured ``MSE = 0.00000``,
            ``PSNR = 72 dB``, ``SSIM = 0.9996`` -- while ``R^2`` is *worse*
            (+0.632 vs +0.706 for ``"peak"``). Any metric derived from a sum
            normalisation is measuring the normalisation, not the fit. Judge this
            setting on ``R^2`` / ``correlation`` / beam metrics, never on MSE.

            ``"none"`` is unusable with real data: the raw FFT amplitude carries
            an arbitrary scale, giving ``R^2 = -159`` and an overflowing
            perplexity.
        far_field_padding: Far-field grid is ``grid * far_field_padding``, so the
            predicted pattern spans a much larger angular extent than the target.
        center_crop: Crop the predicted far field back to the centre ``grid x grid``
            so the output is comparable pixel-for-pixel with the dataset's
            ``image``. Defaults to ``True``.

            This is not cosmetic. :func:`ml.hwdataset.transforms
            .farfield_frame_to_grid` takes a ``grid x grid`` window **centred on
            the 0-order spot** out of the full camera frame (``_anchored_window``)
            -- it does not resize. For the measured ``slm_zernike_shaping`` family
            that frame is 248 px wide, so the target covers only ``64 / 248`` of
            the camera's field of view. A prediction that spans the pupil's full
            diffraction field therefore has ~4x the angular scale of the target,
            the two cannot be compared pixel-for-pixel, and the score is
            *worse than predicting the mean* (measured R^2 = -0.20 at
            ``far_field_padding=1``).

            Calibrating the extent and cropping the centre fixes it. Measured on
            128 real samples at ``n_max=4``, ``grid=64``, R^2 at ``Z = 0``:

            ==========  ========
            padding      R^2
            ==========  ========
            1            -0.20
            4            -0.12
            8            +0.33
            10           +0.46
            12           +0.53
            16           +0.49
            20           +0.18
            ==========  ========

            A clear interior optimum, so the extent is a real calibration rather
            than a monotone "sharper is better". The right value depends on the
            family's ``fov_px``, so re-run the sweep for a new family rather than
            copying the default.
        attention: Add a trainable self-attention refinement **after** the far
            field is formed. Off by default, because the pure physics path is the
            model's whole value proposition: it is a closed-form function of its
            coefficients, so the phase it implies (``Σ Z_k B_k``) is directly
            implementable on the SLM. An attention block is **not** invertible
            back to a phase, so enabling it trades that guarantee away.
        attention_grid: Token grid the attention runs on. The far field is
            adaptively pooled to ``attention_grid x attention_grid`` before
            attention and bilinearly restored after. At ``grid=64`` the full
            field is 4096 tokens, which would make attention quadratic in a very
            large number; 16x16 = 256 tokens keeps it ~1% of the cost. Default 16.
        attention_dim: Bottleneck width of the attention block. Default 32.
        attention_heads: Self-attention heads. Default 1.
    """

    n_max: int = 4
    grid: int = 64
    radius: float | None = None
    observable: Observable = "intensity"
    normalization: Normalization = "peak"
    far_field_padding: int = 10
    center_crop: bool = True
    attention: bool = False
    attention_grid: int = 16
    attention_dim: int = 32
    attention_heads: int = 1


class FarFieldAttention(torch.nn.Module):
    """Self-attention refinement of a far field, as an identity-initialised residual.

    Placed after the physics (the Zernike phase and the Fraunhofer FFT) and
    applied to the *intensity*, so the coefficients keep their meaning as the
    physical description of the bench and attention learns a bounded correction on
    top of it.

    Two properties are deliberate:

    * **Identity at initialisation.** ``out_proj`` is zero-initialised, so the
      block returns exactly its input before training. A randomly-initialised
      residual would start by *destroying* a model that is already good -- the
      measured pure-physics fit reaches val R^2 +0.79 on this corpus -- and the
      optimiser would have to recover that from scratch. With identity init the
      attention can only ever improve on the physics it starts from.
    * **Cannot annihilate the field.** An earlier version used an *additive*
      residual followed by ``clamp(min=0)``. That is a silent dead end: the
      optimiser's first sign-based step (Adam) pushes ``out_proj`` negative
      enough that the clamp zeroes the entire far field, so the output is
      identically 0, its gradient is identically 0, and **every** parameter --
      the coefficients included -- stops receiving gradient. Measured: nonzero
      gradient at step 0 for ``coefficients``/``out_proj``, then empty from step 1
      onwards, with no error raised anywhere. The correction is therefore
      *multiplicative*, ``intensity * (1 + tanh(.))``, which is bounded to
      ``(0, 2x)``: it can reshape the beam but can never zero it, lose positivity,
      or block the gradient path.
    * **Cheap tokens.** The far field is pooled to ``attention_grid`` first
      (256 tokens at the 16x16 default instead of 4096 at 64x64), because
      attention cost is quadratic in token count and 4096 would dominate the
      forward pass for no measurable gain at this output resolution.
    """

    def __init__(self, grid: int, token_grid: int, dim: int, heads: int) -> None:
        super().__init__()
        if heads < 1:
            raise ValueError(f"attention_heads must be >= 1, got {heads}")
        if token_grid < 1:
            raise ValueError(f"attention_grid must be >= 1, got {token_grid}")
        self.grid = int(grid)
        self.token_grid = int(token_grid)
        self.dim = int(dim)
        self.heads = int(heads)

        # (B, 1, g, g) -> (B, dim, t, t): the bottleneck also does the pooling.
        self.down = torch.nn.Conv2d(1, dim, kernel_size=1)
        self.qkv = torch.nn.Conv2d(dim, 3 * dim, kernel_size=1)
        self.proj = torch.nn.Conv2d(dim, dim, kernel_size=1)
        self.out_proj = torch.nn.Conv2d(dim, 1, kernel_size=1)
        # Identity at init: the block starts as an exact no-op.
        torch.nn.init.zeros_(self.out_proj.weight)
        torch.nn.init.zeros_(self.out_proj.bias)

    def forward(self, intensity: torch.Tensor) -> torch.Tensor:
        if intensity.dim() != 4 or intensity.shape[1] != 1:
            raise ValueError(
                f"expected (B, 1, g, g) intensity, got {tuple(intensity.shape)}"
            )
        tokens = self.down(intensity)
        if tokens.shape[-2:] != (self.token_grid, self.token_grid):
            tokens = torch.nn.functional.adaptive_avg_pool2d(
                tokens, (self.token_grid, self.token_grid)
            )
        b, dim = tokens.shape[0], self.dim
        n = self.token_grid * self.token_grid
        qkv = self.qkv(tokens).reshape(b, 3, self.heads, dim // self.heads, n)
        # (B, heads, n, head_dim)
        query, key, value = qkv.unbind(dim=1)
        scale = (dim // self.heads) ** -0.5
        weights = torch.softmax(
            query.transpose(-2, -1) @ key * scale, dim=-1
        )  # (B, heads, head_dim, head_dim)
        attended = (weights @ value.transpose(-2, -1)).transpose(-2, -1)
        merged = attended.reshape(b, dim, self.token_grid, self.token_grid)
        merged = self.proj(merged)
        if merged.shape[-2:] != intensity.shape[-2:]:
            merged = torch.nn.functional.interpolate(
                merged,
                size=intensity.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        # Multiplicative, bounded gate: identity at init (out_proj is zero), and
        # incapable of driving the field to zero -- see the class docstring for
        # why the additive+clamp form was a silent dead end.
        gate = 1.0 + torch.tanh(self.out_proj(merged))
        return intensity * gate


@dataclass
class ZernikeAmpFitConfig:
    """Optimisation hyper-parameters for :meth:`ZernikeAmpModel.fit`.

    Attributes:
        epochs: Number of passes over the data.
        lr: Learning rate.
        optimizer: ``"adam"``, ``"adamw"``, or ``"sgd"``.
        weight_decay: L2 penalty passed to the torch optimizer.
        momentum: Momentum, used only by ``"sgd"``.
        batch_size: Mini-batch size; ``0`` means full batch.
        log_every: Log the running loss every N epochs; ``0`` disables logging.
        normalize_target: Apply the model's own ``normalization`` to the target
            before computing the MSE. Defaults to ``True``.

            The raw FFT amplitude carries an arbitrary scale (``norm="ortho"``
            fixes total power, not peak brightness), so a peak-normalised
            prediction can only be compared fairly against a peak-normalised
            target. Leaving this on and passing a raw CCD frame would add a
            constant brightness offset to the loss that no coefficient can
            reduce -- the model would descend but never reach zero. Turn it off
            only when the target is already on the same scale, e.g. a
            pre-normalised array or ``normalization="none"``.
    """

    epochs: int = 200
    lr: float = 0.05
    optimizer: OptimizerName = "adam"
    weight_decay: float = 0.0
    momentum: float = 0.9
    batch_size: int = 0
    log_every: int = 10
    normalize_target: bool = True


@dataclass
class ZernikeAmpResult:
    """Outcome of a :meth:`ZernikeAmpModel.fit` run.

    Attributes:
        coefficients: Best coefficient vector seen, in raw radians.
        history: Per-epoch mean training loss, in order.
        best_loss: Smallest per-epoch mean loss observed.
        best_epoch: Epoch index (0-based) that produced ``best_loss``; ``-1``
            when the run was empty.
        seconds: Wall-clock duration of the fit.
        observable: Observable the loss was computed against.
        normalization: Output normalisation in force during the fit.
        n_max: Radial order used.
        K: Number of learned (non-piston) modes.
    """

    coefficients: np.ndarray
    history: list[float]
    best_loss: float
    best_epoch: int
    seconds: float
    observable: str
    normalization: str
    n_max: int
    K: int


class ZernikeBasis:
    """Stacked Zernike mode maps in raw unwrapped radians, piston excluded.

    All Zernike mathematics is delegated to the canonical engine
    :class:`~ao_shaping.utils.wavefront.zernike_calc.ZernikeGenerator`; no
    Noll table or radial polynomial is re-derived here (repo red line, see
    ``AGENTS.md`` -> ``## Zernike 使用规范``).

    Two canonical behaviours are reproduced deliberately:

    * ``ZernikeGenerator`` returns **NaN outside the circular aperture**, so
      every mode is passed through ``np.nan_to_num(..., nan=0.0)`` -- NaN must
      never reach a training sample.
    * The values are **raw unwrapped radians**. ``mod 2pi`` happens only at the
      SLM driver on radian->grayscale conversion, so generators must not wrap.
    """

    def __init__(self, grid: int, n_max: int, radius: float | None = None) -> None:
        """Build the basis once.

        Args:
            grid: Side length of the square grid; must be >= 2.
            n_max: Maximum radial order; must be >= 1.
            radius: Aperture radius in pixels, or ``None`` for the inscribed
                circle (``grid / 2``).

        Raises:
            ValueError: If ``grid`` < 2, ``n_max`` < 1, ``radius`` is not
                positive, or ``n_max`` yields no non-piston mode.
        """
        if not isinstance(grid, int) or isinstance(grid, bool) or grid < 2:
            raise ValueError(f"grid must be an int >= 2, got {grid!r}")
        if not isinstance(n_max, int) or isinstance(n_max, bool) or n_max < 1:
            raise ValueError(f"n_max must be an int >= 1, got {n_max!r}")
        if radius is not None and radius <= 0:
            raise ValueError(f"radius must be positive or None, got {radius!r}")

        n_terms = int(calc_n_zernike_terms(n_max))
        k_modes = n_terms - 1  # drop Noll 1 == (0, 0)
        if k_modes <= 0:
            raise ValueError(
                f"n_max={n_max} leaves {k_modes} non-piston Zernike mode(s); need >= 1"
            )

        generator = ZernikeGenerator((grid, grid), radius=radius, n_orders=n_max)

        modes: list[np.ndarray] = []
        for slot in range(k_modes):
            coefficients = np.zeros(generator.n_modes, dtype=np.float64)
            # `slot + 1` skips the piston sitting at Noll slot 0.
            coefficients[slot + _PISTON_SLOT + 1] = 1.0
            mode = generator.generate_noll(coefficients)
            modes.append(np.nan_to_num(mode, nan=0.0).astype(np.float32))

        self.grid = grid
        self.n_max = n_max
        self.radius = float(generator.radius)
        self.n_terms = n_terms
        self.K = k_modes
        self.modes = np.stack(modes, axis=0)
        logger.debug(
            "ZernikeBasis grid={} n_max={} K={} radius={}", grid, n_max, k_modes, self.radius
        )

    @property
    def n_modes(self) -> int:
        """Number of learned (non-piston) modes."""
        return self.K

    def __len__(self) -> int:
        return self.K

    def as_tensor(self) -> torch.Tensor:
        """Return a fresh ``(K, grid, grid)`` float32 tensor of the basis."""
        return torch.from_numpy(self.modes.copy())


class ZernikeAmpModel(nn.Module):
    """Global Zernike coefficient vector evaluated through a Fraunhofer model.

    The only learnable state is ``self.coefficients`` -- **one vector shared
    across the entire dataset**, initialised to zeros so the model starts from
    the measured (uncorrected) bench. The mode stack ``self.basis`` is a
    non-persistent buffer: it is fully determined by
    ``(grid, n_max, radius)``, so it is cheap to rebuild rather than to store
    in every checkpoint.

    Forward convention (mirrors
    ``iterative_zernike_shaping.IterativeZernikeShaper._far_field``)::

        field  = complex(phase_cos, phase_sin) * exp(i * sum_k c_k Z_k)
        focal  = fftshift(fft2(ifftshift(center_pad(field))))
        out    = observable(|focal|^2) normalised per frame

    The FFT itself carries **no implicit scale**: it is a plain
    ``fft2``/``ifft2`` pair (the same one the sim bench uses), and all scaling
    is expressed explicitly by the ``normalization`` setting.
    """

    def __init__(
        self,
        config: ZernikeAmpConfig | None = None,
        **overrides: Any,
    ) -> None:
        """Validate configuration and materialise the basis.

        Args:
            config: A :class:`ZernikeAmpConfig`. Mutually exclusive with
                ``overrides``.
            **overrides: Field values forwarded to :class:`ZernikeAmpConfig`.

        Raises:
            TypeError: If both ``config`` and ``overrides`` are supplied.
            ValueError: On any invalid configuration field, including
                ``n_max < 1`` and a non-piston mode count of zero.
        """
        super().__init__()
        if config is None:
            config = ZernikeAmpConfig(**overrides)
        elif overrides:
            raise TypeError(
                "pass either `config` or keyword overrides to ZernikeAmpConfig, not both"
            )
        self._validate_choices(config)

        # ZernikeBasis owns the grid / n_max / radius / K validation, so building
        # it here surfaces a bad config before any state is registered.
        basis = ZernikeBasis(config.grid, config.n_max, config.radius)

        self.config = config
        self.n_max = config.n_max
        self.grid = config.grid
        self.radius = config.radius
        self.observable: Observable = config.observable
        self.normalization: Normalization = config.normalization
        self.far_field_padding = config.far_field_padding
        self.center_crop = config.center_crop

        self.K = basis.K
        self.register_buffer("basis", basis.as_tensor(), persistent=False)

        # ONE global vector, shared by every sample in the dataset.
        self.coefficients = nn.Parameter(torch.zeros(self.K))

        # Optional trainable refinement of the far field. Identity at init, so
        # enabling it cannot degrade the physics before training has done
        # anything. Note this makes the output a function of the attention
        # weights too, so the coefficients stop being a closed-form description
        # of the phase -- see ZernikeAmpConfig.attention.
        self.attention: FarFieldAttention | None = None
        if config.attention:
            if config.attention_dim % config.attention_heads:
                raise ValueError(
                    f"attention_dim ({config.attention_dim}) must be divisible by "
                    f"attention_heads ({config.attention_heads})"
                )
            self.attention = FarFieldAttention(
                grid=self.grid,
                token_grid=config.attention_grid,
                dim=config.attention_dim,
                heads=config.attention_heads,
            )

        logger.info(
            "ZernikeAmpModel grid={} n_max={} K={} observable={} normalization={}",
            self.grid,
            self.n_max,
            self.K,
            self.observable,
            self.normalization,
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_choices(config: ZernikeAmpConfig) -> None:
        """Reject an invalid config before any state is built."""
        if config.observable not in _OBSERVABLES:
            raise ValueError(
                f"observable must be one of {list(_OBSERVABLES)}, got {config.observable!r}"
            )
        if config.normalization not in _NORMALIZATIONS:
            raise ValueError(
                f"normalization must be one of {list(_NORMALIZATIONS)}, "
                f"got {config.normalization!r}"
            )
        if (
            not isinstance(config.far_field_padding, int)
            or isinstance(config.far_field_padding, bool)
            or config.far_field_padding < 1
        ):
            raise ValueError(
                f"far_field_padding must be an int >= 1, got {config.far_field_padding!r}"
            )

    # ------------------------------------------------------------------
    # Forward model
    # ------------------------------------------------------------------
    def forward(self, phase_cos: torch.Tensor, phase_sin: torch.Tensor) -> torch.Tensor:
        """Predict the far-field observable for the measured pupil phasor.

        Args:
            phase_cos: Real part of the measured phasor, ``(B, 1, g, g)``.
            phase_sin: Imaginary part, same shape as ``phase_cos``.

        Returns:
            ``(B, 1, g, g)`` when :attr:`center_crop` is set (the default, and what
            makes the output comparable with the dataset's ``image``), else
            ``(B, 1, g * far_field_padding, g * far_field_padding)``. Differentiable
            w.r.t. ``self.coefficients``.

        Raises:
            ValueError: On shape/dtype mismatch or a grid that disagrees with the
                basis resolution.
        """
        self._validate_inputs(phase_cos, phase_sin)

        measured = torch.complex(phase_cos, phase_sin)
        correction = self.correction_phase()
        unit_phase = torch.polar(torch.ones_like(correction), correction)
        field = measured * unit_phase

        focal = self._propagate(field)
        if self.center_crop:
            focal = _centre_crop(focal, self.grid)
        intensity = focal.real.pow(2) + focal.imag.pow(2)

        if self.attention is not None:
            # Refine the *intensity*, then take the observable branch exactly as
            # the pure-physics path does. Doing the branch first (and squaring
            # the attention output here) would insert a spurious sqrt into the
            # "intensity" observable and break the identity-at-init guarantee.
            # The block is a bounded multiplicative gate, so the refined field is
            # already positive and needs no clamp here.
            intensity = self.attention(intensity)

        if self.observable == "intensity":
            observable = intensity
        else:
            # `sqrt` has a singular derivative at 0, and a far field has many
            # exact zeros (sidelobes, and every pixel but one for a delta-like
            # pupil), so a bare `sqrt(intensity)` yields NaN gradients.
            # Clamping the argument bounds the slope to 1/(2*sqrt(_EPS)) and
            # makes the zero-intensity pixels inert.
            observable = torch.sqrt(torch.clamp(intensity, min=_EPS))

        return self._normalize(observable)

    def _validate_inputs(self, phase_cos: torch.Tensor, phase_sin: torch.Tensor) -> None:
        """Check the phasor pair matches the basis resolution."""
        for name, tensor in (("phase_cos", phase_cos), ("phase_sin", phase_sin)):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
            if tensor.dim() != 4:
                raise ValueError(
                    f"{name} must be (B, 1, grid, grid), got shape {tuple(tensor.shape)}"
                )
            if not tensor.is_floating_point():
                raise ValueError(
                    f"{name} must be a floating point tensor, got {tensor.dtype}"
                )
        if phase_cos.shape != phase_sin.shape:
            raise ValueError(
                f"phase_cos {tuple(phase_cos.shape)} and phase_sin "
                f"{tuple(phase_sin.shape)} must have identical shapes"
            )
        if tuple(phase_cos.shape[-2:]) != (self.grid, self.grid):
            raise ValueError(
                f"input grid {tuple(phase_cos.shape[-2:])} does not match the basis "
                f"resolution ({self.grid}, {self.grid}); rebuild the model with "
                f"grid={phase_cos.shape[-1]}"
            )

    def _propagate(self, field: torch.Tensor) -> torch.Tensor:
        """Centre-pad to the far-field grid and apply the Fraunhofer FFT.

        Mirrors ``slm_shaping_bench.forward_intensity`` / ``_far_field``: the
        pupil is zero-padded symmetrically, ``ifftshift`` recentres the input
        so the zero-frequency term lands at the array centre, and ``fftshift``
        restores that centring on the output.
        """
        n = field.shape[-1]
        m = n * self.far_field_padding
        if m > n:
            start = (m - n) // 2
            padded = torch.zeros(
                field.shape[:-2] + (m, m), dtype=field.dtype, device=field.device
            )
            padded[..., start : start + n, start : start + n] = field
            field = padded

        return torch.fft.fftshift(
            torch.fft.fft2(
                torch.fft.ifftshift(field, dim=(-2, -1)),
                norm="ortho",
            ),
            dim=(-2, -1),
        )

    def _normalize(self, observable: torch.Tensor) -> torch.Tensor:
        """Apply the configured per-frame normalisation."""
        if self.normalization == "peak":
            scale = observable.amax(dim=(-2, -1), keepdim=True)
        elif self.normalization == "sum":
            scale = observable.sum(dim=(-2, -1), keepdim=True)
        else:
            return observable
        return observable / torch.clamp(scale, min=_EPS)

    def correction_phase(self) -> torch.Tensor:
        """Return the current pupil phase correction as ``(g, g)`` radians.

        This is ``sum_k c_k Z_k`` -- the raw unwrapped phase the coefficients
        currently command. Useful for inspection and for saving an SLM pattern
        (convert with ``utils.slm.phase_display.phase_to_slm_grayscale``).
        """
        return torch.einsum("k,kgh->gh", self.coefficients, self.basis)

    @torch.no_grad()
    def coefficients_array(self) -> np.ndarray:
        """Return the current coefficients as a numpy array in raw radians."""
        return self.coefficients.detach().cpu().numpy().astype(np.float64)

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------
    def update(
        self,
        phase_cos: torch.Tensor,
        phase_sin: torch.Tensor,
        target: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Perform exactly one MSE gradient step on the global coefficients.

        The loss is plain MSE between the predicted observable and the CCD
        frame, matching the training contract.

        Args:
            phase_cos: Real part of the measured phasor, ``(B, 1, g, g)``.
            phase_sin: Imaginary part, same shape as ``phase_cos``.
            target: CCD frame the loss is measured against, shaped like
                :meth:`forward`'s output.
            optimizer: Torch optimizer holding ``self.coefficients``.

        Returns:
            The scalar loss the step was computed from (before the update).
        """
        optimizer.zero_grad(set_to_none=True)
        prediction = self(phase_cos, phase_sin)
        loss = F.mse_loss(prediction, target)
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    def fit(
        self,
        phase_cos: torch.Tensor,
        phase_sin: torch.Tensor,
        target: torch.Tensor,
        config: ZernikeAmpFitConfig | None = None,
    ) -> ZernikeAmpResult:
        """Drive :meth:`update` over the data and report the best coefficients.

        Args:
            phase_cos: Real part of the measured phasor, ``(B, 1, g, g)``.
            phase_sin: Imaginary part, same shape as ``phase_cos``.
            target: CCD frames, ``(B, 1, h, w)`` matching :meth:`forward`.
            config: Fit hyper-parameters; defaults to :class:`ZernikeAmpFitConfig`.

        Returns:
            A :class:`ZernikeAmpResult` carrying the best coefficient vector.

        Raises:
            ValueError: If ``target`` does not match the forward output shape or
                the optimizer name is unknown.
        """
        config = config or ZernikeAmpFitConfig()
        if config.optimizer not in _OPTIMIZERS:
            raise ValueError(
                f"optimizer must be one of {list(_OPTIMIZERS)}, got {config.optimizer!r}"
            )

        expected = self.forward_shape(phase_cos.shape[0])
        if tuple(target.shape) != expected:
            raise ValueError(
                f"target shape {tuple(target.shape)} does not match the forward "
                f"output shape {expected} for this configuration"
            )

        optimizer = self._build_optimizer(config)
        n_samples = phase_cos.shape[0]
        batch_size = config.batch_size if config.batch_size > 0 else n_samples

        # Put the target on the same scale as the prediction, otherwise MSE
        # carries a constant the coefficients cannot influence.
        fit_target = target
        if config.normalize_target and self.normalization != "none":
            fit_target = self._normalize(target.detach().clone())

        history: list[float] = []
        best_loss = float("inf")
        best_epoch = -1
        best_coefficients = self.coefficients_array()

        started = time.perf_counter()
        for epoch in range(config.epochs):
            running = 0.0
            steps = 0
            for start in range(0, n_samples, batch_size):
                stop = min(start + batch_size, n_samples)
                running += self.update(
                    phase_cos[start:stop],
                    phase_sin[start:stop],
                    fit_target[start:stop],
                    optimizer,
                )
                steps += 1
            mean_loss = running / max(steps, 1)
            history.append(mean_loss)

            if mean_loss < best_loss:
                best_loss = mean_loss
                best_epoch = epoch
                best_coefficients = self.coefficients_array()

            if config.log_every and epoch % config.log_every == 0:
                logger.info(
                    "epoch {}/{} loss={:.6e} max|c|={:.4f} rad",
                    epoch + 1,
                    config.epochs,
                    mean_loss,
                    float(np.abs(best_coefficients).max()) if best_coefficients.size else 0.0,
                )

        seconds = time.perf_counter() - started
        logger.info(
            "fit done in {:.2f}s: best loss {:.6e} at epoch {}",
            seconds,
            best_loss,
            best_epoch,
        )
        return ZernikeAmpResult(
            coefficients=best_coefficients,
            history=history,
            best_loss=best_loss,
            best_epoch=best_epoch,
            seconds=seconds,
            observable=self.observable,
            normalization=self.normalization,
            n_max=self.n_max,
            K=self.K,
        )

    def forward_shape(self, batch: int = 1) -> tuple[int, int, int, int]:
        """Return the ``(B, 1, h, w)`` shape :meth:`forward` produces."""
        side = self.grid if self.center_crop else self.grid * self.far_field_padding
        return (batch, 1, side, side)

    def _build_optimizer(self, config: ZernikeAmpFitConfig) -> torch.optim.Optimizer:
        """Construct the optimizer over the trainable parameters.

        Uses ``self.parameters()`` rather than ``[self.coefficients]`` so a
        subclass carrying extra learnable tensors (see
        :class:`ZernikeAmpHybrid`) is optimised too. For the base class these are
        exactly the coefficients.
        """
        params = [p for p in self.parameters() if p.requires_grad]
        if config.optimizer == "adam":
            return torch.optim.Adam(
                params, lr=config.lr, weight_decay=config.weight_decay
            )
        if config.optimizer == "adamw":
            return torch.optim.AdamW(
                params, lr=config.lr, weight_decay=config.weight_decay
            )
        return torch.optim.SGD(
            params,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )


class ZernikeAmpHybrid(ZernikeAmpModel):
    """Physics envelope plus a learned residual speckle term.

    Motivation, measured: against a U-Net on the same split the base physics model
    is already statistically tied on R² (+0.8796 +- 0.0586 vs +0.9012 +- 0.0711,
    a gap far inside the 0.06-0.10 seed noise) but loses on **SSIM** (0.755 vs
    0.868, non-overlapping ranges). SSIM rewards local high-frequency texture, and
    a 135-mode *smooth* Zernike correction cannot synthesise speckle -- it fits
    the envelope only. That is a capacity limit of a global smooth basis, not
    something a larger ``n_max`` fixes (n_max=20 adds modes, not grain).

    So: keep the physics term for the envelope, and add a small CNN for the grain.

        pred = physics(phase_cos, phase_sin) + residual(phase_cos, phase_sin)

    The residual's last convolution is **zero-initialised**, so at step 0 the
    hybrid is *exactly* the physics model. Training therefore starts from the
    physical solution and can only depart from it if that reduces the loss --
    the physics is a strict starting point, not a competing guess.

    ``coefficients`` remains a first-class parameter, so the fitted Zernike
    vector is still directly readable and realisable on the SLM.
    """

    def __init__(
        self,
        config: ZernikeAmpConfig | None = None,
        *,
        residual_width: int = 32,
        **overrides: Any,
    ) -> None:
        """Build the physics model and attach a zero-initialised residual CNN.

        Args:
            config: Physics configuration; see :class:`ZernikeAmpModel`.
            residual_width: Channels in the residual CNN's hidden layers.
            **overrides: Field values forwarded to :class:`ZernikeAmpConfig`.

        Raises:
            ValueError: If ``residual_width`` < 1, or on any invalid physics config.
        """
        super().__init__(config, **overrides)
        if int(residual_width) < 1:
            raise ValueError(f"residual_width must be >= 1, got {residual_width!r}")
        width = int(residual_width)
        self.residual_width = width
        self.residual = nn.Sequential(
            nn.Conv2d(2, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.GELU(),
            nn.Conv2d(width, 1, 3, padding=1),
        )
        # Zero the output layer: the hybrid starts bit-identical to the physics
        # model, so any improvement is attributable to what training adds.
        # `self.residual[-1]` is typed as Tensor | Module, so reach the child
        # module explicitly rather than relying on narrowing.
        head = self.residual[-1]
        if not isinstance(head, nn.Conv2d):  # pragma: no cover - structural guard
            raise TypeError(f"residual head must be a Conv2d, got {type(head).__name__}")
        nn.init.zeros_(head.weight)
        if head.bias is None:  # pragma: no cover - constructed with bias=True
            raise RuntimeError("residual head needs a bias term to zero-initialise")
        nn.init.zeros_(head.bias)
        logger.info(
            "ZernikeAmpHybrid grid={} n_max={} K={} residual_width={} "
            "(residual zero-initialised)",
            self.grid,
            self.n_max,
            self.K,
            width,
        )

    def forward(self, phase_cos: torch.Tensor, phase_sin: torch.Tensor) -> torch.Tensor:
        """Predict the envelope from physics and add the learned residual.

        Args:
            phase_cos: Real part of the measured phasor, ``(B, 1, g, g)``.
            phase_sin: Imaginary part, same shape.

        Returns:
            ``(B, 1, g, g)`` (when :attr:`center_crop`), physics plus residual.
        """
        base = super().forward(phase_cos, phase_sin)
        stacked = torch.cat([phase_cos, phase_sin], dim=1)
        return base + self.residual(stacked)