"""Differentiable beam shaping via gradient-descent SLM phase optimization.

This module provides a fully differentiable forward model (phase → FFT/ASM
propagation → far-field intensity) implemented in PyTorch, enabling direct
gradient-based optimization of an SLM phase map.  The objective minimises a
weighted combination of uniformity, efficiency, zero-order suppression, and
smoothness losses against a target intensity mask.

PyTorch is **optional** at import time — every torch symbol is accessed
through the :func:`_torch` lazy accessor so that the rest of the project can
``import`` this module without torch installed.  Callers who invoke any
torch-dependent function will receive a clear :class:`ImportError` if the
dependency is missing.

Example:
    >>> from ao_shaping.algorithm.differentiable_shaping import (
    ...     create_target_mask, train_beam_shaping,
    ... )
    >>> mask = create_target_mask("square", (128, 128), 40)
    >>> result = train_beam_shaping(mask, (128, 128), iterations=200, seed=42)
    >>> result.phase.shape
    (128, 128)
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:  # pragma: no cover – type-only imports
    from typing import Any

    from torch import Tensor


# ---------------------------------------------------------------------------
# Lazy torch accessor
# ---------------------------------------------------------------------------

def _torch():
    """Return the ``torch`` module, raising :class:`ImportError` when absent.

    Returns:
        The ``torch`` top-level module.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    try:
        import torch as _t
    except ImportError:
        raise ImportError(
            "differentiable_shaping requires PyTorch. "
            "Install with: uv sync --extra ml"
        ) from None
    return _t


# ---------------------------------------------------------------------------
# Target mask creation
# ---------------------------------------------------------------------------

def create_target_mask(
    shape: str,
    grid_size: tuple[int, int],
    size: int,
    *,
    sigma: float | None = None,
) -> np.ndarray:
    """Create a normalised target intensity mask.

    Args:
        shape: One of ``"square"``, ``"circle"``, ``"gaussian"``, ``"spot"``.
        grid_size: ``(H, W)`` of the output array.
        size: Characteristic dimension in pixels.
            * square — side length
            * circle — diameter
            * gaussian — ``sigma`` defaults to ``size / 6`` if not given
            * spot — diameter (focused spot, same as circle)
        sigma: Override for the Gaussian standard deviation (pixels).

    Returns:
        ``(H, W)`` ``float64`` mask with values in ``[0, 1]``, centred.

    Raises:
        ValueError: On invalid *shape* or non-2-D *grid_size*.
    """
    valid_shapes = {"square", "circle", "gaussian", "spot"}
    if shape not in valid_shapes:
        raise ValueError(
            f"Invalid shape {shape!r}. Must be one of {valid_shapes}"
        )
    if len(grid_size) != 2:
        raise ValueError(
            f"grid_size must be a 2-tuple, got {len(grid_size)}D"
        )

    H, W = grid_size
    cy, cx = H / 2.0, W / 2.0
    yy, xx = np.mgrid[0:H, 0:W]

    if shape == "square":
        half = size / 2.0
        mask = (
            (np.abs(xx - cx) <= half) & (np.abs(yy - cy) <= half)
        ).astype(np.float64)

    elif shape in {"circle", "spot"}:
        radius = size / 2.0
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        mask = (r2 <= radius**2).astype(np.float64)

    else:  # gaussian
        sig = sigma if sigma is not None else size / 6.0
        mask = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sig**2))
        peak = mask.max()
        if peak > 0:
            mask = mask / peak

    return mask.astype(np.float64)


# ---------------------------------------------------------------------------
# Loss functions (torch tensors in → scalar tensor out)
# ---------------------------------------------------------------------------

def uniformity_loss(intensity: "Tensor", target: "Tensor") -> "Tensor":
    """Coefficient of variation of intensity inside the target region.

    Lower values mean more uniform illumination.  Returns zero when the
    target region is empty or contains only zeros (NaN-safe).
    """
    torch = _torch()
    mask = target > 0
    vals = intensity[mask]
    if vals.numel() == 0:
        return torch.tensor(0.0, device=intensity.device, dtype=intensity.dtype)
    mu = vals.mean()
    if mu.abs() < 1e-12:
        return torch.tensor(0.0, device=intensity.device, dtype=intensity.dtype)
    return vals.std() / (mu.abs() + 1e-12)


def efficiency_loss(intensity: "Tensor", target: "Tensor") -> "Tensor":
    """One minus the fraction of total energy falling inside the target.

    Lower values mean more energy is concentrated where desired.
    """
    torch = _torch()
    total = intensity.sum()
    if total < 1e-12:
        return torch.tensor(1.0, device=intensity.device, dtype=intensity.dtype)
    target_energy = (intensity * target).sum()
    encircled = target_energy / (total + 1e-12)
    return 1.0 - encircled.clamp(0.0, 1.0)


def zero_order_penalty(intensity: "Tensor") -> "Tensor":
    """Penalty for a strong DC / zero-order spike at the field centre.

    Returns the mean intensity within a small central window (5×5 pixels)
    normalised by the total intensity — higher means worse suppression.
    """
    torch = _torch()
    H, W = intensity.shape[-2:]
    cy, cx = H // 2, W // 2
    r = 2  # half-window → 5×5 region
    region = intensity[..., cy - r : cy + r + 1, cx - r : cx + r + 1]
    total = intensity.sum()
    if total < 1e-12:
        return torch.tensor(0.0, device=intensity.device, dtype=intensity.dtype)
    return region.mean() / (total / (H * W) + 1e-12)


def smoothness_regularization(phase: "Tensor") -> "Tensor":
    """Finite-difference penalty on phase gradients.

    Returns zero for a constant phase map and grows with high-frequency
    spatial variation.
    """
    torch = _torch()
    dy = phase[:, 1:] - phase[:, :-1]
    dx = phase[1:, :] - phase[:-1, :]
    return (dy**2).mean() + (dx**2).mean()


def total_loss(
    intensity: "Tensor",
    phase: "Tensor",
    target: "Tensor",
    *,
    w_uniformity: float = 0.4,
    w_efficiency: float = 0.6,
    w_zero_order: float = 0.0,
    w_smoothness: float = 0.0,
) -> "Tensor":
    """Weighted sum of all loss components.

    All individual terms are non-negative and roughly comparable in scale.

    .. note::
        Empirically verified weights (docs/slm_differential_shaping/): the
        zero-order penalty MUST be 0 for targets centred on the beam origin —
        suppressing the DC (centre) pushes energy OUT of a centred square/spot
        and collapses encircled energy (EE 0.84 -> 0.07 at
        ``w_zero_order=0.1``).  Smoothness regularisation similarly fights the
        high-frequency phase content needed for sharp square edges; a 0 weight
        gives the best result.  ``w_efficiency >= w_uniformity`` concentrates
        energy first, then uniformity flattens it (600+ iterations reach
        CV<0.1; see the report for learning curves).
    """
    return (
        w_uniformity * uniformity_loss(intensity, target)
        + w_efficiency * efficiency_loss(intensity, target)
        + w_zero_order * zero_order_penalty(intensity)
        + w_smoothness * smoothness_regularization(phase)
    )


# ---------------------------------------------------------------------------
# Differentiable propagation
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=32)
def _asm_propagator_torch(
    grid_shape: tuple[int, int],
    dx: float,
    z: float,
    wavelength: float,
    device_str: str,
    dtype_name: str,
):
    """Cache the ASM propagator for a given (shape, dx, z, λ, device, dtype).

    The propagator depends only on the optical geometry, not on the field
    itself, so it can be safely shared across calls.

    Returns:
        A complex tensor of shape ``(H, W)``.
    """
    torch = _torch()
    dtype = getattr(torch, dtype_name)
    H, W = grid_shape
    # Compute in float64 for numerical parity with the numpy reference
    # implementation, then cast to the field's dtype.
    fx = torch.fft.fftfreq(W, dx, device=device_str, dtype=torch.float64)
    fy = torch.fft.fftfreq(H, dx, device=device_str, dtype=torch.float64)
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")
    k = 2.0 * math.pi / wavelength
    kx = 2.0 * math.pi * FX
    ky = 2.0 * math.pi * FY
    kz_sq = k**2 - kx**2 - ky**2
    evanescent = kz_sq < 0
    kz_sq = torch.where(evanescent, torch.zeros_like(kz_sq), kz_sq)
    H_prop = torch.exp(1j * torch.sqrt(kz_sq) * z)
    H_prop[evanescent] = 0.0
    # ifftshift aligns the propagator with the fft2 output ordering, matching
    # the numpy reference in ao_shaping.algorithm.gerchberg_saxton.
    return torch.fft.ifftshift(H_prop).to(dtype=dtype)


def angular_spectrum_propagate_torch(
    field: "Tensor",
    dx: float,
    z: float,
    wavelength: float,
) -> "Tensor":
    """Angular Spectrum Method propagation using PyTorch FFTs.

    Mirrors the numpy ``angular_spectrum_propagate`` in
    ``ao_shaping.algorithm.gerchberg_saxton`` but operates on torch tensors
    with a cached propagator.

    Args:
        field: Complex field tensor ``(H, W)``.
        dx: Pixel spacing in metres.
        z: Propagation distance (positive = forward).
        wavelength: Wavelength in metres.

    Returns:
        Propagated complex field, same shape as *field*.
    """
    torch = _torch()
    if field.ndim != 2:
        raise ValueError(f"Field must be 2D, got {field.ndim}D")
    H, W = field.shape[-2:]
    device_str = str(field.device)
    dtype_name = str(field.dtype).split(".")[-1]  # e.g. "complex64"
    H_prop = _asm_propagator_torch(
        (H, W), dx, z, wavelength, device_str, dtype_name,
    )
    F = torch.fft.fft2(field)
    return torch.fft.ifft2(F * H_prop)


# ---------------------------------------------------------------------------
# Forward model (factory — avoids nn.Module at import time)
# ---------------------------------------------------------------------------

def _build_forward(
    propagation: str,
    cell_spacing: float,
    distance: float,
    wavelength: float,
) -> Callable[["Tensor", "Tensor | None"], "Tensor"]:
    """Build the differentiable forward model as a closure.

    This avoids class definitions that depend on torch at import time while
    still keeping the propagation parameters local.

    Returns:
        Callable ``(phase, source_amplitude=None) -> real_intensity``.
    """
    torch = _torch()

    if propagation not in ("fft", "asm"):
        raise ValueError(
            f"propagation must be 'fft' or 'asm', got {propagation!r}"
        )

    def forward(
        phase: "Tensor",
        source_amplitude: "Tensor | None" = None,
    ) -> "Tensor":
        H, W = phase.shape[-2:]
        if source_amplitude is None:
            amp = torch.ones((H, W), device=phase.device, dtype=phase.dtype)
        else:
            amp = source_amplitude

        field = amp * torch.exp(1j * phase)

        if propagation == "fft":
            out_field = torch.fft.fftshift(torch.fft.fft2(field))
        else:  # asm
            out_field = angular_spectrum_propagate_torch(
                field, cell_spacing, distance, wavelength,
            )

        intensity = out_field.real**2 + out_field.imag**2
        return intensity

    return forward


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DifferentiableShapingResult:
    """Outcome of a differentiable beam-shaping optimisation run.

    Attributes:
        phase: Best/final ``(H, W)`` ``float64`` phase in radians.
        target_intensity: ``(H, W)`` target mask used for optimisation.
        simulated_intensity: ``(H, W)`` intensity produced by the final phase.
        loss_history: Per-iteration total loss values.
        iterations: Number of iterations performed.
        converged: Whether the run completed all requested iterations.
    """
    phase: np.ndarray
    target_intensity: np.ndarray
    simulated_intensity: np.ndarray
    loss_history: list[float]
    iterations: int
    converged: bool


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_beam_shaping(
    target: "np.ndarray | Tensor",
    grid_size: tuple[int, int],
    *,
    source_amplitude: "np.ndarray | Tensor | None" = None,
    initial_phase: "np.ndarray | Tensor | None" = None,
    propagation: str = "fft",
    optimizer: str = "adam",
    iterations: int = 300,
    lr: float = 3e-2,
    w_uniformity: float = 0.4,
    w_efficiency: float = 0.6,
    w_zero_order: float = 0.0,
    w_smoothness: float = 0.0,
    cell_spacing: float = 8e-6,
    distance: float = 0.1,
    wavelength: float = 1064e-9,
    device: str | None = None,
    seed: int | None = None,
    progress_callback: Callable[[int, float], None] | None = None,
    phase_callback: Callable[["Tensor"], None] | None = None,
) -> DifferentiableShapingResult:
    """Optimise an SLM phase map via gradient descent.

    Args:
        target: ``(H, W)`` target intensity mask (numpy or torch).
        grid_size: ``(H, W)`` grid dimensions.
        source_amplitude: Illumination amplitude. Uniform if *None*.
        initial_phase: Starting phase (radians). Zeros if *None*.
        propagation: ``"fft"`` (Fraunhofer) or ``"asm"`` (Angular Spectrum).
        optimizer: ``"adam"`` or ``"lbfgs"``.
        iterations: Number of optimisation steps.
        lr: Learning rate.
        w_uniformity: Weight for uniformity loss.
        w_efficiency: Weight for efficiency loss.
        w_zero_order: Weight for zero-order penalty.
        w_smoothness: Weight for smoothness regularisation.
        cell_spacing: Pixel pitch in metres.
        distance: Propagation distance in metres.
        wavelength: Wavelength in metres.
        device: ``"cuda"``, ``"cpu"``, or *None* for auto-detect.
        seed: Random seed for reproducibility.
        progress_callback: ``fn(iteration, loss)`` called each step.
        phase_callback: ``fn(phase_tensor)`` called each step.

    Returns:
        :class:`DifferentiableShapingResult` with all fields populated.

    Raises:
        ValueError: On invalid inputs.
    """
    torch = _torch()

    # -- Validate -----------------------------------------------------------
    if propagation not in ("fft", "asm"):
        raise ValueError(
            f"propagation must be 'fft' or 'asm', got {propagation!r}"
        )
    if optimizer not in ("adam", "lbfgs"):
        raise ValueError(
            f"optimizer must be 'adam' or 'lbfgs', got {optimizer!r}"
        )
    if len(grid_size) != 2:
        raise ValueError(
            f"grid_size must be a 2-tuple, got {len(grid_size)}D"
        )
    H, W = grid_size
    if H <= 0 or W <= 0:
        raise ValueError(
            f"grid_size must have positive dimensions, got {(H, W)}"
        )
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")

    # Validate target dimensionality and that it matches grid_size
    if isinstance(target, np.ndarray):
        if target.ndim != 2:
            raise ValueError(
                f"target must be 2D, got {target.ndim}D"
            )
        target_shape = target.shape
    else:  # torch tensor
        if target.dim() != 2:
            raise ValueError(
                f"target must be 2D, got {target.dim()}D"
            )
        target_shape = tuple(target.shape)

    if target_shape != (H, W):
        raise ValueError(
            f"target shape {target_shape} does not match grid_size {(H, W)}"
        )

    # -- Reproducibility ----------------------------------------------------
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # -- Device -------------------------------------------------------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    dtype = torch.float32

    # -- Prepare tensors ----------------------------------------------------
    if isinstance(target, np.ndarray):
        t_tensor = torch.from_numpy(target).to(device=dev, dtype=dtype)
    else:
        t_tensor = target.to(device=dev, dtype=dtype)

    if source_amplitude is not None:
        if isinstance(source_amplitude, np.ndarray):
            amp = torch.from_numpy(source_amplitude).to(device=dev, dtype=dtype)
        else:
            amp = source_amplitude.to(device=dev, dtype=dtype)
    else:
        amp = torch.ones((H, W), device=dev, dtype=dtype)

    if initial_phase is not None:
        if isinstance(initial_phase, np.ndarray):
            phase_init = torch.from_numpy(initial_phase).to(
                device=dev, dtype=dtype,
            )
        else:
            phase_init = initial_phase.to(device=dev, dtype=dtype)
    else:
        # Small random noise avoids the degenerate zero-gradient start where
        # the field is purely real (intensity is quadratic in phase there).
        phase_init = 0.1 * torch.randn((H, W), device=dev, dtype=dtype)

    phase = torch.nn.Parameter(phase_init.clone())

    # -- Forward model ------------------------------------------------------
    forward = _build_forward(propagation, cell_spacing, distance, wavelength)

    # -- Optimiser ----------------------------------------------------------
    loss_history: list[float] = []

    def _closure():
        """Single forward+backward pass shared by both Adam and L-BFGS."""
        opt.zero_grad()
        intensity = forward(phase, amp)
        loss = total_loss(
            intensity,
            phase,
            t_tensor,
            w_uniformity=w_uniformity,
            w_efficiency=w_efficiency,
            w_zero_order=w_zero_order,
            w_smoothness=w_smoothness,
        )
        loss.backward()
        return loss

    if optimizer == "adam":
        # `Any` because torch is optional — the concrete optimizer type is
        # only known at runtime from the `optimizer` string.
        opt: Any = torch.optim.Adam([phase], lr=lr)
    else:
        opt = torch.optim.LBFGS(
            [phase],
            lr=lr,
            max_iter=20,
            history_size=10,
            line_search_fn="strong_wolfe",
        )

    # -- Training loop ------------------------------------------------------
    converged = False
    logger.info(
        "Starting differentiable beam shaping: {} iterations, "
        "propagation={}, optimizer={}, device={}",
        iterations, propagation, optimizer, device,
    )

    for it in range(iterations):
        if optimizer == "adam":
            opt.zero_grad()
            loss = _closure()
            if torch.isnan(loss):
                logger.warning("NaN loss at iteration {}; stopping early", it)
                break
            opt.step()
        else:  # lbfgs — closure is invoked internally by the optimiser
            loss = opt.step(_closure)
            if loss is None or torch.isnan(loss):
                logger.warning("L-BFGS returned NaN/None at iteration {}; stopping early", it)
                break

        # Wrap phase into [0, 2π).  For Adam this is safe each step; for
        # L-BFGS wrapping mid-run would break its gradient-history bookkeeping,
        # so we only wrap at the very end.
        if optimizer == "adam":
            with torch.no_grad():
                phase.data = phase.data % (2.0 * math.pi)

        loss_val = float(loss.item())
        loss_history.append(loss_val)

        if progress_callback is not None:
            progress_callback(it, loss_val)

        if phase_callback is not None:
            phase_callback(phase.detach())

        if (it + 1) % max(1, iterations // 10) == 0 or it == 0:
            logger.debug("Iter {}/{}  loss={:.6f}", it + 1, iterations, loss_val)

    if len(loss_history) >= iterations:
        converged = True

    # -- Extract results ----------------------------------------------------
    with torch.no_grad():
        final_intensity = forward(phase, amp)
        final_phase = (phase.detach() % (2.0 * math.pi)).cpu().numpy().astype(
            np.float64,
        )
        final_intensity_np = final_intensity.detach().cpu().numpy().astype(
            np.float64,
        )
        if isinstance(target, np.ndarray):
            target_np = target.astype(np.float64)
        else:
            target_np = target.detach().cpu().numpy().astype(np.float64)

    logger.info(
        "Beam shaping complete: final loss={:.6f}, converged={}",
        loss_history[-1] if loss_history else float("nan"),
        converged,
    )

    return DifferentiableShapingResult(
        phase=final_phase,
        target_intensity=target_np,
        simulated_intensity=final_intensity_np,
        loss_history=loss_history,
        iterations=len(loss_history),
        converged=converged,
    )
