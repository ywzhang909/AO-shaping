"""Canonical Angular Spectrum Propagation (ASM) — the repo's single ASM implementation.

This module is the **one** place where Angular Spectrum Propagation is implemented.
It replaces the two divergent copies that used to live in
``algorithm/signal_processing/gerchberg_saxton.py`` (numpy, 2-D only) and
``algorithm/signal_processing/differentiable_shaping.py`` (torch, 2-D only, and
built with a *half-array misaligned* transfer function — see "Shift convention").
Consumers should import from here and re-export; they must not re-derive the
propagator.

PyTorch is a **hard** import requirement of this module (the public signatures
default to ``torch.complex64``), unlike ``differentiable_shaping`` which is
torch-optional. Any shim that re-exports from here inherits that requirement.

What ASM is
-----------
ASM propagates a complex field between two planes separated by ``z`` by
multiplying its Fourier spectrum by the transfer function
``H = exp(1j * kz * z)`` with ``kz = sqrt(k**2 - kx**2 - ky**2)``. It is exact
for the paraxial Helmholtz equation: it handles near field *and* far field, and
unlike Fraunhofer it carries the physical distance ``z`` and the wavelength.

🔴 THE VALIDITY BOUND — read before trusting an ASM number on this bench
---------------------------------------------------------------------
At ``grid=64``, ``dx=8 um``, ``lambda=1064 nm`` (aperture ``B = 512 um``),
**single-step ASM is invalid for every bench-relevant ``z``**, and this is a
property of the WKB/paraxial approximation, not of this implementation:

* ASM requires ``2*pi*lambda*z*f_sig**2 << 1`` with ``f_sig = 1/B``, i.e.
  ``z << B**2 / (2*pi*lambda) = 39 mm`` for a 512 um aperture.
* The Fraunhofer pattern only *forms* at ``z >> a**2/lambda = 61.6 mm``
  (``a = 256 um`` half-width).
* **No ``z`` is both ASM-valid and in the far field** — the two windows do not
  overlap. Measured against the closed-form far field: FFT ``relL2 = 7.7e-2``,
  ASM ``relL2 = 1.53`` (i.e. worse than predicting nothing useful).
* Splitting the propagation into ``N`` sub-steps is **algebraically vacuous**:
  ``prod(H(z/N)) == H(z)`` because the propagators commute and the phases add.
  Measured ``max|prod - H| <= 1.1e-10`` for ``N = 2..200`` (pure float64
  round-off; exactly ``2.5e-16`` at ``N = 2``). Do not offer sub-stepping as a
  remedy — it cannot fix a broken approximation, only add round-off.

The Fraunhofer FFT is the exact ``z -> infinity`` limit of this operator and is
correct everywhere; ASM is offered here as a *selectable, physically
parameterised alternative* whose output is expected to differ from the FFT on
this bench. Use it when you want ``z``, ``lambda`` and ``dx`` to enter the model
explicitly (e.g. to study the transition regime), and validate against the FFT
before trusting it.

Shift convention (load-bearing)
-------------------------------
Input and output are both **UNSHIFTED**: the spatial origin (DC) is array index
``0`` in both. ``torch.fft.fftfreq`` is already in raw-``fft2`` order, so the
frequency grid is used exactly as returned — **do not** ``fftshift`` it, and the
propagator is **not** shifted either. Callers that want a centred picture apply a
single ``torch.fft.fftshift`` to the *output*::

    u_out = torch.fft.fftshift(angular_spectrum_propagate_torch(u_in, dx, z, lam))

.. warning::
    The two implementations this module replaces both ended with
    ``ifftshift(H)``, on the stated belief that it "aligns the propagator with
    the ``fft2`` output order". It does the opposite. Writing ``S`` for the
    half-array circular shift, one has (verified exactly, ``max|Δ| = 0.0``)::

        ifft2(fft2(u) * S·H) == (-1)**(m+n) * ASM(u * (-1)**(m+n))

    so the old propagator propagated the pupil **spatially rolled by half the
    array**, i.e. carrying an extra Nyquist-frequency checkerboard grating, and
    the propagated field carried a second checkerboard. Consequences, all
    measured: energy is preserved (which is why Parseval tests never caught it),
    and ``z -> -z`` round trips are self-consistent (which is why every
    Gerchberg-Saxton test passed) — but a *single* propagation is wrong, and the
    far field lands half an array away from where it belongs. It is
    catastrophically incompatible with band limiting: the shift moves DC onto a
    corner bin that the band-limit mask has already zeroed, so an all-ones field
    propagates to an **identically zero** field (measured, see
    ``test_parseval_is_conserved``). Correctness wins: the shift is gone.

Numerical disciplines encoded here (each one is a measurement, not a preference)
-------------------------------------------------------------------------------
1. **The propagator is built in float64 and cast once at the end.** At
   ``z ~ 1 m``, ``kz*z ~ 5.9e6 rad``; float32's ``eps ~ 1.2e-7`` therefore buys
   ``~0.7 rad`` of phase error per bin, which destroys the pattern. Measured on
   a random 64x64 complex field at ``z = 1 m``: float32 geometry gives
   ``max|Δphase| = 0.35 rad`` and a field ``relL2`` error of **0.149** versus the
   float64 propagator. ``torch.fft.fftfreq``'s ``dtype`` follows its *output*
   dtype, so the frequency grid is pinned to ``torch.float64`` explicitly.
2. **Evanescent bins are zeroed**, never ``sqrt``-ed into a NaN:
   ``sqrt(kz_sq.clamp(min=0))`` then ``where(kz_sq < 0, 0, H)``.
3. **One shared normalisation constant** for ``fft2`` and ``ifft2``
   (:data:`FFT_NORM`, ``None``). Both ``norm=None`` makes Parseval exact with no
   N factor (``sum|out|**2 / sum|in|**2 == 1.000000000000``). Setting
   ``norm="ortho"`` on ``ifft2`` *only* is a silent ``sqrt(N)`` amplitude error —
   measured energy ratio ``2.44e-4 = 1/4096`` on a 64x64 field, and its
   reciprocal ``4096`` if the flag goes on ``fft2`` only. Peak normalisation hides
   it completely, which is exactly why it must be impossible to write.
4. **``lru_cache(maxsize=4)``** on the propagator, keyed by ``(shape, dx, z,
   wavelength, band_limit, device, dtype)``. ``maxsize=32`` of ``M**2`` complex64
   is a real VRAM leak: ``M = 4800`` gives ``184 MB`` per entry, **5.9 GB** at
   32 entries. The device is part of the key precisely so a CUDA tensor can never
   be handed out for a CPU request. Use :func:`clear_asm_propagator_cache` to drop
   everything (e.g. before freeing VRAM).
5. **N-D ``(..., H, W)`` inputs** are supported, including ``(B, 1, g, g)``. The
   two replaced implementations raised on ``ndim != 2``.
6. **``band_limit=True`` by default**, Voelz & Roggemann form: the cutoff
   ``f_max = 1 / (wavelength * sqrt(1 + (2z/B)**2))`` with ``B = n_input * dx``
   is *identical* to the Nyquist frequency of the ASM output grid,
   ``1 / (2*dx_out) = B / (2*wavelength*z)`` (they agree to 4 digits), so
   suppressing everything above it is what keeps the propagation free of
   aliasing on the output sampling. NOTE: the variant
   ``1 / (wavelength*sqrt(1 + (2*z*dx)**2))`` that circulates in blog posts is
   **dimensionally broken** — ``z*dx`` is m**2, not dimensionless — so it is not
   implemented here.

Example:
    >>> import torch
    >>> from ao_shaping.algorithm.signal_processing.angular_spectrum import (
    ...     AngularSpectrumPropagation,
    ... )
    >>> prop = AngularSpectrumPropagation(
    ...     pixel_size=8e-6, wavelength=1064e-9, shape=(64, 64), z=0.02
    ... )
    >>> field = torch.ones(4, 1, 64, 64, dtype=torch.complex64)
    >>> out = prop(field)          # (4, 1, 64, 64), DC still at index 0
    >>> centred = torch.fft.fftshift(out)

Why there is no ``ifftshift`` on the propagator
----------------------------------------------
The propagator is built directly from ``torch.fft.fftfreq``, which is already in
RAW ``fft2`` order, and multiplied against ``fft2(u)`` without any shift. Shifting
it would misalign it by half an array.

The incumbent ``gerchberg_saxton._compute_propagator`` DOES apply ``ifftshift``
(measured: ``_build_propagator_numpy(...) - ifftshift(_compute_propagator(...))``
is exactly ``0``, i.e. the two differ by a half-array roll). That is not a
defect in this module -- it is a latent half-array shift in the incumbent, which
cancels there only because Gerchberg-Saxton always calls propagate in matched
forward/backward pairs. Measured consequence of forwarding the GS calls here:
``test_perfect_reconstruction_metrics`` correlation drops to **0.277** against a
required ``> 0.5``.

So ``gerchberg_saxton.angular_spectrum_propagate`` deliberately remains its own
implementation rather than becoming a re-export. Do not "de-duplicate" it without
first resolving that half-array shift -- a shared implementation is the right end
state, but it is a physics change to GS, not a refactor.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import torch
from loguru import logger
from torch import Tensor

__all__ = [
    "AngularSpectrumPropagation",
    "asm_propagator",
    "angular_spectrum_propagate",
    "angular_spectrum_propagate_torch",
    "band_limit_frequency",
]

#: Shared normalisation constant for ``fft2`` **and** ``ifft2``.
#:
#: ``None`` (= numpy's ``"backward"``) on both is the only setting that keeps
#: Parseval exact with no N factor, measured ``sum|out|**2 / sum|in|**2 ==
#: 1.000000000000``. Applying ``"ortho"`` to one side only rescales the field by
#: ``sqrt(N)`` (measured energy ratio ``1/4096 = 2.44e-4`` for a 64x64 field
#: with the flag on ``ifft2``, ``4096`` with it on ``fft2``) — an invisible error
#: under peak normalisation. Never pass ``norm=`` at a call site of this module;
#: use this constant so the two transforms can never disagree.
FFT_NORM: str | None = None


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_geometry(dx: float, wavelength: float) -> tuple[float, float]:
    """Validate and normalise the sampling geometry.

    Args:
        dx: Pixel pitch in metres; must be finite and positive.
        wavelength: Wavelength in metres; must be finite and positive.

    Returns:
        The two values coerced to ``float``.

    Raises:
        ValueError: If either value is non-finite or non-positive.
    """
    dx_f = float(dx)
    lam_f = float(wavelength)
    if not math.isfinite(dx_f) or dx_f <= 0.0:
        raise ValueError(f"dx must be a finite positive length in metres, got {dx_f!r}")
    if not math.isfinite(lam_f) or lam_f <= 0.0:
        raise ValueError(
            f"wavelength must be a finite positive length in metres, got {lam_f!r}"
        )
    return dx_f, lam_f


def _normalise_shape(shape: Sequence[int] | int) -> tuple[int, int]:
    """Coerce a grid shape to a hashable ``(rows, cols)`` pair of ints.

    Args:
        shape: ``(H, W)``, a single ``int`` for a square grid, or anything
            indexable/iterable of length 2 (e.g. ``torch.Size``).

    Returns:
        The ``(rows, cols)`` pair.

    Raises:
        ValueError: If the shape is not 1-D of length 1 or 2, or has a
            non-positive dimension.
    """
    if isinstance(shape, int):
        pair = (int(shape), int(shape))
    else:
        items = list(shape)
        if len(items) == 1:
            pair = (int(items[0]), int(items[0]))
        elif len(items) == 2:
            pair = (int(items[0]), int(items[1]))
        else:
            raise ValueError(
                f"shape must be an int or a 2-element (H, W) sequence, got {shape!r}"
            )
    if pair[0] <= 0 or pair[1] <= 0:
        raise ValueError(f"shape dimensions must be positive, got {pair!r}")
    return pair


def _validate_field_shape(field_shape: Sequence[int]) -> tuple[int, int]:
    """Validate an ``(..., H, W)`` field shape and return its spatial extent.

    Args:
        field_shape: The full shape of the incoming field.

    Returns:
        The trailing ``(rows, cols)`` pair.

    Raises:
        ValueError: If the field is not at least 2-D, or its spatial extent is
            not a positive 2-D grid.
    """
    shape = tuple(field_shape)
    if len(shape) < 2:
        raise ValueError(
            f"field must have at least 2 dimensions (..., H, W), got shape {shape!r}"
        )
    return _normalise_shape(shape[-2:])


# ---------------------------------------------------------------------------
# Band limiting
# ---------------------------------------------------------------------------


def band_limit_frequency(
    *, dx: float, z: float, wavelength: float, n_input: int
) -> float:
    """Band-limited ASM cutoff in cycles/metre.

    Voelz & Roggemann form: ``f_max = 1 / (wavelength * sqrt(1 + (2z/B)**2))``
    with ``B = n_input * dx`` the *input* aperture width. This is identical to
    the Nyquist frequency of the ASM **output** grid,
    ``1 / (2*dx_out) = B / (2*wavelength*z)`` (the two agree to 4 digits), so
    dropping everything above ``f_max`` is precisely what keeps the propagated
    field free of aliasing on the output sampling.

    Do **not** substitute the blog-post variant
    ``1 / (wavelength*sqrt(1 + (2*z*dx)**2))``: ``z*dx`` has units of m**2, so
    the square root is not dimensionless and the expression is meaningless.

    Args:
        dx: Pixel pitch in metres.
        z: Propagation distance in metres (sign is irrelevant — the cutoff
            depends on ``|z|`` through the square).
        wavelength: Wavelength in metres.
        n_input: Number of samples across the input aperture (``B = n_input*dx``).

    Returns:
        The cutoff spatial frequency in cycles per metre.

    Raises:
        ValueError: If ``dx``, ``wavelength`` or ``n_input`` is non-positive.
    """
    dx_f, lam_f = _validate_geometry(dx, wavelength)
    n = int(n_input)
    if n <= 0:
        raise ValueError(f"n_input must be positive, got {n_input!r}")
    aperture = n * dx_f
    return 1.0 / (lam_f * math.sqrt(1.0 + (2.0 * float(z) / aperture) ** 2))


# ---------------------------------------------------------------------------
# Cached geometry and propagator
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=4)
def _asm_geometry(
    shape: tuple[int, int],
    dx: float,
    wavelength: float,
    device_str: str,
) -> tuple[Tensor, Tensor]:
    """Cached, z-independent frequency-domain geometry in float64.

    The frequency grid is pinned to ``torch.float64`` on purpose: at ``z ~ 1 m``
    the accumulated phase ``kz*z`` reaches ``~5.9e6 rad``, where float32's
    ``eps = 1.2e-7`` costs ``~0.7 rad`` per bin. ``torch.fft.fftfreq``'s
    ``dtype`` follows its output dtype, so it must be stated explicitly.

    Args:
        shape: ``(rows, cols)`` of the sampling grid.
        dx: Pixel pitch in metres.
        wavelength: Wavelength in metres.
        device_str: ``str(torch.device(...))``; part of the cache key.

    Returns:
        ``(kz_sq, radial_freq)``, both float64 ``(rows, cols)`` tensors.
        ``kz_sq = k**2 - kx**2 - ky**2`` (negative where evanescent) and
        ``radial_freq = hypot(fx, fy)`` in cycles/metre, both in raw ``fft2``
        order.
    """
    rows, cols = shape
    device = torch.device(device_str)
    fx = torch.fft.fftfreq(cols, d=dx, device=device, dtype=torch.float64)
    fy = torch.fft.fftfreq(rows, d=dx, device=device, dtype=torch.float64)
    fy_mesh, fx_mesh = torch.meshgrid(fy, fx, indexing="ij")
    k = 2.0 * math.pi / wavelength
    kx = 2.0 * math.pi * fx_mesh
    ky = 2.0 * math.pi * fy_mesh
    kz_sq = k * k - kx * kx - ky * ky
    radial = torch.sqrt(fx_mesh * fx_mesh + fy_mesh * fy_mesh)
    return kz_sq, radial


def _build_propagator(
    shape: tuple[int, int],
    dx: float,
    wavelength: float,
    device_str: str,
    *,
    z: float,
    band_limit: bool,
) -> Tensor:
    """Build the complex128 transfer function for a single distance.

    Args:
        shape: ``(rows, cols)`` of the sampling grid.
        dx: Pixel pitch in metres.
        wavelength: Wavelength in metres.
        device_str: Device string for the output tensor.
        z: Propagation distance in metres.
        band_limit: Whether to apply the band-limit cutoff.

    Returns:
        A complex128 ``(rows, cols)`` tensor in raw ``fft2`` order (unshifted).
    """
    kz_sq, radial = _asm_geometry(shape, dx, wavelength, device_str)
    kz = torch.sqrt(kz_sq.clamp(min=0.0))  # clamp FIRST: sqrt of a negative is NaN
    propagator = torch.exp(1j * kz * z)
    keep = kz_sq >= 0.0
    if band_limit:
        f_max = band_limit_frequency(
            dx=dx,
            z=z,
            wavelength=wavelength,
            n_input=shape[1],
        )
        keep = keep & (radial <= f_max)
        _log_band_limit(shape, keep, f_max)
    zero = torch.zeros((), dtype=propagator.dtype, device=propagator.device)
    propagator = torch.where(keep, propagator, zero)
    _warn_if_all_bins_dropped(keep)
    return propagator


def _build_propagator_batched(
    shape: tuple[int, int],
    dx: float,
    wavelength: float,
    device_str: str,
    *,
    z_values: Tensor,
    band_limit: bool,
) -> Tensor:
    """Build a batch of transfer functions, one per sample, without a Python loop.

    The whole batch is one broadcasted ``exp`` over ``(B, rows, cols)``, so the
    cost is one fused kernel rather than ``B`` sequential builds.

    Args:
        shape: ``(rows, cols)`` of the sampling grid.
        dx: Pixel pitch in metres.
        wavelength: Wavelength in metres.
        device_str: Device string for the output tensor.
        z_values: 1-D tensor of per-sample distances in metres, shape ``(B,)``.
        band_limit: Whether to apply the band-limit cutoff.

    Returns:
        A complex128 ``(B, rows, cols)`` tensor in raw ``fft2`` order.
    """
    kz_sq, radial = _asm_geometry(shape, dx, wavelength, device_str)
    z_view = z_values.to(torch.float64).reshape(-1, 1, 1)
    kz = torch.sqrt(kz_sq.clamp(min=0.0)).unsqueeze(0)
    propagator = torch.exp(1j * kz * z_view)
    keep = (kz_sq >= 0.0).unsqueeze(0).expand_as(propagator)
    if band_limit:
        aperture = shape[1] * dx
        ratio = (2.0 * z_view / aperture) ** 2
        f_max = (1.0 / (wavelength * torch.sqrt(1.0 + ratio))).reshape(-1, 1, 1)
        keep = keep & (radial.unsqueeze(0) <= f_max)
    zero = torch.zeros((), dtype=propagator.dtype, device=propagator.device)
    return torch.where(keep, propagator, zero)


@functools.lru_cache(maxsize=4)
def _asm_propagator_cached(
    shape: tuple[int, int],
    dx: float,
    z: float,
    wavelength: float,
    band_limit: bool,
    device_str: str,
    dtype: torch.dtype,
) -> Tensor:
    """Cached transfer function ``H`` for a single distance.

    ``maxsize=4`` is deliberate: each entry is ``rows*cols*8`` bytes of VRAM for
    complex64, so ``maxsize=32`` at ``M = 4800`` would hold **5.9 GB** of
    transfer functions. The cache key contains the device, so a CUDA tensor can
    never be handed back for a CPU request.

    Args:
        shape: ``(rows, cols)`` of the sampling grid.
        dx: Pixel pitch in metres.
        z: Propagation distance in metres.
        wavelength: Wavelength in metres.
        band_limit: Whether to apply the band-limit cutoff.
        device_str: ``str(torch.device(...))``; part of the cache key.
        dtype: Target complex dtype (the cast happens exactly once, here).

    Returns:
        A ``(rows, cols)`` complex tensor in raw ``fft2`` order, **unshifted** so
        it lines up with ``torch.fft.fft2`` output directly.
    """
    propagator = _build_propagator(
        shape,
        dx,
        wavelength,
        device_str,
        z=z,
        band_limit=band_limit,
    )
    return propagator.to(dtype=dtype)


def _log_band_limit(shape: tuple[int, int], keep: Tensor, f_max: float) -> None:
    """Log how much of the spectrum the band-limit cutoff removed.

    Args:
        shape: ``(rows, cols)`` of the sampling grid.
        keep: Boolean keep-mask actually used (evanescent bins already excluded).
        f_max: The cutoff that was applied, in cycles/metre.
    """
    total = shape[0] * shape[1]
    kept = int(keep.sum().item())
    logger.debug(
        "ASM band limit: f_max={:.4e} 1/m keeps {}/{} bins ({:.1%})",
        f_max,
        kept,
        total,
        kept / total,
    )


def _warn_if_all_bins_dropped(keep: Tensor) -> None:
    """Warn when the transfer function is identically zero.

    Args:
        keep: The boolean keep-mask used to build the propagator.
    """
    if int(keep.sum().item()) == 0:
        logger.warning(
            "ASM band limit removed every bin (z is far outside the sampling "
            "grid's usable range); the propagated field will be identically zero. "
            "Reduce z, use band_limit=False, or zoom in the input aperture."
        )


def clear_asm_propagator_cache() -> None:
    """Drop every cached ASM propagator and frequency grid.

    Call this before freeing VRAM, after switching devices, or in tests that
    assert on cache behaviour. Both the propagator cache and the shared
    frequency-geometry cache are cleared.
    """
    _asm_propagator_cached.cache_clear()
    _asm_geometry.cache_clear()
    logger.debug("ASM propagator cache cleared")


# ---------------------------------------------------------------------------
# Public propagator
# ---------------------------------------------------------------------------


def asm_propagator(
    shape: Sequence[int] | int,
    dx: float,
    z: float,
    wavelength: float,
    *,
    band_limit: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.complex64,
) -> Tensor:
    """Cached ASM transfer function ``H``, already aligned with raw ``fft2``.

    Multiply ``torch.fft.fft2(field)`` by the result element-wise and pass it to
    ``torch.fft.ifft2`` to propagate; there is no shift on either side, because
    ``torch.fft.fftfreq`` is already in raw ``fft2`` order.

    The cached tensor is shared between calls — **do not mutate it in place**.

    Args:
        shape: ``(H, W)`` grid, or an int for a square grid.
        dx: Pixel pitch in metres.
        z: Propagation distance in metres (negative propagates backwards).
        wavelength: Wavelength in metres.
        band_limit: Apply the Voelz & Roggemann cutoff. Default ``True``.
        device: Device for the returned tensor; defaults to CPU.
        dtype: Complex dtype of the returned tensor. The geometry is always
            evaluated in float64 and cast **once** to this dtype.

    Returns:
        ``(H, W)`` complex tensor, unshifted (DC at index 0).

    Raises:
        ValueError: If ``dx``/``wavelength`` are non-positive or the shape is
            not 2-D.
        TypeError: If ``dtype`` is not a complex dtype.
    """
    dx_f, lam_f = _validate_geometry(dx, wavelength)
    shape_t = _normalise_shape(shape)
    if dtype not in (torch.complex64, torch.complex128):
        raise TypeError(
            f"dtype must be torch.complex64 or torch.complex128, got {dtype!r}"
        )
    z_f = float(z)
    device_t = torch.device(device) if device is not None else torch.device("cpu")
    return _asm_propagator_cached(
        shape_t,
        dx_f,
        z_f,
        lam_f,
        bool(band_limit),
        str(device_t),
        dtype,
    )


# ---------------------------------------------------------------------------
# Torch propagation
# ---------------------------------------------------------------------------


def _resolve_z_torch(z: float | Tensor, field: Tensor) -> tuple[Tensor | float, bool]:
    """Normalise the ``z`` argument into either a scalar float or a batch.

    Args:
        z: Scalar (``float``/``int``/0-dim tensor) or a per-sample tensor whose
            ``numel`` matches ``field.shape[0]`` for a batched field.
        field: The incoming field, used only to check batch agreement.

    Returns:
        ``(z_value, is_batched)`` where ``z_value`` is a ``float`` or a 1-D
        tensor.

    Raises:
        ValueError: If ``z`` is neither scalar nor batch-compatible.
    """
    if not isinstance(z, Tensor):
        try:
            return float(z), False
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"z must be a float or a tensor, got {type(z).__name__}"
            ) from exc

    if z.ndim == 0:
        return float(z.item()), False
    if z.numel() == 1:
        return float(z.reshape(()).item()), False
    if field.ndim < 3:
        raise ValueError(
            "batched z requires a batched field (..., H, W) with at least 3 "
            f"dimensions, got shape {tuple(field.shape)!r}"
        )
    if z.numel() != field.shape[0]:
        raise ValueError(
            f"batched z must have one entry per sample: got {z.numel()} values "
            f"for a batch of {field.shape[0]}"
        )
    return z.reshape(-1).to(field.device), True


def angular_spectrum_propagate_torch(
    field: Tensor,
    dx: float,
    z: float | Tensor,
    wavelength: float,
    *,
    band_limit: bool = True,
) -> Tensor:
    """Propagate an N-D complex field with the Angular Spectrum Method (torch).

    Args:
        field: Complex field of shape ``(..., H, W)``. ``(B, 1, g, g)`` works
            unchanged. Must be complex — torch's FFT rejects real input with a
            message that does not say which argument was wrong.
        dx: Pixel pitch in metres (square sampling in both axes).
        z: Propagation distance in metres; negative propagates backwards. May be
            a scalar, a 0-dim tensor, or a 1-D tensor with one distance per
            sample (batched, applied without a Python loop).
        wavelength: Wavelength in metres.
        band_limit: Apply the Voelz & Roggemann cutoff. Default ``True``.

    Returns:
        The propagated complex field, **unshifted** (spatial origin / DC at
        index 0) and of the same shape and dtype as ``field``. Apply a single
        ``torch.fft.fftshift`` if you want a centred picture.

    Raises:
        TypeError: If ``field`` is not a tensor or is not complex, or ``z`` is
            neither a number nor a tensor.
        ValueError: If ``dx``/``wavelength`` are non-positive, ``field`` has
            fewer than 2 dimensions, or a batched ``z`` does not match the batch.

    Example:
        >>> import torch
        >>> u = torch.ones(8, 1, 64, 64, dtype=torch.complex64)
        >>> out = angular_spectrum_propagate_torch(u, 8e-6, 0.02, 1064e-9)
        >>> out.shape
        torch.Size([8, 1, 64, 64])
    """
    if not isinstance(field, Tensor):
        raise TypeError(
            f"field must be a torch.Tensor, got {type(field).__name__}; the "
            "numpy entry point is angular_spectrum_propagate()"
        )
    if not field.is_complex():
        raise TypeError(
            f"field must be a complex tensor, got dtype {field.dtype}; ASM needs "
            "the complex envelope, so build it with e.g. "
            "torch.exp(1j * phase) rather than passing a phase map"
        )
    if field.ndim < 2:
        raise ValueError(
            f"field must have shape (..., H, W), got {tuple(field.shape)!r}"
        )
    dx_f, lam_f = _validate_geometry(dx, wavelength)
    shape = _validate_field_shape(tuple(field.shape))
    z_value, _is_batched = _resolve_z_torch(z, field)

    # `isinstance(z_value, Tensor)` is exactly `is_batched`: `_resolve_z_torch`
    # collapses a scalar / 0-dim / single-element `z` to a python `float` and only
    # returns a Tensor for the one-per-sample case. Narrowing on the value rather
    # than the flag is what lets the type checker see that `z_values` gets a Tensor.
    spectrum = torch.fft.fft2(field, norm=FFT_NORM)
    if isinstance(z_value, Tensor):
        propagator = _build_propagator_batched(
            shape,
            dx_f,
            lam_f,
            str(field.device),
            z_values=z_value,
            band_limit=bool(band_limit),
        )
    else:
        propagator = asm_propagator(
            shape,
            dx_f,
            z_value,
            lam_f,
            band_limit=bool(band_limit),
            device=field.device,
            dtype=field.dtype,
        )
    return torch.fft.ifft2(spectrum * propagator, norm=FFT_NORM)


# ---------------------------------------------------------------------------
# Numpy propagation
# ---------------------------------------------------------------------------


def _build_propagator_numpy(
    shape: tuple[int, int],
    dx: float,
    z: np.ndarray | float,
    wavelength: float,
    band_limit: bool,
) -> np.ndarray:
    """Build the complex128 transfer function(s) with numpy.

    Mirrors :func:`_build_propagator` / :func:`_build_propagator_batched`
    operation-for-operation so that the two entry points agree to float64
    round-off (verified ``max|Δ| = 8.6e-17`` on a random complex field).

    Args:
        shape: ``(rows, cols)`` of the sampling grid.
        dx: Pixel pitch in metres.
        z: Scalar distance, or a ``(B,)`` array of per-sample distances.
        wavelength: Wavelength in metres.
        band_limit: Whether to apply the band-limit cutoff.

    Returns:
        ``(rows, cols)`` or ``(B, rows, cols)`` complex128 array, unshifted.
    """
    rows, cols = shape
    k = 2.0 * math.pi / wavelength
    fx = np.fft.fftfreq(cols, d=dx)
    fy = np.fft.fftfreq(rows, d=dx)
    fy_mesh, fx_mesh = np.meshgrid(fy, fx, indexing="ij")
    kx = 2.0 * math.pi * fx_mesh
    ky = 2.0 * math.pi * fy_mesh
    kz_sq = k * k - kx * kx - ky * ky
    kz = np.sqrt(np.maximum(kz_sq, 0.0))  # clamp FIRST, never sqrt a negative

    z_arr = np.asarray(z, dtype=np.float64)
    batched = z_arr.ndim == 1 and z_arr.size > 1
    if batched:
        propagator = np.exp(1j * kz[None, :, :] * z_arr.reshape(-1, 1, 1))
        keep = (kz_sq >= 0.0)[None, :, :] & np.ones_like(propagator, dtype=bool)
    else:
        propagator = np.exp(1j * kz * float(z_arr))
        keep = kz_sq >= 0.0
    if band_limit:
        aperture = cols * dx
        f_max = 1.0 / (wavelength * np.sqrt(1.0 + (2.0 * z_arr / aperture) ** 2))
        if batched:
            keep = keep & (np.hypot(fx_mesh, fy_mesh)[None, :, :] <= f_max.reshape(-1, 1, 1))
        else:
            keep = keep & (np.hypot(fx_mesh, fy_mesh) <= float(f_max))
        logger.debug(
            "ASM band limit: f_max={:.4e} 1/m keeps {}/{} bins",
            float(np.ravel(f_max)[0]),
            int(np.count_nonzero(keep)),
            rows * cols,
        )
    propagator = np.where(keep, propagator, 0.0)
    if not np.any(keep):
        logger.warning(
            "ASM band limit removed every bin; the propagated field will be "
            "identically zero. Reduce z, use band_limit=False, or zoom in the "
            "input aperture."
        )
    return propagator


def angular_spectrum_propagate(
    field: npt.NDArray[np.floating] | npt.NDArray[np.complexfloating],
    dx: float,
    z: float | np.ndarray,
    wavelength: float,
    *,
    band_limit: bool = True,
) -> npt.NDArray[np.complexfloating]:
    """Propagate an N-D field with the Angular Spectrum Method (numpy).

    Signature-compatible with the historical implementation this replaces
    (``field, dx, z, wavelength`` positionally); ``band_limit`` is the only
    addition and defaults to the physically-correct ``True``.

    Real input is accepted — ``numpy.fft.fft2`` handles it — unlike the torch
    entry point, which requires a complex field.

    Args:
        field: Field of shape ``(..., H, W)``; real or complex.
        dx: Pixel pitch in metres.
        z: Propagation distance in metres; negative propagates backwards. May
            also be a ``(B,)`` array, one distance per leading sample.
        wavelength: Wavelength in metres.
        band_limit: Apply the Voelz & Roggemann cutoff. Default ``True``.

    Returns:
        The propagated complex field, **unshifted** (spatial origin / DC at
        index 0), same shape as ``field``.

    Raises:
        ValueError: If ``dx``/``wavelength`` are non-positive, or ``field`` has
            fewer than 2 dimensions.

    Example:
        >>> field = np.ones((64, 64), dtype=np.complex128)
        >>> out = angular_spectrum_propagate(field, 8e-6, 0.02, 1064e-9)
        >>> out.shape
        (64, 64)
    """
    arr = np.asarray(field)
    if arr.ndim < 2:
        raise ValueError(
            f"field must have shape (..., H, W), got {arr.ndim}D"
        )
    dx_f, lam_f = _validate_geometry(dx, wavelength)
    rows, cols = int(arr.shape[-2]), int(arr.shape[-1])
    propagator = _build_propagator_numpy(
        (rows, cols), dx_f, z, lam_f, bool(band_limit)
    )
    spectrum = np.fft.fft2(arr, norm="backward")
    return np.fft.ifft2(spectrum * propagator, norm="backward")


# ---------------------------------------------------------------------------
# Module class
# ---------------------------------------------------------------------------


class AngularSpectrumPropagation(torch.nn.Module):
    """Reusable ASM propagator bound to one sampling geometry.

    This is the class-based entry point recommended by
    ``algorithm/README.md``: construction validates the geometry and the
    geometry-dependent state lives on the instance, while ``forward`` is the
    single-step call. It holds no trainable parameters and no state tensors, so
    ``.to(device)``/``state_dict()`` need no special handling — the transfer
    function is looked up per call for the field's own device and dtype, which
    also means an instance can be reused across devices safely.

    Args:
        pixel_size: Pixel pitch ``dx`` in metres; must be positive.
        wavelength: Wavelength in metres; must be positive.
        shape: ``(H, W)`` grid, or an int for a square grid.
        z: Default propagation distance in metres.
        band_limit: Apply the Voelz & Roggemann cutoff. Default ``True``.

    Raises:
        ValueError: If the geometry is invalid.

    Example:
        >>> import torch
        >>> prop = AngularSpectrumPropagation(8e-6, 1064e-9, (64, 64), z=0.02)
        >>> u = torch.ones(64, 64, dtype=torch.complex64)
        >>> prop(u).shape
        torch.Size([64, 64])
        >>> prop(u, z=0.05).shape   # per-call override
        torch.Size([64, 64])
    """

    def __init__(
        self,
        pixel_size: float,
        wavelength: float,
        shape: Sequence[int] | int,
        *,
        z: float = 1.0,
        band_limit: bool = True,
    ) -> None:
        super().__init__()
        pixel, lam = _validate_geometry(pixel_size, wavelength)
        self.pixel_size: float = pixel
        self.wavelength: float = lam
        self.shape: tuple[int, int] = _normalise_shape(shape)
        self.z: float = float(z)
        self.band_limit: bool = bool(band_limit)

    def forward(self, u_in: Tensor, z: float | Tensor | None = None) -> Tensor:
        """Propagate ``u_in`` by ``z`` (or by the instance default).

        Args:
            u_in: Complex field of shape ``(..., H, W)``.
            z: Optional per-call distance in metres; ``None`` uses the
                instance default. May also be a 1-D tensor of per-sample
                distances.

        Returns:
            The propagated complex field, unshifted, same shape and dtype as
            ``u_in``.

        Raises:
            TypeError: If ``u_in`` is not a complex tensor.
            ValueError: If ``u_in`` has fewer than 2 dimensions.
        """
        return angular_spectrum_propagate_torch(
            u_in,
            self.pixel_size,
            self.z if z is None else z,
            self.wavelength,
            band_limit=self.band_limit,
        )

    def extra_repr(self) -> str:
        """Render the bound geometry for ``repr``.

        Returns:
            A short ``key=value`` summary string.
        """
        rows, cols = self.shape
        return (
            f"pixel_size={self.pixel_size!r}, wavelength={self.wavelength!r}, "
            f"shape=({rows}, {cols}), z={self.z!r}, band_limit={self.band_limit!r}"
        )