"""Canonical Fraunhofer far-field propagation (centre-zero-pad + centred FFT).

This module is the **single** implementation of the ``fftshift(fft2(ifftshift(·)))``
far-field chain used for far-field synthesis. The same chain previously existed as
two copies — ``ZernikeCoefficientOptimizer._far_field_intensity``
(``ao_shaping.algorithm.signal_processing``) and ``ZernikeAmpModel._propagate``
(``ml/zernike/models.py``) — which were verified **bitwise identical** before the
call sites were switched over, so this change is a de-duplication with no
numerical effect. Both now call this module instead.

Only the *target grid size* differed between them (an absolute ``far_field_size``
versus ``n * far_field_padding``); the padding arithmetic, the shift convention,
``norm="ortho"`` and the intensity expression were already the same. What the two
surrounding models still differ in -- piston handling, pupil amplitude, aperture
masking, and whether the far field is centre-cropped back to ``grid`` -- is
deliberately **not** unified here.

This module is deliberately **not** re-exported from
``utils/wavefront/__init__.py``: that ``__init__`` eagerly imports every
submodule, and ``ao_shaping.utils`` is currently a 100% torch-free leaf package.
Re-exporting here would make torch — an optional dependency group
(``uv sync --group ml``) — a hard requirement of the whole utils package.
Callers import the submodule directly::

    from ao_shaping.utils.wavefront.fraunhofer import focal_field
"""

from __future__ import annotations

import torch

__all__ = ["focal_field", "focal_intensity"]


def focal_field(field: torch.Tensor, out_size: int) -> torch.Tensor:
    """Centre-pad the last two dims to ``out_size`` and propagate to the far field.

    Applies symmetric zero-padding (a no-op when ``out_size <= field.shape[-1]``,
    so the call never downsamples or raises), then the canonical centred FFT::

        fftshift(fft2(ifftshift(field, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))

    For an odd pad remainder the extra row/column goes to the right/bottom edge
    (``half = (out_size - n) // 2``), matching both historical copies.

    Zero-padding here is a *sampling* change only — it does not change the
    physical field of view (still ``lambda * f / d_slm``), it merely samples the
    same far field at ``out_size / n`` times the fineness (documented with
    bench measurements in ``report/slm/model_in_loop_bench_calibration.md``).

    Args:
        field: Complex or real field whose last two dims are the square pupil
            grid (2-D ``(g, g)`` or any batched shape ``(B, ..., g, g)``).
        out_size: Target side length of the far-field grid.

    Returns:
        Complex far-field spectrum on the ``out_size`` grid.
    """
    n = field.shape[-1]
    if out_size > n:
        half = (out_size - n) // 2
        field = torch.nn.functional.pad(
            field, (half, out_size - n - half, half, out_size - n - half)
        )
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(field, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )


def focal_intensity(field: torch.Tensor, out_size: int) -> torch.Tensor:
    """Far-field intensity ``|F|²`` of ``field`` on the ``out_size`` grid.

    Args:
        field: Complex or real field whose last two dims are the square pupil
            grid (2-D ``(g, g)`` or any batched shape ``(B, ..., g, g)``).
        out_size: Target side length of the far-field grid.

    Returns:
        Real, non-negative intensity map on the ``out_size`` grid.
    """
    focal = focal_field(field, out_size)
    return focal.real.pow(2) + focal.imag.pow(2)
