"""Fit a GS-refine SLM phase into Noll-order Zernike coefficients.

The GS-refine experiment (1) runs a GS + free-phase SPGD optimisation on the
SLM, (2) fits the resulting best phase into a Zernike coefficient vector at
several truncation orders, and (3) re-optimises those coefficients with SPGD.
This module is step (2): a pure-math, hardware-free bridge between a full SLM
panel and a Noll-order coefficient vector.

Design notes
------------
* **No Zernike math is implemented here.** Every polynomial evaluation, index
  conversion and least-squares fit is delegated to the canonical engine
  :class:`~ao_shaping.utils.wavefront.zernike_calc.ZernikeGenerator`; the
  panel-domain reconstruction goes through
  :class:`~ao_shaping.utils.wavefront.pattern_helper.PatternHelper`. The repo
  has exactly two canonical Zernike layers and this module adds none.
* **Raw unwrapped radians only.** Phases are never ``mod 2*pi`` wrapped here;
  the single wrap point is the SLM driver's ``create_phase_from_array``.
* **One grid convention.** ``fit`` and ``generate_noll`` share the generator's
  cached grid, so a fit of a generated phase round-trips to ~1e-15. The
  module-level ``fit_zernike`` helper is deliberately *not* used: it builds a
  ``linspace(-1, 1)`` grid (radius ``(N-1)/2``) that disagrees with the
  generator's default ``N/2`` and would introduce a ~0.04 rad mismatch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ao_shaping.utils.slm.phase_embed import pupil_side
from ao_shaping.utils.wavefront.pattern_helper import PatternHelper
from ao_shaping.utils.wavefront.zernike_calc import (
    ZernikeGenerator,
    calc_n_zernike_terms,
    noll_to_nm,
)

__all__ = [
    "ZernikeFitResult",
    "pupil_square_from_panel",
    "fit_phase_to_zernike",
    "fit_best_phase",
    "panel_phase_from_coefficients",
]


@dataclass(frozen=True)
class ZernikeFitResult:
    """Outcome of fitting a pupil-square phase into Noll-order coefficients.

    Attributes:
        coefficients: Noll-order coefficient vector, ``float64``, length
            ``n_modes`` (index 0 = Noll 1 = piston, index 3 = Noll 4 = defocus,
            index 10 = Noll 11 = spherical).
        n_max: Maximum radial order the fit was truncated to.
        radius_px: Aperture radius in pixels used for the fit.
        side: Edge length of the pupil square, ``pupil_side(radius_px)``.
        n_modes: Number of coefficients, ``calc_n_zernike_terms(n_max)``.
        rms_residual_rad: RMS of ``reconstructed - input`` over the finite
            (in-aperture) pixels, in radians.
        max_abs_coefficient: Largest absolute coefficient magnitude.
        reconstructed: ``(side, side)`` ``float64`` raw-radian phase rebuilt
            from ``coefficients``; ``NaN`` outside the circular aperture.
    """

    coefficients: np.ndarray
    n_max: int
    radius_px: float
    side: int
    n_modes: int
    rms_residual_rad: float
    max_abs_coefficient: float
    reconstructed: np.ndarray


def _validate_radius(radius_px: float) -> float:
    """Return ``radius_px`` as a float, raising if it is not positive and finite."""
    radius = float(radius_px)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(
            f"radius_px must be a positive finite number, got {radius_px!r}"
        )
    return radius


def _validate_panel_shape(panel_shape: tuple[int, int]) -> tuple[int, int]:
    """Return ``(height, width)`` as ints, raising if either is not finite/positive."""
    if len(panel_shape) != 2:
        raise ValueError(f"panel_shape must be (height, width), got {panel_shape!r}")
    height, width = panel_shape
    if not (np.isfinite(height) and np.isfinite(width)):
        raise ValueError(f"panel_shape must be finite, got {panel_shape!r}")
    if height <= 0 or width <= 0:
        raise ValueError(f"panel_shape must be positive, got {panel_shape!r}")
    return int(height), int(width)


def _n_max_for_length(length: int) -> int:
    """Largest ``n`` with ``calc_n_zernike_terms(n) <= length``.

    This is the inverse of the triangular coefficient count: a vector of
    ``length`` coefficients is interpreted as the highest complete Zernike
    order it can hold, and any trailing partial-order entries are dropped.
    """
    n_max = 0
    while calc_n_zernike_terms(n_max + 1) <= length:
        n_max += 1
    return n_max


def pupil_square_from_panel(
    panel: np.ndarray,
    radius_px: float,
    center_xy: tuple[int, int] | None = None,
) -> np.ndarray:
    """Crop the pupil square out of a full-panel phase.

    The square is the exact inverse of
    :func:`ao_shaping.utils.slm.phase_embed.embed_pupil_square`: it is
    ``pupil_side(radius_px)`` pixels on a side, centred on ``center_xy``.

    Args:
        panel: Full-panel phase, shape ``(height, width)``.
        radius_px: Aperture radius in pixels.
        center_xy: ``(x, y)`` centre of the pupil square, in the same
            convention as ``embed_pupil_square``. Defaults to the panel centre.

    Returns:
        ``(side, side)`` ``float64`` copy of the pupil square.

    Raises:
        ValueError: If ``radius_px`` is not positive/finite, ``panel`` is not
            2-D, or the square would extend beyond the panel.
    """
    radius = _validate_radius(radius_px)
    arr = np.asarray(panel, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"panel must be 2-D, got shape {arr.shape}")

    panel_h, panel_w = arr.shape
    side = pupil_side(radius)
    cx, cy = center_xy if center_xy is not None else (panel_w // 2, panel_h // 2)
    x0, y0 = int(cx) - side // 2, int(cy) - side // 2
    if x0 < 0 or y0 < 0 or x0 + side > panel_w or y0 + side > panel_h:
        raise ValueError("pupil square extends beyond the SLM panel")
    return np.array(arr[y0 : y0 + side, x0 : x0 + side], dtype=np.float64)


def fit_phase_to_zernike(
    pupil_phase: np.ndarray,
    radius_px: float,
    n_max: int,
) -> ZernikeFitResult:
    """Least-squares fit a pupil-square phase into Noll-order coefficients.

    Args:
        pupil_phase: ``(side, side)`` phase in raw radians, where
            ``side = pupil_side(radius_px)``. Pixels outside the circular
            aperture may be ``NaN`` (as produced by ``generate_noll``) or any
            finite value; they are excluded from the fit either way.
        radius_px: Aperture radius in pixels.
        n_max: Maximum Zernike radial order (``>= 1``).

    Returns:
        A :class:`ZernikeFitResult` with the fitted coefficients and the
        reconstructed pupil-square phase.

    Raises:
        ValueError: If ``radius_px`` is not positive/finite, ``n_max < 1``,
            ``pupil_phase`` is not a ``(side, side)`` square, or it contains no
            finite pixel at all.
    """
    radius = _validate_radius(radius_px)
    if n_max < 1:
        raise ValueError(f"n_max must be >= 1, got {n_max}")

    side = pupil_side(radius)
    arr = np.asarray(pupil_phase, dtype=np.float64)
    if arr.ndim != 2 or arr.shape != (side, side):
        raise ValueError(
            f"pupil_phase must be a square ({side}, {side}) array, got shape "
            f"{arr.shape}"
        )

    generator = ZernikeGenerator(
        resolution=(side, side), radius=radius, n_orders=n_max
    )
    aperture = generator.mask.astype(bool)
    if not (np.isfinite(arr) & aperture).any():
        raise ValueError("pupil_phase has no finite in-aperture pixels to fit")

    coefficients = np.asarray(generator.fit(arr), dtype=np.float64)
    reconstructed = generator.generate_noll(coefficients)

    residual = reconstructed - arr
    finite = np.isfinite(residual)
    rms_residual = float(np.sqrt(np.mean(residual[finite] ** 2)))

    return ZernikeFitResult(
        coefficients=coefficients,
        n_max=int(n_max),
        radius_px=radius,
        side=side,
        n_modes=generator.n_modes,
        rms_residual_rad=rms_residual,
        max_abs_coefficient=float(np.max(np.abs(coefficients))),
        reconstructed=reconstructed,
    )


def fit_best_phase(
    panel: np.ndarray,
    n_max: int,
    radius_px: float,
    center_xy: tuple[int, int] | None = None,
) -> ZernikeFitResult:
    """Crop the pupil square from a full panel and fit it into Zernike modes.

    Convenience wrapper around :func:`pupil_square_from_panel` followed by
    :func:`fit_phase_to_zernike`.

    Args:
        panel: Full-panel phase, shape ``(height, width)``.
        n_max: Maximum Zernike radial order (``>= 1``).
        radius_px: Aperture radius in pixels.
        center_xy: ``(x, y)`` pupil-square centre; defaults to the panel centre.

    Returns:
        A :class:`ZernikeFitResult` for the cropped pupil square.
    """
    box = pupil_square_from_panel(panel, radius_px, center_xy)
    return fit_phase_to_zernike(box, radius_px, n_max)


def panel_phase_from_coefficients(
    coefficients: np.ndarray,
    radius_px: float,
    panel_shape: tuple[int, int],
) -> np.ndarray:
    """Rebuild a full-panel raw-radian phase from Noll-order coefficients.

    The coefficient vector may be of **any** length. Because ``n_max`` is not
    passed, it is derived from the length as the largest ``n`` with
    ``calc_n_zernike_terms(n) <= len(coefficients)``; the vector is then
    zero-padded or truncated to exactly ``calc_n_zernike_terms(n_max)`` entries.
    A length that is not a triangular number therefore drops its trailing
    partial-order coefficients.

    Args:
        coefficients: Noll-order coefficient vector (any length).
        radius_px: Aperture radius in pixels.
        panel_shape: Output panel shape ``(height, width)``.

    Returns:
        ``(height, width)`` ``float64`` raw-radian phase, zero outside the
        circular aperture.

    Raises:
        ValueError: If ``radius_px`` is not positive/finite or ``panel_shape``
            is not a finite positive ``(height, width)`` pair.
    """
    radius = _validate_radius(radius_px)
    height, width = _validate_panel_shape(panel_shape)

    coeffs = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    n_max = _n_max_for_length(coeffs.size)
    n_terms = calc_n_zernike_terms(n_max)

    truncated = np.zeros(n_terms, dtype=np.float64)
    n_copy = min(coeffs.size, n_terms)
    truncated[:n_copy] = coeffs[:n_copy]

    mode_map: dict[tuple[int, int], float] = {}
    for j in range(1, n_terms + 1):
        n, m = noll_to_nm(j)
        mode_map[(n, m)] = float(truncated[j - 1])

    helper = PatternHelper(resolution=(width, height))
    return helper.generate_zernike_polynomial(mode_map, radius, n_max)
