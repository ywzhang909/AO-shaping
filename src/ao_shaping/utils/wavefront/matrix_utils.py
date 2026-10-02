"""Matrix utility functions for Zernike response matrix operations.

This module contains pure mathematical operations that don't depend on hardware.
Use this for testing without triggering hardware import chains.
"""

from __future__ import annotations

import numpy as np


def compute_pinv(matrix: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """Compute SVD pseudoinverse.

    Args:
        matrix: Input matrix of shape (m, n).
        rcond: Singular value truncation threshold.

    Returns:
        Pseudoinverse matrix of shape (n, m).
    """
    return np.linalg.pinv(matrix, rcond=rcond)


def compute_lstsq(matrix: np.ndarray) -> np.ndarray:
    """Compute least-squares inverse.

    For square matrices, directly computes the inverse.
    For rectangular matrices, computes the minimum-norm solution.

    Args:
        matrix: Input matrix of shape (m, n).

    Returns:
        Least-squares inverse matrix of shape (n, m).
    """
    m, n = matrix.shape

    if m == n:
        return np.linalg.inv(matrix)
    else:
        identity = np.eye(m)
        result = np.zeros((n, m))
        for i in range(m):
            # np.linalg.lstsq returns (solution, residuals, rank, singular_values)
            solution, _, _, _ = np.linalg.lstsq(matrix, identity[:, i], rcond=None)
            result[:, i] = solution
        return result


from ao_shaping.utils.wavefront.zernike_calc import (  # noqa: F401
    calc_n_zernike_terms as _calc_n_zernike_terms,
)


def calc_n_zernike_terms(n_max: int) -> int:
    """Calculate number of Zernike terms up to order n_max (re-export from zernike_calc)."""
    return _calc_n_zernike_terms(n_max)


def noll_to_index(j: int) -> int:
    """Convert Noll index to array index (0-based)."""
    return j - 1


def index_to_noll(i: int) -> int:
    """Convert array index to Noll index (1-based)."""
    return i + 1


def camera_pixel_um_from_focal_scale(
    *,
    wavelength_nm: float,
    focal_length_m: float,
    slm_pixel_um: float,
    focal_scale_px: float,
) -> float:
    """Recover the CCD pixel pitch from a *measured* 2f focal scale.

    Lives in ``utils/`` (not ``tools/``) because both the optimizer layer and the
    tools layer need it, and ``utils`` is the leaf.

    The tilt fitters use ``shift_px = focal_scale / period``, where a 2*pi phase
    ramp over ``period`` SLM pixels displaces the spot by that many camera
    pixels. That focal scale is measured on the bench, which makes it far more
    trustworthy than a pixel pitch copied from a different camera: it already
    folds in the obliquity factor that the naive ``f*lambda/d_slm`` estimate
    misses.

    Inverting the relation:

    .. code-block:: text

        focal_scale = wavelength * f / (d_slm * p_cam)
        => p_cam     = wavelength * f / (d_slm * focal_scale)

    On this bench (1064 nm, f = 125 mm, d_slm = 8 um, measured scale 7400) this
    returns 2.247 um, consistent with the 2.2 um measured independently.

    Args:
        wavelength_nm: Laser wavelength in nanometres.
        focal_length_m: 2f lens focal length in metres.
        slm_pixel_um: SLM pixel pitch in micrometres.
        focal_scale_px: Measured focal scale in camera px per 2*pi ramp per
            panel pixel (e.g. ``slm_bench_probe.TILT_SHIFT_SCALE``).

    Returns:
        Camera pixel pitch in micrometres.

    Raises:
        ValueError: If any argument is not strictly positive.
    """
    lam = float(wavelength_nm) * 1e-9
    f = float(focal_length_m)
    d = float(slm_pixel_um) * 1e-6
    k = float(focal_scale_px)
    if not all(v > 0.0 for v in (lam, f, d, k)):
        raise ValueError(
            "wavelength_nm, focal_length_m, slm_pixel_um and focal_scale_px "
            "must all be positive, got "
            f"({wavelength_nm!r}, {focal_length_m!r}, {slm_pixel_um!r}, {focal_scale_px!r})"
        )
    return float(lam * f / (d * k) * 1e6)


def focal_length_from_camera_pixel(
    *,
    wavelength_nm: float,
    camera_pixel_um: float,
    slm_pixel_um: float,
    focal_scale_px: float,
    scale_measured_at_nm: float | None = None,
) -> float:
    """Recover the 2f lens focal length from a *measured* focal scale.

    Exact inverse of :func:`camera_pixel_um_from_focal_scale`. Exists so the lens
    focal length — a **bench assembly choice** that changes whenever somebody
    swaps optics — can be *derived* instead of baked in as a default, anchoring
    instead on the CCD pixel pitch, which is a per-camera datasheet constant.

    Why one anchor is unavoidable: the measured focal scale constrains only the
    ratio ``f / p_cam``. Recovering either one therefore needs one quantity from
    outside the alignment measurement, and this function takes it to be the
    camera pitch.

    Wavelength handling matters here. The scale is ``K = lambda*f/(d*p_cam)``, so
    **K is proportional to wavelength**: a scale measured at one wavelength does
    not transfer unchanged to another. Pass ``scale_measured_at_nm`` (the
    wavelength at which ``focal_scale_px`` was actually measured) and the scale is
    referred to ``wavelength_nm`` first, which makes the returned focal length
    wavelength-INVARIANT — as a lens focal length must be, since ``f``, ``d`` and
    ``p_cam`` do not depend on the laser. Omitting it keeps ``focal_scale_px``
    pinned to ``wavelength_nm``, which silently scales ``f`` with wavelength.

    On this bench (p_cam = 2.2 um, d_slm = 8 um, scale 7400 measured at 1064 nm)
    this returns 0.1224 m at 1064 nm *and* at 532 nm, i.e. the 125 mm nominal
    lens to within 2% — the same 2% by which the three datasheet values
    (125 mm, 2.2 um, 8 um) disagree with the measured scale.

    Note the GS *target side* is immune to the anchor choice either way: it
    consumes ``f / p_cam`` only, which the scale pins by itself.

    Args:
        wavelength_nm: Laser wavelength in nanometres.
        camera_pixel_um: CCD pixel pitch in micrometres (datasheet/measured).
        slm_pixel_um: SLM pixel pitch in micrometres.
        focal_scale_px: Measured focal scale in camera px per 2*pi ramp per
            panel pixel (e.g. ``slm_bench_probe.TILT_SHIFT_SCALE``).
        scale_measured_at_nm: Wavelength at which ``focal_scale_px`` was measured.
            ``None`` (the default) means "already referred to ``wavelength_nm``".

    Returns:
        2f lens focal length in metres.

    Raises:
        ValueError: If any argument is not strictly positive.
    """
    lam = float(wavelength_nm) * 1e-9
    p = float(camera_pixel_um) * 1e-6
    d = float(slm_pixel_um) * 1e-6
    k = float(focal_scale_px)
    if not all(v > 0.0 for v in (lam, p, d, k)):
        raise ValueError(
            "wavelength_nm, camera_pixel_um, slm_pixel_um and focal_scale_px "
            "must all be positive, got "
            f"({wavelength_nm!r}, {camera_pixel_um!r}, {slm_pixel_um!r}, {focal_scale_px!r})"
        )
    if scale_measured_at_nm is not None:
        lam_ref = float(scale_measured_at_nm) * 1e-9
        if lam_ref <= 0.0:
            raise ValueError(
                f"scale_measured_at_nm must be positive, got {scale_measured_at_nm!r}"
            )
        # K(lambda) = K(lambda_ref) * lambda / lambda_ref
        k = k * (lam / lam_ref)
    return float(p * d * k / lam)
