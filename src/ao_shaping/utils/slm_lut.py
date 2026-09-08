"""Pure-math SLM gray-to-phase LUT calibration utilities.

This module contains only pure NumPy math for building and inverting
gray-to-phase lookup tables for a spatial light modulator (SLM). It is a
leaf module in ``utils/`` and must not import any hardware drivers or
higher-level packages.

The physical model used for the depth-scan inversion is the blazed-grating
efficiency::

    eta(D) = sinc^2((D - 2*pi) / (2*pi))

with ``D = phi(g)`` the phase depth in gray, peaking at ``D = 2*pi``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def blaze_ramp_gray(period: int, peak_gray: int, width: int) -> np.ndarray:
    """1D blaze gray ramp, shape (1, width), phase varies along x (vertical stripes).

    ``gray(x) = round(frac * peak_gray)`` with ``frac = (x mod period)/period``
    for ``x in [0, width)``.

    Args:
        period: Number of pixels per blaze period.
        peak_gray: Peak grayscale value of the ramp.
        width: Number of pixels along x.

    Returns:
        Float array of shape (1, width) with the blaze gray ramp.
    """
    x = np.arange(width)
    frac = (x % period) / period
    return np.round(frac * peak_gray).reshape(1, width)


def depth_pattern(period: int, peak_gray: int, height: int, width: int) -> np.ndarray:
    """Depth-scan TEST half-pattern (full screen).

    Tiles ``blaze_ramp_gray(period, peak_gray, width)`` to ``(height, width)``,
    dtype uint16. ``peak_gray`` is in ``[0, grayscale_max]``.

    Args:
        period: Number of pixels per blaze period.
        peak_gray: Peak grayscale value of the ramp.
        height: Number of rows.
        width: Number of columns.

    Returns:
        uint16 array of shape (height, width).
    """
    ramp = blaze_ramp_gray(period, peak_gray, width)
    return np.tile(ramp, (height, 1)).astype(np.uint16)


def offset_pattern(
    period: int,
    peak_gray: int,
    gray_offset: int,
    bits: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Offset-scan TEST half-pattern.

    ``mod(blaze_ramp_gray(period, peak_gray, width) + gray_offset, 2**bits)``,
    tiled to ``(height, width)``, dtype uint16.

    Args:
        period: Number of pixels per blaze period.
        peak_gray: Peak grayscale value of the ramp.
        gray_offset: Grayscale offset added before the modulo wrap.
        bits: Bit depth of the grayscale range (modulus is ``2**bits``).
        height: Number of rows.
        width: Number of columns.

    Returns:
        uint16 array of shape (height, width).
    """
    ramp = blaze_ramp_gray(period, peak_gray, width)
    shifted = np.mod(ramp + gray_offset, 2**bits)
    return np.tile(shifted, (height, 1)).astype(np.uint16)


def stack_halves(ref: np.ndarray, test: np.ndarray, axis: int = 0) -> np.ndarray:
    """Vertical stack (axis=0, ref on top, test below).

    Each half is ``(H/2, W)``; output is ``(H, W)``. Validates that all
    dimensions except the stacking axis match.

    Args:
        ref: Top half array.
        test: Bottom half array.
        axis: Axis along which to stack (default 0).

    Returns:
        Stacked array.

    Raises:
        ValueError: If shapes are incompatible for stacking along ``axis``.
    """
    ref = np.asarray(ref)
    test = np.asarray(test)
    if ref.ndim != test.ndim:
        raise ValueError(
            f"ref and test must have the same number of dims, "
            f"got {ref.ndim} and {test.ndim}"
        )
    for i in range(ref.ndim):
        if i != axis and ref.shape[i] != test.shape[i]:
            raise ValueError(
                f"shapes {ref.shape} and {test.shape} incompatible for "
                f"stacking along axis {axis}"
            )
    return np.concatenate([ref, test], axis=axis)


def normalize_efficiency(eta: np.ndarray) -> np.ndarray:
    """Normalize efficiency by its maximum.

    Returns ``eta / max(eta)``; if ``max(eta) <= 0`` returns zeros. Guards
    against division by zero.

    Args:
        eta: Efficiency array.

    Returns:
        Normalized efficiency array (same shape).
    """
    eta = np.asarray(eta, dtype=float)
    m = np.max(eta)
    if m <= 0:
        return np.zeros_like(eta)
    return eta / m


def sinc_inv(y: np.ndarray) -> np.ndarray:
    """Inverse of ``sinc(x) = sin(pi*x)/(pi*x)`` restricted to branch ``x in [-1, 0]``.

    Returns non-positive values. Vectorized numerical root-finding via
    bisection on the fixed interval ``[-1, 0]`` (pure NumPy, no scipy).

    Edge cases: ``y=1 -> 0``; ``y<=0 -> -1``; ``y>1 -> ValueError``;
    ``NaN -> NaN``.

    Args:
        y: Array of values in ``(0, 1]``.

    Returns:
        Array of ``x in [-1, 0]`` with ``sinc(x) = y``.

    Raises:
        ValueError: If any ``y > 1``.
    """
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    nan_mask = np.isnan(y)
    out[nan_mask] = np.nan
    finite = ~nan_mask
    if np.any(y[finite] > 1.0):
        raise ValueError("sinc_inv: y must be in (0, 1]")
    le0 = finite & (y <= 0.0)
    out[le0] = -1.0
    eq1 = finite & (y == 1.0)
    out[eq1] = 0.0
    valid = finite & (y > 0.0) & (y < 1.0)
    if np.any(valid):
        ys = y[valid]
        lo = np.full_like(ys, -1.0)
        hi = np.zeros_like(ys)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            smid = np.sinc(mid)
            lo = np.where(smid < ys, mid, lo)
            hi = np.where(smid < ys, hi, mid)
        out[valid] = 0.5 * (lo + hi)
    return out


def invert_depth_scan(eta: np.ndarray) -> np.ndarray:
    """Oracle-verified depth-scan inversion.

    ``eta`` is the measured efficiency array (parallel to the gray scan
    ``g_values``). Let ``p = argmax(eta)``, ``etap = normalize_efficiency(eta)``,
    ``u = sinc_inv(sqrt(etap))`` (in ``[-1, 0]``).

    - ``i < p`` (rising flank, phi < 2pi): ``phi[i] = 2*pi*(1 + u[i])``
    - ``i == p`` (peak, phi == 2pi): ``phi[p] = 2*pi``
    - ``i > p`` (falling flank, phi > 2pi): ``phi[i] = 2*pi*(1 - u[i])``

    The final result is clipped to ``[0, 2*pi*1.04]``.

    Args:
        eta: Measured efficiency array.

    Returns:
        float64 array of recovered phase in radians (same shape as ``eta``).
    """
    eta = np.asarray(eta, dtype=float)
    etap = normalize_efficiency(eta)
    p = int(np.argmax(eta))
    u = sinc_inv(np.sqrt(etap))
    phi = np.empty_like(eta, dtype=float)
    phi[:p] = 2 * np.pi * (1.0 + u[:p])
    phi[p] = 2 * np.pi
    phi[p + 1 :] = 2 * np.pi * (1.0 - u[p + 1 :])
    phi = np.clip(phi, 0.0, 2 * np.pi * 1.04)
    return phi


def invert_offset_scan(
    eta: np.ndarray, g_values: np.ndarray, gray_for_2pi: int
) -> np.ndarray:
    """User-spec offset-scan inversion (kept for method parity).

    ``etap = normalize_efficiency(eta)``; ``dphi = 2*arcsin(sqrt(etap))``;
    ``phi_raw = 2*pi*(g/gray_for_2pi) + dphi`` where ``g = g_values``.
    Then monotonicity is enforced: ``phi = np.maximum.accumulate(phi_raw)``.

    For ``g_values > gray_for_2pi`` the gray is duplicated to ``gray_for_2pi``
    and the phase is clipped at ``2*pi`` there (phase saturates).

    Args:
        eta: Measured efficiency array.
        g_values: Gray values parallel to ``eta``.
        gray_for_2pi: Gray value corresponding to a ``2*pi`` phase depth.

    Returns:
        float64 array of recovered phase in radians (same shape as ``eta``).
    """
    etap = normalize_efficiency(eta)
    dphi = 2 * np.arcsin(np.sqrt(etap))
    g = np.asarray(g_values, dtype=float)
    g_eff = np.minimum(g, gray_for_2pi)
    phi_raw = 2 * np.pi * (g_eff / gray_for_2pi) + dphi
    phi = np.maximum.accumulate(phi_raw)
    mask = g > gray_for_2pi
    phi = np.where(mask, np.minimum(phi, 2 * np.pi), phi)
    return phi


def build_inverse_lut(
    phi: np.ndarray, g_values: np.ndarray, n: int = 1024
) -> tuple[np.ndarray, np.ndarray]:
    """Build an inverse LUT (phase -> gray).

    Uses a uniform target grid ``t`` in ``[0, 2*pi]`` (``n`` points) and sets
    ``gray_lut[i] = g_values[argmin(|phi - t[i]|)]`` (first occurrence on ties).

    Args:
        phi: Measured phase array (must be sorted ascending).
        g_values: Gray values parallel to ``phi``.
        n: Number of points in the target grid.

    Returns:
        Tuple ``(t_grid, gray_lut)`` where ``t_grid`` is the float target grid
        of size ``n`` and ``gray_lut`` is a uint16 index array of size ``n``.

    Raises:
        ValueError: If ``phi`` is not sorted ascending.
    """
    phi = np.asarray(phi, dtype=float)
    g_values = np.asarray(g_values)
    if np.any(np.diff(phi) < 0):
        raise ValueError("phi must be sorted ascending")
    t = np.linspace(0.0, 2 * np.pi, n)
    idx = np.argmin(np.abs(phi[:, None] - t[None, :]), axis=0)
    gray_lut = g_values[idx].astype(np.uint16)
    return t, gray_lut


def save_lut(
    dir_path: Path,
    g_values: np.ndarray,
    phi: np.ndarray,
    inverse_gray: np.ndarray,
    meta: dict | None = None,
) -> Path:
    """Write the LUT files into ``dir_path``.

    Writes three files:
    - ``lut_forward.csv``: header ``g,phi``, g/phi float columns.
    - ``lut_inverse.csv``: header ``target_phi,gray``, inverse table.
    - ``lut.npz``: arrays ``g_values``, ``phi``, ``inverse_gray`` plus ``meta``.

    Args:
        dir_path: Directory to write into (created if missing).
        g_values: Gray values (measured gray -> phase response).
        phi: Measured phase values (radians).
        inverse_gray: Inverse table gray values.
        meta: Optional metadata dict.

    Returns:
        The ``dir_path``.
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    g_values = np.asarray(g_values, dtype=float)
    phi = np.asarray(phi, dtype=float)
    inverse_gray = np.asarray(inverse_gray)

    np.savetxt(
        dir_path / "lut_forward.csv",
        np.column_stack([g_values, phi]),
        delimiter=",",
        header="g,phi",
        comments="",
    )
    target_phi = np.linspace(0.0, 2 * np.pi, inverse_gray.size)
    np.savetxt(
        dir_path / "lut_inverse.csv",
        np.column_stack([target_phi, inverse_gray]),
        delimiter=",",
        header="target_phi,gray",
        comments="",
    )
    np.savez_compressed(
        dir_path / "lut.npz",
        g_values=g_values,
        phi=phi,
        inverse_gray=inverse_gray,
        meta=np.asarray(meta if meta is not None else {}, dtype=object),
        allow_pickle=True,
    )
    return dir_path


@dataclass
class LUTData:
    """Container for a loaded LUT.

    Attributes:
        g_values: Gray values (measured gray -> phase response).
        phi: Measured phase values (radians).
        inverse_gray: Inverse table gray values.
        meta: Metadata dict.
    """

    g_values: np.ndarray
    phi: np.ndarray
    inverse_gray: np.ndarray
    meta: dict = field(default_factory=dict)


def load_lut(dir_path: Path) -> LUTData:
    """Load a LUT from ``dir_path``.

    Loads ``lut.npz`` if present, otherwise falls back to the CSVs.

    Args:
        dir_path: Directory containing the LUT files.

    Returns:
        A :class:`LUTData` with the loaded arrays and metadata.

    Raises:
        FileNotFoundError: If neither the npz nor the CSVs are present.
    """
    dir_path = Path(dir_path)
    npz = dir_path / "lut.npz"
    if npz.exists():
        data = np.load(npz, allow_pickle=True)
        meta = data["meta"].item() if "meta" in data else {}
        return LUTData(
            g_values=data["g_values"],
            phi=data["phi"],
            inverse_gray=data["inverse_gray"],
            meta=meta,
        )
    fwd = dir_path / "lut_forward.csv"
    inv = dir_path / "lut_inverse.csv"
    if not fwd.exists() or not inv.exists():
        raise FileNotFoundError(f"LUT files not found in {dir_path}")
    gphi = np.loadtxt(fwd, delimiter=",", skiprows=1)
    tphi = np.loadtxt(inv, delimiter=",", skiprows=1)
    return LUTData(
        g_values=gphi[:, 0],
        phi=gphi[:, 1],
        inverse_gray=tphi[:, 1].astype(np.uint16),
        meta={},
    )
