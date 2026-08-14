"""Capture-exact Zernike phase generation and DCT-based 2D phase unwrapping.

This module implements the exact Zernike phase math used by the AO-shaping SLM
capture pipeline (old code at commit 76f611c, `slm_pattern_helper.py`), verified
to reproduce stored `phase.csv` files with a 100% exact match (`np.array_equal`).

Key facts verified empirically against stored data:

- The radial polynomial uses ``factorial(n - k)`` in the numerator, NOT ``n!``::

      R_n^|m|(r) = sum_k (-1)^k (n-k)! / (k! ((n+|m|)/2 - k)! ((n-|m|)/2 - k)!) r^(n-2k)

- The grayscale quantization is 10-bit (SLM Gray_Scale_bits=10): ``max_val = 1023``.
- Coefficient order is the metadata insertion order (``(n, m)`` for ``n`` in
  ``0..n_max``, ``m`` in ``-n..n`` with ``(n-|m|) % 2 == 0``); index 0 is the
  piston ``(0, 0)`` which is always 1.0.
- Unwrapping uses a Ghiglia-Romero least-squares approach (Poisson solve via
  DCT-II, ``scipy.fftpack`` only). For masked inputs the wrapped phase is first
  extended outside the mask with nearest-neighbor values from the rim, which was
  empirically found to eliminate the rim contamination of the masked DCT solve
  (recovered coefficient error < 1e-5 waves for moderate amplitudes).

Only numpy/scipy are used here (no torch, no aotools/zernike dependency).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from scipy.fftpack import dct, idct
from scipy.ndimage import distance_transform_edt
from scipy.special import factorial

__all__ = [
    "COEFF_ORDER_NAMES",
    "build_basis_maps",
    "coefficients_to_phase_radians",
    "coefficients_to_wrapped_gray",
    "count_zernike_terms",
    "gray_to_wrapped",
    "iter_nm_terms",
    "load_stored_gray",
    "metadata_order_to_noll",
    "non_piston_indices",
    "unwrap_phase_lsq",
    "zernike_radial",
]

_MAX_GRAY_LEVEL = 1023  # SLM 10-bit grayscale quantization (capture device)
_TWO_PI = 2.0 * np.pi
_RADIAL_LIMIT = 1.0
_AMP_EPS = 1e-10
_LOG_MSG = "zernike_prediction: {}"


# ---------------------------------------------------------------------------
# Zernike math (capture-exact)
# ---------------------------------------------------------------------------


def zernike_radial(n: int, m: int, r: np.ndarray | float) -> np.ndarray | float:
    """Zernike radial polynomial ``R_n^|m|(r)`` (capture-exact numerator).

    Uses ``factorial(n - k)`` in the numerator — the form that reproduces the
    stored ``phase.csv`` files bit-for-bit (standard textbook values, e.g.
    ``R_20 = 2r^2 - 1``, ``R_31 = 3r^3 - 2r``).

    Args:
        n: Radial degree (>= 0).
        m: Azimuthal frequency; the polynomial depends only on ``|m|``.
        r: Radial coordinate(s) in [0, 1].

    Returns:
        Polynomial value(s), same shape as ``r``.

    Raises:
        ValueError: If ``|m| > n`` or ``(n - |m|)`` is odd (invalid Zernike pair).
    """
    m_abs = abs(m)
    if m_abs > n or (n - m_abs) % 2 != 0:
        raise ValueError(f"Invalid Zernike indices: n={n}, m={m} require (n-|m|) even and |m| <= n")
    r = np.asarray(r, dtype=np.float64)
    result = np.zeros_like(r)
    for k in range((n - m_abs) // 2 + 1):
        coef = _radial_coefficient(n, m_abs, k)
        result += coef * r ** (n - 2 * k)
    return result


def _radial_coefficient(n: int, m_abs: int, k: int) -> float:
    """Coefficient ``(-1)^k (n-k)! / (k! ((n+|m|)/2-k)! ((n-|m|)/2-k)!)``."""
    return ((-1.0) ** k * factorial(n - k)) / (
        factorial(k) * factorial((n + m_abs) // 2 - k) * factorial((n - m_abs) // 2 - k)
    )


def iter_nm_terms(n_max: int) -> list[tuple[int, int]]:
    """Ordered ``(n, m)`` pairs in metadata order (for ``n`` in ``0..n_max``, ``m`` ascending).

    Matches the insertion order of ``phase_params.coefficients`` in the capture
    metadata. ``n_max=10`` yields exactly 66 terms; index 0 is piston ``(0, 0)``.
    """
    if n_max < 0:
        raise ValueError(f"n_max must be >= 0, got {n_max}")
    return [(n, m) for n in range(n_max + 1) for m in range(-n, n + 1) if (n - abs(m)) % 2 == 0]


def count_zernike_terms(n_max: int) -> int:
    """Number of ``(n, m)`` pairs with ``(n-|m|) % 2 == 0`` up to ``n_max``."""
    if n_max < 0:
        raise ValueError(f"n_max must be >= 0, got {n_max}")
    return (n_max + 1) * (n_max + 2) // 2


def non_piston_indices(n_max: int) -> list[int]:
    """Indices of the non-piston coefficients (1..count-1); piston is at 0."""
    return list(range(1, count_zernike_terms(n_max)))


def COEFF_ORDER_NAMES(n_max: int = 10) -> list[str]:
    """Human-readable labels ``"n{n}m{m}"`` per term, in metadata order."""
    return [f"n{n}m{m}" for n, m in iter_nm_terms(n_max)]


@lru_cache(maxsize=4)
def _coordinate_maps(height: int, width: int, radius: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit-disk coordinate maps: ``(R, Theta, mask)``.

    ``x = (arange(W) - W/2) / radius``, ``y = (arange(H) - H/2) / radius``,
    ``R = sqrt(X^2+Y^2)``, ``Theta = arctan2(Y, X)``, ``mask = R <= 1.0``.
    Cached keyed on ``(height, width, radius)`` (R + Theta float64 ~148 MB at
    1200x1920).
    """
    x = (np.arange(width, dtype=np.float64) - width / 2.0) / radius
    y = (np.arange(height, dtype=np.float64) - height / 2.0) / radius
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)
    mask = R <= _RADIAL_LIMIT
    return R, Theta, mask


def _radial_powers(R: np.ndarray, n_max: int) -> list[np.ndarray]:
    """``R**p`` for p in ``0..n_max`` (R**0 = ones)."""
    powers = [np.ones_like(R)]
    for p in range(1, n_max + 1):
        powers.append(R**p)
    return powers


def _zernike_mode(
    n: int,
    m: int,
    R: np.ndarray,
    Theta: np.ndarray,
    mask: np.ndarray,
    powers: list[np.ndarray],
) -> np.ndarray:
    """Single aperture-limited Zernike mode ``Z_n^m`` on the unit disk."""
    m_abs = abs(m)
    rn = np.zeros_like(R)
    for k in range((n - m_abs) // 2 + 1):
        rn += _radial_coefficient(n, m_abs, k) * powers[n - 2 * k]
    if m >= 0:
        z = rn * np.cos(m * Theta)
    else:
        z = rn * np.sin(-m * Theta)
    return z * mask


def _as_coeffs(coeffs: np.ndarray, n_terms: int) -> np.ndarray:
    """Validate/normalize a coefficient vector (65 is allowed -> prepend piston 1.0)."""
    arr = np.asarray(coeffs, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"coefficients must be 1-D, got shape {arr.shape}")
    if arr.size == n_terms - 1:
        arr = np.concatenate([[1.0], arr])
    elif arr.size != n_terms:
        raise ValueError(f"expected {n_terms} or {n_terms - 1} coefficients, got {arr.size}")
    return arr


def coefficients_to_phase_radians(
    coeffs: np.ndarray,
    n_max: int = 10,
    size: tuple[int, int] = (1200, 1920),
    radius: int | None = None,
) -> np.ndarray:
    """Zernike coefficients -> wrapped phase in radians (capture-exact math).

    Args:
        coeffs: ``(N,)`` or ``(N-1,)`` coefficients in metadata order; if 65 are
            given, piston 1.0 is prepended. Terms with ``|amp| < 1e-10`` are skipped.
        n_max: Radial degree bound (default 10 -> 66 terms).
        size: ``(height, width)`` output resolution.
        radius: Unit-disk radius in pixels; defaults to ``min(size) // 2``.

    Returns:
        Wrapped phase ``mod(phase_total, 2*pi)`` as ``(H, W)`` float64.
    """
    terms = iter_nm_terms(n_max)
    coeffs = _as_coeffs(coeffs, len(terms))
    height, width = size
    radius = radius if radius is not None else min(height, width) // 2
    R, Theta, mask = _coordinate_maps(height, width, radius)
    powers = _radial_powers(R, n_max)
    phase_total = np.zeros((height, width), dtype=np.float64)
    for (n, m), amp in zip(terms, coeffs):
        if abs(amp) < _AMP_EPS:
            continue
        z = _zernike_mode(n, m, R, Theta, mask, powers)
        phase_total += z * amp * _TWO_PI
    return np.mod(phase_total, _TWO_PI)


def coefficients_to_wrapped_gray(
    coeffs: np.ndarray,
    n_max: int = 10,
    height: int = 1200,
    width: int = 1920,
    radius: int | None = None,
) -> np.ndarray:
    """Zernike coefficients -> SLM grayscale image (uint16, 10-bit, capture-exact).

    Applies ``(wrapped / (2*pi) * 1023).astype(np.uint16)``. Verified to match
    stored ``phase.csv`` files with ``np.array_equal`` (100% exact).
    """
    wrapped = coefficients_to_phase_radians(
        coeffs, n_max=n_max, size=(height, width), radius=radius
    )
    return _wrapped_to_gray(wrapped)


def _wrapped_to_gray(wrapped_rad: np.ndarray) -> np.ndarray:
    """Radians -> 10-bit grayscale (capture quantization)."""
    return (wrapped_rad / _TWO_PI * _MAX_GRAY_LEVEL).astype(np.uint16)


def gray_to_wrapped(gray: np.ndarray) -> np.ndarray:
    """Inverse capture quantization: grayscale (0..1023) -> wrapped radians."""
    return np.asarray(gray, dtype=np.float64) / _MAX_GRAY_LEVEL * _TWO_PI


@lru_cache(maxsize=4)
def build_basis_maps(
    n_max: int = 10,
    height: int = 1200,
    width: int = 1920,
    radius: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Basis of Zernike modes (metadata order incl. piston) plus the disk mask.

    Args:
        n_max: Radial degree bound.
        height: Grid height in pixels.
        width: Grid width in pixels.
        radius: Unit-disk radius in pixels; defaults to ``min(height, width) // 2``.

    Returns:
        ``(basis, mask)``: ``basis`` is ``(N, H, W)`` float32 (metadata order,
        piston first) and ``mask`` is ``(H, W)`` uint8 with 1 inside the disk.

    Memory: the full 66x1200x1920 basis is ~610 MB in float32 (1.2 GB in
    float64); it is cached via ``lru_cache`` keyed on ``(n_max, height, width,
    radius)``. Use smaller sizes where possible.
    """
    terms = iter_nm_terms(n_max)
    radius = radius if radius is not None else min(height, width) // 2
    R, Theta, mask = _coordinate_maps(height, width, radius)
    powers = _radial_powers(R, n_max)
    basis = np.empty((len(terms), height, width), dtype=np.float32)
    for i, (n, m) in enumerate(terms):
        basis[i] = _zernike_mode(n, m, R, Theta, mask, powers).astype(np.float32)
    return basis, mask.astype(np.uint8)


# ---------------------------------------------------------------------------
# 2D least-squares phase unwrapping (Ghiglia-Romero via DCT)
# ---------------------------------------------------------------------------


def _wrap_to_pi(d: np.ndarray) -> np.ndarray:
    """Wrap a difference into (-pi, pi]."""
    return (d + np.pi) % _TWO_PI - np.pi


def _extend_wrapped(wrapped: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Nearest-neighbor extension of the wrapped phase into the masked-out region.

    Assigns each outside-mask pixel the wrapped value of its nearest in-mask
    pixel (Euclidean nearest via distance transform), leaving in-mask values
    untouched. This makes the wrapped differences across the disk rim small,
    eliminating the rim contamination that a plain zero-padded masked DCT solve
    suffers from (verified: 10-100x smaller recovered-coefficient error).
    """
    idx = distance_transform_edt(~mask, return_indices=True, return_distances=False)
    ext = wrapped[idx[0], idx[1]].copy()
    ext[mask] = wrapped[mask]
    return ext


def _wrapped_laplacian(field: np.ndarray) -> np.ndarray:
    """Discrete Laplacian of the wrapped (wrapped-difference) field."""
    dy = _wrap_to_pi(field[1:, :] - field[:-1, :])
    dx = _wrap_to_pi(field[:, 1:] - field[:, :-1])
    rho = np.zeros_like(field)
    rho[:-1, :] += dy
    rho[1:, :] -= dy
    rho[:, :-1] += dx
    rho[:, 1:] -= dx
    return rho


def _solve_poisson_dct(rho: np.ndarray) -> np.ndarray:
    """Solve the Poisson equation with Neumann BCs via DCT-II (Ghiglia-Romero).

    The solution's mean is zeroed (the DCT DC component is dropped).
    """
    height, width = rho.shape
    dct_rho = dct(dct(rho, axis=0, norm="ortho"), axis=1, norm="ortho")
    dct_rho[0, 0] = 0.0
    ky = np.cos(np.pi * np.arange(height) / height)[:, None]
    kx = np.cos(np.pi * np.arange(width) / width)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        solution = dct_rho / (2.0 * (kx + ky - 2.0))
    solution[0, 0] = 0.0
    return idct(idct(solution, axis=1, norm="ortho"), axis=0, norm="ortho")


def unwrap_phase_lsq(wrapped: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """2D least-squares phase unwrapping (Ghiglia-Romero via DCT).

    Solves the wrapped-difference Poisson equation with Neumann boundary
    conditions using a DCT-II (``scipy.fftpack`` only, fully vectorized; ~0.23 s
    on 1200x1920). For a masked region the wrapped phase is extended with
    nearest-neighbor rim values before solving so the rim does not contaminate
    the interior solution.

    Args:
        wrapped: Wrapped phase in radians, shape ``(H, W)``.
        mask: Boolean ``(H, W)``; unwrap only inside (False outside is ignored).
            If None, the whole rectangle is used.

    Returns:
        Unwrapped phase (radians) as ``(H, W)`` float64, zeroed outside the mask
        and with a zero mean (unwrapping determines phase up to a constant).
    """
    wrapped = np.asarray(wrapped, dtype=np.float64)
    height, width = wrapped.shape
    if mask is None:
        mask = np.ones((height, width), dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    ext = _extend_wrapped(wrapped, mask)
    phi = _solve_poisson_dct(_wrapped_laplacian(ext))
    return np.where(mask, phi, 0.0)


# ---------------------------------------------------------------------------
# Stored-data I/O and Noll ordering
# ---------------------------------------------------------------------------


def load_stored_gray(sample_dir: str | Path) -> np.ndarray:
    """Load a stored ``phase.csv`` (1200x1920 comma-separated ints) as uint16."""
    path = Path(sample_dir) / "phase.csv"
    if not path.exists():
        raise FileNotFoundError(f"phase.csv not found in {sample_dir}")
    return np.loadtxt(path, delimiter=",", dtype=np.int64).astype(np.uint16)


def _noll_to_nm(j: int) -> tuple[int, int]:
    """Standard Noll index ``j`` (1-based) -> ``(n, m)`` pair.

    Standard table: j=1 (0,0), j=2 tip (1,-1), j=3 tilt (1,1), j=4 defocus
    (2,0), j=5 astig (2,-2), j=6 astig (2,2), ... The base formula below matches
    the standard table everywhere except j=2/3 (which it swaps), so those two
    entries are corrected explicitly.
    """
    if j < 1:
        raise ValueError(f"Noll index must be >= 1, got {j}")
    n = 0
    j1 = j - 1
    while j1 > n:
        n += 1
        j1 -= n
    m = int((-1) ** j * ((n % 2) + 2 * int((j1 + ((n + 1) % 2)) / 2.0)))
    if (n, m) == (1, 1) and j == 2:
        return (1, -1)
    if (n, m) == (1, -1) and j == 3:
        return (1, 1)
    return (n, m)


def metadata_order_to_noll(coeffs_66: np.ndarray, n_max: int = 10) -> np.ndarray:
    """Convert a metadata-ordered coefficient vector to standard Noll order.

    Args:
        coeffs_66: ``(N,)`` coefficients in metadata ``(n, m)`` insertion order.
        n_max: Radial degree bound.

    Returns:
        ``(N,)`` float64 vector indexed by the standard Noll index ``j`` (1-based
        -> array position ``j - 1``). ``N = count_zernike_terms(n_max)``.
    """
    coeffs = np.asarray(coeffs_66, dtype=np.float64)
    if coeffs.ndim != 1:
        raise ValueError(f"coeffs must be 1-D, got shape {coeffs.shape}")
    terms = iter_nm_terms(n_max)
    if coeffs.size != len(terms):
        raise ValueError(f"expected {len(terms)} coefficients for n_max={n_max}, got {coeffs.size}")
    meta_index = {term: i for i, term in enumerate(terms)}
    out = np.zeros(len(terms), dtype=np.float64)
    for j in range(1, len(terms) + 1):
        nm = _noll_to_nm(j)
        if nm in meta_index:
            out[j - 1] = coeffs[meta_index[nm]]
    return out
