"""Regression metrics for Zernike-coefficient prediction (T2).

Evaluates predicted vs. true Zernike coefficient vectors (65 non-piston
coefficients in metadata ``(n, m)`` order, ``n_max=10``) with scalar regression
metrics, per-coefficient / per-radial-order breakdowns, and circular phase
metrics computed on the reconstructed wrapped phase (via
``phase_gen.coefficients_to_phase_radians``, which handles both 65- and 66-wide
input by prepending piston 1.0 internally).

Piston-alignment convention
---------------------------
The pipeline predicts the 65 non-piston coefficients; the piston (metadata
index 0) is always ``1.0`` by construction and carries no prediction error.
Every coefficient metric therefore compares **non-piston coefficients only**:

- ``(N, 65)`` vs ``(N, 65)``: used as-is.
- ``(N, 65)`` vs ``(N, 66)``: the true piston column (index 0) is skipped.
- ``(N, 66)`` vs ``(N, 66)``: the piston column is skipped on both sides.
- Any other shape mismatch raises ``ValueError``.

This single alignment helper (``_align_non_piston``) is shared by all metrics.
Phase metrics need no alignment: the phase reconstruction already treats the
piston as a fixed 1.0 term.

Only numpy/scipy are used here (no torch); matplotlib is intentionally not
imported (plotting lives in the companion ``plots`` module).
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from ml.zernike_prediction.phase_gen import (
    COEFF_ORDER_NAMES,
    build_basis_maps,
    coefficients_to_phase_radians,
    iter_nm_terms,
)

__all__ = [
    "alignment_ok",
    "coefficient_names",
    "mae",
    "metrics_summary",
    "mse",
    "per_coeff_mae",
    "per_order_mae",
    "phase_mae",
    "phase_rmse",
    "r2",
    "rmse",
]

_N_MAX = 10
_N_NON_PISTON = 65
_N_FULL = 66
_PHASE_SIZE_DEFAULT = (192, 192)


# ---------------------------------------------------------------------------
# Piston-aware alignment
# ---------------------------------------------------------------------------


def _is_width(arr: np.ndarray, n_coeffs: int) -> bool:
    """True if ``arr`` is a 1-D vector or 2-D matrix with ``n_coeffs`` columns."""
    return (arr.ndim == 1 and arr.size == n_coeffs) or (arr.ndim == 2 and arr.shape[1] == n_coeffs)


def _align_non_piston(pred: np.ndarray, true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align ``pred``/``true`` onto the 65 non-piston coefficients.

    See the module docstring for the piston-alignment convention. Raises
    ``ValueError`` with a clear message for any unsupported shape mismatch.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)

    if pred.shape == true.shape:
        if _is_width(pred, _N_FULL):
            return pred[..., 1:], true[..., 1:]
        return pred, true

    if _is_width(pred, _N_NON_PISTON) and _is_width(true, _N_FULL):
        _check_row_counts(pred, true)
        return pred, true[..., 1:]

    if _is_width(pred, _N_FULL) and _is_width(true, _N_NON_PISTON):
        _check_row_counts(pred, true)
        return pred[..., 1:], true

    raise ValueError(
        f"Cannot align pred {pred.shape} with true {true.shape}: expected matching "
        "shapes or a 65/66 non-piston alignment (piston at column 0)"
    )


def _check_row_counts(pred: np.ndarray, true: np.ndarray) -> None:
    """Raise when 2-D arrays with different row counts are combined."""
    if pred.ndim == true.ndim and pred.ndim >= 2 and pred.shape[0] != true.shape[0]:
        raise ValueError(
            f"Cannot align pred {pred.shape} with true {true.shape}: row counts differ "
            f"({pred.shape[0]} vs {true.shape[0]})"
        )


def alignment_ok(pred: np.ndarray, true: np.ndarray) -> bool:
    """True if ``pred``/``true`` are shape-compatible for metric computation.

    Compatible means either identical shapes or the documented 65/66
    piston-alignment case.
    """
    try:
        _align_non_piston(pred, true)
    except ValueError:
        return False
    return True


# ---------------------------------------------------------------------------
# Scalar regression metrics
# ---------------------------------------------------------------------------


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean absolute error over all (non-piston) elements."""
    p, t = _align_non_piston(pred, true)
    return float(np.mean(np.abs(p - t)))


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean squared error over all (non-piston) elements."""
    p, t = _align_non_piston(pred, true)
    return float(np.mean((p - t) ** 2))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Root mean squared error over all (non-piston) elements."""
    return float(np.sqrt(mse(pred, true)))


def r2(pred: np.ndarray, true: np.ndarray) -> float:
    """Coefficient of determination ``R^2 = 1 - SS_res / SS_tot``.

    Returns ``nan`` (instead of crashing) when the true values have zero
    variance (degenerate ``SS_tot``).
    """
    p, t = _align_non_piston(pred, true)
    ss_res = float(np.sum((p - t) ** 2))
    ss_tot = float(np.sum((t - np.mean(t)) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Per-coefficient / per-radial-order breakdowns
# ---------------------------------------------------------------------------


def per_coeff_mae(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """MAE per non-piston coefficient, shape ``(65,)`` in metadata (n, m) order."""
    p, t = _align_non_piston(pred, true)
    if p.ndim == 1:
        p = p.reshape(1, -1)
        t = t.reshape(1, -1)
    return np.mean(np.abs(p - t), axis=0)


def per_order_mae(pred: np.ndarray, true: np.ndarray) -> dict[int, float]:
    """MAE grouped by radial order ``n`` (1..10), as ``{n: float}``.

    Grouping follows ``phase_gen.iter_nm_terms(10)``, skipping the piston
    ``(0, 0)`` term; ``per_order_mae[n]`` is the mean of ``per_coeff_mae`` over
    the terms with radial degree ``n``.
    """
    pcm = per_coeff_mae(pred, true)
    terms = iter_nm_terms(_N_MAX)
    return {
        n: float(np.mean([pcm[i - 1] for i, (nn, _m) in enumerate(terms) if i >= 1 and nn == n]))
        for n in range(1, _N_MAX + 1)
    }


# ---------------------------------------------------------------------------
# Circular phase metrics (masked to the aperture)
# ---------------------------------------------------------------------------


def _single_phase_diff(
    pred_coeffs: np.ndarray, true_coeffs: np.ndarray, size: tuple[int, int]
) -> np.ndarray:
    """Circular wrapped-phase difference (radians) inside the aperture disk."""
    pred_phase = coefficients_to_phase_radians(pred_coeffs, n_max=_N_MAX, size=size)
    true_phase = coefficients_to_phase_radians(true_coeffs, n_max=_N_MAX, size=size)
    diff = np.angle(np.exp(1j * (pred_phase - true_phase)))  # (-pi, pi]
    height, width = size
    _basis, mask = build_basis_maps(n_max=_N_MAX, height=height, width=width)
    return diff[mask.astype(bool)]


def _phase_metric(
    pred_coeffs: np.ndarray,
    true_coeffs: np.ndarray,
    size: tuple[int, int],
    reduce: str,
) -> float:
    """Phase metric per sample (``mae``/``rmse``); 2-D input averages over samples."""
    pred = np.asarray(pred_coeffs, dtype=np.float64)
    true = np.asarray(true_coeffs, dtype=np.float64)
    if pred.ndim != true.ndim:
        raise ValueError(
            f"Cannot compare phase inputs with different ranks: pred {pred.shape} vs true {true.shape}"
        )
    if pred.ndim == 2:
        if pred.shape[0] != true.shape[0]:
            raise ValueError(
                f"Cannot compare phase inputs with different sample counts: "
                f"pred {pred.shape} vs true {true.shape}"
            )
        per_sample = [_phase_metric(p, t, size, reduce) for p, t in zip(pred, true)]
        return float(np.mean(per_sample))
    diff = _single_phase_diff(pred, true, size)
    if reduce == "mae":
        return float(np.mean(np.abs(diff)))
    return float(np.sqrt(np.mean(diff**2)))


def phase_rmse(
    pred_coeffs: np.ndarray, true_coeffs: np.ndarray, size: tuple[int, int] = _PHASE_SIZE_DEFAULT
) -> float:
    """RMSE of the circular wrapped-phase difference, masked to the aperture.

    Accepts ``(N,)`` coefficient vectors or ``(N, 65/66)`` batches (averaged
    over samples).
    """
    return _phase_metric(pred_coeffs, true_coeffs, size, "rmse")


def phase_mae(
    pred_coeffs: np.ndarray, true_coeffs: np.ndarray, size: tuple[int, int] = _PHASE_SIZE_DEFAULT
) -> float:
    """MAE of the circular wrapped-phase difference, masked to the aperture.

    Accepts ``(N,)`` coefficient vectors or ``(N, 65/66)`` batches (averaged
    over samples).
    """
    return _phase_metric(pred_coeffs, true_coeffs, size, "mae")


# ---------------------------------------------------------------------------
# Summary + names
# ---------------------------------------------------------------------------


def metrics_summary(
    pred: np.ndarray, true: np.ndarray, phase_size: tuple[int, int] = _PHASE_SIZE_DEFAULT
) -> dict:
    """One-stop metrics dict for logging/JSON serialization.

    Keys: ``mae``, ``rmse``, ``mse``, ``r2`` (floats), ``per_coeff_mae``
    (``(65,)`` array), ``per_order_mae`` (``{1..10: float}``), ``phase_mae``,
    ``phase_rmse`` (floats on the wrapped-phase difference), ``n_samples``.
    """
    p, _t = _align_non_piston(pred, true)
    n_samples = int(p.shape[0]) if p.ndim >= 2 else 1
    logger.debug("computing metrics summary over {} samples (phase grid {})", n_samples, phase_size)
    return {
        "mae": mae(pred, true),
        "rmse": rmse(pred, true),
        "mse": mse(pred, true),
        "r2": r2(pred, true),
        "per_coeff_mae": per_coeff_mae(pred, true),
        "per_order_mae": per_order_mae(pred, true),
        "phase_mae": phase_mae(pred, true, phase_size),
        "phase_rmse": phase_rmse(pred, true, phase_size),
        "n_samples": n_samples,
    }


def coefficient_names(n_max: int = 10) -> list[str]:
    """Labels ``"n{n}m{m}"`` for the non-piston coefficients (``COEFF_ORDER_NAMES`` minus piston)."""
    return list(COEFF_ORDER_NAMES(n_max)[1:])
