"""Zernike label recovery from wrapped SLM phase images (T4).

Recovers the 65 non-piston Zernike coefficients (metadata (n,m) order,
n_max=10, piston index 0 always 1.0) from stored ``phase.csv`` wrapped-phase
grayscale images that were generated without recorded coefficients (the
20260402 runs).

Why not unwrap-then-fit?
    The superposition of many Zernike terms produces phase gradients up to
    ~100 rad/px — far beyond the π/px Nyquist limit of the wrapped
    representation. A global least-squares unwrap (Ghiglia-Romero DCT)
    spreads the resulting errors over the whole aperture (residual std ~21
    rad). Instead we fit DIRECTLY in the gradient domain, where the circular
    difference of the wrapped phase equals the true phase gradient on
    >98.5% of pixel edges (verified empirically: mean error 0.002 rad).

Algorithm (two stages):
    Stage 1 — gradient-domain IRLS fit (robust initialization):
        Solve ``min_a sum_x w(x) |wrap(diff(phi_obs)) - (dZ/dx, dZ/dy) a|^2``
        with Huber weights. The Zernike gradient columns carry the same 2*pi
        factor as the generator (phase_total = sum Z_k * amp_k * 2*pi).
        Piston (index 0) is EXCLUDED from the fit (its gradient is identically
        zero; it is set back to 1.0 by metadata).
    Stage 2 — wrapped-domain Gauss-Newton refinement (optional):
        Iterate ``r = wrap(phi_obs - Za)``, robust weights on |r|, fit the
        residual onto the basis, ``a += delta``. Fixes the residual bias of
        Stage 1 caused by quantization and the 1.5% gradient-ambiguous pixels.

Reference: the companion ``phase_gen.py`` reproduces stored ``phase.csv``
files bit-for-bit (golden test), so the basis convention here is exact.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger

from ml.zernike_prediction.phase_gen import (
    _coordinate_maps,
    _radial_powers,
    _zernike_mode,
    coefficients_to_wrapped_gray,
    gray_to_wrapped,
    iter_nm_terms,
    load_stored_gray,
)

__all__ = [
    "recover_coefficients",
    "recover_run",
    "fit_error_metrics",
    "acceptance_check",
]

_N_MAX = 10
_N_TERMS = 66
_TWO_PI = 2.0 * np.pi
_STRIDE = 3  # pixel subsampling stride for the gradient fit
_AMP_EPS = 1e-10


# ---------------------------------------------------------------------------
# Gradient-domain helpers
# ---------------------------------------------------------------------------


def _circ_diff(x: np.ndarray, axis: int) -> np.ndarray:
    """Circular (wrapped) difference along ``axis``: values in (-pi, pi]."""
    d = np.diff(x, axis=axis)
    return np.mod(d + np.pi, _TWO_PI) - np.pi


def _build_gradient_system(
    wrapped: np.ndarray,
    mask: np.ndarray,
    n_max: int = _N_MAX,
    stride: int = _STRIDE,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the gradient-domain least-squares system ``(G, b)``.

    Returns the design matrix ``G`` (N_edges x 65, Zernike gradient columns
    scaled by 2*pi, piston excluded) and the observation vector ``b``
    (N_edges, circular differences of the wrapped phase). Edges are the
    horizontal/vertical pixel pairs inside the aperture on a stride-subsampled
    grid.
    """
    height, width = wrapped.shape
    radius = min(height, width) // 2
    rows = np.arange(0, height, stride)
    cols = np.arange(0, width, stride)

    R, Theta, mask64 = _coordinate_maps(height, width, radius)
    R_s = R[rows][:, cols]
    Theta_s = Theta[rows][:, cols]
    mask_s = mask64[rows][:, cols] > 0
    powers = _radial_powers(R_s, n_max)

    h2, w2 = R_s.shape
    basis = np.zeros((_N_TERMS, h2, w2), dtype=np.float64)
    for i, (n, m) in enumerate(iter_nm_terms(n_max)):
        basis[i] = _zernike_mode(n, m, R_s, Theta_s, mask_s, powers)

    gx = np.diff(basis[1:], axis=2) * _TWO_PI  # (65, h2, w2-1)
    gy = np.diff(basis[1:], axis=1) * _TWO_PI  # (65, h2-1, w2)
    mx = mask_s[:, :-1] & mask_s[:, 1:]
    my = mask_s[:-1, :] & mask_s[1:, :]

    wrapped_s = wrapped[rows][:, cols]
    bx = _circ_diff(wrapped_s, axis=1)[mx]
    by = _circ_diff(wrapped_s, axis=0)[my]
    G = np.vstack([gx[:, mx].T, gy[:, my].T])
    b = np.concatenate([bx, by])
    return G, b


def _gradient_irls(G: np.ndarray, b: np.ndarray, iters: int = 70) -> np.ndarray:
    """Iteratively reweighted least squares with Huber weights.

    Iteration count matters: the Huber weight switching can take a long time
    to escape a bad basin. Observed convergence iterations: 29 (155508/0025),
    56 (164456/0000). 70 iterations covers the observed spread with margin.

    Returns the 65 non-piston coefficients (metadata order, piston excluded).
    """
    n_coeffs = G.shape[1]
    a = np.zeros(n_coeffs, dtype=np.float64)
    sigma = 0.05  # rad/px; matches the measured gradient noise floor
    reg = 1e-10 * np.eye(n_coeffs)
    for _ in range(iters):
        r = b - G @ a
        abs_r = np.abs(r)
        w = np.ones_like(abs_r)
        huber = abs_r > sigma
        w[huber] = sigma / abs_r[huber]
        w = np.where(abs_r > 0.5 * np.pi, 0.0, w)  # gradient-ambiguous edges
        GtW = (G * w[:, None]).T
        delta = np.linalg.solve(GtW @ G + reg, GtW @ r)
        a += delta
        if np.abs(delta).max() < 1e-7:
            break
    return a


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def recover_coefficients(
    sample_dir: str | Path,
    n_max: int = _N_MAX,
    refine: bool = False,
    stride: int = _STRIDE,
) -> np.ndarray:
    """Recover the 66 Zernike coefficients (piston=1.0) from a sample dir.

    Args:
        sample_dir: Directory containing ``phase.csv`` and ``metadata.json``.
        n_max: Radial degree (default 10 -> 66 terms).
        refine: Run the wrapped-domain Gauss-Newton polish (default False;
            the gradient-domain IRLS alone recovers coefficients to
            ~5e-5 accuracy on labeled data, while the GN polish is still
            experimental and can degrade the solution).
        stride: Pixel subsampling for the gradient fit (default 3).

    Returns:
        ``(66,)`` float64 coefficients in metadata (n,m) order.
    """
    sample_dir = Path(sample_dir)
    gray = load_stored_gray(sample_dir)
    wrapped = gray_to_wrapped(gray)
    mask = np.ones_like(wrapped, dtype=bool)
    radius = min(wrapped.shape) // 2
    _, _, mask64 = _coordinate_maps(wrapped.shape[0], wrapped.shape[1], radius)
    mask = mask64 > 0

    G, b = _build_gradient_system(wrapped, mask, n_max=n_max, stride=stride)
    a_np = _gradient_irls(G, b)
    coeffs = np.concatenate([[1.0], a_np])  # piston index 0 = 1.0

    if refine:
        coeffs = _wrapped_gn_refine(wrapped, mask, coeffs, n_max=n_max)
    return coeffs


def _wrapped_gn_refine(
    wrapped: np.ndarray,
    mask: np.ndarray,
    coeffs: np.ndarray,
    n_max: int = _N_MAX,
    stride: int = _STRIDE,
    iters: int = 30,
) -> np.ndarray:
    """Gauss-Newton polish in the wrapped domain (Stage 2).

    Minimizes ``sum_x rho(wrap(phi_obs - Za))`` with Huber weights, where
    ``a`` includes piston at index 0 (held fixed at 1.0). Subsampled grid.
    """
    height, width = wrapped.shape
    radius = min(height, width) // 2
    rows = np.arange(0, height, stride)
    cols = np.arange(0, width, stride)
    R, Theta, mask64 = _coordinate_maps(height, width, radius)
    R_s = R[rows][:, cols]
    Theta_s = Theta[rows][:, cols]
    mask_s = mask64[rows][:, cols] > 0
    powers = _radial_powers(R_s, n_max)
    h2, w2 = R_s.shape

    basis = np.zeros((_N_TERMS, h2, w2), dtype=np.float64)
    for i, (n, m) in enumerate(iter_nm_terms(n_max)):
        basis[i] = _zernike_mode(n, m, R_s, Theta_s, mask_s, powers)

    # Zernike design matrix over masked pixels (incl. piston col held at 1.0)
    m = mask_s
    Z = basis[:, m].T  # (N_px, 66)
    target = wrapped[rows][:, cols][m]
    a = coeffs.copy()
    sigma = 0.1  # Huber scale in radians
    reg = 1e-8 * np.eye(_N_TERMS)
    lam = 1e-3  # Levenberg-Marquardt damping
    for _ in range(iters):
        phase = (Z @ a) * _TWO_PI  # generator: phase_total = sum Z_k * amp_k * 2pi
        r = np.mod(target - phase + np.pi, _TWO_PI) - np.pi
        cost = float(np.sqrt((r**2).mean()))
        if cost < 1e-3:  # already at quantization floor — stop
            break
        w = np.where(np.abs(r) < sigma, 1.0, sigma / np.abs(r))
        w = np.where(np.abs(r) > 0.5 * np.pi, 1e-4, w)  # keep ambiguous px weakly
        # Piston column held fixed: zero out its row/col in the normal equations
        Zw = Z * w[:, None]
        H = Zw.T @ Zw + lam * np.eye(_N_TERMS) + reg
        H[0, :] = 0.0
        H[:, 0] = 0.0
        H[0, 0] = 1.0
        g = Zw.T @ r
        g[0] = 0.0
        delta = np.linalg.solve(H, g)
        a_new = a + delta
        phase_new = (Z @ a_new) * _TWO_PI
        r_new = np.mod(target - phase_new + np.pi, _TWO_PI) - np.pi
        cost_new = float(np.sqrt((r_new**2).mean()))
        if cost_new >= cost and np.abs(delta).max() < 1e-5:
            break
        if cost_new < cost:
            a = a_new
            lam = max(lam / 3.0, 1e-6)
        else:
            lam *= 3.0  # reject step, damp harder
    a[0] = 1.0
    return a


def recover_run(
    data_root: str | Path,
    run_id: str,
    out_suffix: str = "labels.npy",
    overwrite: bool = False,
    refine: bool = False,
    max_samples: int | None = None,
) -> dict:
    """Recover labels for all samples of one run, writing ``labels.npy`` sidecars.

    Args:
        data_root: Root dir containing run dirs.
        run_id: Run dir name (e.g. "20260402_155508").
        out_suffix: Sidecar filename written per sample.
        overwrite: Re-recover samples that already have a sidecar.
        refine: Run wrapped-domain GN polish (default False; see
            ``recover_coefficients``).
        max_samples: Cap on samples processed (for smoke tests).

    Returns:
        Summary dict: {run_id, total, recovered, failed, skipped}.
    """
    data_root = Path(data_root)
    run_dir = data_root / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run dir not found: {run_dir}")
    sample_dirs = sorted(run_dir.glob("sample_*"))
    if max_samples is not None:
        sample_dirs = sample_dirs[:max_samples]

    summary: dict = {"run_id": run_id, "total": len(sample_dirs), "recovered": 0, "failed": [], "skipped": 0}
    for i, sd in enumerate(sample_dirs):
        sidecar = sd / out_suffix
        if sidecar.exists() and not overwrite:
            summary["skipped"] += 1
            continue
        try:
            coeffs = recover_coefficients(sd, refine=refine)
            np.save(sidecar, coeffs[1:].astype(np.float32))  # 65 non-piston
            summary["recovered"] += 1
        except Exception as exc:  # noqa: BLE001 - per-sample isolation
            summary["failed"].append((sd.name, str(exc)))
            logger.warning("recovery failed for {}: {}", sd, exc)
        if (i + 1) % 25 == 0:
            logger.info("recover_run {}: {}/{} done ({} ok)",
                        run_id, i + 1, len(sample_dirs), summary["recovered"])
    logger.info("recover_run {} complete: {} recovered, {} failed, {} skipped",
                run_id, summary["recovered"], len(summary["failed"]), summary["skipped"])
    return summary


def fit_error_metrics(stored_gray: np.ndarray, recovered_coeffs: np.ndarray, n_max: int = _N_MAX) -> dict:
    """Reconstruction fidelity of recovered coefficients vs stored grayscale.

    Args:
        stored_gray: Stored ``phase.csv`` uint16 grayscale (H, W).
        recovered_coeffs: ``(66,)`` or ``(65,)`` coefficients (metadata order).

    Returns:
        Dict with masked-pixel metrics: mae_gray, max_gray, mse_gray,
        phase_rmse_rad, rel_l2.
    """
    height, width = stored_gray.shape
    radius = min(height, width) // 2
    _, _, mask64 = _coordinate_maps(height, width, radius)
    mask = mask64 > 0
    regen = coefficients_to_wrapped_gray(recovered_coeffs, n_max, height, width, radius)
    d = np.abs(regen.astype(np.float32) - stored_gray.astype(np.float32))[mask]
    phase_err = np.mod(
        gray_to_wrapped(regen) - gray_to_wrapped(stored_gray) + np.pi, _TWO_PI
    ) - np.pi
    rel_l2 = float(np.linalg.norm(d)) / (float(np.linalg.norm(stored_gray[mask].astype(np.float32))) + 1e-12)
    return {
        "mae_gray": float(d.mean()),
        "max_gray": float(d.max()),
        "mse_gray": float((d**2).mean()),
        "phase_rmse_rad": float(np.sqrt((phase_err[mask] ** 2).mean())),
        "rel_l2": rel_l2,
    }


def acceptance_check(metrics: dict, max_coeff: float, threshold: float = 0.9) -> bool:
    """A sample is acceptable iff phase_rmse_rad < 0.1 and rel_l2 < 0.05.

    Args:
        metrics: Output of ``fit_error_metrics``.
        max_coeff: Generation amplitude bound (metadata ``phase_params.max_coeff``).
        threshold: Reserved for batch acceptance fraction reporting (unused
            in the per-sample decision).

    Returns:
        True when the sample passes the quality gates.
    """
    return metrics["phase_rmse_rad"] < 0.1 and metrics["rel_l2"] < 0.05


def read_metadata_max_coeff(sample_dir: str | Path) -> float:
    """Read ``phase_params.max_coeff`` from a sample's metadata.json (default 5.0)."""
    try:
        meta = json.loads((Path(sample_dir) / "metadata.json").read_text())
        return float(meta["phase_params"].get("max_coeff", 5.0))
    except (OSError, KeyError, ValueError):
        return 5.0
