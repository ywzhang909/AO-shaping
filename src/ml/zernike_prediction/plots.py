"""Publication-quality matplotlib plots for Zernike-coefficient regression.

Visualizes predictions of the 65 non-piston Zernike coefficients (``n_max=10``,
metadata ``(n, m)`` order, piston at index 0 = 1.0) produced by the
``ml.zernike_prediction`` pipeline:

- ``predict_true_scatter``: per-coefficient predicted-vs-true scatter grid.
- ``per_order_mae_bar`` / ``per_coeff_mae_bar``: MAE bar charts.
- ``phase_grid`` / ``phase_error_map``: wrapped-phase heatmaps (via
  :func:`ml.zernike_prediction.phase_gen.coefficients_to_phase_radians`).
- ``loss_curves``: train/val loss on a log scale.
- ``all_plots``: one-call convenience for the standard evaluation figure set.

Design notes:

- Backend is forced to ``Agg`` at import time (headless CI safety; no
  interactive windows, no tkinter/Qt).
- Coefficient vectors may be 65 (non-piston) or 66 (piston-first) elements
  everywhere; phase generation accepts both, and metrics are always computed
  on the 65 non-piston coefficients.
- Metrics come from :mod:`ml.zernike_prediction.metrics` when available
  (``per_coeff_mae`` / ``per_order_mae``); until that module lands, equivalent
  local fallbacks are used. The module import is defensive either way.
- All text is English-only (CJK-safe on headless renderers).
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

try:  # headless CI safety — never open an interactive window
    matplotlib.use("Agg")
except Exception:  # pragma: no cover - backend already locked elsewhere
    pass

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib import colormaps

from ml.zernike_prediction.phase_gen import (
    COEFF_ORDER_NAMES,
    build_basis_maps,
    coefficients_to_phase_radians,
    count_zernike_terms,
    iter_nm_terms,
)

# metrics.py lands alongside this module; use it when present, fall back to
# local MAE helpers until it exists (see _per_coeff_mae / _per_order_mae).
try:
    from ml.zernike_prediction import metrics
except ImportError:  # pragma: no cover - metrics not yet written
    metrics = None

__all__ = [
    "all_plots",
    "loss_curves",
    "per_coeff_mae_bar",
    "per_order_mae_bar",
    "phase_error_map",
    "phase_grid",
    "predict_true_scatter",
]

N_MAX = 10  # radial degree bound -> 66 terms (65 non-piston)
_TWO_PI = 2.0 * np.pi
_DPI = 150
_GRID_ALPHA = 0.3
_LOG_MSG = "zernike_prediction.plots: {}"

# 66 "nXmY" labels (piston first); [1:] is the 65 non-piston set.
_NON_PISTON_LABELS = COEFF_ORDER_NAMES(N_MAX)[1:]
_N_NON_PISTON = count_zernike_terms(N_MAX) - 1  # 65


def _pad_coeffs(coeffs: np.ndarray) -> np.ndarray:
    """Zero-pad a (possibly truncated) coefficient vector to the full 65 terms."""
    arr = np.asarray(coeffs, dtype=np.float64).ravel()
    if arr.size >= _N_NON_PISTON:
        return arr[:_N_NON_PISTON]
    return np.pad(arr, (0, _N_NON_PISTON - arr.size))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _as_float_pair(
    pred: np.ndarray, true: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a (pred, true) pair: 2-D, identical shapes, 65/66 columns."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(
            "pred and true must have identical shapes; "
            f"got pred.shape={pred.shape}, true.shape={true.shape}"
        )
    if pred.ndim != 2:
        raise ValueError(f"pred/true must be 2-D (N, C), got shape {pred.shape}")
    if not (1 <= pred.shape[1] <= _N_NON_PISTON + 1):
        raise ValueError(
            f"expected 1..{_N_NON_PISTON + 1} coefficients per sample (n_max={N_MAX}), "
            f"got {pred.shape[1]}"
        )
    return pred, true


def _non_piston(arr: np.ndarray) -> np.ndarray:
    """Drop the piston column (index 0) when present -> (N, 65)."""
    if arr.shape[1] == _N_NON_PISTON + 1:
        return arr[:, 1:]
    return arr


def _fallback_per_coeff_mae(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """MAE per non-piston coefficient, (65,). Temporary until metrics lands."""
    return np.abs(pred - true).mean(axis=0)


def _per_coeff_mae(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """MAE per non-piston coefficient (65,), from metrics when available."""
    pred_np, true_np = _non_piston(pred), _non_piston(true)
    if metrics is not None:
        mae = np.asarray(metrics.per_coeff_mae(pred_np, true_np), dtype=np.float64).ravel()
        if mae.size == _N_NON_PISTON + 1:  # metrics returned piston-inclusive
            mae = mae[1:]
        return mae
    return _fallback_per_coeff_mae(pred_np, true_np)


def _per_order_mae(pred: np.ndarray, true: np.ndarray) -> dict[int, float]:
    """MAE per radial order 1..10, from metrics when available."""
    pred_np, true_np = _non_piston(pred), _non_piston(true)
    if metrics is not None:
        out = {int(k): float(v) for k, v in metrics.per_order_mae(pred_np, true_np).items()}
    else:
        coeff_mae = _fallback_per_coeff_mae(pred_np, true_np)
        terms = iter_nm_terms(N_MAX)[1:]  # 65 non-piston (n, m) pairs
        out = {}
        for n in range(1, N_MAX + 1):
            idx = [i for i, (nn, _m) in enumerate(terms) if nn == n]
            out[n] = float(coeff_mae[idx].mean())
    # Always expose a complete 1..10 axis, filling gaps with 0.0.
    return {n: out.get(n, 0.0) for n in range(1, N_MAX + 1)}


def _palette_color(i: int) -> tuple[float, float, float, float]:
    """Consistent tab10 color, cycled."""
    return colormaps["tab10"](i % 10)


def _finalize(fig: plt.Figure, title: str | None, save_path: str | Path) -> Path:
    """Optional suptitle, tight layout, save at dpi=150, close the figure."""
    if title:
        fig.suptitle(title, fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.debug(_LOG_MSG, f"saved {path}")
    return path


# ---------------------------------------------------------------------------
# Public plot API
# ---------------------------------------------------------------------------


def predict_true_scatter(
    pred: np.ndarray,
    true: np.ndarray,
    save_path: str | Path,
    title: str = "Predict vs True",
    n_show_terms: int = 12,
) -> Path:
    """Per-coefficient predicted-vs-true scatter grid (one scatter per term).

    Args:
        pred: Predicted coefficients, shape ``(N, C)`` with ``C`` in (65, 66).
        true: Ground-truth coefficients, same shape as ``pred``.
        save_path: Output PNG path (parent dirs are created).
        title: Figure suptitle.
        n_show_terms: Number of coefficient subplots (grid ~3x4, clamped to C).

    Returns:
        ``save_path`` as a resolved :class:`pathlib.Path`.
    """
    pred_a, true_a = _as_float_pair(pred, true)
    pred_np, true_np = _non_piston(pred_a), _non_piston(true_a)
    mae = _per_coeff_mae(pred_np, true_np)
    n_coeffs = pred_np.shape[1]
    n_show = max(1, min(int(n_show_terms), n_coeffs))

    n_cols = 4
    n_rows = math.ceil(n_show / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.6, n_rows * 2.6))
    ax_list = np.atleast_1d(axes).ravel()

    for ax, i in zip(ax_list, range(n_show)):
        t = true_np[:, i]
        p = pred_np[:, i]
        lo = min(float(t.min()), float(p.min()))
        hi = max(float(t.max()), float(p.max()))
        ax.scatter(t, p, s=8, alpha=0.6, color=_palette_color(i), edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(f"{_NON_PISTON_LABELS[i]}  MAE={mae[i]:.4f}", fontsize=8)
        ax.grid(True, alpha=_GRID_ALPHA, lw=0.4)
        ax.tick_params(labelsize=7)
        if i % n_cols == 0:
            ax.set_ylabel("predicted", fontsize=8)
        if i // n_cols == n_rows - 1:
            ax.set_xlabel("true", fontsize=8)

    for ax in ax_list[n_show:]:
        ax.set_visible(False)

    return _finalize(fig, title, save_path)


def per_order_mae_bar(
    pred: np.ndarray,
    true: np.ndarray,
    save_path: str | Path,
    title: str = "Per-order MAE",
) -> Path:
    """Bar chart of per-radial-order MAE (orders 1..10), values labeled."""
    pred_a, true_a = _as_float_pair(pred, true)
    order_mae = _per_order_mae(pred_a, true_a)
    orders = list(range(1, N_MAX + 1))
    vals = [order_mae[n] for n in orders]
    ymax = max(vals) if vals else 1.0

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(orders, vals, width=0.7, color=[_palette_color(i) for i in range(len(orders))])
    ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1.0)
    for x, v in zip(orders, vals):
        ax.text(x, v + ymax * 0.02, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Radial order n", fontsize=9)
    ax.set_ylabel("MAE (rad)", fontsize=9)
    ax.set_xticks(orders)
    ax.set_xticklabels([str(n) for n in orders], fontsize=8)
    ax.grid(axis="y", alpha=_GRID_ALPHA, lw=0.4)
    ax.tick_params(labelsize=8)
    return _finalize(fig, title, save_path)


def per_coeff_mae_bar(
    pred: np.ndarray,
    true: np.ndarray,
    save_path: str | Path,
    title: str = "Per-coefficient MAE",
) -> Path:
    """Bar chart of MAE for all 65 non-piston coefficients ("nXmY" labels)."""
    pred_a, true_a = _as_float_pair(pred, true)
    mae = _per_coeff_mae(pred_a, true_a)
    n = mae.size
    labels = _NON_PISTON_LABELS[:n]
    ymax = float(mae.max()) if n else 1.0

    fig, ax = plt.subplots(figsize=(max(8.0, n * 0.18), 4.2))
    ax.bar(np.arange(n), mae, width=0.8, color=[_palette_color(i) for i in range(n)])
    ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1.0)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_xlabel("Zernike term (nXmY, metadata order)", fontsize=9)
    ax.set_ylabel("MAE (rad)", fontsize=9)
    ax.grid(axis="y", alpha=_GRID_ALPHA, lw=0.4)
    ax.tick_params(labelsize=7)
    return _finalize(fig, title, save_path)


def phase_grid(
    coefficients_list: list[np.ndarray],
    labels_list: list[str],
    save_path: str | Path,
    size: tuple[int, int] = (192, 192),
    n_cols: int = 4,
    title: str | None = None,
    cmap: str = "twilight",
    radius: int | None = None,
) -> Path:
    """Grid of wrapped-phase heatmaps, one row per coefficient array.

    Args:
        coefficients_list: Coefficient vectors (65 or 66 elements each).
        labels_list: Subplot titles, one per coefficient array.
        save_path: Output PNG path.
        size: ``(height, width)`` phase-grid resolution.
        n_cols: Subplots per row.
        title: Optional figure suptitle.
        cmap: Colormap for the wrapped phase (default "twilight").
        radius: Unit-disk radius in pixels (default ``min(size) // 2``).
    """
    if not coefficients_list:
        raise ValueError("coefficients_list must be non-empty")
    if len(labels_list) != len(coefficients_list):
        raise ValueError(
            f"labels_list ({len(labels_list)}) must match "
            f"coefficients_list ({len(coefficients_list)})"
        )
    n_rows = math.ceil(len(coefficients_list) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.4, n_rows * 2.6))
    ax_list = np.atleast_1d(axes).ravel()

    im = None
    for ax, coeffs, label in zip(ax_list, coefficients_list, labels_list):
        phase = coefficients_to_phase_radians(_pad_coeffs(coeffs), n_max=N_MAX, size=size, radius=radius)
        im = ax.imshow(phase, cmap=cmap, vmin=0.0, vmax=_TWO_PI)
        ax.set_title(label, fontsize=9)
        # Hide frame/ticks via spines (tight_layout-compatible; set_axis_off is not)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in ax_list[len(coefficients_list) :]:
        ax.set_visible(False)

    if im is not None:
        # Single-ax colorbar keeps a SubplotSpec (ax-list spans via make_axes,
        # which tight_layout flags as incompatible).
        cbar = fig.colorbar(im, ax=ax_list[0], fraction=0.03, pad=0.03)
        cbar.set_label("wrapped phase (rad)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    return _finalize(fig, title, save_path)


def phase_error_map(
    pred_coeffs: np.ndarray,
    true_coeffs: np.ndarray,
    save_path: str | Path,
    size: tuple[int, int] = (192, 192),
    radius: int | None = None,
    title: str = "Phase error (rad)",
) -> Path:
    """Heatmap of the circular wrapped-phase difference, masked to the aperture.

    The error is the wrapped difference ``(pred - true + pi) mod 2pi - pi`` of
    the two wrapped phase maps, shown with a symmetric diverging colormap.

    Args:
        pred_coeffs: Predicted coefficients (65 or 66 elements).
        true_coeffs: Ground-truth coefficients (65 or 66 elements).
        save_path: Output PNG path.
        size: ``(height, width)`` phase-grid resolution.
        radius: Unit-disk radius in pixels (default ``min(size) // 2``).
        title: Figure suptitle.
    """
    pred_phase = coefficients_to_phase_radians(_pad_coeffs(pred_coeffs), n_max=N_MAX, size=size, radius=radius)
    true_phase = coefficients_to_phase_radians(_pad_coeffs(true_coeffs), n_max=N_MAX, size=size, radius=radius)
    err = (pred_phase - true_phase + np.pi) % _TWO_PI - np.pi  # wrap to [-pi, pi)

    _, mask = build_basis_maps(n_max=N_MAX, height=size[0], width=size[1], radius=radius)
    err_masked = np.ma.masked_where(~mask.astype(bool), err)
    vmax = max(float(np.abs(err_masked).max()), 1e-12)

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    im = ax.imshow(err_masked, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("phase error (rad)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    return _finalize(fig, title, save_path)


def loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    save_path: str | Path,
    title: str = "Loss curves",
) -> Path:
    """Train/val loss curves on a log-scaled y axis."""
    train = np.maximum(np.asarray(train_losses, dtype=np.float64).ravel(), 1e-12)
    val = np.maximum(np.asarray(val_losses, dtype=np.float64).ravel(), 1e-12)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(np.arange(train.size), train, label="train", color=_palette_color(0), lw=1.2)
    ax.plot(np.arange(val.size), val, label="val", color=_palette_color(1), lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Loss (log)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=_GRID_ALPHA, lw=0.4)
    ax.tick_params(labelsize=8)
    return _finalize(fig, title, save_path)


def all_plots(
    pred: np.ndarray,
    true: np.ndarray,
    out_dir: str | Path,
    prefix: str = "eval",
    phase_size: tuple[int, int] = (192, 192),
) -> dict[str, Path]:
    """Render the standard evaluation figure set into ``out_dir``.

    Produces: per-coefficient scatter, per-order MAE bar, per-coefficient MAE
    bar, and the phase-error map of the mean coefficients (overall phase error).

    Returns:
        ``{name: path}`` map with keys ``scatter``, ``order_mae``,
        ``coeff_mae``, ``phase_error``.
    """
    pred_a, true_a = _as_float_pair(pred, true)
    out = Path(out_dir)
    paths: dict[str, Path] = {
        "scatter": predict_true_scatter(
            pred_a, true_a, out / f"{prefix}_predict_true_scatter.png"
        ),
        "order_mae": per_order_mae_bar(pred_a, true_a, out / f"{prefix}_per_order_mae.png"),
        "coeff_mae": per_coeff_mae_bar(pred_a, true_a, out / f"{prefix}_per_coeff_mae.png"),
        "phase_error": phase_error_map(
            pred_a.mean(axis=0),
            true_a.mean(axis=0),
            out / f"{prefix}_phase_error.png",
            size=phase_size,
        ),
    }
    logger.debug(_LOG_MSG, f"wrote {len(paths)} plots to {out}")
    return paths
