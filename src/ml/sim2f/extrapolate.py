"""Out-of-window evaluation for a forward model that predicts the whole sensor.

The experiment this module exists for: a camera reads only a small ROI, the
forward model is fitted on that ROI, and it is then asked for the full frame --
including light the camera never saw.  Because the simulator knows the true
full frame, the prediction can be scored outside the window, which is the only
honest way to test "predict what the CCD cannot see".

Everything here is post-processing on already-computed frames; the physics stays
in :mod:`ml.sim2f.optics`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ml.sim2f.window import CameraWindow

__all__ = [
    "ExtrapolationMetrics",
    "extrapolation_metrics",
    "frame_np",
    "moment_metrics",
    "open_loop_frame",
]


def frame_np(value: Any, *, normalise: bool = True) -> np.ndarray:
    """Convert a frame to a 2-D float64 numpy array.

    Args:
        value: ``(H, W)``, ``(C, H, W)`` or ``(B, C, H, W)`` numpy array or
            torch tensor (the first batch/channel is taken).
        normalise: Divide by the total so the frame sums to one.

    Returns:
        2-D float64 array, non-negative.

    Raises:
        ValueError: If the input has fewer than two dimensions or is empty.
    """
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float64)
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"expected a frame, got shape {np.shape(value)}")
    arr = np.clip(arr, 0.0, None)
    if normalise:
        total = float(arr.sum())
        if total > 0:
            arr = arr / total
    return arr


def _window_mask(window: CameraWindow | None, shape: tuple[int, int]) -> np.ndarray:
    """Boolean full-frame mask that is ``True`` inside the window."""
    if window is None:
        return np.ones(shape, dtype=bool)
    return window.mask((shape[1], shape[0])).astype(bool)


def _ncc(pred: np.ndarray, truth: np.ndarray) -> float:
    """Zero-mean normalised cross-correlation of two frames."""
    a = pred - pred.mean()
    b = truth - truth.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def _l1(pred: np.ndarray, truth: np.ndarray) -> float:
    """Relative L1: ``sum|p - t| / sum t`` (both frames sum to one)."""
    denom = float(truth.sum())
    return float(np.abs(pred - truth).sum() / denom) if denom > 0 else float("nan")


def _centroid(
    frame: np.ndarray, *, threshold_frac: float = 0.01
) -> tuple[float, float] | None:
    """Intensity centroid ``(x, y)`` above a peak-relative threshold."""
    peak = float(frame.max())
    if peak <= 0:
        return None
    yy, xx = np.nonzero(frame >= threshold_frac * peak)
    if xx.size == 0:
        return None
    weights = frame[yy, xx]
    total = float(weights.sum())
    if total <= 0:
        return None
    return float((xx * weights).sum() / total), float((yy * weights).sum() / total)


def _rms_radius(frame: np.ndarray, *, threshold_frac: float = 0.01) -> float:
    """Second-moment radius about the centroid, pixels."""
    centre = _centroid(frame, threshold_frac=threshold_frac)
    if centre is None:
        return float("nan")
    cx, cy = centre
    yy, xx = np.mgrid[0 : frame.shape[0], 0 : frame.shape[1]]
    weights = np.where(frame >= threshold_frac * frame.max(), frame, 0.0)
    total = float(weights.sum())
    if total <= 0:
        return float("nan")
    return float(np.sqrt((weights * ((xx - cx) ** 2 + (yy - cy) ** 2)).sum() / total))


def _d86_radius(frame: np.ndarray) -> float:
    """Radius enclosing 86.6% of the energy (the D86 spot size), pixels."""
    centre = _centroid(frame)
    if centre is None:
        return float("nan")
    cx, cy = centre
    yy, xx = np.mgrid[0 : frame.shape[0], 0 : frame.shape[1]]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    order = np.argsort(r, axis=None)
    cumulative = np.cumsum(frame.reshape(-1)[order])
    total = float(cumulative[-1])
    if total <= 0:
        return float("nan")
    idx = int(np.searchsorted(cumulative, 0.866 * total))
    idx = min(idx, order.size - 1)
    return float(r.reshape(-1)[order][idx])


def moment_metrics(frame: Any) -> dict[str, float]:
    """Spot moments of a single frame: centroid, RMS radius, D86.

    Args:
        frame: Any frame :func:`frame_np` accepts.

    Returns:
        ``{"cx", "cy", "rms_radius_px", "d86_px"}``; unusable frames give NaN.
    """
    arr = frame_np(frame)
    centre = _centroid(arr)
    cx, cy = centre if centre is not None else (float("nan"), float("nan"))
    return {
        "cx": cx,
        "cy": cy,
        "rms_radius_px": _rms_radius(arr),
        "d86_px": _d86_radius(arr),
    }


@dataclass(frozen=True)
class ExtrapolationMetrics:
    """Scores for a whole-sensor prediction against the true frame.

    All errors are computed on unit-sum frames, so ``l1_*`` is a fraction of the
    measured energy.  ``*_in``/``*_out`` split the frame by the camera window:
    ``in`` is what the model could be fitted on, ``out`` is the extrapolation.

    Attributes:
        l1_all: Relative L1 over the whole frame.
        l1_in: Relative L1 inside the window (denominator: true energy inside).
        l1_out: Relative L1 outside the window (denominator: true energy outside).
        ncc_all: Zero-mean NCC over the whole frame.
        ncc_in: NCC inside the window.
        ncc_out: NCC outside the window.
        ee_in_true: True fraction of energy inside the window.
        ee_in_pred: Predicted fraction of energy inside the window.
        ee_out_err: ``|ee_in_pred - ee_in_true|``, i.e. the energy-split error.
        centroid_err_px: Distance between predicted and true centroids.
        d86_err_px: Absolute difference of the D86 spot sizes.
        peak_ratio: Predicted peak over true peak.
        energy_total_ratio: Predicted total over true total (1.0 for unit-sum).
    """

    l1_all: float
    l1_in: float
    l1_out: float
    ncc_all: float
    ncc_in: float
    ncc_out: float
    ee_in_true: float
    ee_in_pred: float
    ee_out_err: float
    centroid_err_px: float
    d86_err_px: float
    peak_ratio: float
    energy_total_ratio: float

    def as_dict(self) -> dict[str, float]:
        """Plain dict, for JSON/CSV artifacts."""
        return asdict(self)


def extrapolation_metrics(
    pred: Any,
    truth: Any,
    window: CameraWindow | None = None,
) -> ExtrapolationMetrics:
    """Score a whole-sensor prediction, split by the camera window.

    Args:
        pred: Predicted full frame.
        truth: True full frame (same shape).
        window: The camera window; ``None`` means the whole frame was measured
            (``*_out`` metrics are then meaningless and reported as NaN).

    Returns:
        The :class:`ExtrapolationMetrics`.

    Raises:
        ValueError: If the frames do not have the same shape.
    """
    p = frame_np(pred)
    t = frame_np(truth, normalise=False)
    total = float(t.sum())
    if total <= 0:
        raise ValueError("truth frame has no energy")
    t = t / total
    if p.shape != t.shape:
        raise ValueError(f"prediction {p.shape} and truth {t.shape} disagree")

    mask = _window_mask(window, t.shape)
    inside = mask
    outside = ~mask
    t_in, p_in = t[inside], p[inside]
    t_out, p_out = t[outside], p[outside]

    truth_moments = moment_metrics(t)
    pred_moments = moment_metrics(p)
    centroid_err = float(
        np.hypot(
            pred_moments["cx"] - truth_moments["cx"],
            pred_moments["cy"] - truth_moments["cy"],
        )
    )
    peak_ratio = float(p.max() / t.max()) if t.max() > 0 else float("nan")

    have_outside = bool(outside.any()) and float(t_out.sum()) > 0
    return ExtrapolationMetrics(
        l1_all=_l1(p, t),
        l1_in=_l1(p_in, t_in) if float(t_in.sum()) > 0 else float("nan"),
        l1_out=_l1(p_out, t_out) if have_outside else float("nan"),
        ncc_all=_ncc(p, t),
        ncc_in=_ncc(p_in, t_in) if float(t_in.sum()) > 0 else float("nan"),
        ncc_out=_ncc(p_out, t_out) if have_outside else float("nan"),
        ee_in_true=float(t_in.sum()),
        ee_in_pred=float(p_in.sum()),
        ee_out_err=float(abs(p_in.sum() - t_in.sum())),
        centroid_err_px=centroid_err,
        d86_err_px=float(abs(pred_moments["d86_px"] - truth_moments["d86_px"])),
        peak_ratio=peak_ratio,
        energy_total_ratio=float(p.sum()),
    )


def open_loop_frame(
    bench: Any,
    phase: Any,
    *,
    full_ccd_px: int | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """One open-loop forward prediction on the whole sensor.

    This is the "single shot, no measurement" path: the caller supplies the
    commanded phase and gets the model's full-frame prediction, which may extend
    far outside any camera window.

    Args:
        bench: Bench with ``forward`` (a :class:`~ml.sim2f.optics.Sim2f`, a
            :class:`~ml.sim2f.calibrate.CalibratedBench`, or a
            :class:`~ml.sim2f.window.WindowedBench`).
        phase: Commanded phase, radians.
        full_ccd_px: Expected full sensor size; checked when given.
        **kwargs: Forward-pass options (``aberration``, ``illumination``,
            ``detector``, ...).

    Returns:
        Unit-sum ``(ccd_px, ccd_px)`` prediction.

    Raises:
        ValueError: If the prediction is not the expected full-frame size.
    """
    forward = getattr(bench, "full_forward", bench.forward)
    pred = frame_np(forward(phase, **kwargs))
    if full_ccd_px is not None and pred.shape != (full_ccd_px, full_ccd_px):
        raise ValueError(
            f"expected a {full_ccd_px}x{full_ccd_px} prediction, got {pred.shape}"
        )
    return pred
