"""Hardware helpers for SLM+CCD closed-loop runners.

Migrated from :mod:`ao_shaping.algorithm.beam_shaping_utils`; the old module
re-exports these names for backward compatibility.

Contains:
- ``capture_amplitude``: CCD far-field capture -> float32 amplitude.
- ``call_with_timeout``: watchdog timeout for hardware SDK calls that can hang.
- Auto-exposure helpers (pure target computation + camera application).
- Frame recording (``frames/`` dir + PNG + JSONL metadata).

``utils`` is a leaf layer: hardware classes are only referenced under
``TYPE_CHECKING`` (or duck-typed via ``Any``), never imported at runtime.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from loguru import logger

from ao_shaping.utils.spots_calc import centroid

if TYPE_CHECKING:  # pragma: no cover
    from ao_shaping.drivers.ccd import BaseCamera

__all__ = [
    "capture_amplitude",
    "init_frame_recording",
    "save_frame_png",
    "record_frame",
    "auto_exposure_target_ms",
    "auto_exposure_possible",
    "apply_auto_exposure",
    "call_with_timeout",
]


def capture_amplitude(
    camera: "BaseCamera",
    center: Sequence[int] | None = None,
    size: Sequence[int] | None = None,
    n_sample: int = 1,
) -> np.ndarray:
    """Capture the far-field intensity from a CCD and return the amplitude.

    If ``center``/``size`` are provided, the camera window is reset to that
    region first (matching the target pattern's footprint). The captured
    intensity (uint16) is converted to float amplitude ``sqrt(I)`` and
    normalized to ``[0, 1]``.

    Args:
        camera: An open CCD camera object (must expose ``get_numpy_image``
            and optionally ``reset_window``).
        center: ``(cx, cy)`` window center in pixels (None to leave as-is).
        size: ``(h, w)`` window size in pixels (None to leave as-is).
        n_sample: Number of frames to average.

    Returns:
        Float32 2D amplitude array, normalized to ``[0, 1]``.
    """
    if center is not None and size is not None:
        try:
            # Sequences may arrive as lists; the driver expects 2-tuples.
            camera.reset_window(
                (int(center[0]), int(center[1])),
                (int(size[0]), int(size[1])),
            )
        except Exception:  # pragma: no cover - driver may not support
            logger.debug("Camera does not support reset_window; using full frame")

    img = camera.get_numpy_image(n_sample=n_sample, skip_first=True)
    intensity = np.asarray(img, dtype=np.float32)
    intensity = np.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)
    amp = np.sqrt(intensity)
    amax = amp.max()
    if amax > 0:
        amp = amp / amax
    return amp.astype(np.float32)


# ---------------------------------------------------------------------------
# Frame recording (shared by diff_beam_runner and diff_shaping_runner)
# ---------------------------------------------------------------------------
_frames_dir: Path | None = None
_frame_counter: int = 0


def init_frame_recording(out_dir: Path) -> None:
    """Create the ``frames/`` directory under ``out_dir`` and reset the counter.

    Args:
        out_dir: Result directory that will contain the ``frames/`` subdir.
    """
    global _frames_dir, _frame_counter
    _frames_dir = Path(out_dir) / "frames"
    _frames_dir.mkdir(parents=True, exist_ok=True)
    _frame_counter = 0


def save_frame_png(frame: np.ndarray, path: Path, title: str) -> None:
    """Render one CCD frame to a PNG (inferno colormap + colorbar).

    Args:
        frame: 2D array to render.
        path: Output PNG path.
        title: Figure title.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4.5))
        im = ax.imshow(np.asarray(frame), cmap="inferno")
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)
    except Exception:  # pragma: no cover - plotting must never break the loop
        logger.warning("PNG 保存失败: {}", path.name)


def record_frame(
    raw: np.ndarray,
    phase_desc: str,
    exposure_ms: float,
) -> dict:
    """Save one raw CCD frame plus a JSONL meta line; log per-frame stats.

    Also returns the meta dict so callers can reuse the recorded stats.

    Args:
        raw: Raw CCD frame array.
        phase_desc: Description of the phase that produced this frame.
        exposure_ms: Exposure time in milliseconds.

    Returns:
        Metadata dict for this frame.
    """
    global _frame_counter
    _frame_counter += 1
    idx = _frame_counter
    frame = np.asarray(raw, dtype=np.float32)
    peak = float(frame.max()) if frame.size else 0.0
    total = float(frame.sum()) if frame.size else 0.0
    if frame.size:
        cx, cy = centroid(frame, return_float=True)
    else:
        cx = cy = 0.0
    meta = {
        "frame": idx,
        "phase": phase_desc,
        "exposure_ms": exposure_ms,
        "peak": peak,
        "sum": total,
        "centroid": [float(cy), float(cx)],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    frames_dir = _frames_dir
    if frames_dir is not None:
        np.save(frames_dir / f"frame_{idx:05d}.npy", np.asarray(raw))
        save_frame_png(
            np.asarray(raw),
            frames_dir / f"frame_{idx:05d}.png",
            f"frame {idx:04d} - {phase_desc} (peak={peak:.0f})",
        )
        with open(frames_dir / "frame_meta.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    logger.info(
        "帧 {:04d}: phase={} peak={:.1f} sum={:.0f} centroid=({:.1f}, {:.1f})",
        idx,
        phase_desc,
        peak,
        total,
        cy,
        cx,
    )
    return meta


# ---------------------------------------------------------------------------
# Auto-exposure helpers (shared by diff_beam_runner and diff_shaping_runner)
# ---------------------------------------------------------------------------
def auto_exposure_target_ms(
    current_ms: float,
    peak: float,
    target_brightness: float = 180.0,
    tol: float = 0.2,
    *,
    min_ms: float = 0.02,
    max_ms: float = 1000.0,
    sat_floor: float = 245.0,
    max_boost: float = 4.0,
    max_cut: float = 0.25,
) -> float:
    """Compute the next exposure (ms) that drives ``peak`` into ``target ± tol``.

    Pure function — no hardware access. The caller applies the result via
    ``camera.reset_exposure_time()``.

    Args:
        current_ms: Current exposure time in milliseconds.
        peak: Measured peak brightness.
        target_brightness: Target peak brightness (default 180).
        tol: Relative tolerance band (default 0.2).
        min_ms: Minimum exposure clamp.
        max_ms: Maximum exposure clamp.
        sat_floor: Hard saturation guard for 8-bit CCD.
        max_boost: Maximum boost factor.
        max_cut: Maximum cut factor.

    Returns:
        Suggested exposure time in milliseconds.
    """
    low, high = target_brightness * (1.0 - tol), target_brightness * (1.0 + tol)
    if low <= peak <= high:
        return float(current_ms)
    if peak >= sat_floor:
        scale = max(target_brightness / peak, 0.1)
    elif peak < low:
        scale = min(target_brightness / peak, max_boost)
    else:  # peak > high
        scale = max(target_brightness / peak, max_cut)
    return float(np.clip(current_ms * scale, min_ms, max_ms))


def auto_exposure_possible(camera: Any) -> bool:
    """Return True if the camera supports runtime exposure adjustment."""
    return callable(getattr(camera, "reset_exposure_time", None))


def apply_auto_exposure(
    camera: Any,
    peak: float,
    current_ms: float,
    target_brightness: float,
    tol: float,
) -> tuple[float, bool]:
    """Adjust camera exposure toward the target peak band.

    Args:
        camera: Camera object with ``reset_exposure_time``.
        peak: Current measured peak.
        current_ms: Current exposure time.
        target_brightness: Target peak brightness.
        tol: Tolerance band.

    Returns:
        ``(actual_exposure_ms, changed)`` tuple.
    """
    next_ms = auto_exposure_target_ms(current_ms, peak, target_brightness, tol)
    if abs(next_ms - current_ms) < 1e-9:
        return current_ms, False
    actual_ms = float(camera.reset_exposure_time(next_ms))
    logger.info(
        "自动曝光: peak={:.1f} -> 曝光 {:.3f} -> {:.3f} ms",
        peak,
        current_ms,
        actual_ms,
    )
    return actual_ms, True


# ---------------------------------------------------------------------------
# Hardware timeout helper
# ---------------------------------------------------------------------------
def call_with_timeout(fn: Any, timeout_s: float, desc: str) -> Any:
    """Run ``fn`` in a daemon thread with a watchdog timeout.

    Hardware SDK calls can hang forever; this bounds the wait and raises
    ``TimeoutError`` if the call does not return in time.

    Args:
        fn: Callable to run.
        timeout_s: Timeout in seconds.
        desc: Description for error messages.

    Returns:
        Return value of ``fn``.

    Raises:
        TimeoutError: If ``fn`` does not complete within ``timeout_s``.
    """
    result: list[Any] = []
    error: list[BaseException] = []

    def _runner() -> None:
        try:
            result.append(fn())
        except BaseException as exc:  # noqa: BLE001
            error.append(exc)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise TimeoutError(f"{desc} 超时 ({timeout_s}s)")
    if error:
        raise error[0]
    return result[0] if result else None