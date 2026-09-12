"""Capture a raw flat-phase far-field spot TIFF and debug centroid algorithms.

2026-09-11: user re-requested capturing the 1200us raw spot TIFF and debugging
the centroid algorithm against it ("请重新采集一个1200um下原始光斑的tiff图片，
基于该图片调试质心算法").

This script, run with the hardware online (SLM200 + Daheng CCD):
  1. Opens the SLM in memory mode (never DVI) and the CCD.
  2. Displays a flat (zero-radian) phase via ``create_phase_from_array`` +
     memory-slot rotation (random slot in 2..125, never the currently shown one).
  3. Captures the raw frame at ``--exposure-us`` (default 1200us = 1.2ms).
  4. Saves the frame as BOTH a 16-bit TIFF (Pillow; tifffile is not installed)
     and a .npy array.
  5. Runs several centroid algorithms on the SAME frame and prints a comparison:
       - argmax (frame global maximum)  -> true 0-order spot (AGENTS.md rule:
         always locate 0-order by argmax, never by geometry)
       - raw intensity centroid          (spots_calc.centroid, (cx,cy)=(col,row))
       - de-backgrounded intensity centroid (median subtraction)
       - corner-max threshold binary centroid (threshold = max of 4 corner blocks)
  6. Saves an annotated comparison PNG (full frame + zoomed ROI).

Usage:
    uv run --group ml python scripts/diff_beam_capture_flat.py \
        --exposure-us 1200 --output data/diff_beam/flat_tiff
"""

from __future__ import annotations

import argparse
import random
import threading
import time
from pathlib import Path

import numpy as np
from loguru import logger

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - hardware path only
    raise SystemExit(f"Pillow required: {exc}")

try:
    from ao_shaping.drivers.ccd.daheng import DahengCamManager
    from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200
    from ao_shaping.utils.spots_calc import centroid as spots_centroid
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False

# ── watchdog helpers (same pattern as diff_beam_runner) ────────────────────


def call_with_timeout(fn: object, timeout_s: float, desc: str) -> object:
    """Run ``fn`` in a daemon thread with a watchdog timeout.

    Hardware SDK calls (SLM open, CCD capture) can hang forever with no
    Python-visible timeout; this helper bounds the wait and raises
    ``TimeoutError`` if the call does not return in time.
    """
    result: list[object] = []
    error: list[BaseException] = []

    def _runner() -> None:
        try:
            result.append(fn())  # type: ignore[operator]
        except BaseException as exc:  # noqa: BLE001 - propagate any exception
            error.append(exc)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise TimeoutError(f"{desc} 超时 ({timeout_s}s) — 原生 SDK 调用可能挂起")
    if error:
        raise error[0]
    return result[0] if result else None


def pick_slm_slot(slm: object, last_slot: int | None) -> tuple[int, int | None]:
    """Pick a random SLM memory slot in [2, 125] different from the last used.

    On first use, reads the currently displayed slot from the device so the
    rotation survives process restarts (memory mode only).
    """
    if last_slot is None:
        try:
            last_slot = int(slm.get_displayed_memory_number())  # type: ignore[attr-defined]
            logger.info("SLM当前显示槽: {}", last_slot)
        except Exception as exc:
            logger.debug("读取 SLM 当前显示槽失败: {}", exc)
            last_slot = None
    candidates = [s for s in range(2, 126) if s != last_slot]
    slot = random.choice(candidates)
    return slot, slot


def display_flat_phase(
    slm: object, settle_time_s: float, last_slot: int | None
) -> int | None:
    """Display a zero-radian flat phase (full panel) with slot rotation."""
    height = int(getattr(slm, "height", 1200))
    width = int(getattr(slm, "width", 1920))
    phase_rad = np.zeros((height, width), dtype=np.float64)
    gray = slm.create_phase_from_array(phase_rad)  # type: ignore[attr-defined]
    slot, new_last = pick_slm_slot(slm, last_slot)
    slm.write_phase(gray, memory_number=slot)  # type: ignore[attr-defined]
    time.sleep(0.05)
    slm.display_memory(slot)  # type: ignore[attr-defined]
    time.sleep(settle_time_s)
    return new_last


# ── centroid algorithms (all return (row, col)) ────────────────────────────


def argmax_center(img: np.ndarray) -> tuple[int, int]:
    """Frame global maximum — the true 0-order spot (AGENTS.md rule)."""
    cy, cx = np.unravel_index(np.argmax(img), img.shape)
    return int(cy), int(cx)


def intensity_centroid(img: np.ndarray) -> tuple[float, float]:
    """Raw intensity-weighted centroid, (row, col).

    NOTE: matches spots_calc.centroid semantics but returns (row, col) so all
    algorithms here share one coordinate convention.
    """
    total = float(img.sum())
    if total <= 0:
        return (float("nan"), float("nan"))
    rows = np.arange(img.shape[0], dtype=np.float64)[:, None]
    cols = np.arange(img.shape[1], dtype=np.float64)[None, :]
    cy = float((img.astype(np.float64) * rows).sum() / total)
    cx = float((img.astype(np.float64) * cols).sum() / total)
    return cy, cx


def debg_centroid(img: np.ndarray) -> tuple[float, float]:
    """Intensity centroid after median-background subtraction (floor at 0)."""
    bg = float(np.median(img))
    signal = np.where(img > bg, img.astype(np.float64) - bg, 0.0)
    return intensity_centroid(signal)


def corner_max_threshold(img: np.ndarray, block: int = 64) -> float:
    """Threshold = max gray of the four corner blocks (stray-light floor)."""
    h, w = img.shape
    corners = [
        img[:block, :block],
        img[:block, -block:],
        img[-block:, :block],
        img[-block:, -block:],
    ]
    return float(max(c.max() for c in corners))


def binary_centroid(img: np.ndarray, threshold: float) -> tuple[float, float]:
    """Centroid of the binary mask ``img > threshold``, (row, col)."""
    mask = img > threshold
    return intensity_centroid(mask.astype(np.float64))


# ── main ───────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture raw flat-phase spot TIFF + debug centroid algorithms"
    )
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--slm-wavelength", type=int, default=1064)
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--exposure-us", type=int, default=1200)
    parser.add_argument("--n-sample", type=int, default=3)
    parser.add_argument("--settle-time", type=float, default=0.3)
    parser.add_argument("--capture-timeout", type=float, default=30.0)
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("data/diff_beam/flat_tiff")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not HARDWARE_AVAILABLE:
        raise SystemExit("硬件驱动不可用 — 需要 SLM200 + 大恒相机")

    exp_ms = args.exposure_us / 1000.0
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("初始化 SLM #{} (lambda={}nm)…", args.slm_number, args.slm_wavelength)
    slm = SantecSLM200(slm_number=args.slm_number)
    call_with_timeout(slm.open, args.capture_timeout, "SLM open")
    slm.set_wavelength(args.slm_wavelength)

    camera = DahengCamManager(cam_id=args.cam_id, exposure_time_ms=exp_ms)
    call_with_timeout(camera.open, args.capture_timeout, "CCD open")
    camera.reset_exposure_time(exp_ms)
    logger.info("CCD 就绪: ID={} exposure={:.3f}ms", args.cam_id, exp_ms)

    try:
        last_slot = display_flat_phase(slm, args.settle_time, None)
        logger.info("平场相位已显示 (slot={})", last_slot)

        # Capture the raw frame. Single frame keeps it truly "raw" (no averaging
        # blur from beam drift), but n_sample>1 reduces read noise; both are saved.
        frame = call_with_timeout(
            lambda: camera.get_numpy_image(
                n_sample=args.n_sample, skip_first=True
            ),
            args.capture_timeout,
            "CCD 采集",
        )
        if frame is None:
            raise RuntimeError("CCD 采集返回空帧")
        frame = np.asarray(frame)

        # ── save TIFF + npy ──
        tiff_path = out_dir / f"flat_{args.exposure_us}us.tiff"
        npy_path = out_dir / f"flat_{args.exposure_us}us.npy"
        img16 = frame.astype(np.uint16)
        Image.fromarray(img16).save(tiff_path, format="TIFF")
        np.save(npy_path, img16)
        logger.info("已保存: {} ({}), {}", tiff_path, img16.shape, npy_path)

        # ── centroid comparison on the SAME frame ──
        logger.info("帧统计: shape={} dtype={} min={} max={} sum={:.0f}",
                    img16.shape, img16.dtype, int(img16.min()),
                    int(img16.max()), float(img16.sum()))
        bg = float(np.median(img16))
        nz_frac = float((img16 > bg).mean())
        logger.info("背景中位数={:.2f} 非背景像素占比={:.1%} frame_sum={:.0f}",
                    bg, nz_frac, float(img16.sum()))

        a_cy, a_cx = argmax_center(img16)
        raw_cy, raw_cx = intensity_centroid(img16)
        debg_cy, debg_cx = debg_centroid(img16)
        thr = corner_max_threshold(img16)
        bin_cy, bin_cx = binary_centroid(img16, thr)
        # spots_calc reference (runner's recorded "centroid"): (cx, cy) = (col, row)
        sc_cx, sc_cy = spots_centroid(img16)

        rows = [
            ("argmax (0级光斑)", f"({a_cy:.1f}, {a_cx:.1f})"),
            ("原始强度质心", f"({raw_cy:.1f}, {raw_cx:.1f})"),
            ("去背景强度质心", f"({debg_cy:.1f}, {debg_cx:.1f})"),
            (f"corner-max二值质心 (thr={thr:.1f})", f"({bin_cy:.1f}, {bin_cx:.1f})"),
            ("spots_calc.centroid (ref)", f"({sc_cy:.1f}, {sc_cx:.1f})"),
        ]
        logger.info("── 质心算法对比 (row, col) ──")
        for name, pos in rows:
            logger.info("  {:<28} {}", name, pos)

        # ── annotated comparison PNG ──
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            full = axes[0]
            im0 = full.imshow(img16, cmap="hot", aspect="equal")
            full.set_title(f"Full frame {img16.shape} (flat @ {exp_ms:.3f}ms)")
            fig.colorbar(im0, ax=full, fraction=0.046)

# zoomed ROI around argmax
            roi = 100
            y0, y1 = max(0, a_cy - roi), min(img16.shape[0], a_cy + roi)
            x0, x1 = max(0, a_cx - roi), min(img16.shape[1], a_cx + roi)
            zoom = axes[1]
            zoom.imshow(img16[y0:y1, x0:x1], cmap="hot", aspect="equal",
                        extent=[x0, x1, y1, y0])
            zoom.set_title(f"Zoomed {2 * roi}px around argmax ({a_cy}, {a_cx})")

            marks = [
                ("argmax", (a_cy, a_cx), "w", "o"),
                ("raw centroid", (raw_cy, raw_cx), "c", "x"),
                ("debg centroid", (debg_cy, debg_cx), "m", "+"),
                ("corner-max binary", (bin_cy, bin_cx), "y", "s"),
            ]
            for name, (cy, cx), color, mk in marks:
                for ax in axes:
                    ax.plot(cx, cy, marker=mk, color=color, markersize=8,
                            markeredgewidth=1.5, linestyle="None",
                            label=name if ax is axes[0] else None)
            axes[0].legend(loc="upper right")
            fig.suptitle(
                f"Centroid comparison — flat phase, {exp_ms:.3f}ms, "
                f"max={int(img16.max())} sum={float(img16.sum()):.0f}"
            )
            fig.tight_layout()
            png_path = out_dir / "centroid_comparison.png"
            fig.savefig(png_path, dpi=150)
            plt.close(fig)
            logger.info("已保存对比图: {}", png_path)
        except Exception as exc:  # plotting is best-effort
            logger.warning("对比图生成失败: {}", exc)

    finally:
        for name, dev in (("SLM", slm), ("CCD", camera)):
            try:
                call_with_timeout(dev.close, 10.0, f"{name} close")
            except Exception as exc:
                logger.warning("{} 关闭失败: {}", name, exc)


if __name__ == "__main__":
    main()