"""Dual-algorithm differentiable beam-shaping runner.

Lets the user pick, via ``--algorithm {backprop,gs}``, between two
algorithms that share the *same* SLM -> CCD closed-loop plumbing and the
*same* metric definitions (so their results are directly comparable):

- ``backprop``: a PyTorch autograd loop that treats the SLM phase ``φ`` as a
  learnable tensor and backpropagates a far-field intensity loss through a
  differentiable FFT far-field model (see
  ``ao_shaping.algorithm.differentiable_beam``).

- ``gs``: the alternating-projection Gerchberg-Saxton closed loop
  (``ao_shaping.algorithm.gerchberg_saxton``), optionally with CCD feedback.

Both algorithms produce an SLM phase pattern; in ``--use-hardware`` mode the
final phase is displayed on the SLM and the far field is captured with a CCD.

Square shaping is supported end-to-end: ``--target-shape square`` with
``--target-size`` (SLM-grid FFT bins), ``--target-px`` (camera pixels, manual
hardware override) or ``--target-brightness`` (硬件亮度自动模式：基于当前
实测质心 + 亮度总和自动定位/定大小，默认平均亮度 = 实测最大亮度的 1/10).
The target grid is the full SLM200 panel (1200, 1920).

Hardware runs record every CCD frame (``frames/`` + ``frame_meta.jsonl``).

Usage:
    python src/ao_shaping/main.py diff-beam --algorithm backprop --epochs 200
    python src/ao_shaping/main.py diff-beam --algorithm gs --iterations 50
    python src/ao_shaping/main.py diff-beam --algorithm backprop --use-hardware
    python src/ao_shaping/main.py diff-beam --algorithm backprop --target-shape square --target-size 40 --use-hardware -e 200 --cam-exposure-us 900
    python src/ao_shaping/main.py diff-beam --algorithm gs --target-shape square --use-hardware -i 50 --cam-exposure-us 900 --adaptive
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

# Ensure src is in path when running directly
if __name__ == "__main__":
    _script_dir = Path(__file__).resolve().parent
    _src_root = _script_dir.parent
    if str(_src_root) not in sys.path:
        sys.path.insert(0, str(_src_root))

import click
import numpy as np
from loguru import logger

# Import shared beam-shaping helpers (single source of truth for metrics/IO)
from ao_shaping.utils.beam_metrics import (
    compute_metrics,
    intensity_to_amplitude,
)
from ao_shaping.utils.slm_utils import phase_to_slm_grayscale
from ao_shaping.utils.targets import (
    create_target_shape,
    crop_resize_to_grid,
    load_target_image,
    square_target_from_measurement,
)

# Import the two algorithms
from ao_shaping.optimizer import differentiable_beam_optimize
from ao_shaping.algorithm.gerchberg_saxton import (
    adaptive_gerchberg_saxton,
    gerchberg_saxton,
)
from ao_shaping.utils.cli_helpers import get_debug_mode

# Signal centroid for frame-by-frame recording (uses the same definition as
# the beam-shaping utilities so recorded centroids are comparable with them).
from ao_shaping.utils.spots_calc import centroid

get_debug_mode()

# Import hardware drivers with graceful fallback
SantecSLM200: Any = None
DahengCamManager: Any = None
SLM_AVAILABLE = False
CCD_AVAILABLE = False

try:
    from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200 as _SantecSLM200

    SantecSLM200 = _SantecSLM200
    SLM_AVAILABLE = True
except ImportError:
    logger.debug("SLM driver not available")

try:
    from ao_shaping.drivers.ccd.daheng import DahengCamManager as _DahengCamManager

    DahengCamManager = _DahengCamManager
    CCD_AVAILABLE = True
except ImportError:
    logger.debug("CCD driver not available")


# SLM memory-slot rotation (2~125, never reuse the currently displayed slot).
# The Santec firmware treats display_memory(same slot) as a no-op, so
# consecutive writes must land on different slots.
_SLOT_MIN, _SLOT_MAX = 2, 125
_last_slot_used: int | None = None


def _pick_slm_slot(slm: Any) -> int:
    """Pick a random SLM memory slot in [2, 125] different from the last used.

    On first use the currently displayed slot is read from the device so the
    rotation also survives process restarts (memory mode only).
    """
    global _last_slot_used
    if _last_slot_used is None:
        try:
            _last_slot_used = slm.get_displayed_memory_number()
            logger.info("SLM当前显示槽: {}", _last_slot_used)
        except Exception as exc:
            logger.debug("读取 SLM 当前显示槽失败: {}", exc)
            _last_slot_used = None
    candidates = [s for s in range(_SLOT_MIN, _SLOT_MAX + 1) if s != _last_slot_used]
    slot = random.choice(candidates)
    _last_slot_used = slot
    return slot


def _display_phase(slm: Any, phase_rad: np.ndarray, settle_time_s: float) -> None:
    """Display a radian phase pattern on the SLM using memory-slot rotation.

    Uses ``create_phase_from_array`` (device-authentic 2π conversion +
    correction/LUT) and the memory mode only (never DVI).
    """
    gray = slm.create_phase_from_array(phase_rad)
    slot = _pick_slm_slot(slm)
    slm.write_phase(gray, memory_number=slot)
    time.sleep(0.05)
    slm.display_memory(slot)
    time.sleep(settle_time_s)


def _call_with_timeout(fn: Any, timeout_s: float, desc: str) -> Any:
    """Run ``fn`` in a daemon thread with a watchdog timeout.

    Hardware SDK calls (SLM open, CCD capture) can hang forever with no
    Python-visible timeout; this helper bounds the wait and raises
    ``TimeoutError`` if the call does not return in time.
    """
    result: list[Any] = []
    error: list[BaseException] = []

    def _runner() -> None:
        try:
            result.append(fn())
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


# ---------------------------------------------------------------------------
# Frame-by-frame recording (hardware runs only): every CCD frame is saved to
# ``frames/frame_NNNNN.npy`` with a JSONL meta line for offline analysis.
# ---------------------------------------------------------------------------
_frames_dir: Path | None = None
_frame_counter: int = 0


def _init_frame_recording(out_dir: Path) -> None:
    """Create the ``frames/`` directory under ``out_dir`` and reset the counter."""
    global _frames_dir, _frame_counter
    _frames_dir = Path(out_dir) / "frames"
    _frames_dir.mkdir(parents=True, exist_ok=True)
    _frame_counter = 0


def _save_frame_png(frame: np.ndarray, path: Path, title: str) -> None:
    """Render one CCD frame to a PNG (inferno colormap + colorbar).

    Cheap enough for the hardware loop (matplotlib import is cached after the
    first call); each PNG lands next to its .npy in ``frames/``.
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


def _record_frame(raw: np.ndarray, phase_desc: str, exposure_ms: float) -> dict:
    """Save one raw CCD frame plus a JSONL meta line; log per-frame stats.

    Also returns the meta dict so callers can reuse the recorded stats
    (e.g. the brightness-driven target builder logs the same frame).
    """
    global _frame_counter
    _frame_counter += 1
    idx = _frame_counter
    frame = np.asarray(raw, dtype=np.float32)
    peak = float(frame.max()) if frame.size else 0.0
    total = float(frame.sum()) if frame.size else 0.0
    if frame.size:
        sy, sx = np.unravel_index(np.argmax(frame), frame.shape)
        cx, cy = centroid(frame, return_float=True)
    else:
        sy = sx = 0
        cx = cy = 0.0
    meta = {
        "frame": idx,
        "phase": phase_desc,
        "exposure_ms": exposure_ms,
        "peak": peak,
        "sum": total,
        # True 0-order spot location (AGENTS.md rule: locate by argmax).
        # The intensity centroid is NOT the spot on this bench — a pervasive
        # stray-light halo drags it 150-450 px off (measured 2026-09-10).
        "spot": [int(sy), int(sx)],
        "centroid": [float(cy), float(cx)],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    frames_dir = _frames_dir
    if frames_dir is not None:
        np.save(frames_dir / f"frame_{idx:05d}.npy", np.asarray(raw))
        _save_frame_png(
            np.asarray(raw),
            frames_dir / f"frame_{idx:05d}.png",
            f"frame {idx:04d} - {phase_desc} (peak={peak:.0f})",
        )
        with open(frames_dir / "frame_meta.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    logger.info(
        "帧 {:04d}: phase={} peak={:.1f} sum={:.0f} spot=({}, {}) centroid=({:.1f}, {:.1f})",
        idx,
        phase_desc,
        peak,
        total,
        sy,
        sx,
        cy,
        cx,
    )
    return meta


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
    """Compute the next exposure (ms) that drives ``peak`` into target±tol.

    Pure function — no hardware access. The caller applies the result via
    ``camera.reset_exposure_time()`` (Daheng clamps to its own [min, max] and
    reports the actual value).

    - Already inside ``[target*(1-tol), target*(1+tol)]`` → unchanged.
    - ``peak >= sat_floor`` (hard saturation guard for 8-bit CCD) → scale down
      by ``target/peak`` regardless of step caps.
    - ``peak`` too low → boost by ``target/peak``, capped at ``max_boost``.
    - ``peak`` too high → cut by ``target/peak``, floored at ``max_cut``.

    Returns:
        Suggested exposure time in ms, clamped to ``[min_ms, max_ms]``.
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


def _auto_exposure_possible(camera: Any) -> bool:
    """True if the camera object supports runtime exposure adjustment."""
    return callable(getattr(camera, "reset_exposure_time", None))


def _apply_auto_exposure(
    camera: Any,
    peak: float,
    current_ms: float,
    target_brightness: float,
    tol: float,
) -> tuple[float, bool]:
    """Adjust camera exposure toward the target peak band (匀化过程自动曝光).

    Applies ``auto_exposure_target_ms`` via ``camera.reset_exposure_time`` and
    returns the camera-reported actual exposure plus whether it changed.

    Returns:
        (actual_exposure_ms, changed)
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


def _parse_tuple(ctx, param, value: str | None) -> tuple[int, int] | None:
    """Parse a Click ``"x,y"`` option into a 2-tuple."""
    if value is None:
        return None
    s_clean = re.sub(r"[()\s]", "", str(value))
    try:
        parts = s_clean.split(",")
        if len(parts) != 2:
            raise ValueError("Must have exactly two integers")
        x, y = map(int, parts)
        return (x, y)
    except Exception:
        raise click.BadParameter(f"Invalid format: {value}. Expected: 'x,y' or '(x,y)'")


@click.command(name="diff-beam")
@click.option(
    "--algorithm",
    type=click.Choice(["backprop", "gs"]),
    default="backprop",
    show_default=True,
    help="优化算法: backprop (PyTorch 反向传播) 或 gs (Gerchberg-Saxton 闭环)",
)
@click.option(
    "--target-image",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="目标图像路径 (灰度图 / .npy，归一化后作为目标强度)",
)
@click.option(
    "--target-shape",
    type=click.Choice(["gaussian", "circle", "square"]),
    default="gaussian",
    show_default=True,
    help="预设目标形状 (当未指定 --target-image 时使用)",
)
@click.option(
    "--target-size",
    type=int,
    default=40,
    show_default=True,
    help="方形目标边长 (SLM 网格 FFT bin)",
)
@click.option(
    "--target-px",
    type=int,
    default=None,
    help=(
        "方形目标在相机上的固定边长 (CCD 像素; 硬件模式, 方形)。"
        "目标方形 = CCD 图片空间: 边长 --target-px、中心 = 实测质心、"
        "亮度 = 1/边长² (总和=1)。loss 用实测帧/总亮度 归一化后对比, "
        "曝光/总亮度变化不影响 target。默认: 由 --target-size 换算"
    ),
)
@click.option(
    "-e",
    "--epochs",
    default=200,
    show_default=True,
    help="backprop 算法的 Adam 优化步数",
)
@click.option(
    "--lr",
    default=0.01,
    show_default=True,
    help="backprop 算法的 Adam 学习率",
)
@click.option(
    "-i",
    "--iterations",
    default=50,
    show_default=True,
    help="gs 算法的迭代次数",
)
@click.option(
    "-d",
    "--distance",
    default=0.1,
    show_default=True,
    help="传播距离 (米) (gs 算法使用)",
)
@click.option(
    "-l",
    "--wavelength",
    default=1064.0,
    show_default=True,
    help="激光波长 (纳米)",
)
@click.option(
    "--slm-wavelength",
    default=1064,
    show_default=True,
    help="SLM 工作波长 (纳米，用于设置 SLM)",
)
@click.option(
    "--slm-number",
    default=1,
    show_default=True,
    help="SLM 设备编号",
)
@click.option(
    "--cam-id",
    default=lambda: os.environ.get("FAR_CAM_ID", "0"),
    show_default="FAR_CAM_ID/0",
    help="CCD 相机 ID",
)
@click.option(
    "--cam-center",
    callback=_parse_tuple,
    default=None,
    help="CCD 中心位置 'x,y' (default: 自动检测)",
)
@click.option(
    "--cam-size",
    default=400,
    show_default=True,
    help="CCD 开窗大小 (像素)",
)
@click.option(
    "--cam-exposure",
    default=50.0,
    show_default=True,
    help="CCD 曝光时间 (毫秒)",
)
@click.option(
    "--cam-exposure-us",
    type=float,
    default=None,
    show_default=True,
    help="CCD 曝光时间 (微秒; 900=0.9ms; 与 --cam-exposure 二选一, 优先)",
)
@click.option(
    "--auto-exposure/--no-auto-exposure",
    "auto_exposure",
    default=False,
    show_default=True,
    help="匀化过程中自动调整相机曝光 (目标峰值带内; 信号弱增曝光/饱和减曝光)",
)
@click.option(
    "--auto-exposure-target",
    type=float,
    default=180.0,
    show_default=True,
    help="自动曝光目标峰值 (8-bit 相机计数; 默认: 180)",
)
@click.option(
    "--auto-exposure-tol",
    type=float,
    default=0.2,
    show_default=True,
    help="自动曝光目标带宽容差 (默认: 0.2 → 峰值 [144, 216])",
)
@click.option(
    "--p-cam",
    type=float,
    default=5.5e-6,
    show_default=True,
    help="相机像素间距 (米)",
)
@click.option(
    "--settle-time",
    type=float,
    default=0.3,
    show_default=True,
    help="SLM 显示相位后等待时间 (秒)",
)
@click.option(
    "--capture-timeout",
    type=float,
    default=30.0,
    show_default=True,
    help="SLM 打开与相机采集的看门狗超时 (秒)",
)
@click.option(
    "--adaptive",
    is_flag=True,
    help="gs 算法启用 CCD 反馈自适应 (需要 --use-hardware)",
)
@click.option(
    "--adaptive-iterations",
    default=3,
    show_default=True,
    help="gs 自适应外层迭代次数",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda"]),
    default=None,
    help="backprop 计算设备 (default: 自动选择)",
)
@click.option(
    "--seed",
    type=int,
    default=0,
    show_default=True,
    help="backprop 初始相位随机种子 (用于可复现)",
)
@click.option(
    "-s",
    "--save-dir",
    default="data/diff_beam",
    show_default=True,
    help="结果保存目录",
)
@click.option(
    "--use-hardware",
    is_flag=True,
    help="使用实际硬件 (SLM+CCD)，否则仅模拟计算",
)
@click.option(
    "--show",
    is_flag=True,
    help="显示结果图像",
)
def run(
    algorithm: str,
    target_image: Path | None,
    target_shape: str,
    target_size: int,
    target_px: int | None,
    epochs: int,
    lr: float,
    iterations: int,
    distance: float,
    wavelength: float,
    slm_wavelength: int,
    slm_number: int,
    cam_id: str,
    cam_center: tuple[int, int] | None,
    cam_size: int,
    cam_exposure: float,
    cam_exposure_us: float | None,
    auto_exposure: bool,
    auto_exposure_target: float,
    auto_exposure_tol: float,
    p_cam: float,
    settle_time: float,
    capture_timeout: float,
    adaptive: bool,
    adaptive_iterations: int,
    device: str | None,
    seed: int,
    save_dir: str,
    use_hardware: bool,
    show: bool,
) -> None:
    """Dual-algorithm differentiable beam shaping.

    通过 ``--algorithm {backprop,gs}`` 选择优化算法。两种算法共享相同的
    SLM->CCD 闭环硬件路径和指标定义，因此结果可直接对比。
    """
    

    # Wavelength in meters (gs angular spectrum uses SI units).
    wavelength_m = wavelength * 1e-9

    # SLM200 native panel resolution (width, height).
    slm_resolution = (1920, 1200)
    grid_h, grid_w = slm_resolution[1], slm_resolution[0]  # (1200, 1920)

    # Effective CCD exposure: --cam-exposure-us takes priority over --cam-exposure.
    if cam_exposure_us is not None:
        effective_exposure_ms = cam_exposure_us / 1000.0
    else:
        effective_exposure_ms = cam_exposure
    logger.info(
        "CCD exposure: {:.0f} us ({:.3f} ms)",
        effective_exposure_ms * 1000.0,
        effective_exposure_ms,
    )
    current_exposure_ms = effective_exposure_ms
    if auto_exposure:
        logger.info(
            "自动曝光开启: 目标峰值 {} ± {} -> [{:.0f}, {:.0f}]",
            auto_exposure_target,
            auto_exposure_tol,
            auto_exposure_target * (1.0 - auto_exposure_tol),
            auto_exposure_target * (1.0 + auto_exposure_tol),
        )

    logger.info("=" * 60)
    logger.info("Dual-Algorithm Differentiable Beam Shaping")
    logger.info("=" * 60)
    logger.info("Algorithm: {}", algorithm)
    logger.info("Target: {}", target_image or target_shape)
    logger.info("Wavelength: {:.0f} nm", wavelength)
    logger.info("Hardware: {}", "enabled" if use_hardware else "disabled")

    # Square target sizing (SLM-grid FFT bins) and bin<->camera-px mapping.
    # Hardware square mode uses a FIXED side in CCD image space: the target
    # square is side `side_ccd_px` camera pixels, centered at the measured beam
    # centroid, with value 1/side**2 (sum == 1). The loss is computed against
    # the normalized measured frame (frame / total), so exposure/brightness
    # changes never alter the target square.
    target_size_bins: int = min(target_size, grid_h)
    cam_px_per_bin_h: float | None = None
    cam_px_per_bin_v: float | None = None
    use_measured_sizing = False
    side_ccd_px: float | None = None

    if target_shape == "square":
        d_slm = 8e-6  # SLM pixel pitch (m)
        cam_px_per_bin_v = wavelength_m * distance / (grid_h * d_slm * p_cam)
        cam_px_per_bin_h = wavelength_m * distance / (grid_w * d_slm * p_cam)
        if use_hardware:
            if target_px is not None:
                side_ccd_px = float(target_px)
                target_size_bins = max(
                    1, min(grid_h, int(round(side_ccd_px / cam_px_per_bin_v)))
                )
                logger.info(
                    "Square target: --target-px {} CCD px (固定边长, 质心定位) "
                    "≈ {} grid bins",
                    round(side_ccd_px),
                    target_size_bins,
                )
            else:
                # Fallback: convert the grid-bin side to camera pixels and warn
                # so the user can pin the real CCD-space side explicitly.
                side_ccd_px = float(target_size) * cam_px_per_bin_v
                target_size_bins = min(target_size, grid_h)
                logger.warning(
                    "未指定 --target-px; 由 --target-size {} grid bins 换算 "
                    "边长 ≈ {:.1f} CCD px (推荐显式 --target-px 固定 CCD 边长)",
                    target_size,
                    side_ccd_px,
                )
            use_measured_sizing = True
            logger.info(
                "硬件方形: target 在 CCD 图片空间 (边长 {:.1f} px, 质心定位, "
                "值=1/边长²), loss 用实测帧/总亮度 归一化对比",
                side_ccd_px,
            )
        elif target_px is not None:
            logger.warning(
                "--target-px 仅在 --use-hardware 模式下有效; 回退到 --target-size"
            )
            target_size_bins = min(target_size, grid_h)
        else:
            target_size_bins = min(target_size, grid_h)
        if not use_measured_sizing:
            logger.info(
                "Square target: {} grid bins ≈ {:.0f}x{:.0f} CCD px "
                "(cam_px_per_bin h={:.2f} v={:.2f})",
                target_size_bins,
                target_size_bins * cam_px_per_bin_v,
                target_size_bins * cam_px_per_bin_h,
                cam_px_per_bin_h,
                cam_px_per_bin_v,
            )
    elif target_px is not None:
        logger.warning("--target-px 仅对 --target-shape square 有效; 已忽略")

    # Prepare the target intensity pattern (shared between both algorithms).
    # In measured-sizing (hardware square) mode the target is built from the
    # initial hardware measurement inside the hardware block below; keep a
    # placeholder here.
    target_info: dict | None = None
    if use_measured_sizing:
        target_intensity: np.ndarray | None = None
    elif target_image is not None:
        target_intensity = load_target_image(target_image)
    else:
        target_intensity = create_target_shape(
            cast(Literal["gaussian", "circle", "square"], target_shape),
            (grid_h, grid_w),
            side=target_size_bins,
        )

    # Uniform illumination at the SLM plane, shaped to the target footprint.
    # Assigned once target_intensity is final (after the measured initial
    # measurement in hardware mode).
    source_amplitude: np.ndarray | None = None

    # Hardware initialization.
    slm: Any = None
    camera: Any = None
    captured_frame: np.ndarray | None = None

    phase: np.ndarray
    loss_history: list[float]
    final_loss: float
    converged: bool
    steps: int
    run_device: str = device or "cpu"

    # The output directory is created up-front so that frame-by-frame
    # recordings can be written while the hardware loop is running.
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_label = target_image.stem if target_image else target_shape
    result_dir = save_path / f"{algorithm}_{target_label}_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)
    _init_frame_recording(result_dir)

    try:
        if use_hardware:
            if not SLM_AVAILABLE:
                raise RuntimeError("SLM driver not available. Install Santec SLM SDK.")
            if not CCD_AVAILABLE:
                raise RuntimeError("CCD driver not available. Install Daheng SDK.")

            logger.info("Initializing hardware...")
            slm = SantecSLM200(slm_number=slm_number, wavelength=slm_wavelength)
            _call_with_timeout(slm.open, capture_timeout, "SLM open")
            logger.info("SLM initialized: #{}, lambda={}nm", slm_number, slm_wavelength)

            cam_id_int = int(cam_id)
            camera = DahengCamManager(
                cam_id=cam_id_int, exposure_time_ms=effective_exposure_ms
            )
            _call_with_timeout(camera.open, capture_timeout, "CCD open")
            if cam_center is not None:
                camera.reset_window(center=cam_center, size=(cam_size, cam_size))
            # Force the exposure (re-applies without restart on Daheng).
            camera.reset_exposure_time(effective_exposure_ms)
            logger.info(
                "CCD initialized: ID={}, exposure={:.3f}ms",
                cam_id_int,
                effective_exposure_ms,
            )

            # Fixed-side square target (CCD image space): display the flat phase
            # once and measure the current beam centroid, then place a square
            # of fixed side `side_ccd_px` camera pixels centered on the measured
            # beam. Square value = 1/side**2 (sum == 1). The loss is evaluated
            # against the normalized measured frame (frame / total brightness),
            # so exposure/total-brightness changes never alter the target.
            if use_measured_sizing:
                logger.info("测量当前光束 (平场相位) 以定位方形中心 (质心)…")
                _display_phase(
                    slm, np.zeros((grid_h, grid_w), dtype=np.float32), settle_time
                )
                # Auto-exposure probe: one frame to judge the current signal,
                # adjust exposure, then take the *official* baseline frame used
                # for brightness-driven square sizing/positioning.
                if auto_exposure and _auto_exposure_possible(camera):
                    probe = _call_with_timeout(
                        lambda: camera.get_numpy_image(n_sample=1, skip_first=True),
                        capture_timeout,
                        "CCD 曝光探测",
                    )
                    probe_peak = float(np.asarray(probe).max())
                    current_exposure_ms, _ = _apply_auto_exposure(
                        camera,
                        probe_peak,
                        current_exposure_ms,
                        auto_exposure_target,
                        auto_exposure_tol,
                    )
                    logger.info(
                        "初始帧曝光调整: peak={:.1f} -> {} ms (继续正式采集)",
                        probe_peak,
                        current_exposure_ms,
                    )
                initial_frame = _call_with_timeout(
                    lambda: camera.get_numpy_image(n_sample=5, skip_first=True),
                    capture_timeout,
                    "CCD 初始采集",
                )
                _record_frame(initial_frame, "flat_initial", current_exposure_ms)
                target_intensity, target_info = square_target_from_measurement(
                    np.asarray(initial_frame),
                    side_ccd_px if side_ccd_px is not None else float(target_size_bins),
                    grid_h,
                    grid_w,
                )
                _save_frame_png(
                    np.asarray(target_intensity),
                    result_dir / "target_square.png",
                    f"fixed-side square target (grid, side={target_info['side_cam_px']:.0f} CCD px)",
                )
                side_grid = int(round(target_info["side_grid_bins"][0]))
                target_size_bins = max(1, min(grid_h, side_grid))
                logger.info(
                    "固定边长方形 target: side={:.1f} CCD px (1/{} 归一化) "
                    "≈ {} grid bins 质心=({:.1f}, {:.1f}) 亮度总和={:.0f} "
                    "(背景={:.1f} 峰值={:.1f})",
                    target_info["side_cam_px"],
                    target_info["n_pixels"],
                    target_size_bins,
                    target_info["centroid"][0],
                    target_info["centroid"][1],
                    target_info["total_intensity"],
                    target_info["background"],
                    target_info["max_brightness"],
                )

        if target_intensity is None:
            raise RuntimeError("目标强度未构建 (硬件方形模式缺少初始测量)")
        source_amplitude = np.ones(target_intensity.shape, dtype=np.float32)

        if algorithm == "backprop":
            logger.info(
                "Running backprop optimization: epochs={}, lr={}, device={}",
                epochs,
                lr,
                run_device,
            )
            result = differentiable_beam_optimize(
                target_intensity=target_intensity,
                source_amplitude=source_amplitude,
                lr=lr,
                epochs=epochs,
                device=device,
                seed=seed,
            )
            phase = result.phase
            loss_history = result.loss_history
            final_loss = result.final_loss
            converged = result.converged
            steps = result.steps
            run_device = result.device
        else:  # gs
            logger.info(
                "Running Gerchberg-Saxton: iterations={}, adaptive={}",
                iterations,
                adaptive,
            )

            target_amp = intensity_to_amplitude(target_intensity) * source_amplitude

            if adaptive:
                # CCD-feedback adaptive GS: display each candidate phase on the
                # SLM and capture the real far field with the CCD to drive
                # closed-loop refinement.
                if not (use_hardware and slm is not None and camera is not None):
                    logger.warning(
                        "--adaptive requires --use-hardware with SLM+CCD; "
                        "falling back to plain GS."
                    )
                    gs_result = gerchberg_saxton(
                        source_amplitude=source_amplitude,
                        target_amplitude=target_amp,
                        iterations=iterations,
                        distance=distance,
                        wavelength=wavelength_m,
                    )
                else:

                    def _capture_amplitude(phase_local: np.ndarray) -> np.ndarray:
                        _display_phase(slm, phase_local, settle_time)
                        nonlocal current_exposure_ms
                        img = _call_with_timeout(
                            lambda: camera.get_numpy_image(n_sample=3, skip_first=True),
                            capture_timeout,
                            "CCD 采集",
                        )
                        # Auto-exposure: drives the next capture into the target
                        # peak band (the amplitude feedback is normalized, so an
                        # exposure change only improves SNR — never skews the
                        # normalized loop signal).
                        if auto_exposure and _auto_exposure_possible(camera):
                            current_exposure_ms, _ = _apply_auto_exposure(
                                camera,
                                float(np.asarray(img).max()),
                                current_exposure_ms,
                                auto_exposure_target,
                                auto_exposure_tol,
                            )
                        _record_frame(img, "adaptive", current_exposure_ms)
                        resized = crop_resize_to_grid(img)
                        amp = np.sqrt(resized)
                        amax = amp.max()
                        return amp / amax if amax > 0 else amp

                    gs_result = adaptive_gerchberg_saxton(
                        source_amplitude=source_amplitude,
                        target_amplitude=target_amp,
                        measured_amplitude_callback=_capture_amplitude,
                        outer_iterations=adaptive_iterations,
                        inner_iterations=iterations,
                        distance=distance,
                        wavelength=wavelength_m,
                    )
            else:
                gs_result = gerchberg_saxton(
                    source_amplitude=source_amplitude,
                    target_amplitude=target_amp,
                    iterations=iterations,
                    distance=distance,
                    wavelength=wavelength_m,
                )

            phase = gs_result.phase
            loss_history = list(gs_result.error_history)
            final_loss = loss_history[-1] if loss_history else float("nan")
            converged = False
            steps = len(loss_history)
            run_device = "numpy"

        # Display the final phase on the SLM for both algorithms (hardware mode).
        if use_hardware and slm is not None:
            logger.info("Displaying phase pattern on SLM...")
            _display_phase(slm, phase, settle_time)

        # Measured capture: display the final phase again (fresh slot) and grab
        # the REAL CCD image. This must happen while the devices are still open
        # (the finally block below closes them).
        if use_hardware and camera is not None:
            logger.info("Capturing measured far field...")
            _display_phase(slm, phase, settle_time)
            captured_frame = _call_with_timeout(
                lambda: camera.get_numpy_image(n_sample=3, skip_first=True),
                capture_timeout,
                "CCD 采集",
            )
            if captured_frame is not None:
                # Auto-exposure: if the final frame is out of band, re-tune the
                # exposure and re-capture once so the recorded final frame is
                # neither saturated nor buried in noise.
                if auto_exposure and _auto_exposure_possible(camera):
                    new_ms, changed = _apply_auto_exposure(
                        camera,
                        float(np.asarray(captured_frame).max()),
                        current_exposure_ms,
                        auto_exposure_target,
                        auto_exposure_tol,
                    )
                    if changed:
                        current_exposure_ms = new_ms
                        logger.info(
                            "final 帧曝光重调: {} ms, 重新采集…", current_exposure_ms
                        )
                        retry = _call_with_timeout(
                            lambda: camera.get_numpy_image(n_sample=3, skip_first=True),
                            capture_timeout,
                            "CCD 采集",
                        )
                        if retry is not None:
                            captured_frame = retry
                if captured_frame is not None:  # narrowing re-check after re-capture
                    _record_frame(captured_frame, "final", current_exposure_ms)

    finally:
        # Cleanup hardware with exception isolation: one failing close must
        # never prevent the other device from being closed.
        if slm is not None:
            try:
                _call_with_timeout(slm.close, 10.0, "SLM close")
            except Exception as exc:
                logger.warning("SLM 关闭失败: {}", exc)
        if camera is not None:
            try:
                _call_with_timeout(camera.close, 10.0, "CCD close")
            except Exception as exc:
                logger.warning("CCD 关闭失败: {}", exc)

    # Capture the measured far field (hardware or simulated) for metrics.
    # Both branches below assign measured_intensity, so it is always bound.
    measured_intensity: np.ndarray
    measured_frame: np.ndarray | None = None
    square_footprint_px: tuple[int, int] | None = None

    if use_hardware and camera is not None and captured_frame is not None:
        # Measured capture path: process the REAL CCD image (the simulated FFT
        # far field is never used in hardware mode).
        frame = np.asarray(captured_frame, dtype=np.float32)
        frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
        measured_frame = frame.copy()

        crop_resized = crop_resize_to_grid(frame)
        cmax = crop_resized.max()
        measured_intensity = (crop_resized / cmax if cmax > 0 else crop_resized).astype(
            np.float32
        )

        # Measure the shaped-square footprint in camera pixels: threshold the
        # resized intensity at 0.5*max and scale the mask extents back to CCD px.
        mask = measured_intensity > 0.5 * measured_intensity.max()
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if rows.size and cols.size:
            n_rows = int(rows[-1] - rows[0] + 1)
            n_cols = int(cols[-1] - cols[0] + 1)
            frame_h, frame_w = frame.shape
            crop_h = min(frame_h, round(frame_w * grid_h / grid_w))
            crop_w = round(crop_h * grid_w / grid_h)
            footprint_h = n_rows * crop_h / grid_h
            footprint_w = n_cols * crop_w / grid_w
            square_footprint_px = (int(round(footprint_h)), int(round(footprint_w)))
            logger.info(
                "Measured square footprint on CCD: {}x{} px",
                square_footprint_px[0],
                square_footprint_px[1],
            )
        else:
            logger.warning("未检测到 0.5×max 以上的能量区域, 无法测量方形足迹")
    else:
        # Simulated far field via the FFT model, matching the backprop loss.
        import torch

        from ao_shaping.optimizer import far_field_intensity

        phase_t = torch.from_numpy(phase.astype(np.float32)).to("cpu")
        i_far = far_field_intensity(source_amplitude, phase_t)
        i_far_np = i_far.cpu().numpy()
        imax = i_far_np.max()
        measured_intensity = (i_far_np / imax if imax > 0 else i_far_np).astype(
            np.float32
        )

    # Compute shared metrics (backprop and gs are directly comparable).
    # Hardware measured-sizing (fixed-side square) mode compares in *CCD image
    # space*: measured = normalized intensity (raw frame / total brightness),
    # target = sum-1 square. Both sides are normalized, so exposure and total
    # brightness changes never alter the target square.
    if (
        use_hardware
        and camera is not None
        and captured_frame is not None
        and use_measured_sizing
        and target_info is not None
    ):
        frame_sum = float(frame.sum())
        if frame_sum > 0:
            measured_norm = np.asarray(frame / frame_sum, dtype=np.float32)
        else:
            measured_norm = np.asarray(frame, dtype=np.float32)
        metrics = compute_metrics(measured_norm, target_info["target_ccd"])
        logger.info(
            "CCD 空间评估 (曝光无关): 实测帧/总亮度 vs 边长 {:.0f} px 方形",
            target_info["side_cam_px"],
        )
    else:
        metrics = compute_metrics(measured_intensity, target_intensity)

    logger.info("-" * 60)
    logger.info("Results:")
    logger.info("  Algorithm: {} ({} steps, converged={})", algorithm, steps, converged)
    logger.info("  Final loss: {:.6f}", final_loss)
    logger.info("  MSE: {:.6f}", metrics["mse"])
    logger.info("  Correlation: {:.4f}", metrics["correlation"])
    logger.info("  Efficiency: {:.4f}", metrics["efficiency"])
    logger.info("-" * 60)

    # Save results. (result_dir/frames set up before the hardware loop ran.)
    # Seed a JSON-serializable copy of target_info: the full CCD-space square
    # (target_ccd) is saved as .npy below; config only carries the scalar
    # summary fields (the ndarray is not JSON-serializable).
    config = {
        "algorithm": algorithm,
        "target": str(target_image) if target_image else target_shape,
        "target_size": target_size_bins,
        "target_px": target_px,
        "target_info": (
            {
                k: v
                for k, v in target_info.items()
                if k != "target_ccd" and not isinstance(v, np.ndarray)
            }
            if use_measured_sizing and target_info is not None
            else None
        ),
        "cam_exposure_us": cam_exposure_us,
        "effective_exposure_ms": effective_exposure_ms,
        "auto_exposure": auto_exposure,
        "auto_exposure_target": auto_exposure_target,
        "auto_exposure_tol": auto_exposure_tol,
        "final_exposure_ms": current_exposure_ms,
        "p_cam": p_cam,
        "settle_time": settle_time,
        "capture_timeout": capture_timeout,
        "epochs": epochs,
        "lr": lr,
        "iterations": iterations,
        "distance_m": distance,
        "wavelength_nm": wavelength,
        "slm_wavelength_nm": slm_wavelength,
        "use_hardware": use_hardware,
        "adaptive": adaptive,
        "steps": steps,
        "converged": converged,
        "final_loss": final_loss,
        "metrics": metrics,
        "frames_recorded": _frame_counter,
        "cam_px_per_bin": (
            {"h": cam_px_per_bin_h, "v": cam_px_per_bin_v}
            if use_hardware and cam_px_per_bin_h is not None
            else None
        ),
        "square_footprint_px": square_footprint_px if use_hardware else None,
    }
    with open(result_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    slm_phase = phase_to_slm_grayscale(phase)
    np.save(result_dir / "phase_pattern.npy", slm_phase)
    np.save(result_dir / "target_intensity.npy", target_intensity)
    np.save(result_dir / "measured_intensity.npy", measured_intensity)
    np.save(result_dir / "loss_history.npy", np.array(loss_history))
    if target_info is not None and "target_ccd" in target_info:
        # CCD-image-space normalized target square (sum == 1): the reference
        # used for the exposure-invariant evaluation in hardware mode.
        np.save(result_dir / "target_ccd.npy", target_info["target_ccd"])
        _save_frame_png(
            target_info["target_ccd"],
            result_dir / "target_ccd.png",
            f"CCD-space target square (side={target_info['side_cam_px']:.0f} px, "
            "1/side² normalized)",
        )
    if measured_frame is not None:
        np.save(result_dir / "measured_frame.npy", measured_frame)

    logger.info("Results saved to: {}", result_dir)
    click.echo("Differentiable beam shaping complete!")
    click.echo(f"  Algorithm: {algorithm}")
    click.echo(f"  Steps: {steps} (converged={converged})")
    click.echo(f"  Final loss: {final_loss:.6f}")
    click.echo(f"  Correlation: {metrics['correlation']:.4f}")
    click.echo(f"  Efficiency: {metrics['efficiency']:.4f}")
    click.echo(f"  Results saved to: {result_dir}")


if __name__ == "__main__":
    run()
