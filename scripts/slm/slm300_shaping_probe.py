"""SLM-300 + MiiCam beam-shaping probe: verify the SLM drives correctly and
capture one far-field frame per distinct shaping pattern.

Goal (this round): hard-drive verification first -- prove the Santec panel
(SLM-300, memory mode) actually displays what we ask, and that
``display_and_average`` returns a sane frame per phase.

Each pattern is built as a raw radians phase array (panel-shaped, (1200, 1920));
``display_and_average`` runs its rad→gray pipeline internally (via
``slm.create_phase_from_array`` + ``slm.display_data``, which rotates the memory
slot itself -- no firmware no-op).

Layout conventions are shared with the rest of the repo:
- 0 order / peak is located by ``argmax`` over the full frame (2f Fourier bench,
  the optical axis is *not* assumed to sit at the camera centre).
- Flat phase is sent as an all-zero *radian* array to ``display_and_average``
  (never routed through a ``mod 2π``-style path that would corrupt it; the
  rad→gray converter handles 0 rad correctly, so flat is safe here).

Default is **plan-only**: no device is opened, nothing is written. Pass
``--execute`` to actually run the acquisition (SLM open first, then camera).

All device data goes to ``data/debug`` (per repo discipline).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ao_shaping.utils.slm.bench_constants import (
    SLM_PANEL_H,
    SLM_PANEL_W,
    BEAM_CENTER_PANEL,
    BEAM_RADIUS_PANEL,
)

# Panel-shaped radians target.
PANEL_SHAPE = (SLM_PANEL_H, SLM_PANEL_W)  # (1200, 1920)
_CX, _CY = BEAM_CENTER_PANEL  # nominal illumination centre (960, 600)
_R = float(BEAM_RADIUS_PANEL)  # nominal beam radius on the panel (450)

DEFAULT_EXPOSURE_MS = 20.0
DEFAULT_FRAMES = 3
DEFAULT_OUT_PREFIX = Path("data/debug/slm300_shaping_probe")


# ---------------------------------------------------------------------------
# Pattern constructors. Each builds a raw-unwrapped radian phase array
# (panel-shaped), and returns (label, phase_rad, short_description).
# ---------------------------------------------------------------------------
def _grid(shape: tuple[int, int]):
    """Return (y, x) coordinate grids in pixel units. Panel rows/cols."""
    hs, ws = shape
    yy, xx = np.mgrid[0:hs, 0:ws].astype(np.float64)
    return yy, xx


def flat_pattern() -> tuple[str, np.ndarray, str]:
    """All-zero phase -> plain 0 order (no shaping). Verifies the flat base line."""
    return "flat", np.zeros(PANEL_SHAPE, dtype=np.float64), "全 0 相位 (0 级基准)"


def defocus_pattern(a: float = 6.0) -> tuple[str, np.ndarray, str]:
    """Defocus 2r^2 - 1 -> ring/donut far field (annular intensity)."""
    yy, xx = _grid(PANEL_SHAPE)
    r2 = ((xx - _CX) ** 2 + (yy - _CY) ** 2) / _R**2
    phase = a * (2.0 * r2 - 1.0)
    return "defocus", phase, f"离焦 6.0 rad (甜甜圈形远场)"


def tilt_x_pattern(a: float = 6.0) -> tuple[str, np.ndarray, str]:
    """Linear phase ramp along x -> blazed grating, pushes 0 order to +1 order."""
    yy, xx = _grid(PANEL_SHAPE)
    phase = a * (xx - _CX) / _R
    return "tilt_x", phase, f"一维倾斜光栅 {a:.1f} rad (0 级转 +1 级)"


def square_pattern(a: float = 5.0, k: float = 20.0) -> tuple[str, np.ndarray, str]:
    """Near-field 2D square-wave (bidirectional phase grating) -> square far field.

    Zernike modes (n<=4) are smooth, circularly symmetric and cannot synthesize a
    square far field; we need the high-frequency 2D-sinc content a square-wave
    phase grating provides. Each sin/cos sign flips between 0 and a full swing of
    ``a`` radians, amplitude is well within the 10-bit code space.
    """
    yy, xx = _grid(PANEL_SHAPE)
    gx = np.sign(np.cos(2 * np.pi * k * (xx - _CX) / _R))
    gy = np.sign(np.cos(2 * np.pi * k * (yy - _CY) / _R))
    phase = a * (gx + gy) / 2.0  # in [-a, a]
    return "square", phase, f"双向方波相位光栅 k={k:.0f} (方形远场)"


def astig_pattern(a: float = 6.0) -> tuple[str, np.ndarray, str]:
    """Zernike astigmatism (2x^2 - 2y^2) -> splitting/fan of the PSF core."""
    yy, xx = _grid(PANEL_SHAPE)
    nx = (xx - _CX) / _R
    ny = (yy - _CY) / _R
    phase = a * (2.0 * nx**2 - 2.0 * ny**2)
    return "astig", phase, f"Zernike 像散 {a:.1f} rad (PSF 核心劈裂/扇形)"


def build_patterns(args) -> tuple[list[tuple[str, np.ndarray, str]], int]:
    """Build the pattern list (all radian arrays). Returns (patterns, n_phys)."""
    patterns = [
        ("flat", np.zeros(PANEL_SHAPE, dtype=np.float64), "全 0 相位 (0 级基准)"),
        defocus_pattern(args.defocus_amp),
        tilt_x_pattern(args.tilt_amp),
        square_pattern(args.square_amp, args.square_k),
        astig_pattern(args.astig_amp),
    ]
    return patterns, len(patterns)


# ---------------------------------------------------------------------------
# Plan-only reporting
# ---------------------------------------------------------------------------
def plan_only(patterns) -> int:
    """Print the pattern list without touching any hardware or filesystem."""
    print("Plan-only: SLM-300 shaping probe 将显示以下图案 (不占设备, 不写盘):")
    print(f"  面板尺寸 (px): {PANEL_SHAPE[0]} x {PANEL_SHAPE[1]}")
    print(f"  图案数: {len(patterns)}")
    for label, phase, desc in patterns:
        print(
            f"    - {label:<8} 相位峰值 {np.nanmax(np.abs(phase)):.2f} rad "
            f"尺寸 {phase.shape[0]}x{phase.shape[1]}  <= {desc}"
        )
    print("\n未传 --execute: 未打开 SLM/相机, 未创建任何输出目录。")
    print("如需真实采集: python scripts/slm/slm300_shaping_probe.py --execute")
    return 0


# ---------------------------------------------------------------------------
# Executive (hardware) path
# ---------------------------------------------------------------------------
def fresh_out_dir(base: Path) -> Path:
    """Create a new empty run dir ``<base>/<timestamp>`. Refuses collisions."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = base / ts
    if out.exists():
        # Pathological: same-second rerun. Bump with a suffix rather than clobber.
        i = 1
        while (base / f"{ts}_{i}").exists():
            i += 1
        out = base / f"{ts}_{i}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def _locate_peak(frame: np.ndarray) -> tuple[int, int]:
    """Locate 0 order / peak: global max in the full frame (2f bench)."""
    flat_idx = int(np.argmax(frame))
    return np.unravel_index(flat_idx, frame.shape)


def run_execute(args) -> int:
    """Open SLM (memory mode) then camera, display each pattern, save npz + summary."""
    from ao_shaping.drivers.slm.factory import (
        open_slm,
        open_camera,
        sync_device_wavelength,
        require_drive_board_entry,
    )
    from ao_shaping.tools.kernels.bench_kernels import display_and_average
    from types import SimpleNamespace

    # Duck-typed BenchDeviceParams carrier.
    params = SimpleNamespace(
        cam_type="miicam",
        cam_id=args.cam_id,
        exposure_time_ms=args.exposure_ms,
        cam_size=0,  # not used by miicam backend
        slm_number=args.slm_number,
        slm_wavelength=args.slm_wavelength,
        video_mode=0,  # Memory mode; never DVI.
    )

    patterns, _ = build_patterns(args)

    slm = None
    cam = None
    try:
        # SLM first (memory mode via from_params default), then camera.
        print(f"正在占用 SLM #{args.slm_number} (内存模式): open → ...")
        slm = open_slm(params)
        slm.open()

        # Adopt wavelength (0 = ask device) and guard drive board before any measure.
        sync_device_wavelength(params, slm)
        require_drive_board_entry(slm)

        print(f"正在占用 miicam (released cam #{args.cam_id}): open → ...")
        cam = open_camera(params)
        cam.open()
        exposure_ms = args.exposure_ms
        if exposure_ms <= 0:
            # Sentinel: resolve the camera's hardware minimum exposure.
            from ao_shaping.drivers.ccd.common import get_camera_exposure_range
            lo, _hi = get_camera_exposure_range(cam)
            exposure_ms = lo
            print(f"--exposure-ms 0 → 使用相机最低曝光 {exposure_ms:.6g} ms")
        cam.reset_exposure_time(exposure_ms)

        out = fresh_out_dir(Path(args.out))
        print(f"输出目录: {out}")

        summary = {
            "slm_number": args.slm_number,
            "slm_wavelength_resolved_nm": params.slm_wavelength,
            "cam_type": "miicam",
            "cam_id": args.cam_id,
            "exposure_ms": exposure_ms,
            "pixel_size_um": 2.9,
            "panel_shape": list(PANEL_SHAPE),
            "frames_per_pattern": args.frames,
            "patterns": [],
        }

        for label, phase, desc in patterns:
            print(f"-- [{label}] 显示并采集 ({args.frames} 帧平均) ...")
            frame = np.asarray(display_and_average(
                cam, slm, phase,
                n_frames=args.frames,
            ))
            peak_value = float(np.max(frame))
            spot_y, spot_x = _locate_peak(frame)
            mean_value = float(np.mean(frame))
            meta = {
                "label": label,
                "desc": desc,
                "phase_peak_rad": float(np.nanmax(np.abs(phase))),
                "slm_number": args.slm_number,
                "slm_wavelength_resolved_nm": params.slm_wavelength,
                "exposure_ms": exposure_ms,
                "pixel_size_um": 2.9,
                "peak_value": peak_value,
                "mean_value": mean_value,
                "spot_xy": [int(spot_x), int(spot_y)],
                "frames": args.frames,
                "frame_shape": list(frame.shape),
            }
            np.savez(out / f"{label}.npz", frame=frame, meta=meta)
            print(
                f"  [{label}] 峰值 {peak_value:.1f} mean {mean_value:.1f} "
                f"0级@({spot_x},{spot_y}) -> {label}.npz"
            )
            summary["patterns"].append(meta)

        with open(out / "probe_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"完成: 摘要已写入 {out / 'probe_summary.json'}")
        return 0
    except Exception as exc:  # noqa: BLE001 - report and let finally close
        print(f"执行失败: {exc!r}", file=sys.stderr)
        return 1
    finally:
        # Devices are closed in finally even on error/kill-adjacent unwinds.
        if cam is not None:
            try:
                cam.close()
                print("相机已关闭 (miicam released).")
            except Exception as exc:  # noqa: BLE001
                print(f"相机关闭失败: {exc!r}", file=sys.stderr)
        if slm is not None:
            try:
                slm.close()
                print(f"SLM #{args.slm_number} 已关闭 (释放).")
            except Exception as exc:  # noqa: BLE001
                print(f"SLM 关闭失败: {exc!r}", file=sys.stderr)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "SLM-300 + MiiCam beam-shaping probe: verify the SLM drives and "
            "capture one far-field frame per shaping pattern. Plan-only by "
            "default; pass --execute to acquire."
        )
    )
    ap.add_argument("--slm-number", type=int, default=1, help="SLM 编号 (默认 1)")
    ap.add_argument(
        "--slm-wavelength", type=int, default=0,
        help="SLM 波长 nm (默认 0 = 询问设备实际编程波长)",
    )
    ap.add_argument("--cam-id", type=int, default=0, help="miicam 相机 id (默认 0)")
    ap.add_argument(
        "--exposure-ms", type=float, default=DEFAULT_EXPOSURE_MS,
        help=f"相机曝光 ms (默认 {DEFAULT_EXPOSURE_MS}; 传 0 = 使用相机硬件最低曝光)",
    )
    ap.add_argument(
        "--out", type=str,
        default=None,
        help="输出目录 (默认 data/debug/slm300_shaping_probe/<timestamp>, 每次新 run)",
    )
    ap.add_argument(
        "--frames", type=int, default=DEFAULT_FRAMES,
        help=f"每形状采集平均帧数 (默认 {DEFAULT_FRAMES})",
    )
    ap.add_argument(
        "--execute", action="store_true",
        help="真正打开 SLM + 相机采集；否则只打印图案清单",
    )
    ap.add_argument("--defocus-amp", type=float, default=6.0, help="defocus 幅度 rad")
    ap.add_argument("--tilt-amp", type=float, default=6.0, help="tilt_x 幅度 rad")
    ap.add_argument("--square-amp", type=float, default=5.0, help="square 光栅幅度 rad")
    ap.add_argument("--square-k", type=float, default=20.0, help="square 光栅空间频率 (无量纲)")
    ap.add_argument("--astig-amp", type=float, default=6.0, help="astig 幅度 rad")
    args = ap.parse_args(argv)
    if args.out is None:
        args.out = str(DEFAULT_OUT_PREFIX)
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    patterns, _ = build_patterns(args)
    if not args.execute:
        return plan_only(patterns)
    return run_execute(args)


if __name__ == "__main__":
    sys.exit(main())