"""Reusable multi-exposure dark-frame capture CLI.

Migrates the one-off ``capture_darks_20261119.py`` diagnostic into a permanent
tool.  It opens **only the camera** (no SLM, no laser) under the bench
:class:`~ao_shaping.tools.kernels.device_mutex.DeviceLock`, captures a ladder of
exposures and stores each level through
:func:`~ao_shaping.drivers.ccd.dark_frames.save_dark_capture` in the serial-keyed
CCD config directory.

The acquisition maths lives in :func:`capture_exposure`, a pure function over a
camera-like object, so the whole capture loop is unit-testable offline.  Only
:func:`main` touches hardware / the device lock.

Typical use::

    python -m ao_shaping.tools.ccd.capture_darks --no-hw
    python -m ao_shaping.tools.ccd.capture_darks --exposures-ms 0.024,0.043 --frames 16

Hardware capture is gated by ``--no-hw`` for self-checks; see the bench hardware
discipline notes before running against a real device.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from ao_shaping.drivers.ccd.dark_frames import dark_frames_dir

DEFAULT_EXPOSURES_MS = "0.024,0.043,0.047"
DEFAULT_FRAMES = 16
DEFAULT_CAM_TYPE = "daheng"
DEFAULT_CAM_ID = 0
LOCK_TIMEOUT_S = 60.0


# --------------------------------------------------------------------------- #
# Pure helpers (no hardware, no I/O beyond path resolution)
# --------------------------------------------------------------------------- #
def parse_exposures(text: str) -> tuple[float, ...]:
    """Parse ``--exposures-ms`` (comma-separated) into positive finite floats."""
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if not parts:
        raise ValueError("--exposures-ms 不能为空")
    values: list[float] = []
    for part in parts:
        try:
            value = float(part)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"--exposures-ms 含非法值 {part!r}") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--exposures-ms 必须是正有限值, 收到 {part!r}")
        values.append(value)
    return tuple(values)


@dataclass(frozen=True)
class AcquisitionPlan:
    """Everything the CLI will do, resolved without opening a device."""

    exposures_ms: tuple[float, ...]
    frames: int
    cam_type: str
    cam_id: int
    serial: str | None
    target_dir: Path | None
    bit_depth: int | None = None


def build_plan(
    exposures_ms: tuple[float, ...],
    frames: int,
    *,
    cam_type: str,
    cam_id: int,
    serial: str | None = None,
    config_root: str | Path | None = None,
    bit_depth: int | None = None,
) -> AcquisitionPlan:
    """Resolve the target directory (via :func:`dark_frames_dir`) for a run."""
    if frames <= 0:
        raise ValueError(f"--frames 必须 > 0, 收到 {frames!r}")
    if not exposures_ms:
        raise ValueError("曝光列表不能为空")
    if bit_depth is not None and int(bit_depth) <= 0:
        raise ValueError(f"--bit-depth 必须为正数, 收到 {bit_depth!r}")
    resolved_serial = str(serial).strip() if serial is not None and str(serial).strip() else None
    target = dark_frames_dir(resolved_serial, root=config_root) if resolved_serial else None
    return AcquisitionPlan(
        exposures_ms=tuple(exposures_ms),
        frames=int(frames),
        cam_type=cam_type,
        cam_id=cam_id,
        serial=resolved_serial,
        target_dir=target,
        bit_depth=None if bit_depth is None else int(bit_depth),
    )


def format_plan(plan: AcquisitionPlan) -> str:
    """Human-readable acquisition plan (used by ``--no-hw``)."""
    serial = plan.serial or "<由相机回读, 或 --serial 指定>"
    target = str(plan.target_dir) if plan.target_dir is not None else "<serial 确定后解析>"
    lines = [
        "暗帧采集计划 (不打开设备):",
        f"  相机        : {plan.cam_type}#{plan.cam_id}",
        f"  序列号      : {serial}",
        f"  位深        : {plan.bit_depth if plan.bit_depth is not None else '<相机配置/回读>'}",
        f"  曝光阶梯 ms : {', '.join(f'{ms:g}' for ms in plan.exposures_ms)}",
        f"  每档平均帧数: {plan.frames}",
        f"  目标目录    : {target}",
    ]
    return "\n".join(lines)


def capture_exposure(
    cam: Any, requested_ms: float, n_frames: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Capture ``n_frames`` reads at ``requested_ms`` and reduce them.

    Uses the camera-agnostic exposure helpers so any supported backend (and a
    plain fake camera in tests) works.  The authoritative exposure stored with
    the dark frame is the device **readback**, not the request (issue #32).

    Returns:
        ``(mean, std, effective_ms)`` -- per-pixel float32 statistics over the
        stacked 2-D frames, and the effective exposure reported by the device.
    """
    if n_frames <= 0:
        raise ValueError(f"n_frames 必须 > 0, 收到 {n_frames!r}")

    # Imported here so that importing this module never drags in the driver
    # package as a side effect (relevant for --no-hw self-checks).
    from ao_shaping.drivers.ccd.common import (
        get_camera_exposure_ms,
        set_camera_exposure_ms,
    )

    # Prefer the BaseCamera contract entry point (all real backends implement it,
    # and duck-typed fakes in tests do too); fall back to the module-level helper
    # for bare camera-like objects that only expose a backend setter/attribute.
    set_fn = getattr(cam, "set_exposure_ms", None)
    if callable(set_fn):
        set_fn(float(requested_ms))
    else:
        set_camera_exposure_ms(cam, float(requested_ms))
    readback = float(get_camera_exposure_ms(cam))
    if not math.isfinite(readback) or readback <= 0:
        raise ValueError(f"设备回读曝光非法: {readback!r}")

    frames: list[np.ndarray] = []
    for _ in range(n_frames):
        frame = np.asarray(cam.get_numpy_image(1, True), dtype=np.float64)
        if frame.ndim == 3:
            frame = frame.mean(axis=2)
        if frame.ndim != 2:
            raise ValueError(f"相机返回了非二维帧 (shape={frame.shape})")
        frames.append(frame)

    stack = np.stack(frames, axis=0)
    mean = stack.mean(axis=0).astype(np.float32)
    std = stack.std(axis=0).astype(np.float32)
    return mean, std, readback


def _resolve_full_scale(cam: Any, mean: np.ndarray, bit_depth: int | None) -> float | None:
    """Effective full scale from bit depth, else from one frame's annotation."""
    if bit_depth is not None:
        try:
            return float((1 << int(bit_depth)) - 1)
        except (TypeError, ValueError, OverflowError):
            return None
    fs_fn = getattr(cam, "effective_full_scale", None)
    if callable(fs_fn):
        try:
            return float(fs_fn(mean))
        except Exception:
            # Metadata must never abort a capture; fall back to no full scale.
            return None
    return None


def _collect_metadata(cam: Any, cam_type: str, mean: np.ndarray) -> dict[str, Any]:
    """Collect the camera metadata block stored in the dark-frame manifest."""
    bit_depth = getattr(cam, "bit_depth", None)
    if bit_depth is not None:
        try:
            bit_depth = int(bit_depth)
        except (TypeError, ValueError):
            bit_depth = None
    full_scale = _resolve_full_scale(cam, mean, bit_depth)
    return {
        "model_name": getattr(cam, "model_name", None),
        "cam_type": cam_type,
        "bit_depth": bit_depth,
        "pixel_size_um": getattr(cam, "pixel_size_um", None),
        "full_scale": full_scale,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--exposures-ms",
        default=DEFAULT_EXPOSURES_MS,
        help=f"逗号分隔的曝光阶梯 (ms), 默认 {DEFAULT_EXPOSURES_MS!r}",
    )
    parser.add_argument(
        "--frames", type=int, default=DEFAULT_FRAMES,
        help=f"每档平均的原始帧数, 默认 {DEFAULT_FRAMES}",
    )
    parser.add_argument("--cam-type", default=DEFAULT_CAM_TYPE, help="相机后端类型")
    parser.add_argument("--cam-id", type=int, default=DEFAULT_CAM_ID, help="相机 id")
    parser.add_argument(
        "--bit-depth", type=int, default=None,
        help="请求的输出位深 (8/12/16); 必须经相机回读校验, 缺省沿用相机配置",
    )
    parser.add_argument("--config-root", default=None, help="CCD 配置根目录 (覆盖默认解析)")
    parser.add_argument(
        "--serial", default=None,
        help="手动指定序列号 (后端无回读时的逃生口); 缺省从相机读取",
    )
    parser.add_argument(
        "--no-hw", action="store_true",
        help="只打印采集计划, 不打开设备、不取设备锁",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns a process exit code."""
    args = _parser().parse_args(argv)

    exposures = parse_exposures(args.exposures_ms)
    plan = build_plan(
        exposures, args.frames,
        cam_type=args.cam_type, cam_id=args.cam_id,
        serial=args.serial, config_root=args.config_root,
        bit_depth=args.bit_depth,
    )

    if args.no_hw:
        print(format_plan(plan))
        return 0

    # Imported only for the real hardware path.
    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.tools.kernels.device_mutex import DeviceLock

    lock = DeviceLock.default()
    acquired = lock.acquire(timeout=LOCK_TIMEOUT_S)
    logger.info(
        "[capture_darks] DeviceLock acquired={} holder={}", acquired, lock.holder()
    )
    if not acquired:
        logger.error("[capture_darks] 台面设备被其它进程占用, 终止")
        return 2

    try:
        logger.info(
            "[capture_darks] 设备占用开始: 打开相机 {}#{} 采暗帧 (不碰 SLM/激光)",
            args.cam_type, args.cam_id,
        )
        camera_kwargs: dict[str, Any] = {"exposure_time_ms": exposures[0]}
        if plan.bit_depth is not None:
            camera_kwargs["bit_depth"] = plan.bit_depth
        with create_camera(args.cam_type, args.cam_id, **camera_kwargs) as cam:
            # 位深是设备回读真值才可信: 请求 12-bit 若静默落到 Mono8, 满量程不同,
            # 暗帧与 12-bit 配对曝光不可减, 必须在这里拦下 (ccd/AGENTS.md 位深契约)。
            if plan.bit_depth is not None:
                actual = getattr(cam, "bit_depth", None)
                if actual is None or int(actual) != plan.bit_depth:
                    raise RuntimeError(
                        f"位深回读校验失败: 请求 {plan.bit_depth}-bit, 设备报告 {actual}; "
                        "拒绝写入位深不符的暗帧"
                    )
            serial = (
                getattr(cam, "serial_number", None)
                or getattr(cam, "sn", None)
                or args.serial
            )
            if not serial:
                raise RuntimeError(
                    "相机未报告序列号且未指定 --serial; 拒绝写入目录 'unknown', "
                    "请使用 --serial 显式指定"
                )

            root = args.config_root
            from ao_shaping.drivers.ccd.dark_frames import save_dark_capture

            for requested in exposures:
                mean, std, effective_ms = capture_exposure(cam, requested, args.frames)
                metadata = _collect_metadata(cam, args.cam_type, mean)
                entry = save_dark_capture(
                    str(serial),
                    exposure_ms=effective_ms,
                    mean=mean,
                    std=std,
                    frames=args.frames,
                    requested_exposure_ms=requested,
                    bit_depth=metadata["bit_depth"],
                    full_scale=metadata["full_scale"],
                    camera=metadata,
                    root=root,
                )
                logger.info(
                    "[capture_darks] 已保存 serial={} requested={:g}ms "
                    "effective={:g}ms peak={:.1f} mean={:.4f} -> {}",
                    serial, requested, effective_ms, entry.peak or float(mean.max()),
                    float(mean.mean()), entry.file,
                )
        logger.info("[capture_darks] 相机已正常关闭; 设备占用结束")
    finally:
        lock.release()
        logger.info(
            "[capture_darks] DeviceLock 已释放; locked={} holder={}",
            lock.is_locked(), lock.holder(),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
