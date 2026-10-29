"""Capture current-path phase diversity without reusing historical geometry.

Raw frames, displayed grayscale, raw phase and per-frame clocks are retained.
Defaults to a device-free plan; hardware is opened only with --execute.

The measurements are handed to the fit stage **in memory** by default: one
``sample_%04d.npz`` per probe carries ``raw_frames`` (frames x 1944 x 2592 uint8),
a float64 ``frame`` (40 MB), ``panel_phase`` (9 MB) and the exposure-probe stack,
i.e. ~60-80 MB and most of a probe's wall clock. ``fit.load_dataset`` only reads
``model_phase`` and ``raw_frames``, so the default path keeps exactly those two
arrays per probe in RAM and writes none of the npz files. The on-disk dataset
stays available as an opt-in (``--save-dataset`` / ``TransferSession.save_dataset``)
because the offline ``--dataset`` path and the analysis scripts under
``scripts/slm/`` read it; when enabled it writes the same keys and the same
content as before, so a recorded dataset stays byte-compatible.

Exposure is armed with **one** multiplicative correction plus one verification
read (the sensor is linear in exposure, so the correction is exact); the bound is
widened for the cold start, where the first probe reads a few percent of full
scale and needs a factor of ~9.
"""
from __future__ import annotations
from loguru import logger

import argparse
import hashlib
import json
import pickle
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from .light_guard import LightGuard, OpticalPathInvalid, verify_light

#: Exposure search for one probe: one exact correction + one verification read.
#: The old loop allowed 12 multiplicative steps under a (0.15, 6.0) clamp; a cold
#: start at the camera floor (peak ~5% of full scale) needs a factor of ~9, which
#: that clamp cannot do in one step. Both values are recorded in ``config.json``.
EXPOSURE_MAX_STEPS = 2
EXPOSURE_RATIO_BOUNDS = (0.1, 50.0)



def clock() -> dict:
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                monotonic_ns=time.monotonic_ns())


def exposure_setter(cam) -> Callable[[float], float]:
    """Wrap a camera into the ``set_exposure_ms`` callback the light guard wants.

    ``reset_exposure_time`` returns the effective exposure on the hardware driver,
    but the leaf backends (sim / image_folder) are allowed to return ``None``, and
    the value that matters for the ``exposure_ceiling`` diagnosis is the one the
    **sensor** reports, not the one we asked for — ``get_camera_exposure_ms`` is
    already the single readback entry point (issue #32, ``回读恒为 3.0``). So set,
    then read back, instead of trusting a return value that half the backends
    omit.
    """
    from ao_shaping.drivers.ccd.common import get_camera_exposure_ms

    def set_exposure(ms: float) -> float:
        cam.reset_exposure_time(float(ms))
        return get_camera_exposure_ms(cam)

    return set_exposure


def to_panel(grid: np.ndarray) -> np.ndarray:
    """Expand the exact model controls, without averaging wrapped phases."""
    size = grid.shape[0]
    if grid.shape != (size, size) or 1920 % size or 1200 % (1920 // size):
        raise ValueError("grid must exactly tile the physical panel")
    pitch = 1920 // size
    return np.repeat(np.repeat(grid, pitch, axis=0), pitch, axis=1)[:1200]


def locate(frame: np.ndarray) -> tuple[float, float]:
    """Locate the dominant component using a smoothed peak and local moments."""
    smooth = gaussian_filter(np.asarray(frame, dtype=float), 3)
    row, col = np.unravel_index(smooth.argmax(), smooth.shape)
    y0, y1 = max(0, row - 30), min(frame.shape[0], row + 31)
    x0, x1 = max(0, col - 30), min(frame.shape[1], col + 31)
    weights = np.maximum(frame[y0:y1, x0:x1] - np.median(frame), 0)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return float((xx * weights).sum() / weights.sum()), float((yy * weights).sum() / weights.sum())


def probes(size: int, count: int, seed: int, periods=(64, 128)):
    from ao_shaping.tools.slm.transfer.probes import probe_grid

    zero = np.zeros((size, size), np.float32)
    yield "reference_start", "calibration", zero
    yy, xx = np.mgrid[:size, :size] * (1920 / size)
    for axis, coord in (("x", xx), ("y", yy)):
        for period in periods:
            for sign in (-1, 1):
                yield f"ramp_{axis}_{period}_{sign}", "calibration", sign * 2 * np.pi * coord / period
    for index in range(count):
        family, grid = probe_grid(index, size, seed)
        # All families occur in both disjoint splits.
        yield f"probe_{index:03d}_{family}", "validation" if index % 6 >= 3 else "train", grid
    yield "reference_end", "calibration", zero


def capture(args):
    """Measure the probe plan; return ``{manifest file name: fit payload}``.

    The returned mapping is what ``fit.load_dataset(samples=...)`` consumes, so
    the fit stage never needs the npz files. Payload keys are exactly the ones
    the fit reads (``model_phase`` on the control grid, ``raw_frames`` averaged
    by the fit); ``panel_phase`` / ``panel_gray`` / ``frame`` /
    ``exposure_probe_frames`` are written only on the ``save_dataset`` path.

    Memory: ``raw_frames`` is kept as the uint8 stack (~15 MB for 3 frames of
    1944x2592), i.e. ~0.5 GB for the 34-entry default plan. The fit only ever
    averages it, so a mean-frame payload would cut that to ~40 MB — deliberately
    not done, because that would change what the fit consumes.
    """
    from .hardware import prepare_camera_sdk
    prepare_camera_sdk(args.camera)
    from ao_shaping.drivers.ccd.common import create_camera, get_camera_exposure_ms, get_camera_exposure_range
    from ao_shaping.drivers.slm.santec import Santec
    from ao_shaping.tools.slm.panel.focus_probe import adapt_exposure
    from ao_shaping.utils.image.exposure_control import configured_daheng_exposure_limit

    save_dataset = bool(getattr(args, "save_dataset", False))
    exposure_steps = int(getattr(args, "exposure_max_steps", EXPOSURE_MAX_STEPS))
    if exposure_steps < 2:
        raise ValueError("exposure_max_steps must be >= 2")
    args.out.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), **clock(),
              "save_dataset": save_dataset,
              "exposure_adaptation": dict(max_steps=exposure_steps,
                                          ratio_bounds=list(EXPOSURE_RATIO_BOUNDS)),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.out / "config.json").write_text(json.dumps(config, default=str, indent=2), encoding="utf-8")
    manifest, samples = [], {}
    early_stopped = False
    with create_camera(args.camera, args.cam_id, exposure_time_ms=.02) as cam:
        with Santec(slm_number=args.slm_number, wavelength=1064, video_mode=0) as slm:
            slm.set_phase_overlay(None)
            try:
                floor, ceiling = get_camera_exposure_range(cam)
                limit = configured_daheng_exposure_limit() if args.camera == "daheng" else 0
                if limit > 0:
                    ceiling = min(ceiling, limit)
                exposure = max(floor, .02)

                def read():
                    raw = np.asarray(cam.get_numpy_image(n_sample=1, skip_first=False))
                    if raw.ndim != 2:
                        raise ValueError(f"Expected a monochrome 2D camera frame, got {raw.shape}")
                    return raw

                # 便宜的光路检查, 在探针族之前: 2026-10-28 光路失效那次, 单条探针
                # 先烧掉整个阶梯 (每档 0.6 s settle + 落盘) 才报「曝光不收敛」,
                # 一条 lane 几十秒换来一句无法诊断的话。这里先显示平场 (第一号
                # 探针 reference_start 本来就是它, 所以不多一次显示语义), 再用
                # 0.05/0.4/3.2 ms 三档少数几帧回答「有没有光」; 没有光就直接抛
                # OpticalPathInvalid (exit code 3), 不进探针族。
                slm.display_data(slm.create_phase_from_array(to_panel(
                    np.zeros((args.grid, args.grid), np.float32))))
                time.sleep(.6)
                light = verify_light(read, exposure_setter(cam),
                                     floor_ms=floor, ceiling_ms=ceiling,
                                     label=f"capture {args.out.name}")
                (args.out / "light_check.json").write_text(
                    json.dumps({**light, **clock()}, default=str, indent=2), encoding="utf-8")

                for label, split, grid in probes(args.grid, args.count, args.seed, getattr(args, "ramp_periods", (64, 128))):
                    # The parallel-fit worker (or an operator) may ask capture to
                    # stop early; honour it before taking the next probe.
                    if (args.out / "stop_acquisition.request").is_file():
                        early_stopped = True
                        break
                    panel = to_panel(grid)
                    gray = slm.create_phase_from_array(panel)
                    slm.display_data(gray)
                    time.sleep(.6)
                    exposure_frames, exposure_clocks = [], []

                    def probe_read(_clocks=exposure_clocks, _frames=exposure_frames):
                        _clocks.append({**clock(), "exposure_ms": float(get_camera_exposure_ms(cam))})
                        frame = read()
                        if save_dataset:
                            _frames.append(frame.copy())
                        return frame

                    setting = adapt_exposure(cam.reset_exposure_time, probe_read, start_ms=exposure,
                                             floor_ms=floor, ceiling_ms=ceiling, full_scale=255,
                                             max_steps=exposure_steps,
                                             ratio_bounds=EXPOSURE_RATIO_BOUNDS,
                                             on_step=LightGuard(label=f"probe {label}",
                                                                max_steps=exposure_steps).observe)
                    exposure = float(get_camera_exposure_ms(cam))
                    frames, clocks = [], []
                    for _ in range(args.frames):
                        stamp = clock()
                        frames.append(read().copy())
                        clocks.append({**stamp, "end_monotonic_ns": time.monotonic_ns(),
                                       "exposure_ms": float(get_camera_exposure_ms(cam))})
                    raw = np.stack(frames)
                    frame = raw.mean(0)
                    path = f"sample_{len(manifest):04d}.npz"
                    if save_dataset:
                        np.savez_compressed(args.out / path, raw_frames=raw, frame=frame,
                                            model_phase=grid.astype(np.float32), panel_phase=panel.astype(np.float32),
                                            panel_gray=gray, exposure_probe_frames=np.stack(exposure_frames))
                    samples[path] = dict(model_phase=grid.astype(np.float32), raw_frames=raw)
                    x, y = locate(frame)
                    row = dict(file=path, label=label, split=split, exposure_ms=exposure,
                               frames=clocks, exposure_probe_clocks=exposure_clocks,
                               exposure_setting=asdict(setting), peak=float(raw.max()),
                               saturation_pixels=int((raw >= 255).sum()), spot_xy=[x, y], **clock())
                    manifest.append(row)
                    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                    with (args.out / "capture_records.pkl").open("wb") as stream:
                        pickle.dump(manifest, stream, protocol=5)
                    logger.info(f'{label}: exposure={exposure:.5f} peak={raw.max():.1f} spot=({x:.1f},{y:.1f}) {setting.status}')
                    if setting.status != "ok" or raw.max() >= 250:
                        raise RuntimeError("Unusable exposure; data retained, releasing devices")
            finally:
                slm.display_data(slm.create_phase_from_array(np.zeros((1200, 1920))))
                time.sleep(.3)
    (args.out / ".capture_done").write_text(
        json.dumps(dict(early_stopped=early_stopped, probes=len(manifest), **clock()), indent=2),
        encoding="utf-8")
    return samples


def precheck_light(*, camera: str, cam_id: int, label: str = "启动前光路检查") -> dict:
    """开一次相机, 读少数几帧, 回答「这一路现在有没有光」。

    只给**跳过采集**的启动路径用 (``--model-dir`` 复用一份已拟合的模型)。正常
    路径里 :func:`capture` 已经在它自己打开的相机上做了同一件事 —— 同一实现
    :func:`ao_shaping.tools.slm.transfer.light_guard.verify_light`, 所以这里不再
    重复判据, 也不该被正常路径调用 (那会白开一次设备)。

    为什么跳过采集的路径需要它: 那份模型是在**原来那束光**上拟合的, 光路失效后
    继续用, 第一个变暗的读数要等到整形段第一块才出现, 而模型加载、目标与种子
    准备都已经跑完。这一步把坏光路挡在这些分钟级工作之前。

    面板上显示的是上一次会话留下的相位 —— 不影响「有没有光」的判定 (只影响光斑
    形状), 所以这里不为此再打开 SLM。

    Args:
        camera: ``daheng`` / ``sim`` 等相机类型。
        cam_id: 相机下标。
        label: 写进异常的对象名。

    Returns:
        :func:`verify_light` 的判定字典 (已带实测数字)。

    Raises:
        OpticalPathInvalid: 每一档都只有噪声底。
    """
    from .hardware import prepare_camera_sdk
    prepare_camera_sdk(camera)
    from ao_shaping.drivers.ccd.common import create_camera, get_camera_exposure_range
    from ao_shaping.utils.image.exposure_control import configured_daheng_exposure_limit

    with create_camera(camera, cam_id, exposure_time_ms=.02) as cam:
        floor, ceiling = get_camera_exposure_range(cam)
        limit = configured_daheng_exposure_limit() if camera == "daheng" else 0
        if limit > 0:
            ceiling = min(ceiling, limit)
        return verify_light(
            lambda: np.asarray(cam.get_numpy_image(n_sample=1, skip_first=False)),
            exposure_setter(cam), floor_ms=floor, ceiling_ms=ceiling, label=label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--camera", default="daheng")
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--ramp-periods", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--seed", type=int, default=102319)
    parser.add_argument("--exposure-max-steps", type=int, default=EXPOSURE_MAX_STEPS)
    parser.add_argument("--save-dataset", dest="save_dataset", action="store_true", default=True,
                        help="Write one sample_*.npz per probe (the standalone recorder's "
                             "default). Off keeps the measurements in memory only.")
    parser.add_argument("--no-save-dataset", dest="save_dataset", action="store_false",
                        help="Skip the per-probe npz files; the fit stage consumes the "
                             "in-memory payload instead")
    args = parser.parse_args()
    if args.frames < 2 or args.count < 6:
        parser.error("frames >= 2 and count >= 6 required")
    plan = [(label, split, list(grid.shape)) for label, split, grid in probes(args.grid, args.count, args.seed)]
    if args.execute:
        # 光路无光是实验系统上的预期状态, 不是缺陷: 只报实测数字, 退出码 3, 不打
        # traceback (与探针的 2、普通异常的 1 分开)。
        try:
            capture(args)
        except OpticalPathInvalid as exc:
            logger.error(exc.report())
            raise SystemExit(exc.exit_code) from None
    else:
        logger.info(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
