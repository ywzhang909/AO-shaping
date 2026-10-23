"""Capture current-path phase diversity without reusing historical geometry.

Raw frames, displayed grayscale, raw phase and per-frame clocks are retained.
Defaults to a device-free plan; hardware is opened only with --execute.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter



def clock() -> dict:
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                monotonic_ns=time.monotonic_ns())


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


def probes(size: int, count: int, seed: int):
    from ao_shaping.tools.slm.transfer.probes import probe_grid

    zero = np.zeros((size, size), np.float32)
    yield "reference_start", "calibration", zero
    yy, xx = np.mgrid[:size, :size] * (1920 / size)
    for axis, coord in (("x", xx), ("y", yy)):
        for period in (64, 128):
            for sign in (-1, 1):
                yield f"ramp_{axis}_{period}_{sign}", "calibration", sign * 2 * np.pi * coord / period
    for index in range(count):
        family, grid = probe_grid(index, size, seed)
        # All families occur in both disjoint splits.
        yield f"probe_{index:03d}_{family}", "validation" if index % 6 >= 3 else "train", grid
    yield "reference_end", "calibration", zero


def capture(args):
    from .hardware import prepare_camera_sdk
    prepare_camera_sdk(args.camera)
    from ao_shaping.drivers.ccd.common import create_camera, get_camera_exposure_ms, get_camera_exposure_range
    from ao_shaping.drivers.slm.santec import Santec
    from ao_shaping.tools.slm.panel.focus_probe import adapt_exposure
    from ao_shaping.utils.image.exposure_control import configured_daheng_exposure_limit

    args.out.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), **clock(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.out / "config.json").write_text(json.dumps(config, default=str, indent=2), encoding="utf-8")
    manifest = []
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

                for label, split, grid in probes(args.grid, args.count, args.seed):
                    panel = to_panel(grid)
                    gray = slm.create_phase_from_array(panel)
                    slm.display_data(gray)
                    time.sleep(.6)
                    exposure_frames, exposure_clocks = [], []

                    def probe_read(_clocks=exposure_clocks, _frames=exposure_frames):
                        _clocks.append({**clock(), "exposure_ms": float(get_camera_exposure_ms(cam))})
                        frame = read()
                        _frames.append(frame.copy())
                        return frame

                    setting = adapt_exposure(cam.reset_exposure_time, probe_read, start_ms=exposure,
                                             floor_ms=floor, ceiling_ms=ceiling, full_scale=255,
                                             max_steps=12)
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
                    np.savez_compressed(args.out / path, raw_frames=raw, frame=frame,
                                        model_phase=grid.astype(np.float32), panel_phase=panel.astype(np.float32),
                                        panel_gray=gray, exposure_probe_frames=np.stack(exposure_frames))
                    x, y = locate(frame)
                    row = dict(file=path, label=label, split=split, exposure_ms=exposure,
                               frames=clocks, exposure_probe_clocks=exposure_clocks,
                               exposure_setting=asdict(setting), peak=float(raw.max()),
                               saturation_pixels=int((raw >= 255).sum()), spot_xy=[x, y], **clock())
                    manifest.append(row)
                    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                    with (args.out / "capture_records.pkl").open("wb") as stream:
                        pickle.dump(manifest, stream, protocol=5)
                    print(f"{label}: exposure={exposure:.5f} peak={raw.max():.1f} spot=({x:.1f},{y:.1f}) {setting.status}", flush=True)
                    if setting.status != "ok" or raw.max() >= 250:
                        raise RuntimeError("Unusable exposure; data retained, releasing devices")
            finally:
                slm.display_data(slm.create_phase_from_array(np.zeros((1200, 1920))))
                time.sleep(.3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--camera", default="daheng")
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--seed", type=int, default=102319)
    args = parser.parse_args()
    if args.frames < 2 or args.count < 6:
        parser.error("frames >= 2 and count >= 6 required")
    plan = [(label, split, list(grid.shape)) for label, split, grid in probes(args.grid, args.count, args.seed)]
    if args.execute:
        capture(args)
    else:
        print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
