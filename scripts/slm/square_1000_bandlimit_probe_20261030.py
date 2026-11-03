"""Band-limited transport/continuation probe at 640 and 320 control cells."""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import fftconvolve

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    FrozenDefocusForward, load_frozen_parameters,
)
from ao_shaping.algorithm.signal_processing.multishape_objective import ShapeObjective
from ao_shaping.tools.slm.transfer.capture import clock, to_panel

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/debug/slm_transfer/multishape_nogs_20261030"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def bandlimit(phase, plane, sigma):
    """Low-pass the shaping slope while preserving the calibrated carrier exactly."""
    shaping = np.asarray(phase - plane, dtype=np.float64)
    kernel = np.exp(-.5 * (np.arange(-24, 25) / sigma) ** 2)
    kernel /= kernel.sum()
    pad = len(kernel) // 2
    padded = np.pad(shaping, pad, mode="edge")
    smooth = fftconvolve(padded, np.outer(kernel, kernel), mode="valid")
    return np.asarray(plane + smooth, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--wait-until-idle", type=float, default=0.)
    args = parser.parse_args()
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location("square_protocol", ROOT / "scripts/slm/square_1000_diff_abba_20261030.py")
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    seed_run = DATA / "p20261030_s1000_square_plane_continuation"
    with np.load(seed_run / "proposals.npz") as saved:
        plane, direction, xy = saved["plane"], saved["direction"], saved["camera_xy"]
    phases = {"A": plane + .5 * direction}
    phases["B_blur2"] = bandlimit(plane + .75 * direction, plane, 2.)
    phases["C_blur4"] = bandlimit(plane + .75 * direction, plane, 4.)
    phases["D_blur6"] = bandlimit(plane + .75 * direction, plane, 6.)
    phases["E_blur4_half"] = bandlimit(plane + .5 * direction, plane, 4.)
    phases["F_blur4_full"] = bandlimit(plane + direction, plane, 4.)
    phases["G_blur4_alpha125"] = bandlimit(plane + 1.25 * direction, plane, 4.)
    phases["H_blur4_alpha15"] = bandlimit(plane + 1.5 * direction, plane, 4.)
    sequence = ["A", "A", "B_blur2", "C_blur4", "D_blur6", "E_blur4_half",
                "F_blur4_full", "G_blur4_alpha125", "H_blur4_alpha15", "A"]
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.out / "plan.npz", camera_xy=xy, **phases)
    write_json(args.out / "config.json", dict(sequence=sequence, seed_alpha=.5,
        bandlimit="edge-padded Gaussian on raw shaping phase; carrier unchanged",
        sigmas_cells=[2., 4., 6.], alphas=[.5, .75, 1., 1.25, 1.5],
        execute=args.execute, **clock()))
    if not args.execute:
        for label, phase in phases.items():
            print(label, "max_dx", float(np.abs(np.diff(phase, axis=1)).max()),
                  "max_dy", float(np.abs(np.diff(phase, axis=0)).max()))
        return
    from ao_shaping.tools.kernels.device_mutex import DeviceLock, wait_until_idle
    verdict = wait_until_idle(args.wait_until_idle, lock=DeviceLock.default(), own_paths=[Path(__file__).resolve()])
    if not verdict.ok or verdict.lock is None:
        raise RuntimeError(verdict.headline())
    records = []
    try:
        from ao_shaping.tools.slm.transfer.hardware import prepare_camera_sdk
        prepare_camera_sdk("daheng")
        from ao_shaping.drivers.ccd.common import create_camera, get_camera_exposure_ms
        from ao_shaping.drivers.slm.santec import Santec
        yy, xx = np.mgrid[:1944, :2592]
        native_masks = protocol.masks(np.stack([xx, yy], -1))
        objective = ShapeObjective(protocol.masks(xy)["box"].astype(np.float32), formation=True)
        with create_camera("daheng", 0, exposure_time_ms=.5, bit_depth=12) as cam:
            with Santec(slm_number=1, wavelength=1064, video_mode=0) as slm:
                slm.set_phase_overlay(None)
                try:
                    def capture(phase, exposure, label):
                        gray = slm.create_phase_from_array(to_panel(phase))
                        slm.display_data(gray)
                        cam.reset_exposure_time(exposure)
                        time.sleep(.8)
                        for _ in range(3):
                            cam.get_numpy_image(n_sample=1, skip_first=False)
                        raw = np.stack([np.asarray(cam.get_numpy_image(n_sample=1, skip_first=False)).copy() for _ in range(5)])
                        row = dict(label=label, exposure_ms=float(get_camera_exposure_ms(cam)),
                                   file=f"capture_{len(records):04d}.npz", **clock())
                        np.savez_compressed(args.out / row["file"], raw_frames=raw, model_phase=phase, panel_gray=gray, camera_xy=xy)
                        records.append(row)
                        write_json(args.out / "manifest.json", records)
                        return row, raw
                    row, raw = capture(phases["A"], .5, "A_safe")
                    peak = float(raw[:, native_masks["box"]].max())
                    if peak < 20 or peak >= .98 * 4095:
                        raise RuntimeError("unsafe or unresolved metering")
                    exposure = min(16.87, row["exposure_ms"] * 1400 / peak)
                    row, raw = capture(phases["A"], exposure, "A_calibration")
                    bg = float(np.median(raw.mean(0)[native_masks["background"]]))
                    noise = float(np.sqrt(raw[:, native_masks["background"]].var(0, ddof=1).mean() / len(raw)))
                    mean = float(np.maximum(raw.mean(0) - bg, 0)[native_masks["core"]].mean())
                    threshold = max(3 * noise, .2 * mean)
                    write_json(args.out / "measurement_contract.json", dict(background_counts=bg,
                        coverage_threshold_counts=threshold, exposure_ms=exposure, noise_counts=noise,
                        full_scale=4095, smoothing_sigma_native_px=8))
                    for label in sequence:
                        row, raw = capture(phases[label], exposure, label)
                        row["metrics"] = protocol.metrics(raw, row["exposure_ms"], bg, threshold, native_masks, objective, xy)
                        write_json(args.out / "manifest.json", records)
                        print(json.dumps(dict(label=label, **row["metrics"])), flush=True)
                finally:
                    slm.display_data(np.zeros((1200, 1920), dtype=np.uint16))
                    time.sleep(.3)
    finally:
        verdict.lock.release()
    write_json(args.out / "released.json", clock())


if __name__ == "__main__":
    main()
