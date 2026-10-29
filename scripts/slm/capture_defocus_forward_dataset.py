"""Capture paired phase-diverse CCD data at the current and compensated plane.

Defaults to a device-free plan. Raw phases, repeated frames, settle verdicts and
actual exposure readbacks are retained. A failed settle remains provisional.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.utils.slm.panel_geometry import phase_to_panel
from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase


from ao_shaping.tools.slm.transfer.probes import probe_grid
from loguru import logger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--focus-directory", type=Path, required=True)
    parser.add_argument("--count", type=int, default=18)
    parser.add_argument("--seed", type=int, default=1022)
    parser.add_argument("--region", type=int, default=64)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    saved = json.loads((args.focus_directory / "focus.json").read_text(encoding="utf-8"))
    settings = saved["settings"]
    focus = saved["focus"]
    if args.count < 6 or args.region < 24:
        parser.error("count >= 6 and region >= 24 required")
    plan = {**vars(args), "settings": settings, "best_diopter_m1": focus["best_diopter_m1"]}
    logger.info(json.dumps(plan, default=str, indent=2))
    if not args.execute:
        for index in range(args.count):
            probe_grid(index, args.region, args.seed)
        return 0

    from ao_shaping.drivers.ccd.common import create_camera, get_camera_exposure_range
    from ao_shaping.drivers.slm.santec import Santec
    from ao_shaping.tools.kernels.bench_kernels import (
        SLM_PANEL_H, SLM_PANEL_W, capture_settled_report, fresnel_panel, measure_spot,
    )
    from ao_shaping.tools.slm.panel.focus_probe import adapt_exposure
    from ao_shaping.utils.image.exposure_control import configured_daheng_exposure_limit

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "run_config.json").write_text(json.dumps(plan, default=str, indent=2), encoding="utf-8")
    shape = (SLM_PANEL_H, SLM_PANEL_W)
    center = (settings["center_x"], settings["center_y"])
    radius = settings["radius"]
    corrected = fresnel_panel(focus["best_focal_mm"], radius=radius,
                              pupil_center=center, wavelength_nm=settings["wavelength"])
    np.save(out / "focus_phase.npy", corrected)
    rows = []
    with Santec(slm_number=settings["slm_number"], wavelength=settings["wavelength"], video_mode=0) as slm:
        try:
            with create_camera(settings["cam_type"], settings["cam_id"], exposure_time_ms=1) as cam:
                floor, ceiling = get_camera_exposure_range(cam)
                limit = configured_daheng_exposure_limit() if settings["cam_type"] == "daheng" else 0
                if limit > 0:
                    ceiling = min(ceiling, limit)
                current = 1.0

                def capture(phase, label, branch, family, split, model_phase=None):
                    nonlocal current
                    started = time.monotonic_ns()
                    utc_start = datetime.now(timezone.utc).isoformat(timespec="microseconds")
                    slm.display_data(slm.create_phase_from_array(phase), .8)
                    setting = adapt_exposure(cam.reset_exposure_time,
                                             lambda: cam.get_numpy_image(n_sample=1),
                                             start_ms=current, floor_ms=floor, ceiling_ms=ceiling,
                                             full_scale=255)
                    current = float(cam.exposure_time)
                    acquisition = capture_settled_report(cam, slm, phase, n_frames=3,
                                                         stable_tol=.05, max_wait_s=12)
                    raw = np.stack([np.asarray(cam.get_numpy_image(n_sample=1)) for _ in range(3)])
                    number = len(rows)
                    file = f"sample_{number:04d}.npz"
                    np.savez_compressed(out / file, frame=acquisition.frame, raw_frames=raw,
                                        panel_phase=phase.astype(np.float32),
                                        model_phase=np.asarray(model_phase) if model_phase is not None else np.zeros((0, 0)))
                    row = dict(index=number, file=file, label=label, branch=branch, family=family,
                               split=split, exposure_ms=float(cam.exposure_time),
                               exposure_status=setting.status, settled=asdict(acquisition.settle),
                               utc_start=utc_start,
                               timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                               monotonic_start_ns=started, monotonic_end_ns=time.monotonic_ns(),
                               peak=float(raw.max()), spot=asdict(measure_spot(acquisition.frame)))
                    rows.append(row)
                    (out / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
                    with (out / "capture_records.pkl").open("wb") as stream:
                        pickle.dump(rows, stream, protocol=5)
                    logger.info(f"{number:03d} {label} {branch}: exposure={current:.4f} "
                          f"peak={raw.max()} settled={acquisition.settle.settled}")
                    if setting.status != "ok" or raw.max() >= 250:
                        raise RuntimeError(f"unusable exposure: {label} {setting.status}")

                # Full-panel ramps determine the CCD axes without assuming the pupil centre.
                y, x = np.indices(shape)
                capture(np.zeros(shape), "reference_start", "U", "reference", "calibration")
                capture(corrected, "reference_start", "C", "reference", "calibration")
                for axis, grid in (("x", x), ("y", y)):
                    for period in (480, 960):
                        for sign in (-1, 1):
                            capture(corrected + sign * 2 * np.pi * grid / period,
                                    f"ramp_{axis}_{period}_{sign}", "C", "ramp", "calibration")
                for index in range(args.count):
                    family, model_phase = probe_grid(index, args.region, args.seed)
                    panel = phase_to_panel(model_phase, radius, center)
                    split = "test" if index % 6 >= 3 else "train"
                    order = ("U", "C") if index % 2 == 0 else ("C", "U")
                    for branch in order:
                        capture(panel + (corrected if branch == "C" else 0),
                                f"probe_{index:03d}", branch, family, split, model_phase)
                    if index % 6 == 5:
                        capture(np.zeros(shape), f"reference_{index}", "U", "reference", "calibration")
                        capture(corrected, f"reference_{index}", "C", "reference", "calibration")
        finally:
            slm.display_data(slm.create_phase_from_array(np.zeros(shape)), .5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
