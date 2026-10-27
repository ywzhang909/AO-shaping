"""Repeat fixed-phase exposures without assuming a permanent exposure ceiling.

Requires an existing paired dataset. Defaults to a plan without opening devices.
All raw frames, exposure readbacks and explicit stability verdicts are retained.
"""
from __future__ import annotations

import argparse
from loguru import logger
import json
import pickle
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--fractions", nargs="+", type=float, default=[.5, 1., .25, .5],
                        help="Relative exposures per branch; repeat a level for independent validation")
    parser.add_argument("--branches", nargs="+", choices=("U", "C"), default=["C", "U"])
    parser.add_argument("--max-wait-s", type=float, default=12.)
    args = parser.parse_args()
    config = json.loads((args.directory / "run_config.json").read_text())
    settings = config["settings"]
    manifest = json.loads((args.directory / "manifest.json").read_text())
    references = {r["branch"]: r for r in manifest if r["label"] == "reference_start"}
    # Three relative levels, bounded by each phase's already observed level.
    if len(args.fractions) < 3 or any(f <= 0 for f in args.fractions):
        parser.error("at least three positive relative exposures required")
    if args.max_wait_s <= 0:
        parser.error("--max-wait-s must be positive")
    plan = [(branch, fraction) for branch in args.branches for fraction in args.fractions]
    logger.info(json.dumps({"source": str(args.directory), "plan": plan}, indent=2))
    if not args.execute:
        return

    from ao_shaping.drivers.ccd.common import create_camera, get_camera_exposure_range
    from ao_shaping.drivers.slm.santec import Santec
    from ao_shaping.tools.kernels.bench_kernels import capture_settled_report
    from ao_shaping.utils.image.exposure_control import configured_daheng_exposure_limit

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps({"source": str(args.directory.resolve()),
                                                "plan": plan, "settings": settings,
                                                "max_wait_s": args.max_wait_s}, indent=2))
    rows = []
    flat = np.zeros((1200, 1920), np.float32)
    with Santec(slm_number=settings["slm_number"], wavelength=settings["wavelength"], video_mode=0) as slm:
        try:
            with create_camera(settings["cam_type"], settings["cam_id"], exposure_time_ms=1) as cam:
                floor, ceiling = get_camera_exposure_range(cam)
                limit = configured_daheng_exposure_limit() if settings["cam_type"] == "daheng" else 0
                if limit > 0:
                    ceiling = min(ceiling, limit)
                for index, (branch, fraction) in enumerate(plan):
                    reference = references[branch]
                    with np.load(args.directory / reference["file"]) as data:
                        phase = data["panel_phase"]
                    exposure = float(reference["exposure_ms"]) * fraction
                    if not floor <= exposure <= ceiling:
                        raise ValueError(f"requested exposure outside current bounds: {exposure}")
                    started = time.monotonic_ns()
                    slm.display_data(slm.create_phase_from_array(phase), .8)
                    cam.reset_exposure_time(exposure)
                    for _ in range(3):
                        cam.get_numpy_image(n_sample=1)
                    settled = capture_settled_report(cam, slm, phase, n_frames=3,
                                                     stable_tol=.05, max_wait_s=args.max_wait_s)
                    raw = np.stack([cam.get_numpy_image(n_sample=1) for _ in range(5)])
                    file = f"response_{index:03d}.npz"
                    np.savez_compressed(out / file, raw_frames=raw, panel_phase=phase)
                    frame = raw.mean(0)
                    cx, cy = [round(reference["spot"][f"centroid_{k}"]) for k in ("x", "y")]
                    roi = frame[cy-120:cy+121, cx-120:cx+121]
                    yy, xx = np.indices(roi.shape)
                    signal = (xx-120)**2 + (yy-120)**2 <= 100**2
                    # This is an illuminated reference annulus, never a dark frame.
                    annulus = (xx-120)**2 + (yy-120)**2 >= 110**2
                    background = float(np.median(roi[annulus]))
                    row = dict(index=index, branch=branch, fraction=fraction, file=file,
                               exposure_ms=float(cam.exposure_time), peak=float(raw.max()),
                               roi_sum=float(roi[signal].sum()), annulus_median=background,
                               corrected_sum=float((roi[signal]-background).sum()),
                               settled=asdict(settled.settle),
                               timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                               monotonic_start_ns=started, monotonic_end_ns=time.monotonic_ns())
                    rows.append(row)
                    (out / "manifest.json").write_text(json.dumps(rows, indent=2))
                    with (out / "capture_records.pkl").open("wb") as stream:
                        pickle.dump(rows, stream, protocol=5)
                    print(json.dumps(row), flush=True)
                    if raw.max() >= 250:
                        raise RuntimeError("saturated exposure; response experiment stopped")
        finally:
            slm.display_data(slm.create_phase_from_array(flat), .5)


if __name__ == "__main__":
    main()
