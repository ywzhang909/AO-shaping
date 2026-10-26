"""Replay exact archived phases at common native bit depth and exposures."""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from ao_shaping.algorithm.signal_processing.multishape_objective import ShapeObjective, image_metrics
from .capture import clock, to_panel
from .hardware import prepare_camera_sdk


def replay(args):
    prepare_camera_sdk("daheng")
    from ao_shaping.drivers.ccd.common import create_camera, get_camera_exposure_ms, get_camera_exposure_range
    from ao_shaping.drivers.slm.santec import Santec
    from ao_shaping.utils.image.exposure_control import configured_daheng_exposure_limit

    sources = []
    selections = [(args.initial_run, "square_seed")] + [(d, "square_final") for d in args.runs]
    if getattr(args, "include_best", False):
        for directory in args.runs:
            if not (directory / "summary.json").exists():
                continue
            rows = json.loads((directory / "manifest.json").read_text())
            blocks = json.loads((directory / "summary.json").read_text())
            eligible = {"square_seed"}
            eligible.update(f"square_{b['block']}_B" for b in blocks
                            if b["shape"] == "square" and b["accepted"] and
                            b.get("acceptance") in {"sequential_ccd", "nonmonotone_ccd"})
            selected = min((r for r in rows if r["label"] in eligible), key=lambda r: r["metrics"]["cv"])
            selections.append((directory, selected["label"]))
    for directory, label in selections:
        rows = json.loads((directory / "manifest.json").read_text())
        row = next(r for r in rows if r["label"] == label)
        with np.load(directory / row["file"]) as saved:
            phase = saved["model_phase"].copy()
            if not sources:
                xy, target, core = (saved[k].copy() for k in ("camera_xy", "target", "core"))
        sources.append((directory.name + "_" + label, phase))
    objective = ShapeObjective(target, cv_only=True)
    objective.core = core
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "config.json").write_text(json.dumps(dict(**vars(args), sources=[s[0] for s in sources], **clock()), default=str, indent=2), encoding="utf-8")
    records = []
    with create_camera("daheng", 0, exposure_time_ms=.02, bit_depth=12) as camera:
        floor, ceiling = get_camera_exposure_range(camera)
        limit = configured_daheng_exposure_limit()
        if min(args.exposures) < floor or max(args.exposures) > min(ceiling, limit if limit > 0 else ceiling):
            raise ValueError("replay exposure outside current session range")
        with Santec(slm_number=1, wavelength=1064, video_mode=0) as slm:
            slm.set_phase_overlay(None)
            try:
                for exposure in args.exposures:
                    camera.reset_exposure_time(exposure)
                    for name, phase in sources:
                        gray = slm.create_phase_from_array(to_panel(phase))
                        slm.display_data(gray)
                        time.sleep(.6)
                        frames, stamps = [], []
                        for _ in range(args.frames):
                            stamp = clock()
                            frame = np.asarray(camera.get_numpy_image(n_sample=1, skip_first=False)).copy()
                            frames.append(frame)
                            stamps.append(dict(**stamp, end_monotonic_ns=time.monotonic_ns(), exposure_ms=float(get_camera_exposure_ms(camera))))
                        raw = np.stack(frames)
                        sample = raw.mean(0)[xy[..., 1].astype(int), xy[..., 0].astype(int)]
                        filename = f"replay_{len(records):04d}.npz"
                        np.savez_compressed(args.out / filename, raw_frames=raw, model_phase=phase,
                            panel_gray=gray, camera_xy=xy, target=target, core=core)
                        record = dict(label=name, file=filename, exposure_ms=float(get_camera_exposure_ms(camera)),
                            full_scale=4095, bit_depth=12, counts_scale=16., frames=stamps,
                            metrics=image_metrics(sample, objective), core_peak=float(sample[core].max()), **clock())
                        records.append(record)
                        (args.out / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
                        with (args.out / "capture_records.pkl").open("wb") as stream:
                            pickle.dump(records, stream, protocol=5)
                        print(json.dumps(record | dict(frames=len(stamps))), flush=True)
                        if record["core_peak"] >= 4014:
                            raise RuntimeError("core saturated; raw replay retained")
            finally:
                slm.display_data(slm.create_phase_from_array(np.zeros((1200, 1920))))
                time.sleep(.3)
    (args.out / "released.json").write_text(json.dumps(dict(status="camera and SLM contexts closed", **clock()), indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-run", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exposures", type=float, nargs="+", default=[50., 100.])
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--include-best", action="store_true", help="Also replay the lowest measured accepted sequential candidate from each run")
    replay(parser.parse_args())
