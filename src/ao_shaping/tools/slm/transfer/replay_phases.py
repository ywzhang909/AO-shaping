"""Replay exact archived phases at common native bit depth and exposures."""
from __future__ import annotations

import argparse
from loguru import logger
import json
import pickle
import time
from pathlib import Path

import numpy as np

from ao_shaping.algorithm.signal_processing.multishape_objective import ShapeObjective, image_metrics
from .capture import clock, to_panel
from .hardware import prepare_camera_sdk


def _selections(args) -> list[tuple[Path, str]]:
    """Every (run, label) pair to replay: the initial seed plus each run's final.

    The labels are per shape (``<shape>_seed`` / ``<shape>_final``); the shape
    set is taken from the runs themselves, so a multi-shape session replays all
    of its shapes and a single-shape session keeps its old behaviour.
    """
    shapes: list[str] = []
    runs = [args.initial_run, *args.runs]
    for directory in runs:
        rows = json.loads((directory / "manifest.json").read_text())
        for row in rows:
            label = row["label"]
            if label.endswith("_seed") and not label.endswith("_seed_replay"):
                shape = label[: -len("_seed")]
                if shape not in shapes:
                    shapes.append(shape)
    selections: list[tuple[Path, str]] = []
    for shape in shapes:
        for directory in args.runs:  # the seed lives inside a stage run
            rows = json.loads((directory / "manifest.json").read_text())
            if any(row["label"] == f"{shape}_seed" for row in rows):
                selections.append((directory, f"{shape}_seed"))
                break
    for directory in args.runs:
        rows = json.loads((directory / "manifest.json").read_text())
        present = {row["label"] for row in rows}
        for shape in shapes:
            if f"{shape}_final" in present:
                selections.append((directory, f"{shape}_final"))
    if getattr(args, "include_best", False):
        for directory in args.runs:
            if not (directory / "summary.json").exists():
                continue
            rows = json.loads((directory / "manifest.json").read_text())
            blocks = json.loads((directory / "summary.json").read_text())
            for shape in shapes:
                eligible = {f"{shape}_seed"}
                eligible.update(f"{shape}_{b['block']}_B" for b in blocks
                                if b["shape"] == shape and b["accepted"] and
                                b.get("acceptance") in {"sequential_ccd", "nonmonotone_ccd"})
                selected = min((r for r in rows if r["label"] in eligible),
                               key=lambda r: r["metrics"]["cv"])
                selections.append((directory, selected["label"]))
    return selections


def replay(args):
    prepare_camera_sdk(getattr(args, "camera", "daheng"))
    from ao_shaping.drivers.ccd.common import create_camera, get_camera_exposure_ms, get_camera_exposure_range
    from ao_shaping.drivers.slm.santec import Santec
    from ao_shaping.utils.image.exposure_control import configured_daheng_exposure_limit

    sources = []
    for directory, label in _selections(args):
        rows = json.loads((directory / "manifest.json").read_text())
        row = next(r for r in rows if r["label"] == label)
        with np.load(directory / row["file"]) as saved:
            phase = saved["model_phase"].copy()
            grid, target, core = (saved[k].copy() for k in ("camera_xy", "target", "core"))
        sources.append(dict(name=directory.name + "_" + label, phase=phase, grid=grid,
                            target=target, core=core))
    if not sources:
        raise ValueError("nothing to replay")
    # One camera frame serves every grid. The stages deliberately sample the
    # same window at different steps (stage 4 uses the finer recipe step), so a
    # run's `camera_xy` arrays are equal *per stage* and differ across stages.
    # Rejecting that made a finished multi-shape session un-replayable; instead
    # each phase is sampled through its own grid, which is exactly the
    # comparison the replay is for.
    groups: dict[tuple, list[dict]] = {}
    for source in sources:
        key = (source["grid"].shape, float(source["grid"][0, 0, 0]), float(source["grid"][0, 0, 1]))
        groups.setdefault(key, []).append(source)
    for source in sources:
        grid = source["grid"]
        if (grid[..., 0].max() >= 2592 or grid[..., 1].max() >= 1944
                or grid[..., 0].min() < 0 or grid[..., 1].min() < 0):
            raise ValueError(f"{source['name']}: camera grid extends beyond the sensor")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "config.json").write_text(json.dumps(dict(
        **vars(args), sources=[s["name"] for s in sources],
        grids={f"{key[0]}|{key[1]:.1f},{key[2]:.1f}": [s["name"] for s in members]
               for key, members in groups.items()}, **clock()), default=str, indent=2),
        encoding="utf-8")
    records = []
    with create_camera(getattr(args, "camera", "daheng"), int(getattr(args, "cam_id", 0)),
                       exposure_time_ms=.02, bit_depth=12) as camera:
        floor, ceiling = get_camera_exposure_range(camera)
        limit = configured_daheng_exposure_limit()
        if min(args.exposures) < floor or max(args.exposures) > min(ceiling, limit if limit > 0 else ceiling):
            raise ValueError("replay exposure outside current session range")
        with Santec(slm_number=int(getattr(args, "slm_number", 1)), wavelength=1064, video_mode=0) as slm:
            slm.set_phase_overlay(None)
            try:
                for exposure in args.exposures:
                    camera.reset_exposure_time(exposure)
                    for source in sources:
                        phase, grid = source["phase"], source["grid"]
                        target, core = source["target"], source["core"]
                        name = source["name"]
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
                        sample = raw.mean(0)[grid[..., 1].astype(int), grid[..., 0].astype(int)]
                        filename = f"replay_{len(records):04d}.npz"
                        np.savez_compressed(args.out / filename, raw_frames=raw, model_phase=phase,
                            panel_gray=gray, camera_xy=grid, target=target, core=core)
                        # Each source carries its own target / core, so the metrics
                        # are evaluated against the shape that was replayed.
                        shape_objective = ShapeObjective(target, cv_only=True)
                        shape_objective.core = core
                        record = dict(label=name, file=filename, exposure_ms=float(get_camera_exposure_ms(camera)),
                            full_scale=4095, bit_depth=12, counts_scale=16., frames=stamps,
                            camera_grid=list(grid.shape),
                            metrics=image_metrics(sample, shape_objective), core_peak=float(sample[core].max()), **clock())
                        records.append(record)
                        (args.out / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
                        with (args.out / "capture_records.pkl").open("wb") as stream:
                            pickle.dump(records, stream, protocol=5)
                        logger.info(json.dumps(record | dict(frames=len(stamps))))
                        if record["core_peak"] >= 4014:
                            raise RuntimeError("core saturated; raw replay retained")
            finally:
                slm.display_data(slm.create_phase_from_array(np.zeros((1200, 1920))))
                time.sleep(.3)
    (args.out / "released.json").write_text(json.dumps(dict(status="camera and SLM contexts closed", **clock()), indent=2), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-run", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exposures", type=float, nargs="+", default=[50., 100.])
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--camera", default="daheng")
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--include-best", action="store_true", help="Also replay the lowest measured accepted sequential candidate from each run")
    replay(parser.parse_args())
