"""Recompute paired-run metrics with the optical axis frozen from each flat."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np

from ao_shaping.tools.slm.compare_disturbance import _common_quality
from ao_shaping.utils.image.beam_metrics import zero_order_center
from loguru import logger

from ao_shaping.utils.io.recorder_keys import (
    RECORD_IMAGE,
)


def analyze(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for entry in manifest["runs"]:
        run = root / entry["directory"]
        if entry["algorithm"] == "gs_refine":
            with next(run.glob("*.pkl")).open("rb") as handle:
                history = pickle.load(handle)
            flat = np.asarray(history[0][RECORD_IMAGE])
            best = np.load(run / "best_frame.npy")
            timestamps = [history[index]["timestamp_utc"] for index in sorted(history)]
        else:
            with next(run.glob("*.pkl")).open("rb") as handle:
                history = pickle.load(handle)
            with np.load(run / "probes" / "round_00_flat_00.npz") as data:
                flat = data["frame"].copy()
            accepted = bool(history[1].get("accepted", 0))
            if accepted:
                with np.load(run / "probes" / "round_01_shaped_00.npz") as data:
                    best = data["frame"].copy()
            else:
                best = flat
            timestamps = [history[index].get("timestamp_utc") for index in sorted(history)]
        center = zero_order_center(flat, refine=False)
        rows.append({
            "algorithm": entry["algorithm"], "condition": entry["condition"],
            "directory": entry["directory"], "duration_s": entry["duration_s"],
            "flat_center_xy": list(center),
            "flat_quality": _common_quality(flat, manifest["settings"]["target_side"], center),
            "best_quality": _common_quality(best, manifest["settings"]["target_side"], center),
            "iteration_timestamps_utc": timestamps,
        })
    analysis = {"source_manifest": "manifest.json", "fixed_flat_roi": True, "runs": rows}
    (root / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    result = analyze(parser.parse_args().run_dir)
    logger.info(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
