"""Compare trusted local GS debug runs using their saved CCD frames.

Usage: python scripts/slm/evaluate_gs_variant_runs.py RUN_DIR ... --output summary.json

删除条件 (AGENTS.md 「scripts/ 目录」规则 3): 可重跑的离线比较器 (summary.json
写到调用者指定路径), 其 GS 变体对比结论未被任何 report/ 或 issue 引用,
规则 3① 不满足; 且 H-33 (GS → Zernike 阶数扫描, 见 TODO.md 与
report/shape_trials/gs_refine/gs_refine_zernike_hardware_plan.md) 未闭环,
GS 变体 run 可能还需复评。当 GS 变体对比结论已入
``report/shape_trials/gs_refine/`` 且存量 ``slm_gs_refine`` run 不再需要复评时, 可删。
"""

from __future__ import annotations

import argparse
from loguru import logger
import csv
import json
import pickle
from pathlib import Path

import numpy as np

from ao_shaping.utils.image.beam_metrics import compute_shape_iou
from ao_shaping.utils.image.targets import square_mask
from ao_shaping.utils.io.recorder_keys import (
    RECORD_IMAGE,
)


def summarize_run(run_dir: Path) -> dict:
    csv_file = next(run_dir.glob("slm_gs_refine_*.csv"))
    pkl_file = next(run_dir.glob("slm_gs_refine_*.pkl"))
    config_file = next(run_dir.glob("slm_gs_refine_*.json"))
    with csv_file.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    # Hardware debug pickles are trusted local experiment artifacts only.
    with pkl_file.open("rb") as stream:
        frames = pickle.load(stream)
    if len(rows) != len(frames):
        raise ValueError(f"CSV/PKL record count differs in {run_dir}")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    flat = rows[0]
    cy, cx = int(float(flat["cy"])), int(float(flat["cx"]))
    measurements = []
    for index, row in enumerate(rows):
        frame = np.asarray(frames[index][RECORD_IMAGE])
        mask = square_mask(frame.shape, cy, cx, int(float(row["target_side"])))
        measurements.append({
            "stage": row["stage"],
            "timestamp_utc": row["timestamp_utc"],
            "score": float(row["score"]),
            "pib": float(row["pib"]),
            "cv": float(row["cv"]),
            "shape_iou": compute_shape_iou(frame, mask),
            "frame_peak": float(np.max(frame)),
        })
    return {
        "run_dir": str(run_dir),
        "gs_method": config["gs_method"],
        "exposure_ms_requested": config["exposure_time_ms"],
        "target_side": config["target_side"],
        "seed": config["seed"],
        "measurements": measurements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = [summarize_run(path) for path in args.run_dirs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(args.output)


if __name__ == "__main__":
    main()
