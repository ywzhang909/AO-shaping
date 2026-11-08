"""Evaluate differentiable CCD-grid registration on archived frames (issue #237).

Offline only: no SLM, camera, or laser is opened, and no ``ao_shaping.drivers``
module is imported. The evaluation uses the frozen fit and archived NPZ frames
from ``registration_gain_probe_20261107.py`` on CPU.

The archived probe applied ``scipy.ndimage.shift`` to completed output arrays.
This evaluation instead applies its subpixel ``(dy, dx)`` directly in CCD pixels
to the differentiable camera sampling grid. These are deliberately different
operators; deviations from the archived metrics are reported, not hidden by
changing units or camera-grid semantics.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    FrozenDefocusForward,
)

PROBE_PATH = ROOT / "scripts" / "slm" / "registration_gain_probe_20261107.py"
ARCHIVED_RESULTS = ROOT / "data" / "debug" / "slm_transfer" / "registration_gain_probe_20261107" / "results.json"
DEFAULT_OUTPUT = ROOT / "data" / "debug" / "slm_transfer" / "registration_eval_20261108" / "results.json"
DEVICE_STATEMENT = (
    "离线分析，未占用任何设备：未打开 SLM / 相机 / 激光，未 import 任何 "
    "ao_shaping.drivers 模块；全部输入来自磁盘归档 NPZ 帧与冻结拟合 JSON，"
    "前向与指标计算均在 CPU 上确定性执行。"
)


def _load_probe() -> ModuleType:
    """Load the archived data/metric helpers without running its CLI."""
    spec = importlib.util.spec_from_file_location("registration_gain_probe_helpers", PROBE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load archived probe helpers: {PROBE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _predict(probe: ModuleType, forward: FrozenDefocusForward, frames: list[object]) -> dict[str, np.ndarray]:
    return {frame.key: probe._predict(forward, frame.phase) for frame in frames}


def evaluate(output: Path) -> dict[str, object]:
    """Run both registration arms and write a transparent archived comparison."""
    probe = _load_probe()
    archived = json.loads(ARCHIVED_RESULTS.read_text(encoding="utf-8"))
    target = archived["by_sign"]["-1"]
    registration = target["registration"]
    shift = (float(registration["subpixel_dy"]), float(registration["subpixel_dx"]))
    fit = probe._TRANSFER.load_fit()
    base_forward, offset = probe._TRANSFER.make_forward(fit, -1.0)
    registered_params = replace(base_forward.params, registration_shift_ccd_px=shift)
    registered_forward = FrozenDefocusForward(
        registered_params, base_forward.camera_xy, device="cpu"
    )
    frames = probe._load_frames(offset, np.asarray(fit["camera_crop_center"], dtype=np.float64))
    pairs = probe._TRANSFER.direction_pairs(frames["ladder_all_seqs"])
    all_frames = [*frames["ladder_pass1"], *frames["ladder_pass2"], *frames["r2_usable"]]
    baseline = _predict(probe, base_forward, all_frames)
    registered = _predict(probe, registered_forward, all_frames)

    rows: dict[str, dict[str, object]] = {}
    metric_comparisons: dict[str, dict[str, dict[str, float]]] = {}
    largest_deviation = 0.0
    for name in ("ladder_pass2", "r2_usable", "r2_heldout"):
        current_baseline = probe.evaluate(frames[name], baseline)
        current_registered = probe.evaluate(frames[name], registered)
        rows[name] = {"baseline": current_baseline, "registration": current_registered}
        archived_registered = target["evaluations"][name]["registration"]
        metric_comparisons[name] = {}
        for metric in ("full_pearson", "core_pearson", "rel_l2"):
            deviation = float(current_registered[metric] - archived_registered[metric])
            metric_comparisons[name][metric] = {
                "current": float(current_registered[metric]),
                "archived": float(archived_registered[metric]),
                "deviation": deviation,
            }
            largest_deviation = max(largest_deviation, abs(deviation))
        rows[name]["increments"] = {
            f"{metric}_delta": float(current_registered[metric] - current_baseline[metric])
            for metric in ("full_pearson", "core_pearson", "rel_l2")
        }

    current_direction = probe._direction_stats(registered, pairs)
    archived_direction = target["direction"]["registration"]
    count_deviations = {
        key: int(current_direction[key]) - int(archived_direction[key])
        for key in ("n_ge_0.5", "n")
    }
    pearson_pass = all(
        abs(item[metric]["deviation"]) <= 0.005
        for item in metric_comparisons.values()
        for metric in ("full_pearson", "core_pearson")
    )
    direction_pass = all(abs(value) <= 1 for value in count_deviations.values())

    result: dict[str, object] = {
        "schema": "registration_eval/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "command": "scripts/slm/registration_eval_20261108.py",
        "device_occupied": False,
        "device_statement": DEVICE_STATEMENT,
        "device_released": "not_applicable_no_device_opened",
        "config": {
            "device": "cpu",
            "phase_sign": -1.0,
            "registration_shift_ccd_px": list(shift),
            "registration_shift_order": "(dy, dx), down/right",
            "fit": str(probe.FIT_PATH.relative_to(ROOT)),
            "archived_results": str(ARCHIVED_RESULTS.relative_to(ROOT)),
            "archived_shift_interpolation_order": archived["config"]["shift_interpolation_order"],
            "current_shift_operator": "differentiable camera_sample_grids displacement before grid_sample",
            "archived_shift_operator": "scipy.ndimage.shift on completed prediction arrays",
            "note": "Archived subpixel values are used directly as CCD pixels; no sampled-grid conversion.",
        },
        "archived_training_peak": float(registration["integer_peak_score"]),
        "results": rows,
        "direction": {
            "current": current_direction,
            "archived": archived_direction,
            "count_deviations": count_deviations,
            "mean_cosine_deviation": float(current_direction["mean"] - archived_direction["mean"]),
        },
        "archived_comparison": {
            "registration_metrics": metric_comparisons,
            "largest_abs_metric_deviation": largest_deviation,
            "thresholds": {"pearson_abs_deviation": 0.005, "direction_count_abs_deviation": 1},
            "pearson_within_threshold": pearson_pass,
            "direction_counts_within_threshold": direction_pass,
            "passed": pearson_pass and direction_pass,
            "interpretation": (
                "Differences are expected because camera-grid displacement and output-array "
                "translation use different coordinates and boundary/interpolation semantics."
            ),
        },
    }
    imported_drivers = sorted(
        name for name in sys.modules
        if name == "ao_shaping.drivers" or name.startswith("ao_shaping.drivers.")
    )
    if imported_drivers:
        raise RuntimeError(f"offline evaluation unexpectedly imported drivers: {imported_drivers}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    result = evaluate(args.output.resolve())
    comparison = result["archived_comparison"]
    print(f"registration (dy, dx) CCD px: {result['config']['registration_shift_ccd_px']}")
    print(f"largest archived metric deviation: {comparison['largest_abs_metric_deviation']:.9g}")
    print(f"Pearson threshold passed: {comparison['pearson_within_threshold']}")
    print(f"direction-count threshold passed: {comparison['direction_counts_within_threshold']}")
    print(DEVICE_STATEMENT)
    print(f"wrote: {args.output.resolve()}")


if __name__ == "__main__":
    main()
