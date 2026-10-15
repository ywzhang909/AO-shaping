"""Interleaved CCD replay of two saved model-in-loop phases."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from ao_shaping.optimizer.wfless.slm_model_in_loop import (
    SlmModelInLoopConfig,
    _HardwareBench,
    _metrics_at,
    _quality,
)
from ao_shaping.utils.image.beam_metrics import prepare_frame, zero_order_center
from ao_shaping.utils.image.exposure_control import configured_daheng_exposure_limit


SEQUENCE = ("flat", "A", "B", "flat", "B", "A", "flat")
MATCH_FIELDS = (
    "cam_type", "cam_id", "cam_size", "slm_number", "slm_wavelength",
    "wavelength_nm", "region", "panel_span_px", "pupil_center",
    "target_shape", "target_side", "w_efficiency", "w_uniformity", "far_field_padding",
    "phase_disturbance_file",
)


@dataclass(frozen=True)
class ReplayPlan:
    config: SlmModelInLoopConfig
    sources: tuple[Path, Path]
    phases: dict[str, np.ndarray]
    output_dir: Path
    fixed_shaped_exposure_ms: float | None = None
    fixed_flat_exposure_ms: float | None = None


def load_plan(
    source_a: Path, source_b: Path, output_dir: Path,
    fixed_shaped_exposure_ms: float | None = None,
    fixed_flat_exposure_ms: float | None = None,
) -> ReplayPlan:
    """Validate saved phases and bench settings before opening any device."""
    sources = (Path(source_a).resolve(), Path(source_b).resolve())
    configs = [json.loads((source / "run_config.json").read_text(encoding="utf-8"))
               for source in sources]
    for name in MATCH_FIELDS:
        if configs[0][name] != configs[1][name]:
            raise ValueError(f"incompatible saved runs: {name}")
    if configs[0]["cam_type"] != "daheng":
        raise ValueError("replay currently requires Daheng CCD")
    phases = {name: np.load(source / "best_phase.npy", allow_pickle=False)
              for name, source in zip(("A", "B"), sources)}
    shape = (int(configs[0]["region"]),) * 2
    for name, phase in phases.items():
        if phase.shape != shape or not np.issubdtype(phase.dtype, np.number):
            raise ValueError(f"{name} phase must be a numeric {shape} array")
        if not np.isfinite(phase).all():
            raise ValueError(f"{name} phase contains non-finite values")
    config_keys = {field.name for field in fields(SlmModelInLoopConfig)}
    values = {key: value for key, value in configs[0].items() if key in config_keys}
    values["pupil_center"] = tuple(values["pupil_center"])
    if "frozen_modes" in values:
        values["frozen_modes"] = tuple(values["frozen_modes"])
    env_ceiling = configured_daheng_exposure_limit()
    if env_ceiling > 0:
        # The current session's ceiling wins when it is the safer (lower) one;
        # a saved explicit ceiling stays authoritative when no env is set, and
        # a saved 0 means "auto" (resolved by probe at run time).
        values["max_exposure_ms"] = min(float(values["max_exposure_ms"]), env_ceiling)
    config = SlmModelInLoopConfig(**values)
    if config.max_exposure_ms > 0:
        if fixed_shaped_exposure_ms is not None and not 0 < fixed_shaped_exposure_ms <= config.max_exposure_ms:
            raise ValueError("fixed shaped exposure exceeds the Daheng limit")
        if fixed_flat_exposure_ms is not None and not 0 < fixed_flat_exposure_ms <= config.max_exposure_ms:
            raise ValueError("fixed flat exposure exceeds the Daheng limit")
    return ReplayPlan(config, sources, phases, Path(output_dir),
                      fixed_shaped_exposure_ms, fixed_flat_exposure_ms)


def run_replay(plan: ReplayPlan) -> list[dict[str, float | int | str]]:
    """Capture an ABBA comparison with flat references and saved raw frames."""
    plan.output_dir.mkdir(parents=True, exist_ok=False)
    (plan.output_dir / "config.json").write_text(
        json.dumps({"sources": [str(source) for source in plan.sources],
                    "sequence": SEQUENCE, "config": {
                        name: getattr(plan.config, name) for name in MATCH_FIELDS},
                    "max_exposure_ms": plan.config.max_exposure_ms,
                    "fixed_shaped_exposure_ms": plan.fixed_shaped_exposure_ms,
                    "fixed_flat_exposure_ms": plan.fixed_flat_exposure_ms},
                   indent=2, default=list), encoding="utf-8",
    )
    flat = np.zeros_like(plan.phases["A"], dtype=np.float64)
    phases = {**plan.phases, "flat": flat}
    rows: list[dict[str, float | int | str]] = []
    bench = _HardwareBench(plan.config)
    try:
        center: tuple[int, int] | None = None
        for index, name in enumerate(SEQUENCE):
            bench.display(phases[name])
            fixed_exposure = (plan.fixed_flat_exposure_ms if name == "flat"
                              else plan.fixed_shaped_exposure_ms)
            if fixed_exposure is not None:
                frame = prepare_frame(
                    bench.measure_fixed_exposure(fixed_exposure)
                )
            else:
                frame = prepare_frame(bench.measure())
            if center is None:
                center = zero_order_center(frame, refine=False)
            metrics = _metrics_at(
                frame, plan.config.target_side, center,
                target_shape=plan.config.target_shape,
                camera_rotation_quarters=1,
            )
            path = plan.output_dir / f"{index:02d}_{name}"
            np.save(path.with_suffix(".npy"), frame)
            if bench.last_raw_frame is not None:
                np.save(plan.output_dir / f"{index:02d}_{name}_raw.npy", bench.last_raw_frame)
            row = {
                "index": index, "phase": name,
                "time_local": datetime.now().astimezone().isoformat(),
                "capture_utc": bench.last_capture_utc,
                "exposure_ms": float(bench.last_exposure_ms),
                "score": _quality(metrics, plan.config.w_efficiency,
                                  plan.config.w_uniformity),
                "cv": float(metrics["uniformity_cv"]),
                "ee": float(metrics["encircled_energy"]),
                "total": float(frame.sum()),
                "peak": float(frame.max()),
                "raw_peak": float(np.max(bench.last_raw_frame)),
                "center_x": center[0], "center_y": center[1],
            }
            rows.append(row)
            (plan.output_dir / "measurements.json").write_text(
                json.dumps(rows, indent=2), encoding="utf-8"
            )
            print(row, flush=True)
    finally:
        try:
            bench.display(flat)
            if plan.fixed_flat_exposure_ms is None:
                bench.measure()
            else:
                bench.measure_fixed_exposure(plan.fixed_flat_exposure_ms)
        finally:
            bench.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_a", type=Path)
    parser.add_argument("source_b", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--fixed-shaped-exposure-ms", type=float)
    parser.add_argument("--fixed-flat-exposure-ms", type=float)
    args = parser.parse_args()
    try:
        plan = load_plan(args.source_a, args.source_b, args.output_dir,
                         args.fixed_shaped_exposure_ms, args.fixed_flat_exposure_ms)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    run_replay(plan)


if __name__ == "__main__":
    main()
