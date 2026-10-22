"""Paired GS-refine / model-in-loop comparison with a fixed SLM aberration.

Run from the repository root with ``python -m ao_shaping.tools.slm.compare_disturbance``.
Each optimizer opens and closes its own devices. The same saved panel map is
added to *every* phase in both disturbed runs, including their flat baseline.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ao_shaping.optimizer.wfless.slm_gs_refine import (
    SlmGsRefineConfig, optimize_slm_gs_refine,
)
from ao_shaping.optimizer.wfless.slm_model_in_loop import (
    SlmModelInLoopConfig, _metrics_at, _quality, optimize_slm_model_in_loop,
)
from ao_shaping.runners.runner_common import (
    SlmModelInLoopParams, parse_point, save_model_in_loop_results,
)
from ao_shaping.tools.kernels.bench_kernels import SLM_PANEL_H, SLM_PANEL_W
from ao_shaping.utils.image.beam_metrics import zero_order_center
from ao_shaping.utils.image.exposure_control import (
    DAHENG_EXPOSURE_LIMIT_ENV,
    configured_daheng_exposure_limit,
)
from ao_shaping.utils.io.file import save_recorder_debug_artifacts
from ao_shaping.utils.slm.phase_disturbance import zernike_panel_disturbance
from loguru import logger

from ao_shaping.utils.io.recorder_keys import (
    RECORD_IMAGE,
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _common_quality(
    frame: np.ndarray, target_side: int, center: tuple[int, int]
) -> dict[str, float]:
    metrics = _metrics_at(np.asarray(frame), target_side, center)
    return {
        "score": float(_quality(metrics, 0.6, 0.4)),
        "cv": float(metrics["uniformity_cv"]),
        "ee": float(metrics["encircled_energy"]),
    }


def _disturbance(
    seed: int, rms_rad: float, center: tuple[int, int] = (960, 720),
    radius: int = 450,
) -> tuple[dict[int, float], np.ndarray]:
    rng = np.random.default_rng(seed)
    coefficients = {noll: float(rng.normal()) for noll in range(4, 16)}
    panel = zernike_panel_disturbance(
        coefficients, (SLM_PANEL_H, SLM_PANEL_W), center, radius
    )
    x, y = center
    illuminated = panel[y - radius:y + radius, x - radius:x + radius]
    inside = illuminated != 0
    scale = rms_rad / max(float(np.std(illuminated[inside])), 1e-12)
    panel *= scale
    coefficients = {noll: value * scale for noll, value in coefficients.items()}
    return coefficients, panel


def run_comparison(args: argparse.Namespace) -> Path:
    if args.exposure_ms <= 0:
        raise ValueError("Daheng exposure must be positive")
    env_ceiling = configured_daheng_exposure_limit()
    if env_ceiling > 0 and args.exposure_ms > env_ceiling:
        raise ValueError(
            f"Daheng exposure {args.exposure_ms} ms exceeds the {env_ceiling} ms "
            f"ceiling ({DAHENG_EXPOSURE_LIMIT_ENV})"
        )
    if args.rms_rad <= 0:
        raise ValueError("disturbance RMS must be positive")
    center = parse_point(args.pupil_center)
    radius = int(args.beam_radius_px)
    coefficients, disturbance = _disturbance(args.seed, args.rms_rad, center, radius)
    root = Path(args.output) / datetime.now().strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=False)
    disturbance_file = root / "disturbance_panel_rad.npy"
    np.save(disturbance_file, disturbance)
    (root / "disturbance.json").write_text(
        json.dumps({"seed": args.seed, "noll_coefficients_rad": coefficients,
                    "rms_rad": args.rms_rad, "center_xy": list(center),
                    "radius_px": radius, "panel_file": disturbance_file.name}, indent=2),
        encoding="utf-8",
    )
    # ABBA order reduces a monotonic bench drift's confounding with condition.
    runs = (("gs_refine", False), ("model_in_loop", False),
            ("model_in_loop", True), ("gs_refine", True))
    manifest = {"started_utc": _utc(), "runs": [], "settings": vars(args)}
    for algorithm, disturbed in runs:
        condition = "disturbed" if disturbed else "clean"
        run_dir = root / f"{len(manifest['runs']) + 1:02d}_{algorithm}_{condition}"
        run_dir.mkdir()
        phase_file = str(disturbance_file) if disturbed else None
        started = _utc()
        tick = perf_counter()
        entry = {"algorithm": algorithm, "condition": condition,
                 "started_utc": started, "directory": run_dir.name}
        manifest["runs"].append(entry)
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        try:
            if algorithm == "gs_refine":
                config = SlmGsRefineConfig(
                    epochs=args.gs_epochs, seed=args.seed, target_side=args.target_side,
                    pupil_center=center, beam_radius_px=radius,
                    cam_type="daheng", cam_id=0, exposure_time_ms=args.exposure_ms,
                    cam_size=256,
                    slm_number=1, slm_wavelength=1064, record_debug=True,
                    phase_disturbance_file=phase_file,
                )
                recorder = optimize_slm_gs_refine(config)
                save_recorder_debug_artifacts(
                    recorder, str(root.parent.parent), "slm_gs_refine", output_dir=run_dir,
                    scalar_keys=("score", "pib", "cv", "pos", "neg", "epoch"),
                    text_keys=("timestamp_utc", "pos_timestamp_utc", "neg_timestamp_utc"),
                    img_keys=("_img", "_phase", "_img_pos", "_phase_pos",
                              "_img_neg", "_phase_neg"),
                    top_right_key="_phase", json_payload=asdict(config),
                )
                np.save(run_dir / "best_phase_rad.npy", recorder.best_phase)
                np.save(run_dir / "best_frame.npy", recorder.best_frame)
                entry["algorithm_best_score"] = float(max(row["score"] for row in recorder.history))
                entry["iterations"] = sum(row["stage"] == "refine" for row in recorder.history)
                best_frame = recorder.best_frame
                flat_frame = recorder.history[0][RECORD_IMAGE]
            else:
                config = SlmModelInLoopConfig(
                    n_rounds=args.model_rounds, seed=args.seed,
                    probe_count=args.probes, step_a_iterations=args.step_a_iters,
                    step_b_iterations=args.step_b_iters, target_side=args.target_side,
                    pupil_center=center, panel_span_px=2 * radius,
                    cam_type="daheng", cam_id=0,
                    exposure_time_ms=args.exposure_ms,
                    max_exposure_ms=args.exposure_ms, slm_number=1, slm_wavelength=1064,
                    cam_size=256, far_field_padding=3,
                    geometry_file=args.geometry_file, prior_run_dir=args.prior_run_dir,
                    phase_disturbance_file=phase_file,
                )
                result = optimize_slm_model_in_loop(config)
                save_model_in_loop_results(
                    result, SlmModelInLoopParams(), root_dir=root.parent.parent,
                    debug=True, save_dir=run_dir,
                )
                entry["status"] = str(result.status)
                entry["algorithm_best_score"] = float(result.best_score)
                entry["iterations"] = len([row for row in result.recorder.history if row.get("stage") == "round"])
                best_frame = result.best_frame
                flat_frame = result.probe_records[0]["frame"]
            center = zero_order_center(np.asarray(flat_frame), refine=False)
            entry["flat_center_xy"] = list(center)
            entry["common_quality"] = _common_quality(best_frame, args.target_side, center)
            entry["result"] = "completed"
        except Exception as exc:
            entry["result"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            entry["ended_utc"] = _utc()
            entry["duration_s"] = perf_counter() - tick
            (root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/debug/slm_disturbance_compare")
    parser.add_argument("--geometry-file", required=True)
    parser.add_argument("--prior-run-dir")
    parser.add_argument("--seed", type=int, default=78)
    parser.add_argument("--rms-rad", type=float, default=0.5)
    parser.add_argument("--exposure-ms", type=float, default=0.02)
    parser.add_argument("--target-side", type=int, default=100)
    parser.add_argument("--pupil-center", default="960,720",
                        help="Measured SLM panel beam center as x,y.")
    parser.add_argument("--beam-radius-px", type=int, default=450)
    parser.add_argument("--gs-epochs", type=int, default=20)
    parser.add_argument("--model-rounds", type=int, default=1)
    parser.add_argument("--probes", type=int, default=4)
    parser.add_argument("--step-a-iters", type=int, default=25)
    parser.add_argument("--step-b-iters", type=int, default=200)
    args = parser.parse_args()
    logger.info(run_comparison(args))


if __name__ == "__main__":
    main()
