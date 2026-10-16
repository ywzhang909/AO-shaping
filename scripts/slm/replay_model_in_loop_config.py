"""Run a saved model-in-loop configuration with a new probe seed on hardware."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

from ao_shaping.optimizer.wfless.slm_model_in_loop import (
    SlmModelInLoopConfig,
    optimize_slm_model_in_loop,
)
from ao_shaping.runners.runner_common import (
    SlmModelInLoopParams,
    save_model_in_loop_config,
    save_model_in_loop_results,
)
from ao_shaping.utils.image.exposure_control import (
    DAHENG_EXPOSURE_LIMIT_ENV,
    configured_daheng_exposure_limit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--step-a-iterations", type=int, default=0)
    parser.add_argument("--max-exposure-ms", type=float, default=None,
                        help="Session limit in ms; 0 probes the current optical path")
    parser.add_argument("--w-uniformity", type=float, default=None)
    parser.add_argument("--w-efficiency", type=float, default=None)
    args = parser.parse_args()

    source = json.loads((args.source_run / "run_config.json").read_text(encoding="utf-8"))
    names = {field.name for field in fields(SlmModelInLoopConfig)} - {"callback"}
    values = {key: value for key, value in source.items() if key in names}
    values.update(seed=args.seed, step_a_iterations=args.step_a_iterations,
                  debug_dir=str(args.output_dir.resolve()))
    if args.max_exposure_ms is not None:
        values["max_exposure_ms"] = args.max_exposure_ms
    if args.w_uniformity is not None:
        values["w_uniformity"] = args.w_uniformity
    if args.w_efficiency is not None:
        values["w_efficiency"] = args.w_efficiency
    for key in ("geometry_file", "prior_run_dir", "forward_checkpoint"):
        if values.get(key) and not Path(values[key]).is_absolute():
            values[key] = str((Path(__file__).resolve().parents[2] / values[key]).resolve())
    config = SlmModelInLoopConfig(**values)
    if config.cam_type != "daheng" or config.slm_number != 1:
        parser.error("this saved hardware replay requires Daheng CCD and SLM #1")
    env_ceiling = configured_daheng_exposure_limit()
    if env_ceiling > 0 and config.max_exposure_ms > env_ceiling:
        parser.error(
            f"saved config exposes {config.max_exposure_ms} ms, above the current "
            f"{env_ceiling} ms ceiling ({DAHENG_EXPOSURE_LIMIT_ENV})"
        )
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    save_model_in_loop_config(config, args.output_dir)
    result = optimize_slm_model_in_loop(config)
    saved = save_model_in_loop_results(
        result, SlmModelInLoopParams(),
        root_dir=args.output_dir.parent, debug=True, save_dir=args.output_dir,
    )
    print(f"status={result.status} score={result.best_score:.4f} data={saved.save_dir}")


if __name__ == "__main__":
    main()
