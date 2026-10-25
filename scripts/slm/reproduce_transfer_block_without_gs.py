"""Offline reproduction of one anchored block, seeded from an accepted phase (no GS).

Why this exists
---------------
The transfer pipeline's cold start runs ``gs_seed()`` to synthesize the first phase,
then optimizes it against the CCD-anchored local objective. That makes a reproduction
conflate two things: the Gerchberg-Saxton initialization and the differentiable
optimization. This script removes the GS term entirely:

* the seed comes from an **already accepted phase of a previous run**
  (``<seed-run>/<shape>_final``), exactly what ``validate.py --seed-run`` does;
* the reference irradiance and geometry come from the **archived A1 capture** of the
  run being reproduced, so no camera is opened;
* the update loop is the production loop (same lr, same step RMS cap, same
  ``anchored_intensity`` proxy), so the candidate can be compared **bit for bit**
  against the archived ``B1`` phase.

If the candidate matches the archive, the algorithm is reproduced without GS. If it
does not, the mismatch is reported with numbers instead of a claim.

Everything here is offline: no SLM, no camera, no ``--execute``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_CHECKPOINT = "data/debug/calibrate_transfer_blur/shapes_20261023_1956/frozen_parameters.json"
DEFAULT_SEED_RUN = "data/debug/validate_transfer_shapes/paired_20261023_1942"
DEFAULT_TARGET_RUN = "data/debug/validate_transfer_shapes/anchored_20261023_2002"
SHAPES = ("square", "rectangle", "triangle", "star")


def _manifest(run_dir: Path) -> list[dict]:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def _row(records: list[dict], label: str) -> dict:
    for row in records:
        if row["label"] == label:
            return row
    raise KeyError(f"label {label!r} not in manifest")


def _load_capture(run_dir: Path, records: list[dict], label: str) -> tuple[dict, dict]:
    row = _row(records, label)
    with np.load(run_dir / row["file"]) as saved:
        payload = {key: saved[key] for key in saved.files}
    return row, payload


def _sampled(payload: dict, xy: np.ndarray) -> np.ndarray:
    frame = payload["raw_frames"].mean(0)
    return frame[xy[..., 1].astype(int), xy[..., 0].astype(int)]


def reproduce_shape(*, shape: str, block: int, updates: int, checkpoint: Path,
                    seed_run: Path, target_run: Path, lr: float,
                    step_rms_cap: float, seed_from_run: bool = True,
                    capture_frames: bool = False) -> dict:
    """Reproduce one block's candidate phase without calling ``gs_seed()``.

    ``capture_frames=True`` additionally returns ``model_frames`` (``updates + 1``
    model predictions: the seed state plus every post-update state) and
    ``measured_frames`` (the four archived CCD frames). Those keys hold numpy
    arrays and therefore are NOT JSON serialisable -- ``main()`` leaves the flag
    off so ``result.json`` keeps its exact shape.
    """
    import torch

    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        DifferentiableUniformityOptimizer, FrozenDefocusForward, FrozenDefocusParameters,
    )
    from ao_shaping.algorithm.signal_processing.multishape_objective import (
        ShapeObjective, anchored_intensity, image_metrics,
    )

    torch.set_num_threads(4)
    spec = json.loads(checkpoint.read_text(encoding="utf-8"))
    params = FrozenDefocusParameters.from_dicts(
        spec, spec["ramp_matrix"], grid_size=int(spec["grid_size"]),
        fft_size=int(spec["fft_size"]), pixel_samples=int(spec["pixel_samples"]))

    target_records = _manifest(target_run)
    seed_records = _manifest(seed_run)
    a1_row, a1 = _load_capture(target_run, target_records, f"{shape}_{block}_A1")
    b1_row, b1 = _load_capture(target_run, target_records, f"{shape}_{block}_B1")
    b2_row, b2 = _load_capture(target_run, target_records, f"{shape}_{block}_B2")
    a2_row, a2 = _load_capture(target_run, target_records, f"{shape}_{block}_A2")

    xy = np.asarray(a1["camera_xy"])
    objective = ShapeObjective(np.asarray(a1["target"]))
    forward = FrozenDefocusForward(params, xy, device="cpu")

    # --- seed: never gs_seed(). Either a previous run's accepted phase, or the phase
    # actually displayed at A1 (which is the accepted phase carried into this block).
    if seed_from_run:
        seed_row, seed_payload = _load_capture(seed_run, seed_records, f"{shape}_final")
        seed_phase = np.asarray(seed_payload["model_phase"], dtype=np.float64)
    else:
        seed_row = a1_row
        seed_phase = np.asarray(a1["model_phase"], dtype=np.float64)
    displayed = np.asarray(a1["model_phase"], dtype=np.float64)
    seed_vs_a1 = float(np.abs(seed_phase - displayed).max())

    exposure = float(a1_row["exposure_ms"])
    anchor = _sampled(a1, xy) / exposure
    with torch.no_grad():
        anchor_prediction = forward.predict(seed_phase)[0].numpy()

    optimizer = DifferentiableUniformityOptimizer(forward, objective, seed_phase, lr=lr,
                                                  step_rms_cap=step_rms_cap, device="cpu")
    losses = []
    model_frames: list[np.ndarray] = []
    for _ in range(updates):
        with torch.no_grad():
            current = forward.predict(optimizer.current_phase)[0].numpy()
        if capture_frames:
            model_frames.append(np.asarray(current, dtype=np.float64))
        optimizer.update(anchored_intensity(anchor, current, anchor_prediction))
        losses.append(float(optimizer.loss_history[-1]["loss"]))
    candidate = np.asarray(optimizer.current_phase)

    archived_b1_phase = np.asarray(b1["model_phase"], dtype=np.float64)
    phase_delta = np.abs(candidate - archived_b1_phase)
    with torch.no_grad():
        predicted_candidate = forward.predict(candidate)[0].numpy()
        predicted_seed = forward.predict(seed_phase)[0].numpy()
    # The optimizer minimizes the anchored proxy, not the pure model prediction. Report both,
    # otherwise "model CV went up" and "CCD CV went down" read as a contradiction.
    anchored_seed = anchored_intensity(anchor, predicted_seed, anchor_prediction)
    anchored_candidate = anchored_intensity(anchor, predicted_candidate, anchor_prediction)

    def _loss(image: np.ndarray) -> dict:
        image_t = torch.tensor(np.asarray(image, dtype=np.float32))
        with torch.no_grad():
            value, terms = objective.evaluate(image_t, torch.zeros_like(image_t))
        return dict(loss=float(value), **{k: float(v) if np.isscalar(v) else v
                                          for k, v in terms.items()})

    baseline = 0.5 * (float(a1_row["loss"]) + float(a2_row["loss"]))
    proposed = 0.5 * (float(b1_row["loss"]) + float(b2_row["loss"]))
    drift = abs(float(a2_row["loss"]) - float(a1_row["loss"]))
    result = dict(
        shape=shape, block=block, updates=updates, gs_calls=0,
        seed_source=f"{seed_run.name}/{shape}_final" if seed_from_run else "archived A1 phase",
        seed_exposure_ms=float(seed_row["exposure_ms"]),
        seed_cv=float(seed_row["metrics"]["cv"]),
        seed_vs_archived_a1_max_abs=seed_vs_a1,
        phase_reproduction=dict(max_abs=float(phase_delta.max()),
                                rms=float(np.sqrt((phase_delta ** 2).mean())),
                                bitwise_equal=bool(np.array_equal(candidate, archived_b1_phase))),
        predicted=dict(seed=image_metrics(predicted_seed, objective),
                       candidate=image_metrics(predicted_candidate, objective),
                       loss_first=losses[0], loss_last=losses[-1]),
        anchored_proxy=dict(seed=dict(metrics=image_metrics(anchored_seed, objective),
                                      **_loss(anchored_seed)),
                            candidate=dict(metrics=image_metrics(anchored_candidate, objective),
                                           **_loss(anchored_candidate))),
        ccd=dict(anchor_at_a1=a1_row["metrics"], b1=b1_row["metrics"], b2=b2_row["metrics"],
                 a2=a2_row["metrics"]),
        acceptance=dict(baseline=baseline, proposed=proposed, drift=drift,
                        accepted_archived=bool(proposed < baseline - drift)),
    )
    if capture_frames:
        model_frames.append(np.asarray(predicted_candidate, dtype=np.float64))
        result["model_frames"] = model_frames
        result["target"] = np.asarray(a1["target"], dtype=np.float64)
        result["measured_frames"] = {
            name: _sampled(payload, xy) / float(row["exposure_ms"])
            for name, row, payload in (("A1", a1_row, a1), ("B1", b1_row, b1),
                                       ("B2", b2_row, b2), ("A2", a2_row, a2))
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / DEFAULT_CHECKPOINT)
    parser.add_argument("--seed-run", type=Path, default=ROOT / DEFAULT_SEED_RUN)
    parser.add_argument("--target-run", type=Path, default=ROOT / DEFAULT_TARGET_RUN)
    parser.add_argument("--shapes", nargs="+", default=list(SHAPES))
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--seed-source", choices=("run", "a1"), default="run",
                        help="'run' = <seed-run>/<shape>_final (block 0); "
                             "'a1' = the phase actually displayed at A1 (later blocks)")
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--step-rms-cap", type=float, default=0.2)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    results = []
    available = {row["label"] for row in _manifest(args.target_run)}
    for shape in args.shapes:
        if f"{shape}_{args.block}_A1" not in available:
            print(f"[skip] {shape}: no block {args.block} in {args.target_run.name}")
            continue
        result = reproduce_shape(shape=shape, block=args.block, updates=args.updates,
                                 checkpoint=args.checkpoint, seed_run=args.seed_run,
                                 target_run=args.target_run, lr=args.lr,
                                 step_rms_cap=args.step_rms_cap,
                                 seed_from_run=args.seed_source == "run")
        results.append(result)
        repro = result["phase_reproduction"]
        proxy = result["anchored_proxy"]
        print(f"{shape:10s} seed a1Δ={result['seed_vs_archived_a1_max_abs']:.3e} "
              f"B1 phase maxΔ={repro['max_abs']:.3e} bitwise={repro['bitwise_equal']} "
              f"| proxy CV {proxy['seed']['metrics']['cv']:.4f}→"
              f"{proxy['candidate']['metrics']['cv']:.4f} "
              f"loss {proxy['seed']['loss']:.4f}→{proxy['candidate']['loss']:.4f} "
              f"| model CV {result['predicted']['seed']['cv']:.4f}→"
              f"{result['predicted']['candidate']['cv']:.4f} "
              f"| CCD CV {result['ccd']['anchor_at_a1']['cv']:.4f}→"
              f"{result['ccd']['b1']['cv']:.4f}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "result.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {args.out / 'result.json'}")


if __name__ == "__main__":
    main()

