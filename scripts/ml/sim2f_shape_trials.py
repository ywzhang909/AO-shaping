"""Offline arbitrary-shape closed-loop trials on a calibrated bench.

The ``sim2f`` stack claims one thing this script exists to test: a bench
calibrated from the device's own measurements can stand in for the device well
enough that a *local*, **device-accepted** phase update improves an arbitrary
far-field shape.  Everything here runs offline (the camera is unreachable), on
two surrogates built from the recorded hardware session
``runs/sim2f_bench/session_hw``:

* ``nominal``   -- device = the calibrated bench, model = a bare
  :class:`~ml.sim2f.optics.Sim2f`.  The model knows **nothing** of the beam, the
  camera warp or the aberration: the pessimistic case, where only the measured
  acceptance test can keep the loop honest.
* ``calibrated``-- device = the calibrated bench **plus** a drift aberration the
  calibration never saw, model = the calibrated bench.  The realistic case:
  the model is right about the bench and wrong about today's drift.

The two are never the same object, which is the one rule a surrogate trial
cannot break: a loop that improves because ``device is model`` has measured
nothing.

Targets are arbitrary bitmaps (:func:`ml.sim2f.targets.mask_target`), not
corpus families -- the point of the loop is shapes the network was never
trained on.

Usage::

    python scripts/ml/sim2f_shape_trials.py --execute \
        --out report/shape_trials/mask_closed_loop
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from loguru import logger

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from ml.sim2f.calibrate import (
    CalibratedBench,
    CalibrateConfig,
    CalibrationResult,
    calibrate_bench,
    load_bench_session,
    save_calibration,
)
from ml.sim2f.config import RefineConfig, ShapeLoopConfig
from ml.sim2f.loop import _load_network
from ml.sim2f.losses import from_phasor
from ml.sim2f.model import network_input
from ml.sim2f.optics import Sim2f
from ml.sim2f.shape_loop import run_shape_trial
from ml.sim2f.targets import mask_target

#: Default recorded session this script calibrates against.
SESSION = REPO / "runs" / "sim2f_bench" / "session_hw"

#: Camera-space FWHM the calibrated twin measures (``report/shape_trials/sim_vs_hw``).
#: A feature thinner than ~1.5x this is a delta function to the bench, not a shape.
MIN_FEATURE_PX = 24.0

#: Drift the ``calibrated`` condition's device carries beyond the calibration.
DRIFT_RMS_RAD = 0.30


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
def _axes(ccd_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre coordinates of a ``(ccd_px, ccd_px)`` window.

    Args:
        ccd_px: Window side, pixels.

    Returns:
        ``(x, y)`` each ``(ccd_px, ccd_px)``, origin at the window centre.
    """
    axis = np.arange(ccd_px, dtype=np.float32) - (ccd_px - 1) / 2.0
    return np.broadcast_to(axis, (ccd_px, ccd_px)).copy(), np.broadcast_to(
        axis.reshape(ccd_px, 1), (ccd_px, ccd_px)
    ).copy()


def build_masks(ccd_px: int) -> dict[str, np.ndarray]:
    """The arbitrary shapes this script trials, as boolean bitmaps.

    Every feature is at least :data:`MIN_FEATURE_PX` thick on the reference
    256-px window, scaled with ``ccd_px``.

    Args:
        ccd_px: Window side, pixels.

    Returns:
        ``{name: (ccd_px, ccd_px) bool}``.
    """
    scale = float(ccd_px) / 256.0
    x, y = _axes(ccd_px)
    radius = np.hypot(x, y)
    half = ccd_px / 2.0

    ring = (radius > 44.0 * scale) & (radius < 76.0 * scale)
    # a notch, so the shape is not centrosymmetric and the sign of the phase matters
    ring &= ~((x > 0.0) & (np.abs(y) < 26.0 * scale))

    bar = 26.0 * scale
    arm = 96.0 * scale
    cross = (np.abs(x) < bar) & (np.abs(y) < arm)
    cross |= (np.abs(y) < bar) & (np.abs(x) < arm)

    frame_side = 104.0 * scale
    frame_thick = 26.0 * scale
    inner = frame_side - 2.0 * frame_thick
    frame = (np.abs(x) < frame_side) & (np.abs(y) < frame_side)
    frame &= ~((np.abs(x) < inner) & (np.abs(y) < inner))

    spot_r = 30.0 * scale
    two_spot = np.zeros_like(ring)
    for cx in (-0.36, 0.36):
        two_spot |= np.hypot(x - cx * 2 * half, y - 0.15 * 2 * half) < spot_r

    return {"ring": ring, "cross": cross, "frame": frame, "two_spot": two_spot}


def make_target(mask: np.ndarray, ccd_px: int) -> object:
    """Wrap a bitmap as a :class:`~ml.sim2f.targets.TargetSample`.

    Args:
        mask: ``(ccd_px, ccd_px)`` boolean.
        ccd_px: Window side, pixels.

    Returns:
        The target sample.
    """
    return mask_target(
        mask.astype(np.float32), ccd_px, min_feature_px=MIN_FEATURE_PX * ccd_px / 256.0
    )


# ---------------------------------------------------------------------------
# Bench construction
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Condition:
    """One (device, model) pairing of a trial.

    Attributes:
        name: ``"nominal"`` or ``"calibrated"``.
        note: What the surrogate stands for, for the report.
    """

    name: str
    note: str


CONDITIONS: tuple[Condition, ...] = (
    Condition(
        "nominal",
        "device = calibrated bench, model = bare Sim2f (no beam, no camera warp, "
        "no aberration): the pessimistic pairing",
    ),
    Condition(
        "calibrated",
        "device = calibrated bench + a drift aberration the calibration never saw, "
        "model = calibrated bench: the realistic pairing",
    ),
)


def build_benches(
    calibration: CalibrationResult,
    sim_config: object,
    condition: Condition,
    *,
    seed: int,
) -> tuple[object, object]:
    """The (device, model) pair for one condition.

    Args:
        calibration: The fitted bench state.
        sim_config: Geometry the session was recorded on.
        condition: Which pairing.
        seed: Seed for the drift the calibrated condition adds.

    Returns:
        ``(device, model)`` -- two distinct objects, always.

    Raises:
        ValueError: On an unknown condition name.
    """
    sim = Sim2f(sim_config, device="cuda" if torch.cuda.is_available() else "cpu")
    if condition.name == "nominal":
        device: object = CalibratedBench(sim, calibration)
        model: object = Sim2f(
            sim_config, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        return device, model
    if condition.name == "calibrated":
        model = CalibratedBench(sim, calibration)
        rng = np.random.default_rng(seed)
        drift_coeffs = np.zeros(12, dtype=np.float32)  # Noll 4..15
        drift_coeffs[2:] = rng.normal(0.0, 1.0, size=10).astype(np.float32)
        drift = sim.aberration(
            torch.as_tensor(drift_coeffs[None], dtype=sim.dtype, device=sim.device)
        )[0]
        drift = drift * (DRIFT_RMS_RAD / float(drift.std().clamp_min(1e-6)))
        drifted = replace(
            calibration,
            aberration=(calibration.aberration + drift.detach().cpu().numpy()).astype(
                np.float32
            ),
        )
        device = CalibratedBench(Sim2f(sim_config, device=sim.device), drifted)
        return device, model
    raise ValueError(f"unknown condition {condition.name!r}")


# ---------------------------------------------------------------------------
# Trial driver
# ---------------------------------------------------------------------------
def _net_phase(net: object, context: torch.Tensor, intensity: np.ndarray) -> np.ndarray:
    """The network's open-loop phase for one target, raw radians.

    The network was trained on unit-sum targets on the aberration-free bench,
    so the target is renormalised here and the phase is passed to
    :func:`run_shape_trial` uncomposed -- aberration handling is the loop's
    job, not the caller's.

    Args:
        net: The loaded :class:`~ml.sim2f.model.Sim2fUNet`.
        context: The checkpoint's PSF-context channel, ``(1, 1, ccd_px, ccd_px)``.
        intensity: ``(ccd_px, ccd_px)`` target intensity.

    Returns:
        ``(grid, grid)`` float32 radians.
    """
    unit = intensity.astype(np.float32)
    unit /= max(float(unit.sum()), 1e-12)
    device = next(net.parameters()).device  # type: ignore[attr-defined]
    target = torch.as_tensor(unit[None], device=device)
    with torch.no_grad():
        phasor = net(network_input(target, context))  # type: ignore[operator]
    return from_phasor(phasor)[0].detach().cpu().numpy().astype(np.float32)


def run_trials(
    calibration: CalibrationResult,
    sim_config: object,
    *,
    shapes: tuple[str, ...],
    loop_config: ShapeLoopConfig,
    seed: int,
    net: object | None = None,
    context: torch.Tensor | None = None,
    artifacts: Path | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Run every (condition, shape) trial.

    Args:
        calibration: The fitted bench state.
        sim_config: Session geometry.
        shapes: Which built-in masks to trial.
        loop_config: The loop contract.
        seed: Drift seed.
        net: Optional loaded network; with ``context`` it adds the ``net_open``
            and ``net_finetune`` arms to every trial.
        context: The checkpoint's PSF-context channel.
        artifacts: Directory for the per-arm phase dumps; ``None`` skips them.

    Returns:
        ``(rows, detail)``: flat per-arm rows for the CSV, and the per-trial
        detail (rounds included) for the JSON.
    """
    ccd_px = int(sim_config.ccd_px)  # type: ignore[attr-defined]
    masks = build_masks(ccd_px)
    rows: list[dict[str, object]] = []
    detail: dict[str, object] = {}
    for condition in CONDITIONS:
        device, model = build_benches(calibration, sim_config, condition, seed=seed)
        for shape in shapes:
            target = make_target(masks[shape], ccd_px)
            net_command = (
                _net_phase(net, context, np.asarray(target.intensity))
                if net is not None and context is not None
                else None
            )
            started = time.time()
            result = run_shape_trial(
                device, model, target, net_phase=net_command, config=loop_config
            )
            wall = time.time() - started
            key = f"{condition.name}/{shape}"
            logger.info(
                "{}: flat {:.4f} -> finetune {:.4f} ({:.1f} s)",
                key,
                result.arm("flat").final["objective"],
                result.arms[-1].final["objective"],
                wall,
            )
            for arm in result.arms:
                rows.append(
                    {
                        "condition": condition.name,
                        "shape": shape,
                        "arm": arm.arm,
                        "objective_start": round(arm.start["objective"], 6),
                        "objective_final": round(arm.final["objective"], 6),
                        "l1_start": round(arm.start["l1"], 6),
                        "l1_final": round(arm.final["l1"], 6),
                        "ee_start": round(arm.start["ee"], 6),
                        "ee_final": round(arm.final["ee"], 6),
                        "ncc_start": round(arm.start["ncc"], 6),
                        "ncc_final": round(arm.final["ncc"], 6),
                        "cv_start": round(arm.start["cv_in_support"], 6),
                        "cv_final": round(arm.final["cv_in_support"], 6),
                        "rounds": len(arm.rounds),
                        "accepted": sum(1 for r in arm.rounds if r.accepted_device),
                        "wall_s": round(wall, 2),
                    }
                )
            detail[key] = {
                "kind": result.target_kind,
                "support_frac": round(float(np.asarray(target.support).mean()), 6),
                "arms": [
                    {
                        "arm": arm.arm,
                        "start": {k: round(v, 6) for k, v in arm.start.items()},
                        "final": {k: round(v, 6) for k, v in arm.final.items()},
                        "rounds": [
                            {
                                "round": r.round,
                                "accepted_model": r.accepted_model,
                                "accepted_device": r.accepted_device,
                                "gain_device": round(r.gain_device, 6),
                                "objective": round(r.objective, 6),
                                "ee": round(r.ee, 6),
                                "ncc": round(r.ncc, 6),
                                "cv_in_support": round(r.cv_in_support, 6),
                                "fit_residual_start": round(r.fit_residual_start, 6),
                                "fit_residual": round(r.fit_residual, 6),
                                "coeff_rms_rad": round(r.coeff_rms_rad, 6),
                                "delta_rms_rad": round(r.delta_rms_rad, 6),
                            }
                            for r in arm.rounds
                        ],
                    }
                    for arm in result.arms
                ],
            }
            if artifacts is not None:
                phase_dir = artifacts / "phases"
                phase_dir.mkdir(parents=True, exist_ok=True)
                for arm in result.arms:
                    np.save(
                        phase_dir / f"{condition.name}_{shape}_{arm.arm}.npy",
                        arm.phase.astype(np.float32),
                    )
                    # Per-round snapshots replay the optimisation as a sequence
                    # of camera frames; the final dump above is the last one.
                    for record in arm.rounds:
                        if record.phase is not None:
                            np.save(
                                phase_dir / f"{condition.name}_{shape}_{arm.arm}"
                                f"_r{record.round}.npy",
                                record.phase,
                            )
    return rows, detail


def main(argv: list[str] | None = None) -> int:
    """Calibrate the recorded session, run the trials, write the artifacts.

    Args:
        argv: Argument vector; ``None`` reads ``sys.argv``.

    Returns:
        ``0`` on success.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--execute", action="store_true", help="actually run")
    parser.add_argument("--session", default=str(SESSION), help="recorded session")
    parser.add_argument(
        "--out", default="report/shape_trials/mask_closed_loop", help="output directory"
    )
    parser.add_argument("--shapes", default="ring,cross,frame,two_spot")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--refine-iters", type=int, default=120)
    parser.add_argument("--fit-iters", type=int, default=80)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--calib-iters", type=int, default=300)
    parser.add_argument(
        "--checkpoint",
        default=str(REPO / "runs" / "sim2f_fair" / "train" / "model_best.pt"),
        help="trained network for the net_open/net_finetune arms",
    )
    args = parser.parse_args(argv)
    if not args.execute:
        logger.info("dry run: pass --execute to run the trials")
        return 0

    out = Path(args.out)
    artifacts = out / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    session = load_bench_session(args.session)
    logger.info(
        "session {}: {} samples, backend {}, grid {} ccd_px {}",
        session.root.name,
        len(session.samples),
        session.backend,
        session.sim_config.grid,
        session.sim_config.ccd_px,
    )
    sim = Sim2f(
        session.sim_config, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    started = time.time()
    calibration = calibrate_bench(
        sim, session, CalibrateConfig(iters=args.calib_iters, seed=args.seed)
    )
    logger.info(
        "calibration ok={} (n_train={}, n_holdout={}, aberration rms {:.3f} rad) ({:.1f} s)",
        calibration.ok,
        calibration.n_train,
        calibration.n_holdout,
        float(np.sqrt(np.mean(calibration.aberration**2))),
        time.time() - started,
    )
    save_calibration(artifacts / "calibration", calibration)
    if not calibration.ok:
        logger.warning(
            "calibration did not pass its holdout gate: {}", calibration.reason
        )

    net = None
    context = None
    checkpoint = Path(args.checkpoint)
    if checkpoint.is_file():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net, context, net_cfg, _ = _load_network(checkpoint, device)
        if (net_cfg.grid, net_cfg.ccd_px) != (
            session.sim_config.grid,
            session.sim_config.ccd_px,
        ):
            raise ValueError(
                f"checkpoint geometry ({net_cfg.grid}/{net_cfg.ccd_px}) does not "
                f"match the session ({session.sim_config.grid}/"
                f"{session.sim_config.ccd_px})"
            )
        logger.info("checkpoint {}: net_open/net_finetune arms enabled", checkpoint)
    else:
        logger.info(
            "no checkpoint at {}: running the flat/opt_finetune arms", checkpoint
        )

    loop_config = ShapeLoopConfig(
        rounds=args.rounds,
        refine=RefineConfig(iters=args.refine_iters),
        fit_iters=args.fit_iters,
        seed=args.seed,
    )
    shapes = tuple(name.strip() for name in args.shapes.split(",") if name.strip())
    rows, detail = run_trials(
        calibration,
        session.sim_config,
        shapes=shapes,
        loop_config=loop_config,
        seed=args.seed,
        net=net,
        context=context,
        artifacts=artifacts,
    )

    csv_path = artifacts / "trials.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "session": str(args.session),
        "backend": session.backend,
        "calibration": {
            "ok": calibration.ok,
            "reason": calibration.reason,
            "n_train": calibration.n_train,
            "n_holdout": calibration.n_holdout,
            "score": {k: vars(v) for k, v in calibration.score.items()},
            "score_start": {k: vars(v) for k, v in calibration.score_start.items()},
            "beam": calibration.beam,
            "camera": calibration.camera,
            "aberration_rms_rad": round(
                float(np.sqrt(np.mean(calibration.aberration**2))), 6
            ),
        },
        "config": loop_config.to_dict(),
        "trials": detail,
    }
    (artifacts / "trials.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("wrote {} and {}", csv_path.name, "trials.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
