"""One session directory for capture, transfer, shaping and independent audit."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .capture import clock


@dataclass(frozen=True)
class TransferSession:
    out: Path
    dataset: Path | None = None
    checkpoint: Path | None = None
    shapes: tuple[str, ...] = ("square", "rectangle", "triangle")
    count: int = 24
    iterations: int = 60
    blocks: int = 4
    updates: int = 10
    frames: int = 3
    side: float = 80.
    period: float = 64.
    seed: int = 102319
    camera: str = "daheng"
    cam_id: int = 0
    slm_number: int = 1
    calibration: bool = True

    def validate(self):
        if min(self.count, self.frames) < 3 or self.count < 6 or self.iterations < 0:
            raise ValueError("count >= 6, frames >= 3 and iterations >= 0 required")
        if min(self.blocks, self.updates) < 1 or self.side <= 0 or self.period < 4:
            raise ValueError("invalid target or optimization parameters")
        if not self.shapes or len(set(self.shapes)) != len(self.shapes) or not set(self.shapes) <= {"square", "rectangle", "triangle", "star"}:
            raise ValueError("unique supported shapes required")
        if self.dataset and not (self.dataset / "manifest.json").is_file():
            raise FileNotFoundError(self.dataset / "manifest.json")
        if self.checkpoint and not self.checkpoint.exists():
            raise FileNotFoundError(self.checkpoint)
        if self.calibration and not {"square", "triangle", "rectangle"}.issubset(self.shapes):
            raise ValueError("calibration requires square/triangle/rectangle; use --skip-calibration for other selections")

    def plan(self, offline=False):
        self.validate()
        return dict(settings=asdict(self), opens_devices=not offline,
                    stages=(["reuse_capture"] if self.dataset else ["capture"])
                    + ([] if self.checkpoint else ["nominal_initialization"])
                    + ["fit"]
                    + (["GS_and_differentiable_synthesis"] if offline else ["CCD_model_blocks"]
                       + (["exposure_response_capture", "exposure_response_analysis", "blur_calibration"] if self.calibration else [])
                       + ["CCD_anchored_blocks", "raw_data_audit", "summary", "direction_check"]),
                    note="Device stages require --execute; nominal cold start is not a calibrated model")


def run_session(config: TransferSession, *, offline=False):
    """No subprocess chains; each completed/failed stage is journaled immediately."""
    config.validate()
    if offline and config.dataset is None:
        raise ValueError("--offline requires a recorded --dataset")
    out = config.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "session.json").write_text(json.dumps(dict(**config.plan(offline), **clock()), default=str, indent=2), encoding="utf-8")
    journal = []

    def stage(name, operation):
        entry = dict(stage=name, status="running", **clock())
        journal.append(entry)
        write = lambda: (out / "journal.json").write_text(json.dumps(journal, default=str, indent=2), encoding="utf-8")
        write()
        print(f"[{name}] started", flush=True)
        try:
            result = operation()
        except BaseException as exc:
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}", end=clock())
            write()
            raise
        entry.update(status="complete", end=clock())
        write()
        return result

    from . import bootstrap, capture, fit, validate
    dataset = config.dataset.resolve() if config.dataset else out / "capture"
    if config.dataset is None:
        stage("capture", lambda: capture.capture(SimpleNamespace(out=dataset, camera=config.camera,
              cam_id=config.cam_id, slm_number=config.slm_number, grid=128, count=config.count,
              frames=config.frames, seed=config.seed, execute=True)))
    checkpoint = config.checkpoint.resolve() if config.checkpoint else out / "initial/frozen_parameters.json"
    if config.checkpoint is None:
        def initialize():
            params, provenance = bootstrap.bootstrap(dataset)
            checkpoint.parent.mkdir()
            checkpoint.write_text(json.dumps(params, indent=2), encoding="utf-8")
            (checkpoint.parent / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        stage("nominal_initialization", initialize)
    stage("fit", lambda: fit.fit(SimpleNamespace(dataset=dataset, checkpoint=checkpoint,
          out=out / "fit", iterations=config.iterations, span=120)))
    checkpoint = out / "fit/frozen_parameters.json"
    rows = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    reference = next(r for r in rows if r["label"] == "reference_start" and fit.valid(r))
    zero = np.round(fit.spot(reference)).tolist()
    options = dict(checkpoint=checkpoint, zero_xy=zero, period=config.period, side=config.side,
                   shapes=list(config.shapes), updates=config.updates, frames=config.frames,
                   camera=config.camera, cam_id=config.cam_id, slm_number=config.slm_number)
    if offline:
        def synthesize():
            from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
                FrozenDefocusForward, load_frozen_parameters, recentre_window,
            )
            params = load_frozen_parameters(checkpoint)
            xy = recentre_window(tuple(zero), half_span_px=220., step_px=2.)
            forward = FrozenDefocusForward(params, xy, device="cpu")
            center = np.asarray(zero)+np.asarray(params.ramp_matrix)[:, 0]/config.period
            for shape in config.shapes:
                phase = validate.gs_seed(forward, center, config.side, shape)
                target = validate.target_at(xy, shape, center, config.side)
                objective = validate.ShapeObjective(target)
                from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import DifferentiableUniformityOptimizer
                optimizer = DifferentiableUniformityOptimizer(forward, objective, phase, lr=.01, step_rms_cap=.2, device="cpu")
                for _ in range(config.updates):
                    optimizer.update()
                if not np.isfinite(optimizer.current_phase).all():
                    raise RuntimeError("nonfinite offline optimized phase")
                np.savez_compressed(out / f"{shape}_offline_seed.npz", model_phase=phase,
                                    optimized_phase=optimizer.current_phase,
                                    prediction=forward.predict(phase).detach().numpy(),
                                    optimized_prediction=forward.predict(optimizer.current_phase).detach().numpy(),
                                    target=target, camera_xy=xy)
                (out / f"{shape}_offline_iterations.json").write_text(json.dumps(optimizer.loss_history, indent=2), encoding="utf-8")
        stage("GS_and_differentiable_synthesis", synthesize)
    else:
        from . import direction, summary
        from .artifacts import audit_run
        stage("CCD_model_blocks", lambda: validate.run(SimpleNamespace(**options, out=out / "model",
              seed_run=None, mode="model", blocks=2, execute=True)))
        if config.calibration:
            from . import blur, capture_response, response
            stage("exposure_response_capture", lambda: capture_response.capture(SimpleNamespace(
                  source=out / "model", out=out / "response", shape="square", frames=config.frames,
                  camera=config.camera, cam_id=config.cam_id, slm_number=config.slm_number)))
            stage("exposure_response_analysis", lambda: response.analyze(out / "response", out / "response_analysis"))
            stage("blur_calibration", lambda: blur.calibrate(SimpleNamespace(source=out / "model",
                  checkpoint=checkpoint, out=out / "blur", sigmas=[0., .5, 1., 2., 3.515916109085083])))
            options["checkpoint"] = out / "blur/frozen_parameters.json"
        stage("CCD_anchored_blocks", lambda: validate.run(SimpleNamespace(**options, out=out / "anchored",
              seed_run=out / "model", mode="anchored", blocks=config.blocks, execute=True)))
        def audit():
            result = audit_run(out / "anchored")
            (out / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        stage("raw_data_audit", audit)
        stage("summary", lambda: summary.summarize(out / "anchored", out / "summary"))
        stage("direction_check", lambda: direction.analyze(out / "anchored", out / "direction"))
    print(f"Session complete: {out}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        p = commands.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--dataset", type=Path)
        p.add_argument("--checkpoint", type=Path)
        p.add_argument("--shapes", nargs="+", default=["square", "rectangle", "triangle"])
        for key, default in (("count", 24), ("iterations", 60), ("blocks", 4), ("updates", 10), ("frames", 3), ("seed", 102319), ("cam-id", 0), ("slm-number", 1)):
            p.add_argument("--"+key, type=int, default=default)
        p.add_argument("--camera", default="daheng")
        p.add_argument("--side", type=float, default=80.)
        p.add_argument("--period", type=float, default=64.)
        p.add_argument("--skip-calibration", dest="calibration", action="store_false")
        mode = p.add_mutually_exclusive_group()
        mode.add_argument("--offline", action="store_true")
        mode.add_argument("--execute", action="store_true")
    replay = commands.add_parser("replay", help="Audit and recompute saved measurements, without devices")
    replay.add_argument("--run", type=Path, required=True)
    replay.add_argument("--out", type=Path, required=True)
    audit = commands.add_parser("audit", help="Check raw frames, exposures, phases, masks and A/B decisions")
    audit.add_argument("--run", type=Path, required=True)
    audit.add_argument("--out", type=Path)
    doctor = commands.add_parser("doctor", help="Enumerate Daheng without opening a stream or accessing SLM")
    doctor.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "doctor":
        from .hardware import enumerate_camera
        result = enumerate_camera()
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, default=str, indent=2)
        print(json.dumps(result, default=str, indent=2))
        return
    if args.command in {"audit", "replay"}:
        from .artifacts import audit_run
        result = audit_run(args.run)
        if args.command == "replay":
            from . import direction, summary
            args.out.mkdir(parents=True, exist_ok=False)
            (args.out / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            summary.summarize(args.run, args.out / "summary")
            direction.analyze(args.run, args.out / "direction")
        elif args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2)
        print(json.dumps({k: v for k, v in result.items() if k != "file_sha256"}, indent=2))
        return
    config = TransferSession(**{k: v for k, v in vars(args).items() if k not in {"command", "offline", "execute"}})
    if args.command == "plan" or not (args.execute or args.offline):
        print(json.dumps(config.plan(args.offline), default=str, indent=2))
    else:
        # The CLI owns its logging; keep INFO on stdout so PowerShell does not
        # turn routine loguru stderr messages into NativeCommandError records.
        import sys
        from loguru import logger
        logger.remove()
        logger.add(sys.stdout, level="INFO", colorize=False)
        run_session(config, offline=args.offline)
