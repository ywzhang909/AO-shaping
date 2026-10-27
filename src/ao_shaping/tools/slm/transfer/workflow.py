"""One session directory for capture, transfer, shaping and independent audit."""
from __future__ import annotations
from loguru import logger

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ao_shaping.utils.image.target.metrics import POLYGON_SHAPE_CHOICES

from .capture import clock
from .device import DEVICE_CHOICES, resolve_device
from .preflight import check_geometry, check_parameters


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
    calibration: bool | None = None
    adaptation: str = "auto"
    accept_metric: str = "loss"
    objective: str = "shape"
    seed_run: Path | None = None
    seed_policy: str = "strict"
    #: Optional A–B–B–A paired CCD acceptance (default off). Off: the
    #: candidate is accepted on the optimizer's own loss and no CCD verdict is
    #: measured -- the summary then says ``acceptance="model_only"``.
    abba: bool = False
    #: Write one ``sample_*.npz`` per probe. Off (default) keeps the measurements
    #: in memory and hands them straight to the fit; on, the capture writes the
    #: same keys/content as before, for the offline ``--dataset`` path and the
    #: ``scripts/slm/`` analyses.
    save_dataset: bool = False
    #: Compute device for the torch stages (fit, synthesis, CCD blocks):
    #: ``"auto"`` follows the machine (CUDA when available, else CPU), ``"cpu"``
    #: reproduces the historical numbers, ``"cuda"`` fails loudly if unavailable.
    #: The *resolved* value is what ``session.json`` and the stage configs record.
    device: str = "auto"
    initial_phase: str = "gs"
    exposure_max_steps: int = 2
    phase_lr: float = .01
    phase_rms_cap: float = .2
    adaptive_step: bool = False
    adaptive_exposure: bool = False
    #: CCD sampling window for the anchored blocks: half-span and step in
    #: native CCD pixels. The 220/2 default is a small-target setting; a
    #: 1200-px target needs a half-span of at least ~640 or its box leaves the
    #: window and the run fails the boundary check.
    window_span: float = 220.
    window_step: float = 2.
    #: Radius (native CCD px) excluded from the core metric around the zero
    #: order, and the extra erosion applied to the core beyond the window step.
    zero_radius: float = 0.
    core_margin: float | None = None
    #: Control grid of the probe capture; must tile the panel exactly.
    capture_grid: int = 128
    #: ADC bit depth of the CCD blocks (8 or 12).
    bit_depth: int = 8
    #: Control resolution used for the shaping stages. ``0`` keeps the fit's own
    #: resolution; a finer grid is a *sampling* export of the same optical
    #: parameters (geometric_seed.export_frozen_parameters), which is what the
    #: 1200-px runs use so the phase is not limited by the probe grid.
    control_grid: int = 0
    fft_size: int = 2048
    #: Learning rate of the illumination-position parameter (panel px/step).
    #: ``None`` keeps the fit default; a disturbed bench whose beam moved on the
    #: panel needs more than the default to reach the measured position.
    amplitude_center_lr: float | None = None

    @property
    def calibration_enabled(self):
        return (self.calibration if self.calibration is not None else
                {"square", "triangle", "rectangle"}.issubset(self.shapes))

    def validate(self):
        if str(self.device).strip().lower() not in DEVICE_CHOICES:
            raise ValueError(f"unknown device {self.device!r}; expected one of {list(DEVICE_CHOICES)}")
        if self.exposure_max_steps < 2 or self.phase_lr <= 0 or self.phase_rms_cap <= 0:
            raise ValueError("invalid exposure search or phase step budget")
        if self.initial_phase not in {"gs", "plane", "random"}:
            raise ValueError("unknown initial phase")
        if self.seed_run and self.initial_phase != "gs":
            raise ValueError("cold initial phase cannot use a historical seed run")
        if self.objective not in {"shape", "cv", "formation"}:
            raise ValueError("unknown objective")
        if self.accept_metric not in {"loss", "cv"}:
            raise ValueError("unknown acceptance metric")
        if self.seed_policy not in {"strict", "transfer"}:
            raise ValueError("unknown seed policy")
        if self.seed_run:
            for name in ("manifest.json", "config.json"):
                if not (self.seed_run / name).is_file():
                    raise FileNotFoundError(self.seed_run / name)
        if self.adaptation not in {"auto", "alignment", "full"}:
            raise ValueError("unknown adaptation mode")
        if min(self.count, self.frames) < 3 or self.count < 6 or self.iterations < 0:
            raise ValueError("count >= 6, frames >= 3 and iterations >= 0 required")
        if min(self.blocks, self.updates) < 1 or self.side <= 0 or self.period < 4:
            raise ValueError("invalid target or optimization parameters")
        if self.window_span <= 0 or self.window_step <= 0 or self.zero_radius < 0:
            raise ValueError("invalid camera window or zero-order exclusion")
        if self.bit_depth not in (8, 12):
            raise ValueError("bit depth must be 8 or 12")
        if self.capture_grid < 16 or 1920 % self.capture_grid or 1200 % (1920 // self.capture_grid):
            raise ValueError("capture grid must exactly tile the 1920x1200 panel")
        if self.control_grid and (1920 % self.control_grid
                                  or 1200 % (1920 // self.control_grid)
                                  or self.fft_size < self.control_grid):
            raise ValueError("control grid must tile the panel exactly and FFT must cover it")
        if not self.shapes or len(set(self.shapes)) != len(self.shapes) or not set(self.shapes) <= set(POLYGON_SHAPE_CHOICES):
            raise ValueError("unique supported shapes required")
        # Parameter-level pre-flight. A carrier period the calibration never
        # measured (or a target that cannot fit the sampling window at all) used
        # to be discovered minutes later, in the first CCD block, as "empty
        # target core" (2026-10-26: period 4 px -> 1865 px displacement).
        problems = check_parameters(
            side=float(self.side),
            period=float(self.period),
            count=int(self.count),
            iterations=int(self.iterations),
            blocks=int(self.blocks),
            updates=int(self.updates),
            frames=int(self.frames),
            shapes=tuple(self.shapes),
            half_span_px=float(self.window_span),
        )
        errors = [problem for problem in problems if problem.severity == "error"]
        if errors:
            first = errors[0]
            raise ValueError(f"{first.message} → {first.hint}")
        if self.dataset and not (self.dataset / "manifest.json").is_file():
            raise FileNotFoundError(self.dataset / "manifest.json")
        if self.checkpoint and not self.checkpoint.exists():
            raise FileNotFoundError(self.checkpoint)
        if self.calibration_enabled and not {"square", "triangle", "rectangle"}.issubset(self.shapes):
            raise ValueError("calibration requires square/triangle/rectangle; use --skip-calibration for other selections")

    def plan(self, offline=False):
        self.validate()
        note = "Device stages require --execute; nominal cold start is not a calibrated model"
        if not self.abba:
            note += ("; A–B–B–A acceptance skipped: candidates are accepted on the model loss "
                     "only and no CCD verdict is measured (summary acceptance=\"model_only\")")
        return dict(settings=asdict(self), opens_devices=not offline,
                    resolved=dict(adaptation=self.adaptation if self.checkpoint else "full",
                                  calibration=self.calibration_enabled,
                                  acceptance="paired_abba" if self.abba else "model_only",
                                  save_dataset=self.save_dataset,
                                  # The request; ``run_session`` replaces it with the
                                  # resolved device before writing session.json.
                                  device=self.device),
                    stages=(["reuse_capture"] if self.dataset else ["capture"])
                    + ([] if self.checkpoint else ["nominal_initialization"])
                    + ["fit"]
                    + (["control_resolution_export"] if self.control_grid else [])
                    + ([("transferred_phase_differentiable_synthesis" if self.seed_run else "GS_and_differentiable_synthesis" if self.initial_phase == "gs" else "cold_differentiable_synthesis")] if offline else ([] if (self.seed_run or self.initial_phase != "gs") and not self.calibration_enabled else ["CCD_model_blocks"])
                       + (["exposure_response_capture", "exposure_response_analysis", "blur_calibration"] if self.calibration_enabled else [])
                       + ["CCD_anchored_blocks", "raw_data_audit", "summary"]
                       + (["direction_check"] if self.abba else [])),
                    note=note)


def _notify(on_stage, name, status):
    """Best-effort observer call; the journal remains the source of truth."""
    if on_stage is None:
        return
    try:
        on_stage(name, status)
    except Exception as exc:  # a UI hook must never abort a measured session
        from loguru import logger

        logger.warning("transfer stage hook failed for {} ({}): {}", name, status, exc)


def run_session(config: TransferSession, *, offline=False, on_stage=None):
    """No subprocess chains; each completed/failed stage is journaled immediately.

    ``on_stage(name, status)`` is an optional observer called whenever a stage
    starts / completes / fails -- the same moment the journal line is written.
    It exists so the GUI can show progress through *this* entry point instead of
    re-implementing the stage sequence (a second orchestrator would drift).
    A raising observer must not wreck the session, so its errors are swallowed
    with a warning; the journal and the run itself stay authoritative.
    """
    config.validate()
    if offline and config.dataset is None:
        raise ValueError("--offline requires a recorded --dataset")
    out = config.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # Resolve the compute device once, here, so ``session.json`` records the one
    # the stages will actually use (``config.device`` is the *request*: "auto").
    device = resolve_device(config.device)
    plan = config.plan(offline)
    plan["resolved"]["device"] = device
    (out / "session.json").write_text(json.dumps(dict(**plan, **clock()), default=str, indent=2), encoding="utf-8")
    journal = []

    def stage(name, operation):
        entry = dict(stage=name, status="running", **clock())
        journal.append(entry)
        write = lambda: (out / "journal.json").write_text(json.dumps(journal, default=str, indent=2), encoding="utf-8")
        write()
        print(f"[{name}] started", flush=True)
        _notify(on_stage, name, "running")
        try:
            result = operation()
        except BaseException as exc:
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}", end=clock())
            write()
            _notify(on_stage, name, "failed")
            raise
        entry.update(status="complete", end=clock())
        write()
        _notify(on_stage, name, "complete")
        return result

    from . import bootstrap, capture, fit, validate
    dataset = config.dataset.resolve() if config.dataset else out / "capture"
    samples = None
    if config.dataset is None:
        samples = stage("capture", lambda: capture.capture(SimpleNamespace(out=dataset, camera=config.camera,
              cam_id=config.cam_id, slm_number=config.slm_number, grid=config.capture_grid,
              count=config.count,
              frames=config.frames, seed=config.seed, execute=True,
              exposure_max_steps=config.exposure_max_steps,
              save_dataset=config.save_dataset)))
    checkpoint = config.checkpoint.resolve() if config.checkpoint else out / "initial/frozen_parameters.json"
    if config.checkpoint is None:
        def initialize():
            params, provenance = bootstrap.bootstrap(dataset)
            checkpoint.parent.mkdir()
            checkpoint.write_text(json.dumps(params, indent=2), encoding="utf-8")
            (checkpoint.parent / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        stage("nominal_initialization", initialize)
    # The capture stage's in-memory payloads go straight into the fit; only a
    # reused (recorded) dataset is read back from disk.
    stage("fit", lambda: fit.fit(SimpleNamespace(dataset=dataset, checkpoint=checkpoint,
          out=out / "fit", iterations=config.iterations, span=120, samples=samples,
          device=device, amplitude_center_lr=config.amplitude_center_lr,
          adaptation=config.adaptation if config.checkpoint else "full")))
    checkpoint = out / "fit/frozen_parameters.json"
    if config.control_grid:
        from .geometric_seed import export_frozen_parameters

        control_out = out / f"model{config.control_grid}"
        stage("control_resolution_export", lambda: export_frozen_parameters(
            checkpoint, control_out, grid=config.control_grid, fft=config.fft_size))
        checkpoint = control_out / "frozen_parameters.json"
    rows = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    reference = next(r for r in rows if r["label"] == "reference_start" and fit.valid(r))
    zero = np.round(fit.spot(reference)).tolist()
    # Geometry pre-flight: with the *measured* zero and the *fitted* ramp gain,
    # recompute where the target lands in this session's sampling window. The
    # parameter check above cannot see that combination (it needs the measured
    # gain), and the alternative was discovering it in the first CCD block as
    # "empty target core" (2026-10-26). Fail here, with the numbers that say
    # which knob to turn.
    frozen = json.loads(checkpoint.read_text(encoding="utf-8"))
    geometry_problems = check_geometry(
        zero_xy=zero,
        ramp_matrix=np.asarray(frozen["ramp_matrix"], dtype=float),
        side=float(config.side),
        period=float(config.period),
        shapes=tuple(config.shapes),
        half_span_px=float(config.window_span),
        step_px=float(config.window_step),
    )
    geometry_errors = [problem for problem in geometry_problems if problem.severity == "error"]
    if geometry_errors:
        first = geometry_errors[0]
        raise ValueError(f"{first.message} → {first.hint}")
    options = dict(checkpoint=checkpoint, zero_xy=zero, period=config.period, side=config.side,
                   window_span=config.window_span, window_step=config.window_step,
                   zero_radius=config.zero_radius, core_margin=config.core_margin,
                   bit_depth=config.bit_depth,
                   initial_phase=config.initial_phase, random_seed=config.seed,
                   phase_lr=config.phase_lr, phase_rms_cap=config.phase_rms_cap, adaptive_step=config.adaptive_step,
                   adaptive_exposure=config.adaptive_exposure,
                   accept_metric=config.accept_metric, objective=config.objective,
                   shapes=list(config.shapes), updates=config.updates, frames=config.frames,
                   abba=config.abba, device=device,
                   camera=config.camera, cam_id=config.cam_id, slm_number=config.slm_number)
    if offline:
        def synthesize():
            from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
                FrozenDefocusForward, load_frozen_parameters, recentre_window,
            )
            params = load_frozen_parameters(checkpoint)
            xy = recentre_window(tuple(zero), half_span_px=220., step_px=2.)
            forward = FrozenDefocusForward(params, xy, device=device)
            forward.exposure_ms = float(reference["exposure_ms"])
            center = np.asarray(zero)+np.asarray(params.ramp_matrix)[:, 0]/config.period
            for shape in config.shapes:
                target = validate.target_at(xy, shape, center, config.side)
                if config.seed_run:
                    from .seeding import accepted_seed
                    phase, _ = accepted_seed(config.seed_run, shape, forward, xy, target,
                                             policy=config.seed_policy, centre=center)
                elif config.initial_phase != "gs":
                    from .cold_start import cold_phase
                    phase, _ = cold_phase(forward, center, kind=config.initial_phase, seed=config.seed)
                else:
                    phase = validate.gs_seed(forward, center, config.side, shape)
                objective = validate.ShapeObjective(target, cv_only=config.objective == "cv",
                                                     formation=config.objective == "formation")
                from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import DifferentiableUniformityOptimizer
                optimizer = DifferentiableUniformityOptimizer(forward, objective, phase,
                    lr=config.phase_lr, step_rms_cap=config.phase_rms_cap, device=device)
                for _ in range(config.updates):
                    optimizer.update()
                if not np.isfinite(optimizer.current_phase).all():
                    raise RuntimeError("nonfinite offline optimized phase")
                np.savez_compressed(out / f"{shape}_offline_seed.npz", model_phase=phase,
                                    optimized_phase=optimizer.current_phase,
                                    prediction=forward.predict(phase).detach().cpu().numpy(),
                                    optimized_prediction=forward.predict(optimizer.current_phase).detach().cpu().numpy(),
                                    target=target, camera_xy=xy, exposure_ms=forward.exposure_ms)
                (out / f"{shape}_offline_iterations.json").write_text(json.dumps(optimizer.loss_history, indent=2), encoding="utf-8")
            # Self-describing: which device actually produced these phases.
            (out / "synthesis.json").write_text(json.dumps(dict(
                device=device, device_requested=config.device, shapes=list(config.shapes),
                initial_phase=config.initial_phase, updates=config.updates, **clock()), indent=2), encoding="utf-8")
        stage("transferred_phase_differentiable_synthesis" if config.seed_run else "GS_and_differentiable_synthesis" if config.initial_phase == "gs" else "cold_differentiable_synthesis", synthesize)
    else:
        from . import direction, summary
        from .artifacts import audit_run
        if (not config.seed_run and config.initial_phase == "gs") or config.calibration_enabled:
            stage("CCD_model_blocks", lambda: validate.run(SimpleNamespace(**options, out=out / "model",
                  seed_run=config.seed_run, seed_policy=config.seed_policy, mode="model", blocks=2, execute=True)))
        if config.calibration_enabled:
            from . import blur, capture_response, response
            stage("exposure_response_capture", lambda: capture_response.capture(SimpleNamespace(
                  source=out / "model", out=out / "response", shape="square", frames=config.frames,
                  camera=config.camera, cam_id=config.cam_id, slm_number=config.slm_number)))
            stage("exposure_response_analysis", lambda: response.analyze(out / "response", out / "response_analysis"))
            stage("blur_calibration", lambda: blur.calibrate(SimpleNamespace(source=out / "model",
                  checkpoint=checkpoint, out=out / "blur", sigmas=[0., .5, 1., 2., 3.515916109085083])))
            options["checkpoint"] = out / "blur/frozen_parameters.json"
        stage("CCD_anchored_blocks", lambda: validate.run(SimpleNamespace(**options, out=out / "anchored",
              seed_run=config.seed_run if not config.calibration_enabled and (config.seed_run or config.initial_phase != "gs") else out / "model",
              seed_policy=config.seed_policy if config.seed_run and not config.calibration_enabled else "strict",
              mode="anchored", blocks=config.blocks, execute=True)))
        def audit():
            result = audit_run(out / "anchored")
            (out / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        stage("raw_data_audit", audit)
        stage("summary", lambda: summary.summarize(out / "anchored", out / "summary"))
        if config.abba:
            # The direction check reads B1/B2 and the minus probe; with the paired
            # blocks switched off those captures do not exist.
            stage("direction_check", lambda: direction.analyze(out / "anchored", out / "direction"))
    print(f"Session complete: {out}", flush=True)
    return out


def _log_to_stdout():
    """Keep the CLI's own INFO on stdout.

    PowerShell turns loguru's default stderr lines into NativeCommandError
    records, and the plan/execute output *is* the command's result (the
    ``--help``-frozen CLI contract asserts it on stdout). ``logger`` stays the
    module-level one: importing it inside a function makes it a local and
    breaks every other call in ``main``.
    """
    import sys

    logger.remove()
    logger.add(sys.stdout, level="INFO", colorize=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        p = commands.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--dataset", type=Path)
        p.add_argument("--checkpoint", type=Path)
        p.add_argument("--seed-run", type=Path, help="Start from accepted phases; never invoke GS")
        p.add_argument("--seed-policy", choices=("strict", "transfer"), default="strict")
        p.add_argument("--initial-phase", choices=("gs", "plane", "random"), default="gs")
        p.add_argument("--exposure-max-steps", type=int, default=2)
        p.add_argument("--phase-lr", type=float, default=.01)
        p.add_argument("--phase-rms-cap", type=float, default=.2)
        p.add_argument("--adaptive-step", action="store_true")
        p.add_argument("--adaptive-exposure", action="store_true")
        p.add_argument("--shapes", nargs="+", default=["square", "rectangle", "triangle"])
        for key, default in (("count", 24), ("iterations", 60), ("blocks", 4), ("updates", 10), ("frames", 3), ("seed", 102319), ("cam-id", 0), ("slm-number", 1)):
            p.add_argument("--"+key, type=int, default=default)
        p.add_argument("--camera", default="daheng")
        p.add_argument("--device", choices=DEVICE_CHOICES, default="auto",
                       help="Compute device for fit/synthesis/CCD blocks: auto = CUDA when "
                            "available else CPU; cpu reproduces the historical numbers")
        p.add_argument("--side", type=float, default=80.)
        p.add_argument("--period", type=float, default=64.)
        p.add_argument("--window-span", type=float, default=220.,
                       help="CCD window half-span (native px); a 1200-px target needs >= ~640")
        p.add_argument("--window-step", type=float, default=2.,
                       help="CCD sampling step (native px) used for the mask and metrics")
        p.add_argument("--zero-radius", type=float, default=0.,
                       help="Radius (native px) excluded from the core around the zero order")
        p.add_argument("--core-margin", type=float, default=None,
                       help="Extra erosion (native px) of the core beyond the window step")
        p.add_argument("--capture-grid", type=int, default=128,
                       help="Control grid of the probe capture (must tile the panel)")
        p.add_argument("--bit-depth", type=int, choices=(8, 12), default=8,
                       help="ADC bit depth of the CCD blocks")
        p.add_argument("--control-grid", type=int, default=0,
                       help="Export the fitted optics at this control grid before shaping "
                            "(0 = keep the fit resolution; a 1200-px target uses 640)")
        p.add_argument("--fft-size", type=int, default=2048,
                       help="FFT size of the exported control-resolution checkpoint")
        p.add_argument("--amplitude-center-lr", type=float, default=None,
                       help="Learning rate of the illumination position (panel px/step); "
                            "a disturbed bench needs more than the default to reach it")
        p.add_argument("--adaptation", choices=("auto", "alignment", "full"), default="auto")
        p.add_argument("--accept-metric", choices=("loss", "cv"), default="loss")
        p.add_argument("--objective", choices=("shape", "cv", "formation"), default="shape")
        p.add_argument("--no-abba", dest="abba", action="store_false",
                       help="Skip the A–B–B–A acceptance; accept on the model loss only "
                            "(summary acceptance=model_only, no CCD verdict)")
        p.add_argument("--abba", dest="abba", action="store_true", help="Enable optional A–B–B–A CCD acceptance")
        p.set_defaults(abba=False)
        p.add_argument("--save-dataset", action="store_true",
                       help="Write one sample_*.npz per probe (needed by the offline --dataset "
                            "path and scripts/slm analyses); off keeps them in memory")
        p.add_argument("--skip-calibration", dest="calibration", action="store_false")
        p.add_argument("--calibration", dest="calibration", action="store_true")
        p.set_defaults(calibration=None)
        mode = p.add_mutually_exclusive_group()
        mode.add_argument("--offline", action="store_true")
        mode.add_argument("--execute", action="store_true")
    replay = commands.add_parser("replay", help="Audit and recompute saved measurements, without devices")
    replay.add_argument("--run", type=Path, required=True)
    replay.add_argument("--out", type=Path, required=True)
    audit = commands.add_parser("audit", help="Check raw frames, exposures, phases, masks and A/B decisions")
    audit.add_argument("--run", type=Path, required=True)
    audit.add_argument("--out", type=Path)
    record = commands.add_parser("record", help="Rebuild a finished run's Recorder "
                                 "table and GIF; no devices")
    record.add_argument("--run", type=Path, required=True,
                        help="the anchored/ or model/ directory of a finished run")
    record.add_argument("--out", type=Path, help="output directory (default <run>/recording)")
    record.add_argument("--fps", type=float, default=2.0)
    doctor = commands.add_parser("doctor", help="Enumerate Daheng without opening a stream or accessing SLM")
    doctor.add_argument("--out", type=Path)
    simulate = commands.add_parser("simulate", help="Create known-truth linear-camera transfer probes; no devices")
    simulate.add_argument("--out", type=Path, required=True)
    simulate.add_argument("--power", type=float, default=1.2)
    simulate.add_argument("--shift", type=float, nargs=2, default=(12., -8.))
    simulate.add_argument("--count", type=int, default=12)
    simulate.add_argument("--seed", type=int, default=102419)
    pretrain = commands.add_parser("pretrain", help="Fit the low-dimensional estimator on a "
                                  "matched-model synthetic library; no devices")
    pretrain.add_argument("--out", type=Path, required=True)
    pretrain.add_argument("--combos", type=int, default=60)
    pretrain.add_argument("--probes", type=int, default=8)
    pretrain.add_argument("--fft", type=int, default=128)
    pretrain.add_argument("--grid", type=int, default=32)
    pretrain.add_argument("--alpha", type=float, default=None,
                          help="ridge penalty; default is pretrain.DEFAULT_ALPHA")
    pretrain.add_argument("--seed", type=int, default=102419)
    args = parser.parse_args()
    if args.command == "pretrain":
        from .pretrain import DEFAULT_ALPHA, pretrain as run_pretrain
        if args.probes < 2:
            parser.error("probes >= 2 required")
        result = run_pretrain(args.out, combos=args.combos, probes_per_combo=args.probes,
                              fft_size=args.fft, grid_size=args.grid,
                              alpha=DEFAULT_ALPHA if args.alpha is None else args.alpha,
                              seed=args.seed)
        logger.info(json.dumps(dict(out=str(result["out"]), combinations=result["combinations"],
                              timings=result["timings"], metrics=result["metrics"]),
                         indent=2))
        return

    if args.command == "simulate":
        from .simulation import generate_dataset
        if args.count < 6:
            parser.error("count >= 6 required")
        checkpoint = generate_dataset(args.out, power=args.power, shift=args.shift, count=args.count, seed=args.seed)
        logger.info(f"Synthetic probes complete; nominal checkpoint: {checkpoint}")
        return
    if args.command == "doctor":
        from .hardware import enumerate_camera
        result = enumerate_camera()
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, default=str, indent=2)
        logger.info(json.dumps(result, default=str, indent=2))
        return
    if args.command == "record":
        from .recording import rebuild_run

        _log_to_stdout()
        logger.info(json.dumps(rebuild_run(args.run, args.out, fps=args.fps), indent=2))
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
        logger.info(json.dumps({k: v for k, v in result.items() if k != "file_sha256"}, indent=2))
        return
    config = TransferSession(**{k: v for k, v in vars(args).items() if k not in {"command", "offline", "execute"}})
    if args.command == "plan" or not (args.execute or args.offline):
        _log_to_stdout()
        logger.info(json.dumps(config.plan(args.offline), default=str, indent=2))
    else:
        _log_to_stdout()
        run_session(config, offline=args.offline)
