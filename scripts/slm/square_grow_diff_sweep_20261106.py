"""Growing-square differentiable shaping sweep: 100 -> 1200 px, +100 px per side.

Why this script exists
----------------------
`transfer` blocks optimise **one** fixed target per session; the GS path and the
differentiable path were only ever compared at a single scale (side 80-1200 in
separate campaigns, each with its own fit). The open question is whether the
*differentiable* (autograd VJP on the fitted forward) path can track a target
that **grows inside one session**, i.e. whether a warm-started phase survives a
change of target scale instead of having to be rebuilt from a non-GS seed.

So: one session, one forward model, twelve targets. Each side is optimised on
the frozen forward with `DifferentiableUniformityOptimizer` (torch autograd, no
Gerchberg-Saxton anywhere). Two seeding modes are recorded in `sides.csv`:

* default -- every side is seeded by `square_transport_phase` (the non-GS,
  separable construction), then refined for `--steps` steps. Measured offline on
  this bench (2026-11-06): model cv 0.036 / 0.009 / 0.003 at side 100 / 450 /
  900 with 40 steps, pearson >= 0.97.
* `--warm-start` -- only the first side is seeded; every later side starts from
  the previous side's accepted phase. This is the "does one phase survive a
  target scale change" question. Offline it does *not* converge in 40 steps at
  lr 0.05 / cap 0.1 rad (side 450 reached cv 0.42 from a 100 px warm start,
  against 0.009 from a fresh transport seed), because the warm start carries the
  small target's phase ramp (rms 22 rad vs 101 rad needed) and 40 steps of a
  0.1 rad increment cannot rebuild it. Use it only with many more steps.

With `--execute` the phase is displayed on the SLM and measured, so each row of
`sides.csv` carries predicted *and* measured numbers.

Camera window
-------------
The window tracks the target -- ``max(220, side/2 + 150)`` px half-span -- but is
**capped by what the light path actually supports**, never by a hardcoded number:
``camera_edge_half_span`` (distance from the model origin to the CCD edge, from
the driver's full-frame size with ``--execute`` or the per-backend table offline)
and ``far_field_half_span`` (Nyquist x the ramp's spectral norm, i.e. the
inscribed half-side of the model's valid far-field support). See
``max_window_half_span``. A fixed +/-820 window was measured to break the
objective -- at side 100 the box covers 2601 of 674041 samples, so `ee_roi` sits
at 0.011 and the energy term is background rather than control -- and it can also
run past the CCD edge (the MIICAM path had only 643 px of headroom above the
origin) or past the model's support, where ``grid_sample`` pads zeros and the
"prediction" is really background.

Model caveat (read before trusting a prediction)
------------------------------------------------
The checkpoint is fit at one control grid and then *lifted* to a finer grid with
`export_frozen_parameters` when the target needs more far-field reach. That is a
sampling change, not a refit. Two live checkpoints exist:

* `20261105_203528/fit` -- the retired SLM#1 + MIICAM path, fit at grid 128 /
  FFT 1024, so its valid support is only `|ramp| / (2 * pitch)` = 282 px:
  a 1200 px square is far outside it. Kept as the offline baseline.
* `multishape_nogs_20261104/p20261104/model640` -- the current SLM-200 +
  Daheng path, fit at grid 384 and lifted to 640 / FFT 2048, support 1254 px,
  which covers the whole sweep (the CCD edge at 940 px binds first).

Support is set by the control grid alone (`index = frequency * fft_size *
pitch`); `far_field_half_span` computes it from the checkpoint itself rather
than a magic number. Beyond the support (or beyond the CCD) the prediction is
extrapolation, and the measured column is the authority -- quantifying exactly
that gap per scale is this script's secondary purpose.

Device discipline
-----------------
Without ``--execute`` nothing is imported from the driver layer and no device is
touched: the whole optimisation runs offline and writes predicted artifacts.
With ``--execute`` it occupies SLM #1 (memory mode, ``video_mode=0``; never DVI)
and the requested CCD (``--camera``/``--cam-id``), reports every open/close in
``device_log.json`` in order, and clears the SLM panel on the way out. Do not run
it concurrently with another process that holds either device.

Usage
-----
    python scripts/slm/square_grow_diff_sweep_20261106.py                 # plan
    python scripts/slm/square_grow_diff_sweep_20261106.py --optimise      # offline sweep
    python scripts/slm/square_grow_diff_sweep_20261106.py --execute       # + hardware
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    ROOT / "data/debug/slm_transfer/20261105_203528/fit/frozen_parameters.json"
)
DEFAULT_OUT_ROOT = ROOT / "data/debug/slm_transfer/diff_square_grow"

#: Production 1200-px tier (``campaign.SIZE_WINDOWS`` / ``RUNNER_CLI_DEFAULTS``):
#: a control grid fine enough that the far field reaches +/-600 px, and a camera
#: window that contains a 1200 px target plus margin.
CONTROL_GRID = 640
FFT_SIZE = 2048
#: Full-frame sensor geometry per camera backend, in px. Offline (no camera
#: open) this is the only source; with ``--execute`` the driver's own
#: ``cam_width``/``cam_height`` (full-frame ROI) replace these, so the window
#: never has to assume the CCD edge. Override with ``--camera-width/--height``.
CAMERA_SENSOR_PX = {
    "miicam": (2688, 1520),
    "daheng": (2592, 1944),  # MER2-507-23GM NIR
}
#: Smallest half-span, so even the 100 px tier keeps a peripheral ring for the
#: objective's outside-the-box terms (the transfer campaign used +/-110 for an
#: 80 px target, i.e. a 1.4x margin on each side; 220 keeps that spirit).
MIN_WINDOW_SPAN_PX = 220.0
#: Extra margin beyond the target half-side.
WINDOW_MARGIN_PX = 150.0
WINDOW_STEP_PX = 2.0

#: Fraction of the camera full scale the exposure ladder aims for per side.
EXPOSURE_TARGET_FRACTION = 0.6
#: Footprint threshold, as a fraction of the peak rate inside the ROI.
COVERAGE_FRACTION_OF_PEAK = 0.15


def clock() -> dict:
    return dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        monotonic_ns=time.monotonic_ns(),
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sides_of(args: argparse.Namespace) -> list[float]:
    if args.side_step <= 0:
        raise ValueError("--side-step must be positive")
    if args.side_max < args.side_min:
        raise ValueError("--side-max must be >= --side-min")
    count = int(round((args.side_max - args.side_min) / args.side_step)) + 1
    return [round(args.side_min + index * args.side_step, 6) for index in range(count)]


def build_plan(args: argparse.Namespace) -> dict:
    "Everything that decides the run, before any device or model is built."
    sides = sides_of(args)
    return dict(
        checkpoint=str(args.checkpoint),
        checkpoint_sha256=(
            sha256_file(args.checkpoint) if Path(args.checkpoint).is_file() else None
        ),
        out=str(args.out),
        sides=sides,
        steps_per_side=int(args.steps),
        lr=float(args.lr),
        phase_rms_cap=float(args.phase_rms_cap),
        objective=str(args.objective),
        control_grid=int(args.control_grid),
        fft_size=int(args.fft_size),
        window_span_px=None if args.window_span is None else float(args.window_span),
        camera_sensor_px=CAMERA_SENSOR_PX.get(args.camera),
        window_min_span_px=MIN_WINDOW_SPAN_PX,
        window_margin_px=WINDOW_MARGIN_PX,
        window_step_px=float(args.window_step),
        seed="transport_square" if args.seed_transport else "zeros",
        warm_start=bool(args.warm_start),
        device=str(args.device),
        execute=bool(args.execute),
        optimise=bool(args.optimise or args.execute),
        camera=str(args.camera),
        cam_id=int(args.cam_id),
        slm_number=int(args.slm_number),
        frames=int(args.frames),
        exposure_ms=None if args.exposure_ms is None else float(args.exposure_ms),
        opens_devices=bool(args.execute),
        radiometry=dict(
            every=int(args.radiometry_every),
            phases=max(int(args.radiometry_phases), 3),
        ),
    )


def print_plan(plan: dict) -> None:
    print("growing-square differentiable sweep (no GS)")
    print(f"  checkpoint     : {plan['checkpoint']}")
    print(f"  out            : {plan['out']}")
    print(
        f"  sides          : {plan['sides'][0]:g} -> {plan['sides'][-1]:g} px "
        f"({len(plan['sides'])} targets, step {plan['sides'][1] - plan['sides'][0]:g})"
        if len(plan["sides"]) > 1
        else f"  sides          : {plan['sides'][0]:g} px"
    )
    print(
        f"  model          : grid {plan['control_grid']} / fft {plan['fft_size']} "
        f"(lifted from the checkpoint; not refit)"
    )
    print(
        f"  camera window  : {'user cap +/-' + format(plan['window_span_px'], 'g') + ' px, ' if plan['window_span_px'] is not None else 'auto (CCD edge + far-field support), '}"
        f">= +/-{plan['window_min_span_px']:g} px, side/2 + "
        f"{plan['window_margin_px']:g} px margin @ {plan['window_step_px']:g} px"
    )
    print(
        f"  optimiser      : {plan['objective']}, {plan['steps_per_side']} steps/side, "
        f"lr {plan['lr']:g}, rms cap {plan['phase_rms_cap']:g} rad"
    )
    print(
        f"  seed           : first side {plan['seed']}, later sides "
        f"{'warm start' if plan['warm_start'] else plan['seed']}"
    )
    print(f"  devices        : {'OCCUPY SLM + CAMERA' if plan['opens_devices'] else 'none (offline)'}")
    if not plan["optimise"]:
        print("  (plan only: pass --optimise for the offline sweep, --execute for hardware)")


def camera_edge_half_span(
    centre, camera: str, width=None, height=None
) -> float:
    """Largest symmetric half-span the CCD edge allows at ``centre`` (px).

    Uses the driver's full-frame ``width``/``height`` when given, else the
    per-backend ``CAMERA_SENSOR_PX`` table (offline). An unknown sensor returns
    ``inf`` so only the far-field support constrains the window.
    """
    if not width or not height:
        width, height = CAMERA_SENSOR_PX.get(camera, (0, 0))
    if not (width and height):
        return float("inf")
    return float(min(centre[0], width - centre[0], centre[1], height - centre[1]))


def far_field_half_span(params) -> float:
    """Conservative half-span of the model's valid far-field support (px).

    ``grid_sample`` pads with zeros outside the FFT grid, so camera points with
    ``|frequency| > 1/(2*pitch)`` (Nyquist) read background. The support
    ellipse's inscribed-square half-side is ``nyquist * sigma_max(ramp) *
    axial_scale`` -- the spectral norm bounds the ellipse from inside, making
    this the safest half-span for a square window.
    """
    from ao_shaping.algorithm.signal_processing.defocus_forward_core import (
        PANEL_PITCH_PX,
        axial_plane_correction,
    )

    pitch = PANEL_PITCH_PX / float(params.grid_size)
    nyquist = 1.0 / (2.0 * pitch)
    ramp = np.asarray(params.ramp_matrix, dtype=np.float64)
    sigma = float(np.linalg.svd(ramp, compute_uv=False)[0])
    axial_scale = axial_plane_correction(
        params.defocus_diopter, params.slm_axial_offset_m
    )[1]
    return float(nyquist * sigma * axial_scale)


def max_window_half_span(args, params, centre) -> float:
    """Largest camera half-span the model can trust on this light path.

    The window must stay inside the CCD **and** inside the far-field support;
    both are read from the device/checkpoint, never hardcoded. ``--window-span``
    remains an explicit user cap, applied on top.
    """
    max_half = min(
        camera_edge_half_span(
            centre,
            args.camera,
            getattr(args, "camera_width", None),
            getattr(args, "camera_height", None),
        ),
        far_field_half_span(params),
    )
    if args.window_span is not None:
        max_half = min(max_half, float(args.window_span))
    return float(max_half)


def window_span_for(side: float, max_half_span: float) -> float:
    """Camera half-span that keeps the target a small fraction of the window.

    A fixed +/-820 window was measured to break the objective: at side 100 the
    box covers 2601 of 674041 samples, so ``ee_roi`` sits at 0.011 and
    ``shape_loss`` at 0.98 whatever the phase does -- the loss is background,
    not control. Verified offline: side 100 at +/-220 reaches model cv 0.002 /
    pearson 0.985 in 80 steps, side 600 at +/-620 reaches cv 0.001 /
    pearson 0.993. Hence a span that tracks the target, capped by what the
    CCD edge and the far-field support actually allow (``max_half_span``).
    """
    return float(
        min(max_half_span, max(MIN_WINDOW_SPAN_PX, 0.5 * side + WINDOW_MARGIN_PX))
    )


def load_model(args: argparse.Namespace, out: Path):
    """Lift the fitted checkpoint to the control grid and return its parameters."""
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        load_frozen_parameters,
    )
    from ao_shaping.tools.slm.transfer.device import resolve_device
    from ao_shaping.tools.slm.transfer.geometric_seed import export_frozen_parameters

    # "auto" is a transfer-stage policy, not a torch device string: resolve it in
    # the one place that owns the fallback rules (device.py) before building the
    # forward, which would otherwise hand "auto" straight to ``torch.device``.
    args.device = resolve_device(args.device)
    model_dir = out / "model"
    # ``export_frozen_parameters`` creates the directory itself (exist_ok=False)
    # and records the sampling change in ``provenance.json``; reuse it on a
    # resumed run only when the lift matches what this run asked for.
    exported = model_dir / "frozen_parameters.json"
    if exported.is_file():
        existing = json.loads(exported.read_text(encoding="utf-8"))
        if (
            int(existing.get("grid_size", 0)) != args.control_grid
            or int(existing.get("fft_size", 0)) != args.fft_size
        ):
            raise SystemExit(
                f"{exported} holds grid {existing.get('grid_size')}/fft "
                f"{existing.get('fft_size')}, not the requested "
                f"{args.control_grid}/{args.fft_size}; use a fresh --out"
            )
        print(f"model: reusing {exported}")
    else:
        export_frozen_parameters(
            args.checkpoint, model_dir, grid=args.control_grid, fft=args.fft_size
        )
    params = load_frozen_parameters(model_dir)
    centre = tuple(float(v) for v in params.origin_xy)
    return params, centre


def build_forward(args: argparse.Namespace, params, centre, side: float):
    """The frozen forward sampled on a camera window proportional to ``side``."""
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusForward,
        recentre_window,
    )

    max_half = max_window_half_span(args, params, centre)
    span = window_span_for(side, max_half)
    camera_xy = recentre_window(centre, half_span_px=span, step_px=args.window_step)
    forward = FrozenDefocusForward(params, camera_xy, device=args.device)
    # The checkpoint carries ``bias_counts`` (3.92 on this bench), so ``predict``
    # refuses to run without an exposure: the bias is divided by it. Use the
    # campaign's representative exposure; with ``--execute`` the per-side measured
    # readback replaces it before the next prediction is reported.
    forward.exposure_ms = float(args.model_exposure_ms)
    return forward, camera_xy, span, max_half


def square_target(camera_xy: np.ndarray, centre, side: float) -> np.ndarray:
    from ao_shaping.algorithm.signal_processing.multishape_objective import target_at

    return target_at(camera_xy, "square", centre, side).astype(np.float32)


def make_objective(target: np.ndarray, kind: str):
    from ao_shaping.algorithm.signal_processing.multishape_objective import ShapeObjective

    if kind == "cv":
        return ShapeObjective(target, cv_only=True), "cv"
    if kind == "formation":
        return ShapeObjective(target, formation=True), "formation"
    return ShapeObjective(target), "shape"


def footprint_metrics(image: np.ndarray, box: np.ndarray, core: np.ndarray) -> dict:
    "Scale-free footprint numbers on a rate image (counts / ms)."
    peak = float(image[box].max()) if box.any() else 0.0
    if not np.isfinite(peak) or peak <= 0:
        return dict(coverage=None, iou=None, footprint_side_px=None)
    filled = image > COVERAGE_FRACTION_OF_PEAK * peak
    union = np.logical_or(filled, box).sum()
    return dict(
        coverage=float(filled[box].mean()),
        iou=float((filled & box).sum() / max(union, 1)),
        footprint_side_px=float(np.sqrt(filled.sum())),
        cv=float(image[core].std() / max(abs(image[core].mean()), 1e-9)),
        peak_rate=peak,
    )


def predicted_numbers(forward, phase, objective, box, core) -> dict:
    prediction = forward.predict(np.asarray(phase, dtype=np.float32))
    if prediction.ndim == 3:
        prediction = prediction[0]
    image = prediction.detach().cpu().numpy().astype(np.float64)
    image = np.maximum(image, 0.0)
    from ao_shaping.algorithm.signal_processing.multishape_objective import image_metrics

    base = image_metrics(image, objective)
    numbers = footprint_metrics(image, box, core)
    numbers.update(model_cv=base["cv"], model_pearson=base["target_pearson"],
                   model_ee_roi=base["ee_roi"])
    return numbers, image


def optimise_side(forward, objective, seed_phase, *, steps: int, lr: float, cap: float):
    """Run `steps` differentiable steps; keep the lowest-loss phase."""
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        DifferentiableUniformityOptimizer,
    )

    opt = DifferentiableUniformityOptimizer(
        forward, objective, seed_phase, lr=lr, step_rms_cap=cap
    )
    best_phase, best_loss, history = seed_phase, None, []
    for _ in range(int(steps)):
        phase = opt.update()
        loss = opt.last_loss
        history.append(None if loss is None else float(loss))
        if loss is not None and (best_loss is None or loss < best_loss):
            best_loss, best_phase = float(loss), np.array(phase, copy=True)
    return np.asarray(best_phase, dtype=np.float32), history


def sample_ccd_to_window(frame: np.ndarray, camera_xy: np.ndarray) -> np.ndarray:
    from scipy.ndimage import map_coordinates

    coords = np.stack(
        [camera_xy[..., 1] - 0.5, camera_xy[..., 0] - 0.5], axis=0
    ).reshape(2, -1)
    return map_coordinates(
        np.asarray(frame, dtype=np.float64), coords, order=1, mode="constant"
    ).reshape(camera_xy.shape[:2])


def measured_numbers(
    frame: np.ndarray, *, centre, side: float, exposure_ms: float, noise: float
) -> dict:
    "Footprint metrics on the *native* CCD frame, for a square of `side` at `centre`."
    height, width = frame.shape
    rows, cols = np.mgrid[:height, :width]
    box = (np.abs(cols - centre[0]) <= side / 2) & (np.abs(rows - centre[1]) <= side / 2)
    core = (
        np.abs(cols - centre[0]) <= side / 2 - 2
    ) & (np.abs(rows - centre[1]) <= side / 2 - 2)
    if not box.any():
        return dict(measured_coverage=None, measured_iou=None,
                    measured_footprint_side_px=None, measured_cv=None,
                    measured_peak_rate=None, measured_saturated=0)
    rate = np.maximum(np.asarray(frame, dtype=np.float64) - noise, 0.0) / max(
        exposure_ms, 1e-9
    )
    numbers = footprint_metrics(rate, box, core)
    return dict(
        measured_coverage=numbers["coverage"],
        measured_iou=numbers["iou"],
        measured_footprint_side_px=numbers["footprint_side_px"],
        measured_cv=numbers["cv"],
        measured_peak_rate=numbers["peak_rate"],
        measured_saturated=int((np.asarray(frame) >= 0.98 * 255).sum()),
    )


def csv_row(side: float, predicted: dict, measured: dict | None, extra: dict) -> dict:
    row = dict(side_px=side, **extra, **predicted)
    if measured is not None:
        row.update(measured)
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def device_compat_check(checkpoint: Path, *, live: dict | None = None,
                        args: argparse.Namespace | None = None) -> dict:
    """Compare the checkpoint's calibration record against the live bench.

    Answers "does this checkpoint still describe the device + optical path that
    is physically in front of us?" without touching any device. Every field is
    best-effort: a missing calibration record is a *gap*, never an error, so an
    old session simply reports ``unverifiable`` dimensions instead of blocking.

    The verdict has two independent halves:

    * ``model`` -- is the checkpoint itself internally consistent with the
      calibration session's fingerprint (same ramp / origin / grid / fft)?
      A ``mismatch`` here means the checkpoint was edited or rebuilt after the
      fingerprint was written.
    * ``bench`` -- does the live bench (``live``) match the calibration-time
      record? Only meaningful when the calibration session recorded device
      identity; otherwise ``unverifiable``.

    ``live`` keys (all optional): ``camera``, ``cam_id``, ``slm_number``,
    ``slm_serial``, ``camera_serial``, ``camera_model``, ``sensor`` (w, h),
    ``centre`` (x, y).
    """
    from ao_shaping.tools.slm.transfer.optical_session import (
        _matrix_sha256,
        load_fingerprint,
    )

    def read_json(path: Path) -> dict | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    checkpoint = Path(checkpoint)
    session = checkpoint.resolve().parent.parent
    fingerprint = load_fingerprint(session) or {}
    capture_cfg = read_json(session / "capture" / "config.json") or {}
    manifest = read_json(session / "capture" / "manifest.json") or []

    def manifest_field(key: str, default=None):
        if isinstance(manifest, list) and manifest and isinstance(manifest[0], dict):
            return manifest[0].get(key, default)
        return default

    model = load_model_identity(checkpoint, fingerprint)
    result: dict = {
        "session": str(session),
        "fingerprint": bool(fingerprint),
        "recorded": {
            "camera": capture_cfg.get("camera"),
            "cam_id": capture_cfg.get("cam_id"),
            "slm_number": capture_cfg.get("slm_number"),
            "grid": capture_cfg.get("grid"),
            "slm_serial": manifest_field("slm_serial"),
            "camera_serial": manifest_field("camera_serial"),
        },
        "model": model,
        "bench": {"status": "unverifiable", "checks": []},
    }

    # --- bench half: only when the live bench is supplied -------------------
    if live is not None:
        checks: list[dict] = []
        problems: list[str] = []

        def check(name: str, recorded, live_value, *, equal=None):
            if recorded is None and live_value is None:
                checks.append({"name": name, "status": "unrecorded"})
                return
            if recorded is None:
                checks.append({"name": name, "status": "unverifiable",
                               "live": live_value})
                return
            if live_value is None:
                checks.append({"name": name, "status": "unverifiable",
                               "recorded": recorded})
                return
            match = equal(recorded, live_value) if equal else recorded == live_value
            checks.append({"name": name, "status": "match" if match else "mismatch",
                           "recorded": recorded, "live": live_value})
            if not match:
                problems.append(name)

        check("camera_backend", capture_cfg.get("camera"), live.get("camera"))
        check("cam_id", capture_cfg.get("cam_id"), live.get("cam_id"))
        check("slm_number", capture_cfg.get("slm_number"), live.get("slm_number"))
        check("slm_serial", manifest_field("slm_serial"), live.get("slm_serial"))
        check("camera_serial", manifest_field("camera_serial"),
              live.get("camera_serial"))
        if live.get("sensor"):
            check("sensor", None, tuple(live["sensor"]))
        result["bench"] = {
            "status": ("mismatch" if problems else "ok"),
            "checks": checks,
        }

    # --- verdict ------------------------------------------------------------
    if model.get("status") == "mismatch":
        result["verdict"] = "recalibrate"
    elif result["bench"]["status"] == "mismatch":
        result["verdict"] = "recalibrate"
    elif not fingerprint:
        result["verdict"] = "unverifiable"
    elif result["bench"]["status"] == "unverifiable":
        result["verdict"] = "review"
    else:
        result["verdict"] = "ok"
    return result


def load_model_identity(checkpoint: Path,
                        fingerprint: dict) -> dict:
    """Model half of :func:`device_compat_check`: checkpoint vs fingerprint."""
    from ao_shaping.tools.slm.transfer.optical_session import _matrix_sha256

    payload = _read_json_quiet(checkpoint)
    params = payload.get("parameters") if isinstance(payload, dict) else None
    if not isinstance(params, dict):
        params = payload if isinstance(payload, dict) else {}

    checks: list[dict] = []
    problems: list[str] = []
    fp_ramp = fingerprint.get("ramp_matrix_sha256")
    cp_ramp = _matrix_sha256(params.get("ramp_matrix"))

    def add(name, recorded, live, *, equal=None):
        if recorded is None and live is None:
            checks.append({"name": name, "status": "unrecorded"})
        elif recorded is None or live is None:
            checks.append({"name": name, "status": "unverifiable",
                           **({"recorded": recorded} if recorded is not None else {}),
                           **({"live": live} if live is not None else {})})
        else:
            match = equal(recorded, live) if equal else recorded == live
            checks.append({"name": name, "status": "match" if match else "mismatch",
                           "recorded": recorded, "live": live})
            if not match:
                problems.append(name)

    add("ramp_matrix", fp_ramp, cp_ramp)
    add("control_grid", fingerprint.get("control_grid"),
        params.get("grid_size") or payload.get("grid_size"))
    add("fft_size", fingerprint.get("fft_size"),
        params.get("fft_size") or payload.get("fft_size"))

    fp_origin = fingerprint.get("origin_px")
    cp_origin = params.get("origin")
    if fp_origin and cp_origin and len(fp_origin) >= 2 and len(cp_origin) >= 2:
        dx = float(fp_origin[0]) - float(cp_origin[0])
        dy = float(fp_origin[1]) - float(cp_origin[1])
        match = abs(dx) <= 1.0 and abs(dy) <= 1.0
        checks.append({"name": "origin_px", "status": "match" if match else "mismatch",
                       "recorded": fp_origin, "live": cp_origin,
                       "delta_px": [round(dx, 3), round(dy, 3)]})
        if not match:
            problems.append("origin_px")
    else:
        add("origin_px", fp_origin, cp_origin)

    return {
        "status": "mismatch" if problems else "ok",
        "checks": checks,
        "fingerprint_ramp_sha": fp_ramp,
    }


def _read_json_quiet(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def print_compat_verdict(compat: dict) -> None:
    """One-line, human-readable rendering of a ``device_compat_check`` verdict."""
    verdict = compat.get("verdict", "unknown")
    if verdict == "ok":
        print("[compat] OK: checkpoint matches the recorded calibration record")
        return
    if verdict == "unverifiable":
        print(
            f"[compat] UNVERIFIABLE: checkpoint session has no device record "
            f"(session={compat.get('session')})"
        )
        return
    problems: list[str] = []
    for half in ("model", "bench"):
        checks = (compat.get(half) or {}).get("checks") or []
        problems += [c["name"] for c in checks if c.get("status") == "mismatch"]
    if verdict == "recalibrate":
        print(
            f"[compat] RECALIBRATE required: {', '.join(sorted(problems)) or 'record mismatch'}"
        )
    else:
        print(
            f"[compat] REVIEW: calibration record lacks device identity "
            f"({', '.join(problems) or 'unrecorded'})"
        )


def save_png(array: np.ndarray, path: Path, *, title: str, log: bool = True) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.asarray(array, dtype=np.float64)
    figure, axis = plt.subplots(figsize=(4.6, 4.2), dpi=110)
    image = axis.imshow(
        np.log10(np.maximum(data, data.max() * 1e-6)) if log else data, cmap="inferno"
    )
    axis.set_title(title, fontsize=9)
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def run_sweep(args: argparse.Namespace, plan: dict) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase").mkdir(exist_ok=True)
    (out / "predicted").mkdir(exist_ok=True)

    params, centre = load_model(args, out)
    max_half = max_window_half_span(args, params, centre)
    print(
        f"model: grid {params.grid_size} / fft {params.fft_size}, "
        f"origin ({centre[0]:.1f}, {centre[1]:.1f}); "
        f"window +/-{window_span_for(plan['sides'][0], max_half):g} .. "
        f"+/-{window_span_for(plan['sides'][-1], max_half):g} px "
        f"(max {max_half:.0f} from CCD edge + support)"
    )

    # Device / calibration compatibility verdict: model half + recorded record
    # are device-free, so run it before touching anything; the bench half is
    # filled in once the devices are open (or left "unverifiable" offline).
    compat = device_compat_check(args.checkpoint)
    print_compat_verdict(compat)
    plan["device_compat"] = compat
    (out / "plan.json").write_text(
        json.dumps({**plan, **clock()}, indent=2), encoding="utf-8"
    )

    exposure_ms = None
    cam = slm = None
    device_log: list[dict] = []
    rows: list[dict] = []
    phase = None
    try:
        if args.execute:
            cam, slm, device_log, exposure_ms = open_devices(args, centre)
            # The driver opened a full-frame ROI; its sensor geometry is the
            # authority for the CCD edge, replacing the per-backend table.
            sensor_w = int(getattr(cam, "cam_width", 0) or 0)
            sensor_h = int(getattr(cam, "cam_height", 0) or 0)
            if sensor_w and sensor_h:
                args.camera_width, args.camera_height = sensor_w, sensor_h
            compat_live = device_compat_check(
                args.checkpoint,
                live={
                    "camera": args.camera,
                    "cam_id": args.cam_id,
                    "slm_number": args.slm_number,
                    "slm_serial": getattr(slm, "serial_number", None),
                    "camera_serial": getattr(cam, "sn", None),
                    "camera_model": getattr(cam, "model_name", None),
                    "sensor": (sensor_w, sensor_h),
                    "centre": list(centre),
                },
            )
            plan["device_compat"] = compat_live
            print_compat_verdict(compat_live)
            (out / "plan.json").write_text(
                json.dumps({**plan, **clock()}, indent=2), encoding="utf-8"
            )
        for index, side in enumerate(plan["sides"]):

            # One forward per side: the camera window (hence the objective's
            # support and sample density in the target) tracks the target size.
            forward, camera_xy, span, max_half = build_forward(
                args, params, centre, side
            )
            # Periodic online radiometry refit (index > 0, every N-th side): the
            # forward's gain/background/bias_counts are re-estimated from live
            # CCD frames of already-optimised phases, so this and later sides
            # optimise against the measured camera response.
            every = int(plan.get("radiometry", {}).get("every", 0) or 0)
            if every and index > 0 and index % every == 0:
                params, exposure_ms = refit_radiometry(
                    args, plan, params, centre, camera_xy, cam, slm,
                    exposure_ms, device_log, out, side,
                )
                if exposure_ms is not None:
                    args.model_exposure_ms = float(exposure_ms)
                forward, camera_xy, span, max_half = build_forward(
                    args, params, centre, side
                )
            target = square_target(camera_xy, centre, side)
            objective, mode = make_objective(target, args.objective)
            carry = index > 0 and phase is not None and args.warm_start
            if carry:
                seed = phase
                seed_source = "warm_start"
            else:
                if args.seed_transport:
                    from ao_shaping.tools.slm.transfer.geometric_seed import (
                        square_transport_phase,
                    )

                    seed = square_transport_phase(forward, centre, side)
                else:
                    seed = np.zeros((params.grid_size, params.grid_size), np.float32)
                seed_source = "transport" if args.seed_transport else "zeros"

            started = time.monotonic()
            phase, history = optimise_side(
                forward, objective, seed,
                steps=args.steps, lr=args.lr, cap=args.phase_rms_cap,
            )
            predicted, image = predicted_numbers(
                forward, phase, objective, objective.box, objective.core
            )
            np.save(out / "phase" / f"side_{int(round(side)):04d}.npy", phase)
            save_png(
                image, out / "predicted" / f"side_{int(round(side)):04d}.png",
                title=f"predicted side {side:g} px, loss {history[-1]:.4f}",
            )
            extra = dict(
                objective=mode,
                seed=seed_source,
                window_half_span_px=round(float(span), 3),
                final_loss=history[-1] if history else None,
                first_loss=history[0] if history else None,
                seconds=round(time.monotonic() - started, 1),
            )
            measured = None
            if args.execute:
                measured, exposure_ms = measure_side(
                    args, plan, out, side, centre, camera_xy, cam, slm,
                    exposure_ms, device_log,
                )
                # ``predict`` divides ``bias_counts`` by the exposure: afterwards
                # the measured exposure is the right value for later predictions.
                args.model_exposure_ms = float(exposure_ms)
            rows.append(csv_row(side, predicted, measured, extra))
            write_csv(out / "sides.csv", rows)
            print(
                f"side {side:6g} px (+/-{span:g}) | loss {history[-1]:.4f} <- "
                f"{history[0]:.4f} | "
                f"model cv {predicted['model_cv']} cov {predicted['coverage']} "
                f"iou {predicted['iou']}"
                + (
                    f" | measured cv {measured['measured_cv']} "
                    f"cov {measured['measured_coverage']} "
                    f"iou {measured['measured_iou']}"
                    if measured
                    else ""
                )
            )
    finally:
        if args.execute:
            close_devices(slm, cam, device_log,
                          slm_number=args.slm_number, camera=args.camera)
            (out / "device_log.json").write_text(
                json.dumps(device_log, indent=2), encoding="utf-8"
            )
        write_csv(out / "sides.csv", rows)
    print(f"artifacts: {out}")
    return 0


def open_devices(args, centre):
    """Occupy the SLM (memory mode) + the requested CCD; log both sides."""
    from ao_shaping.drivers.ccd.common import (
        create_camera,
        get_camera_exposure_ms,
        get_camera_exposure_range,
    )
    from ao_shaping.drivers.slm.santec import Santec

    log: list[dict] = []

    def note(action: str, **fields) -> None:
        entry = dict(action=action, **fields, **clock())
        log.append(entry)
        print(f"[device] {action} {fields}")

    note("open", device=f"camera:{args.camera}", cam_id=args.cam_id)
    cam = create_camera(args.camera, args.cam_id, exposure_time_ms=0.05)
    cam.open()
    try:
        floor, ceiling = get_camera_exposure_range(cam)
        exposure = float(args.exposure_ms) if args.exposure_ms else max(floor, 0.5)
        exposure = float(np.clip(exposure, floor, ceiling))
        cam.reset_exposure_time(exposure)
        exposure = float(get_camera_exposure_ms(cam))
        note("open", device=f"slm#{args.slm_number}", mode="memory (video_mode=0)")
        slm = Santec(slm_number=args.slm_number, wavelength=1064, video_mode=0)
        slm.open()
        slm.set_phase_overlay(None)
        note("ready", exposure_ms=exposure,
             sensor=(int(getattr(cam, "cam_width", 0) or 0),
                     int(getattr(cam, "cam_height", 0) or 0)),
             centre=list(centre))
        return cam, slm, log, exposure
    except Exception:
        cam.close()
        raise


def close_devices(slm, cam, log, *, slm_number=None, camera=None) -> None:
    if slm is not None:
        try:
            panel = np.zeros((1200, 1920), np.float32)
            slm.display_data(slm.create_phase_from_array(panel))
        except Exception as exc:  # policy: 清理失败不掩盖主流程结果, 但要留痕
            log.append(dict(action="clear_failed", error=repr(exc), **clock()))
        try:
            slm.close()
        finally:
            log.append(dict(action="close", device=f"slm#{slm_number}", **clock()))
    if cam is not None:
        try:
            cam.close()
        finally:
            log.append(dict(action="close", device=f"camera:{camera}", **clock()))
    print(f"[device] released SLM #{slm_number} and camera {camera}")


def measure_side(
    args, plan, out: Path, side, centre, camera_xy, cam, slm, exposure_ms,
    device_log,
) -> tuple[dict, float]:
    "Display the optimised phase, adapt exposure to this side, capture, measure."
    from ao_shaping.drivers.ccd.common import (
        get_camera_exposure_ms,
        get_camera_exposure_range,
    )
    from ao_shaping.tools.slm.transfer.capture import to_panel

    grid = int(plan["control_grid"])
    phase = np.load(out / "phase" / f"side_{int(round(side)):04d}.npy")
    slm.display_data(slm.create_phase_from_array(to_panel(np.asarray(phase, np.float32))))
    time.sleep(args.settle_s)

    def grab() -> np.ndarray:
        frames = [
            np.asarray(cam.get_numpy_image(n_sample=1, skip_first=False), dtype=np.float64)
            for _ in range(max(int(args.frames), 1))
        ]
        return np.mean(np.stack(frames), axis=0), np.stack(frames)

    frame, stack = grab()
    grid_rows, grid_cols = np.mgrid[: frame.shape[0], : frame.shape[1]]

    if not args.fixed_exposure:
        box = (np.abs(grid_cols - centre[0]) <= side / 2) & (
            np.abs(grid_rows - centre[1]) <= side / 2
        )
        peak = float(frame[box].max()) if box.any() else 0.0
        floor, ceiling = get_camera_exposure_range(cam)
        if peak > 1.0:
            wanted_target = EXPOSURE_TARGET_FRACTION * 255.0
            wanted = float(np.clip(exposure_ms * wanted_target / peak, floor, ceiling))
            if abs(wanted - exposure_ms) > 0.02 * exposure_ms:
                cam.reset_exposure_time(wanted)
                exposure_ms = float(get_camera_exposure_ms(cam))
                time.sleep(args.settle_s)
                frame, stack = grab()

    # Background is re-estimated for every side and every exposure: the offset
    # scales with exposure, so a value carried over from the previous side would
    # bias the rate image (and hence the measured cv) by an unknown amount.
    outside = np.ones(frame.shape, bool)
    outside &= np.abs(grid_cols - centre[0]) > side
    outside &= np.abs(grid_rows - centre[1]) > side
    noise = float(np.median(frame[outside])) if outside.any() else 0.0

    measured = measured_numbers(
        frame, centre=centre, side=side, exposure_ms=exposure_ms, noise=noise
    )
    np.savez_compressed(
        out / "predicted" / f"measured_{int(round(side)):04d}.npz",
        frames=stack.astype(np.uint8),
        exposure_ms=exposure_ms,
    )
    save_png(
        frame, out / "predicted" / f"measured_{int(round(side)):04d}.png",
        title=f"measured {side:g} px @ {exposure_ms:.2f} ms",
    )
    device_log.append(
        dict(action="capture", side_px=side, exposure_ms=exposure_ms,
             frames=int(args.frames), **clock())
    )
    return {**measured, "exposure_ms": exposure_ms}, exposure_ms


def refit_radiometry(
    args,
    plan: dict,
    params,
    centre,
    camera_xy: np.ndarray,
    cam,
    slm,
    exposure_ms,
    device_log: list,
    out: Path,
    side: float,
) -> tuple[object, float]:
    """Refit the frozen forward's radiometric weights from live CCD frames.

    While the sweep runs, the already-optimised phases are re-displayed and
    captured; the forward (on the current side's window) predicts the same
    phases, and a non-negative least squares solves
    ``rate = gain * predicted + background + bias_counts / exposure`` over the
    window's core. The refitted parameters replace ``params`` so later sides
    optimise against a camera response measured on the live bench.

    Needs ``--execute`` and at least 3 distinct saved phases; otherwise
    ``(params, exposure_ms)`` is returned unchanged.
    """
    if not args.execute:
        return params, exposure_ms

    from dataclasses import replace

    from scipy.optimize import nnls

    from ao_shaping.tools.slm.transfer.capture import to_panel

    # Collect distinct saved phases (one per side), ascending, up to the cap.
    want = max(int(plan["radiometry"]["phases"]), 3)
    distinct: list[tuple[int, Path]] = []
    seen: set[int] = set()
    for path in sorted((out / "phase").glob("side_*.npy")):
        side_px = int(Path(path.stem).stem.split("_")[1])
        if side_px in seen:
            continue
        seen.add(side_px)
        distinct.append((side_px, path))
        if len(distinct) >= want:
            break
    if len(distinct) < 3:
        print(f"[radiometry] skipped: need >=3 distinct phases, have {len(distinct)}")
        return params, exposure_ms

    # The forward for the side about to be built; the phases are re-simulated on
    # this window and the measured CCD frame is sampled onto the same grid.
    forward, _, _, _ = build_forward(args, params, centre, side)
    window_x = camera_xy[..., 0]
    window_y = camera_xy[..., 1]
    core = (np.abs(window_x - centre[0]) <= side / 2 - 2) & (
        np.abs(window_y - centre[1]) <= side / 2 - 2
    )
    if not core.any():
        print("[radiometry] skipped: core window empty")
        return params, exposure_ms

    import torch

    exposure_used = float(exposure_ms)
    measured: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    for _side_px, path in distinct:
        phase = np.load(path)
        slm.display_data(
            slm.create_phase_from_array(to_panel(np.asarray(phase, np.float32)))
        )
        time.sleep(args.settle_s)
        frame = np.asarray(
            cam.get_numpy_image(n_sample=1, skip_first=False), dtype=np.float64
        )
        rows, cols = np.mgrid[: frame.shape[0], : frame.shape[1]]
        outside = (np.abs(cols - centre[0]) > side) & (
            np.abs(rows - centre[1]) > side
        )
        bg = float(np.median(frame[outside])) if outside.any() else 0.0
        rate = np.maximum(frame - bg, 0.0) / exposure_used
        sampled = sample_ccd_to_window(rate, camera_xy)
        with torch.no_grad():
            pred = forward.predict(np.asarray(phase, np.float32))[0].cpu().numpy()
        measured.append(np.asarray(sampled[core], dtype=np.float64))
        predicted.append(np.asarray(pred[core], dtype=np.float64))

    y = np.concatenate(measured)
    a = np.concatenate(predicted)
    design = np.stack([a, np.ones_like(a), np.full_like(a, 1.0 / exposure_used)], axis=1)
    coeff, _res = nnls(design, y)
    gain, background, bias_counts = map(float, coeff)

    # Physical clamps: positive gain, non-negative background/bias (bias is a
    # count offset on a byte camera, so it cannot exceed the saturation level).
    gain = float(max(gain, 1e-9))
    background = float(max(background, 0.0))
    bias_counts = float(np.clip(bias_counts, 0.0, 255.0))
    new_params = replace(
        params, gain=gain, background=background, bias_counts=bias_counts
    )

    print(
        f"[radiometry] refit on {len(distinct)} phases: gain {gain:.4f} "
        f"(was {params.gain:.4f}), background {background:.4f} "
        f"(was {params.background:.4f}), bias {bias_counts:.3f} "
        f"(was {params.bias_counts:.3f}) @ {exposure_used:.3f} ms"
    )
    device_log.append(
        dict(action="radiometry_refit", phases=len(distinct), gain=gain,
             background=background, bias_counts=bias_counts,
             exposure_ms=exposure_used, **clock())
    )
    with (out / "radiometry_updates.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                dict(**clock(), phases=len(distinct), gain=gain,
                     background=background, bias_counts=bias_counts,
                     exposure_ms=exposure_used)
            )
            + "\n"
        )
    return new_params, exposure_used


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--side-min", type=float, default=100.0)
    parser.add_argument("--side-max", type=float, default=1200.0)
    parser.add_argument("--side-step", type=float, default=100.0)
    parser.add_argument("--steps", type=int, default=40,
                        help="differentiable steps per side")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--phase-rms-cap", type=float, default=0.1,
                        help="per-step increment cap in weighted RMS radians")
    parser.add_argument("--objective", choices=("cv", "shape", "formation"),
                        default="shape")
    parser.add_argument("--control-grid", type=int, default=CONTROL_GRID)
    parser.add_argument("--fft-size", type=int, default=FFT_SIZE)
    parser.add_argument("--window-span", type=float, default=None,
                        help="explicit cap on the camera half-span (px); default "
                             "derives it from the CCD edge + far-field support")
    parser.add_argument("--camera-width", type=int, default=None,
                        help="full-frame sensor width (px); default per-backend "
                             "table, replaced by the driver when --execute")
    parser.add_argument("--camera-height", type=int, default=None,
                        help="full-frame sensor height (px); default per-backend "
                             "table, replaced by the driver when --execute")
    parser.add_argument("--window-step", type=float, default=WINDOW_STEP_PX)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-seed-transport", dest="seed_transport",
                        action="store_false")
    parser.add_argument("--optimise", action="store_true",
                        help="run the offline sweep")
    parser.add_argument("--execute", action="store_true",
                        help="occupy SLM #1 + camera and measure each side")
    parser.add_argument("--camera", default="miicam")
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--exposure-ms", type=float, default=None)
    parser.add_argument(
        "--model-exposure-ms", type=float, default=8.0,
        help="exposure the forward's additive bias is divided by (checkpoint "
             "bias_counts); the capture median of the 2026-11-05 session was ~6.9 ms",
    )
    parser.add_argument("--fixed-exposure", action="store_true",
                        help="keep one exposure for every side")
    parser.add_argument(
        "--warm-start", action="store_true",
        help="seed sides 2+ from the previous side's phase instead of a fresh "
             "transport seed (the scale-change question; needs far more steps "
             "than the default 40 to converge)",
    )
    parser.add_argument("--settle-s", type=float, default=0.6)
    parser.add_argument(
        "--radiometry-every", type=int, default=0,
        help="refit the frozen forward's radiometric weights (gain/background/"
             "bias_counts) from live CCD frames every N-th side (0-based index "
             "%% N == 0, index > 0); requires --execute; 0 disables (default)",
    )
    parser.add_argument(
        "--radiometry-phases", type=int, default=3,
        help="distinct intermediate phases to display+capture per radiometry "
             "refit (default 3; needs >= 3 to be identifiable)",
    )
    args = parser.parse_args(argv)
    if args.out is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = DEFAULT_OUT_ROOT / stamp
    if not Path(args.checkpoint).is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    plan = build_plan(args)
    print_plan(plan)
    if not plan["optimise"]:
        return 0
    return run_sweep(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
