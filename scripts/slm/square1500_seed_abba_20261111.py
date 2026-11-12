"""Measured A–B–B–A comparison of three square seed constructions at side 1500 px.

Purpose
-------
Plans (and, only with ``--execute``, performs) a measured CCD comparison of three
square-seed constructions --- Gerchberg–Saxton, separable-irradiance transport and
target back-projection --- at a requested target side length (default 1500 px) under
**one frozen exposure**. The acceptance is a fixed-exposure paired (ABBA) decision,
so each unordered pair of seeds is captured as a contiguous ``A,B,B,A`` window and
judged with :func:`ao_shaping.tools.slm.transfer.paired.paired_decision`.

The reference construction (default ``gs``) carries the exposure ladder; the frozen
exposure it lands on is reused verbatim for every other capture. The per-seed peak at
that exposure is recorded, because the **spread** of peaks is the evidence that
separates a real shaping gap from an illumination / saturation artifact.

Default is offline
------------------
Without ``--execute`` the script only resolves the frozen checkpoint, builds the three
seeds on the CPU forward model, verifies the grid-480 reach covers the requested side
(``fft_max_side_px``), computes per-seed phase statistics and writes ``plan.json`` into
``--out``. No SDK import and no device is opened in that mode.

``--execute`` performs the hardware run: exactly one SLM #1 + camera #0 open/close
cycle (the bench fault forbids back-to-back SLM open/close), an SLM self-check right
after open, a flat-phase light precheck, the reference-seed exposure ladder, the frozen
ABBA captures, the three pairwise verdicts and the ``captures/`` / ``manifest.json`` /
``abba.json`` / ``summary.md`` / ``config.json`` outputs.

Deviations from the naive schedule (documented here and in ``summary.md``)
-------------------------------------------------------------------------
* ``--rounds`` is accepted for interface symmetry but the ABBA schedule is always the
  three disjoint contiguous ``A,B,B,A`` windows (one per unordered pair) = 12 captures,
  4 per seed. A monotonic 4-round x 3-capture schedule cannot yield three disjoint ABBA
  windows, so ``--rounds`` is not consumed beyond being recorded.
* ``--seed-int`` is recorded for reproducibility. All three constructions are
  deterministic given the checkpoint (no RNG), so the integer is not consumed by the
  seed builders today.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    FrozenDefocusForward,
    load_frozen_parameters,
    recentre_window,
)
from ao_shaping.algorithm.signal_processing.multishape_objective import (
    ShapeObjective,
    assert_target_within_fft_window,
    fft_max_side_px,
    gs_seed,
    target_at,
)
from ao_shaping.tools.slm.panel.display_state import slm_self_check
from ao_shaping.tools.slm.transfer.geometric_seed import (
    square_transport_phase,
    target_backprojection_phase,
)
from ao_shaping.tools.slm.transfer.paired import (
    abba_schedule,
    paired_decision,
    pair_window,
)

ROOT = Path(__file__).resolve().parents[2]

#: Frozen checkpoint from the 2026-11-11 session (the reference the default geometry
#: was measured against).
DEFAULT_CHECKPOINT = (
    ROOT / "data/debug/slm_transfer/20261111_140924/fit/frozen_parameters.json"
)

#: Accepted construction names, in the default comparison order.
SEEDS = ("gs", "transport", "backprojection")

SEED_LABEL = {
    "gs": "Gerchberg–Saxton",
    "transport": "square transport",
    "backprojection": "target backprojection",
}

#: Fraction of full scale at which a frame counts as saturated (production meter:
#: ``raw.max() >= full_scale * 250/255``).
SATURATION_FRACTION = 250.0 / 255.0

#: Settle after each ``display_data`` (matches the production capture ladder).
DISPLAY_SETTLE_S = 0.6

#: Multiplicative correction bounds for the exposure ladder (production default).
EXPOSURE_RATIO_BOUNDS = (0.15, 6.0)


def write_json(path: Path, value: object) -> None:
    """Write JSON with the project style (indent 2, UTF-8, numpy scalars coerced)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def digest(path: Path) -> str:
    """sha256 of one file's bytes (used for checkpoint provenance)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def phase_stats(phase: np.ndarray) -> dict[str, float | bool]:
    """Per-seed raw-radian phase statistics plus the finiteness check."""
    phase = np.asarray(phase, dtype=np.float64)
    return {
        "min_rad": float(phase.min()),
        "max_rad": float(phase.max()),
        "rms_rad": float(np.sqrt(np.mean(np.square(phase)))),
        "finite": bool(np.isfinite(phase).all()),
    }


def parse_seeds(value: str) -> tuple[str, ...]:
    """Parse the comma-separated ``--seeds`` value and validate membership."""
    names = tuple(str(part).strip() for part in value.split(",") if part.strip())
    if not names:
        raise ValueError("--seeds must list at least one construction")
    for name in names:
        if name not in SEEDS:
            raise ValueError(f"unknown seed {name!r}; choose from {', '.join(SEEDS)}")
    if len(set(names)) != len(names):
        raise ValueError("--seeds must not repeat a construction")
    return names


def build_pairs(seed_names: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    """All unordered pairs among the requested seeds (ABBA needs >= 2)."""
    if len(seed_names) < 2:
        raise ValueError("ABBA comparison needs at least two seeds")
    return tuple(
        (seed_names[i], seed_names[j])
        for i in range(len(seed_names))
        for j in range(i + 1, len(seed_names))
    )


def build_seeds(
    forward: FrozenDefocusForward, centre: tuple[float, float], side: float, grid: int
) -> dict[str, dict]:
    """Build all requested seeds as raw-radian phase grids plus diagnostics/stats.

    Return shapes are production-verified: ``gs_seed`` and ``square_transport_phase``
    return phase only; ``target_backprojection_phase`` returns ``(phase, diagnostics)``.
    """
    seeds: dict[str, dict] = {}
    for name in SEEDS:
        if name == "gs":
            phase = gs_seed(forward, centre, side, "square")
            diag: dict = {"kind": "gs", "gs_calls": 80}
        elif name == "transport":
            phase = square_transport_phase(forward, centre, side)
            diag = {"kind": "transport", "gs_calls": 0}
        else:  # backprojection
            phase, back_diag = target_backprojection_phase(
                forward, centre, side, "square"
            )
            diag = {"kind": "backprojection", "gs_calls": 0}
            diag.update(
                {
                    key: (
                        float(value)
                        if isinstance(value, (int, float, np.floating))
                        else value
                    )
                    for key, value in dict(back_diag).items()
                }
            )
        phase = np.asarray(phase, dtype=np.float64)
        if phase.shape != (int(grid), int(grid)):
            raise ValueError(
                f"seed {name!r} shape {phase.shape} != expected {(int(grid), int(grid))}"
            )
        seeds[name] = {"phase": phase, "diagnostics": diag, "stats": phase_stats(phase)}
    return seeds


@dataclass
class Resolved:
    """Live objects needed by both the plan writer and the execute path."""

    params: object
    xy: np.ndarray
    forward: FrozenDefocusForward
    objective: ShapeObjective
    seeds: dict[str, dict]
    zero_xy: tuple[float, float]
    slope_px: float
    reach_px: float
    max_side_px: float
    pairs: tuple[tuple[str, str], ...] = field(default_factory=tuple)


def resolve(args: argparse.Namespace) -> Resolved:
    """Load the checkpoint and build the CPU forward model, window and seeds."""
    params = load_frozen_parameters(args.checkpoint)
    # Zero order: with --no-tilt the target sits at the zero order; with tilt the
    # checkpoint's measured zero is the window centre (the carrier moves the target).
    if args.tilt:
        zero_xy = tuple(float(v) for v in params.origin_xy)
    else:
        zero_xy = tuple(float(v) for v in args.target_xy)
    xy = recentre_window(
        zero_xy, half_span_px=args.window_span / 2.0, step_px=args.window_step
    )
    forward = FrozenDefocusForward(params, xy, device="cpu")
    centre = tuple(float(v) for v in args.target_xy)

    # Authoritative production reach gate: raises if side exceeds the FFT window.
    assert_target_within_fft_window(
        forward, centre, args.side, "square", aspect_ratio=2.0
    )

    ramp = np.asarray(params.ramp_matrix, dtype=float)
    slope_px = float(np.linalg.norm(ramp, axis=1).max())
    grid = float(params.grid_size)
    reach_px = slope_px * (0.5 / (1920.0 / grid))
    max_side_px = fft_max_side_px(grid, slope_px)
    if max_side_px < args.side:
        raise ValueError(
            f"grid {int(grid)} reach covers at most {max_side_px:.1f} px < requested "
            f"side {args.side:g} px; use a larger grid or a smaller side"
        )

    objective = ShapeObjective(
        target_at(xy, "square", centre, args.side, aspect_ratio=2.0)
    )
    if args.zero_radius > 0:
        radius = np.linalg.norm(xy - np.asarray(zero_xy), axis=-1)
        objective.core &= radius > args.zero_radius
        if not objective.core.any():
            raise ValueError("zero_radius masks out the entire target core")

    seeds = build_seeds(forward, centre, args.side, int(grid))
    pairs = build_pairs(parse_seeds(args.seeds))
    return Resolved(
        params=params,
        xy=xy,
        forward=forward,
        objective=objective,
        seeds=seeds,
        zero_xy=zero_xy,
        slope_px=slope_px,
        reach_px=reach_px,
        max_side_px=max_side_px,
        pairs=pairs,
    )


def build_plan(resolved: Resolved, args: argparse.Namespace) -> dict:
    """Serialize the resolved run into a JSON-safe plan dict (no live arrays)."""
    seed_names = parse_seeds(args.seeds)
    schedule = abba_schedule(resolved.pairs)
    reference = args.reference
    if reference not in seed_names:
        raise ValueError(f"--reference {reference!r} not in --seeds {seed_names}")
    return {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": digest(args.checkpoint),
        "out": str(args.out),
        "geometry": {
            "side": float(args.side),
            "window_span": float(args.window_span),
            "window_step": float(args.window_step),
            "zero_radius": float(args.zero_radius),
            "period": float(args.period),
            "tilt": bool(args.tilt),
            "target_xy": [float(v) for v in args.target_xy],
            "zero_xy": [float(v) for v in resolved.zero_xy],
        },
        "grid": {
            "grid_size": int(resolved.params.grid_size),
            "fft_size": int(resolved.params.fft_size),
            "pixel_samples": int(resolved.params.pixel_samples),
        },
        "reach": {
            "slope_px": float(resolved.slope_px),
            "reach_px": float(resolved.reach_px),
            "max_side_px": float(resolved.max_side_px),
            "side_px": float(args.side),
            "covers_side": bool(resolved.max_side_px >= float(args.side)),
        },
        "seeds": {
            name: {
                "diagnostics": dict(resolved.seeds[name]["diagnostics"]),
                "stats": dict(resolved.seeds[name]["stats"]),
            }
            for name in seed_names
        },
        "reference": reference,
        "capture_order": schedule,
        "pairs": [list(pair) for pair in resolved.pairs],
        "rounds_requested": int(args.rounds),
        "captures_per_seed": {name: schedule.count(name) for name in seed_names},
        "frames_per_capture": int(args.frames),
        "bit_depth": int(args.bit_depth),
        "full_scale": float((1 << int(args.bit_depth)) - 1),
        "camera": args.camera,
        "cam_id": int(args.cam_id),
        "slm_number": int(args.slm_number),
        "seed_int": int(args.seed_int),
        "fixed_exposure_ms": float(args.fixed_exposure_ms),
        "wait_until_idle": float(args.wait_until_idle),
        "execute": bool(args.execute),
    }


def _centroid_xy(sampled: np.ndarray, objective: ShapeObjective) -> list[float] | None:
    """Intensity centroid over the target box (diagnostic; None when dark).

    ``sampled`` is the camera-window image (already indexed at ``camera_xy``),
    so ``objective.box`` is the right mask shape for it.
    """
    values = np.asarray(sampled, dtype=float)[objective.box]
    total = float(values.sum())
    if total <= 0:
        return None
    yy, xx = np.nonzero(objective.box)
    return [float((xx * values).sum() / total), float((yy * values).sum() / total)]


def make_row(
    frame: np.ndarray,
    objective: ShapeObjective,
    *,
    xy: np.ndarray,
    label: str,
    seed: str,
    round_index: int,
    full_scale: float,
) -> dict:
    """One measured capture row: production metrics + saturation + centroid.

    Mirrors the transfer seam exactly (``tools/slm/transfer/validate.py:652``):
    the raw native frame is first sampled at the camera-window grid ``xy``, and
    metrics run on that sampled (window-sized) image — never on the full frame,
    whose 1944 rows do not match ``objective.core``/``objective.box``.
    ``peak``/``saturated`` keep the **raw** frame semantics (production uses
    ``float(raw.max())``); ``core_peak`` is the sampled core maximum.
    """
    from ao_shaping.algorithm.signal_processing.multishape_objective import (
        image_metrics,
    )

    raw = np.asarray(frame, dtype=float)
    sampled = raw[xy[..., 1].astype(int), xy[..., 0].astype(int)]
    metrics = image_metrics(sampled, objective)
    cv = metrics["cv"]
    return {
        "label": label,
        "seed": seed,
        "round": int(round_index),
        "metrics": {
            "cv": None if cv is None else float(cv),
            "mean": float(metrics["mean"]),
            "ee_roi": float(metrics["ee_roi"]),
            "target_peak": float(metrics["target_peak"]),
            "target_pearson": (
                None
                if metrics["target_pearson"] is None
                else float(metrics["target_pearson"])
            ),
        },
        # paired_decision(metric="cv") reads metrics["cv"]; loss mirrors cv for
        # the row contract (loss=float, finite when measurable).
        "loss": float(cv) if cv is not None else float("nan"),
        "peak_counts": float(raw.max()),
        "core_peak": float(sampled[objective.core].max()),
        "saturated": bool(raw.max() >= full_scale * SATURATION_FRACTION),
        "centroid_xy": _centroid_xy(sampled, objective),
    }


def _dark_floor(frame: np.ndarray) -> float:
    """Robust "no measurable signal" floor: peak must exceed 2x the background median."""
    return 2.0 * max(float(np.median(np.asarray(frame, dtype=float))), 1.0)


def _landing_status(
    peak: float, fraction: float, frame: np.ndarray, full_scale: float
) -> str:
    """Classify one seed's peak at the frozen exposure.

    ``saturated`` and ``dark`` are hard stops (band unattainable / shaping gap);
    out-of-band but measurable is recorded as evidence (peak spread), not a stop.
    """
    if fraction >= SATURATION_FRACTION:
        return "saturated"
    if peak <= _dark_floor(frame):
        return "dark"
    if fraction >= 0.25 and fraction <= 0.80:
        return "ok"
    return "out_of_band"


def perform_measurement(
    resolved: Resolved,
    plan: dict,
    args: argparse.Namespace,
    cam: object,
    slm: object,
    full_scale: float,
    schedule: list[str],
) -> tuple[dict, dict, dict]:
    """Run the frozen-exposure ABBA capture sequence on an open camera + SLM.

    Returns ``(manifest, abba, landing)``.  Raises ``RuntimeError`` when the
    reference-seed ladder cannot land, a seed is saturated or dark, or any row's
    CV is non-finite (a capture is retained only for the record, never for the
    paired decision).
    """
    from ao_shaping.drivers.ccd.common import (
        get_camera_exposure_ms,
        get_camera_exposure_range,
    )
    from ao_shaping.tools.slm.panel.exposure_adapt import adapt_exposure, robust_peak
    from ao_shaping.tools.slm.transfer.capture import exposure_setter, to_panel
    from ao_shaping.tools.slm.transfer.light_guard import LightGuard, verify_light

    def read_raw() -> np.ndarray:
        return np.asarray(cam.get_numpy_image(n_sample=1, skip_first=False)).copy()

    out = args.out
    floor, ceiling = get_camera_exposure_range(cam)
    floor = max(floor, float(args.ladder_floor_ms))
    # Daheng configured exposure ceiling (production behaviour: min with config).
    if args.camera == "daheng":
        try:
            from ao_shaping.utils.image.exposure_control import (
                configured_daheng_exposure_limit,
            )

            configured = configured_daheng_exposure_limit()
            if configured > 0:
                ceiling = min(ceiling, configured)
        except Exception:
            logger.warning("daheng exposure limit query failed; using camera ceiling")

    grid = int(resolved.params.grid_size)
    reference = plan["reference"]
    seeds = resolved.seeds

    # --- flat-phase light precheck (production seam; raises OpticalPathInvalid) ---
    slm.display_data(
        slm.create_phase_from_array(to_panel(np.zeros((grid, grid), np.float32)))
    )
    time.sleep(DISPLAY_SETTLE_S)
    light = verify_light(
        read_raw,
        exposure_setter(cam),
        floor_ms=floor,
        ceiling_ms=ceiling,
        label=f"square1500 {out.name}",
    )
    write_json(out / "light_check.json", {**dict(light), "clock": _clock_dict()})

    # --- reference-seed exposure ladder (or fixed-exposure override) ---
    ref_phase = seeds[reference]["phase"]
    slm.display_data(slm.create_phase_from_array(to_panel(ref_phase)))
    time.sleep(DISPLAY_SETTLE_S)
    if args.fixed_exposure_ms > 0:
        frozen_ms = float(args.fixed_exposure_ms)
        cam.reset_exposure_time(frozen_ms)
        frozen_ms = float(get_camera_exposure_ms(cam))
        slm.display_data(slm.create_phase_from_array(to_panel(ref_phase)))
        time.sleep(DISPLAY_SETTLE_S)
        ref_raw = read_raw()
        if float(ref_raw.max()) >= full_scale * SATURATION_FRACTION:
            raise RuntimeError(
                f"reference {reference!r} saturated at fixed {frozen_ms:.5f} ms; record and stop"
            )
    else:
        steps = int(args.max_ladder_steps)
        setting = adapt_exposure(
            cam.reset_exposure_time,
            read_raw,
            start_ms=float(args.ladder_start_ms),
            floor_ms=floor,
            ceiling_ms=ceiling,
            full_scale=full_scale,
            max_steps=steps,
            ratio_bounds=EXPOSURE_RATIO_BOUNDS,
            on_step=LightGuard(label=f"reference {reference}", max_steps=steps).observe,
        )
        # Verify the landed reference frame is not saturated before freezing.
        slm.display_data(slm.create_phase_from_array(to_panel(ref_phase)))
        time.sleep(DISPLAY_SETTLE_S)
        ref_raw = read_raw()
        if setting.status != "ok":
            raise RuntimeError(
                f"reference {reference!r} exposure ladder ended {setting.status!r} "
                f"(peak={setting.peak:.1f}, fraction={setting.fraction:.3f}); record and stop"
            )
        if float(ref_raw.max()) >= full_scale * SATURATION_FRACTION:
            raise RuntimeError(
                "reference seed saturated at landed exposure; record and stop"
            )
        frozen_ms = float(get_camera_exposure_ms(cam))
        cam.reset_exposure_time(frozen_ms)
    logger.info(
        "frozen exposure {:.5f} ms on reference seed {!r}", frozen_ms, reference
    )

    # --- per-seed landing probe at the frozen exposure (record; stop if hard-fail) ---
    landing: dict[str, dict] = {}
    for name in list(dict.fromkeys(schedule)):
        slm.display_data(slm.create_phase_from_array(to_panel(seeds[name]["phase"])))
        time.sleep(DISPLAY_SETTLE_S)
        frame = read_raw()
        peak = robust_peak(frame)
        fraction = peak / full_scale
        status = _landing_status(peak, fraction, frame, full_scale)
        landing[name] = {
            "peak_counts": float(peak),
            "fraction_of_full_scale": float(fraction),
            "status": status,
        }
        if status in ("saturated", "dark"):
            write_json(
                out / "landing.json",
                {"frozen_exposure_ms": frozen_ms, "landing": landing},
            )
            raise RuntimeError(
                f"seed {name!r} landing {status} at {frozen_ms:.5f} ms "
                f"(peak={peak:.1f}, fraction={fraction:.3f}); record and stop"
            )

    # --- frozen ABBA captures ---
    rows: list[dict] = []
    capture_dir = out / "captures"
    capture_dir.mkdir(parents=True, exist_ok=True)
    for idx, seed_name in enumerate(schedule, start=1):
        slm.display_data(
            slm.create_phase_from_array(to_panel(seeds[seed_name]["phase"]))
        )
        time.sleep(DISPLAY_SETTLE_S)
        frames: list[np.ndarray] = []
        for _ in range(int(args.frames)):
            frames.append(read_raw().copy())
        raw = np.stack(frames)
        frame = raw.mean(0)
        if float(raw.max()) >= full_scale * SATURATION_FRACTION:
            raise RuntimeError("saturated capture retained; stop paired comparison")
        label = f"r{idx}_{seed_name}"
        row = make_row(
            frame,
            resolved.objective,
            xy=resolved.xy,
            label=label,
            seed=seed_name,
            round_index=idx,
            full_scale=full_scale,
        )
        rows.append(row)
        np.savez_compressed(
            capture_dir / f"{label}.npz",
            raw_frames=raw,
            frame=frame,
            exposure_ms=np.array([frozen_ms]),
            full_scale=np.array([full_scale]),
        )
        logger.info(
            "capture {}: {} peak={:.1f} cv={}",
            label,
            seed_name,
            row["peak_counts"],
            row["metrics"]["cv"],
        )

    # CV finiteness guard (paired_decision raises on nonfinite).
    for row in rows:
        if row["metrics"]["cv"] is None:
            raise RuntimeError(
                f"capture {row['label']} has non-finite CV (dark core); record and stop"
            )

    # --- pairwise ABBA verdicts ---
    abba: dict[str, object] = {"metric": "cv", "pairs": {}}
    for a, b in resolved.pairs:
        i0, i1, i2, i3 = pair_window(schedule, a, b)
        verdict = paired_decision(
            rows[i0],
            rows[i1],
            rows[i2],
            rows[i3],
            metric="cv",
            rel_tol=0.0,
            abs_tol=0.0,
        )
        abba["pairs"][f"{a}_vs_{b}"] = {
            "a": a,
            "b": b,
            "indices": [i0 + 1, i1 + 1, i2 + 1, i3 + 1],
            **verdict,
        }

    manifest = {
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "frozen_exposure_ms": frozen_ms,
        "full_scale": full_scale,
        "bit_depth": int(args.bit_depth),
        "schedule": schedule,
        "reference": reference,
        "landing": landing,
        "captures": rows,
    }
    return manifest, abba, landing


def _clock_dict() -> dict:
    from ao_shaping.tools.slm.transfer.capture import clock

    return clock()


def run_execute(resolved: Resolved, plan: dict, args: argparse.Namespace) -> None:
    """Hardware path: exactly one SLM + camera open/close cycle, then write outputs."""
    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm.santec import Santec
    from ao_shaping.tools.slm.transfer.hardware import opened_device, prepare_camera_sdk

    out = args.out
    if (out / "manifest.json").exists():
        raise FileExistsError(f"refusing reused out dir {out}: manifest.json present")

    device_lock = None
    if args.wait_until_idle > 0:
        from ao_shaping.tools.kernels.device_mutex import wait_until_idle

        verdict = wait_until_idle(
            float(args.wait_until_idle),
            slm_number=int(args.slm_number),
            cam_id=int(args.cam_id),
            cam_type=str(args.camera),
            exposure_ms=0.2,
        )
        if not verdict.ok:
            raise RuntimeError(f"device not idle: {verdict.reason}")
        device_lock = verdict.lock

    out.mkdir(parents=True, exist_ok=True)
    full_scale = float((1 << int(args.bit_depth)) - 1)
    schedule = plan["capture_order"]
    release: dict[str, object] = {
        "devices": f"SLM #{args.slm_number}, camera #{args.cam_id} ({args.camera})",
        "started_utc": None,
        "released_utc": None,
        "run_completed": False,
        "closed_normally": False,
        "device_lock_released": "not_acquired",
    }
    logger.info(
        "[occupancy] start: SLM #{} + camera #{} ({}) for square1500 ABBA",
        args.slm_number,
        args.cam_id,
        args.camera,
    )
    release["started_utc"] = _clock_dict()["timestamp_utc"]
    try:
        prepare_camera_sdk(str(args.camera))
        with opened_device(
            create_camera(str(args.camera), int(args.cam_id), exposure_time_ms=0.02)
        ) as cam:
            with opened_device(
                Santec(slm_number=int(args.slm_number), wavelength=1064, video_mode=0)
            ) as slm:
                slm.set_phase_overlay(None)
                selfcheck = slm_self_check(slm)
                logger.info("SLM self-check after open: {}", selfcheck)
                manifest, abba, landing = perform_measurement(
                    resolved, plan, args, cam, slm, full_scale, schedule
                )
                write_json(out / "manifest.json", manifest)
                write_json(out / "abba.json", abba)
                write_json(out / "config.json", plan)
                write_json(
                    out / "selfcheck.json",
                    {"selfcheck": selfcheck, **manifest["landing"]},
                )
                _write_summary(out, plan, manifest, abba)
                # Best-effort restore to flat before close.
                try:
                    slm.display_data(
                        slm.create_phase_from_array(np.zeros((1200, 1920)))
                    )
                    time.sleep(0.3)
                    release["restore_flat"] = "ok"
                except Exception as exc:
                    release["restore_flat"] = f"failed: {exc}"
        # Both `with opened_device(...)` blocks exited: the context managers
        # closed camera and SLM (one open/close cycle) without error.
        release["closed_normally"] = True
        release["run_completed"] = True
    except Exception as exc:
        # The context managers still close both devices while unwinding; record
        # that separately from whether the measurement itself completed.
        release["closed_normally"] = "closed on unwind (context manager)"
        release["run_completed"] = False
        release["error"] = f"{type(exc).__name__}: {exc}"
        # If the run failed before the success summary was written, leave a
        # failure summary.md carrying the reason (e.g. a saturated capture).
        if not (out / "summary.md").exists():
            _write_failure_summary(out, plan, release)
        raise
    finally:
        release["released_utc"] = _clock_dict()["timestamp_utc"]
        if device_lock is not None:
            try:
                device_lock.release()
                release["device_lock_released"] = "ok"
            except Exception as exc:
                release["device_lock_released"] = f"failed: {exc}"
        write_json(out / "release.json", release)
        logger.info("[occupancy] released: {}", release)


def _write_failure_summary(out: Path, plan: dict, release: dict) -> None:
    """Leave a summary.md carrying the failure reason when the run did not complete."""
    reason = str(release.get("error", "unknown failure"))
    lines: list[str] = [
        "# square1500 seed ABBA comparison — FAILED",
        "",
        f"- Checkpoint sha256: `{plan['checkpoint_sha256']}`",
        f"- Reason: **{reason}**",
        f"- Devices: {release.get('devices', 'n/a')}",
        f"- Started (UTC): {release.get('started_utc', 'n/a')}",
        f"- Released (UTC): {release.get('released_utc', 'n/a')}",
        f"- Run completed: {release.get('run_completed', False)}",
        f"- Closed normally: {release.get('closed_normally', False)}",
        "",
        "The paired comparison was stopped; the retained captures are for diagnosis only.",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_summary(out: Path, plan: dict, manifest: dict, abba: dict) -> None:
    """Human-readable summary.md (schedule, stats, peaks, verdicts, deviations)."""
    lines: list[str] = [
        "# square1500 seed ABBA comparison — summary",
        "",
        f"- Checkpoint sha256: `{plan['checkpoint_sha256']}`",
        f"- Frozen exposure: **{manifest['frozen_exposure_ms']:.5f} ms** (reference seed `{plan['reference']}`)",
        f"- Bit depth: {plan['bit_depth']} (full scale {manifest['full_scale']:.0f})",
        f"- Reach: slope {plan['reach']['slope_px']:.2f} px, reach {plan['reach']['reach_px']:.1f} px, "
        f"max side {plan['reach']['max_side_px']:.1f} px (requested {plan['reach']['side_px']:g} px)",
        f"- Capture order: `{' '.join(plan['capture_order'])}`",
        f"- Frames per capture: {plan['frames_per_capture']}",
        "",
        "## Per-seed phase stats (raw radians)",
        "",
        "| seed | min | max | rms | finite |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, info in plan["seeds"].items():
        s = info["stats"]
        lines.append(
            f"| {name} | {s['min_rad']:.4f} | {s['max_rad']:.4f} | "
            f"{s['rms_rad']:.4f} | {s['finite']} |"
        )
    lines += [
        "",
        "## Landing peaks at frozen exposure",
        "",
        "| seed | peak | fraction | status |",
        "| --- | --- | --- | --- |",
    ]
    for name, info in manifest["landing"].items():
        lines.append(
            f"| {name} | {info['peak_counts']:.1f} | {info['fraction_of_full_scale']:.3f} | "
            f"{info['status']} |"
        )
    lines += [
        "",
        "## Captures",
        "",
        "| # | seed | cv | peak | ee_roi | centroid |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in manifest["captures"]:
        c = row["centroid_xy"]
        lines.append(
            f"| {row['round']} | {row['seed']} | {row['metrics']['cv']:.4f} | "
            f"{row['peak_counts']:.1f} | {row['metrics']['ee_roi']:.4f} | "
            f"{('%.1f, %.1f' % (c[0], c[1])) if c else 'n/a'} |"
        )
    lines += ["", "## ABBA verdicts (metric = cv)", ""]
    for key, value in abba["pairs"].items():
        lines.append(
            f"- **{key}**: baseline {value['baseline']:.4f}, candidate {value['candidate']:.4f}, "
            f"drift {value['drift']:.4f}, tolerance {value['tolerance']:.4f}, "
            f"accepted = **{value['accepted']}**"
        )
    lines += [
        "",
        "## Deviations from the naive schedule",
        "",
        "- `--rounds` is accepted for interface symmetry only: the ABBA schedule is the "
        "three disjoint contiguous A–B–B–A windows (one per unordered pair) = 12 captures, "
        "4 per seed. A monotonic 4-round × 3-capture schedule cannot yield three disjoint "
        "ABBA windows, so `--rounds` is recorded, not consumed.",
        "- `--seed-int` is recorded for reproducibility; all three constructions are "
        "deterministic given the checkpoint (no RNG), so the integer is not consumed by "
        "the seed builders today.",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output directory (refused when reused with manifest.json)",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--side", type=float, default=1500.0)
    parser.add_argument("--window-span", type=float, default=758.0)
    parser.add_argument("--window-step", type=float, default=2.0)
    parser.add_argument("--zero-radius", type=float, default=0.0)
    parser.add_argument("--period", type=float, default=64.0)
    tilt = parser.add_mutually_exclusive_group()
    tilt.add_argument(
        "--tilt",
        dest="tilt",
        action="store_true",
        help="carrier the target away from the zero order",
    )
    tilt.add_argument(
        "--no-tilt",
        dest="tilt",
        action="store_false",
        help="target sits at the zero order (default)",
    )
    parser.set_defaults(tilt=False)
    parser.add_argument(
        "--target-xy",
        type=float,
        nargs=2,
        default=(1221.0, 1104.0),
        metavar=("X", "Y"),
        help="target centre (CCD px)",
    )
    parser.add_argument(
        "--seeds",
        default="gs,transport,backprojection",
        help="comma-separated constructions to compare",
    )
    parser.add_argument(
        "--reference", default="gs", help="seed that carries the exposure ladder"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=4,
        help="recorded only (ABBA schedule is fixed at 4 captures/seed)",
    )
    parser.add_argument(
        "--frames", type=int, default=3, help="frames per capture (mean)"
    )
    parser.add_argument("--bit-depth", type=int, default=8)
    parser.add_argument("--camera", default="daheng")
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument(
        "--seed-int",
        type=int,
        default=0,
        help="reproducibility integer (recorded; constructions are deterministic)",
    )
    parser.add_argument(
        "--fixed-exposure-ms",
        type=float,
        default=0.0,
        help="override the ladder with a fixed frozen exposure (ms); 0 = ladder",
    )
    parser.add_argument(
        "--ladder-start-ms",
        type=float,
        default=0.5,
        help="reference-seed ladder start (ms)",
    )
    parser.add_argument(
        "--ladder-floor-ms",
        type=float,
        default=0.05,
        help="ladder floor (ms); max with the camera floor",
    )
    parser.add_argument("--max-ladder-steps", type=int, default=12)
    parser.add_argument(
        "--execute", action="store_true", help="open devices and capture"
    )
    parser.add_argument(
        "--wait-until-idle",
        type=float,
        default=0.0,
        help="seconds to wait for the device lane before opening",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.frames < 1:
        raise ValueError("--frames must be >= 1")
    if args.bit_depth < 8 or args.bit_depth > 16:
        raise ValueError("--bit-depth must be in 8..16")
    if args.max_ladder_steps < 1:
        raise ValueError("--max-ladder-steps must be >= 1")
    seed_names = parse_seeds(args.seeds)
    if args.reference not in seed_names:
        raise ValueError(f"--reference {args.reference!r} not in --seeds {seed_names}")
    if not np.isfinite(args.wait_until_idle) or args.wait_until_idle < 0:
        raise ValueError("--wait-until-idle must be finite and non-negative")

    resolved = resolve(args)
    plan = build_plan(resolved, args)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "plan.json", plan)

    print("square1500 seed ABBA plan")
    print(
        f"  checkpoint: {plan['checkpoint']}  sha256={plan['checkpoint_sha256'][:16]}..."
    )
    print(
        f"  geometry: side {plan['geometry']['side']:g} px, target "
        f"{plan['geometry']['target_xy']}, zero {plan['geometry']['zero_xy']}, "
        f"tilt {plan['geometry']['tilt']}"
    )
    print(
        f"  grid {plan['grid']['grid_size']}: slope {plan['reach']['slope_px']:.1f} px, "
        f"reach {plan['reach']['reach_px']:.1f} px, max side "
        f"{plan['reach']['max_side_px']:.1f} px"
    )
    if not plan["reach"]["covers_side"]:
        raise ValueError(
            f"grid {plan['grid']['grid_size']} cannot reach side {plan['reach']['side_px']:g} px"
        )
    print(
        f"  reach covers side {plan['reach']['side_px']:g} px: "
        f"{plan['reach']['covers_side']}"
    )
    for name, info in plan["seeds"].items():
        s = info["stats"]
        print(
            f"  seed {name:14s} min {s['min_rad']:+.4f} max {s['max_rad']:+.4f} "
            f"rms {s['rms_rad']:.4f} rad finite={s['finite']}"
        )
    print(f"  capture order: {' '.join(plan['capture_order'])}")
    print(f"  captures per seed: {plan['captures_per_seed']}")
    if not args.execute:
        print(f"  offline plan written to {args.out / 'plan.json'} (no device opened)")
        return 0
    print(f"  executing hardware run into {args.out}")
    run_execute(resolved, plan, args)
    print(f"  done: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
