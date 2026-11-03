"""Measured broadening seed -> differentiable increments -> frozen-exposure ABBA.

Default is offline: no SDK imports, probing, waiting or device access. Remove
after the accompanying 2026-10-30 report is final and no rerun is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    DifferentiableUniformityOptimizer, FrozenDefocusForward,
    FrozenDefocusParameters, load_frozen_parameters,
)
from ao_shaping.algorithm.signal_processing.multishape_objective import ShapeObjective
from ao_shaping.tools.slm.transfer.capture import clock, to_panel
from ao_shaping.tools.slm.transfer.radiometry import count_rate

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/debug/slm_transfer/multishape_nogs_20261030"
SEED_RUN = DATA / "square1000_bandlimit_live_20261030_v2"
SEED_KEY = "F_blur4_full"
CHECKPOINT = DATA / "p20261030_s1000_square_plane_formation/model640/frozen_parameters.json"
NATIVE_CCD_SHAPE = (1944, 2592)
FULL_SCALE = 4095.


@dataclass(frozen=True)
class SeedArchive:
    """Validated phase seed and its matching archived measurement."""

    seed: np.ndarray
    camera_xy: np.ndarray
    capture_row: dict
    capture_path: Path
    measurement: dict
    parameters: FrozenDefocusParameters


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    path = Path(path)
    if path.is_dir():
        sha = hashlib.sha256()
        files = sorted(item for item in path.rglob("*") if item.is_file())
        if not files:
            raise ValueError(f"cannot digest empty checkpoint directory {path}")
        for item in files:
            sha.update(item.relative_to(path).as_posix().encode("utf-8"))
            sha.update(b"\0")
            sha.update(item.read_bytes())
        return sha.hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc


def _require_keys(saved, path: Path, keys):
    missing = [key for key in keys if key not in saved.files]
    if missing:
        raise ValueError(f"{path} missing arrays {missing}; found {saved.files}")


def _finite_measurement(contract: dict, key: str, *, positive=False,
                        nonnegative=False) -> float:
    if key not in contract:
        raise ValueError(f"measurement contract missing {key!r}")
    try:
        value = float(contract[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"measurement contract {key!r} must be numeric") from exc
    if not np.isfinite(value):
        raise ValueError(f"measurement contract {key!r} must be finite")
    if positive and value <= 0:
        raise ValueError(f"measurement contract {key!r} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"measurement contract {key!r} must be nonnegative")
    return value


def _validate_camera_grid(camera_xy: np.ndarray, native_shape: tuple[int, int]):
    if camera_xy.ndim != 3 or camera_xy.shape[-1] != 2:
        raise ValueError(f"camera_xy must be (H, W, 2), got {camera_xy.shape}")
    if camera_xy.shape[0] < 2 or camera_xy.shape[1] < 2:
        raise ValueError(f"camera_xy grid is too small: {camera_xy.shape}")
    if not np.isfinite(camera_xy).all():
        raise ValueError("camera_xy contains nonfinite coordinates")
    if not np.array_equal(camera_xy, np.rint(camera_xy)):
        raise ValueError("camera_xy must contain integer-valued native pixel coordinates")
    x_axis = camera_xy[0, :, 0]
    y_axis = camera_xy[:, 0, 1]
    if not np.array_equal(camera_xy[..., 0], np.broadcast_to(x_axis, camera_xy.shape[:2])):
        raise ValueError("camera_xy x coordinates must vary only along columns")
    if not np.array_equal(camera_xy[..., 1], np.broadcast_to(y_axis[:, None], camera_xy.shape[:2])):
        raise ValueError("camera_xy y coordinates must vary only along rows")
    for name, axis in (("x", x_axis), ("y", y_axis)):
        step = axis[1] - axis[0]
        if step <= 0 or not np.allclose(np.diff(axis), step, rtol=0., atol=1e-9):
            raise ValueError(f"camera_xy {name} axis must be increasing and uniform")
    height, width = native_shape
    if x_axis[0] < 0 or x_axis[-1] > width - 1:
        raise ValueError("camera_xy x coordinates fall outside the native CCD")
    if y_axis[0] < 0 or y_axis[-1] > height - 1:
        raise ValueError("camera_xy y coordinates fall outside the native CCD")


def _validate_native_shape(native_shape) -> tuple[int, int]:
    try:
        height, width = (int(v) for v in native_shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("native_shape must contain two integers") from exc
    if height < 2 or width < 2:
        raise ValueError("native_shape must be at least 2x2")
    return height, width


def load_seed_contract(seed_run: Path, checkpoint: Path, *, seed_key: str = SEED_KEY,
                       native_shape: tuple[int, int] = NATIVE_CCD_SHAPE) -> SeedArchive:
    """Load one uniquely labelled seed and verify its archived measurement contract.

    The phase and camera grid come from ``plan.npz``. The matching capture is
    resolved by manifest label, never by a hard-coded file number. The
    measurement contract is authoritative for radiometry and morphology.
    """
    seed_run = Path(seed_run)
    if not isinstance(seed_key, str) or not seed_key:
        raise ValueError("seed key must be a nonempty string")
    native_shape = _validate_native_shape(native_shape)
    manifest_path = seed_run / "manifest.json"
    rows = _read_json(manifest_path)
    if not isinstance(rows, list):
        raise ValueError(f"{manifest_path} must contain a list")
    matches = [row for row in rows if isinstance(row, dict) and row.get("label") == seed_key]
    if len(matches) != 1:
        raise ValueError(
            f"seed label {seed_key!r} must match exactly one manifest row; "
            f"found {len(matches)}"
        )
    capture_row = matches[0]
    capture_name = capture_row.get("file")
    if (not isinstance(capture_name, str) or not capture_name
            or Path(capture_name).name != capture_name):
        raise ValueError(f"seed label {seed_key!r} has invalid capture filename")
    capture_path = seed_run / capture_name
    if not capture_path.is_file():
        raise ValueError(f"capture for seed label {seed_key!r} is missing: {capture_path}")

    parameters = load_frozen_parameters(checkpoint).effective_parameters()
    grid_size = int(parameters.grid_size)
    if grid_size < 2:
        raise ValueError(f"checkpoint grid_size must be at least 2, got {grid_size}")

    plan_path = seed_run / "plan.npz"
    with np.load(plan_path) as saved:
        _require_keys(saved, plan_path, (seed_key, "camera_xy"))
        seed = np.asarray(saved[seed_key], dtype=np.float32)
        camera_xy = np.asarray(saved["camera_xy"], dtype=np.float64)
    if seed.shape != (grid_size, grid_size):
        raise ValueError(
            f"seed shape {seed.shape} does not match checkpoint grid "
            f"{(grid_size, grid_size)}"
        )
    if not np.isfinite(seed).all():
        raise ValueError("seed phase contains nonfinite values")
    _validate_camera_grid(camera_xy, native_shape)

    with np.load(capture_path) as saved:
        _require_keys(saved, capture_path, ("raw_frames", "model_phase", "camera_xy"))
        capture_phase = np.asarray(saved["model_phase"], dtype=np.float32)
        capture_xy = np.asarray(saved["camera_xy"], dtype=np.float64)
        raw_frames = saved["raw_frames"]
        if raw_frames.ndim != 3 or raw_frames.shape[0] < 3:
            raise ValueError(f"{capture_path} must contain at least three raw frames")
        if tuple(raw_frames.shape[1:]) != tuple(native_shape):
            raise ValueError(
                f"{capture_path} native frame shape {raw_frames.shape[1:]} "
                f"does not match {native_shape}"
            )
        if not np.issubdtype(raw_frames.dtype, np.integer):
            raise ValueError(f"{capture_path} raw frames must be integer counts")
        if raw_frames.size and float(np.min(raw_frames)) < 0:
            raise ValueError(f"{capture_path} raw frames must be nonnegative counts")
    if not np.isfinite(capture_phase).all():
        raise ValueError(f"{capture_name} model_phase contains nonfinite values")
    if capture_phase.shape != seed.shape or not np.array_equal(capture_phase, seed):
        maxdiff = (float(np.max(np.abs(capture_phase - seed)))
                   if capture_phase.shape == seed.shape else float("nan"))
        raise ValueError(
            f"plan seed {seed_key!r} does not exactly match {capture_name} "
            f"model_phase (maxdiff={maxdiff})"
        )
    if not np.array_equal(capture_xy, camera_xy):
        raise ValueError(f"{capture_name} camera_xy does not exactly match plan.npz")

    raw_contract = _read_json(seed_run / "measurement_contract.json")
    if not isinstance(raw_contract, dict):
        raise ValueError("measurement_contract.json must contain an object")
    measurement = dict(raw_contract)
    measurement["background_counts"] = _finite_measurement(
        raw_contract, "background_counts")
    measurement["coverage_threshold_counts"] = _finite_measurement(
        raw_contract, "coverage_threshold_counts", positive=True)
    measurement["exposure_ms"] = _finite_measurement(
        raw_contract, "exposure_ms", positive=True)
    measurement["noise_counts"] = _finite_measurement(
        raw_contract, "noise_counts", nonnegative=True)
    measurement["full_scale"] = _finite_measurement(
        raw_contract, "full_scale", positive=True)
    measurement["smoothing_sigma_native_px"] = _finite_measurement(
        raw_contract, "smoothing_sigma_native_px", positive=True)
    try:
        manifest_exposure = float(capture_row["exposure_ms"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"seed label {seed_key!r} has invalid exposure") from exc
    if (not np.isfinite(manifest_exposure) or manifest_exposure <= 0
            or abs(manifest_exposure - measurement["exposure_ms"])
            > max(1e-6, 1e-3 * measurement["exposure_ms"])):
        raise ValueError("manifest and measurement-contract exposures disagree")
    if float(np.max(raw_frames)) > measurement["full_scale"]:
        raise ValueError("raw frames exceed the declared full scale")

    return SeedArchive(
        seed=seed,
        camera_xy=camera_xy,
        capture_row=capture_row,
        capture_path=capture_path,
        measurement=measurement,
        parameters=parameters,
    )


def measurement_kwargs(measurement: dict) -> dict:
    """Radiometry and morphology values that must come from the frozen contract."""
    return dict(
        smoothing_sigma=float(measurement["smoothing_sigma_native_px"]),
        full_scale=float(measurement["full_scale"]),
    )


def masks(xy):
    x, y = xy[..., 0], xy[..., 1]
    box = (abs(x-1917.) < 500) & (abs(y-943.) < 500)
    core = (abs(x-1917.) < 480) & (abs(y-943.) < 480)
    inner = (abs(x-1917.) < 430) & (abs(y-943.) < 430)
    roi = (x >= 1300) & (x < 2540) & (y >= 300) & (y < 1588)
    zero = (x-1267.)**2 + (y-943.)**2 < 120.**2
    return dict(box=box, core=core, edge=core & ~inner, roi=roi & ~zero,
                background=roi & ~box & ~zero)


def candidate_step(forward, objective, seed, measured, cap):
    """One existing measured-image VJP/Adam step, then smooth raw-radian control."""
    if not np.isfinite(cap) or cap <= 0 or cap > .1:
        raise ValueError("increment cap must lie in (0, 0.1] rad")
    opt = DifferentiableUniformityOptimizer(forward, objective, seed, lr=cap,
                                          step_rms_cap=cap)
    delta = gaussian_filter(opt.update(measured)-seed, 3.)
    weights = forward.illumination_weights()
    yy, xx = np.mgrid[:seed.shape[0], :seed.shape[1]]
    design = np.stack([np.ones_like(xx), xx/seed.shape[1], yy/seed.shape[0]], -1)
    # Joint weighted plane removal preserves the steering carrier in the seed.
    a = design.reshape(-1, 3)
    w = weights.ravel()
    coeff = np.linalg.solve(a.T @ (w[:, None]*a), a.T @ (w*delta.ravel()))
    delta -= design @ coeff
    rms = float(np.sqrt(np.sum(weights*delta**2)/weights.sum()))
    scale = min(1., cap/max(rms, 1e-12), .3/max(float(abs(delta).max()), 1e-12))
    delta = np.asarray(delta*scale, dtype=np.float32)
    if not np.isfinite(delta).all() or rms < 1e-10:
        raise ValueError("nonfinite or unresolved differentiable increment")
    return np.asarray(seed+delta, np.float32), dict(
        optimizer=opt.loss_history, smoothing_sigma_cells=3.,
        rms_rad=float(np.sqrt(np.sum(weights*delta**2)/weights.sum())),
        max_abs_rad=float(abs(delta).max()))


def metrics(raw, exposure, background, threshold, native_masks, objective, xy, *,
            smoothing_sigma=8., full_scale=FULL_SCALE,
            expected_shape=NATIVE_CCD_SHAPE):
    """Frozen background/coverage threshold; native-pixel morphology and energy."""
    import torch
    expected_shape = _validate_native_shape(expected_shape)
    if raw.ndim != 3 or raw.shape[0] < 3 or tuple(raw.shape[1:]) != expected_shape:
        raise ValueError(
            f"three or more native Mono12 frames of shape {expected_shape} required"
        )
    if not np.isfinite([smoothing_sigma, full_scale]).all() or smoothing_sigma <= 0 or full_scale <= 0:
        raise ValueError("positive finite smoothing and full scale required")
    _validate_camera_grid(np.asarray(xy, dtype=np.float64), expected_shape)
    for key in ("box", "core", "roi"):
        if not np.asarray(native_masks[key], dtype=bool).any():
            raise ValueError(f"native mask {key!r} must be nonempty")
    frame = raw.mean(0)
    signal = np.maximum(frame-background, 0.)
    m = native_masks
    blurred = gaussian_filter(signal, float(smoothing_sigma))
    filled = (blurred >= threshold) & m["roi"]
    box, core = m["box"], m["core"]
    values = signal[core]
    mean = float(values.mean())
    total = max(float(signal[m["roi"]].sum()), 1e-9)
    rate = count_rate(signal[xy[..., 1].astype(int), xy[..., 0].astype(int)],
                      dict(exposure_ms=exposure)).astype(np.float32)
    with torch.no_grad():
        loss, terms = objective.evaluate(torch.from_numpy(rate), torch.zeros((2, 2)))
    edge = m["edge"]
    boundary_error = (float(abs(blurred[edge]/max(mean, 1e-9)-1).mean())
                      if np.asarray(edge, dtype=bool).any() else 0.)
    return dict(coverage=float(filled[core].mean()),
                iou=float((filled & box).sum()/max((filled | box).sum(), 1)),
                boundary_error=boundary_error,
                cv=float(values.std()/max(mean, 1e-9)),
                target_rate=float(signal[box].sum()/exposure),
                local_ee=float(signal[box].sum()/total), mean=mean,
                clipped_target=int((raw[:, box] >= .98*full_scale).sum()),
                clipped_full=int((raw >= .98*full_scale).sum()),
                loss=float(loss), terms=terms)


def accept_abba(rows, repeats):
    """Both B repeats must beat time-interpolated A beyond observed drift."""
    if len(rows) != 4 or len(repeats) < 3:
        raise ValueError("ABBA plus at least three seed repeats required")
    a1, b1, b2, a2 = rows
    keys = ("coverage", "iou", "boundary_error", "cv", "target_rate", "local_ee", "loss")
    if not all(np.isfinite(r["metrics"][k]) for r in [*rows, *repeats] for k in keys):
        return dict(accepted=False, reasons=["nonfinite metric"])
    if any(r["metrics"]["clipped_target"] for r in [*rows, *repeats]):
        return dict(accepted=False, reasons=["full target saturated"])
    exposures = [r["exposure_ms"] for r in rows]
    if min(exposures) <= 0 or max(exposures)/min(exposures) > 1.001:
        return dict(accepted=False, reasons=["exposure readback mismatch"])
    t0, t1 = a1["monotonic_ns"], a2["monotonic_ns"]
    if not t0 < b1["monotonic_ns"] < b2["monotonic_ns"] < t1:
        return dict(accepted=False, reasons=["invalid ABBA timestamps"])
    if abs(a2["metrics"]["target_rate"]-a1["metrics"]["target_rate"]) > .05*max(a1["metrics"]["target_rate"], 1e-9):
        return dict(accepted=False, reasons=["same-phase energy drift >5%; rebaseline in a new session"])
    floors = dict(coverage=.002, iou=.002, boundary_error=.005, cv=.01,
                  target_rate=0., local_ee=.005, loss=.001)
    resolution = {k: max(floors[k], 3*float(np.std(
        [r["metrics"][k] for r in repeats], ddof=1)),
        3*abs(a2["metrics"][k]-a1["metrics"][k])) for k in keys}
    reasons = []
    for b in (b1, b2):
        fraction = (b["monotonic_ns"]-t0)/(t1-t0)
        ref = {k: a1["metrics"][k]*(1-fraction)+a2["metrics"][k]*fraction for k in keys}
        v = b["metrics"]
        for k in ("coverage", "iou"):
            if v[k]-ref[k] <= resolution[k]:
                reasons.append(f"{b['label']}: unresolved {k} improvement")
        if ref["loss"]-v["loss"] <= resolution["loss"]:
            reasons.append(f"{b['label']}: unresolved formation loss improvement")
        for k in ("boundary_error", "cv"):
            if v[k] > ref[k]+resolution[k]:
                reasons.append(f"{b['label']}: {k} regression")
        if v["target_rate"] < ref["target_rate"]*.98:
            reasons.append(f"{b['label']}: target energy loss >2%")
        if v["local_ee"] < ref["local_ee"]-.005:
            reasons.append(f"{b['label']}: local EE loss >0.005")
    return dict(accepted=not reasons, reasons=reasons, resolution=resolution)


def execute(args, seed, xy, forward, objective, measurement):
    # No driver import or availability probe is reachable on the offline path.
    from ao_shaping.tools.kernels.device_mutex import DeviceLock, wait_until_idle
    verdict = wait_until_idle(args.wait_until_idle, lock=DeviceLock.default(),
                              own_paths=[Path(__file__).resolve()])
    if not verdict.ok:
        raise RuntimeError(f"{verdict.status}: {verdict.detail}; {verdict.remedy}")
    lock = verdict.lock
    if lock is None:
        raise RuntimeError("idle verdict did not return an owned lock")
    records = []
    try:
        from ao_shaping.tools.slm.transfer.hardware import prepare_camera_sdk
        prepare_camera_sdk("daheng")
        from ao_shaping.drivers.ccd.common import create_camera, get_camera_exposure_ms
        from ao_shaping.drivers.slm.santec import Santec
        native_shape = _validate_native_shape(NATIVE_CCD_SHAPE)
        yy, xx = np.mgrid[:native_shape[0], :native_shape[1]]
        native_masks = masks(np.stack([xx, yy], -1))
        with create_camera("daheng", 0, exposure_time_ms=.5, bit_depth=12) as cam:
            with Santec(slm_number=1, wavelength=1064, video_mode=0) as slm:
                slm.set_phase_overlay(None)
                try:
                    def capture(phase, exposure, label):
                        gray = slm.create_phase_from_array(to_panel(phase))
                        slm.display_data(gray)
                        cam.reset_exposure_time(exposure)
                        time.sleep(.8)
                        for _ in range(3):
                            cam.get_numpy_image(n_sample=1, skip_first=False)
                        frames, stamps = [], []
                        for _ in range(5):
                            stamp = clock()
                            frame = np.asarray(cam.get_numpy_image(n_sample=1, skip_first=False)).copy()
                            stamp.update(end_monotonic_ns=time.monotonic_ns(),
                                         exposure_ms=float(get_camera_exposure_ms(cam)))
                            if frame.shape != native_shape or frame.dtype != np.uint16:
                                raise RuntimeError("expected native unpacked Mono12 uint16 frame")
                            frames.append(frame)
                            stamps.append(stamp)
                        raw = np.stack(frames)
                        row = dict(label=label, exposure_ms=stamps[0]["exposure_ms"], frames=stamps,
                                   file=f"capture_{len(records):04d}.npz", **clock())
                        np.savez_compressed(args.out/row["file"], raw_frames=raw,
                                            model_phase=phase, panel_gray=gray, camera_xy=xy)
                        records.append(row)
                        write_json(args.out/"manifest.json", records)
                        if max(s["exposure_ms"] for s in stamps)/min(s["exposure_ms"] for s in stamps) > 1.001:
                            raise RuntimeError("exposure changed inside capture")
                        return row, raw

                    row, raw = capture(seed, .5, "seed_safe_meter")
                    peak = float(raw[:, native_masks["box"]].max())
                    if peak >= .98*4095 or peak < 20:
                        raise RuntimeError("unsafe or unresolved metering; inspect archived frames")
                    exposure = min(16.87, row["exposure_ms"]*1400/max(peak, 1))
                    row, raw = capture(seed, exposure, "seed_calibration_0")
                    bg = float(np.median(raw.mean(0)[native_masks["background"]]))
                    noise = float(np.sqrt(raw[:, native_masks["background"]].var(0, ddof=1).mean()/len(raw)))
                    mean = float(np.maximum(raw.mean(0)-bg, 0)[native_masks["core"]].mean())
                    threshold = max(3*noise, .2*mean)
                    write_json(args.out/"measurement_contract.json", dict(background_counts=bg,
                        coverage_threshold_counts=threshold, exposure_ms=row["exposure_ms"],
                        noise_counts=noise, full_scale=measurement["full_scale"],
                        smoothing_sigma_native_px=measurement["smoothing_sigma_native_px"]))
                    if mean <= noise or raw[:, native_masks["box"]].max() >= .98*measurement["full_scale"]:
                        raise RuntimeError("seed signal unresolved or full target saturated")

                    def evaluate(row, raw):
                        row["metrics"] = metrics(raw, row["exposure_ms"], bg, threshold,
                                                 native_masks, objective, xy,
                                                 expected_shape=native_shape,
                                                 **measurement_kwargs(measurement))
                        write_json(args.out/"manifest.json", records)
                        return row

                    repeats = [evaluate(row, raw)]
                    for i in (1, 2):
                        row, raw = capture(seed, exposure, f"seed_calibration_{i}")
                        repeats.append(evaluate(row, raw))
                    accepted = seed.copy()
                    np.save(args.out/"accepted_phase.npy", accepted)
                    rejected = 0
                    decisions = []
                    for block in range(args.blocks):
                        row, raw = capture(accepted, exposure, f"block{block}_anchor")
                        evaluate(row, raw)
                        if row["metrics"]["clipped_target"]:
                            raise RuntimeError("anchor saturated; start a new lower-exposure session")
                        forward.exposure_ms = row["exposure_ms"]
                        sampled = np.maximum(raw.mean(0)-bg, 0)[xy[..., 1].astype(int), xy[..., 0].astype(int)]
                        rate = count_rate(sampled, row)
                        candidate, proposal = candidate_step(forward, objective, accepted, rate,
                                                              .03/(2**rejected))
                        np.savez_compressed(args.out/f"proposal_{block:02d}.npz",
                                            seed=accepted, candidate=candidate, delta=candidate-accepted)
                        write_json(args.out/f"proposal_{block:02d}.json", proposal)

                        def abba(phase, tag, baseline=accepted, block_index=block):
                            rows = []
                            for label, p in (("A1", baseline), ("B1", phase),
                                             ("B2", phase), ("A2", baseline)):
                                r, f = capture(p, exposure, f"block{block_index}_{tag}_{label}")
                                rows.append(evaluate(r, f))
                            return accept_abba(rows, repeats)

                        plus = abba(candidate, "gradient")
                        # Opposite direction is diagnostic only, never auto-promoted.
                        minus = abba(2*accepted-candidate, "reverse") if block == 0 else None
                        direction_failed = minus is not None and minus["accepted"]
                        take = plus["accepted"] and not direction_failed
                        decisions.append(dict(block=block, plus=plus, reverse=minus, accepted=take,
                                              direction_failed=direction_failed))
                        write_json(args.out/"decisions.json", decisions)
                        if take:
                            accepted = candidate
                            rejected = 0
                            np.save(args.out/"accepted_phase.npy", accepted)
                        else:
                            rejected += 1
                        if direction_failed or rejected >= 2:
                            break
                finally:
                    # Direct zero gray is the driver contract for a flat reset.
                    slm.display_data(np.zeros((1200, 1920), dtype=np.uint16))
                    time.sleep(.3)
    finally:
        # This encloses both device context managers, including setup failures.
        lock.release()
    write_json(args.out/"released.json", clock())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--seed-run", type=Path, default=SEED_RUN)
    parser.add_argument("--seed-key", default=SEED_KEY)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--wait-until-idle", type=float, default=0.)
    parser.add_argument("--blocks", type=int, default=3)
    args = parser.parse_args()
    if args.blocks < 1 or args.blocks > 5 or not np.isfinite(args.wait_until_idle) or args.wait_until_idle < 0:
        parser.error("1..5 blocks and a finite nonnegative wait required")
    import torch
    torch.set_num_threads(4)
    archive = load_seed_contract(args.seed_run, args.checkpoint, seed_key=args.seed_key)
    params, seed, xy = archive.parameters, archive.seed, archive.camera_xy
    measurement = archive.measurement
    anchor_row = dict(archive.capture_row, exposure_ms=measurement["exposure_ms"])
    with np.load(archive.capture_path) as saved:
        raw = saved["raw_frames"]
        historical_frame = raw.mean(0)
    forward = FrozenDefocusForward(params, xy, device="cpu")
    forward.exposure_ms = measurement["exposure_ms"]
    objective = ShapeObjective(masks(xy)["box"].astype(np.float32), formation=True)
    measured = count_rate(np.maximum(
        historical_frame-measurement["background_counts"], 0)[
            xy[..., 1].astype(int), xy[..., 0].astype(int)], anchor_row)
    candidate, info = candidate_step(forward, objective, seed, measured, .03)
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.out/"offline_plan.npz", seed=seed, candidate=candidate,
                        reverse=2*seed-candidate, camera_xy=xy)
    config = dict(execute=args.execute, checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=digest(args.checkpoint), parameters=params.to_dict(),
        seed_run=str(args.seed_run.resolve()), seed_key=args.seed_key,
        seed_sha256=digest(args.seed_run/"plan.npz"),
        seed_capture=archive.capture_row["file"],
        seed_capture_sha256=digest(archive.capture_path),
        measurement_contract=measurement,
        measurement_contract_sha256=digest(args.seed_run/"measurement_contract.json"),
        centre=[1917, 943], side=1000, video_mode=0,
        formation=True, weights=objective.weights, units="native counts/ms, no counts_scale division",
        historical_anchor=archive.capture_row, offline_proposal=info,
        blocks=args.blocks, wait_until_idle_s=args.wait_until_idle,
        acceptance="both B improve coverage/IoU/loss beyond 3x repeats and drift; no boundary/CV regression; energy >=98%; EE loss <=0.005; full-target clipping veto",
        note="offline proposal is not an experimental result; execute reacquires anchor and recomputes",
        **clock())
    write_json(args.out/"config.json", config)
    print(json.dumps(dict(out=str(args.out), execute=args.execute, proposal=info)), flush=True)
    if args.execute:
        execute(args, seed, xy, forward, objective, measurement)


if __name__ == "__main__":
    main()
