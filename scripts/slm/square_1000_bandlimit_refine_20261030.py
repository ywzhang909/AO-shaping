"""Measured low-frequency refinement of the 1000 px band-limited square seed.

The seed is ``F_blur4_full`` from the 2026-10-30 band-limit session.  Updates
are restricted to a smooth DCT basis, with piston and global tilt removed, so
the calibrated steering carrier is never replaced by a black-box search.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from ao_shaping.algorithm.signal_processing.multishape_objective import ShapeObjective
from ao_shaping.tools.slm.transfer.capture import clock, to_panel

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/debug/slm_transfer/multishape_nogs_20261030"
SEED_RUN = DATA / "square1000_bandlimit_live_20261030_v2"
SEED_KEY = "F_blur4_full"
FULL_SCALE = 4095.0


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def dct_basis(n: int, size: int) -> np.ndarray:
    """Return unit-RMS low-frequency 2-D DCT modes, excluding piston."""
    x = np.arange(size) + 0.5
    modes = []
    for v in range(n):
        for u in range(n):
            if u == v == 0:
                continue
            mode = np.cos(np.pi * u * x[:, None] / size) * np.cos(
                np.pi * v * x[None, :] / size
            )
            modes.append(mode / np.sqrt(np.mean(mode**2)))
    return np.stack(modes)


def remove_plane(values: np.ndarray, sigma: float = 0.0) -> np.ndarray:
    """Remove piston and global tilt from a control correction."""
    result = np.asarray(values, dtype=np.float64)
    if sigma > 0:
        result = gaussian_filter(result, sigma=sigma, mode="nearest")
    yy, xx = np.mgrid[: result.shape[0], : result.shape[1]]
    design = np.stack(
        [np.ones_like(xx), xx / result.shape[1], yy / result.shape[0]], axis=-1
    ).reshape(-1, 3)
    coeff, *_ = np.linalg.lstsq(design, result.ravel(), rcond=None)
    return np.asarray(result - (design @ coeff).reshape(result.shape), dtype=np.float32)


def edge_positions(blurred: np.ndarray) -> dict[str, float]:
    """Locate the four target edges from central intensity profiles."""
    x_profile = blurred[700:1200, :].mean(axis=0)
    y_profile = blurred[:, 1700:2200].mean(axis=1)

    def peak(values: np.ndarray, start: int, stop: int, negative: bool = False) -> float:
        section = values[start:stop]
        grad = np.gradient(section)
        if negative:
            grad = -grad
        return float(start + int(np.argmax(grad)))

    return dict(
        left=peak(x_profile, 1350, 1650),
        right=peak(x_profile, 2200, 2480, negative=True),
        top=peak(y_profile, 350, 650),
        bottom=peak(y_profile, 1250, 1520, negative=True),
    )


def metrics(raw, exposure, background, threshold, native_masks, objective, xy):
    """Native-pixel morphology, energy, stripe, and edge metrics."""
    if raw.ndim != 3 or raw.shape[0] < 3 or raw.shape[1:] != (1944, 2592):
        raise ValueError("three or more native Mono12 frames required")
    frame = raw.mean(0)
    signal = np.maximum(frame - background, 0.0)
    m = native_masks
    blurred = gaussian_filter(signal, 8.0)
    filled = (blurred >= threshold) & m["roi"]
    box, core, inner = m["box"], m["core"], m["inner"]
    values = signal[core]
    mean = float(values.mean())
    total = max(float(signal[m["roi"]].sum()), 1e-9)
    rate = protocol_count_rate(signal[xy[..., 1].astype(int), xy[..., 0].astype(int)],
                               exposure)
    with torch.no_grad():
        loss, terms = objective.evaluate(
            torch.from_numpy(rate.astype(np.float32)), torch.zeros((2, 2))
        )
    smooth = gaussian_filter(signal, 30.0)
    stripe = signal - smooth
    edges = edge_positions(blurred)
    expected = dict(left=1417.0, right=2417.0, top=443.0, bottom=1443.0)
    return dict(
        coverage=float(filled[core].mean()),
        iou=float((filled & box).sum() / max((filled | box).sum(), 1)),
        boundary_error=float(abs(blurred[m["edge"]] / max(mean, 1e-9) - 1).mean()),
        cv=float(values.std() / max(mean, 1e-9)),
        stripe_rms=float(stripe[inner].std() / max(mean, 1e-9)),
        target_rate=float(signal[box].sum() / exposure),
        local_ee=float(signal[box].sum() / total),
        mean=mean,
        clipped_target=int((raw[:, box] >= 0.98 * FULL_SCALE).sum()),
        clipped_full=int((raw >= 0.98 * FULL_SCALE).sum()),
        edge_positions=edges,
        edge_position_error=float(
            np.mean([abs(edges[k] - expected[k]) for k in expected])
        ),
        loss=float(loss),
        terms=terms,
    )


def protocol_count_rate(values: np.ndarray, exposure_ms: float) -> np.ndarray:
    """Convert background-subtracted native counts to counts/ms."""
    return np.maximum(values, 0.0) / exposure_ms


def accept_refinement(rows, repeats):
    """Accept only repeatable interior improvement at essentially fixed energy."""
    if len(rows) != 4 or len(repeats) < 3:
        raise ValueError("ABBA plus at least three seed repeats required")
    a1, b1, b2, a2 = rows
    keys = (
        "coverage",
        "iou",
        "boundary_error",
        "cv",
        "stripe_rms",
        "target_rate",
        "local_ee",
        "edge_position_error",
        "loss",
    )
    all_rows = [*rows, *repeats]
    if not all(np.isfinite(r["metrics"][k]) for r in all_rows for k in keys):
        return dict(accepted=False, reasons=["nonfinite metric"])
    if any(r["metrics"]["clipped_target"] for r in all_rows):
        return dict(accepted=False, reasons=["full target saturated"])
    exposures = [r["exposure_ms"] for r in rows]
    if min(exposures) <= 0 or max(exposures) / min(exposures) > 1.001:
        return dict(accepted=False, reasons=["exposure readback mismatch"])
    t0, t1 = a1["monotonic_ns"], a2["monotonic_ns"]
    if not t0 < b1["monotonic_ns"] < b2["monotonic_ns"] < t1:
        return dict(accepted=False, reasons=["invalid ABBA timestamps"])
    if abs(a2["metrics"]["target_rate"] - a1["metrics"]["target_rate"]) > 0.05 * max(
        a1["metrics"]["target_rate"], 1e-9
    ):
        return dict(
            accepted=False,
            reasons=["same-phase energy drift >5%; rebaseline in a new session"],
        )
    floors = dict(
        coverage=0.003,
        iou=0.003,
        boundary_error=0.006,
        cv=0.006,
        stripe_rms=0.004,
        target_rate=0.0,
        local_ee=0.005,
        edge_position_error=2.0,
        loss=0.001,
    )
    resolution = {
        k: max(
            floors[k],
            3.0 * float(np.std([r["metrics"][k] for r in repeats], ddof=1)),
            3.0 * abs(a2["metrics"][k] - a1["metrics"][k]),
        )
        for k in keys
    }
    reasons = []
    for b in (b1, b2):
        fraction = (b["monotonic_ns"] - t0) / (t1 - t0)
        ref = {
            k: a1["metrics"][k] * (1.0 - fraction) + a2["metrics"][k] * fraction
            for k in keys
        }
        v = b["metrics"]
        if v["coverage"] < ref["coverage"] - resolution["coverage"]:
            reasons.append(f"{b['label']}: coverage regression")
        if v["iou"] < ref["iou"] - resolution["iou"]:
            reasons.append(f"{b['label']}: IoU regression")
        if v["cv"] > ref["cv"] - resolution["cv"]:
            reasons.append(f"{b['label']}: unresolved CV improvement")
        if v["boundary_error"] > ref["boundary_error"] + resolution["boundary_error"]:
            reasons.append(f"{b['label']}: boundary error regression")
        if v["stripe_rms"] > ref["stripe_rms"] + resolution["stripe_rms"]:
            reasons.append(f"{b['label']}: stripe RMS regression")
        if v["edge_position_error"] > ref["edge_position_error"] + resolution[
            "edge_position_error"
        ]:
            reasons.append(f"{b['label']}: edge position regression")
        if v["target_rate"] < 0.98 * ref["target_rate"]:
            reasons.append(f"{b['label']}: target energy loss >2%")
        if v["local_ee"] < ref["local_ee"] - 0.005:
            reasons.append(f"{b['label']}: local EE loss >0.005")
        if ref["loss"] - v["loss"] <= resolution["loss"]:
            reasons.append(f"{b['label']}: unresolved formation loss improvement")
    return dict(accepted=not reasons, reasons=reasons, resolution=resolution)


def control_score(row, anchor_rate):
    """Measured ascent score for a near-formed square."""
    q = row["metrics"]
    energy_shortfall = max(0.0, 1.0 - q["target_rate"] / max(anchor_rate, 1e-9))
    return (
        -q["cv"]
        - 0.35 * q["boundary_error"]
        - 0.20 * q["stripe_rms"]
        + 0.20 * q["iou"]
        + 0.10 * q["coverage"]
        + 0.50 * q["local_ee"]
        - 2.0 * energy_shortfall
        - 0.005 * q["edge_position_error"]
    )


def load_protocol():
    path = ROOT / "scripts/slm/square_1000_diff_abba_20261030.py"
    spec = importlib.util.spec_from_file_location("square_protocol", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--wait-until-idle", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--delta", type=float, default=0.015)
    parser.add_argument("--step", type=float, default=0.012)
    parser.add_argument("--grid", type=int, default=6)
    parser.add_argument("--target-peak", type=float, default=1100.0)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 20:
        parser.error("epochs must be 1..20")
    if not 0.005 <= args.delta <= 0.03 or not 0.005 <= args.step <= 0.03:
        parser.error("delta and step must be in [0.005, 0.03] rad")
    if not 3 <= args.grid <= 8:
        parser.error("grid must be 3..8")
    if not 300.0 <= args.target_peak <= 3000.0:
        parser.error("target peak must be 300..3000 counts")

    torch.set_num_threads(4)
    protocol = load_protocol()
    with np.load(SEED_RUN / "plan.npz") as saved:
        seed = np.asarray(saved[SEED_KEY], dtype=np.float32)
        xy = saved["camera_xy"].copy()
    basis = dct_basis(args.grid, len(seed)).astype(np.float32)
    objective = ShapeObjective(protocol.masks(xy)["box"].astype(np.float32), formation=True)
    rng = np.random.default_rng(20261030)
    preview_sign = rng.choice([-1.0, 1.0], len(basis))
    preview_delta = remove_plane(np.tensordot(preview_sign * args.delta, basis, axes=1))
    preview_delta *= args.delta / max(
        float(np.sqrt(np.mean(preview_delta**2))), 1e-12
    )

    args.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.out / "plan.npz",
        seed=seed,
        basis=basis,
        camera_xy=xy,
        preview_delta=preview_delta,
    )
    write_json(
        args.out / "config.json",
        dict(
            execute=args.execute,
            seed_run=str(SEED_RUN),
            seed_key=SEED_KEY,
            seed_sha256=hashlib.sha256((SEED_RUN / "plan.npz").read_bytes()).hexdigest(),
            basis="unit-RMS 2D DCT excluding piston; global tilt removed",
            grid=args.grid,
            modes=len(basis),
            delta_rad=args.delta,
            step_rad=args.step,
            target_peak_counts=args.target_peak,
            optimizer="measured two-sided low-frequency SPGD; no model Jacobian",
            acceptance="energy >=98%; target unsaturation; CV improves; coverage/IoU/boundary/stripe/edge no material regression",
            **clock(),
        ),
    )
    if not args.execute:
        print(
            json.dumps(
                dict(
                    out=str(args.out),
                    modes=len(basis),
                    preview_rms=float(np.sqrt(np.mean(preview_delta**2))),
                )
            )
        )
        return

    from ao_shaping.tools.kernels.device_mutex import DeviceLock, wait_until_idle

    verdict = wait_until_idle(
        args.wait_until_idle,
        lock=DeviceLock.default(),
        own_paths=[Path(__file__).resolve()],
    )
    if not verdict.ok or verdict.lock is None:
        raise RuntimeError(verdict.headline())
    records, decisions = [], []
    try:
        from ao_shaping.tools.slm.transfer.hardware import prepare_camera_sdk

        prepare_camera_sdk("daheng")
        from ao_shaping.drivers.ccd.common import create_camera, get_camera_exposure_ms
        from ao_shaping.drivers.slm.santec import Santec

        yy, xx = np.mgrid[:1944, :2592]
        native_masks = protocol.masks(np.stack([xx, yy], axis=-1))
        native_masks["inner"] = (np.abs(xx - 1917.0) < 430.0) & (
            np.abs(yy - 943.0) < 430.0
        )
        with create_camera("daheng", 0, exposure_time_ms=0.5, bit_depth=12) as cam:
            with Santec(slm_number=1, wavelength=1064, video_mode=0) as slm:
                slm.set_phase_overlay(None)
                try:

                    def capture(phase, exposure, label):
                        gray = slm.create_phase_from_array(to_panel(phase))
                        slm.display_data(gray)
                        cam.reset_exposure_time(exposure)
                        time.sleep(0.8)
                        for _ in range(3):
                            cam.get_numpy_image(n_sample=1, skip_first=False)
                        raw = np.stack(
                            [
                                np.asarray(
                                    cam.get_numpy_image(n_sample=1, skip_first=False)
                                ).copy()
                                for _ in range(7)
                            ]
                        )
                        row = dict(
                            label=label,
                            exposure_ms=float(get_camera_exposure_ms(cam)),
                            file=f"capture_{len(records):04d}.npz",
                            **clock(),
                        )
                        if raw.shape != (7, 1944, 2592) or raw.dtype != np.uint16:
                            raise RuntimeError("expected native Mono12 uint16 frames")
                        np.savez_compressed(
                            args.out / row["file"],
                            raw_frames=raw,
                            model_phase=phase,
                            panel_gray=gray,
                            camera_xy=xy,
                        )
                        records.append(row)
                        write_json(args.out / "manifest.json", records)
                        return row, raw

                    row, raw = capture(seed, 0.5, "seed_safe_meter")
                    peak = float(raw[:, native_masks["box"]].max())
                    if peak < 20:
                        raise RuntimeError("unresolved target signal at safe exposure")
                    exposure = min(16.87, row["exposure_ms"] * args.target_peak / peak)
                    for attempt in range(4):
                        row, raw = capture(seed, exposure, f"seed_calibration_{attempt}")
                        peak = float(raw[:, native_masks["box"]].max())
                        if peak < 0.98 * FULL_SCALE:
                            break
                        exposure *= 0.80
                    else:
                        raise RuntimeError("target remained saturated after exposure search")
                    bg = float(np.median(raw.mean(0)[native_masks["background"]]))
                    noise = float(
                        np.sqrt(
                            raw[:, native_masks["background"]].var(0, ddof=1).mean()
                            / len(raw)
                        )
                    )
                    mean = float(
                        np.maximum(raw.mean(0) - bg, 0.0)[native_masks["core"]].mean()
                    )
                    threshold = max(3.0 * noise, 0.2 * mean)
                    write_json(
                        args.out / "measurement_contract.json",
                        dict(
                            background_counts=bg,
                            coverage_threshold_counts=threshold,
                            exposure_ms=float(row["exposure_ms"]),
                            noise_counts=noise,
                            full_scale=FULL_SCALE,
                            smoothing_sigma_native_px=8,
                            seed_key=SEED_KEY,
                        ),
                    )

                    def evaluate(row, raw):
                        row["metrics"] = metrics(
                            raw,
                            row["exposure_ms"],
                            bg,
                            threshold,
                            native_masks,
                            objective,
                            xy,
                        )
                        write_json(args.out / "manifest.json", records)
                        print(json.dumps(dict(label=row["label"], **row["metrics"])), flush=True)
                        return row

                    repeats = []
                    for i in range(5):
                        r, f = capture(seed, row["exposure_ms"], f"seed_repeat_{i}")
                        repeats.append(evaluate(r, f))
                    accepted = seed.copy()
                    np.save(args.out / "accepted_phase.npy", accepted)
                    m = np.zeros(len(basis))
                    v = np.zeros(len(basis))
                    rejected = 0
                    for epoch in range(args.epochs):
                        row, raw = capture(accepted, row["exposure_ms"], f"epoch{epoch}_A")
                        anchor = evaluate(row, raw)
                        sign = rng.choice([-1.0, 1.0], len(basis))
                        perturb = remove_plane(
                            np.tensordot(args.delta * sign, basis, axes=1), sigma=1.0
                        )
                        perturb *= args.delta / max(
                            float(np.sqrt(np.mean(perturb**2))), 1e-12
                        )
                        rp, fp = capture(accepted + perturb, row["exposure_ms"], f"epoch{epoch}_plus")
                        plus = evaluate(rp, fp)
                        rm, fm = capture(accepted - perturb, row["exposure_ms"], f"epoch{epoch}_minus")
                        minus = evaluate(rm, fm)
                        anchor_rate = anchor["metrics"]["target_rate"]
                        gradient = (
                            (control_score(plus, anchor_rate) - control_score(minus, anchor_rate))
                            / (2.0 * args.delta)
                            * sign
                        )
                        m = 0.9 * m + 0.1 * gradient
                        v = 0.999 * v + 0.001 * gradient**2
                        m_hat = m / (1.0 - 0.9 ** (epoch + 1))
                        v_hat = v / (1.0 - 0.999 ** (epoch + 1))
                        direction = m_hat / (np.sqrt(v_hat) + 1e-8)
                        delta = remove_plane(
                            np.tensordot(args.step * direction, basis, axes=1), sigma=1.0
                        )
                        rms = float(np.sqrt(np.mean(delta**2)))
                        scale = min(
                            1.0,
                            0.025 / max(rms, 1e-12),
                            0.06 / max(float(np.abs(delta).max()), 1e-12),
                        )
                        delta = np.asarray(delta * scale, dtype=np.float32)
                        candidate = np.asarray(accepted + delta, dtype=np.float32)
                        rows = []
                        for label, phase in (
                            ("A1", accepted),
                            ("B1", candidate),
                            ("B2", candidate),
                            ("A2", accepted),
                        ):
                            r, f = capture(phase, row["exposure_ms"], f"epoch{epoch}_{label}")
                            rows.append(evaluate(r, f))
                        verdict_abba = accept_refinement(rows, repeats)
                        take = bool(verdict_abba["accepted"])
                        decisions.append(
                            dict(
                                epoch=epoch,
                                perturb_rms=float(np.sqrt(np.mean(perturb**2))),
                                update_rms=float(np.sqrt(np.mean(delta**2))),
                                gradient_rms=float(np.sqrt(np.mean(gradient**2))),
                                accepted=take,
                                verdict=verdict_abba,
                            )
                        )
                        write_json(args.out / "decisions.json", decisions)
                        if take:
                            accepted = candidate
                            rejected = 0
                            np.save(args.out / "accepted_phase.npy", accepted)
                        else:
                            rejected += 1
                        if rejected >= 2:
                            break
                    write_json(
                        args.out / "summary.json",
                        dict(
                            seed_key=SEED_KEY,
                            baseline_metrics=[r["metrics"] for r in repeats],
                            final_accepted=bool(decisions and decisions[-1]["accepted"]),
                            epochs_run=len(decisions),
                            final_phase_file="accepted_phase.npy",
                        ),
                    )
                finally:
                    slm.display_data(np.zeros((1200, 1920), dtype=np.uint16))
                    time.sleep(0.3)
    finally:
        verdict.lock.release()
    write_json(args.out / "released.json", clock())


if __name__ == "__main__":
    main()
