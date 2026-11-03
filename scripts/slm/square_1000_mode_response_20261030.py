"""Measure local low-frequency phase responses of the 1000 px square seed.

The seed is ``F_blur4_full`` from the 2026-10-30 band-limit session.  This
probe does not run a high-dimensional optimizer.  It applies one unit-RMS DCT
mode at a time in an ``A, P, M, A, M, P, A`` sequence, so the local
phase-to-image response can be separated from slow measurement drift.
"""
from __future__ import annotations

import argparse
import contextlib
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

# ``(v, u)`` in the same convention as ``dct_basis`` in the refine script:
# mode[y, x] = cos(pi * v * y / N) * cos(pi * u * x / N).
DEFAULT_MODES = (
    (0, 2),
    (2, 0),
    (1, 1),
    (0, 3),
    (3, 0),
    (2, 1),
    (1, 2),
    (0, 4),
    (4, 0),
    (2, 2),
)
ROLE_SEQUENCE = ("A1", "P1", "M1", "A2", "M2", "P2", "A3")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_protocols():
    protocol = load_module(
        "square1000_protocol",
        ROOT / "scripts/slm/square_1000_diff_abba_20261030.py",
    )
    metrics_module = load_module(
        "square1000_metrics",
        ROOT / "scripts/slm/square_1000_bandlimit_refine_20261030.py",
    )
    return protocol, metrics_module


def dct_mode(size: int, v: int, u: int) -> np.ndarray:
    """Return one unit-RMS 2-D DCT mode, excluding the piston mode."""
    if v == 0 and u == 0:
        raise ValueError("piston cannot be used as a control mode")
    x = np.arange(size, dtype=np.float64) + 0.5
    mode = np.cos(np.pi * u * x[:, None] / size) * np.cos(
        np.pi * v * x[None, :] / size
    )
    return mode / np.sqrt(np.mean(mode**2))


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


def make_mode_delta(seed: np.ndarray, mode: tuple[int, int], amplitude_rad: float):
    """Build a plane-removed DCT perturbation with an exact requested RMS."""
    delta = remove_plane(amplitude_rad * dct_mode(len(seed), *mode), sigma=1.0)
    rms = float(np.sqrt(np.mean(delta**2)))
    if not np.isfinite(rms) or rms <= 1e-12:
        raise ValueError(f"unresolved mode delta for {mode}")
    delta *= amplitude_rad / rms
    return np.asarray(delta, dtype=np.float32)


def parse_modes(value: str, size: int) -> tuple[tuple[int, int], ...]:
    """Parse ``v:u,v:u`` while rejecting piston and out-of-range modes."""
    modes = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("mode must be written as v:u")
        v, u = (int(part) for part in parts)
        if v == 0 and u == 0:
            raise argparse.ArgumentTypeError("piston (0:0) cannot be controlled")
        if not (0 <= v < size and 0 <= u < size):
            raise argparse.ArgumentTypeError(f"mode {(v, u)} outside {size}x{size}")
        modes.append((v, u))
    if not modes:
        raise argparse.ArgumentTypeError("at least one mode is required")
    if len(set(modes)) != len(modes):
        raise argparse.ArgumentTypeError("duplicate modes are not allowed")
    return tuple(modes)


def role_phase(seed: np.ndarray, delta: np.ndarray, role: str) -> np.ndarray:
    if role.startswith("A"):
        return seed
    if role.startswith("P"):
        return np.asarray(seed + delta, dtype=np.float32)
    if role.startswith("M"):
        return np.asarray(seed - delta, dtype=np.float32)
    raise ValueError(f"unknown role {role}")


def role_sign(role: str) -> int:
    if role.startswith("A"):
        return 0
    if role.startswith("P"):
        return 1
    if role.startswith("M"):
        return -1
    raise ValueError(f"unknown role {role}")


def scalar_score(metrics_module, row: dict, reference_rate: float) -> float:
    return float(metrics_module.control_score(row, reference_rate))


def metric_keys() -> tuple[str, ...]:
    return (
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


def favorable_sign(key: str) -> float:
    """Return +1 when larger is better and -1 when smaller is better."""
    return 1.0 if key in {"coverage", "iou", "target_rate", "local_ee"} else -1.0


def paired_response(rows: list[dict], reference_rate: float, metrics_module, delta_rad: float):
    """Estimate one mode response from adjacent P/M contrasts and A drift."""
    by_role = {row["role"]: row for row in rows}
    anchors = [by_role["A1"], by_role["A2"], by_role["A3"]]
    anchor_scores = np.asarray(
        [scalar_score(metrics_module, row, reference_rate) for row in anchors],
        dtype=np.float64,
    )
    # Three anchors give one residual degree of freedom.  Use the RMS deviation
    # from a straight line as a conservative local-drift scale.
    anchor_times = np.asarray([row["monotonic_ns"] for row in anchors], dtype=np.float64)
    anchor_times = (anchor_times - anchor_times[0]) / max(anchor_times[-1] - anchor_times[0], 1.0)
    design = np.stack([np.ones_like(anchor_times), anchor_times], axis=-1)
    anchor_fit = design @ np.linalg.lstsq(design, anchor_scores, rcond=None)[0]
    anchor_rms = float(np.sqrt(np.mean((anchor_scores - anchor_fit) ** 2)))

    p1 = scalar_score(metrics_module, by_role["P1"], reference_rate)
    p2 = scalar_score(metrics_module, by_role["P2"], reference_rate)
    m1 = scalar_score(metrics_module, by_role["M1"], reference_rate)
    m2 = scalar_score(metrics_module, by_role["M2"], reference_rate)
    contrast_1 = p1 - m1
    contrast_2 = p2 - m2
    mean_contrast = 0.5 * (contrast_1 + contrast_2)
    pair_se = 0.5 * abs(contrast_2 - contrast_1)
    # ``control_score`` is dimensionless.  The floor prevents an exact
    # repeat from being labelled detectable on numerical noise alone.
    detected = bool(
        np.isfinite([contrast_1, contrast_2, anchor_rms]).all()
        and contrast_1 * contrast_2 > 0
        and abs(mean_contrast) > max(0.01, 3.0 * pair_se, 3.0 * np.sqrt(2.0) * anchor_rms)
    )
    score_response = float(mean_contrast / (2.0 * delta_rad))

    metric_response = {}
    for key in metric_keys():
        values = [
            favorable_sign(key) * (by_role[r]["metrics"][key] - by_role[m]["metrics"][key])
            for r, m in (("P1", "M1"), ("P2", "M2"))
        ]
        metric_response[key] = {
            "paired_favorable_delta": [float(values[0]), float(values[1])],
            "mean_favorable_delta": float(np.mean(values)),
            "per_rad": float(np.mean(values) / (2.0 * delta_rad)),
            "same_sign": bool(values[0] * values[1] > 0),
        }
    return {
        "score_response_per_rad": score_response,
        "score_contrasts": [float(contrast_1), float(contrast_2)],
        "pair_se": float(pair_se),
        "anchor_score_rms": anchor_rms,
        "detected": detected,
        "metrics": metric_response,
    }


def build_suggested_update(
    seed: np.ndarray,
    modes: tuple[tuple[int, int], ...],
    deltas: dict[tuple[int, int], np.ndarray],
    responses: dict[str, dict],
    max_modes: int,
    rms_cap: float,
):
    """Build a small measured-response proposal for a later ABBA check."""
    candidates = [
        (label, info["response"]["score_response_per_rad"])
        for label, info in responses.items()
        if info["response"]["detected"] and info["response"]["score_response_per_rad"] > 0
    ]
    candidates.sort(key=lambda item: item[1], reverse=True)
    selected = candidates[:max_modes]
    if not selected:
        return None, {"selected_modes": [], "rms_rad": 0.0}
    direction = np.zeros_like(seed, dtype=np.float64)
    for label, _ in selected:
        v, u = next(mode for mode in modes if f"v{mode[0]}_u{mode[1]}" == label)
        direction += deltas[(v, u)].astype(np.float64)
    direction = remove_plane(direction, sigma=1.0)
    rms = float(np.sqrt(np.mean(direction**2)))
    if rms <= 1e-12:
        return None, {"selected_modes": [label for label, _ in selected], "rms_rad": 0.0}
    direction *= min(rms_cap, rms) / rms
    proposal = np.asarray(seed + direction, dtype=np.float32)
    return proposal, {
        "selected_modes": [label for label, _ in selected],
        "rms_rad": float(np.sqrt(np.mean(direction**2))),
        "max_abs_rad": float(np.abs(direction).max()),
        "score_response_per_rad": dict(selected),
    }


def capture_frames(
    cam, slm, phase, exposure, args, records, masks, output_dir, camera_xy, **metadata
):
    """Apply one phase, settle, discard flush frames, and archive native frames."""
    gray = slm.create_phase_from_array(to_panel(phase))
    slm.display_data(gray)
    cam.reset_exposure_time(exposure)
    time.sleep(args.settle_s)
    for _ in range(args.flush_frames):
        cam.get_numpy_image(n_sample=1, skip_first=False)
    frames = []
    stamps = []
    for _ in range(args.frames):
        stamp = clock()
        frame = np.asarray(cam.get_numpy_image(n_sample=1, skip_first=False)).copy()
        stamp.update(end_monotonic_ns=time.monotonic_ns())
        frames.append(frame)
        stamps.append(stamp)
    raw = np.stack(frames)
    if raw.shape != (args.frames, 1944, 2592) or raw.dtype != np.uint16:
        raise RuntimeError("expected native unpacked Mono12 uint16 frames")
    exposure_readback = float(args.exposure_reader(cam))
    row = dict(
        label=metadata.pop("label"),
        role=metadata.pop("role"),
        mode=metadata.pop("mode"),
        signed_amplitude_rad=float(metadata.pop("signed_amplitude_rad")),
        exposure_ms=exposure_readback,
        file=f"capture_{len(records):04d}.npz",
        frames=stamps,
        **metadata,
        **clock(),
    )
    np.savez_compressed(
        output_dir / row["file"],
        raw_frames=raw,
        model_phase=phase,
        panel_gray=gray,
        camera_xy=camera_xy,
    )
    records.append(row)
    write_json(output_dir / "manifest.json", records)
    target_peak = float(raw[:, masks["box"]].max())
    if target_peak >= 0.98 * FULL_SCALE:
        raise RuntimeError(f"target saturated in {row['label']} (peak={target_peak})")
    return row, raw


def execute(args, seed: np.ndarray, xy: np.ndarray, modes, deltas, protocol, metrics_module):
    from ao_shaping.tools.kernels.device_mutex import DeviceLock, wait_until_idle

    verdict = wait_until_idle(
        args.wait_until_idle,
        lock=DeviceLock.default(),
        own_paths=[Path(__file__).resolve()],
    )
    if not verdict.ok or verdict.lock is None:
        raise RuntimeError(verdict.headline())
    lock = verdict.lock
    records: list[dict] = []
    device_slm = None
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
        objective = ShapeObjective(
            protocol.masks(xy)["box"].astype(np.float32), formation=True
        )
        args.exposure_reader = get_camera_exposure_ms

        with create_camera("daheng", 0, exposure_time_ms=0.5, bit_depth=12) as cam:
            with Santec(slm_number=1, wavelength=1064, video_mode=0) as slm:
                device_slm = slm
                slm.set_phase_overlay(None)
                try:
                    def capture(phase, exposure, label, role, mode, amplitude, **extra):
                        return capture_frames(
                            cam,
                            slm,
                            phase,
                            exposure,
                            args,
                            records,
                            native_masks,
                            args.out,
                            xy,
                            label=label,
                            role=role,
                            mode=mode,
                            signed_amplitude_rad=amplitude,
                            **extra,
                        )

                    def evaluate(row, raw, background, threshold, reference_rate):
                        row["metrics"] = metrics_module.metrics(
                            raw,
                            row["exposure_ms"],
                            background,
                            threshold,
                            native_masks,
                            objective,
                            xy,
                        )
                        row["score"] = scalar_score(metrics_module, row, reference_rate)
                        write_json(args.out / "manifest.json", records)
                        print(
                            json.dumps(
                                dict(
                                    label=row["label"],
                                    mode=row["mode"],
                                    role=row["role"],
                                    exposure_ms=row["exposure_ms"],
                                    score=row["score"],
                                    coverage=row["metrics"]["coverage"],
                                    iou=row["metrics"]["iou"],
                                    cv=row["metrics"]["cv"],
                                    loss=row["metrics"]["loss"],
                                    edge_position_error=row["metrics"][
                                        "edge_position_error"
                                    ],
                                    target_rate=row["metrics"]["target_rate"],
                                )
                            ),
                            flush=True,
                        )
                        return row

                    row, raw = capture(
                        seed, 0.5, "seed_safe_meter", "calibration", "seed", 0.0
                    )
                    peak = float(raw[:, native_masks["box"]].max())
                    if peak < 20:
                        raise RuntimeError("unresolved target signal at safe exposure")
                    exposure = min(
                        args.max_exposure_ms,
                        row["exposure_ms"] * args.target_peak / peak,
                    )
                    row, raw = capture(
                        seed, exposure, "seed_calibration", "calibration", "seed", 0.0
                    )
                    background = float(
                        np.median(raw.mean(0)[native_masks["background"]])
                    )
                    noise = float(
                        np.sqrt(
                            raw[:, native_masks["background"]].var(0, ddof=1).mean()
                            / len(raw)
                        )
                    )
                    core_mean = float(
                        np.maximum(raw.mean(0) - background, 0.0)[
                            native_masks["core"]
                        ].mean()
                    )
                    threshold = max(3.0 * noise, 0.2 * core_mean)
                    if core_mean <= noise:
                        raise RuntimeError("seed signal unresolved after exposure selection")
                    write_json(
                        args.out / "measurement_contract.json",
                        dict(
                            background_counts=background,
                            coverage_threshold_counts=threshold,
                            exposure_ms=float(row["exposure_ms"]),
                            noise_counts=noise,
                            core_mean_counts=core_mean,
                            full_scale=FULL_SCALE,
                            smoothing_sigma_native_px=8,
                            target_peak_counts=args.target_peak,
                            settle_s=args.settle_s,
                            flush_frames=args.flush_frames,
                            frames_per_capture=args.frames,
                        ),
                    )
                    reference_rate = float(
                        np.maximum(raw.mean(0) - background, 0.0)[
                            native_masks["box"]
                        ].sum()
                        / row["exposure_ms"]
                    )

                    seed_rows = []
                    for index in range(5):
                        r, f = capture(
                            seed,
                            row["exposure_ms"],
                            f"seed_repeat_{index}",
                            "A_ref",
                            "seed",
                            0.0,
                        )
                        seed_rows.append(
                            evaluate(r, f, background, threshold, reference_rate)
                        )

                    responses = {}
                    for mode_index, (v, u) in enumerate(modes):
                        label = f"v{v}_u{u}"
                        delta = deltas[(v, u)]
                        rows = []
                        for sequence_index, role in enumerate(ROLE_SEQUENCE):
                            phase = role_phase(seed, delta, role)
                            r, f = capture(
                                phase,
                                row["exposure_ms"],
                                f"mode{mode_index:02d}_{label}_{sequence_index}_{role}",
                                role,
                                label,
                                role_sign(role) * args.delta,
                                mode_v=v,
                                mode_u=u,
                                mode_index=mode_index,
                                sequence_index=sequence_index,
                                delta_rms_rad=args.delta,
                            )
                            rows.append(
                                evaluate(r, f, background, threshold, reference_rate)
                            )
                        response = paired_response(
                            rows, reference_rate, metrics_module, args.delta
                        )
                        responses[label] = dict(
                            mode=[v, u],
                            delta_rms_rad=args.delta,
                            response=response,
                            rows=[
                                dict(
                                    role=row["role"],
                                    score=row["score"],
                                    metrics=row["metrics"],
                                )
                                for row in rows
                            ],
                        )
                        write_json(args.out / "responses.json", responses)
                        print(
                            json.dumps(
                                dict(
                                    mode=label,
                                    score_response_per_rad=response[
                                        "score_response_per_rad"
                                    ],
                                    detected=response["detected"],
                                    pair_se=response["pair_se"],
                                    anchor_score_rms=response["anchor_score_rms"],
                                )
                            ),
                            flush=True,
                        )

                    proposal, proposal_info = build_suggested_update(
                        seed,
                        modes,
                        deltas,
                        responses,
                        args.proposal_modes,
                        args.proposal_rms,
                    )
                    if proposal is not None:
                        np.save(args.out / "suggested_update.npy", proposal)
                    summary = dict(
                        seed_key=SEED_KEY,
                        seed_run=str(SEED_RUN),
                        seed_sha256=digest(SEED_RUN / "plan.npz"),
                        modes=[f"v{v}_u{u}" for v, u in modes],
                        delta_rad=args.delta,
                        detected_modes=[
                            label
                            for label, info in responses.items()
                            if info["response"]["detected"]
                        ],
                        proposal=proposal_info,
                        decision=(
                            "local response measurable; prepare a measured-response ABBA test"
                            if any(
                                info["response"]["detected"]
                                for info in responses.values()
                            )
                            else "no local response resolved; stop model/black-box gradient search"
                        ),
                    )
                    write_json(args.out / "summary.json", summary)
                finally:
                    with contextlib.suppress(Exception):
                        slm.display_data(np.zeros((1200, 1920), dtype=np.uint16))
                        time.sleep(0.3)
    except Exception as exc:
        write_json(
            args.out / "failure.json",
            dict(error=repr(exc), stage="execute", **clock()),
        )
        raise
    finally:
        if device_slm is not None:
            with contextlib.suppress(Exception):
                device_slm.display_data(np.zeros((1200, 1920), dtype=np.uint16))
        lock.release()
        write_json(args.out / "released.json", clock())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--wait-until-idle", type=float, default=0.0)
    parser.add_argument("--modes", default=",".join(f"{v}:{u}" for v, u in DEFAULT_MODES))
    parser.add_argument("--delta", type=float, default=0.03)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--flush-frames", type=int, default=3)
    parser.add_argument("--settle-s", type=float, default=0.8)
    parser.add_argument("--target-peak", type=float, default=1800.0)
    parser.add_argument("--max-exposure-ms", type=float, default=16.87)
    parser.add_argument("--proposal-modes", type=int, default=6)
    parser.add_argument("--proposal-rms", type=float, default=0.02)
    args = parser.parse_args()
    if not 0.005 <= args.delta <= 0.05:
        parser.error("delta must be in [0.005, 0.05] rad")
    if not 3 <= args.frames <= 9:
        parser.error("frames must be 3..9")
    if not 0 <= args.flush_frames <= 8:
        parser.error("flush_frames must be 0..8")
    if not 0.1 <= args.settle_s <= 5.0:
        parser.error("settle_s must be in [0.1, 5.0] s")
    if not 300.0 <= args.target_peak <= 3000.0:
        parser.error("target_peak must be 300..3000 counts")
    if not 0.0 <= args.wait_until_idle:
        parser.error("wait_until_idle must be nonnegative")
    if not 1 <= args.proposal_modes <= len(DEFAULT_MODES):
        parser.error("proposal_modes outside supported range")
    if not 0.005 <= args.proposal_rms <= 0.05:
        parser.error("proposal_rms must be in [0.005, 0.05] rad")

    torch.set_num_threads(4)
    protocol, metrics_module = load_protocols()
    with np.load(SEED_RUN / "plan.npz") as saved:
        seed = np.asarray(saved[SEED_KEY], dtype=np.float32)
        xy = saved["camera_xy"].copy()
    modes = parse_modes(args.modes, len(seed))
    deltas = {mode: make_mode_delta(seed, mode, args.delta) for mode in modes}
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.out / "plan.npz",
        seed=seed,
        camera_xy=xy,
        modes=np.asarray(modes, dtype=np.int16),
        **{f"delta_v{v}_u{u}": delta for (v, u), delta in deltas.items()},
    )
    write_json(
        args.out / "config.json",
        dict(
            execute=args.execute,
            seed_run=str(SEED_RUN),
            seed_key=SEED_KEY,
            seed_sha256=digest(SEED_RUN / "plan.npz"),
            modes=[f"v{v}_u{u}" for v, u in modes],
            role_sequence=list(ROLE_SEQUENCE),
            delta_rad=args.delta,
            frames_per_capture=args.frames,
            flush_frames=args.flush_frames,
            settle_s=args.settle_s,
            target_peak_counts=args.target_peak,
            max_exposure_ms=args.max_exposure_ms,
            basis="unit-RMS 2-D DCT, piston/global tilt removed, Gaussian sigma=1 control cell",
            detection="both P/M contrasts same sign and |mean| > max(0.01, 3*paired SE, 3*sqrt(2)*anchor drift RMS)",
            proposal_modes=args.proposal_modes,
            proposal_rms_rad=args.proposal_rms,
            **clock(),
        ),
    )
    if not args.execute:
        print(
            json.dumps(
                dict(
                    out=str(args.out),
                    modes=[f"v{v}_u{u}" for v, u in modes],
                    delta_rad=args.delta,
                    delta_rms={f"v{v}_u{u}": float(np.sqrt(np.mean(delta**2))) for (v, u), delta in deltas.items()},
                    max_abs={
                        f"v{v}_u{u}": float(np.abs(delta).max())
                        for (v, u), delta in deltas.items()
                    },
                )
            ),
            flush=True,
        )
        return
    execute(args, seed, xy, modes, deltas, protocol, metrics_module)


if __name__ == "__main__":
    main()
