"""Replay a saved GS phase with circular smoothing and a zero-order-free CV.

Each candidate is bracketed by the original phase at a fixed exposure. All
frames and UTC timestamps are saved under data/debug for paired analysis.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from ao_shaping.algorithm.signal_processing.square_shaping import uniformity_cv
from ao_shaping.optimizer.wfless._measure import display_and_average
from ao_shaping.tools.slm.panel.display_state import preserve_slm_display
from ao_shaping.utils.image.beam_metrics import (
    prepare_frame,
    zero_order_center,
    zero_order_exclusion_mask,
)
from ao_shaping.utils.image.exposure_control import (
    configured_daheng_exposure_limit,
    probe_exposure_ceiling_ms,
)
from ao_shaping.utils.image.targets import square_mask
from ao_shaping.utils.slm.phase_embed import panel_shape_from_slm
from ao_shaping.utils.slm.phase_uniformity import (
    circular_smooth_phase,
    wrapped_gradient_rms,
)


def pupil_mask(shape: tuple[int, int], center_xy: tuple[int, int], radius: float) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (xx - center_xy[0]) ** 2 + (yy - center_xy[1]) ** 2 <= radius**2


def frame_metrics(
    frame: np.ndarray, target: np.ndarray, zero_core: np.ndarray
) -> dict[str, float]:
    valid = target & ~zero_core
    clean = prepare_frame(frame)
    return {
        "cv_raw": uniformity_cv(frame, target),
        "cv_no_zero": uniformity_cv(frame, valid),
        "cv_no_zero_background_subtracted": uniformity_cv(clean, valid),
        "target_mean": float(np.mean(frame[target])),
        "target_mean_no_zero": float(np.mean(frame[valid])),
        "frame_peak": float(np.max(frame)),
        "frame_median": float(np.median(frame)),
        "zero_core_pixels": int(np.count_nonzero(zero_core)),
        "target_pixels": int(np.count_nonzero(target)),
    }


def summarize_pairs(rows: list[dict]) -> dict:
    """Compare each smoothed frame against its bracketing raw-phase frames."""
    if len(rows) % 3:
        raise ValueError("expected baseline/candidate/baseline triples")
    pairs = []
    for before, candidate, after in zip(rows[::3], rows[1::3], rows[2::3]):
        if (before["label"], candidate["label"], after["label"]) != (
            "baseline_before", "smoothed", "baseline_after"
        ):
            raise ValueError("invalid paired measurement order")
        reference = 0.5 * (before["cv_no_zero"] + after["cv_no_zero"])
        pairs.append({
            "sigma_px": candidate["sigma_px"],
            "cv_no_zero_candidate": candidate["cv_no_zero"],
            "cv_no_zero_reference": reference,
            "delta_cv_no_zero": candidate["cv_no_zero"] - reference,
            "phase_gradient_rms_rad_per_pixel": candidate["phase_gradient_rms_rad_per_pixel"],
        })
    by_sigma = {}
    for sigma in sorted({pair["sigma_px"] for pair in pairs}):
        values = [pair["delta_cv_no_zero"] for pair in pairs if pair["sigma_px"] == sigma]
        by_sigma[str(sigma)] = {
            "n_pairs": len(values),
            "mean_delta_cv_no_zero": float(np.mean(values)),
            "pair_deltas": values,
        }
    return {"pairs": pairs, "by_sigma": by_sigma}


def collect(
    phase_file: Path,
    output_dir: Path,
    *,
    sigmas: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0),
    repeats: int = 2,
    exposure_ms: float = 0.12,
    frames: int = 4,
    zero_radius_px: float | None = None,
) -> list[dict]:
    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm.santec import Santec, SantecError

    if not phase_file.is_file() or phase_file.name != "best_phase.npy":
        raise ValueError("phase_file must be a saved GS best_phase.npy")
    if not sigmas or any(not np.isfinite(s) or s <= 0 for s in sigmas):
        raise ValueError("sigmas must be finite and positive")
    if repeats < 1 or frames < 1 or exposure_ms < 0.02:
        raise ValueError("repeats and frames must be positive; exposure >= 0.02 ms")
    source_config = next(phase_file.parent.glob("slm_gs_refine_*.json"))
    config = json.loads(source_config.read_text(encoding="utf-8"))
    raw = np.load(phase_file, allow_pickle=False)
    radius = float(config["beam_radius_px"])
    center_xy = tuple(int(v) for v in config["pupil_center"])
    target_side = int(config["target_side"])
    pupil = pupil_mask(raw.shape, center_xy, radius)
    candidates = {sigma: circular_smooth_phase(raw, sigma, pupil) for sigma in sigmas}
    gradients = {0.0: wrapped_gradient_rms(raw, pupil)}
    gradients.update({sigma: wrapped_gradient_rms(phase, pupil) for sigma, phase in candidates.items()})
    output_dir.mkdir(parents=True, exist_ok=False)
    for sigma, phase in candidates.items():
        np.save(output_dir / f"phase_sigma_{sigma:g}.npy", phase)
    rows: list[dict] = []
    with Santec(slm_number=1, wavelength=1064, video_mode=0) as slm:
        with create_camera("daheng", 0, exposure_time_ms=0.02) as cam:
            with preserve_slm_display(slm):
                if tuple(panel_shape_from_slm(slm)) != raw.shape:
                    raise ValueError("saved phase does not match the SLM panel")
                try:
                    protected_slot = slm.get_displayed_memory_number()
                except SantecError as exc:
                    if exc.code != 1:
                        raise
                    protected_slot = None
                slots = [s for s in (126, 127, 124) if s != protected_slot][:2]
                slot_index = 0

                def capture(phase: np.ndarray) -> np.ndarray:
                    nonlocal slot_index
                    image = display_and_average(
                        cam, slm, phase, n_frames=frames, settle_wait_s=0.5,
                        settle_tol=0.02, settle_max_wait_s=6.0,
                        settle_max_discard=40,
                        memory_number=slots[slot_index % len(slots)],
                    )
                    slot_index += 1
                    if np.max(image) >= 250:
                        raise RuntimeError("CCD frame near saturation; stop and lower exposure")
                    return image

                flat_phase = np.zeros(raw.shape, dtype=np.float64)
                cam.reset_exposure_time(0.02)
                capture(flat_phase)
                ceiling = configured_daheng_exposure_limit()
                if not ceiling:
                    ceiling = probe_exposure_ceiling_ms(
                        cam.reset_exposure_time,
                        lambda: np.asarray(cam.get_numpy_image(n_sample=1)),
                    ).ceiling_ms
                working_ms = min(exposure_ms, ceiling)
                if working_ms < 0.02:
                    raise RuntimeError("resolved exposure ceiling is below camera minimum")
                actual = float(cam.reset_exposure_time(working_ms))
                if not 0 < actual <= ceiling + 1e-9:
                    raise RuntimeError(f"unsafe exposure readback: {actual} ms")
                full_flat = capture(flat_phase)
                cx_full, cy_full = zero_order_center(full_flat, refine=False, despike_k=3)
                half = 125
                if (cx_full - half < 0 or cy_full - half < 0
                        or cx_full + half >= full_flat.shape[1]
                        or cy_full + half >= full_flat.shape[0]):
                    raise ValueError("zero order too close to CCD edge for 250 px ROI")
                cam.reset_window(center=(cx_full, cy_full), size=(250, 250))
                flat = capture(flat_phase)
                cx, cy = zero_order_center(flat, refine=False, despike_k=3)
                target = square_mask(flat.shape, cy, cx, target_side)
                zero_core = zero_order_exclusion_mask(
                    flat, target, radius_px=zero_radius_px
                )
                np.savez_compressed(output_dir / "flat_reference.npz",
                                    frame=flat, target=target, zero_core=zero_core)
                run_config = {
                    "source_phase": str(phase_file),
                    "source_config": str(source_config),
                    "source_pupil_center_xy": center_xy,
                    "source_pupil_radius_px": radius,
                    "target_side_px": target_side,
                    "ccd_full_zero_center_xy": [cx_full, cy_full],
                    "ccd_roi_zero_center_xy": [cx, cy],
                    "exposure_ms_requested": exposure_ms,
                    "exposure_ms_working": actual,
                    "exposure_ceiling_ms": ceiling,
                    "frames_per_measurement": frames,
                    "repeats": repeats,
                    "sigmas_px": sigmas,
                    "zero_core_pixels": int(zero_core.sum()),
                    "phase_gradient_rms_rad_per_pixel": gradients,
                }
                (output_dir / "config.json").write_text(
                    json.dumps(run_config, indent=2), encoding="utf-8"
                )

                def record(label: str, sigma: float, pair: int) -> None:
                    phase = raw if sigma == 0 else candidates[sigma]
                    frame = capture(phase)
                    readback = float(cam.get_exposure_time_ms())
                    if abs(readback - actual) > 1e-9 or readback > ceiling + 1e-9:
                        raise RuntimeError(f"CCD exposure changed during comparison: {readback} ms")
                    index = len(rows)
                    row = {
                        "index": index,
                        "pair": pair,
                        "label": label,
                        "sigma_px": sigma,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                        "exposure_ms_readback": readback,
                        "phase_gradient_rms_rad_per_pixel": gradients[sigma],
                        **frame_metrics(frame, target, zero_core),
                    }
                    np.savez_compressed(output_dir / f"frame_{index:03d}.npz", frame=frame)
                    rows.append(row)
                    (output_dir / "measurements.json").write_text(
                        json.dumps(rows, indent=2), encoding="utf-8"
                    )
                    print(json.dumps(row), flush=True)

                for repeat in range(repeats):
                    order = sigmas if repeat % 2 == 0 else tuple(reversed(sigmas))
                    for sigma in order:
                        pair = len(rows) // 3
                        record("baseline_before", 0.0, pair)
                        record("smoothed", sigma, pair)
                        record("baseline_after", 0.0, pair)
    (output_dir / "summary.json").write_text(
        json.dumps(summarize_pairs(rows), indent=2), encoding="utf-8"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase_file", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sigma", nargs="+", type=float, default=[2.0, 5.0, 10.0, 20.0])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--exposure-ms", type=float, default=0.12)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--zero-radius-px", type=float)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        rows = json.loads((args.output_dir / "measurements.json").read_text(encoding="utf-8"))
        (args.output_dir / "summary.json").write_text(
            json.dumps(summarize_pairs(rows), indent=2), encoding="utf-8"
        )
        return
    collect(
        args.phase_file, args.output_dir, sigmas=tuple(args.sigma),
        repeats=args.repeats, exposure_ms=args.exposure_ms,
        frames=args.frames, zero_radius_px=args.zero_radius_px,
    )


if __name__ == "__main__":
    main()
