"""Measure focus and two coma corrections before locating the CCD spot.

Reuses the canonical Fresnel sweep and Zernike generator. All tested phases,
raw frames, exposure readbacks and UTC timestamps are retained for replay.
The returned correction is never left on the panel implicitly.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, median_filter

from ao_shaping.tools.slm.panel.focus_probe import adapt_exposure, sweep_focus


def symmetry_metrics(frame: np.ndarray, half_width: int = 100) -> dict:
    """Exposure invariant mirror errors about a local intensity centroid.

    Restrict the calculation to the brightest lobe; subtract a border median
    without claiming that it is a measured dark frame. Re-centre before the
    mirror comparison so pure image translation is not mistaken for coma.
    """
    image = np.asarray(frame, dtype=float)
    if image.ndim != 2 or not np.isfinite(image).all() or min(image.shape) < 9:
        raise ValueError("need a finite two-dimensional spot image")
    clean = median_filter(image, size=3)
    if float(np.ptp(clean)) <= 0:
        raise ValueError("spot has no detectable energy")
    py, px = np.unravel_index(np.argmax(gaussian_filter(clean, 2)), clean.shape)
    h = int(half_width)
    if h < 4 or min(px, py, image.shape[1] - 1 - px, image.shape[0] - 1 - py) < h:
        raise ValueError("spot window reaches the sensor edge")
    crop = clean[py-h:py+h+1, px-h:px+h+1]
    border = np.concatenate((crop[0], crop[-1], crop[:, 0], crop[:, -1]))
    background = float(np.median(border))
    weights = np.maximum(crop - background, 0)
    weights[weights < 0.02 * weights.max()] = 0
    energy = weights.sum()
    if energy <= 0:
        raise ValueError("spot has no detectable energy")
    yy, xx = np.indices(weights.shape)
    cx, cy = float((xx * weights).sum() / energy), float((yy * weights).sum() / energy)
    # Keep identical support after centring; clipping cannot improve the score.
    radius = int(min(cx, cy, 2*h-cx, 2*h-cy))
    if radius < h // 2:
        raise ValueError("spot centre is not contained in the evaluation window")
    gy, gx = np.mgrid[-radius:radius+1, -radius:radius+1]
    centered = map_coordinates(weights, [gy + cy, gx + cx], order=1, mode="constant")
    total = float(centered.sum())
    lr = float(np.abs(centered - centered[:, ::-1]).sum() / (2 * total))
    ud = float(np.abs(centered - centered[::-1, :]).sum() / (2 * total))
    rms = float(np.sqrt((centered * (gx*gx + gy*gy)).sum() / total))
    return dict(center_x=px-h+cx, center_y=py-h+cy,
                left_right=lr, up_down=ud, asymmetry=(lr+ud)/2,
                rms_radius_px=rms, peak=float(clean.max()), background=background)


def scan_coma(capture, phase_for, *, span_rad=1.5, points=7, passes=2,
              half_width=100, max_radius_growth=1.15, on_measure=None):
    """Coordinate scans with repeated incumbent controls and a width guard.

    Mirror asymmetry alone can favour a diffuse spot. Require its RMS radius
    to remain within the specified ratio of the focused baseline. A bracket
    winner is a measured correction, not a unique estimate of optical coma.
    """
    if span_rad <= 0 or points < 3 or passes < 1 or max_radius_growth < 1:
        raise ValueError("invalid coma scan parameters")
    coefficients = np.zeros(2)
    rows = []

    def measure(c, label):
        frame, exposure = capture(phase_for(c), label)
        row = {**symmetry_metrics(frame, half_width), "coefficients_rad": c.tolist(),
               "exposure_ms": exposure, "label": label,
               "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds")}
        rows.append(row)
        if on_measure is not None:
            on_measure(rows)
        return row

    baseline = measure(coefficients, "focused_baseline")
    radius_limit = baseline["rms_radius_px"] * max_radius_growth
    for turn in range(passes):
        span = span_rad / (2 ** turn)
        for axis in range(2):
            incumbent = measure(coefficients, f"p{turn}_axis{axis}_before")
            candidates = [(incumbent, coefficients.copy())]
            for delta in np.linspace(-span, span, points):
                if abs(delta) < 1e-10:
                    continue
                trial = coefficients.copy()
                trial[axis] += delta
                row = measure(trial, f"p{turn}_axis{axis}_{delta:+.3f}")
                if row["rms_radius_px"] <= radius_limit:
                    candidates.append((row, trial))
            after = measure(coefficients, f"p{turn}_axis{axis}_after")
            candidate, trial = min(candidates, key=lambda item: (
                round(item[0]["asymmetry"], 5), item[0]["rms_radius_px"]))
            # Both bracketing incumbent controls must be worse than the trial.
            if candidate["asymmetry"] < min(incumbent["asymmetry"], after["asymmetry"]):
                validation = measure(trial, f"p{turn}_axis{axis}_verify")
                if (validation["asymmetry"] < min(incumbent["asymmetry"], after["asymmetry"])
                        and validation["rms_radius_px"] <= radius_limit):
                    coefficients = trial
    control = measure(np.zeros(2), "focused_baseline_control")
    final = measure(coefficients, "corrected_final")
    accepted = (final["asymmetry"] < min(baseline["asymmetry"], control["asymmetry"])
                and final["rms_radius_px"] <= radius_limit)
    if not accepted:
        coefficients = np.zeros(2)
        final = measure(coefficients, "focused_fallback")
    return dict(coefficients_rad=coefficients.tolist(), baseline=baseline,
                baseline_control=control, final=final, accepted=accepted, rows=rows)


def run_probe(args) -> dict:
    from ao_shaping.drivers.ccd.common import create_camera, get_camera_exposure_range
    from ao_shaping.drivers.slm.santec import Santec
    from ao_shaping.tools.kernels.bench_kernels import (
        SLM_PANEL_H, SLM_PANEL_W, capture_settled, fresnel_panel, zernike_panel,
    )
    from ao_shaping.utils.image.exposure_control import configured_daheng_exposure_limit

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    shape = (SLM_PANEL_H, SLM_PANEL_W)
    center = (args.center_x, args.center_y)
    pending = np.zeros(shape)
    manifest = []
    phase_index = 0
    current_ms = args.exposure_ms
    with Santec(slm_number=args.slm_number, wavelength=args.wavelength, video_mode=0) as slm:
        try:
            with create_camera(args.cam_type, args.cam_id, exposure_time_ms=current_ms) as cam:
                floor, hardware_max = get_camera_exposure_range(cam)
                ceiling = min(args.max_exposure_ms or hardware_max, hardware_max)
                configured = configured_daheng_exposure_limit() if args.cam_type == "daheng" else 0
                if configured > 0:
                    ceiling = min(ceiling, configured)

                def set_exposure(ms):
                    nonlocal current_ms
                    current_ms = float(cam.reset_exposure_time(ms))
                    return current_ms

                def apply(phase):
                    nonlocal pending, phase_index
                    pending = np.asarray(phase)
                    phase_index += 1
                    np.save(out / f"phase_{phase_index:04d}.npy", pending.astype(np.float32))
                    capture_settled(cam, slm, pending, n_frames=1)

                def read():
                    frame = np.asarray(cam.get_numpy_image(n_sample=1))
                    stamp = datetime.now(timezone.utc).isoformat(timespec="microseconds")
                    name = f"frame_{len(manifest):05d}.npz"
                    np.savez_compressed(out / name, frame=frame)
                    manifest.append(dict(file=name, phase_file=f"phase_{phase_index:04d}.npy",
                                         exposure_ms=float(cam.exposure_time),
                                         timestamp_utc=stamp))
                    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                    return frame

                apply(pending)
                focus = sweep_focus(apply_phase=apply, read_frame=read, set_exposure_ms=set_exposure,
                                    radius_px=args.radius, centre_px=center,
                                    coarse_points=args.focus_points, diopter_range=args.diopter_range,
                                    refine_points=5, frames=2, floor_ms=floor, ceiling_ms=ceiling,
                                    start_exposure_ms=current_ms, wavelength_nm=args.wavelength)
                result = dict(focus=focus.as_evidence(), settings=vars(args))
                (out / "focus.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                if focus.best is None or focus.linearity_ratio is None or not 0.8 <= focus.linearity_ratio <= 1.2:
                    raise RuntimeError("focus exposure linearity could not be validated; inspect focus.json")
                focused = fresnel_panel(focus.best_focal_mm, radius=args.radius,
                                        pupil_center=center, wavelength_nm=args.wavelength)

                def phase_for(c):
                    return focused + zernike_panel({(3, -1): float(c[0]), (3, 1): float(c[1])},
                                                    args.radius, center)

                def capture(phase, label):
                    apply(phase)
                    setting = adapt_exposure(set_exposure, read, start_ms=current_ms,
                                             floor_ms=floor, ceiling_ms=ceiling,
                                             full_scale=float(np.iinfo(read().dtype).max))
                    if setting.status != "ok":
                        raise RuntimeError(f"unusable coma exposure at {label}: {setting.status}")
                    frame = np.mean([read().astype(float) for _ in range(3)], axis=0)
                    return frame, current_ms

                coma = scan_coma(capture, phase_for, span_rad=args.coma_span,
                                 points=args.coma_points, passes=2, half_width=args.half_width,
                                 on_measure=lambda rows: (out / "coma_measurements.json").write_text(
                                     json.dumps(rows, indent=2), encoding="utf-8"))
                np.save(out / "correction_phase.npy", phase_for(coma["coefficients_rad"]))
                result["coma"] = coma
                (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                return result
        finally:
            slm.display_data(slm.create_phase_from_array(np.zeros(shape)), 0.4)


def verify_saved(args) -> dict:
    """Independent fixed-exposure, reversed-order replay of a saved correction."""
    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm.santec import Santec
    from ao_shaping.tools.kernels.bench_kernels import capture_settled, fresnel_panel

    previous = Path(args.verify_directory)
    saved = json.loads((previous / "result.json").read_text(encoding="utf-8"))
    settings = saved["settings"]
    corrected = np.load(previous / "correction_phase.npy", allow_pickle=False)
    focused = fresnel_panel(saved["focus"]["best_focal_mm"], radius=settings["radius"],
                            pupil_center=(settings["center_x"], settings["center_y"]),
                            wavelength_nm=settings["wavelength"])
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    # Reuse the actual, unsaturated exposure chosen in the independent scan.
    exposure = float(saved["coma"]["final"]["exposure_ms"])
    rows = []
    with Santec(slm_number=settings["slm_number"], wavelength=settings["wavelength"], video_mode=0) as slm:
        try:
            with create_camera(settings["cam_type"], settings["cam_id"], exposure_time_ms=exposure) as cam:
                actual = float(cam.reset_exposure_time(exposure))
                if not np.isclose(actual, exposure, rtol=.01):
                    raise RuntimeError("fixed exposure readback differs from requested replay")
                for pair in range(3):
                    order = ("focused", "corrected") if pair % 2 == 0 else ("corrected", "focused")
                    for label in order:
                        phase = focused if label == "focused" else corrected
                        image = capture_settled(cam, slm, phase, n_frames=3)
                        stamp = datetime.now(timezone.utc).isoformat(timespec="microseconds")
                        raw = np.asarray(cam.get_numpy_image(n_sample=1))
                        full_scale = float(np.iinfo(raw.dtype).max)
                        if float(raw.max()) / full_scale >= .9:
                            raise RuntimeError("fixed-exposure replay saturated")
                        row = dict(pair=pair, label=label, exposure_ms=float(cam.exposure_time),
                                   timestamp_utc=stamp, **symmetry_metrics(image, settings["half_width"]))
                        rows.append(row)
                        np.savez_compressed(out / f"{pair}_{label}.npz", frame=image, raw=raw)
                        (out / "measurements.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        finally:
            slm.display_data(slm.create_phase_from_array(np.zeros_like(corrected)), .4)
    summary = {}
    for label in ("focused", "corrected"):
        group = [row for row in rows if row["label"] == label]
        summary[label] = {name: dict(mean=float(np.mean([row[name] for row in group])),
                                    std=float(np.std([row[name] for row in group], ddof=1)))
                          for name in ("center_x", "center_y", "asymmetry", "rms_radius_px")}
    result = dict(source=str(previous.resolve()), exposure_ms=actual, summary=summary, rows=rows)
    (out / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify-directory", help="Replay an existing result at one fixed exposure")
    parser.add_argument("--cam-type", default="daheng")
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--wavelength", type=int, default=1064)
    parser.add_argument("--center-x", type=int)
    parser.add_argument("--center-y", type=int)
    parser.add_argument("--radius", type=int)
    parser.add_argument("--exposure-ms", type=float, default=0.02)
    parser.add_argument("--max-exposure-ms", type=float, default=0)
    parser.add_argument("--focus-points", type=int, default=13)
    parser.add_argument("--diopter-range", type=float, default=1.0)
    parser.add_argument("--coma-span", type=float, default=1.5)
    parser.add_argument("--coma-points", type=int, default=7)
    parser.add_argument("--half-width", type=int, default=100)
    args = parser.parse_args()
    if not args.verify_directory and any(getattr(args, name) is None for name in ("center_x", "center_y", "radius")):
        parser.error("new scans require --center-x, --center-y and --radius")
    result = verify_saved(args) if args.verify_directory else run_probe(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
