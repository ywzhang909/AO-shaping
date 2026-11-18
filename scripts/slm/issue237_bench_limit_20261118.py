"""Issue #237 acceptance (4)(5) and the crop-window prerequisite, measured offline.

Two independent checks on the archived ``--scan`` capture:

1. **Dynamic range** -- is any exposure simultaneously unsaturated at the zero
   order and usable in the shaped field?  (Acceptance items 4 and 5 need this.)
2. **Crop window** -- does the frozen calibration's camera window even contain
   the zero-order core that the model is compared against?  The zero-order
   position is taken from *unsaturated wings* inside a fixed annulus, so the
   clipped core never enters the centroid.

Pure offline: no SLM/CCD/laser access, no repo imports.

Usage::

    uv run python scripts/slm/issue237_bench_limit_20261118.py \
        --scan-root data/debug/slm_transfer/measured_forward_hw_probe_20261118_scan
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

#: Saturation level of the archived 8-bit capture.
SATURATION_LEVEL = 255
#: Detector-fixed exclusion halo around the saturated union (CCD px); covers the
#: blooming skirt without letting any candidate shift move the mask.
DC_HALO_PX = 90
#: Field signal level the LUT/PSF criteria need before the pedestal stops mattering.
USABLE_FIELD_COUNTS = 20.0
#: Annulus (CCD px around the saturated blob) used for the wing centroid.  The core
#: is clipped in every archived frame, so the position comes from the wings only.
WING_INNER_PX = 26.0
WING_OUTER_PX = 90.0
#: Default source of the frozen calibration's window geometry.
DEFAULT_FIT_SOURCES = Path(
    "data/debug/slm_transfer/measured_forward_hw_probe_20261117_run4/results.json"
)


def load_stack(scan_root: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    points = json.loads((scan_root / "points.json").read_text(encoding="utf-8"))
    frames = sorted((scan_root / "frames").glob("scan_*.npz"))
    if len(frames) != len(points):
        raise ValueError(f"{len(frames)} frame files vs {len(points)} points")
    means = []
    for path in frames:
        raw = np.load(path)["raw_frames"].astype(np.float32)
        means.append(raw.mean(axis=0))
    return np.stack(means), points


def measure(cube: np.ndarray, points: list[dict[str, Any]]) -> dict[str, Any]:
    sat_any = (cube >= SATURATION_LEVEL).any(axis=0)
    dc_mask = ndimage.binary_dilation(sat_any, iterations=DC_HALO_PX)
    field = ~dc_mask
    ys, xs = np.nonzero(sat_any)
    per_point = []
    for i, point in enumerate(points):
        frame = cube[i]
        per_point.append(
            {
                "index": i,
                "phase": point["phase"],
                "exposure_ms": float(point["exposure_ms"]),
                "dc_max": float(frame[dc_mask].max()),
                "dc_saturated_px": int((frame[dc_mask] >= SATURATION_LEVEL).sum()),
                "field_max": float(frame[field].max()),
                "field_p999": float(np.percentile(frame[field], 99.9)),
                "field_mean": float(frame[field].mean()),
            }
        )
    return {
        "n_points": len(points),
        "n_saturated_union_px": int(sat_any.sum()),
        "n_saturated_all_px": int((cube >= SATURATION_LEVEL).all(axis=0).sum()),
        "saturated_bbox_y": [int(ys.min()), int(ys.max())],
        "saturated_bbox_x": [int(xs.min()), int(xs.max())],
        "dc_mask_px": int(dc_mask.sum()),
        "field_px": int(field.sum()),
        "per_point": per_point,
    }


def wing_centroid(
    frame: np.ndarray, seed_yx: tuple[float, float]
) -> tuple[float, float, float]:
    """Centroid of ``frame - ring background`` inside a fixed annulus around ``seed``.

    The saturated core is excluded by construction, so clipping cannot pull the
    estimate towards the blob's flat top.
    """
    yy, xx = np.mgrid[0:frame.shape[0], 0:frame.shape[1]]
    r2 = (yy - seed_yx[0]) ** 2 + (xx - seed_yx[1]) ** 2
    ring = (r2 >= WING_INNER_PX**2) & (r2 <= WING_OUTER_PX**2)
    vals = frame[ring]
    bg = float(np.median(vals))
    weights = np.clip(vals - bg, 0.0, None)
    if weights.sum() <= 0.0:
        return float("nan"), float("nan"), bg
    return (
        float((yy[ring] * weights).sum() / weights.sum()),
        float((xx[ring] * weights).sum() / weights.sum()),
        bg,
    )


def check_crop_window(cube: np.ndarray, fit_sources: Path) -> dict[str, Any]:
    """Is the measured zero order inside the frozen calibration's camera window?"""
    sat_any = (cube >= SATURATION_LEVEL).any(axis=0)
    ys, xs = np.nonzero(sat_any)
    seed = ((ys.min() + ys.max()) / 2.0, (xs.min() + xs.max()) / 2.0)
    rows = []
    for frame in cube:
        y, x, bg = wing_centroid(frame, seed)
        rows.append({"wing_y": y, "wing_x": x, "ring_background": bg})
    finite = [r for r in rows if np.isfinite(r["wing_y"])]
    wing_y = float(np.median([r["wing_y"] for r in finite]))
    wing_x = float(np.median([r["wing_x"] for r in finite]))
    spread_y = float(
        np.percentile([r["wing_y"] for r in finite], 90)
        - np.percentile([r["wing_y"] for r in finite], 10)
    )
    spread_x = float(
        np.percentile([r["wing_x"] for r in finite], 90)
        - np.percentile([r["wing_x"] for r in finite], 10)
    )

    sources = json.loads(fit_sources.read_text(encoding="utf-8"))["sources"]
    center_x, center_y = (float(v) for v in sources["camera_crop_center"])
    half = int(sources["crop_half"])
    dy, dx = wing_y - center_y, wing_x - center_x
    inside = abs(dx) <= half and abs(dy) <= half
    return {
        "blob_bbox_centre_yx": [seed[0], seed[1]],
        "wing_centroid_yx": [wing_y, wing_x],
        "wing_centroid_spread_px_p10_p90": [spread_y, spread_x],
        "frozen_crop_center_xy": [center_x, center_y],
        "frozen_crop_half_px": half,
        "frozen_window_x": [center_x - half, center_x + half],
        "frozen_window_y": [center_y - half, center_y + half],
        "measured_minus_frozen_dy_dx": [dy, dx],
        "measured_minus_frozen_norm_px": float(np.hypot(dy, dx)),
        "measured_zero_order_inside_window": bool(inside),
        "beyond_half_width_px": [abs(dy) - half, abs(dx) - half],
        "per_frame": rows,
    }


def verdict(measured: dict[str, Any]) -> dict[str, Any]:
    per_phase: dict[str, Any] = {}
    for phase in sorted({p["phase"] for p in measured["per_point"]}):
        rows = [p for p in measured["per_point"] if p["phase"] == phase]
        brightest = max(rows, key=lambda p: p["field_max"])
        first_dc_saturation = min(
            (p["exposure_ms"] for p in rows if p["dc_saturated_px"] > 0), default=None
        )
        need_ms = (
            brightest["exposure_ms"] * USABLE_FIELD_COUNTS / brightest["field_max"]
            if brightest["field_max"] > 0 else None
        )
        per_phase[phase] = {
            "field_max": brightest["field_max"],
            "field_max_exposure_ms": brightest["exposure_ms"],
            "field_mean_at_brightest": brightest["field_mean"],
            "dc_first_saturated_at_ms": first_dc_saturation,
            "exposure_for_usable_field_ms": need_ms,
            "dc_overexposure_factor_at_usable_field": (
                need_ms / first_dc_saturation
                if need_ms is not None and first_dc_saturation else None
            ),
            "dc_saturated_px_growth": [rows[0]["dc_saturated_px"], rows[-1]["dc_saturated_px"]],
        }
    any_clean = any(
        p["dc_saturated_px"] == 0 and p["field_max"] >= USABLE_FIELD_COUNTS
        for p in measured["per_point"]
    )
    return {
        "usable_field_counts": USABLE_FIELD_COUNTS,
        "per_phase": per_phase,
        "any_simultaneously_usable_exposure": bool(any_clean),
        "criterion_4_establishable": bool(any_clean),
        "criterion_5_establishable": bool(any_clean),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan-root", type=Path,
        default=Path("data/debug/slm_transfer/measured_forward_hw_probe_20261118_scan"),
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--fit-sources", type=Path, default=DEFAULT_FIT_SOURCES,
        help="results.json whose 'sources' block holds camera_crop_center/crop_half",
    )
    args = parser.parse_args(argv)

    cube, points = load_stack(args.scan_root)
    measured = measure(cube, points)
    payload = {"scan_root": str(args.scan_root), "measured": measured, "verdict": verdict(measured)}
    if args.fit_sources.exists():
        payload["crop_window"] = check_crop_window(cube, args.fit_sources)
    else:
        payload["crop_window"] = {"skipped": f"missing {args.fit_sources}"}

    out = args.out or Path("data/debug/slm_transfer/issue237_bench_limit_20261118/results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"points={measured['n_points']} sat_union_px={measured['n_saturated_union_px']}")
    print(f"dc_mask_px={measured['dc_mask_px']} field_px={measured['field_px']}")
    for phase, row in payload["verdict"]["per_phase"].items():
        print(
            f"{phase}: field_max={row['field_max']:.1f} @ {row['field_max_exposure_ms']:.0f} ms, "
            f"dc_sat_from={row['dc_first_saturated_at_ms']} ms, "
            f"usable needs {row['exposure_for_usable_field_ms']:.1f} ms "
            f"(DC x{row['dc_overexposure_factor_at_usable_field']:.1f}), "
            f"dc_sat_px {row['dc_saturated_px_growth']}"
        )
    print(f"simultaneously usable exposure: {payload['verdict']['any_simultaneously_usable_exposure']}")
    crop = payload["crop_window"]
    if "measured_zero_order_inside_window" in crop:
        print()
        print(f"zero order from wings (y,x) = ({crop['wing_centroid_yx'][0]:.2f}, "
              f"{crop['wing_centroid_yx'][1]:.2f}) "
              f"p10-p90 spread {np.round(crop['wing_centroid_spread_px_p10_p90'], 2)}")
        print(f"frozen window x={crop['frozen_window_x']} y={crop['frozen_window_y']}")
        print(f"measured - frozen = dy {crop['measured_minus_frozen_dy_dx'][0]:+.2f}, "
              f"dx {crop['measured_minus_frozen_dy_dx'][1]:+.2f} "
              f"(|d| = {crop['measured_minus_frozen_norm_px']:.2f} px)")
        print(f"zero order inside window: {crop['measured_zero_order_inside_window']} "
              f"(beyond half-width by {np.round(crop['beyond_half_width_px'], 1)} px)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
