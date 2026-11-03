"""Diagnostic: why does the sim2f full-frame 0-order argmax move between opens?

This is a *measurement* script, not acceptance evidence.  It opens the real
Santec SLM + Daheng CCD bench, captures 30 native full frames with a flat
(all-zero) phase, and compares six position estimators on the same frames:

* ``argmax_raw``            -- the plain ``argmax`` used by
  :meth:`HwFullFrameMixin._locate_zero_order`;
* ``argmax_despiked``       -- ``argmax`` after a 3x3 median filter;
* ``argmax_blurred``        -- ``argmax`` after a Gaussian blur (sigma 1.5);
* ``centroid_box``          -- intensity-weighted centroid over a +/-48 px box
  around ``argmax_raw``, box median subtracted, negatives clipped;
* ``centroid_box_halfmax``  -- the same, restricted to pixels >= 50 % of the
  (background-subtracted) box peak;
* ``centroid_despiked_box`` -- ``centroid_box`` evaluated on the 3x3-median-
  filtered frame.

It also reopens the bench 3 times to measure the spread of the open-time
argmax, and records per-frame diagnostics (peak counts, peak-over-8-neighbour
ratio, ceiling proximity, ...) that explain *why* plain argmax is unstable.

Outputs (a new ``data/debug/sim2f_zero_order_diag_<timestamp>/`` directory):
``diag.json``, ``frames/`` (one representative full frame + a 256x256 crop
per captured frame), ``spot.png``.

Run with the exposure ceiling set, e.g.::

    AO_DAHENG_MAX_EXPOSURE_MS=18.8846 \\
    PYTHONPATH=<repo>/src;<repo>/libs python scripts/ml/sim2f_fov_zero_order_diag.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.sim2f.bench import HwInstrument

N_FRAMES = 30
BOX_HALF = 48
CROP = 256
SPOT_HALF = 64


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _median_filter_3x3(frame: np.ndarray) -> np.ndarray:
    """3x3 median filter, returned as float64."""
    return ndimage.median_filter(np.asarray(frame, dtype=np.float64), size=3)


def argmax_raw(frame: np.ndarray) -> tuple[float, float]:
    """Plain global argmax, returned as ``(x_px, y_px)`` = (column, row)."""
    arr = np.asarray(frame)
    row, col = np.unravel_index(int(np.argmax(arr)), arr.shape)
    return float(col), float(row)


def argmax_despiked(frame: np.ndarray) -> tuple[float, float]:
    """Argmax after a 3x3 median filter (hot-pixel suppression)."""
    filtered = _median_filter_3x3(frame)
    row, col = np.unravel_index(int(np.argmax(filtered)), filtered.shape)
    return float(col), float(row)


def argmax_blurred(frame: np.ndarray) -> tuple[float, float]:
    """Argmax after a Gaussian blur (sigma = 1.5)."""
    filtered = ndimage.gaussian_filter(np.asarray(frame, dtype=np.float64), sigma=1.5)
    row, col = np.unravel_index(int(np.argmax(filtered)), filtered.shape)
    return float(col), float(row)


def _box_centroid(
    frame: np.ndarray,
    center_xy: tuple[float, float] | None = None,
    half: int = BOX_HALF,
    halfmax: bool = False,
) -> tuple[float, float]:
    """Intensity-weighted centroid over a ``+/-half`` box.

    The box is centred on ``center_xy`` (defaults to ``argmax_raw(frame)``).
    The box median is subtracted and negatives are clipped.  When ``halfmax``
    is True only pixels with background-subtracted value >= 50 % of the box
    peak contribute.
    """
    if center_xy is None:
        cx0, cy0 = argmax_raw(frame)
    else:
        cx0, cy0 = center_xy
    cx0 = int(round(cx0))
    cy0 = int(round(cy0))
    arr = np.asarray(frame, dtype=np.float64)
    height, width = arr.shape
    r0 = max(0, cy0 - half)
    r1 = min(height, cy0 + half + 1)
    c0 = max(0, cx0 - half)
    c1 = min(width, cx0 + half + 1)
    box = arr[r0:r1, c0:c1]
    median = float(np.median(box))
    body = box - median
    body[body < 0.0] = 0.0
    if halfmax:
        peak = float(body.max()) if body.size else 0.0
        if peak > 0.0:
            body = np.where(body >= 0.5 * peak, body, 0.0)
    total = float(body.sum())
    if total <= 0.0:
        return float(cx0), float(cy0)
    rr, cc = np.indices(body.shape)
    x = c0 + float((cc * body).sum()) / total
    y = r0 + float((rr * body).sum()) / total
    return x, y


def centroid_box(frame: np.ndarray) -> tuple[float, float]:
    """Classical background-subtracted centroid over a +/-48 px box."""
    return _box_centroid(frame, halfmax=False)


def centroid_box_halfmax(frame: np.ndarray) -> tuple[float, float]:
    """Background-subtracted centroid restricted to pixels >= 50 % of box peak."""
    return _box_centroid(frame, halfmax=True)


def centroid_despiked_box(frame: np.ndarray) -> tuple[float, float]:
    """``centroid_box`` evaluated on the 3x3-median-filtered frame."""
    return centroid_box(_median_filter_3x3(frame))


ESTIMATORS = (
    "argmax_raw",
    "argmax_despiked",
    "argmax_blurred",
    "centroid_box",
    "centroid_box_halfmax",
    "centroid_despiked_box",
)


def _estimator_fn(name: str):
    return {
        "argmax_raw": argmax_raw,
        "argmax_despiked": argmax_despiked,
        "argmax_blurred": argmax_blurred,
        "centroid_box": centroid_box,
        "centroid_box_halfmax": centroid_box_halfmax,
        "centroid_despiked_box": centroid_despiked_box,
    }[name]


def _summarize(points: list[tuple[float, float]]) -> dict[str, Any]:
    xs = np.asarray([p[0] for p in points], dtype=np.float64)
    ys = np.asarray([p[1] for p in points], dtype=np.float64)
    mx = float(xs.mean())
    my = float(ys.mean())
    rads = np.sqrt((xs - mx) ** 2 + (ys - my) ** 2)
    return {
        "n": int(len(points)),
        "mean_x": mx,
        "std_x": float(xs.std(ddof=1)) if len(points) > 1 else 0.0,
        "min_x": float(xs.min()),
        "max_x": float(xs.max()),
        "ptp_x": float(xs.max() - xs.min()),
        "mean_y": my,
        "std_y": float(ys.std(ddof=1)) if len(points) > 1 else 0.0,
        "min_y": float(ys.min()),
        "max_y": float(ys.max()),
        "ptp_y": float(ys.max() - ys.min()),
        "radial_mean_px": float(rads.mean()),
        "radial_std_px": float(rads.std(ddof=1)) if len(points) > 1 else 0.0,
        "radial_min_px": float(rads.min()),
        "radial_max_px": float(rads.max()),
        "radial_ptp_px": float(rads.max() - rads.min()),
        "radial_spread_px": float(rads.max()),
    }


def _peak_over_ring(frame: np.ndarray, peak_rc: tuple[int, int]) -> float | None:
    r, c = peak_rc
    height, width = frame.shape
    vals: list[float] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < height and 0 <= cc < width:
                vals.append(float(frame[rr, cc]))
    if not vals:
        return None
    med = float(np.median(vals))
    if med <= 0.0:
        return None
    peak = float(frame[r, c])
    return peak / med


def _crop256(frame: np.ndarray, center_xy: tuple[int, int]) -> np.ndarray:
    cx, cy = center_xy
    height, width = frame.shape
    half = CROP // 2
    r0 = cy - half
    r1 = cy + half
    c0 = cx - half
    c1 = cx + half
    out = np.zeros((CROP, CROP), dtype=np.float32)
    rs = max(0, r0)
    re = min(height, r1)
    cs = max(0, c0)
    ce = min(width, c1)
    if re > rs and ce > cs:
        out[rs - r0 : re - r0, cs - c0 : ce - c0] = frame[rs:re, cs:ce]
    return out


def _open_instrument() -> HwInstrument:
    return HwInstrument(
        exposure_ms=0.182,
        ccd_px=256,
        cam_type="daheng",
        cam_id=0,
        pupil_center_xy=(960, 720),
        device="cpu",
    )


def run(out_override: Path | None = None) -> dict[str, Any]:
    if out_override is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / "data" / "debug" / f"sim2f_zero_order_diag_{stamp}"
    else:
        out_dir = out_override
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    flat = np.zeros((128, 128), dtype=np.float32)

    instrument = _open_instrument()
    try:
        initial_zo = instrument.zero_order_xy
        initial_zo_list = [int(initial_zo[0]), int(initial_zo[1])]
        describe = instrument.describe()
        full_scale = float(instrument.full_scale)
        instrument.enable_full_frame()

        per_frame: list[dict[str, Any]] = []
        est_points: dict[str, list[tuple[float, float]]] = {n: [] for n in ESTIMATORS}
        for i in range(N_FRAMES):
            t_cap = _utc_now()
            frame = np.asarray(instrument.capture_full_frame(flat), dtype=np.float32)
            frame64 = frame.astype(np.float64)

            peak_rc = np.unravel_index(int(np.argmax(frame64)), frame64.shape)
            peak_counts = float(frame64.max())
            por = _peak_over_ring(frame64, peak_rc)
            near_ceiling = bool(peak_counts >= 0.95 * full_scale)

            estimates: dict[str, list[float]] = {}
            for name in ESTIMATORS:
                pt = _estimator_fn(name)(frame64)
                estimates[name] = [float(pt[0]), float(pt[1])]
                est_points[name].append(pt)

            argmax_pt = est_points["argmax_raw"][-1]
            centroid_pt = est_points["centroid_box"][-1]
            diff = float(np.hypot(argmax_pt[0] - centroid_pt[0], argmax_pt[1] - centroid_pt[1]))

            per_frame.append(
                {
                    "index": i,
                    "utc": t_cap,
                    "peak_counts": peak_counts,
                    "peak_over_ring": por,
                    "near_ceiling": near_ceiling,
                    "estimates": estimates,
                    "argmax_centroid_box_distance_px": diff,
                }
            )

            crop = _crop256(frame, initial_zo_list)
            np.save(frames_dir / f"crop_{i:03d}.npy", crop)
            if i == 0:
                np.save(frames_dir / "representative_full.npy", frame)
    finally:
        instrument.close()

    # argmax stability across frames
    argmax_pixels = [(int(round(p[0])), int(round(p[1]))) for p in est_points["argmax_raw"]]
    distinct = sorted(set(argmax_pixels))
    argmax_stable_all = len(distinct) == 1

    median_diff = float(
        np.median([pf["argmax_centroid_box_distance_px"] for pf in per_frame])
    )

    # reopen repeatability
    reopen_records: list[dict[str, Any]] = []
    for k in range(3):
        inst = _open_instrument()
        try:
            zo = inst.zero_order_xy
            reopen_records.append(
                {
                    "index": k,
                    "zero_order_xy": [int(zo[0]), int(zo[1])],
                    "utc": _utc_now(),
                }
            )
        finally:
            inst.close()

    reopen_points = [(float(r["zero_order_xy"][0]), float(r["zero_order_xy"][1])) for r in reopen_records]

    # per-estimator summaries
    estimator_summaries = {name: _summarize(est_points[name]) for name in ESTIMATORS}

    reopen_summary = _summarize(reopen_points)

    # combined all-opens (initial + 3 reopens) for context
    all_open_points = [ (float(initial_zo_list[0]), float(initial_zo_list[1])) ] + reopen_points
    all_opens_summary = _summarize(all_open_points)

    diag = {
        "schema": "sim2f_zero_order_diag_v1",
        "created_utc": _utc_now(),
        "params": {
            "exposure_ms": 0.182,
            "ccd_px": 256,
            "cam_type": "daheng",
            "cam_id": 0,
            "pupil_center_xy": [960, 720],
            "device": "cpu",
            "n_frames": N_FRAMES,
            "box_half_px": BOX_HALF,
            "crop_px": CROP,
        },
        "device": describe,
        "full_scale": full_scale,
        "initial_open_zero_order_xy": initial_zo_list,
        "frames_saved": {
            "mode": "one_representative_full_frame_plus_256x256_crop_per_frame",
            "representative_full": "frames/representative_full.npy",
            "crop_pattern": "frames/crop_{index:03d}.npy",
            "n_crops": N_FRAMES,
            "crop_size_px": CROP,
            "crop_centre_xy": initial_zo_list,
            "padded_to_square": True,
        },
        "argmax_stability": {
            "all_30_same_pixel": bool(argmax_stable_all),
            "n_distinct_pixels": int(len(distinct)),
            "distinct_pixels": [[int(p[0]), int(p[1])] for p in distinct],
        },
        "median_abs_diff_argmax_vs_centroid_box_px": median_diff,
        "per_frame": per_frame,
        "estimator_summaries": estimator_summaries,
        "reopen": {
            "opens": reopen_records,
            "summary_3_reopens": reopen_summary,
            "all_opens_including_initial": {
                "points": [[float(p[0]), float(p[1])] for p in all_open_points],
                "summary": all_opens_summary,
            },
        },
        "estimators_documented": list(ESTIMATORS),
    }

    with (out_dir / "diag.json").open("w", encoding="utf-8") as fh:
        json.dump(diag, fh, indent=2, allow_nan=True)

    # spot.png
    rep = np.load(frames_dir / "representative_full.npy")
    zo = initial_zo_list
    r0 = max(0, zo[1] - SPOT_HALF)
    r1 = min(rep.shape[0], zo[1] + SPOT_HALF)
    c0 = max(0, zo[0] - SPOT_HALF)
    c1 = min(rep.shape[1], zo[0] + SPOT_HALF)
    crop_spot = rep[r0:r1, c0:c1]
    rx, ry = argmax_raw(rep)
    cx, cy = centroid_despiked_box(rep)
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.imshow(np.sqrt(np.clip(crop_spot, 0.0, None)), cmap="gray", origin="upper")
    ax.plot(rx - c0, ry - r0, "rx", markersize=12, markeredgewidth=2, label="argmax_raw")
    ax.plot(cx - c0, cy - r0, "co", markersize=8, label="centroid_despiked_box")
    ax.set_title("sqrt stretch; 128x128 around reported zero order")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "spot.png", dpi=120)
    plt.close(fig)

    # compact stdout summary
    print(f"output_dir: {out_dir}")
    print(f"full_scale: {full_scale:g}  initial_zero_order_xy: {initial_zo_list}")
    print(f"argmax stable across 30 frames: {argmax_stable_all}  distinct: {len(distinct)}")
    print(f"median |argmax - centroid_box| per frame: {median_diff:.3f} px")
    print("")
    header = f"{'estimator':<24}{'radial_std_px':>16}{'radial_spread_px':>18}{'ptp_x':>10}{'ptp_y':>10}"
    print(header)
    print("-" * len(header))
    for name in ESTIMATORS:
        s = estimator_summaries[name]
        print(
            f"{name:<24}{s['radial_std_px']:>16.3f}{s['radial_spread_px']:>18.3f}"
            f"{s['ptp_x']:>10.3f}{s['ptp_y']:>10.3f}"
        )
    print("")
    print("reopen zero_order_xy:")
    for r in reopen_records:
        print(f"  open {r['index']}: {r['zero_order_xy']}")
    rs = reopen_summary
    print(
        f"  spread (3 reopens): radial_std={rs['radial_std_px']:.3f} px, "
        f"ptp_x={rs['ptp_x']:.3f}, ptp_y={rs['ptp_y']:.3f}, "
        f"radial_spread={rs['radial_spread_px']:.3f} px"
    )
    # diagnostics summary
    peaks = [pf["peak_counts"] for pf in per_frame]
    rings = [pf["peak_over_ring"] for pf in per_frame if pf["peak_over_ring"] is not None]
    near = sum(1 for pf in per_frame if pf["near_ceiling"])
    print("")
    print(f"peak_counts: min={min(peaks):.1f} max={max(peaks):.1f} median={float(np.median(peaks)):.1f}")
    if rings:
        print(f"peak_over_ring (8-nn median): min={min(rings):.2f} max={max(rings):.2f} median={float(np.median(rings)):.2f}")
    print(f"frames near ceiling (>=0.95*full_scale): {near}/{N_FRAMES}")

    return diag


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="override output dir")
    args = parser.parse_args(argv)
    run(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
