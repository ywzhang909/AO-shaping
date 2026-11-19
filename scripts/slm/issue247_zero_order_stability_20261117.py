"""Issue #247 offline analysis of the *true* tilt capture (Mono12 full frame).

Input is the output of ``issue247_capture_tilt_20261117.py``: a manifest with
``label`` / ``kind`` entries and ``NN_label.npz`` files holding ``raw_frames``
(5x native uint16 Mono12), ``phase_rad`` and ``panel_gray``.

Physics this tests (see GitHub issue #247):

* The **zero order** is the unmodulated / stationary component: it does NOT
  move when a linear phase ramp is applied.  In a pair of opposite tilts it is
  present in both frames at the same place, so ``min(rate(+), rate(-))``
  isolates it.  Its centre must coincide with the flat-frame focus and stay
  stable across the tilt sweep.

* The **target (first diffracted order)** IS steered by the ramp.  The
  antisymmetric difference ``rate(+) - rate(-)`` cancels the stationary zero
  order and leaves two lobes at ``Z +/- d``; their separation gives the target
  displacement ``d`` and its direction must follow the applied tilt axis.  The
  difference is background-subtracted (median) and the lobes are selected with
  a peak-relative threshold, so a frame-wide offset cannot bias the centroids
  and the measurement is invariant under the rotation/canvas change below.

Everything is rate-normalised (counts/ms) so roles with the same exposure
compare directly.  Drift is checked with the repeated flat baselines.

Panel/CCD mount rotation is **measured**, not assumed: the two tilt pairs give
the panel +x and +y displacement vectors on the unrotated frames, the correction
angle comes from those vectors (:mod:`ao_shaping.utils.image.rotation`), and the
loaded frames are rotated to panel-aligned axes before any centre/lobe analysis.
If either axis pair is unavailable or invalid, the raw frames are analysed and
the reason is recorded in ``image_alignment`` instead of inventing an angle.

OFFLINE ONLY: never touches SLM, camera, laser or DeviceLock.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ao_shaping.utils.image.rotation import (
    ROTATION_CONVENTION,
    AxisRotationError,
    AxisRotationEstimate,
    estimate_axis_rotation,
    rotate_image_about_center,
    rotated_canvas_shape,
)
from ao_shaping.utils.image.zero_order import zero_order_center_by_method

ZERO_METHODS = ("argmax", "centroid", "box_centroid", "ellipse")
PAIRS = (
    ("tilt_x_plus", "tilt_x_minus", "panel_x"),
    ("tilt_y_plus", "tilt_y_minus", "panel_y"),
)
# Peak-relative threshold (fraction of the background-subtracted |rp - rm|
# maximum) selecting lobe pixels for the centroid measurement.
LOBE_THRESHOLD_FRAC = 0.02


def _rate(session: Path, row: dict) -> np.ndarray:
    """Mean raw frame in counts/ms (rate-normalised), float64."""
    data = np.load(session / row["file"], allow_pickle=False)
    raw = np.asarray(data["raw_frames"], dtype=np.float64)
    exp = float(np.median(row["exposure_ms_readback"]))
    return raw.mean(axis=0) / max(exp, 1e-12)


def _centroid(frame: np.ndarray, *, positive: bool = True) -> tuple[float, float] | None:
    w = np.maximum(frame, 0.0) if positive else np.maximum(-frame, 0.0)
    total = float(w.sum())
    if total <= 0:
        return None
    yy, xx = np.mgrid[0 : frame.shape[0], 0 : frame.shape[1]]
    return (float((xx * w).sum() / total), float((yy * w).sum() / total))


def _pair_displacement(
    rp: np.ndarray,
    rm: np.ndarray,
    *,
    lobe_frac: float = LOBE_THRESHOLD_FRAC,
) -> dict | None:
    """Antisymmetric-difference lobes: target displacement from minus to plus.

    ``rp - rm`` cancels the stationary zero order and leaves a positive lobe at
    the target's ``+`` position and a negative lobe at its ``-`` position.  The
    robust background of the difference (its median) is subtracted first so a
    frame-wide offset cannot bias the centroids; lobe pixels are then selected
    with a peak-relative threshold ``lobe_frac * max(|d|)`` and weighted by
    ``d`` (positive lobe) / ``-d`` (negative lobe).  Half their separation is
    the panel-axis displacement vector in CCD x/y pixels.  Also returns the
    lobe midpoint, which must sit on the stationary zero order.
    """
    d = np.asarray(rp, dtype=np.float64) - np.asarray(rm, dtype=np.float64)
    d = d - float(np.median(d))
    peak = float(np.abs(d).max())
    if peak <= 0:
        return None
    thr = lobe_frac * peak
    pos = _centroid(np.where(d >= thr, d, 0.0), positive=True)
    neg = _centroid(np.where(d <= -thr, -d, 0.0), positive=True)
    if pos is None or neg is None:
        return None
    return {
        "pos": pos,
        "neg": neg,
        "mid": ((pos[0] + neg[0]) / 2.0, (pos[1] + neg[1]) / 2.0),
        "dx": (pos[0] - neg[0]) / 2.0,
        "dy": (pos[1] - neg[1]) / 2.0,
        "sep": float(np.hypot(pos[0] - neg[0], pos[1] - neg[1])),
    }


def _calibrate_alignment(
    rates: dict[str, np.ndarray],
    *,
    lobe_frac: float = LOBE_THRESHOLD_FRAC,
) -> tuple[AxisRotationEstimate | None, dict]:
    """Measure both panel-axis vectors on unrotated frames and estimate rotation.

    A missing or invalid axis pair returns ``(None, metadata_with_reason)``; the
    caller then keeps the raw frames instead of inventing an angle.
    """
    first = next(iter(rates.values())) if rates else None
    source_shape = list(first.shape) if first is not None else None
    meta: dict = {
        "available": False,
        "convention": ROTATION_CONVENTION,
        "vector_definition": (
            "half the (tilt_*_plus - tilt_*_minus) target-lobe displacement, "
            "taken as the panel +axis direction"
        ),
        "correction_deg": None,
        "frames_rotated": False,
        "analysis_frame": "raw_unrotated",
        "source_shape": source_shape,
        "aligned_shape": source_shape,
        "interpolation_order": None,
        "pad_value": None,
        "lobe_threshold_frac": float(lobe_frac),
    }

    vectors: dict[str, tuple[float, float]] = {}
    unusable: list[str] = []
    for plus, minus, axis in PAIRS:
        if plus not in rates or minus not in rates:
            missing = plus if plus not in rates else minus
            unusable.append(f"{axis} (missing {missing})")
            continue
        disp = _pair_displacement(rates[plus], rates[minus], lobe_frac=lobe_frac)
        if disp is None:
            unusable.append(f"{axis} (no separable +/- lobes)")
            continue
        vectors[axis] = (disp["dx"], disp["dy"])
        meta[f"{axis}_delta_xy"] = [float(disp["dx"]), float(disp["dy"])]
        meta[f"{axis}_lobes_mid_xy"] = [float(disp["mid"][0]), float(disp["mid"][1])]

    if unusable:
        meta["reason"] = "unusable axis pair(s): " + ", ".join(unusable)
        return None, meta
    try:
        estimate = estimate_axis_rotation(vectors["panel_x"], vectors["panel_y"])
    except AxisRotationError as exc:
        meta["reason"] = str(exc)
        return None, meta

    meta.update(
        available=True,
        correction_deg=float(estimate.correction_deg),
        panel_to_ccd_deg=float(estimate.panel_to_ccd_deg),
        panel_x_angle_deg=float(estimate.panel_x_angle_deg),
        panel_y_angle_deg=float(estimate.panel_y_angle_deg),
        panel_x_magnitude_px=float(estimate.panel_x_magnitude_px),
        panel_y_magnitude_px=float(estimate.panel_y_magnitude_px),
        orthogonality_error_deg=float(estimate.orthogonality_error_deg),
        scale_ratio=float(estimate.scale_ratio),
        max_residual_px=float(estimate.max_residual_px),
    )
    return estimate, meta


def _align_rates(
    rates: dict[str, np.ndarray],
    *,
    lobe_frac: float = LOBE_THRESHOLD_FRAC,
) -> tuple[dict[str, np.ndarray], dict]:
    """Rotate every loaded rate frame to panel-aligned axes when calibration works."""
    estimate, meta = _calibrate_alignment(rates, lobe_frac=lobe_frac)
    if estimate is None:
        return rates, meta

    first = next(iter(rates.values()))
    aligned_shape = rotated_canvas_shape(first.shape, estimate.correction_deg)
    rotated = {
        label: rotate_image_about_center(
            frame,
            estimate.correction_deg,
            output_shape=aligned_shape,
            order=1,
        )
        for label, frame in rates.items()
    }
    meta.update(
        frames_rotated=True,
        analysis_frame="panel_aligned_rotated",
        aligned_shape=list(aligned_shape),
        interpolation_order=1,
        pad_value=0.0,
    )
    return rotated, meta


def analyse(
    session: Path,
    methods: tuple[str, ...],
    *,
    lobe_frac: float = LOBE_THRESHOLD_FRAC,
) -> dict:
    manifest = json.loads((session / "manifest.json").read_text())
    by_label = {r["label"]: r for r in manifest}
    rates = {label: _rate(session, row) for label, row in by_label.items()}
    # Calibrate on unrotated frames, then rotate everything before analysis.
    # Source .npz captures are never modified.
    rates, image_alignment = _align_rates(rates, lobe_frac=lobe_frac)

    out: dict = {
        "session": str(session),
        "captures": len(manifest),
        "exposure_ms": float(np.median([r["exposure_ms_readback"][0] for r in manifest])),
        "peak_counts": {lbl: int(r["peak"]) for lbl, r in by_label.items()},
        "saturated_pixels": {lbl: int(r["saturated_pixels"]) for lbl, r in by_label.items()},
        "image_alignment": image_alignment,
        "flat_centres": {},
        "tilt": {},
        "defocus": {},
    }

    # --- flats: drift + zero-order centre estimate -------------------------
    flat_labels = [lbl for lbl in by_label if by_label[lbl]["kind"] == "flat"]
    for lbl in flat_labels:
        frame = rates[lbl]
        out["flat_centres"][lbl] = {
            m: [int(c[0]), int(c[1])]
            for m, c in ((m, zero_order_center_by_method(frame, m, despike_k=3))
                         for m in methods)
        }
    # Representative zero-order centre: median of the elliptical centres of
    # the pre/mid/post flats (analysis-frame pixels, x/y; see image_alignment).
    flat_xy = np.array([
        out["flat_centres"][lbl]["ellipse"] for lbl in flat_labels
    ], dtype=float)
    zero_xy = np.median(flat_xy, axis=0)
    out["zero_order_xy"] = [float(zero_xy[0]), float(zero_xy[1])]
    out["flat_drift_px"] = {
        "max_from_median": float(np.max(np.linalg.norm(flat_xy - zero_xy, axis=1))),
        "per_label": {lbl: float(np.linalg.norm(xy - zero_xy))
                      for lbl, xy in zip(flat_labels, flat_xy)},
    }

    # --- tilt pairs --------------------------------------------------------
    for plus, minus, axis in PAIRS:
        if plus not in rates or minus not in rates:
            continue
        rp, rm = rates[plus], rates[minus]
        # Stationary zero order: present in both -> min suppresses the target.
        common = np.minimum(rp, rm)
        z_centre = zero_order_center_by_method(common, "ellipse", despike_k=3)
        # Antisymmetric difference: stationary part cancels, target lobes at Z±d.
        disp = _pair_displacement(rp, rm, lobe_frac=lobe_frac)
        info: dict = {
            "axis": axis,
            "zero_from_min_xy": [int(z_centre[0]), int(z_centre[1])],
            "zero_stability_px": float(np.linalg.norm(
                np.array(z_centre, dtype=float) - zero_xy)),
        }
        if disp is not None:
            pos, neg = disp["pos"], disp["neg"]
            dx, dy = disp["dx"], disp["dy"]
            info.update(
                target_plus_lobe_xy=[round(pos[0], 2), round(pos[1], 2)],
                target_minus_lobe_xy=[round(neg[0], 2), round(neg[1], 2)],
                target_lobes_midpoint_xy=[round(disp["mid"][0], 2), round(disp["mid"][1], 2)],
                target_displacement_px=[round(dx, 2), round(dy, 2)],
                target_separation_px=round(disp["sep"], 2),
                displacement_along_axis_px=round(dx if axis == "panel_x" else dy, 2),
                displacement_cross_axis_px=round(dy if axis == "panel_x" else dx, 2),
            )
        out["tilt"][f"{plus}/{minus}"] = info

    # --- defocus: centre stability + width ---------------------------------
    for label in ("defocus_plus", "defocus_minus"):
        if label not in rates:
            continue
        frame = rates[label]
        c = zero_order_center_by_method(frame, "ellipse", despike_k=3)
        bg = float(np.median(frame[:8, :8]))
        w = np.maximum(frame - bg, 0.0)
        total = float(w.sum())
        cycx = float((w * np.indices(w.shape)[0]).sum() / total) if total else float("nan")
        out["defocus"][label] = {
            "centre_ellipse_xy": [int(c[0]), int(c[1])],
            "centroid_y": round(cycx, 2),
            "peak_counts_ms": float(frame.max()),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", type=Path, required=True,
                    help="issue247 tilt capture dir (contains manifest.json)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--methods", default=",".join(ZERO_METHODS))
    args = ap.parse_args()

    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    res = analyse(args.session, methods)

    print(f"=== {args.session.name} : {res['captures']} captures, "
          f"exposure {res['exposure_ms']:.3f} ms ===")
    sat = sum(res["saturated_pixels"].values())
    print(f"saturated pixels total = {sat}; peak counts "
          f"{min(res['peak_counts'].values())}..{max(res['peak_counts'].values())}")
    align = res["image_alignment"]
    if align["frames_rotated"]:
        print(f"image alignment: rotated {align['correction_deg']:+.3f} deg to panel axes; "
              f"{tuple(align['source_shape'])} -> {tuple(align['aligned_shape'])}")
    else:
        print(f"image alignment: unavailable ({align['reason']}); using raw frames")
    print("flat centres (ellipse, x/y) and drift:")
    for lbl, m in res["flat_centres"].items():
        print(f"  {lbl:16s} {m['ellipse']}  "
              f"|dZ|={res['flat_drift_px']['per_label'][lbl]:.2f} px")
    print(f"zero-order centre (median) = {res['zero_order_xy']}; "
          f"max flat drift = {res['flat_drift_px']['max_from_median']:.2f} px")
    for name, t in res["tilt"].items():
        print(f"  {name}: zero_from_min={t['zero_from_min_xy']} "
              f"|dZ|={t['zero_stability_px']:.2f} px; "
              f"target d={t.get('target_displacement_px')} "
              f"(along={t.get('displacement_along_axis_px')}, "
              f"cross={t.get('displacement_cross_axis_px')})")
    for label, d in res["defocus"].items():
        print(f"  {label}: centre={d['centre_ellipse_xy']} "
              f"peak={d['peak_counts_ms']:.0f}/ms")

    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / "issue247_analysis.json"
    res["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dest.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
