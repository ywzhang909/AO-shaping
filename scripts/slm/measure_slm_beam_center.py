"""Measure the SLM-panel beam center via background-subtracted knife-edge scans.

Issue #235: the previous knife-edge probe gave an inconsistent SLM beam center
twice on 2026-11-10 -- run1 (518.7, 788.5) vs run2 (612.2, 890.2), ~100 px
apart -- and the 0-order was not suppressed (high E0/E1, blaze-side orders
unresolvable), so the cumulative-distribution fit was contaminated by
background. This script re-measures the center on the *same* knife-edge
measurement (``SLMCCDCalibrator.find_beam_on_slm`` style: one side cleared to
0 phase, the other a 0/pi checkerboard that scatters out of the 0-order) but:

* subtracts a real dark frame (``--dark-frame``) when given; otherwise falls
  back to the per-frame median, which is recorded as an explicit gap,
* records the 0-order window power at 1/2/3/6/12 x FWHM for every frame, so the
  offline analysis can see how much the answer depends on the window,
* records a CCD-side 0-order box centroid as *reference evidence only* (it is
  the camera-side spot, NOT the SLM-panel center),
* emits ``reliable: true/false`` together with per-item criteria; when false it
  does NOT emit a usable center (``beam_center_cy_cx`` is null) and prints a
  loud warning and exits 1; raw evidence is preserved.

The panel geometric center is reported alongside (cy, cx) = (600, 960), which
matches ``ao_shaping.utils.slm.bench_constants.BEAM_CENTER_PANEL``.

Default exposure is 0.05 ms per the task spec (non-saturating). Bench notes
(2026-10-22) show this bench's usable window is 1-10 ms and that 0.05 ms sits at
the read-noise floor (2 counts) -- so at the default this script will most
likely report ``reliable: false``. That is the correct outcome, not a bug: the
raw frames are saved and the analysis can be re-run with a real exposure.

Plan-only by default: no ``--execute`` prints the plan as JSON and exits 0
before creating ``--out`` or touching any device. ``--out`` must be a directory
that does not yet exist (``exist_ok=False``).

Usage (plan-only, then hardware)::

    python scripts/slm/measure_slm_beam_center.py --out <new-dir>
    python scripts/slm/measure_slm_beam_center.py --out <new-dir> --execute

Devices are acquired under ``DeviceLock`` (bench mutex). The SLM is guaranteed
to be reset to uniform gray 0 and closed, and the camera closed, in ``finally``
even when a pattern fails. Per-frame failures are collected and reported in the
summary JSON; they do not abort the session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

#: Panel resolution (w, h); matched to the Santec ``Panel_Res`` (1920, 1200).
PANEL_RES_WH = (1920, 1200)
#: Panel geometric center in (cy, cx) panel px; equals BEAM_CENTER_PANEL (960, 600) as (x, y).
PANEL_GEOMETRIC_CENTER_CY_CX = (600.0, 960.0)
#: Window sizes recorded for every knife frame, as multiples of the flat-field FWHM.
WINDOW_FACTORS = (1.0, 2.0, 3.0, 6.0, 12.0)
#: The window factor whose center/sigma is reported as *the* answer (library default).
REFERENCE_WINDOW_FACTOR = 6.0
#: CCD 0-order box (full side, px) used for the box-centroid reference evidence.
CCD_BOX_SIDE = 120
#: Knife-edge checkerboard period (panel px), matching find_beam_on_slm.
CHECKER_PERIOD = 8

#: Reliability thresholds. Documented here so the offline reader can audit them.
MIN_FLAT_PEAK_SIGNAL_COUNTS = 20.0  # flat peak minus background median (8-bit counts)
MAX_BACKGROUND_FRACTION = 0.50  # background energy / total energy in the ref window
MIN_RELATIVE_MODULATION = 0.30  # (Pmax-Pmin)/Pmax of a knife curve at the ref window
MIN_SPEARMAN = 0.90  # signed Spearman rank of P(s) vs s (power rises with s)
MIN_MONOTONE_SEGMENT_FRACTION = 0.80  # fraction of adjacent steps with non-negative slope
MAX_CENTER_WINDOW_SPREAD_PX = 10.0  # spread of the 50% crossing across window factors
MAX_SIGMA_RELATIVE_SPREAD = 0.25  # spread of sigma / median sigma across window factors
EDGE_MARGIN_FRACTION = 0.05  # the 50% crossing must lie inside this margin of the scan range


def _fwhm1d(profile: np.ndarray) -> float:
    """FWHM (px) of a 1-D peak profile, interpolated at half max.

    The profile is assumed background-subtracted and non-negative. Returns 0.0
    when no crossing can be found (flat/degenerate input).
    """
    p = np.asarray(profile, dtype=np.float64)
    if p.size < 2:
        return 0.0
    peak = float(p.max())
    if peak <= 0.0:
        return 0.0
    half = 0.5 * peak
    imax = int(np.argmax(p))
    # left crossing
    i_left = imax
    while i_left > 0 and p[i_left] >= half:
        i_left -= 1
    if p[i_left] < half and i_left < imax:
        frac = (half - p[i_left]) / (p[i_left + 1] - p[i_left] + 1e-12)
        x_left = i_left + frac
    else:
        x_left = float(imax)
    # right crossing
    i_right = imax
    while i_right < p.size - 1 and p[i_right] >= half:
        i_right += 1
    if p[i_right] < half and i_right > imax:
        frac = (half - p[i_right]) / (p[i_right - 1] - p[i_right] + 1e-12)
        x_right = i_right + frac
    else:
        x_right = float(imax)
    width = x_right - x_left
    return float(width) if width > 0.0 else 0.0


def _window_slices(cy: int, cx: int, side: int, h: int, w: int):
    """Clipped window slices around (cy, cx) plus the actual bounds.

    Returns (y0, y1, x0, x1) such that ``frame[y0:y1, x0:x1]`` is the window.
    """
    half = max(1, side // 2)
    y0 = max(0, cy - half)
    y1 = min(h, cy + half)
    x0 = max(0, cx - half)
    x1 = min(w, cx + half)
    return y0, y1, x0, x1


def _window_power(frame: np.ndarray, cy: int, cx: int, side: int) -> tuple[float, tuple]:
    """Sum of ``frame`` inside the ``side x side`` window centered at (cy, cx).

    Returns ``(power, (y0, y1, x0, x1))``. The frame is assumed already
    background-subtracted; the power is the sum of the (possibly clipped) window.
    """
    h, w = frame.shape
    y0, y1, x0, x1 = _window_slices(cy, cx, side, h, w)
    return float(frame[y0:y1, x0:x1].sum()), (y0, y1, x0, x1)


def _box_centroid(frame: np.ndarray, cy: int, cx: int, side: int) -> tuple[float, float]:
    """Weighted centroid (cy, cx) of a background-subtracted frame in a box.

    Thresholds at 15% of the box max to suppress residual noise, mirroring
    ``SLMCCDCalibrator._moments``. Returns the centroid in full-frame coords.
    """
    h, w = frame.shape
    y0, y1, x0, x1 = _window_slices(cy, cx, side, h, w)
    box = np.asarray(frame[y0:y1, x0:x1], dtype=np.float64)
    box = np.maximum(box, 0.0)
    thr = float(box.max()) * 0.15
    m = box >= thr
    if int(m.sum()) < 5:
        return float(cy), float(cx)
    yy, xx = np.nonzero(m)
    wv = box[m]
    cy_b = float((yy * wv).sum() / wv.sum()) + y0
    cx_b = float((xx * wv).sum() / wv.sum()) + x0
    return cy_b, cx_b


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Signed Spearman rank correlation (scipy), NaN-safe."""
    try:
        from scipy.stats import spearmanr
    except ImportError:  # pragma: no cover - scipy is a project dependency
        return float("nan")
    r = spearmanr(x, y).statistic
    if r is None or np.isnan(r):
        return float("nan")
    return float(r)


def _curve_fit(P: np.ndarray, s: np.ndarray) -> dict:
    """Cumulative-distribution fit of a knife-edge curve P(s).

    Mirrors ``find_beam_on_slm``: P is rescaled to [0, 1] on the *intensity*
    axis (legitimate -- this is a 1-D scalar intensity curve, not a phase map),
    then the 50% crossing and the (84%-16%)/2 sigma are read off via ``interp``.
    """
    Pn = (P - P.min()) / (np.ptp(P) + 1e-12)
    c50 = float(np.interp(0.50, Pn, s))
    c16 = float(np.interp(0.16, Pn, s))
    c84 = float(np.interp(0.84, Pn, s))
    sigma = 0.5 * (c84 - c16)
    return {
        "c50": c50,
        "c16": c16,
        "c84": c84,
        "sigma": float(sigma),
    }


def _curve_analysis(P: np.ndarray, s: np.ndarray) -> dict:
    """Monotonicity + modulation + CDF fit for one (axis, window) curve."""
    P = np.asarray(P, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    rng = float(P.max() - P.min())
    mod = rng / float(P.max()) if P.max() > 0.0 else 0.0
    # Power should rise as the clear region grows (s increasing).
    diffs = np.diff(P)
    seg_frac = float(np.mean(diffs >= 0.0)) if diffs.size else 0.0
    return {
        "power_min": float(P.min()),
        "power_max": float(P.max()),
        "power_range": rng,
        "relative_modulation": mod,
        "spearman": _spearman(s, P),
        "monotone_segment_fraction": seg_frac,
        "fit": _curve_fit(P, s),
        "points": [[float(a), float(b)] for a, b in zip(s, P)],
    }


def _jsonable(obj):
    """Recursively convert numpy scalars/arrays to JSON-safe Python objects."""
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _plan(args: argparse.Namespace) -> dict:
    n_knife = 2 * args.n_scan
    return {
        "plan_only": True,
        "background": _background_info(args.dark_frame),
        "dark_frame": str(args.dark_frame) if args.dark_frame else None,
        "script": "scripts/slm/measure_slm_beam_center.py",
        "issue": 235,
        "devices": {
            "slm": args.slm_number,
            "slm_video_mode": 0,
            "camera": "daheng",
            "cam_id": args.cam_id,
            "exposure_ms_requested": args.exposure_ms,
        },
        "action": (
            "knife-edge scan (background-subtracted, %d patterns/axis, checker "
            "period %d px) + flat-field CCD 0-order box centroid" % (args.n_scan, CHECKER_PERIOD)
        ),
        "n_patterns": 1 + n_knife,
        "window_factors": list(WINDOW_FACTORS),
        "reference_window_factor": REFERENCE_WINDOW_FACTOR,
        "ccd_box_side_px": CCD_BOX_SIDE,
        "panel_res_wh": list(PANEL_RES_WH),
        "panel_geometric_center_cy_cx": list(PANEL_GEOMETRIC_CENTER_CY_CX),
        "thresholds": {
            "min_flat_peak_signal_counts": MIN_FLAT_PEAK_SIGNAL_COUNTS,
            "max_background_fraction": MAX_BACKGROUND_FRACTION,
            "min_relative_modulation": MIN_RELATIVE_MODULATION,
            "min_spearman": MIN_SPEARMAN,
            "min_monotone_segment_fraction": MIN_MONOTONE_SEGMENT_FRACTION,
            "max_center_window_spread_px": MAX_CENTER_WINDOW_SPREAD_PX,
            "max_sigma_relative_spread": MAX_SIGMA_RELATIVE_SPREAD,
            "edge_margin_fraction": EDGE_MARGIN_FRACTION,
        },
        "writes": [
            "summary.json",
            "plan.json",
            "flat_full.npy",
            "frames/<axis>_<index>.npy",
        ],
        "notes": [
            "CCD 0-order box centroid is camera-side reference evidence only; it is NOT the SLM panel center.",
            "Default exposure 0.05 ms is non-saturating but sits at this bench's read-noise floor; at that exposure reliable=false is expected and correct.",
            "actual_gray_available=false for phase patterns: get_displayed_phase() returns the driver-side write cache, not a device readback.",
        ],
    }


def _background_info(path: Path | None) -> dict:
    return {
        "method": "dark_frame" if path is not None else "frame_median",
        "dark_frame": str(path.resolve()) if path is not None else None,
        "gaps": [] if path is not None else [
            "背景为每帧全图 median，非真实暗场，固定热点未剥离"
        ],
    }


def _subtract_background(raw: np.ndarray, dark: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if dark is not None and dark.shape != raw.shape:
        raise ValueError(f"Dark frame shape {dark.shape} != CCD frame shape {raw.shape}")
    background = dark if dark is not None else np.full(raw.shape, np.median(raw))
    return np.maximum(raw - background, 0.0), background


def _exit_code(reliable: bool) -> int:
    return 0 if reliable else 1


def _prepare_out_dir(out: Path) -> Path:
    """Create the fresh run directory and its ``frames/`` subdirectory.

    ``np.save(out / "frames/x_00.npy", ...)`` does not create parents, so the
    subdirectory must exist before the first pattern is captured — otherwise the
    whole scan dies after the payload is already acquired (2026-11-11 run).
    Refuses to reuse an existing run directory: raw frames must never be
    silently overwritten.
    """
    out.mkdir(parents=True, exist_ok=False)
    (out / "frames").mkdir()
    return out


def _capture_flat(slm, cam, dark: np.ndarray | None) -> tuple[dict, np.ndarray]:
    """Flat-field (all-zero) frame and CCD 0-order box-centroid reference evidence.

    Returns ``(record, raw_float64_frame)`` so the caller can save the raw frame.
    """
    from ao_shaping.drivers.ccd.common import is_saturated

    # Flat phase must be sent as raw uint16 gray, never via create_phase_from_array.
    h, w = slm.Panel_Res[1], slm.Panel_Res[0]
    slm.display_data(np.zeros((h, w), dtype=np.uint16))
    raw = np.asarray(cam.get_numpy_image(1, True), dtype=np.float64)
    med = float(np.median(dark)) if dark is not None else float(np.median(raw))
    peak = float(raw.max())
    saturated = bool(is_saturated(raw))

    bs, background = _subtract_background(raw, dark)
    cy, cx = map(int, np.unravel_index(np.argmax(bs), bs.shape))
    cy_c, cx_c = _box_centroid(bs, cy, cx, CCD_BOX_SIDE)
    fwhm_row = _fwhm1d(bs[cy, :])
    fwhm_col = _fwhm1d(bs[:, cx])
    # window powers for the flat frame at every factor (evidence of how spread the spot is)
    windows = {}
    for f in WINDOW_FACTORS:
        side = int(round(f * max((fwhm_row + fwhm_col) / 2.0, 5.0)))
        p, bounds = _window_power(raw, cy, cx, side)
        bg_frac = _window_power(background, cy, cx, side)[0] / (p + 1e-12)
        windows[str(int(f * 100) / 100)] = {
            "side": side,
            "bounds": list(bounds),
            "power_raw": float(p),
            "background_fraction": float(bg_frac),
        }

    displayed = _displayed_provenance(slm)
    rec = {
        "kind": "flat",
        "frame_file": "flat_full.npy",
        "frame_shape": list(raw.shape),
        "frame_dtype": str(raw.dtype),
        "background_median": med,
        "background_source": "dark_frame" if dark is not None else "frame_median",
        "peak": peak,
        "peak_background_subtracted": float(bs.max()),
        "saturated": saturated,
        "argmax_cy_cx": [float(cy), float(cx)],
        "ccd_box_centroid_cy_cx": [cy_c, cx_c],
        "ccd_box_side_px": CCD_BOX_SIDE,
        "fwhm_row_px": fwhm_row,
        "fwhm_col_px": fwhm_col,
        "fwhm_geometric_px": float(np.hypot(fwhm_row, fwhm_col)),
        "windows": windows,
        "actual_gray_available": displayed["actual_gray_available"],
        "displayed_phase_sha256": displayed["displayed_phase_sha256"],
        "displayed_gray_source": displayed["displayed_gray_source"],
        "gray_readback_note": displayed["gray_readback_note"],
    }
    return rec, raw


def _displayed_provenance(slm) -> dict:
    """Commanded/cached gray hash + provenance string; never fabricate a readback."""
    try:
        phase_gray, source = slm.get_displayed_phase()
    except Exception as exc:
        return {
            "actual_gray_available": False,
            "displayed_phase_sha256": None,
            "displayed_gray_source": f"get_displayed_phase() failed: {exc}",
            "gray_readback_note": (
                "get_displayed_phase() returned the driver-side write cache, not a device "
                "readback; no panel gray value can be trusted from it."
            ),
        }
    if phase_gray is None:
        return {
            "actual_gray_available": False,
            "displayed_phase_sha256": None,
            "displayed_gray_source": source,
            "gray_readback_note": (
                "No phase image available from get_displayed_phase() (SDK cannot read back "
                "panel memory); recorded commanded pattern only."
            ),
        }
    return {
        "actual_gray_available": False,
        "displayed_phase_sha256": _sha256(phase_gray),
        "displayed_gray_shape": list(phase_gray.shape),
        "displayed_gray_dtype": str(phase_gray.dtype),
        "displayed_gray_source": source,
        "gray_readback_note": (
            "get_displayed_phase() returned the driver-side write cache ("
            + source
            + "), not a device readback; the hash is of the cached/commanded uint16 gray, "
            "and actual_gray_available=false."
        ),
    }


def _run(args: argparse.Namespace) -> int:
    """Hardware acquisition body. Devices are opened/closed inside a lock."""
    # Lazy device imports: the ONLY place drivers are loaded.
    from ao_shaping.drivers.ccd.common import (
        create_camera,
        get_camera_exposure_ms,
        is_saturated,
        set_camera_exposure_ms,
    )
    from ao_shaping.drivers.slm.factory import require_drive_board_entry
    from ao_shaping.drivers.slm.santec import Santec
    from ao_shaping.tools.kernels import device_mutex
    from ao_shaping.tools.kernels.device_mutex import DeviceLock
    from ao_shaping.tools.slm.transfer.hardware import prepare_camera_sdk

    # --- bench mutex ---------------------------------------------------------
    lock = DeviceLock.default()
    if not lock.acquire(timeout=0.0):
        holder = lock.holder()
        print(
            "ERROR: bench device lock held by %s; refusing to start (hardware discipline)."
            % (holder if holder else "an unknown process"),
            file=sys.stderr,
        )
        return 2

    dark = args.dark
    out = _prepare_out_dir(args.out)
    failures: list[dict] = []
    frames: list[dict] = []
    release: dict = {"slm_flat_gray0": "not-attempted", "slm_closed": "not-attempted", "cam_closed": "not-attempted"}

    slm = None
    cam = None
    t0 = time.time()
    try:
        prepare_camera_sdk("daheng")
        slm = Santec(slm_number=args.slm_number, video_mode=0)
        slm.open()
        # Acquisition-entry guard: refuse to measure if the drive board is dead.
        require_drive_board_entry(slm)
        print("SLM #%d open (drive board OK)" % args.slm_number)

        cam = create_camera("daheng", args.cam_id, exposure_time_ms=args.exposure_ms)
        cam.open()
        set_camera_exposure_ms(cam, args.exposure_ms)
        exposure_readback = get_camera_exposure_ms(cam)
        print("camera open, exposure readback %.4f ms" % exposure_readback)

        panel_h, panel_w = slm.Panel_Res[1], slm.Panel_Res[0]

        # ---- flat field + CCD 0-order box centroid (reference evidence) ------
        print("capturing flat field ...")
        flat_rec, raw_flat = _capture_flat(slm, cam, dark)
        np.save(out / "flat_full.npy", raw_flat)
        frames.append(flat_rec)

        # The knife-edge window center and size come from the flat-field spot.
        cy = int(round(flat_rec["argmax_cy_cx"][0]))
        cx = int(round(flat_rec["argmax_cy_cx"][1]))
        fwhm_geo = flat_rec["fwhm_geometric_px"]
        ref_side = int(round(REFERENCE_WINDOW_FACTOR * max(fwhm_geo, 5.0)))

        # ---- knife-edge scans -------------------------------------------------
        def scan_pattern(s: int, axis: str) -> np.ndarray:
            yy, xx = np.mgrid[0:panel_h, 0:panel_w]
            clear = (xx < s) if axis == "x" else (yy < s)
            checker = ((xx // CHECKER_PERIOD + yy // CHECKER_PERIOD) % 2) * np.pi
            return np.where(clear, 0.0, checker).astype(np.float32)

        curves: dict[str, dict] = {}
        for axis in ("x", "y"):
            extent = panel_w if axis == "x" else panel_h
            ss = np.linspace(0, extent, args.n_scan)
            axis_curves: dict[str, list] = {str(f): [] for f in WINDOW_FACTORS}
            for idx, s in enumerate(ss):
                s = int(round(s))
                phase = scan_pattern(s, axis)
                try:
                    slm.display_phase(phase)
                except Exception as exc:
                    failures.append({"stage": "knife", "axis": axis, "s": float(s), "error": str(exc)})
                    print("FAILED knife %s s=%.0f: %s" % (axis, s, exc), file=sys.stderr)
                    continue
                raw = np.asarray(cam.get_numpy_image(1, True), dtype=np.float64)
                med = float(np.median(dark)) if dark is not None else float(np.median(raw))
                bs, _ = _subtract_background(raw, dark)
                prov = _displayed_provenance(slm)
                rec = {
                    "kind": "knife",
                    "axis": axis,
                    "index": idx,
                    "s_panel_px": float(s),
                    "frame_file": "frames/%s_%02d.npy" % (axis, idx),
                    "frame_shape": list(raw.shape),
                    "frame_dtype": str(raw.dtype),
                    "background_median": med,
                    "background_source": "dark_frame" if dark is not None else "frame_median",
                    "frame_max": float(raw.max()),
                    "saturated": bool(is_saturated(raw)),
                    "window_powers": {},
                    "actual_gray_available": prov["actual_gray_available"],
                    "displayed_phase_sha256": prov["displayed_phase_sha256"],
                    "displayed_gray_source": prov["displayed_gray_source"],
                }
                for f in WINDOW_FACTORS:
                    side = int(round(f * max(fwhm_geo, 5.0)))
                    p, bounds = _window_power(bs, cy, cx, side)
                    key = str(f)
                    rec["window_powers"][key] = {
                        "side": side,
                        "bounds": list(bounds),
                        "power": float(p),
                    }
                    axis_curves[key].append({"s": float(s), "power": float(p)})
                np.save(out / rec["frame_file"], raw)
                frames.append(rec)
            # analysis per window factor
            curves[axis] = {}
            for f in WINDOW_FACTORS:
                key = str(f)
                pts = axis_curves[key]
                P = np.array([pt["power"] for pt in pts], dtype=np.float64)
                s = np.array([pt["s"] for pt in pts], dtype=np.float64)
                curves[axis][key] = _curve_analysis(P, s)
            print("knife axis %s done (%d frames)" % (axis, len(axis_curves[list(axis_curves)[0]])))

        # ---- window dependence ------------------------------------------------
        window_dependence: dict[str, dict] = {}
        for axis in ("x", "y"):
            cs = [curves[axis][str(f)]["fit"]["c50"] for f in WINDOW_FACTORS]
            sg = [curves[axis][str(f)]["fit"]["sigma"] for f in WINDOW_FACTORS]
            c_spread = float(max(cs) - min(cs)) if cs else float("nan")
            med_sigma = float(np.median(sg)) if sg else float("nan")
            rel_spread = float((max(sg) - min(sg)) / med_sigma) if med_sigma and med_sigma > 0 else float("nan")
            window_dependence[axis] = {
                "centers_px": cs,
                "center_spread_px": c_spread,
                "sigmas_px": sg,
                "median_sigma_px": med_sigma,
                "sigma_relative_spread": rel_spread,
            }

        # ---- reliability criteria ---------------------------------------------
        reasons: list[str] = []
        criteria: list[dict] = []

        def add_criterion(name: str, passed: bool, detail: str) -> None:
            criteria.append({"name": name, "passed": bool(passed), "detail": detail})
            if not passed:
                reasons.append("%s: %s" % (name, detail))

        med_bg = flat_rec["background_median"]
        peak_sig = flat_rec["peak_background_subtracted"]
        add_criterion(
            "flat_signal_above_background",
            peak_sig >= MIN_FLAT_PEAK_SIGNAL_COUNTS,
            "flat peak %s counts, background median %s counts -> signal %s (need >= %.0f)"
            % (flat_rec["peak"], med_bg, peak_sig, MIN_FLAT_PEAK_SIGNAL_COUNTS),
        )
        add_criterion("flat_not_saturated", not flat_rec["saturated"], "flat frame saturated=%s" % flat_rec["saturated"])
        add_criterion(
            "no_acquisition_failures",
            not failures,
            "%d acquisition failure(s) recorded" % len(failures),
        )
        add_criterion(
            "background_fraction",
            flat_rec["windows"][str(REFERENCE_WINDOW_FACTOR)]["background_fraction"] <= MAX_BACKGROUND_FRACTION,
            "background fraction at %dx window = %.3f (need <= %.2f)"
            % (
                REFERENCE_WINDOW_FACTOR,
                flat_rec["windows"][str(REFERENCE_WINDOW_FACTOR)]["background_fraction"],
                MAX_BACKGROUND_FRACTION,
            ),
        )
        for axis in ("x", "y"):
            ref = curves[axis][str(REFERENCE_WINDOW_FACTOR)]
            add_criterion(
                "monotonicity_%s" % axis,
                ref["spearman"] >= MIN_SPEARMAN
                and ref["monotone_segment_fraction"] >= MIN_MONOTONE_SEGMENT_FRACTION,
                "spearman=%.3f monotone-segment-fraction=%.3f (need >= %.2f / >= %.2f)"
                % (ref["spearman"], ref["monotone_segment_fraction"], MIN_SPEARMAN, MIN_MONOTONE_SEGMENT_FRACTION),
            )
            add_criterion(
                "modulation_%s" % axis,
                ref["relative_modulation"] >= MIN_RELATIVE_MODULATION,
                "relative modulation=%.3f (need >= %.2f)" % (ref["relative_modulation"], MIN_RELATIVE_MODULATION),
            )
            wd = window_dependence[axis]
            add_criterion(
                "window_stability_%s" % axis,
                wd["center_spread_px"] <= MAX_CENTER_WINDOW_SPREAD_PX
                and (np.isnan(wd["sigma_relative_spread"]) or wd["sigma_relative_spread"] <= MAX_SIGMA_RELATIVE_SPREAD),
                "center spread=%.1f px, sigma relative spread=%s (need <= %.1f px / <= %.2f)"
                % (
                    wd["center_spread_px"],
                    "nan" if np.isnan(wd["sigma_relative_spread"]) else "%.3f" % wd["sigma_relative_spread"],
                    MAX_CENTER_WINDOW_SPREAD_PX,
                    MAX_SIGMA_RELATIVE_SPREAD,
                ),
            )
            extent = panel_w if axis == "x" else panel_h
            ss = np.linspace(0, extent, args.n_scan)
            margin = EDGE_MARGIN_FRACTION * (ss[-1] - ss[0])
            c50 = ref["fit"]["c50"]
            inside = bool(ss[0] + margin <= c50 <= ss[-1] - margin)
            add_criterion(
                "crossing_inside_scan_%s" % axis,
                inside,
                "50%% crossing=%.1f px within scan range [%.1f, %.1f] margin %.1f px"
                % (c50, ss[0], ss[-1], margin),
            )

        reliable = all(c["passed"] for c in criteria)
        background_info = _background_info(args.dark_frame)
        gaps = background_info["gaps"]
        if reliable:
            center = [
                curves["y"][str(REFERENCE_WINDOW_FACTOR)]["fit"]["c50"],
                curves["x"][str(REFERENCE_WINDOW_FACTOR)]["fit"]["c50"],
            ]
            sigma = [
                curves["y"][str(REFERENCE_WINDOW_FACTOR)]["fit"]["sigma"],
                curves["x"][str(REFERENCE_WINDOW_FACTOR)]["fit"]["sigma"],
            ]
        else:
            center = None
            sigma = None

        summary = {
            "script": "scripts/slm/measure_slm_beam_center.py",
            "issue": 235,
            "created_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "elapsed_s": time.time() - t0,
            "devices": {
                "slm_number": args.slm_number,
                "cam_id": args.cam_id,
                "cam_type": "daheng",
                "exposure_ms_requested": args.exposure_ms,
                "exposure_ms_readback": exposure_readback,
                "panel_res_wh": [panel_w, panel_h],
            },
            "panel_geometric_center_cy_cx": list(PANEL_GEOMETRIC_CENTER_CY_CX),
            "window_factors": list(WINDOW_FACTORS),
            "reference_window_factor": REFERENCE_WINDOW_FACTOR,
            "ccd_zero_order": {
                "argmax_cy_cx": flat_rec["argmax_cy_cx"],
                "box_centroid_cy_cx": flat_rec["ccd_box_centroid_cy_cx"],
                "box_side_px": flat_rec["ccd_box_side_px"],
                "fwhm_row_px": flat_rec["fwhm_row_px"],
                "fwhm_col_px": flat_rec["fwhm_col_px"],
                "fwhm_geometric_px": flat_rec["fwhm_geometric_px"],
                "peak": flat_rec["peak"],
                "background_median": med_bg,
                "saturated": flat_rec["saturated"],
                "note": (
                    "CCD-side reference evidence only; NOT the SLM panel center. "
                    "The SLM panel center comes from the knife-edge scan (beam_center_cy_cx)."
                ),
            },
            "background": {
                **background_info,
                "flat_median": med_bg,
                "flat_peak_minus_median": peak_sig,
                "background_fraction_ref_window": flat_rec["windows"][str(REFERENCE_WINDOW_FACTOR)]["background_fraction"],
            },
            "curves": _jsonable(curves),
            "window_dependence": _jsonable(window_dependence),
            "criteria": criteria,
            "reliable": reliable,
            "reasons": reasons,
            "gaps": gaps,
            "beam_center_cy_cx": center,
            "beam_sigma": sigma,
            "beam_center_units": (
                "panel px (cy, cx). The apodization patterns are defined on the panel "
                "grid (clear = xx < s / yy < s) and the 50% crossing is the panel "
                "coordinate s at which half the far-field power is cleared; the CCD-side "
                "1-D scale factor cancels in the cumulative power fraction, so no "
                "CCD->panel mapping is needed. The CCD centroid is an independent "
                "reference only, NOT the panel center."
            ),
            "beam_center_null_note": (
                "When reliable=false, beam_center_cy_cx is None and MUST NOT be "
                "used downstream as a calibration value."
            ),
            "frames": _jsonable(frames),
            "failures": failures,
        }

        # --- write artifacts ---------------------------------------------------
        (out / "plan.json").write_text(
            json.dumps(_plan(args), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("wrote summary.json (%d frames, %d failures)" % (len(frames), len(failures)))

        if not reliable:
            print("!" * 78)
            print("WARNING: SLM beam center NOT reliable -- no usable center emitted.")
            print("reasons:")
            for r in reasons:
                print("  - " + r)
            print("Raw frames and per-window curves are saved; re-run with a real exposure (1-10 ms).")
            print("!" * 78)
        else:
            print("beam_center_cy_cx = %s, beam_sigma = %s" % (center, sigma))
        return _exit_code(reliable)
    finally:
        # --- guaranteed release: SLM to uniform gray 0, then close; camera close.
        if slm is not None:
            try:
                slm.set_grayscale(0)
                release["slm_flat_gray0"] = "ok"
            except Exception as exc:
                release["slm_flat_gray0"] = "failed: %s" % exc
                print("slm flat-set warning: %s" % exc, file=sys.stderr)
            try:
                slm.close()
                release["slm_closed"] = "ok"
            except Exception as exc:
                release["slm_closed"] = "failed: %s" % exc
                print("slm close warning: %s" % exc, file=sys.stderr)
        if cam is not None:
            try:
                cam.close()
                release["cam_closed"] = "ok"
            except Exception as exc:
                release["cam_closed"] = "failed: %s" % exc
                print("camera close warning: %s" % exc, file=sys.stderr)
        try:
            lock.release()
        except Exception:
            pass
        print("released: SLM #%d closed, camera closed" % args.slm_number)
        try:
            (out / "release.json").write_text(
                json.dumps(release, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            print("could not write release.json: %s" % exc, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="new output directory (must not exist)")
    ap.add_argument("--slm-number", type=int, default=1)
    ap.add_argument("--cam-id", type=int, default=0)
    ap.add_argument("--exposure-ms", type=float, default=0.05)
    ap.add_argument("--n-scan", type=int, default=25)
    ap.add_argument("--dark-frame", type=Path, help="real dark frame .npy from capture_dark_flat_baselines.py; missing => median gap recorded")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args(argv)

    dark = None
    if args.dark_frame is not None:
        if not args.dark_frame.is_file():
            print(f"ERROR: dark frame not found: {args.dark_frame}", file=sys.stderr)
            return 1
        try:
            dark = np.asarray(np.load(args.dark_frame, allow_pickle=False), dtype=np.float64)
        except (OSError, ValueError) as exc:
            print(f"ERROR: invalid dark frame: {exc}", file=sys.stderr)
            return 1
        if dark.ndim != 2 or not np.isfinite(dark).all():
            print("ERROR: dark frame must be a 2-D array", file=sys.stderr)
            return 1
    args.dark = dark
    plan = _plan(args)
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
