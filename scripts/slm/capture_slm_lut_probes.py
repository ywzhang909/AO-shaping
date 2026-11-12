"""Capture LUT-identifiability and spatial-crosstalk probes for issue #238.

Issue #238 symptom: a uniform gray sweep produces only piston (measured
``spot_power`` 5.2-5.8e4 on 2026-11-10, +-5% variation, no structural response),
and a 200 px square probe response is weak (sq512 down 10%). The hypothesis is
that the SLM's gray -> phase LUT is not behaving as assumed: uniform gray is
piston (invisible in the far field) and the spatial response does not match the
commanded pattern. This script captures *physical evidence* for that:

1. **Gray sweep** (``set_grayscale``, 10-bit): uniform gray in {0,64,...,960,1023}
   records the 0-order window power, frame sum, max, saturation flag and one
   cropped frame per level. This is the raw LUT-identifiability dataset: piston
   is invisible, so a *flat* response is expected for a phase-only panel, and any
   amplitude/structure dependence exposes an amplitude (non-phase-only) coupling
   or a broken modulation path.

2. **Multi-carrier blaze probes** (``display_phase``): a phase ramp/grating at
   direction {0,45,90,135} deg x period {8,16,32,64} panel px x gray amplitude
   {256,512,768} records the 0-order window power, per-diffraction-order
   energies, frame sum and one cropped frame per probe. A working gray->phase
   LUT drives energy into the +1 order as the ramp depth approaches 2*pi; the
   E0/E1 ratio is the ``zero_order_suppressed`` estimate.

3. **Spatial squares** (``display_phase``): a 200 px square at the panel center
   and at four corners, gray {256,512,768,1023}, records the 0-order window
   power, frame sum, max, saturation flag and one cropped frame per probe. This
   is the spatial-crosstalk dataset: a localized phase region should produce
   only a localized (edge) response, not a whole-panel change.

All frames are captured at 0.05 ms (non-saturating per the task spec; bench
notes show the usable window is 1-10 ms, so at the default the data is faint but
valid evidence). Every pattern records the sha256 of the uint16 gray phase from
``get_displayed_phase`` plus a provenance string.

**Actual gray availability**:
- Gray sweep runs in ``set_grayscale`` mode, where ``get_current_grayscale()``
  is a genuine device readback of the panel gray -> ``actual_gray_available=true``.
- Phase patterns (carriers, squares) use ``display_phase``; ``get_displayed_phase``
  returns the **driver-side write cache**, not a device readback (the Santec SDK
  cannot read panel memory), so ``actual_gray_available=false`` and only the hash
  of the commanded/cached gray is recorded. No panel gray is ever fabricated.

Plan-only by default: no ``--execute`` prints the plan as JSON and exits 0 before
creating ``--out`` or touching any device. ``--out`` must be a directory that does
not yet exist (``exist_ok=False``).

Usage (plan-only, then hardware)::

    python scripts/slm/capture_slm_lut_probes.py --out <new-dir>
    python scripts/slm/capture_slm_lut_probes.py --out <new-dir> --execute

Devices are acquired under ``DeviceLock`` (bench mutex). The SLM is guaranteed to
be reset to uniform gray 0 and closed, and the camera closed, in ``finally`` even
when a pattern fails. Per-frame failures are collected and reported; they do not
abort the session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

#: Panel full-scale gray (10-bit, 0..1023). 1024 wraps to 0 on this device.
MAX_GRAYSCALE = 1023
#: Uniform-gray sweep levels (10-bit). 960 is the last multiple of 64 below 1024.
GRAY_SWEEP = tuple(range(0, 1024, 64)) + (1023,)
#: Blaze (phase ramp) directions in degrees.
CARRIER_DIRECTIONS = (0, 45, 90, 135)
#: Blaze periods in panel px.
CARRIER_PERIODS = (8, 16, 32, 64)
#: Blaze gray amplitudes (peak-to-peak phase depth in gray units; <1023 = sub-2*pi).
CARRIER_GRAY_AMPS = (256, 512, 768)
#: Spatial square side (panel px).
SQUARE_SIDE = 200
#: Offsets from a measured PANEL beam center, never from the CCD zero order.
SQUARE_OFFSETS_CX_CY = ((0, 0), (-200, -200), (200, -200), (-200, 200), (200, 200))
#: Square gray levels (10-bit).
SQUARE_GRAYS = (256, 512, 768, 1023)
#: 0-order window side for the E0 of the zero_order_suppressed estimate (px).
BLAZE_ZERO_ORDER_WINDOW = 24
#: Side-order search window side (px).
SIDE_ORDER_WINDOW = 24
#: Max distance to search for a side order from the 0-order center (px).
SIDE_ORDER_SEARCH_PX = 1000
#: A side-order peak must hold >= this fraction of E0 to count as resolved.
SIDE_ORDER_MIN_FRACTION = 0.05
#: E0/E1 at/below which the 0-order is considered suppressed.
ZERO_ORDER_SUPPRESSED_MAX_RATIO = 0.5
#: Saved crop side centered on the 0-order (px).
CROP_SIDE = 1100
#: Side-order peak must exceed this multiple of the scan-line median to be a real bump.
PEAK_VS_MEDIAN = 1.5
#: Panel->CCD geometry calibration (rotation_deg / mirror / center), produced by
#: scripts/slm/calib_geometry_from_orders.py. If this file is missing the script
#: refuses to run rather than silently assuming an identity panel->CCD mapping.
DEFAULT_GEOMETRY_CALIB = Path(__file__).resolve().parents[2] / "data/slm_geometry_calib.npz"
#: Max fraction of blaze probes allowed to end up with status "unknown" (side
#: order not resolved) before the whole run is judged unreliable. Rationale: the
#: E0/E1 suppression estimate is the primary LUT-identifiability evidence; if
#: more than half of the probes cannot resolve a side order, the dataset cannot
#: answer issue #238's core question and downstream conclusions must not be drawn.
MAX_UNRESOLVED_SIDE_ORDER_FRACTION = 0.5


def _panel_direction_to_ccd(direction_deg: float, rotation_deg: float, mirror: bool) -> tuple[float, float]:
    """Return CCD (row, col) unit vector; mirror reflects panel x BEFORE rotation.

    atan2(d_row, d_col) = panel angle + rotation (orders calibration lines 203-206).
    Mirror is x reflection, consistent with calibration.py's x-mirror declaration.
    """
    th = np.deg2rad(direction_deg)
    x, y = float(np.cos(th)), float(np.sin(th))
    if mirror:
        x = -x
    rot = np.deg2rad(rotation_deg)
    return float(x * np.sin(rot) + y * np.cos(rot)), float(x * np.cos(rot) - y * np.sin(rot))


def _load_geometry(path: Path) -> dict:
    path = path.resolve(strict=True)
    with np.load(path, allow_pickle=False) as data:
        rotation = float(data["rotation_deg"])
        mirror = bool(data["mirror"])
        center = np.asarray(data["center"], dtype=float)
        if center.shape != (2,) or not np.isfinite(center).all() or not np.isfinite(rotation):
            raise ValueError("geometry rotation/center must be finite, center must be CCD (row,col)")
        provenance = json.loads(str(data["provenance"])) if "provenance" in data else {}
    return {
        "path": str(path), "mtime_ns": path.stat().st_mtime_ns,
        "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime)),
        "rotation_deg": rotation, "mirror": mirror, "center_ccd_cy_cx": center.tolist(),
        "calibration_date": provenance.get("measured_at", "unknown"),
        "status": "historical calibration; 需要实验复核; not verified after 2026-11-10 adjustment",
        "mapping": "reflect panel x if mirror, then rotate by rotation_deg; CCD=(row,col)",
        "directions": [{"panel_direction_deg": d, "ccd_unit_dcy_dcx": list(_panel_direction_to_ccd(d, rotation, mirror))} for d in CARRIER_DIRECTIONS],
    }


def _square_layout(path: Path | None, panel_w: int, panel_h: int) -> tuple[list, list[str]]:
    # A far-field zero-order CCD position does NOT identify the panel illumination
    # center: translation of a pupil leaves Fourier intensity unchanged. A separate
    # panel-grid knife-edge measurement is required; never reuse CCD pixels here.
    if path is None:
        return [], ["Panel beam center unknown: flat/gray CCD zero order cannot locate illumination on the panel; square coverage unverified, squares skipped. Supply a reliable #235 summary via --beam-center-summary."]
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("reliable") is not True or data.get("beam_center_cy_cx") is None:
        return [], ["Beam-center summary is unreliable/null; squares skipped (coverage gap)."]
    cy, cx = map(float, data["beam_center_cy_cx"])
    if not np.isfinite([cy, cx]).all():
        return [], ["Nonfinite panel beam center; squares skipped."]
    locations = [(int(round(cx + dx)), int(round(cy + dy))) for dx, dy in SQUARE_OFFSETS_CX_CY]
    valid = [(x, y) for x, y in locations if SQUARE_SIDE // 2 <= x <= panel_w - SQUARE_SIDE // 2 and SQUARE_SIDE // 2 <= y <= panel_h - SQUARE_SIDE // 2]
    gaps = [] if len(valid) == len(locations) else ["Some beam-relative square positions do not fit on the panel and were skipped."]
    return valid, gaps


def _assess_reliability(failures: list, gray_records: list, estimates: dict, gaps: list[str]) -> tuple[bool, list[str]]:
    reasons = list(gaps)
    if failures:
        reasons.append("One or more acquisition/probe failures occurred.")
    if any(r.get("gray_readback_matches_commanded") is False for r in gray_records):
        reasons.append("Gray readback does not match commanded gray.")
    if any(not r.get("gray_readback_verified", False) for r in gray_records):
        reasons.append("Gray readback unavailable/unverified.")
    fraction = sum(v["status"] != "ok" for v in estimates.values()) / len(estimates) if estimates else 1.0
    if fraction > MAX_UNRESOLVED_SIDE_ORDER_FRACTION:
        reasons.append(f"Unresolved side-order fraction {fraction:.3f} exceeds {MAX_UNRESOLVED_SIDE_ORDER_FRACTION:.3f}.")
    return not reasons, reasons


def _exit_code(reliable: bool) -> int:
    return 0 if reliable else 1


def _window_slices(cy: int, cx: int, side: int, h: int, w: int):
    half = max(1, side // 2)
    y0 = max(0, cy - half)
    y1 = min(h, cy + half)
    x0 = max(0, cx - half)
    x1 = min(w, cx + half)
    return y0, y1, x0, x1


def _window_power(frame: np.ndarray, cy: int, cx: int, side: int) -> tuple[float, tuple]:
    """Sum of ``frame`` in the ``side x side`` window centered at (cy, cx)."""
    h, w = frame.shape
    y0, y1, x0, x1 = _window_slices(cy, cx, side, h, w)
    return float(frame[y0:y1, x0:x1].sum()), (y0, y1, x0, x1)


def _crop(frame: np.ndarray, cy: int, cx: int, side: int) -> tuple[np.ndarray, tuple]:
    """Clipped ``side x side`` crop centered at (cy, cx); returns (crop, bounds)."""
    y0, y1, x0, x1 = _window_slices(cy, cx, side, frame.shape[0], frame.shape[1])
    return frame[y0:y1, x0:x1], (y0, y1, x0, x1)


def _fwhm1d(profile: np.ndarray) -> float:
    """FWHM (px) of a 1-D peak profile at half max (index-based, matches calibration)."""
    p = np.asarray(profile, dtype=np.float64)
    if p.size < 2:
        return 0.0
    peak = float(p.max())
    if peak <= 0.0:
        return 0.0
    above = np.nonzero(p >= 0.5 * peak)[0]
    return float(above[-1] - above[0] + 1) if len(above) >= 2 else 0.0


def _sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _jsonable(obj):
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


def _displayed_provenance(slm) -> dict:
    """Commanded/cached gray hash + provenance; never fabricate a device readback."""
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


def _gray_sweep_provenance(slm, commanded_gray: int) -> dict:
    """Gray-sweep provenance: true device readback via get_current_grayscale().

    The readback is compared against the commanded gray: a mismatch means the
    panel is not showing what was commanded and ``actual_gray_available`` must
    NOT be treated as trustworthy (``gray_readback_matches_commanded=false``).
    """
    try:
        readback = slm.get_current_grayscale()
        actual = readback is not None
        note = "device readback via get_current_grayscale()"
    except Exception as exc:
        readback = None
        actual = False
        note = f"get_current_grayscale() failed: {exc}"
    delta = None
    matches = False
    if readback is not None:
        delta = int(readback) - int(commanded_gray)
        matches = bool(delta == 0)
        if not matches:
            note += (
                "; MISMATCH: readback %d != commanded %d (|delta|=%d) -- "
                "actual_gray_available is reported but the commanded value was "
                "NOT confirmed on the panel"
                % (int(readback), int(commanded_gray), abs(delta))
            )
    prov = _displayed_provenance(slm)
    return {
        "actual_gray_available": actual,
        "commanded_gray": int(commanded_gray),
        "gray_readback": readback,
        "gray_readback_matches_commanded": matches,
        "gray_readback_delta": delta,
        "gray_readback_abs_delta": abs(delta) if delta is not None else None,
        "gray_readback_verified": bool(matches),
        "displayed_phase_sha256": prov["displayed_phase_sha256"],
        "displayed_gray_source": prov["displayed_gray_source"],
        "gray_readback_note": note,
    }


def _blaze_phase(h: int, w: int, direction_deg: float, period: float, gray_amp: int) -> np.ndarray:
    """Blazed phase ramp: depth = gray_amp/1023 * 2*pi over one period along direction."""
    th = np.deg2rad(direction_deg)
    ux, uy = np.cos(th), np.sin(th)
    yy, xx = np.mgrid[0:h, 0:w]
    coord = xx * ux + yy * uy
    depth = (gray_amp / MAX_GRAYSCALE) * 2.0 * np.pi
    return (depth * (coord / period)).astype(np.float32)


def _square_phase(h: int, w: int, cx: int, cy: int, side: int, gray: int) -> np.ndarray:
    """Square piston region of depth gray/1023 * 2*pi; zero outside (raw radians)."""
    phase = np.zeros((h, w), dtype=np.float32)
    y0 = max(0, cy - side // 2)
    y1 = min(h, cy + side // 2)
    x0 = max(0, cx - side // 2)
    x1 = min(w, cx + side // 2)
    phase[y0:y1, x0:x1] = (gray / MAX_GRAYSCALE) * 2.0 * np.pi
    return phase


def _find_diffraction_orders(bs, cy, cx, dcy, dcx, window_side, max_search) -> list[dict]:
    """Scan along +-u for side-order energy peaks (local maxima of the line profile).

    ``(dcy, dcx)`` is the CCD-side direction unit vector (row, col) of the blaze.
    Returns a list of found orders, each ``{sign, distance_px, energy, center_cy_cx}``.
    """
    h, w = bs.shape
    start = max(2, int(round(window_side / 2)) + 2)
    orders: list[dict] = []
    for sign in (1.0, -1.0):
        profile = []
        for d in range(start, max_search):
            py = int(round(cy + sign * d * dcy))
            px = int(round(cx + sign * d * dcx))
            if not (0 <= py < h and 0 <= px < w):
                break
            p = _window_power(bs, py, px, window_side)[0]
            profile.append((float(d), p, py, px))
        if len(profile) < 5:
            continue
        energies = np.array([pt[1] for pt in profile], dtype=np.float64)
        med = float(np.median(energies)) if energies.size else 0.0
        for i in range(2, len(profile) - 2):
            d0, p0, py, px = profile[i]
            is_peak = p0 >= profile[i - 1][1] and p0 >= profile[i + 1][1]
            is_bump = p0 >= PEAK_VS_MEDIAN * max(med, 1e-12)
            if is_peak and is_bump:
                orders.append({
                    "sign": int(sign),
                    "distance_px": d0,
                    "energy": p0,
                    "center_cy_cx": [float(py), float(px)],
                })
    return orders


def _estimate_zero_order_suppressed(bs, cy, cx, dcy, dcx, e0_window, side_window, max_search) -> dict:
    """E0/E1 suppression estimate; marks ``status: unknown`` when side orders are not resolved."""
    e0 = _window_power(bs, cy, cx, e0_window)[0]
    orders = _find_diffraction_orders(bs, cy, cx, dcy, dcx, side_window, max_search)
    e1 = max((o["energy"] for o in orders), default=0.0)
    if e1 <= 0.0 or e1 < SIDE_ORDER_MIN_FRACTION * e0:
        return {
            "status": "unknown",
            "reason": "side order not resolved (no peak above %.2f of E0)" % SIDE_ORDER_MIN_FRACTION,
            "e0": e0,
            "e1": None,
            "ratio_e0_e1": None,
            "suppressed": None,
            "orders": orders,
        }
    ratio = e0 / e1
    return {
        "status": "ok",
        "e0": e0,
        "e1": e1,
        "ratio_e0_e1": ratio,
        "suppressed": bool(ratio <= ZERO_ORDER_SUPPRESSED_MAX_RATIO),
        "orders": orders,
    }


def _plan(args: argparse.Namespace) -> dict:
    n_carrier = len(CARRIER_DIRECTIONS) * len(CARRIER_PERIODS) * len(CARRIER_GRAY_AMPS)
    n_square = len(SQUARE_OFFSETS_CX_CY) * len(SQUARE_GRAYS)
    return {
        "plan_only": True,
        "script": "scripts/slm/capture_slm_lut_probes.py",
        "issue": 238,
        "devices": {
            "slm": args.slm_number,
            "slm_video_mode": 0,
            "camera": "daheng",
            "cam_id": args.cam_id,
            "exposure_ms_requested": args.exposure_ms,
        },
        "action": "gray sweep + multi-carrier blaze probes + spatial square probes",
        "n_flat": 1,
        "n_gray_levels": len(GRAY_SWEEP),
        "n_carrier_probes": n_carrier,
        "n_square_probes": n_square,
        "n_patterns_total": 1 + len(GRAY_SWEEP) + n_carrier + n_square,
        "gray_sweep": list(GRAY_SWEEP),
        "carrier_directions": list(CARRIER_DIRECTIONS),
        "carrier_periods": list(CARRIER_PERIODS),
        "carrier_gray_amps": list(CARRIER_GRAY_AMPS),
        "square_side_px": SQUARE_SIDE,
        "square_offsets_cx_cy": [list(loc) for loc in SQUARE_OFFSETS_CX_CY],
        "square_anchor": "reliable measured panel beam center; otherwise skip with gap (CCD zero order is NOT panel center)",
        "beam_center_summary": str(args.beam_center_summary) if args.beam_center_summary else None,
        "geometry_calibration": _load_geometry(args.geometry_calib),
        "square_grays": list(SQUARE_GRAYS),
        "blaze_zero_order_window_px": BLAZE_ZERO_ORDER_WINDOW,
        "side_order_window_px": SIDE_ORDER_WINDOW,
        "side_order_search_px": SIDE_ORDER_SEARCH_PX,
        "crop_side_px": CROP_SIDE,
        "thresholds": {
            "side_order_min_fraction": SIDE_ORDER_MIN_FRACTION,
            "zero_order_suppressed_max_ratio": ZERO_ORDER_SUPPRESSED_MAX_RATIO,
            "peak_vs_median": PEAK_VS_MEDIAN,
            "max_unresolved_side_order_fraction": MAX_UNRESOLVED_SIDE_ORDER_FRACTION,
        },
        "writes": [
            "summary.json",
            "plan.json",
            "frames/flat.npy",
            "frames/gray_<level>.npy",
            "frames/carrier_<dir>_<period>_<amp>.npy",
            "frames/square_<cx>_<cy>_<gray>.npy",
        ],
        "notes": [
            "Uniform gray is piston (invisible in the far field); a flat gray-sweep response is expected for a phase-only panel.",
            "Default exposure 0.05 ms is non-saturating but sits at this bench's read-noise floor; data is faint but valid evidence.",
            "actual_gray_available=true only in the gray sweep (get_current_grayscale readback); phase patterns are false (get_displayed_phase is a driver-side write cache, not a device readback).",
        ],
    }


def _capture_frame(cam, cy, cx) -> dict:
    """Grab one frame, subtract background, compute 0-order windows; returns record + raw + crop."""
    from ao_shaping.drivers.ccd.common import is_saturated

    raw = np.asarray(cam.get_numpy_image(1, True), dtype=np.float64)
    med = float(np.median(raw))
    bs = np.maximum(raw - med, 0.0)
    saturated = bool(is_saturated(raw))
    return {
        "background_median": med,
        "frame_max": float(raw.max()),
        "frame_sum": float(raw.sum()),
        "saturated": saturated,
    }, raw, bs


def _run(args: argparse.Namespace) -> int:
    # Lazy device imports: the ONLY place drivers are loaded.
    from ao_shaping.drivers.ccd.common import (
        create_camera,
        get_camera_exposure_ms,
        set_camera_exposure_ms,
    )
    from ao_shaping.drivers.slm.factory import require_drive_board_entry
    from ao_shaping.drivers.slm.santec import Santec
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

    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    (out / "frames").mkdir(parents=True, exist_ok=False)
    failures: list[dict] = []
    release: dict = {"slm_flat_gray0": "not-attempted", "slm_closed": "not-attempted", "cam_closed": "not-attempted"}

    slm = None
    cam = None
    t0 = time.time()
    try:
        prepare_camera_sdk("daheng")
        slm = Santec(slm_number=args.slm_number, video_mode=0)
        slm.open()
        require_drive_board_entry(slm)
        print("SLM #%d open (drive board OK)" % args.slm_number)

        cam = create_camera("daheng", args.cam_id, exposure_time_ms=args.exposure_ms)
        cam.open()
        set_camera_exposure_ms(cam, args.exposure_ms)
        exposure_readback = get_camera_exposure_ms(cam)
        print("camera open, exposure readback %.4f ms" % exposure_readback)

        panel_h, panel_w = slm.Panel_Res[1], slm.Panel_Res[0]

        # ---- flat reference: uniform gray 0 -> 0-order position + FWHM ---------
        slm.set_grayscale(0)
        flat_rec, raw_flat, bs_flat = _capture_frame(cam, 0, 0)
        cy, cx = map(int, np.unravel_index(np.argmax(bs_flat), bs_flat.shape))
        fwhm_row = _fwhm1d(bs_flat[cy, :])
        fwhm_col = _fwhm1d(bs_flat[:, cx])
        fwhm = 0.5 * (fwhm_row + fwhm_col)
        ref_window = max(BLAZE_ZERO_ORDER_WINDOW, int(round(2.0 * fwhm)))
        zero_power, _ = _window_power(bs_flat, cy, cx, ref_window)
        prov = _gray_sweep_provenance(slm, 0)
        crop_flat, flat_bounds = _crop(raw_flat, cy, cx, CROP_SIDE)
        np.save(out / "frames" / "flat.npy", crop_flat)
        flat_info = {
            "kind": "flat",
            "frame_file": "frames/flat.npy",
            "zero_order_cy_cx": [float(cy), float(cx)],
            "fwhm_row_px": fwhm_row,
            "fwhm_col_px": fwhm_col,
            "fwhm_geometric_px": float(np.hypot(fwhm_row, fwhm_col)),
            "reference_window_px": ref_window,
            "zero_order_power": zero_power,
            "crop_side_px": CROP_SIDE,
            "crop_bounds": list(flat_bounds),
            "background_median": flat_rec["background_median"],
            "frame_max": flat_rec["frame_max"],
            "saturated": flat_rec["saturated"],
            "actual_gray_available": prov["actual_gray_available"],
            "commanded_gray": 0,
            "gray_readback": prov["gray_readback"],
            **{k: v for k, v in prov.items() if k.startswith("gray_readback_")},
            "displayed_phase_sha256": prov["displayed_phase_sha256"],
            "displayed_gray_source": prov["displayed_gray_source"],
        }
        print("flat: 0-order (cy,cx)=(%d,%d), FWHM %.1f px" % (cy, cx, fwhm))

        gray_sweep: list[dict] = []
        for g in GRAY_SWEEP:
            rec_name = "frames/gray_%d.npy" % g
            try:
                slm.set_grayscale(g)
                rec, raw, bs = _capture_frame(cam, cy, cx)
                p24, _ = _window_power(bs, cy, cx, BLAZE_ZERO_ORDER_WINDOW)
                pw, _ = _window_power(bs, cy, cx, ref_window)
                crop, bounds = _crop(raw, cy, cx, CROP_SIDE)
                np.save(out / rec_name, crop)
                prov = _gray_sweep_provenance(slm, g)
                gray_sweep.append({
                    "kind": "gray_sweep",
                    "gray": g,
                    "frame_file": rec_name,
                    "crop_bounds": list(bounds),
                    "zero_order_power_24px": p24,
                    "zero_order_power_refwindow": pw,
                    "reference_window_px": ref_window,
                    "background_median": rec["background_median"],
                    "frame_max": rec["frame_max"],
                    "frame_sum": rec["frame_sum"],
                    "saturated": rec["saturated"],
                    "actual_gray_available": prov["actual_gray_available"],
                    "commanded_gray": g,
                    "gray_readback": prov["gray_readback"],
            **{k: v for k, v in prov.items() if k.startswith("gray_readback_")},
                    "displayed_phase_sha256": prov["displayed_phase_sha256"],
                    "displayed_gray_source": prov["displayed_gray_source"],
                    "gray_readback_note": prov["gray_readback_note"],
                })
            except Exception as exc:
                failures.append({"stage": "gray_sweep", "gray": g, "error": str(exc)})
                print("FAILED gray=%d: %s" % (g, exc), file=sys.stderr)
        print("gray sweep done (%d levels)" % len(gray_sweep))

        carriers: list[dict] = []
        suppressed_summary: dict = {}
        for direction in CARRIER_DIRECTIONS:
            geometry = args.geometry
            dcy, dcx = _panel_direction_to_ccd(direction, geometry["rotation_deg"], geometry["mirror"])
            for period in CARRIER_PERIODS:
                for gray_amp in CARRIER_GRAY_AMPS:
                    rec_name = "frames/carrier_%d_%d_%d.npy" % (direction, period, gray_amp)
                    key = "d%d_p%d_a%d" % (direction, period, gray_amp)
                    try:
                        phase = _blaze_phase(panel_h, panel_w, direction, period, gray_amp)
                        slm.display_phase(phase)
                        rec, raw, bs = _capture_frame(cam, cy, cx)
                        p24, _ = _window_power(bs, cy, cx, BLAZE_ZERO_ORDER_WINDOW)
                        pw, _ = _window_power(bs, cy, cx, ref_window)
                        est = _estimate_zero_order_suppressed(
                            bs, cy, cx, dcy, dcx,
                            BLAZE_ZERO_ORDER_WINDOW, SIDE_ORDER_WINDOW, SIDE_ORDER_SEARCH_PX,
                        )
                        crop, bounds = _crop(raw, cy, cx, CROP_SIDE)
                        np.save(out / rec_name, crop)
                        prov = _displayed_provenance(slm)
                        carriers.append({
                            "kind": "carrier",
                            "direction_deg": direction,
                            "ccd_unit_dcy_dcx": [dcy, dcx],
                            "period_px": period,
                            "gray_amp": gray_amp,
                            "frame_file": rec_name,
                            "crop_bounds": list(bounds),
                            "zero_order_power_24px": p24,
                            "zero_order_power_refwindow": pw,
                            "reference_window_px": ref_window,
                            "background_median": rec["background_median"],
                            "frame_max": rec["frame_max"],
                            "frame_sum": rec["frame_sum"],
                            "saturated": rec["saturated"],
                            "zero_order_suppressed": est,
                            "actual_gray_available": prov["actual_gray_available"],
                            "displayed_phase_sha256": prov["displayed_phase_sha256"],
                            "displayed_gray_source": prov["displayed_gray_source"],
                            "gray_readback_note": prov["gray_readback_note"],
                        })
                        suppressed_summary[key] = {
                            "direction_deg": direction,
                            "ccd_unit_dcy_dcx": [dcy, dcx],
                            "period_px": period,
                            "gray_amp": gray_amp,
                            "status": est["status"],
                            "ratio_e0_e1": est["ratio_e0_e1"],
                            "suppressed": est["suppressed"],
                            "e0": est["e0"],
                            "e1": est["e1"],
                        }
                    except Exception as exc:
                        failures.append({"stage": "carrier", "key": key, "error": str(exc)})
                        print("FAILED carrier %s: %s" % (key, exc), file=sys.stderr)
        print("carrier probes done (%d)" % len(carriers))

        squares: list[dict] = []
        square_locations, gaps = _square_layout(args.beam_center_summary, panel_w, panel_h)
        for cx_sq, cy_sq in square_locations:
            for gray in SQUARE_GRAYS:
                rec_name = "frames/square_%d_%d_%d.npy" % (cx_sq, cy_sq, gray)
                try:
                    phase = _square_phase(panel_h, panel_w, cx_sq, cy_sq, SQUARE_SIDE, gray)
                    slm.display_phase(phase)
                    rec, raw, bs = _capture_frame(cam, cy, cx)
                    p24, _ = _window_power(bs, cy, cx, BLAZE_ZERO_ORDER_WINDOW)
                    pw, _ = _window_power(bs, cy, cx, ref_window)
                    crop, bounds = _crop(raw, cy, cx, CROP_SIDE)
                    np.save(out / rec_name, crop)
                    prov = _displayed_provenance(slm)
                    squares.append({
                        "kind": "square",
                        "square_center_cx_cy": [cx_sq, cy_sq],
                        "square_side_px": SQUARE_SIDE,
                        "gray": gray,
                        "frame_file": rec_name,
                        "crop_bounds": list(bounds),
                        "zero_order_power_24px": p24,
                        "zero_order_power_refwindow": pw,
                        "reference_window_px": ref_window,
                        "background_median": rec["background_median"],
                        "frame_max": rec["frame_max"],
                        "frame_sum": rec["frame_sum"],
                        "saturated": rec["saturated"],
                        "actual_gray_available": prov["actual_gray_available"],
                        "displayed_phase_sha256": prov["displayed_phase_sha256"],
                        "displayed_gray_source": prov["displayed_gray_source"],
                        "gray_readback_note": prov["gray_readback_note"],
                    })
                except Exception as exc:
                    failures.append({
                        "stage": "square", "cx": cx_sq, "cy": cy_sq, "gray": gray, "error": str(exc),
                    })
                    print("FAILED square cx=%d cy=%d gray=%d: %s" % (cx_sq, cy_sq, gray, exc), file=sys.stderr)
        print("square probes done (%d)" % len(squares))

        # ---- consolidated suppression summary ---------------------------------
        n_ok = sum(1 for v in suppressed_summary.values() if v["status"] == "ok")
        n_unknown = sum(1 for v in suppressed_summary.values() if v["status"] == "unknown")
        n_suppressed = sum(1 for v in suppressed_summary.values() if v.get("suppressed") is True)

        reliable, reasons = _assess_reliability(failures, [flat_info, *gray_sweep], suppressed_summary, gaps)
        summary = {
            "reliable": reliable, "reasons": reasons, "gaps": gaps,
            "geometry_calibration": args.geometry,
            "square_anchor_summary": str(args.beam_center_summary) if args.beam_center_summary else None,
            "script": "scripts/slm/capture_slm_lut_probes.py",
            "issue": 238,
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
            "flat": flat_info,
            "reference_window_px": ref_window,
            "gray_sweep": _jsonable(gray_sweep),
            "carriers": _jsonable(carriers),
            "squares": _jsonable(squares),
            "zero_order_suppressed": _jsonable(suppressed_summary),
            "zero_order_suppressed_summary": {
                "n_probes": len(suppressed_summary),
                "n_ok": n_ok,
                "n_unknown": n_unknown,
                "n_suppressed": n_suppressed,
                "max_ratio_e0_e1_ok": max(
                    (v["ratio_e0_e1"] for v in suppressed_summary.values() if v["status"] == "ok"),
                    default=None,
                ),
            },
            "failures": _jsonable(failures),
        }

        (out / "plan.json").write_text(
            json.dumps(_plan(args), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            "wrote summary.json (%d gray, %d carriers, %d squares, %d failures)"
            % (len(gray_sweep), len(carriers), len(squares), len(failures))
        )
        print(
            "zero_order_suppressed: %d/%d probes resolved (E0/E1 ratio known), %d suppressed"
            % (n_ok, len(suppressed_summary), n_suppressed)
        )
        if n_unknown:
            print("  %d probe(s) could not resolve a side order -> status unknown" % n_unknown)
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
    ap.add_argument("--geometry-calib", type=Path, default=DEFAULT_GEOMETRY_CALIB)
    ap.add_argument("--beam-center-summary", type=Path, help="reliable #235 panel-grid measurement; otherwise squares skipped")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args(argv)

    try:
        args.geometry = _load_geometry(args.geometry_calib)
        if args.beam_center_summary is not None and not args.beam_center_summary.is_file():
            raise FileNotFoundError(args.beam_center_summary)
        plan = _plan(args)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: invalid/missing calibration input: {exc}", file=sys.stderr)
        return 1
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
