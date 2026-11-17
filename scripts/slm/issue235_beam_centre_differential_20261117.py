"""Issue #235: differential knife-edge beam centre on the SLM panel (hardware).

Why not the plain knife edge
----------------------------
``find_beam_on_slm`` clears one side of the panel and turns the other side into
a 0/pi checker, assuming the checker scatters *all* blocked light out of the
0-order window. ``data/debug/knife_block_20261111*`` measured how much actually
stays: ``retained_fraction`` = 0.41-0.43 for every checker period (and 0.59 for
blaze, 0.61-0.65 for random phase). So the knife curve keeps a pedestal of
~40-60% of the flat power no matter where the split sits, the 50% crossing is
not the beam centre, and the curve need not be monotonic.

What this script measures instead
---------------------------------
For every split ``s`` on an axis, capture **both** complements:

* ``A``: panel coordinate ``< s`` clear, the rest checkerboard;
* ``B``: panel coordinate ``< s`` checkerboard, the rest clear.

Whatever light the blocking pattern fails to remove is common to A and B (same
pattern area, same geometry), so it cancels in ``D(s) = P_A(s) - P_B(s)``.
``D`` is an odd function of ``s`` about the beam centre and is zero *at* that
centre, independent of the pedestal, of the exposure and of the window:

    D(s) = amp * erf((s - c) / (sqrt(2) * sigma))

``c`` is the beam centre on the panel in panel pixels, ``sigma`` its width. The
same free-baseline erf fit on ``P_A`` alone is reported alongside as a
cross-check (it assumes a *constant* pedestal, which the differential form does
not).

Device use: SLM #1 (Santec, memory mode) and the Daheng camera (cam_id 0).
Plan-only unless ``--execute``; ``--out`` must not exist. The SLM and camera are
closed and the bench lock released in ``finally``; ``released.json`` records it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf

from ao_shaping.drivers.ccd.common import (
    create_camera,
    full_scale,
    set_camera_exposure_ms,
)
from ao_shaping.drivers.slm.santec import Santec
from ao_shaping.tools.kernels import device_mutex
from ao_shaping.tools.kernels.device_mutex import DeviceLock

CHECKER_PERIOD = 8
BOX_SIDE = 120
SETTLE_S = 0.2


def _wait_for_devices(args) -> DeviceLock:
    """Block until the bench is free, then hand back the held lock."""
    verdict = device_mutex.wait_until_idle(
        float(args.wait_until_idle),
        slm_number=int(args.slm_number),
        cam_id=int(args.cam_id),
        cam_type="daheng",
        exposure_ms=float(args.exposure_ms),
        own_paths=[Path(__file__).resolve()],
    )
    if not verdict.ok:
        raise SystemExit(f"devices never became idle: {getattr(verdict, 'reason', verdict)}")
    lock = verdict.lock if verdict.lock is not None else DeviceLock.default()
    if not lock.is_locked():
        lock.acquire(timeout=0.0)
    return lock


def _unsaturated_exposure(cam, requested_ms: float, saturation: float, max_steps: int = 8) -> tuple[float, float]:
    """Halve the exposure until the flat field stops clipping (only ever down)."""
    ms = float(requested_ms)
    for _ in range(max(1, max_steps)):
        set_camera_exposure_ms(cam, ms)
        frame = np.asarray(cam.get_numpy_image(5))
        peak = float(np.max(frame))
        full = float(full_scale(frame, cam))
        if peak < saturation * full:
            return ms, peak
        ms /= 2.0
    raise SystemExit(f"flat field still saturated at {peak:.0f}/{full:.0f} counts down to {ms:.4f} ms")


def _fwhm1d(profile: np.ndarray) -> float:
    p = np.asarray(profile, dtype=np.float64)
    if p.size == 0 or p.max() <= 0:
        return 0.0
    half = 0.5 * float(p.max())
    above = np.flatnonzero(p >= half)
    if above.size < 2:
        return 0.0
    return float(above[-1] - above[0] + 1)


def _box_centroid(frame: np.ndarray, cy: int, cx: int, side: int) -> tuple[float, float]:
    h, w = frame.shape
    half = max(1, side // 2)
    y0, y1 = max(0, cy - half), min(h, cy + half)
    x0, x1 = max(0, cx - half), min(w, cx + half)
    box = np.maximum(np.asarray(frame[y0:y1, x0:x1], dtype=np.float64), 0.0)
    thr = float(box.max()) * 0.15
    m = box >= thr
    if int(m.sum()) < 5:
        return float(cy), float(cx)
    yy, xx = np.nonzero(m)
    wv = box[m]
    return float((yy * wv).sum() / wv.sum()) + y0, float((xx * wv).sum() / wv.sum()) + x0


def _window(frame: np.ndarray, cy: int, cx: int, side: int) -> float:
    h, w = frame.shape
    half = max(1, side // 2)
    y0, y1 = max(0, cy - half), min(h, cy + half)
    x0, x1 = max(0, cx - half), min(w, cx + half)
    return float(np.asarray(frame[y0:y1, x0:x1], dtype=np.float64).sum())


def _knife_pattern(s: int, axis: str, clear_low: bool, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    coord = xx if axis == "x" else yy
    low = coord < s
    clear = low if clear_low else ~low
    checker = ((xx // CHECKER_PERIOD + yy // CHECKER_PERIOD) % 2) * np.pi
    return np.where(clear, 0.0, checker).astype(np.float32)


def _erf_model(s: np.ndarray, amp: float, centre: float, sigma: float) -> np.ndarray:
    return amp * erf((s - centre) / (np.sqrt(2.0) * sigma))


def _fit_differential(s: np.ndarray, D: np.ndarray) -> dict:
    """Zero crossing first (robust), then an erf fit seeded with it."""
    crossing = None
    for i in range(1, s.size):
        if D[i - 1] == 0.0:
            crossing = float(s[i - 1])
            break
        if D[i - 1] * D[i] < 0.0:
            t = D[i - 1] / (D[i - 1] - D[i])
            crossing = float(s[i - 1] + t * (s[i] - s[i - 1]))
            break
    out: dict = {"zero_crossing": crossing, "sign_changes": int(np.sum(np.diff(np.sign(D)) != 0))}
    if crossing is None:
        out["fit"] = None
        return out
    try:
        p, _ = curve_fit(
            _erf_model,
            s,
            D,
            p0=[float(np.max(np.abs(D))), crossing, 0.25 * float(np.ptp(s))],
            bounds=([0.0, float(s.min()) - 500.0, 5.0], [np.inf, float(s.max()) + 500.0, 2000.0]),
            maxfev=20000,
        )
        resid = _erf_model(s, *p) - D
        out["fit"] = {
            "amplitude": float(p[0]),
            "centre": float(p[1]),
            "sigma": float(p[2]),
            "rms_residual": float(np.sqrt(np.mean(resid**2))),
            "residual_over_amplitude": float(np.sqrt(np.mean(resid**2)) / (p[0] + 1e-12)),
        }
    except (RuntimeError, ValueError) as exc:
        out["fit"] = None
        out["fit_error"] = str(exc)
    return out


def _fit_pedestal(s: np.ndarray, P: np.ndarray) -> dict | None:
    """Free-baseline erf on one complement (assumes a constant pedestal)."""
    def model(x, b, a, c, sg):
        return b + a * 0.5 * (1.0 + erf((x - c) / (np.sqrt(2.0) * sg)))

    best = None
    for c0 in np.linspace(s.min(), s.max(), 11):
        try:
            p, _ = curve_fit(
                model,
                s,
                P,
                p0=[float(P.min()), float(P.max() - P.min()), c0, 150.0],
                bounds=([0.0, 0.0, s.min() - 500.0, 5.0], [2 * P.max(), 2 * P.max(), s.max() + 500.0, 2000.0]),
                maxfev=20000,
            )
        except (RuntimeError, ValueError):
            continue
        rms = float(np.sqrt(np.mean((model(s, *p) - P) ** 2)))
        if best is None or rms < best[1]:
            best = (p, rms)
    if best is None:
        return None
    p, rms = best
    return {
        "baseline": float(p[0]),
        "amplitude": float(p[1]),
        "centre": float(p[2]),
        "sigma": float(p[3]),
        "rms_residual": rms,
        "pedestal_fraction": float(p[0] / (p[0] + p[1])),
    }


def _plan(args) -> dict:
    return {
        "plan_only": True,
        "issue": 235,
        "script": "scripts/slm/issue235_beam_centre_differential_20261117.py",
        "devices": {"slm": args.slm_number, "slm_video_mode": 0, "camera": "daheng", "cam_id": args.cam_id},
        "exposure_ms_requested": args.exposure_ms,
        "n_scan": args.n_scan,
        "frames": 2 * args.n_scan * 2 + 4,
        "window_factors": [float(x) for x in args.window_factors.split(",")],
        "checker_period_px": CHECKER_PERIOD,
        "action": "differential knife edge: clear(<s) minus clear(>=s) per split, per axis",
        "writes": ["summary.json", "flat.npy", "release.json"],
    }


def _run(args) -> int:
    from ao_shaping.drivers.slm.factory import require_drive_board_entry

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    factors = [float(x) for x in args.window_factors.split(",")]
    release = {"slm_closed": False, "camera_closed": False, "lock_released": False}
    lock = None
    slm = None
    cam = None
    t0 = time.time()
    try:
        lock = _wait_for_devices(args)
        print("[占用] SLM #%d + Daheng CCD cam_id=%d" % (args.slm_number, args.cam_id), flush=True)
        slm = Santec(slm_number=args.slm_number, video_mode=0)
        slm.open()
        require_drive_board_entry(slm)
        panel_w, panel_h = slm.Panel_Res[0], slm.Panel_Res[1]
        shape = (panel_h, panel_w)

        cam = create_camera("daheng", args.cam_id, exposure_time_ms=float(args.exposure_ms))
        cam.open()
        used_ms, peak = _unsaturated_exposure(cam, float(args.exposure_ms), float(args.saturation_fraction))

        # flat reference
        slm.display_data(np.zeros(shape, dtype=np.uint16))
        time.sleep(SETTLE_S)
        flat = np.asarray(cam.get_numpy_image(int(args.samples)), dtype=np.float64)
        np.save(out / "flat.npy", flat)
        bg = float(np.median(flat))
        peak = float(flat.max())
        full = float(full_scale(flat, cam))
        cy, cx = np.unravel_index(int(np.argmax(flat)), flat.shape)
        cy_f, cx_f = _box_centroid(flat - bg, int(cy), int(cx), BOX_SIDE)
        fwhm_row = _fwhm1d((flat - bg)[int(cy), :])
        fwhm_col = _fwhm1d((flat - bg)[:, int(cx)])
        fwhm_geo = float(np.hypot(fwhm_row, fwhm_col))
        sides = {f: int(round(f * max(fwhm_geo, 5.0))) for f in factors}
        y0, x0 = int(round(cy_f)), int(round(cx_f))

        # pedestal reference: everything blocked
        slm.display_phase(_knife_pattern(0, "x", True, shape))
        time.sleep(SETTLE_S)
        blocked = np.asarray(cam.get_numpy_image(int(args.samples)), dtype=np.float64) - bg

        summary = {
            "elapsed_s": None,
            "exposure_ms_requested": float(args.exposure_ms),
            "exposure_ms_used": used_ms,
            "flat_peak_counts": peak,
            "flat_full_scale": full,
            "saturated": bool(peak >= 0.98 * full),
            "background_median": bg,
            "zero_order_cy_cx": [float(cy), float(cx)],
            "zero_order_box_centroid_cy_cx": [cy_f, cx_f],
            "fwhm_row_px": fwhm_row,
            "fwhm_col_px": fwhm_col,
            "fwhm_geometric_px": fwhm_geo,
            "window_side_px": {str(f): sides[f] for f in factors},
            "pedestal": {
                str(f): {
                    "flat_power": _window(flat - bg, y0, x0, sides[f]),
                    "blocked_power": _window(blocked, y0, x0, sides[f]),
                    "retained_fraction": _window(blocked, y0, x0, sides[f])
                    / (_window(flat - bg, y0, x0, sides[f]) + 1e-12),
                }
                for f in factors
            },
            "axes": {},
            "panel_res_wh": [panel_w, panel_h],
            "device": {
                "slm_number": args.slm_number,
                "slm_video_mode": 0,
                "serial_source": str(slm.serial_source()),
            },
        }

        for axis in ("x", "y"):
            extent = panel_w if axis == "x" else panel_h
            ss = np.linspace(0, extent, int(args.n_scan))
            curves = {str(f): {"A": [], "B": []} for f in factors}
            frame_max: list[float] = []
            for s in ss:
                si = int(round(s))
                row = {}
                for tag, clear_low in (("A", True), ("B", False)):
                    slm.display_phase(_knife_pattern(si, axis, clear_low, shape))
                    time.sleep(SETTLE_S)
                    img = np.asarray(cam.get_numpy_image(int(args.samples)), dtype=np.float64) - bg
                    frame_max.append(float(img.max()))
                    row[tag] = img
                for f in factors:
                    curves[str(f)]["A"].append(_window(row["A"], y0, x0, sides[f]))
                    curves[str(f)]["B"].append(_window(row["B"], y0, x0, sides[f]))
                print("  %s s=%4d done" % (axis, si), flush=True)

            axis_rec: dict = {"s": [float(x) for x in ss], "frame_max": max(frame_max), "windows": {}}
            for f in factors:
                A = np.asarray(curves[str(f)]["A"], dtype=np.float64)
                B = np.asarray(curves[str(f)]["B"], dtype=np.float64)
                D = A - B
                axis_rec["windows"][str(f)] = {
                    "A": A.tolist(),
                    "B": B.tolist(),
                    "differential": D.tolist(),
                    "differential_fit": _fit_differential(np.asarray(ss, dtype=np.float64), D),
                    "A_free_baseline_fit": _fit_pedestal(np.asarray(ss, dtype=np.float64), A),
                    "monotone_drop_steps_A": int(np.sum(np.diff(A) < 0)),
                }
            summary["axes"][axis] = axis_rec

        summary["elapsed_s"] = time.time() - t0
        (out / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )

        print("\n=== differential centre (panel px, cy/cx convention) ===", flush=True)
        for axis in ("x", "y"):
            for f in factors:
                rec = summary["axes"][axis]["windows"][str(f)]
                fit = rec["differential_fit"].get("fit")
                cross = rec["differential_fit"]["zero_crossing"]
                ped = rec["A_free_baseline_fit"]
                line = "  %s window=%2dx side=%4d  crossing=%s" % (axis, int(f), sides[f], f"{cross:.1f}" if cross else "none")
                if fit:
                    line += "  fit_c=%.1f sigma=%.1f resid/amp=%.3f" % (
                        fit["centre"],
                        fit["sigma"],
                        fit["residual_over_amplitude"],
                    )
                if ped:
                    line += "  | pedestal c=%.1f ped=%.2f" % (ped["centre"], ped["pedestal_fraction"])
                print(line, flush=True)
        return 0
    finally:
        if cam is not None:
            try:
                cam.close()
                release["camera_closed"] = True
            except Exception as exc:
                print("camera close failed:", exc)
        if slm is not None:
            try:
                slm.set_grayscale(0)
                slm.close()
                release["slm_closed"] = True
            except Exception as exc:
                print("SLM close failed:", exc)
        if lock is not None:
            try:
                lock.release()
                release["lock_released"] = True
            except Exception as exc:
                print("lock release failed:", exc)
        release["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            (Path(args.out) / "release.json").write_text(json.dumps(release, indent=2), encoding="utf-8")
        except OSError:
            pass
        print("release:", release, flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="new output directory (must not exist)")
    ap.add_argument("--slm-number", type=int, default=1)
    ap.add_argument("--cam-id", type=int, default=0)
    ap.add_argument("--exposure-ms", type=float, default=0.05)
    ap.add_argument("--saturation-fraction", type=float, default=0.98)
    ap.add_argument("--n-scan", type=int, default=41)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--window-factors", default="6,12,24")
    ap.add_argument("--wait-until-idle", type=float, default=0.0)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args(argv)

    if not Path(args.out).exists():
        pass
    elif args.execute:
        raise SystemExit(f"--out already exists: {args.out}")
    if not args.execute:
        print(json.dumps(_plan(args), indent=2, ensure_ascii=False))
        return 0
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
