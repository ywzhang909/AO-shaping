"""One-session optical-path diagnostic capture (raw evidence, no issue experiments).

Why this exists: issues #235 / #237 / #238 / #230 all need *physical* facts about
the current bench that are currently guessed from older runs:

* what exposure is actually linear (the previous run showed ``0.5 ms -> max 255``,
  i.e. the whole old ladder was clipped, so no exposure setting on record is safe),
* where the 0-order spot is and how big it is, measured on a non-clipped frame,
* where the beam sits on the SLM panel (knife-edge scan) at that exposure,
* the gray -> far-field response curve (#238's 1x1 LUT) and its spatial variation,
* the defocus PSF shape (#237's measured PSF) and a speckle-averaged envelope that
  is the only route to a gain field on this single-spot bench,
* the blaze-order geometry (camera px <-> panel px scale, diffraction baseline).

Steps (all at the *auto-chosen* exposure unless stated):

1. ``exposure`` -- ladder from the camera's own minimum up to 30 ms; the chosen
   exposure is the brightest one with no saturated pixel and peak in [120, 230].
2. ``flat`` -- raw frame + stats computed on a window around ``argmax`` (the whole
   frame centroid is polluted by a 0.375-count background, which is what made the
   first run report a 2585 px "row FWHM" -- a defect, not a measurement).
3. ``knife`` -- ``SLMCCDCalibrator.find_beam_on_slm`` with per-frame window powers
   at 1/2/3/6/12 x FWHM and every window crop saved for offline re-fit.
4. ``gray`` -- gray sweep 0..1023 (uniform) and a spatial check (200 px square at
   gray g on an otherwise flat panel) -> #238 LUT identifiability.
5. ``psf`` -- quadratic defocus sweep A = 0..60 rad at R = 600 px, frames saved ->
   #237 measured PSF.
6. ``speckle`` -- 8 random-phase frames, summed -> speckle-averaged envelope for a
   gain-field estimate (the coherent beam is the confound; still the only option).
7. ``orders`` -- blazed grating (period 24 and 12 panel px) order positions/energies.

Not captured here, and why: a true 2D-registration target (#237 (1)) is a pure
offline transform on existing windows; ``panel_gray`` replay (#235) needs a
``--save-dataset`` transfer capture, not this script.

Device use: SLM #1 (Santec, memory mode) + Daheng camera. Plan-only unless
``--execute``; ``--out`` must be new (``exist_ok=False``). The SLM is set to gray 0
and both devices are closed in ``finally``; failures are collected and reported.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ao_shaping.drivers.ccd.common import (
    get_camera_exposure_ms,
    get_camera_exposure_range,
)
from ao_shaping.drivers.slm.santec import Santec

CAMERA_SAMPLES = 10
#: Ladder from the camera minimum (0.02 ms on the MER2-507) upwards. The bench has
#: been recorded as "30 ms -> max 140" and also "0.5 ms -> max 255"; probing both
#: ends settles it instead of trusting either.
DEFAULT_EXPOSURES = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0, 2.0, 8.0, 30.0)
WINDOW_FACTORS = (1, 2, 3, 6, 12)
GRAY_LEVELS = (0, 64, 128, 192, 256, 384, 512, 640, 768, 896, 1023)
GRAY_SQUARE_LEVELS = (0, 256, 512, 768, 1023)
DEFOCUS_A_RAD = (0.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0)
DEFOCUS_R_PX = 600.0
SPECKLE_FRAMES = 8


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (tuple, list)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v


def _argmax_center(frame: np.ndarray) -> tuple[float, float]:
    """Spot centre as the intensity centroid inside a box around ``argmax``.

    Deliberately *not* the whole-frame centroid: on this bench the background
    median is ~0.4 counts over 5 Mpixel, which is enough to drag the centroid
    ~150 px away from the spot (observed: full-frame 989.9/1270.3 vs spot
    1101.8/1233.7).
    """
    f = np.asarray(frame, np.float64)
    h, w = f.shape
    py, px = np.unravel_index(int(np.argmax(f)), f.shape)
    # 120 px box: bigger than the observed core (30x24 px) plus its halo.
    half = 120
    y0, y1 = max(0, py - half), min(h, py + half + 1)
    x0, x1 = max(0, px - half), min(w, px + half + 1)
    sub = f[y0:y1, x0:x1]
    tot = float(sub.sum())
    if tot <= 0:
        return float(py), float(px)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return float((sub * yy).sum() / tot), float((sub * xx).sum() / tot)


def _frame_stats(frame: np.ndarray) -> dict:
    f = np.asarray(frame, np.float64)
    peak = float(f.max())
    cy, cx = _argmax_center(f)
    return {
        "shape": list(f.shape),
        "max": peak,
        "mean": float(f.mean()),
        "p99": float(np.percentile(f, 99)),
        "sum": float(f.sum()),
        # A pixel at the top code cannot be distinguished from a brighter one.
        "saturated_px": int(np.sum(f >= 255.0)) if peak >= 255.0 else 0,
        "background_median": float(np.median(f)),
        "spot_centroid_cy_cx": [cy, cx],
    }


def _fwhm1d(profile: np.ndarray) -> float:
    peak = float(profile.max())
    if peak <= 0:
        return 0.0
    above = np.nonzero(profile >= peak / 2)[0]
    return float(above[-1] - above[0] + 1) if len(above) >= 2 else 0.0


def _spot_shape(frame: np.ndarray) -> dict:
    """Row/column FWHM and radial profile, measured through the spot centre."""
    f = np.asarray(frame, np.float64)
    cy, cx = _argmax_center(f)
    h, w = f.shape
    row = f[min(max(int(round(cy)), 0), h - 1), :]
    col = f[:, min(max(int(round(cx)), 0), w - 1)]
    yy, xx = np.mgrid[0:h, 0:w]
    rad = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    prof = []
    for b in range(0, 400, 20):
        sel = (rad >= b) & (rad < b + 20)
        if sel.any():
            prof.append([b, b + 20, float(f[sel].mean())])
    total = float(f.sum())
    return {
        "center_cy_cx": [cy, cx],
        "fwhm_row_px": _fwhm1d(row),
        "fwhm_col_px": _fwhm1d(col),
        "radial_mean_20px_bins": prof,
        "frac_beyond_150px": float(f[rad > 150].sum() / total) if total else 0.0,
        "frac_beyond_300px": float(f[rad > 300].sum() / total) if total else 0.0,
    }


def _classify_pattern(phase: np.ndarray) -> tuple[str | None, float | None]:
    """Recover (axis, split) from the knife-edge pattern (clear side = 0)."""
    zero = np.asarray(phase) == 0
    rows = int(zero.all(axis=1).sum())
    cols = int(zero.all(axis=0).sum())
    h, w = zero.shape
    if rows == 0 and cols > 0:
        return "x", float(cols)
    if cols == 0 and rows > 0:
        return "y", float(rows)
    if rows == h and cols == w:
        return "clear", None
    if rows == 0 and cols == 0:
        return "checker", None
    return None, None


def _window_power(frame: np.ndarray, center: tuple[float, float], win: int) -> float:
    """Sum inside a ``win`` x ``win`` box around ``center`` (edge-padded)."""
    cy, cx = int(round(center[0])), int(round(center[1]))
    hh, ww = frame.shape
    y0, x0 = cy - win // 2, cx - win // 2
    pt, pb = max(0, -y0), max(0, y0 + win - hh)
    pl, pr = max(0, -x0), max(0, x0 + win - ww)
    p = frame[max(0, y0) : min(hh, y0 + win), max(0, x0) : min(ww, x0 + win)]
    if pt or pb or pl or pr:
        p = np.pad(p, ((pt, pb), (pl, pr)), mode="edge")
    return float(p.sum())


class RecordingCamera:
    """Camera proxy: records the knife-edge pattern and every frame's powers."""

    def __init__(self, cam, out_dir: Path, center: tuple[float, float], windows: list[int]):
        self._cam = cam
        self._out = out_dir
        self._patterns: list[tuple[str | None, float | None]] = []
        self.frames: list[dict] = []
        self.center = center  # fixed on the flat frame: the scan must not move it
        self.windows = windows
        self.crop_size = int(min(max(2 * max(windows), 256), 1500))
        self._n = 0
        (out_dir / "frames").mkdir(parents=True, exist_ok=True)

    def __getattr__(self, name):
        return getattr(self._cam, name)

    def show(self, phase) -> None:
        self._patterns.append(_classify_pattern(np.asarray(phase, np.float32)))

    def get_numpy_image(self, n_sample: int = 1, skip_first: bool = True, **kw):
        frame = np.asarray(self._cam.get_numpy_image(n_sample, skip_first, **kw), np.float64)
        axis, s = self._patterns.pop(0) if self._patterns else (None, None)
        powers = {str(w): _window_power(frame, self.center, w) for w in self.windows}
        crop = self._crop(frame)
        name = "frames/frame_%03d.npy" % self._n
        self._n += 1
        np.save(self._out / name, crop.astype(np.uint8))
        self.frames.append(
            {
                "axis": axis,
                "s": s,
                "powers": powers,
                "max": float(frame.max()),
                "mean": float(frame.mean()),
                "sat": int((frame >= 255.0).sum()),
                "crop": name,
            }
        )
        return frame

    def _crop(self, frame: np.ndarray) -> np.ndarray:
        cy, cx = int(round(self.center[0])), int(round(self.center[1]))
        half = self.crop_size // 2
        h, w = frame.shape
        y0, x0 = cy - half, cx - half
        return frame[max(0, y0) : min(h, y0 + self.crop_size), max(0, x0) : min(w, x0 + self.crop_size)]


def _curve_summary(frames: list[dict]) -> dict:
    out = {}
    for axis in ("x", "y"):
        sel = sorted(
            [f for f in frames if f["axis"] == axis and f["s"] is not None],
            key=lambda f: f["s"],
        )
        if len(sel) < 3:
            out[axis] = {"n": len(sel)}
            continue
        s = np.array([f["s"] for f in sel], float)
        block = {"n": len(sel), "s_range": [float(s.min()), float(s.max())], "windows": {}}
        for key in sel[0]["powers"]:
            P = np.array([f["powers"][key] for f in sel], float)
            rng = float(P.max() - P.min())
            block["windows"][key] = {
                "power_min": float(P.min()),
                "power_max": float(P.max()),
                "power_range": rng,
                "relative_modulation": rng / float(P.max()) if P.max() > 0 else 0.0,
                "points": [[float(a), float(b)] for a, b in zip(s, P)],
            }
        out[axis] = block
    return out


def _order_profile(frame: np.ndarray) -> dict:
    """Column-sum runs: 0 order and the +/-1 blaze orders."""
    f = np.asarray(frame, np.float64)
    col = f.sum(axis=0)
    peak = float(col.max()) if col.size else 0.0
    out = {"col_sum_peaks": [], "peak": peak}
    if peak <= 0:
        return out
    above = col >= 0.15 * peak
    runs, start = [], None
    for i, v in enumerate(above):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(above) - 1))
    idx = np.arange(col.size, dtype=np.float64)
    for a, b in runs:
        seg = col[a : b + 1]
        if seg.sum() <= 0:
            continue
        out["col_sum_peaks"].append(
            {
                "x0": int(a),
                "x1": int(b),
                "width": int(b - a + 1),
                "centroid_x": float((seg * idx[a : b + 1]).sum() / seg.sum()),
                "energy": float(seg.sum()),
                "peak": float(seg.max()),
            }
        )
    return out


def _save_png(frame: np.ndarray, path: Path) -> str | None:
    try:
        from PIL import Image

        img = np.asarray(frame, np.float64)
        lo, hi = float(img.min()), float(img.max())
        norm = (img - lo) / (hi - lo + 1e-12)
        h, w = norm.shape
        scale = max(1, int(np.ceil(max(h, w) / 1296)))
        Image.fromarray((norm * 255).astype(np.uint8)).resize(
            (w // scale, h // scale), Image.BILINEAR
        ).save(path)
        return None
    except Exception as exc:
        return str(exc)


def _choose_exposure(ladder: list[dict]) -> float:
    """Brightest exposure with no clipping and peak in [120, 230]."""
    clean = [r for r in ladder if r["saturated_px"] == 0 and r["max"] >= 120.0]
    if clean:
        return max(clean, key=lambda r: r["max"])["requested_ms"]
    clean = [r for r in ladder if r["saturated_px"] == 0]
    if clean:
        return max(clean, key=lambda r: r["max"])["requested_ms"]
    return min(ladder, key=lambda r: r["saturated_px"] or r["max"])["requested_ms"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--slm-number", type=int, default=1)
    ap.add_argument("--cam-id", type=int, default=0)
    ap.add_argument("--n-scan", type=int, default=25)
    ap.add_argument("--exposures", type=float, nargs="*", default=list(DEFAULT_EXPOSURES))
    ap.add_argument("--skip", nargs="*", default=[], help="steps to skip: knife gray psf speckle orders")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    plan = {
        "device": {"slm": args.slm_number, "camera": "daheng", "cam_id": args.cam_id},
        "steps": [
            "exposure ladder (camera min .. 30 ms), pick the brightest unclipped one",
            "flat frame at that exposure (raw npy + png + spot stats)",
            "knife-edge scan: panel beam centre, multi-window powers, per-frame crops",
            "gray sweep (uniform) + 200 px square sweep -> 1x1 LUT and its spatial part",
            "defocus sweep A=0..60 rad @ R=600 px -> measured PSF",
            "speckle-averaged envelope (8 random phases) -> gain-field input",
            "blaze orders (period 24, 12 panel px)",
        ],
        "exposures_ms": list(args.exposures),
        "n_scan": args.n_scan,
        "skipped": list(args.skip),
    }
    if not args.execute:
        print(json.dumps({"plan_only": True, **plan}, ensure_ascii=False, indent=2))
        return 0

    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "meta.json").write_text(
        json.dumps({"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2),
        encoding="utf-8",
    )
    summary: dict = {"plan": plan}
    errors: list[str] = []
    slm = Santec(slm_number=args.slm_number, video_mode=0)
    slm.open()
    cam = None
    try:
        from ao_shaping.drivers.ccd.common import create_camera, set_camera_exposure_ms

        cam = create_camera("daheng", args.cam_id, exposure_time_ms=float(args.exposures[1]))
        cam.open()
        summary["camera"] = {
            "exposure_range_ms": list(get_camera_exposure_range(cam)),
            "full_frame": list(np.asarray(cam.get_numpy_image(1, True)).shape),
        }
        slm.set_grayscale(0)
        time.sleep(0.5)

        # ---- 1. exposure ladder ----------------------------------------------
        ladder = []
        for ms in args.exposures:
            applied = float(set_camera_exposure_ms(cam, float(ms)))
            frame = np.asarray(cam.get_numpy_image(3, True), np.float64)
            st = _frame_stats(frame)
            ladder.append(
                {
                    "requested_ms": float(ms),
                    "applied_ms": applied,
                    "readback_ms": float(get_camera_exposure_ms(cam)),
                    "max": st["max"],
                    "p99": st["p99"],
                    "mean": st["mean"],
                    "saturated_px": st["saturated_px"],
                    "sum": st["sum"],
                }
            )
            print("ladder %7.2f ms -> max %5.0f  p99 %6.2f  sat %5d  sum %.3e"
                  % (ms, st["max"], st["p99"], st["saturated_px"], st["sum"]))
        summary["exposure_ladder"] = ladder
        chosen = _choose_exposure(ladder)
        summary["chosen_exposure_ms"] = chosen
        summary["chosen_readback_ms"] = float(set_camera_exposure_ms(cam, chosen))
        print("chosen exposure: %.3f ms (readback %.3f ms)" % (chosen, summary["chosen_readback_ms"]))

        # ---- 2. flat ---------------------------------------------------------
        slm.set_grayscale(0)
        time.sleep(0.5)
        flat = np.asarray(cam.get_numpy_image(8, True), np.float64).astype(np.float32)
        np.save(args.out / "flat.npy", flat)
        fst = _frame_stats(flat)
        fst["spot"] = _spot_shape(flat)
        fst["png_error"] = _save_png(flat, args.out / "flat.png")
        fst["histogram"] = np.histogram(flat, bins=32, range=(0, 256))[0].tolist()
        summary["flat"] = fst
        (args.out / "flat_stats.json").write_text(
            json.dumps(_jsonable(fst), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        cy0, cx0 = fst["spot"]["center_cy_cx"]
        print("flat: max %.0f sat %d spot %s fwhm row/col %.1f/%.1f"
              % (fst["max"], fst["saturated_px"], (round(cy0, 1), round(cx0, 1)),
                 fst["spot"]["fwhm_row_px"], fst["spot"]["fwhm_col_px"]))

        from ao_shaping.tools.slm.calibration.calibration import SLMCCDCalibrator

        cal = SLMCCDCalibrator(slm, cam)

        # ---- 3. knife-edge ---------------------------------------------------
        if "knife" not in args.skip:
            base = max(float(min(fst["spot"]["fwhm_row_px"], fst["spot"]["fwhm_col_px"])), 5.0)
            windows = [int(f * base) for f in WINDOW_FACTORS]
            rec = RecordingCamera(cam, args.out, (cy0, cx0), windows)
            cal.ccd = rec  # type: ignore[assignment]
            orig_show = cal._show

            def show(phase):
                rec.show(phase)
                orig_show(phase)

            cal._show = show  # type: ignore[method-assign]
            t0 = time.time()
            knife = cal.find_beam_on_slm(n_scan=args.n_scan)
            elapsed = time.time() - t0
            summary["knife_edge"] = {
                "elapsed_s": elapsed,
                "result": _jsonable(knife),
                "window_center_cy_cx": [cy0, cx0],
                "windows_px": windows,
                "crop_size_px": rec.crop_size,
                "curves": _curve_summary(rec.frames),
                "frames": rec.frames,
                "max_sat_px": max([f["sat"] for f in rec.frames], default=0),
            }
            (args.out / "knifedge.json").write_text(
                json.dumps(_jsonable(summary["knife_edge"]), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print("knife-edge: center %s sigma %s (%.1f s)"
                  % (knife["beam_center"], knife["beam_sigma"], elapsed))

        h, w = slm.Panel_Res[1], slm.Panel_Res[0]
        win = int(max(3 * max(fst["spot"]["fwhm_row_px"], fst["spot"]["fwhm_col_px"]), 60))

        def spot_power(frame):
            return _window_power(np.asarray(frame, np.float64), (cy0, cx0), win)

        # ---- 4. gray sweep ---------------------------------------------------
        if "gray" not in args.skip:
            slm.set_grayscale(0)
            time.sleep(0.4)
            base_power = spot_power(cam.get_numpy_image(4, True))
            levels = []
            for g in GRAY_LEVELS:
                slm.set_grayscale(int(g))
                time.sleep(0.35)
                frame = np.asarray(cam.get_numpy_image(4, True), np.float64)
                levels.append(
                    {
                        "gray": int(g),
                        "spot_power": spot_power(frame),
                        "total": float(frame.sum()),
                        "max": float(frame.max()),
                        "sat": int((frame >= 255.0).sum()),
                    }
                )
                print("gray %4d -> spot %.4e  total %.4e" % (g, levels[-1]["spot_power"], levels[-1]["total"]))
            np.save(args.out / "gray_sweep_last.npy", frame.astype(np.uint8))

            # Spatial part: 200 px square at gray g, rest flat 0.
            sq = 200
            sy, sx = h // 2 - sq // 2, w // 2 - sq // 2
            squares = []
            for g in GRAY_SQUARE_LEVELS:
                arr = np.zeros((h, w), np.uint16)
                arr[sy : sy + sq, sx : sx + sq] = int(g)
                slm.display_data(arr, wait_time_s=0.4)
                frame = np.asarray(cam.get_numpy_image(4, True), np.float64)
                squares.append(
                    {
                        "gray": int(g),
                        "spot_power": spot_power(frame),
                        "max": float(frame.max()),
                        "total": float(frame.sum()),
                        "sat": int((frame >= 255.0).sum()),
                        "frame_png": str(args.out / ("gray_square_%04d.png" % g)),
                    }
                )
                _save_png(frame, args.out / ("gray_square_%04d.png" % g))
                print("square %4d -> spot %.4e" % (g, squares[-1]["spot_power"]))
            summary["gray"] = {
                "window_px": win,
                "base_power": base_power,
                "uniform": levels,
                "square": {"size_px": sq, "origin_yx": [sy, sx], "levels": squares},
            }
            (args.out / "gray.json").write_text(
                json.dumps(_jsonable(summary["gray"]), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        # ---- 5. defocus PSF --------------------------------------------------
        if "psf" not in args.skip:
            yy, xx = np.mgrid[0:h, 0:w]
            rad = np.sqrt((yy - 600.0) ** 2 + (xx - 960.0) ** 2)
            rr = np.clip(rad / DEFOCUS_R_PX, 0.0, 1.0) ** 2
            psf = []
            for A in DEFOCUS_A_RAD:
                phase = (A * rr).astype(np.float32)
                slm.display_phase(phase, wait_time_s=0.5)
                frame = np.asarray(cam.get_numpy_image(4, True), np.float64)
                np.save(args.out / ("psf_a%04.1f.npy" % A), frame.astype(np.uint8))
                st = _frame_stats(frame)
                st["spot"] = _spot_shape(frame)
                psf.append({"A_rad": float(A), "stats": st, "spot_power": spot_power(frame)})
                print("psf A=%5.1f rad -> fwhm %.1f/%.1f  max %.0f sat %d"
                      % (A, st["spot"]["fwhm_row_px"], st["spot"]["fwhm_col_px"],
                         st["max"], st["saturated_px"]))
            summary["psf"] = {"R_px": DEFOCUS_R_PX, "sweep": psf}
            (args.out / "psf.json").write_text(
                json.dumps(_jsonable(summary["psf"]), ensure_ascii=False, indent=2), encoding="utf-8"
            )

        # ---- 6. speckle-averaged envelope -----------------------------------
        if "speckle" not in args.skip:
            rng = np.random.default_rng(20261110)
            acc = None
            for i in range(SPECKLE_FRAMES):
                phase = rng.uniform(0.0, 2.0 * np.pi, size=(h, w)).astype(np.float32)
                slm.display_phase(phase, wait_time_s=0.35)
                frame = np.asarray(cam.get_numpy_image(3, True), np.float64)
                acc = frame if acc is None else acc + frame
            mean_frame = acc / SPECKLE_FRAMES
            np.save(args.out / "speckle_mean.npy", mean_frame.astype(np.float32))
            _save_png(mean_frame, args.out / "speckle_mean.png")
            summary["speckle"] = {
                "frames": SPECKLE_FRAMES,
                "max": float(mean_frame.max()),
                "mean": float(mean_frame.mean()),
                "sum": float(mean_frame.sum()),
                "frac_above_1pct_peak": float((mean_frame > 0.01 * mean_frame.max()).mean()),
            }
            print("speckle: max %.1f mean %.3f coverage %.3f"
                  % (summary["speckle"]["max"], summary["speckle"]["mean"],
                     summary["speckle"]["frac_above_1pct_peak"]))

        # ---- 7. blaze orders -------------------------------------------------
        if "orders" not in args.skip:
            slm.set_grayscale(0)
            time.sleep(0.4)
            orders = {"reference": _frame_stats(cam.get_numpy_image(4, True))}
            np.save(args.out / "orders_flat.npy", np.asarray(cam.get_numpy_image(1, True), np.uint8))
            for period in (24, 12):
                phase = (2.0 * np.pi * ((xx % period) / period)).astype(np.float32)
                slm.display_phase(phase, wait_time_s=0.5)
                frame = np.asarray(cam.get_numpy_image(4, True), np.float64)
                np.save(args.out / ("orders_blaze%02d.npy" % period), frame.astype(np.uint8))
                orders["blaze_%02d" % period] = {
                    "stats": _frame_stats(frame),
                    "profile": _order_profile(frame),
                }
                print("blaze %02d px: peaks %s" % (
                    period,
                    [(round(p["centroid_x"], 1), round(p["energy"], -3))
                     for p in orders["blaze_%02d" % period]["profile"]["col_sum_peaks"]],
                ))
            summary["orders"] = orders
            (args.out / "orders.json").write_text(
                json.dumps(_jsonable(orders), ensure_ascii=False, indent=2), encoding="utf-8"
            )

        summary["done_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        summary["errors"] = errors
        (args.out / "summary.json").write_text(
            json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("wrote", args.out / "summary.json")
    except BaseException:
        errors.append("aborted: see traceback above")
        raise
    finally:
        if cam is not None:
            try:
                cam.close()
                print("camera closed")
            except Exception as exc:
                errors.append("camera close: %s" % exc)
                print("camera close warning:", exc)
        try:
            slm.set_grayscale(0)
            print("slm flat (gray 0)")
        except Exception as exc:
            errors.append("slm flat: %s" % exc)
            print("slm flat warning:", exc)
        try:
            slm.close()
            print("slm closed")
        except Exception as exc:
            errors.append("slm close: %s" % exc)
            print("slm close warning:", exc)
        if errors:
            print("RELEASE ERRORS:", errors)
        else:
            print("released: SLM #%d gray 0 + closed, camera closed" % args.slm_number)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
