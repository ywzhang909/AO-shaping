"""Issue #235 hardware step: measure the SLM beam centre on the panel.

Calls ``SLMCCDCalibrator.find_beam_on_slm`` (the only supported path to the
knife-edge measurement) and, in parallel, records the evidence needed to
*judge* its answer:

* the flat-field frame the calibrator starts from (``F0``), saved as ``.npy``
  plus a downscaled ``.png`` for eyeballing;
* the raw knife-edge power curve per axis, recomputed from the frames the
  calibrator actually captured, using the same window rule as the library
  (``calibration.py`` ``find_beam_on_slm``);
* the library's own ``beam_center`` / ``beam_sigma``.

The curve is the point: a monotonic, well-spread power-vs-split curve means
the fit is meaningful, while a flat or one-sided curve means the scan never
crossed the beam and the returned centre is an artefact.

Device use: SLM #1 (Santec, memory mode) and the Daheng camera (cam_id 0).
Plan-only unless ``--execute``. ``--out`` must be new (exist_ok=False).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ao_shaping.drivers.ccd.common import create_camera
from ao_shaping.drivers.slm.santec import Santec
from ao_shaping.tools.slm.calibration.calibration import SLMCCDCalibrator

CAMERA_SAMPLES = 10


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, tuple):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v


def _classify_pattern(phase: np.ndarray) -> tuple[str | None, float | None]:
    """Recover (axis, split position) from the knife-edge pattern.

    ``scan_pattern`` clears one side and checkerboards the other, so the count
    of all-zero rows/columns identifies the axis and the split.
    """
    zero = phase == 0
    rows_all_zero = zero.all(axis=1)
    cols_all_zero = zero.all(axis=0)
    n_rows = int(rows_all_zero.sum())
    n_cols = int(cols_all_zero.sum())
    h, w = zero.shape
    if n_rows == 0 and n_cols > 0:
        return "x", float(n_cols)
    if n_cols == 0 and n_rows > 0:
        return "y", float(n_rows)
    if n_rows == h and n_cols == w:
        return "both", None  # fully clear, ambiguous but harmless
    return None, None


#: Window sizes (multiples of the flat-field FWHM) whose power is recorded for
#: every knife-edge frame. The library uses 6*FWHM only; recording several lets
#: the offline analysis check how much the answer depends on the window, which
#: is what decides whether the fit is a beam edge or an artefact.
WINDOW_FACTORS = (1, 2, 3, 6, 12)


class RecordingCamera:
    """Proxy around the camera that records frames and window power."""

    def __init__(self, cam, calibrator):
        self._cam = cam
        self._cal = calibrator
        self._pattern: np.ndarray | None = None
        self._pending: list[tuple[str | None, float | None]] = []
        self.curves: list[dict] = []
        self.flat: np.ndarray | None = None
        self.window: tuple[int, int, int] | None = None  # (cy, cx, win)
        self.windows: list[int] = []
        self._first = True

    def __getattr__(self, name):
        return getattr(self._cam, name)

    def show(self, phase: np.ndarray) -> None:
        self._pattern = np.asarray(phase, np.float32)
        axis, s = _classify_pattern(self._pattern)
        self._pending.append((axis, s))

    def get_numpy_image(self, n_sample: int = 1, skip_first: bool = True, **kw):
        frame = np.asarray(self._cam.get_numpy_image(n_sample, skip_first, **kw), np.float64)
        axis, s = self._pending.pop(0) if self._pending else (None, None)
        if self._first:
            self.flat = frame
            c0 = self._cal._moments(frame)
            fwhm = float(self._cal.spot_fwhm(frame, c0))
            win = int(6 * max(fwhm, 5))
            cy, cx = int(round(c0[0])), int(round(c0[1]))
            self.window = (cy, cx, win)
            self.windows = [int(f * max(fwhm, 5)) for f in WINDOW_FACTORS]
            self._first = False
            return frame
        powers = {str(w): self._window_power(frame, w) for w in self.windows}
        self.curves.append(
            {"axis": axis, "s": s, "power": powers["%d" % self.window[2]],
             "powers": powers, "max": float(frame.max())}
        )
        return frame

    def _window_power(self, frame: np.ndarray, win: int) -> float:
        cy, cx = self.window[0], self.window[1]
        hh, ww = frame.shape
        y0, x0 = cy - win // 2, cx - win // 2
        pt, pb = max(0, -y0), max(0, y0 + win - hh)
        pl, pr = max(0, -x0), max(0, x0 + win - ww)
        p = frame[max(0, y0) : min(hh, y0 + win), max(0, x0) : min(ww, x0 + win)]
        if pt or pb or pl or pr:
            p = np.pad(p, ((pt, pb), (pl, pr)), mode="edge")
        return float(p.sum())


def _curve_summary(points: list[dict]) -> dict:
    """Judge the curve: spread, monotonicity, and where 50% falls."""
    pts = [p for p in points if p["axis"] in ("x", "y") and p["s"] is not None]
    out: dict = {}
    for axis in ("x", "y"):
        sel = sorted([p for p in pts if p["axis"] == axis], key=lambda p: p["s"])
        if len(sel) < 3:
            out[axis] = {"n": len(sel)}
            continue
        P = np.array([p["power"] for p in sel], float)
        s = np.array([p["s"] for p in sel], float)
        rng = float(P.max() - P.min())
        # count monotone-decreasing steps (power should fall as the split crosses)
        drops = int(np.sum(np.diff(P) < 0))
        out[axis] = {
            "n": len(sel),
            "s_min": float(s.min()),
            "s_max": float(s.max()),
            "power_min": float(P.min()),
            "power_max": float(P.max()),
            "power_range": rng,
            "monotone_drop_steps": drops,
            "points": [{"s": float(a), "power": float(b)} for a, b in zip(s, P)],
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--slm-number", type=int, default=1)
    ap.add_argument("--cam-id", type=int, default=0)
    ap.add_argument("--exposure-ms", type=float, default=30.0)
    ap.add_argument("--n-scan", type=int, default=25)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    plan = {
        "device": {"slm": args.slm_number, "camera": "daheng", "cam_id": args.cam_id},
        "action": "find_beam_on_slm (knife-edge scan, n_scan=%d)" % args.n_scan,
        "exposure_ms": args.exposure_ms,
        "writes": ["beam_center.json", "flat.npy", "flat.png"],
    }
    if not args.execute:
        print(json.dumps({"plan_only": True, **plan}, ensure_ascii=False, indent=2))
        return 0

    args.out.mkdir(parents=True, exist_ok=False)
    slm = Santec(slm_number=args.slm_number, video_mode=0)
    slm.open()
    cam = None
    try:
        cam = create_camera("daheng", args.cam_id, exposure_time_ms=args.exposure_ms)
        cam.open()
        cal = SLMCCDCalibrator(slm, cam)
        rec = RecordingCamera(cam, cal)
        cal.ccd = rec  # type: ignore[assignment]
        orig_show = cal._show

        def show(phase):  # capture the pattern the calibrator puts on the panel
            rec.show(phase)
            orig_show(phase)

        cal._show = show  # type: ignore[method-assign]
        t0 = time.time()
        res = cal.find_beam_on_slm(n_scan=args.n_scan)
        elapsed = time.time() - t0

        summary = {
            "elapsed_s": elapsed,
            "result": _jsonable(res),
            "panel_res": list(slm.Panel_Res),
            "exposure_ms": args.exposure_ms,
            "window": rec.window,
            "curves": _curve_summary(rec.curves),
            "raw_curve": rec.curves,
        }
        if rec.flat is not None:
            flat = rec.flat
            c0 = cal._moments(flat)
            summary["flat_stats"] = {
                "max": float(flat.max()),
                "mean": float(flat.mean()),
                "centroid": [float(c0[0]), float(c0[1])],
                "fwhm": float(cal.spot_fwhm(flat, c0)),
                "saturated_px": int(np.sum(flat >= flat.max() - 1e-9)),
            }
            np.save(args.out / "flat.npy", flat.astype(np.float32))
            try:
                from PIL import Image

                img = flat.astype(np.float64)
                lo, hi = float(img.min()), float(img.max())
                norm = (img - lo) / (hi - lo + 1e-12)
                Image.fromarray((norm * 255).astype(np.uint8)).resize(
                    (1296, 972), Image.BILINEAR
                ).save(args.out / "flat.png")
            # The PNG render is a convenience artifact; a failure here must not
            # abort the capture that already succeeded.
            except Exception as exc:
                summary["png_error"] = str(exc)
        (args.out / "beam_center.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if cam is not None:
            try:
                cam.close()
            except Exception as exc:
                # The device API is polymorphic; a close failure must not mask
                # the capture result that was already written to disk.
                print(f"camera close warning: {exc}")
        try:
            slm.set_grayscale(0)
        except Exception as exc:
            print(f"slm flat-set warning: {exc}")
        try:
            slm.close()
        except Exception as exc:
            print(f"slm close warning: {exc}")
        print("released: SLM #%d closed, camera closed" % args.slm_number)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
