"""Hardware verification of the focus probe's answer, and the evidence behind it.

The probe itself only reports *one* number: the Fresnel power that maximises
``peak / exposure``. That is a peak statistic, and a peak statistic can rise for
two very different reasons -- the spot got sharper, or the light got
concentrated back into the frame. This script re-measures the same
configurations and adds the quantities that tell the two apart, so the probe's
answer can be checked instead of trusted:

* **exposure ladder** on a displayed flat phase. Establishes that the sensor is
  linear in exposure (which is the assumption the ranking rests on) and records
  how much light the bench is delivering right now.
* **steering check** at three focal lengths. A lens centred off the beam carries
  a linear phase, and on this bench that steering is huge -- so the centroid
  shift is a direct readout of whether the assumed pupil centre is usable.
* **verification sweep** with two exposure-invariant additions: whole-frame
  energy per ms (constant unless light is leaving the frame) and the fraction of
  that energy inside a box on the spot (immune to exposure). A peak gain that is
  *not* matched by an energy gain would mean the probe was measuring an aperture
  effect.

Every configuration is first adapted to ~50% of full scale, exactly as the probe
does, so the comparison is on equal footing rather than on one saturated frame.

Usage:
    python scripts/slm/verify_focus_plane_hw.py
    python scripts/slm/verify_focus_plane_hw.py --focals 1418 --no-ladder
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.drivers.ccd.common import create_camera
from ao_shaping.drivers.slm.santec import Santec
from ao_shaping.tools.kernels.bench_kernels import (
    despike_frame,
    fresnel_panel,
    measure_spot,
)
from ao_shaping.tools.slm.panel.focus_probe import adapt_exposure

PANEL = (1200, 1920)
DEFAULT_OUT = ROOT / "data" / "debug" / "bench_settings" / "focus_probe"

#: Box radius (camera px) for the "is it really a spot" energy fraction. Wide
#: enough to hold the focused spot plus its wings, narrow enough that a diffuse
#: defocused halo does not fill it.
BOX_PX = 60.0
FULL_SCALE = 255.0
LADDER_MS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)


@dataclass
class Rig:
    """The three device callables the measurement loop needs, bundled.

    Keeping them in one object makes the read/display/exposure order explicit at
    every call site -- the order is load-bearing (settle, discard, average) and
    is easy to get wrong when the three are loose parameters.
    """

    cam: Any
    slm: Any

    def apply(self, phase: np.ndarray) -> None:
        # ``display_data`` with no memory_number: a fixed slot is a firmware no-op.
        self.slm.display_data(self.slm.create_phase_from_array(phase), 0.4)

    def apply_flat(self) -> None:
        self.apply(np.zeros(PANEL, dtype=np.float64))

    def set_exposure(self, ms: float) -> float:
        return float(self.cam.reset_exposure_time(float(ms)) or ms)

    def read(self) -> np.ndarray:
        return np.asarray(self.cam.get_numpy_image(n_sample=1), dtype=np.float64)

    def average(self, n: int = 3) -> np.ndarray:
        acc: np.ndarray | None = None
        for _ in range(n):
            frame = self.read()
            acc = frame if acc is None else acc + frame
        assert acc is not None
        return acc / n


def robust_peak(frame: np.ndarray) -> float:
    """Frame peak after a 3x3 median, so a single hot pixel cannot win."""
    return float(despike_frame(np.asarray(frame, dtype=np.float64), 3).max())


def background_free(frame: np.ndarray) -> np.ndarray:
    """Median-subtracted, clipped signal.

    The raw sum carries a per-frame pedestal that scales with exposure and would
    otherwise fake a difference in *total* energy between configurations.
    """
    return np.clip(np.asarray(frame, dtype=np.float64) - np.median(frame), 0.0, None)


def in_box_fraction(frame: np.ndarray, cx: float, cy: float) -> tuple[float, float]:
    """``(fraction, total_counts)`` of the signal inside ``BOX_PX`` of the spot."""
    signal = background_free(frame)
    total = float(signal.sum())
    if total <= 0:
        return 0.0, 0.0
    iy, ix = np.mgrid[0 : frame.shape[0], 0 : frame.shape[1]]
    box = (iy - cy) ** 2 + (ix - cx) ** 2 <= BOX_PX**2
    return float(signal[box].sum() / total), total


def ladder(rig: Rig, repeats: int = 3) -> list[dict[str, float]]:
    """Peak vs exposure on a displayed flat phase: linearity and current level."""
    rig.apply_flat()
    rows: list[dict[str, float]] = []
    for ms in LADDER_MS:
        rig.set_exposure(ms)
        rig.read()
        rig.read()
        frame = rig.average(repeats)
        m = measure_spot(frame)
        peak = robust_peak(frame)
        frac, _ = in_box_fraction(frame, m.centroid_x, m.centroid_y)
        rows.append(
            {
                "exposure_ms": float(ms),
                "peak": peak,
                "peak_fraction": peak / FULL_SCALE,
                "median": float(np.median(frame)),
                "counts_per_ms": peak / float(ms),
                "fwhm_px": float(m.fwhm_px),
                "centroid": [float(m.centroid_x), float(m.centroid_y)],
                "in_box_frac": frac,
            }
        )
        print(
            f"  ladder {ms:6.2f} ms  peak={peak:6.1f}  "
            f"counts/ms={peak / float(ms):7.1f}  fwhm={m.fwhm_px:6.1f}",
            flush=True,
        )
    return rows


def measure_focal(rig: Rig, focal_mm: float, frames: int = 3) -> dict[str, Any]:
    """One focal length, adapted to ~50% full scale, with exposure-free extras."""
    rig.apply(
        fresnel_panel(
            focal_mm,
            radius=450,
            pupil_center=(960, 690),
            panel_shape=PANEL,
        )
    )
    setting = adapt_exposure(
        rig.set_exposure,
        rig.read,
        start_ms=0.2,
        floor_ms=0.02,
        ceiling_ms=20.0,
        full_scale=FULL_SCALE,
        discard=2,
    )
    frame = rig.average(frames)
    m = measure_spot(frame)
    peak = robust_peak(frame)
    frac, total = in_box_fraction(frame, m.centroid_x, m.centroid_y)
    return {
        "focal_mm": None if math.isinf(focal_mm) else float(focal_mm),
        "is_plain": bool(math.isinf(focal_mm)),
        "exposure_ms": setting.exposure_ms,
        "peak": peak,
        "peak_fraction": peak / FULL_SCALE,
        "peak_per_ms": peak / setting.exposure_ms,
        "fwhm_px": float(m.fwhm_px),
        "hollowness": float(m.hollowness),
        "centroid": [float(m.centroid_x), float(m.centroid_y)],
        "energy_per_ms": total / setting.exposure_ms,
        "in_box_frac": frac,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--focals",
        default="inf,3000,2000,1418,1000,666,-1000",
        help="comma-separated focal lengths in mm ('inf' = plain phase)",
    )
    ap.add_argument("--frames", type=int, default=3, help="frames averaged (default 3)")
    ap.add_argument("--repeats", type=int, default=3, help="ladder frames (default 3)")
    ap.add_argument("--no-ladder", action="store_true", help="skip the exposure ladder")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT), help="evidence directory")
    args = ap.parse_args(argv)

    focals = [
        float("inf") if v.strip() == "inf" else float(v)
        for v in str(args.focals).split(",")
        if v.strip()
    ]

    payload: dict[str, Any] = {
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "slm_number": 1,
        "cam_type": "daheng",
        "cam_id": 0,
        "centre_panel_px": [960, 690],
        "radius_px": 450,
        "box_px": BOX_PX,
        "frames": int(args.frames),
    }

    with Santec(slm_number=1, wavelength=1064, video_mode=0) as slm:
        with create_camera("daheng", 0, exposure_time_ms=0.2) as cam:
            rig = Rig(cam=cam, slm=slm)
            if not args.no_ladder:
                print("exposure ladder on a flat phase:", flush=True)
                payload["ladder"] = ladder(rig, int(args.repeats))

            print("\nverification sweep (each point adapted to ~50% full scale):")
            rows = [measure_focal(rig, f, int(args.frames)) for f in focals]
            rig.apply_flat()

    plain = next((r for r in rows if r["is_plain"]), None)
    for row in rows:
        if plain and plain["peak_per_ms"] > 0:
            row["peak_gain_vs_plain"] = row["peak_per_ms"] / plain["peak_per_ms"]
        if plain and plain["energy_per_ms"] > 0:
            row["energy_ratio_vs_plain"] = row["energy_per_ms"] / plain["energy_per_ms"]

    # Re-read the winner at half its exposure: the ranking is only valid if the
    # sensor is linear, and this is the cheapest place to find out it is not.
    best = max(
        (r for r in rows if r["focal_mm"] and not math.isinf(float(r["focal_mm"]))),
        key=lambda r: r["peak_per_ms"],
        default=None,
    )
    linearity = None
    if best is not None and best["exposure_ms"] * 0.5 >= 0.02:
        with Santec(slm_number=1, wavelength=1064, video_mode=0) as slm:
            with create_camera("daheng", 0, exposure_time_ms=best["exposure_ms"]) as cam:
                rig = Rig(cam=cam, slm=slm)
                rig.apply(
                    fresnel_panel(
                        float(best["focal_mm"]),
                        radius=450,
                        pupil_center=(960, 690),
                        panel_shape=PANEL,
                    )
                )
                half = best["exposure_ms"] * 0.5
                rig.set_exposure(half)
                rig.read()
                peak_half = robust_peak(rig.average(int(args.frames)))
                linearity = (peak_half / half) / best["peak_per_ms"]
                rig.apply_flat()
    payload["rows"] = rows
    payload["linearity_at_best"] = linearity

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out / f"verify_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n  f (mm)     exp     peak   peak/ms    fwhm   E/ms      in_box  gain")
    for row in rows:
        print(
            f"  {row['focal_mm'] if row['focal_mm'] else 'plain':>6}"
            f"{row['exposure_ms']:9.3f}{row['peak']:8.1f}{row['peak_per_ms']:9.1f}"
            f"{row['fwhm_px']:9.1f}{row['energy_per_ms']:10.0f}"
            f"{row['in_box_frac']:9.3f}"
            f"{row.get('peak_gain_vs_plain', 0.0):7.2f}"
        )
    print(f"\nlinearity at the best non-plain point: {linearity}")
    logger.info("wrote {}", path)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())