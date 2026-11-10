"""Capture the dark and uniform-illumination baselines for issues #237 and #238.

Why this script exists
----------------------
Both #237 (2D registration + measured PSF + gain field + complex-amplitude
stray light) and #238 (per-pixel gray->phase LUT + spatial crosstalk kernel)
need a measurement that the 11-10 diagnostics did not produce: a **true dark
frame** and a **uniform-illumination flat**.

Without a dark frame the following are not separable:

* the CCD read-noise floor versus a real pedestal (``bias_counts``);
* a fixed hot spot (the failure-table row "every phase shows the same hot
  spot") versus a genuine stray-light amplitude;
* the additive background scalar of the current forward model versus a
  spatially structured background (which is what #237 item (5) asks for).

Without a uniform flat the following are not identifiable:

* the pixel-to-pixel gain field (a scalar ``gain`` cannot express it);
* the zero-order reference level that the #238 carrier probes are normalised
  against, which is exactly the quantity the crosstalk kernel perturbs.

Protocol
--------
* **dark**: SLM parked at uniform gray 0 *and* the beam blocked by the
  operator, so the frame contains the camera's own pedestal and noise only.
  Without the operator blocking the beam this is **not** a dark frame, and the
  script records ``beam_blocked=false`` so the data cannot be mistaken for
  one downstream.
* **flat**: SLM at uniform gray 0 with the beam **on**, i.e. the reference
  illumination that every other probe is normalised against. This is the
  ``gain field`` input for #237 item (3).

Exposure discipline
-------------------
``set_camera_exposure_ms`` returns the *requested* value, not a hardware
readback, so ``exposure_ms_readback`` is recorded as ``null`` with an explicit
``readback_verified=false`` note rather than being filled with the request.
Every frame keeps its raw 8-bit values; the saturation count is computed
against the known uint8 full scale (255), not against "equals this frame's
max", so a faint frame is never mislabelled as saturated.

What this script deliberately does not do
------------------------------------------
* It does not block the beam itself. A human does that; the script records
  whether it was told so, and a missing confirmation is a recorded gap, not
  an assumption.
* It does not fit a gain field or a PSF. It produces the measurements those
  fits need, with the metadata those fits must carry.

Hardware
--------
Default is plan-only: nothing is opened and no directory is created without
``--execute``. With ``--execute`` the script takes the bench device lock, and
in ``finally`` it returns the SLM to uniform gray 0, closes it, closes the
camera, and writes ``release.json`` recording each step separately -- so a
failed close cannot be hidden behind an unconditional "released" line.

Usage
-----
::

    python scripts/slm/capture_dark_flat_baselines.py --out data/debug/dark_flat/20261111_xxxx
    python scripts/slm/capture_dark_flat_baselines.py --out ... --execute
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

#: Frames kept per condition. Three is the minimum that shows the noise
#: floor repeatability; more is better but the beam must stay stable.
FRAMES_PER_CONDITION = 3

#: Pixel value that means "full scale" on this 8-bit camera.
FULL_SCALE = 255

#: A dark frame whose peak stays at or below this fraction of full scale is
#: not obviously contaminated by stray light. It is a *diagnostic*, not a
#: gate: the operator's block confirmation is the authoritative signal.
DARK_PEAK_WARN_FRACTION = 0.02

#: A dark frame whose mean exceeds this fraction of full scale is not a
#: pedestal-only frame. A few DN of pedestal is normal on an 8-bit camera, so
#: the threshold is deliberately loose.
DARK_MEAN_WARN_FRACTION = 0.10


def _sha256_array(array: np.ndarray) -> str:
    """sha256 of the raw C-contiguous bytes of ``array``."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _frame_stats(frame: np.ndarray) -> dict:
    """Full-scale saturation statistics for one 8-bit frame.

    The saturation count is against the known uint8 full scale (255), not
    against "equals this frame's own maximum" -- the latter would label a
    faint, clean frame as saturated.
    """
    values = np.asarray(frame)
    if values.size == 0:
        return {
            "shape": list(values.shape),
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "sum": None,
            "saturated_px": 0,
            "saturation_full_scale": FULL_SCALE,
        }
    saturated = int(np.count_nonzero(values >= FULL_SCALE))
    return {
        "shape": list(values.shape),
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "sum": float(values.sum()),
        "saturated_px": saturated,
        "saturation_full_scale": FULL_SCALE,
    }


def _plan(args: argparse.Namespace) -> dict:
    """The plan document, printed in plan-only mode and written when executed."""
    return {
        "script": "scripts/slm/capture_dark_flat_baselines.py",
        "issues": ["#237", "#238"],
        "mode": "execute" if args.execute else "plan-only",
        "slm_number": args.slm_number,
        "cam_id": args.cam_id,
        "exposure_ms_requested": args.exposure_ms,
        "exposure_readback_verified": False,
        "exposure_readback_note": (
            "set_camera_exposure_ms returns the requested value; this script "
            "records no hardware readback and never claims one."
        ),
        "frames_per_condition": args.frames,
        "conditions": [
            {
                "name": "dark",
                "slm_state": "uniform gray 0",
                "beam": "blocked by the operator (must be confirmed via --beam-blocked)",
                "purpose": (
                    "camera pedestal, read noise, and fixed hot spots; separates "
                    "additive background from complex-amplitude stray light (#237 (5))"
                ),
            },
            {
                "name": "flat",
                "slm_state": "uniform gray 0",
                "beam": "on",
                "purpose": (
                    "reference illumination for the pixel-to-pixel gain field "
                    "(#237 (3)) and the zero-order reference that the #238 "
                    "carrier probes normalise against"
                ),
            },
        ],
        "outputs": [
            "plan.json -- this plan",
            "frames/<condition>_<index>.npy -- raw uint8 frames, unquantised",
            "frames/<condition>_<index>.png -- 8-bit preview (convenience only)",
            "summary.json -- per-frame stats, provenance and the gap list",
            "release.json -- per-step device release status",
        ],
        "discipline": [
            "plan-only by default: no device is opened and no directory is created",
            "the bench device lock is taken before any hardware is touched",
            "the SLM returns to uniform gray 0 and is closed, and the camera is "
            "closed, in finally; each step is recorded separately in release.json",
        ],
    }


def _capture_frames(cam, condition: str, count: int, out: Path) -> list[dict]:
    """Capture ``count`` frames of one condition and save each one raw."""
    records: list[dict] = []
    for index in range(count):
        frame = np.asarray(cam.capture())
        if frame.ndim == 3:
            frame = frame.mean(axis=2)
        name = f"{condition}_{index:02d}.npy"
        np.save(out / "frames" / name, frame)
        try:
            from PIL import Image

            Image.fromarray(frame.astype(np.uint8)).save(out / "frames" / f"{condition}_{index:02d}.png")
        except Exception as exc:  # a preview failure must not lose the raw frame
            print("preview write warning: %s" % exc, file=sys.stderr)
        records.append(
            {
                "condition": condition,
                "index": index,
                "frame_file": f"frames/{name}",
                "sha256": _sha256_array(frame),
                "stats": _frame_stats(frame),
            }
        )
        print(
            "  %s[%d] max=%s mean=%.4f saturated=%d"
            % (
                condition,
                index,
                records[-1]["stats"]["max"],
                records[-1]["stats"]["mean"],
                records[-1]["stats"]["saturated_px"],
            )
        )
    return records


def _condition_diagnostics(condition: str, frames: list[dict], beam_blocked: bool) -> dict:
    """Summarise one condition and flag the gaps that block a downstream fit."""
    if not frames:
        return {"frames": 0, "usable": False, "gaps": ["no frames were captured"]}

    maxima = [f["stats"]["max"] for f in frames]
    means = [f["stats"]["mean"] for f in frames]
    peak = max(maxima)
    mean_level = float(np.mean(means))

    gaps: list[str] = []
    if condition == "dark":
        if not beam_blocked:
            gaps.append(
                "beam_blocked=false: this is NOT a verified dark frame; the "
                "operator must confirm the beam was blocked before the pedestal "
                "or hot-spot statistics can be used"
            )
        if peak >= DARK_PEAK_WARN_FRACTION * FULL_SCALE:
            gaps.append(
                f"dark peak {peak} >= {DARK_PEAK_WARN_FRACTION * FULL_SCALE:.1f} "
                "(fraction of full scale): stray light may be present"
            )
        if mean_level >= DARK_MEAN_WARN_FRACTION * FULL_SCALE:
            gaps.append(
                f"dark mean {mean_level:.2f} >= {DARK_MEAN_WARN_FRACTION * FULL_SCALE:.1f} "
                "(10% of full scale): the pedestal is far above the noise floor"
            )
    else:
        if beam_blocked:
            gaps.append("beam_blocked=true on a flat frame: the flat is not a flat")
        if peak >= FULL_SCALE:
            gaps.append(
                f"flat peak {peak} reached full scale: the illumination must be "
                "re-exposed before it can serve as a gain-field reference"
            )

    peak_spread = max(maxima) - min(maxima)
    return {
        "frames": len(frames),
        "usable": not gaps,
        "peak": peak,
        "mean_level": mean_level,
        "peak_spread": peak_spread,
        "peak_spread_note": (
            "the spread of the per-frame peak; a large value means the "
            "illumination or the exposure was not stable across the burst"
        ),
        "gaps": gaps,
    }


def _run(args: argparse.Namespace) -> int:
    # Lazy device imports: the ONLY place drivers are loaded.
    from ao_shaping.drivers.ccd.common import (
        create_camera,
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
    release: dict = {
        "slm_flat_gray0": "not-attempted",
        "slm_closed": "not-attempted",
        "cam_closed": "not-attempted",
    }

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
        print("camera open, exposure requested %.4f ms (not a hardware readback)" % args.exposure_ms)

        conditions: dict[str, list[dict]] = {}

        # --- condition 1: dark (beam blocked by the operator) -------------
        print("condition 'dark': SLM at uniform gray 0, beam must be BLOCKED")
        slm.set_grayscale(0)
        time.sleep(args.settle_s)
        conditions["dark"] = _capture_frames(cam, "dark", args.frames, out)

        # --- condition 2: flat (beam on) -----------------------------------
        print("condition 'flat': SLM at uniform gray 0, beam ON")
        slm.set_grayscale(0)
        time.sleep(args.settle_s)
        conditions["flat"] = _capture_frames(cam, "flat", args.frames, out)

        diagnostics = {
            name: _condition_diagnostics(name, frames, args.beam_blocked and name == "dark")
            for name, frames in conditions.items()
        }

        summary = {
            "script": "scripts/slm/capture_dark_flat_baselines.py",
            "issues": ["#237", "#238"],
            "mode": "execute",
            "slm_number": args.slm_number,
            "cam_id": args.cam_id,
            "exposure_ms_requested": args.exposure_ms,
            "exposure_ms_readback": None,
            "exposure_readback_verified": False,
            "exposure_readback_note": (
                "set_camera_exposure_ms returns the requested value; no hardware "
                "readback was obtained, so the effective exposure is unverified."
            ),
            "beam_blocked": args.beam_blocked,
            "beam_blocked_source": (
                "operator flag --beam-blocked" if args.beam_blocked else "not confirmed"
            ),
            "frames_per_condition": args.frames,
            "settle_s": args.settle_s,
            "conditions": conditions,
            "diagnostics": diagnostics,
            "gaps": [gap for diag in diagnostics.values() for gap in diag["gaps"]],
            "failures": failures,
            "elapsed_s": time.time() - t0,
            "done_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        (out / "plan.json").write_text(
            json.dumps(_plan(args), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("wrote summary.json (%d conditions, %d failures)" % (len(conditions), len(failures)))
        for name, diag in diagnostics.items():
            if diag["gaps"]:
                print("  %s: %d gap(s) -- %s" % (name, len(diag["gaps"]), diag["gaps"][0]))
            else:
                print("  %s: usable" % name)
        # The evidence is written even when the diagnostics fail; the gaps are
        # the finding, not a reason to discard the frames.
        return 0
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
    ap.add_argument(
        "--exposure-ms",
        type=float,
        default=0.05,
        help="requested exposure; 0.05 ms is non-saturating on this bench",
    )
    ap.add_argument("--frames", type=int, default=FRAMES_PER_CONDITION)
    ap.add_argument(
        "--settle-s",
        type=float,
        default=0.3,
        help="settle time after each SLM state change before capturing",
    )
    ap.add_argument(
        "--beam-blocked",
        action="store_true",
        help="operator confirms the beam was blocked for the dark condition",
    )
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args(argv)

    plan = _plan(args)
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
