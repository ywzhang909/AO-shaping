"""Capture the dark baseline and the SLM gray-0 far-field illumination reference.

Why this script exists
----------------------
Both #237 (2D registration + measured PSF + gain field + complex-amplitude
stray light) and #238 (per-pixel gray->phase LUT + spatial crosstalk kernel)
need a measurement that the 11-10 diagnostics did not produce: a **true dark
frame** and an **SLM gray-0 far-field illumination reference** (not a uniform
flat).

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

The ``flat`` condition below is **not** that uniform flat. It is the SLM gray-0
far-field image: a spatial illumination reference. The illumination is
non-uniform, so it cannot separate pixel gain from illumination structure and
must not be used as a pixel-gain field input.

Protocol
--------
* **dark**: SLM parked at uniform gray 0 *and* the beam blocked by the
  operator, so the frame contains the camera's own pedestal and noise only.
  ``--execute --condition dark`` requires ``--beam-blocked``. Without the
  operator blocking the beam this is **not** a dark frame, and the script
  records ``beam_blocked=false`` so the data cannot be mistaken for one
  downstream.
* **flat**: SLM at uniform gray 0 with the beam **on** (``--beam-blocked`` is
  rejected for this condition). This is the SLM gray-0 *far-field illumination
  reference*: the spatial distribution of light that reaches the camera with
  the SLM parked, not a uniform field. It is insufficient on its own to
  identify the per-pixel gain -- gain cannot be separated from the non-uniform
  illumination -- so ``gain_field_valid`` is recorded as false. It also does
  not by itself establish the #238 zero-order reference.

Exposure discipline
-------------------
``set_camera_exposure_ms`` returns the *requested* value, so it is never used
as the effective exposure. After setting it, ``--execute`` calls
``get_camera_exposure_ms`` and records ``exposure_ms_readback`` together with
its source: the backend's live ``get_exposure_time_ms()`` device readback where
one exists, otherwise an explicitly labelled cache fallback. The verified value,
not the request, is what the dark calibration is keyed by. Every frame keeps its
raw 8-bit values and its own UTC and monotonic start/end timestamps; the
saturation count is computed against the known uint8 full scale (255), not
against "equals this frame's max", so a faint frame is never mislabelled as
saturated.

What this script deliberately does not do
------------------------------------------
* It does not block the beam itself. A human does that; ``--execute
  --condition dark`` refuses to run without ``--beam-blocked``, and the flag is
  recorded in the summary.
* It does not fit a gain field or a PSF. It produces the dark frame those fits
  need and an SLM gray-0 far-field illumination reference; the latter is not a
  uniform flat, so it is not by itself a gain-field fit input.

Dark calibration cache
----------------------
``--condition dark --execute`` is a **per-camera one-time calibration keyed by
the camera serial number**:

* the calibration is cached at ``data/calib/dark/<camera_sn>.npz`` with the
  per-pixel ``mean`` frame (fixed pattern/background: dark current + bias +
  fixed hot spots) and ``median`` frame (robust: a hot pixel or a single-frame
  transient does not move it), plus the exposure, both serials, capture time
  and the script's sha256;
* a cached calibration is reused **only for dark** and only when the serial,
  the exposure (within 10% or 0.005 ms, whichever is looser) and the frame
  shape agree (the shape is checked whenever the camera reports it without
  being opened). A reuse opens neither the camera nor the SLM, captures
  nothing, and writes a ``dark_calib_<sn>.npz`` copy into the run directory;
* a serial or exposure mismatch captures a fresh dark and overwrites the
  cache; a camera without a serial number is refused because the cache cannot
  be keyed;
* ``flat`` is never cache-keyed: the illumination reference depends on that
  day's beam and geometry, so it is always captured fresh.

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

    # plan only (default condition: dark)
    python scripts/slm/capture_dark_flat_baselines.py --out data/debug/dark_flat/20261111_xxxx
    # capture the dark frame (operator-confirmed blocked beam)
    python scripts/slm/capture_dark_flat_baselines.py --out ... --execute --condition dark --beam-blocked
    # capture the SLM gray-0 far-field illumination reference (beam on)
    python scripts/slm/capture_dark_flat_baselines.py --out ... --execute --condition flat
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

#: One-time dark calibration cache, keyed by the camera serial number.
DARK_CALIB_DIR = Path("data/calib/dark")

#: A cached dark calibration is reused when the exposure agrees within this
#: relative tolerance, or within this absolute tolerance in ms, whichever is
#: looser (the SDK quantises and clamps the request).
DARK_EXPOSURE_REL_TOL = 0.10
DARK_EXPOSURE_ABS_TOL_MS = 0.005


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


def _condition_entry(condition: str) -> dict:
    """Accurate description of the single condition a run captures."""
    if condition == "dark":
        return {
            "name": "dark",
            "slm_state": "uniform gray 0",
            "beam": "blocked by the operator (required via --beam-blocked at execute)",
            "purpose": (
                "camera pedestal, read noise, and fixed hot spots; separates "
                "additive background from complex-amplitude stray light (#237 (5))"
            ),
        }
    return {
        "name": "flat",
        "slm_state": "uniform gray 0",
        "beam": "on, unblocked (--beam-blocked is rejected for flat)",
        "purpose": (
            "SLM gray-0 far-field illumination reference: the spatial distribution "
            "of light reaching the camera with the SLM parked. The illumination is "
            "non-uniform, so this is not a uniform flat and is insufficient on its "
            "own to identify per-pixel gain; it is not a valid gain-field input and "
            "does not by itself fix the #238 zero-order reference."
        ),
        "gain_field_valid": False,
    }


def _plan(
    args: argparse.Namespace,
    camera_sn: str | None = None,
    slm_sn: str | None = None,
    camera_sn_source: str | None = None,
    slm_sn_source: str | None = None,
) -> dict:
    """The plan document, printed in plan-only mode and written when executed."""
    return {
        "script": "scripts/slm/capture_dark_flat_baselines.py",
        "issues": ["#237", "#238"],
        "mode": "execute" if args.execute else "plan-only",
        "slm_number": args.slm_number,
        "cam_id": args.cam_id,
        "camera_sn": camera_sn,
        "camera_sn_source": camera_sn_source
        or "unknown in plan-only: the device is not opened",
        "slm_sn": slm_sn,
        "slm_sn_source": slm_sn_source
        or "unknown in plan-only: the device is not opened",
        "exposure_ms_requested": args.exposure_ms,
        "exposure_readback_verified": False,
        "exposure_readback_note": (
            "plan-only records no readback; --execute reads the effective "
            "exposure via get_camera_exposure_ms() and records its source "
            "(device get_exposure_time_ms() where the backend has it)"
        ),
        "frames_per_condition": args.frames,
        "condition": args.condition,
        "beam_blocked": args.beam_blocked,
        "conditions": [_condition_entry(args.condition)],
        "dark_calibration_cache": {
            "applies_to": "dark only",
            "path": str(DARK_CALIB_DIR / "<camera_sn>.npz"),
            "reuse_rule": (
                "serial + frame shape (when known without opening) and exposure "
                "within %g%% or %g ms; flat is never cache-keyed"
                % (DARK_EXPOSURE_REL_TOL * 100, DARK_EXPOSURE_ABS_TOL_MS)
            ),
        },
        "outputs": [
            "plan.json -- this plan",
            "frames/<condition>_<index>.npy -- raw uint8 frames, unquantised",
            "frames/<condition>_<index>.png -- 8-bit preview (convenience only)",
            "dark_calib_<camera_sn>.npz -- dark calibration copy (dark only)",
            "summary.json -- per-frame stats, provenance and the gap list",
            "release.json -- per-step device release status",
        ],
        "discipline": [
            "plan-only by default: no device is opened and no directory is created",
            "the bench device lock is taken before any hardware is touched",
            "a matching dark cache is reused without opening any device; flat is "
            "always captured fresh",
            "the SLM returns to uniform gray 0 and is closed, and the camera is "
            "closed, in finally; each step is recorded separately in release.json",
        ],
    }


def _capture_frames(cam, condition: str, count: int, out: Path) -> list[dict]:
    """Capture ``count`` frames of one condition and save each one raw.

    Each frame records its own UTC and monotonic start/end time: UTC is the
    wall-clock provenance, monotonic is the only safe source for the duration
    (a system clock step must not produce a negative interval).
    """
    records: list[dict] = []
    for index in range(count):
        started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        started_monotonic_s = time.monotonic()
        frame = np.asarray(cam.get_numpy_image(1, True))
        ended_monotonic_s = time.monotonic()
        ended_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
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
                "started_utc": started_utc,
                "ended_utc": ended_utc,
                "started_monotonic_s": started_monotonic_s,
                "ended_monotonic_s": ended_monotonic_s,
                "duration_s": ended_monotonic_s - started_monotonic_s,
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


def _usable_as(condition: str) -> str:
    if condition == "dark":
        return "dark-frame pedestal/noise/hot-spot statistics"
    return "SLM gray-0 far-field illumination reference; not a pixel-gain field"


def _dark_gaps(peak: float, mean_level: float, beam_blocked: bool) -> list[str]:
    """Gaps that make a dark calibration unusable for its declared purpose."""
    gaps: list[str] = []
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
    return gaps


def _condition_diagnostics(condition: str, frames: list[dict], beam_blocked: bool) -> dict:
    """Summarise one condition and flag the gaps that block a downstream fit."""
    if not frames:
        result = {
            "frames": 0,
            "usable": False,
            "usable_as": _usable_as(condition),
            "gaps": ["no frames were captured"],
        }
        if condition == "flat":
            result["gain_field_valid"] = False
        return result

    maxima = [f["stats"]["max"] for f in frames]
    means = [f["stats"]["mean"] for f in frames]
    peak = max(maxima)
    mean_level = float(np.mean(means))

    gaps: list[str] = []
    if condition == "dark":
        gaps.extend(_dark_gaps(peak, mean_level, beam_blocked))
    else:
        if beam_blocked:
            gaps.append(
                "beam_blocked=true on a flat frame: the flat must be captured "
                "with the beam on"
            )
        if peak >= FULL_SCALE:
            gaps.append(
                f"flat peak {peak} reached full scale: the illumination reference "
                "is saturated and must be re-exposed before it is usable as an "
                "illumination reference"
            )

    peak_spread = max(maxima) - min(maxima)
    result = {
        "frames": len(frames),
        "usable": not gaps,
        "usable_as": _usable_as(condition),
        "peak": peak,
        "mean_level": mean_level,
        "peak_spread": peak_spread,
        "peak_spread_note": (
            "the spread of the per-frame peak; a large value means the "
            "illumination or the exposure was not stable across the burst"
        ),
        "gaps": gaps,
    }
    if condition == "flat":
        result["gain_field_valid"] = False
    return result


def _script_sha256() -> str:
    """sha256 of this script, recorded in the cache for provenance."""
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def _camera_serial(cam, cam_id: int) -> tuple[str | None, str]:
    """Resolve the camera serial without opening the device when possible.

    ``DahengCamera.sn`` is ``None`` until the camera is opened, so fall back to
    the backend's device enumeration, which lists serials without opening a
    device. Returns ``(serial, source)``; ``serial`` is ``None`` when neither
    route yields one, and the caller must then refuse (the cache is keyed by
    serial).
    """
    sn = getattr(cam, "sn", None)
    if isinstance(sn, str) and sn.strip():
        return sn.strip(), "cam.sn"
    lister = getattr(cam, "get_cam_list", None)
    if callable(lister):
        try:
            infos = lister()
            entry = (
                infos[cam_id]
                if isinstance(infos, list) and len(infos) > cam_id
                else None
            )
            if isinstance(entry, dict):
                sn = entry.get("sn")
        except Exception as exc:
            print("camera SN enumeration warning: %s" % exc, file=sys.stderr)
    if isinstance(sn, str) and sn.strip():
        return sn.strip(), "cam.get_cam_list() enumeration (device not opened)"
    return None, "unavailable"


def _current_frame_shape(cam) -> tuple[int, int] | None:
    """The camera's frame shape when it is known before a capture, else None."""
    width = int(getattr(cam, "cam_width", 0) or 0)
    height = int(getattr(cam, "cam_height", 0) or 0)
    if width > 0 and height > 0:
        return (height, width)
    return None


def _exposure_agrees(cached_ms: float, requested_ms: float) -> bool:
    """Whether a cached exposure is close enough to this run's request."""
    diff = abs(float(cached_ms) - float(requested_ms))
    return diff < DARK_EXPOSURE_ABS_TOL_MS or diff <= (
        DARK_EXPOSURE_REL_TOL * abs(float(requested_ms))
    )


def _load_dark_calibration(path: Path) -> dict | None:
    """Load and validate one cache file; ``None`` when absent or unusable."""
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            required = (
                "mean",
                "median",
                "n_frames",
                "exposure_ms",
                "camera_sn",
                "slm_sn",
                "capture_utc",
                "peak",
                "mean_level",
                "script_sha256",
            )
            missing = [key for key in required if key not in data.files]
            if missing:
                print(
                    "dark cache ignored (missing %s): %s"
                    % (", ".join(missing), path),
                    file=sys.stderr,
                )
                return None
            mean = np.asarray(data["mean"], dtype=np.float32)
            median = np.asarray(data["median"], dtype=np.float32)
            if mean.ndim != 2 or mean.shape != median.shape:
                print(
                    "dark cache ignored (mean/median shape mismatch): %s" % path,
                    file=sys.stderr,
                )
                return None
            return {
                "mean": mean,
                "median": median,
                "n_frames": int(data["n_frames"]),
                "exposure_ms": float(data["exposure_ms"]),
                "camera_sn": str(data["camera_sn"].item()),
                "slm_sn": str(data["slm_sn"].item()),
                "capture_utc": str(data["capture_utc"].item()),
                "peak": int(data["peak"]),
                "mean_level": float(data["mean_level"]),
                "script_sha256": str(data["script_sha256"].item()),
                "path": str(path),
            }
    except Exception as exc:
        print("dark cache unreadable (%s): %s" % (exc, path), file=sys.stderr)
        return None


def _dark_cache_match(
    calib: dict,
    camera_sn: str,
    requested_ms: float,
    frame_shape: tuple[int, int] | None,
) -> tuple[bool, str]:
    """Decide whether a cache entry can stand in for this dark run."""
    if calib["camera_sn"] != camera_sn:
        return False, "serial mismatch"
    if not _exposure_agrees(calib["exposure_ms"], requested_ms):
        return False, "exposure changed"
    if frame_shape is not None and tuple(calib["mean"].shape) != tuple(frame_shape):
        return False, "frame size mismatch"
    return True, "serial+exposure match"


def _write_dark_calibration(path: Path, mean, median, meta: dict) -> None:
    """Atomic write: same-directory ``.tmp`` file, then ``replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as handle:
            np.savez(
                handle,
                mean=np.asarray(mean, dtype=np.float32),
                median=np.asarray(median, dtype=np.float32),
                n_frames=np.int64(meta["n_frames"]),
                exposure_ms=np.float64(meta["exposure_ms"]),
                camera_sn=np.array(meta["camera_sn"]),
                slm_sn=np.array(meta["slm_sn"]),
                capture_utc=np.array(meta["capture_utc"]),
                peak=np.int64(meta["peak"]),
                mean_level=np.float64(meta["mean_level"]),
                script_sha256=np.array(meta["script_sha256"]),
            )
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _calibration_stats(frame) -> dict:
    values = np.asarray(frame)
    return {
        "shape": list(values.shape),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
    }


def _calibration_summary(calib: dict, output_copy: str | None) -> dict:
    """The declared purpose and provenance of both calibration frames."""
    return {
        "cache_path": calib["path"],
        "output_copy": output_copy,
        "n_frames": calib["n_frames"],
        "exposure_ms": calib["exposure_ms"],
        "capture_utc": calib["capture_utc"],
        "script_sha256": calib["script_sha256"],
        "mean_frame": "float32 per-pixel mean of the raw frames",
        "mean_use": (
            "fixed-pattern/background calibration for this camera: dark current, "
            "bias/pedestal and fixed hot spots"
        ),
        "median_frame": "float32 per-pixel median of the raw frames",
        "median_use": (
            "robust background for hot-spot/transient rejection: a hot pixel or a "
            "single-frame transient does not move the median"
        ),
        "mean_stats": _calibration_stats(calib["mean"]),
        "median_stats": _calibration_stats(calib["median"]),
        "mean_sha256": _sha256_array(np.asarray(calib["mean"])),
        "median_sha256": _sha256_array(np.asarray(calib["median"])),
    }


def _cached_dark_diagnostics(calib: dict, beam_blocked: bool) -> dict:
    """Diagnostics for a dark calibration loaded from cache, not captured now."""
    gaps = _dark_gaps(calib["peak"], calib["mean_level"], beam_blocked)
    return {
        "frames": calib["n_frames"],
        "usable": not gaps,
        "usable_as": _usable_as("dark"),
        "peak": calib["peak"],
        "mean_level": calib["mean_level"],
        "peak_spread": None,
        "peak_spread_note": "not available from the cached calibration",
        "gaps": gaps,
        "reused_from_cache": True,
        "cache_path": calib["path"],
    }


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
    failures: list[dict] = []
    release: dict = {
        "slm_flat_gray0": "not-attempted",
        "slm_closed": "not-attempted",
        "cam_closed": "not-attempted",
    }

    slm = None
    cam = None
    slm_opened = False
    cam_opened = False
    condition = args.condition
    dark_reused = False
    dark_updated = False
    dark_calibration: dict | None = None
    cache_path: Path | None = None
    camera_sn: str | None = None
    camera_sn_source = "unavailable"
    slm_sn: str | None = None
    slm_sn_source = "unavailable"
    exposure_ms = float(args.exposure_ms)
    exposure_ms_source = "requested"
    exposure_ms_readback: float | None = None
    exposure_readback_verified = False
    exposure_readback_note = (
        "not attempted: no device was opened (cached dark reuse)"
    )
    reuse_reason = (
        "not applicable: flat is never cache-keyed"
        if condition == "flat"
        else "no cache"
    )
    conditions: dict[str, list[dict]] = {}
    diagnostics: dict[str, dict] = {}
    t0 = time.time()
    try:
        # Create the output tree inside the try: a failure here (for example an
        # existing directory) must still reach finally and release the bench lock.
        out.mkdir(parents=True, exist_ok=False)
        (out / "frames").mkdir(parents=True, exist_ok=False)

        # --- dark: try the one-time per-camera calibration cache first --------
        if condition == "dark":
            prepare_camera_sdk("daheng")
            cam = create_camera(
                "daheng", args.cam_id, exposure_time_ms=args.exposure_ms
            )
            camera_sn, camera_sn_source = _camera_serial(cam, args.cam_id)
            if camera_sn is not None:
                cache_path = DARK_CALIB_DIR / ("%s.npz" % camera_sn)
                cached = _load_dark_calibration(cache_path)
                if cached is None:
                    use_cache, reuse_reason = False, "no cache"
                else:
                    use_cache, reuse_reason = _dark_cache_match(
                        cached,
                        camera_sn,
                        args.exposure_ms,
                        _current_frame_shape(cam),
                    )
                if use_cache:
                    dark_reused = True
                    dark_calibration = cached
                    exposure_ms = cached["exposure_ms"]
                    exposure_ms_source = "cache metadata (device not opened)"
                    slm_sn = cached["slm_sn"]
                    slm_sn_source = "cache metadata (SLM not opened on reuse)"
                    cam = None  # nothing was opened, so nothing to close
                    print(
                        "dark calibration reuse: %s (%s); no device opened, "
                        "no frames captured" % (cache_path, reuse_reason)
                    )
            if camera_sn is None:
                # Refuse before opening the SLM: a serial-less camera cannot key
                # the one-time dark calibration, and opening hardware to discover
                # that would be pointless.
                print(
                    "ERROR: camera serial number unavailable; the dark calibration "
                    "is keyed by camera SN and cannot be stored or matched. Refusing.",
                    file=sys.stderr,
                )
                return 2

        # --- devices: only when the cache did not stand in --------------------
        if not dark_reused:
            if condition == "flat":
                prepare_camera_sdk("daheng")
                cam = create_camera(
                    "daheng", args.cam_id, exposure_time_ms=args.exposure_ms
                )
            slm = Santec(slm_number=args.slm_number, video_mode=0)
            slm.open()
            slm_opened = True
            require_drive_board_entry(slm)
            slm_sn = slm.get_serial_number()
            if not slm_sn:
                print(
                    "ERROR: SLM serial number unavailable; the calibration "
                    "provenance cannot be recorded. Refusing.",
                    file=sys.stderr,
                )
                return 2
            slm_sn_source = "slm.get_serial_number()"
            print("SLM #%d open (drive board OK), sn=%s" % (args.slm_number, slm_sn))

            cam.open()
            cam_opened = True
            set_camera_exposure_ms(cam, args.exposure_ms)
            exposure_ms = float(args.exposure_ms)
            exposure_ms_source = (
                "requested via set_camera_exposure_ms; readback below is authoritative"
            )
            print("camera open, exposure set to %.4f ms; reading it back" % args.exposure_ms)
            try:
                readback = float(get_camera_exposure_ms(cam))
            except Exception as exc:
                exposure_readback_note = (
                    "get_camera_exposure_ms failed: %s" % exc
                )
                print(
                    "exposure readback warning: %s" % exc,
                    file=sys.stderr,
                )
            else:
                exposure_ms_readback = readback
                exposure_ms = readback
                if callable(getattr(cam, "get_exposure_time_ms", None)):
                    exposure_readback_note = (
                        "cam.get_exposure_time_ms() live device readback via "
                        "get_camera_exposure_ms"
                    )
                else:
                    exposure_readback_note = (
                        "driver-cache fallback via get_camera_exposure_ms; this "
                        "backend exposes no device-side readback"
                    )
                exposure_readback_verified = callable(
                    getattr(cam, "get_exposure_time_ms", None)
                )
                print(
                    "exposure readback %.4f ms (%s)"
                    % (readback, exposure_readback_note)
                )

            actual_sn = getattr(cam, "sn", None)
            if isinstance(actual_sn, str) and actual_sn.strip():
                if camera_sn is not None and actual_sn.strip() != camera_sn:
                    print(
                        "ERROR: camera serial changed while opening (%r -> %r); "
                        "refusing to mix calibrations." % (camera_sn, actual_sn),
                        file=sys.stderr,
                    )
                    return 2
                camera_sn = actual_sn.strip()
                camera_sn_source = "cam.sn (opened)"
            if camera_sn is None:
                camera_sn, camera_sn_source = _camera_serial(cam, args.cam_id)
            if camera_sn is None and condition == "dark":
                print(
                    "ERROR: camera serial number unavailable; the dark calibration "
                    "is keyed by camera SN and cannot be stored or matched. Refusing.",
                    file=sys.stderr,
                )
                return 2
            if camera_sn is None:
                print(
                    "warning: camera serial unavailable; flat is not cache-keyed, "
                    "so it is recorded as null.",
                    file=sys.stderr,
                )
                camera_sn_source = "unavailable (flat does not need a key)"

            if condition == "dark":
                cache_path = DARK_CALIB_DIR / ("%s.npz" % camera_sn)
                cached = _load_dark_calibration(cache_path)
                if cached is not None:
                    use_cache, reuse_reason = _dark_cache_match(
                        cached,
                        camera_sn,
                        args.exposure_ms,
                        _current_frame_shape(cam),
                    )
                    if use_cache:
                        dark_reused = True
                        dark_calibration = cached
                        exposure_ms = cached["exposure_ms"]
                        exposure_ms_source = (
                            "cache metadata (device opened for serial/shape check, "
                            "no capture)"
                        )
                        exposure_readback_note = (
                            "device readback of the requested exposure; the reused "
                            "calibration keeps its own cached exposure_ms"
                        )
                        print(
                            "dark calibration reuse: %s (%s); no frames captured"
                            % (cache_path, reuse_reason)
                        )

        if not dark_reused:
            print(
                "condition '%s': SLM at uniform gray 0, beam %s"
                % (condition, "BLOCKED" if condition == "dark" else "ON")
            )
            slm.set_grayscale(0)
            time.sleep(args.settle_s)
            conditions = {
                condition: _capture_frames(cam, condition, args.frames, out)
            }
            diagnostics = {
                condition: _condition_diagnostics(
                    condition, conditions[condition], args.beam_blocked
                )
            }
            if condition == "dark":
                stack = np.stack(
                    [
                        np.asarray(np.load(out / record["frame_file"]))
                        for record in conditions["dark"]
                    ]
                )
                mean = np.mean(stack, axis=0).astype(np.float32)
                median = np.median(stack, axis=0).astype(np.float32)
                diag = diagnostics["dark"]
                dark_calibration = {
                    "mean": mean,
                    "median": median,
                    "n_frames": len(conditions["dark"]),
                    "exposure_ms": exposure_ms,
                    "camera_sn": camera_sn,
                    "slm_sn": slm_sn,
                    "capture_utc": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                    "peak": diag["peak"],
                    "mean_level": diag["mean_level"],
                    "script_sha256": _script_sha256(),
                    "path": str(cache_path),
                }
                _write_dark_calibration(
                    cache_path, mean, median, dark_calibration
                )
                dark_updated = True
                print(
                    "dark calibration written: %s (%d frames, atomic replace)"
                    % (cache_path, dark_calibration["n_frames"])
                )
        else:
            diagnostics = {
                "dark": _cached_dark_diagnostics(dark_calibration, args.beam_blocked)
            }

        dark_output_copy = None
        if condition == "dark" and dark_calibration is not None:
            dark_output_copy = "dark_calib_%s.npz" % camera_sn
            _write_dark_calibration(
                out / dark_output_copy,
                dark_calibration["mean"],
                dark_calibration["median"],
                dark_calibration,
            )

        summary = {
            "script": "scripts/slm/capture_dark_flat_baselines.py",
            "issues": ["#237", "#238"],
            "mode": "execute",
            "slm_number": args.slm_number,
            "cam_id": args.cam_id,
            "camera_sn": camera_sn,
            "camera_sn_source": camera_sn_source,
            "slm_sn": slm_sn,
            "slm_sn_source": slm_sn_source,
            "exposure_ms_requested": args.exposure_ms,
            "exposure_ms": exposure_ms,
            "exposure_ms_source": exposure_ms_source,
            "exposure_ms_readback": exposure_ms_readback,
            "exposure_readback_verified": exposure_readback_verified,
            "exposure_readback_note": exposure_readback_note,
            "beam_blocked": args.beam_blocked,
            "beam_blocked_source": (
                "operator flag --beam-blocked (dark only)"
                if args.condition == "dark"
                else "not applicable: flat requires the beam on"
            ),
            "condition": args.condition,
            "frames_per_condition": args.frames,
            "frames_captured": sum(len(v) for v in conditions.values()),
            "settle_s": args.settle_s,
            "dark_reused": dark_reused,
            "dark_updated": dark_updated,
            "dark_cache_path": (
                str(cache_path) if condition == "dark" else None
            ),
            "reuse_reason": reuse_reason,
            "dark_calibration": (
                _calibration_summary(dark_calibration, dark_output_copy)
                if dark_calibration is not None
                else None
            ),
            "conditions": conditions,
            "diagnostics": diagnostics,
            "gaps": [gap for diag in diagnostics.values() for gap in diag["gaps"]],
            "failures": failures,
            "elapsed_s": time.time() - t0,
            "done_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        (out / "plan.json").write_text(
            json.dumps(
                _plan(
                    args,
                    camera_sn=camera_sn,
                    slm_sn=slm_sn,
                    camera_sn_source=camera_sn_source,
                    slm_sn_source=slm_sn_source,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
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
        if slm is not None and slm_opened:
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
        if cam is not None and cam_opened:
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
        # Report what actually happened, per step; "not-attempted" only means the
        # device was never opened. A failed close must not read as "released".
        statuses = ", ".join("%s=%s" % (key, value) for key, value in release.items())
        complete = all(value in ("ok", "not-attempted") for value in release.values())
        print(
            ("released: " if complete else "release INCOMPLETE: ") + statuses,
            file=sys.stdout if complete else sys.stderr,
        )
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
        "--condition",
        choices=("dark", "flat"),
        default="dark",
        help="the single condition to capture (default: dark)",
    )
    ap.add_argument(
        "--beam-blocked",
        action="store_true",
        help=(
            "operator confirms the beam is blocked; required for "
            "--execute --condition dark and rejected for flat"
        ),
    )
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args(argv)

    if not args.execute:
        print(json.dumps(_plan(args), ensure_ascii=False, indent=2))
        return 0

    if args.condition == "dark" and not args.beam_blocked:
        print(
            "ERROR: --execute --condition dark requires --beam-blocked "
            "(the operator must confirm the beam is blocked).",
            file=sys.stderr,
        )
        return 2
    if args.condition == "flat" and args.beam_blocked:
        print(
            "ERROR: --execute --condition flat forbids --beam-blocked; "
            "the flat must be captured with the beam on.",
            file=sys.stderr,
        )
        return 2

    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
