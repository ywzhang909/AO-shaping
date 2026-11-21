"""Issue #247: capture true flat / ±linear tilt / ±defocus full-frame Mono12 frames.

This script acquires a *deterministic, model-free* calibration set on the bench
(Santec SLM-200 #1 @1064 nm + Daheng MER2-507-23GM): a flat phase, real panel-coordinate
linear ramps along the two panel axes (both signs), quadratic defocus (both signs),
with baseline-flat repeats before and after for drift tracking.

Unlike the archived square-1000 diagnostic (whose ``RX``/``RY`` were 64-px-period
*cosine* tilts of the frozen model, not true ramps), here the phase is generated
directly in panel coordinates and displayed raw -- no correction application, no
model changes, no frozen checkpoint.

Device discipline (docs/dev/hardware_discipline.md):
* Without ``--execute`` this script NEVER imports a driver or opens a device. It
  builds the deterministic phase plan, records phase digests/parameters into
  ``plan.json``, and prints ``skip: requires --execute``.
* With ``--execute`` it opens the SLM and CCD exactly once in a single process,
  guarded by ``DeviceLock`` + ``--wait-until-idle``; ``require_drive_board()`` is a
  hard entry assertion. A ``finally`` always releases the devices and writes
  ``released.json``.
* Full native Mono12 is asserted by the *effective pixel-format readback*
  (``set_bit_depth(12)`` return + ``cam.bit_depth``), not by frame dtype alone
  (issue #246: MONO12 frames arrive in uint16 containers). Full scale is taken
  from ``cam.effective_full_scale()`` (4095), never from ``np.iinfo(dtype)``.
* A native frame at/above ``--saturation-fraction`` of full scale aborts the
  session (after the current frame is saved), so no saturated run is trusted.
* The CCD is never physically windowed; every frame is the full native frame.

Usage::

    python scripts/slm/issue247_capture_tilt_20261117.py --out <fresh-dir>            # plan only
    python scripts/slm/issue247_capture_tilt_20261117.py --execute --out <fresh-dir>  # hardware

Hardware PYTHONPATH (offline validation needs none)::

    $env:PYTHONPATH = "D:\\Projects\\TIFO\\AO-shaping\\libs"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "libs"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger

#: SLM-200 panel default; overwritten by the driver's actual Panel_Res at execute time.
DEFAULT_PANEL_SHAPE = (1200, 1920)
#: Native, never-windowed Daheng frame shape.
NATIVE_CCD_SHAPE = (1944, 2592)
SATURATION_FRACTION = 0.98
TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Deterministic phase plan (raw radians, unwrapped; the driver wraps at rad->gray)
# ---------------------------------------------------------------------------
def _flat(shape: tuple[int, int]) -> np.ndarray:
    return np.zeros(shape, dtype=np.float64)


def _tilt(
    shape: tuple[int, int], *, period_px: float, axis: int, sign: int
) -> np.ndarray:
    """Linear ramp: ``sign * 2*pi * (coord % period) / period`` along ``axis``.

    A genuine linear phase ramp (prismatic tilt), 2*pi every ``period_px`` panel
    pixels. ``axis=1`` -> panel x (column), ``axis=0`` -> panel y (row). Negative
    sign is left raw (negative radians); the driver's rad->gray step wraps mod 2*pi.
    """
    height, width = int(shape[0]), int(shape[1])
    period = float(period_px)
    if period <= 0:
        raise ValueError("tilt period_px must be positive")
    n = width if axis == 1 else height
    t = (np.arange(n, dtype=np.float64) % period) / period
    ramp = TWO_PI * t
    out = ramp.reshape((1, width) if axis == 1 else (height, 1))
    return (sign * np.broadcast_to(out, (height, width))).astype(np.float64)


def _defocus(
    shape: tuple[int, int], *, amplitude_rad: float, sign: int
) -> np.ndarray:
    """Quadratic (lens) phase centred on the panel, ``sign * A * r**2 / R**2``.

    ``R`` is half the shorter panel edge, so the panel-corner phase magnitude is
    about ``2 * amplitude_rad``. Raw radians; wrapped by the driver. No pupil
    apodisation, no correction map.
    """
    height, width = int(shape[0]), int(shape[1])
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    radius = max(min(height, width) / 2.0, 1.0)
    yy = (np.arange(height, dtype=np.float64) - cy)[:, None]
    xx = (np.arange(width, dtype=np.float64) - cx)[None, :]
    r2 = xx * xx + yy * yy
    return (sign * float(amplitude_rad) * r2 / (radius * radius)).astype(np.float64)


def build_plan(period_px: float, defocus_rad: float) -> list[dict]:
    """Ordered capture plan. Baselines repeat before/after for drift tracking."""

    def entry(label: str, kind: str, **params: Any) -> dict:
        return dict(label=label, kind=kind, params=params)

    return [
        entry("flat_before_1", "flat"),
        entry("flat_before_2", "flat"),
        entry("tilt_x_plus", "tilt", axis=1, sign=+1, period_px=period_px),
        entry("flat_mid_1", "flat"),
        entry("tilt_x_minus", "tilt", axis=1, sign=-1, period_px=period_px),
        entry("tilt_y_plus", "tilt", axis=0, sign=+1, period_px=period_px),
        entry("flat_mid_2", "flat"),
        entry("tilt_y_minus", "tilt", axis=0, sign=-1, period_px=period_px),
        entry("defocus_plus", "defocus", sign=+1, amplitude_rad=defocus_rad),
        entry("flat_mid_3", "flat"),
        entry("defocus_minus", "defocus", sign=-1, amplitude_rad=defocus_rad),
        entry("flat_after_1", "flat"),
        entry("flat_after_2", "flat"),
    ]


def phase_for(entry: dict, shape: tuple[int, int]) -> np.ndarray:
    kind = entry["kind"]
    params = entry["params"]
    if kind == "flat":
        return _flat(shape)
    if kind == "tilt":
        return _tilt(shape, **params)
    if kind == "defocus":
        return _defocus(shape, **params)
    raise ValueError(f"unknown phase kind {kind!r}")


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def clock() -> dict:
    return dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        monotonic_ns=time.monotonic_ns(),
    )


# ---------------------------------------------------------------------------
# Hardware execution
# ---------------------------------------------------------------------------
def _read_gain(cam: Any) -> float | None:
    """Best-effort effective gain readback; ``None`` is recorded when unavailable."""
    for getter in ("get_gain", "get_gain_raw"):
        fn = getattr(cam, getter, None)
        if callable(fn):
            try:
                return float(fn())
            except Exception:
                pass
    value = getattr(cam, "gain", None)
    if value is not None:
        try:
            return float(value)
        except Exception:
            pass
    sdk_cam = getattr(cam, "cam", None)
    gain_feature = getattr(sdk_cam, "Gain", None) if sdk_cam is not None else None
    if gain_feature is not None and hasattr(gain_feature, "get"):
        try:
            return float(gain_feature.get())
        except Exception:
            pass
    return None


def execute(args: argparse.Namespace, plan: list[dict]) -> None:
    from ao_shaping.tools.kernels.device_mutex import DeviceLock, wait_until_idle

    verdict = wait_until_idle(
        args.wait_until_idle,
        lock=DeviceLock.default(),
        own_paths=[Path(__file__).resolve()],
    )
    if not verdict.ok or verdict.lock is None:
        raise RuntimeError(verdict.headline())
    lock = verdict.lock

    records: list[dict] = []
    monotonic_log: list[int] = []
    released: dict | None = None
    try:
        from ao_shaping.tools.slm.transfer.hardware import prepare_camera_sdk

        prepare_camera_sdk("daheng")
        from ao_shaping.drivers.ccd.common import create_camera, get_camera_exposure_ms
        from ao_shaping.drivers.slm.santec import Santec

        with create_camera(
            "daheng", args.cam_id, exposure_time_ms=args.exposure_ms, bit_depth=12
        ) as cam:
            with Santec(
                slm_number=args.slm_number,
                wavelength=args.wavelength_nm,
                video_mode=0,
            ) as slm:
                # Hard entry assertion: a silent drive board means no modulation.
                drive_board = slm.require_drive_board()

                # Assert the *actual* pixel format via effective-bit-depth readback.
                effective_bits = cam.set_bit_depth(12)
                if effective_bits != 12 or getattr(cam, "bit_depth", None) != 12:
                    raise RuntimeError(
                        "full native Mono12 not active: "
                        f"set_bit_depth returned {effective_bits}, "
                        f"cam.bit_depth={getattr(cam, 'bit_depth', None)}"
                    )

                panel_res = tuple(slm.Panel_Res)  # (width, height)
                panel_shape = (int(panel_res[1]), int(panel_res[0]))

                try:
                    serial = slm.get_serial_number()
                except Exception:
                    serial = None
                try:
                    wavelength_info = slm.get_wavelength_info()
                except Exception:
                    wavelength_info = None
                try:
                    encoding_max_gray = slm.encoding_max_grayscale()
                except Exception:
                    encoding_max_gray = None

                write_json(
                    args.out / "device.json",
                    dict(
                        drive_board=str(drive_board),
                        slm_serial=serial,
                        slm_number=args.slm_number,
                        wavelength_nm=args.wavelength_nm,
                        wavelength_info=wavelength_info,
                        encoding_max_grayscale=encoding_max_gray,
                        panel_res=list(panel_res),
                        cam_id=args.cam_id,
                        camera_bit_depth_effective=int(effective_bits),
                        camera_full_scale_expected=4095.0,
                        camera_gain=_read_gain(cam),
                        opened_utc=datetime.now(timezone.utc).isoformat(
                            timespec="microseconds"
                        ),
                    ),
                )

                # No overlay / correction is ever applied.
                slm.set_phase_overlay(None)

                for index, item in enumerate(plan):
                    phase = phase_for(item, panel_shape)
                    if not np.all(np.isfinite(phase)):
                        raise RuntimeError(f"{item['label']} phase is non-finite")
                    gray = slm.create_phase_from_array(phase)
                    slm.display_data(gray)

                    cam.reset_exposure_time(float(args.exposure_ms))
                    time.sleep(args.settle_s)
                    # Discard frames left in the pipeline by display / exposure writes.
                    for _ in range(3):
                        cam.get_numpy_image(n_sample=1, skip_first=False)

                    frames: list[np.ndarray] = []
                    stamps: list[dict] = []
                    for _ in range(args.frames):
                        stamp = clock()
                        frame = np.asarray(
                            cam.get_numpy_image(n_sample=1, skip_first=False)
                        ).copy()
                        stamp.update(
                            end_monotonic_ns=time.monotonic_ns(),
                            exposure_ms=float(get_camera_exposure_ms(cam)),
                        )
                        if frame.shape != NATIVE_CCD_SHAPE or frame.dtype != np.uint16:
                            raise RuntimeError(
                                f"{item['label']}: expected native unpacked Mono12 "
                                f"uint16 frame {NATIVE_CCD_SHAPE}, got "
                                f"{frame.shape}/{frame.dtype}"
                            )
                        full_scale = float(cam.effective_full_scale(frame))
                        if full_scale != 4095.0:
                            raise RuntimeError(
                                f"{item['label']}: Mono12 full scale readback is "
                                f"{full_scale}, expected 4095"
                            )
                        frames.append(frame)
                        stamps.append(stamp)
                        monotonic_log.append(int(stamp["monotonic_ns"]))
                        monotonic_log.append(int(stamp["end_monotonic_ns"]))

                    raw = np.stack(frames)
                    peak = float(raw.max())
                    saturated = int((raw >= args.saturation_fraction * 4095.0).sum())
                    exposures = [s["exposure_ms"] for s in stamps]
                    filename = f"{index:02d}_{item['label']}.npz"
                    row = dict(
                        index=index,
                        label=item["label"],
                        kind=item["kind"],
                        phase_params=item["params"],
                        requested_exposure_ms=float(args.exposure_ms),
                        exposure_ms_readback=exposures,
                        frame_shape=list(NATIVE_CCD_SHAPE),
                        frame_dtype=str(raw.dtype),
                        full_scale=4095.0,
                        peak=peak,
                        saturated_pixels=saturated,
                        phase_sha256=_digest(phase),
                        panel_gray_sha256=_digest(gray),
                        panel_shape=list(panel_shape),
                        stamps=stamps,
                        file=filename,
                        **clock(),
                    )
                    np.savez_compressed(
                        args.out / filename,
                        raw_frames=raw,
                        phase_rad=phase,
                        panel_gray=gray,
                    )
                    records.append(row)
                    write_json(args.out / "manifest.json", records)

                    if saturated:
                        raise RuntimeError(
                            f"{item['label']} saturated {saturated} native pixels "
                            f"(peak {peak} >= {args.saturation_fraction} * 4095)"
                        )
                    if max(exposures) / min(exposures) > 1.001:
                        raise RuntimeError(
                            f"{item['label']} exposure readback not constant: {exposures}"
                        )

                if monotonic_log != sorted(monotonic_log):
                    raise RuntimeError("timestamps are not monotonic")

                write_json(
                    args.out / "summary.json",
                    dict(
                        captures=len(records),
                        exposure_ms=args.exposure_ms,
                        frames_per_capture=args.frames,
                        tilt_period_px=args.tilt_period_px,
                        defocus_rad=args.defocus_rad,
                        all_timestamps_monotonic=True,
                        saturated_total=sum(r["saturated_pixels"] for r in records),
                        **clock(),
                    ),
                )
                logger.info(f"issue247: {len(records)} captures written to {args.out}")
                print(
                    f"acquisition complete: {len(records)} captures -> {args.out}"
                )
                # Devices close via the context managers below, before this finally
                # block runs; mark released only once that has happened.
                released = dict(
                    slm_closed=True,
                    camera_closed=True,
                    finished_unix=time.time(),
                    finished_utc=datetime.now(timezone.utc).isoformat(
                        timespec="microseconds"
                    ),
                )
    finally:
        if released is None:
            released = dict(
                slm_closed=None,
                camera_closed=None,
                aborted=True,
                finished_unix=time.time(),
                finished_utc=datetime.now(timezone.utc).isoformat(
                    timespec="microseconds"
                ),
            )
        write_json(args.out / "released.json", released)
        lock.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="fresh output directory")
    parser.add_argument("--execute", action="store_true", help="open hardware and acquire")
    parser.add_argument("--exposure-ms", type=float, default=0.05)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--settle-s", type=float, default=0.8)
    parser.add_argument("--tilt-period-px", type=float, default=128.0)
    parser.add_argument("--defocus-rad", type=float, default=float(TWO_PI))
    parser.add_argument("--saturation-fraction", type=float, default=SATURATION_FRACTION)
    parser.add_argument("--wait-until-idle", type=float, default=1800.0)
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--wavelength-nm", type=int, default=1064)
    parser.add_argument("--cam-id", type=int, default=0)
    args = parser.parse_args()

    if args.frames < 1 or args.exposure_ms <= 0 or args.settle_s < 0:
        parser.error("--frames>=1, --exposure-ms>0, --settle-s>=0 required")
    if not 0 < args.saturation_fraction <= 1:
        parser.error("--saturation-fraction must be in (0, 1]")

    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"--out must be a fresh/empty directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    plan = build_plan(args.tilt_period_px, args.defocus_rad)

    # Offline plan: phase parameters + digests, no driver import, no device access.
    plan_rows = []
    for item in plan:
        phase = phase_for(item, DEFAULT_PANEL_SHAPE)
        plan_rows.append(
            dict(
                label=item["label"],
                kind=item["kind"],
                params=item["params"],
                phase_sha256=_digest(phase),
                phase_shape=list(phase.shape),
            )
        )
    write_json(
        args.out / "plan.json",
        dict(
            execute=bool(args.execute),
            created_utc=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            exposure_ms=args.exposure_ms,
            frames=args.frames,
            settle_s=args.settle_s,
            tilt_period_px=args.tilt_period_px,
            defocus_rad=args.defocus_rad,
            saturation_fraction=args.saturation_fraction,
            default_panel_shape=list(DEFAULT_PANEL_SHAPE),
            native_ccd_shape=list(NATIVE_CCD_SHAPE),
            entries=plan_rows,
        ),
    )

    if not args.execute:
        print(f"plan written to {args.out/'plan.json'} ({len(plan_rows)} captures)")
        print("skip: requires --execute")
        return

    execute(args, plan)


if __name__ == "__main__":
    main()
