"""Probe the frozen 1000 px-square model without running an optimizer.

The session follows the 14-frame contract consumed by
``analyze_square1000_model_diagnostic_20261030.py``:

``A, A, N4+, N4-, A, RX+, RX-, A, RY+, RY-, A, A(0.5x), A(1.5x), A``

``A`` is the archived broadening seed.  ``N4`` is the model's own Noll-4
defocus basis, and ``RX`` / ``RY`` are 64-panel-px-period cosine tilts.  The
base exposure is selectable because the analyzer reconstructs radiometry from
the recorded 0.5/1/1.5 multipliers.  Any native frame at or above 98% of the
Mono12 full scale stops the session; optimization is never attempted here.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    FrozenDefocusForward,
)
from ao_shaping.tools.slm.transfer.capture import clock, to_panel
from square_1000_diff_abba_20261030 import (
    CHECKPOINT,
    NATIVE_CCD_SHAPE,
    SEED_KEY,
    SEED_RUN,
    load_seed_contract,
    write_json,
)

FULL_SCALE = 4095.0
SATURATION_FRACTION = 0.98
BASE_EXPOSURE_MS = 10.0
PROBE_AMPLITUDE_RAD = 0.03
TILT_PERIOD_PANEL_PX = 64.0
ROLE_SEQUENCE = (
    ("A", "anchor"),
    ("A", "anchor_repeat"),
    ("N4+", "noll4_plus"),
    ("N4-", "noll4_minus"),
    ("A", "anchor"),
    ("RX+", "tilt_x_plus"),
    ("RX-", "tilt_x_minus"),
    ("A", "anchor"),
    ("RY+", "tilt_y_plus"),
    ("RY-", "tilt_y_minus"),
    ("A", "anchor"),
    ("A", "exposure_half"),
    ("A", "exposure_one_and_half"),
    ("A", "anchor_final"),
)
EXPOSURE_MULTIPLIERS = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 0.5, 1.5, 1.0)


def probe_phases(
    seed: np.ndarray, forward, amplitude_rad: float
) -> dict[str, np.ndarray]:
    """Return exact raw-radian phases for every model probe."""
    z4 = forward.z4.detach().cpu().numpy().astype(np.float64)
    panel_x = forward.panel_x.detach().cpu().numpy().astype(np.float64)
    panel_y = forward.panel_y.detach().cpu().numpy().astype(np.float64)
    rx = np.cos(2.0 * np.pi * panel_x / TILT_PERIOD_PANEL_PX)
    ry = np.cos(2.0 * np.pi * panel_y / TILT_PERIOD_PANEL_PX)
    phase = {
        "A": seed.astype(np.float32),
        "N4+": (seed + amplitude_rad * z4).astype(np.float32),
        "N4-": (seed - amplitude_rad * z4).astype(np.float32),
        "RX+": (seed + amplitude_rad * rx).astype(np.float32),
        "RX-": (seed - amplitude_rad * rx).astype(np.float32),
        "RY+": (seed + amplitude_rad * ry).astype(np.float32),
        "RY-": (seed - amplitude_rad * ry).astype(np.float32),
    }
    for name, values in phase.items():
        if values.shape != seed.shape:
            raise RuntimeError(f"{name} probe shape {values.shape} != seed {seed.shape}")
        if not np.isfinite(values).all():
            raise RuntimeError(f"{name} probe contains nonfinite phase")
    return phase


def predictions(forward, phase: dict[str, np.ndarray], exposure_ms: float) -> dict[str, np.ndarray]:
    """Predict the same nine camera-grid rate maps the analyzer consumes."""
    output = {}
    forward.exposure_ms = exposure_ms
    with torch.no_grad():
        for name in ("A", "N4+", "N4-", "RX+", "RX-", "RY+", "RY-"):
            tensor = torch.as_tensor(phase[name], dtype=torch.float32)
            value = forward.predict(tensor, exposure_ms=exposure_ms)
            array = value.detach().cpu().numpy()
            if array.ndim == 3:
                array = array[0]
            expected = tuple(forward.camera_xy.shape[:2])
            if array.shape != expected:
                raise RuntimeError(
                    f"{name} prediction shape {array.shape} != {expected}"
                )
            output[name] = np.asarray(array, dtype=np.float32)
    return output


def execute(args, archive, phase) -> None:
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
    released = None
    try:
        from ao_shaping.tools.slm.transfer.hardware import prepare_camera_sdk

        prepare_camera_sdk("daheng")
        from ao_shaping.drivers.ccd.common import create_camera, get_camera_exposure_ms
        from ao_shaping.drivers.slm.santec import Santec

        with create_camera(
            "daheng", 0, exposure_time_ms=0.5, bit_depth=12
        ) as cam:
            with Santec(slm_number=1, wavelength=1064, video_mode=0) as slm:
                drive_board = slm.require_drive_board()
                write_json(
                    args.out / "device.json",
                    dict(
                        drive_board=str(drive_board),
                        slm_number=1,
                        wavelength_nm=1064,
                        opened_utc=time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                    ),
                )
                slm.set_phase_overlay(None)
                try:
                    for index, ((role, label), multiplier) in enumerate(
                        zip(ROLE_SEQUENCE, EXPOSURE_MULTIPLIERS, strict=True)
                    ):
                        exposure = args.base_exposure_ms * multiplier
                        values = phase[role]
                        gray = slm.create_phase_from_array(to_panel(values))
                        slm.display_data(gray)
                        cam.reset_exposure_time(exposure)
                        time.sleep(0.8)
                        for _ in range(3):
                            cam.get_numpy_image(n_sample=1, skip_first=False)
                        frames, stamps = [], []
                        for _ in range(5):
                            stamp = clock()
                            frame = np.asarray(
                                cam.get_numpy_image(n_sample=1, skip_first=False)
                            ).copy()
                            stamp.update(
                                end_monotonic_ns=time.monotonic_ns(),
                                exposure_ms=float(get_camera_exposure_ms(cam)),
                            )
                            if (
                                frame.shape != tuple(NATIVE_CCD_SHAPE)
                                or frame.dtype != np.uint16
                            ):
                                raise RuntimeError(
                                    "expected native unpacked Mono12 uint16 frame"
                                )
                            frames.append(frame)
                            stamps.append(stamp)
                        raw = np.stack(frames)
                        row = dict(
                            index=index,
                            label=label,
                            role=role,
                            requested_exposure_ms=float(exposure),
                            requested_multiplier=float(multiplier),
                            exposure_ms=float(stamps[0]["exposure_ms"]),
                            frames=stamps,
                            file=f"capture_{index:04d}.npz",
                            maximum=int(raw.max()),
                            saturated_pixels=int(
                                (raw >= SATURATION_FRACTION * FULL_SCALE).sum()
                            ),
                            **clock(),
                        )
                        np.savez_compressed(
                            args.out / row["file"],
                            raw_frames=raw,
                            model_phase=values,
                            panel_gray=gray,
                            camera_xy=archive.camera_xy,
                        )
                        records.append(row)
                        write_json(args.out / "manifest.json", records)
                        if row["saturated_pixels"]:
                            raise RuntimeError(
                                f"{label} at {row['exposure_ms']:.3f} ms saturated "
                                f"{row['saturated_pixels']} native pixels"
                            )
                        ratios = [s["exposure_ms"] for s in stamps]
                        if max(ratios) / min(ratios) > 1.001:
                            raise RuntimeError(
                                f"{label} exposure readback is not constant: {ratios}"
                            )
                finally:
                    released = dict(
                        slm_closed=True,
                        camera_closed=True,
                        finished_unix=time.time(),
                        finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    )
                    write_json(args.out / "released.json", released)
    finally:
        lock.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--seed-run", type=Path, default=SEED_RUN)
    parser.add_argument("--seed-key", default=SEED_KEY)
    parser.add_argument("--base-exposure-ms", type=float, default=BASE_EXPOSURE_MS)
    parser.add_argument("--probe-amplitude-rad", type=float, default=PROBE_AMPLITUDE_RAD)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--wait-until-idle", type=float, default=0.0)
    args = parser.parse_args()
    if not np.isfinite(args.base_exposure_ms) or args.base_exposure_ms <= 0:
        parser.error("base-exposure-ms must be finite and positive")
    if not np.isfinite(args.probe_amplitude_rad) or args.probe_amplitude_rad <= 0:
        parser.error("probe-amplitude-rad must be finite and positive")
    if not np.isfinite(args.wait_until_idle) or args.wait_until_idle < 0:
        parser.error("wait-until-idle must be finite and nonnegative")
    torch.set_num_threads(4)
    archive = load_seed_contract(
        args.seed_run, args.checkpoint, seed_key=args.seed_key
    )
    forward = FrozenDefocusForward(archive.parameters, archive.camera_xy, device="cpu")
    phase = probe_phases(archive.seed, forward, args.probe_amplitude_rad)
    predicted = predictions(forward, phase, args.base_exposure_ms)
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.out / "probe_phases.npz", **phase)
    np.savez_compressed(args.out / "predictions.npz", **predicted)
    write_json(
        args.out / "config.json",
        dict(
            seed_run=str(args.seed_run),
            seed_key=args.seed_key,
            checkpoint=str(args.checkpoint),
            base_exposure_ms=args.base_exposure_ms,
            exposure_multipliers=list(EXPOSURE_MULTIPLIERS),
            probe_amplitude_rad=args.probe_amplitude_rad,
            tilt_period_panel_px=TILT_PERIOD_PANEL_PX,
            role_sequence=[role for role, _ in ROLE_SEQUENCE],
            native_ccd_shape=list(NATIVE_CCD_SHAPE),
            full_scale=FULL_SCALE,
            saturation_fraction=SATURATION_FRACTION,
        ),
    )
    print(json.dumps(dict(out=str(args.out), probes=len(ROLE_SEQUENCE)), indent=2))
    if not args.execute:
        print("offline plan only; pass --execute to acquire CCD frames")
        return
    execute(args, archive, phase)


if __name__ == "__main__":
    main()
