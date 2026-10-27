"""Short-exposure fixed-phase probe with raw frames and exposure readback."""

from __future__ import annotations

import argparse
from loguru import logger
from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path
from typing import Iterator

import numpy as np

from ao_shaping.tools.kernels.bench_kernels import (
    SLM_PANEL_H, SLM_PANEL_W, capture_settled, phase_to_panel,
)
from ao_shaping.utils.image.beam_metrics import prepare_frame, zero_order_center
from ao_shaping.utils.image.exposure_control import (
    DAHENG_EXPOSURE_LIMIT_ENV,
    configured_daheng_exposure_limit,
    resolve_daheng_exposure_limit,
)


@contextmanager
def _flat_on_exit(slm) -> Iterator[None]:
    try:
        yield
    finally:
        flat = np.zeros((SLM_PANEL_H, SLM_PANEL_W), dtype=np.float64)
        slm.display_data(slm.create_phase_from_array(flat))


def validate_exposures(exposures: list[float]) -> None:
    if len(exposures) < 2 or any(not np.isfinite(ms) or ms < 0.02
                                 for ms in exposures):
        raise ValueError(
            "provide at least two finite Daheng exposures of at least 0.02 ms"
        )
    ceiling = configured_daheng_exposure_limit()
    if ceiling > 0 and any(ms > ceiling for ms in exposures):
        raise ValueError(
            f"Daheng exposure exceeds the {ceiling} ms ceiling "
            f"({DAHENG_EXPOSURE_LIMIT_ENV})"
        )


def _phase_panel(phase_run_dir: Path | None) -> np.ndarray:
    if phase_run_dir is None:
        return np.zeros((SLM_PANEL_H, SLM_PANEL_W), dtype=np.float64)
    source = Path(phase_run_dir)
    config = json.loads((source / "run_config.json").read_text(encoding="utf-8"))
    if config["cam_type"] != "daheng" or config["slm_number"] != 1 or config["slm_wavelength"] != 1064:
        raise ValueError("saved phase requires Daheng CCD and SLM #1 at 1064 nm")
    phase = np.load(source / "best_phase.npy", allow_pickle=False)
    region = int(config["region"])
    if phase.shape != (region, region) or not np.isfinite(phase).all():
        raise ValueError("saved phase is invalid for its recorded model grid")
    return phase_to_panel(phase, int(round(config["panel_span_px"] / 2)),
                          tuple(config["pupil_center"]))


def run_probe(
    output_dir: Path, exposures: list[float], phase_run_dir: Path | None = None,
) -> list[dict]:
    """Measure a fixed flat or saved phase at bounded CCD exposures."""
    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm.santec import Santec

    validate_exposures(exposures)
    panel = _phase_panel(phase_run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(
        json.dumps({"exposures_ms": exposures, "n_frames": 4,
                    "phase_run_dir": str(Path(phase_run_dir).resolve()) if phase_run_dir else None,
                    "max_daheng_exposure_ms": configured_daheng_exposure_limit()},
        indent=2),
        encoding="utf-8",
    )
    rows: list[dict] = []
    center: tuple[int, int] | None = None
    with (
        Santec(slm_number=1, wavelength=1064, video_mode=0) as slm,
        create_camera("daheng", 0, exposure_time_ms=exposures[0]) as cam,
        _flat_on_exit(slm),
    ):
        ceiling = resolve_daheng_exposure_limit(camera=cam)
        if any(ms > ceiling + 1e-9 for ms in exposures):
            raise ValueError(
                f"Daheng exposure ladder exceeds the resolved ceiling "
                f"{ceiling} ms"
            )
        for index, requested_ms in enumerate(exposures):
            actual_ms = float(cam.reset_exposure_time(requested_ms))
            if not 0 < actual_ms <= ceiling + 1e-9:
                raise RuntimeError(
                    f"unsafe Daheng exposure readback: {actual_ms} ms "
                    f"(ceiling {ceiling} ms)"
                )
            frame = np.asarray(capture_settled(
                cam, slm, panel, n_frames=4, n_discard=3,
                wait_time_s=None, stable_tol=0.02, max_wait_s=6.0,
            ), dtype=np.float32)
            raw = np.asarray(cam.get_numpy_image(n_sample=1))
            peak = max(float(frame.max()), float(raw.max()))
            if peak >= 250:
                raise RuntimeError(f"CCD frame saturated at {requested_ms} ms: {peak}/255")
            if center is None:
                center = zero_order_center(prepare_frame(frame), refine=False,
                                           despike_k=3)
            cx, cy = center
            y0, y1 = max(0, cy - 96), min(frame.shape[0], cy + 96)
            x0, x1 = max(0, cx - 96), min(frame.shape[1], cx + 96)
            signal = prepare_frame(frame)
            row = {
                "index": index,
                "time_local": datetime.now().astimezone().isoformat(),
                "requested_ms": requested_ms, "readback_ms": actual_ms,
                "center_x": cx, "center_y": cy,
                "raw_peak": float(raw.max()), "average_peak": float(frame.max()),
                "background_median": float(np.median(frame)),
                "raw_roi_sum": float(frame[y0:y1, x0:x1].sum()),
                "signal_roi_sum": float(signal[y0:y1, x0:x1].sum()),
                "signal_peak": float(signal.max()),
            }
            np.savez_compressed(output_dir / f"{index:02d}.npz", frame=frame,
                                raw=raw, roi_bounds=np.array([x0, y0, x1, y1]))
            rows.append(row)
            (output_dir / "measurements.json").write_text(
                json.dumps(rows, indent=2), encoding="utf-8"
            )
            logger.info(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--exposures-ms", default="0.03,0.06,0.06,0.03")
    parser.add_argument("--phase-run-dir", type=Path)
    args = parser.parse_args()
    try:
        exposures = [float(item) for item in args.exposures_ms.split(",")]
        validate_exposures(exposures)
        _phase_panel(args.phase_run_dir)
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    run_probe(args.output_dir, exposures, args.phase_run_dir)


if __name__ == "__main__":
    main()
