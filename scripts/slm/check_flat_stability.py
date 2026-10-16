"""Record repeated fixed-exposure Santec flat phases on a Daheng CCD.

The complete frames and per-read exposure readbacks are retained for diagnosing
laser or optical-path drift before a model-in-loop run.

The safe exposure ceiling is optical-path dependent: it comes from
``AO_DAHENG_MAX_EXPOSURE_MS`` when set, otherwise it is probed from the image
with a doubling exposure ladder before any measurement.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from ao_shaping.drivers.ccd.common import create_camera
from ao_shaping.drivers.slm.santec import Santec
from ao_shaping.tools.kernels.bench_kernels import measure_flat_reference
from ao_shaping.tools.slm.panel.display_state import preserve_slm_display
from ao_shaping.utils.image.exposure_control import (
    DAHENG_EXPOSURE_LIMIT_ENV,
    configured_daheng_exposure_limit,
    resolve_daheng_exposure_limit,
)


def collect(output_dir: Path, *, exposure_ms: float, repeats: int, frames: int) -> list[dict]:
    env_ceiling = configured_daheng_exposure_limit()
    if exposure_ms < 0.02:
        raise ValueError("Daheng exposure must be at least the 0.02 ms camera minimum")
    if env_ceiling > 0 and exposure_ms > env_ceiling:
        raise ValueError(
            f"Daheng exposure {exposure_ms} ms exceeds the {env_ceiling} ms "
            f"ceiling ({DAHENG_EXPOSURE_LIMIT_ENV})"
        )
    if repeats < 2 or frames < 1:
        raise ValueError("repeats must be >= 2 and frames must be >= 1")
    output_dir.mkdir(parents=True, exist_ok=False)
    config = {"slm_number": 1, "slm_wavelength_nm": 1064,
              "cam_type": "daheng", "cam_id": 0, "exposure_ms": exposure_ms,
              "exposure_ceiling_ms": env_ceiling,
              "repeats": repeats, "frames_per_repeat": frames}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    rows: list[dict] = []
    center: tuple[int, int] | None = None
    with Santec(slm_number=1, wavelength=1064, video_mode=0) as slm:
        with create_camera("daheng", 0, exposure_time_ms=exposure_ms) as cam:
            with preserve_slm_display(slm):
                ceiling = resolve_daheng_exposure_limit(camera=cam)
                if exposure_ms > ceiling + 1e-9:
                    raise ValueError(
                        f"Daheng exposure {exposure_ms} ms exceeds the resolved "
                        f"ceiling {ceiling} ms"
                    )
                actual = float(cam.reset_exposure_time(exposure_ms))
                if not 0 < actual <= ceiling + 1e-9:
                    raise RuntimeError(
                        f"unsafe Daheng exposure readback: {actual} ms "
                        f"(ceiling {ceiling} ms)"
                    )
                for index in range(repeats):
                    image, spot = measure_flat_reference(cam, slm, n_frames=frames)
                    actual = float(cam.get_exposure_time_ms())
                    if not 0 < actual <= ceiling + 1e-9:
                        raise RuntimeError(
                            f"unsafe Daheng exposure readback: {actual} ms "
                            f"(ceiling {ceiling} ms)"
                        )
                    if center is None:
                        center = (int(round(spot.centroid_x)), int(round(spot.centroid_y)))
                    x, y = center
                    yy, xx = np.ogrid[:image.shape[0], :image.shape[1]]
                    roi = (xx - x) ** 2 + (yy - y) ** 2 <= 60**2
                    row = {
                        "index": index,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                        "exposure_ms_readback": actual,
                        "peak": float(np.max(image)),
                        "total": float(np.sum(image)),
                        "fixed_roi_energy": float(np.sum(image[roi])),
                        "fixed_roi_center_xy": [x, y],
                        "measured_centroid_xy": [float(spot.centroid_x), float(spot.centroid_y)],
                    }
                    np.savez_compressed(output_dir / f"flat_{index:02d}.npz", frame=image)
                    rows.append(row)
                    (output_dir / "measurements.json").write_text(
                        json.dumps(rows, indent=2), encoding="utf-8"
                    )
                    print(json.dumps(row), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--exposure-ms", type=float, default=0.02)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--frames", type=int, default=2)
    args = parser.parse_args()
    collect(args.output_dir, exposure_ms=args.exposure_ms,
            repeats=args.repeats, frames=args.frames)


if __name__ == "__main__":
    main()
