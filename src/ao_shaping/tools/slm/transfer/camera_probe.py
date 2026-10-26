"""Record actual camera formats and raw precision without opening an SLM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .capture import clock
from .hardware import prepare_camera_sdk


def probe(out: Path):
    prepare_camera_sdk("daheng")
    from ao_shaping.drivers.ccd.common import create_camera, get_camera_exposure_ms

    out.mkdir(parents=True, exist_ok=False)
    with create_camera("daheng", 0, exposure_time_ms=.02) as camera:
        formats = camera.cam.PixelFormat.get_range()
        evidence = dict(formats=formats, observations=[], **clock())
        (out / "formats.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        for name in ("MONO8", "MONO12"):
            feature = next((v for k, v in formats.items() if k.upper() == name), None)
            if feature is None:
                continue
            camera.cam.stream_off()
            try:
                camera.cam.PixelFormat.set(feature)
            finally:
                camera.cam.stream_on()
            for _ in range(3):
                camera.get_numpy_image(n_sample=1, skip_first=False)
            for index in range(3):
                start = clock()
                frame = np.asarray(camera.get_numpy_image(n_sample=1, skip_first=False)).copy()
                filename = f"{name.lower()}_{index}.npy"
                np.save(out / filename, frame)
                evidence["observations"].append(dict(file=filename, format=camera.cam.PixelFormat.get(),
                    dtype=str(frame.dtype), shape=list(frame.shape), peak=int(frame.max()),
                    exposure_ms=float(get_camera_exposure_ms(camera)), **start))
                (out / "formats.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(probe(args.out), indent=2))
