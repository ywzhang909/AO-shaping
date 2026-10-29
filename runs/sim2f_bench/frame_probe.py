"""One frame at a fixed exposure, with the raw numbers a dark bench needs.

No phase is written, no averaging, no centroid: on a bench that may have lost its
light, every one of those steps hides the cause.  Prints peak, mean, the argmax
pixel, and the exposure readback so "the camera is fine, the beam is gone" and
"the camera is stuck at 0.02 ms" are different outputs.
"""

from __future__ import annotations

import argparse

import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exposure-ms", type=float, action="append", default=None,
                        help="repeatable; each runs once (default 0.02, 0.2, 1, 5)")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--cam-type", default="daheng")
    parser.add_argument("--cam-id", type=int, default=0)
    args = parser.parse_args(argv)
    exposures = args.exposure_ms or [0.02, 0.2, 1.0, 5.0]

    from ao_shaping.drivers.ccd.common import create_camera

    with create_camera(args.cam_type, args.cam_id, exposure_time_ms=0.02) as cam:
        for ms in exposures:
            back = cam.reset_exposure_time(float(ms))
            for _ in range(int(args.frames)):
                frame = np.asarray(cam.get_numpy_image(n_sample=1), dtype=np.float64)
            peak = float(frame.max())
            row, col = np.unravel_index(int(frame.argmax()), frame.shape)
            print(
                f"requested {ms:>7.3f} ms  readback {float(back):>7.3f} ms  "
                f"peak {peak:>6.1f}  mean {frame.mean():>8.3f}  "
                f"sum {frame.sum():>12.0f}  argmax=({row},{col})  "
                f"shape={frame.shape}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
