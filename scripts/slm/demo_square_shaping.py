#!/usr/bin/env python3
"""SPGD square-beam uniformity optimization on an SLM (extracted from
``ao_shaping.optimizer.wfless.slm_square_shaping`` — see issue #90).

The library module keeps only the algorithm; this script is the runnable
CLI entry point for the square-beam demo / real-device run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np

from ao_shaping.optimizer.wfless.slm_square_shaping import optimize_slm_square
from ao_shaping.utils import logger
from ao_shaping.utils.io.file import gen_date_dir, gen_date_str
from ao_shaping.utils.io.recorder_keys import (
    RECORD_EPOCH,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SPGD square beam uniformity optimization using SLM with Zernike coefficients"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=2000, help="Number of iterations"
    )
    parser.add_argument(
        "-n", "--n-max", type=int, default=4, help="Max Zernike radial order"
    )
    parser.add_argument(
        "-c",
        "--center",
        type=str,
        default="max",
        help="Center detection: shape/max/mass/centroid_thresh or 'x,y'",
    )
    parser.add_argument(
        "--target-side", type=int, default=0, help="Target square side (0=auto)"
    )
    parser.add_argument(
        "--side-factor", type=float, default=1.5, help="Side length factor for auto"
    )
    parser.add_argument(
        "-d", "--delta", type=float, default=0.1, help="Perturbation amplitude"
    )
    parser.add_argument("--lr", type=float, default=0, help="Learning rate (0=auto)")
    parser.add_argument(
        "-t",
        "--exposure-time-ms",
        type=float,
        default=0.0,
        help="Exposure time in ms; 0 = keep device setting / auto (Daheng clamps 0 to device min)",
    )
    parser.add_argument("--cam-id", type=int, default=0, help="Camera device ID")
    parser.add_argument("--slm-number", type=int, default=1, help="SLM device number")
    parser.add_argument(
        "--slm-wavelength",
        type=int,
        default=0,
        help=(
            "SLM wavelength (nm); 0 (the default) asks the device which "
            "wavelength it is programmed for instead of assuming a bench value"
        ),
    )
    parser.add_argument(
        "--optimizer", type=str, default="adamod", help="Optimizer type"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--show", action="store_true", help="Show images during optimization"
    )
    parser.add_argument("--cam-size", type=int, default=300, help="Camera window size")
    parser.add_argument(
        "--zernike-mask",
        type=str,
        default=None,
        help="0/1 binary mask for Zernike modes (comma-separated), e.g. '0,0,0,1,0,0,0,0,0,0,0,1' for defocus+spherical only. Noll 1-3 forced to 0.",
    )
    parser.add_argument(
        "--rotation-search",
        type=float,
        default=0.0,
        help="SLM↔相机相对旋转搜索范围(度, 0~360; 0=关闭旋转校正)",
    )

    args = parser.parse_args()

    center_arg: tuple[int, int] | str | None = args.center
    try:
        parts = args.center.split(",")
        if len(parts) == 2:
            center_arg = (int(parts[0]), int(parts[1]))
    except (ValueError, AttributeError):
        pass

    zernike_mask_arr = None
    if args.zernike_mask is not None:
        zernike_mask_arr = np.array(
            [int(x.strip()) for x in args.zernike_mask.split(",")],
            dtype=int,
        )

    recorder = optimize_slm_square(
        center=center_arg,
        epochs=args.epochs,
        n_max=args.n_max,
        target_side=args.target_side,
        side_factor=args.side_factor,
        delta=args.delta,
        lr=args.lr,
        exposure_time_ms=args.exposure_time_ms,
        cam_id=args.cam_id,
        slm_number=args.slm_number,
        slm_wavelength=args.slm_wavelength,
        optimizer_type=args.optimizer,
        random_seed=args.seed,
        show=args.show,
        cam_size=args.cam_size,
        zernike_mask=zernike_mask_arr,
        rotation_search_deg=args.rotation_search,
    )

    best_iter, (_, best_val) = recorder.get_best_iter()
    logger.info(
        f"Optimization complete. Best quality: {best_val:.4f} "
        f"@ epoch {best_iter.get(RECORD_EPOCH, 'N/A')}"
    )
    save_file = gen_date_dir("data") / f"slm_square_{gen_date_str()}.csv"
    recorder.save_dataframe(save_file)
    logger.info(f"Results saved to: {save_file}")


if __name__ == "__main__":
    main()
