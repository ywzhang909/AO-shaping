"""
Validation script for flat-phase (平场) gray-level fix.

Usage (on hardware):
    python scripts/validate_flat_phase_gray.py                     # quick 5-point test
    python scripts/validate_flat_phase_gray.py --scan --gray-step 30  # full-range scan
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
from loguru import logger

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ao_shaping.drivers.ccd.miicam_driver import CameraStreamManager
from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200


def _flat_phase(slm: SantecSLM200, gray_value: int) -> np.ndarray:
    height, width = slm.Panel_Res[1], slm.Panel_Res[0]
    return np.full((height, width), gray_value, dtype=np.uint16)


def _capture_brightness(camera: CameraStreamManager, discard: int = 3) -> dict:
    for _ in range(discard):
        camera.get_numpy_image(n_sample=1)
    frame = camera.get_numpy_image(n_sample=1)
    return {
        "max": int(frame.max()),
        "min": int(frame.min()),
        "mean": float(frame.mean()),
        "sum": float(frame.sum()),
    }


@click.command()
@click.option("--slm-number", type=click.IntRange(1, 8), default=1, show_default=True)
@click.option("--miicam-id", type=click.IntRange(0), default=0, show_default=True)
@click.option("--wavelength", type=click.IntRange(450, 1600), default=1064, show_default=True)
@click.option("--exposure-ms", type=float, default=3.0, show_default=True)
@click.option("--wait-time-s", type=float, default=0.3, show_default=True)
@click.option("--discard-count", type=int, default=3, show_default=True)
@click.option("--bit-depth", type=click.Choice(["8", "16"]), default="8", show_default=True)
@click.option("--scan", is_flag=True, default=False, help="Scan full range [0..2π_gray]")
@click.option("--gray-step", type=int, default=50, help="Step for scan mode")
def run(
    slm_number: int,
    miicam_id: int,
    wavelength: int,
    exposure_ms: float,
    wait_time_s: float,
    discard_count: int,
    bit_depth: int,
    scan: bool,
    gray_step: int,
) -> None:
    """Test flat-phase gray values and verify camera sees different brightness."""

    with SantecSLM200(
        slm_number=slm_number,
        wavelength=wavelength,
        video_mode=0,
    ) as slm, CameraStreamManager(
        cam_id=miicam_id,
        exposure_time_ms=exposure_ms,
        bit_depth=int(bit_depth),
    ) as camera:
        wl, g_2pi = slm.get_wavelength_info()
        g_pi = g_2pi // 2
        g_pi2 = g_2pi // 4
        g_3pi2 = 3 * g_2pi // 4

        if scan:
            gray_values = list(range(0, g_2pi + 1, gray_step))
            if gray_values[-1] != g_2pi:
                gray_values.append(g_2pi)
            mode_str = f"scan (step={gray_step}, {len(gray_values)} pts)"
        else:
            gray_values = [0, g_pi2, g_pi, g_3pi2, g_2pi, 1023]
            mode_str = f"quick: 0, {g_pi2}(π/2), {g_pi}(π), {g_3pi2}(3π/2), {g_2pi}(2π), 1023"

        logger.info(f"SLM: wl={wl}nm, 2π_gray={g_2pi}")
        logger.info(f"Mode: {mode_str} | exp={exposure_ms}ms wait={wait_time_s}s discard={discard_count}")
        logger.info(f"{'gray':>6} {'max':>6} {'min':>6} {'mean':>10} {'sum':>14}")
        logger.info("-" * 50)

        results = []
        for gv in gray_values:
            slm.display_data(_flat_phase(slm, gv), wait_time_s)
            stats = _capture_brightness(camera, discard=discard_count)
            results.append((gv, stats))
            logger.info(
                f"{gv:>6} {stats['max']:>6} {stats['min']:>6} "
                f"{stats['mean']:>10.3f} {stats['sum']:>14.0f}"
            )

        logger.info("-" * 50)
        distinct = len(set(s["max"] for _, s in results))
        if distinct >= 2:
            logger.info(f"✅ {distinct} distinct brightness levels detected.")
            if distinct >= 5:
                logger.info("✅ Excellent dynamic range across phase.")
        else:
            logger.warning("❌ All identical. Try --exposure-ms 30 --wait-time-s 1.0 --discard-count 5")


if __name__ == "__main__":
    run()
