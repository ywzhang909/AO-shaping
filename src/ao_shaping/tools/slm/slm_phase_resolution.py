"""Can the panel actually apply *pixel-scale* phase? Compare against smooth modes.

The question
------------
Every speckle-based model-in-the-loop method needs a random *per-pixel* pupil
phase to produce a far-field speckle field, and then correlates the model's
prediction against the measurement. On this bench that correlation never rose
above ~0.07, and the reason is not a geometry bug: under a per-pixel random phase
the measured far field is a **single tight focus on a dim halo**, exactly like the
flat phase. There is no speckle to correlate.

What this measures
------------------
Drive the pupil with phase of decreasing smoothness and watch the focus degrade:

* per-pixel random phase -- the finest structure the SLM can address
* random Zernike n<=4, n<=8, n<=14 -- progressively smoother

Measured 2026-09-30 on this bench (3.0 ms, beam r=450 px at panel centre):

===========================  =====  ==========
pupil phase                  peak   core40
===========================  =====  ==========
flat                           86    0.0261
per-pixel random (1 SLM px)     92    0.0247   <-- unchanged
random Zernike n<=4            35    0.0195
random Zernike n<=8            24    0.0250
random Zernike n<=14           12    0.0127
===========================  =====  ==========

Per-pixel random phase does **nothing**; smooth Zernike phase degrades the focus
monotonically with order. So the panel's effective phase resolution is far
coarser than one pixel -- high spatial frequencies are strongly low-pass filtered
by the LCOS and the optics.

Consequences
------------
1. Speckle-correlation calibration is unavailable here. Do not spend more time
   tuning aperture candidates, ``far_field_size`` or a waist search against it.
2. The **sweep** route works instead: smooth Zernike modes produce a measurable
   response, and a tilt sweep fixes the far-field scale with no assumption about
   focal length, pixel pitch or the camera pixel size. See
   ``scripts/model_in_loop_hw_runbook.py --stage sweep``.
3. Any pixel-level work needs the panel transfer function characterised and
   pre-compensated first (``slm_lut_runner`` does grayscale->phase).

Usage
-----
::

    python -m ao_shaping.tools.slm.slm_phase_resolution --exposure-ms 3.0
    python -m ao_shaping.tools.slm.slm_phase_resolution --orders 4,8,14 --scale 0.5
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import click
import numpy as np
from loguru import logger

from ao_shaping.tools.slm.bench_kernels import (
    BEAM_RADIUS_PANEL,
    SLM_PANEL_H,
    SLM_PANEL_W,
    core_fraction,
    display_and_average,
    measure_flat_reference,
    measure_spot,
    random_phase,
)
from ao_shaping.utils.cli.params import option, with_params


@dataclass
class PhaseResolutionParams:
    """CLI surface of the phase-resolution probe.

    Field order *is* the ``--help`` order, and the whole eleven-option surface
    is frozen by ``tests/ao_shaping/runners/_cli_help_golden.json``.

    Declared locally rather than spliced from :mod:`ao_shaping.tools.slm.params`:
    the shared ``SlmBenchParams`` types ``--cam-type`` as a ``click.Choice``
    (renders ``[daheng|miicam]``, not ``TEXT``) and leaves ``--exposure-ms`` /
    ``--frames`` without help text, so reuse would change the frozen help.
    """

    slm_number: Annotated[
        int, option("--slm-number", help="SLM 设备编号 (默认 1)")
    ] = 1
    slm_wavelength: Annotated[
        int, option("--slm-wavelength", help="SLM 波长 nm (默认 1064)")
    ] = 1064
    cam_type: Annotated[
        str, option("--cam-type", help="相机类型 (daheng/miicam, 默认 daheng)")
    ] = "daheng"
    cam_id: Annotated[int, option("--cam-id", help="相机 ID (默认 0)")] = 0
    exposure_ms: Annotated[
        float, option("--exposure-ms", help="相机曝光 ms (默认 3.0)")
    ] = 3.0
    zernike_radius: Annotated[
        int,
        option("--zernike-radius", help="Zernike 孔径半径 px (默认 450)"),
    ] = BEAM_RADIUS_PANEL
    pupil_center: Annotated[
        str,
        option("--pupil-center", help="光斑中心 (面板 px 'x,y', 默认 960,600)"),
    ] = "960,600"
    orders: Annotated[
        str,
        option(
            "--orders",
            help="随机 Zernike 的最高阶 (逗号分隔)。阶数越低越光滑, 默认 4,8,14",
        ),
    ] = "4,8,14"
    scale: Annotated[
        float,
        option(
            "--scale",
            help="每阶系数的高斯 sigma (rad)。默认 0.8 对应高阶自动衰减, 保持总 RMS 相当",
        ),
    ] = 0.8
    frames: Annotated[int, option("--frames", help="每帧平均张数 (默认 4)")] = 4
    seed: Annotated[int, option("--seed", help="随机种子 (默认 11)")] = 11


def zernike_random_panel(
    seed: int,
    n_orders: int,
    scale: float,
    radius: int,
    pupil_center: tuple[int, int],
    panel_shape: tuple[int, int],
) -> np.ndarray:
    """A random combination of Zernike modes up to ``n_orders`` on the panel.

    Smooth by construction: the coarsest structure is set by ``n_orders``, not by
    the pixel grid, which is exactly the contrast with a per-pixel random phase.
    Piston is skipped -- it carries no slope and only shifts the gray histogram.
    """
    from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator

    rng = np.random.default_rng(int(seed))
    size = 2 * int(radius) + 1
    gen = ZernikeGenerator((size, size), radius=int(radius), n_orders=int(n_orders))
    acc = np.zeros((size, size), dtype=np.float64)
    for n in range(0, int(n_orders) + 1):
        for m in range(-n, n + 1, 2):
            if (n, m) == (0, 0):
                continue
            z = np.asarray(gen.generate_polynomial({(n, m): 1.0}), dtype=np.float64)
            if z.shape != (size, size):
                z = z.T
            finite = np.isfinite(z)
            patch = np.zeros(z.shape, dtype=np.float64)
            patch[finite] = float(rng.normal(0.0, float(scale))) * z[finite]
            acc += patch
    return _paste(acc, radius, pupil_center, panel_shape)


def _paste(
    local: np.ndarray,
    radius: int,
    pupil_center: tuple[int, int],
    panel_shape: tuple[int, int],
) -> np.ndarray:
    """Paste a local ``(2r+1, 2r+1)`` array onto the panel at the pupil centre."""
    height, width = int(panel_shape[0]), int(panel_shape[1])
    out = np.zeros((height, width), dtype=np.float64)
    cx, cy = int(pupil_center[0]), int(pupil_center[1])
    x0, y0 = max(cx - int(radius), 0), max(cy - int(radius), 0)
    x1 = min(cx + int(radius) + 1, width)
    y1 = min(cy + int(radius) + 1, height)
    out[y0:y1, x0:x1] = local[
        y0 - (cy - int(radius)) : y1 - (cy - int(radius)),
        x0 - (cx - int(radius)) : x1 - (cx - int(radius)),
    ]
    return out


@click.command()
@with_params(PhaseResolutionParams, kw_name="params")
def main(params: PhaseResolutionParams) -> None:
    """比较逐像素随机相位与光滑 Zernike 相位, 判定面板的等效相位分辨率。"""
    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm.santec import Santec

    # Local aliases keep the measurement body below verbatim.
    slm_number = params.slm_number
    slm_wavelength = params.slm_wavelength
    cam_type = params.cam_type
    cam_id = params.cam_id
    exposure_ms = params.exposure_ms
    zernike_radius = params.zernike_radius
    pupil_center = params.pupil_center
    orders = params.orders
    scale = params.scale
    frames = params.frames
    seed = params.seed

    panel = (SLM_PANEL_H, SLM_PANEL_W)
    pupil = tuple(int(float(v)) for v in str(pupil_center).split(","))
    if len(pupil) != 2:
        raise SystemExit("--pupil-center must be 'x,y' in panel pixels")
    order_list = [int(float(v)) for v in str(orders).split(",") if v.strip()]

    with Santec(
        slm_number=slm_number, wavelength=slm_wavelength, video_mode=0
    ) as slm, create_camera(cam_type, cam_id, exposure_time_ms=exposure_ms) as cam:
        cam.reset_exposure_time(float(exposure_ms))
        ref, flat = measure_flat_reference(cam, slm, n_frames=frames, panel_shape=panel)
        flat_core = core_fraction(ref, flat.centroid_x, flat.centroid_y, 40.0)
        logger.info("flat: {}  core40={:.4f}", flat.as_row(), flat_core)

        cases: list[tuple[str, np.ndarray]] = [
            (
                "per-pixel random",
                _paste(
                    random_phase((2 * zernike_radius + 1, 2 * zernike_radius + 1), seed),
                    zernike_radius,
                    pupil,
                    panel,
                ),
            )
        ]
        for k, n in enumerate(order_list):
            cases.append((
                f"Zernike n<={n} random",
                zernike_random_panel(
                    seed + k + 1, n, scale, zernike_radius, pupil, panel
                ),
            ))

        rows: list[tuple[str, float, float, float]] = []
        for name, phase in cases:
            img = display_and_average(cam, slm, phase, n_frames=frames)
            m = measure_spot(img)
            c = core_fraction(img, flat.centroid_x, flat.centroid_y, 40.0)
            rows.append((name, m.peak, c, m.fwhm_px))
            logger.info(
                "  {:<24} peak={:>5.0f} core40={:.4f} fwhm={:.1f}px",
                name, m.peak, c, m.fwhm_px,
            )

    logger.info("")
    logger.info("=== verdict ===")
    per_pixel = next((r for r in rows if r[0] == "per-pixel random"), None)
    if per_pixel is not None and flat.peak > 0:
        ratio = per_pixel[1] / flat.peak if flat.peak else float("nan")
        if 0.85 <= ratio <= 1.15:
            logger.error(
                "per-pixel random phase left the peak at {:.0f} vs flat {:.0f} -- "
                "NO effect. The panel cannot resolve pixel-scale phase, so the "
                "speckle-correlation calibration route is unavailable on this "
                "bench. Use the smooth-mode sweep instead, or calibrate the panel "
                "transfer function (slm_lut_runner) and pre-compensate.",
                per_pixel[1], flat.peak,
            )
        else:
            logger.info(
                "per-pixel random phase changed the peak to {:.0f} vs flat {:.0f} "
                "({:+.0%}) -- pixel-scale phase IS being applied, so speckle "
                "calibration is worth retrying.",
                per_pixel[1], flat.peak, ratio - 1.0,
            )
    smooth = [r for r in rows if r[0].startswith("Zernike")]
    if smooth:
        logger.info(
            "smooth modes degrade the focus: {}",
            ", ".join(f"{r[0]} peak={r[1]:.0f}" for r in smooth),
        )


if __name__ == "__main__":
    main()
