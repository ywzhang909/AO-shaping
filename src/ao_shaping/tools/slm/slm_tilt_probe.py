"""Decisive "is the panel actually modulating?" probe, via SLM tilt.

Why a tilt
----------
A blazed or binary grating is a *bad* probe: its first-order efficiency is low
and oblique incidence moves where the order lands, so "no order appeared" is
ambiguous between a dead panel and a badly chosen period. A tilt has neither
problem — a 2*pi phase ramp over ``P`` panel pixels moves the focal spot by a
known amount regardless of efficiency, and the shift is measured by tracking the
spot, not by looking for a new one.

This is the probe that settled a false fault on this bench. Flat vs grating vs
full-panel random phase all appeared to give *identical* frames (normalised
full-frame L2 ~5e-5), which reads as a dead LCOS. They were not: the metric could
not see a moving spot, because a 60 px spot moving 63 px changes ~1e-4 of a
5.0-Mpx frame. The tilt showed the spot moving immediately.

What it reports
---------------
For each ramp period: the displayed memory slot, the spot, the measured shift,
and the shift the empirical bench scale predicts. A working panel matches; a
frozen one shows zero shift against a non-zero prediction.

Usage
-----
::

    python -m ao_shaping.tools.slm.slm_tilt_probe --exposure-ms 3.0
    python -m ao_shaping.tools.slm.slm_tilt_probe --periods 480,240,120 --axis y

The axis is in **panel** coordinates. The panel and camera axes are swapped 90
degrees on this bench, so a panel-x ramp moves the spot in camera-y; the probe
reports the movement on whichever camera axis it actually sees and flags the
mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import click
import numpy as np
from loguru import logger

from ao_shaping.tools.slm.bench_kernels import (
    SLM_PANEL_H,
    SLM_PANEL_W,
    SLM_PITCH_M,
    TILT_SHIFT_SCALE,
    display_and_average,
    fit_linear_slope,
    measure_flat_reference,
    measure_spot,
    ramp_panel,
)
from ao_shaping.utils.cli.params import option, with_params


@dataclass
class TiltProbeParams:
    """CLI surface of the tilt probe.

    Field order *is* the ``--help`` order, and the whole nine-option surface is
    frozen by ``tests/ao_shaping/runners/_cli_help_golden.json``.

    These five device/acquisition knobs are declared locally rather than spliced
    from :mod:`ao_shaping.tools.slm.params`. The shared groups are deliberately
    *not* a drop-in here: :class:`~ao_shaping.tools.slm.params.SlmBenchParams`
    types ``--cam-type`` as a ``click.Choice`` (which renders
    ``[daheng|miicam]``, not ``TEXT``) and leaves ``--exposure-ms``/``--frames``
    without help text, so reusing it would silently change the frozen help.
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
        float,
        option(
            "--exposure-ms",
            help="相机曝光 ms (默认 3.0; 1.1 ms 落在 0 阶峰值 ~60, 但散斑帧太暗)",
        ),
    ] = 3.0
    periods: Annotated[
        str,
        option(
            "--periods",
            help="2*pi 斜坡周期 (面板 px, 逗号分隔)。位移 = 7600/period 相机 px, "
            "所以周期必须大——周期 1 会把光斑甩出 5.7 mm 画框",
        ),
    ] = "480,240,120"
    axis: Annotated[
        str,
        option(
            "--axis",
            type=click.Choice(["x", "y"]),
            help="倾斜轴 (面板坐标, 默认 x)",
        ),
    ] = "x"
    frames: Annotated[int, option("--frames", help="每帧平均张数 (默认 4)")] = 4
    repeat: Annotated[
        int, option("--repeat", help="每个周期重复次数 (默认 2)")
    ] = 2


@click.command()
@with_params(TiltProbeParams, kw_name="params")
def main(params: TiltProbeParams) -> None:
    """用相位倾斜斜坡判定面板是否真的在调制 (比光栅可靠得多)。"""
    # Local aliases keep the position-sensitive measurement body below verbatim.
    slm_number = params.slm_number
    slm_wavelength = params.slm_wavelength
    cam_type = params.cam_type
    cam_id = params.cam_id
    exposure_ms = params.exposure_ms
    periods = params.periods
    axis = params.axis
    frames = params.frames
    repeat = params.repeat

    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm.santec import MEMORY_MODE_INTERNAL, Santec

    period_list = [int(float(p)) for p in str(periods).split(",") if p.strip()]
    if not period_list:
        raise SystemExit("--periods 没有解析出任何周期")
    panel = (SLM_PANEL_H, SLM_PANEL_W)
    axis_index = 1 if axis == "x" else 0
    points: list[tuple[float, float, float, int]] = []  # (period, cx, cy, slot)

    with Santec(
        slm_number=slm_number, wavelength=slm_wavelength, video_mode=0
    ) as slm, create_camera(cam_type, cam_id, exposure_time_ms=exposure_ms) as cam:
        cam.reset_exposure_time(float(exposure_ms))

        # Flat FIRST: the panel retains the last displayed pattern, so a "flat"
        # read before any write is the previous run's speckle.
        _, flat = measure_flat_reference(cam, slm, n_frames=frames, panel_shape=panel)
        logger.info(
            "flat reference: {}  (0-order is the frame's brightest point, never "
            "the geometric centre)", flat.as_row()
        )

        for period in period_list:
            for rep in range(int(repeat)):
                gray = slm.create_phase_from_array(ramp_panel(period, axis_index, panel))
                slot = slm.display_data(gray, memory_mode=MEMORY_MODE_INTERNAL)
                img = display_and_average(
                    cam, slm, ramp_panel(period, axis_index, panel), n_frames=frames
                )
                m = measure_spot(img)
                points.append((float(period), m.centroid_x, m.centroid_y, int(slot)))
                logger.info(
                    "panel-{a} ramp period {p:>4} px  rep {r}  slot={s:<4} {m}",
                    a=axis, p=period, r=rep + 1, s=slot, m=m.as_row(),
                )

    # The panel-x ramp moves the spot along one *camera* axis; find out which.
    periods_arr = np.array([p[0] for p in points], dtype=np.float64)
    cx = np.array([p[1] for p in points], dtype=np.float64)
    cy = np.array([p[2] for p in points], dtype=np.float64)
    # The focal shift is proportional to 1/period, NOT to period -- a 2*pi ramp
    # over P px deflects by lambda/(P*d_slm). Regressing against `period` (which
    # is what this did before) fits a near-constant and reports a meaningless
    # slope: it read 0.025 cam px per period where the correct figure against
    # 1/period is ~7400.
    inv_periods = 1.0 / periods_arr
    kx, _ = fit_linear_slope(inv_periods, cx)
    ky, _ = fit_linear_slope(inv_periods, cy)
    slope = kx if abs(kx) >= abs(ky) else ky
    moved_axis = "camera-x" if abs(kx) >= abs(ky) else "camera-y"

    logger.info("")
    logger.info("=== verdict ===")
    logger.info(
        "displacement vs 1/period: camera-x {:+.1f}, camera-y {:+.1f} cam px per 1/px",
        kx, ky,
    )
    scale = abs(slope)
    logger.info(
        "empirical shift_px = {:.0f}/period  (the TILT_SHIFT_SCALE constant, which "
        "this run measures at {:.0f})", scale, scale,
    )
    logger.info(
        "the bench constant in slm_bench_probe is {}. Ratio {:.2f} -- update it if "
        "this is a different setup.",
        int(TILT_SHIFT_SCALE), scale / TILT_SHIFT_SCALE,
    )
    logger.info(
        "a panel-{a} ramp moved the spot along {b} -- {v} the expected 90 degree "
        "axis swap",
        a=axis, b=moved_axis,
        v=(
            "CONFIRMING"
            if (axis == "x") == (moved_axis == "camera-y")
            else "NOT the"
        ),
    )
    mean_slot = float(np.mean([p[3] for p in points]))
    logger.info(
        "displayed memory slot advanced to ~{avg:.0f} over {n} writes -- if it had "
        "not advanced, the firmware no-op would have made every frame stale",
        avg=mean_slot, n=len(points),
    )
    if not np.isfinite(slope) or abs(slope) < 1.0:
        logger.error(
            "NO measurable shift: the spot did not move with the tilt. The panel "
            "is not applying phase (or the beam is not on the addressed area)."
        )
    else:
        logger.info(
            "the spot DOES move with the tilt, so the panel is applying phase. "
            "Measured scale {:.0f} cam px per 1/period; theory for a {:.1f} mm "
            "aperture is f*lambda/(pi*R*camera_pixel) = {:.0f}.",
            scale, 1e3 * 450 * SLM_PITCH_M,
            0.125 * 1064e-9 / (np.pi * 450 * SLM_PITCH_M * 2.2e-6),
        )


if __name__ == "__main__":
    main()
