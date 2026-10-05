"""Measure the beam's centre and radius **on the panel**, by boundary sweep.

Why not just look
-----------------
The illuminated footprint decides the modelled aperture, the Zernike radius of
any injected mode, and where the pupil phase must be written. A stale number in a
doc is worse than none: this bench's calibration notes claimed "~192 SLM px" for
months while the beam was in fact ~450 px, and a run using the stale number
modulated under half the pupil.

Method
------
Sweep the boundary of a half-plane of random phase and watch the 0-order energy
concentration fall. Where the boundary crosses the beam, light is scattered out of
the core and the concentration drops; past the beam edge, moving the boundary
changes nothing. The knee is the beam edge. This is a *location* measurement --
no peak-picking, no centroid, no FWHM.

Flat must be displayed first
----------------------------
The panel retains whatever pattern was last displayed, so a "flat" reference read
before any write is the previous run's speckle. That single mistake made the beam
look four times dimmer than it is between runs and sent the first version of this
measurement down a dead end.

Usage
-----
::

    python -m ao_shaping.tools.slm.slm_beam_extent --exposure-ms 3.0
    python -m ao_shaping.tools.slm.slm_beam_extent --axis x --steps 16
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import click
import numpy as np
from loguru import logger

from ao_shaping.tools.slm.slm_bench_probe import (
    SLM_PANEL_H,
    SLM_PANEL_W,
    core_fraction,
    display_and_average,
    measure_flat_reference,
    random_phase,
)
from ao_shaping.utils.cli.params import option, with_params


@dataclass
class BeamExtentParams:
    """CLI surface of the beam-extent probe.

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
    axis: Annotated[
        str,
        option("--axis", type=click.Choice(["x", "y"]), help="扫描轴 (默认 x)"),
    ] = "x"
    steps: Annotated[int, option("--steps", help="边界位置数 (默认 12)")] = 12
    frames: Annotated[int, option("--frames", help="每帧平均张数 (默认 4)")] = 4
    repeats: Annotated[
        int, option("--repeats", help="每个边界重复次数 (默认 2)")
    ] = 2
    core_radius: Annotated[
        float, option("--core-radius", help="中心盘半径 px (默认 40)")
    ] = 40.0
    seed: Annotated[int, option("--seed", help="随机相位种子 (默认 2024)")] = 2024


@click.command()
@with_params(BeamExtentParams, kw_name="params")
def main(params: BeamExtentParams) -> None:
    """用半平面随机相位边界扫描测光斑在面板上的中心与半径。"""
    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm.santec import Santec

    # Local aliases keep the measurement body below verbatim.
    slm_number = params.slm_number
    slm_wavelength = params.slm_wavelength
    cam_type = params.cam_type
    cam_id = params.cam_id
    exposure_ms = params.exposure_ms
    axis = params.axis
    steps = params.steps
    frames = params.frames
    repeats = params.repeats
    core_radius = params.core_radius
    seed = params.seed

    panel = (SLM_PANEL_H, SLM_PANEL_W)
    span = SLM_PANEL_W if axis == "x" else SLM_PANEL_H
    positions = np.linspace(0, span, int(steps) + 1)[1:]
    rng = np.random.default_rng(int(seed))
    base_rand = random_phase(panel, seed=int(seed))
    yy, xx = np.mgrid[0 : panel[0], 0 : panel[1]]

    with Santec(
        slm_number=slm_number, wavelength=slm_wavelength, video_mode=0
    ) as slm, create_camera(cam_type, cam_id, exposure_time_ms=exposure_ms) as cam:
        cam.reset_exposure_time(float(exposure_ms))
        ref, flat = measure_flat_reference(cam, slm, n_frames=frames, panel_shape=panel)
        f0 = core_fraction(ref, flat.centroid_x, flat.centroid_y, core_radius)
        logger.info(
            "flat reference: {}  core_fraction={:.4f}", flat.as_row(), f0
        )
        if f0 <= 0:
            raise SystemExit("flat core fraction is zero; cannot score the sweep")

        rows: list[tuple[int, float]] = []
        for t in positions:
            mask = (xx < t) if axis == "x" else (yy < t)
            fracs: list[float] = []
            for _ in range(int(repeats)):
                phase = np.where(mask, base_rand, 0.0)
                img = display_and_average(cam, slm, phase, n_frames=frames)
                fracs.append(
                    core_fraction(img, flat.centroid_x, flat.centroid_y, core_radius)
                )
            cf = float(np.mean(fracs))
            rows.append((int(t), cf))
            logger.info(
                "  boundary at {a}={t:>5}  area={p:>5.1f}%  core_fraction={c:.4f}  "
                "rel={r:.3f}",
                a=axis, t=int(t), p=100.0 * mask.sum() / mask.size,
                c=cf, r=cf / f0,
            )

    rel = [c / f0 for _, c in rows]
    logger.info("")
    logger.info("profile (rel): {}", " ".join(f"{v:.2f}" for v in rel))
    # The beam edge is where rel stops falling; report the last index still
    # meaningfully above the fully-randomised floor.
    floor = float(np.min(rel))
    saturated = [i for i, v in enumerate(rel) if v <= floor * 1.10]
    edge = positions[saturated[0]] if saturated else positions[-1]
    logger.info(
        "randomising up to {a}={e} px saturates the response (floor rel={f:.2f}), "
        "so the beam's {a} extent ends near {e} px",
        a=axis, e=int(edge), f=floor,
    )
    if axis == "x":
        logger.info(
            "if the beam is centred, that implies centre ~{c} px, radius ~{r} px",
            c=int(edge) // 2, r=int(edge) // 2,
        )
    else:
        logger.info("panel height is {}, so compare with the x scan", SLM_PANEL_H)
    logger.info(
        "run both axes before trusting a radius: the first few points of a scan "
        "are non-monotonic because each is only {} random draw(s)", int(repeats)
    )


if __name__ == "__main__":
    main()
