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

Acquisition policy
------------------
The flat reference and every boundary position are taken with
``ProbePolicy(mode="frames")``: a fixed budget of frames combined **per pixel**
by median, no settle criterion. The criterion (peak, centroid and width all
steady) serves measurements that read position and shape; this one reads a *core
fraction*, i.e. an energy ratio at a frozen centroid, so a varying centroid is
irrelevant to it and a settling gate that waits on one can never open.

A median rather than a mean because a transient frame biases ``core_fraction``
**downward only** -- light scattered out of the core is not compensated by any
frame that had extra light in it -- and a downward bias inflates the reported
width. Rejection removes that failure mode where averaging would bake it in.

``--frames`` is the kept-frame budget's floor: a median needs enough samples to
reject one corrupted frame, so the probe never keeps fewer than seven.

Usage
-----
::

    python -m ao_shaping.tools.slm.panel.beam_extent --exposure-ms 3.0
    python -m ao_shaping.tools.slm.panel.beam_extent --axis x --steps 16
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import click
import numpy as np
from loguru import logger

from ao_shaping.tools.kernels.bench_kernels import (
    SLM_PANEL_H,
    SLM_PANEL_W,
    core_fraction,
    display_and_average,
    measure_flat_reference,
    random_phase,
)
from ao_shaping.tools.slm.panel.probe_policy import probe_policy
from ao_shaping.utils.cli.params import option, with_params


@dataclass
class BeamExtentParams:
    """CLI surface of the beam-extent probe.

    Field order *is* the ``--help`` order, and the whole eleven-option surface
    is frozen by ``tests/ao_shaping/runners/cli_help_golden.json``.

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
    frames: Annotated[
        int, option("--frames", help="中值合并保留的帧数 (默认 7, 最少 7)")
    ] = 7
    repeats: Annotated[
        int, option("--repeats", help="每个边界重复次数 (默认 2)")
    ] = 2
    core_radius: Annotated[
        float, option("--core-radius", help="中心盘半径 px (默认 40)")
    ] = 40.0
    seed: Annotated[int, option("--seed", help="随机相位种子 (默认 2024)")] = 2024


#: How far below the flat response counts as "the fall has started". The first
#: couple of points of a scan are non-monotonic (one random draw each), so a
#: tiny threshold would latch onto noise instead of the beam's near edge.
_ONSET_DROP = 0.03


@dataclass(frozen=True)
class BeamExtentResult:
    """One axis of the boundary sweep, and the edge it implies.

    ``edge_px`` is where the response stopped falling: the **far** edge of the
    illuminated area along ``axis``, measured from the panel origin. The beam's
    centre is *not* measured here -- take it from ``panel_locate``.
    ``radius_px`` assumes the illumination starts at the origin, which holds
    only for a beam parked at the panel centre, so it is an **upper bound**
    otherwise, not a radius.
    """

    axis: str
    edge_px: int
    radius_px: int
    centre_px: float
    fall_lo_px: int
    fall_hi_px: int
    step_px: float
    floor_rel: float
    profile_rel: tuple[float, ...]
    positions_px: tuple[int, ...]
    core_fraction_flat: float
    flat_centroid: tuple[float, float]
    flat_peak: float


def sweep_axis(
    *,
    axis: str,
    steps: int,
    frames: int,
    repeats: int,
    core_radius: float,
    seed: int,
    slm_number: int,
    slm_wavelength: int,
    cam_type: str,
    cam_id: int,
    exposure_ms: float,
) -> BeamExtentResult:
    """Run the boundary sweep on one axis and return what it implies.

    Split out of the Click command so the GUI can call the measurement directly:
    ``beam_extent`` is the tool the model-in-loop geometry warning names when its
    two focal-scale routes disagree, and an operator chasing that warning should
    not have to leave the page to re-measure the illuminated aperture.
    """
    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm.santec import Santec

    panel = (SLM_PANEL_H, SLM_PANEL_W)
    span = SLM_PANEL_W if axis == "x" else SLM_PANEL_H
    positions = np.linspace(0, span, int(steps) + 1)[1:]
    base_rand = random_phase(panel, seed=int(seed))
    yy, xx = np.mgrid[0 : panel[0], 0 : panel[1]]
    policy = probe_policy(frames)

    with Santec(
        slm_number=slm_number, wavelength=slm_wavelength, video_mode=0
    ) as slm, create_camera(cam_type, cam_id, exposure_time_ms=exposure_ms) as cam:
        cam.reset_exposure_time(float(exposure_ms))
        ref, flat = measure_flat_reference(
            cam, slm, n_frames=frames, panel_shape=panel,
            policy=policy, wait_time_s=None,
        )
        f0 = core_fraction(ref, flat.centroid_x, flat.centroid_y, core_radius)
        logger.info(
            "flat reference: {}  core_fraction={:.4f}", flat.as_row(), f0
        )
        if f0 <= 0:
            raise RuntimeError(
                "flat core fraction is zero; the beam is not reaching the camera "
                "core, so there is nothing to sweep"
            )

        rows: list[tuple[int, float]] = []
        for t in positions:
            mask = (xx < t) if axis == "x" else (yy < t)
            fracs: list[float] = []
            for _ in range(int(repeats)):
                phase = np.where(mask, base_rand, 0.0)
                img = display_and_average(
                    cam, slm, phase, n_frames=frames,
                    policy=policy, wait_time_s=None,
                )
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
    # The beam edge is where rel stops falling; the last index still
    # meaningfully above the fully-randomised floor.
    floor = float(np.min(rel))
    saturated = [i for i, v in enumerate(rel) if v <= floor * 1.10]
    edge = positions[saturated[0]] if saturated else positions[-1]

    # Centre-free width: the response holds at ~1 until the boundary crosses the
    # beam's near edge, falls across the illuminated width, then pins at the
    # floor. So the *width of the fall* is the beam's diameter along this axis
    # and its midpoint is the centre -- both from this one scan, with no
    # dependency on where ``panel_locate`` happened to put the centre.
    #
    # This matters because ``far_edge - centre`` inherits every error in the
    # centre. Measured 2026-10-17: the same scan gave a radius of 180 px with a
    # previous session's centre and 300 px with the panel's geometric one -- a
    # 1.7x swing from the centre alone, far larger than the step size.
    onset = 0
    for i, v in enumerate(rel):
        if v < 1.0 - _ONSET_DROP:
            onset = i
            break
    fall_lo = positions[onset]
    fall_hi = positions[saturated[0]] if saturated else positions[-1]
    width_px = float(fall_hi - fall_lo)
    centre_px = float(fall_lo + fall_hi) / 2.0

    return BeamExtentResult(
        axis=axis,
        edge_px=int(edge),
        radius_px=int(width_px / 2.0),
        centre_px=centre_px,
        fall_lo_px=int(fall_lo),
        fall_hi_px=int(fall_hi),
        step_px=float(positions[1] - positions[0]) if len(positions) > 1 else 0.0,
        floor_rel=floor,
        profile_rel=tuple(float(v) for v in rel),
        positions_px=tuple(int(t) for t in positions),
        core_fraction_flat=float(f0),
        flat_centroid=(float(flat.centroid_x), float(flat.centroid_y)),
        flat_peak=float(flat.peak),
    )


@click.command()
@with_params(BeamExtentParams, kw_name="params")
def main(params: BeamExtentParams) -> None:
    """用半平面随机相位边界扫描测光斑在面板上的中心与半径。"""
    axis = params.axis
    repeats = params.repeats

    result = sweep_axis(
        axis=axis,
        steps=params.steps,
        frames=params.frames,
        repeats=repeats,
        core_radius=params.core_radius,
        seed=params.seed,
        slm_number=params.slm_number,
        slm_wavelength=params.slm_wavelength,
        cam_type=params.cam_type,
        cam_id=params.cam_id,
        exposure_ms=params.exposure_ms,
    )
    rel = list(result.profile_rel)
    logger.info("")
    logger.info("profile (rel): {}", " ".join(f"{v:.2f}" for v in rel))
    edge = result.edge_px
    logger.info(
        "randomising up to {a}={e} px saturates the response (floor rel={f:.2f}), "
        "so the beam's {a} extent ends near {e} px",
        a=axis, e=edge, f=result.floor_rel,
    )
    if axis == "x":
        logger.info(
            "if the beam is centred, that implies centre ~{c} px, radius ~{r} px",
            c=edge // 2, r=result.radius_px,
        )
    else:
        logger.info("panel height is {}, so compare with the x scan", SLM_PANEL_H)
    logger.info(
        "run both axes before trusting a radius: the first few points of a scan "
        "are non-monotonic because each is only {} random draw(s)", int(repeats)
    )


if __name__ == "__main__":
    main()
