"""Locate the beam on the SLM panel, in panel coordinates.

Why this is needed
------------------
Everything downstream -- the pupil phase, the injected defocus, the modelled
aperture -- has to be placed where the beam actually is. The camera's 0-order
position is **not** that place: on this bench the two axes are swapped 90
degrees and the scales differ by more than an order of magnitude. The runbook
used to derive a panel offset from the camera 0-order and computed ``(+426, -290)``
panel px, which rolled the pupil phase clean off the beam.

So measure it instead of deriving it: write a random-phase disc at candidate
panel positions and keep the one whose frame differs most from flat. Only a patch
that actually covers the beam can scatter the 0-order.

Metric
------
The change in energy inside a small box at the 0-order. Two alternatives were
tried and both failed:

* *total* frame energy is conserved -- measured 0.999-1.005 across aperture radii
  from 100 to 850 px -- so it carries no position or size information;
* the 0-order box energy is **not** monotone in patch size, because a disc both
  scatters light out of the box and redistributes light into it.

Usage
-----
::

    python -m ao_shaping.tools.slm.slm_panel_locate --exposure-ms 3.0
    python -m ao_shaping.tools.slm.slm_panel_locate --grid-xs 400,700,1000,1300 --grid-ys 300,600,900
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import click
import numpy as np
from loguru import logger

from ao_shaping.tools.slm.slm_bench_probe import (
    BEAM_RADIUS_PANEL,
    SLM_PANEL_H,
    SLM_PANEL_W,
    display_and_average,
    measure_flat_reference,
    random_phase,
)
from ao_shaping.utils.cli_params import option, with_params


def _box_mask(shape: tuple[int, int], cx: int, cy: int, r: int) -> np.ndarray:
    iy, ix = np.mgrid[0 : shape[0], 0 : shape[1]]
    return (iy - cy) ** 2 + (ix - cx) ** 2 <= r * r


@dataclass
class PanelLocateParams:
    """面板坐标光斑定位探针的 CLI 参数。

    ``--slm-wavelength`` 默认 1064 nm (本台架红外工位) 与 ``--patch-radius``
    默认 ``BEAM_RADIUS_PANEL`` (实测光斑半径 450 面板 px) 都是**本台架**标定的
    物理常数; 候选网格 ``--grid-xs``/``--grid-ys`` 的默认值覆盖 240..1680 px,
    即整块面板, 因为相机 0 阶坐标与面板坐标轴互换 90° 且尺度差 >10 倍 ——
    面板偏移必须实测, 不能从相机坐标推导。
    """

    slm_number: Annotated[int, option("--slm-number", help="SLM 设备编号 (默认 1)")] = 1
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
    patch_radius: Annotated[
        int,
        option(
            "--patch-radius",
            help="随机相位圆盘半径 (面板 px, 默认 450=实测光斑半径)",
        ),
    ] = BEAM_RADIUS_PANEL
    grid_xs: Annotated[
        str, option("--grid-xs", help="候选 x (面板 px, 逗号分隔)")
    ] = "240,600,960,1320,1680"
    grid_ys: Annotated[
        str, option("--grid-ys", help="候选 y (面板 px, 逗号分隔)")
    ] = "220,480,720,980"
    frames: Annotated[int, option("--frames", help="每帧平均张数 (默认 4)")] = 4
    seed: Annotated[int, option("--seed", help="随机相位种子 (默认 7)")] = 7


@click.command()
@with_params(PanelLocateParams, kw_name="params")
def main(params: PanelLocateParams) -> None:
    """在面板坐标上定位光斑 (扫描随机相位圆盘, 取 0 阶能量变化最大的位置)。"""
    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm.santec import Santec

    xs = [int(float(v)) for v in str(params.grid_xs).split(",") if v.strip()]
    ys = [int(float(v)) for v in str(params.grid_ys).split(",") if v.strip()]
    panel = (SLM_PANEL_H, SLM_PANEL_W)
    rng = np.random.default_rng(int(params.seed))

    with Santec(
        slm_number=params.slm_number, wavelength=params.slm_wavelength, video_mode=0
    ) as slm, create_camera(
        params.cam_type, params.cam_id, exposure_time_ms=params.exposure_ms
    ) as cam:
        cam.reset_exposure_time(float(params.exposure_ms))
        ref, flat = measure_flat_reference(
            cam, slm, n_frames=params.frames, panel_shape=panel
        )
        box = _box_mask(ref.shape, int(round(flat.centroid_x)),
                        int(round(flat.centroid_y)), 60)
        base = float(ref[box].sum())
        logger.info(
            "flat reference: {}  0-order box energy={:.0f}", flat.as_row(), base
        )
        if base <= 0:
            raise SystemExit("0-order box energy is zero; cannot score patches")

        results: list[tuple[float, int, int, float]] = []
        for yi, py in enumerate(ys):
            row: list[float] = []
            for xi, px in enumerate(xs):
                phase = np.zeros(panel, dtype=np.float64)
                yy, xx = np.mgrid[0 : panel[0], 0 : panel[1]]
                m = (yy - py) ** 2 + (xx - px) ** 2 <= int(params.patch_radius) ** 2
                phase[m] = rng.uniform(0.0, 2.0 * np.pi, int(m.sum()))
                img = display_and_average(cam, slm, phase, n_frames=params.frames)
                delta = float((img[box].sum() - base) / base)
                row.append(delta)
                results.append((abs(delta), py, px, delta))
                del xi, yi
            logger.info(
                "panel y={:>4}: {}", py,
                "  ".join(f"x={px}:{v:+7.3f}" for px, v in zip(xs, row)),
            )

    results.sort(reverse=True)
    logger.info("")
    logger.info("=== strongest patch positions (|0-order energy change|) ===")
    for score, py, px, delta in results[:5]:
        logger.info("  panel (x={:>4}, y={:>4})  dE/E={:+.4f}", px, py, delta)
    _, by, bx, _ = results[0]
    logger.info("")
    logger.info(
        "beam is near panel (x={}, y={}); panel centre is ({}, {})",
        bx, by, SLM_PANEL_W // 2, SLM_PANEL_H // 2,
    )
    logger.info(
        "offset from panel centre: dx={:+d}, dy={:+d} panel px -- pass this to the "
        "runbook as --pupil-center, never a camera coordinate",
        bx - SLM_PANEL_W // 2, by - SLM_PANEL_H // 2,
    )


if __name__ == "__main__":
    main()
