"""用单帧 ±1 级落点反推几何标定 (Kx/Ky/零级中心/旋转)。

为什么需要这个脚本
------------------
:mod:`ao_shaping.tools.slm.calibration` 的 ``align``/``calibrate`` 是一条
多阶段流程 (装配 → 光束位置 → 扫 K → 验证), 其中 ``find_beam_on_slm`` 在光斑
不聚焦/散斑时会把光束判成"铺满整面板" (返回面板角), 之后 ``calibrate`` 的
0 级质心落到暗区 ⇒ ``_moments`` 除零 ⇒ ``K = nan`` ⇒ ``int(round(nan))`` 崩溃
(2026-10-23 实测)。

衍射效率扫描其实只需要三样东西: 零级在哪 (``center``)、预期间距
(``Kx``/``Ky``)、有没有旋转。这三样**一帧光栅图就能量出来**: 0 级质心给中心,
±1 级落点给间距 (``K = 间距 × 周期``), ±1 级相对 0 级的横向分量给旋转角。
本脚本只做这件事, 不碰装配/光束位置, 因此不依赖那条会崩的路径。

判据
----
* 预测间距只作为**搜索窗口**, 实测间距才是结果: ``K_实测 = 间距_实测 × 周期``。
* 残差 ``|间距_实测 − 间距_预测|`` 写进 npz (``resid_x``/``resid_y``), 供下游
  判断这份标定是否可信。
* 过曝 (峰值 > 满量程 80%) 会导致 ±1 级能量不可比, 直接报错而不是照写。

用法
----
``python scripts/slm/calib_geometry_from_orders.py --out data/slm_geometry_calib.npz``
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.tools.kernels.bench_kernels import (
    fresnel_panel,
    measure_spot,
)
from ao_shaping.tools.slm.calibration.diffraction_efficiency.core import (
    BlazeSpec,
    blaze_phase,
    measure_orders,
    phase_to_gray,
    prepare_frame,
)

#: 满量程占比超过它即认为过曝 (与效率扫描同一判据)。
SATURATION_FRAC = 0.8


def _parse_tuple(text: str) -> tuple[float, float]:
    a, b = text.replace("(", "").replace(")", "").split(",")
    return float(a), float(b)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/slm_geometry_calib.npz")
    parser.add_argument("--period-px", type=int, default=32)
    parser.add_argument("--exposure-ms", type=float, default=0.4)
    parser.add_argument("--defocus-focal-mm", type=float, default=1500.0)
    parser.add_argument("--pupil-center", default="960,600")
    parser.add_argument("--pupil-radius", type=int, default=480)
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--slm-wavelength", type=int, default=1064)
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--discard", type=int, default=3)
    parser.add_argument("--settle-s", type=float, default=0.4)
    parser.add_argument(
        "--predicted-k",
        type=float,
        default=7546.4,
        help="K 的先验 (只用作搜索窗口); 实测值才写盘",
    )
    args = parser.parse_args()

    pupil_center = tuple(int(v) for v in _parse_tuple(args.pupil_center))
    pupil_radius = int(args.pupil_radius)

    from ao_shaping.drivers.ccd.common import create_camera, full_scale
    from ao_shaping.drivers.slm.santec import Santec, SlotRotator

    with (
        Santec(
            slm_number=int(args.slm_number),
            video_mode=0,
            wavelength=int(args.slm_wavelength),
        ) as slm,
        create_camera("daheng", int(args.cam_id), exposure_time_ms=float(args.exposure_ms)) as cam,
    ):
        max_gray = int(getattr(slm, "_max_gray", None) or slm.MAX_GRAYSCALE_VALUE)
        panel_shape = (int(slm.Panel_Res[1]), int(slm.Panel_Res[0]))
        rotator = SlotRotator(slm)
        logger.info(
            "SLM #{} serial={} 2π灰度={} 面板={} 周期={}px 曝光={}ms 补偿={}mm",
            args.slm_number,
            getattr(slm, "_serial_number", None),
            max_gray,
            panel_shape,
            args.period_px,
            args.exposure_ms,
            args.defocus_focal_mm,
        )

        defocus = (
            fresnel_panel(
                float(args.defocus_focal_mm),
                radius=pupil_radius,
                pupil_center=pupil_center,
                panel_shape=panel_shape,
                wavelength_nm=float(args.slm_wavelength),
            )
            if math.isfinite(float(args.defocus_focal_mm))
            else np.zeros(panel_shape, dtype=np.float64)
        )

        def show(extra_phase: np.ndarray | None) -> None:
            phase = defocus if extra_phase is None else defocus + extra_phase
            gray = phase_to_gray(phase, max_gray)
            rotator.display_data(slm, gray, float(args.settle_s))

        def capture() -> np.ndarray:
            for _ in range(int(args.discard)):
                cam.get_numpy_image(n_sample=1, skip_first=False)
            frames = [
                np.asarray(
                    cam.get_numpy_image(n_sample=1, skip_first=False), dtype=np.float64
                )
                for _ in range(int(args.frames))
            ]
            return np.mean(np.stack(frames, axis=0), axis=0)

        # --- 0 级 (平场) -------------------------------------------------
        show(None)
        flat = capture()
        scale = float(full_scale(flat))
        spot = measure_spot(flat)
        if spot.peak <= 0:
            logger.error("平场画面无信号 (峰值 {:.1f})", spot.peak)
            return 2
        if spot.peak > SATURATION_FRAC * scale:
            logger.error(
                "过曝: 峰值 {:.0f} / 满量程 {:.0f} —— 降低 --exposure-ms",
                spot.peak,
                scale,
            )
            return 2
        center = (float(spot.centroid_y), float(spot.centroid_x))
        logger.info(
            "0级: 中心=(行 {:.1f}, 列 {:.1f}) FWHM={:.1f}px 峰值={:.0f}/{:.0f}",
            center[0],
            center[1],
            spot.fwhm_px,
            spot.peak,
            scale,
        )

        # --- 两轴光栅 ----------------------------------------------------
        predicted_sep = float(args.predicted_k) / float(args.period_px)
        measured: dict[str, dict[str, float]] = {}
        for axis in ("x", "y"):
            spec = BlazeSpec(
                period_px=int(args.period_px),
                axis=axis,
                depth_rad=math.pi,  # 半波: ±1 级最强
                pupil_center=pupil_center,
                pupil_radius=pupil_radius,
            )
            show(blaze_phase(spec, panel_shape))
            frame = prepare_frame(capture())
            result = measure_orders(
                frame,
                expected_center=center,
                predicted_separation_px=predicted_sep,
            )
            if result.separation_px is None or (
                result.plus_one is None and result.minus_one is None
            ):
                logger.error(
                    "{} 轴: 没量到任何一级 (notes={})", axis, list(result.notes)
                )
                return 3
            sep = float(result.separation_px)
            # 本光路的 SLM 轴与传感器轴**交叉** (实测: x 光栅 ±1 落在行、y 光栅落在列),
            # 所以旋转角按"实测 ±1 连线方向 相对 该轴应有的方向"来量, ±90° 就是交叉。
            plus = result.plus_one
            minus = result.minus_one
            if plus is not None and minus is not None:
                d_row = plus.cy - minus.cy
                d_col = plus.cx - minus.cx
            else:
                other = plus if plus is not None else minus
                d_row = float(other.cy) - center[0]
                d_col = float(other.cx) - center[1]
            expected = math.atan2(0.0, 1.0) if axis == "x" else math.atan2(1.0, 0.0)
            angle = math.degrees(math.atan2(d_row, d_col) - expected)
            # 折到 (-180, 180]
            angle = (angle + 180.0) % 360.0 - 180.0
            measured[axis] = {
                "separation_px": sep,
                "K": sep * float(args.period_px),
                "separation_row_col": [d_row, d_col],
                "angle_vs_axis_deg": angle,
                "residual_px": float(abs(sep - predicted_sep)),
                "eta_plus_total": float(result.eta_plus_total or 0.0),
                "eta_zero": float(result.eta_zero or 0.0),
                "both_orders": plus is not None and minus is not None,
            }
            logger.info(
                "{} 轴: 间距={:.2f}px (预测 {:.2f}) K={:.1f} 方向(行,列)=({:+.1f},{:+.1f}) "
                "相对轴向={:+.1f}° η±1={:.3f} η0={:.3f} 两侧可辨={}",
                axis,
                sep,
                predicted_sep,
                measured[axis]["K"],
                d_row,
                d_col,
                angle,
                measured[axis]["eta_plus_total"],
                measured[axis]["eta_zero"],
                measured[axis]["both_orders"],
            )

        # --- 落盘 --------------------------------------------------------
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        provenance = {
            "source": "scripts/slm/calib_geometry_from_orders.py",
            "measured_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            "period_px": int(args.period_px),
            "exposure_ms": float(args.exposure_ms),
            "defocus_focal_mm": float(args.defocus_focal_mm),
            "pupil_center": list(pupil_center),
            "pupil_radius": int(pupil_radius),
            "slm_serial": str(getattr(slm, "_serial_number", "")),
            "method": "single-frame ±1 order separation, K = separation × period",
            "predicted_k": float(args.predicted_k),
            "axes": measured,
        }
        np.savez(
            out,
            center=np.asarray(center, dtype=np.float64),
            Kx=np.float64(measured["x"]["K"]),
            Ky=np.float64(measured["y"]["K"]),
            rotation_deg=np.float64(
                0.5 * (measured["x"]["angle_vs_axis_deg"] + measured["y"]["angle_vs_axis_deg"])
            ),
            fwhm0=np.float64(spot.fwhm_px),
            mirror=np.bool_(False),
            resid_x=np.float64(measured["x"]["residual_px"]),
            resid_y=np.float64(measured["y"]["residual_px"]),
            crop_side=np.int64(0),
            beam_center=np.zeros(2, dtype=np.float64),
            beam_sigma=np.zeros(2, dtype=np.float64),
            provenance=np.asarray(json.dumps(provenance, ensure_ascii=False)),
        )
        logger.info("几何标定已写出: {} (中心={}, Kx={:.1f}, Ky={:.1f})", out, center, measured["x"]["K"], measured["y"]["K"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
