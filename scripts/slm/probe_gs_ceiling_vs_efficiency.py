"""用 SLM_Ctrl_WriteGS 扫"面板灰阶上限"对闪耀光栅效率的影响。

背景
----
用户 2026-10-23 指出两点, 本探针直接验证:

1. **图案要真的到 1023**: 之前的效率扫描用 ``slm._max_gray`` (=993, 设备按波长
   标称的 2π 灰度) 当斜坡上界, 于是光栅最大灰阶只有 993 —— 没用满 10 bit。
   本探针把上界**显式设成 1023**。
2. **面板灰阶由 ``SLM_Ctrl_WriteGS(gray)`` 控制**: 手册 §3.2.7 写的是
   "全屏均匀灰度 0-1023 (0π-2π)", 即把整屏设成某个均匀相位。用户的意思是该值
   同时充当面板灰阶上限 (增益)。两种解释只能用实测判定: 对每个 gs, 先用
   WriteGS 设值, **再下发 0..1023 的闪耀光栅**, 看效率是否随 gs 变化 ——
   若 gs 只是"均匀相位", 光栅会原样显示, 效率对 gs 不敏感; 若 gs 是灰阶上限,
   效率会在真实 2π 对应的 gs 处取极值。

用法::

    python scripts/slm/probe_gs_ceiling_vs_efficiency.py --gs-min 900 --gs-max 1023 --gs-step 7
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.tools.slm.calibration.diffraction_efficiency.core import (
    BlazeSpec,
    blaze_gray,
    measure_orders,
    prepare_frame,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gs-min", type=int, default=900)
    ap.add_argument("--gs-max", type=int, default=1023)
    ap.add_argument("--gs-step", type=int, default=7)
    ap.add_argument("--period-px", type=int, default=32)
    ap.add_argument("--axis", default="x")
    ap.add_argument("--exposure-ms", type=float, default=1.0)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--discard", type=int, default=4)
    ap.add_argument("--settle-s", type=float, default=0.5)
    ap.add_argument("--pupil-center", default="960,600")
    ap.add_argument("--pupil-radius", type=int, default=600)
    ap.add_argument("--predicted-k", type=float, default=7510.0)
    ap.add_argument("--defocus-focal-mm", type=float, default=float("inf"))
    ap.add_argument("--calib", default="data/slm_geometry_calib.npz")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out_dir = Path(
        args.out
        or f"data/diffraction_efficiency/gs_ceiling_{datetime.now():%Y%m%d_%H%M%S}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.add(out_dir / "probe.log", level="INFO")

    from ao_shaping.drivers.ccd.common import create_camera, full_scale
    from ao_shaping.drivers.slm.santec import Santec, SlotRotator

    pupil_center = tuple(int(v) for v in args.pupil_center.replace("(", "").replace(")", "").split(","))
    gs_values = list(range(int(args.gs_min), int(args.gs_max) + 1, max(1, int(args.gs_step))))
    if gs_values[-1] != int(args.gs_max):
        gs_values.append(int(args.gs_max))

    rows: list[dict] = []
    with (
        Santec(slm_number=1, video_mode=0, wavelength=1064) as slm,
        create_camera("daheng", 0, exposure_time_ms=float(args.exposure_ms)) as cam,
    ):
        rot = SlotRotator(slm)
        panel = (int(slm.Panel_Res[1]), int(slm.Panel_Res[0]))
        # 🔴 关键: 图案上界显式 1023, 不用驱动标称的 2π 灰度 993
        max_gray = int(slm.MAX_GRAYSCALE_VALUE)
        logger.info(
            "面板 {} 图案最大灰阶 = {} (驱动标称 2π 灰度 = {})",
            panel,
            max_gray,
            getattr(slm, "_max_gray", None),
        )
        spec = BlazeSpec(
            period_px=int(args.period_px),
            axis=str(args.axis),
            depth_rad=2.0 * math.pi,  # 满闪耀: 0..1023 走完一个 2π 斜坡
            pupil_center=pupil_center,
            pupil_radius=int(args.pupil_radius),
        )
        pattern = blaze_gray(spec, panel, max_gray)
        logger.info(
            "光栅图案: 灰阶 {}..{} (周期 {} px, 瞳面半径 {})",
            int(pattern.min()),
            int(pattern.max()),
            args.period_px,
            args.pupil_radius,
        )
        predicted_sep = float(args.predicted_k) / float(args.period_px)

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

        center = None
        calib_path = Path(args.calib)
        if calib_path.is_file():
            with np.load(calib_path) as data:
                calib_center = np.asarray(data["center"], dtype=np.float64).ravel()
            center = (float(calib_center[0]), float(calib_center[1]))
            logger.info("零级中心来自几何标定 {}: {}", calib_path, center)
        else:
            logger.warning("没有几何标定 {}，退化为首帧最大点", calib_path)
        for gs in gs_values:
            # 1) 用 WriteGS 设"面板灰阶上限"
            slm.set_grayscale(int(gs))
            uniform = capture()
            uniform_peak = float(uniform.max())
            # 2) 再下发 0..1023 的闪耀光栅 (memory 模式), 看图案是否照常显示
            rot.display_data(slm, pattern, float(args.settle_s))
            frame = prepare_frame(capture())
            peak = float(full_scale(uniform))
            if center is None:
                # 用满闪耀这一帧的零级位置当锚点 (此时零级很弱, 故用首帧最大点)
                idx = int(np.argmax(frame))
                center = (
                    float(divmod(idx, frame.shape[1])[0]),
                    float(divmod(idx, frame.shape[1])[1]),
                )
                logger.info("锚点 (首帧最大点) = {}", center)
            try:
                m = measure_orders(
                    frame,
                    expected_center=center,
                    predicted_separation_px=predicted_sep,
                )
                row = {
                    "gs": int(gs),
                    "eta_zero": float(m.eta_zero or 0.0),
                    "eta_plus_total": float(m.eta_plus_total or 0.0),
                    "separation_px": float(m.separation_px or float("nan")),
                    "residual_px": float(m.residual_px or float("nan")),
                    "blaze_peak_adu": float(frame.max()),
                    "uniform_peak_adu": uniform_peak,
                    "notes": " | ".join(m.notes)[:120],
                }
            except Exception as exc:  # 单点失败不中断
                row = {
                    "gs": int(gs),
                    "eta_zero": float("nan"),
                    "eta_plus_total": float("nan"),
                    "separation_px": float("nan"),
                    "residual_px": float("nan"),
                    "blaze_peak_adu": float(frame.max()),
                    "uniform_peak_adu": uniform_peak,
                    "notes": f"measure_orders 失败: {exc}"[:120],
                }
            rows.append(row)
            logger.info(
                "gs={:4d}  η0={:6.4f}  η±1={:6.4f}  间距={:7.2f}px  光栅峰值={:5.0f}  均匀峰={:5.0f}",
                row["gs"],
                row["eta_zero"],
                row["eta_plus_total"],
                row["separation_px"],
                row["blaze_peak_adu"],
                row["uniform_peak_adu"],
            )

    # 落盘
    csv_path = out_dir / "gs_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    valid = [r for r in rows if np.isfinite(r["eta_plus_total"])]
    best = max(valid, key=lambda r: r["eta_plus_total"]) if valid else None
    summary = {
        "canvas_max_gray": int(max_gray),
        "driver_two_pi_gray": getattr(slm, "_max_gray", None),
        "period_px": int(args.period_px),
        "axis": str(args.axis),
        "pupil_radius": int(args.pupil_radius),
        "exposure_ms": float(args.exposure_ms),
        "predicted_separation_px": float(predicted_sep),
        "gs_values": gs_values,
        "best": best,
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    logger.info(f"产物: {csv_path} / {out_dir / 'summary.json'}")
    if best:
        logger.info(
            f"最优 gs = {best['gs']}  η±1 = {best['eta_plus_total']:.4f}  "
            f"η0 = {best['eta_zero']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
