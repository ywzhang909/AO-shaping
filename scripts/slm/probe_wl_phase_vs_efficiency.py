"""扫 ``SLM_Ctrl_WriteWL`` 的 ``phase`` 参数 (= 满量程灰度 1023 处的相位), 定 2π 灰度。

手册依据
--------
``docs/slm-200/dev.md`` §3.2.4::

    SLM_Ctrl_WriteWL(SLMNumber, wavelength, phase)
    phase: Specify phase value multiplied by 100 (0-999).  e.g. 2.00π => 200
    It cannot be set to a value that causes internal calculation result of SLM
    to be abnormal.  e.g. Set phase 2.00π => calculation result 2.01π

所以 ``phase`` 是**面板满灰度 (1023) 处能给出的相位**:
  * ``phase = 200`` ⇒ 1023 处正好 2π ⇒ 2π 灰度 = 1023
  * ``phase = 206`` ⇒ 1023 处 2.06π ⇒ 2π 灰度 = 1023·2/2.06 = 993 (本机标称)
驱动在 :meth:`Santec.get_wavelength_info` 里正是这么算 ``_max_gray`` 的。

做法
----
对每个 ``phase``: 先 ``WriteWL(1, 1064, phase)`` (不写 EEPROM, 断电即恢复),
读回设备实际接受的值与 ``_max_gray``, 再下发**上界固定 1023** 的闪耀光栅, 测效率。
η±1 取极大值处的 ``phase`` 就是正确的设置值 —— 因为那时斜坡刚好走满 2π。

用法::

    python scripts/slm/probe_wl_phase_vs_efficiency.py --phases 180,190,196,200,206,212,220
"""

from __future__ import annotations

import argparse
import csv
import ctypes
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
    ap.add_argument("--phases", default="180,190,196,200,206,212,220")
    ap.add_argument("--slm-wavelength", type=int, default=1064)
    ap.add_argument("--period-px", type=int, default=32)
    ap.add_argument("--axis", default="x")
    ap.add_argument("--exposure-ms", type=float, default=1.0)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--discard", type=int, default=4)
    ap.add_argument("--settle-s", type=float, default=0.5)
    ap.add_argument("--pupil-center", default="960,600")
    ap.add_argument("--pupil-radius", type=int, default=600)
    ap.add_argument("--predicted-k", type=float, default=7510.0)
    ap.add_argument("--calib", default="data/slm_geometry_calib.npz")
    ap.add_argument("--wait-after-write-s", type=float, default=45.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    phases = [int(v) for v in str(args.phases).split(",") if v.strip()]
    out_dir = Path(
        args.out or f"data/diffraction_efficiency/wl_phase_{datetime.now():%Y%m%d_%H%M%S}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.add(out_dir / "probe.log", level="INFO")

    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm.santec import SLM_OK, Santec, SlotRotator

    pupil_center = tuple(
        int(v) for v in args.pupil_center.replace("(", "").replace(")", "").split(",")
    )
    calib_path = Path(args.calib)
    if not calib_path.is_file():
        logger.error("缺几何标定 {} (零级中心无从得知)", calib_path)
        return 2
    with np.load(calib_path) as data:
        calib_center = np.asarray(data["center"], dtype=np.float64).ravel()
    center = (float(calib_center[0]), float(calib_center[1]))
    predicted_sep = float(args.predicted_k) / float(args.period_px)

    rows: list[dict] = []
    with (
        Santec(slm_number=1, video_mode=0, wavelength=int(args.slm_wavelength)) as slm,
        create_camera("daheng", 0, exposure_time_ms=float(args.exposure_ms)) as cam,
    ):
        rot = SlotRotator(slm)
        panel = (int(slm.Panel_Res[1]), int(slm.Panel_Res[0]))
        max_gray = int(slm.MAX_GRAYSCALE_VALUE)  # 图案上界固定 1023
        spec = BlazeSpec(
            period_px=int(args.period_px),
            axis=str(args.axis),
            depth_rad=2.0 * math.pi,
            pupil_center=pupil_center,
            pupil_radius=int(args.pupil_radius),
        )
        pattern = blaze_gray(spec, panel, max_gray)
        logger.info(
            "图案 {}..{} (上界固定 1023), 周期 {} px, 零级中心 {}",
            int(pattern.min()),
            int(pattern.max()),
            args.period_px,
            center,
        )

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

        for phase in phases:
            # 1) 写波长+满量程相位 (不写 EEPROM)
            res = slm._slm.SLM_Ctrl_WriteWL(
                slm.slm_number, int(args.slm_wavelength), int(phase)
            )
            if res != SLM_OK:
                logger.error("phase={} 写入失败 (code={})", phase, res)
                continue
            logger.info("phase={} 已写入, 等 {}s 展开相位表 ...", phase, args.wait_after_write_s)
            import time

            time.sleep(float(args.wait_after_write_s))
            # 2) 读回设备实际接受的值
            wl_out = ctypes.c_uint32(0)
            ph_out = ctypes.c_uint32(0)
            slm._slm.SLM_Ctrl_ReadWL(
                slm.slm_number, ctypes.byref(wl_out), ctypes.byref(ph_out)
            )
            accepted = int(ph_out.value)
            _, two_pi_gray = slm.get_wavelength_info()
            # 3) 下发固定上界 1023 的光栅并测量
            rot.display_data(slm, pattern, float(args.settle_s))
            frame = prepare_frame(capture())
            try:
                m = measure_orders(
                    frame,
                    expected_center=center,
                    predicted_separation_px=predicted_sep,
                )
                eta0 = float(m.eta_zero or 0.0)
                eta1 = float(m.eta_plus_total or 0.0)
                sep = float(m.separation_px or float("nan"))
                notes = " | ".join(m.notes)[:100]
            except Exception as exc:  # 保留宽泛捕获: 内部调用链 measure_orders 的失败类型不可枚举
                eta0 = eta1 = sep = float("nan")
                notes = f"measure_orders 失败: {exc}"[:100]
            row = {
                "phase_requested": int(phase),
                "phase_accepted": accepted,
                "phase_accepted_pi": accepted / 100.0,
                "two_pi_gray": int(two_pi_gray),
                "eta_zero": eta0,
                "eta_plus_total": eta1,
                "separation_px": sep,
                "blaze_peak_adu": float(frame.max()),
                "notes": notes,
            }
            rows.append(row)
            logger.info(
                "phase={:3d} -> 设备接受 {:3d} (={:.2f}pi, 2π灰度={:4d})  η0={:6.4f}  η±1={:6.4f}  峰值={:5.0f}",
                phase,
                accepted,
                accepted / 100.0,
                two_pi_gray,
                eta0,
                eta1,
                row["blaze_peak_adu"],
            )

    # 恢复设备标称值 (206 = 2.06π), 不写 EEPROM
    if rows:
        try:
            from ao_shaping.drivers.slm.santec import Santec as _S

            with _S(slm_number=1, video_mode=0, wavelength=int(args.slm_wavelength)) as s2:
                s2._slm.SLM_Ctrl_WriteWL(s2.slm_number, int(args.slm_wavelength), 206)
                logger.warning("已把 phase 恢复为 206 (设备标称 2.06π); 未写 EEPROM")
        except Exception as exc:  # 恢复失败只告警
            logger.warning("恢复 phase=206 失败: {}", exc)

    csv_path = out_dir / "phase_sweep.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    valid = [r for r in rows if np.isfinite(r["eta_plus_total"])]
    best = max(valid, key=lambda r: r["eta_plus_total"]) if valid else None
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "canvas_max_gray": int(max_gray),
                "period_px": int(args.period_px),
                "exposure_ms": float(args.exposure_ms),
                "best": best,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    logger.info(f"产物: {csv_path}")
    if best:
        logger.info(
            f"最优 phase = {best['phase_accepted']} (={best['phase_accepted_pi']:.2f}pi, "
            f"2π灰度={best['two_pi_gray']})  η±1 = {best['eta_plus_total']:.4f}  η0 = {best['eta_zero']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
