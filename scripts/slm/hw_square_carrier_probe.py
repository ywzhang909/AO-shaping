"""正方形目标 + 倾斜载波（把 0 级让开）+ **不开窗** 的可用上限探针。

为什么这样设计（2026-10-23 的实测教训）：
* **当前光路的方向检验未通过**（位置对齐后 Δ 反号且低于 3σ），所以这一轮定位是
  **探索性上限测量**，不是"可微优化有效"的证明；报告必须按此口径写。
* **载波**：给总相位叠加一维线性斜坡（闪耀光栅），把目标搬到离 0 级 Δx 处。
  斜坡与位移预测一律走 canonical
  :mod:`ao_shaping.utils.wavefront.zero_order_overlay`（issue #199 R5：脚本
  曾经自带一份 1D ramp 和朴素的 ``λf/(p·period)`` 公式，而实测台架常数是
  ``TILT_SHIFT_SCALE``，且面板↔相机轴相差 90°）。
* **不开窗**：整帧 1944×2592 采（不调 `reset_window`），0 级与目标同帧可见、可分别度量。
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "libs"))

from ao_shaping.utils.wavefront.zero_order_overlay import (
    OverlayParams,
    apply_zero_order_overlay,
    predicted_shift_px,
)


def square_target(size_grid: int, side_px: float, centre: tuple[int, int]) -> np.ndarray:
    """在模型网格上做归一化方形目标强度（0/1）。"""
    yy, xx = np.mgrid[:size_grid, :size_grid]
    half = side_px / 2.0
    mask = (np.abs(xx - centre[1]) <= half) & (np.abs(yy - centre[0]) <= half)
    return mask.astype(np.float32)


def carrier_overlay_params(period_px: float, axis: str) -> OverlayParams:
    """载波 overlay 参数（canonical 实现，issue #199 R5）。

    ``axis="x"`` 是沿面板 x 的斜坡，即 overlay 约定的 ``"vertical"`` 竖条纹；
    ``axis="y"`` 对应 ``"horizontal"``。位移大小由实测台架常数
    ``TILT_SHIFT_SCALE`` 决定（``predicted_shift_px``），面板 x 向斜坡按 90°
    轴交换搬到相机 y 向。
    """
    return OverlayParams(
        mode="blaze",
        blaze_period_px=float(period_px),
        blaze_direction="vertical" if axis == "x" else "horizontal",
    )


def metrics(frame: np.ndarray, centre_xy: tuple[float, float], side_px: float,
            zero_xy: tuple[float, float]) -> dict:
    """整帧上算：目标框 CV/EE/均值/峰值 + 与 0 级的距离与峰值比。"""
    height, width = frame.shape
    col, row = centre_xy
    half = side_px / 2.0
    yy, xx = np.mgrid[:height, :width]
    box = (np.abs(xx - col) <= half) & (np.abs(yy - row) <= half)
    inner = (np.abs(xx - col) <= max(1.0, half - 3.0)) & (np.abs(yy - row) <= max(1.0, half - 3.0))
    values = frame[inner]
    mean = float(values.mean()) if values.size else float("nan")
    cv = float(np.sqrt((((values / mean) - 1) ** 2).mean())) if mean > 0 else float("nan")
    zero_col, zero_row = zero_xy
    zero_box = (np.abs(xx - zero_col) <= 8) & (np.abs(yy - zero_row) <= 8)
    return dict(
        target_mean=mean, target_peak=float(frame[box].max()) if box.any() else float("nan"),
        cv=float(cv), ee=float(frame[box].sum() / max(frame.sum(), 1.0)),
        zero_peak=float(frame[zero_box].max()) if zero_box.any() else float("nan"),
        separation_px=float(np.hypot(col - zero_col, row - zero_row)),
        zero_in_target=float(frame[box & zero_box].sum() / max(frame[box].sum(), 1.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--spot-xy", type=float, nargs=2, required=True, help="当前实测零点 (x, y)")
    parser.add_argument("--target-side-px", type=float, default=84.85)
    parser.add_argument("--carrier-period-px", type=float, default=32.0)
    parser.add_argument("--carrier-axis", choices=("x", "y"), default="x")
    parser.add_argument("--steps", type=int, default=0, help="可微步数；0 = 纯 GS 基线")
    parser.add_argument("--cap", type=float, default=0.06)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--exposure-ms", type=float, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import torch

    torch.set_num_threads(4)

    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusForward, UniformityObjective, build_masks, load_frozen_parameters, recentre_window,
    )

    params = load_frozen_parameters(args.checkpoint)
    grid = params.grid_size
    # 目标放在模型网格中心（GS 在中心求），载波在面板域叠加 ⇒ 远场整体平移
    target = square_target(grid, args.target_side_px * grid / 1920.0, (grid // 2, grid // 2))

    from ao_shaping.algorithm.signal_processing.gerchberg_saxton import gerchberg_saxton

    seed = gerchberg_saxton(target, iterations=100) if callable(gerchberg_saxton) else None
    if seed is None:
        raise SystemExit("gerchberg_saxton 不可用")
    seed_grid = np.asarray(seed, dtype=np.float64)

    forward = FrozenDefocusForward(params, recentre_window((args.spot_xy[0], args.spot_xy[1]),
                                                          half_span_px=400.0, step_px=2.0), device="cpu")
    centre = forward.sample_grid_index(args.spot_xy[0], args.spot_xy[1])
    box, core = build_masks(forward.camera_xy.shape[:2], centre, args.target_side_px, 3.0,
                            forward.sample_step_ccd())
    objective = UniformityObjective(box, core, w_phase=0.01)

    record = dict(recorded_utc=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                  checkpoint=str(args.checkpoint), spot_xy=list(args.spot_xy),
                  target_side_px=args.target_side_px, carrier_period_px=args.carrier_period_px,
                  carrier_axis=args.carrier_axis, steps=args.steps, mode="exploratory")
    if args.steps > 0:
        from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
            DifferentiableUniformityOptimizer,
        )

        optimizer = DifferentiableUniformityOptimizer(forward, objective, seed_grid, lr=0.01,
                                                     step_rms_cap=args.cap, device="cpu")
        for _ in range(args.steps):
            optimizer.update()
        phase_grid = seed_grid + optimizer.delta_phase
        record["model_cv"] = [r["l_cv"] for r in optimizer.loss_history]
    else:
        phase_grid = seed_grid

    pitch = 1920 // grid
    rows = 1200 // pitch
    panel = np.zeros((1200, 1920), dtype=np.float64)
    panel[: rows * pitch] = np.repeat(np.repeat(phase_grid[:rows], pitch, axis=0), pitch, axis=1)
    # 载波走 canonical overlay（raw 弧度叠加，唯一的 mod-2π 在驱动灰度换算）
    overlay = carrier_overlay_params(args.carrier_period_px, args.carrier_axis)
    panel = apply_zero_order_overlay(panel, overlay)
    np.save(args.out / "phase_panel_rad.npy", panel)

    # 载波带来的远场平移（实测台架常数 TILT_SHIFT_SCALE，含面板↔相机 90° 轴交换）
    shift_dx, shift_dy = predicted_shift_px(overlay)
    record["carrier_shift_px_theory"] = [float(shift_dx), float(shift_dy)]
    print(f"[carrier] period {args.carrier_period_px} px ⇒ 远场平移理论 "
          f"({shift_dx:.0f}, {shift_dy:.0f}) CCD px", flush=True)

    if not args.execute:
        (args.out / "probe.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[dry] 计划已写入 {args.out}")
        return

    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm.santec.driver import Santec
    from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale

    exposure = float(args.exposure_ms) if args.exposure_ms else 4.0
    slm = Santec(slm_number=1, video_mode=0)
    slm.open()
    slm.set_phase_overlay(None)
    shots = {}
    try:
        with create_camera("daheng", 0, exposure_time_ms=exposure) as cam:   # 不开窗：整帧
            for label, array in (("flat", np.zeros((1200, 1920), dtype=np.uint16)),
                                 ("shaped", phase_to_slm_grayscale(panel, slm=slm))):
                slm.display_data(array)
                time.sleep(0.4)
                for _ in range(2):
                    cam.get_numpy_image(n_sample=1)
                stack = []
                for _ in range(args.frames):
                    frame = np.asarray(cam.get_numpy_image(n_sample=1), dtype=float)
                    if frame.ndim == 1:
                        frame = frame.reshape(1944, 2592)
                    stack.append(frame.mean(0) if frame.ndim == 3 else frame)
                shots[label] = np.mean(stack, axis=0)
                np.save(args.out / f"ccd_{label}.npy", shots[label].astype(np.float32))
                row, col = np.unravel_index(int(np.argmax(shots[label])), shots[label].shape)
                print(f"  {label}: peak {shots[label].max():.0f} argmax (row {row}, col {col})", flush=True)
    finally:
        try:
            slm.set_phase_overlay(None)
        except Exception:
            pass
        slm.close()

    zero_row, zero_col = np.unravel_index(int(np.argmax(shots["flat"])), shots["flat"].shape)
    # 目标应在 0 级 + 载波位移处；预测位移来自实测台架常数（含 90° 轴交换），
    # 两个方向都试，取更亮的那个作为"目标已成形"的证据
    candidates = []
    for sign in (+1, -1):
        shifted = (zero_col + sign * shift_dx, zero_row + sign * shift_dy)
        candidates.append((sign, shifted, metrics(shots["shaped"], shifted, args.target_side_px,
                                                  (zero_col, zero_row))))
    best = max(candidates, key=lambda item: item[2]["target_peak"])
    record["zero_order_xy"] = [int(zero_col), int(zero_row)]
    record["target_xy"] = [float(v) for v in best[1]]
    record["carrier_sign"] = int(best[0])
    record["metrics_shaped"] = best[2]
    record["metrics_flat"] = metrics(shots["flat"], best[1], args.target_side_px, (zero_col, zero_row))
    record["exposure_ms"] = float(exposure)
    (args.out / "probe.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k.startswith(("metrics", "zero_", "target_", "carrier_"))},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
