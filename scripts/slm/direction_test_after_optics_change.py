"""换光路后的 B± 方向检验：模型下降方向在实验系统上是否仍指对。

流程：
1. 用**新检查点**在给定初值处算 CV 梯度（固定 60 px 掩膜），归一到照明加权 RMS
   ``--target-rms``（默认 0.03 rad），导出 B± 两张面板相位；
2. ``--execute`` 时按 A / B+ / A / B− / A 上屏，各 ``--frames`` 帧，算掩膜 CV 并
   与 3σ 噪声地板比较。

为什么必须先做这一步：2026-10-23 实测，重拟合能把 fit loss 从 0.93 压到 0.36，却
让梯度**完全不可迁移**（4/4 轮实测 Δ≈0）。fit loss 不能判模型可用，方向检验才能。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed-panel", type=Path, required=True, help="面板分辨率初值 .npy")
    parser.add_argument("--spot-xy", type=float, nargs=2, required=True, help="实测零点 (x, y)")
    parser.add_argument("--window-half-span-px", type=float, default=200.0)
    parser.add_argument("--step-px", type=float, default=2.0)
    parser.add_argument("--target-side-px", type=float, default=60.0)
    parser.add_argument("--target-rms", type=float, default=0.03)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--exposure-ms", type=float, default=None)
    parser.add_argument("--noise-floor", type=Path, default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import torch

    torch.set_num_threads(4)

    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusForward, UniformityObjective, load_frozen_parameters, recentre_window,
    )
    from ao_shaping.utils.image.target.masks import build_box_core_masks

    params = load_frozen_parameters(args.checkpoint)
    xy = recentre_window((args.spot_xy[0], args.spot_xy[1]), half_span_px=args.window_half_span_px,
                         step_px=args.step_px)
    forward = FrozenDefocusForward(params, xy, device="cpu")

    seed_panel = np.load(args.seed_panel).astype(np.float64)
    pitch = 1920 // params.grid_size
    rows = seed_panel.shape[0] // pitch
    seed = np.zeros((params.grid_size, params.grid_size))
    seed[:rows] = seed_panel[: rows * pitch].reshape(rows, pitch, params.grid_size, pitch).mean(axis=(1, 3))

    centre = forward.sample_grid_index(args.spot_xy[0], args.spot_xy[1])
    box, core = build_box_core_masks(xy.shape[:2], centre, args.target_side_px, 3.0, forward.sample_step_ccd())
    objective = UniformityObjective(box, core, w_phase=0.01)
    logger.info(f"[window] centre={centre} box={int(box.sum())} core={int(core.sum())} "
          f"d={params.defocus_diopter:+.4f}")

    phase = torch.tensor(seed[None], dtype=torch.float32)
    phase.requires_grad_(True)
    image = forward.predict(phase)
    image = image[0] if image.ndim == 3 else image
    loss, terms = objective.evaluate(image, phase[0])
    loss.backward()
    grad = phase.grad[0].numpy().astype(np.float64)

    weights = forward.illumination_weights()
    rms = float(np.sqrt((weights * grad ** 2).sum() / weights.sum()))
    direction = -grad * (args.target_rms / max(rms, 1e-12))
    applied = float(np.sqrt((weights * direction ** 2).sum() / weights.sum()))

    def to_panel(grid: np.ndarray) -> np.ndarray:
        panel = np.zeros((1200, 1920), dtype=np.float64)
        rows_local = min(1200 // pitch, grid.shape[0])
        panel[: rows_local * pitch] = np.repeat(np.repeat(grid[:rows_local], pitch, axis=0), pitch, axis=1)
        return panel

    np.save(args.out / "seed_panel.npy", to_panel(seed))
    np.save(args.out / "b_plus_panel.npy", to_panel(seed + direction))
    np.save(args.out / "b_minus_panel.npy", to_panel(seed - direction))

    def model_cv(grid: np.ndarray) -> float:
        with torch.no_grad():
            image = forward.predict(torch.tensor(grid[None], dtype=torch.float32))[0]
            values = image[torch.as_tensor(core)]
            mean = values.mean().clamp_min(1e-8)
            return float(np.sqrt((((values / mean) - 1) ** 2).mean()))

    record = dict(checkpoint=str(args.checkpoint), spot_xy=list(args.spot_xy), centre=list(centre),
                  target_side_px=args.target_side_px, target_rms=args.target_rms, applied_rms=applied,
                  loss_terms=terms, defocus_diopter=params.defocus_diopter,
                  model_cv=dict(seed=model_cv(seed), b_plus=model_cv(seed + direction),
                                b_minus=model_cv(seed - direction)))
    logger.info(f"[model] CV: seed {record['model_cv']['seed']:.4f} "
          f"B+ {record['model_cv']['b_plus']:.4f} B− {record['model_cv']['b_minus']:.4f}")

    if args.execute:
        from ao_shaping.drivers.ccd.common import create_camera
        from ao_shaping.drivers.slm.santec.driver import Santec
        from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale

        phases = {"A": np.load(args.out / "seed_panel.npy"), "B+": np.load(args.out / "b_plus_panel.npy"),
                  "B-": np.load(args.out / "b_minus_panel.npy")}
        order = ["A", "B+", "A", "B-", "A"]
        exposure = float(args.exposure_ms) if args.exposure_ms else 4.0
        slm = Santec(slm_number=1, video_mode=0)
        slm.open()
        slm.set_phase_overlay(None)
        records = []
        try:
            gray = {k: phase_to_slm_grayscale(v, slm=slm) for k, v in phases.items()}
            with create_camera("daheng", 0, exposure_time_ms=exposure) as cam:
                slm.display_data(gray["A"])
                time.sleep(0.4)
                cam.get_numpy_image(n_sample=1)
                first = np.asarray(cam.get_numpy_image(n_sample=1), dtype=float)
                if first.ndim == 1:
                    first = first.reshape(1944, 2592)
                row, col = np.unravel_index(int(np.argmax(first)), first.shape)
                # 掩膜一致性：模型按 --spot-xy 建掩膜、实测按首帧 argmax 定中心。
                # 两者偏离时 CV 不可比 —— 告警并记录，绝不静默重定中心。
                offset_px = float(np.hypot(col - args.spot_xy[0], row - args.spot_xy[1]))
                if offset_px > 5.0:
                    logger.warning(
                        "实测零点 ({}, {}) 与 --spot-xy ({:.1f}, {:.1f}) 相差 {:.1f} px；"
                        "模型与实测掩膜不在同一位置，本次结论不可比",
                        col, row, args.spot_xy[0], args.spot_xy[1], offset_px)
                record["mask_offset_px"] = offset_px
                half = int(round(args.target_side_px / 2))
                yy, xx = np.mgrid[: first.shape[0], : first.shape[1]]
                core_mask = (np.abs(yy - row) <= half - 3) & (np.abs(xx - col) <= half - 3)
                for label in order:
                    slm.display_data(gray[label])
                    time.sleep(0.4)
                    for _ in range(2):
                        cam.get_numpy_image(n_sample=1)
                    for _ in range(args.frames):
                        frame = np.asarray(cam.get_numpy_image(n_sample=1), dtype=float)
                        if frame.ndim == 1:
                            frame = frame.reshape(1944, 2592)
                        values = frame[core_mask]
                        mean = float(values.mean())
                        cv = float(np.sqrt((((values / mean) - 1) ** 2).mean())) if mean > 0 else float("nan")
                        records.append(dict(label=label, cv=cv, mean=mean, peak=float(frame.max())))
                    logger.info(f"  {label}: cv={records[-1]['cv']:.5f} mean={records[-1]['mean']:.2f}")
        finally:
            try:
                slm.set_phase_overlay(None)
            except Exception:  # 保留宽泛捕获: 设备 API 是多态实现 (真机/仿真/mock), 抛出的类型不可枚举
                pass
            slm.close()
        seed_cv = float(np.mean([r["cv"] for r in records if r["label"] == "A"]))
        plus = float(np.mean([r["cv"] for r in records if r["label"] == "B+"]))
        minus = float(np.mean([r["cv"] for r in records if r["label"] == "B-"]))
        record["measured"] = dict(seed=seed_cv, b_plus=plus, b_minus=minus,
                                  delta_plus=plus - seed_cv, delta_minus=minus - seed_cv,
                                  frames=[r for r in records])
        record["mask"] = dict(centre_yx=[int(row), int(col)], side_px=args.target_side_px)
        if args.noise_floor and args.noise_floor.exists():
            resolution = float(json.loads(args.noise_floor.read_text(encoding="utf-8"))["three_sigma_resolution"])
            record["three_sigma_resolution"] = resolution
            dp = record["measured"]["delta_plus"]
            dm = record["measured"]["delta_minus"]
            # 判决必须查**方向**：Δ+ 应下降、Δ− 应上升，且都超过 3σ。
            # 只查"异号"会放过方向反了的记录（例如 Δ+ +0.08 / Δ− −0.08）。
            record["verdict"] = bool(dp < -resolution and dm > resolution)
        logger.info(json.dumps({k: v for k, v in record["measured"].items() if k != "frames"},
                         indent=2, ensure_ascii=False))

    (args.out / "direction_record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False),
                                                    encoding="utf-8")
    logger.info(f"[done] {args.out}")


if __name__ == "__main__":
    main()


