"""SLM + CCD 硬件闭环光束整形最小测试脚本（仅在有设备时运行，无仿真分支）。

流程: 连接 SLM/CCD → 显示 flat 相位 → 采集初始帧 → 基于初始光强构建
方形目标 (曝光无关) → 微分优化器 (backprop+Adam) 得到整形相位 → 下发 →
重采帧 → compute_metrics 评估 → Recorder 记录。

运行:  $env:PYTHONPATH="src;libs"; python shaping_test.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from loguru import logger

from ao_shaping.drivers.ccd import DahengCamera
from ao_shaping.drivers.slm import SantecSLM200
from ao_shaping.optimizer import differentiable_beam_optimize
from ao_shaping.utils.beam_metrics import compute_metrics
from ao_shaping.utils.file import Recorder
from ao_shaping.utils.hardware_utils import call_with_timeout
from ao_shaping.utils.targets import square_target_from_measurement

# ---------------------------------------------------------------------------
# 可调参数 (调试时改这里即可)
# ---------------------------------------------------------------------------
EXPOSURE_MS = 1.2          # CCD 曝光 (ms)
SIDE_PX = 30.0             # 目标方形在 CCD 上的边长 (px) — 按光斑大小调整
EPOCHS = 200               # 微分优化器 Adam 迭代步数 (调试时先给 10-50)
LR = 0.01                  # Adam 学习率
SEED = 0                   # 初始相位随机种子 (可复现)
SETTLE_S = 0.5             # SLM 显示后等待 (s)
CAPTURE_TIMEOUT_S = 30.0   # SLM/CCD SDK 调用看门狗超时 (防原生挂起)
N_SAMPLE = 3               # 相机每次采样的平均帧数
OUT_DIR = Path("data/shaping_test")

# --- pygame 可视化 (参考 ao_shaping.display 包) ---
SHOW_DISPLAY = True        # 结束后打开 4 面板窗口对比 初始/目标/相位/结果
DISPLAY_SIZE = (1280, 640)  # 窗口总尺寸
FRAME_SIZE = (300, 300)    # 单面板尺寸


def _display_phase(slm: SantecSLM200, phase_rad: np.ndarray, settle_s: float) -> None:
    """弧度相位 → 灰度下发 (create_phase_from_array 含波前矫正/LUT)。

    display_data 内部自动轮换内存槽, 满足 AGENTS.md 槽轮换约束。
    """
    gray = slm.create_phase_from_array(phase_rad)
    slm.display_data(gray, wait_time_s=settle_s)


def _norm_gray(arr: np.ndarray) -> np.ndarray:
    """min-max 归一化到 0~255 uint8 (Image2DFrame 不归一化, make_surface 需要 uint8)."""
    a = np.asarray(arr, dtype=np.float64)
    lo, hi = np.percentile(a, 1.0), np.percentile(a, 99.0)
    if hi - lo < 1e-12:
        return np.zeros(a.shape, dtype=np.uint8)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.uint8) * 255


def _show_result(init_img, target_ccd, phase, result_img, metrics) -> None:
    """4 面板 pygame 对比窗口: 初始帧 / 目标方形 / 最终相位 / 整形结果.

    使用 ao_shaping.display.AutoDisplay 自动排版; 关闭窗口或按 ESC 退出。
    """
    import pygame

    from ao_shaping.display import AutoDisplay, FrameInfo

    frame_list = [
        FrameInfo("init", "Initial Frame", "Image2DFrame"),
        FrameInfo("target", "Target Square", "Image2DFrame"),
        FrameInfo("phase", "Final Phase", "Image2DFrame"),
        FrameInfo("result", "Shaped Result", "Image2DFrame"),
    ]
    frame_data = {
        "init": {"img": _norm_gray(init_img)},
        "target": {"img": _norm_gray(target_ccd)},
        "phase": {"img": _norm_gray(phase)},
        "result": {"img": _norm_gray(result_img)},
    }
    info = (f"mse={metrics['mse']:.4f} corr={metrics['correlation']:.4f} "
            f"eff={metrics['efficiency']:.4f}  — 关闭窗口或 ESC 退出")
    with AutoDisplay(frame_list, frame_size=FRAME_SIZE,
                     display_size=DISPLAY_SIZE, margin=10) as win:
        win.render(frame_data, info=info)
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    return
            pygame.time.wait(30)


def main() -> None:
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    slm = SantecSLM200()
    ccd = DahengCamera(exposure_time_ms=EXPOSURE_MS)
    recorder = Recorder("mse", "min")
    grid_h, grid_w = slm.Panel_Res  # (1200, 1920) SLM200 全面板

    try:
        call_with_timeout(slm.open, CAPTURE_TIMEOUT_S, "SLM open")
        call_with_timeout(ccd.open, CAPTURE_TIMEOUT_S, "CCD open")
        logger.info("设备已连接: SLM {}x{} / CCD 曝光 {}ms",
                    grid_h, grid_w, EXPOSURE_MS)

        # 1) flat 相位 → 采集初始帧 (0-order 光斑 = 帧全局最大, 见 AGENTS.md)
        _display_phase(slm, np.zeros((grid_h, grid_w), dtype=np.float32), SETTLE_S)
        init_img = call_with_timeout(
            lambda: ccd.get_numpy_image(n_sample=N_SAMPLE), CAPTURE_TIMEOUT_S, "CCD capture"
        )
        logger.info("初始帧 shape={} peak={:.0f}",
                    init_img.shape, float(np.max(init_img)))

        # 2) 基于初始光强构建方形目标 (曝光无关: target_ccd.sum()==1)
        target_grid, info = square_target_from_measurement(
            init_img, SIDE_PX, grid_h, grid_w,
        )
        y0, x0 = info["centroid"]
        logger.info("target: side_cam_px={} 质心=({}, {}) target_ccd.sum()={:.4f}",
                    info["side_cam_px"], y0, x0, float(info["target_ccd"].sum()))

        # 3) 微分优化器: 优化 SLM 相位匹配目标远场强度
        source_amp = np.ones((grid_h, grid_w), dtype=np.float32)
        res = differentiable_beam_optimize(
            target_intensity=target_grid,
            source_amplitude=source_amp,
            lr=LR,
            epochs=EPOCHS,
            seed=SEED,
            log_every=EPOCHS // 4,
        )
        logger.info("优化完成: final_loss={:.4f} converged={}",
                    res.final_loss, res.converged)

        # 4) 下发整形相位 → 重采帧
        _display_phase(slm, np.asarray(res.phase, dtype=np.float32), SETTLE_S)
        current_img = call_with_timeout(
            lambda: ccd.get_numpy_image(n_sample=N_SAMPLE), CAPTURE_TIMEOUT_S, "CCD capture"
        )

        # 5) 曝光无关评估: 实测帧/总亮度 vs target_ccd (sum==1)
        metrics = compute_metrics(current_img, info["target_ccd"])
        logger.info("metrics: mse={:.5f} correlation={:.4f} efficiency={:.4f}",
                    metrics["mse"], metrics["correlation"], metrics["efficiency"])

        # 6) 记录: Recorder(mark="mse") — record 含 metrics 即满足约束
        recorder.append({"ccd": current_img, "phase": res.phase, **metrics})
        recorder.save_dataframe(out_dir / "history.csv", sidecar_dir=out_dir)
        logger.info("结果已保存: {}", out_dir)

        # 7) pygame 4 面板对比: 初始帧 / 目标方形 / 最终相位 / 整形结果
        if SHOW_DISPLAY:
            _show_result(init_img, info["target_ccd"], res.phase, current_img, metrics)

    finally:
        call_with_timeout(slm.close, 10.0, "SLM close")
        call_with_timeout(ccd.close, 10.0, "CCD close")


if __name__ == "__main__":
    main()