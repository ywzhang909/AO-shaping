"""可微分闭环光束整形优化器

利用 PyTorch 可微分优化 (梯度下降) + CCD 反馈，迭代优化 SLM 相位图案，
将远场光斑整形为方形/圆形/高斯/聚焦光斑。

闭环流程 (每个外迭代):
    1. CCD 采集当前光束图像
    2. 测量光斑直径
    3. 构建目标强度掩模 (circle/square/gaussian/spot)
    4. 运行可微分梯度优化 (内迭代) 生成相位
    5. 下发相位到 SLM (内存槽轮换)
    6. 等待 SLM 稳定
    7. 重新采集光束图像
    8. 计算质量指标 (长宽比、均匀性、环围能量)
    9. 记录进度并检查收敛

用法:
    python -m ao_shaping.runners.diff_shaping_runner --target-shape square --outer-iterations 10
    python -m ao_shaping.runners.diff_shaping_runner --target-shape gaussian --dl-iterations 500 --optimizer lbfgs
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import numpy as np

from loguru import logger

from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200
from ao_shaping.utils.file import Recorder
from ao_shaping.utils.resample import resample_to_grid

if TYPE_CHECKING:
    import pygame


# ==================== 全局运行状态 ====================

_running = True


def _signal_handler(signum, frame) -> None:
    """优雅退出信号处理 (SIGINT/SIGTERM)."""
    global _running
    _running = False
    logger.info("收到信号 {}，正在优雅退出...", signum)


# ==================== 相机工厂 ====================

def _get_daheng_camera(cam_id: int, exposure_ms: float):
    """Import and create Daheng camera instance."""
    try:
        from ao_shaping.drivers.ccd.daheng import DahengCamManager

        cam = DahengCamManager(cam_id=cam_id, exposure_time_ms=exposure_ms)
        cam.open()
        return cam
    except ImportError as e:
        logger.warning("Daheng相机不可用: {}", e)
        raise
    except Exception as e:
        logger.error("Daheng相机初始化失败: {}", e)
        raise


def _get_miicam_camera(cam_id: int, exposure_ms: float, bit_depth: int = 8):
    """Import and create MiiCam camera instance."""
    try:
        from ao_shaping.drivers.ccd.miicam.driver import CameraStreamManager

        cam = CameraStreamManager(
            cam_id=cam_id,
            exposure_time_ms=exposure_ms,
            bit_depth=bit_depth,
        )
        cam.open()
        return cam
    except ImportError as e:
        logger.warning("MiiCam相机不可用: {}", e)
        raise
    except Exception as e:
        logger.error("MiiCam相机初始化失败: {}", e)
        raise


# ==================== pygame 可视化 ====================

class _DiffDisplay:
    """可微分迭代过程 pygame 可视化窗口.

    显示四个面板:
        - 左上: 当前相位图案 (0~1 归一化)
        - 右上: 目标强度掩模
        - 左下: 远场光斑图像 (CCD 采集)
        - 右下: 损失收敛曲线

    用法:
        with _DiffDisplay() as disp:
            disp.update_phase(phase)
            disp.update_target(target_mask)
            disp.update_image(captured_image)
            disp.update_loss(iteration, loss)
            disp.update_status(score, side, spot_d)
    """

    PANEL_W = 480
    PANEL_H = 480
    PAD = 8
    TITLE_H = 28
    BG = (25, 25, 25)

    def __init__(self) -> None:
        w = self.PANEL_W * 2 + self.PAD * 3
        h = self.PANEL_H * 2 + self.PAD * 3 + self.TITLE_H

        self._phase: np.ndarray | None = None
        self._target: np.ndarray | None = None
        self._image: np.ndarray | None = None
        self._loss_curve: list[float] = []
        self._status_lines: list[str] = []
        self._dl_iter = 0
        self._window_size = (w, h)

    def __enter__(self) -> _DiffDisplay:
        import pygame

        pygame.init()
        pygame.display.set_caption("Diff Shaping 闭环优化")
        self._screen = pygame.display.set_mode(self._window_size)
        self._font = pygame.font.SysFont("consolas", 16)
        self._clock = pygame.time.Clock()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        import pygame

        pygame.quit()

    def _check_quit(self) -> bool:
        """检查用户是否点击关闭窗口. 返回 True 表示应停止."""
        import pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        return False

    def _draw_panel(
        self, index: int, title: str, surface: "pygame.Surface"
    ) -> None:
        import pygame

        col = index % 2
        row = index // 2
        x = self.PAD + col * (self.PANEL_W + self.PAD)
        y = self.TITLE_H + self.PAD + row * (self.PANEL_H + self.PAD)

        title_surf = self._font.render(title, True, (255, 255, 0))
        self._screen.blit(title_surf, (x + 4, y - self.TITLE_H + 2))

        scaled = pygame.transform.scale(
            surface, (self.PANEL_W, self.PANEL_H)
        )
        self._screen.blit(scaled, (x, y))
        pygame.draw.rect(self._screen, (120, 120, 120), (x, y, self.PANEL_W, self.PANEL_H), 1)

    @staticmethod
    def _to_surface(arr: np.ndarray, cmap: str = "gray") -> "pygame.Surface":
        """将2D数组归一化到 0~255 并转为 pygame Surface."""
        import pygame

        arr = np.asarray(arr, dtype=np.float64)
        if arr.size == 0:
            arr = np.zeros((16, 16))
        vmin, vmax = float(arr.min()), float(arr.max())
        if vmax - vmin < 1e-9:
            norm = np.zeros_like(arr, dtype=np.uint8)
        else:
            norm = ((arr - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
        if cmap == "heat" and norm.ndim == 2:
            f = norm.astype(np.float64) / 255.0
            r = np.clip(1.5 - np.abs(4 * f - 3.0), 0.0, 1.0)
            g = np.clip(1.5 - np.abs(4 * f - 2.0), 0.0, 1.0)
            b = np.clip(1.5 - np.abs(4 * f - 1.0), 0.0, 1.0)
            rgb = np.dstack((r, g, b))
            rgb = (rgb * 255.0).astype(np.uint8)
            return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        if norm.ndim == 2:
            norm = np.dstack((norm, norm, norm))
        return pygame.surfarray.make_surface(norm.swapaxes(0, 1))

    def update_phase(self, phase: np.ndarray) -> None:
        """更新相位图案面板."""
        self._phase = np.asarray(phase)

    def update_target(self, target: np.ndarray) -> None:
        """更新目标强度掩模面板."""
        self._target = np.asarray(target)

    def update_image(self, image: np.ndarray) -> None:
        """更新远场光斑图像面板."""
        self._image = np.asarray(image)

    def update_loss(self, iteration: int, loss: float) -> None:
        """更新损失收敛曲线 (在 progress_callback 中调用)."""
        self._dl_iter = iteration
        self._loss_curve.append(float(loss))
        self._render()

    def update_status(self, score: float, side: int, spot_d: float) -> None:
        """更新状态文本 (评分, 边长, 光斑直径)."""
        self._status_lines = [
            f"外迭代评分: {score:.4f}",
            f"目标边长: {side} px",
            f"光斑直径: {spot_d:.1f} px",
        ]

    def render_static(self) -> None:
        """渲染当前所有面板 (每次外迭代后调用)."""
        self._render()

    def _draw_loss_curve(self) -> None:
        import pygame

        col = 1
        row = 1
        x = self.PAD + col * (self.PANEL_W + self.PAD)
        y = self.TITLE_H + self.PAD + row * (self.PANEL_H + self.PAD)
        area = pygame.Rect(x, y, self.PANEL_W, self.PANEL_H)
        self._screen.fill((0, 0, 0), area)

        curve = self._loss_curve
        if len(curve) < 2:
            t = self._font.render("Diff loss curve...", True, (150, 150, 150))
            self._screen.blit(t, (x + 20, y + 20))
            return

        cmin, cmax = min(curve), max(curve)
        span = (cmax - cmin) or 1.0
        points = []
        for i, val in enumerate(curve):
            px = x + int(i * (self.PANEL_W - 20) / max(len(curve) - 1, 1)) + 10
            py = y + self.PANEL_H - 10 - int((val - cmin) / span * (self.PANEL_H - 20))
            points.append((px, py))
        pygame.draw.lines(self._screen, (0, 255, 0), False, points, 2)

        t = self._font.render(
            f"DL iter {self._dl_iter + 1}, loss={curve[-1]:.2e}", True, (255, 255, 255)
        )
        self._screen.blit(t, (x + 10, y + 6))

    def _render(self) -> None:
        """渲染全部面板."""
        import pygame

        self._screen.fill(self.BG)

        header = " | ".join(self._status_lines) if self._status_lines else "Diff Shaping 闭环优化"
        head_surf = self._font.render(header, True, (0, 255, 255))
        self._screen.blit(head_surf, (self.PAD, 6))

        if self._phase is not None:
            self._draw_panel(0, "DL Phase", self._to_surface(self._phase, "gray"))
        if self._target is not None:
            self._draw_panel(1, "Target Mask", self._to_surface(self._target, "gray"))
        if self._image is not None:
            self._draw_panel(2, "Far-field Image", self._to_surface(self._image, "heat"))
        self._draw_loss_curve()

        pygame.display.update()
        self._clock.tick(30)


# ==================== 质量指标 (纯函数, 可独立测试) ====================

def compute_square_metrics(
    intensity: np.ndarray,
    target_side: int,
    center: tuple[float, float],
    energy: float = 0.90,
) -> dict[str, float]:
    """计算方形光束的质量指标.

    Args:
        intensity: 2D远场强度图像.
        target_side: 目标方形边长 (像素).
        center: 光束中心 (cx, cy).
        energy: 环围能量分数 (default 0.90).

    Returns:
        包含以下指标的字典:
            - aspect_ratio: 亮区长宽比 (>= 1, 1=完美方形)
            - squareness: abs(1 - aspect_ratio) (0=完美方形)
            - uniformity_cv: 方形区域内强度变异系数 (越小越均匀)
            - encircled_energy: 目标方形内能量占总能量比例
            - flatness_factor: 方形内均值 / 峰值 (平坦度)
            - intensity_max: 图像最大强度
            - intensity_mean: 图像平均强度
    """
    intensity = np.asarray(intensity, dtype=np.float64)
    total = float(np.sum(intensity))
    if total <= 0:
        return {
            "aspect_ratio": 1.0,
            "squareness": 0.0,
            "uniformity_cv": 0.0,
            "encircled_energy": 0.0,
            "flatness_factor": 0.0,
            "intensity_max": 0.0,
            "intensity_mean": 0.0,
        }

    cx, cy = int(round(center[0])), int(round(center[1]))
    h, w = intensity.shape

    # --- 长宽比: 亮区 (50%峰阈值) 的 x/y 跨度 ---
    peak = float(np.max(intensity))
    threshold = 0.5 * peak
    bright = intensity >= threshold
    if bright.any():
        ys, xs = np.nonzero(bright)
        width_bright = int(xs.max()) - int(xs.min()) + 1
        height_bright = int(ys.max()) - int(ys.min()) + 1
        aspect_ratio = max(width_bright, height_bright) / max(min(width_bright, height_bright), 1)
    else:
        aspect_ratio = 1.0

    # --- 方形区域内均匀性 ---
    half = max(target_side // 2, 1)
    y0 = max(cy - half, 0)
    y1 = min(cy + half, h)
    x0 = max(cx - half, 0)
    x1 = min(cx + half, w)
    region = intensity[y0:y1, x0:x1]
    region_mean = float(np.mean(region))
    region_std = float(np.std(region))
    uniformity_cv = region_std / max(region_mean, 1e-10) if region_mean > 0 else 0.0

    # --- 环围能量: 目标方形内能量占比 ---
    encircled_energy = float(np.sum(region)) / max(total, 1e-10)

    # --- 平坦度 ---
    flatness_factor = region_mean / max(peak, 1e-10) if peak > 0 else 0.0

    return {
        "aspect_ratio": float(aspect_ratio),
        "squareness": float(abs(1.0 - aspect_ratio)),
        "uniformity_cv": float(uniformity_cv),
        "encircled_energy": float(encircled_energy),
        "flatness_factor": float(flatness_factor),
        "intensity_max": float(peak),
        "intensity_mean": float(np.mean(intensity)),
    }


def compute_quality_score(metrics: dict[str, float]) -> float:
    """由质量指标计算综合评分 (0~1, 越高越好).

    加权组合三个主要指标:
        - 长宽比 (aspect_ratio 接近 1)
        - 均匀性 (uniformity_cv 接近 0)
        - 环围能量 (encircled_energy 接近 1)

    Args:
        metrics: compute_square_metrics 返回的指标字典.

    Returns:
        0~1 的综合评分.
    """
    f_ar = math.exp(-((metrics["aspect_ratio"] - 1.0) / 0.3) ** 2)
    f_uni = math.exp(-(metrics["uniformity_cv"] / 0.3) ** 2)
    f_ee = float(np.clip(metrics["encircled_energy"], 0.0, 1.0))
    return float(0.3 * f_ar + 0.4 * f_uni + 0.3 * f_ee)


def _centroid(intensity: np.ndarray) -> tuple[float, float]:
    """计算强度质心 (cx, cy)."""
    intensity = np.asarray(intensity, dtype=np.float64)
    total = float(np.sum(intensity))
    if total <= 0:
        h, w = intensity.shape
        return w / 2.0, h / 2.0
    ys, xs = np.mgrid[0 : intensity.shape[0], 0 : intensity.shape[1]]
    cx = float(np.sum(xs * intensity) / total)
    cy = float(np.sum(ys * intensity) / total)
    return cx, cy


def _bright_span(intensity: np.ndarray, peak_frac: float = 0.5) -> tuple[int, int]:
    """50%峰阈值亮区的外接宽高, 用于像素缩放标定.

    Args:
        intensity: 2D强度图像.
        peak_frac: 峰阈值比例 (default 0.5).

    Returns:
        (宽, 高) 亮区包围盒尺寸; 无亮区时返回 (0, 0).
    """
    intensity = np.asarray(intensity, dtype=np.float64)
    peak = float(np.max(intensity))
    if peak <= 0:
        return 0, 0
    bright = intensity >= peak_frac * peak
    if not bright.any():
        return 0, 0
    ys, xs = np.nonzero(bright)
    return int(xs.max()) - int(xs.min()) + 1, int(ys.max()) - int(ys.min()) + 1


def _clamp_side(side: int, height: int, width: int) -> int:
    """方形边长钳制到SLM网格内 (留8px边距)."""
    max_side = min(height, width) - 8
    if side > max_side:
        logger.warning("方形边长 {}px 超出SLM网格, 钳制为 {}px", side, max_side)
        return max_side
    return side


# ==================== 闭环核心 ====================

def _run_closed_loop(
    slm: SantecSLM200,
    camera,
    *,
    target_shape: str,
    target_size: int,
    target_cam_px: int | None,
    gs_factor: float,
    focal_length_m: float,
    gs_energy: float,
    p_cam: float | None,
    pixel_scale: float | None,
    propagation: str,
    dl_iterations: int,
    optimizer: str,
    lr: float,
    w_uniformity: float,
    w_efficiency: float,
    w_zero_order: float,
    w_smoothness: float,
    cell_spacing_um: float,
    wavelength_nm: int,
    seed: int | None,
    device: str | None = None,
    outer_iterations: int,
    convergence_threshold: float,
    settle_time: float,
    n_sample: int,
    refine: bool,
    output_dir: Path,
    display: bool = False,
) -> dict:
    """执行可微分闭环光束整形.

    Args:
        slm: 已打开的SLM实例.
        camera: 已打开的相机实例.
        target_shape: 目标形状 ("square", "circle", "gaussian", "spot").
        target_size: 目标尺寸 (SLM 网格像素).
        target_cam_px: 目标在相机上的像素宽度; None=按 gs_factor×光斑直径.
        gs_factor: 目标/光斑尺寸因子 (仅当 target_cam_px 未指定时使用).
        focal_length_m: 焦距/传播距离 (米).
        gs_energy: 光斑测量的环围能量.
        p_cam: 相机像素间距 (米); None=使用SLM间距.
        pixel_scale: 像素缩放比 k=SLM网格边长/相机亮区宽度; None=自动标定.
        propagation: 传播模型 ("asm"=角谱法, "fft"=单FFT夫琅禾费).
        dl_iterations: 每次外迭代的梯度优化内迭代次数.
        optimizer: 梯度优化器 ("adam" 或 "lbfgs").
        lr: 梯度优化学习率.
        w_uniformity: 均匀性损失权重.
        w_efficiency: 效率损失权重.
        w_zero_order: 零级损失权重.
        w_smoothness: 平滑损失权重.
        cell_spacing_um: SLM 像素间距 (微米).
        wavelength_nm: 工作波长 (纳米).
        seed: 随机种子.
        outer_iterations: 最大外迭代次数.
        convergence_threshold: 收敛评分阈值 (0~1).
        settle_time: SLM稳定等待时间 (秒).
        n_sample: 相机每次采样平均帧数.
        refine: 是否在每次外迭代后运行额外细化梯度通道.
        output_dir: 输出目录.
        display: 是否启用pygame实时可视化.

    Returns:
        结果字典, 包含 best_phase, best_score, convergence_history 等.
    """
    # Lazy import: torch 可选, 仅在进入硬件循环时检查
    try:
        from ao_shaping.algorithm.differentiable_shaping import (
            create_target_mask,
            train_beam_shaping,
        )
    except ImportError as e:
        raise RuntimeError(
            "PyTorch 未安装或 differentiable_shaping 模块不可用. "
            "请运行 'uv sync --extra ml' 安装 PyTorch 后重试."
        ) from e

    from ao_shaping.gui.slm.multi_slm_controller import (
        build_square_target_amplitude,
        measure_spot_diameter_cam,
    )

    width = slm.Panel_Res[0]
    height = slm.Panel_Res[1]
    d_slm = float(slm.Pitch_um) * 1e-6  # um -> m
    wavelength = float(slm.wavelength) if slm.wavelength is not None else 1064.0
    wavelength_m = wavelength * 1e-9
    cell_spacing_m = cell_spacing_um * 1e-6

    # SLM内存槽随机选取 (2~125, 禁止连续使用同一槽位, 含跨进程重启边界)
    # 驱动固件: display_memory(同一槽) 是 no-op, LCOS 不刷新 → 前后两次相位
    # 写入必须落在不同槽。在 2~125 大范围内随机选槽, 单次运行内相邻两次写入
    # 几乎必然不同; 启动时读取 SLM 当前显示的槽号 (memory 模式 SLM_Ctrl_ReadDS
    # 有效) 并排除之, 使跨进程重启第一轮也不会与上次末槽相同。
    _SLOT_MIN, _SLOT_MAX = 2, 125
    _last_slot_used: int | None = None
    try:
        _last_slot_used = slm.get_displayed_memory_number()
        logger.info("SLM当前显示槽: {}", _last_slot_used)
    except Exception:
        # 读取失败 (如 set_grayscale 模式, 无内存槽概念), 首次随机选取即可
        _last_slot_used = None

    def _pick_next_slot() -> int:
        """从 2~125 随机选取一个与上次写入不同的 SLM 内存槽 (防连续同槽 no-op)."""
        nonlocal _last_slot_used
        candidates = [s for s in range(_SLOT_MIN, _SLOT_MAX + 1) if s != _last_slot_used]
        slot = random.choice(candidates)
        _last_slot_used = slot
        return slot

    # 可选pygame实时显示
    display_stack = contextlib.ExitStack()
    display_ctx: _DiffDisplay | None = None
    if display:
        try:
            display_ctx = _DiffDisplay()
            display_stack.enter_context(display_ctx)
        except Exception as e:
            logger.warning("pygame显示初始化失败, 本次运行禁用显示: {}", e)
            display_ctx = None
            display_stack.close()

    def render_display() -> None:
        """渲染显示面板 (外迭代后调用)."""
        if display_ctx is None:
            return
        try:
            display_ctx.render_static()
        except Exception as e:
            logger.warning("pygame渲染失败: {}", e)

    # 初始光源振幅: 均匀 (SLM分辨率)
    source_amplitude: np.ndarray | None = None
    initial_phase: np.ndarray | None = None

    convergence_history: list[dict] = []
    loss_history: list[float] = []
    all_images: list[np.ndarray] = []
    all_phases: list[np.ndarray] = []
    all_metrics: list[dict] = []

    best_phase = None
    best_phase_rad = None
    best_score = -1.0
    best_image = None
    best_iter = -1

    # 首次采集, 用于测量光斑
    init_image = np.asarray(
        camera.get_numpy_image(n_sample=n_sample, skip_first=True), dtype=np.float64
    )
    all_images.append(init_image)
    cx, cy = _centroid(init_image)

    logger.info("初始光斑: center=({:.1f}, {:.1f}), max={}", cx, cy, init_image.max())
    spot_d = measure_spot_diameter_cam(init_image, energy=gs_energy)

    # 目标尺寸 (相机像素宽):
    #   1) --target-px 显式给定
    #   2) 未给定: 兼容旧语义 target_px = gs_factor × spot_d
    if target_cam_px is not None:
        target_px = float(target_cam_px)
        logger.info("目标尺寸: {}px (相机像素宽)", int(target_px))
    else:
        target_px = gs_factor * spot_d
        logger.info(
            "目标尺寸: {:.0f}px = gs_factor {} × 光斑直径 {:.0f}px",
            target_px, gs_factor, spot_d,
        )

    # 像素缩放比 k = side_slm_px / 亮区宽度_cam_px
    if pixel_scale is not None:
        auto_calib = False
        side = _clamp_side(
            int(round(target_px * float(pixel_scale))), height, width
        )
    elif p_cam is not None:
        auto_calib = False
        k_guess = float(p_cam) / d_slm
        side = _clamp_side(int(round(target_px * k_guess)), height, width)
    else:
        auto_calib = True
        side = int(round(target_px * 0.4))  # k 初猜 0.4
        logger.info("自动标定像素缩放: 首轮以探测边长 {}px 运行, 采集后更新目标边长", side)

    # 如果用户直接指定 --target-size, 用它覆盖 side
    if target_size > 0:
        side = _clamp_side(target_size, height, width)
        auto_calib = False
        logger.info("使用指定目标尺寸: {}px (SLM 网格)", side)

    logger.info("测得光斑直径 {:.1f}px, 目标边长 {}px", spot_d, side)

    # 亮度基线提示 (启动诊断): 当前亮度 vs 理想整形后的目标最优亮度.
    # 假设光总能量不变 (sum 守恒): 若能量均匀分布于目标方形亮区
    # (target_px × target_px 相机像素), 理论最大亮度 = E / target_px².
    # 若当前 max ≪ 该值, 说明能量仍集中在零级/未展开, 或曝光过暗.
    _total_energy = float(init_image.sum())
    _ideal_max = _total_energy / (target_px * target_px) if target_px > 0 else 0.0
    logger.info(
        "亮度基线: 当前max={:.0f} mean={:.3f} 总能量E={:.0f}; "
        "目标最优最大亮度(总能量不变, 均匀分布于{:.0f}px方形)={:.1f}",
        init_image.max(),
        init_image.mean(),
        _total_energy,
        target_px,
        _ideal_max,
    )

    # 迭代记录器
    recorder = Recorder(mark="quality_score", mode="max")

    for outer_iter in range(outer_iterations):
        if not _running:
            logger.info("检测到停止信号, 提前终止 (iter={})", outer_iter + 1)
            break

        logger.info("\n" + "=" * 60)
        logger.info("外迭代 {}/{}", outer_iter + 1, outer_iterations)

        # --- 1. 构建目标强度掩模 ---
        target_mask = create_target_mask(target_shape, (height, width), side)

        # --- 2. 运行可微分梯度优化 ---
        logger.info(
            "运行可微分优化 ({}/{} 外迭代, {} 内迭代, {})...",
            outer_iter + 1, outer_iterations, dl_iterations, optimizer,
        )

        def _dl_progress(iteration: int, loss: float) -> None:
            """梯度优化内迭代进度回调 (每个迭代输出日志 + 更新pygame损失曲线)."""
            logger.info(
                "内迭代 {:4d}/{:d} loss={:.6f}",
                iteration + 1, dl_iterations, loss,
            )
            if display_ctx is None:
                return
            try:
                display_ctx.update_loss(iteration, loss)
            except Exception:
                pass

        def _dl_phase_callback(phase) -> None:
            """梯度优化相位回调 (可选: 实时写相位到SLM)."""
            pass  # 不在内迭代中写SLM, 等求解完成后再写, 避免SLM磨损

        result = train_beam_shaping(
            target=target_mask,
            grid_size=(height, width),
            source_amplitude=source_amplitude,
            initial_phase=initial_phase,
            propagation=propagation,
            optimizer=optimizer,
            iterations=dl_iterations,
            lr=lr,
            w_uniformity=w_uniformity,
            w_efficiency=w_efficiency,
            w_zero_order=w_zero_order,
            w_smoothness=w_smoothness,
            cell_spacing=cell_spacing_m,
            distance=focal_length_m,
            wavelength=wavelength_m,
            seed=seed,
            device=device,
            progress_callback=_dl_progress,
            phase_callback=_dl_phase_callback,
        )

        # 可选: 额外细化梯度通道
        if refine:
            logger.info("运行细化梯度通道 ({}/{} 迭代)...", dl_iterations, optimizer)
            result = train_beam_shaping(
                target=target_mask,
                grid_size=(height, width),
                source_amplitude=source_amplitude,
                initial_phase=result.phase,
                propagation=propagation,
                optimizer=optimizer,
                iterations=dl_iterations,
                lr=lr * 0.1,  # 细化阶段降低学习率
                w_uniformity=w_uniformity,
                w_efficiency=w_efficiency,
                w_zero_order=w_zero_order,
                w_smoothness=w_smoothness,
                cell_spacing=cell_spacing_m,
                distance=focal_length_m,
                wavelength=wavelength_m,
                seed=seed,
                progress_callback=_dl_progress,
                phase_callback=_dl_phase_callback,
            )

        # --- 3. 转换相位为灰度并下发 ---
        phase_gray = slm.create_phase_from_array(result.phase)
        slot = _pick_next_slot()
        slm.write_phase(phase_gray, memory_number=slot)
        time.sleep(0.05)
        slm.display_memory(slot)
        logger.info("相位已写入SLM内存槽 {}", slot)

        # --- 4. 等待SLM稳定 ---
        if settle_time > 0:
            time.sleep(settle_time)

        # --- 5. 重新采集 ---
        new_image = np.asarray(
            camera.get_numpy_image(n_sample=n_sample, skip_first=True), dtype=np.float64
        )
        cx, cy = _centroid(new_image)

        # --- 6. 计算质量 ---
        metrics: dict[str, Any] = compute_square_metrics(
            new_image, side, (cx, cy), energy=gs_energy
        )
        score = compute_quality_score(metrics)

        metrics["iteration"] = outer_iter + 1
        metrics["quality_score"] = float(score)
        metrics["side"] = int(side)
        metrics["center"] = [float(cx), float(cy)]
        metrics["dl_loss_final"] = (
            float(result.loss_history[-1]) if result.loss_history else None
        )
        metrics["dl_converged"] = bool(result.converged)

        convergence_history.append(metrics)
        all_images.append(new_image)
        all_phases.append(phase_gray)
        all_metrics.append(metrics)

        # 累积损失历史
        if result.loss_history:
            loss_history.extend(result.loss_history)

        logger.info(
            "iter {}: 评分={:.4f} AR={:.3f} squareness={:.3f} CV={:.3f} EE={:.3f} dl_loss={}",
            outer_iter + 1,
            score,
            metrics["aspect_ratio"],
            metrics["squareness"],
            metrics["uniformity_cv"],
            metrics["encircled_energy"],
            metrics["dl_loss_final"],
        )

        # --- 6.5 更新pygame显示面板 ---
        if display_ctx is not None:
            try:
                display_ctx.update_phase(result.phase)
                display_ctx.update_target(target_mask)
                display_ctx.update_image(new_image)
                display_ctx.update_status(score, int(side), float(spot_d))
                display_ctx.render_static()
            except Exception as e:
                logger.warning("pygame显示更新失败: {}", e)

        # --- 7. 保存本次迭代产物 ---
        iter_dir = output_dir / f"iter_{outer_iter + 1:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        np.save(iter_dir / "captured_image.npy", new_image)
        np.save(iter_dir / "dl_phase.npy", phase_gray)
        np.save(iter_dir / "target_mask.npy", target_mask)
        if result.phase is not None:
            np.save(iter_dir / "dl_phase_rad.npy", result.phase)
        with (iter_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        if result.loss_history:
            with (iter_dir / "dl_loss_history.json").open("w", encoding="utf-8") as f:
                json.dump(result.loss_history, f)

        # 追加本次迭代 record 并增量保存 pkl
        recorder.append(
            {
                "quality_score": float(score),
                "iteration": outer_iter + 1,
                "side": int(side),
                "spot_d": float(spot_d),
                "center": [float(cx), float(cy)],
                "dl_loss_final": (
                    float(result.loss_history[-1]) if result.loss_history else None
                ),
                "dl_converged": bool(result.converged),
                "aspect_ratio": float(metrics["aspect_ratio"]),
                "squareness": float(metrics["squareness"]),
                "uniformity_cv": float(metrics["uniformity_cv"]),
                "encircled_energy": float(metrics["encircled_energy"]),
                "phase": np.asarray(result.phase, dtype=np.float64),
                "phase_gray": np.asarray(phase_gray, dtype=np.uint16),
                "image": np.asarray(new_image, dtype=np.float64),
                "target": np.asarray(target_mask, dtype=np.float64),
            }
        )
        recorder.dataframe.to_pickle(output_dir / "diff_shaping_records.pkl", compression="zip")

        # --- 8. 跟踪最优 ---
        if score > best_score:
            best_score = float(score)
            best_phase = phase_gray
            best_phase_rad = result.phase
            best_image = new_image
            best_iter = outer_iter + 1
            logger.info("更新最优: iter={}, score={:.4f}", best_iter, best_score)

        # --- 9. 收敛判断 ---
        if score >= convergence_threshold:
            logger.info("达到收敛阈值 {:.3f}, 停止迭代", convergence_threshold)
            break

        # 更新光源振幅为当前实测光强 (闭环核心)
        source_amplitude = resample_to_grid(new_image, (height, width))
        source_amplitude = source_amplitude / (np.max(source_amplitude) + 1e-10)

        # 梯度优化继续从当前相位开始 (相位延续)
        initial_phase = result.phase

        # 自动标定模式
        if auto_calib:
            bright_w, bright_h = _bright_span(new_image)
            bright_w = max(bright_w, bright_h)
            if bright_w > 5:
                k_meas = side / bright_w
                next_side = _clamp_side(
                    int(round(target_px * k_meas)), height, width
                )
                logger.info(
                    "像素标定: k={:.4f} (边长{}px→亮区{:.0f}px), 下次目标边长 {}px",
                    k_meas,
                    side,
                    bright_w,
                    next_side,
                )
                side = next_side
            else:
                logger.warning(
                    "本轮亮区过小/未检测到 ({:.0f}px), 保持目标边长 {}px", bright_w, side
                )

    # 汇总结果
    result_dict = {
        "best_phase": best_phase,
        "best_phase_rad": best_phase_rad if best_phase is not None else None,
        "best_score": float(best_score) if best_score >= 0 else 0.0,
        "best_iter": best_iter,
        "best_image": best_image,
        "convergence_history": convergence_history,
        "loss_history": loss_history,
        "all_images": all_images,
        "all_phases": all_phases,
        "all_metrics": all_metrics,
        "side": int(side),
        "spot_diameter_px": float(spot_d),
        "resolution": [width, height],
    }
    display_stack.close()
    return result_dict


def _save_results(result: dict, output_dir: Path, params: dict) -> None:
    """保存最终结果 (最优相位, 收敛历史, 元数据)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if result["best_phase"] is not None:
        np.save(output_dir / "best_phase.npy", result["best_phase"])
        if result.get("best_phase_rad") is not None:
            np.save(output_dir / "best_phase_rad.npy", result["best_phase_rad"])
        if result["best_image"] is not None:
            np.save(output_dir / "best_image.npy", result["best_image"])

    # 保存最优目标掩模和模拟强度 (如果有)
    if result["all_images"]:
        np.save(output_dir / "best_image.npy", result["all_images"][result["best_iter"]])

    with (output_dir / "convergence_history.json").open("w", encoding="utf-8") as f:
        json.dump(result["convergence_history"], f, ensure_ascii=False, indent=2)

    with (output_dir / "loss_history.json").open("w", encoding="utf-8") as f:
        json.dump(result["loss_history"], f)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_score": result["best_score"],
        "best_iteration": result["best_iter"],
        "side": result["side"],
        "spot_diameter_px": result["spot_diameter_px"],
        "resolution": result["resolution"],
        "converged": result["best_score"] >= params.get("convergence_threshold", 0.95),
        "params": params,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("最终结果已保存到 {}", output_dir)


# ==================== CLI ====================

@click.command()
# 相机
@click.option(
    "--camera-type",
    type=click.Choice(["daheng", "miicam"]),
    default="daheng",
    help="相机类型 (default: daheng)",
)
@click.option(
    "--cam-id",
    default=None,
    type=int,
    help="相机ID (default: FAR_CAM_ID 环境变量或 0)",
)
@click.option("--exposure-ms", default=None, type=float,
              help="相机曝光时间 ms (default: 自动 miicam=0.02 / daheng=50)")
@click.option("--cam-bit-depth", default=8, type=click.IntRange(8, 16),
              help="MiiCam输出位深 (default: 8)")
# SLM
@click.option("--slm-number", default=1, type=int, help="SLM设备编号 1-8 (default: 1)")
@click.option("--slm-wavelength", default=1064, type=int, help="SLM工作波长 nm (default: 1064)")
# 目标参数
@click.option(
    "--target-shape",
    type=click.Choice(["square", "circle", "gaussian", "spot"]),
    default="square",
    help="目标形状 (default: square)",
)
@click.option("--target-size", default=0, type=int,
              help="目标尺寸 (SLM 网格 px); 0=自动 (default: 0)")
@click.option(
    "--target-px",
    default=None,
    type=int,
    help="目标在相机上的像素宽度; 推荐显式指定 (default: 按 --gs-factor×光斑直径)",
)
@click.option("--gs-factor", default=1.5, type=float,
              help="目标/光斑尺寸因子 (default: 1.5)")
# 物理参数
@click.option("--focal-length", default=0.1, type=float, help="焦距/传播距离 m (default: 0.1)")
@click.option("--gs-energy", default=0.90, type=float, help="光斑测量环围能量 (default: 0.90)")
@click.option("--cell-spacing", default=8.0, type=float,
              help="SLM 像素间距 um (default: 8)")
@click.option("--p-cam", default=None, type=float,
              help="相机像素间距 m (default: 使用SLM间距)")
@click.option("--pixel-scale", default=None, type=float,
              help="像素缩放比 k=SLM网格边长/相机亮区宽度; None=自动标定 (default: 自动)")
@click.option(
    "--propagation",
    type=click.Choice(["fft", "asm"], case_sensitive=False),
    default="fft",
    show_default=True,
    help="传播模型: fft=单FFT夫琅禾费(高速), asm=角谱法(精确)",
)
# 可微分优化器参数
# ---------------------------------------------------------------------------
# 默认权重/学习率经实验标定 (docs/slm_differential_shaping/):
#   * w_zero_order=0 是必须的 —— 零级惩罚会把能量从居中目标推出
#     (默认 [.4,.4,.1,.1] 下 EE 崩到 0.07, 而 w=[.4,.6,0,0] 达 EE≈0.84)
#   * lr=3e-2 + 600 次内迭代: fft/adam 方形 CV<0.1 EE≈0.84 (seed 1-3 稳健),
#     asm 亦能收敛 (CV≈0.001 EE≈0.90); lbfgs (lr=1.0) 60 步即近平顶
#   * 若想手动校验, 可用 --dl-iterations 500+ 或 --optimizer lbfgs
@click.option("--dl-iterations", "-i", default=500, type=int,
              help="每次外迭代的梯度优化内迭代次数 (default: 500; 600 达 CV<0.1)")
@click.option(
    "--optimizer",
    type=click.Choice(["adam", "lbfgs"]),
    default="adam",
    show_default=True,
    help="梯度优化器 (default: adam)",
)
@click.option("--lr", default=3e-2, type=float, help="梯度优化学习率 (default: 3e-2)")
@click.option("--w-uniformity", default=0.4, type=float,
              help="均匀性损失权重 (default: 0.4)")
@click.option("--w-efficiency", default=0.6, type=float,
              help="效率损失权重 (default: 0.6; >= 均匀性权重先集中能量再展平)")
@click.option("--w-zero-order", default=0.0, type=float,
              help="零级损失权重 (default: 0.0; 非零会把能量推出居中目标)")
@click.option("--w-smoothness", default=0.0, type=float,
              help="平滑损失权重 (default: 0.0; 抑制方形锐边所需的高频相位)")
@click.option("--seed", default=None, type=int, help="随机种子 (default: None)")
@click.option(
    "--device",
    type=click.Choice(["auto", "cuda", "cpu"]),
    default="auto",
    help="计算设备: auto=有CUDA则用GPU否则CPU (default: auto)",
)
# 闭环参数
@click.option("--outer-iterations", "-n", default=10, type=int,
              help="最大外迭代次数 (default: 10)")
@click.option("--convergence-threshold", default=0.95, type=float,
              help="收敛评分阈值 0~1 (default: 0.95)")
@click.option("--settle-time", default=0.5, type=float,
              help="SLM稳定等待时间 s (default: 0.5)")
@click.option("--n-sample", default=3, type=int,
              help="相机每次采样平均帧数 (default: 3)")
@click.option("--refine/--no-refine", default=False,
              help="每次外迭代后运行额外细化梯度通道 (default: False)")
# 输出
@click.option("-o", "--output", default="data/diff_shaping",
              help="输出目录 (default: data/diff_shaping)")
@click.option(
    "--display/--no-display",
    default=False,
    help="启用pygame实时可视化迭代过程 (default: False)",
)
def run(
    camera_type: str,
    cam_id: int | None,
    exposure_ms: float,
    cam_bit_depth: int,
    slm_number: int,
    slm_wavelength: int,
    target_shape: str,
    target_size: int,
    target_px: int | None,
    gs_factor: float,
    focal_length: float,
    gs_energy: float,
    cell_spacing: float,
    p_cam: float | None,
    pixel_scale: float | None,
    propagation: str,
    dl_iterations: int,
    optimizer: str,
    lr: float,
    w_uniformity: float,
    w_efficiency: float,
    w_zero_order: float,
    w_smoothness: float,
    seed: int | None,
    device: str,
    outer_iterations: int,
    convergence_threshold: float,
    settle_time: float,
    n_sample: int,
    refine: bool,
    output: str,
    display: bool,
) -> None:
    """可微分闭环光束整形优化器

    利用 PyTorch 可微分优化 + CCD 反馈迭代优化 SLM 相位,
    将远场光斑整形为目标形状 (square/circle/gaussian/spot).
    """
    global _running
    _running = True
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # 相机ID默认值
    if cam_id is None:
        cam_id = int(os.environ.get("FAR_CAM_ID", "0"))

    # 曝光时间默认值: miicam 按本光路实测近饱和基线 0.02ms (1064nm 2f 傅里叶光路),
    # daheng 保持 50ms
    if exposure_ms is None:
        exposure_ms = 0.02 if camera_type == "miicam" else 50.0
        logger.info("曝光时间默认: {}ms (camera_type={})", exposure_ms, camera_type)
    elif camera_type == "miicam" and exposure_ms > 50.0:
        logger.warning(
            "miicam 曝光 {:.3f}ms 偏大, 建议 0.02~1ms 避免饱和", exposure_ms
        )

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    slm: SantecSLM200 | None = None
    camera = None

    try:
        # --- 打开相机 ---
        if camera_type == "daheng":
            camera = _get_daheng_camera(cam_id, exposure_ms)
        else:
            camera = _get_miicam_camera(cam_id, exposure_ms, cam_bit_depth)
        logger.info("相机已连接: type={}, id={}", camera_type, cam_id)

        # --- 打开SLM ---
        logger.info("正在连接SLM #{}...", slm_number)
        slm = SantecSLM200(
            slm_number=slm_number,
            wavelength=slm_wavelength,
            video_mode=0,  # 内存模式
        )
        slm.open()
        # 2026-09-08 实测 (与相机无关的运维记录):
        #   diff-shaping 硬件闭环可能在此行之后无限阻塞 —— 日志止于 SLM open 成功,
        #   而下方首次 camera.get_numpy_image() 无任何输出挂死 (本例 900s 超时被强杀)。
        #   WaitImageV3 的原生等待由 SDK 内部驱动, 不受 Python 侧超时保护; 排查挂点
        #   请用分步计时的最小探针脚本逐调用计时, 不要直接跑完整闭环。
        #   强杀 (外部 timeout/硬断) 不会经过本函数 finally 的 close, SLM 控制器可能
        #   残留异常状态; 重跑前先确认无残留 python 进程 (Get-Process python*),
        #   若 memory 模式 open() 超过数秒无日志, 对 SLM 控制器物理断电重置
        #   (与 DVI 模式挂起同一处置)。
        logger.info(
            "SLM #{} 已连接, 分辨率: {}x{}, 位深: {}bit",
            slm_number,
            slm.Panel_Res[0],
            slm.Panel_Res[1],
            slm.Gray_Scale_bits,
        )

        params = {
            "camera_type": camera_type,
            "cam_id": cam_id,
            "exposure_ms": exposure_ms,
            "cam_bit_depth": cam_bit_depth,
            "slm_number": slm_number,
            "slm_wavelength": slm_wavelength,
            "target_shape": target_shape,
            "target_size": target_size,
            "target_px": target_px,
            "gs_factor": gs_factor,
            "focal_length_m": focal_length,
            "gs_energy": gs_energy,
            "cell_spacing_um": cell_spacing,
            "p_cam": p_cam,
            "pixel_scale": pixel_scale,
            "propagation": propagation,
            "dl_iterations": dl_iterations,
            "optimizer": optimizer,
            "lr": lr,
            "w_uniformity": w_uniformity,
            "w_efficiency": w_efficiency,
            "w_zero_order": w_zero_order,
            "w_smoothness": w_smoothness,
            "seed": seed,
            "device": device,
            "outer_iterations": outer_iterations,
            "convergence_threshold": convergence_threshold,
            "settle_time": settle_time,
            "n_sample": n_sample,
            "refine": refine,
            "display": display,
        }

        # --- 执行闭环 ---
        result = _run_closed_loop(
            slm,
            camera,
            target_shape=target_shape,
            target_size=target_size,
            target_cam_px=target_px,
            gs_factor=gs_factor,
            focal_length_m=focal_length,
            gs_energy=gs_energy,
            p_cam=p_cam,
            pixel_scale=pixel_scale,
            propagation=propagation,
            dl_iterations=dl_iterations,
            optimizer=optimizer,
            lr=lr,
            w_uniformity=w_uniformity,
            w_efficiency=w_efficiency,
            w_zero_order=w_zero_order,
            w_smoothness=w_smoothness,
            cell_spacing_um=cell_spacing,
            wavelength_nm=slm_wavelength,
            seed=seed,
            device=None if device == "auto" else device,
            outer_iterations=outer_iterations,
            convergence_threshold=convergence_threshold,
            settle_time=settle_time,
            n_sample=n_sample,
            refine=refine,
            output_dir=output_dir,
            display=display,
        )

        # --- 保存结果 ---
        _save_results(result, output_dir, params)

        n_iters = len(result["convergence_history"])
        logger.info("闭环完成: 运行 {} 次迭代, 最优评分 {:.4f}", n_iters, result["best_score"])

    except Exception as e:
        logger.error("Diff shaping runner failed: {}", e)
        raise
    finally:
        logger.info("正在清理设备...")
        if slm is not None:
            try:
                slm.close()
                logger.info("SLM已断开")
            except Exception as e:
                logger.warning("SLM断开失败: {}", e)
        if camera is not None:
            try:
                camera.close()
                logger.info("相机已断开")
            except Exception as e:
                logger.warning("相机断开失败: {}", e)


def main() -> None:
    """CLI主入口, 统一异常处理."""
    try:
        run()
    except SystemExit:
        raise
    except Exception as e:
        click.echo(f"Error: {e}")
        logger.exception("Diff shaping runner failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
