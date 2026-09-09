"""GS闭环光束整形优化器

利用Gerchberg-Saxton(GS)算法 + CCD反馈，迭代优化SLM相位图案，
将远场光斑整形为方形。

闭环流程 (每个外迭代):
    1. CCD采集当前光束图像
    2. 测量光斑直径
    3. 计算方形目标尺寸 (factor × 光斑直径)
    4. 运行GS算法 (内迭代) 生成相位
    5. 下发相位到SLM (内存槽轮换)
    6. 等待SLM稳定
    7. 重新采集光束图像
    8. 计算方形质量指标 (长宽比、均匀性、环围能量)
    9. 记录进度并检查收敛

用法:
    python -m ao_shaping.runners.gs_square_runner --camera-type daheng --outer-iterations 10
    python -m ao_shaping.runners.gs_square_runner --camera-type miicam --gs-iterations 150 -o data/gs_square
"""

from __future__ import annotations

import contextlib
import itertools
import json
import math
import os
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

class _GSDisplay:
    """GS迭代过程pygame可视化窗口.

    显示四个面板:
        - 左上: GS相位图案 (0~1 归一化)
        - 右上: 目标方形振幅
        - 左下: 远场光斑图像 (CCD采集)
        - 右下: GS误差收敛曲线

    用法:
        with _GSDisplay() as disp:
            disp.update_phase(phase)
            disp.update_target(target_amplitude)
            disp.update_image(captured_image)
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

        # 缓存的当前状态
        self._phase: np.ndarray | None = None
        self._target: np.ndarray | None = None
        self._image: np.ndarray | None = None
        self._error_curve: list[float] = []
        self._status_lines: list[str] = []
        self._gs_iter = 0
        self._window_size = (w, h)

    def __enter__(self) -> "_GSDisplay":
        import pygame

        pygame.init()
        pygame.display.set_caption("GS Square 闭环优化")
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

        # 标题
        title_surf = self._font.render(title, True, (255, 255, 0))
        self._screen.blit(title_surf, (x + 4, y - self.TITLE_H + 2))

        # 图像
        scaled = pygame.transform.scale(
            surface, (self.PANEL_W, self.PANEL_H)
        )
        self._screen.blit(scaled, (x, y))
        # 边框
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
            # 伪彩色 (jet 简化近似)
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
        """更新GS相位图案面板."""
        self._phase = np.asarray(phase)

    def update_target(self, target: np.ndarray) -> None:
        """更新目标方形振幅面板."""
        self._target = np.asarray(target)

    def update_image(self, image: np.ndarray) -> None:
        """更新远场光斑图像面板."""
        self._image = np.asarray(image)

    def update_error(self, iteration: int, error: float) -> None:
        """更新GS误差收敛曲线 (在progress_callback中调用)."""
        self._gs_iter = iteration
        self._error_curve.append(float(error))
        self._render()

    def update_status(self, score: float, side: int, spot_d: float) -> None:
        """更新状态文本 (评分, 边长, 光斑直径)."""
        self._status_lines = [
            f"外迭代评分: {score:.4f}",
            f"方形边长: {side} px",
            f"光斑直径: {spot_d:.1f} px",
        ]

    def render_static(self) -> None:
        """渲染当前所有面板 (每次外迭代后调用)."""
        self._render()

    def _draw_error_curve(self) -> None:
        import pygame

        col = 1
        row = 1
        x = self.PAD + col * (self.PANEL_W + self.PAD)
        y = self.TITLE_H + self.PAD + row * (self.PANEL_H + self.PAD)
        area = pygame.Rect(x, y, self.PANEL_W, self.PANEL_H)
        self._screen.fill((0, 0, 0), area)

        curve = self._error_curve
        if len(curve) < 2:
            # 显示占位文本
            t = self._font.render("GS error curve...", True, (150, 150, 150))
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

        # 当前迭代文本
        t = self._font.render(
            f"GS iter {self._gs_iter + 1}, MSE={curve[-1]:.2e}", True, (255, 255, 255)
        )
        self._screen.blit(t, (x + 10, y + 6))

    def _render(self) -> None:
        """渲染全部面板."""
        import pygame

        self._screen.fill(self.BG)

        # 标题栏 (总进度)
        header = " | ".join(self._status_lines) if self._status_lines else "GS Square 闭环优化"
        head_surf = self._font.render(header, True, (0, 255, 255))
        self._screen.blit(head_surf, (self.PAD, 6))

        if self._phase is not None:
            self._draw_panel(0, "GS Phase", self._to_surface(self._phase, "gray"))
        if self._target is not None:
            self._draw_panel(1, "Target Square", self._to_surface(self._target, "gray"))
        if self._image is not None:
            self._draw_panel(2, "Far-field Image", self._to_surface(self._image, "heat"))
        self._draw_error_curve()

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


def _resample_to_grid(img: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """将相机图像重采样到SLM网格尺寸 (height, width), 作为GS光源振幅.

    以光斑峰值为中心, 按目标宽高比裁剪相机帧后双线性缩放, 避免整帧
    直接缩放产生的比例失真 (相机与SLM宽高比不同).
    """
    from scipy.ndimage import zoom

    th, tw = target_shape
    img = np.asarray(img, dtype=np.float64)
    ih, iw = img.shape
    if (ih, iw) == (th, tw):
        return img

    cy, cx = np.unravel_index(int(np.argmax(img)), img.shape)
    target_ar = tw / th

    if iw / ih > target_ar:
        # 相机过宽: 裁剪宽度以匹配目标宽高比
        crop_w = max(1, int(ih * target_ar))
        x0 = max(0, min(iw - crop_w, int(cx - crop_w / 2)))
        crop = img[:, x0 : x0 + crop_w]
    else:
        # 相机过高: 裁剪高度以匹配目标宽高比
        crop_h = max(1, int(iw / target_ar))
        y0 = max(0, min(ih - crop_h, int(cy - crop_h / 2)))
        crop = img[y0 : y0 + crop_h, :]

    zoom_y, zoom_x = th / crop.shape[0], tw / crop.shape[1]
    return np.asarray(zoom(crop, (zoom_y, zoom_x), order=1), dtype=np.float64)


# ==================== 闭环核心 ====================

def _run_closed_loop(
    slm: SantecSLM200,
    camera,
    *,
    gs_iterations: int,
    gs_factor: float,
    focal_length_m: float,
    gs_energy: float,
    p_cam: float | None,
    outer_iterations: int,
    convergence_threshold: float,
    settle_time: float,
    n_sample: int,
    output_dir: Path,
    display: bool = False,
) -> dict:
    """执行GS闭环光束整形.

    Args:
        slm: 已打开的SLM实例.
        camera: 已打开的相机实例.
        gs_iterations: GS内迭代次数.
        gs_factor: 方形/光斑尺寸因子.
        focal_length_m: 焦距/传播距离 (米).
        gs_energy: 光斑测量的环围能量.
        p_cam: 相机像素间距 (米); None=使用SLM间距.
        outer_iterations: 最大外迭代次数.
        convergence_threshold: 收敛评分阈值 (0~1).
        settle_time: SLM稳定等待时间 (秒).
        n_sample: 相机每次采样平均帧数.
        output_dir: 输出目录.
        display: 是否启用pygame实时可视化.

    Returns:
        结果字典, 包含 best_phase, best_score, convergence_history 等.
    """
    from ao_shaping.algorithm.gerchberg_saxton import gerchberg_saxton
    from ao_shaping.gui.slm.multi_slm_controller import (
        build_square_target_amplitude,
        compute_square_side,
        measure_spot_diameter_cam,
    )

    width = slm.Panel_Res[0]
    height = slm.Panel_Res[1]
    d_slm = float(slm.Pitch_um) * 1e-6  # um -> m
    p_cam_eff = d_slm if p_cam is None else float(p_cam)
    wavelength = float(slm.wavelength) if slm.wavelength is not None else 1064.0
    wavelength_m = wavelength * 1e-9

    # SLM内存槽轮换 (禁止连续使用同一槽位)
    slot_cycle = itertools.cycle([3, 4, 5])

    # 可选pygame实时显示 (需上下文管理器管理窗口生命周期)
    display_stack = contextlib.ExitStack()
    display_ctx: _GSDisplay | None = None
    if display:
        try:
            display_ctx = _GSDisplay()
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
        except Exception as e:  # 显示失败不应中断实验
            logger.warning("pygame渲染失败: {}", e)

    # 初始光源振幅: 均匀 (SLM分辨率)
    source_amplitude = np.ones((height, width), dtype=np.float64)

    convergence_history: list[dict] = []
    all_images: list[np.ndarray] = []
    all_phases: list[np.ndarray] = []
    all_metrics: list[dict] = []

    best_phase = None
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
    side = compute_square_side(spot_d, factor=gs_factor, p_cam=p_cam_eff, d_slm=d_slm)
    max_side = min(height, width) - 8
    if side > max_side:
        logger.warning("方形边长 {}px 超出SLM网格, 钳制为 {}px", side, max_side)
        side = max_side
    logger.info("测得光斑直径 {:.1f}px, 方形边长 {}px", spot_d, side)

    # 迭代记录器: 每次迭代保存相位/图片/指标为 pkl record (:ref:`axis_beam_runner` 保存 record 模式)
    recorder = Recorder(mark="quality_score", mode="max")

    for outer_iter in range(outer_iterations):
        if not _running:
            logger.info("检测到停止信号, 提前终止 (iter={})", outer_iter + 1)
            break

        logger.info("\n" + "=" * 60)
        logger.info("外迭代 {}/{}", outer_iter + 1, outer_iterations)

        # --- 1. 构建方形目标 ---
        target_amplitude = build_square_target_amplitude(height, width, side)

        # --- 2. 运行 GS ---
        logger.info("运行GS ({}/{} 迭代)...", outer_iter + 1, outer_iterations)

        def _gs_progress(iteration: int, error: float) -> None:
            """GS内迭代进度回调 (更新pygame误差曲线)."""
            if display_ctx is None:
                return
            try:
                display_ctx.update_error(iteration, error)
            except Exception:  # 显示失败不应中断GS
                pass

        result = gerchberg_saxton(
            source_amplitude=source_amplitude,
            target_amplitude=target_amplitude,
            iterations=gs_iterations,
            cell_spacing=d_slm,
            distance=focal_length_m,
            wavelength=wavelength_m,
            progress_callback=_gs_progress,
        )

        # --- 3. 转换相位为灰度并下发 ---
        phase_gray = slm.create_phase_from_array(result.phase)
        slot = next(slot_cycle)
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
        metrics["gs_final_error"] = float(result.error_history[-1]) if result.error_history else None

        convergence_history.append(metrics)
        all_images.append(new_image)
        all_phases.append(phase_gray)
        all_metrics.append(metrics)

        logger.info(
            "iter {}: 评分={:.4f} AR={:.3f} squareness={:.3f} CV={:.3f} EE={:.3f}",
            outer_iter + 1,
            score,
            metrics["aspect_ratio"],
            metrics["squareness"],
            metrics["uniformity_cv"],
            metrics["encircled_energy"],
        )

        # --- 6.5 更新pygame显示面板 ---
        if display_ctx is not None:
            try:
                display_ctx.update_phase(result.phase)
                display_ctx.update_target(target_amplitude)
                display_ctx.update_image(new_image)
                display_ctx.update_status(score, int(side), float(spot_d))
                # 单独绘制一次相位面板 (显示当前GS结果)
                display_ctx.render_static()
            except Exception as e:
                logger.warning("pygame显示更新失败: {}", e)

        # --- 7. 保存本次迭代产物 ---
        iter_dir = output_dir / f"iter_{outer_iter + 1:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        np.save(iter_dir / "captured_image.npy", new_image)
        np.save(iter_dir / "gs_phase.npy", phase_gray)
        with (iter_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        # 追加本次迭代 record (相位/图片/目标/指标) 并增量保存 pkl, Ctrl+C 也不丢数据
        recorder.append(
            {
                "quality_score": float(score),
                "iteration": outer_iter + 1,
                "side": int(side),
                "spot_d": float(spot_d),
                "center": [float(cx), float(cy)],
                "gs_final_error": (
                    float(result.error_history[-1]) if result.error_history else None
                ),
                "aspect_ratio": float(metrics["aspect_ratio"]),
                "squareness": float(metrics["squareness"]),
                "uniformity_cv": float(metrics["uniformity_cv"]),
                "encircled_energy": float(metrics["encircled_energy"]),
                "phase": np.asarray(result.phase, dtype=np.float64),
                "phase_gray": np.asarray(phase_gray, dtype=np.uint16),
                "image": np.asarray(new_image, dtype=np.float64),
                "target": np.asarray(target_amplitude, dtype=np.float64),
            }
        )
        recorder.dataframe.to_pickle(output_dir / "gs_square_records.pkl", compression="zip")

        # --- 8. 跟踪最优 ---
        if score > best_score:
            best_score = float(score)
            best_phase = phase_gray
            best_image = new_image
            best_iter = outer_iter + 1
            logger.info("更新最优: iter={}, score={:.4f}", best_iter, best_score)

        # --- 9. 收敛判断 ---
        if score >= convergence_threshold:
            logger.info("达到收敛阈值 {:.3f}, 停止迭代", convergence_threshold)
            break

        # 更新光源振幅为当前实测光强 (闭环核心: 使用实测作为下一次GS源)
        # 相机帧需先重采样到SLM网格尺寸, 与GS目标网格一致
        source_amplitude = _resample_to_grid(new_image, (height, width))
        source_amplitude = source_amplitude / (np.max(source_amplitude) + 1e-10)

    # 汇总结果
    result = {
        "best_phase": best_phase,
        "best_score": float(best_score) if best_score >= 0 else 0.0,
        "best_iter": best_iter,
        "best_image": best_image,
        "convergence_history": convergence_history,
        "all_images": all_images,
        "all_phases": all_phases,
        "all_metrics": all_metrics,
        "side": int(side),
        "spot_diameter_px": float(spot_d),
        "resolution": [width, height],
    }
    display_stack.close()
    return result


def _save_results(result: dict, output_dir: Path, params: dict) -> None:
    """保存最终结果 (最优相位, 收敛历史, 元数据)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if result["best_phase"] is not None:
        np.save(output_dir / "best_phase.npy", result["best_phase"])
        if result["best_image"] is not None:
            np.save(output_dir / "best_image.npy", result["best_image"])

    with (output_dir / "convergence_history.json").open("w", encoding="utf-8") as f:
        json.dump(result["convergence_history"], f, ensure_ascii=False, indent=2)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_score": result["best_score"],
        "best_iteration": result["best_iter"],
        "side": result["side"],
        "spot_diameter_px": result["spot_diameter_px"],
        "resolution": result["resolution"],
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
@click.option("--exposure-ms", default=50.0, type=float, help="相机曝光时间 ms (default: 50)")
# SLM
@click.option("--slm-number", default=1, type=int, help="SLM设备编号 1-8 (default: 1)")
@click.option("--slm-wavelength", default=1064, type=int, help="SLM工作波长 nm (default: 1064)")
# GS参数
@click.option("--gs-iterations", "-i", default=100, type=int, help="GS内迭代次数 (default: 100)")
@click.option("--gs-factor", default=1.5, type=float, help="方形/光斑尺寸因子 (default: 1.5)")
@click.option("--focal-length", default=0.1, type=float, help="焦距/传播距离 m (default: 0.1)")
@click.option("--gs-energy", default=0.90, type=float, help="光斑测量环围能量 (default: 0.90)")
@click.option(
    "--p-cam",
    default=None,
    type=float,
    help="相机像素间距 m (default: 使用SLM间距)",
)
# 闭环参数
@click.option("--outer-iterations", "-n", default=10, type=int, help="最大外迭代次数 (default: 10)")
@click.option(
    "--convergence-threshold",
    default=0.95,
    type=float,
    help="收敛评分阈值 0~1 (default: 0.95)",
)
@click.option("--settle-time", default=0.5, type=float, help="SLM稳定等待时间 s (default: 0.5)")
@click.option("--n-sample", default=3, type=int, help="相机每次采样平均帧数 (default: 3)")
# 输出
@click.option("-o", "--output", default="data/gs_square", help="输出目录 (default: data/gs_square)")
@click.option("--cam-bit-depth", default=8, type=click.IntRange(8, 16), help="MiiCam输出位深 (default: 8)")
@click.option(
    "--display/--no-display",
    default=False,
    help="启用pygame实时可视化GS迭代过程 (default: False)",
)
def run(
    camera_type: str,
    cam_id: int | None,
    exposure_ms: float,
    slm_number: int,
    slm_wavelength: int,
    gs_iterations: int,
    gs_factor: float,
    focal_length: float,
    gs_energy: float,
    p_cam: float | None,
    outer_iterations: int,
    convergence_threshold: float,
    settle_time: float,
    n_sample: int,
    output: str,
    cam_bit_depth: int,
    display: bool,
) -> None:
    """GS闭环光束整形优化器

    利用GS算法 + CCD反馈迭代优化SLM相位, 将远场光斑整形为方形.
    """
    global _running
    _running = True
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # 相机ID默认值
    if cam_id is None:
        cam_id = int(os.environ.get("FAR_CAM_ID", "0"))

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
            "slm_number": slm_number,
            "slm_wavelength": slm_wavelength,
            "gs_iterations": gs_iterations,
            "gs_factor": gs_factor,
            "focal_length_m": focal_length,
            "gs_energy": gs_energy,
            "p_cam": p_cam,
            "outer_iterations": outer_iterations,
            "convergence_threshold": convergence_threshold,
            "settle_time": settle_time,
            "n_sample": n_sample,
            "display": display,
        }

        # --- 执行闭环 ---
        result = _run_closed_loop(
            slm,
            camera,
            gs_iterations=gs_iterations,
            gs_factor=gs_factor,
            focal_length_m=focal_length,
            gs_energy=gs_energy,
            p_cam=p_cam,
            outer_iterations=outer_iterations,
            convergence_threshold=convergence_threshold,
            settle_time=settle_time,
            n_sample=n_sample,
            output_dir=output_dir,
            display=display,
        )

        # --- 保存结果 ---
        _save_results(result, output_dir, params)

        n_iters = len(result["convergence_history"])
        logger.info("闭环完成: 运行 {} 次迭代, 最优评分 {:.4f}", n_iters, result["best_score"])

    except Exception as e:
        logger.error("GS square runner failed: {}", e)
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
        logger.exception("GS square runner failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
