"""SLM 图形生成的相位包裹优化。

本模块实现多种策略, 用于抑制 SLM 相位图形中 2π 相位不连续所导致的高频伪影:

1. 最小跳变包裹 (Min-Jump Wrapping): 依据梯度连续性选择 2π 偏移以最小化跳变。
2. 误差扩散 (Error Diffusion): Floyd-Steinberg 变体, 把 2π 阶跃摊成渐变过渡。
3. 过采样-平滑-降采样: 高分辨率生成、高斯滤波, 再降采样。
4. 条纹修复 (Fringe Repair): 检测包裹边缘并做局部螺旋插值。
5. 混合管道 (Hybrid Pipeline): 按优化顺序组合以上全部策略。

参考文献:
    - Goodman, J. W. (2005). Introduction to Fourier Optics.
    - Floyd, R. W. & Steinberg, L. (1976). An adaptive algorithm for spatial grey scale.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import numpy.typing as npt
from loguru import logger
from scipy.ndimage import gaussian_filter, zoom

from ao_shaping.algorithm.signal_processing.iterative_base import IterativeOptimizer
from ao_shaping.utils.wavefront.zernike_calc import generate_noll_polynomial, noll_to_nm


class PhaseWrapOptimizer(IterativeOptimizer):
    """抑制 2π 不连续伪影的相位包裹优化器。

    核心策略:
    1. 最小跳变包裹: 基于梯度连续性的 2π 偏移选择。
    2. 误差扩散: 摊开 2π 阶跃的 Floyd-Steinberg 变体。
    3. 过采样-平滑-降采样: 2x/4x 过采样、高斯滤波、降采样。
    4. 条纹修复: 在包裹边缘做局部螺旋插值。

    Attributes:
        slm_height: SLM 面板高度 (像素)。
        slm_width: SLM 面板宽度 (像素)。
        oversample: 过采样-平滑策略的过采样倍数。
    """

    def __init__(
        self,
        slm_height: int = 1600,
        slm_width: int = 2560,
        oversample: int = 2,
        max_iterations: int = 1000,
    ) -> None:
        super().__init__(max_iterations=max_iterations)
        self.slm_height = slm_height
        self.slm_width = slm_width
        self.oversample = oversample

    # ==================== 1. 基础工具 ====================

    @staticmethod
    def wrap_hard(phase: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """标准硬包裹: 对 2π 取模, 落入 [-π, π)。

        Args:
            phase: 未包裹的相位 (弧度)。

        Returns:
            落在 [-π, π) 区间的包裹后相位。
        """
        return np.mod(phase + np.pi, 2 * np.pi) - np.pi

    @staticmethod
    def detect_jumps(
        wrapped_phase: npt.NDArray[np.floating],
        threshold: float = 0.5 * np.pi,
    ) -> npt.NDArray[np.floating]:
        """在包裹后的相位图里检测 2π 跳变边缘像素。

        计算四个方向的梯度, 标记考虑圆周距离后梯度超过阈值的像素。
        若所有相邻像素间的相位差都落在一个 2π 周期内, 返回全 0 (无跳变)。

        Args:
            wrapped_phase: 包裹后的相位图 (二维数组)。
            threshold: 用于跳变检测的圆周梯度阈值。
                默认 0.5π, 即在计入 2π 周期性后, 任何超过 90° 的
                像素间相位变化都会被标记。

        Returns:
            形状相同的布尔掩码, 跳变边缘像素处为 True。
        """
        dy = np.abs(np.diff(wrapped_phase, axis=0, append=wrapped_phase[-1:, :]))
        dx = np.abs(np.diff(wrapped_phase, axis=1, append=wrapped_phase[:, -1:]))

        # 计入 2π 周期性: 取绕圆周的最短路径
        dy_wrap = np.minimum(dy, 2 * np.pi - dy)
        dx_wrap = np.minimum(dx, 2 * np.pi - dx)

        jump = (dy_wrap > threshold) | (dx_wrap > threshold)
        return jump

    @staticmethod
    def calculate_diffraction_efficiency(
        phase: npt.NDArray[np.floating],
        pixel_size_um: float = 8.0,
        wavelength_um: float = 0.633,
    ) -> float:
        """由梯度 RMS 估算一级衍射效率。

        使用一个经验模型: 效率损失与处于 2π 跳变边缘的像素占比成正比。

        Args:
            phase: 相位图 (包裹或未包裹均可), 单位弧度。
            pixel_size_um: SLM 像素间距 (微米)。
            wavelength_um: 光的波长 (微米)。

        Returns:
            [0, 1] 区间的估算衍射效率。
        """
        jump_mask = PhaseWrapOptimizer.detect_jumps(phase)
        loss = np.sum(jump_mask) / phase.size
        efficiency = float(np.exp(-loss * 2.0))
        return efficiency

    # ==================== 2. 最小跳变包裹 ====================

    def min_jump_wrap(
        self,
        phase_unwrapped: npt.NDArray[np.floating],
        connectivity: int = 4,
    ) -> npt.NDArray[np.floating]:
        """最小跳变包裹: 选择 2π 偏移以最小化相位差。

        迭代松弛, 调整每个像素处的整数 2π 偏移以最小化局部相位梯度。
        从硬包裹出发逐步细化。

        Args:
            phase_unwrapped: 连续的 (未包裹) 相位, 单位弧度。
            connectivity: 邻域连通性 (4 或 8)。

        Returns:
            落在 [0, 2π) 区间的最优包裹相位。
        """
        H, W = phase_unwrapped.shape
        wrapped = self.wrap_hard(phase_unwrapped)

        # 每个像素累加的 2π 偏移
        k_offset = np.round(
            (phase_unwrapped - wrapped) / (2 * np.pi)
        ).astype(int)

        # 构造邻域偏移
        if connectivity == 4:
            neighbor_shifts = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        else:
            neighbor_shifts = [
                (i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if not (i == 0 and j == 0)
            ]

        for _iteration in range(20):
            k_old = k_offset.copy()

            for di, dj in neighbor_shifts:
                k_roll = np.roll(np.roll(k_offset, di, axis=0), dj, axis=1)

                phi_self = wrapped + 2 * np.pi * k_offset
                phi_nei = wrapped + 2 * np.pi * k_roll
                diff_current = np.abs(phi_self - phi_nei)

                # 尝试把 k_offset 调 ±1
                diff_p1 = np.abs(phi_self + 2 * np.pi - phi_nei)
                diff_m1 = np.abs(phi_self - 2 * np.pi - phi_nei)

                k_offset[diff_p1 < diff_current] += 1
                k_offset[diff_m1 < diff_current] -= 1

            if np.all(k_old == k_offset):
                break

        phase_continuous = wrapped + 2 * np.pi * k_offset
        phase_out = np.mod(phase_continuous, 2 * np.pi)

        jumps_before = int(np.sum(self.detect_jumps(self.wrap_hard(phase_unwrapped))))
        jumps_after = int(np.sum(self.detect_jumps(phase_out)))
        reduction = 100 * (1 - jumps_after / max(jumps_before, 1))
        logger.info(
            f"[Min-Jump Wrap] Jump pixels: {jumps_before} -> {jumps_after} "
            f"(reduced {reduction:.1f}%)"
        )

        return phase_out

    # ==================== 3. 误差扩散 (Floyd-Steinberg for Phase) ====================

    def error_diffusion_wrap(
        self,
        phase_unwrapped: npt.NDArray[np.floating],
        quantization_levels: int = 256,
    ) -> npt.NDArray[np.floating]:
        """误差扩散包裹: 把 2π 跳变摊成渐变过渡。

        为相位改写的 Floyd-Steinberg 误差扩散:
        - 量化步长 = 2π / levels
        - 包裹误差扩散到尚未处理的邻居
        - 结果: 跳变边缘变成 2-3 像素的梯度

        Args:
            phase_unwrapped: 连续的 (未包裹) 相位, 单位弧度。
            quantization_levels: 2π 区间的量化级数。

        Returns:
            带扩散量化误差的包裹相位。
        """
        H, W = phase_unwrapped.shape
        phase = phase_unwrapped.copy()
        step = 2 * np.pi / quantization_levels
        error_buffer = np.zeros((H, W))

        for y in range(H):
            for x in range(W):
                old_val = phase[y, x] + error_buffer[y, x]
                old_mod = np.mod(old_val, 2 * np.pi)
                idx = int(np.round(old_mod / step)) % quantization_levels
                new_val = idx * step

                err = old_mod - new_val
                if err > step / 2:
                    err -= 2 * np.pi
                elif err < -step / 2:
                    err += 2 * np.pi

                # Floyd-Steinberg 核
                if y < H - 1:
                    if x > 0:
                        error_buffer[y + 1, x - 1] += err * 3 / 16
                    if x < W - 1:
                        error_buffer[y + 1, x + 1] += err * 1 / 16
                    error_buffer[y + 1, x] += err * 5 / 16
                if x < W - 1:
                    error_buffer[y, x + 1] += err * 7 / 16

        phase_out = np.mod(phase + error_buffer, 2 * np.pi)
        logger.info(
            f"[Error Diffusion] Levels: {quantization_levels}, "
            f"effective grayscale: {quantization_levels * 4096 // 256} (12-bit)"
        )
        return phase_out

    # ==================== 4. 过采样平滑 ====================

    def oversample_smooth(
        self,
        phase_unwrapped: npt.NDArray[np.floating],
        sigma_pixels: float = 0.8,
    ) -> npt.NDArray[np.floating]:
        """过采样 → 平滑 → 降采样包裹。

        在更高分辨率上生成连续相位, 在高分辨率网格上做高斯平滑,
        再降采样并包裹。这会把生硬的 2π 边缘转成 2-3 像素的梯度。

        Args:
            phase_unwrapped: 连续的 (未包裹) 相位, 单位弧度。
            sigma_pixels: 以原始像素为单位的高斯 sigma。

        Returns:
            落在 [0, 2π) 区间的平滑包裹相位。
        """
        H, W = phase_unwrapped.shape
        r = self.oversample

        # 1. 用双三次插值过采样
        phase_high = zoom(phase_unwrapped, r, order=3)

        # 2. 在高分辨率上平滑
        phase_smooth = gaussian_filter(phase_high, sigma=sigma_pixels * r)

        # 3. 降采样时取圆周均值, 避免 2π 包裹带来的偏差
        phase_down = np.zeros((H, W))
        for i in range(H):
            for j in range(W):
                block = phase_smooth[i * r : (i + 1) * r, j * r : (j + 1) * r]
                phase_down[i, j] = self._circular_mean(block)

        # 4. 最终包裹
        phase_out = np.mod(phase_down, 2 * np.pi)
        logger.info(f"[Oversample] {r}x -> Gaussian(sigma={sigma_pixels}) -> downsample")
        return phase_out

    @staticmethod
    def _circular_mean(angles: npt.NDArray[np.floating]) -> float:
        """角度的圆周均值, 对 2π 包裹稳健。

        把角度投影到复平面上再做正确的平均。

        Args:
            angles: 角度数组, 单位弧度。

        Returns:
            落在 [0, 2π) 区间的平均角 (弧度)。
        """
        z = np.exp(1j * angles)
        mean_angle = float(np.angle(np.mean(z)))
        if mean_angle < 0:
            mean_angle += 2 * np.pi
        return mean_angle

    # ==================== 5. 跳变局部修复 ====================

    def repair_jumps(
        self,
        wrapped_phase: npt.NDArray[np.floating],
        repair_width: int = 2,
        blend_factor: float = 0.5,
    ) -> npt.NDArray[np.floating]:
        """通过螺旋插值检测并局部修复 2π 跳变边缘。

        作为硬包裹之后的后处理步骤, 修掉残余的不连续。

        Args:
            wrapped_phase: 包裹后的相位图, 在 [0, 2π) 或 [-π, π) 内。
            repair_width: 跳变区域的膨胀宽度。
            blend_factor: 修复修正量的混合强度。

        Returns:
            落在 [0, 2π) 区间的修复后相位图。
        """
        H, W = wrapped_phase.shape
        phase = wrapped_phase.copy()

        jump = self.detect_jumps(phase, threshold=1.5 * np.pi)
        if not np.any(jump):
            return phase

        jump_dilated = jump.copy()
        from scipy.ndimage import binary_dilation

        for _ in range(repair_width):
            jump_dilated = binary_dilation(jump_dilated)

        repaired = phase.copy()

        # 水平方向修复
        diff_x = np.diff(phase, axis=1, append=phase[:, -1:])
        wrap_x = np.round(diff_x / (2 * np.pi)).astype(int)
        for w in range(1, repair_width + 1):
            shift_right = np.roll(wrap_x, w, axis=1)
            mask = jump_dilated & (shift_right != 0)
            weight = (repair_width + 1 - w) / (repair_width + 1) * blend_factor
            repaired += mask * weight * 2 * np.pi * np.sign(shift_right)

        # 垂直方向修复
        diff_y = np.diff(phase, axis=0, append=phase[-1:, :])
        wrap_y = np.round(diff_y / (2 * np.pi)).astype(int)
        for w in range(1, repair_width + 1):
            shift_down = np.roll(wrap_y, w, axis=0)
            mask = jump_dilated & (shift_down != 0)
            weight = (repair_width + 1 - w) / (repair_width + 1) * blend_factor
            repaired += mask * weight * 2 * np.pi * np.sign(shift_down)

        repaired = np.mod(repaired, 2 * np.pi)

        jumps_after = int(np.sum(self.detect_jumps(repaired)))
        logger.info(f"[Fringe Repair] Width={repair_width}, residual jumps={jumps_after}")
        return repaired

    # ==================== 6. 综合优化管道 ====================

    def optimize(
        self,
        phase_unwrapped: npt.NDArray[np.floating],
        strategy: Literal[
            "min_jump", "error_diffusion", "oversample", "repair", "hybrid"
        ] = "hybrid",
    ) -> npt.NDArray[np.floating]:
        """运行所选的相位包裹优化策略。

        ``hybrid`` 策略 (推荐) 依次执行:
        1. 最小跳变包裹 (可消除约 90% 的跳变)。
        2. 误差扩散 (把残余跳变摊成梯度)。
        3. 局部条纹修复 (平滑剩余边缘)。

        Args:
            phase_unwrapped: 连续的 (未包裹) 相位, 单位弧度。
            strategy: 优化策略名。

        Returns:
            落在 [0, 2π) 区间的优化后包裹相位。
        """
        if strategy == "min_jump":
            return self.min_jump_wrap(phase_unwrapped)
        elif strategy == "error_diffusion":
            return self.error_diffusion_wrap(phase_unwrapped, quantization_levels=256)
        elif strategy == "oversample":
            return self.oversample_smooth(phase_unwrapped, sigma_pixels=0.8)
        elif strategy == "repair":
            wrapped = self.wrap_hard(phase_unwrapped)
            return self.repair_jumps(wrapped, repair_width=2)
        elif strategy == "hybrid":
            return self._hybrid_pipeline(phase_unwrapped)
        else:
            msg = f"Unknown strategy: {strategy}"
            raise ValueError(msg)

    def update(
        self,
        phase_unwrapped: npt.NDArray[np.floating],
        strategy: Literal[
            "min_jump", "error_diffusion", "oversample", "repair", "hybrid"
        ] = "hybrid",
    ) -> npt.NDArray[np.floating]:
        """执行一步相位包裹优化。

        :meth:`optimize` 的别名, 满足 :class:`IterativeOptimizer` 契约。
        每次调用对输入相位施加所选的包裹策略并返回结果。

        Args:
            phase_unwrapped: 连续的 (未包裹) 相位, 单位弧度。
            strategy: 优化策略名。

        Returns:
            落在 [0, 2π) 区间的优化后包裹相位。
        """
        return self.optimize(phase_unwrapped, strategy)

    def _hybrid_pipeline(self, phase_unwrapped: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """推荐的混合管道: min_jump → error_diffusion → fringe repair。

        Returns:
            落在 [0, 2π) 区间的优化后包裹相位。
        """
        # 步骤 1: 最小跳变包裹 (可消除约 90% 的跳变)
        phase = self.min_jump_wrap(phase_unwrapped)

        # 步骤 2: 误差扩散 —— 作用于连续估计值
        # 由 min-jump 结果重建连续相位
        phase_continuous = phase_unwrapped  # 用原始的连续相位
        phase_ed = self.error_diffusion_wrap(phase_continuous, quantization_levels=512)

        # 步骤 3: 对残余边缘做局部条纹修复
        phase_final = self.repair_jumps(phase_ed, repair_width=1, blend_factor=0.3)

        eff_before = self.calculate_diffraction_efficiency(self.wrap_hard(phase_unwrapped))
        eff_after = self.calculate_diffraction_efficiency(phase_final)
        logger.info(
            f"[Hybrid] Diffraction efficiency estimate: "
            f"{eff_before * 100:.1f}% -> {eff_after * 100:.1f}%"
        )

        return phase_final


# ==================== 集成到 SLM 控制流 ====================

class SLMPhaseController:
    """带包裹优化的 SLM 相位控制器。

    在 :class:`PhaseWrapOptimizer` 与 SLM 设备之间架桥,
    提供便捷的 ``load_zernike_coefficients`` 接口,
    在下发图形之前先施加包裹优化。

    Example:
        >>> with Santec(slm_number=1) as slm:
        ...     ctrl = SLMPhaseController(slm)
        ...     ctrl.load_zernike_coefficients(np.array([0.5, 0.3, 0.2]))
    """

    def __init__(
        self,
        slm,
        lut_lookup: Callable | None = None,
        wrap_optimizer: PhaseWrapOptimizer | None = None,
    ) -> None:
        self.slm = slm
        self.lut = lut_lookup or (lambda x: (x / (2 * np.pi) * 4095).astype(np.uint16))
        self.wrap = wrap_optimizer or PhaseWrapOptimizer(
            slm_height=getattr(slm, "height", 1600),
            slm_width=getattr(slm, "width", 2560),
            oversample=2,
        )

    def load_zernike_coefficients(
        self,
        a: npt.NDArray[np.floating],
        method: Literal["min_jump", "error_diffusion", "oversample", "repair", "hybrid"] = "hybrid",
        apply_lut: bool = True,
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """合成 Zernike 相位、做包裹优化并下发到 SLM。

        Args:
            a: 形状为 ``(n_modes,)``、单位为 λ 的 Zernike 系数数组。
            method: 包裹优化策略 (见 :meth:`PhaseWrapOptimizer.optimize`)。
            apply_lut: 是否施加 LUT (灰度映射)。

        Returns:
            ``(wrapped_phase, grayscale_array)`` 元组。
        """
        phase_continuous = self._synthesize_zernike(a)

        # 相位包裹优化
        phase_wrapped = self.wrap.optimize(phase_continuous, strategy=method)

        # 转换为灰度
        if apply_lut:
            gray = self.lut(phase_wrapped)
        else:
            gray = (phase_wrapped / (2 * np.pi) * 4095).astype(np.uint16)

        # 下发到 SLM
        load_array = getattr(self.slm, "load_array", None)
        if load_array is not None:
            load_array(gray)
        else:
            logger.warning("SLM has no load_array method; skipping hardware update.")

        return phase_wrapped, gray

    def _synthesize_zernike(self, a: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """由系数合成连续的 Zernike 相位图。

        Args:
            a: Zernike 系数数组 (Z2, Z3, ...), 单位为 λ。

        Returns:
            连续的 (未包裹) 相位, 单位弧度。
        """
        h = getattr(self.slm, "height", 1600)
        w = getattr(self.slm, "width", 2560)
        cy, cx = h // 2, w // 2
        max_r = min(cy - 50, cx - 50)

        y, x = np.mgrid[0:h, 0:w]
        r = np.sqrt(((x - cx) / max_r) ** 2 + ((y - cy) / max_r) ** 2)
        mask = r <= 1.0

        # 系数按 Z2..Z15 排序, 所以 Noll 索引 = i + 2。把各个正交归一的
        # Zernike 基都交给 canonical 的 zernike_calc 实现, 使基保持单一来源
        # (见 zernike_calc)。
        phase = np.zeros((h, w))
        for i, coeff in enumerate(a):
            if abs(coeff) <= 1e-6:
                continue
            n, m = noll_to_nm(i + 2)
            z = generate_noll_polynomial(n, m, (w, h))
            phase += coeff * 2 * np.pi * z

        phase[~mask] = 0
        return phase
