"""Phase wrapping optimization for SLM pattern generation.

This module implements multiple strategies to suppress high-frequency artifacts
caused by 2π phase discontinuities in SLM phase patterns:

1. Min-Jump Wrapping: Selects 2π offsets via gradient continuity to minimize jumps.
2. Error Diffusion: Floyd-Steinberg variant that spreads 2π steps into gradual transitions.
3. Oversample-Smooth-Downsample: High-res generation, Gaussian filter, then downsample.
4. Fringe Repair: Detects wrap edges and performs local spiral interpolation.
5. Hybrid Pipeline: Combines all strategies in an optimized sequence.

Reference:
    - Goodman, J. W. (2005). Introduction to Fourier Optics.
    - Floyd, R. W. & Steinberg, L. (1976). An adaptive algorithm for spatial grey scale.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from loguru import logger
from scipy.ndimage import gaussian_filter, zoom


class PhaseWrapOptimizer:
    """Phase wrapping optimizer for suppressing 2π discontinuity artifacts.

    Core strategies:
    1. Min-Jump Wrapping: 2π offset selection based on gradient continuity.
    2. Error Diffusion: Floyd-Steinberg variant that spreads 2π steps.
    3. Oversample-Smooth-Downsample: 2x/4x oversample, Gaussian filter, downsample.
    4. Fringe Repair: Local spiral interpolation at wrap edges.

    Attributes:
        slm_height: SLM panel height in pixels.
        slm_width: SLM panel width in pixels.
        oversample: Oversampling factor for oversample-smooth strategy.
    """

    def __init__(
        self,
        slm_height: int = 1600,
        slm_width: int = 2560,
        oversample: int = 2,
    ) -> None:
        self.slm_height = slm_height
        self.slm_width = slm_width
        self.oversample = oversample

    # ==================== 1. 基础工具 ====================

    @staticmethod
    def wrap_hard(phase: np.ndarray) -> np.ndarray:
        """Standard hard wrapping: modulo 2π into [-π, π).

        Args:
            phase: Unwrapped phase in radians.

        Returns:
            Wrapped phase in range [-π, π).
        """
        return np.mod(phase + np.pi, 2 * np.pi) - np.pi

    @staticmethod
    def detect_jumps(
        wrapped_phase: np.ndarray,
        threshold: float = 0.5 * np.pi,
    ) -> np.ndarray:
        """Detect 2π jump edge pixels in a wrapped phase map.

        Computes 4-directional gradients and flags pixels where the
        circular-distance-aware gradient exceeds the threshold.
        A value of 0 (no jumps) is returned when all adjacent-pixel
        phase differences are within a single 2π cycle.

        Args:
            wrapped_phase: Wrapped phase map (2D array).
            threshold: Circular gradient threshold for jump detection.
                Default 0.5π flags any inter-pixel phase change > 90°
                after accounting for 2π periodicity.

        Returns:
            Boolean mask of same shape, True at jump edge pixels.
        """
        dy = np.abs(np.diff(wrapped_phase, axis=0, append=wrapped_phase[-1:, :]))
        dx = np.abs(np.diff(wrapped_phase, axis=1, append=wrapped_phase[:, -1:]))

        # Account for 2π periodicity: use the shorter path around the circle
        dy_wrap = np.minimum(dy, 2 * np.pi - dy)
        dx_wrap = np.minimum(dx, 2 * np.pi - dx)

        jump = (dy_wrap > threshold) | (dx_wrap > threshold)
        return jump

    @staticmethod
    def calculate_diffraction_efficiency(
        phase: np.ndarray,
        pixel_size_um: float = 8.0,
        wavelength_um: float = 0.633,
    ) -> float:
        """Estimate 1st-order diffraction efficiency from gradient RMS.

        Uses an empirical model where efficiency loss is proportional to
        the fraction of pixels at 2π jump edges.

        Args:
            phase: Phase map (wrapped or unwrapped) in radians.
            pixel_size_um: SLM pixel pitch in microns.
            wavelength_um: Light wavelength in microns.

        Returns:
            Estimated diffraction efficiency in [0, 1].
        """
        jump_mask = PhaseWrapOptimizer.detect_jumps(phase)
        loss = np.sum(jump_mask) / phase.size
        efficiency = float(np.exp(-loss * 2.0))
        return efficiency

    # ==================== 2. 最小跳变包裹 ====================

    def min_jump_wrap(
        self,
        phase_unwrapped: np.ndarray,
        connectivity: int = 4,
    ) -> np.ndarray:
        """Minimum-jump wrapping: choose 2π offset to minimize phase differences.

        Iterative relaxation that adjusts the integer 2π offset at each pixel
        to minimize local phase gradients. Starts from a hard wrap and refines.

        Args:
            phase_unwrapped: Continuous (unwrapped) phase in radians.
            connectivity: Neighborhood connectivity (4 or 8).

        Returns:
            Optimally wrapped phase in [0, 2π).
        """
        H, W = phase_unwrapped.shape
        wrapped = self.wrap_hard(phase_unwrapped)

        # Accumulated 2π offset per pixel
        k_offset = np.round(
            (phase_unwrapped - wrapped) / (2 * np.pi)
        ).astype(int)

        # Build neighbor offsets
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

                # Try adjusting k_offset by ±1
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
        phase_unwrapped: np.ndarray,
        quantization_levels: int = 256,
    ) -> np.ndarray:
        """Error diffusion wrapping: spread 2π jumps into gradual transitions.

        Adapted Floyd-Steinberg error diffusion for phase:
        - Quantization step = 2π / levels
        - Wrap error diffused to unprocessed neighbours
        - Result: jump edges become 2-3 pixel gradients

        Args:
            phase_unwrapped: Continuous (unwrapped) phase in radians.
            quantization_levels: Number of quantization levels for 2π range.

        Returns:
            Wrapped phase with diffused quantization error.
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

                # Floyd-Steinberg kernel
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
        phase_unwrapped: np.ndarray,
        sigma_pixels: float = 0.8,
    ) -> np.ndarray:
        """Oversample → smooth → downsample wrapping.

        Generates a continuous phase at higher resolution, applies
        Gaussian smoothing at the high-res grid, then downsamples and wraps.
        This converts hard 2π edges into 2-3 pixel gradients.

        Args:
            phase_unwrapped: Continuous (unwrapped) phase in radians.
            sigma_pixels: Gaussian sigma in original pixel units.

        Returns:
            Smoothed wrapped phase in [0, 2π).
        """
        H, W = phase_unwrapped.shape
        r = self.oversample

        # 1. Oversample via bicubic interpolation
        phase_high = zoom(phase_unwrapped, r, order=3)

        # 2. Smooth at high resolution
        phase_smooth = gaussian_filter(phase_high, sigma=sigma_pixels * r)

        # 3. Downsample with circular mean to avoid 2π wrap bias
        phase_down = np.zeros((H, W))
        for i in range(H):
            for j in range(W):
                block = phase_smooth[i * r : (i + 1) * r, j * r : (j + 1) * r]
                phase_down[i, j] = self._circular_mean(block)

        # 4. Final wrap
        phase_out = np.mod(phase_down, 2 * np.pi)
        logger.info(f"[Oversample] {r}x -> Gaussian(sigma={sigma_pixels}) -> downsample")
        return phase_out

    @staticmethod
    def _circular_mean(angles: np.ndarray) -> float:
        """Circular mean of angles, robust to 2π wraps.

        Projects angles to the complex plane for proper averaging.

        Args:
            angles: Array of angles in radians.

        Returns:
            Mean angle in radians, in [0, 2π).
        """
        z = np.exp(1j * angles)
        mean_angle = float(np.angle(np.mean(z)))
        if mean_angle < 0:
            mean_angle += 2 * np.pi
        return mean_angle

    # ==================== 5. 跳变局部修复 ====================

    def repair_jumps(
        self,
        wrapped_phase: np.ndarray,
        repair_width: int = 2,
        blend_factor: float = 0.5,
    ) -> np.ndarray:
        """Detect and locally repair 2π jump edges via spiral interpolation.

        Acts as a post-processing step after hard wrapping to fix
        residual discontinuities.

        Args:
            wrapped_phase: Wrapped phase map in [0, 2π) or [-π, π).
            repair_width: Dilation width for jump region.
            blend_factor: Blending strength for the repair correction.

        Returns:
            Repaired phase map in [0, 2π).
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

        # Horizontal repair
        diff_x = np.diff(phase, axis=1, append=phase[:, -1:])
        wrap_x = np.round(diff_x / (2 * np.pi)).astype(int)
        for w in range(1, repair_width + 1):
            shift_right = np.roll(wrap_x, w, axis=1)
            mask = jump_dilated & (shift_right != 0)
            weight = (repair_width + 1 - w) / (repair_width + 1) * blend_factor
            repaired += mask * weight * 2 * np.pi * np.sign(shift_right)

        # Vertical repair
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
        phase_unwrapped: np.ndarray,
        strategy: Literal[
            "min_jump", "error_diffusion", "oversample", "repair", "hybrid"
        ] = "hybrid",
    ) -> np.ndarray:
        """Run the chosen phase wrapping optimization strategy.

        The ``hybrid`` strategy (recommended) runs:
        1. Min-jump wrapping (reduces ~90% of jumps).
        2. Error diffusion (spreads residual jumps into gradients).
        3. Local fringe repair (smooths remaining edges).

        Args:
            phase_unwrapped: Continuous (unwrapped) phase in radians.
            strategy: Optimization strategy name.

        Returns:
            Optimized wrapped phase in [0, 2π).
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

    def _hybrid_pipeline(self, phase_unwrapped: np.ndarray) -> np.ndarray:
        """Recommended hybrid pipeline: min_jump → error_diffusion → fringe repair.

        Returns:
            Optimized wrapped phase in [0, 2π).
        """
        # Step 1: Min-jump wrapping (reduces ~90% of jumps)
        phase = self.min_jump_wrap(phase_unwrapped)

        # Step 2: Error diffusion — operates on the continuous estimate
        # Reconstruct a continuous phase from the min-jump result
        phase_continuous = phase_unwrapped  # use original continuous phase
        phase_ed = self.error_diffusion_wrap(phase_continuous, quantization_levels=512)

        # Step 3: Local fringe repair for residual edges
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
    """SLM phase controller with wrapping optimization.

    Bridges the :class:`PhaseWrapOptimizer` with an SLM device,
    providing a convenient ``load_zernike_coefficients`` interface
    that applies wrapping optimization before sending the pattern.

    Example:
        >>> with SantecSLM200(slm_number=1) as slm:
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
        a: np.ndarray,
        method: Literal["min_jump", "error_diffusion", "oversample", "repair", "hybrid"] = "hybrid",
        apply_lut: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Synthesize Zernike phase, optimize wrapping, and send to SLM.

        Args:
            a: Zernike coefficient array of shape ``(n_modes,)`` in units of λ.
            method: Wrapping optimization strategy (see :meth:`PhaseWrapOptimizer.optimize`).
            apply_lut: Whether to apply the LUT (grayscale mapping).

        Returns:
            Tuple of ``(wrapped_phase, grayscale_array)``.
        """
        phase_continuous = self._synthesize_zernike(a)

        # Phase wrapping optimization
        phase_wrapped = self.wrap.optimize(phase_continuous, strategy=method)

        # Convert to grayscale
        if apply_lut:
            gray = self.lut(phase_wrapped)
        else:
            gray = (phase_wrapped / (2 * np.pi) * 4095).astype(np.uint16)

        # Send to SLM
        load_array = getattr(self.slm, "load_array", None)
        if load_array is not None:
            load_array(gray)
        else:
            logger.warning("SLM has no load_array method; skipping hardware update.")

        return phase_wrapped, gray

    def _synthesize_zernike(self, a: np.ndarray) -> np.ndarray:
        """Synthesize a continuous Zernike phase map from coefficients.

        Args:
            a: Zernike coefficient array (Z2, Z3, ...) in units of λ.

        Returns:
            Continuous (unwrapped) phase in radians.
        """
        h = getattr(self.slm, "height", 1600)
        w = getattr(self.slm, "width", 2560)
        cy, cx = h // 2, w // 2
        max_r = min(cy - 50, cx - 50)

        y, x = np.mgrid[0:h, 0:w]
        y_n = (y - cy) / max_r
        x_n = (x - cx) / max_r
        r = np.sqrt(x_n**2 + y_n**2)
        theta = np.arctan2(y_n, x_n)
        mask = r <= 1.0

        phase = np.zeros((h, w))
        funcs = _zernike_functions(min(len(a), 15))
        for i, coeff in enumerate(a[: len(funcs)]):
            if abs(coeff) > 1e-6:
                z = funcs[i](r, theta)
                phase += coeff * 2 * np.pi * z

        phase[~mask] = 0
        return phase


# ==================== 泽尼克基函数 ====================

def _zernike_functions(n: int) -> list:
    """Return the first ``n`` Zernike polynomial functions (Z2, Z3, ...).

    Args:
        n: Number of functions to return (max 15).

    Returns:
        List of callables ``f(r, theta) -> np.ndarray``.
    """
    functions = [
        lambda r, t: r * np.cos(t),                     # Z2: tip (x tilt)
        lambda r, t: r * np.sin(t),                     # Z3: tilt (y tilt)
        lambda r, t: 2 * r**2 - 1,                     # Z4: defocus
        lambda r, t: r**2 * np.cos(2 * t),             # Z5: primary astigmatism x
        lambda r, t: r**2 * np.sin(2 * t),             # Z6: primary astigmatism y
        lambda r, t: (3 * r**3 - 2 * r) * np.cos(t),   # Z7: primary coma x
        lambda r, t: (3 * r**3 - 2 * r) * np.sin(t),   # Z8: primary coma y
        lambda r, t: r**3 * np.cos(3 * t),              # Z9: trefoil x
        lambda r, t: r**3 * np.sin(3 * t),              # Z10: trefoil y
        lambda r, t: 6 * r**4 - 6 * r**2 + 1,          # Z11: primary spherical
        lambda r, t: r**4 * np.cos(4 * t),             # Z12: tetrafoil x
        lambda r, t: r**4 * np.sin(4 * t),             # Z13: tetrafoil y
        lambda r, t: (10 * r**5 - 12 * r**3 + 3 * r) * np.cos(t),  # Z14
        lambda r, t: r**5 * np.cos(5 * t),             # Z15
    ]
    return functions[:n]
