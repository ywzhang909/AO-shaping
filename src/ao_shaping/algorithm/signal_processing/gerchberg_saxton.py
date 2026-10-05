"""用于全息图生成的 Gerchberg-Saxton 算法。

本模块实现经典的 Gerchberg-Saxton 相位恢复算法, 并采用角谱法 (ASM)
完成光的波前传播。

算法迭代地约束源面 (SLM) 与目标面 (远场) 的振幅, 从而算出 SLM 的最优相位图形。

参考文献:
    - Gerchberg, R. W., & Saxton, W. O. (1972). A practical algorithm for the
      determination of phase from image and diffraction plane pictures.
      Optik, 35, 237-246.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import numpy.typing as npt
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from loguru import logger


@dataclass
class GSResult:
    """Gerchberg-Saxton 算法的结果容器。

    Attributes:
        phase: 为 SLM 算出的相位图形 (弧度, 0-2π)
        amplitude: 目标面上的最终振幅
        error_history: 每次迭代对应的误差值列表
        iterations: 实际执行的迭代次数
        converged: 算法是否收敛
    """

    phase: npt.NDArray[np.floating]
    amplitude: npt.NDArray[np.floating]
    error_history: list[float]
    iterations: int
    converged: bool


@lru_cache(maxsize=32)
def _compute_propagator(
    ny: int,
    nx: int,
    dx: float,
    z: float,
    wavelength: float,
) -> npt.NDArray[np.complexfloating]:
    """预计算 ASM 传播子 (已 ifftshift 对齐 fft2 输出), 跨调用复用.

    传播子只依赖 (网格形状, 像素间距, 距离, 波长), 与输入场无关. GS 迭代中
    正/反向传播反复调用 ``angular_spectrum_propagate``, 缓存避免每迭代重复
    构建 meshgrid 与 exp 传播因子 (2.3M 像素网格上的纯标量浪费).
    """
    k = 2 * np.pi / wavelength
    fx = np.fft.fftfreq(nx, dx)
    fy = np.fft.fftfreq(ny, dx)
    FX, FY = np.meshgrid(fx, fy)
    kx = 2 * np.pi * FX
    ky = 2 * np.pi * FY
    kz_squared = k**2 - kx**2 - ky**2
    kz = np.sqrt(np.maximum(kz_squared, 0))
    H = np.exp(1j * kz * z)
    H[kz_squared < 0] = 0  # 倏逝波
    return ifftshift(H)


def angular_spectrum_propagate(
    field: npt.NDArray,
    dx: float,
    z: float,
    wavelength: float,
) -> npt.NDArray[np.complexfloating]:
    """用角谱法 (ASM) 传播光场。

    角谱法借助傅里叶光学把复光场从一个平面传播到另一个平面。
    它对近场与远场传播都是精确的。

    Args:
        field: 复光场数组 (二维 numpy 数组)
        dx: 像素间距 (米)
        z: 传播距离 (米, 正=正向, 负=反向)
        wavelength: 光的波长 (米)

    Returns:
        传播后的复光场 (形状与输入相同)

    Example:
        >>> # 正向传播 10cm
        >>> propagated = angular_spectrum_propagate(field, dx=8e-6, z=0.1, wavelength=633e-9)
    """
    if field.ndim != 2:
        raise ValueError(f"Field must be 2D array, got {field.ndim}D")

    # FFT → 乘以传播子 → IFFT
    F = fft2(field)
    H = _compute_propagator(field.shape[0], field.shape[1], dx, z, wavelength)
    F_propagated = F * H

    return ifft2(F_propagated)


def gerchberg_saxton(
    source_amplitude: npt.NDArray[np.floating],
    target_amplitude: npt.NDArray[np.floating],
    iterations: int = 50,
    cell_spacing: float = 8e-6,
    distance: float = 0.1,
    wavelength: float = 1064e-9,
    error_threshold: float | None = None,
    progress_callback: Callable[[int, float], None] | None = None,
    phase_callback: Callable[[int, npt.NDArray[np.floating]], None] | None = None,
    propagation: str = "asm",
) -> GSResult:
    """用于相位恢复的 Gerchberg-Saxton 算法。

    计算应施加在源面 (SLM) 的最优相位图形, 使目标面上产生期望的强度分布。

    算法:
        1. 在源面初始化光场 A
        2. 每次迭代:
           a. 施加源面振幅约束: B = source_amp * exp(i*phase(A))
           b. 正向传播到目标面: C = ASM(B, +z)
           c. 施加目标面振幅约束: D = target_amp * exp(i*phase(C))
           d. 反向传播回源面: A = ASM(D, -z)
        3. 提取最终相位: phase = angle(A)

    Args:
        source_amplitude: 二维数组, SLM 面上的振幅约束
                         (通常为均匀照明, 形状与 SLM 一致)
        target_amplitude: 二维数组, 目标面上期望的振幅
                         (目标强度图的平方根)
        iterations: GS 迭代次数 (默认: 50)
        cell_spacing: 像素尺寸, 单位米 (SLM200 默认 8e-6)
        distance: 传播距离, 单位米 (默认: 0.1)
        wavelength: 光的波长, 单位米 (YAG 激光默认 1064e-9)
        error_threshold: 可选的收敛阈值 (均方误差)
        progress_callback: 可选的回调函数 (迭代数, 误差), 用于监控
        phase_callback: 可选的回调函数 (迭代数, 相位), 每次迭代后以当前
            源面相位 (弧度) 调用, 从而能把演化中的相位图形实时显示到硬件上。
        propagation: 传播模型, ``"asm"`` (角谱法, 默认) 或 ``"fft"``
            (单次 FFT 的夫琅禾费平面 —— 焦平面被视作源平面的傅里叶变换)。
            ``"fft"`` 完全省掉每次迭代构建传播子的开销, 速度与单对 FFT/IFFT
            的远场 GS 相当。

    Returns:
        含相位、振幅、误差历史与收敛信息的 GSResult

    Raises:
        ValueError: 输入数组维度不对或参数无效时

    Example:
        >>> # 由图像创建目标振幅
        >>> target_img = np.loadtxt('target_pattern.csv', delimiter=',')
        >>> target_amp = np.sqrt(target_img / target_img.max())  # 归一化并开方
        >>>
        >>> # 均匀源振幅
        >>> source_amp = np.ones((1200, 1920))
        >>>
        >>> # 运行 GS 算法
        >>> result = gerchberg_saxton(
        ...     source_amplitude=source_amp,
        ...     target_amplitude=target_amp,
        ...     iterations=100,
        ...     cell_spacing=8e-6,
        ...     distance=0.15,
        ...     wavelength=1064e-9,
        ... )
        >>>
        >>> # 使用算出的相位
        >>> slm_phase = result.phase  # 弧度, 0-2π
    """
    # 校验输入
    if source_amplitude.ndim != 2 or target_amplitude.ndim != 2:
        raise ValueError("Input amplitudes must be 2D arrays")

    if source_amplitude.shape != target_amplitude.shape:
        raise ValueError(
            f"Source and target shapes must match: "
            f"{source_amplitude.shape} vs {target_amplitude.shape}"
        )

    if iterations < 1:
        raise ValueError(f"Iterations must be >= 1, got {iterations}")

    if cell_spacing <= 0 or distance <= 0 or wavelength <= 0:
        raise ValueError("Physical parameters must be positive")

    if propagation not in ("asm", "fft"):
        raise ValueError(f"propagation must be 'asm' or 'fft', got {propagation!r}")

    logger.info(
        f"Starting Gerchberg-Saxton algorithm: "
        f"iterations={iterations}, distance={distance * 1000:.1f}mm, "
        f"λ={wavelength * 1e9:.0f}nm, pixel={cell_spacing * 1e6:.1f}µm, "
        f"propagation={propagation}"
    )

    Ny, Nx = source_amplitude.shape

    # 用反向传播到源面的目标来初始化光场 A
    # 这比随机初始化给出更好的起点
    logger.debug("Initializing field with back-propagated target")
    if propagation == "asm":
        A = angular_spectrum_propagate(
            target_amplitude.astype(np.complex128),
            cell_spacing,
            -distance,  # 反向传播
            wavelength,
        )
    else:
        # 夫琅禾费反向传播: 目标平面的逆 FFT
        A = ifft2(ifftshift(target_amplitude.astype(np.complex128)))

    error_history = []

    # GS 主迭代循环
    for i in range(iterations):
        # 步骤 1: 施加源面振幅约束
        # B = source_amplitude * exp(i * phase(A))
        phase_A = np.angle(A)
        B = source_amplitude * np.exp(1j * phase_A)

        if propagation == "asm":
            # 步骤 2: 正向传播到目标面
            C = angular_spectrum_propagate(B, cell_spacing, distance, wavelength)
        else:
            # 步骤 2: 正向传播到夫琅禾费 (焦) 平面 —— 单次 FFT
            C = fftshift(fft2(B))

        # 步骤 3: 施加目标面振幅约束
        # D = target_amplitude * exp(i * phase(C))
        phase_C = np.angle(C)
        D = target_amplitude * np.exp(1j * phase_C)

        if propagation == "asm":
            # 步骤 4: 反向传播回源面
            A = angular_spectrum_propagate(D, cell_spacing, -distance, wavelength)
        else:
            # 步骤 4: 反向传播回源面 —— 单次 IFFT
            A = ifft2(ifftshift(D))

        # 计算误差 (|C| 与目标之间的均方误差)
        amplitude_C = np.abs(C)
        mse = np.mean((amplitude_C - target_amplitude) ** 2)
        error_history.append(float(mse))

        # 进度回调
        if progress_callback is not None:
            progress_callback(i, float(mse))

        # 实时相位回调 —— 把当前源面相位推送到硬件
        if phase_callback is not None:
            phase_callback(i, phase_A)

        # 每 10 次迭代记录一次进度
        if (i + 1) % 10 == 0 or i == 0:
            logger.debug(f"Iteration {i + 1}/{iterations}, MSE={mse:.6f}")

        # 检查收敛
        if error_threshold is not None and mse < error_threshold:
            logger.info(f"Converged at iteration {i + 1} with MSE={mse:.6f}")
            break

    # 提取最终结果
    final_phase = np.angle(A)

    # 再正向传播一次以得到目标面振幅
    final_B = source_amplitude * np.exp(1j * final_phase)
    if propagation == "asm":
        final_C = angular_spectrum_propagate(
            final_B, cell_spacing, distance, wavelength
        )
    else:
        final_C = fftshift(fft2(final_B))
    final_amplitude = np.abs(final_C)

    # 检查是否收敛
    converged = error_threshold is not None and error_history[-1] < error_threshold

    logger.info(
        f"GS algorithm completed: final MSE={error_history[-1]:.6f}, "
        f"converged={converged}"
    )

    return GSResult(
        phase=final_phase,
        amplitude=final_amplitude,
        error_history=error_history,
        iterations=len(error_history),
        converged=converged,
    )


def adaptive_gerchberg_saxton(
    source_amplitude: npt.NDArray[np.floating],
    target_amplitude: npt.NDArray[np.floating],
    measured_amplitude_callback: Callable[
        [npt.NDArray[np.floating]], npt.NDArray[np.floating]
    ],
    outer_iterations: int = 5,
    inner_iterations: int = 30,
    cell_spacing: float = 8e-6,
    distance: float = 0.1,
    wavelength: float = 1064e-9,
    feedback_weight: float = 0.3,
) -> GSResult:
    """带实验反馈的自适应 Gerchberg-Saxton。

    这个变体把实验装置实测到的振幅引入进来, 迭代细化相位图形。
    当理论模型与实际不完全吻合时它很有用。

    Args:
        source_amplitude: SLM 面上的振幅约束
        target_amplitude: 目标面上期望的振幅
        measured_amplitude_callback: 接受相位图形、把它显示到 SLM 上、
            用 CCD 拍图, 并返回实测振幅 (强度的平方根) 的函数
        outer_iterations: 自适应反馈外层循环的次数
        inner_iterations: 每个反馈循环内的 GS 迭代次数
        cell_spacing: 像素间距, 单位米
        distance: 传播距离, 单位米
        wavelength: 光的波长, 单位米
        feedback_weight: 实测与仿真混合时的权重 (0-1)

    Returns:
        含最终相位的 GSResult

    Example:
        >>> def capture_amplitude(phase_pattern):
        ...     slm.display_phase(phase_pattern)
        ...     img = camera.get_image()
        ...     return np.sqrt(img)  # 由强度得到振幅
        >>>
        >>> result = adaptive_gerchberg_saxton(
        ...     source_amplitude,
        ...     target_amplitude,
        ...     capture_amplitude,
        ...     outer_iterations=5,
        ...     inner_iterations=20,
        ... )
    """
    logger.info(f"Starting adaptive GS: {outer_iterations} outer loops")

    # 先跑标准 GS
    result = gerchberg_saxton(
        source_amplitude,
        target_amplitude,
        iterations=inner_iterations,
        cell_spacing=cell_spacing,
        distance=distance,
        wavelength=wavelength,
    )

    current_phase = result.phase

    for outer_i in range(outer_iterations):
        logger.info(f"Adaptive iteration {outer_i + 1}/{outer_iterations}")

        # 从实验装置取实测振幅
        measured_amp = measured_amplitude_callback(current_phase)

        # 把目标与实测混合 (反馈)
        # 这让算法能适应真实世界中的不完美
        blended_target = (
            1 - feedback_weight
        ) * target_amplitude + feedback_weight * measured_amp * target_amplitude / (
            measured_amp + 1e-10
        )

        # 用混合后的目标跑 GS
        result = gerchberg_saxton(
            source_amplitude,
            blended_target,
            iterations=inner_iterations,
            cell_spacing=cell_spacing,
            distance=distance,
            wavelength=wavelength,
        )

        current_phase = result.phase

    return result


def calculate_reconstruction_error(
    computed_phase: npt.NDArray[np.floating],
    source_amplitude: npt.NDArray[np.floating],
    target_amplitude: npt.NDArray[np.floating],
    cell_spacing: float = 8e-6,
    distance: float = 0.1,
    wavelength: float = 1064e-9,
) -> dict[str, float]:
    """计算衡量 GS 重建质量的各种误差指标。

    Args:
        computed_phase: GS 算法算出的相位图形
        source_amplitude: 源面振幅约束
        target_amplitude: 目标面振幅约束
        cell_spacing: 像素间距, 单位米
        distance: 传播距离, 单位米
        wavelength: 光的波长, 单位米

    Returns:
        含各误差指标的字典:
            - mse: 均方误差
            - nmse: 归一化 MSE
            - correlation: 相关系数
            - efficiency: 光学效率
    """
    # 把算出的相位传播到目标面
    field = source_amplitude * np.exp(1j * computed_phase)
    propagated = angular_spectrum_propagate(field, cell_spacing, distance, wavelength)
    computed_amplitude = np.abs(propagated)

    # 归一化以便比较
    target_norm = target_amplitude / (target_amplitude.max() + 1e-10)
    computed_norm = computed_amplitude / (computed_amplitude.max() + 1e-10)

    # MSE
    mse = np.mean((computed_norm - target_norm) ** 2)

    # 归一化 MSE
    nmse = mse / (np.mean(target_norm**2) + 1e-10)

    # 相关系数
    correlation = np.corrcoef(computed_norm.flatten(), target_norm.flatten())[0, 1]

    # 光学效率 (目标区域内的能量 / 总能量)
    efficiency = np.sum(computed_amplitude**2) / (np.sum(source_amplitude**2) + 1e-10)

    return {
        "mse": float(mse),
        "nmse": float(nmse),
        "correlation": float(correlation),
        "efficiency": float(efficiency),
    }
