"""Gerchberg-Saxton Hologram Runner

基于Gerchberg-Saxton算法的全息图生成和优化Runner。
使用Santec SLM200空间光调制器和Daheng CCD相机。

该runner通过GS算法计算最优相位图案，在SLM上显示，
并使用CCD捕获实际远场图像进行验证。

Usage:
    python gs_hologram_runner.py --target-image target.png --iterations 100
    python gs_hologram_runner.py --target-shape gaussian --slm-wavelength 1064
    
    # 通过main.py调用
    python main.py gs --target-image target.png --distance 0.15
"""

from __future__ import annotations

import os
import sys
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Any

# Ensure src is in path when running directly
if __name__ == "__main__":
    _script_dir = Path(__file__).resolve().parent
    _src_root = _script_dir.parent
    if str(_src_root) not in sys.path:
        sys.path.insert(0, str(_src_root))

import click
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from loguru import logger

# Import GS algorithm
from ao_shaping.algorithm.gerchberg_saxton import (
    gerchberg_saxton,
    adaptive_gerchberg_saxton,
    calculate_reconstruction_error,
    GSResult,
)

# Import hardware drivers with graceful fallback
SantecSLM200: Any = None
DahengCamManager: Any = None
SLM_AVAILABLE = False
CCD_AVAILABLE = False

try:
    from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200 as _SantecSLM200
    SantecSLM200 = _SantecSLM200
    SLM_AVAILABLE = True
except ImportError:
    logger.debug("SLM driver not available")

try:
    from ao_shaping.drivers.ccd.daheng import DahengCamManager as _DahengCamManager
    DahengCamManager = _DahengCamManager
    CCD_AVAILABLE = True
except ImportError:
    logger.debug("CCD driver not available")

# Import utilities
from ao_shaping.utils.file import gen_date_dir, gen_file_path_uuid


# Default physical parameters
DEFAULT_WAVELENGTH = 1064e-9  # 1064 nm (YAG laser)
DEFAULT_SLM_PIXEL_SIZE = 8e-6  # 8 µm (SLM200 pixel pitch)
DEFAULT_DISTANCE = 0.1  # 10 cm propagation distance


def parse_tuple(ctx, param, value: str | None) -> Tuple[int, int] | None:
    """解析元组格式的参数，支持 'x,y' 或 '(x,y)' 格式"""
    if value is None:
        return None
    # 移除空格和括号
    s_clean = re.sub(r"[()\s]", "", str(value))
    try:
        parts = s_clean.split(",")
        if len(parts) != 2:
            raise ValueError("Must have exactly two integers")
        x, y = map(int, parts)
        return (x, y)
    except Exception as e:
        raise click.BadParameter(
            f"Invalid format: {value}. Expected: 'x,y' or '(x,y)'"
        )


def load_target_image(
    image_path: Path,
    target_size: Tuple[int, int] = (1920, 1200),
    invert: bool = False,
) -> np.ndarray:
    """加载目标图像并预处理为振幅分布。
    
    Args:
        image_path: 图像文件路径
        target_size: 目标尺寸 (width, height)
        invert: 是否反转图像 (黑底白字→白底黑字)
    
    Returns:
        归一化的振幅分布数组 (0-1)
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Target image not found: {image_path}")
    
    # 加载图像
    img = Image.open(image_path).convert("L")  # 转为灰度
    
    # 调整尺寸
    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # 转为numpy数组并归一化
    img_array = np.array(img_resized, dtype=np.float64)
    img_array = img_array / 255.0  # 归一化到 0-1
    
    if invert:
        img_array = 1.0 - img_array
    
    # 振幅 = sqrt(强度)
    amplitude = np.sqrt(img_array)
    
    logger.info(f"Loaded target image: {image_path}, shape: {amplitude.shape}")
    return amplitude


def create_target_shape(
    shape: str,
    size: Tuple[int, int] = (1920, 1200),
    **kwargs,
) -> np.ndarray:
    """生成预设的目标形状振幅分布。
    
    Args:
        shape: 形状名称 ('gaussian', 'circle', 'square', 'annular', 'grid')
        size: 输出尺寸 (width, height)
        **kwargs: 形状特定参数
    
    Returns:
        振幅分布数组
    """
    width, height = size
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    if shape == "gaussian":
        sigma = kwargs.get("sigma", 0.3)
        intensity = np.exp(-(R**2) / (2 * sigma**2))
    
    elif shape == "circle":
        radius = kwargs.get("radius", 0.5)
        intensity = (R <= radius).astype(float)
    
    elif shape == "square":
        side = kwargs.get("side", 0.8)
        intensity = ((np.abs(X) <= side/2) & (np.abs(Y) <= side/2)).astype(float)
    
    elif shape == "annular":
        inner_r = kwargs.get("inner_radius", 0.2)
        outer_r = kwargs.get("outer_radius", 0.5)
        intensity = ((R >= inner_r) & (R <= outer_r)).astype(float)
    
    elif shape == "grid":
        nx = kwargs.get("nx", 5)
        ny = kwargs.get("ny", 5)
        line_width = kwargs.get("line_width", 0.02)
        intensity = np.zeros((height, width))
        for i in range(nx):
            x_pos = -1 + 2 * i / (nx - 1)
            intensity[np.abs(X - x_pos) < line_width] = 1.0
        for j in range(ny):
            y_pos = -1 + 2 * j / (ny - 1)
            intensity[np.abs(Y - y_pos) < line_width] = 1.0
    
    elif shape == "cross":
        thickness = kwargs.get("thickness", 0.05)
        intensity = ((np.abs(X) < thickness) | (np.abs(Y) < thickness)).astype(float)
    
    else:
        raise ValueError(f"Unknown shape: {shape}")
    
    amplitude = np.sqrt(intensity)
    logger.info(f"Created target shape: {shape}, size: {size}")
    return amplitude


def phase_to_slm_grayscale(
    phase: np.ndarray,
    max_grayscale: int = 1023,
) -> np.ndarray:
    """将弧度相位转换为SLM灰度值。
    
    Args:
        phase: 相位数组 (radians, typically 0-2π)
        max_grayscale: 最大灰度值 (10-bit SLM = 1023)
    
    Returns:
        uint16灰度值数组
    """
    # Wrap phase to 0-2π
    phase_wrapped = np.mod(phase, 2 * np.pi)
    
    # Convert to grayscale
    grayscale = (phase_wrapped / (2 * np.pi)) * max_grayscale
    
    return grayscale.astype(np.uint16)


def capture_amplitude_with_ccd(
    camera: Any,
    n_samples: int = 3,
) -> np.ndarray:
    """使用CCD捕获图像并转为振幅分布。
    
    Args:
        camera: 已初始化的相机对象
        n_samples: 采样平均次数
    
    Returns:
        振幅分布 (sqrt of normalized intensity)
    """
    # 捕获图像
    img = camera.get_numpy_image(n_sample=n_samples, skip_first=True)
    
    # 转为float并归一化
    img_float = img.astype(np.float64)
    img_normalized = img_float / (img_float.max() + 1e-10)
    
    # 振幅 = sqrt(强度)
    amplitude = np.sqrt(img_normalized)
    
    return amplitude


@click.command(name="gs")
@click.option(
    "--target-image",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="目标图像路径 (灰度图，将转为振幅分布)",
)
@click.option(
    "--target-shape",
    type=click.Choice(["gaussian", "circle", "square", "annular", "grid", "cross"]),
    default="gaussian",
    help="预设目标形状 (当未指定--target-image时使用)",
)
@click.option(
    "-i", "--iterations",
    default=50,
    help="GS算法迭代次数 (default: 50)",
)
@click.option(
    "-d", "--distance",
    default=0.1,
    help="传播距离 (米) (default: 0.1)",
)
@click.option(
    "-l", "--wavelength",
    default=1064.0,
    help="激光波长 (纳米) (default: 1064)",
)
@click.option(
    "--slm-wavelength",
    default=1064,
    help="SLM工作波长 (纳米，用于设置SLM) (default: 1064)",
)
@click.option(
    "--slm-number",
    default=1,
    help="SLM设备编号 (default: 1)",
)
@click.option(
    "--cam-id",
    default=lambda: os.environ.get("FAR_CAM_ID", "0"),
    help="CCD相机ID (default: FAR_CAM_ID/0)",
)
@click.option(
    "--cam-center",
    callback=parse_tuple,
    default=None,
    help="CCD中心位置 'x,y' (default: 自动检测)",
)
@click.option(
    "--cam-size",
    default=400,
    help="CCD开窗大小 (像素) (default: 400)",
)
@click.option(
    "--cam-exposure",
    default=50.0,
    help="CCD曝光时间 (毫秒) (default: 50)",
)
@click.option(
    "-s", "--save-dir",
    default="data/gs_hologram",
    help="结果保存目录 (default: data/gs_hologram)",
)
@click.option(
    "--use-hardware",
    is_flag=True,
    help="使用实际硬件 (SLM+CCD)，否则仅模拟计算",
)
@click.option(
    "--adaptive",
    is_flag=True,
    help="启用自适应GS (使用CCD反馈迭代优化)",
)
@click.option(
    "--adaptive-iterations",
    default=3,
    help="自适应迭代次数 (default: 3)",
)
@click.option(
    "--show",
    is_flag=True,
    help="显示结果图像",
)
@click.option(
    "--debug",
    is_flag=True,
    help="调试模式 (保存详细数据)",
)
def run(
    target_image: Optional[Path],
    target_shape: str,
    iterations: int,
    distance: float,
    wavelength: float,
    slm_wavelength: int,
    slm_number: int,
    cam_id: str,
    cam_center: Optional[Tuple[int, int]],
    cam_size: int,
    cam_exposure: float,
    save_dir: str,
    use_hardware: bool,
    adaptive: bool,
    adaptive_iterations: int,
    show: bool,
    debug: bool,
):
    """Gerchberg-Saxton全息图生成器
    
    使用GS算法计算最优相位图案，生成目标远场光强分布。
    可选择使用实际硬件(SLM+CCD)或仅进行模拟计算。
    
    Examples:
        # 仅模拟计算高斯光斑
        python gs_hologram_runner.py --target-shape gaussian
        
        # 使用硬件生成自定义图像
        python gs_hologram_runner.py --target-image logo.png --use-hardware
        
        # 自适应优化
        python gs_hologram_runner.py --target-shape circle --use-hardware --adaptive
    """
    # 转换波长单位 (nm -> m)
    wavelength_m = wavelength * 1e-9
    
    # SLM参数 (SLM200分辨率)
    slm_resolution = (1920, 1200)  # (width, height)
    pixel_size = DEFAULT_SLM_PIXEL_SIZE
    
    logger.info("=" * 60)
    logger.info("Gerchberg-Saxton Hologram Generator")
    logger.info("=" * 60)
    logger.info(f"Target: {target_image or target_shape}")
    logger.info(f"Distance: {distance*1000:.1f} mm")
    logger.info(f"Wavelength: {wavelength:.0f} nm")
    logger.info(f"Iterations: {iterations}")
    logger.info(f"Hardware: {'enabled' if use_hardware else 'disabled'}")
    if adaptive:
        logger.info(f"Adaptive mode: {adaptive_iterations} outer iterations")
    
    # 准备目标振幅分布
    if target_image is not None:
        target_amplitude = load_target_image(target_image, slm_resolution)
    else:
        target_amplitude = create_target_shape(target_shape, slm_resolution)
    
    # 归一化
    target_amplitude = target_amplitude / (target_amplitude.max() + 1e-10)
    
    # 源平面振幅 (假设均匀照明)
    source_amplitude = np.ones(slm_resolution[::-1])  # (height, width)
    
    # 硬件初始化
    slm = None
    camera = None
    
    if use_hardware:
        if not SLM_AVAILABLE:
            raise RuntimeError("SLM driver not available. Install Santec SLM SDK.")
        if not CCD_AVAILABLE:
            raise RuntimeError("CCD driver not available. Install Daheng SDK.")
        
        logger.info("Initializing hardware...")
        
        # 初始化SLM
        slm = SantecSLM200(slm_number=slm_number)
        slm.open()
        slm.set_wavelength(slm_wavelength)
        logger.info(f"SLM initialized: #{slm_number}, λ={slm_wavelength}nm")
        
        # 初始化CCD
        cam_id_int = int(cam_id)
        camera = DahengCamManager(cam_id=cam_id_int, exposure_time_ms=cam_exposure)
        camera.open()
        
        # 设置ROI
        if cam_center is not None:
            camera.reset_window(center=cam_center, size=(cam_size, cam_size))
        else:
            # 使用全帧
            pass
        
        logger.info(f"CCD initialized: ID={cam_id_int}, exposure={cam_exposure}ms")
    
    # 运行GS算法
    result: GSResult | None = None
    
    try:
        if adaptive and use_hardware:
            # Hardware is required for adaptive mode
            assert slm is not None, "SLM must be initialized for adaptive mode"
            assert camera is not None, "Camera must be initialized for adaptive mode"
            
            logger.info("Running adaptive Gerchberg-Saxton...")
            
            def make_capture_callback(slm_device: Any, cam_device: Any):
                """Create capture callback with bound devices."""
                def capture_callback(phase: np.ndarray) -> np.ndarray:
                    """Capture actual amplitude with CCD"""
                    # Convert phase to SLM grayscale
                    slm_phase = phase_to_slm_grayscale(phase)
                    
                    # Display on SLM
                    slm_device.display_data(slm_phase)
                    
                    # Wait for SLM response
                    import time
                    time.sleep(0.1)
                    
                    # Capture with CCD
                    return capture_amplitude_with_ccd(cam_device)
                return capture_callback
            
            result = adaptive_gerchberg_saxton(
                source_amplitude=source_amplitude,
                target_amplitude=target_amplitude,
                measured_amplitude_callback=make_capture_callback(slm, camera),
                outer_iterations=adaptive_iterations,
                inner_iterations=iterations,
                cell_spacing=pixel_size,
                distance=distance,
                wavelength=wavelength_m,
            )
        else:
            logger.info("Running standard Gerchberg-Saxton...")
            result = gerchberg_saxton(
                source_amplitude=source_amplitude,
                target_amplitude=target_amplitude,
                iterations=iterations,
                cell_spacing=pixel_size,
                distance=distance,
                wavelength=wavelength_m,
            )
            
            # If using hardware, display result on SLM
            if use_hardware:
                if slm is None or camera is None:
                    raise RuntimeError("Hardware not properly initialized")
                
                logger.info("Displaying phase pattern on SLM...")
                slm_phase = phase_to_slm_grayscale(result.phase)
                slm.display_data(slm_phase)
                
                # Capture actual result
                logger.info("Capturing result with CCD...")
                import time
                time.sleep(0.2)  # Wait for SLM
                actual_amplitude = capture_amplitude_with_ccd(camera)
    
    except Exception as e:
        logger.error(f"GS algorithm failed: {e}")
        raise
    
    finally:
        # Cleanup hardware
        if slm is not None:
            slm.close()
        if camera is not None:
            camera.close()
    
    # 计算误差指标
    metrics = calculate_reconstruction_error(
        result.phase,
        source_amplitude,
        target_amplitude,
        pixel_size,
        distance,
        wavelength_m,
    )
    
    logger.info("-" * 60)
    logger.info("Results:")
    logger.info(f"  MSE: {metrics['mse']:.6f}")
    logger.info(f"  NMSE: {metrics['nmse']:.6f}")
    logger.info(f"  Correlation: {metrics['correlation']:.4f}")
    logger.info(f"  Efficiency: {metrics['efficiency']:.4f}")
    logger.info("-" * 60)
    
    # 保存结果
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_name = f"gs_{target_shape if target_image is None else target_image.stem}_{timestamp}"
    result_dir = save_path / result_name
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    config = {
        "target": str(target_image) if target_image else target_shape,
        "iterations": iterations,
        "distance_m": distance,
        "wavelength_nm": wavelength,
        "slm_wavelength_nm": slm_wavelength,
        "slm_resolution": slm_resolution,
        "use_hardware": use_hardware,
        "adaptive": adaptive,
        "metrics": metrics,
    }
    
    with open(result_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # 保存相位图案 (灰度值)
    slm_phase = phase_to_slm_grayscale(result.phase)
    np.save(result_dir / "phase_pattern.npy", slm_phase)
    
    # 保存目标振幅
    np.save(result_dir / "target_amplitude.npy", target_amplitude)
    
    # 保存仿真结果振幅
    np.save(result_dir / "simulated_amplitude.npy", result.amplitude)
    
    # 保存误差历史
    np.save(result_dir / "error_history.npy", np.array(result.error_history))
    
    # 可视化并保存
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 目标振幅
    im0 = axes[0, 0].imshow(target_amplitude, cmap="gray")
    axes[0, 0].set_title("Target Amplitude")
    plt.colorbar(im0, ax=axes[0, 0])
    
    # 计算相位 (0-2π)
    phase_display = np.mod(result.phase, 2 * np.pi)
    im1 = axes[0, 1].imshow(phase_display, cmap="hsv", vmin=0, vmax=2*np.pi)
    axes[0, 1].set_title(f"SLM Phase Pattern ({result.iterations} iter)")
    plt.colorbar(im1, ax=axes[0, 1])
    
    # SLM灰度值
    im2 = axes[0, 2].imshow(slm_phase, cmap="gray")
    axes[0, 2].set_title(f"SLM Grayscale (0-{slm_phase.max()})")
    plt.colorbar(im2, ax=axes[0, 2])
    
    # 仿真结果
    im3 = axes[1, 0].imshow(result.amplitude, cmap="gray")
    axes[1, 0].set_title("Simulated Result")
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 误差历史
    axes[1, 1].plot(result.error_history)
    axes[1, 1].set_title("Error History (MSE)")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("MSE")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True, alpha=0.3)
    
    # 指标文本
    axes[1, 2].axis("off")
    metrics_text = (
        f"MSE: {metrics['mse']:.6f}\\n"
        f"NMSE: {metrics['nmse']:.6f}\\n"
        f"Correlation: {metrics['correlation']:.4f}\\n"
        f"Efficiency: {metrics['efficiency']:.4f}"
    )
    axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=14, family="monospace",
                    verticalalignment="center")
    
    plt.tight_layout()
    plt.savefig(result_dir / "result_overview.png", dpi=150)
    
    if show:
        plt.show()
    else:
        plt.close()
    
    logger.info(f"Results saved to: {result_dir}")
    click.echo(f"GS hologram generation complete!")
    click.echo(f"  Results saved to: {result_dir}")
    click.echo(f"  MSE: {metrics['mse']:.6f}")
    click.echo(f"  Correlation: {metrics['correlation']:.4f}")


if __name__ == "__main__":
    run()
