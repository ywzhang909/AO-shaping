"""
CCD Real-time Image Analyzer (Streamlit page)

Features:
1. Real-time CCD image capture (Daheng camera by default)
2. Automatic enclosing ellipse calculation
3. X/Y intensity cross-sections from centroid
4. Gaussian fit curves for cross-sections
"""

import streamlit as st
import numpy as np
from pathlib import Path
import sys
import time
import logging
from typing import Tuple, Optional

# Add the src directory to the path when running this file directly via Streamlit.
import sys
import types
from pathlib import Path

# ccd_analyzer.py is at: src/ao_shaping/gui/steamlit_helper/ccd_analyzer.py
SRC_ROOT = Path(__file__).resolve().parents[3]  # src/
PROJECT_ROOT = Path(__file__).resolve().parents[4]  # AO-shaping/

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Patch miicam module before importing ccd package
if "miicam" not in sys.modules:
    sys.modules["miicam"] = types.ModuleType("miicam")

from loguru import logger
from ao_shaping.drivers.ccd.daheng import DahengCamManager
from ao_shaping.utils.spots_calc import centroid
from scipy import ndimage
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import BytesIO
from PIL import Image


def _initialize_camera_state() -> None:
    """Initialize camera session state variables."""
    if "camera" not in st.session_state:
        st.session_state.camera = None
        st.session_state.camera_connected = False
        st.session_state.camera_id = 0
        st.session_state.exposure_time_ms = 50
        st.session_state.auto_exposure = False
        st.session_state.update_interval = 0.5  # seconds
        st.session_state.roi_size = 0  # 0 means full frame
        st.session_state.roi_center = (0, 0)
        st.session_state.auto_refresh = (
            True  # Default to auto-refresh for better responsiveness
        )
        st.session_state.last_update_time = 0


def gaussian(
    x: np.ndarray, amplitude: float, center: float, sigma: float, offset: float
) -> np.ndarray:
    """高斯函数用于曲线拟合。"""
    return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma**2)) + offset


def fit_gaussian(
    x: np.ndarray, y: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    对数据进行高斯拟合。

    Returns:
        (popt, residual): 拟合参数和残差，如果拟合失败返回(None, None)
    """
    try:
        # 初始参数估计
        amplitude = np.max(y) - np.min(y)
        center = x[np.argmax(y)]
        sigma = (x.max() - x.min()) / 6
        offset = np.min(y)

        # 确保sigma为正
        if sigma <= 0:
            sigma = 1.0

        # 确保bounds有效：x_max > x_min
        x_min, x_max = float(x.min()), float(x.max())
        if x_max <= x_min + 0.1:
            x_max = x_min + 10.0  # 确保有足够的范围

        popt, pcov = curve_fit(
            gaussian,
            x,
            y,
            p0=[amplitude, center, sigma, offset],
            bounds=([0, x_min, 0.5, -np.inf], [np.inf, x_max, np.inf, np.inf]),
        )

        # 计算残差
        y_fit = gaussian(x, *popt)
        residual = np.sum((y - y_fit) ** 2)

        return popt, residual
    except Exception as e:
        logger.warning(f"Gaussian fitting failed: {e}")
        return None, None


def calculate_enclosing_ellipse(
    img: np.ndarray, threshold_ratio: float = 0.1
) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """
    计算包围椭圆。

    Args:
        img: 输入图像
        threshold_ratio: 阈值比率（相对于最大值的比例）

    Returns:
        (center, axes, angle): 椭圆中心、半轴长度、旋转角度
    """
    # 应用阈值
    threshold = threshold_ratio * np.max(img)
    binary = img > threshold

    # 计算质量属性
    labeled, num_features = ndimage.label(binary)

    if num_features == 0:
        # 如果没有找到区域，返回默认椭圆
        h, w = img.shape
        return ((w / 2, h / 2), (w / 4, h / 4), 0)

    # 获取最大连通区域
    sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
    max_label = np.argmax(sizes) + 1
    largest_component = labeled == max_label

    # 计算质心
    cy, cx = ndimage.center_of_mass(largest_component)

    # 计算二阶矩来确定主轴方向
    y_coords, x_coords = np.where(largest_component)
    if len(x_coords) < 3:
        h, w = img.shape
        return ((cx, cy), (w / 4, h / 4), 0)

    # 中心化坐标
    x_centered = x_coords - cx
    y_centered = y_coords - cy

    # 计算二阶矩
    m20 = np.sum(x_centered**2)
    m02 = np.sum(y_centered**2)
    m11 = np.sum(x_centered * y_centered)

    # 计算特征值和特征向量
    delta = np.sqrt((m20 - m02) ** 2 + 4 * m11**2)
    eigenvalues = [(m20 + m02 + delta) / 2, (m20 + m02 - delta) / 2]

    # 短轴和长轴（2倍标准差）
    a = 2 * np.sqrt(max(eigenvalues))
    b = 2 * np.sqrt(min(eigenvalues))

    # 计算旋转角度
    if m11 != 0 or m20 != m02:
        angle = 0.5 * np.arctan2(2 * m11, m20 - m02)
    else:
        angle = 0

    return ((cx, cy), (a, b), np.degrees(angle))


def draw_ellipse_on_image(
    img: np.ndarray,
    ellipse_params: Tuple[Tuple[float, float], Tuple[float, float], float],
) -> np.ndarray:
    """在图像上绘制椭圆并返回带标记的图像。"""
    import matplotlib.patches as mpatches
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    # 归一化图像到0-255
    img_normalized = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(
        np.uint8
    )

    # 创建RGB图像
    h, w = img_normalized.shape
    rgb_img = np.stack([img_normalized] * 3, axis=-1)

    # 创建图形并绘制
    fig, ax = plt.subplots(1, 1, figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(img_normalized, cmap="gray")

    # 获取椭圆参数
    center, axes, angle = ellipse_params

    # 创建椭圆patch
    ellipse = mpatches.Ellipse(
        center,
        axes[0] * 2,
        axes[1] * 2,
        angle=angle,
        fill=False,
        edgecolor="red",
        linewidth=2,
    )
    ax.add_patch(ellipse)

    # 标记中心点
    ax.plot(center[0], center[1], "r+", markersize=10, markeredgewidth=2)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    plt.tight_layout(pad=0)

    # 转换为numpy数组
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    result = plt.imread(buf)
    plt.close(fig)

    # 如果是灰度图像，转换为RGB
    if result.ndim == 2:
        result = np.stack([result] * 3, axis=-1)

    return (result * 255).astype(np.uint8)


def get_cross_sections(
    img: np.ndarray, center: Tuple[float, float], width: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    获取X和Y方向的强度分布截面（从质心位置提取）。

    对质心周围一定宽度范围内的像素进行平均，以获得更平滑的截面数据。

    Args:
        img: 输入图像
        center: 质心坐标 (x, y) - 必须是质心位置
        width: 截面提取宽度（从质心左右各取width个像素进行平均）

    Returns:
        (x_profile, y_profile): X和Y方向的强度分布
    """
    cy, cx = int(round(center[1])), int(round(center[0]))
    h, w = img.shape

    # 计算提取范围
    half_width = width // 2

    # X方向截面（水平穿过质心的一行，对width范围内的像素取平均）
    x_start = max(0, cx - half_width)
    x_end = min(w, cx + half_width + 1)
    if x_end > x_start:
        # 对指定宽度范围内的列取平均，得到单行数据
        x_profile = np.mean(
            img[max(0, cy - half_width) : min(h, cy + half_width + 1), x_start:x_end],
            axis=0,
        )
    else:
        x_profile = np.array([img[cy, cx]])

    # Y方向截面（垂直穿过质心的一列，对width范围内的像素取平均）
    y_start = max(0, cy - half_width)
    y_end = min(h, cy + half_width + 1)
    if y_end > y_start:
        # 对指定宽度范围内的行取平均，得到单列数据
        y_profile = np.mean(
            img[y_start:y_end, max(0, cx - half_width) : min(w, cx + half_width + 1)],
            axis=1,
        )
    else:
        y_profile = np.array([img[cy, cx]])

    return x_profile, y_profile


def main():
    st.set_page_config(page_title="CCD Image Analyzer", page_icon="📷", layout="wide")

    _initialize_camera_state()

    st.title("📷 CCD Real-time Image Analyzer")
    st.markdown("Real-time CCD camera display and beam analysis")

    # Sidebar - Camera Settings
    with st.sidebar:
        st.header("Camera Settings")

        st.session_state.camera_id = st.number_input(
            "Camera ID",
            min_value=0,
            value=st.session_state.camera_id,
            help="Camera device ID",
        )

        new_exposure = st.slider(
            "Exposure Time (ms)",
            min_value=1,
            max_value=1000,
            value=st.session_state.exposure_time_ms,
            help="Camera exposure time",
        )

        # Update camera exposure time if changed and camera is connected
        if new_exposure != st.session_state.exposure_time_ms:
            st.session_state.exposure_time_ms = new_exposure
            if (
                st.session_state.camera is not None
                and st.session_state.camera_connected
            ):
                try:
                    # Call camera's internal method to update exposure
                    st.session_state.camera._CameraStreamManager__reset_exposure_time(
                        new_exposure
                    )
                    logger.info(f"Exposure time updated to {new_exposure}ms")

                    # Force immediate refresh after exposure change
                    # Force rerun to get new image immediately
                    st.session_state.last_update_time = 0  # Reset to force refresh
                    st.rerun()  # Force immediate rerun to apply exposure change
                except Exception as e:
                    logger.warning(f"Failed to update exposure time: {e}")

        st.session_state.auto_exposure = st.checkbox(
            "Auto Exposure",
            value=st.session_state.auto_exposure,
            help="Auto adjust exposure",
        )

        st.session_state.update_interval = st.slider(
            "Update Interval (s)",
            min_value=0.1,
            max_value=5.0,
            value=st.session_state.update_interval,
            step=0.1,
            help="Image refresh interval",
        )

        st.divider()

        # Check if camera object exists in session state (persists across page refreshes)
        # If camera exists but camera_connected is False, try to recover the connection
        if (
            st.session_state.camera is not None
            and not st.session_state.camera_connected
        ):
            # Try to initialize the existing camera object
            try:
                if (
                    hasattr(st.session_state.camera, "cam")
                    and st.session_state.camera.cam is not None
                ):
                    st.session_state.camera_connected = True
                    logger.info("Recovered existing camera connection")
            except Exception:
                # Camera object exists but not valid, reset it
                st.session_state.camera = None

        # Connect/Disconnect Camera
        if not st.session_state.camera_connected:
            if st.button("Connect Camera", type="primary"):
                try:
                    st.session_state.camera = DahengCamManager(
                        cam_id=st.session_state.camera_id,
                        exposure_time_ms=st.session_state.exposure_time_ms,
                    )
                    st.session_state.camera.initialize()
                    st.session_state.camera_connected = True
                    st.success("Camera connected")
                except Exception as e:
                    st.error(f"Connection failed: {e}")
        else:
            if st.button("Disconnect Camera", type="secondary"):
                try:
                    if st.session_state.camera is not None:
                        # Safely close the camera
                        try:
                            # Try __exit__ first (context manager)
                            st.session_state.camera.__exit__(None, None, None)
                        except AttributeError:
                            # If no __exit__, try direct close
                            if hasattr(st.session_state.camera, "close"):
                                st.session_state.camera.close()
                            elif (
                                hasattr(st.session_state.camera, "cam")
                                and st.session_state.camera.cam
                            ):
                                # Manually close the device
                                st.session_state.camera.cam.stream_off()
                                st.session_state.camera.cam.close_device()
                        # Clear the camera reference
                        st.session_state.camera = None
                    st.session_state.camera_connected = False
                    # Clear image data
                    if "current_image" in st.session_state:
                        st.session_state.current_image = None
                    st.info("Camera disconnected")
                except Exception as e:
                    logger.error(f"Disconnect failed: {e}")
                    st.session_state.camera = None
                    st.session_state.camera_connected = False
                    st.error(f"Disconnect failed: {e}")

    # Main area
    if st.session_state.camera_connected and st.session_state.camera is not None:
        # Real-time refresh mode
        st.subheader("Real-time Image")

        # Initialize session state for image data
        if "current_image" not in st.session_state:
            st.session_state.current_image = None
            st.session_state.current_analysis = None

        # Refresh control - add auto-refresh checkbox
        col_auto, col_manual, col_status = st.columns([1, 1, 2])

        with col_auto:
            st.session_state.auto_refresh = st.checkbox(
                "Auto Refresh",
                value=st.session_state.auto_refresh,
                help="Automatically refresh image",
            )

        with col_manual:
            # Manual refresh button
            manual_refresh = st.button("🔄 Refresh Image", type="primary")

        # DEBUG: Log checkbox and button state
        logger.debug(
            f"[DEBUG] auto_refresh={st.session_state.auto_refresh}, "
            f"manual_refresh={manual_refresh}, "
            f"last_update_time={st.session_state.last_update_time:.3f}, "
            f"update_interval={st.session_state.update_interval}"
        )

        with col_status:
            if st.session_state.current_image is not None:
                # Show capture timestamp to verify fresh image
                capture_time = st.session_state.current_image.get("capture_time", 0)
                if capture_time > 0:
                    from datetime import datetime

                    ts = datetime.fromtimestamp(capture_time)
                    st.success(f"✓ Live at {ts.strftime('%H:%M:%S.%f')[:-3]}")
            else:
                st.info("Click to capture")

        # Check if we need to update (auto-refresh or manual button)
        current_time = time.time()
        should_update = False

        # DEBUG: Log decision factors
        logger.debug(
            f"[DEBUG] current_time={current_time:.3f}, "
            f"elapsed={current_time - st.session_state.last_update_time:.3f}, "
            f"time_check_passed={current_time - st.session_state.last_update_time >= st.session_state.update_interval}"
        )

        if manual_refresh:
            logger.debug(
                "[DEBUG] Manual refresh button pressed - setting should_update=True"
            )
            should_update = True
        elif st.session_state.auto_refresh:
            # Auto-refresh: check if enough time has passed
            if (
                current_time - st.session_state.last_update_time
                >= st.session_state.update_interval
            ):
                logger.debug(
                    "[DEBUG] Auto-refresh time interval passed - setting should_update=True"
                )
                should_update = True
            else:
                logger.debug(
                    "[DEBUG] Auto-refresh but time interval not passed - no update"
                )

        logger.debug(f"[DEBUG] should_update={should_update}")

        if should_update:
            try:
                # DEBUG: Log before camera capture
                logger.debug("[DEBUG] Starting camera capture...")
                capture_start = time.time()

                # Get image directly - skip_first=False reduces delay, n_sample=1 for speed
                img = st.session_state.camera.get_numpy_image(
                    n_sample=1, skip_first=False
                )

                # DEBUG: Log after camera capture
                capture_end = time.time()
                logger.debug(
                    f"[DEBUG] Camera capture completed in {(capture_end - capture_start) * 1000:.1f}ms, "
                    f"image shape={img.shape}, dtype={img.dtype}"
                )

                # Calculate centroid
                cx, cy = centroid(img, moment=1, threshold=0.01)

                # Calculate enclosing ellipse
                ellipse_params = calculate_enclosing_ellipse(img)

                # Get cross-section data from centroid
                x_profile, y_profile = get_cross_sections(img, (cx, cy))

                # Gaussian fitting
                x = np.arange(len(x_profile))
                y = np.arange(len(y_profile))

                x_popt, x_residual = fit_gaussian(x, x_profile)
                y_popt, y_residual = fit_gaussian(y, y_profile)

                # Save current image and analysis results to session state
                # Include capture_time to verify it's fresh each time
                capture_time = time.time()
                st.session_state.current_image = {
                    "img": img,
                    "ellipse_params": ellipse_params,
                    "cx": cx,
                    "cy": cy,
                    "x_profile": x_profile,
                    "y_profile": y_profile,
                    "x_popt": x_popt,
                    "y_popt": y_popt,
                    "x": x,
                    "y": y,
                    "capture_time": capture_time,  # Add timestamp to verify fresh capture
                }

                # Update last update time AFTER capture
                st.session_state.last_update_time = time.time()

                # Debug: Log capture to console
                logger.info(f"Captured new image at {capture_time:.3f}")

                # For auto-refresh, trigger rerun AFTER successful capture to create polling loop
                # This creates the time-based polling effect
                if st.session_state.auto_refresh:
                    logger.debug("[DEBUG] Auto-refresh enabled, triggering st.rerun()")
                    st.rerun()

            except Exception as e:
                logger.error(f"[DEBUG] Failed to get image: {e}")
                st.error(f"Failed to get image: {e}")

        # Status display moved above

        # 显示结果
        if st.session_state.current_image is not None:
            data = st.session_state.current_image
            img = data["img"]
            ellipse_params = data["ellipse_params"]
            cx = data["cx"]
            cy = data["cy"]
            x_profile = data["x_profile"]
            y_profile = data["y_profile"]
            x_popt = data["x_popt"]
            y_popt = data["y_popt"]
            x = data["x"]
            y = data["y"]

            # Create two column layout
            col1, col2 = st.columns([2, 1])

            with col1:
                # Option 1: Display raw image directly (faster, more responsive)
                # Convert numpy to PIL for direct display
                pil_img = Image.fromarray(img.astype("uint8"))
                st.image(
                    pil_img,
                    caption=f"Raw CCD Image ({img.shape[1]}x{img.shape[0]})",
                    width="stretch",
                )

                # Option 2: If you want ellipse overlay, uncomment below:
                # annotated_img = draw_ellipse_on_image(img, ellipse_params)
                # st.image(annotated_img, caption="...", width="stretch")

            with col2:
                # Display analysis results
                st.markdown("### 📊 Analysis Results")
                st.markdown(f"**Centroid**: ({cx:.1f}, {cy:.1f})")
                st.markdown(
                    f"**Ellipse Center**: ({ellipse_params[0][0]:.1f}, {ellipse_params[0][1]:.1f})"
                )
                st.markdown(
                    f"**Ellipse Axes**: Major={ellipse_params[1][0]:.1f}, Minor={ellipse_params[1][1]:.1f}"
                )
                st.markdown(f"**Rotation Angle**: {ellipse_params[2]:.1f}°")

                st.markdown("---")

                if x_popt is not None:
                    st.markdown("#### X-direction Gaussian Fit")
                    st.metric("Amplitude", f"{x_popt[0]:.2f}")
                    st.metric("Center", f"{x_popt[1]:.2f}")
                    st.metric("σ (Sigma)", f"{x_popt[2]:.2f}")

                if y_popt is not None:
                    st.markdown("#### Y-direction Gaussian Fit")
                    st.metric("Amplitude", f"{y_popt[0]:.2f}")
                    st.metric("Center", f"{y_popt[1]:.2f}")
                    st.metric("σ (Sigma)", f"{y_popt[2]:.2f}")

            # Plot cross-section profiles
            st.markdown("### 📈 Intensity Profile")

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

            ax1.plot(x_profile, "b-", linewidth=1.5, label="X-profile")
            if x_popt is not None:
                x_fit = gaussian(x, *x_popt)
                ax1.plot(
                    x_fit,
                    "r--",
                    linewidth=1.5,
                    label=f"Gaussian Fit (σ={x_popt[2]:.2f})",
                )
            ax1.set_xlabel("X pixels")
            ax1.set_ylabel("Intensity")
            ax1.set_title("X-direction Intensity Profile")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(y_profile, "b-", linewidth=1.5, label="Y-profile")
            if y_popt is not None:
                y_fit = gaussian(y, *y_popt)
                ax2.plot(
                    y_fit,
                    "r--",
                    linewidth=1.5,
                    label=f"Gaussian Fit (σ={y_popt[2]:.2f})",
                )
            ax2.set_xlabel("Y pixels")
            ax2.set_ylabel("Intensity")
            ax2.set_title("Y-direction Intensity Profile")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
    else:
        # Camera not connected, show instructions
        st.info("Please connect camera in sidebar to start real-time monitoring")

        # Show available camera list
        st.subheader("Available Cameras")
        try:
            cam_list = DahengCamManager.get_cam_list()
            if cam_list:
                st.write(f"Found {len(cam_list)} camera device(s)")
                for i, cam in enumerate(cam_list):
                    st.write(f"  - Camera {i}: {cam}")
            else:
                st.warning("No camera devices found")
        except Exception as e:
            st.warning(f"Cannot get camera list: {e}")


if __name__ == "__main__":
    main()
