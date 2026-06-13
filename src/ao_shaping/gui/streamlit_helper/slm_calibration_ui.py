"""SLM LUT (Look-Up Table) Calibration UI

Features:
1. Connect to SLM and camera devices
2. Configure calibration parameters (grating period, grayscale range, ROI)
3. Run blazed grating calibration to find 2π phase grayscale value
4. Real-time visualization of calibration curve
5. Save/load calibration results

Usage:
    streamlit run src/ao_shaping/gui/streamlit_helper/slm_calibration_ui.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from loguru import logger

from ao_shaping.utils.file import ROOT_DIR as PROJECT_ROOT

# Mock miicam module before importing ccd package
import types

if "miicam" not in sys.modules:
    sys.modules["miicam"] = types.ModuleType("miicam")

# Import drivers and calibration module
from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200
from ao_shaping.drivers.slm.slm_calibration import (
    CalibrationMethod,
    CalibrationResult,
    SantecSLM200Calibrator,
)
from ao_shaping.drivers.ccd import CameraStreamManager


def _initialize_state() -> None:
    """Initialize session state variables."""
    # Device states
    if "slm_cal_slm" not in st.session_state:
        st.session_state.slm_cal_slm = None
        st.session_state.slm_cal_slm_connected = False

    if "slm_cal_camera" not in st.session_state:
        st.session_state.slm_cal_camera = None
        st.session_state.slm_cal_camera_connected = False

    # Calibration parameters
    if "slm_cal_method" not in st.session_state:
        st.session_state.slm_cal_method = CalibrationMethod.BLAZED_GRATING.value

    if "slm_cal_grating_period" not in st.session_state:
        st.session_state.slm_cal_grating_period = 8

    if "slm_cal_grayscale_min" not in st.session_state:
        st.session_state.slm_cal_grayscale_min = 100

    if "slm_cal_grayscale_max" not in st.session_state:
        st.session_state.slm_cal_grayscale_max = 800

    if "slm_cal_step" not in st.session_state:
        st.session_state.slm_cal_step = 20

    if "slm_cal_fine_step" not in st.session_state:
        st.session_state.slm_cal_fine_step = 2

    if "slm_cal_fine_range" not in st.session_state:
        st.session_state.slm_cal_fine_range = 50

    if "slm_cal_n_samples" not in st.session_state:
        st.session_state.slm_cal_n_samples = 3

    if "slm_cal_auto_exposure" not in st.session_state:
        st.session_state.slm_cal_auto_exposure = True

    if "slm_cal_target_min" not in st.session_state:
        st.session_state.slm_cal_target_min = 80

    if "slm_cal_target_max" not in st.session_state:
        st.session_state.slm_cal_target_max = 220

    if "slm_cal_roi_center" not in st.session_state:
        st.session_state.slm_cal_roi_center = None

    if "slm_cal_roi_size" not in st.session_state:
        st.session_state.slm_cal_roi_size = 100

    if "slm_cal_storage_dir" not in st.session_state:
        st.session_state.slm_cal_storage_dir = str(
            PROJECT_ROOT / "data" / "slm_calibration"
        )

    # Calibration state
    if "slm_cal_result" not in st.session_state:
        st.session_state.slm_cal_result = None

    if "slm_cal_running" not in st.session_state:
        st.session_state.slm_cal_running = False

    if "slm_cal_progress" not in st.session_state:
        st.session_state.slm_cal_progress = {
            "percent": 0,
            "message": "",
            "current_gs": None,
            "intensity": None,
        }

    # SLM device settings
    if "slm_cal_slm_number" not in st.session_state:
        st.session_state.slm_cal_slm_number = 1

    if "slm_cal_wavelength" not in st.session_state:
        st.session_state.slm_cal_wavelength = 1064

    if "slm_cal_exposure_time" not in st.session_state:
        st.session_state.slm_cal_exposure_time = 50


def connect_slm() -> bool:
    """Connect to SLM device."""
    try:
        if st.session_state.slm_cal_slm_connected:
            try:
                st.session_state.slm_cal_slm.close()
            except Exception:
                pass

        slm = SantecSLM200(
            slm_number=st.session_state.slm_cal_slm_number,
            wavelength=st.session_state.slm_cal_wavelength,
        )
        slm.open()

        st.session_state.slm_cal_slm = slm
        st.session_state.slm_cal_slm_connected = True

        st.success(
            f"SLM {st.session_state.slm_cal_slm_number} connected (wavelength={slm.wavelength}nm)"
        )
        return True

    except Exception as e:
        st.error(f"SLM connection failed: {e}")
        logger.exception(f"SLM connection failed: {e}")
        return False


def disconnect_slm() -> None:
    """Disconnect from SLM."""
    had_slm = st.session_state.slm_cal_slm is not None
    try:
        if had_slm:
            st.session_state.slm_cal_slm.close()
    except Exception as e:
        st.error(f"SLM disconnect failed: {e}")
        logger.exception(f"SLM disconnect failed: {e}")
    finally:
        st.session_state.slm_cal_slm = None
        st.session_state.slm_cal_slm_connected = False
        if had_slm:
            st.success("SLM disconnected")


def connect_camera() -> bool:
    """Connect to camera device."""
    try:
        if st.session_state.slm_cal_camera is not None:
            try:
                st.session_state.slm_cal_camera.close()
            except Exception:
                pass

        camera = CameraStreamManager(
            cam_id=0,
            exposure_time_ms=st.session_state.slm_cal_exposure_time,
        )
        camera.open()

        st.session_state.slm_cal_camera = camera
        st.session_state.slm_cal_camera_connected = True
        st.success(
            f"Camera connected (exposure={st.session_state.slm_cal_exposure_time}ms)"
        )
        return True

    except Exception as e:
        st.error(f"Camera connection failed: {e}")
        logger.exception(f"Camera connection failed: {e}")
        return False


def disconnect_camera() -> None:
    """Disconnect from camera."""
    had_camera = st.session_state.slm_cal_camera is not None
    try:
        if had_camera:
            st.session_state.slm_cal_camera.close()
    except Exception as e:
        st.error(f"Camera disconnect failed: {e}")
        logger.exception(f"Camera disconnect failed: {e}")
    finally:
        st.session_state.slm_cal_camera = None
        st.session_state.slm_cal_camera_connected = False
        if had_camera:
            st.success("Camera disconnected")


def _calibration_worker(
    calib: SantecSLM200Calibrator,
    grayscale_range: tuple[int, int],
    step: int,
    n_samples: int,
    fine_search: bool,
    fine_step: int,
    fine_range: int,
    measure_background: bool,
    auto_exposure: bool,
    target_min: int,
    target_max: int,
    progress_file: str,
) -> None:
    """Run calibration in background thread with progress updates via JSON file."""
    try:
        progress_path = Path(progress_file)

        def write_progress(data: dict) -> None:
            """Write progress to JSON file for UI polling."""
            try:
                with open(progress_path, "w") as f:
                    json.dump(data, f)
            except Exception as e:
                logger.warning(f"Failed to write progress file: {e}")

        if auto_exposure and st.session_state.slm_cal_camera is not None:
            result = calib.calibrate_with_auto_exposure(
                grayscale_range=grayscale_range,
                step=step,
                n_samples=n_samples,
                fine_search=fine_search,
                fine_step=fine_step,
                fine_range=fine_range,
                measure_background=measure_background,
                auto_exposure=True,
                target_min=target_min,
                target_max=target_max,
            )
        else:
            result = calib.calibrate_with_background(
                grayscale_range=grayscale_range,
                step=step,
                n_samples=n_samples,
                fine_search=fine_search,
                fine_step=fine_step,
                fine_range=fine_range,
                measure_background=measure_background,
            )

        st.session_state.slm_cal_result = result
        write_progress(
            {
                "status": "complete",
                "percent": 100.0,
                "message": "Calibration complete!",
            }
        )

    except Exception as e:
        logger.exception(f"Calibration error: {e}")
        st.session_state.slm_cal_progress["status"] = "error"
        st.session_state.slm_cal_progress["message"] = f"Calibration failed: {e}"

    finally:
        st.session_state.slm_cal_running = False


def render_sidebar() -> None:
    """Render sidebar configuration."""
    with st.sidebar:
        st.header("SLM LUT 标定")

        # Storage directory
        st.subheader("存储设置")
        st.session_state.slm_cal_storage_dir = st.text_input(
            "存储目录",
            value=st.session_state.slm_cal_storage_dir,
            help="标定结果保存目录",
        )

        # Calibration method selection
        st.subheader("标定方法")
        method_labels = {
            CalibrationMethod.BLAZED_GRATING.value: "闪耀光栅法 (Blazed Grating)",
            CalibrationMethod.INTERFEROMETER.value: "干涉法 (Interferometer)",
            CalibrationMethod.DIFFRACTION_EFFICIENCY.value: "衍射效率法 (Diffraction Efficiency)",
            CalibrationMethod.TWIN_BEAM.value: "双光束法 (Twin Beam)",
        }
        st.session_state.slm_cal_method = st.selectbox(
            "标定方法",
            options=[m.value for m in CalibrationMethod],
            format_func=lambda x: method_labels.get(x, x) or x,
            index=0,
            help="选择SLM相位-灰度响应标定方法",
        )

        st.divider()

        # SLM settings
        st.subheader("SLM参数")
        st.session_state.slm_cal_slm_number = st.number_input(
            "SLM编号",
            min_value=1,
            max_value=8,
            value=st.session_state.slm_cal_slm_number,
            step=1,
        )

        st.session_state.slm_cal_wavelength = st.number_input(
            "波长 (nm)",
            min_value=450,
            max_value=1600,
            value=st.session_state.slm_cal_wavelength,
            step=1,
        )

        st.divider()

        # Camera settings
        st.subheader("相机参数")
        st.session_state.slm_cal_exposure_time = st.number_input(
            "曝光时间 (ms)",
            min_value=1,
            max_value=1000,
            value=st.session_state.slm_cal_exposure_time,
            step=1,
        )

        st.divider()

        # Calibration parameters
        st.subheader("标定参数")

        st.session_state.slm_cal_grating_period = st.number_input(
            "光栅周期 (像素)",
            min_value=1,
            max_value=100,
            value=st.session_state.slm_cal_grating_period,
            step=1,
            help="闪耀光栅的周期，影响衍射效率",
        )

        col1, col2 = st.columns(2)
        with col1:
            st.session_state.slm_cal_grayscale_min = st.number_input(
                "灰度最小值",
                min_value=0,
                max_value=1023,
                value=st.session_state.slm_cal_grayscale_min,
                step=10,
            )
        with col2:
            st.session_state.slm_cal_grayscale_max = st.number_input(
                "灰度最大值",
                min_value=0,
                max_value=1023,
                value=st.session_state.slm_cal_grayscale_max,
                step=10,
            )

        st.session_state.slm_cal_step = st.number_input(
            "扫描步长",
            min_value=1,
            max_value=100,
            value=st.session_state.slm_cal_step,
            step=1,
        )

        st.session_state.slm_cal_n_samples = st.number_input(
            "采样次数",
            min_value=1,
            max_value=10,
            value=st.session_state.slm_cal_n_samples,
            step=1,
        )

        st.session_state.slm_cal_fine_step = st.number_input(
            "精细搜索步长",
            min_value=1,
            max_value=20,
            value=st.session_state.slm_cal_fine_step,
            step=1,
        )

        st.session_state.slm_cal_fine_range = st.number_input(
            "精细搜索范围",
            min_value=10,
            max_value=200,
            value=st.session_state.slm_cal_fine_range,
            step=10,
        )

        st.divider()

        # ROI settings
        st.subheader("感兴趣区域 (ROI)")
        roi_auto = st.checkbox(
            "自动检测光斑中心",
            value=st.session_state.get("slm_cal_roi_auto", True),
            key="slm_cal_roi_auto",
            help="启用后自动寻找最亮点作为ROI中心",
        )

        if roi_auto:
            st.caption("将自动检测光斑中心")
        else:
            col1, col2 = st.columns(2)
            with col1:
                roi_x = st.number_input(
                    "ROI中心 X",
                    min_value=0,
                    max_value=1920,
                    value=960,
                    step=10,
                    key="slm_cal_roi_x_input",
                )
            with col2:
                roi_y = st.number_input(
                    "ROI中心 Y",
                    min_value=0,
                    max_value=1080,
                    value=540,
                    step=10,
                    key="slm_cal_roi_y_input",
                )
            st.session_state.slm_cal_roi_center = (roi_x, roi_y)

        st.session_state.slm_cal_roi_size = st.number_input(
            "ROI大小",
            min_value=10,
            max_value=500,
            value=st.session_state.slm_cal_roi_size,
            step=10,
        )

        st.divider()

        # Auto-exposure settings
        st.subheader("自动曝光")
        st.session_state.slm_cal_auto_exposure = st.checkbox(
            "启用自动曝光",
            value=st.session_state.slm_cal_auto_exposure,
            help="在标定前自动调整相机曝光时间",
        )

        if st.session_state.slm_cal_auto_exposure:
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.slm_cal_target_min = st.number_input(
                    "目标最小值",
                    min_value=10,
                    max_value=200,
                    value=st.session_state.slm_cal_target_min,
                    step=5,
                )
            with col2:
                st.session_state.slm_cal_target_max = st.number_input(
                    "目标最大值",
                    min_value=10,
                    max_value=250,
                    value=st.session_state.slm_cal_target_max,
                    step=5,
                )

        st.divider()

        # Device connection
        st.subheader("设备连接")

        st.write("**SLM**")
        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state.slm_cal_slm_connected:
                if st.button("连接SLM", type="primary", key="slm_cal_connect_slm"):
                    connect_slm()
            else:
                st.success("SLM已连接")
                if st.button("断开SLM", key="slm_cal_disconnect_slm"):
                    disconnect_slm()

        st.write("**相机**")
        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state.slm_cal_camera_connected:
                if st.button("连接相机", type="primary", key="slm_cal_connect_cam"):
                    connect_camera()
            else:
                st.success("相机已连接")
                if st.button("断开相机", key="slm_cal_disconnect_cam"):
                    disconnect_camera()


def render_calibration() -> None:
    """Render calibration mode UI."""
    st.header("SLM相位-灰度标定")
    st.markdown("""
    通过闪耀光栅法测量SLM的2π相位对应的灰度值。
    
    **原理**:
    1. 在SLM上显示不同灰度深度的闪耀光栅
    2. 用相机测量衍射光斑强度
    3. 找到最大衍射效率对应的灰度值即为2π相位值
    """)

    # Check device connection
    if (
        not st.session_state.slm_cal_slm_connected
        or not st.session_state.slm_cal_camera_connected
    ):
        st.warning("请先连接SLM和相机设备")
        return

    # Show current configuration
    n_points = (
        st.session_state.slm_cal_grayscale_max - st.session_state.slm_cal_grayscale_min
    ) // st.session_state.slm_cal_step + 1
    st.info(f"""
    配置: 灰度范围 [{st.session_state.slm_cal_grayscale_min}, {st.session_state.slm_cal_grayscale_max}],
    步长 {st.session_state.slm_cal_step}, 预计 {n_points} 个测量点
    """)

    # Start calibration button
    progress_bar = st.empty()
    status_text = st.empty()
    chart_placeholder = st.empty()

    if st.button("开始标定", type="primary", disabled=st.session_state.slm_cal_running):
        st.session_state.slm_cal_running = True
        st.session_state.slm_cal_progress = {
            "percent": 0,
            "message": "准备中...",
            "current_gs": None,
            "intensity": None,
        }

        # Setup progress file for thread communication
        progress_file = (
            Path(st.session_state.slm_cal_storage_dir)
            / f"progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        progress_file.parent.mkdir(parents=True, exist_ok=True)

        # ROI center (stored as tuple or None for auto-detect)
        roi_center = st.session_state.get("slm_cal_roi_center")

        calib = SantecSLM200Calibrator(
            slm=st.session_state.slm_cal_slm,
            camera=st.session_state.slm_cal_camera,
            grating_period=st.session_state.slm_cal_grating_period,
            roi_center=roi_center,
            roi_size=(
                st.session_state.slm_cal_roi_size,
                st.session_state.slm_cal_roi_size,
            ),
        )

        # Run calibration in background thread
        import threading

        thread = threading.Thread(
            target=_calibration_worker,
            args=(
                calib,
                (
                    st.session_state.slm_cal_grayscale_min,
                    st.session_state.slm_cal_grayscale_max,
                ),
                st.session_state.slm_cal_step,
                st.session_state.slm_cal_n_samples,
                True,
                st.session_state.slm_cal_fine_step,
                st.session_state.slm_cal_fine_range,
                True,
                st.session_state.slm_cal_auto_exposure,
                st.session_state.slm_cal_target_min,
                st.session_state.slm_cal_target_max,
                str(progress_file),
            ),
            daemon=True,
        )
        thread.start()
        st.rerun()

    # Poll progress if running
    if st.session_state.slm_cal_running:
        progress = st.session_state.slm_cal_progress

        if "status" in progress and progress["status"] == "complete":
            status_text.success("标定完成!")

        elif "status" in progress and progress["status"] == "error":
            status_text.error(progress.get("message", "标定失败"))

        else:
            percent = progress.get("percent", 0)
            message = progress.get("message", "标定中...")
            current_gs = progress.get("current_gs")
            intensity = progress.get("intensity")

            if percent > 0:
                progress_bar.progress(min(percent / 100.0, 1.0), text=message)

            if current_gs is not None:
                st.caption(
                    f"当前灰度值: {current_gs}, 强度: {intensity:.2f}"
                    if intensity
                    else f"当前灰度值: {current_gs}"
                )

        time.sleep(0.5)
        st.rerun()

    # Display result if complete
    if st.session_state.slm_cal_result is not None:
        result = st.session_state.slm_cal_result

        st.divider()
        st.subheader("标定结果")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("2π相位灰度值", result.grayscale_2pi)
        with col2:
            st.metric("波长", f"{result.wavelength_nm} nm")
        with col3:
            st.metric("SLM型号", result.slm_model)

        # Plot calibration curve
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))

            # Raw data
            gs_vals = result.grayscale_values
            intensities = result.intensities

            ax.plot(gs_vals, intensities, "b-", label="衍射效率", linewidth=2)
            ax.axvline(
                x=result.grayscale_2pi,
                color="r",
                linestyle="--",
                label=f"2π相位 = {result.grayscale_2pi}",
            )

            ax.set_xlabel("灰度值", fontsize=12)
            ax.set_ylabel("衍射光强 (a.u.)", fontsize=12)
            ax.set_title("SLM相位-灰度标定曲线", fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

        except ImportError:
            st.warning("matplotlib未安装，无法显示曲线图")

        # Save button
        storage_dir = Path(st.session_state.slm_cal_storage_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)

        if st.button("保存结果"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = (
                storage_dir
                / f"slm_calibration_{result.wavelength_nm}nm_{timestamp}.json"
            )
            result.save(save_path)
            st.success(f"结果已保存到: {save_path}")


def render_load_view() -> None:
    """Render load and view mode UI."""
    st.header("加载校准结果")

    storage_dir = Path(st.session_state.slm_cal_storage_dir)

    if not storage_dir.exists():
        st.warning(f"存储目录不存在: {storage_dir}")
        return

    json_files = list(storage_dir.glob("*.json"))

    if not json_files:
        st.warning("未找到校准结果文件")
        return

    file_options = [f.stem for f in json_files]
    selected = st.selectbox("选择校定结果", file_options)

    if selected and st.button("加载", type="primary"):
        try:
            file_path = storage_dir / f"{selected}.json"
            result = CalibrationResult.load(file_path)
            st.session_state.slm_cal_result = result
            st.success(f"已加载: {selected}")
        except Exception as e:
            st.error(f"加载失败: {e}")

    if st.session_state.slm_cal_result is not None:
        result = st.session_state.slm_cal_result

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("2π相位灰度值", result.grayscale_2pi)
        with col2:
            st.metric("波长", f"{result.wavelength_nm} nm")
        with col3:
            st.metric("SLM型号", result.slm_model)

        # Show curve
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(result.grayscale_values, result.intensities, "b-", linewidth=2)
            ax.axvline(
                x=result.grayscale_2pi,
                color="r",
                linestyle="--",
                label=f"2π相位 = {result.grayscale_2pi}",
            )
            ax.set_xlabel("灰度值")
            ax.set_ylabel("衍射光强 (a.u.)")
            ax.set_title("SLM相位-灰度标定曲线")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

        except ImportError:
            st.warning("matplotlib未安装")


def main() -> None:
    """Main entry point for SLM calibration Streamlit UI."""
    st.set_page_config(
        page_title="SLM LUT 标定",
        page_icon=":test_tube:",
        layout="wide",
    )

    # Initialize state
    _initialize_state()

    # Sidebar
    render_sidebar()

    # Main area - mode selection
    mode = st.radio(
        "操作模式",
        options=["calibrate", "load_view"],
        format_func=lambda x: "标定" if x == "calibrate" else "加载查看",
        horizontal=True,
    )

    if mode == "calibrate":
        render_calibration()
    else:
        render_load_view()


if __name__ == "__main__":
    main()
