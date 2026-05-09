"""
Zernike Response Matrix Calibration UI (Streamlit)

Features:
1. Calibrate SLM→WFS Zernike response matrix with configurable parameters
2. Real-time visualization during calibration
3. Load and view saved calibration results

Usage:
    streamlit run src/ao_shaping/gui/streamlit_helper/zernike_response_matrix_ui.py
"""

from __future__ import annotations

import sys
import types
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import streamlit as st

from loguru import logger

# Import drivers
from ao_shaping.drivers.slm.zernike_slm import ZernikeSLM
from ao_shaping.drivers.wfs.thorlab_wfs import WFSManager

from ao_shaping.optimizer.wf.zernike_response_matrix import (
    calibrate_zernike_response_matrix,
    load_zernike_response_matrix,
    save_zernike_response_matrix,
    DEFAULT_N_MAX,
    DEFAULT_MAGNITUDE,
    DEFAULT_N_AVERAGES,
    DEFAULT_N_CYCLES,
    DEFAULT_WAIT_TIME,
)
from ao_shaping.utils.matrix_utils import calc_n_zernike_terms

# Determine project root (file is at src/ao_shaping/gui/streamlit_helper/zernike_response_matrix_ui.py)
PROJECT_ROOT = Path(__file__).resolve().parents[4]  # AO-shaping/
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Patch miicam module before importing ccd package
if "miicam" not in sys.modules:
    sys.modules["miicam"] = types.ModuleType("miicam")

def _initialize_state() -> None:
    """Initialize session state variables."""
    # Device states
    if "zrm_slm" not in st.session_state:
        st.session_state.zrm_slm = None
        st.session_state.zrm_slm_connected = False

    if "zrm_wfs" not in st.session_state:
        st.session_state.zrm_wfs = None
        st.session_state.zrm_wfs_connected = False

    # Configuration (defaults)
    if "zrm_n_max" not in st.session_state:
        st.session_state.zrm_n_max = DEFAULT_N_MAX

    if "zrm_magnitude" not in st.session_state:
        st.session_state.zrm_magnitude = DEFAULT_MAGNITUDE

    if "zrm_n_cycles" not in st.session_state:
        st.session_state.zrm_n_cycles = DEFAULT_N_CYCLES

    if "zrm_n_averages" not in st.session_state:
        st.session_state.zrm_n_averages = DEFAULT_N_AVERAGES

    if "zrm_wait_time" not in st.session_state:
        st.session_state.zrm_wait_time = DEFAULT_WAIT_TIME

    if "zrm_excluded_piston" not in st.session_state:
        st.session_state.zrm_excluded_piston = True

    if "zrm_excluded_tip_tilt" not in st.session_state:
        st.session_state.zrm_excluded_tip_tilt = False

    if "zrm_compute_inverses" not in st.session_state:
        st.session_state.zrm_compute_inverses = True

    if "zrm_verbose" not in st.session_state:
        st.session_state.zrm_verbose = True

    if "zrm_storage_dir" not in st.session_state:
        st.session_state.zrm_storage_dir = str(PROJECT_ROOT / "data" / "zernike_calibration")

    if "zrm_slm_wavelength" not in st.session_state:
        st.session_state.zrm_slm_wavelength = 1064

    if "zrm_slm_number" not in st.session_state:
        st.session_state.zrm_slm_number = 1

    if "zrm_slm_selection" not in st.session_state:
        st.session_state.zrm_slm_selection = 1  # Default to SLM 1

    if "zrm_slm_n_max" not in st.session_state:
        st.session_state.zrm_slm_n_max = 10

    if "zrm_shift_x" not in st.session_state:
        st.session_state.zrm_shift_x = 0

    if "zrm_shift_y" not in st.session_state:
        st.session_state.zrm_shift_y = 0

    # SLM properties (populated after connection)
    if "zrm_slm_width" not in st.session_state:
        st.session_state.zrm_slm_width = None

    if "zrm_slm_height" not in st.session_state:
        st.session_state.zrm_slm_height = None

    if "zrm_slm_pixel_size_um" not in st.session_state:
        st.session_state.zrm_slm_pixel_size_um = None

    if "zrm_slm_bits" not in st.session_state:
        st.session_state.zrm_slm_bits = None

    # WFS settings
    if "zrm_wfs_mla_res" not in st.session_state:
        st.session_state.zrm_wfs_mla_res = "768"

    if "zrm_wfs_exp_time" not in st.session_state:
        st.session_state.zrm_wfs_exp_time = 0.01  # Valid default within [0.002, 86]

    # WFS auto-exposure display value (separate from actual exp_time used)
    if "zrm_wfs_exp_time_display" not in st.session_state:
        st.session_state.zrm_wfs_exp_time_display = None  # Will be set on first render

    # WFS auto-exposure toggle
    if "zrm_wfs_auto_exposure" not in st.session_state:
        st.session_state.zrm_wfs_auto_exposure = True  # Default to auto-exposure

    # WFS exposure time range (detected from device)
    if "zrm_wfs_exp_time_min" not in st.session_state:
        st.session_state.zrm_wfs_exp_time_min = 0.002  # Default from WFS driver

    if "zrm_wfs_exp_time_max" not in st.session_state:
        st.session_state.zrm_wfs_exp_time_max = 86.0  # Default from WFS driver

    if "zrm_wfs_pupil_diameter" not in st.session_state:
        st.session_state.zrm_wfs_pupil_diameter = 2.0

    if "zrm_wfs_pupil_center_x" not in st.session_state:
        st.session_state.zrm_wfs_pupil_center_x = 0.0

    if "zrm_wfs_pupil_center_y" not in st.session_state:
        st.session_state.zrm_wfs_pupil_center_y = 0.0

    # Calibration state
    if "zrm_calibration_result" not in st.session_state:
        st.session_state.zrm_calibration_result = None

    if "zrm_calibration_running" not in st.session_state:
        st.session_state.zrm_calibration_running = False

    if "zrm_current_mode" not in st.session_state:
        st.session_state.zrm_current_mode = "calibrate"

    # Progress tracking
    if "zrm_progress_file" not in st.session_state:
        # Create a unique progress file path in the storage directory
        progress_dir = Path(st.session_state.get("zrm_storage_dir", str(PROJECT_ROOT / "data" / "zernike_calibration")))
        progress_dir.mkdir(parents=True, exist_ok=True)
        st.session_state.zrm_progress_file = str(progress_dir / f"progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    # Calibration parameters
    if "zrm_verbose" not in st.session_state:
        st.session_state.zrm_verbose = True


def _get_zernike_name(n_max: int) -> dict[tuple[int, int], str]:
    """Generate Zernike name mapping for given n_max."""
    names = {
        (0, 0): "Piston",
        (1, -1): "Tip",
        (1, 1): "Tilt",
        (2, 0): "Defocus",
        (2, -2): "Astig 45°",
        (2, 2): "Astig 0°",
        (3, -1): "Coma Y",
        (3, 1): "Coma X",
        (3, -3): "Trefoil Y",
        (3, 3): "Trefoil X",
        (4, 0): "Spherical",
        (4, -2): "Sec Astig 45°",
        (4, 2): "Sec Astig 0°",
        (4, -4): "Tetrafoil Y",
        (4, 4): "Tetrafoil X",
    }
    # Filter to only valid modes for given n_max
    valid = {}
    for n in range(n_max + 1):
        for m in range(-n, n + 1):
            if (n - abs(m)) % 2 == 0:
                valid[n, m] = names.get((n, m), f"Z{n},{m}")
    return valid


def set_slm_shift() -> None:
    """Apply shift values to connected SLM."""
    try:
        slm = st.session_state.zrm_slm
        if slm is None:
            st.error("SLM未连接")
            return
        shift_x = st.session_state.zrm_shift_x
        shift_y = st.session_state.zrm_shift_y
        slm.set_shift(shift_x, shift_y)
        st.success(f"SLM shift set to ({shift_x}, {shift_y})")
    except Exception as e:
        st.error(f"设置shift失败: {e}")
        logger.exception(f"Failed to set SLM shift: {e}")


def connect_slm() -> bool:
    """Connect to ZernikeSLM."""
    try:
        if st.session_state.zrm_slm is not None:
            st.session_state.zrm_slm.close()

        # Use selected SLM number from dropdown
        slm_number = st.session_state.zrm_slm_selection

        slm = ZernikeSLM(
            slm_number=slm_number,
            wavelength=st.session_state.zrm_slm_wavelength,
            n_max=st.session_state.zrm_slm_n_max,
            shift_x=st.session_state.zrm_shift_x,
            shift_y=st.session_state.zrm_shift_y,
        )
        slm.open()

        # Store SLM properties for pattern generation and display
        st.session_state.zrm_slm = slm
        st.session_state.zrm_slm_connected = True
        st.session_state.zrm_slm_width = slm._slm.Panel_Res[0]
        st.session_state.zrm_slm_height = slm._slm.Panel_Res[1]
        st.session_state.zrm_slm_pixel_size_um = slm._slm.Pitch_um
        st.session_state.zrm_slm_bits = slm._slm.Gray_Scale_bits
        st.session_state.zrm_slm_number = slm_number  # Store the actual connected number

        st.success(f"SLM {slm_number} connected (wavelength={slm.wavelength}nm, resolution={st.session_state.zrm_slm_width}×{st.session_state.zrm_slm_height})")
        return True

    except Exception as e:
        st.error(f"SLM connection failed: {e}")
        logger.exception(f"ZernikeSLM connection failed: {e}")
        return False


def disconnect_slm() -> None:
    """Disconnect from ZernikeSLM."""
    try:
        if st.session_state.zrm_slm is not None:
            st.session_state.zrm_slm.close()
        st.session_state.zrm_slm = None
        st.session_state.zrm_slm_connected = False
        st.success("SLM disconnected")
    except Exception as e:
        st.error(f"SLM disconnect failed: {e}")
        logger.exception(f"ZernikeSLM disconnect failed: {e}")


def connect_wfs() -> bool:
    """Connect to Thorlab WFS with auto-retry on failure."""
    try:
        # Ensure previous connection is fully closed
        if st.session_state.zrm_wfs is not None:
            try:
                st.session_state.zrm_wfs.close()
            except Exception as e:
                logger.warning(f"Previous WFS close warning: {e}")
            st.session_state.zrm_wfs = None
            st.session_state.zrm_wfs_connected = False

        # Determine exposure time: auto-exposure uses 0, otherwise use configured value
        exp_time = 0.0 if st.session_state.zrm_wfs_auto_exposure else st.session_state.zrm_wfs_exp_time

        wfs = WFSManager(
            mla_index=st.session_state.zrm_wfs_mla_res,
            exp_time=exp_time,
            high_speed=False,
            pupil_diameter=st.session_state.zrm_wfs_pupil_diameter,
            pupil_center=(st.session_state.zrm_wfs_pupil_center_x, st.session_state.zrm_wfs_pupil_center_y),
        )
        wfs.initialize()
        min_exp, max_exp, _ = wfs.get_exposure_time_range()
        if max_exp > 0:
            st.session_state.zrm_wfs_exp_time_min = min_exp*1000
            st.session_state.zrm_wfs_exp_time_max = max_exp*1000
            logger.info(f"WFS exposure range: {min_exp*1000:.3f} ~ {max_exp*1000:.3f} ms")
        else:
            logger.warning("Failed to get WFS exposure range, using defaults")

        st.session_state.zrm_wfs = wfs
        st.session_state.zrm_wfs_connected = True
        mode_str = "自动曝光" if st.session_state.zrm_wfs_auto_exposure else f"手动曝光 {exp_time:.3f}ms"
        st.success(f"WFS connected ({mode_str})")
        return True
    except Exception as e:
        logger.exception(f"fail to connect wfs: {e}")
        st.error(f"fail to connect wfs: {e}")
        return False

def disconnect_wfs() -> None:
    """Disconnect from Thorlab WFS."""
    try:
        if st.session_state.zrm_wfs is not None:
            try:
                st.session_state.zrm_wfs.close()
            except Exception as e:
                logger.warning(f"WFS close warning: {e}")
            finally:
                st.session_state.zrm_wfs = None
                st.session_state.zrm_wfs_connected = False
        st.success("WFS disconnected")
    except Exception as e:
        st.error(f"WFS disconnect failed: {e}")
        logger.exception(f"Failed to disconnect WFS: {e}")


def _progress_callback(progress_data: dict) -> None:
    """Write progress data to JSON file for UI polling.

    Args:
        progress_data: Dictionary containing progress information with keys like:
            - current_mode: Current Zernike mode being calibrated
            - total_modes: Total number of modes to calibrate
            - percent: Progress percentage (0-100)
            - message: Status message
            - timestamp: ISO format timestamp
    """
    try:
        progress_file = st.session_state.get("zrm_progress_file")
        if progress_file:
            with open(progress_file, "w") as f:
                json.dump(progress_data, f)
    except Exception as e:
        logger.warning(f"Failed to write progress file: {e}")


def _run_calibration_thread(
    zslm,
    wfs,
    n_max,
    magnitude,
    n_cycles,
    n_averages,
    wait_time,
    excluded_piston,
    excluded_tip_tilt,
    compute_inverses,
    verbose,
    save_path,
) -> None:
    """Run calibration in a background thread and update progress.

    This function runs the calibration in a separate thread, writing progress
    updates to a JSON file that the UI can poll.
    """
    try:
        total_modes = calc_n_zernike_terms(n_max) - (1 if excluded_piston else 0) - (2 if excluded_tip_tilt else 0)

        # Progress callback wrapper (matches backend signature)
        def callback(mode_index: int, total_modes: int, mean_resp: np.ndarray, var_resp: np.ndarray) -> None:
            # Calculate progress percentage
            percent = ((mode_index + 1) / total_modes) * 100.0
            _progress_callback({
                "current_mode": mode_index,
                "total_modes": total_modes,
                "percent": percent,
                "message": f"Calibrating mode {mode_index + 1}/{total_modes}",
                "timestamp": datetime.now().isoformat(),
                "mean_response": mean_resp.tolist() if mean_resp is not None else None,
                "variance": var_resp.tolist() if var_resp is not None else None,
            })

        # Run calibration with progress callback
        result = calibrate_zernike_response_matrix(
            zslm=zslm,
            wfs=wfs,
            n_max=n_max,
            magnitude=magnitude,
            n_cycles=n_cycles,
            n_averages=n_averages,
            wait_time=wait_time,
            excluded_piston=excluded_piston,
            excluded_tip_tilt=excluded_tip_tilt,
            compute_inverses=compute_inverses,
            verbose=verbose,
            callback=callback if verbose else None,
        )

        # Save result
        save_zernike_response_matrix(result, str(save_path))

        # Write completion status
        _progress_callback({
            "current_mode": total_modes,
            "total_modes": total_modes,
            "percent": 100.0,
            "message": "Calibration complete!",
            "timestamp": datetime.now().isoformat(),
            "status": "complete",
            "result_path": str(save_path),
        })

        # Store result in session state (will be picked up on next rerun)
        st.session_state.zrm_calibration_result = result

    except Exception as e:
        logger.exception(f"Calibration thread error: {e}")
        _progress_callback({
            "percent": -1,
            "message": f"Calibration failed: {e}",
            "timestamp": datetime.now().isoformat(),
            "status": "error",
        })
    finally:
        st.session_state.zrm_calibration_running = False


def render_sidebar() -> None:
    """Render sidebar configuration."""
    with st.sidebar:
        st.header("Zernike Response Matrix")

        # Mode selection
        st.session_state.zrm_current_mode = st.radio(
            "Operation Mode",
            options=["calibrate", "load_view"],
            format_func=lambda x: "校准" if x == "calibrate" else "加载查看",
            horizontal=True,
        )

        st.divider()

        # Storage directory
        st.subheader("存储设置")
        st.session_state.zrm_storage_dir = st.text_input(
            "存储目录",
            value=st.session_state.zrm_storage_dir,
            help="校准结果保存目录",
        )

        st.divider()

        # Calibration parameters (only show in calibrate mode)
        st.subheader("校准参数")

        st.session_state.zrm_n_max = st.number_input(
            "Zernike最大阶数 (n_max)",
            min_value=1,
            max_value=10,
            value=st.session_state.zrm_n_max,
            step=1,
        )

        st.session_state.zrm_magnitude = st.number_input(
            "扰动幅度 (波长)",
            min_value=0.01,
            max_value=20.0,
            value=st.session_state.zrm_magnitude,
            format="%.3f",
        )

        st.session_state.zrm_n_cycles = st.number_input(
            "正负循环次数",
            min_value=1,
            max_value=5,
            value=st.session_state.zrm_n_cycles,
            step=1,
        )

        st.session_state.zrm_n_averages = st.number_input(
            "WFS平均次数",
            min_value=1,
            max_value=100,
            value=st.session_state.zrm_n_averages,
            step=1,
        )

        st.session_state.zrm_wait_time = st.number_input(
            "等待时间 (s)",
            min_value=0.01,
            max_value=5.0,
            value=st.session_state.zrm_wait_time,
            step=0.1,
            format="%.2f",
        )

        st.session_state.zrm_excluded_piston = st.checkbox(
            "排除Piston (Z1)",
            value=st.session_state.zrm_excluded_piston,
            help="排除Zernike第1项（Piston）的校准，通常不需要校准Piston模式",
        )

        st.session_state.zrm_excluded_tip_tilt = st.checkbox(
            "排除Tip/Tilt (Z2, Z3)",
            value=st.session_state.zrm_excluded_tip_tilt,
            help="排除Zernike第2、3项（Tip/Tilt）的校准，通常由光路对准补偿",
        )

        st.session_state.zrm_compute_inverses = st.checkbox(
            "计算逆矩阵",
            value=st.session_state.zrm_compute_inverses,
            help="计算并保存响应矩阵的逆矩阵，用于后续波前校正",
        )

        st.session_state.zrm_verbose = st.checkbox(
            "显示详细进度",
            value=st.session_state.zrm_verbose,
            help="显示校准过程中的详细日志和进度信息",
        )

        st.divider()

        # WFS settings (parameters applied on connect)
        st.subheader("WFS参数")

        st.session_state.zrm_wfs_mla_res = st.selectbox(
            "MLA分辨率",
            options=[0, 1, 2, 3, 4],
            format_func=lambda x: {0: "1280x1024", 1: "1024x1024", 2: "768x768", 3: "512x512", 4: "320x320"}.get(x, str(x)),
            index=2,  # Default Res768
        )

        # Show current exposure range (default or detected)
        exp_min = st.session_state.get("zrm_wfs_exp_time_min", 0.002)
        exp_max = st.session_state.get("zrm_wfs_exp_time_max", 86.0)
        st.caption(f"设备曝光范围: {exp_min:.3f} ~ {exp_max:.3f} ms")

        # Auto-exposure toggle
        auto_exp = st.checkbox(
            "自动曝光",
            value=st.session_state.zrm_wfs_auto_exposure,
            help="开启时使用WFS自动曝光，曝光时间设置无效",
        )
        # Update session state only if changed (avoid unnecessary resets)
        if auto_exp != st.session_state.zrm_wfs_auto_exposure:
            st.session_state.zrm_wfs_auto_exposure = auto_exp
            # Only reset to 0.0 when enabling auto-exposure (for driver),
            # but keep a separate display value
            if auto_exp:
                st.session_state.zrm_wfs_exp_time_display = 0.0

        # Determine display value: use stored display value if in auto mode, else use actual exp_time
        if st.session_state.zrm_wfs_auto_exposure:
            # When auto, show the minimum valid value (since actual value is 0.0 for driver)
            display_value = exp_min
            st.session_state.zrm_wfs_exp_time_display = exp_min
        else:
            display_value = st.session_state.zrm_wfs_exp_time
            # Ensure manual value is within [exp_min, exp_max] for UI safety
            if display_value < exp_min:
                display_value = exp_min
                st.session_state.zrm_wfs_exp_time = exp_min
            elif display_value > exp_max:
                display_value = exp_max
                st.session_state.zrm_wfs_exp_time = exp_max
            st.session_state.zrm_wfs_exp_time_display = display_value

        # Exposure time input (disabled when auto-exposure is on)
        new_exp = st.number_input(
            "曝光时间 (ms)" + (" (自动)" if st.session_state.zrm_wfs_auto_exposure else ""),
            min_value=float(exp_min),
            max_value=float(exp_max),
            value=float(display_value),
            step=0.1,
            format="%.3f",
            help="自动曝光模式下此设置无效" if st.session_state.zrm_wfs_auto_exposure else "手动曝光时间，0=自动曝光",
            disabled=st.session_state.zrm_wfs_auto_exposure,
        )

        # Update exposure time state if not auto and value changed
        if not st.session_state.zrm_wfs_auto_exposure:
            st.session_state.zrm_wfs_exp_time = new_exp
        # Clamp to device range (should be enforced by number_input but ensure safety)
        st.session_state.zrm_wfs_exp_time = min(
            max(st.session_state.zrm_wfs_exp_time, float(exp_min)),
            float(exp_max)
        )

        st.session_state.zrm_wfs_pupil_diameter = st.number_input(
            "瞳孔直径 (mm)",
            min_value=0.5,
            max_value=10.0,
            value=st.session_state.zrm_wfs_pupil_diameter,
            step=0.1,
            format="%.1f"
        )

        col1, col2 = st.columns(2)
        with col1:
            st.session_state.zrm_wfs_pupil_center_x = st.number_input("瞳孔中心 X", value=st.session_state.zrm_wfs_pupil_center_x, step=0.1, format="%.1f")
        with col2:
            st.session_state.zrm_wfs_pupil_center_y = st.number_input("瞳孔中心 Y", value=st.session_state.zrm_wfs_pupil_center_y, step=0.1, format="%.1f")

        st.divider()

        # SLM settings (parameters applied on connect)
        st.subheader("SLM参数")

        st.session_state.zrm_slm_selection = st.selectbox(
            "选择SLM编号",
            options=[1, 2],
            index=st.session_state.get("zrm_slm_selection", 1) - 1,
            format_func=lambda x: f"SLM {x}",
            help="选择要连接的SLM设备编号",
        )

        st.session_state.zrm_slm_wavelength = st.number_input(
            "波长 (nm)",
            min_value=450,
            max_value=1600,
            value=st.session_state.zrm_slm_wavelength,
            step=1,
        )

        st.session_state.zrm_slm_n_max = st.number_input(
            "SLM Zernike阶数",
            min_value=1,
            max_value=10,
            value=st.session_state.zrm_slm_n_max,
            step=1,
        )

        st.session_state.zrm_shift_x = st.number_input(
            "SLM Shift X (像素)",
            min_value=-500,
            max_value=500,
            value=st.session_state.zrm_shift_x,
            step=1,
        )

        st.session_state.zrm_shift_y = st.number_input(
            "SLM Shift Y (像素)",
            min_value=-500,
            max_value=500,
            value=st.session_state.zrm_shift_y,
            step=1,
        )

        # Apply shift button (only enabled when SLM connected)
        if st.session_state.zrm_slm_connected:
            if st.button("应用Shift设置", key="zrm_apply_shift", type="secondary"):
                set_slm_shift()
        else:
            st.caption("连接SLM后可设置shift")

        st.divider()

        # Device connection
        st.subheader("设备连接")

        # SLM connection section
        st.write("**SLM**")
        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state.zrm_slm_connected:
                if st.button("连接SLM", type="primary", key="zrm_connect_slm"):
                    connect_slm()
            else:
                st.success("SLM已连接")
                if st.button("断开SLM", key="zrm_disconnect_slm"):
                    disconnect_slm()

        # Display SLM info when connected
        if st.session_state.zrm_slm_connected:
            width = st.session_state.get("zrm_slm_width")
            height = st.session_state.get("zrm_slm_height")
            pixel_size = st.session_state.get("zrm_slm_pixel_size_um")
            bits = st.session_state.get("zrm_slm_bits")
            if width and height:
                st.caption(f"分辨率: {width}×{height}")
            if pixel_size:
                st.caption(f"像素尺寸: {pixel_size} μm")
            if bits:
                st.caption(f"Bit: {bits}")

        st.divider()

        # WFS connection section
        st.write("**WFS**")
        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state.zrm_wfs_connected:
                if st.button("连接WFS", type="primary", key="zrm_connect_wfs"):
                    connect_wfs()
            else:
                st.success("WFS已连接")
                if st.button("断开WFS", key="zrm_disconnect_wfs"):
                    disconnect_wfs()

def render_calibrate_mode() -> None:
    """Render calibration mode UI."""
    st.header("Zernike响应矩阵校准")
    st.markdown("通过逐一施加各阶Zernike相位，测量对应的WFS响应，建立响应矩阵。")

    # Check device connection
    if not st.session_state.zrm_slm_connected or not st.session_state.zrm_wfs_connected:
        st.warning("请先在侧边栏连接SLM和WFS设备")
        return

    # Show configuration summary
    n_terms = calc_n_zernike_terms(st.session_state.zrm_n_max) - (1 if st.session_state.zrm_excluded_piston else 0) - (2 if st.session_state.zrm_excluded_tip_tilt else 0)
    st.info(
        f"配置: n_max={st.session_state.zrm_n_max}, "
        f"magnitude={st.session_state.zrm_magnitude}λ, "
        f"cycles={st.session_state.zrm_n_cycles}, "
        f"averages={st.session_state.zrm_n_averages}, "
        f"排除piston={st.session_state.zrm_excluded_piston}, "
        f"排除tip/tilt={st.session_state.zrm_excluded_tip_tilt}, "
        f"预计校准 {n_terms} 个模式"
    )

    # Auto-save path
    storage_dir = Path(st.session_state.zrm_storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)

    # Generate default filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_filename = f"zernike_response_{st.session_state.zrm_n_max}_{timestamp}"

    # File name input
    filename = st.text_input(
        "保存文件名 (不含扩展名)",
        value=default_filename,
        help="校准结果将保存为 {filename}.response.npy 等",
    )

    save_path = storage_dir / filename

    # Progress bar placeholder (created once, reused during polling)
    progress_bar = st.empty()
    status_text = st.empty()
    plot_placeholder = st.empty()

    # Start calibration button
    if st.button("开始校准", type="primary", disabled=st.session_state.zrm_calibration_running):
        if filename.strip() == "":
            st.error("请输入文件名")
            return

        # Initialize progress file
        progress_file = Path(st.session_state.zrm_storage_dir) / f"progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        st.session_state.zrm_progress_file = str(progress_file)

        # Clear any old progress data
        if progress_file.exists():
            progress_file.unlink()

        st.session_state.zrm_calibration_running = True

        # Start calibration in background thread
        import threading

        calibration_thread = threading.Thread(
            target=_run_calibration_thread,
            args=(
                st.session_state.zrm_slm,
                st.session_state.zrm_wfs,
                st.session_state.zrm_n_max,
                st.session_state.zrm_magnitude,
                st.session_state.zrm_n_cycles,
                st.session_state.zrm_n_averages,
                st.session_state.zrm_wait_time,
                st.session_state.zrm_excluded_piston,
                st.session_state.zrm_excluded_tip_tilt,
                st.session_state.zrm_compute_inverses,
                st.session_state.zrm_verbose,
                save_path,
            ),
            daemon=True,
        )
        calibration_thread.start()

        # Trigger rerun to start polling
        st.rerun()

    # Polling: Check progress if calibration is running
    if st.session_state.zrm_calibration_running:
        status_text.warning("校准进行中...")

        # Read progress from JSON file
        progress_file = st.session_state.get("zrm_progress_file")
        if progress_file and Path(progress_file).exists():
            try:
                with open(progress_file) as f:
                    progress_data = json.load(f)

                percent = progress_data.get("percent", 0)
                message = progress_data.get("message", "校准进行中...")
                current_mode = progress_data.get("current_mode", 0)
                total_modes = progress_data.get("total_modes", 1)
                mode_name = progress_data.get("mode_name", "")

                # Update progress bar
                if percent >= 0:
                    progress_bar.progress(
                        min(percent / 100.0, 1.0),
                        text=f"{message} ({percent:.1f}%)" if mode_name else message,
                    )
                else:
                    status_text.error(message)

                # Check for completion
                if progress_data.get("status") == "complete":
                    status_text.success("校准完成!")
                    progress_bar.empty()

                    # Load and display result
                    try:
                        from ao_shaping.optimizer.wf.zernike_response_matrix import load_zernike_response_matrix

                        result = load_zernike_response_matrix(str(save_path))
                        st.session_state.zrm_calibration_result = result

                        st.success(f"校准完成! 结果已保存到: {save_path.parent}")

                        # Display summary
                        st.subheader("校准结果摘要")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("矩阵形状", f"{result.matrix.shape}")
                        with col2:
                            st.metric("平均方差", f"{result.mean_variance:.6f}")
                        with col3:
                            st.metric("最大方差", f"{result.max_variance:.6f}")
                        with col4:
                            st.metric("条件数", f"{result.condition_number:.2e}" if result.condition_number else "N/A")

                        # Auto-plot
                        try:
                            from ao_shaping.optimizer.wf.zernike_response_matrix import plot_response_matrix

                            plot_response_matrix(result, save_path.parent)
                            st.success("可视化图表已生成")
                        except Exception as e:
                            logger.warning(f"可视化生成失败: {e}")

                    except Exception as e:
                        st.error(f"加载结果失败: {e}")
                        logger.exception(f"Failed to load calibration result: {e}")

                    st.session_state.zrm_calibration_running = False

                    # Clean up progress file
                    try:
                        if progress_file and Path(progress_file).exists():
                            Path(progress_file).unlink()
                    except Exception:
                        pass

                elif progress_data.get("status") == "error":
                    status_text.error(f"校准失败: {message}")
                    progress_bar.empty()
                    st.session_state.zrm_calibration_running = False

            except json.JSONDecodeError:
                # File might be partially written, retry on next poll
                pass
            except Exception as e:
                logger.warning(f"Failed to read progress file: {e}")

        # Rerun to poll again (with a small delay to avoid excessive reruns)
        import time
        time.sleep(0.5)
        st.rerun()

    # Display result if already loaded (and not currently running)
    if not st.session_state.zrm_calibration_running and st.session_state.zrm_calibration_result is not None:
        result = st.session_state.zrm_calibration_result

        st.subheader("校准结果摘要")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("矩阵形状", f"{result.matrix.shape}")
        with col2:
            st.metric("平均方差", f"{result.mean_variance:.6f}")
        with col3:
            st.metric("最大方差", f"{result.max_variance:.6f}")
        with col4:
            st.metric("条件数", f"{result.condition_number:.2e}" if result.condition_number else "N/A")

def render_load_view_mode() -> None:
    """Render load and view mode UI."""
    st.header("加载并查看校准结果")

    # File selector
    storage_dir = Path(st.session_state.zrm_storage_dir)

    if not storage_dir.exists():
        st.warning(f"存储目录不存在: {storage_dir}")
        st.info("请在侧边栏设置正确的存储目录")
        return

    # Find available result files (HDF5 format from backend)
    h5_files = list(storage_dir.glob("*.h5"))
    if not h5_files:
        st.warning("未找到校准结果文件 (.h5)")
        return

    # Show available files
    file_options = [f.stem for f in h5_files]
    selected_file = st.selectbox("选择校准结果", file_options)

    if selected_file:
        file_path = storage_dir / selected_file

        # Load button
        if st.button("加载", type="primary"):
            try:
                result = load_zernike_response_matrix(str(file_path))
                st.session_state.zrm_calibration_result = result
                st.success(f"已加载: {selected_file}")
            except Exception as e:
                st.error(f"加载失败: {e}")
                return

    # Display result if loaded
    if st.session_state.zrm_calibration_result is not None:
        result = st.session_state.zrm_calibration_result

        st.divider()
        st.subheader("校准结果")

        # Metadata
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("n_max", result.n_max)
        with col2:
            st.metric("magnitude", f"{result.magnitude}λ")
        with col3:
            st.metric("WFS terms", result.n_wfs_terms)
        with col4:
            st.metric("SLM terms", result.n_slm_terms)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("n_averages", result.n_averages)
        with col2:
            st.metric("n_cycles", result.n_cycles)
        with col3:
            st.metric("平均方差", f"{result.mean_variance:.6f}")
        with col4:
            st.metric("条件数", f"{result.condition_number:.2e}" if result.condition_number else "N/A")

        st.caption(
            f"时间戳: {result.timestamp} | "
            f"排除piston: {result.excluded_piston} | "
            f"排除tip/tilt: {result.excluded_tip_tilt}"
        )

        # Visualization
        st.divider()
        st.subheader("可视化")

        # Response matrix heatmap
        try:
            import matplotlib.pyplot as plt

            # Response matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(result.matrix, aspect="auto", cmap="RdBu_r")
            ax.set_xlabel("SLM Zernike Mode Index")
            ax.set_ylabel("WFS Zernike Mode Index")
            ax.set_title(f"Response Matrix (n_max={result.n_max})")
            fig.colorbar(im, ax=ax, label="Response")
            st.pyplot(fig)

            # Variance matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(result.variance_matrix, aspect="auto", cmap="YlOrRd")
            ax.set_xlabel("SLM Zernike Mode Index")
            ax.set_ylabel("WFS Zernike Mode Index")
            ax.set_title(f"Variance Matrix (mean={result.mean_variance:.6f})")
            fig.colorbar(im, ax=ax, label="Variance")
            st.pyplot(fig)

            # Per-mode variance
            col_var = np.mean(result.variance_matrix, axis=0)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(range(len(col_var)), col_var)
            ax.set_xlabel("SLM Zernike Mode Index")
            ax.set_ylabel("Mean Variance")
            ax.set_title("Measurement Stability per Mode")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # SVD singular values (if available)
            if result.pinv_matrix is not None:
                _, s, _ = np.linalg.svd(result.matrix)
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(s, "o-")
                ax.set_xlabel("Singular Value Index")
                ax.set_ylabel("Singular Value")
                ax.set_title(f"SVD Singular Values (condition={result.condition_number:.2e})")
                ax.set_yscale("log")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

            # Subaperture mask visualization
            if result.subaperture_mask is not None:
                st.divider()
                st.subheader("子孔径掩膜")
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(result.subaperture_mask, cmap="gray")
                ax.set_title(f"Valid Subapertures: {np.sum(result.subaperture_mask)}/{result.subaperture_mask.size}")
                fig.colorbar(im, ax=ax, label="Valid")
                st.pyplot(fig)

            # Deviation response matrix visualization
            if result.deviation_response_matrix is not None:
                st.divider()
                st.subheader("子孔径斜率响应矩阵")
                dev_matrix = result.deviation_response_matrix
                n_spots = dev_matrix.shape[0] // 2
                st.info(f"子孔径斜率维度: {n_spots} spots (X+Y)")
                fig, ax = plt.subplots(figsize=(12, 8))
                im = ax.imshow(dev_matrix, aspect="auto", cmap="RdBu_r")
                ax.set_xlabel("SLM Zernike Mode Index")
                ax.set_ylabel("Subaperture Slope Index (X then Y)")
                ax.set_title(f"Deviation Response Matrix ({n_spots}×2 spots)")
                fig.colorbar(im, ax=ax, label="Slope Response")
                st.pyplot(fig)

            # Amplitude optimization visualization
            if result.amplitude_optimization is not None:
                st.divider()
                st.subheader("幅度优化结果")
                opt_data = result.amplitude_optimization
                n_modes = len(opt_data)
                st.info(f"优化了 {n_modes} 个模式的幅度")
                optimal_amps = [opt_data[i]["optimal_amplitude"] for i in range(n_modes)]
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(range(n_modes), optimal_amps)
                ax.set_xlabel("SLM Zernike Mode Index")
                ax.set_ylabel("Optimal Amplitude (λ)")
                ax.set_title("Optimal Perturbation Amplitude per Mode")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                with st.expander("查看详细优化数据"):
                    for mode_idx, diag in opt_data.items():
                        st.write(f"Mode {mode_idx}: optimal={diag['optimal_amplitude']:.4f}λ, best_idx={diag['best_idx']}")

        except ImportError:
            st.warning("matplotlib未安装，无法生成可视化")
        except Exception as e:
            st.error(f"可视化生成失败: {e}")


def main():
    st.set_page_config(
        page_title="Zernike响应矩阵校准",
        page_icon="🔬",
        layout="wide",
    )

    # Initialize state
    _initialize_state()

    # Render sidebar
    render_sidebar()

    # Render main area based on mode
    if st.session_state.zrm_current_mode == "calibrate":
        render_calibrate_mode()
    else:
        render_load_view_mode()


if __name__ == "__main__":
    main()
