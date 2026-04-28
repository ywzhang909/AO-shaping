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

import streamlit as st
import numpy as np
from pathlib import Path
import sys
import time
from datetime import datetime
from typing import Any

# Add src to path when running directly via Streamlit
SRC_ROOT = Path(__file__).resolve().parents[3]  # src/
PROJECT_ROOT = Path(__file__).resolve().parents[4]  # AO-shaping/

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from loguru import logger

# Import drivers
from ao_shaping.drivers.slm.zernike_slm import ZernikeSLM
from ao_shaping.drivers.wfs.thorlab_wfs import WFSManager

# Import calibration functions
from ao_shaping.optimizer.wf.zernike_response_matrix import (
    ZernikeResponseMatrixResult,
    calibrate_zernike_response_matrix,
    load_zernike_response_matrix,
    save_zernike_response_matrix,
    plot_response_matrix,
    DEFAULT_N_MAX,
    DEFAULT_MAGNITUDE,
    DEFAULT_N_AVERAGES,
    DEFAULT_N_CYCLES,
    DEFAULT_WAIT_TIME,
)

# Import utilities
from ao_shaping.utils.matrix_utils import calc_n_zernike_terms


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

    if "zrm_storage_dir" not in st.session_state:
        st.session_state.zrm_storage_dir = str(PROJECT_ROOT / "data" / "zernike_calibration")

    if "zrm_slm_wavelength" not in st.session_state:
        st.session_state.zrm_slm_wavelength = 1064

    if "zrm_slm_number" not in st.session_state:
        st.session_state.zrm_slm_number = 1

    if "zrm_slm_n_max" not in st.session_state:
        st.session_state.zrm_slm_n_max = 10

    # Calibration state
    if "zrm_calibration_result" not in st.session_state:
        st.session_state.zrm_calibration_result = None

    if "zrm_calibration_running" not in st.session_state:
        st.session_state.zrm_calibration_running = False

    if "zrm_current_mode" not in st.session_state:
        st.session_state.zrm_current_mode = "calibrate"


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
                if (n, m) in names:
                    valid[(n, m)] = names[(n, m)]
                else:
                    valid[(n, m)] = f"Z{n},{m}"
    return valid


def connect_slm() -> bool:
    """Connect to ZernikeSLM."""
    try:
        if st.session_state.zrm_slm is not None:
            st.session_state.zrm_slm.close()

        slm = ZernikeSLM(
            slm_number=st.session_state.zrm_slm_number,
            wavelength=st.session_state.zrm_slm_wavelength,
            n_max=st.session_state.zrm_slm_n_max,
        )
        slm.open()

        st.session_state.zrm_slm = slm
        st.session_state.zrm_slm_connected = True
        st.success(f"SLM {st.session_state.zrm_slm_number} connected")
        return True

    except Exception as e:
        st.error(f"SLM connection failed: {e}")
        logger.error(f"ZernikeSLM connection failed: {e}")
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
        logger.error(f"ZernikeSLM disconnect failed: {e}")


def connect_wfs() -> bool:
    """Connect to Thorlab WFS."""
    try:
        if st.session_state.zrm_wfs is not None:
            st.session_state.zrm_wfs.close()

        wfs = WFSManager()
        wfs.initialize()

        st.session_state.zrm_wfs = wfs
        st.session_state.zrm_wfs_connected = True
        st.success("WFS connected")
        return True

    except Exception as e:
        st.error(f"WFS connection failed: {e}")
        logger.error(f"WFS connection failed: {e}")
        return False


def disconnect_wfs() -> None:
    """Disconnect from Thorlab WFS."""
    try:
        if st.session_state.zrm_wfs is not None:
            st.session_state.zrm_wfs.close()
        st.session_state.zrm_wfs = None
        st.session_state.zrm_wfs_connected = False
        st.success("WFS disconnected")
    except Exception as e:
        st.error(f"WFS disconnect failed: {e}")
        logger.error(f"WFS disconnect failed: {e}")


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
            max_value=2.0,
            value=st.session_state.zrm_magnitude,
            step=0.05,
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

        st.divider()

        # SLM settings
        st.subheader("SLM设置")

        st.session_state.zrm_slm_number = st.number_input(
            "SLM编号",
            min_value=1,
            max_value=2,
            value=st.session_state.zrm_slm_number,
            step=1,
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

        st.divider()

        # Device connection
        st.subheader("设备连接")

        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state.zrm_slm_connected:
                if st.button("连接SLM", type="primary"):
                    connect_slm()
            else:
                st.success("SLM已连接")
                if st.button("断开SLM"):
                    disconnect_slm()

        with col2:
            if not st.session_state.zrm_wfs_connected:
                if st.button("连接WFS", type="primary"):
                    connect_wfs()
            else:
                st.success("WFS已连接")
                if st.button("断开WFS"):
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
    n_terms = calc_n_zernike_terms(st.session_state.zrm_n_max) - 1
    st.info(
        f"配置: n_max={st.session_state.zrm_n_max}, "
        f"magnitude={st.session_state.zrm_magnitude}λ, "
        f"cycles={st.session_state.zrm_n_cycles}, "
        f"averages={st.session_state.zrm_n_averages}, "
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

    # Start calibration button
    if st.button("开始校准", type="primary", disabled=st.session_state.zrm_calibration_running):
        if filename.strip() == "":
            st.error("请输入文件名")
            return

        st.session_state.zrm_calibration_running = True

        try:
            result = calibrate_zernike_response_matrix(
                zslm=st.session_state.zrm_slm,
                wfs=st.session_state.zrm_wfs,
                n_max=st.session_state.zrm_n_max,
                magnitude=st.session_state.zrm_magnitude,
                n_cycles=st.session_state.zrm_n_cycles,
                n_averages=st.session_state.zrm_n_averages,
                wait_time=st.session_state.zrm_wait_time,
                verbose=True,
            )

            # Save result
            save_zernike_response_matrix(result, str(save_path))
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
                plot_response_matrix(result, save_path.parent)
                st.success("可视化图表已生成")
            except Exception as e:
                logger.warning(f"可视化生成失败: {e}")

        except Exception as e:
            st.error(f"校准失败: {e}")
            logger.error(f"Calibration failed: {e}")

        finally:
            st.session_state.zrm_calibration_running = False

    # Status
    if st.session_state.zrm_calibration_running:
        st.warning("校准进行中...")


def render_load_view_mode() -> None:
    """Render load and view mode UI."""
    st.header("加载并查看校准结果")

    # File selector
    storage_dir = Path(st.session_state.zrm_storage_dir)

    if not storage_dir.exists():
        st.warning(f"存储目录不存在: {storage_dir}")
        st.info("请在侧边栏设置正确的存储目录")
        return

    # Find available result files
    json_files = list(storage_dir.glob("*.json"))
    if not json_files:
        st.warning("未找到校准结果文件")
        return

    # Show available files
    file_options = [f.stem for f in json_files]
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

        st.caption(f"时间戳: {result.timestamp}")

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