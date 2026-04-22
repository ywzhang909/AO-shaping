import streamlit as st
import numpy as np
from pathlib import Path
import sys
from loguru import logger
from typing import Any

# Add the src directory to the path when running this file directly via Streamlit.
SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200
from ao_shaping.drivers.slm.slm_pattern_helper import PatternHelper, SLMPatternHelper

# Global pattern helpers (will be recreated per-SLM based on resolution)
# Note: Resolution and bit depth now come from the SLM object when generating patterns


def _initialize_slm_state() -> None:
    if "slm1" not in st.session_state:
        st.session_state.slm1 = None
        st.session_state.slm1_connected = False
        st.session_state.slm1_wavelength = 1064
        st.session_state.slm1_video_mode = 0
        st.session_state.slm1_next_memory = 1
        st.session_state.slm1_phase_preview = None
        st.session_state.slm1_phase_source = "暂无"

    if "slm2" not in st.session_state:
        st.session_state.slm2 = None
        st.session_state.slm2_connected = False
        st.session_state.slm2_wavelength = 1064
        st.session_state.slm2_video_mode = 0
        st.session_state.slm2_next_memory = 1
        st.session_state.slm2_phase_preview = None
        st.session_state.slm2_phase_source = "暂无"

    for cam_num in (1, 2):
        prefix = f"cam{cam_num}"
        if prefix not in st.session_state:
            st.session_state[prefix] = None
            st.session_state[f"{prefix}_connected"] = False
            st.session_state[f"{prefix}_driver"] = "MIICAM"
            st.session_state[f"{prefix}_id"] = cam_num - 1
            st.session_state[f"{prefix}_exposure_ms"] = 20
            st.session_state[f"{prefix}_exposure_min_ms"] = 1
            st.session_state[f"{prefix}_exposure_max_ms"] = 1000


def _phase_to_preview(phase_gray: np.ndarray) -> np.ndarray:
    normalized = phase_gray.astype(np.float32) / max(
        SantecSLM200.MAX_GRAYSCALE_VALUE, 1
    )
    return np.clip(normalized, 0.0, 1.0)


def refresh_phase_preview(slm_num: int) -> None:
    phase_key = f"slm{slm_num}_phase_preview"
    source_key = f"slm{slm_num}_phase_source"
    slm = st.session_state.get(f"slm{slm_num}")
    if slm is None:
        st.session_state[phase_key] = None
        st.session_state[source_key] = "暂无"
        return

    phase_gray, source = slm.get_displayed_phase()
    st.session_state[phase_key] = (
        None if phase_gray is None else _phase_to_preview(phase_gray)
    )
    st.session_state[source_key] = source


def render_phase_preview(slm_num: int) -> None:
    preview = st.session_state.get(f"slm{slm_num}_phase_preview")
    source = st.session_state.get(f"slm{slm_num}_phase_source", "暂无")
    st.caption(f"当前显示来源: {source}")
    if preview is None:
        st.info(
            "当前无法精确展示相位预览。只有通过本页面下发并缓存过的相位，才能保证预览与设备显示一致。"
        )
        return

    st.image(
        preview,
        caption=f"SLM {slm_num} 当前显示相位预览",
        clamp=True,
        use_container_width=True,
    )


def render_pattern_controls(slm_num: int) -> tuple[str, dict[str, int | float | str]]:
    prefix = f"slm{slm_num}"

    # Get SLM properties from session state (set during connection)
    slm_width = st.session_state.get(f"slm{slm_num}_width", 1920)
    slm_height = st.session_state.get(f"slm{slm_num}_height", 1200)
    slm_pixel_size = st.session_state.get(f"slm{slm_num}_pixel_size_um", 8.0)

    pattern_type = st.selectbox(
        "选择相位图类型",
        options=[
            "平场",
            "线性光栅",
            "圆形光栅",
            "透镜",
            "全息光栅",
            "棋盘格",
            "二元光栅",
            "微透镜阵列",
            "湍流相位屏",
            "Zernike",
            "达曼光栅",
        ],
        key=f"{prefix}_pattern_type",
    )

    params: dict[str, Any] = {}

    if pattern_type in {"线性光栅", "全息光栅"}:
        params["period"] = st.number_input(
            "周期 (像素)",
            min_value=1.0,
            max_value=1000.0,
            value=50.0,
            step=1.0,
            key=f"{prefix}_{pattern_type}_period",
        )
        params["phase_range"] = st.number_input(
            "相位范围 (rad)",
            min_value=0.1,
            max_value=float(2 * np.pi),
            value=float(2 * np.pi),
            key=f"{prefix}_{pattern_type}_phase_range",
        )
    elif pattern_type == "圆形光栅":
        params["radius"] = st.number_input(
            "圆形周期半径 (像素)",
            min_value=1.0,
            max_value=2000.0,
            value=300.0,
            step=10.0,
            key=f"{prefix}_circular_radius",
        )
        params["phase_range"] = st.number_input(
            "相位范围 (rad)",
            min_value=0.1,
            max_value=float(2 * np.pi),
            value=float(2 * np.pi),
            key=f"{prefix}_circular_phase_range",
        )
    elif pattern_type == "透镜":
        params["focal_length_mm"] = st.number_input(
            "焦距 (mm)",
            min_value=1.0,
            max_value=100000.0,
            value=1000.0,
            step=10.0,
            key=f"{prefix}_lens_focal_length",
        )
        params["pixel_size_um"] = st.number_input(
            "像素尺寸 (um)",
            min_value=0.1,
            max_value=100.0,
            value=slm_pixel_size,
            step=0.1,
            key=f"{prefix}_lens_pixel_size",
        )
    elif pattern_type == "棋盘格":
        params["period"] = st.number_input(
            "棋盘格周期 (像素)",
            min_value=1,
            max_value=1000,
            value=100,
            step=1,
            key=f"{prefix}_checker_period",
        )
    elif pattern_type == "二元光栅":
        params["a"] = st.number_input(
            "亮条纹宽度 a (像素)",
            min_value=1,
            max_value=1000,
            value=2,
            step=1,
            key=f"{prefix}_binary_a",
        )
        params["b"] = st.number_input(
            "暗条纹宽度 b (像素)",
            min_value=1,
            max_value=1000,
            value=3,
            step=1,
            key=f"{prefix}_binary_b",
        )
        params["direction"] = st.selectbox(
            "方向",
            options=["horizontal", "vertical"],
            format_func=lambda x: "水平" if x == "horizontal" else "垂直",
            key=f"{prefix}_binary_direction",
        )
    elif pattern_type == "微透镜阵列":
        params["lens_size"] = st.number_input(
            "微透镜尺寸 (像素)",
            min_value=8,
            max_value=1000,
            value=200,
            step=1,
            key=f"{prefix}_microlens_size",
        )
        params["focal_length_mm"] = st.number_input(
            "焦距 (mm)",
            min_value=1.0,
            max_value=100000.0,
            value=100.0,
            step=1.0,
            key=f"{prefix}_microlens_focal_length",
        )
        params["pixel_size_um"] = st.number_input(
            "像素尺寸 (um)",
            min_value=0.1,
            max_value=100.0,
            value=slm_pixel_size,
            step=0.1,
            key=f"{prefix}_microlens_pixel_size",
        )
    elif pattern_type == "湍流相位屏":
        params["Cn2"] = st.number_input(
            "Cn²",
            min_value=1e-18,
            max_value=1e-10,
            value=1e-14,
            format="%.1e",
            key=f"{prefix}_turbulence_cn2",
        )
        params["L"] = st.number_input(
            "传播距离 L (m)",
            min_value=0.1,
            max_value=1e6,
            value=1000.0,
            step=10.0,
            key=f"{prefix}_turbulence_length",
        )
        params["pixel_size_um"] = st.number_input(
            "像素尺寸 (um)",
            min_value=0.1,
            max_value=100.0,
            value=slm_pixel_size,
            step=0.1,
            key=f"{prefix}_turbulence_pixel_size",
        )
    elif pattern_type == "Zernike":
        # Maximum radial order
        n_max = st.number_input(
            "最大径向阶数 N",
            min_value=1,
            max_value=10,
            value=4,
            step=1,
            key=f"{prefix}_zernike_n_max",
        )
        params["n_max"] = n_max
        params["radius"] = st.number_input(
            "孔径半径 (像素)",
            min_value=1,
            max_value=2000,
            value=min(slm_width, slm_height) // 2,
            step=1,
            key=f"{prefix}_zernike_radius",
        )

        # Collect coefficients for all (n, m) pairs up to n_max
        st.caption("各阶系数 (n, m):")
        coefficients = {}

        # Zernike polynomial naming (Noll's scheme)
        zernike_names = {
            (0, 0): "Piston",
            (1, -1): "Tip",
            (1, 1): "Tilt",
            (2, 0): "Defocus",
            (2, -2): "Astigmatism 45°",
            (2, 2): "Astigmatism 0°",
            (3, -1): "Coma Y",
            (3, 1): "Coma X",
            (3, -3): "Trefoil Y",
            (3, 3): "Trefoil X",
            (4, 0): "Spherical",
            (4, -2): "Secondary Astig 45°",
            (4, 2): "Secondary Astig 0°",
            (4, -4): "Tetrafoil Y",
            (4, 4): "Tetrafoil X",
        }

        # Generate all valid (n, m) pairs for orders up to n_max
        for n in range(n_max + 1):
            for m in range(-n, n + 1):
                if (n - abs(m)) % 2 == 0:  # Valid Zernike order
                    key = f"{prefix}_zernike_{n}_{m}"

                    # Default value: 1.0 for piston (0,0), 0.0 for others
                    default_val = 1.0 if n == 0 and m == 0 else 0.0

                    # Get value from session state if exists, otherwise use default
                    current_val = st.session_state.get(key, default_val)

                    col1, col2, col3 = st.columns([1, 2, 2])
                    with col1:
                        st.write(f"Z{n},{m}")
                    with col2:
                        name = zernike_names.get((n, m), "")
                        st.caption(name if name else f"n={n},m={m}")
                    with col3:
                        coeff = st.number_input(
                            "系数",
                            min_value=-5.0,
                            max_value=5.0,
                            value=float(current_val),
                            step=0.1,
                            key=key,
                        )
                    coefficients[(n, m)] = coeff

        params["coefficients"] = coefficients
    elif pattern_type == "达曼光栅":
        params["order"] = st.number_input(
            "衍射级数",
            min_value=2,
            max_value=8,
            value=3,
            step=1,
            key=f"{prefix}_dammann_order",
        )
        params["fill_factor"] = st.slider(
            "填充因子",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            key=f"{prefix}_dammann_fill_factor",
        )

    return pattern_type, params


def generate_phase_gray(
    slm: SantecSLM200,
    pattern_type: str,
    params: dict[str, int | float | str],
) -> np.ndarray:
    """Generate phase pattern using SLM properties.

    Automatically reads resolution, pixel size, bit depth, and wavelength from the SLM object.
    """
    # Get SLM properties
    width = slm.Panel_Res[0]
    height = slm.Panel_Res[1]
    pixel_size_um = slm.Pixel_Size_um
    bits = slm.Gray_Scale_bits
    wavelength_nm = slm.wavelength

    # Create pattern helper with correct resolution and bit depth
    slm_pattern_helper = SLMPatternHelper()
    gray_helper = PatternHelper((width, height), bits=bits)

    if pattern_type == "平场":
        phase_rad = np.zeros((height, width))
        return slm.create_phase_from_array(phase_rad)
    if pattern_type == "线性光栅":
        phase_rad = slm_pattern_helper.linear_grating(
            width=width,
            height=height,
            period=float(params["period"]),
            phase_range=float(params["phase_range"]),
        )
        return slm.create_phase_from_array(phase_rad)
    if pattern_type == "圆形光栅":
        phase_rad = slm_pattern_helper.circular_grating(
            width=width,
            height=height,
            radius=float(params["radius"]),
            phase_range=float(params["phase_range"]),
        )
        return slm.create_phase_from_array(phase_rad)
    if pattern_type == "透镜":
        # Use pixel size from SLM if not provided in params
        pixel_size = params.get("pixel_size_um", pixel_size_um)
        phase_rad = slm_pattern_helper.lens(
            width=width,
            height=height,
            focal_length=float(params["focal_length_mm"]) * 1e-3,  # mm -> m
            wavelength=float(wavelength_nm) * 1e-9,  # nm -> m
            pixel_size=float(pixel_size) * 1e-6,  # um -> m
        )
        return slm.create_phase_from_array(phase_rad)
    if pattern_type == "全息光栅":
        phase_rad = slm_pattern_helper.hologram(
            width=width,
            height=height,
            period=float(params["period"]),
            phase_range=float(params["phase_range"]),
        )
        return slm.create_phase_from_array(phase_rad)
    if pattern_type == "棋盘格":
        return gray_helper.generate_checkerboard(period=int(params["period"]))
    if pattern_type == "二元光栅":
        return gray_helper.generate_binary_grating(
            a=int(params["a"]),
            b=int(params["b"]),
            direction=str(params["direction"]),
        )
    if pattern_type == "微透镜阵列":
        # Use pixel size from SLM if not provided in params
        pixel_size = params.get("pixel_size_um", pixel_size_um)
        return gray_helper.generate_microlens_array(
            lens_size=int(params["lens_size"]),
            focal_length=float(params["focal_length_mm"]) * 1e-3,
            wavelength=float(wavelength_nm) * 1e-9,
            pixel_size=float(pixel_size) * 1e-6,
        )
    if pattern_type == "湍流相位屏":
        # Use pixel size from SLM if not provided in params
        pixel_size = params.get("pixel_size_um", pixel_size_um)
        return gray_helper.generate_turbulence_screen(
            Cn2=float(params["Cn2"]),
            L=float(params["L"]),
            wavelength=float(wavelength_nm) * 1e-9,
            pixel_size=float(pixel_size) * 1e-6,
        )
    if pattern_type == "Zernike":
        n_max = int(params.get("n_max", 4))
        raw_coeffs = params.get("coefficients")
        coefficients: dict[tuple[int, int], float] | None = None
        if raw_coeffs is not None and isinstance(raw_coeffs, dict):
            coefficients = {
                k: float(v)
                for k, v in raw_coeffs.items()
                if isinstance(k, tuple) and isinstance(v, (int, float))
            }
        radius = float(params.get("radius", min(width, height) // 2))
        return gray_helper.generate_zernike_polynomial(
            n_max=n_max,
            coefficients=coefficients,
            radius=radius,
        )
    if pattern_type == "达曼光栅":
        order = int(params.get("order", 3))
        fill_factor = float(params.get("fill_factor", 0.5))
        return gray_helper.generate_dammann_grating(
            order=order, fill_factor=fill_factor
        )
    raise ValueError(f"未知相位图类型: {pattern_type}")


def main():
    st.title("双SLM200控制器")
    _initialize_slm_state()

    # Sidebar for controls
    with st.sidebar:
        st.header("SLM 1 设置")
        slm1_conn_button = st.button("连接 SLM 1", key="slm1_connect")
        slm1_disc_button = st.button("断开 SLM 1", key="slm1_disconnect")

        if slm1_conn_button:
            connect_slm(1)
        if slm1_disc_button:
            disconnect_slm(1)

        # Only show wavelength and mode settings if connected
        if st.session_state.slm1_connected:
            # Display device info
            st.caption("设备信息")
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.write(
                    f"分辨率: {st.session_state.slm1_width}×{st.session_state.slm1_height}"
                )
                st.write(f"像素尺寸: {st.session_state.slm1_pixel_size_um} μm")
            with col_info2:
                st.write(f"Bit数: {st.session_state.slm1_bits}")
                st.write(f"SLM编号: {st.session_state.slm1.slm_number}")

            st.divider()

            st.number_input(
                "波长 (nm)",
                min_value=450,
                max_value=1600,
                value=st.session_state.slm1_wavelength,
                key="slm1_wavelength",
            )
            if st.button("设置波长", key="slm1_set_wl"):
                set_wavelength(1)

            st.selectbox(
                "视频模式",
                options=[0, 1],
                format_func=lambda x: "内存模式" if x == 0 else "DVI模式",
                index=st.session_state.slm1_video_mode,
                key="slm1_video_mode",
            )
            if st.button("设置模式", key="slm1_set_mode"):
                set_video_mode(1, st.session_state.slm1_video_mode)

        st.divider()

        st.header("SLM 2 设置")
        slm2_conn_button = st.button("连接 SLM 2", key="slm2_connect")
        slm2_disc_button = st.button("断开 SLM 2", key="slm2_disconnect")

        if slm2_conn_button:
            connect_slm(2)
        if slm2_disc_button:
            disconnect_slm(2)

        if st.session_state.slm2_connected:
            # Display device info
            st.caption("设备信息")
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.write(
                    f"分辨率: {st.session_state.slm2_width}×{st.session_state.slm2_height}"
                )
                st.write(f"像素尺寸: {st.session_state.slm2_pixel_size_um} μm")
            with col_info2:
                st.write(f"Bit数: {st.session_state.slm2_bits}")
                st.write(f"SLM编号: {st.session_state.slm2.slm_number}")

            st.divider()

            st.number_input(
                "波长 (nm)",
                min_value=450,
                max_value=1600,
                value=st.session_state.slm2_wavelength,
                key="slm2_wavelength",
            )
            if st.button("设置波长", key="slm2_set_wl"):
                set_wavelength(2)

            st.selectbox(
                "视频模式",
                options=[0, 1],
                format_func=lambda x: "内存模式" if x == 0 else "DVI模式",
                index=st.session_state.slm2_video_mode,
                key="slm2_video_mode",
            )
            if st.button("设置模式", key="slm2_set_mode"):
                set_video_mode(2, st.session_state.slm2_video_mode)

    # Main area: Phase control for each SLM
    col1, col2 = st.columns(2)

    with col1:
        st.header("SLM 1 相位控制")
        display_slm_status(1)
        if st.session_state.slm1_connected:
            if st.button("刷新当前显示相位", key="slm1_refresh_phase"):
                refresh_phase_preview(1)
            render_phase_preview(1)
            slm1_phase_control()

    with col2:
        st.header("SLM 2 相位控制")
        display_slm_status(2)
        if st.session_state.slm2_connected:
            if st.button("刷新当前显示相位", key="slm2_refresh_phase"):
                refresh_phase_preview(2)
            render_phase_preview(2)
            slm2_phase_control()

    st.divider()
    st.header("相机可视化")
    cam_col1, cam_col2 = st.columns(2)
    with cam_col1:
        render_camera_panel(1)
    with cam_col2:
        render_camera_panel(2)


def _get_camera_class(driver_name: str):
    if driver_name == "MIICAM":
        try:
            from ao_shaping.drivers.ccd.miicam_driver import CameraStreamManager

            return CameraStreamManager
        except ImportError as e:
            raise ImportError(
                f"MIICAM 驱动不可用: {e}. 请确保已安装 MIICAM SDK 或将 SDK 文件放置在正确的位置。"
            )
    if driver_name == "Daheng":
        try:
            from ao_shaping.drivers.ccd.daheng import DahengCamManager

            return DahengCamManager
        except ImportError as e:
            raise ImportError(f"Daheng 驱动不可用: {e}. 请确保已安装 GxIPy 库。")
    raise ValueError(f"未知相机驱动: {driver_name}")


def _detect_exposure_range(camera: Any, driver_name: str) -> tuple[int, int]:
    if driver_name == "Daheng" and getattr(camera, "cam", None) is not None:
        try:
            float_range = camera.cam.ExposureTime.get_range()
            if float_range:
                return int(float_range["min"]), int(float_range["max"])
        except Exception as e:
            logger.warning(f"Daheng曝光范围读取失败，使用默认范围: {e}")
        return 20, 1_000_000

    if driver_name == "MIICAM" and getattr(camera, "cam", None) is not None:
        try:
            max_us, min_us, _, _ = camera.cam.get_ExpTimeRange()
            return max(int(min_us / 1000), 1), max(int(max_us / 1000), 1)
        except Exception as e:
            logger.warning(f"MIICAM曝光范围读取失败，使用默认范围: {e}")
        return 1, 1000

    return 1, 1000


def connect_camera(cam_num: int) -> None:
    prefix = f"cam{cam_num}"
    driver_name = st.session_state[f"{prefix}_driver"]
    cam_id = st.session_state[f"{prefix}_id"]
    exposure_ms = st.session_state[f"{prefix}_exposure_ms"]
    try:
        camera_class = _get_camera_class(driver_name)
        camera = camera_class(cam_id=cam_id, exposure_time_ms=exposure_ms)
        camera.open()
        exposure_min_ms, exposure_max_ms = _detect_exposure_range(camera, driver_name)

        st.session_state[prefix] = camera
        st.session_state[f"{prefix}_connected"] = True
        st.session_state[f"{prefix}_exposure_min_ms"] = exposure_min_ms
        st.session_state[f"{prefix}_exposure_max_ms"] = exposure_max_ms
        st.session_state[f"{prefix}_exposure_ms"] = int(
            min(max(exposure_ms, exposure_min_ms), exposure_max_ms)
        )
        st.success(f"相机 {cam_num} 连接成功（{driver_name}）")
    except Exception as e:
        st.error(f"相机 {cam_num} 连接失败: {e}")
        logger.error(f"Failed to connect camera {cam_num}: {e}")


def disconnect_camera(cam_num: int) -> None:
    prefix = f"cam{cam_num}"
    camera = st.session_state.get(prefix)
    try:
        if camera is not None:
            camera.close()
        st.session_state[prefix] = None
        st.session_state[f"{prefix}_connected"] = False
        st.success(f"相机 {cam_num} 已断开")
    except Exception as e:
        st.error(f"相机 {cam_num} 断开失败: {e}")
        logger.error(f"Failed to disconnect camera {cam_num}: {e}")


def render_camera_panel(cam_num: int) -> None:
    prefix = f"cam{cam_num}"
    st.subheader(f"相机 {cam_num}")
    st.selectbox(
        "驱动类型",
        options=["MIICAM", "Daheng"],
        key=f"{prefix}_driver",
    )
    st.number_input(
        "相机ID",
        min_value=0,
        max_value=8,
        step=1,
        key=f"{prefix}_id",
    )

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button("连接", key=f"{prefix}_connect"):
            connect_camera(cam_num)
    with action_col2:
        if st.button("断开", key=f"{prefix}_disconnect"):
            disconnect_camera(cam_num)

    if not st.session_state.get(f"{prefix}_connected", False):
        st.info("相机未连接")
        return

    exposure_min = int(st.session_state.get(f"{prefix}_exposure_min_ms", 1))
    exposure_max = int(st.session_state.get(f"{prefix}_exposure_max_ms", 1000))
    st.caption(f"曝光范围：{exposure_min} ~ {exposure_max} ms（不同相机范围不同）")
    st.slider(
        "曝光时间 (ms)",
        min_value=exposure_min,
        max_value=exposure_max,
        key=f"{prefix}_exposure_ms",
    )

    camera = st.session_state[prefix]
    if st.button("设置曝光", key=f"{prefix}_set_exp"):
        try:
            actual_exposure = camera.reset_exposure_time(
                st.session_state[f"{prefix}_exposure_ms"]
            )
            # Note: Cannot modify session_state for widgets after instantiation in Streamlit
            # Just show the actual value that was set
            st.success(
                f"曝光设置成功: 期望值={st.session_state[f'{prefix}_exposure_ms']}ms, 实际值={actual_exposure}ms"
            )
        except Exception as e:
            st.error(f"设置曝光失败: {e}")
            logger.error(f"Failed to set exposure for camera {cam_num}: {e}")

    sample_count = st.number_input(
        "平均帧数",
        min_value=1,
        max_value=16,
        value=1,
        step=1,
        key=f"{prefix}_sample_count",
    )
    skip_first = st.checkbox("跳过首帧", value=True, key=f"{prefix}_skip_first")

    if st.button("采集并显示图像", key=f"{prefix}_capture"):
        try:
            frame = camera.get_numpy_image(
                n_sample=int(sample_count), skip_first=bool(skip_first)
            )
            st.image(
                frame,
                caption=f"相机 {cam_num} 实时图像",
                clamp=True,
                use_container_width=True,
            )
            st.write(
                f"形状: {frame.shape}, dtype: {frame.dtype}, min/max: {frame.min()}/{frame.max()}"
            )
        except Exception as e:
            st.error(f"采集图像失败: {e}")
            logger.error(f"Failed to capture image for camera {cam_num}: {e}")


def connect_slm(slm_num: int):
    """Connect to the specified SLM"""
    try:
        if slm_num == 1:
            slm = SantecSLM200(
                slm_number=1,
                wavelength=st.session_state.slm1_wavelength,
                video_mode=st.session_state.slm1_video_mode,
            )
            slm.open()
            st.session_state.slm1 = slm
            st.session_state.slm1_connected = True
            # Update wavelength and mode from the object (in case they changed during open)
            st.session_state.slm1_wavelength = slm.wavelength
            st.session_state.slm1_video_mode = slm.video_mode
            st.session_state.slm1_phase_preview = None
            st.session_state.slm1_phase_source = "连接成功，等待读取或下发相位"
            # Store SLM properties from driver
            st.session_state.slm1_width = slm.Panel_Res[0]
            st.session_state.slm1_height = slm.Panel_Res[1]
            st.session_state.slm1_pixel_size_um = slm.Pixel_Size_um
            st.session_state.slm1_bits = slm.Gray_Scale_bits
            st.success(f"SLM {slm_num} 连接成功")
        else:
            slm = SantecSLM200(
                slm_number=2,
                wavelength=st.session_state.slm2_wavelength,
                video_mode=st.session_state.slm2_video_mode,
            )
            slm.open()
            st.session_state.slm2 = slm
            st.session_state.slm2_connected = True
            st.session_state.slm2_wavelength = slm.wavelength
            st.session_state.slm2_video_mode = slm.video_mode
            st.session_state.slm2_phase_preview = None
            st.session_state.slm2_phase_source = "连接成功，等待读取或下发相位"
            # Store SLM properties from driver
            st.session_state.slm2_width = slm.Panel_Res[0]
            st.session_state.slm2_height = slm.Panel_Res[1]
            st.session_state.slm2_pixel_size_um = slm.Pixel_Size_um
            st.session_state.slm2_bits = slm.Gray_Scale_bits
            st.success(f"SLM {slm_num} 连接成功")
    except Exception as e:
        st.error(f"SLM {slm_num} 连接失败: {e}")
        logger.error(f"Failed to connect SLM {slm_num}: {e}")


def disconnect_slm(slm_num: int):
    """Disconnect from the specified SLM"""
    try:
        if slm_num == 1 and st.session_state.slm1 is not None:
            st.session_state.slm1.close()
            st.session_state.slm1 = None
            st.session_state.slm1_connected = False
            st.session_state.slm1_phase_preview = None
            st.session_state.slm1_phase_source = "暂无"
            st.success(f"SLM {slm_num} 已断开")
        elif slm_num == 2 and st.session_state.slm2 is not None:
            st.session_state.slm2.close()
            st.session_state.slm2 = None
            st.session_state.slm2_connected = False
            st.session_state.slm2_phase_preview = None
            st.session_state.slm2_phase_source = "暂无"
            st.success(f"SLM {slm_num} 已断开")
    except Exception as e:
        st.error(f"SLM {slm_num} 断开失败: {e}")
        logger.error(f"Failed to disconnect SLM {slm_num}: {e}")


def set_wavelength(slm_num: int):
    """Set wavelength for the specified SLM"""
    try:
        if slm_num == 1 and st.session_state.slm1 is not None:
            st.session_state.slm1.set_wavelength(st.session_state.slm1_wavelength)
            st.success(f"SLM 1 波长设置为 {st.session_state.slm1_wavelength} nm")
        elif slm_num == 2 and st.session_state.slm2 is not None:
            st.session_state.slm2.set_wavelength(st.session_state.slm2_wavelength)
            st.success(f"SLM 2 波长设置为 {st.session_state.slm2_wavelength} nm")
    except Exception as e:
        st.error(f"设置波长失败: {e}")
        logger.error(f"Failed to set wavelength for SLM {slm_num}: {e}")


def set_video_mode(slm_num: int, mode: int):
    """Set video mode for the specified SLM"""
    try:
        if slm_num == 1 and st.session_state.slm1 is not None:
            st.session_state.slm1._set_memory_mode(mode)
            st.session_state.slm1_video_mode = mode
            st.success(f"SLM 1 模式设置为 {'内存模式' if mode == 0 else 'DVI模式'}")
        elif slm_num == 2 and st.session_state.slm2 is not None:
            st.session_state.slm2._set_memory_mode(mode)
            st.session_state.slm2_video_mode = mode
            st.success(f"SLM 2 模式设置为 {'内存模式' if mode == 0 else 'DVI模式'}")
    except Exception as e:
        st.error(f"设置模式失败: {e}")
        logger.error(f"Failed to set video mode for SLM {slm_num}: {e}")


def display_slm_status(slm_num: int):
    """Display the current status of the specified SLM"""
    if slm_num == 1:
        if st.session_state.slm1_connected and st.session_state.slm1 is not None:
            slm = st.session_state.slm1
            st.write(f"**状态**: 已连接")
            st.write(f"**波长**: {slm.wavelength} nm")
            st.write(f"**模式**: {'内存模式' if slm.video_mode == 0 else 'DVI模式'}")
            st.write(f"**下一个内存槽**: {st.session_state.slm1_next_memory}")
        else:
            st.write("**状态**: 未连接")
    else:
        if st.session_state.slm2_connected and st.session_state.slm2 is not None:
            slm = st.session_state.slm2
            st.write(f"**状态**: 已连接")
            st.write(f"**波长**: {slm.wavelength} nm")
            st.write(f"**模式**: {'内存模式' if slm.video_mode == 0 else 'DVI模式'}")
            st.write(f"**下一个内存槽**: {st.session_state.slm2_next_memory}")
        else:
            st.write("**状态**: 未连接")


def slm1_phase_control():
    """Phase control UI for SLM 1"""
    st.subheader("设置相位")
    pattern_type, params = render_pattern_controls(1)

    if st.button("从模式生成器生成相位", key="slm1_gen_pattern"):
        try:
            phase_gray = generate_phase_gray(
                st.session_state.slm1,
                pattern_type,
                params,
            )

            # Write to next memory slot and immediately display
            mem_slot = st.session_state.slm1_next_memory
            st.session_state.slm1.write_phase(phase_gray, memory_number=mem_slot)
            st.session_state.slm1.display_memory(mem_slot)
            refresh_phase_preview(1)

            # Update next memory slot (avoid using same slot consecutively)
            st.session_state.slm1_next_memory = (
                st.session_state.slm1_next_memory % 128
            ) + 1

            st.success(f"相位已写入内存槽 {mem_slot} 并显示")
        except Exception as e:
            st.error(f"生成或显示相位失败: {e}")
            logger.error(f"Failed to generate/display phase for SLM 1: {e}")

    # Option to load from CSV
    uploaded_file = st.file_uploader("上传CSV相位文件", type=["csv"], key="slm1_csv")
    if uploaded_file is not None:
        if st.button("从CSV加载相位", key="slm1_load_csv"):
            try:
                # Save uploaded file temporarily
                temp_path = Path("temp_slm1_phase.csv")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Load phase from CSV
                phase_gray = st.session_state.slm1.load_phase_from_csv(temp_path)

                # Write to next memory slot and immediately display
                mem_slot = st.session_state.slm1_next_memory
                st.session_state.slm1.write_phase(phase_gray, memory_number=mem_slot)
                st.session_state.slm1.display_memory(mem_slot)
                refresh_phase_preview(1)

                # Update next memory slot
                st.session_state.slm1_next_memory = (
                    st.session_state.slm1_next_memory % 128
                ) + 1

                st.success(f"相位已从CSV加载到内存槽 {mem_slot} 并显示")

                # Clean up temp file
                temp_path.unlink()
            except Exception as e:
                st.error(f"加载CSV相位失败: {e}")
                logger.error(f"Failed to load CSV phase for SLM 1: {e}")


def slm2_phase_control():
    """Phase control UI for SLM 2"""
    st.subheader("设置相位")
    pattern_type, params = render_pattern_controls(2)

    if st.button("从模式生成器生成相位", key="slm2_gen_pattern"):
        try:
            phase_gray = generate_phase_gray(
                st.session_state.slm2,
                pattern_type,
                params,
            )

            # Write to next memory slot and immediately display
            mem_slot = st.session_state.slm2_next_memory
            st.session_state.slm2.write_phase(phase_gray, memory_number=mem_slot)
            st.session_state.slm2.display_memory(mem_slot)
            refresh_phase_preview(2)

            # Update next memory slot (avoid using same slot consecutively)
            st.session_state.slm2_next_memory = (
                st.session_state.slm2_next_memory % 128
            ) + 1

            st.success(f"相位已写入内存槽 {mem_slot} 并显示")
        except Exception as e:
            st.error(f"生成或显示相位失败: {e}")
            logger.error(f"Failed to generate/display phase for SLM 2: {e}")

    # Option to load from CSV
    uploaded_file = st.file_uploader("上传CSV相位文件", type=["csv"], key="slm2_csv")
    if uploaded_file is not None:
        if st.button("从CSV加载相位", key="slm2_load_csv"):
            try:
                # Save uploaded file temporarily
                temp_path = Path("temp_slm2_phase.csv")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Load phase from CSV
                phase_gray = st.session_state.slm2.load_phase_from_csv(temp_path)

                # Write to next memory slot and immediately display
                mem_slot = st.session_state.slm2_next_memory
                st.session_state.slm2.write_phase(phase_gray, memory_number=mem_slot)
                st.session_state.slm2.display_memory(mem_slot)
                refresh_phase_preview(2)

                # Update next memory slot
                st.session_state.slm2_next_memory = (
                    st.session_state.slm2_next_memory % 128
                ) + 1

                st.success(f"相位已从CSV加载到内存槽 {mem_slot} 并显示")

                # Clean up temp file
                temp_path.unlink()
            except Exception as e:
                st.error(f"加载CSV相位失败: {e}")
                logger.error(f"Failed to load CSV phase for SLM 2: {e}")


if __name__ == "__main__":
    main()
