import sys
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from loguru import logger

from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200
from ao_shaping.utils.pattern_helper import PatternHelper

# Global pattern helpers (will be recreated per-SLM based on resolution)
# Note: Resolution and bit depth now come from the SLM object when generating patterns


def _initialize_slm_state() -> None:
    for slm_num in (1, 2):
        prefix = f"slm{slm_num}"
        if prefix not in st.session_state:
            st.session_state[prefix] = None
            st.session_state[f"{prefix}_connected"] = False
            st.session_state[f"{prefix}_wavelength"] = 532
            st.session_state[f"{prefix}_video_mode"] = "内存模式"
            st.session_state[f"{prefix}_next_memory"] = np.random.randint(1, 128)
            st.session_state[f"{prefix}_phase_preview"] = None
            st.session_state[f"{prefix}_phase_source"] = "暂无"
            st.session_state[f"{prefix}_shift_x"] = 0
            st.session_state[f"{prefix}_shift_y"] = 0
        else:
            slm = st.session_state[prefix]
            if slm is not None and not getattr(slm, "is_open", False):
                st.session_state[prefix] = None
                st.session_state[f"{prefix}_connected"] = False

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


def _apply_shift(phase_gray: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
    shifted = np.zeros_like(phase_gray)
    y_src_start = max(0, -shift_y)
    y_src_end = min(phase_gray.shape[0], phase_gray.shape[0] - shift_y)
    y_dst_start = max(0, shift_y)
    y_dst_end = min(phase_gray.shape[0], phase_gray.shape[0] + shift_y)
    x_src_start = max(0, -shift_x)
    x_src_end = min(phase_gray.shape[1], phase_gray.shape[1] - shift_x)
    x_dst_start = max(0, shift_x)
    x_dst_end = min(phase_gray.shape[1], phase_gray.shape[1] + shift_x)
    if y_src_end > y_src_start and x_src_end > x_src_start:
        shifted[y_dst_start:y_dst_end, x_dst_start:x_dst_end] = phase_gray[
            y_src_start:y_src_end, x_src_start:x_src_end
        ]
    return shifted


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
        width="stretch",
    )


def render_pattern_controls(slm_num: int) -> tuple[str, dict[str, int | float | str]]:
    prefix = f"slm{slm_num}"

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
            "涡旋相位",
        ],
        key=f"{prefix}_pattern_type",
    )

    params: dict[str, Any] = {}

    if pattern_type == "平场":
        params["flat_gray"] = st.number_input(
            "灰度",
            min_value=0,
            max_value=1024,
            step=1,
            key=f"{prefix}_{pattern_type}_gray"
        )
    elif pattern_type in {"线性光栅", "全息光栅"}:
        params["period"] = st.number_input(
            "周期 (像素)",
            min_value=1.0,
            max_value=1000.0,
            step=1.0,
            key=f"{prefix}_{pattern_type}_period",
        )
        params["phase_range"] = st.number_input(
            "相位范围 (rad)",
            min_value=0.1,
            max_value=float(2 * np.pi),
            step=0.1,
            key=f"{prefix}_{pattern_type}_phase_range",
        )
    elif pattern_type == "圆形光栅":
        params["radius"] = st.number_input(
            "圆形周期半径 (像素)",
            min_value=1.0,
            max_value=2000.0,
            step=10.0,
            key=f"{prefix}_circular_radius",
        )
        params["phase_range"] = st.number_input(
            "相位范围 (rad)",
            min_value=0.1,
            max_value=float(2 * np.pi),
            step=0.1,
            key=f"{prefix}_circular_phase_range",
        )
    elif pattern_type == "透镜":
        params["focal_length_mm"] = st.number_input(
            "焦距 (mm)",
            min_value=1.0,
            max_value=100000.0,
            step=10.0,
            key=f"{prefix}_lens_focal_length",
        )
        params["pixel_pitch_um"] = st.number_input(
            "像素间距 (um)",
            min_value=0.1,
            max_value=100.0,
            step=0.1,
            key=f"{prefix}_lens_pixel_pitch",
        )
        params["lens_radius"] = st.number_input(
            "透镜半径 (像素)",
            min_value=1,
            max_value=2000,
            step=1,
            key=f"{prefix}_lens_radius",
        )
    elif pattern_type == "棋盘格":
        params["period"] = st.number_input(
            "棋盘格周期 (像素)",
            min_value=1,
            max_value=1000,
            step=1,
            key=f"{prefix}_checker_period",
        )
    elif pattern_type == "二元光栅":
        params["a"] = st.number_input(
            "亮条纹宽度 a (像素)",
            min_value=1,
            max_value=1000,
            step=1,
            key=f"{prefix}_binary_a",
        )
        params["b"] = st.number_input(
            "暗条纹宽度 b (像素)",
            min_value=1,
            max_value=1000,
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
            step=1,
            key=f"{prefix}_microlens_size",
        )
        params["focal_length_mm"] = st.number_input(
            "焦距 (mm)",
            min_value=1.0,
            max_value=100000.0,
            step=1.0,
            key=f"{prefix}_microlens_focal_length",
        )
        params["pixel_pitch_um"] = st.number_input(
            "像素间距 (um)",
            min_value=0.1,
            max_value=100.0,
            step=0.1,
            key=f"{prefix}_microlens_pixel_pitch",
        )
    elif pattern_type == "湍流相位屏":
        params["Cn2"] = st.number_input(
            "Cn²",
            min_value=1e-18,
            max_value=1e-10,
            format="%.1e",
            key=f"{prefix}_turbulence_cn2",
        )
        params["L"] = st.number_input(
            "传播距离 L (m)",
            min_value=0.1,
            max_value=1e6,
            step=10.0,
            key=f"{prefix}_turbulence_length",
        )
        params["pixel_pitch_um"] = st.number_input(
            "像素间距 (um)",
            min_value=0.1,
            max_value=100.0,
            step=0.1,
            key=f"{prefix}_turbulence_pixel_pitch",
        )
    elif pattern_type == "涡旋相位":
        params["topological_charge"] = st.number_input(
            "拓扑荷",
            min_value=-10,
            max_value=10,
            step=1,
            key=f"{prefix}_vortex_charge",
        )
        # 从sidebar设置读取默认值
        default_wavelength = st.session_state.get(f"{prefix}_wavelength", 1064)
        default_pixel_pitch = st.session_state.get(f"{prefix}_pixel_pitch_um", 8.0)

        params["wavelength_nm"] = st.number_input(
            "波长 (nm)",
            value=default_wavelength,
            step=1,
            key=f"{prefix}_vortex_wavelength",
        )
        params["pixel_pitch_um"] = st.number_input(
            "像素间距 (um)",
            value=default_pixel_pitch,
            min_value=1,
            max_value=100,
            step=1,
            key=f"{prefix}_vortex_pixel_pitch",
        )
        params["wrap_phase"] = st.checkbox(
            "包裹相位",
            key=f"{prefix}_vortex_wrap_phase",
        )
    elif pattern_type == "Zernike":
        # Maximum radial order
        n_max = st.number_input(
            "最大径向阶数 N",
            min_value=1,
            max_value=10,
            step=1,
            key=f"{prefix}_zernike_n_max",
        )
        params["n_max"] = n_max
        params["radius"] = st.number_input(
            "孔径半径 (像素)",
            min_value=1,
            max_value=2000,
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
                    st.session_state.get(key, default_val)

                    col1, col2, col3 = st.columns([1, 2, 2])
                    with col1:
                        st.write(f"Z{n},{m}")
                    with col2:
                        name = zernike_names.get((n, m), "")
                        st.caption(name if name else f"n={n},m={m}")
                    with col3:
                        st.number_input(
                            "系数",
                            min_value=-100.0,
                            max_value=100.0,
                            step=0.001,
                            key=key,
                        )
                    coefficients[(n, m)] = st.session_state.get(key, default_val)

        params["coefficients"] = coefficients
    elif pattern_type == "达曼光栅":
        params["order"] = st.number_input(
            "衍射级数",
            min_value=2,
            max_value=8,
            step=1,
            key=f"{prefix}_dammann_order",
        )
        params["fill_factor"] = st.slider(
            "填充因子",
            min_value=0.1,
            max_value=1.0,
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
    # Get SLM properties (use pixel pitch for diffraction pattern geometry)
    width = slm.Panel_Res[0]
    height = slm.Panel_Res[1]
    pattern_pitch_um = slm.Pitch_um
    bits = slm.Gray_Scale_bits
    wavelength_nm = slm.wavelength

    # Create unified pattern helper
    helper = PatternHelper((width, height), bits=bits)

    # Phase pattern generation mapping
    if pattern_type == "平场":
        _gray = int(params["flat_gray"])
        # Return raw grayscale uint16 — bypass create_phase_from_array which
        # interprets input as radians (mod 2π) and would map the gray value
        # through an unwanted radian-to-grayscale conversion.
        return np.full((height, width), _gray, dtype=np.uint16)
    if pattern_type == "线性光栅":
        phase_rad = helper.linear_grating(
            period=float(params["period"]),
            phase_range=float(params["phase_range"]),                                                                                                                                                           
            wrap_phase=False,
        )
        return slm.create_phase_from_array(phase_rad)
    if pattern_type == "圆形光栅":
        phase_rad = helper.circular_grating(
            radius=float(params["radius"]),
            phase_range=float(params["phase_range"]),
            wrap_phase=False,
        )
        return slm.create_phase_from_array(phase_rad)
    if pattern_type == "透镜":
        # Use pixel pitch (center-to-center spacing) for diffraction geometry
        pattern_pitch = params.get("pixel_pitch_um", pattern_pitch_um)
        lens_radius = float(params.get("lens_radius", 0.0))
        lens_radius = lens_radius if lens_radius > 0 else None
        phase_rad = helper.lens(
            focal_length=float(params["focal_length_mm"]) * 1e-3,  # mm -> m
            wavelength=float(wavelength_nm) * 1e-9,  # nm -> m
            pixel_size=float(pattern_pitch) * 1e-6,  # um -> m
            lens_radius=lens_radius,
        )
        return slm.create_phase_from_array(phase_rad)
    if pattern_type == "全息光栅":
        phase_rad = helper.hologram(
            period=float(params["period"]),
            phase_range=float(params["phase_range"]),
        )
        return slm.create_phase_from_array(phase_rad)
    if pattern_type == "棋盘格":
        return helper.generate_checkerboard(period=int(params["period"]))
    if pattern_type == "二元光栅":
        return helper.generate_binary_grating(
            a=int(params["a"]),
            b=int(params["b"]),
            direction=str(params["direction"]),
        )
    if pattern_type == "微透镜阵列":
        # Use pixel pitch (center-to-center spacing) for diffraction geometry
        pattern_pitch = params.get("pixel_pitch_um", pattern_pitch_um)
        return helper.generate_microlens_array(
            lens_size=int(params["lens_size"]),
            focal_length=float(params["focal_length_mm"]) * 1e-3,
            wavelength=float(wavelength_nm) * 1e-9,
            pixel_size=float(pattern_pitch) * 1e-6,
        )
    if pattern_type == "湍流相位屏":
        # Use pixel pitch (center-to-center spacing) for diffraction geometry
        pattern_pitch = params.get("pixel_pitch_um", pattern_pitch_um)
        return helper.generate_turbulence_screen(
            Cn2=float(params["Cn2"]),
            L=float(params["L"]),
            wavelength=float(wavelength_nm) * 1e-9,
            pixel_size=float(pattern_pitch) * 1e-6,
        )
    if pattern_type == "Zernike":
        raw_coeffs = params.get("coefficients")
        coefficients: dict[tuple[int, int], float] | None = None
        if raw_coeffs is not None and isinstance(raw_coeffs, dict):
            coefficients = {
                k: float(v)
                for k, v in raw_coeffs.items()
                if isinstance(k, tuple) and isinstance(v, (int, float))
            }
        radius = float(params.get("radius", min(width, height) // 2))
        return helper.generate_zernike_polynomial(
            coefficients=coefficients,
            radius=radius,
        )
    if pattern_type == "达曼光栅":
        order = int(params.get("order", 3))
        fill_factor = float(params.get("fill_factor", 0.5))
        return helper.generate_dammann_grating(order=order, fill_factor=fill_factor)
    if pattern_type == "涡旋相位":
        # Convert parameters to appropriate units
        wavelength_m = float(params["wavelength_nm"]) * 1e-9  # nm -> m
        pixel_pitch_m = (
            float(params.get("pixel_pitch_um", pattern_pitch_um)) * 1e-6
        )  # um -> m
        wrap_phase = bool(params.get("wrap_phase", True))
        topological_charge = int(params["topological_charge"])

        # Generate vortex phase (helper returns uint16 when wrap_phase=True, radians when False)
        phase_gray = helper.generate_vortex(
            topological_charge=topological_charge,
            wavelength=wavelength_m,
            pixel_size=pixel_pitch_m,
            wrap_phase=wrap_phase,
        )

        # If wrap_phase=False, helper returns radians; convert to uint16
        if not wrap_phase:
            phase_wrapped = np.mod(phase_gray, 2 * np.pi)
            phase_gray = (phase_wrapped / (2 * np.pi) * (2**bits - 1)).astype(np.uint16)

        return phase_gray
    raise ValueError(f"未知相位图类型: {pattern_type}")


def main():
    st.title("双SLM200控制器")

    _initialize_slm_state()

    with st.sidebar:
        render_slm_sidebar(1)
        st.divider()
        render_slm_sidebar(2)

    # Main area: Phase control for each SLM
    col1, col2 = st.columns(2)

    with col1:
        st.header("SLM 1 相位控制")
        display_slm_status(1)
        if st.session_state.slm1_connected:
            if st.button("刷新当前显示相位", key="slm1_refresh_phase_btn"):
                refresh_phase_preview(1)
            render_phase_preview(1)
            render_phase_control(1)

    with col2:
        st.header("SLM 2 相位控制")
        display_slm_status(2)
        if st.session_state.slm2_connected:
            if st.button("刷新当前显示相位", key="slm2_refresh_phase_btn"):
                refresh_phase_preview(2)
            render_phase_preview(2)
            render_phase_control(2)

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
        logger.exception(f"Failed to connect camera {cam_num}: {e}")


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
        logger.exception(f"Failed to disconnect camera {cam_num}: {e}")


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
        if st.button("连接", key=f"{prefix}_connect_btn"):
            connect_camera(cam_num)
    with action_col2:
        if st.button("断开", key=f"{prefix}_disconnect_btn"):
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
    if st.button("设置曝光", key=f"{prefix}_set_exp_btn"):
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
            logger.exception(f"Failed to set exposure for camera {cam_num}: {e}")

    sample_count = st.number_input(
        "平均帧数",
        min_value=1,
        max_value=16,
        step=1,
        key=f"{prefix}_sample_count",
    )
    skip_first = st.checkbox("跳过首帧", key=f"{prefix}_skip_first_toggle")

    if st.button("采集并显示图像", key=f"{prefix}_capture_btn"):
        try:
            frame = camera.get_numpy_image(
                n_sample=int(sample_count), skip_first=bool(skip_first)
            )
            st.image(
                frame,
                caption=f"相机 {cam_num} 实时图像",
                clamp=True,
                width="stretch",
            )
            st.write(
                f"形状: {frame.shape}, dtype: {frame.dtype}, min/max: {frame.min()}/{frame.max()}"
            )
        except Exception as e:
            st.error(f"采集图像失败: {e}")
            logger.exception(f"Failed to capture image for camera {cam_num}: {e}")


def connect_slm(slm_num: int):
    """Connect to the specified SLM

    Creates the :class:`SantecSLM200` instance with only the SLM number,
    then calls ``open()`` which handles its own init flow:
    reading serial → loading config file (if any) → applying device
    defaults.  After ``open()`` all resolved SLM state is synced back to
    session state so the UI widgets always reflect reality.
    """
    prefix = f"slm{slm_num}"
    wavelength_key = f"{prefix}_wavelength"
    video_mode_key = f"{prefix}_video_mode"
    shift_x_key = f"{prefix}_shift_x"
    shift_y_key = f"{prefix}_shift_y"
    mismatch_key = f"{prefix}_wavelength_mismatch"

    # Clean up any stale SLM object before connecting
    old_slm = st.session_state.get(prefix)
    if old_slm is not None:
        try:
            # Only try to close if the device thinks it's open
            if getattr(old_slm, "is_open", False):
                old_slm.close()
        except Exception:
            pass  # Ignore errors when cleaning up stale state
        # Always reset state in session
        st.session_state[prefix] = None
        st.session_state[f"{prefix}_connected"] = False

    try:
        slm = SantecSLM200(slm_number=slm_num)
        slm.open()

        st.session_state[prefix] = slm
        st.session_state[f"{prefix}_connected"] = True

        st.session_state[video_mode_key] = "内存模式" if slm.video_mode == 0 else "DVI模式"
        st.session_state[shift_x_key] = slm.shift_x
        st.session_state[shift_y_key] = slm.shift_y
        st.session_state[wavelength_key] = slm.wavelength
        st.session_state[f"{prefix}_phase_preview"] = None
        st.session_state[f"{prefix}_phase_source"] = "连接成功，等待读取或下发相位"
        st.session_state[f"{prefix}_width"] = slm.Panel_Res[0]
        st.session_state[f"{prefix}_height"] = slm.Panel_Res[1]
        st.session_state[f"{prefix}_pixel_pitch_um"] = slm.Pitch_um
        st.session_state[f"{prefix}_bits"] = slm.Gray_Scale_bits
        st.session_state[f"{prefix}_use_120hz"] = slm._use_120hz

        serial = slm._serial_number
        if serial:
            st.session_state[mismatch_key] = {
                "serial": serial,
                "wavelength": slm.wavelength,
            }
            st.toast(
                f"SLM {slm_num} 已连接（序列号: {serial}，波长 {slm.wavelength}nm）",
                icon="ℹ️",
            )
        else:
            st.session_state[mismatch_key] = None
            st.toast(
                f"SLM {slm_num} 已连接（波长 {slm.wavelength}nm，序列号未读取）",
                icon="ℹ️",
            )
        st.success(f"SLM {slm_num} 连接成功")
    except Exception as e:
        # Clean up state on failure
        st.session_state[prefix] = None
        st.session_state[f"{prefix}_connected"] = False
        st.error(f"SLM {slm_num} 连接失败: {e}")
        logger.exception(f"Failed to connect SLM {slm_num}: {e}")


def disconnect_slm(slm_num: int):
    """Disconnect from the specified SLM"""
    prefix = f"slm{slm_num}"
    slm = st.session_state.get(prefix)
    try:
        if slm is not None and getattr(slm, "is_open", False):
            slm.close()
            st.success(f"SLM {slm_num} 已断开")
    except Exception as e:
        st.error(f"SLM {slm_num} 断开失败: {e}")
        logger.exception(f"Failed to disconnect SLM {slm_num}: {e}")
    finally:
        st.session_state[prefix] = None
        st.session_state[f"{prefix}_connected"] = False
        st.session_state[f"{prefix}_phase_preview"] = None
        st.session_state[f"{prefix}_phase_source"] = "暂无"


def set_wavelength(slm_num: int):
    """Set wavelength for the specified SLM"""
    prefix = f"slm{slm_num}"
    wavelength_key = f"{prefix}_wavelength"
    try:
        slm = st.session_state.get(prefix)
        if slm is not None:
            wavelength = st.session_state[wavelength_key]
            slm.set_wavelength(wavelength)
            st.success(f"SLM {slm_num} 波长设置为 {wavelength} nm")
    except Exception as e:
        st.error(f"设置波长失败: {e}")
        logger.exception(f"Failed to set wavelength for SLM {slm_num}: {e}")


def set_video_mode(slm_num: int, mode_label: str):
    """Set video mode for the specified SLM"""
    prefix = f"slm{slm_num}"
    video_mode_key = f"{prefix}_video_mode"
    mode = 0 if mode_label == "内存模式" else 1
    try:
        slm = st.session_state.get(prefix)
        if slm is not None:
            slm._set_memory_mode(mode)
            st.session_state[video_mode_key] = mode_label
            st.success(f"SLM {slm_num} 模式设置为 {mode_label}")
    except Exception as e:
        st.error(f"设置模式失败: {e}")
        logger.exception(f"Failed to set video mode for SLM {slm_num}: {e}")


def display_slm_status(slm_num: int):
    """Display the current status of the specified SLM"""
    prefix = f"slm{slm_num}"
    connected_key = f"{prefix}_connected"
    next_memory_key = f"{prefix}_next_memory"
    if st.session_state.get(connected_key) and st.session_state.get(prefix) is not None:
        slm = st.session_state[prefix]
        st.write("**状态**: 已连接")
        st.write(f"**波长**: {slm.wavelength} nm")
        st.write(f"**模式**: {'内存模式' if slm.video_mode == 0 else 'DVI模式'}")
        st.write(f"**下一个内存槽**: {st.session_state[next_memory_key]}")
    else:
        st.write("**状态**: 未连接")


def render_slm_sidebar(slm_num: int):
    """Render sidebar controls for a single SLM (parameterized)."""
    prefix = f"slm{slm_num}"
    connected_key = f"{prefix}_connected"

    st.header(f"SLM {slm_num} 设置")

    is_connected = st.session_state.get(connected_key, False)
    button_label = f"断开 SLM {slm_num}" if is_connected else f"连接 SLM {slm_num}"
    action_button = st.button(button_label, key=f"{prefix}_action_btn")

    if action_button:
        if is_connected:
            disconnect_slm(slm_num)
        else:
            connect_slm(slm_num)

    # Only show wavelength and mode settings if connected
    slm_obj = st.session_state.get(prefix)
    if st.session_state.get(connected_key) and slm_obj is not None:
        # Display device info
        st.caption("设备信息")
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.write(
                f"分辨率: {st.session_state[f'{prefix}_width']}×{st.session_state[f'{prefix}_height']}"
            )
            st.write(f"像素间距: {st.session_state[f'{prefix}_pixel_pitch_um']} μm")
        with col_info2:
            st.write(f"Bit数: {st.session_state[f'{prefix}_bits']}")
            sn_display = st.session_state[prefix]._serial_number
            st.write(f"SLM序列号: {sn_display if sn_display else '-'}")
            st.write(f"SLM编号: {st.session_state[prefix].slm_number}")

        st.divider()

        st.number_input(
            "波长 (nm)",
            min_value=450,
            max_value=1600,
            step=1,
            key=f"{prefix}_wavelength",
        )
        if st.button("设置波长", key=f"{prefix}_set_wl_btn"):
            set_wavelength(slm_num)

        st.selectbox(
            "视频模式",
            options=["内存模式", "DVI模式"],
            key=f"{prefix}_video_mode",
        )
        if st.button("设置模式", key=f"{prefix}_set_mode_btn"):
            set_video_mode(slm_num, st.session_state[f"{prefix}_video_mode"])

        # Shift X/Y configuration
        st.caption("Pattern Shift (像素)")
        col_shift1, col_shift2 = st.columns(2)
        with col_shift1:
            st.number_input(
                "Shift X",
                min_value=-500,
                max_value=500,
                step=1,
                key=f"{prefix}_shift_x",
            )
        with col_shift2:
            st.number_input(
                "Shift Y",
                min_value=-500,
                max_value=500,
                step=1,
                key=f"{prefix}_shift_y",
            )

        # Config file info
        config_info = st.session_state.get(f"{prefix}_wavelength_mismatch")
        if config_info and slm_obj is not None:
            st.caption(f"序列号: {config_info['serial']}")
            loaded_cfg = slm_obj.load_config() if slm_obj._serial_number else {}
            if loaded_cfg:
                st.caption("已加载配置文件:")
                st.json(loaded_cfg)
            else:
                st.caption("未找到匹配的配置文件（使用设备默认值）")

        st.divider()

        st.caption("灰度设置")
        max_gray = int(getattr(slm_obj, "_max_gray", SantecSLM200.MAX_GRAYSCALE_VALUE))
        max_gray_abs = int(SantecSLM200.MAX_GRAYSCALE_VALUE)
        st.caption(f"最大灰度值 (2π对应): **{max_gray}** / {max_gray_abs}")
        new_max_gray = st.number_input(
            "2π 灰度值",
            min_value=1,
            max_value=max_gray_abs,
            step=1,
            value=max_gray,
            key=f"{prefix}_max_gray",
        )
        if st.button("应用灰度设置", key=f"{prefix}_apply_gray_btn"):
            try:
                slm_obj._max_gray = int(new_max_gray)
                st.success(f"SLM {slm_num} 灰度值已更新为 {new_max_gray}")
            except Exception as e:
                st.error(f"更新灰度设置失败: {e}")

        st.divider()

        # Save configuration button
        if st.button("保存配置", key=f"{prefix}_save_config_btn"):
            slm = st.session_state.get(prefix)
            if slm is None:
                st.info("SLM未连接，配置将在连接后写入设备")
            else:
                try:
                    # Sync UI widget values to the SLM object before saving,
                    # so the persisted config reflects what the user sees.
                    ui_shift_x = st.session_state.get(f"{prefix}_shift_x", 0)
                    ui_shift_y = st.session_state.get(f"{prefix}_shift_y", 0)
                    if (ui_shift_x, ui_shift_y) != (slm.shift_x, slm.shift_y):
                        slm.set_shift(shift_x=ui_shift_x, shift_y=ui_shift_y)
                    slm.save_config()
                    st.success(
                        f"SLM {slm_num} 配置已保存"
                        f"（序列号: {slm._serial_number or '未知'}）"
                    )
                except Exception as e:
                    st.error(f"保存配置到设备失败: {e}")

        # Save current phase to CSV
        if st.button("保存当前相位到CSV", key=f"{prefix}_save_csv_btn"):
            try:
                slm = st.session_state.get(prefix)
                if slm is None:
                    st.warning("SLM未连接，无法保存相位")
                else:
                    phase, _ = slm.get_displayed_phase()
                    if phase is None:
                        st.warning("无法获取当前显示的相位数据")
                    else:
                        save_path = Path.home() / ".config" / "ao_shaping" / f"slm{slm_num}_phase.csv"
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        np.savetxt(
                            str(save_path),
                            phase.astype(np.uint16),
                            fmt="%d",
                            delimiter=",",
                        )
                        st.success(f"SLM {slm_num} 相位已保存到 {save_path}")
            except Exception as e:
                st.error(f"保存相位CSV失败: {e}")
                logger.exception(f"Failed to save phase CSV for SLM {slm_num}: {e}")


def render_phase_control(slm_num: int):
    """Phase control UI for SLM (parameterized)"""
    prefix = f"slm{slm_num}"
    st.subheader("设置相位")
    pattern_type, params = render_pattern_controls(slm_num)

    if st.button("从模式生成器生成相位", key=f"{prefix}_gen_pattern_btn"):
        try:
            slm = st.session_state[prefix]
            phase_gray = generate_phase_gray(
                slm,
                pattern_type,
                params,
            )

            # Apply shift if configured (with zero-padding instead of wrap-around)
            shift_x = st.session_state.get(f"{prefix}_shift_x", 0)
            shift_y = st.session_state.get(f"{prefix}_shift_y", 0)
            phase_gray = _apply_shift(phase_gray, shift_x, shift_y)

            # Write to next memory slot and immediately display
            mem_slot = int(st.session_state[f"{prefix}_next_memory"])
            slm.write_phase(phase_gray, memory_number=mem_slot)
            slm.display_memory(mem_slot)
            refresh_phase_preview(slm_num)

            # Update next memory slot (avoid using same slot consecutively)
            st.session_state[f"{prefix}_next_memory"] = (
                st.session_state[f"{prefix}_next_memory"] % 128
            ) + 1

            st.success(f"相位已写入内存槽 {mem_slot} 并显示")
        except Exception as e:
            st.error(f"生成或显示相位失败: {e}")
            logger.exception(f"Failed to generate/display phase for SLM {slm_num}: {e}")

    # Option to load from CSV
    uploaded_file = st.file_uploader(
        "上传CSV相位文件", type=["csv"], key=f"{prefix}_csv_file"
    )
    if uploaded_file is not None:
        if st.button("从CSV加载相位", key=f"{prefix}_load_csv_btn"):
            try:
                # Save uploaded file temporarily
                temp_path = Path(f"temp_{prefix}_phase.csv")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Load phase from CSV
                slm = st.session_state[prefix]
                phase_gray = slm.load_phase_from_csv(temp_path)

                # Apply shift if configured (with zero-padding instead of wrap-around)
                shift_x = st.session_state.get(f"{prefix}_shift_x", 0)
                shift_y = st.session_state.get(f"{prefix}_shift_y", 0)
                phase_gray = _apply_shift(phase_gray, shift_x, shift_y)

                # Write to next memory slot and immediately display
                mem_slot = st.session_state[f"{prefix}_next_memory"]
                slm.write_phase(phase_gray, memory_number=mem_slot)
                slm.display_memory(mem_slot)
                refresh_phase_preview(slm_num)

                # Update next memory slot
                st.session_state[f"{prefix}_next_memory"] = (
                    st.session_state[f"{prefix}_next_memory"] % 128
                ) + 1

                st.success(f"相位已从CSV加载到内存槽 {mem_slot} 并显示")

                # Clean up temp file
                temp_path.unlink()
            except Exception as e:
                st.error(f"加载CSV相位失败: {e}")
                logger.exception(f"Failed to load CSV phase for SLM {slm_num}: {e}")


if __name__ == "__main__":
    main()
