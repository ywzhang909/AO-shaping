from __future__ import annotations

import itertools
import threading
import time
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from loguru import logger

from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200
from ao_shaping.utils.pattern_helper import PatternHelper, calc_blazed_grating_period
from ao_shaping.utils.zernike_calc import get_zernike_name

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
            st.session_state[f"{prefix}_use_correction"] = True
            st.session_state[f"{prefix}_toggle_phase_a"] = None
            st.session_state[f"{prefix}_toggle_phase_b"] = None
            st.session_state[f"{prefix}_toggle_active"] = False
            st.session_state[f"{prefix}_toggle_frequency"] = 1.0
            st.session_state[f"{prefix}_toggle_thread"] = None
            st.session_state[f"{prefix}_toggle_stop_event"] = None
            st.session_state[f"{prefix}_toggle_freq_ref"] = None
            st.session_state[f"{prefix}_toggle_slm_container"] = None
        else:
            slm = st.session_state[prefix]
            if slm is not None and not getattr(slm, "is_open", False):
                st.session_state[prefix] = None
                st.session_state[f"{prefix}_connected"] = False
                st.session_state[f"{prefix}_toggle_active"] = False
                st.session_state[f"{prefix}_toggle_phase_a"] = None
                st.session_state[f"{prefix}_toggle_phase_b"] = None
                st.session_state[f"{prefix}_toggle_thread"] = None
                st.session_state[f"{prefix}_toggle_stop_event"] = None
                st.session_state[f"{prefix}_toggle_freq_ref"] = None
                st.session_state[f"{prefix}_toggle_slm_container"] = None


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


def render_pattern_controls(slm_num: int) -> tuple[str, dict[str, Any]]:
    prefix = f"slm{slm_num}"

    pattern_type = st.selectbox(
        "选择相位图类型",
        options=[
            "平场",
            "线性光栅",
            "圆形光栅",
            "透镜",
            "全息光栅",
            "闪耀光栅",
            "棋盘格",
            "二元光栅",
            "微透镜阵列",
            "湍流相位屏",
            "Zernike",
            "达曼光栅",
            "涡旋相位",
            "半半相位",
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
            key=f"{prefix}_{pattern_type}_gray",
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
    elif pattern_type == "闪耀光栅":
        period_mode = st.radio(
            "周期设置方式",
            options=["direct", "angle"],
            format_func=lambda x: (
                "直接设置周期(像素)" if x == "direct" else "根据衍射角度和波长计算"
            ),
            key=f"{prefix}_blazed_period_mode",
            horizontal=True,
            label_visibility="collapsed",
        )

        if period_mode == "angle":
            default_wl = st.session_state.get(f"{prefix}_wavelength", 1064)
            default_pitch = st.session_state.get(f"{prefix}_pixel_pitch_um", 8.0)

            col_a1, col_a2 = st.columns(2)
            with col_a1:
                angle_deg = st.number_input(
                    "衍射角度 θ (度)",
                    min_value=0.1,
                    max_value=89.0,
                    value=10.0,
                    step=0.5,
                    key=f"{prefix}_blazed_calc_angle",
                )
            with col_a2:
                calc_wl = st.number_input(
                    "波长 λ (nm)",
                    min_value=400,
                    max_value=1600,
                    value=default_wl,
                    step=1,
                    key=f"{prefix}_blazed_calc_wl",
                )

            calc_pitch = st.number_input(
                "像素间距 (μm)",
                min_value=0.1,
                max_value=100.0,
                value=default_pitch,
                step=0.1,
                key=f"{prefix}_blazed_calc_pitch",
            )

            period_pixels = calc_blazed_grating_period(angle_deg, calc_wl, calc_pitch)

            # Sync to the linked session state key for seamless mode switching
            st.session_state[f"{prefix}_blazed_period"] = period_pixels

            st.metric(
                "计算周期",
                f"{period_pixels:.1f} 像素",
                help=f"d = λ / sin(θ)，像素间距 {calc_pitch} μm",
            )
            params["period"] = period_pixels
        else:
            params["period"] = st.number_input(
                "周期 (像素)",
                min_value=1.0,
                max_value=10000.0,
                step=1.0,
                key=f"{prefix}_blazed_period",
            )

        params["phase_range"] = st.number_input(
            "相位范围 (rad)",
            min_value=0.1,
            max_value=float(2 * np.pi),
            step=0.1,
            key=f"{prefix}_blazed_phase_range",
        )
        params["direction"] = st.selectbox(
            "光栅方向",
            options=["vertical", "horizontal"],
            format_func=lambda x: (
                "竖条纹（垂直）" if x == "vertical" else "横条纹（水平）"
            ),
            key=f"{prefix}_blazed_direction",
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

        # Generate all valid (n, m) pairs for orders up to n_max
        for n in range(n_max + 1):
            for m in range(-n, n + 1):
                if (n - abs(m)) % 2 == 0:  # Valid Zernike order
                    key = f"{prefix}_zernike_{n}_{m}"

                    default_val = 1.0 if n == 0 and m == 0 else 0.0

                    st.session_state.get(key, default_val)

                    col1, col2, col3 = st.columns([1, 2, 2])
                    with col1:
                        st.write(f"Z{n},{m}")
                    with col2:
                        name = get_zernike_name(n, m)
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
    elif pattern_type == "半半相位":
        params["flat_gray"] = st.number_input(
            "平面灰度",
            min_value=0,
            max_value=1024,
            step=1,
            key=f"{prefix}_halfhalf_flat_gray",
        )
        params["split_direction"] = st.selectbox(
            "划分方向",
            options=["左右", "上下"],
            format_func=lambda x: (
                "左半平面+右半闪耀光栅" if x == "左右" else "上半平面+下半闪耀光栅"
            ),
            key=f"{prefix}_halfhalf_split",
        )
        st.caption("闪耀光栅参数")
        params["period"] = st.number_input(
            "光栅周期 (像素)",
            min_value=1.0,
            max_value=1000.0,
            step=1.0,
            key=f"{prefix}_halfhalf_period",
        )
        params["phase_range"] = st.number_input(
            "相位范围 (rad)",
            min_value=0.1,
            max_value=float(2 * np.pi),
            step=0.1,
            key=f"{prefix}_halfhalf_phase_range",
        )
        # Direction selector — only meaningful for top/bottom split (left/right forces vertical)
        split = st.session_state.get(f"{prefix}_halfhalf_split", "左右")
        if split == "上下":
            params["blaze_direction"] = st.selectbox(
                "闪耀光栅方向",
                options=["horizontal", "vertical"],
                format_func=lambda x: (
                    "横条纹（水平）" if x == "horizontal" else "竖条纹（垂直）"
                ),
                key=f"{prefix}_halfhalf_blaze_dir",
            )
        else:
            st.caption("左右划分时闪耀光栅固定为竖条纹方向")
            params["blaze_direction"] = "vertical"

    return pattern_type, params


def generate_phase_gray(
    slm: SantecSLM200,
    pattern_type: str,
    params: dict[str, Any],
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
    assert wavelength_nm is not None, "SLM波长未设置，无法生成相位图"

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
        pixel_scale = float(pattern_pitch) * 1e-6  # um -> m
        # Compute Fried parameter r0 from Cn2, L, and wavelength
        # r0 = (0.423 * k^2 * Cn2 * L)^(-3/5), k = 2π/λ
        k_wave = 2.0 * np.pi / (float(wavelength_nm) * 1e-9)
        Cn2 = float(params["Cn2"])
        L = float(params["L"])
        r0 = (0.423 * k_wave**2 * Cn2 * L) ** (-3.0 / 5.0)
        L0 = 10.0  # outer scale (m), typical atmospheric value
        helper.init_turbulence_screen(r0=r0, L0=L0, pixel_scale=pixel_scale)
        return helper.generate_turbulence_screen()
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
    if pattern_type == "闪耀光栅":
        phase_rad = helper.linear_grating(
            period=float(params["period"]),
            phase_range=float(params["phase_range"]),
            wrap_phase=False,
            direction=str(params["direction"]),
        )
        return slm.create_phase_from_array(phase_rad)
    if pattern_type == "半半相位":
        flat_gray = int(params["flat_gray"])
        period = float(params["period"])
        phase_range = float(params["phase_range"])
        split_dir = str(params["split_direction"])
        blaze_dir = str(params.get("blaze_direction", "vertical"))

        # Generate full-frame blazed grating (radians → grayscale via SLM)
        blaze_rad = helper.linear_grating(
            period=period,
            phase_range=phase_range,
            wrap_phase=False,
            direction=blaze_dir,
        )
        blaze_gray = slm.create_phase_from_array(blaze_rad)

        # Generate flat half (direct uint16 grayscale — bypass radian conversion)
        flat_full = np.full((height, width), flat_gray, dtype=np.uint16)

        # Stitch: half flat + half blaze
        if split_dir == "左右":
            mid = width // 2
            return np.concatenate([flat_full[:, :mid], blaze_gray[:, mid:]], axis=1)
        else:  # 上下
            mid = height // 2
            return np.concatenate([flat_full[:mid, :], blaze_gray[mid:, :]], axis=0)
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

        st.session_state[video_mode_key] = (
            "内存模式" if slm.video_mode == 0 else "DVI模式"
        )
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
        stop_event = st.session_state.get(f"{prefix}_toggle_stop_event")
        if stop_event is not None:
            stop_event.set()
        st.session_state[f"{prefix}_toggle_active"] = False
        st.session_state[f"{prefix}_toggle_phase_a"] = None
        st.session_state[f"{prefix}_toggle_phase_b"] = None
        st.session_state[f"{prefix}_toggle_thread"] = None
        st.session_state[f"{prefix}_toggle_stop_event"] = None
        st.session_state[f"{prefix}_toggle_freq_ref"] = None
        st.session_state[f"{prefix}_toggle_slm_container"] = None


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


def toggle_correction(slm_num: int, enabled: bool) -> None:
    """启用或禁用SLM波前误差矫正叠加

    启用时从配置文件重新加载矫正数据；
    禁用时清空矫正对象（write_phase 不再叠加矫正）。
    """
    prefix = f"slm{slm_num}"
    slm = st.session_state.get(prefix)
    if slm is None:
        return

    from ao_shaping.drivers.slm.wavefront_correction import WavefrontCorrection

    if enabled:
        config = slm.load_config()
        slm._load_correction(config)
        if slm._correction.is_valid:
            st.success(f"SLM {slm_num} 矫正已启用: {slm._correction.csv_path.name}")
        else:
            st.warning(f"SLM {slm_num} 矫正已启用，但未找到有效矫正文件")
    else:
        slm._correction = WavefrontCorrection()
        st.info(f"SLM {slm_num} 矫正已禁用")


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

        if st.button("获取当前2π灰度", key=f"{prefix}_read_max_gray_btn"):
            try:
                _wl, current_max_gray = slm_obj.get_wavelength_info()
                slm_obj._max_gray = int(current_max_gray)
                st.success(f"SLM {slm_num} 当前2π灰度: {current_max_gray}")
                st.rerun()
            except Exception as e:
                st.error(f"读取灰度失败: {e}")

        st.divider()

        # 波前误差矫正开关
        st.caption("波前误差矫正")
        use_correction = st.checkbox(
            "叠加矫正CSV",
            value=st.session_state.get(f"{prefix}_use_correction", True),
            key=f"{prefix}_use_correction_cb",
            help="启用时，写入相位会自动叠加波前误差矫正数据",
        )
        if use_correction != st.session_state.get(f"{prefix}_use_correction", True):
            st.session_state[f"{prefix}_use_correction"] = use_correction
            toggle_correction(slm_num, use_correction)

        if slm_obj._correction.is_valid:
            st.caption(f"当前矫正文件: {slm_obj._correction.csv_path.name}")
        else:
            st.caption("未加载矫正文件")

        st.divider()

        st.caption("波前矫正")
        correction_enabled = bool(getattr(slm_obj, "correction_enabled", False))
        correction_path = getattr(slm_obj, "correction_csv_path", None)
        if correction_enabled and correction_path is not None:
            st.success(
                f"矫正已启用: {correction_path.name}",
                icon="✅",
            )
        else:
            st.info("矫正未加载", icon="ℹ️")

        uploaded_correction = st.file_uploader(
            "加载矫正 CSV 文件",
            type=["csv"],
            key=f"{prefix}_correction_csv",
        )
        col_corr1, col_corr2 = st.columns(2)
        with col_corr1:
            if st.button("应用矫正", key=f"{prefix}_apply_corr_btn"):
                try:
                    slm = st.session_state.get(prefix)
                    if slm is None:
                        st.warning("SLM未连接，无法应用矫正")
                    elif uploaded_correction is None:
                        st.warning("请先选择矫正 CSV 文件")
                    else:
                        import tempfile

                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".csv"
                        ) as tmp:
                            tmp.write(uploaded_correction.getbuffer())
                            tmp_path = tmp.name
                        ok = slm.load_correction_from_csv(tmp_path)
                        if ok:
                            st.success(f"矫正已应用: {uploaded_correction.name}")
                        else:
                            st.warning("矫正文件加载失败")
                except Exception as e:
                    st.error(f"应用矫正失败: {e}")
                    logger.exception(
                        f"Failed to apply correction for SLM {slm_num}: {e}"
                    )
        with col_corr2:
            if st.button("清除矫正", key=f"{prefix}_clear_corr_btn"):
                try:
                    slm = st.session_state.get(prefix)
                    if slm is None:
                        st.warning("SLM未连接")
                    else:
                        slm.load_correction_from_csv(None)
                        st.success("矫正已清除")
                except Exception as e:
                    st.error(f"清除矫正失败: {e}")

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
                        save_path = (
                            Path.home()
                            / ".config"
                            / "ao_shaping"
                            / f"slm{slm_num}_phase.csv"
                        )
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


def _verify_phase_displayed(slm: SantecSLM200, expected_slot: int) -> bool:
    try:
        actual_slot = slm.get_displayed_memory_number()
        if actual_slot == expected_slot:
            logger.debug(f"相位显示验证成功: 内存槽 {expected_slot}")
            return True
        logger.warning(
            f"相位显示验证失败: 期望槽 {expected_slot}, 实际槽 {actual_slot}"
        )
        return False
    except Exception as e:
        logger.warning(f"相位显示验证异常: {e}")
        return False


def _toggle_phases_task(
    slm_container: list,
    phase_a: np.ndarray,
    phase_b: np.ndarray,
    stop_event: threading.Event,
    freq_ref: list,
) -> None:
    """Background toggle loop — mirrors R50 `run_loop` + `alt_tick` pattern.

    Alternates between ``phase_a`` and ``phase_b`` at ``freq_ref[0]`` Hz
    using wall-clock timing.  Never touches ``st.session_state``.
    """
    slots = [3, 4]
    t0 = time.time()

    while not stop_event.is_set():
        freq = freq_ref[0] if freq_ref else 1.0
        if freq <= 0:
            time.sleep(0.05)
            continue

        slm = slm_container[0] if slm_container else None
        if slm is None or not getattr(slm, "is_open", False):
            time.sleep(0.05)
            continue

        elapsed = time.time() - t0
        use_phase_a = int(elapsed * 2.0 * freq) % 2 == 0
        target_phase = phase_a if use_phase_a else phase_b
        slot = slots[0] if use_phase_a else slots[1]

        try:
            slm.write_phase(target_phase, memory_number=slot)
            slm.display_memory(slot)
        except Exception as e:
            logger.warning(f"周期切换失败: {e}")

        time.sleep(max(0.01, 1.0 / (2.0 * freq)))


def render_phase_control(slm_num: int):
    """Phase control UI for SLM (parameterized)"""
    prefix = f"slm{slm_num}"
    st.subheader("设置相位")
    pattern_type, params = render_pattern_controls(slm_num)

    if st.button("从模式生成器生成相位", key=f"{prefix}_gen_pattern_btn"):
        try:
            stop_event = st.session_state.get(f"{prefix}_toggle_stop_event")
            if stop_event is not None:
                stop_event.set()
            st.session_state[f"{prefix}_toggle_active"] = False
            st.session_state[f"{prefix}_toggle_thread"] = None
            st.session_state[f"{prefix}_toggle_stop_event"] = None
            st.session_state[f"{prefix}_toggle_freq_ref"] = None
            st.session_state[f"{prefix}_toggle_slm_container"] = None
            if st.session_state.get(f"{prefix}_toggle_active", False):
                st.info("已停止周期切换")

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
            display_ok = _verify_phase_displayed(slm, mem_slot)
            refresh_phase_preview(slm_num)

            # Update next memory slot (avoid using same slot consecutively)
            st.session_state[f"{prefix}_next_memory"] = (
                st.session_state[f"{prefix}_next_memory"] % 128
            ) + 1

            if display_ok:
                st.success(f"相位已写入内存槽 {mem_slot} 并显示（验证通过）")
            else:
                st.warning(
                    f"相位已写入内存槽 {mem_slot}，但显示验证失败（设备可能未刷新）"
                )
        except Exception as e:
            st.error(f"生成或显示相位失败: {e}")
            logger.exception(f"Failed to generate/display phase for SLM {slm_num}: {e}")

    st.divider()
    st.subheader("周期切换")
    st.caption("在相位 A 与相位 B 之间持续来回切换")

    col_ph_a, col_ph_b = st.columns(2)
    with col_ph_a:
        if st.button("设为相位 A", key=f"{prefix}_set_phase_a"):
            slm = st.session_state.get(prefix)
            if slm is not None and getattr(slm, "is_open", False):
                phase, _ = slm.get_displayed_phase()
                if phase is not None:
                    st.session_state[f"{prefix}_toggle_phase_a"] = phase
                    st.success(f"相位 A 已设置 ({phase.shape})")
                else:
                    st.warning("无法获取当前显示相位")
            else:
                st.warning("SLM 未连接")
    with col_ph_b:
        if st.button("设为相位 B", key=f"{prefix}_set_phase_b"):
            slm = st.session_state.get(prefix)
            if slm is not None and getattr(slm, "is_open", False):
                phase, _ = slm.get_displayed_phase()
                if phase is not None:
                    st.session_state[f"{prefix}_toggle_phase_b"] = phase
                    st.success(f"相位 B 已设置 ({phase.shape})")
                else:
                    st.warning("无法获取当前显示相位")
            else:
                st.warning("SLM 未连接")

    freq = st.number_input(
        "切换频率 (Hz)",
        min_value=0.1,
        max_value=100.0,
        value=1.0,
        step=0.1,
        key=f"{prefix}_toggle_frequency",
    )

    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("开始周期切换", key=f"{prefix}_start_toggle"):
            phase_a = st.session_state.get(f"{prefix}_toggle_phase_a")
            phase_b = st.session_state.get(f"{prefix}_toggle_phase_b")
            slm = st.session_state.get(prefix)
            if phase_a is None or phase_b is None:
                st.warning("请先设置相位 A 和相位 B")
            elif slm is None or not getattr(slm, "is_open", False):
                st.warning("SLM 未连接")
            else:
                old_stop = st.session_state.get(f"{prefix}_toggle_stop_event")
                if old_stop is not None:
                    old_stop.set()
                st.session_state[f"{prefix}_toggle_active"] = True
                stop_event = threading.Event()
                freq_ref = [st.session_state.get(f"{prefix}_toggle_frequency", 1.0)]
                slm_container = [st.session_state.get(prefix)]
                st.session_state[f"{prefix}_toggle_stop_event"] = stop_event
                st.session_state[f"{prefix}_toggle_freq_ref"] = freq_ref
                st.session_state[f"{prefix}_toggle_slm_container"] = slm_container
                thread = threading.Thread(
                    target=_toggle_phases_task,
                    args=(
                        slm_container,
                        st.session_state.get(f"{prefix}_toggle_phase_a"),
                        st.session_state.get(f"{prefix}_toggle_phase_b"),
                        stop_event,
                        freq_ref,
                    ),
                    daemon=True,
                )
                st.session_state[f"{prefix}_toggle_thread"] = thread
                thread.start()
                st.success("周期切换已开始")
    with col_stop:
        if st.button("停止周期切换", key=f"{prefix}_stop_toggle"):
            stop_event = st.session_state.get(f"{prefix}_toggle_stop_event")
            if stop_event is not None:
                stop_event.set()
            st.session_state[f"{prefix}_toggle_active"] = False
            st.session_state[f"{prefix}_toggle_thread"] = None
            st.session_state[f"{prefix}_toggle_stop_event"] = None
            st.session_state[f"{prefix}_toggle_freq_ref"] = None
            st.session_state[f"{prefix}_toggle_slm_container"] = None
            st.success("周期切换已停止")

    # Option to load from CSV
    uploaded_file = st.file_uploader(
        "上传CSV相位文件", type=["csv"], key=f"{prefix}_csv_file"
    )
    if uploaded_file is not None:
        if st.button("从CSV加载相位", key=f"{prefix}_load_csv_btn"):
            try:
                stop_event = st.session_state.get(f"{prefix}_toggle_stop_event")
                if stop_event is not None:
                    stop_event.set()
                st.session_state[f"{prefix}_toggle_active"] = False
                st.session_state[f"{prefix}_toggle_thread"] = None
                st.session_state[f"{prefix}_toggle_stop_event"] = None
                st.session_state[f"{prefix}_toggle_freq_ref"] = None
                st.session_state[f"{prefix}_toggle_slm_container"] = None
                if st.session_state.get(f"{prefix}_toggle_active", False):
                    st.info("已停止周期切换")

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
                display_ok = _verify_phase_displayed(slm, mem_slot)
                refresh_phase_preview(slm_num)

                # Update next memory slot
                st.session_state[f"{prefix}_next_memory"] = (
                    st.session_state[f"{prefix}_next_memory"] % 128
                ) + 1

                if display_ok:
                    st.success(f"相位已从CSV加载到内存槽 {mem_slot} 并显示（验证通过）")
                else:
                    st.warning(
                        f"相位已从CSV加载到内存槽 {mem_slot}，但显示验证失败（设备可能未刷新）"
                    )

                # Clean up temp file
                temp_path.unlink()
            except Exception as e:
                st.error(f"加载CSV相位失败: {e}")
                logger.exception(f"Failed to load CSV phase for SLM {slm_num}: {e}")


if __name__ == "__main__":
    main()
