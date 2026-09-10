from __future__ import annotations

import ctypes
import io
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from loguru import logger

from ao_shaping.algorithm.beam_shaping_utils import (
    build_square_target_amplitude,
    compute_square_side,
    measure_spot_diameter_cam,
)
from ao_shaping.algorithm.gerchberg_saxton import gerchberg_saxton
from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200
from ao_shaping.utils.pattern_helper import PatternHelper, calc_blazed_grating_period
from ao_shaping.utils.zernike_calc import get_zernike_name

# Global pattern helpers (will be recreated per-SLM based on resolution)
# Note: Resolution and bit depth now come from the SLM object when generating patterns

# Timeout (s) for probing a single SLM during device discovery.  The Santec SDK
# has historically hung on SLM_Ctrl_ReadSD after rapid open/close cycles, so the
# probe runs in a daemon thread and is abandoned after this interval.
SLM_PROBE_TIMEOUT_S = 3.0


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


def _stop_toggle(slm_num: int) -> None:
    """Stop the background toggle thread and clear its session state.

    Safe to call when no toggle is running (no-op). Extracted to avoid
    duplicating the same 8-line cleanup block in three places.
    """
    prefix = f"slm{slm_num}"
    stop_event = st.session_state.get(f"{prefix}_toggle_stop_event")
    if stop_event is not None:
        stop_event.set()
    st.session_state[f"{prefix}_toggle_active"] = False
    st.session_state[f"{prefix}_toggle_thread"] = None
    st.session_state[f"{prefix}_toggle_stop_event"] = None
    st.session_state[f"{prefix}_toggle_freq_ref"] = None
    st.session_state[f"{prefix}_toggle_slm_container"] = None


def _pick_next_memory(slm_num: int) -> int:
    """Pick the next SLM memory slot in 2..125, excluding the currently
    displayed one.

    The Santec firmware treats ``display_memory(slot)`` as a no-op when
    that slot is already displayed, so consecutive writes MUST target
    different slots.  This helper survives process restarts because the
    currently-displayed slot is queried from the device at call time.
    """
    prefix = f"slm{slm_num}"
    slm = st.session_state.get(prefix)
    last_slot: int | None = None
    if slm is not None and getattr(slm, "is_open", False):
        try:
            last_slot = slm.get_displayed_memory_number()
        except Exception:
            last_slot = None
    candidates = [s for s in range(2, 126) if s != last_slot]
    return int(np.random.choice(candidates))


def _probe_slm(slm_num: int) -> dict | None:
    """Probe a single SLM by number; return device dict or None.

    Returns ``None`` when no device is attached to ``slm_num``.  When a device
    is attached but already open (by this or another instance) the returned
    dict carries ``in_use=True`` / ``status="使用中"`` so the UI can show it
    without attempting a second ``SLM_Ctrl_Open``.

    Runs in a daemon thread with a timeout so a hung SDK call cannot block
    the Streamlit server (the Santec SDK has historically hung on
    ``SLM_Ctrl_ReadSD`` after rapid open/close cycles).
    """
    import ao_shaping.drivers.slm._slm_win as slm_sdk

    result: list[dict | None] = [None]

    def worker() -> None:
        try:
            ret = slm_sdk.SLM_Ctrl_Open(slm_num)
            if ret == 0:
                # Device is free — read its serial then release it.
                device_id = ctypes.create_string_buffer(256)
                ret2 = slm_sdk.SLM_Ctrl_ReadSD(slm_num, device_id)
                serial = device_id.value.decode("utf-8").strip() if ret2 == 0 else None
                result[0] = {
                    "slm_number": slm_num,
                    "serial": serial,
                    "connected": False,
                    "in_use": False,
                    "status": "未连接",
                }
            elif ret > 0:
                # Positive return codes (SLM_NG=1, SLM_IS_BUSY=2, …) mean the
                # SDK recognised a device at this index but could not open it
                # — almost always because another instance already holds it.
                # Negative codes (SLM_NOT_OPEN_USB=-200,
                # FT_DEVICE_NOT_FOUND=-10002, …) mean no device is attached.
                logger.debug(f"SLM #{slm_num} 已被占用 (返回码 {ret})")
                result[0] = {
                    "slm_number": slm_num,
                    "serial": None,
                    "connected": False,
                    "in_use": True,
                    "status": "使用中",
                }
            else:
                logger.debug(f"SLM #{slm_num} 无设备 (返回码 {ret})")
        except Exception:
            logger.debug(f"扫描 SLM #{slm_num} 时发生异常", exc_info=True)
        finally:
            try:
                slm_sdk.SLM_Ctrl_Close(slm_num)
            except Exception:
                pass

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=SLM_PROBE_TIMEOUT_S)
    if t.is_alive():
        logger.warning(f"SLM #{slm_num} 扫描超时 ({SLM_PROBE_TIMEOUT_S}s)，跳过")
    return result[0]


def _scan_available_slms(max_slms: int = 8) -> list[dict]:
    """Scan for available SLM devices without disrupting existing connections."""
    available: list[dict] = []
    for slm_num in range(1, max_slms + 1):
        prefix = f"slm{slm_num}"
        connected = bool(st.session_state.get(f"{prefix}_connected"))
        slm_obj = st.session_state.get(prefix)

        if connected:
            # This session already controls the SLM — always list it, even if
            # the cached object has since been closed (e.g. by a prior rerun
            # cleanup).  Dropping the ``is_open`` guard here was the root cause
            # of connected SLMs disappearing from the device table.
            available.append(
                {
                    "slm_number": slm_num,
                    "serial": getattr(slm_obj, "_serial_number", None)
                    if slm_obj is not None
                    else None,
                    "connected": True,
                    "in_use": bool(getattr(slm_obj, "is_open", False))
                    if slm_obj is not None
                    else False,
                    "status": "已连接",
                }
            )
            continue

        device = _probe_slm(slm_num)
        if device is not None:
            available.append(device)
    return available


def _refresh_device_list() -> None:
    """Refresh the list of available SLM devices without changing connections."""
    st.session_state["available_slms"] = _scan_available_slms()


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


@st.fragment(run_every=1.0)
def render_phase_preview(slm_num: int) -> None:
    """Display the current SLM phase preview, refreshing only this area.

    Wrapped in ``@st.fragment(run_every=1.0)`` so that whenever a phase write
    or SLM config change calls :func:`refresh_phase_preview` (updating the
    session-state cache), this section re-renders automatically within ~1s —
    no full-page ``st.rerun``, no manual click required. Widget interactions
    inside this fragment (the refresh button) also rerun only this fragment.
    """
    if st.button("刷新当前显示相位", key=f"slm{slm_num}_refresh_phase_btn"):
        refresh_phase_preview(slm_num)
        st.rerun(scope="fragment")

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
            "GS方形整形",
            "稳像法整形",
        ],
        key=f"{prefix}_pattern_type",
    )

    params: dict[str, Any] = {}

    if pattern_type == "平场":
        params["flat_gray"] = st.number_input(
            "灰度",
            min_value=0,
            max_value=int(SantecSLM200.MAX_GRAYSCALE_VALUE),
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
            default_wl = int(st.session_state.get(f"{prefix}_wavelength", 1064))
            default_pitch = float(st.session_state.get(f"{prefix}_pixel_pitch_um", 8.0))

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
        # 从 sidebar (connect 时从硬件同步) 读取默认值, 用户只需填焦距;
        # 像素间距/孔径半径默认即 SLM 硬件真实值 (8um / min(w,h)//2).
        default_pitch = float(st.session_state.get(f"{prefix}_pixel_pitch_um", 8.0))
        default_w = int(st.session_state.get(f"{prefix}_width", 1920))
        default_h = int(st.session_state.get(f"{prefix}_height", 1200))
        default_radius = min(default_w, default_h) // 2

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
            value=default_pitch,
            step=0.1,
            key=f"{prefix}_lens_pixel_pitch",
        )
        params["lens_radius"] = st.number_input(
            "透镜半径 (像素)",
            min_value=1,
            max_value=2000,
            value=default_radius,
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
        default_wavelength = int(st.session_state.get(f"{prefix}_wavelength", 1064))
        default_pixel_pitch = float(
            st.session_state.get(f"{prefix}_pixel_pitch_um", 8.0)
        )

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
        # 默认孔径 = SLM 面板短边一半 (与透镜分支一致, 从 sidebar 硬件同步值读取)
        default_w = int(st.session_state.get(f"{prefix}_width", 1920))
        default_h = int(st.session_state.get(f"{prefix}_height", 1200))
        params["radius"] = st.number_input(
            "孔径半径 (像素)",
            value=min(default_w, default_h) // 2,
            min_value=1,
            max_value=2000,
            step=1,
            key=f"{prefix}_zernike_radius",
        )

        # Collect coefficients for all (n, m) pairs up to n_max.
        # Rendered as a single st.data_editor table for performance: the old
        # per-coefficient st.columns loop created ~3 containers per pair
        # (48 for n_max=6, 90 for n_max=10) and was a bottleneck on rerun.
        pairs: list[dict[str, Any]] = []
        for n in range(n_max + 1):
            for m in range(-n, n + 1):
                if (n - abs(m)) % 2 == 0:  # Valid Zernike order
                    default_val = 1.0 if n == 0 and m == 0 else 0.0
                    key = f"{prefix}_zernike_{n}_{m}"
                    pairs.append(
                        {
                            "n": n,
                            "m": m,
                            "name": get_zernike_name(n, m) or f"n={n},m={m}",
                            "coeff": float(st.session_state.get(key, default_val)),
                        }
                    )

        edited = st.data_editor(
            pairs,
            key=f"{prefix}_zernike_table",
            num_rows="fixed",
            column_config={
                "n": st.column_config.NumberColumn("n", disabled=True, width="small"),
                "m": st.column_config.NumberColumn("m", disabled=True, width="small"),
                "name": st.column_config.TextColumn(
                    "名称", disabled=True, width="medium"
                ),
                "coeff": st.column_config.NumberColumn(
                    "系数",
                    min_value=-100.0,
                    max_value=100.0,
                    step=0.001,
                ),
            },
            hide_index=True,
        )

        coefficients: dict[tuple[int, int], float] = {}
        for row in edited:
            coefficients[(int(row["n"]), int(row["m"]))] = float(row["coeff"])
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
        # Direction selector — only meaningful for top/bottom split
        # (left/right split forces vertical direction).
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

    elif pattern_type == "GS方形整形":
        uploaded = st.file_uploader(
            "上传远场光斑图片 (用于自动计算方形大小)",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            key=f"{prefix}_gs_spot_image",
        )
        st.caption("GS 算法将输入光斑整形为方形；方形边长 = 光斑直径 × 尺寸因子")
        if uploaded is not None:
            try:
                params["intensity_cam"] = _upload_to_intensity(uploaded)
            except Exception as e:
                st.error(f"图片解码失败: {e}")
                logger.exception(f"Failed to decode GS spot image: {e}")
        else:
            st.warning("请先上传远场光斑图片")

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            params["gs_factor"] = st.number_input(
                "方形边长/光斑直径 因子",
                min_value=0.5,
                max_value=5.0,
                value=1.5,
                step=0.1,
                key=f"{prefix}_gs_factor",
            )
        with col_g2:
            params["gs_focal_length_mm"] = st.number_input(
                "焦距 (mm)",
                min_value=1.0,
                max_value=5000.0,
                value=100.0,
                step=1.0,
                key=f"{prefix}_gs_focal_mm",
            )

        col_g3, col_g4 = st.columns(2)
        with col_g3:
            params["gs_iterations"] = st.number_input(
                "GS 迭代次数",
                min_value=1,
                max_value=5000,
                value=100,
                step=1,
                key=f"{prefix}_gs_iterations",
            )
        with col_g4:
            params["gs_energy"] = st.number_input(
                "光斑能量占比 (0~1)",
                min_value=0.1,
                max_value=0.999,
                value=0.90,
                step=0.05,
                key=f"{prefix}_gs_energy",
            )

        p_cam_input = st.number_input(
            "相机像素间距 (μm，留空使用 SLM 像素间距)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key=f"{prefix}_gs_p_cam",
            help="输入相机像素间距(μm)。为 0 时默认与 SLM 像素间距一致。",
        )
        if p_cam_input > 0:
            params["gs_p_cam"] = p_cam_input * 1e-6

        params["gs_live_display"] = st.checkbox(
            "实时显示到 SLM（每轮迭代同步下发相位）",
            value=st.session_state.get(f"{prefix}_gs_live_display", False),
            key=f"{prefix}_gs_live_display",
            help="勾选后 GS 每轮迭代都会把当前相位写入 SLM，方便实时观察整形收敛过程。\n"
            "写入会自动轮换内存槽，避免同槽重复写入被固件判为无操作。",
        )
        params["gs_live_interval"] = st.number_input(
            "实时显示间隔（每 N 轮迭代下发一次）",
            min_value=1,
            max_value=50,
            value=int(st.session_state.get(f"{prefix}_gs_live_interval", 1)),
            step=1,
            key=f"{prefix}_gs_live_interval",
        )

    elif pattern_type == "稳像法整形":
        st.caption(
            "稳像法 (Steady Phase Method) 生成方形平顶光束整形相位。"
            "参考: 翟中生等, 应用光学 2023, 44(4), 711-719。"
            "原理: 通过几何稳相法相位 φ(x,y) 将入射高斯光束整形为方形平顶光束, "
            "叠加闪耀光栅使平顶偏移到一级衍射位置, 与零级光空间分离。"
        )

        # Read SLM parameters for Nyquist constraint display
        slm_wavelength_nm = float(st.session_state.get(f"{prefix}_wavelength", 1064))
        slm_pixel_pitch_um = float(
            st.session_state.get(f"{prefix}_pixel_pitch_um", 8.0)
        )

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            params["focal_length_mm"] = st.number_input(
                "傅里叶透镜焦距 f (mm)",
                min_value=10.0,
                max_value=5000.0,
                value=300.0,
                step=10.0,
                key=f"{prefix}_spm_focal_mm",
                help=(
                    "SLM 后方傅里叶透镜的焦距, 即 SLM 到 CCD 之间透镜的焦距。\n"
                    "单位: mm。典型值: 100~500 mm。\n"
                    "该透镜对 SLM 上的相位分布做傅里叶变换, "
                    "CCD 放在其后焦面上接收远场衍射图样。"
                ),
            )
        with col_s2:
            params["waist_radius_um"] = st.number_input(
                "高斯光束束腰半径 w₀ (μm)",
                min_value=100.0,
                max_value=50000.0,
                value=3600.0,
                step=100.0,
                key=f"{prefix}_spm_waist_um",
                help=(
                    "入射到 SLM 表面的高斯光束 1/e² 束腰半径。\n"
                    "单位: μm。典型值: 1~5 mm (1000~5000 μm)。\n"
                    "测量方法: 用 CCD 采集光斑, 拟合高斯分布 I(r)=I₀·exp(-2r²/w₀²), "
                    "或测量光斑直径 D 后取 w₀ ≈ D/2 (强度降至 1/e² 处)。\n"
                    "注意: 此处是 SLM 面上的束腰, 非焦点处束腰。"
                ),
            )

        col_s3, col_s4 = st.columns(2)
        with col_s3:
            params["flat_top_half_length_um"] = st.number_input(
                "平顶光束半边长 L (μm)",
                min_value=10.0,
                max_value=5000.0,
                value=150.0,
                step=10.0,
                key=f"{prefix}_spm_flat_top_um",
                help=(
                    "期望的方形平顶光束的半边长 (远场焦平面上)。\n"
                    "单位: μm。平顶光束总边长 = 2L。\n"
                    "⚠️ 奈奎斯特约束: L 必须满足 L < λ·f·w₀/(π·dx²), "
                    "否则相位梯度过大导致采样混叠, 整形质量下降。\n"
                    "参考代码中 f=300mm, w₀=3.6mm, λ=1064nm, dx=12.5μm 时, "
                    "L_max ≈ 184 μm。"
                ),
            )
        with col_s4:
            params["blaze_period"] = st.number_input(
                "闪耀光栅周期 T (像素)",
                min_value=1.0,
                max_value=1000.0,
                value=4.0,
                step=0.5,
                key=f"{prefix}_spm_blaze_period",
                help=(
                    "用于分离零级光的闪耀光栅周期 (单位: SLM 像素)。\n"
                    "偏转角 θ 满足 sin(θ) = λ/(T·dx)。\n"
                    "• T=4 像素: 偏转角较大, 平顶远离零级\n"
                    "• T=8 像素: 偏转角适中\n"
                    "• T=16+ 像素: 偏转角较小, 平顶靠近零级\n"
                    "较小的 T 值产生更大的偏转角, 但衍射效率可能降低。"
                ),
            )

        params["blaze_angle_deg"] = st.number_input(
            "闪耀光栅方向 θ (度)",
            min_value=0.0,
            max_value=360.0,
            value=45.0,
            step=1.0,
            key=f"{prefix}_spm_blaze_angle",
            help=(
                "闪耀光栅的倾斜方向角 (单位: 度)。\n"
                "控制平顶光束相对于零级光的偏转方向。\n"
                "• 0°: 水平向右偏转\n"
                "• 45°: 右上方偏转 (对角线方向)\n"
                "• 90°: 垂直向上偏转\n"
                "• 180°: 水平向左偏转\n"
                "通常设为 45° 使平顶光束沿对角线方向偏移, "
                "与零级光有最大空间分离。"
            ),
        )

        # Nyquist constraint warning
        wavelength_m = slm_wavelength_nm * 1e-9
        f_m = params["focal_length_mm"] * 1e-3
        w0_m = params["waist_radius_um"] * 1e-6
        dx_m = slm_pixel_pitch_um * 1e-6
        L_max_um = (wavelength_m * f_m * w0_m / (np.pi * dx_m**2)) * 1e6
        L_input_um = params["flat_top_half_length_um"]

        st.divider()
        st.caption("📊 约束检查")
        col_check1, col_check2 = st.columns(2)
        with col_check1:
            st.metric(
                "奈奎斯特 L_max",
                f"{L_max_um:.1f} μm",
                help="L_max = λ·f·w₀/(π·dx²), 超过此值将产生采样混叠",
            )
        with col_check2:
            if L_input_um >= L_max_um:
                st.error(
                    f"⚠️ L={L_input_um:.0f}μm ≥ L_max={L_max_um:.1f}μm, "
                    "相位梯度过大, 将产生采样混叠!"
                )
            else:
                st.success(
                    f"✅ L={L_input_um:.0f}μm < L_max={L_max_um:.1f}μm, "
                    "满足奈奎斯特条件"
                )

        st.caption(
            f"当前参数: λ={slm_wavelength_nm}nm, "
            f"dx={slm_pixel_pitch_um}μm, "
            f"f={params['focal_length_mm']:.0f}mm, "
            f"w₀={params['waist_radius_um']:.0f}μm"
        )

    return pattern_type, params


def _upload_to_intensity(uploaded: Any) -> np.ndarray:
    """Decode an uploaded image file into a 2D float intensity array.

    Args:
        uploaded: A Streamlit ``UploadedFile`` (or any file-like with ``getvalue``).

    Returns:
        2D float intensity array (grayscale; color images converted via luminance).
    """
    from PIL import Image

    raw = uploaded.getvalue()
    img = Image.open(io.BytesIO(raw)).convert("L")
    return np.asarray(img, dtype=np.float64)


def generate_gs_square_phase(
    slm: SantecSLM200,
    intensity_cam: np.ndarray,
    factor: float = 1.5,
    focal_length_m: float = 0.1,
    iterations: int = 100,
    energy: float = 0.90,
    p_cam: float | None = None,
    progress_cb: Callable[[int, int, float], None] | None = None,
    live_display: bool = False,
    live_display_interval: int = 1,
) -> np.ndarray:
    """Full GS beam-shaping pipeline: measure spot -> size square -> GS phase.

    Measures the beam spot from ``intensity_cam``, auto-sizes a square target
    on the SLM grid, runs Gerchberg-Saxton at SLM resolution, and returns the
    uint16 grayscale phase for the SLM.

    Args:
        slm: SLM object (reads resolution, pitch, bits, wavelength).
        intensity_cam: 2D far-field intensity image used to size the square.
        factor: Square-to-spot size factor (default 1.5).
        focal_length_m: Focal length / propagation distance in meters.
        iterations: GS iteration count (default 100).
        energy: Encircled-energy fraction for spot measurement (default 0.90).
        p_cam: Camera pixel pitch in meters; defaults to the SLM pitch.
        progress_cb: Optional callback ``fn(iteration, total, mse)`` invoked
            after each GS iteration (1-based iteration, total iterations,
            current mean-squared error). Enables live progress in the UI.
        live_display: When True, push the evolving phase to the SLM hardware on
            every iteration so the beam-shaping progress is visible in real
            time.  Writes rotate through memory slots automatically (the Santec
            firmware treats a repeat ``display_memory`` on the same slot as a
            no-op, so consecutive writes MUST target different slots).
        live_display_interval: Push to hardware every N-th iteration when
            ``live_display`` is True (default 1 = every iteration).

    Returns:
        uint16 phase grayscale array with shape (height, width).
    """
    width = slm.Panel_Res[0]
    height = slm.Panel_Res[1]
    d_slm = float(slm.Pitch_um) * 1e-6  # um -> m
    p_cam_eff = d_slm if p_cam is None else float(p_cam)
    wavelength_nm = slm.wavelength
    assert wavelength_nm is not None, "SLM波长未设置，无法生成相位图"

    spot_d = measure_spot_diameter_cam(intensity_cam, energy=energy)
    side = compute_square_side(spot_d, factor=factor, p_cam=p_cam_eff, d_slm=d_slm)

    # Clamp the square so it fits on the SLM grid with a small margin.
    max_side = min(height, width) - 8
    if side > max_side:
        logger.warning(
            "Beam-derived square side {}px exceeds SLM grid, clamped to {}px",
            side,
            max_side,
        )
        side = max_side
    if side < 8:
        side = 8

    source = np.ones((height, width), dtype=np.float64)
    target = build_square_target_amplitude(height, width, side)

    logger.info(
        "GS square shaping: spot_d={:.1f}px, side={}px, f={:.3f}m, iters={}",
        spot_d,
        side,
        focal_length_m,
        iterations,
    )

    # Live hardware display — converts the current source-plane phase (radians)
    # to uint16 grayscale and pushes it to the SLM.  ``display_data`` rotates
    # memory slots internally so the LCOS panel actually refreshes.
    def _live_phase_cb(iteration: int, phase_rad: np.ndarray) -> None:
        if not live_display:
            return
        if live_display_interval > 1 and (iteration % live_display_interval) != 0:
            return
        try:
            gray = slm.create_phase_from_array(phase_rad)
            slm.display_data(gray, wait_time_s=0.0)
            # Refresh the phase preview so the UI reflects the latest
            # displayed pattern without waiting for the user to press the
            # manual refresh button.
            refresh_phase_preview(slm.slm_number)
            logger.debug(
                "GS live display: iteration {} phase 已发送到 SLM",
                iteration + 1,
            )
        except Exception as e:
            logger.warning(f"GS live display 失败 (iteration {iteration + 1}): {e}")

    result = gerchberg_saxton(
        source_amplitude=source,
        target_amplitude=target,
        iterations=iterations,
        cell_spacing=d_slm,
        distance=focal_length_m,
        wavelength=float(wavelength_nm) * 1e-9,
        progress_callback=(
            (lambda i, mse: progress_cb(i + 1, iterations, mse))
            if progress_cb is not None
            else None
        ),
        phase_callback=_live_phase_cb,
    )
    return slm.create_phase_from_array(result.phase)


def generate_phase_gray(
    slm: SantecSLM200,
    pattern_type: str,
    params: dict[str, Any],
    progress_cb: Callable[[int, int, float], None] | None = None,
) -> np.ndarray:
    """Generate phase pattern using SLM properties.

    Automatically reads resolution, pixel size, bit depth, and wavelength from
    the SLM object.

    Args:
        slm: Connected SLM object.
        pattern_type: One of the supported pattern names (e.g. "平场", "GS方形整形").
        params: Pattern-specific parameters (see ``render_pattern_controls``).
        progress_cb: Optional callback ``fn(iteration, total, mse)`` forwarded to
            :func:`generate_gs_square_phase` for the "GS方形整形" pattern.
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
        n_max = int(params.get("n_max", 6))
        return helper.generate_zernike_polynomial(
            coefficients=coefficients,
            radius=radius,
            n_max=n_max,
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

        # Generate vortex phase (helper returns uint16 when wrap_phase=True,
        # radians when False).
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
    if pattern_type == "GS方形整形":
        intensity_cam = params.get("intensity_cam")
        if intensity_cam is None:
            raise ValueError("GS方形整形需要先上传远场光斑图片")
        return generate_gs_square_phase(
            slm,
            intensity_cam,
            factor=float(params.get("gs_factor", 1.5)),
            focal_length_m=float(params["gs_focal_length_mm"]) * 1e-3,
            iterations=int(params.get("gs_iterations", 100)),
            energy=float(params.get("gs_energy", 0.90)),
            p_cam=params.get("gs_p_cam"),
            progress_cb=progress_cb,
            live_display=bool(params.get("gs_live_display", False)),
            live_display_interval=int(params.get("gs_live_interval", 1)),
        )
    if pattern_type == "稳像法整形":
        from scipy.special import erf

        # Get SLM parameters
        wavelength_m = float(wavelength_nm) * 1e-9  # nm -> m
        pixel_pitch_m = float(pattern_pitch_um) * 1e-6  # μm -> m

        # Get user parameters
        f = float(params["focal_length_mm"]) * 1e-3  # mm -> m
        w0 = float(params["waist_radius_um"]) * 1e-6  # μm -> m
        L = float(params["flat_top_half_length_um"]) * 1e-6  # μm -> m
        T = float(params["blaze_period"])  # pixels
        theta = float(params["blaze_angle_deg"]) * np.pi / 180  # degrees -> radians

        # Create coordinate grids (in meters)
        x_m = (np.arange(width) - width // 2) * pixel_pitch_m
        y_m = (np.arange(height) - height // 2) * pixel_pitch_m
        X_m, Y_m = np.meshgrid(x_m, y_m)

        # 1D steady phase φ(t)
        def _steady_phase_1d(t, L, lam, f, w0):
            return (
                np.sqrt(np.pi)
                * L
                / (lam * f)
                * (
                    np.sqrt(np.pi) * t / (2 * w0) * erf(t / w0)
                    + 0.5 * np.exp(-(t**2) / w0**2)
                    - 0.5
                )
            )

        # SPM phase (separable 2D)
        phi_1d_x = _steady_phase_1d(x_m, L, wavelength_m, f, w0)
        phi_1d_y = _steady_phase_1d(y_m, L, wavelength_m, f, w0)
        phi_spm = phi_1d_y[:, None] + phi_1d_x[None, :]

        # Blazed grating phase
        px = np.arange(width) - width // 2
        py = np.arange(height) - height // 2
        PX, PY = np.meshgrid(px, py)
        phi_blaze = 2 * np.pi / T * np.mod(PX * np.cos(theta) + PY * np.sin(theta), T)

        # Combined phase
        phi_total = np.mod(phi_spm + phi_blaze, 2 * np.pi)

        return slm.create_phase_from_array(phi_total)
    raise ValueError(f"未知相位图类型: {pattern_type}")


def main():
    _initialize_slm_state()

    if "available_slms" not in st.session_state:
        _refresh_device_list()

    with st.sidebar:
        render_slm_sidebar()

    connected_slms = [
        slm_num
        for slm_num in (1, 2)
        if st.session_state.get(f"slm{slm_num}_connected")
        and st.session_state.get(f"slm{slm_num}") is not None
    ]

    count = len(connected_slms)
    if count == 0:
        st.title("SLM200 控制器")
        st.info("请在左侧连接至少一个 SLM")
        return

    st.title(f"{'双' if count == 2 else '单'}SLM200 控制器")

    if count == 1:
        slm_num = connected_slms[0]
        st.header(f"SLM {slm_num} 相位控制")
        display_slm_status(slm_num)
        render_phase_preview(slm_num)
        render_phase_control(slm_num)
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.header("SLM 1 相位控制")
            display_slm_status(1)
            render_phase_preview(1)
            render_phase_control(1)

        with col2:
            st.header("SLM 2 相位控制")
            display_slm_status(2)
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
        else:
            st.session_state[mismatch_key] = None

        st.rerun()
    except Exception as e:
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
    except Exception as e:
        st.error(f"SLM {slm_num} 断开失败: {e}")
        logger.exception(f"Failed to disconnect SLM {slm_num}: {e}")
    finally:
        st.session_state[prefix] = None
        st.session_state[f"{prefix}_connected"] = False
        st.session_state[f"{prefix}_phase_preview"] = None
        st.session_state[f"{prefix}_phase_source"] = "暂无"
        _stop_toggle(slm_num)
        st.session_state[f"{prefix}_toggle_phase_a"] = None
        st.session_state[f"{prefix}_toggle_phase_b"] = None

        st.rerun()


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
            refresh_phase_preview(slm_num)
    except Exception as e:
        st.error(f"设置波长失败: {e}")
        logger.exception(f"Failed to set wavelength for SLM {slm_num}: {e}")


def set_shift(slm_num: int):
    """Apply shift values from the UI to the SLM driver.

    Syncs the ``shift_x`` / ``shift_y`` values stored in ``st.session_state``
    back to the driver instance so that subsequent ``save_config()`` calls
    persist the latest user-entered values.  After updating the driver, the
    currently displayed phase is re-shifted and pushed to the SLM so the
    change takes effect immediately, and the preview is refreshed.
    """
    prefix = f"slm{slm_num}"
    try:
        # Stop any running toggle loop so the shift write cannot race
        # with the background thread.
        _stop_toggle(slm_num)

        slm = st.session_state.get(prefix)
        if slm is not None:
            sx = st.session_state.get(f"{prefix}_shift_x", 0)
            sy = st.session_state.get(f"{prefix}_shift_y", 0)
            slm.set_shift(shift_x=int(sx), shift_y=int(sy))

            # Auto-apply the new shift to the currently displayed phase.
            # The displayed phase is already the post-shift version (it was
            # displaced by slm._apply_shift() when originally written via
            # create_phase_from_array).  Re-applying the same shift here would
            # double-displace the pattern.  Instead, call slm._apply_shift()
            # directly on the raw cache so the new shift takes effect once.
            current_phase, source = slm.get_displayed_phase()
            if current_phase is not None:
                shifted = slm._apply_shift(current_phase, int(sx), int(sy))
                mem_slot = _pick_next_memory(slm_num)
                slm.write_phase(shifted, memory_number=mem_slot)
                slm.display_memory(mem_slot)
                refresh_phase_preview(slm_num)
                st.success(
                    f"SLM {slm_num} 平移已应用并自动下发: "
                    f"shift_x={sx}, shift_y={sy} (来源: {source})"
                )
            else:
                st.success(f"SLM {slm_num} 平移参数已更新: shift_x={sx}, shift_y={sy}")
                st.info("当前没有缓存的相位可自动下发；请先生成或捕获一个相位图案。")
    except Exception as e:
        st.error(f"应用平移失败: {e}")
        logger.exception(f"Failed to set shift for SLM {slm_num}: {e}")


def set_video_mode(slm_num: int, mode_label: str):
    """Set video mode for the specified SLM (memory mode only).

    DVI mode (``video_mode=1``) is intentionally unsupported: its ``open()``
    can hang for 120s/300s and a hung controller only recovers via physical
    power cycle (see AGENTS.md). Only memory mode is offered.
    """
    prefix = f"slm{slm_num}"
    video_mode_key = f"{prefix}_video_mode"
    mode = 0 if mode_label == "内存模式" else 1
    if mode != 0:
        st.error("DVI 模式已禁用: open() 已知挂起且需物理断电恢复, 仅支持内存模式")
        return
    try:
        slm = st.session_state.get(prefix)
        if slm is not None:
            slm._set_memory_mode(mode)
            st.session_state[video_mode_key] = mode_label
            st.success(f"SLM {slm_num} 模式设置为 {mode_label}")
            refresh_phase_preview(slm_num)
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
    refresh_phase_preview(slm_num)


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


def render_slm_sidebar():
    """Render sidebar with device table and per-SLM settings."""
    st.header("SLM 设备")

    available = st.session_state.get("available_slms", [])
    if not available:
        st.caption("未发现可用设备")

    col_h1, col_h2, col_h3, col_h4 = st.columns([1, 2, 1, 1])
    with col_h1:
        st.caption("**编号**")
    with col_h2:
        st.caption("**序列号**")
    with col_h3:
        st.caption("**状态**")
    with col_h4:
        st.caption("**操作**")

    st.divider()

    for device in available:
        slm_num = device["slm_number"]
        prefix = f"slm{slm_num}"
        # ``available_slms`` is only (re)scanned at startup or via the manual
        # "刷新设备列表" button — ``connect_slm`` / ``disconnect_slm`` end with
        # ``st.rerun()``, so the ``_refresh_device_list()`` calls that used to
        # follow them never ran and the cached entry's ``connected``/``status``/
        # ``in_use`` went stale.  Connection state must always be read live from
        # session state, never from the cached device entry.
        is_connected = bool(st.session_state.get(f"{prefix}_connected", False))
        in_use = device.get("in_use", False)
        # ``in_use=True`` recorded while WE held the handle (a scan performed
        # during our own connection) describes our own handle, not an external
        # holder — it becomes free the moment we disconnect.  Only trust
        # ``in_use`` for entries the scan never marked as connected to us.
        if device.get("connected"):
            in_use = False
        status = "已连接" if is_connected else ("使用中" if in_use else "未连接")

        col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
        with col1:
            st.write(f"**SLM {slm_num}**")
        with col2:
            st.write(device["serial"] or "-")
        with col3:
            if is_connected:
                st.success(status)
            elif in_use:
                st.warning(status)
            else:
                st.caption(status)
        with col4:
            if is_connected:
                if st.button(
                    "断开", key=f"{prefix}_disconnect_table", type="secondary"
                ):
                    disconnect_slm(slm_num)
            elif in_use:
                # Physical device exists but is held by another instance —
                # cannot take over without closing the other handle first.
                st.caption("已被占用")
            else:
                if st.button("连接", key=f"{prefix}_connect_table"):
                    connect_slm(slm_num)

    st.divider()

    if st.button("刷新设备列表", key="refresh_devices_btn", type="secondary"):
        _refresh_device_list()

    st.divider()

    for slm_num in (1, 2):
        prefix = f"slm{slm_num}"
        if (
            st.session_state.get(f"{prefix}_connected")
            and st.session_state.get(prefix) is not None
        ):
            with st.expander(f"SLM {slm_num} 设置", expanded=False):
                _render_slm_settings(slm_num)


@st.fragment(run_every=1.0)
def _render_grayscale_config(slm_num: int, prefix: str, slm_obj: SantecSLM200) -> None:
    """灰度设置区块（fragment 局部刷新）。

    run_every=1.0 使"获取当前2π灰度 / 应用灰度设置"之后约 1s 内重新渲染本区块，
    让 caption 显示最新的 slm_obj._max_gray；无需整页 st.rerun。
    """
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
            refresh_phase_preview(slm_num)
            st.success(f"SLM {slm_num} 灰度值已更新为 {new_max_gray}，预览已刷新")
        except Exception as e:
            st.error(f"更新灰度设置失败: {e}")

    if st.button("获取当前2π灰度", key=f"{prefix}_read_max_gray_btn"):
        try:
            _wl, current_max_gray = slm_obj.get_wavelength_info()
            slm_obj._max_gray = int(current_max_gray)
            refresh_phase_preview(slm_num)
            st.success(f"SLM {slm_num} 当前2π灰度: {current_max_gray}")
        except Exception as e:
            st.error(f"读取灰度失败: {e}")


def _render_slm_settings(slm_num: int):
    """Render settings controls for a single connected SLM."""
    prefix = f"slm{slm_num}"
    slm_obj = st.session_state.get(prefix)
    if slm_obj is None:
        return

    st.caption("设备信息")
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.write(
            "分辨率: "
            f"{st.session_state[f'{prefix}_width']}"
            f"×{st.session_state[f'{prefix}_height']}"
        )
        st.write(f"像素间距: {st.session_state[f'{prefix}_pixel_pitch_um']} μm")
    with col_info2:
        st.write(f"Bit数: {st.session_state[f'{prefix}_bits']}")
        sn_display = slm_obj._serial_number
        st.write(f"SLM序列号: {sn_display if sn_display else '-'}")
        st.write(f"SLM编号: {slm_obj.slm_number}")

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

    # DVI 模式已知挂起 (video_mode=1 open() 可挂 120s/300s, 且挂起后 memory 模式
    # 也挂直到物理断电) —— 强制仅内存模式, 不提供 DVI 选项。
    st.selectbox(
        "视频模式",
        options=["内存模式"],
        key=f"{prefix}_video_mode",
        disabled=True,
    )
    st.caption("仅内存模式可用: DVI 模式 open() 已知挂起 (需物理断电恢复), 已禁用")
    if st.button("设置模式", key=f"{prefix}_set_mode_btn"):
        set_video_mode(slm_num, st.session_state[f"{prefix}_video_mode"])

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
    if st.button("应用平移", key=f"{prefix}_apply_shift_btn"):
        set_shift(slm_num)

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

    _render_grayscale_config(slm_num, prefix, slm_obj)

    st.divider()

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
                        refresh_phase_preview(slm_num)
                    else:
                        st.warning("矫正文件加载失败")
            except Exception as e:
                st.error(f"应用矫正失败: {e}")
                logger.exception(f"Failed to apply correction for SLM {slm_num}: {e}")
    with col_corr2:
        if st.button("清除矫正", key=f"{prefix}_clear_corr_btn"):
            try:
                slm = st.session_state.get(prefix)
                if slm is None:
                    st.warning("SLM未连接")
                else:
                    slm.load_correction_from_csv(None)
                    st.success("矫正已清除")
                    refresh_phase_preview(slm_num)
            except Exception as e:
                st.error(f"清除矫正失败: {e}")

    st.divider()

    if st.button("保存配置", key=f"{prefix}_save_config_btn"):
        slm = st.session_state.get(prefix)
        if slm is None:
            st.info("SLM未连接，配置将在连接后写入设备")
        else:
            try:
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

        # Avoid the no-op trap: never write to the slot that is already
        # displayed.  If the target slot happens to be the currently
        # displayed one, pick an alternative in 2..125.
        try:
            current_slot = slm.get_displayed_memory_number()
        except Exception:
            current_slot = None
        if slot == current_slot:
            alt = [s for s in range(2, 126) if s != current_slot]
            slot = int(np.random.choice(alt))

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
            _stop_toggle(slm_num)
            if st.session_state.get(f"{prefix}_toggle_active", False):
                st.info("已停止周期切换")

            slm = st.session_state[prefix]

            # Show live GS progress when generating a GS square-shaping phase.
            progress_cb = None
            status_ctx = None
            progress_bar = None
            if pattern_type == "GS方形整形":
                status_ctx = st.status(
                    "正在计算 GS 相位…", expanded=True, state="running"
                )
                status_ctx.write("准备 GS 迭代…")
                progress_bar = status_ctx.progress(0.0)
                if params.get("gs_live_display"):
                    status_ctx.write("实时显示已启用：每轮迭代都会把当前相位下发到 SLM")

                def _gs_progress(iteration: int, total: int, mse: float) -> None:
                    # Runs synchronously on the main Streamlit thread.
                    pct = iteration / total if total else 0.0
                    progress_bar.progress(pct)
                    status_ctx.write(f"GS 迭代 {iteration}/{total} — MSE={mse:.6f}")
                    logger.debug("GS iteration {}/{} MSE={:.6f}", iteration, total, mse)

                progress_cb = _gs_progress

            phase_gray = generate_phase_gray(
                slm,
                pattern_type,
                params,
                progress_cb=progress_cb,
            )

            if status_ctx is not None:
                status_ctx.write("GS 迭代完成")
                status_ctx.update(state="complete")

            # generate_phase_gray() already applies the configured shift
            # internally via slm.create_phase_from_array() → _apply_shift().
            # Do NOT apply a second shift here — that would double-displace
            # the phase on the LCOS panel.

            # Write to next memory slot and immediately display
            mem_slot = _pick_next_memory(slm_num)
            slm.write_phase(phase_gray, memory_number=mem_slot)
            slm.display_memory(mem_slot)
            display_ok = _verify_phase_displayed(slm, mem_slot)
            refresh_phase_preview(slm_num)

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

    def _export_phase_csv(phase: np.ndarray, default_name: str) -> None:
        """Save a uint16 phase array as CSV and offer it for download."""
        buf = io.BytesIO()
        np.savetxt(buf, phase, fmt="%d", delimiter=",")
        buf.seek(0)
        st.download_button(
            f"导出 {default_name}",
            data=buf,
            file_name=default_name,
            mime="text/csv",
            key=f"{prefix}_export_{default_name}",
        )

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
        phase_a = st.session_state.get(f"{prefix}_toggle_phase_a")
        if phase_a is not None:
            _export_phase_csv(phase_a, f"slm{slm_num}_phase_a.csv")
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
        phase_b = st.session_state.get(f"{prefix}_toggle_phase_b")
        if phase_b is not None:
            _export_phase_csv(phase_b, f"slm{slm_num}_phase_b.csv")

    _freq = st.number_input(
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
    if uploaded_file is not None and st.button(
        "从CSV加载相位", key=f"{prefix}_load_csv_btn"
    ):
        try:
            _stop_toggle(slm_num)
            if st.session_state.get(f"{prefix}_toggle_active", False):
                st.info("已停止周期切换")

            # Save uploaded file temporarily
            temp_path = Path(f"temp_{prefix}_phase.csv")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load phase from CSV
            slm = st.session_state[prefix]
            phase_gray = slm.load_phase_from_csv(temp_path)

            # CSV loading bypasses create_phase_from_array(), so the driver's
            # internal _apply_shift() is NOT called here — the GUI-side shift
            # is the only application of the configured shift.  Keep it.
            shift_x = st.session_state.get(f"{prefix}_shift_x", 0)
            shift_y = st.session_state.get(f"{prefix}_shift_y", 0)
            phase_gray = _apply_shift(phase_gray, shift_x, shift_y)

            # Write to next memory slot and immediately display
            mem_slot = _pick_next_memory(slm_num)
            slm.write_phase(phase_gray, memory_number=mem_slot)
            slm.display_memory(mem_slot)
            display_ok = _verify_phase_displayed(slm, mem_slot)
            refresh_phase_preview(slm_num)

            if display_ok:
                st.success(f"相位已从CSV加载到内存槽 {mem_slot} 并显示（验证通过）")
            else:
                st.warning(
                    "相位已从CSV加载到内存槽 "
                    f"{mem_slot}，但显示验证失败（设备可能未刷新）"
                )

            # Clean up temp file
            temp_path.unlink()
        except Exception as e:
            st.error(f"加载CSV相位失败: {e}")
            logger.exception(f"Failed to load CSV phase for SLM {slm_num}: {e}")


if __name__ == "__main__":
    main()
