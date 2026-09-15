"""see docs at docs[docs/slm_gui_manual.md]"""

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

from ao_shaping.drivers.slm.santec import Santec
from ao_shaping.gui.slm.pattern_controls import (
    PATTERN_REGISTRY,
    PatternControl,
    generate_phase_gray,
    refresh_phase_preview,
)

# Timeout (s) for probing a single SLM during device discovery.  The Santec SDK
# has historically hung on SLM_Ctrl_ReadSD after rapid open/close cycles, so the
# probe runs in a daemon thread and is abandoned after this interval.
SLM_PROBE_TIMEOUT_S = 3.0


# ============================================================================
# Rerun 时间点分析 — 每个函数应在何时触发 st.rerun()
#
# Streamlit rerun 触发机制:
#   - 用户交互 (按钮/滑块/选择框) → 全页 rerun (main() 被调用)
#   - @st.fragment(run_every=N) → fragment 每 N 秒自动 rerun (不影响全局)
#   - st.rerun() → 立即全页 rerun
#   - st.rerun(scope="fragment") → 立即 fragment rerun (仅限 fragment 内)
#
# 通用原则:
#   - 修改了用户可见状态 (UI 显示或 session_state) → 需要 rerun
#   - 仅修改了后台状态 (SLM 硬件/线程) 但不影响 UI → 可用 fragment 自动刷新代替
# ============================================================================


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
            st.session_state[f"{prefix}_base_phase"] = None
            st.session_state[f"{prefix}_overlay_base"] = False
        else:
            slm = st.session_state[prefix]
            if slm is not None and not getattr(slm, "is_open", False):
                st.session_state[prefix] = None
                st.session_state[f"{prefix}_connected"] = False
                st.session_state[f"{prefix}_toggle_phase_a"] = None
                st.session_state[f"{prefix}_toggle_phase_b"] = None
                st.session_state[f"{prefix}_base_phase"] = None
                st.session_state[f"{prefix}_overlay_base"] = False
            st.session_state[f"{prefix}_toggle_active"] = False
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
    import ao_shaping.drivers.slm.santec._slm_win as slm_sdk

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


def _apply_shift(phase_gray: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
    """平移相位灰度图, 空白区域填 0.

    平移数学的唯一实现位于驱动层 :meth:`Santec.shift_phase` ——
    GUI 预览与驱动重下发共用同一函数, 保证预览与上屏字节级一致。
    """
    return Santec.shift_phase(phase_gray, shift_x, shift_y)


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


def _reset_context_widgets(slm_num: int) -> None:
    """Reset control-space widget keys that mirror live SLM context.

    After the driver's wavelength or 2π max-grayscale changes (set_wavelength,
    grayscale apply/read), Streamlit would otherwise keep the stale
    first-render values in session_state for keys such as ``slm{N}_flat_gray``
    / ``slm{N}_vortex_wavelength`` — popping them makes the next rerun
    re-initialize those widgets from the new control defaults (the controls
    are rebuilt every rerun from the live SLM context in ``_build_control``).

    Rerun: none directly — the caller decides; ``set_wavelength`` rides the
    full rerun its sidebar button already triggers, the grayscale fragment
    handlers call ``st.rerun(scope="app")`` explicitly.
    """
    prefix = f"slm{slm_num}"
    for control_cls in PATTERN_REGISTRY.values():
        for suffix in control_cls.CONTEXT_SYNCED_WIDGETS:
            key = f"{prefix}_{suffix}"
            if key in st.session_state:
                del st.session_state[key]


def _build_control(pattern_type: str, slm_num: int) -> PatternControl:
    """Build the ``PatternControl`` for ``pattern_type`` on SLM ``slm_num``.

    All device geometry (id, wavelength, pixel pitch, panel size, bit depth)
    is resolved here and passed through ``__init__`` — the control classes
    themselves never read ``st.session_state``.  The connected SLM object
    wins; before connection the widget defaults stored in
    ``st.session_state[f"slm{slm_num}_..."]`` (set by ``set_wavelength`` /
    ``connect_slm``) are used.
    """
    prefix = f"slm{slm_num}"
    slm = st.session_state.get(prefix)
    if slm is not None:
        panel_res = getattr(slm, "Panel_Res", (1920, 1200))
        raw_max_gray = getattr(slm, "_max_gray", None)
        return PATTERN_REGISTRY[pattern_type](
            slm_id=slm_num,
            wavelength=float(getattr(slm, "wavelength", None) or 1064.0),
            pixel_pitch_um=float(getattr(slm, "Pitch_um", 8.0)),
            width=int(panel_res[0]),
            height=int(panel_res[1]),
            bits=int(getattr(slm, "Gray_Scale_bits", 10)),
            max_gray=int(raw_max_gray) if isinstance(raw_max_gray, int) else None,
        )
    return PATTERN_REGISTRY[pattern_type](
        slm_id=slm_num,
        wavelength=float(st.session_state.get(f"{prefix}_wavelength", 1064)),
        pixel_pitch_um=float(st.session_state.get(f"{prefix}_pixel_pitch_um", 8.0)),
        width=int(st.session_state.get(f"{prefix}_width", 1920)),
        height=int(st.session_state.get(f"{prefix}_height", 1200)),
        bits=10,
    )


def render_pattern_controls(slm_num: int) -> tuple[str, dict[str, Any]]:
    """Render the per-SLM pattern-type widgets and return ``(pattern_type, params)``.

    The widget tree lives on the :class:`PatternControl` classes in
    ``pattern_controls.py``; this function only picks the pattern type and
    builds the control via ``__init__`` (so widget keys stay ``slm{N}_*`` and
    defaults follow the connected SLM), then delegates the rendering.
    """
    prefix = f"slm{slm_num}"

    pattern_type = st.selectbox(
        "选择相位图类型",
        options=list(PATTERN_REGISTRY.keys()),
        key=f"{prefix}_pattern_type",
    )

    control = _build_control(pattern_type, slm_num)
    return pattern_type, control.render(prefix)


def main():
    if "available_slms" not in st.session_state:
        _initialize_slm_state()
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
    """Connect to the specified SLM.

    Creates the :class:`Santec` instance with only the SLM number,
    then calls ``open()`` which handles its own init flow:
    reading serial → loading config file (if any) → applying device
    defaults.  After ``open()`` all resolved SLM state is synced back to
    session state so the UI widgets always reflect reality.

    Rerun: after successful connection (st.rerun) — needed to update
    sidebar device list and reveal per-SLM controls.
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
        slm = Santec(slm_number=slm_num)
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
        st.session_state[f"{prefix}_loaded_config"] = (
            slm.load_config() if slm._serial_number else {}
        )

        serial = slm._serial_number
        if serial:
            st.session_state[mismatch_key] = {
                "serial": serial,
                "wavelength": slm.wavelength,
            }
        else:
            st.session_state[mismatch_key] = None

        _refresh_device_list()
        st.rerun()
    except Exception as e:
        st.session_state[prefix] = None
        st.session_state[f"{prefix}_connected"] = False
        st.error(f"SLM {slm_num} 连接失败: {e}")
        logger.exception(f"Failed to connect SLM {slm_num}: {e}")


def disconnect_slm(slm_num: int):
    """Disconnect from the specified SLM.

    Rerun: after successful disconnection (st.rerun) — needed to
    update sidebar device list and hide per-SLM controls.
    """
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
        st.session_state[f"{prefix}_base_phase"] = None
        st.session_state[f"{prefix}_overlay_base"] = False
        st.session_state[f"{prefix}_loaded_config"] = None

        st.rerun()


def set_wavelength(slm_num: int):
    """Set wavelength for the specified SLM.

    Rerun: none directly — uses refresh_phase_preview() to update cache,
    then @st.fragment(run_every=1.0) auto-refreshes UI within ~1s. The
    sidebar button already triggers a full rerun, which re-initializes the
    context-derived control widgets (flat_gray / vortex_wavelength / ...)
    from the new wavelength and 2π max-grayscale on the same rerun.
    """
    prefix = f"slm{slm_num}"
    wavelength_key = f"{prefix}_wavelength"
    try:
        slm = st.session_state.get(prefix)
        if slm is not None:
            wavelength = st.session_state[wavelength_key]
            slm.set_wavelength(wavelength)
            st.success(f"SLM {slm_num} 波长设置为 {wavelength} nm")
            refresh_phase_preview(slm_num)
            _reset_context_widgets(slm_num)
    except Exception as e:
        st.error(f"设置波长失败: {e}")
        logger.exception(f"Failed to set wavelength for SLM {slm_num}: {e}")


def set_shift(slm_num: int):
    """Apply shift values from the UI to the SLM driver.

    Delegates to the driver-level :meth:`Santec.apply_shift`, which
    handles absolute-positioning re-display (undo old shift → apply new),
    memory-slot rotation, settle wait and config save — identical semantics
    for every caller (runner, script, other UI).

    Rerun: none directly — uses refresh_phase_preview() to update cache,
    then @st.fragment(run_every=1.0) auto-refreshes UI within ~1s.
    """
    prefix = f"slm{slm_num}"
    try:
        _stop_toggle(slm_num)

        slm = st.session_state.get(prefix)
        if slm is not None:
            sx = st.session_state.get(f"{prefix}_shift_x", 0)
            sy = st.session_state.get(f"{prefix}_shift_y", 0)

            shifted = slm.apply_shift(
                shift_x=int(sx),
                shift_y=int(sy),
                wait_time_s=0.3,
                save_config=True,
            )
            if shifted is not None:
                refresh_phase_preview(slm_num)
                st.success(
                    f"SLM {slm_num} 平移已应用并自动下发: shift_x={sx}, shift_y={sy}"
                )
            else:
                st.success(f"SLM {slm_num} 平移参数已更新: shift_x={sx}, shift_y={sy}")
                st.info("当前没有缓存的相位可自动下发；请先生成或捕获一个相位图案。")
    except Exception as e:
        st.error(f"应用平移失败: {e}")
        logger.exception(f"Failed to set shift for SLM {slm_num}: {e}")


def set_video_mode(slm_num: int, mode_label: str):
    """Set video mode for the specified SLM (memory mode only).

    Rerun: none directly — uses refresh_phase_preview() to update cache,
    then @st.fragment(run_every=1.0) auto-refreshes UI within ~1s.

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

    Rerun: none directly — uses refresh_phase_preview() to update cache,
    then @st.fragment(run_every=1.0) auto-refreshes UI within ~1s.

    启用时从配置文件重新加载矫正数据；
    禁用时清空矫正对象（write_phase 不再叠加矫正）。
    """
    prefix = f"slm{slm_num}"
    slm = st.session_state.get(prefix)
    if slm is None:
        return

    from ao_shaping.drivers.slm.santec.wavefront_correction import WavefrontCorrection

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
def _render_grayscale_config(slm_num: int, prefix: str, slm_obj: Santec) -> None:
    """灰度设置区块（fragment 局部刷新）。

    run_every=1.0 使"获取当前2π灰度 / 应用灰度设置"之后约 1s 内重新渲染本区块，
    让 caption 显示最新的 slm_obj._max_gray。两个按钮处理器在更新 _max_gray 后
    触发整页 rerun（st.rerun(scope="app")），使主区域的控制组件从新的 _max_gray
    重新初始化（仅 fragment 级 rerun 会让 flat_gray 滑块保持旧值）。
    """
    st.caption("灰度设置")
    max_gray = int(getattr(slm_obj, "_max_gray", Santec.MAX_GRAYSCALE_VALUE))
    max_gray_abs = int(Santec.MAX_GRAYSCALE_VALUE)
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
            _reset_context_widgets(slm_num)
            st.rerun(scope="app")
        except Exception as e:
            st.error(f"更新灰度设置失败: {e}")

    if st.button("获取当前2π灰度", key=f"{prefix}_read_max_gray_btn"):
        try:
            _wl, current_max_gray = slm_obj.get_wavelength_info()
            slm_obj._max_gray = int(current_max_gray)
            refresh_phase_preview(slm_num)
            _reset_context_widgets(slm_num)
            st.rerun(scope="app")
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
            loaded_cfg = st.session_state.get(f"{prefix}_loaded_config", {})
            if loaded_cfg:
                st.caption("已加载配置文件:")
                st.json(loaded_cfg)
            else:
                st.caption("未找到匹配的配置文件（使用设备默认值）")
            if st.button("刷新配置", key=f"{prefix}_refresh_config_btn"):
                fresh = slm_obj.load_config() if slm_obj._serial_number else {}
                st.session_state[f"{prefix}_loaded_config"] = fresh
                st.rerun()

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
            slm.display_data(target_phase, memory_number=slot)
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

            slm: Santec = st.session_state[prefix]

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

            phase = generate_phase_gray(
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
            slm.display_data(phase)
            refresh_phase_preview(slm_num)
            st.success(
                f"相位已写入内存槽 {slm.get_displayed_memory_number()} 并显示（验证通过）"
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
        if st.button("设为全0相位 A", key=f"{prefix}_set_zero_phase_a"):
            slm = st.session_state.get(prefix)
            if slm is not None and getattr(slm, "is_open", False):
                h = slm.Panel_Res[1]
                w = slm.Panel_Res[0]
                st.session_state[f"{prefix}_toggle_phase_a"] = np.zeros(
                    (h, w), dtype=np.uint16
                )
                st.success(f"相位 A 已设为全0相位 ({h}×{w})")
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
        if st.button("设为全0相位 B", key=f"{prefix}_set_zero_phase_b"):
            slm = st.session_state.get(prefix)
            if slm is not None and getattr(slm, "is_open", False):
                h = slm.Panel_Res[1]
                w = slm.Panel_Res[0]
                st.session_state[f"{prefix}_toggle_phase_b"] = np.zeros(
                    (h, w), dtype=np.uint16
                )
                st.success(f"相位 B 已设为全0相位 ({h}×{w})")
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

            # 1. 从 CSV 加载灰度矩阵（驱动层已内置格式校验：
            #    标题 "Y/X"、尺寸=面板分辨率、值范围 0..1023）
            slm = st.session_state[prefix]
            phase_gray = slm.load_gray_from_csv(temp_path)

            # 2. 灰度 → 弧度制相位图（静态方法，不做硬件相关转换）
            phase_rad = Santec.csv_to_phase(temp_path)

            # 3. 走 display_phase 标准路径：
            #    create_phase_from_array() 内部自动完成
            #    弧度→灰度 + 矫正 + LUT + 平移（与 GUI 预览共用同一
            #    shift_phase，保证预览与上屏字节级一致），再写入内存槽
            #    并等待像素翻转完成。
            slm.display_phase(phase_rad)

            # 4. 验证相位确实已上屏（读回缓存对比）
            mem_slot = slm.get_displayed_memory_number()
            refresh_phase_preview(slm_num)

            st.success(
                f"相位已从CSV加载（灰度→弧度→create_phase_from_array"
                f"→display_phase）并显示到内存槽 {mem_slot}"
            )

            # Clean up temp file
            temp_path.unlink()
        except Exception as e:
            st.error(f"加载CSV相位失败: {e}")
            logger.exception(f"Failed to load CSV phase for SLM {slm_num}: {e}")

    st.divider()
    st.subheader("底相位")
    st.caption("将当前显示相位保存为底相位，后续发送相位时可自动叠加")

    col_base1, col_base2 = st.columns(2)
    with col_base1:
        if st.button("保存当前相位为底相位", key=f"{prefix}_save_base_phase"):
            slm = st.session_state.get(prefix)
            if slm is not None and getattr(slm, "is_open", False):
                phase, source = slm.get_displayed_phase()
                if phase is not None:
                    st.session_state[f"{prefix}_base_phase"] = phase.copy()
                    overlay = st.session_state.get(f"{prefix}_overlay_base", False)
                    slm._overlay_base_phase = overlay
                    st.success(f"底相位已保存 ({phase.shape}, 来源: {source})")
                else:
                    st.warning("无法获取当前显示相位，请先显示一个相位图案")
            else:
                st.warning("SLM 未连接")
    with col_base2:
        overlay = st.checkbox(
            "发送时自动叠加底相位",
            value=st.session_state.get(f"{prefix}_overlay_base", False),
            key=f"{prefix}_overlay_base_cb",
            help="勾选后，写入SLM的相位会自动叠加已保存的底相位",
        )
        if overlay != st.session_state.get(f"{prefix}_overlay_base", False):
            st.session_state[f"{prefix}_overlay_base"] = overlay
            slm = st.session_state.get(prefix)
            if slm is not None:
                slm._overlay_base_phase = overlay
                st.success(f"底相位叠加已{'启用' if overlay else '禁用'}")

    base_phase = st.session_state.get(f"{prefix}_base_phase")
    if base_phase is not None:
        st.caption(f"当前底相位: 已保存，尺寸 {base_phase.shape}")
        if st.button("清除底相位", key=f"{prefix}_clear_base_phase"):
            st.session_state[f"{prefix}_base_phase"] = None
            st.session_state[f"{prefix}_overlay_base"] = False
            slm = st.session_state.get(prefix)
            if slm is not None:
                slm._overlay_base_phase = False
            st.success("底相位已清除")
    else:
        st.caption("当前底相位: 未设置")


if __name__ == "__main__":
    main()
