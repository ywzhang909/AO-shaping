"""Multi-SLM200 控制器 (Streamlit).

完整操作手册: docs/slm/slm_gui_manual.md
操作时长分类与操作路线图: docs/slm/slm_gui_operation_map.md
"""

from __future__ import annotations

import ctypes
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from loguru import logger

from ao_shaping.drivers.slm.santec import Santec, describe_santec_error
from ao_shaping.gui.slm.bench_settings import (
    DEFAULT_CAMERA_PIXEL_UM,
    KNOWN_CAMERA_PIXEL_UM,
    auto_check_camera_pixel_um,
    build_send_overlay,
    detect_camera_pixel_um,
    render_bench_settings,
    widget_default,
)
from ao_shaping.gui.utils.phase_overlay import (
    OverlaidSantec,
    apply_overlays_to_handle,
)
from ao_shaping.gui.slm.pattern_controls import (
    PATTERN_REGISTRY,
    PREVIEW_COLORMAP_DEFAULT,
    PREVIEW_COLORMAP_LABELS,
    PatternControl,
    generate_phase_gray,
    refresh_phase_preview,
)
from ao_shaping.gui.slm.shaping_controls import render_closed_loop_shaping
from ao_shaping.utils.wavefront.pattern_helper import PatternHelper

# Timeout (s) for probing a single SLM during device discovery.  The Santec SDK
# has historically hung on SLM_Ctrl_ReadSD after rapid open/close cycles, so the
# probe runs in a daemon thread and is abandoned after this interval.
SLM_PROBE_TIMEOUT_S = 3.0

# 像元间距的探测与告警已搬到「CCD参数设置」区块（bench_settings）—— 那是一次
# 实验内固定的量，不该和「刷新相机状态」这种一次性诊断按钮混在一起。下面三个名字
# 是为旧 import 路径保留的再导出。
_DETECT_CAMERA_PIXEL_UM = detect_camera_pixel_um
_AUTO_CHECK_CAMERA_PIXEL_UM = auto_check_camera_pixel_um

__all__ = [
    "DEFAULT_CAMERA_PIXEL_UM",
    "KNOWN_CAMERA_PIXEL_UM",
    "main",
    "render_slm_sidebar",
]


def _scan_camera_status() -> list[dict]:
    """Scan for Daheng cameras and report whether each is free or occupied.

    Returns a list of dicts with keys: ``index``, ``serial``, ``status``
    (``"可用"`, ``"使用中"``, or `"错误"`).  A camera is reported as
    ``"使用中"`` when opening it raises ``InvalidAccess`` or "Access denied"
    (another process already holds the handle); in that case a tip is shown
    to the user rather than offering a force-close button.
    """
    try:
        import gxipy as gx  # type: ignore[import-untyped]
    except (ImportError, OSError) as e:
        logger.debug(f"Daheng SDK 不可用: {e}")
        st.warning("Daheng SDK 未安装，无法扫描相机状态")
        return []

    result: list[dict] = []
    try:
        device_manager = gx.DeviceManager()
        dev_count, dev_info_list = device_manager.update_device_list()
    except Exception as e:  # policy: the vendor enumeration fails wholesale when the runtime is missing or no camera is attached; the scan keeps the statuses already collected instead of blanking the panel
        logger.debug(f"Daheng 相机枚举失败: {e}")
        return result

    if not dev_count or not dev_info_list:
        return result

    for idx, dev_info in enumerate(dev_info_list):
        sn = dev_info.get("sn")
        entry: dict = {"index": idx, "serial": sn, "status": "未知"}
        try:
            cam = device_manager.open_device_by_sn(sn)
            if cam is not None:
                cam.close_device()
            entry["status"] = "可用"
        except Exception as e:  # policy: this reads the vendor message, not a typed error: already-open-in-another-session (Access denied / REPEAT_OPENED) is a normal state that must be shown as in use, not as a failure
            msg = str(e)
            if (
                "Access denied" in msg
                or "REPEAT_OPENED" in msg
                or "device has been open" in msg
            ):
                entry["status"] = "使用中"
            else:
                entry["status"] = "错误"
                logger.debug(f"Daheng 相机 {sn} 探测异常: {msg}")
        result.append(entry)
    return result


# ============================================================================
# Rerun 时间点分析 — 每个函数应在何时触发 st.rerun()
#
# Streamlit rerun 触发机制:
#   - 用户交互 (按钮/滑块/选择框) → 全页 rerun (main() 被调用)
#   - st.rerun() → 立即全页 rerun
#   - st.rerun(scope="fragment") → 立即 fragment rerun (仅限 fragment 内)
#
# 通用原则:
#   - 修改了用户可见状态 (UI 显示或 session_state) → 需要 rerun
#   - 仅修改了后台状态 (SLM 硬件/线程) 但不影响 UI → 无需 rerun
#
# 注意: 本模块的 fragment **不使用 run_every 定时器**。fragment ID 是
#   "模块.函数名 + 调用点容器 delta path" 的哈希 (fragment.py / delta_generator.py):
#   单/双 SLM 布局切换、设备连接/断开都会让调用点路径变化, fragment 在新 ID 下
#   重新注册, 旧 ID 在 fragment storage.clear() 中被剪枝; 若仍挂 run_every 定时器,
#   前端会每秒请求已被剪枝的旧 ID → 反复警告
#   "The fragment with id ... does not exist anymore"。
#   所有更新预览缓存的路径 (refresh_phase_preview 调用点) 都在各自已触发 rerun 的
#   处理器内 (按钮→全页 rerun, fragment 内按钮/选择框→fragment rerun),
#   故定时轮询完全冗余, 且是上述警告的唯一来源。
# ============================================================================


def _initialize_slm_state() -> None:
    for slm_num in (1, 2):
        prefix = f"slm{slm_num}"
        if prefix not in st.session_state:
            st.session_state[prefix] = None
            st.session_state[f"{prefix}_connected"] = False
            st.session_state[f"{prefix}_wavelength"] = 1064
            st.session_state[f"{prefix}_video_mode"] = "内存模式"
            st.session_state[f"{prefix}_next_memory"] = np.random.randint(1, 128)
            st.session_state[f"{prefix}_phase_preview"] = None
            st.session_state[f"{prefix}_phase_source"] = "暂无"
            st.session_state[f"{prefix}_preview_colormap"] = PREVIEW_COLORMAP_DEFAULT
            st.session_state[f"{prefix}_shift_x"] = 0
            st.session_state[f"{prefix}_shift_y"] = 0
            st.session_state[f"{prefix}_width"] = 1920
            st.session_state[f"{prefix}_height"] = 1200
            st.session_state[f"{prefix}_pixel_pitch_um"] = 8.0
            st.session_state[f"{prefix}_bits"] = 10
            st.session_state[f"{prefix}_use_120hz"] = False
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
            st.session_state[f"{prefix}_pattern_helper"] = None
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
        except Exception:  # policy: ctypes SDK open can raise on an invalid handle; a probe that cannot open reports not-responding and still reaches the finally-close, so one dead controller cannot abort scanning the rest
            logger.debug(f"扫描 SLM #{slm_num} 时发生异常", exc_info=True)
        finally:
            try:
                slm_sdk.SLM_Ctrl_Close(slm_num)
            except Exception:  # policy: the handle may never have opened, so this close is best-effort cleanup and its failure carries no information
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


@st.fragment()
def render_phase_preview(slm_num: int) -> None:
    """Display the current SLM phase preview, refreshing only this area.

    Wrapped in ``@st.fragment()`` so widget interactions inside this section
    (refresh button / colormap selectbox) rerun only this fragment instead of
    the whole app. There is **no ``run_every`` timer**: the preview cache
    (:func:`refresh_phase_preview`) is only ever updated inside widget handlers
    that already trigger their own rerun (full-page buttons) or a fragment-
    scoped rerun (this section's own widgets), so the preview always refreshes
    with the accompanying rerun. A periodic timer is not just redundant but
    harmful: the fragment ID derives from the caller container's delta path,
    which changes when the single/dual SLM layout flips — the old ID is pruned
    from fragment storage while the timer keeps requesting it, spamming
    "The fragment with id ... does not exist anymore".
    """
    if st.button(
        "刷新当前显示相位", key=f"slm{slm_num}_refresh_phase_btn", type="secondary"
    ):
        refresh_phase_preview(slm_num)
        st.rerun(scope="fragment")

    colormap = st.radio(
        "预览色图",
        options=PREVIEW_COLORMAP_LABELS,
        index=PREVIEW_COLORMAP_LABELS.index(
            st.session_state.get(
                f"slm{slm_num}_preview_colormap", PREVIEW_COLORMAP_DEFAULT
            )
        ),
        key=f"slm{slm_num}_preview_colormap_widget",
    )
    if (
        st.session_state.get(f"slm{slm_num}_preview_colormap", PREVIEW_COLORMAP_DEFAULT)
        != colormap
    ):
        st.session_state[f"slm{slm_num}_preview_colormap"] = colormap
        refresh_phase_preview(slm_num)
        st.rerun(scope="fragment")

    preview = st.session_state.get(f"slm{slm_num}_phase_preview")
    source = st.session_state.get(f"slm{slm_num}_phase_source", "暂无")
    st.caption(f"当前显示来源: {source}")
    if preview is None:
        st.info(
            "当前无法精确展示相位预览。"
            "只有通过本页面下发并缓存过的相位，才能保证预览与设备显示一致。"
        )
        st.caption("提示: 在「设置相位」区域生成或加载相位后，预览会自动更新。")
        return

    st.image(
        preview,
        caption=f"SLM {slm_num} 当前显示相位预览（{colormap}）",
        clamp=True,
        width="stretch",
    )

    slm = st.session_state.get(f"slm{slm_num}")
    if slm is not None:
        max_gray = int(getattr(slm, "_max_gray", None) or Santec.MAX_GRAYSCALE_VALUE)
        st.caption(f"2π 灰度值: {max_gray} | {colormap}（0→黑蓝, π→青, 2π→黑蓝，循环）")


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


def _get_slm_helper(slm_num: int, width: int, height: int, bits: int) -> PatternHelper:
    """Return the per-SLM shared :class:`PatternHelper`.

    One instance per SLM device is cached in session state; every pattern
    type on that device receives it via ``PatternControl(helper=...)`` so the
    expensive coordinate grids and Zernike generators are built once per
    device instead of once per control per rerun. Rebuilds only when the
    panel geometry or bit depth changed (e.g. reconnecting a different
    device).
    """
    key = f"slm{slm_num}_pattern_helper"
    helper = st.session_state.get(key)
    if (
        helper is None
        or helper.resolution != (int(width), int(height))
        or helper.bits != int(bits)
    ):
        helper = PatternHelper((int(width), int(height)), bits=int(bits))
        st.session_state[key] = helper
    return helper


def _build_control(pattern_type: str, slm_num: int) -> PatternControl:
    """Build the ``PatternControl`` for ``pattern_type`` on SLM ``slm_num``.

    All device geometry (id, wavelength, pixel pitch, panel size, bit depth)
    is resolved here and passed through ``__init__`` — the control classes
    themselves never read ``st.session_state``.  The connected SLM object
    wins; before connection the widget defaults stored in
    ``st.session_state[f"slm{slm_num}_..."]`` (set by ``set_wavelength`` /
    ``connect_slm``) are used.  A per-SLM :class:`PatternHelper` (cached in
    session state, see ``_get_slm_helper``) is injected so all pattern types
    on the device share one set of coordinate/Zernike caches.
    """
    prefix = f"slm{slm_num}"
    slm = st.session_state.get(prefix)
    if slm is not None:
        panel_res = getattr(slm, "Panel_Res", (1920, 1200))
        width, height = int(panel_res[0]), int(panel_res[1])
        bits = int(getattr(slm, "Gray_Scale_bits", 10))
        raw_max_gray = getattr(slm, "_max_gray", None)
        kwargs: dict[str, Any] = dict(
            slm_id=slm_num,
            wavelength=float(getattr(slm, "wavelength", None) or 1064.0),
            pixel_pitch_um=float(getattr(slm, "Pitch_um", 8.0)),
            width=width,
            height=height,
            bits=bits,
            helper=_get_slm_helper(slm_num, width, height, bits),
        )
        if isinstance(raw_max_gray, int):
            kwargs["max_gray"] = int(raw_max_gray)
    else:
        width = int(st.session_state.get(f"{prefix}_width", 1920))
        height = int(st.session_state.get(f"{prefix}_height", 1200))
        bits = 10
        kwargs = dict(
            slm_id=slm_num,
            wavelength=float(st.session_state.get(f"{prefix}_wavelength", 1064)),
            pixel_pitch_um=float(st.session_state.get(f"{prefix}_pixel_pitch_um", 8.0)),
            width=width,
            height=height,
            bits=bits,
            helper=_get_slm_helper(slm_num, width, height, bits),
        )
    return PATTERN_REGISTRY[pattern_type](**kwargs)


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

    # Debug anchor for "页面在选择某个图案后中断" reports: the log line lands
    # immediately before the crashing control renders, so the last entry in the
    # server log names the pattern type that aborted the script run.
    logger.debug(
        "相位图类型选择: slm{} -> {} (control={})",
        slm_num,
        pattern_type,
        PATTERN_REGISTRY[pattern_type].__name__,
    )

    control = _build_control(pattern_type, slm_num)
    return pattern_type, control.render(prefix)


def main():
    st.set_page_config(
        page_title="SLM200 控制器",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded",
    )
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
        render_phase_control(slm_num)
        render_phase_preview(slm_num)
        with st.expander("闭环整形", expanded=False):
            render_closed_loop_shaping(slm_num)
    else:
        for slm_num in (1, 2):
            st.header(f"SLM {slm_num} 相位控制")
            display_slm_status(slm_num)
            render_phase_control(slm_num)
            render_phase_preview(slm_num)
            with st.expander("闭环整形", expanded=False):
                render_closed_loop_shaping(slm_num)


def _safe_set_state(key: str, value: Any) -> None:
    """Set a session_state key only if no widget has claimed it yet.

    Once Streamlit instantiates a widget with ``key``, that key's entry in
    ``st.session_state`` is locked for the remainder of the script run.
    ``connect_slm`` runs before the sidebar widgets are instantiated (the button
    click handler fires before later widgets are rendered, then ``st.rerun()``
    restarts the script), so setting works.  ``_reacquire_slm_handle`` runs from
    the main area *after* all sidebar widgets exist, so the same set would raise
    ``StreamlitAPIException``.  This helper swallows that case — the value was
    already synced on first connect and the next full rerun picks it up.
    """
    try:
        st.session_state[key] = value
    except Exception:  # policy: session_state access can fail outside a script run (background thread, shutdown); letting the caller continue without the value is this helper's whole purpose
        pass


def _sync_slm_session_state(slm_num: int, slm: Any) -> None:
    """Mirror a freshly opened SLM's resolved state into session state.

    ``slm`` is the real :class:`Santec` or the controller-layer
    :class:`~ao_shaping.gui.utils.phase_overlay.OverlaidSantec` proxy that wraps
    it (the proxy forwards every attribute this reads).  Shared by
    ``connect_slm`` and the closed-loop shaping re-acquisition path
    (``shaping_controls._reacquire_slm_handle``): after ``Santec.open()`` the
    device has read its serial, loaded its per-serial config file and applied
    defaults.

    All widget-backed keys are synced here so the sidebar always reflects the
    actual device state after a connect or re-acquire.  ``connect_slm`` calls
    ``st.rerun()`` after this, so the next render instantiates widgets fresh
    from the updated session_state values.
    """
    prefix = f"slm{slm_num}"
    st.session_state[f"{prefix}_phase_preview"] = None
    st.session_state[f"{prefix}_phase_source"] = "连接成功，等待读取或下发相位"
    # ``Santec.open()`` 已经读过每序列号配置: 复用它, 不再 load_config() 二次读盘。
    # 老 driver / 测试替身没有该属性时回落到原来的读取。
    loaded_config = getattr(slm, "loaded_config", None)
    if loaded_config is None:
        loaded_config = slm.load_config() if slm._serial_number else {}
    st.session_state[f"{prefix}_loaded_config"] = loaded_config

    # Sync all widget-backed keys from the device's resolved state so the
    # sidebar shows the actual hardware values (wavelength, shift, geometry…).
    # Guard: once a widget with a given key is instantiated in the current render
    # pass, its session_state entry is locked.  ``_reacquire_slm_handle`` is
    # called from the main area (after sidebar widgets are already rendered),
    # so those keys must not be (re)set here — they were already synced on first
    # connect and the next full rerun will pick them up.  Non-widget keys below
    # are always safe.
    _safe_set_state(
        f"{prefix}_wavelength",
        int(slm.wavelength) if slm.wavelength is not None else None,
    )
    _safe_set_state(f"{prefix}_shift_x", int(getattr(slm, "_shift_x", 0)))
    _safe_set_state(f"{prefix}_shift_y", int(getattr(slm, "_shift_y", 0)))
    _safe_set_state(f"{prefix}_width", int(slm.Panel_Res[0]))
    _safe_set_state(f"{prefix}_height", int(slm.Panel_Res[1]))
    # 这个 key 被图案生成器当**采样间距**用（见 _build_slm_pattern 的
    # pixel_pitch_um），所以必须同步 Pitch_um 而不是有效口径 Pixel_Size_um
    # ——两者相差 0.2 μm，见 slm200_constants 注释。
    _safe_set_state(f"{prefix}_pixel_pitch_um", float(getattr(slm, "Pitch_um", 8.0)))
    _safe_set_state(f"{prefix}_bits", int(getattr(slm, "Gray_Scale_bits", 10)))
    _safe_set_state(f"{prefix}_use_120hz", bool(slm._use_120hz))

    serial = slm._serial_number
    if serial:
        st.session_state[f"{prefix}_wavelength_mismatch"] = {
            "serial": serial,
            "wavelength": slm.wavelength,
        }
    else:
        st.session_state[f"{prefix}_wavelength_mismatch"] = None

    # Restore the two send-pipeline overlay layers (base phase + defocus) onto
    # the freshly (re-)acquired handle, so a closed-loop shaping run (which closes
    # and re-opens the SLM) does not silently drop the operator's T4 overlays.
    # ``_sync_slm_session_state`` is the single restoration point (first connect
    # and re-acquire both funnel through it).
    _apply_slm_overlays(slm_num)


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

    # Clean up any stale SLM object before connecting
    old_slm = st.session_state.get(prefix)
    if old_slm is not None:
        try:
            # Only try to close if the device thinks it's open
            if getattr(old_slm, "is_open", False):
                old_slm.close()
        except Exception:
            pass  # policy: a stale handle that refuses to close is expected after a crash or a re-run; session state is reset regardless so the new connect starts clean
        # Always reset state in session
        st.session_state[prefix] = None
        st.session_state[f"{prefix}_connected"] = False

    try:
        slm = Santec(slm_number=slm_num)
        slm.open()
        # Wrap the fresh Santec in the overlay proxy so the GUI session handle
        # carries the send-pipeline overlay (defocus + base phase) at the
        # controller layer.  Driver slots stay empty on this path (the proxy
        # layer is the single source), so an overlay is applied exactly once.
        slm = OverlaidSantec(slm)

        st.session_state[prefix] = slm
        st.session_state[f"{prefix}_connected"] = True

        _sync_slm_session_state(slm_num, slm)

        _refresh_device_list()
        st.rerun()
    except Exception as e:  # policy: open failure is the expected not-connected path, so state is cleared and the operator sees the vendor's own diagnosis instead of a traceback
        st.session_state[prefix] = None
        st.session_state[f"{prefix}_connected"] = False
        # 只报原始错误码对操作员没有指导意义 (例如 -10003 只是「USB设备已打开」),
        # 故用驱动层的可执行提示替换: 说明占用/断链的原因并给出下一步动作。
        st.error(f"SLM {slm_num} 连接失败。\n\n{describe_santec_error(e)}")
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
    except Exception as e:  # policy: the finally block must still stop the toggle and clear the slots when close fails, or a controller that refuses to close leaves the panel driving a device it believes is gone
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
        # Drop the shared helper — it may hold large cached coordinate/Zernike
        # arrays and is only needed while the device is connected.
        st.session_state.pop(f"{prefix}_pattern_helper", None)

        st.rerun()


def set_wavelength(slm_num: int):
    """Set wavelength for the specified SLM.

    Rerun: none directly — the sidebar button already triggers a full rerun;
    refresh_phase_preview() updates the cache and the preview fragment
    re-renders on that same rerun. The rerun also re-initializes the
    context-derived control widgets (flat_gray / vortex_wavelength / ...)
    from the new wavelength and 2π max-grayscale.
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
    except Exception as e:  # policy: the value reaches the device, so a rejection must be reported with the device message or the panel advertises a wavelength the LCOS is not using
        st.error(f"设置波长失败: {e}")
        logger.exception(f"Failed to set wavelength for SLM {slm_num}: {e}")


def set_shift(slm_num: int):
    """Apply shift values from the UI to the SLM driver.

    Delegates to the driver-level :meth:`Santec.apply_shift`, which
    handles absolute-positioning re-display (undo old shift → apply new),
    memory-slot rotation, settle wait and config save — identical semantics
    for every caller (runner, script, other UI).

    Rerun: none directly — the triggering button already causes a full rerun;
    refresh_phase_preview() updates the cache and the preview fragment
    re-renders on that same rerun.
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
    except Exception as e:  # policy: the periodic toggle is stopped first so a failed shift cannot leave the display swapping between the old and the shifted phase
        st.error(f"应用平移失败: {e}")
        logger.exception(f"Failed to set shift for SLM {slm_num}: {e}")


def set_video_mode(slm_num: int, mode_label: str):
    """Set video mode for the specified SLM (memory mode only).

    Rerun: none directly — the triggering button already causes a full rerun;
    refresh_phase_preview() updates the cache and the preview fragment
    re-renders on that same rerun.

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
    except Exception as e:  # policy: surfaced and never retried automatically: an unresponsive controller can hang this process (documented DVI-mode failure) and only a power cycle recovers it
        st.error(f"设置模式失败: {e}")
        logger.exception(f"Failed to set video mode for SLM {slm_num}: {e}")


def toggle_correction(slm_num: int, enabled: bool) -> None:
    """启用或禁用SLM波前误差矫正叠加

    Rerun: none directly — the triggering button already causes a full rerun;
    refresh_phase_preview() updates the cache and the preview fragment
    re-renders on that same rerun.

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
    """Display the current status of the specified SLM.

    Uses a 2×2 grid layout: (status, wavelength) on the first row,
    (mode, next memory slot) on the second row. Status is shown as a
    monochrome badge (info/success/warning) for a consistent visual language.
    """
    prefix = f"slm{slm_num}"
    connected_key = f"{prefix}_connected"
    next_memory_key = f"{prefix}_next_memory"
    if st.session_state.get(connected_key) and st.session_state.get(prefix) is not None:
        slm = st.session_state[prefix]
        col1, col2 = st.columns(2)
        with col1:
            st.write("**状态**:")
            st.success("已连接")
            st.write("**波长**:")
            st.caption(f"{slm.wavelength} nm")
        with col2:
            st.write("**模式**:")
            st.caption("内存模式" if slm.video_mode == 0 else "DVI 模式")
            st.write("**下一个内存槽**:")
            st.caption(str(st.session_state[next_memory_key]))
    else:
        st.warning("未连接")


def render_slm_sidebar():
    """Render sidebar with device table and per-SLM settings."""
    st.header("SLM 设备")

    available = st.session_state.get("available_slms", [])
    if not available:
        st.caption("未发现可用设备")

    col_h1, col_h2, col_h3, col_h4 = st.columns([1, 2, 1, 1])
    with col_h1:
        st.markdown("**编号**")
    with col_h2:
        st.markdown("**序列号**")
    with col_h3:
        st.markdown("**状态**")
    with col_h4:
        st.markdown("**操作**")
    st.markdown("")

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
            st.markdown(f"**SLM {slm_num}**")
        with col2:
            st.markdown(device["serial"] or "-")
        with col3:
            if is_connected:
                st.success(status)
            elif in_use:
                st.warning(status)
            else:
                st.markdown(status)
        with col4:
            if is_connected:
                if st.button(
                    "断开", key=f"{prefix}_disconnect_table", type="secondary"
                ):
                    disconnect_slm(slm_num)
            elif in_use:
                st.markdown("已被占用")
            else:
                if st.button("连接", key=f"{prefix}_connect_table", type="primary"):
                    connect_slm(slm_num)

    st.markdown("")

    if st.button("刷新设备列表", key="refresh_devices_btn", type="secondary"):
        _refresh_device_list()

    st.markdown("")

    st.caption("相机工具")
    cam_status = st.session_state.get("camera_status")
    if st.button("刷新相机状态", key="refresh_camera_status_btn", type="secondary"):
        cam_status = _scan_camera_status()
        st.session_state["camera_status"] = cam_status

    if cam_status:
        for cam in cam_status:
            label = f"大恒相机 #{cam['index']}"
            if cam["serial"]:
                label += f" (SN: {cam['serial']})"
            status = cam["status"]
            if status == "可用":
                st.success(f"{label}: {status}")
            elif status == "使用中":
                st.warning(f"{label}: {status} — 连接被占用，请在其他实例断开后重试。")
            elif status == "错误":
                st.error(f"{label}: 探测失败 ({status})")
            else:
                st.caption(f"{label}: {status}")
    else:
        st.caption("未发现大恒相机 (点击刷新尝试)")
        st.caption("连接被占用时，此处会显示占用提示")

    st.markdown("")

    st.markdown("")

    # 「CCD参数设置」区块：相机类型/ID、像元间距（+ 探测 + 自动探测告警）、
    # 会话曝光（+ 探测）、读出窗口、去零级载波，以及每台已连接 SLM 的面板几何、
    # 照明半径（+ 探测）与标定按钮。
    # 像元间距的 widget key 仍是 camera_pixel_um（不新建同 key widget），
    # 整个区块只在这里实例化一次 —— 主区的整形参数区不再重复它。
    # 标定入口也只有这一处：光路摘要卡 + 「标定 SLM N 光路」按钮同在一个折叠块里
    # （2026-10-24 删掉原先独立的「一键光路标定」块 —— 它调的是同一个
    # run_session_calibration，纯重复入口）。
    render_bench_settings()

    for slm_num in (1, 2):
        prefix = f"slm{slm_num}"
        if (
            st.session_state.get(f"{prefix}_connected")
            and st.session_state.get(prefix) is not None
        ):
            _render_slm_settings(slm_num)


def _slm_resolution_label(width: int, height: int) -> str:
    """The resolution line of the device-info block."""
    return f"分辨率: {width}×{height}"


def _slm_pitch_label(pixel_pitch_um: float, size_um: float | None) -> str:
    """The pixel-pitch line, with the clear aperture appended when reported.

    7.8 = 有效口径 / 8.0 = 间距, 两个不同的物理量, 并排显示避免混淆
    """
    label = f"像素间距: {pixel_pitch_um} μm"
    if size_um is not None:
        label += f" · 有效口径: {size_um} μm"
    return label


def _slm_serial_label(slm_obj: Santec) -> str:
    """The serial-number line, showing a dash when the driver reports none."""
    sn_display = slm_obj._serial_number
    return f"SLM序列号: {sn_display if sn_display else '-'}"


def _render_slm_wavelength_mismatch(slm_num: int, prefix: str, slm_obj: Santec) -> None:
    """Warn when the sidebar wavelength differs from the device's own value."""
    if slm_obj is None or slm_obj.wavelength is None:
        return
    widget_wl = st.session_state.get(f"{prefix}_wavelength")
    if widget_wl is None or widget_wl == int(slm_obj.wavelength):
        return
    col_wl1, col_wl2 = st.columns([4, 1])
    with col_wl1:
        st.warning(f"波长: 设定值 {widget_wl} nm ≠ 设备值 {int(slm_obj.wavelength)} nm")
    with col_wl2:
        if st.button("同步", key=f"{prefix}_sync_wl_btn", type="primary"):
            st.session_state[f"{prefix}_wavelength"] = int(slm_obj.wavelength)
            st.rerun()


def _render_slm_config_status(slm_num: int, prefix: str, slm_obj: Santec) -> None:
    """Show the matched configuration file and offer to reload it."""
    config_info = st.session_state.get(f"{prefix}_wavelength_mismatch")
    if not config_info or slm_obj is None:
        return
    st.caption(f"序列号: {config_info['serial']}")
    loaded_cfg = st.session_state.get(f"{prefix}_loaded_config", {})
    if loaded_cfg:
        st.caption("已加载配置文件:")
        st.json(loaded_cfg)
    else:
        st.caption("未找到匹配的配置文件（使用设备默认值）")
    if st.button("刷新配置", key=f"{prefix}_refresh_config_btn", type="secondary"):
        fresh = slm_obj.load_config() if slm_obj._serial_number else {}
        st.session_state[f"{prefix}_loaded_config"] = fresh
        st.rerun()


def _render_slm_device_section(slm_num: int, prefix: str, slm_obj: Santec) -> None:
    """Device info, wavelength, video mode, pattern shift and config status."""
    st.markdown("**设备信息**")
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.write(
            _slm_resolution_label(
                st.session_state[f"{prefix}_width"],
                st.session_state[f"{prefix}_height"],
            )
        )
        pitch_um = st.session_state[f"{prefix}_pixel_pitch_um"]
        st.write(_slm_pitch_label(pitch_um, getattr(slm_obj, "Pixel_Size_um", None)))
    with col_info2:
        st.write(f"Bit数: {st.session_state[f'{prefix}_bits']}")
        st.write(_slm_serial_label(slm_obj))
        st.write(f"SLM编号: {slm_obj.slm_number}")

    st.markdown("")

    st.number_input(
        "波长 (nm)",
        min_value=450,
        max_value=1600,
        step=1,
        key=f"{prefix}_wavelength",
    )
    _render_slm_wavelength_mismatch(slm_num, prefix, slm_obj)
    if st.button("设置波长", key=f"{prefix}_set_wl_btn", type="secondary"):
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
    if st.button("设置模式", key=f"{prefix}_set_mode_btn", type="secondary"):
        set_video_mode(slm_num, st.session_state[f"{prefix}_video_mode"])

    st.markdown("**Pattern Shift (像素)**")
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
    if st.button("应用平移", key=f"{prefix}_apply_shift_btn", type="secondary"):
        set_shift(slm_num)

    _render_slm_config_status(slm_num, prefix, slm_obj)


def _write_upload_to_temp_csv(uploaded_correction: Any) -> str:
    """Persist an uploaded file as a temporary CSV and return its path."""
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_correction.getbuffer())
        tmp_path = tmp.name
    return tmp_path


def _load_correction_into_device(
    slm: Any, slm_num: int, uploaded_correction: Any
) -> None:
    """Hand the uploaded CSV to the device and report the outcome."""
    tmp_path = _write_upload_to_temp_csv(uploaded_correction)
    ok = slm.load_correction_from_csv(tmp_path)
    if ok:
        st.success(f"矫正已应用: {uploaded_correction.name}")
        refresh_phase_preview(slm_num)
    else:
        st.warning("矫正文件加载失败")


def _apply_uploaded_correction(slm_num: int, uploaded_correction: Any) -> None:
    """Load an uploaded correction CSV into the device; failures are reported."""
    try:
        slm = st.session_state.get(f"slm{slm_num}")
        if slm is None:
            st.warning("SLM未连接，无法应用矫正")
        elif uploaded_correction is None:
            st.warning("请先选择矫正 CSV 文件")
        else:
            _load_correction_into_device(slm, slm_num, uploaded_correction)
    # policy: the correction is written to the device; on failure the previously displayed phase stays, which is exactly what the message must say
    except Exception as e:
        st.error(f"应用矫正失败: {e}")
        logger.exception(f"Failed to apply correction for SLM {slm_num}: {e}")


def _clear_correction(slm_num: int) -> None:
    """Drop the loaded correction file; failures are reported."""
    try:
        slm = st.session_state.get(f"slm{slm_num}")
        if slm is None:
            st.warning("SLM未连接")
        else:
            slm.load_correction_from_csv(None)
            st.success("矫正已清除")
            refresh_phase_preview(slm_num)
    # policy: a failed clear leaves the correction in place, so the message must not imply the phase was reset
    except Exception as e:
        st.error(f"清除矫正失败: {e}")


def _render_slm_correction_section(slm_num: int, prefix: str, slm_obj: Santec) -> None:
    """Wavefront-correction toggle, CSV upload, and apply / clear."""
    st.caption("叠加矫正")
    use_correction = st.checkbox(
        "叠加矫正CSV",
        # Seed from the mirror key only on the first render; afterwards the
        # widget key owns the value (Streamlit warns on every rerun when
        # both are supplied, and ignores the value anyway).
        value=widget_default(
            f"{prefix}_use_correction_cb",
            st.session_state.get(f"{prefix}_use_correction", True),
        ),
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

    st.markdown("")

    st.caption("加载校正")
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
        if st.button("应用矫正", key=f"{prefix}_apply_corr_btn", type="secondary"):
            _apply_uploaded_correction(slm_num, uploaded_correction)
    with col_corr2:
        if st.button("清除矫正", key=f"{prefix}_clear_corr_btn", type="secondary"):
            _clear_correction(slm_num)


def _save_slm_config(slm_num: int, prefix: str) -> None:
    """Write the sidebar shift back to the device and save its configuration."""
    slm = st.session_state.get(prefix)
    if slm is None:
        st.info("SLM未连接，配置将在连接后写入设备")
        return
    try:
        ui_shift_x = st.session_state.get(f"{prefix}_shift_x", 0)
        ui_shift_y = st.session_state.get(f"{prefix}_shift_y", 0)
        if (ui_shift_x, ui_shift_y) != (slm.shift_x, slm.shift_y):
            slm.set_shift(shift_x=ui_shift_x, shift_y=ui_shift_y)
        slm.save_config()
        st.success(
            f"SLM {slm_num} 配置已保存（序列号: {slm._serial_number or '未知'}）"
        )
    # policy: persisting to the device can be refused (firmware or read-only state); the operator believes the configuration is now stored, so the refusal is reported
    except Exception as e:
        st.error(f"保存配置到设备失败: {e}")


def _save_slm_phase_csv(slm_num: int) -> None:
    """Write the currently displayed phase to a CSV under the app config dir."""
    try:
        slm = st.session_state.get(f"slm{slm_num}")
        if slm is None:
            st.warning("SLM未连接，无法保存相位")
            return
        phase, _ = slm.get_displayed_phase()
        if phase is None:
            st.warning("无法获取当前显示的相位数据")
            return
        save_path = Path.home() / ".config" / "ao_shaping" / f"slm{slm_num}_phase.csv"
        max_gray = getattr(slm, "_max_gray", None) or Santec.MAX_GRAYSCALE_VALUE
        phase_rad = phase.astype(np.float64) / float(max_gray) * 2 * np.pi
        Santec.save_phase_to_csv(phase_rad, save_path)
        st.success(f"SLM {slm_num} 相位已保存到 {save_path}")
    # policy: the CSV write can fail on permissions or disk while the phase stays displayed either way, so the failure is reported without touching the device
    except Exception as e:
        st.error(f"保存相位CSV失败: {e}")
        logger.exception(f"Failed to save phase CSV for SLM {slm_num}: {e}")


def _render_slm_save_section(slm_num: int, prefix: str) -> None:
    """Save the device configuration and the currently displayed phase."""
    st.markdown("")

    st.caption("保存")
    if st.button("保存配置", key=f"{prefix}_save_config_btn", type="secondary"):
        _save_slm_config(slm_num, prefix)

    if st.button("保存当前相位到CSV", key=f"{prefix}_save_csv_btn", type="secondary"):
        _save_slm_phase_csv(slm_num)


def _render_slm_settings(slm_num: int):
    """Render settings controls for a single connected SLM.

    Split into collapsible sections to keep the sidebar scannable:
    device/wavelength-mode-shift, grayscale config, wavefront correction,
    the two long-lived overlays, and save. Each section lives in its own
    function so no single block nests deeply (issue #167).
    """
    prefix = f"slm{slm_num}"
    slm_obj = st.session_state.get(prefix)
    if slm_obj is None:
        return

    # --- Section 1: Device info + wavelength/mode/shift (expanded by default) ---
    with st.expander(f"SLM {slm_num} 设置", expanded=True):
        _render_slm_device_section(slm_num, prefix, slm_obj)

    # --- Section 3: Wavefront correction (overlay layer 1 of the send pipeline) ---
    with st.expander("波前矫正", expanded=False):
        _render_slm_correction_section(slm_num, prefix, slm_obj)

    # --- Section 3b: 相位叠加（底相位 / 载波 / 离焦，可多选） ---
    with st.expander("相位叠加（底相位 / 载波 / 离焦）", expanded=False):
        render_phase_overlay_block(slm_num)

    # --- Section 3c: Save (persist) ---
    with st.expander("保存", expanded=False):
        _render_slm_save_section(slm_num, prefix)


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
        # policy: in set_grayscale mode no slot is displayed and the SDK answers with an error code; that is a normal state, not a fault, so None is the correct value
        except Exception:
            current_slot = None
        if slot == current_slot:
            alt = [s for s in range(2, 126) if s != current_slot]
            slot = int(np.random.choice(alt))

        try:
            slm.display_data(target_phase, memory_number=slot)
        # policy: this runs in the background toggle thread: one failed write is skipped and the next tick retries, because raising here would silence the periodic swap without telling anyone
        except Exception as e:
            logger.warning(f"周期切换失败: {e}")

        time.sleep(max(0.01, 1.0 / (2.0 * freq)))


def _toggle_phase_capture_label(slot: str, shape: tuple[int, ...]) -> str:
    """The confirmation line shown after capturing the displayed phase."""
    return f"相位 {slot} 已设置 ({shape})"


def _toggle_phase_zero_label(slot: str, height: int, width: int) -> str:
    """The confirmation line shown after filling a toggle slot with zeros."""
    return f"相位 {slot} 已设为全0相位 ({height}×{width})"


def _capture_toggle_phase(prefix: str, slot: str) -> None:
    """Store the currently displayed phase as one of the two toggle slots."""
    slm = st.session_state.get(prefix)
    if slm is not None and getattr(slm, "is_open", False):
        phase, _ = slm.get_displayed_phase()
        if phase is not None:
            st.session_state[f"{prefix}_toggle_phase_{slot.lower()}"] = phase
            st.success(_toggle_phase_capture_label(slot, phase.shape))
        else:
            st.warning("无法获取当前显示相位")
    else:
        st.warning("SLM 未连接")


def _fill_toggle_phase_with_zeros(prefix: str, slot: str) -> None:
    """Fill one of the two toggle slots with an all-zero phase."""
    slm = st.session_state.get(prefix)
    if slm is not None and getattr(slm, "is_open", False):
        h = slm.Panel_Res[1]
        w = slm.Panel_Res[0]
        st.session_state[f"{prefix}_toggle_phase_{slot.lower()}"] = np.zeros(
            (h, w), dtype=np.uint16
        )
        st.success(_toggle_phase_zero_label(slot, h, w))
    else:
        st.warning("SLM 未连接")


def _start_phase_toggle(slm_num: int, prefix: str) -> None:
    """Start the background thread that alternates between the two slots."""
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


def _stop_phase_toggle(prefix: str) -> None:
    """Stop the background toggle thread and clear its session state."""
    stop_event = st.session_state.get(f"{prefix}_toggle_stop_event")
    if stop_event is not None:
        stop_event.set()
    st.session_state[f"{prefix}_toggle_active"] = False
    st.session_state[f"{prefix}_toggle_thread"] = None
    st.session_state[f"{prefix}_toggle_stop_event"] = None
    st.session_state[f"{prefix}_toggle_freq_ref"] = None
    st.session_state[f"{prefix}_toggle_slm_container"] = None
    st.success("周期切换已停止")


def _render_phase_generate_section(slm_num: int, prefix: str) -> None:
    """Generate a phase from the pattern generator, or load one from a CSV."""
    pattern_type, params = render_pattern_controls(slm_num)

    if st.button(
        "从模式生成器生成相位", key=f"{prefix}_gen_pattern_btn", type="primary"
    ):
        try:
            _stop_toggle(slm_num)
            if st.session_state.get(f"{prefix}_toggle_active", False):
                st.info("已停止周期切换")

            slm = st.session_state[prefix]

            phase = generate_phase_gray(slm, pattern_type, params)

            # generate_phase_gray() already applies the configured shift
            # internally via slm.create_phase_from_array() → Santec.shift_phase().
            # Do NOT apply a second shift here — that would double-displace
            # the phase on the LCOS panel.

            # Write to next memory slot and immediately display
            slm.display_data(phase)
            refresh_phase_preview(slm_num)
            st.success(
                f"相位已写入内存槽 {slm.get_displayed_memory_number()} 并显示（验证通过）"
            )

        # policy: generation or display failure is reported with the operator's parameters still in the widgets, so the pattern can be corrected and re-sent
        except Exception as e:
            st.error(f"生成或显示相位失败: {e}")
            logger.exception(f"Failed to generate/display phase for SLM {slm_num}: {e}")

    # Option to load from CSV
    uploaded_file = st.file_uploader(
        "上传CSV相位文件", type=["csv"], key=f"{prefix}_csv_file"
    )
    if uploaded_file is not None and st.button(
        "从CSV加载相位", key=f"{prefix}_load_csv_btn", type="secondary"
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
            slm.load_gray_from_csv(temp_path)

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
        # policy: the uploaded file is arbitrary input; a malformed CSV must surface as an error naming this step and leave the displayed phase untouched
        except Exception as e:
            st.error(f"加载CSV相位失败: {e}")
            logger.exception(f"Failed to load CSV phase for SLM {slm_num}: {e}")


def _render_phase_toggle_section(slm_num: int, prefix: str) -> None:
    """The two toggle slots, the toggle frequency, and start / stop."""
    col_ph_a, col_ph_b = st.columns(2)
    with col_ph_a:
        if st.button("设为相位 A", key=f"{prefix}_set_phase_a", type="primary"):
            _capture_toggle_phase(prefix, "A")
        if st.button(
            "设为全0相位 A", key=f"{prefix}_set_zero_phase_a", type="secondary"
        ):
            _fill_toggle_phase_with_zeros(prefix, "A")
    with col_ph_b:
        if st.button("设为相位 B", key=f"{prefix}_set_phase_b", type="primary"):
            _capture_toggle_phase(prefix, "B")
        if st.button(
            "设为全0相位 B", key=f"{prefix}_set_zero_phase_b", type="secondary"
        ):
            _fill_toggle_phase_with_zeros(prefix, "B")
    st.number_input(
        "切换频率 (Hz)",
        min_value=0.1,
        max_value=100.0,
        value=1.0,
        step=0.1,
        key=f"{prefix}_toggle_frequency",
    )

    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("开始周期切换", key=f"{prefix}_start_toggle", type="secondary"):
            _start_phase_toggle(slm_num, prefix)
    with col_stop:
        if st.button("停止周期切换", key=f"{prefix}_stop_toggle", type="secondary"):
            _stop_phase_toggle(prefix)


def render_phase_control(slm_num: int):
    """Phase control UI for SLM (parameterized).

    操作路线图 (docs/slm/slm_gui_operation_map.md) 主区两段:
    一次下发 (图案生成 / CSV 加载) + 持续任务 (周期切换)。
    底相位属长期配置 (T4), 已移至侧栏「底相位 (发送叠加)」区块。
    """
    prefix = f"slm{slm_num}"
    st.subheader("一次下发")
    _render_phase_generate_section(slm_num, prefix)

    st.markdown("")
    st.subheader("持续任务：周期切换")
    st.caption(
        "后台线程在相位 A 与相位 B 之间持续来回切换（运行期间独占面板，需「停止」释放）"
    )
    _render_phase_toggle_section(slm_num, prefix)


def _apply_slm_overlays(slm_num: int) -> None:
    """把该 SLM 的两个发送管线叠加层（底相位 / 离焦）按 session 状态推给句柄。

    单一真源：连接、重连、切换开关、保存底相位后统一走这里，保证叠加层始终
    等于 UI 当前配置。``_sync_slm_session_state``（首连 + 整形后重取）和
    底相位/离焦区块的开关都调它。无连接时直接返回。

    叠加层由 :func:`build_send_overlay` 从 session 现算（与闭环整形 GUI 入口
    共用同一读取点，保证 GUI 会话句柄与优化器自开句柄拿到同一份叠加），再经
    :func:`apply_overlays_to_handle` **单一分发点**写到句柄上，按句柄类型决定
    写入哪个真源（代理句柄 → 代理层并清零其驱动槽；裸驱动句柄 → 驱动槽），
    从而维护「一个句柄只叠加一次」的 invariant，代理 / 裸驱动两条路径互不
    双重叠加：

    * 底相位（灰度）：仅在「叠加底相位」勾选且已保存时应用，否则置 None。
    * 离焦（弧度）：受离焦开关门控（见
      :func:`ao_shaping.gui.slm.bench_settings.build_defocus_overlay`）。
    """
    prefix = f"slm{slm_num}"
    slm = st.session_state.get(prefix)
    if slm is None or not getattr(slm, "is_open", False):
        return
    overlay = build_send_overlay(prefix, slm)
    apply_overlays_to_handle(slm, overlay)


def _render_defocus_overlay_block(slm_num: int):
    """离焦叠加 (T4 发送管线叠加层)，渲染在侧栏「离焦叠加 (发送叠加)」区。

    离焦补偿 = 菲涅尔项（焦距 mm，粗调，默认取 D2「探测焦面」推荐值，操作员可覆盖）
    + Zernike (2,0)（细调，手动）。做成**开关式全局叠加**：主开关一开，驱动
    ``set_phase_overlay`` 就承载该层，作用于其后所有发送（优化器 / 一键下发 / CSV /
    周期切换都走 ``create_phase_from_array``），与底相位同层、同机制。

    控件顺序是硬约束：「用推荐值」按钮必须先于焦距 ``number_input`` 渲染，
    这样点击时写 ``defocus_focal``（widget key）在本 run 内合法（该 key 尚未被
    控件锁定）。
    """
    from ao_shaping.gui.slm import bench_settings

    prefix = f"slm{slm_num}"
    slm = st.session_state.get(prefix)
    connected = slm is not None and getattr(slm, "is_open", False)
    busy = bench_settings.bench_busy()

    auto_focal = bench_settings.bench_defocus_focal_auto(prefix)
    has_probe = auto_focal != 0.0

    enabled = st.checkbox(
        "启用离焦叠加",
        value=widget_default(f"{prefix}_defocus_enabled", False),
        key=f"{prefix}_defocus_enabled",
        help="勾选后，离焦相位叠加到所有发送（发送管线 T4 叠加层，与底相位同机制）",
        disabled=busy,
    )

    if enabled:
        # 推荐值（来自探测）+ 手动覆盖
        auto_text = f"{auto_focal:.1f} mm" if has_probe else "未探测（填 0 = 仅细调）"
        st.caption(f"探测推荐焦距：**{auto_text}**（D2 探测焦面；未探测可手填）")

        if st.button(
            "用推荐值",
            key=f"{prefix}_defocus_use_reco",
            help="把下方焦距重置为探测推荐值",
            disabled=busy or not has_probe,
        ):
            if has_probe:
                st.session_state[f"{prefix}_defocus_focal"] = auto_focal
                _apply_slm_overlays(slm_num)
                st.success(f"已用推荐焦距 {auto_focal:.1f} mm")

        st.number_input(
            "焦距 (mm)",
            value=widget_default(f"{prefix}_defocus_focal", auto_focal),
            key=f"{prefix}_defocus_focal",
            step=0.1,
            help="离焦补偿焦距 (mm)，菲涅尔项；0 = 不加菲涅尔项（仅细调）",
            disabled=busy,
        )
        st.number_input(
            "细调 (Zernike (2,0))",
            value=widget_default(f"{prefix}_defocus_amount", 0.0),
            key=f"{prefix}_defocus_amount",
            step=0.5,
            help="Zernike (2,0) 系数，离焦细调（rad 量纲，面板口径）",
            disabled=busy,
        )

        # 只读几何（半径/中心/波长），来自 bench 取值链 + 设备
        centre = bench_settings.bench_pupil_center(prefix)
        span = float(
            st.session_state.get(f"{prefix}_bench_panel_span_px", 900.0) or 900.0
        )
        radius = max(1, int(round(span / 2.0)))
        wl = getattr(slm, "wavelength", None) if connected else None
        wl_text = f"{wl:.0f} nm" if wl else "1064 nm (默认)"
        st.caption(f"离焦孔径：中心 {centre}，半径 {radius} px，波长 {wl_text}")

    # 幂等同步：把当前离焦开关/参数推给驱动（纯 numpy，无硬件 I/O），
    # 使叠加层始终等于 UI 配置。connected 时才推，避免对空句柄操作。
    if connected and not busy:
        _apply_slm_overlays(slm_num)

    # 只读状态行（信息可见性：不受开关嵌套，永远渲染）。启用时给出当前生效的
    # 焦距/细调，操作者无需回滚到上方控件即可核对正在叠加的数值。
    if not connected:
        return
    applied = bool(st.session_state.get(f"{prefix}_defocus_enabled", False))
    if applied:
        eff_focal = bench_settings.bench_defocus_focal(prefix)
        eff_fine = float(st.session_state.get(f"{prefix}_defocus_amount", 0.0) or 0.0)
        st.caption(
            f"离焦叠加：**已启用**（焦距 {eff_focal:.1f} mm，"
            f"细调 {eff_fine:+.1f}；作用于所有发送）"
        )
    else:
        st.caption("离焦叠加：**已关闭**（未对发送叠加相位）")


def render_phase_overlay_block(slm_num: int) -> None:
    """相位叠加（底相位 / 载波 / 离焦）：三层可分别勾选，勾中的全部相加。

    为什么合到一处
    --------------
    2026-10-23 之前三层散在三个地方、语义还不同：底相位在驱动的灰度槽，载波只
    作为 ``blaze_period_px`` 喂给整形优化器（于是侧栏「一次下发」的
    PatternHelper 相位**从来没带上它** —— 这就是「载波没有叠加倾斜相位」），
    离焦是另一块独立按钮。分开放的后果就是「某一层只对某条发送路径生效」。

    现在三层都经 :func:`build_send_overlay` 汇成一份 ``PhaseOverlay``，由
    :func:`_apply_slm_overlays` 推到句柄。任何走 ``create_phase_from_array`` 的
    发送（一键下发 / CSV / 周期切换 / 整形结果 / 优化器自开句柄）都自动带上。

    两个域不能混
    ------------
    * 底相位 → 灰度槽（矫正/LUT **之后**）：它是「设备上已显示过的那张图」，
      按灰度原样叠加才与当初显示一致。
    * 载波 / 离焦 → 弧度槽（矫正/LUT **之前**）：它们是光路里真实的相位，
      应当被 LUT 重映射。
    """
    prefix = f"slm{slm_num}"
    st.caption("将当前显示相位保存为底相位，后续发送相位时可自动叠加")

    col_base1, col_base2 = st.columns(2)
    with col_base1:
        if st.button(
            "保存当前相位为底相位", key=f"{prefix}_save_base_phase", type="secondary"
        ):
            slm = st.session_state.get(prefix)
            if slm is not None and getattr(slm, "is_open", False):
                phase, source = slm.get_displayed_phase()
                if phase is not None:
                    st.session_state[f"{prefix}_base_phase"] = phase.copy()
                    _apply_slm_overlays(slm_num)
                    st.success(f"底相位已保存 ({phase.shape}, 来源: {source})")
                else:
                    st.warning("无法获取当前显示相位，请先显示一个相位图案")
            else:
                st.warning("SLM 未连接")
    with col_base2:
        overlay = st.checkbox(
            "发送时自动叠加底相位",
            value=widget_default(
                f"{prefix}_overlay_base_cb",
                st.session_state.get(f"{prefix}_overlay_base", False),
            ),
            key=f"{prefix}_overlay_base_cb",
            help="勾选后，写入SLM的相位会自动叠加已保存的底相位",
        )
        if overlay != st.session_state.get(f"{prefix}_overlay_base", False):
            st.session_state[f"{prefix}_overlay_base"] = overlay
            _apply_slm_overlays(slm_num)
            st.success(f"底相位叠加已{'启用' if overlay else '禁用'}")

    base_phase = st.session_state.get(f"{prefix}_base_phase")
    if base_phase is not None:
        st.caption(f"当前底相位: 已保存，尺寸 {base_phase.shape}")
        if st.button("清除底相位", key=f"{prefix}_clear_base_phase", type="secondary"):
            st.session_state[f"{prefix}_base_phase"] = None
            st.session_state[f"{prefix}_overlay_base"] = False
            slm = st.session_state.get(prefix)
            if slm is not None:
                _apply_slm_overlays(slm_num)
            st.success("底相位已清除")
    else:
        st.caption("当前底相位: 未设置")

    # --- 载波（倾斜）-----------------------------------------------------
    st.divider()
    from ao_shaping.gui.slm import bench_settings as _bs

    busy = _bs.bench_busy()
    carrier_on = st.checkbox(
        "叠加载波（倾斜）",
        value=widget_default(f"{prefix}_blaze_enabled", False),
        key=f"{prefix}_blaze_enabled",
        disabled=busy,
        help="把 0 级移出读出窗的离轴载波。作为发送叠加层作用于所有发送；"
        "整形优化器不再另行施加，避免倾斜被加两遍。",
    )
    _bs.render_blaze_controls(prefix, busy or not carrier_on)
    if carrier_on != bool(st.session_state.get(f"{prefix}_blaze_enabled", False)):
        # checkbox 已经用 key 写回 session，这一支只在语义上兜底（键被外部改过）。
        st.session_state[f"{prefix}_blaze_enabled"] = carrier_on
    if not busy:
        _apply_slm_overlays(slm_num)

    # --- 离焦 ------------------------------------------------------------
    st.divider()
    st.caption("离焦（菲涅尔粗调 + Zernike (2,0) 细调）")
    _render_defocus_overlay_block(slm_num)


if __name__ == "__main__":
    main()
