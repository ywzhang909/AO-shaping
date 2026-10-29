"""CCD参数设置：侧栏区块 + 「自动探测但允许人工修改」的 session 契约。

为什么单独一个模块
------------------
**一次实验 = 一条光路**。相机型号 / ID、读出窗口、曝光、去零级载波，以及每台
SLM 的入射位置与几何，都在一次实验内固定。把它们留在主区的整形参数里，
就是「每次都要重填一屏数字，换一条光路就直接失效」的根因 —— 尤其
``slm-model-in-loop`` 的 CLI 每次会话本来会自动做
**全板随机相位盘扫描定位入射中心 → sweep 几何标定**，GUI 却绕过这条流程，
只能让操作员手填 ``pupil_center="960,690"``、``panel_span_px=900``。

本模块只管两件事：

1. :func:`render_bench_settings` —— 侧栏「CCD参数设置」区块（实验级 + 每 SLM 级）；
2. 三态 session 契约 —— 每个可探测字段的 ``unset`` / ``detected`` / ``manual``。

两层光路的分工
--------------
**实验级**共享，**每 SLM 级**分片：双 SLM 串联时第二台的入射面是第一台的出射面，
入射中心与几何**必须**分开标定；共用一套值既错又不报错，所以这里按前缀
``""``（实验级）与 ``slm{N}``（每 SLM）分开存。

三态
----
* ``unset``    —— 从未标定，没有任何探测值；
* ``detected`` —— 有探测值，caption 前缀 ``●``；
* ``manual``   —— 人工改过，前缀 ``✎``，同时显示当前值、探测值与差值。

``manual`` 永远优先，但探测值永远可见。**重新标定不覆盖手动值**，只在 caption
里提示「探测到新几何，当前仍使用手动值」。

写值为什么必须走 pending-flush
------------------------------
Streamlit 里 widget 一旦以 ``key`` 实例化，该 key 在本次脚本运行内被锁，之后写
``st.session_state[widget_key]`` 会抛 ``StreamlitAPIException``。标定按钮是在
侧栏 widget **渲染之后**被点击的（同一脚本运行），所以回填不能直接写 widget key。
两段式：

1. 点击 → 写 ``{prefix}_bench_pending``（非 widget key，不受锁）+ ``st.rerun()``；
2. :func:`render_bench_settings` 的第一行 :func:`flush_pending` 把 pending 写进
   对应 widget key（此刻尚未实例化，允许写），清空 pending，再实例化控件。

``_safe_set_state()``（吞异常）**不**用于此处 —— 吞掉异常会让回填静默失败。
它保留给「运行后重连」那条晚于侧栏渲染的路径。

术语见 ``docs/terminology_glossary.md``（本模块用「实验系统」「瞳面」，
不用「台架」「瞳孔」）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import streamlit as st
from loguru import logger

from ao_shaping.drivers.ccd.common import WindowSize
from ao_shaping.drivers.ccd.pixel_size import (
    KNOWN_MODEL_PIXEL_SIZE_UM,
    lookup_model_pixel_size_um,
    probe_daheng_models,
)
from ao_shaping.utils.cli.exposure import DEFAULT_EXPOSURE_TOKEN
from ao_shaping.drivers.slm.santec.slm200_constants import (
    PANEL_RES,
    PITCH_UM,
    PIXEL_SIZE_UM,
)
from ao_shaping.gui.utils.phase_overlay import PhaseOverlay
from ao_shaping.utils.image.exposure_control import (
    configured_daheng_exposure_limit,
    working_exposure_ms,
)
from ao_shaping.tools.slm.panel.focus_probe import FOCUS_PROBE_DEBUG_DIR

__all__ = [
    "BLAZE_DIRECTIONS",
    "BLAZE_PERIOD_OPTIONS",
    "CAMERA_PIXEL_KEY",
    "CAM_TYPE_OPTIONS",
    "DEFAULT_CAMERA_PIXEL_UM",
    "GEOMETRY_SOURCE_OPTIONS",
    "KNOWN_CAMERA_PIXEL_UM",
    "MIL_TARGET_SIDE_MIN_PX",
    "MIN_TARGET_SIDE_PX",
    "PUPIL_CENTER_FALLBACK",
    "BenchField",
    "WindowBounds",
    "auto_check_camera_pixel_um",
    "bench_busy",
    "bench_fields",
    "bench_fresnel_focal_mm",
    "bench_geometry_source",
    "bench_pupil_center",
    "bench_summary_lines",
    "bench_window_side",
    "measure_illuminated_radius",
    "build_calibration_params",
    "detected_state",
    "effective_state",
    "field_diff",
    "field_hint",
    "field_state",
    "field_states",
    "flush_pending",
    "geometry_region_mismatch",
    "manual_fields",
    "manual_state",
    "merge_bench_values",
    "pending_state",
    "probe_camera_window",
    "publish_session_calibration",
    "BEAM_RADIUS_DEBUG_DIR",
    "EXPOSURE_PROBE_DEBUG_DIR",
    "ExposureProbeFailure",
    "probe_session_exposure_ms",
    "queue_pending",
    "record_detection",
    "render_bench_settings",
    "reset_manual",
    "reusable_geometry_file",
    "run_session_calibration",
    "widget_default",
    "set_bench_busy",
    "set_effective_state",
    "sync_manual_states",
    "uncalibrated_note",
    "validate_bench_inputs",
    "window_bounds",
]

# ---------------------------------------------------------------------------
# 常量（全部来自设备 / 手册，不写死本实验系统的实测值）
# ---------------------------------------------------------------------------

#: CCD 后端，顺序即选择框顺序。``sim`` = 2f-Fourier 数字孪生（无硬件自检）。
CAM_TYPE_OPTIONS: tuple[str, ...] = ("daheng", "miicam", "sim")

#: 去零级载波周期：-1 关闭 / 0 实测自动 / 其余为实测固定周期（px）。
BLAZE_PERIOD_OPTIONS: tuple[int, ...] = (-1, 0, 128, 96, 64)

#: 载波周期的显示口径（与主区历史控件一致，值本身就是选项）。
BLAZE_PERIOD_LABELS: dict[int, str] = {
    -1: "关闭",
    0: "实测自动选择",
    128: "实测 128 px 周期",
    96: "实测 96 px 周期",
    64: "实测 64 px 周期",
}

BLAZE_DIRECTIONS: tuple[str, ...] = ("vertical", "horizontal")

#: 几何来源 —— 文档的是**标定按钮**的作用范围，不是运行路径（运行不测量）。
#: ``auto`` 点按钮时入射中心与 sweep 几何全测；``reuse`` 直接用已有几何文件；
#: ``keep`` 只重测入射中心，沿用上次几何。
GEOMETRY_SOURCE_OPTIONS: tuple[str, ...] = ("auto", "reuse", "keep")
GEOMETRY_SOURCE_LABELS: dict[str, str] = {
    "auto": "自动标定（点按钮时全测）",
    "reuse": "复用已标定几何",
    "keep": "沿用上次几何（只重测中心）",
}

#: 相机像元间距的已知型号表 —— 由相机驱动层给出 (单一来源:
#: ``ao_shaping.drivers.ccd.pixel_size.KNOWN_MODEL_PIXEL_SIZE_UM``), 这里只是
#: 保持侧栏的公开名字不变。**不要**在本模块再添一份型号 → 像元间距的映射。
KNOWN_CAMERA_PIXEL_UM: dict[str, float] = dict(KNOWN_MODEL_PIXEL_SIZE_UM)
#: 侧栏控件在「探不到设备」时的显示默认值 (不是设备真值; 探到就覆盖)。
DEFAULT_CAMERA_PIXEL_UM = KNOWN_MODEL_PIXEL_SIZE_UM["MER2-507"]

#: 目标方边长下界（相机像素）。低于 8 px 的目标框在读出噪声里没有意义。
MIN_TARGET_SIDE_PX = 8
#: 同名再导出，供 shaping_controls 绑定（该常量是「侧栏 → 主区」的契约）。
MIL_TARGET_SIDE_MIN_PX = MIN_TARGET_SIDE_PX

#: 模型覆盖宽度（面板 px）的步长。
PANEL_SPAN_STEP_PX = 10.0

#: 相机不可达时的**保守兜底**窗口上限 / 步长（大恒 MER2-507 传感器短边 1944）。
#: 拿得到相机时一律用真值，这里只保证控件不越界，并在 caption 里说明来源。
FALLBACK_WINDOW_MAX_SIDE = 1944
FALLBACK_WINDOW_STEP = 10

#: 读出窗口边长的**默认选择**（px）。这不是设备值，而是本仓库历史默认的读出窗口
#: （主区 ``MIL_WIDGET_DEFAULTS["cam_size"]`` 同为 250）。控件用它做默认值，上界
#: 仍由设备特征网格决定。**不要**拿 ``WindowBounds.min_side`` 当默认值 —— 那是网格
#: 步长（device 时 = ``inc``，相机不可达时 = 10），会把「当前窗口边长」塌成个位数，
#: 而主区 ``target_side`` 的上界就是它（2026-10-16 启动崩溃）。
DEFAULT_WINDOW_SIDE_PX = 250

# ---------------------------------------------------------------------------
# 差异提示阈值（只在超过阈值时提示，避免 1e-9 级噪声刷屏）
# ---------------------------------------------------------------------------

#: 入射中心：任一坐标差 ≥ 1 px。
CENTER_TOLERANCE_PX = 1.0
#: 模型覆盖宽度：差 ≥ 1 个 step。
PANEL_SPAN_TOLERANCE_PX = PANEL_SPAN_STEP_PX
#: 相机窗口边长：差 ≥ 1 个 step（实际取设备 ``inc``）。
WINDOW_SIDE_TOLERANCE_STEP = 1
#: 像元间距：相对差 ≥ 0.5%。
CAMERA_PIXEL_REL_TOLERANCE = 0.005
#: 曝光：相对差 ≥ 1%。
EXPOSURE_REL_TOLERANCE = 0.01
#: 几何相关度：绝对差 ≥ 0.02。
CORRELATION_ABS_TOLERANCE = 0.02

#: Bare fallback for 入射中心 when a session has **never** been calibrated.
#:
#: This is this bench's measured panel centre, so a different optical path has to
#: calibrate — it is a last resort, not a default.  The sidebar widget starts
#: empty（「留空 = 自动定位」); this only ever appears when nothing was probed,
#: and it is what the sidebar back-fills when a calibration skipped locate.
PUPIL_CENTER_FALLBACK = "960,690"

#: 三态前缀。
MARK_DETECTED = "●"
MARK_MANUAL = "✎"
MARK_UNSET = "○"

# ---------------------------------------------------------------------------
# session key 契约
# ---------------------------------------------------------------------------

#: 标定或整形运行中为真。侧栏光路控件据此全部 ``disabled=True``。
BUSY_KEY = "bench_busy"
#: 相机窗口上限的探测缓存（按 cam_type / cam_id 失效）。
WINDOW_CACHE_KEY = "bench_window_probe"

#: 像元间距探测的会话键（原在 multi_slm_controller 的「相机工具」区，现搬到这里）。
CAMERA_PIXEL_KEY = "camera_pixel_um"

# ---------------------------------------------------------------------------
# 三态字段规格
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchField:
    """一个可探测字段的三态规格。

    Attributes:
        name: 字段名（在 detected / manual / effective 三张表里都用它）。
        widget_key: 承载**生效值**的 Streamlit widget key。
        unit: 显示单位（空串表示无量纲/字符串）。
        kind: 比较方式 —— ``center``（逐坐标）、``abs``（绝对差）、
            ``rel``（相对差）、``text``（字符串相等）。
        tolerance: 阈值（``abs`` / ``rel`` / ``center`` 有效）。
        label: 显示名。
    """

    name: str
    widget_key: str
    unit: str
    kind: str
    tolerance: float
    label: str


def _scoped(prefix: str, name: str) -> str:
    """命名空间化 session key；``prefix=""`` 得到无前缀的实验级 key。"""
    return f"{prefix}_{name}" if prefix else name


def bench_fresnel_focal_mm(prefix: str = "slm1") -> float:
    """该光路**探测到的**菲涅尔焦距补偿 (mm)；0 = 没有可用结果。

    相机一旦离开 2f 焦面，plain 相位就不再是最亮的，补偿量是一个光路事实：
    它随相机位置的毫米级移动而变，也不能从标称 125 mm 透镜推出来（面板相位
    的符号方向与衍射效率都不在纸面上）。所以只读最近一次
    :func:`~ao_shaping.tools.slm.panel.focus_probe.probe_session_focus` 的结果，
    绝不在此处猜。
    """
    probe = st.session_state.get(_scoped(prefix, "bench_focus_probe")) or {}
    try:
        value = float(probe.get("best_focal_mm") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) else 0.0


def detected_key(prefix: str) -> str:
    """探测值表的 session key。"""
    return _scoped(prefix, "bench_detected")


def manual_key(prefix: str) -> str:
    """人工覆盖表的 session key（只列被人工改过的字段）。"""
    return _scoped(prefix, "bench_manual")


def pending_key(prefix: str) -> str:
    """单次 flush 用的待写表（非 widget key，不受 widget 锁影响）。"""
    return _scoped(prefix, "bench_pending")


def effective_key(prefix: str) -> str:
    """运行开始时的生效快照。"""
    return _scoped(prefix, "bench_effective")


_EXPERIMENT_FIELD_NAMES = ("camera_pixel_um", "exposure_ms", "cam_size")
_SLM_FIELD_NAMES = ("pupil_center", "panel_span_px", "geometry_file")


def bench_fields(prefix: str) -> dict[str, BenchField]:
    """该命名空间下参与三态的 widget 字段。

    实验级（``prefix=""``）是相机侧三项（像元间距 / 曝光 / 窗口边长），
    每 SLM 级（``prefix="slm{N}"``）是面板侧三项（入射中心 / 模型覆盖宽度 /
    几何文件）。``geometry_correlation`` / ``evidence_dir`` / ``calibrated_at``
    是**只读**探测信息、没有 widget，因此不在本表里。

    The widget key is **re-derived with the prefix**: the catalog is module-level
    (the SLM-level names have no prefix of their own), and a shared key across
    SLM 1 / SLM 2 would silently merge two different optical paths into one
    value — exactly the dual-SLM mistake the two-level split exists to prevent.
    """
    names = _EXPERIMENT_FIELD_NAMES if not prefix else _SLM_FIELD_NAMES
    return {
        name: replace(field, widget_key=_scoped(prefix, field.widget_key))
        for name, field in ((n, _FIELD_CATALOG[n]) for n in names)
    }


_FIELD_CATALOG: dict[str, BenchField] = {
    "camera_pixel_um": BenchField(
        "camera_pixel_um",
        CAMERA_PIXEL_KEY,
        "μm",
        "rel",
        CAMERA_PIXEL_REL_TOLERANCE,
        "相机像元间距",
    ),
    "exposure_ms": BenchField(
        "exposure_ms",
        "bench_exposure_ms",
        "ms",
        "rel",
        EXPOSURE_REL_TOLERANCE,
        "会话曝光",
    ),
    "cam_size": BenchField(
        "cam_size",
        "bench_cam_size",
        "px",
        "abs",
        float(WINDOW_SIDE_TOLERANCE_STEP),
        "读出窗口边长",
    ),
    "pupil_center": BenchField(
        "pupil_center",
        "bench_pupil_center",
        "px",
        "center",
        CENTER_TOLERANCE_PX,
        "入射中心",
    ),
    "panel_span_px": BenchField(
        "panel_span_px",
        "bench_panel_span_px",
        "px",
        "abs",
        PANEL_SPAN_TOLERANCE_PX,
        "模型覆盖宽度",
    ),
    "geometry_file": BenchField(
        "geometry_file", "bench_geometry_file", "", "text", 0.0, "几何文件"
    ),
}


# ---------------------------------------------------------------------------
# 三张表的读写
# ---------------------------------------------------------------------------


def detected_state(prefix: str) -> dict[str, Any]:
    """探测值表（只读视图）。"""
    return dict(st.session_state.get(detected_key(prefix)) or {})


def manual_state(prefix: str) -> dict[str, Any]:
    """人工覆盖表（只读视图；只列被人工改过的字段）。"""
    return dict(st.session_state.get(manual_key(prefix)) or {})


def pending_state(prefix: str) -> dict[str, Any]:
    """待 flush 表（只读视图）。"""
    return dict(st.session_state.get(pending_key(prefix)) or {})


def effective_state(prefix: str) -> dict[str, Any]:
    """运行开始时的生效快照（只读视图）。"""
    return dict(st.session_state.get(effective_key(prefix)) or {})


def widget_default(key: str, default: Any) -> Any:
    """``value=`` for a keyed widget — ``None`` once the key holds a value.

    Streamlit's rule: a widget with a ``key`` is **initialised from
    ``st.session_state[key]``**, and supplying ``value=`` as well is a conflict
    it resolves in favour of session state while logging
    *"created with a default value but also had its value set via the Session
    State API"*. So passing ``value=st.session_state.get(key, default)`` is
    doubly wrong: it triggers the warning, and the number it passes is ignored
    anyway once the key exists.

    Returning ``None`` means "use the session value"; returning ``default`` only
    on the very first render (key absent) seeds the initial value. Every keyed
    widget in this module goes through here so the warning cannot come back one
    control at a time.
    """
    return None if key in st.session_state else default


def queue_pending(prefix: str, values: dict[str, Any]) -> None:
    """把「下一次 flush 要写进 widget key 的值」排队。

    调用方随后必须 ``st.rerun()``，让 :func:`flush_pending` 在控件实例化之前
    先把值写进去。
    """
    pending = dict(st.session_state.get(pending_key(prefix)) or {})
    pending.update(values)
    st.session_state[pending_key(prefix)] = pending


def flush_pending(prefix: str) -> dict[str, Any]:
    """把待写值写进对应 widget key 并清空待写表；返回实际写入的项。

    必须在**任何相关控件实例化之前**调用 —— 此刻 widget key 尚未被锁。
    直接赋值（不吞异常）：回填静默失败比报错更难查。
    """
    pending = dict(st.session_state.get(pending_key(prefix)) or {})
    if not pending:
        return {}
    for key, value in pending.items():
        st.session_state[key] = value
    st.session_state[pending_key(prefix)] = {}
    return pending


def record_detection(prefix: str, values: dict[str, Any]) -> None:
    """写入 / 合并探测值（**不碰** widget，也不碰人工覆盖表）。

    所以「重新标定不覆盖手动值」是结构性保证，而不是靠调用方记得判断。
    """
    detected = detected_state(prefix)
    detected.update(values)
    st.session_state[detected_key(prefix)] = detected


def set_effective_state(prefix: str, snapshot: dict[str, Any]) -> None:
    """记录一次运行的生效光路参数快照。"""
    st.session_state[effective_key(prefix)] = dict(snapshot)


def reset_manual(prefix: str, *names: str) -> None:
    """从人工覆盖表里删掉给定字段（``恢复探测值`` 用）。

    不直接改 widget：调用方负责 ``queue_pending`` + ``st.rerun()``。
    """
    manual = manual_state(prefix)
    for name in names or tuple(manual):
        manual.pop(name, None)
    st.session_state[manual_key(prefix)] = manual


def bench_busy() -> bool:
    """标定或整形运行中？"""
    return bool(st.session_state.get(BUSY_KEY, False))


def set_bench_busy(busy: bool) -> None:
    """置 / 清「标定或整形运行中」标志。"""
    st.session_state[BUSY_KEY] = bool(busy)


# ---------------------------------------------------------------------------
# 三态判定
# ---------------------------------------------------------------------------


def _coerce_point(value: Any) -> tuple[float, float] | None:
    """``"960,690"`` / ``(960, 690)`` / ``[960, 690]`` → ``(960.0, 690.0)``。"""
    if value is None:
        return None
    if isinstance(value, str):
        parts = [p for p in value.replace("，", ",").split(",") if p.strip() != ""]
        if len(parts) != 2:
            return None
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            return None
    try:
        seq = list(value)
    except TypeError:
        return None
    if len(seq) != 2:
        return None
    try:
        return float(seq[0]), float(seq[1])
    except (TypeError, ValueError):
        return None


def _field_tolerance(field: BenchField) -> float:
    """字段阈值；窗口边长用设备 ``inc`` 折算成「1 个 step」。"""
    if field.name == "cam_size":
        return float(window_bounds().step) * WINDOW_SIDE_TOLERANCE_STEP
    return float(field.tolerance)


def field_diff(field: BenchField, current: Any, detected: Any) -> float | None:
    """``current`` 与 ``detected`` 的差值；不可比 / 视为相同时返回 ``None``。

    ``center`` 取两坐标差的最大值；``rel`` 为相对差（探测值为 0 时退回绝对差）。
    """
    if detected is None:
        return None
    if field.kind == "text":
        return None if str(current or "").strip() == str(detected).strip() else 1.0
    if field.kind == "center":
        a, b = _coerce_point(current), _coerce_point(detected)
        if a is None or b is None:
            return None
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
    try:
        cur, det = float(current), float(detected)
    except (TypeError, ValueError):
        return None
    if field.kind == "rel":
        if det == 0.0:
            return abs(cur)
        return abs(cur - det) / abs(det)
    return abs(cur - det)


def _is_manual(field: BenchField, current: Any, detected: Any) -> bool:
    """当前值是否已偏离探测值超过阈值。"""
    if detected is None:
        return False
    if field.kind == "text":
        return str(current or "").strip() != str(detected).strip()
    diff = field_diff(field, current, detected)
    return diff is not None and diff > _field_tolerance(field)


def sync_manual_states(prefix: str) -> dict[str, Any]:
    """按当前 widget 值重算人工覆盖表；在控件实例化**之前**调用。

    「人工改过」的判据只能是「当前值 ≠ 探测值（超过阈值）」：Streamlit 没有
    变更事件，而 ``flush_pending`` 刚写进去的值恰好等于探测值，所以自动回填
    不会被误判成人工值。
    """
    fields = bench_fields(prefix)
    detected = detected_state(prefix)
    manual = {
        name: st.session_state.get(field.widget_key)
        for name, field in fields.items()
        if _is_manual(field, st.session_state.get(field.widget_key), detected.get(name))
    }
    st.session_state[manual_key(prefix)] = manual
    return manual


def field_state(prefix: str, name: str) -> str:
    """字段三态：``manual`` > ``detected`` > ``unset``。"""
    if name in manual_state(prefix):
        return "manual"
    if name in detected_state(prefix):
        return "detected"
    return "unset"


def field_states(prefix: str) -> dict[str, str]:
    """该命名空间下全部字段的三态。"""
    return {name: field_state(prefix, name) for name in bench_fields(prefix)}


def manual_fields(prefix: str) -> tuple[str, ...]:
    """当前被人工覆盖的字段名（按字段声明顺序，保证 caption 稳定）。"""
    manual = manual_state(prefix)
    return tuple(name for name in bench_fields(prefix) if name in manual)


# ---------------------------------------------------------------------------
# 值的解析（生效值 = widget 值；人工值自动压过探测值）
# ---------------------------------------------------------------------------


def format_value(field: BenchField, value: Any) -> str:
    """字段值的显示文本（带单位）。"""
    if field.kind == "center":
        point = _coerce_point(value)
        text = f"({point[0]:g}, {point[1]:g})" if point is not None else "（未设置）"
    elif field.kind == "text":
        text = str(value).strip() or "（空）"
    else:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "（未设置）"
        text = f"{number:.4g}"
    return f"{text} {field.unit}".rstrip()


def field_hint(prefix: str, name: str) -> str:
    """一行三态说明：``●`` 探测值 / ``✎`` 人工值（附探测值与差值）。"""
    fields = bench_fields(prefix)
    field = fields.get(name)
    if field is None:
        return ""
    state = field_state(prefix, name)
    detected = detected_state(prefix)
    if state == "detected":
        return f"{MARK_DETECTED} {field.label}: 探测值 {format_value(field, detected.get(name))}"
    if state != "manual":
        return f"{MARK_UNSET} {field.label}: 未探测"
    current = manual_state(prefix).get(name, st.session_state.get(field.widget_key))
    diff = field_diff(field, current, detected.get(name))
    if field.kind == "rel" and diff is not None:
        delta = f"{diff:.2%}"
    elif field.kind == "text":
        delta = "值不同"
    else:
        delta = "—" if diff is None else f"{diff:.4g}"
    return (
        f"{MARK_MANUAL} {field.label}: 人工 {format_value(field, current)}"
        f" / 探测 {format_value(field, detected.get(name))} / 差 {delta}"
    )


def bench_value(prefix: str, name: str, fallback: Any = None) -> Any:
    """字段的生效值：widget 值优先，回落到 ``fallback``。"""
    field = bench_fields(prefix).get(name)
    if field is None:
        return fallback
    value = st.session_state.get(field.widget_key)
    return fallback if value is None else value


def bench_pupil_center(
    prefix: str = "slm1", fallback: str = PUPIL_CENTER_FALLBACK
) -> str:
    """入射中心的字符串形式（``"x,y"``）。

    解析顺序：人工填的文本 → 本次标定的探测值 → ``fallback``。
    ``fallback`` 默认是本实验系统的实测值，只在**从未标定过**时兜底；控件本身
    的默认值是空串，placeholder 写「留空 = 自动定位」。
    """
    raw = str(
        st.session_state.get(_scoped(prefix, "bench_pupil_center"), "") or ""
    ).strip()
    if raw:
        return raw
    detected = detected_state(prefix).get("pupil_center")
    point = _coerce_point(detected)
    if point is not None:
        return f"{int(round(point[0]))},{int(round(point[1]))}"
    return fallback


# ---------------------------------------------------------------------------
# 离焦叠加（T4 发送管线叠加层）
# ---------------------------------------------------------------------------
# 离焦补偿 = 相机到 2f 面距离决定的二次相位（菲涅尔项，mm，粗调）+ Zernike
# (2,0) 细调。与发送的图案内容无关——平场 / CSV / GS 整形解都需要同一层，所以
# 做成驱动层叠加（set_phase_overlay），开关一开就作用于其后所有发送。焦距默认
# 取 D2「探测焦面」的实测值，操作员可手动覆盖；半径/中心复用 bench 取值链。

#: 离焦焦距的"自动"哨兵：number_input 里填这个值 = 用探测推荐值，人工填别的即覆盖。
DEFOCUS_FOCAL_AUTO = -9999.0


def bench_defocus_focal_auto(prefix: str = "slm1") -> float:
    """探测推荐的离焦焦距 (mm)；无可用探测结果时返回 0.0。

    与 :func:`bench_fresnel_focal_mm` 同源（D2 探测焦面的 best_focal_mm），
    只在这里多一层"离焦叠加"语义，避免 GUI 直接依赖 probe 的内部 dict 结构。
    """
    return bench_fresnel_focal_mm(prefix)


def bench_defocus_focal(prefix: str = "slm1") -> float:
    """离焦叠加生效的焦距 (mm)。

    ``-9999``（自动）时返回探测推荐值；操作员手动填的值优先。返回值直接喂给
    :func:`build_defocus_overlay` 的菲涅尔项。
    """
    v = st.session_state.get(_scoped(prefix, "defocus_focal"))
    if v is None or (isinstance(v, float) and v == DEFOCUS_FOCAL_AUTO):
        return bench_defocus_focal_auto(prefix)
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return bench_defocus_focal_auto(prefix)
    return fv if np.isfinite(fv) else bench_defocus_focal_auto(prefix)


def build_defocus_overlay(prefix: str = "slm1", slm=None) -> np.ndarray:
    """按当前离焦设置构造**面板坐标下**的离焦叠加相位（弧度，panel_size）。

    纯函数：每次从 session 现算，绝不累积；焦距/细调均为零时返回与全零逐位相同
    的数组。菲涅尔项用 :func:`fresnel_panel`（粗调，mm），Zernike (2,0) 用
    :func:`zernike_panel`（细调）。半径/中心复用 bench 取值链（``panel_span_px``
    的半径 = span/2，``bench_pupil_center``），波长/像元取设备实参（缺省 1064nm /
    8 μm）。
    """
    from ao_shaping.tools.kernels.bench_kernels import (
        SLM_PANEL_H,
        SLM_PANEL_W,
        fresnel_panel,
        zernike_panel,
    )

    panel_shape = (SLM_PANEL_H, SLM_PANEL_W)
    centre = bench_pupil_center(prefix)
    point = _coerce_point(centre) or (960.0, 600.0)
    # 半径 = 照明直径/2（panel_span_px 即 2×半径）；缺省 900 px 口径 → 450 px。
    span = float(
        st.session_state.get(_scoped(prefix, "bench_panel_span_px"), 900.0) or 900.0
    )
    radius = max(1, int(round(span / 2.0)))
    wavelength = getattr(slm, "wavelength", None) if slm is not None else None
    pixel_um = float(getattr(slm, "Pitch_um", 8.0)) if slm is not None else 8.0

    out = np.zeros(panel_shape, dtype=np.float64)
    focal = bench_defocus_focal(prefix)
    if abs(focal) > 1e-9:
        out = out + fresnel_panel(
            float(focal),
            radius=radius,
            pupil_center=(int(round(point[0])), int(round(point[1]))),
            panel_shape=panel_shape,
            wavelength_nm=float(wavelength) if wavelength else 1064.0,
            pixel_um=pixel_um,
        )
    amount = float(st.session_state.get(_scoped(prefix, "defocus_amount"), 0.0) or 0.0)
    if abs(amount) > 1e-9:
        out = out + zernike_panel(
            {(2, 0): float(amount)},
            radius=radius,
            pupil_center=(int(round(point[0])), int(round(point[1]))),
            panel_shape=panel_shape,
        )
    return out


def build_carrier_overlay(prefix: str = "slm1", slm=None) -> np.ndarray | None:
    """构造**载波（倾斜）**叠加相位（弧度，panel_size）；未启用时返回 ``None``。

    这是「载波没有叠加倾斜相位」那个缺陷的落点：载波原先只作为
    ``blaze_period_px`` 喂给整形优化器，由优化器自己在相位里加，于是侧栏
    「一次下发」的 PatternHelper 相位**从来没带上它**。现在载波成为发送管线
    叠加层的一层（与离焦同槽位、同机制），任何走
    ``create_phase_from_array`` 的发送都会带上。

    周期语义沿用侧栏：``-1`` = 关闭、``0`` = 实测自动。自动选择需要相机反馈
    （属于整形运行内部的事），叠加层这里没有画面可依，所以 ``0`` 也按未启用
    处理，而不是猜一个周期。
    """
    if not bool(st.session_state.get(_scoped(prefix, "blaze_enabled"), False)):
        return None
    period = int(st.session_state.get("bench_blaze_period_px", -1))
    if period <= 0:
        return None
    from ao_shaping.tools.kernels.bench_kernels import SLM_PANEL_H, SLM_PANEL_W
    from ao_shaping.utils.wavefront.zero_order_overlay import (
        OverlayParams,
        apply_zero_order_overlay,
    )

    panel_shape = (SLM_PANEL_H, SLM_PANEL_W)
    direction = str(st.session_state.get("bench_blaze_direction", "vertical"))
    params = OverlayParams(
        mode="blaze",
        blaze_period_px=float(period),
        blaze_direction=direction,
        wavelength_nm=float(
            (getattr(slm, "wavelength", None) if slm is not None else None) or 1064.0
        ),
        panel_pixel_um=float(
            (getattr(slm, "Pitch_um", 8.0) if slm is not None else 8.0) or 8.0
        ),
    )
    try:
        out = apply_zero_order_overlay(np.zeros(panel_shape, dtype=np.float64), params)
    except Exception as exc:  # policy: 载波叠加是可选增强：构造失败返回 None，主流程按不叠加继续
        logger.warning("载波叠加构造失败 ({}): {}", prefix, exc)
        return None
    return np.asarray(out, dtype=np.float64)


def rebuild_defocus_overlay(prefix: str, slm) -> None:
    """把当前离焦设置推到驱动（set_phase_overlay），供叠加层在重连/重取后重建。

    受 ``{prefix}_defocus_enabled`` 开关门控：关 = 不叠加（None）；开 = 按焦距+
    细调构造（全零也置 None，等价不叠加）。无连接时直接返回（不碰驱动）。
    """
    if slm is None or not getattr(slm, "is_open", False):
        return
    enabled = bool(st.session_state.get(_scoped(prefix, "defocus_enabled"), False))
    if not enabled:
        slm.set_phase_overlay(None)
        return
    try:
        overlay = build_defocus_overlay(prefix, slm)
    except Exception as exc:  # policy: 重连后离焦叠加重建失败只跳过叠加，不得让重连本身失败
        logger.warning("SLM {} 离焦叠加重建失败: {}", slm.slm_number, exc)
        return
    if np.allclose(overlay, 0.0, atol=1e-12):
        slm.set_phase_overlay(None)
    else:
        slm.set_phase_overlay(overlay)


def build_send_overlay(
    prefix: str = "slm1", slm=None, *, include_carrier: bool = True
) -> PhaseOverlay | None:
    """按当前 session 状态合成**发送管线叠加层**（离焦 + 载波 + 底相位）。

    这是「叠加层真值」的唯一读取点：controller 的 ``_apply_slm_overlays`` 和
    闭环整形（gs-refine / model-in-loop）的 GUI 入口共用它，保证 GUI 会话句柄
    与优化器自开句柄拿到**同一份**叠加。纯读 session，无硬件 I/O：

    * 离焦（弧度）：受 ``{prefix}_defocus_enabled`` 开关门控，全零视同不叠加
      （与 controller 既有语义一致）。
    * 底相位（灰度）：仅「叠加底相位」勾选且已保存时叠加，否则 None。

    返回 ``None`` 表示**无任何叠加**（调用方据此跳过 ``apply_overlays_to_handle``，
    避免把既有叠加误清零）。任一层构建失败（如离焦构造抛错）只记 warning 并
    降级为不叠该层，不卡主流程。
    """
    phase_rad = None
    defocus_enabled = bool(
        st.session_state.get(_scoped(prefix, "defocus_enabled"), False)
    )
    if defocus_enabled:
        try:
            built = build_defocus_overlay(prefix, slm)
        except Exception as exc:  # policy: 下发时的离焦叠加失败按不叠加继续（built=None），不得打断下发
            logger.warning("SLM 离焦叠加重建失败 ({}): {}", prefix, exc)
            built = None
        if built is not None and np.allclose(built, 0.0, atol=1e-12):
            built = None
        phase_rad = built

    # 载波与离焦同为弧度域叠加，直接相加。它们都是「光路里真实存在的相位」，
    # 因此和离焦一样必须经过矫正/LUT（与底相位的灰度域语义不同）。
    # ``include_carrier=False`` 用于**相位本身已经含载波**的发送：迁移学习闭环
    # （``tools/slm/transfer``）把载波写进它的 ``ramp_matrix`` 并烘焙进接受相位
    # （见 ``transfer.validate.run`` 里的 ``to_panel`` 与 ``set_phase_overlay(None)``），
    # 再叠一次就是两份倾斜、0 级跑两倍远。
    carrier = build_carrier_overlay(prefix, slm) if include_carrier else None
    if carrier is not None:
        phase_rad = carrier if phase_rad is None else phase_rad + carrier

    base = st.session_state.get(_scoped(prefix, "base_phase"))
    overlay_on = bool(st.session_state.get(_scoped(prefix, "overlay_base"), False))
    base_gray = base if (base is not None and overlay_on) else None
    if phase_rad is None and base_gray is None:
        return None
    return PhaseOverlay(phase_rad, base_gray)


# ---------------------------------------------------------------------------
# 相机窗口边界（真值来自设备，拿不到才用保守兜底）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowBounds:
    """正方形读出窗口的合法边长区间。"""

    min_side: int
    max_side: int
    step: int
    source: str
    detail: str

    @property
    def is_device(self) -> bool:
        """True = 来自相机特征网格，而不是兜底值。"""
        return self.source == "device"

    def clamp(self, value: int) -> int:
        """把一个边长夹到「网格上的」合法区间。"""
        step = max(1, int(self.step))
        clamped = max(self.min_side, min(int(value), self.max_side))
        steps = max(0, round((clamped - self.min_side) / step))
        return int(self.min_side + steps * step)


def probe_camera_window(cam_type: str = "daheng", cam_id: int = 0) -> WindowSize | None:
    """读相机 ``Width`` / ``Height`` 特征网格，返回正方形窗口的边界。

    ``inc`` 不可省：大恒 gxipy 的 ``IntFeature.set`` 对**非网格值不抛异常、
    只 print 一行噪声然后静默保留旧窗口**（见 ``drivers/ccd/AGENTS.md``），
    所以 step 必须取设备的 ``inc``，正方形窗口上界 = ``min(max_width, max_height)``。

    拿不到（相机未连接 / SDK 不可用 / 非大恒）一律返回 ``None``，由调用方用
    保守兜底并在 caption 说明。
    """
    if cam_type != "daheng":
        return None
    try:
        import gxipy as gx  # type: ignore[import-untyped]
    except (ImportError, OSError):
        return None
    cam: Any = None
    try:
        manager = gx.DeviceManager()
        _, infos = manager.update_device_list()
        if not infos or len(infos) <= int(cam_id):
            return None
        cam = manager.open_device_by_sn(infos[int(cam_id)].get("sn"))
        width_range = cam.Width.get_range()
        height_range = cam.Height.get_range()
        return WindowSize(
            width=int(cam.Width.get()),
            height=int(cam.Height.get()),
            max_width=int(width_range["max"]),
            max_height=int(height_range["max"]),
            inc=int(width_range.get("inc") or 2),
        )
    except Exception as exc:  # policy: 相机窗口探测失败不是错误，只是拿不到参考尺寸：返回 None 让调用方用默认值
        logger.debug("相机窗口上限探测失败 ({}): {}", cam_type, exc)
        return None
    finally:
        if cam is not None:
            try:
                cam.close_device()
            except Exception:  # policy: 探测用的临时相机句柄：关闭失败无可补救，不得掩盖探测结果
                pass


def _fallback_bounds() -> WindowBounds:
    return WindowBounds(
        min_side=FALLBACK_WINDOW_STEP,
        max_side=FALLBACK_WINDOW_MAX_SIDE,
        step=FALLBACK_WINDOW_STEP,
        source="fallback",
        detail=(
            f"相机不可达，用保守兜底 {FALLBACK_WINDOW_MAX_SIDE} px / 步长 "
            f"{FALLBACK_WINDOW_STEP} px；点「探测窗口上限」读设备真值"
        ),
    )


def window_bounds(*, probe: bool = False) -> WindowBounds:
    """当前相机的正方形窗口边界（按 cam_type / cam_id 缓存探测结果）。

    探测只在「换相机」或显式 ``probe=True`` 时发生，避免每次渲染都去打开设备；
    运行中（``bench_busy``）不探测 —— 句柄已交给标定器 / 优化器。
    """
    cam_type = str(st.session_state.get("bench_cam_type", "daheng"))
    cam_id = int(st.session_state.get("bench_cam_id", 0) or 0)
    cached = st.session_state.get(WINDOW_CACHE_KEY)
    stale = (
        not isinstance(cached, dict)
        or cached.get("cam_type") != cam_type
        or int(cached.get("cam_id", -1)) != cam_id
    )
    if (probe or stale) and not bench_busy() and cam_type != "sim":
        window = probe_camera_window(cam_type, cam_id)
        cached = {
            "cam_type": cam_type,
            "cam_id": cam_id,
            "window": None
            if window is None
            else {
                "width": window.width,
                "height": window.height,
                "max_width": window.max_width,
                "max_height": window.max_height,
                "inc": window.inc,
            },
        }
        st.session_state[WINDOW_CACHE_KEY] = cached
    window = (cached or {}).get("window") if isinstance(cached, dict) else None
    if not window:
        return _fallback_bounds()
    inc = max(1, int(window.get("inc") or 1))
    max_side = min(int(window["max_width"]), int(window["max_height"]))
    return WindowBounds(
        min_side=inc,
        max_side=max_side,
        step=inc,
        source="device",
        detail=(
            f"来自相机特征网格：Width/Height.max = {window['max_width']}×"
            f"{window['max_height']}，inc = {inc}（非网格值 SDK 会静默保留旧窗口）"
        ),
    )


def bench_camera() -> tuple[str, int]:
    """当前实验级相机 (类型, ID)。"""
    return (
        str(st.session_state.get("bench_cam_type", "daheng")),
        int(st.session_state.get("bench_cam_id", 0) or 0),
    )


def bench_window_side() -> int:
    """**目标方边长的上界**：开窗 = 窗口边长；关窗 = 传感器短边。"""
    bounds = window_bounds()
    if not bool(st.session_state.get("bench_cam_crop", True)):
        return bounds.max_side
    raw = st.session_state.get("bench_cam_size", bounds.max_side)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return bounds.max_side


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------


def validate_bench_inputs() -> str | None:
    """运行前对侧栏光路设置做一次校验；返回错误文本，无问题返回 ``None``。

    放在主区**按钮之前**渲染，才能把错误和 ``disabled=True`` 一起呈现。
    """
    cam_type, _ = bench_camera()
    crop = bool(st.session_state.get("bench_cam_crop", True))
    exposure = st.session_state.get("bench_exposure_ms", 0.0)
    try:
        exposure_ms = float(exposure)
    except (TypeError, ValueError):
        return f"会话曝光不是数值: {exposure!r}"
    if exposure_ms < 0:
        return f"会话曝光必须 ≥ 0（0 = 自动起曝），当前 {exposure_ms}"
    if cam_type == "daheng":
        ceiling = configured_daheng_exposure_limit()
        if ceiling > 0 and exposure_ms > ceiling:
            return (
                f"会话曝光 {exposure_ms} ms 超过本光路饱和保护上限 {ceiling} ms"
                "（AO_DAHENG_MAX_EXPOSURE_MS）"
            )
        if exposure_ms == 0:
            st.warning(
                "会话曝光 = 0：大恒驱动会钳到约 0.02 ms，比可用区间暗 20–75×，"
                "先按当前激光功率实测曝光再回填。"
            )
    bounds = window_bounds()
    if crop:
        side = st.session_state.get("bench_cam_size", bounds.min_side)
        if side is None:
            side = bounds.min_side
        if int(side) % int(bounds.step) != 0:
            return (
                f"窗口边长 {int(side)} px 不是设备步长 {bounds.step} px 的整数倍；"
                "非网格值 SDK 会静默保留旧窗口"
            )
    return None


# ---------------------------------------------------------------------------
# 曝光探测
# ---------------------------------------------------------------------------

#: 探测证据落在 ``data/debug/<脚本名>/`` 下，与本模块的其它探测分开。
EXPOSURE_PROBE_DEBUG_DIR = Path("data") / "debug" / "bench_settings" / "exposure_probe"

#: 确认帧的合格区间。返回的曝光必须让实测峰值落在这里，否则就是截断值。
_CONFIRM_SATURATION = 0.9
_CONFIRM_TARGET = 0.55


class ExposureProbeFailure(RuntimeError):
    """曝光探测失败，携带一条可直接上屏的中文说明。"""


def _confirm_working_exposure(
    read_frame: Callable[[], np.ndarray],
    reset_exposure: Callable[[float], float | None],
    ceiling_ms: float,
    *,
    work_fraction: float,
    floor_ms: float,
    max_attempts: int = 3,
) -> tuple[float, float, float, list[dict[str, float]], np.dtype]:
    """回读验证工作曝光，返回 ``(工作曝光, 读回曝光, 峰值, 逐次回读记录, 帧 dtype)``。

    阶梯给出的上限只是响应模型；必须用**那一档曝光**回读一帧再判定，否则会把
    截断的曝光当成可用的交出去（实测 2026-10-17：阶梯定上限 0.243 ms，取其一半
    回读却是 255/255）。

    饱和时按比例回退再试；``max_attempts`` 次仍饱和则抛
    :class:`RuntimeError`。本函数不碰设备状态，只用注入的两个回调，因此可以
    用合成帧直接做数值回归。最后一个返回值是回读帧的 dtype，调用方据此算满量程。
    """
    working = working_exposure_ms(ceiling_ms, fraction=work_fraction, floor_ms=floor_ms)
    confirmations: list[dict[str, float]] = []
    readback = working
    peak = 0.0
    dtype: np.dtype = np.dtype(np.uint8)
    for _ in range(max_attempts):
        readback = float(reset_exposure(working) or working)
        frame = np.asarray(read_frame())
        peak = float(np.max(frame)) if frame.size else 0.0
        dtype = frame.dtype
        full = (
            float(np.iinfo(frame.dtype).max)
            if np.issubdtype(frame.dtype, np.integer)
            else 255.0
        )
        frac = peak / full if full else 0.0
        confirmations.append({"ms": working, "peak": peak, "peak_fraction": frac})
        if frac >= _CONFIRM_SATURATION:
            # Clipped: back off proportionally rather than returning it.
            working = max(floor_ms, working * _CONFIRM_TARGET / max(frac, 1e-9))
            continue
        break
    else:
        raise RuntimeError(
            f"工作曝光在 {len(confirmations)} 次回退后仍然饱和"
            f"（最后一次 {confirmations[-1]['peak_fraction']:.0%}）；"
            "光路比阶梯看到的高出很多，请降低激光功率或加衰减后重测"
        )
    return working, readback, peak, confirmations, dtype


def probe_session_exposure_ms(
    cam_type: str = "daheng",
    cam_id: int = 0,
    *,
    work_fraction: float = 0.5,
    floor_ms: float = 0.02,
    debug_dir: Path | None = None,
) -> tuple[float, dict[str, Any]]:
    """在当前光路下实测一个可用的会话曝光（ms），返回 ``(曝光, 证据)``。

    探测用**倍增阶梯**找饱和上限（``probe_exposure_ceiling_ms``：曝光翻倍直到
    峰值达到 ``saturation_fraction``×满量程），再用 ``working_exposure_ms`` 取
    ``work_fraction``×上限作为工作曝光。上限取「峰值将达满量程 90%」的那一档
    再乘 0.9 余量（≈81%），所以工作曝光下峰值约在 40%——既能拿到足够原始
    计数，又留有余量防截断。

    失败时抛 :class:`ExposureProbeFailure`：曝光是**光路属性**，不随传感器变，
    所以宁可中止也不能给一个猜的值。若 ``AO_DAHENG_MAX_EXPOSURE_MS`` 有配置则
    优先用它（不探测），这是既有契约，此处不改变优先级。

    上限一律再受**相机自己的曝光上限**约束：相机最大值是「传感器还能积分多久」，
    不是「何时饱和」，所以它只是上限而不是一次测量——但超过它的曝光根本设不下去
    （驱动会静默钳位，于是实测峰值与请求曝光对不上）。相机报不出范围的后端按
    **无上限**处理，与旧行为一致。

    相机在函数内打开并关闭；``debug_dir`` 给定时把阶梯每一档的
    ``(曝光, 峰值)`` 写成 JSON，便于复盘「当时画面到底多亮」。
    """
    from ao_shaping.drivers.ccd.common import (
        create_camera,
        get_camera_exposure_range,
    )
    from ao_shaping.utils.image.exposure_control import probe_exposure_ceiling_ms

    if cam_type.lower() != "daheng":
        # 其他相机类型走各自的曝光语义，本探测只服务大恒的定量 CCD 路径。
        raise ExposureProbeFailure(
            f"曝光自动探测目前只支持大恒（daheng），当前相机类型为 {cam_type!r}。"
            "请在「会话曝光」直接填入实测值。"
        )
    camera_max_ms: float | None = None
    try:
        with create_camera(cam_type, int(cam_id), exposure_time_ms=floor_ms) as cam:
            # ``n_sample=1`` 返回整数原始帧；取 n>1 会得到 float32 平均帧，
            # 而阶梯判定要求整数帧（见 docs/slm/slm_model_in_loop.md 实测记录）。
            #
            # 相机上限先问设备要：拿不到（后端不支持 / 调用失败）就按无上限
            # 继续，绝不让「读不到范围」把一次可用的探测变成失败。
            try:
                _, camera_max_ms = get_camera_exposure_range(cam)
            except Exception as exc:  # policy: 后端不支持曝光范围查询不是探测失败：按无上限继续本次探测
                logger.debug("相机曝光范围不可用，本次探测按无上限处理: {}", exc)
                camera_max_ms = None
            if camera_max_ms is not None and (
                not np.isfinite(camera_max_ms) or camera_max_ms <= 0
            ):
                camera_max_ms = None

            # Walk the ladder **once**: letting ``resolve_daheng_exposure_limit``
            # probe and then probing again here would cost a second walk and
            # could contradict itself on a bench whose light drifts (measured
            # 2026-10-17: peak at 0.1 ms moved 63 -> 79 within 20 min). Taking
            # the operator's env ceiling when it is set keeps that precedence
            # without touching the camera, exactly as the shared resolver does.
            configured = configured_daheng_exposure_limit()
            if configured > 0:
                # 操作员的取值压过探测，但不压过传感器：取二者中更紧的那个。
                ceiling = float(configured)
                if camera_max_ms is not None:
                    ceiling = min(ceiling, camera_max_ms)
                estimate = None
            else:
                estimate = probe_exposure_ceiling_ms(
                    cam.reset_exposure_time,
                    lambda: np.asarray(cam.get_numpy_image(n_sample=1)),
                    start_ms=floor_ms,
                    hard_max_ms=camera_max_ms,
                )
                ceiling = float(estimate.ceiling_ms)

            working, readback, peak, confirmations, dtype = _confirm_working_exposure(
                lambda: np.asarray(cam.get_numpy_image(n_sample=1)),
                cam.reset_exposure_time,
                ceiling,
                work_fraction=work_fraction,
                floor_ms=floor_ms,
            )
    except ExposureProbeFailure:
        raise
    except (RuntimeError, ValueError, OSError) as exc:
        raise ExposureProbeFailure(
            f"曝光探测失败：{exc}。"
            "请确认激光已出光、光路未遮挡，并在「会话曝光」手动填入实测值。"
        ) from exc

    full_scale = (
        float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 255.0
    )
    evidence: dict[str, Any] = {
        "cam_type": cam_type,
        "cam_id": int(cam_id),
        "ceiling_ms": ceiling,
        "work_fraction": float(work_fraction),
        "working_ms": working,
        "readback_ms": readback,
        "peak": peak,
        "full_scale": full_scale,
        "peak_fraction": peak / full_scale if full_scale else 0.0,
        "ceiling_source": "env" if estimate is None else "image_probe",
        "confirmations": confirmations,
        "last_safe_ms": float(estimate.last_safe_ms) if estimate else 0.0,
        "saturated_ms": None
        if estimate is None or estimate.saturated_ms is None
        else float(estimate.saturated_ms),
        "ladder_peak_fractions": (
            [] if estimate is None else [float(v) for v in estimate.peak_fractions]
        ),
        # How the ceiling was reached, so the next probe can be replayed: the
        # per-rung extrapolations (the old ``min`` of these) with their
        # exposures, the fitted saturation exposure, and the measured bound
        # that clamped it.  ``ladder_peak_fractions`` has one entry per rung;
        # ``ladder_estimates_ms`` only one per rung that carried signal, so zip
        # it with ``ladder_estimate_exposures_ms``.
        "ladder_estimates_ms": (
            [] if estimate is None else [float(v) for v in estimate.estimates_ms]
        ),
        "ladder_estimate_exposures_ms": (
            []
            if estimate is None
            else [float(v) for v in estimate.estimate_exposures_ms]
        ),
        "fit_ms": None if estimate is None else estimate.fit_ms,
        "fit_points": 0 if estimate is None else int(estimate.fit_points),
        "bound_ms": None if estimate is None else estimate.bound_ms,
        "first_clipped_ms": None if estimate is None else estimate.first_clipped_ms,
        # 相机自己报的曝光上限（None = 后端报不出，按无上限处理）。探测路径下它
        # 已被 ``probe_exposure_ceiling_ms`` 用作 ``hard_max_ms``，所以命中时
        # ``bound_ms`` 与它相等；env 路径没有 ``estimate``，这个键是唯一的记录。
        "camera_max_ms": camera_max_ms,
    }
    target = Path(debug_dir) if debug_dir is not None else EXPOSURE_PROBE_DEBUG_DIR
    try:
        target.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        (target / f"{stamp}.json").write_text(
            json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as exc:  # 证据落盘失败不否定已完成的测量
        logger.warning("曝光探测证据写入 {} 失败: {}", target, exc)
    logger.info(
        "曝光探测：上限 {:.4f} ms（最后未饱和 {:.4f} ms），工作曝光 {:.4f} ms，"
        "峰值 {:.0f}/{:.0f} ({:.0%})",
        ceiling,
        evidence["last_safe_ms"],
        working,
        peak,
        full_scale,
        evidence["peak_fraction"],
    )
    return working, evidence


# ---------------------------------------------------------------------------
# 照明半径探测
# ---------------------------------------------------------------------------

#: 半径扫描证据落在 ``data/debug/<脚本名>/`` 下。
BEAM_RADIUS_DEBUG_DIR = Path("data") / "debug" / "bench_settings" / "beam_radius"


def _centre_from_values(state: Any, slm_num: int) -> tuple[float, float] | None:
    """会话里该 SLM 的入射中心；没有就返回 ``None``。

    远边界只是一个位置，半径必须减去中心才是半径 —— 所以这一步失败时宁可
    报错，也不退回 ``edge / 2`` 那种假设光斑贴着面板原点的算法。

    人工填的优先于探测值，与侧栏其它字段一致。
    """
    prefix = f"slm{slm_num}"
    manual = _parse_point_text(
        str(state.get(_scoped(prefix, "bench_pupil_center"), "") or "")
    )
    if manual is not None:
        return manual
    detected = detected_state(prefix)
    point = _coerce_point(detected.get("pupil_center"))
    if point is None:
        return None
    return (float(point[0]), float(point[1]))


def _parse_point_text(text: str) -> tuple[float, float] | None:
    """``"960,690"`` → ``(960.0, 690.0)``；不合法返回 ``None``。"""
    parts = [p for p in str(text).replace("，", ",").split(",") if p.strip()]
    if len(parts) != 2:
        return None
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError:
        return None


def _radius_from_sweep(
    results: dict[str, Any],
    centre: tuple[float, float] | None,
) -> dict[str, Any]:
    """从两轴扫描结果算半径口径，返回可直接并入证据字典的字段。

    两个独立估计**并列**给出而不是相互覆盖：

    1. **scan-internal** —— 响应落差自身的宽度。落差从边界穿过光束近沿开始，
       到越过远沿后钉在地板上，所以落差宽度的一半**就是**半径、中点**就是**中心。
       不依赖 ``panel_locate``。
    2. **centre-subtracted** —— ``远边界 − 已测中心``。口径正确，但它继承中心的
       全部误差：实测 2026-10-17，同一次扫描用上一会话的中心给 180 px、用面板
       几何中心给 300 px —— 单是中心就带来 1.7× 的摆动。

    scan-internal 那个在两者都有时胜出，因为主导误差是中心而非步进；它自身的
    不确定度就是一个步进。
    """
    scan_radius = min(results[a].radius_px for a in ("x", "y"))
    scan_centre = (results["x"].centre_px, results["y"].centre_px)
    out: dict[str, Any] = {
        "radius_px": float(scan_radius),
        "radius_source": "scan_fall_width",
        "scan_centre_panel_px": [float(scan_centre[0]), float(scan_centre[1])],
    }
    step_px = float(max(results[a].step_px for a in ("x", "y")))
    out["step_px"] = step_px
    out["radius_uncertainty_px"] = step_px
    if centre is not None:
        out["centre_panel_px"] = [float(centre[0]), float(centre[1])]
        centred = {
            a: float(results[a].edge_px) - float(centre[0] if a == "x" else centre[1])
            for a in ("x", "y")
        }
        out["radius_x_px"] = centred["x"]
        out["radius_y_px"] = centred["y"]
        out["radius_from_centre_px"] = min(centred.values())
    return out


def measure_illuminated_radius(
    slm_num: int,
    cam_type: str,
    cam_id: int,
    *,
    exposure_ms: float | None = None,
    centre_hint: tuple[float, float] | None = None,
    steps: int = 16,
    frames: int = 3,
    repeats: int = 2,
    debug_dir: Path | None = None,
) -> dict[str, Any]:
    """实测面板上的照明半径，返回可渲染的证据字典（失败时带 ``error`` 键）。

    两轴各跑一次 :func:`ao_shaping.tools.slm.panel.beam_extent.sweep_axis` 的
    半平面随机相位边界扫描，取两轴远边界的**较小值**作半径。

    为什么取较小值：扫描量到的是「远边界」，半径 = 远边界 − **已测中心**，而中心
    来自 ``panel_locate``（本函数优先从会话里取，取不到才用 ``centre_hint``）。
    两轴独立扫描时各自的步进与饱和判据有差异，取较小的一轴给出**不会高估**
    照明口径的值 —— 而这个量最终要喂给物理预测模型的 ``aperture_m`` 交叉校验，
    高估会让校验形同虚设。

    这个量原先没有实测来源：``model_in_loop_hw_runbook --stage fit`` 用 sweep
    的 Zernike 半径（默认 200 px）代替，而两条焦面尺度推断相差 152% 的根因就是
    这个口径未经实测（见 ``model_in_loop_shaping.py`` 中该 warning 的措辞）。

    曝光缺省时先探一次（与 A 组「探测」同一个入口），否则 0 会让大恒钳到
    0.02 ms、边界扫描量到的是噪声。
    """
    from ao_shaping.tools.slm.panel.beam_extent import sweep_axis

    out: dict[str, Any] = {
        "slm_number": slm_num,
        "cam_type": cam_type,
        "cam_id": cam_id,
    }
    try:
        if exposure_ms is None or float(exposure_ms) <= 0:
            exposure_ms, _ = probe_session_exposure_ms(cam_type, cam_id)
        out["exposure_ms"] = float(exposure_ms)

        results = {}
        for axis in ("x", "y"):
            result = sweep_axis(
                axis=axis,
                steps=steps,
                frames=frames,
                repeats=repeats,
                core_radius=40.0,
                seed=2024,
                slm_number=slm_num,
                slm_wavelength=1064,
                cam_type=cam_type,
                cam_id=cam_id,
                exposure_ms=float(exposure_ms),
            )
            results[axis] = result
            out[f"edge_{axis}"] = float(result.edge_px)
            out[f"fall_lo_{axis}"] = float(result.fall_lo_px)
            out[f"fall_hi_{axis}"] = float(result.fall_hi_px)
            out[f"scan_centre_{axis}"] = float(result.centre_px)
            out[f"scan_radius_{axis}"] = float(result.radius_px)
            out[f"floor_rel_{axis}"] = float(result.floor_rel)
            out[f"profile_{axis}"] = [float(v) for v in result.profile_rel]
            out["flat_peak"] = float(result.flat_peak)
            out["flat_centroid"] = [float(v) for v in result.flat_centroid]

        centre = _centre_from_values(st.session_state, slm_num) or centre_hint
        out.update(_radius_from_sweep(results, centre))
        target = Path(debug_dir) if debug_dir is not None else BEAM_RADIUS_DEBUG_DIR
        target.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        (target / f"slm{slm_num}_{stamp}.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(
            "SLM {} 实测照明半径 {:.0f} 面板px（x 边 {:.0f} / y 边 {:.0f}，曝光 {:.3f} ms）",
            slm_num,
            out["radius_px"],
            out["edge_x"],
            out["edge_y"],
            out["exposure_ms"],
        )
    except Exception as exc:  # policy: 照明半径是侧栏诊断量：失败记 exception 并把原因写进 out["error"]，不掀翻面板
        logger.exception("SLM {} 照明半径探测失败", slm_num)
        out["error"] = str(exc)
    return out


# ---------------------------------------------------------------------------
# 像元间距探测（自 multi_slm_controller「相机工具」区整体搬来）
# ---------------------------------------------------------------------------


def detect_camera_pixel_um(cam_type: str = "daheng", cam_id: int = 0) -> float | None:
    """探测相机像元间距 (µm); 探不到返回 ``None``。

    单一来源是相机驱动 (``BaseCamera.pixel_size_um``): 大恒按**型号**从
    ``data/ccd_configs/ccd/models.json`` 解析 (未登记则回落已知型号表), MiiCam
    写死 2.9。本函数只做型号查询, **不打开相机** —— 型号已足够定位型号级配置,
    开相机只为读一个静态属性既慢又会抢设备锁。

    ⚠️ 不要改回读 GenICam 的 ``PixelSize`` 特征: 那是**位深** (``BPP8`` = 8),
    会把 Mono8 相机报成 8 µm。
    """
    if cam_type == "daheng":
        models = probe_daheng_models()
        if cam_id >= len(models):
            return None
        return lookup_model_pixel_size_um(models[cam_id])
    if cam_type == "miicam":
        try:
            from ao_shaping.drivers.ccd.miicam.driver import MIICamera

            return float(MIICamera.pixel_size_um) or None
        except (ImportError, OSError) as exc:
            logger.debug("MiiCam 驱动不可用: {}", exc)
            return None
    return None


def auto_check_camera_pixel_um(
    cam_type: str | None = None, cam_id: int | None = None
) -> None:
    """渲染期比对会话值与硬件探测值；不一致时给一行告警 + 「采用」按钮。

    **从不静默覆盖** —— 已有值一律由操作者决定是否采用（这就是「自动探测但
    允许人工修改」在像元间距上的具体形态）。差异判据复用三态阈值 0.5%。
    """
    configured = st.session_state.get(CAMERA_PIXEL_KEY)
    if configured is None:
        return
    if cam_type is None or cam_id is None:
        cam_type, cam_id = bench_camera()
    if "_cam_pixel_checked" not in st.session_state:
        detected = detect_camera_pixel_um(str(cam_type), int(cam_id))
        tolerance = CAMERA_PIXEL_REL_TOLERANCE
        mismatch = False
        if detected is not None:
            try:
                rel = abs(float(detected) - float(configured)) / abs(
                    float(detected) or 1.0
                )
            except (TypeError, ValueError, ZeroDivisionError):
                rel = float("inf")
            mismatch = rel > tolerance
        if detected is not None:
            st.session_state["_cam_pixel_detected"] = detected
        st.session_state["_cam_pixel_mismatch"] = bool(mismatch)
        st.session_state["_cam_pixel_checked"] = True

    if st.session_state.get("_cam_pixel_mismatch"):
        detected = st.session_state.get("_cam_pixel_detected")
        col1, col2 = st.columns([4, 1])
        with col1:
            st.warning(
                f"相机像元间距：会话值 {configured} μm ≠ 硬件探测值 {detected} μm"
            )
        with col2:
            if st.button("采用", key="adopt_camera_pixel_btn", type="primary"):
                if detected is not None:
                    queue_pending("", {CAMERA_PIXEL_KEY: float(detected)})
                    st.session_state.pop("_cam_pixel_checked", None)
                    st.session_state.pop("_cam_pixel_mismatch", None)
                    st.rerun()


# ---------------------------------------------------------------------------
# 标定（走 CLI 的同一个入口，不复制实现）
# ---------------------------------------------------------------------------


def _calibration_widget_values(prefix: str) -> dict[str, Any]:
    """标定所需的、可能只在主区渲染过的控件值（主区没展开时用默认值）。"""

    def _read(key: str, fallback: Any) -> Any:
        return st.session_state.get(key, fallback)

    from ao_shaping.gui.slm.shaping_controls import MIL_WIDGET_DEFAULTS

    return {
        "region": int(
            _read(f"{prefix}_shp_model-in-loop_region", MIL_WIDGET_DEFAULTS["region"])
        ),
        "far_field_padding": int(
            _read(
                f"{prefix}_shp_model-in-loop_far_field_padding",
                MIL_WIDGET_DEFAULTS["far_field_padding"],
            )
        ),
        "min_holdout_correlation": float(
            _read(f"{prefix}_shp_model-in-loop_min_holdout_correlation", 0.5)
        ),
        "probe_spread": float(
            _read(
                f"{prefix}_shp_model-in-loop_probe_spread",
                MIL_WIDGET_DEFAULTS["probe_spread"],
            )
        ),
        "probe_count": int(
            _read(
                f"{prefix}_shp_model-in-loop_probe_count",
                MIL_WIDGET_DEFAULTS["probe_count"],
            )
        ),
        "n_rounds": int(
            _read(
                f"{prefix}_shp_model-in-loop_n_rounds", MIL_WIDGET_DEFAULTS["n_rounds"]
            )
        ),
        "seed": int(MIL_WIDGET_DEFAULTS["seed"]),
    }


def build_calibration_params(prefix: str, slm_num: int) -> Any:
    """构造 ``calibrate_session`` 所需的 ``SlmModelInLoopParams``。

    只用**实验级 / 每 SLM 级**的光路设置 + 主区已渲染过的探针参数；标定本身
    不关心目标与评分，所以那些字段取默认值。
    """
    from ao_shaping.runners.runner_common import SlmModelInLoopParams

    cam_type, cam_id = bench_camera()
    w = _calibration_widget_values(prefix)
    geometry_file = _geometry_file_for_source(prefix)
    return SlmModelInLoopParams(
        n_rounds=w["n_rounds"],
        seed=w["seed"],
        target_shape="square",
        target_side=60,
        w_uniformity=0.4,
        w_efficiency=0.6,
        probe_count=w["probe_count"],
        probe_spread=w["probe_spread"],
        step_a_iterations=0,
        step_b_iterations=1,
        holdout_probe_count=2,
        freeform_holdout_count=0,
        min_holdout_correlation=w["min_holdout_correlation"],
        region=w["region"],
        panel_span_px=float(bench_value(prefix, "panel_span_px", 900.0)),
        pupil_center=bench_pupil_center(prefix),
        far_field_padding=w["far_field_padding"],
        geometry_file=geometry_file,
        cam_type=cam_type,
        cam_id=cam_id,
        # Unset session exposure used to reach the params as a bare 0.0, which the
        # exposure resolver now rejects; 'default' is the token that means
        # "don't change it", i.e. the same behaviour.
        exposure_time_ms=(
            float(st.session_state.get("bench_exposure_ms", 0.0) or 0.0)
            or DEFAULT_EXPOSURE_TOKEN
        ),
        max_exposure_ms=float(configured_daheng_exposure_limit()),
        cam_size=int(
            st.session_state.get("bench_cam_size", DEFAULT_WINDOW_SIDE_PX) or 0
        ),
        camera_pixel_um=float(
            st.session_state.get(CAMERA_PIXEL_KEY, DEFAULT_CAMERA_PIXEL_UM)
        ),
        slm_number=slm_num,
        blaze_period_px=int(st.session_state.get("bench_blaze_period_px", -1)),
        blaze_direction=str(st.session_state.get("bench_blaze_direction", "vertical")),
    )


def _geometry_file_for_source(prefix: str) -> str | None:
    """几何来源 = 复用时取路径，否则 ``None``（= 本次自动测量）。"""
    if bench_geometry_source(prefix) != "reuse":
        return None
    raw = str(
        st.session_state.get(_scoped(prefix, "bench_geometry_file"), "") or ""
    ).strip()
    return raw or None


def bench_geometry_source(prefix: str) -> str:
    """当前几何来源（非法值回落 ``auto``）。"""
    source = str(st.session_state.get(_scoped(prefix, "bench_geometry_source"), "auto"))
    return source if source in GEOMETRY_SOURCE_OPTIONS else "auto"


def reusable_geometry_file(prefix: str) -> str:
    """本次运行直接复用的 sweep 几何文件（运行路径不再自动标定）。

    解析顺序：几何来源「复用」时文本框里的路径（人工指定的那条）→ 本会话已标定
    的探测产物 → 空串（= 没有可复用的几何，由调用方拒绝运行）。
    """
    if bench_geometry_source(prefix) == "reuse":
        raw = str(
            st.session_state.get(_scoped(prefix, "bench_geometry_file"), "") or ""
        ).strip()
        if raw:
            return raw
    return str(detected_state(prefix).get("geometry_file", "") or "").strip()


def uncalibrated_note(prefix: str) -> str:
    """「还没标定」那一行（状态条与光路摘要共用同一句措辞）。

    It has to point at the sidebar 标定 button: the run path no longer measures, so
    that button is the single action that clears this state. The duration lives on
    that button's help **only** — repeating it here would be a second copy that
    drifts, and the length genuinely differs (the geometry step runs the
    illumination-radius boundary sweep only on some sessions).
    """
    source = bench_geometry_source(prefix)
    return (
        f"几何来源「{GEOMETRY_SOURCE_LABELS.get(source, source)}」"
        " —— 还没标定：先点本区块的「标定 … 光路」按钮（入射定位 → 曝光探测 → "
        "sweep 几何）。闭环整形直接复用这次结果，不再在运行时重测。"
    )


def run_session_calibration(prefix: str, slm_num: int) -> str | None:
    """跑一次会话标定：入射定位 → sweep 几何；返回错误文本，成功返回 ``None``。

    走 :func:`ao_shaping.runners.slm.model_in_loop_session.calibrate_session`
    —— 与 CLI ``slm-model-in-loop`` **同一个入口**，所以 GUI 不会和 CLI 漂移。
    该函数会自己 open SLM + CCD（句柄不可重入），因此经
    :func:`ao_shaping.gui.slm.shaping_controls.with_slm_ownership` 交出句柄，
    标定按钮与整形运行按钮共用这一条交接路径。
    """
    try:
        from ao_shaping.runners.slm.model_in_loop_session import (
            apply_session_calibration,
            calibrate_session,
        )
    except ImportError as exc:  # pragma: no cover - 另一条 lane 的模块缺失
        message = (
            "会话标定入口 "
            "ao_shaping.runners.slm.model_in_loop_session 不可用，无法自动标定："
            f"{exc}"
        )
        logger.error(message)
        st.error(message)
        return message

    from ao_shaping.utils.io.file import debug_run_dir

    # Deferred: shaping_controls imports this module at module level, so the
    # reverse edge can only be resolved inside the function body.
    from ao_shaping.gui.slm.shaping_controls import with_slm_ownership

    source = bench_geometry_source(prefix)
    locate = source in ("auto", "keep")
    geometry = source == "auto"

    save_dir = debug_run_dir(Path("data"), "slm_bench_calibration")
    try:
        params = build_calibration_params(prefix, slm_num)
        from ao_shaping.runners.runner_common import build_model_in_loop_config

        config = build_model_in_loop_config(params, debug=False)
    except Exception as exc:  # policy: 标定参数不可用是操作者可修正的输入问题：界面报错后返回消息，不抛到重跑
        message = f"标定参数不可用: {exc}"
        logger.exception("标定参数不可用: {}", exc)
        st.error(message)
        return message

    try:
        with (
            # 不带耗时: 全 GUI 唯一的耗时口径在按钮 help 里（几何步带不带照明半径
            # 边界扫描差 3 倍），这里再写一个数就是第三份会漂的副本。
            st.spinner("正在标定，请勿触碰设备（入射定位 → 曝光探测 → sweep 几何）"),
            with_slm_ownership(slm_num),
        ):
            calibration = calibrate_session(
                params, config, save_dir, locate=locate, geometry=geometry
            )
            # The canonical merge: the GUI must not re-implement "which value
            # wins".  ``apply_session_calibration`` writes only the fields that
            # were actually measured, so a skipped step keeps the caller's value
            # rather than inventing a detection.
            resolved_params, resolved_config = apply_session_calibration(
                params, config, calibration
            )
    except Exception as exc:  # policy: calibrate_session 抛普通异常：界面报失败并返回原因字符串，不改变已下发的相位
        logger.exception("SLM {} 光路标定失败: {}", slm_num, exc)
        st.error(f"SLM {slm_num} 光路标定失败: {exc}")
        return str(exc)

    # 三态判定只有 publish_session_calibration 一处实现：探测值、实验级曝光与
    # 「跳过人工字段」的 widget 回填都在那里，运行路径共用同一份。
    publish_session_calibration(
        prefix, resolved_config, calibration, source=source, save_dir=save_dir
    )
    sync_manual_states(prefix)
    for line in calibration.summary_lines():
        st.caption(line)
    if manual_fields(prefix):
        st.warning(
            "探测到新光路参数，但以下字段仍使用人工值："
            + "；".join(field_hint(prefix, name) for name in manual_fields(prefix))
        )
    return None


def publish_session_calibration(
    prefix: str,
    resolved_config: Any,
    calibration: Any,
    *,
    source: str,
    save_dir: Any,
) -> dict[str, Any]:
    """把一次会话标定的结果写进三态表，并排好下一次 flush 的 widget 回填。

    三态的判定只有这一处实现：侧栏「标定」按钮
    :func:`run_session_calibration` 是**唯一**的测量路径（运行路径不再标定），
    所以「什么算探测值 / 什么算人工值」不可能分歧。

    回填完把几何来源钉到「复用已标定几何」：标定按钮本身就是一次性动作，之后的
    运行直接复用这次结果。
    """
    exposure_ms = float(st.session_state.get("bench_exposure_ms", 0.0) or 0.0)
    detected: dict[str, Any] = {
        "calibrated_at": _now_text(),
        "geometry_source": source,
        "evidence_dir": str(getattr(calibration, "evidence_dir", save_dir)),
        "pupil_center": list(resolved_config.pupil_center),
        "geometry_file": str(resolved_config.geometry_file or ""),
    }
    correlation = getattr(calibration, "geometry_correlation", None)
    if correlation is not None:
        detected["geometry_correlation"] = float(correlation)
    for skipped in ("pupil_center", "geometry_file"):
        if getattr(calibration, skipped) is None:
            detected.pop(skipped, None)
    record_detection(prefix, detected)
    record_detection("", {"exposure_ms": exposure_ms})
    # 重新标定**不覆盖手动值**：只给没有被人工改过的字段排队回填。
    backfill = _detected_backfill(prefix, resolved_config)
    queue_pending(prefix, backfill)
    return backfill


def _detected_backfill(prefix: str, resolved_config: Any = None) -> dict[str, Any]:
    """探测值 → widget key 的回填表（跳过人工字段）。

    ``resolved_config`` is the object
    :func:`ao_shaping.runners.slm.model_in_loop_session.apply_session_calibration`
    produced, so the value written into the widget is the one the next run will
    actually consume rather than a re-derived copy.
    """
    detected = detected_state(prefix)
    manual = manual_state(prefix)
    fields = bench_fields(prefix)
    backfill: dict[str, Any] = {}
    for name in ("pupil_center", "panel_span_px", "geometry_file"):
        if name not in detected or name in manual:
            continue
        field = fields[name]
        value = detected[name]
        if name == "pupil_center" and resolved_config is not None:
            center = getattr(resolved_config, "pupil_center", None)
            if center is not None:
                value = (int(center[0]), int(center[1]))
        backfill[field.widget_key] = (
            f"{int(round(value[0]))},{int(round(value[1]))}"
            if field.kind == "center"
            else value
        )
    if backfill.get(fields["geometry_file"].widget_key):
        backfill[_scoped(prefix, "bench_geometry_source")] = "reuse"
    return backfill


def _now_text() -> str:
    from ao_shaping.utils.io.file import gen_date_str

    try:
        return str(gen_date_str())
    except Exception:  # policy: 时间戳只是文件名片段：拿不到就返回空串，不得让保存流程失败
        return ""


# ---------------------------------------------------------------------------
# 光路摘要卡（折叠态也要看得到）
# ---------------------------------------------------------------------------


def is_calibrated(prefix: str) -> bool:
    """本会话是否已有可用光路测量（入射中心或 sweep 几何任一即可）。

    ``bench_summary_lines`` 与折叠标题 ``calibration_marker`` 共用这一个判据:
    两处各写一份会让「摘要说已标定、标题说未标定」的漂移无人发现。
    """
    detected = detected_state(prefix)
    return bool(detected.get("pupil_center") or detected.get("geometry_file"))


def calibration_marker(prefix: str) -> str:
    """折叠标题用的标定状态短语（``● 已标定`` / ``○ 未标定``）。

    放在标题里而不是只在块内: 折叠状态下也要能看出这台 SLM 能不能直接整形。
    """
    if is_calibrated(prefix):
        return f"{MARK_DETECTED} 已标定"
    return f"{MARK_UNSET} 未标定"


def bench_summary_lines(prefix: str) -> tuple[str, ...]:
    """光路摘要（``●`` 未标定 / 已标定 + 手动覆盖项）。"""
    detected = detected_state(prefix)
    manual = manual_fields(prefix)
    lines: list[str] = []
    if not is_calibrated(prefix):
        lines.append(f"{MARK_UNSET} 未标定 —— {uncalibrated_note(prefix)}")
    else:
        center = _coerce_point(detected.get("pupil_center"))
        parts = [f"{MARK_DETECTED} 已标定 {detected.get('calibrated_at') or '-'}"]
        parts.append(
            "入射中心 " + (f"({center[0]:g}, {center[1]:g})" if center else "未定位")
        )
        if detected.get("geometry_correlation") is not None:
            parts.append(f"相关度 {float(detected['geometry_correlation']):.4f}")
        parts.append(
            "曝光 "
            + format_value(
                _FIELD_CATALOG["exposure_ms"], detected.get("exposure_ms", 0.0)
            )
        )
        if detected.get("geometry_file"):
            parts.append(f"几何 {detected['geometry_file']}")
        lines.append(" · ".join(parts))
    lines.extend(field_hint(prefix, name) for name in manual)
    return tuple(lines)


# ---------------------------------------------------------------------------
# 主区取值：把侧栏光路设置并进方法参数表
# ---------------------------------------------------------------------------


def merge_bench_values(
    values: dict[str, Any], prefix: str, slm_num: int, method_key: str
) -> dict[str, Any]:
    """把侧栏光路设置并入方法的 values 表（就地修改并返回）。

    ``build_gs_refine_params`` / ``build_model_in_loop_params`` 的映射与语义
    因此**完全不变** —— 它们仍然只读 ``values``，只是 ``values`` 现在带上了
    实验级与每 SLM 级的光路字段。
    """
    crop = bool(st.session_state.get("bench_cam_crop", True))
    cam_type, cam_id = bench_camera()
    values["cam_type"] = cam_type
    values["cam_id"] = cam_id
    values["cam_crop_enabled"] = crop
    # 裁窗关闭 ⇒ ``0``，即整帧。runner 用 ``0`` 表示「不重置窗口」，所以这里
    # 不能退回一个边长 —— 退回边长会在关闭裁窗时反而开出一个 256 px 的窗口。
    values["cam_size"] = (
        int(st.session_state.get("bench_cam_size", 0) or 0) if crop else 0
    )
    values["exposure_time_ms"] = (
        float(st.session_state.get("bench_exposure_ms", 0.0) or 0.0)
        or DEFAULT_EXPOSURE_TOKEN
    )
    values["camera_pixel_um"] = float(
        st.session_state.get(CAMERA_PIXEL_KEY, DEFAULT_CAMERA_PIXEL_UM)
    )
    values["blaze_period_px"] = int(st.session_state.get("bench_blaze_period_px", -1))
    values["blaze_direction"] = str(
        st.session_state.get("bench_blaze_direction", "vertical")
    )
    values["max_exposure_ms"] = float(configured_daheng_exposure_limit())

    if method_key == "gs-refine":
        span = float(bench_value(prefix, "panel_span_px", 2 * 450.0))
        values["panel_span_px"] = span
        values["beam_radius_px"] = span / 2.0
    else:
        values["panel_span_px"] = float(bench_value(prefix, "panel_span_px", 900.0))
        # 运行路径不测量：直接把侧栏已标定的 sweep 几何交出去（人工路径优先，
        # 其次本会话的探测产物）。空串 = 没有可复用的几何，validate 会拒绝运行。
        values["geometry_file"] = reusable_geometry_file(prefix)
    values["pupil_center"] = bench_pupil_center(prefix)
    values["bench_geometry_source"] = bench_geometry_source(prefix)
    values["bench_prefix"] = prefix
    values["bench_slm_number"] = slm_num
    return values


# ---------------------------------------------------------------------------
# 侧栏渲染
# ---------------------------------------------------------------------------


def render_bench_settings() -> None:
    """侧栏「CCD参数设置」区块入口。

    顺序是硬约束：先 flush pending、再重算人工覆盖表，最后才实例化控件。
    调用方（``render_slm_sidebar``）只需在侧栏早期调用一次。
    """
    flush_pending("")
    sync_manual_states("")
    _render_experiment_block()

    for slm_num in (1, 2):
        prefix = f"slm{slm_num}"
        if not (
            st.session_state.get(f"{prefix}_connected") and st.session_state.get(prefix)
        ):
            continue
        flush_pending(prefix)
        sync_manual_states(prefix)
        _render_slm_block(prefix, slm_num)


def _render_camera_block(disabled: bool) -> str:
    """A 相机：类型 / ID / 像元间距。返回选中的相机类型。"""
    st.caption("A 相机")
    c1, c2 = st.columns(2)
    with c1:
        cam_type = st.selectbox(
            "相机类型",
            options=list(CAM_TYPE_OPTIONS),
            index=list(CAM_TYPE_OPTIONS).index(
                str(st.session_state.get("bench_cam_type", "daheng"))
            )
            if str(st.session_state.get("bench_cam_type", "daheng")) in CAM_TYPE_OPTIONS
            else 0,
            key="bench_cam_type",
            disabled=disabled,
            help="sim = 2f-Fourier 数字孪生（无硬件自检）",
        )
    with c2:
        st.number_input(
            "相机 ID",
            min_value=0,
            max_value=8,
            value=widget_default("bench_cam_id", 0),
            step=1,
            key="bench_cam_id",
            disabled=disabled,
        )
    auto_check_camera_pixel_um(
        cam_type, int(st.session_state.get("bench_cam_id", 0) or 0)
    )

    p1, p2 = st.columns([3, 1])
    with p1:
        st.number_input(
            "像元间距 (μm)",
            min_value=0.1,
            max_value=50.0,
            step=0.1,
            value=widget_default(CAMERA_PIXEL_KEY, DEFAULT_CAMERA_PIXEL_UM),
            key=CAMERA_PIXEL_KEY,
            disabled=disabled,
            help="CCD 像元间距 (μm)。点「探测」尝试从硬件读取；"
            "参考：大恒 MER2-507 = 2.2 μm",
        )
    with p2:
        st.markdown("")
        if st.button(
            "探测",
            key="detect_camera_pixel_btn",
            type="secondary",
            disabled=disabled,
            use_container_width=True,
        ):
            probed = detect_camera_pixel_um(
                cam_type, int(st.session_state.get("bench_cam_id", 0) or 0)
            )
            if probed is not None:
                queue_pending("", {CAMERA_PIXEL_KEY: float(probed)})
                st.session_state.pop("_cam_pixel_checked", None)
                st.session_state.pop("_cam_pixel_mismatch", None)
                st.success(f"探测到像元间距: {probed} μm")
                st.rerun()
            else:
                st.warning("无法探测像元间距（相机未连接或 SDK 不可用），保持当前值")
    return cam_type


def _render_exposure_block(disabled: bool, cam_type: str) -> None:
    """A 组续：会话曝光（可实测）+ 饱和上限提示。"""
    e1, e2 = st.columns([3, 1])
    with e1:
        exposure = st.number_input(
            "会话曝光 (ms)",
            min_value=0.0,
            value=widget_default("bench_exposure_ms", 0.0),
            step=0.01,
            format="%.3f",
            key="bench_exposure_ms",
            disabled=disabled,
            help="0 = 自动起曝。**填 0 时大恒驱动会钳到约 0.02 ms，比可用区间暗 "
            "20–75×**；实验前必须按当前激光功率实测曝光。点右侧「探测」按当前"
            "光路实测：倍增阶梯找饱和上限，取其一半作工作曝光。上限取 "
            "AO_DAHENG_MAX_EXPOSURE_MS（本光路饱和保护），缺省/0 时由画面探测。",
        )
    with e2:
        st.markdown("")
        if st.button(
            "探测",
            key="detect_exposure_btn",
            type="secondary",
            disabled=disabled or cam_type != "daheng",
            use_container_width=True,
            help="按当前光路实测曝光：倍增阶梯找饱和上限 → 取一半作工作曝光。"
            "约几秒；换激光功率或改光路后需要重测。",
        ):
            try:
                probed_ms, evidence = probe_session_exposure_ms(
                    cam_type, int(st.session_state.get("bench_cam_id", 0) or 0)
                )
            except ExposureProbeFailure as exc:
                st.error(str(exc))
            else:
                queue_pending("", {"bench_exposure_ms": float(probed_ms)})
                record_detection("", {"exposure_ms": float(probed_ms)})
                source = (
                    "环境变量" if evidence["ceiling_source"] == "env" else "画面探测"
                )
                st.success(
                    f"探测到会话曝光 {probed_ms:.3f} ms"
                    f"（饱和上限 {evidence['ceiling_ms']:.3f} ms，来源：{source}；"
                    f"工作曝光取其 {int(evidence['work_fraction'] * 100)}%）"
                )
                st.caption(
                    f"确认帧峰值 {evidence['peak']:.0f}/{evidence['full_scale']:.0f}"
                    f"（{evidence['peak_fraction']:.0%}）；"
                    f"最后未饱和档 {evidence['last_safe_ms']:.3f} ms；"
                    f"证据 {EXPOSURE_PROBE_DEBUG_DIR}"
                )
                st.rerun()
    ceiling = configured_daheng_exposure_limit()
    if cam_type == "daheng" and ceiling > 0 and float(exposure) > ceiling:
        st.warning(
            f"曝光 {exposure} ms 超过本光路饱和保护上限 {ceiling} ms，运行前会被拦下"
        )
    if cam_type == "daheng" and float(exposure) == 0:
        st.warning("曝光 = 0 → 大恒钳到约 0.02 ms（比可用区间暗 20–75×）")


def _render_window_block(disabled: bool) -> None:
    """B 读出窗口：开窗开关 / 边长 / 窗口上限探测。"""
    st.caption("B 读出窗口")
    bounds = window_bounds()
    crop = st.checkbox(
        "启用开窗",
        value=widget_default("bench_cam_crop", True),
        key="bench_cam_crop",
        disabled=disabled,
        help="关闭则整幅读出；开启后窗口边长必须是设备步长的整数倍",
    )
    st.number_input(
        "窗口边长 (px)",
        min_value=bounds.min_side,
        max_value=bounds.max_side,
        value=widget_default(
            "bench_cam_size",
            bounds.clamp(DEFAULT_WINDOW_SIDE_PX),
        ),
        step=bounds.step,
        key="bench_cam_size",
        disabled=disabled or not crop,
        help=f"正方形 ROI 边长。{bounds.detail}",
    )
    st.caption(bounds.detail)
    if st.button(
        "探测窗口上限",
        key="bench_probe_window_btn",
        type="secondary",
        disabled=disabled,
    ):
        window_bounds(probe=True)
        st.rerun()


def render_blaze_controls(prefix: str, disabled: bool) -> None:
    """载波周期 / 光栅方向控件。

    从实验光路区块搬出：载波现在是一层**发送叠加**（见
    :func:`build_carrier_overlay`），因此与底相位/离焦一起渲染在侧栏
    「相位叠加」区块里。键名不变，仍写 ``bench_blaze_period_px`` /
    ``bench_blaze_direction``。
    """
    d1, d2 = st.columns(2)
    with d1:
        st.selectbox(
            "载波周期",
            options=list(BLAZE_PERIOD_OPTIONS),
            format_func=lambda p: BLAZE_PERIOD_LABELS[p],
            key="bench_blaze_period_px",
            disabled=disabled,
            help="离轴载波把 0 级移出读出窗；-1 = 关闭，0 = 实测自动选择（需相机反馈）",
        )
    with d2:
        st.selectbox(
            "光栅方向",
            options=list(BLAZE_DIRECTIONS),
            key="bench_blaze_direction",
            disabled=disabled,
        )
    if int(st.session_state.get("bench_blaze_period_px", -1)) == 0:
        st.warning(
            "周期 = 0 是「实测自动」，需要相机反馈；发送叠加层这里没有画面可依，"
            "所以不会叠加。请选一个具体周期。"
        )


def _render_summary_block(detected: dict[str, Any], manual: list[str]) -> None:
    """光路摘要卡 + 各字段状态标注。"""
    st.divider()
    st.markdown("**光路摘要**")
    for line in bench_summary_lines(""):
        if line.startswith(MARK_MANUAL):
            st.warning(line)
        else:
            st.caption(line)
    for name in _EXPERIMENT_FIELD_NAMES:
        state = field_state("", name)
        if state == "detected":
            st.caption(field_hint("", name))
    if manual:
        st.warning(f"实验级有 {len(manual)} 项手动覆盖：{'、'.join(manual)}")
    if (
        detected.get("exposure_ms") is not None
        and field_state("", "exposure_ms") == "detected"
    ):
        st.caption("上表曝光为标定时在用的会话曝光（sweep 内部还会按画面自动起曝）")


def _render_experiment_block() -> None:
    """实验级共享的光路设置（相机 / 读出窗口 / 载波）+ 光路摘要卡。"""
    busy = bench_busy()
    disabled = busy
    detected = detected_state("")
    manual = manual_fields("")

    with st.expander("CCD参数设置", expanded=False):
        if busy:
            st.info(
                "标定或闭环整形进行中 —— SLM 句柄已交给标定器 / 优化器"
                "（SDK 不可重入），此时显示未连接属正常。"
            )

        cam_type = _render_camera_block(disabled)
        _render_exposure_block(disabled, cam_type)
        _render_window_block(disabled)
        # 载波控件已并入侧栏「相位叠加」区块（与底相位/离焦同处、可勾选）：
        # 它现在是一层**发送叠加**，而不是实验光路的一个固定参数。
        _render_summary_block(detected, manual)


def _render_slm_geometry_block(
    prefix: str, slm_num: int, disabled: bool, cam_type: str
) -> None:
    """D 面板与瞳面：入射中心 / 覆盖宽度 / 实测照明半径。"""
    st.caption("D 面板与瞳面")
    st.text_input(
        "入射中心 x,y (面板px)",
        value=widget_default(_scoped(prefix, "bench_pupil_center"), ""),
        key=_scoped(prefix, "bench_pupil_center"),
        placeholder="留空 = 自动定位",
        disabled=disabled,
        help="入射光斑在 SLM 面板上的坐标 (x, y)。留空 = 点「标定 … 光路」时由 "
        "calibrate_session 的全板随机相位盘扫描定位；运行路径不测量。",
    )
    st.number_input(
        "模型覆盖宽度 (面板px)",
        min_value=32,
        max_value=int(PANEL_RES[0]),
        value=widget_default(
            _scoped(prefix, "bench_panel_span_px"),
            bounds_clamp_panel_span(prefix, 900.0),
        ),
        step=10,
        key=_scoped(prefix, "bench_panel_span_px"),
        disabled=disabled,
        help="= 2×照明半径（GS 的 beam_radius_px×2 落在同一字段）。"
        "用 SLM 像素直接采样会把孔径算小 ~3.5 倍。",
    )
    # 两个不同的物理量并排显示，避免只看到 7.8 被当成「待修的误差」
    # 改成间距（见 slm200_constants 注释与 GS_REFINE_RECIPE 的不变量）。
    st.caption(
        f"SLM 面板 {PANEL_RES[0]}×{PANEL_RES[1]}，像素间距 {PITCH_UM} μm · "
        f"有效口径 {PIXEL_SIZE_UM} μm（相差 0.2 μm 为真实器件几何）"
    )
    r1, r2 = st.columns([3, 1])
    with r1:
        st.number_input(
            "实测照明半径 (面板px)",
            min_value=0.0,
            value=widget_default(_scoped(prefix, "bench_beam_radius_px"), 0.0),
            step=10.0,
            format="%.0f",
            key=_scoped(prefix, "bench_beam_radius_px"),
            disabled=disabled,
            help="面板上的**实测**照明半径 = 远边界 − 已测入射中心（两轴取较小）。"
            "0 = 未实测。这是 `model_in_loop_hw_runbook --stage fit` 传给 "
            "`aperture_m` 的那个量 —— 它原先取自 sweep 用的 Zernike 半径"
            "（默认 200 px），而两条焦面尺度推断相差 152% 的根因就是"
            "这个口径未经实测。填上它等于给物理交叉校验一个真实的孔径。",
        )
    with r2:
        st.markdown("")
        if st.button(
            "探测半径",
            key=_scoped(prefix, "detect_beam_radius_btn"),
            type="secondary",
            disabled=disabled,
            use_container_width=True,
            help="半平面随机相位边界扫描，两轴各一次（约 5–10 分钟，"
            "每档都要重写全板相位）。产出的远边界见下方 caption；"
            "半径取两轴较小值。",
        ):
            st.session_state[_scoped(prefix, "bench_beam_radius_probe")] = (
                measure_illuminated_radius(
                    slm_num,
                    cam_type,
                    int(st.session_state.get("bench_cam_id", 0) or 0),
                    exposure_ms=(
                        float(st.session_state.get("bench_exposure_ms", 0.0) or 0.0)
                        or None
                    ),
                )
            )
            st.rerun()
    probe = st.session_state.get(_scoped(prefix, "bench_beam_radius_probe"))
    if probe:
        if probe.get("error"):
            st.error(f"半径探测失败: {probe['error']}")
        else:
            unc = probe.get("radius_uncertainty_px", 0.0)
            extra = ""
            if probe.get("radius_from_centre_px") is not None:
                extra = (
                    f"；按入射中心减远边界则为 "
                    f"{probe['radius_from_centre_px']:.0f}（中心 "
                    f"{probe.get('centre_panel_px')}）"
                )
            st.caption(
                f"实测照明半径 ≈ **{probe['radius_px']:.0f} ±{unc:.0f} 面板px**"
                f"（扫描自身落差宽度；x {probe.get('scan_radius_x', float('nan')):.0f}"
                f" / y {probe.get('scan_radius_y', float('nan')):.0f}，"
                f"扫描中心 {probe.get('scan_centre_panel_px')}）"
                f"{extra}；平场峰值 {probe['flat_peak']:.0f}/255，"
                f"曝光 {probe['exposure_ms']:.3f} ms，"
                f"证据 {BEAM_RADIUS_DEBUG_DIR}"
            )


def _run_focus_probe(prefix: str, slm_num: int, cam_type: str) -> None:
    """跑一次焦面扫描并把证据写进会话状态。"""
    # 句柄在 GUI 手里，SDK 不可重入：先交出去，探测自己开设备，
    # 结束后自动收回（与自动标定同一条路径）。
    from ao_shaping.gui.slm.shaping_controls import with_slm_ownership
    from ao_shaping.tools.slm.panel.focus_probe import (
        probe_session_focus,
    )

    centre = _centre_from_values(st.session_state, slm_num)
    radius = float(
        st.session_state.get(_scoped(prefix, "bench_beam_radius_px"), 0.0) or 0.0
    )
    exposure = float(st.session_state.get("bench_exposure_ms", 0.0) or 0.0)
    with (
        st.spinner("正在扫描菲涅尔焦距（每个点都会自动重调曝光）…"),
        with_slm_ownership(slm_num),
    ):
        evidence = probe_session_focus(
            slm_num,
            cam_type,
            int(st.session_state.get("bench_cam_id", 0) or 0),
            exposure_ms=exposure or None,
            radius_px=radius or None,
            centre_px=centre,
        )
    st.session_state[_scoped(prefix, "bench_focus_probe")] = evidence


def _render_slm_focus_block(
    prefix: str, slm_num: int, disabled: bool, cam_type: str
) -> None:
    """D2 焦面：探测按钮 + 补偿量说明。"""
    st.caption("D2 焦面（相机离开 2f 后必须补偿）")
    f1, f2 = st.columns([3, 1])
    with f2:
        st.markdown("")
        if st.button(
            "探测焦面",
            key=_scoped(prefix, "detect_focus_btn"),
            type="secondary",
            disabled=disabled,
            use_container_width=True,
            help="在 **plain 相位** 上扫描菲涅尔焦距（1/f，±6 m⁻¹），"
            "找让光斑最亮的那个；每个点的曝光都自动重调，"
            "否则近焦面会饱和、远焦面只剩读出噪声。约 1–2 分钟。",
        ):
            _run_focus_probe(prefix, slm_num, cam_type)
            st.rerun()
    with f1:
        st.caption(
            "相机挪到 2f 焦面之后，plain 相位的光斑会随离焦量线性涨大；"
            "补偿量来自实测，不从标称 125 mm 推断。"
        )
    focus = st.session_state.get(_scoped(prefix, "bench_focus_probe"))
    if focus:
        if focus.get("error"):
            st.error(f"焦面探测失败: {focus['error']}")
        elif focus.get("best_focal_mm") is None:
            if focus.get("path_blocked"):
                # 光路受阻与「没找到焦面」是两件事: 前者要查三段光路, 后者要
                # 重扫焦距。混成同一句话正是 2026-10-21 那次误判的根源 ——
                # 操作者只看到「没找到可用点」, 不知道光根本没到相机。
                st.error(f"焦面探测：**光路受阻** —— {focus.get('blockage_reason')}")
            else:
                st.warning(
                    "焦面探测没有找到可用点（所有点的曝光都落在可测带之外）；"
                    f"证据 {FOCUS_PROBE_DEBUG_DIR}"
                )
        else:
            gain = focus.get("gain_vs_flat") or 0.0
            lin = focus.get("linearity_ratio")
            lin_text = "" if lin is None else f"，线性度 {lin:.2f}"
            st.caption(
                f"焦面补偿 ≈ **1/f = {focus['best_diopter_m1']:+.3f} m⁻¹**"
                f"（f = {focus['best_focal_mm']:.0f} mm），"
                f"相对 plain 峰值/曝光 ×{gain:.2f}{lin_text}；"
                f"扫描范围 {focus['range_m1'][0]:+.1f} … "
                f"{focus['range_m1'][1]:+.1f} m⁻¹，"
                f"证据 {FOCUS_PROBE_DEBUG_DIR}"
            )
            for note in focus.get("notes", []):
                st.caption(f"· {note}")


def _render_slm_calibration_block(
    prefix: str, slm_num: int, disabled: bool, detected: dict[str, Any]
) -> None:
    """E 标定：几何来源 / 标定按钮 / 手动覆盖提示 / 证据。"""
    st.caption("E 标定")
    source = st.selectbox(
        "几何来源",
        options=list(GEOMETRY_SOURCE_OPTIONS),
        format_func=lambda s: GEOMETRY_SOURCE_LABELS[s],
        key=_scoped(prefix, "bench_geometry_source"),
        disabled=disabled,
        help="自动标定 = 点「标定」按钮时测量入射中心 + sweep 几何；"
        "沿用上次几何 = 只重测中心。运行路径不测量，直接复用这里的结果。",
    )
    if source == "reuse":
        st.text_input(
            "几何文件路径 (bench_geometry.json)",
            value=widget_default(_scoped(prefix, "bench_geometry_file"), ""),
            key=_scoped(prefix, "bench_geometry_file"),
            disabled=disabled,
            help="scripts/slm/model_in_loop_hw_runbook.py --stage sweep/calibrate 的产物",
        )
        st.caption(_geometry_help_text(prefix))

    if st.button(
        f"标定 SLM {slm_num} 光路",
        key=_scoped(prefix, "bench_calibrate_btn"),
        type="primary",
        use_container_width=True,
        # 复用模式下 calibrate_session 两步都是 locate=False/geometry=False,
        # 点下去只会刷新一个 calibrated_at 时间戳, 却让人以为重新量过了。
        disabled=disabled or source == "reuse",
        help=(
            "复用模式下不测量：请改「几何来源」为「自动标定」"
            if source == "reuse"
            # 全块唯一的耗时口径：几何步在会话没带几何文件、非 sim、曝光 > 0 时
            # 还会跑一次照明半径边界扫描（_should_measure_radius），那一步才是大头。
            else "全板入射定位 → 曝光探测 → sweep 几何"
            "（3–8 分钟；几何步不做照明半径边界扫描时 1–3 分钟）"
        ),
    ):
        error = run_session_calibration(prefix, slm_num)
        if error is None:
            st.rerun()

    manual = manual_fields(prefix)
    if manual:
        if st.button(
            "恢复探测值",
            key=_scoped(prefix, "bench_restore_btn"),
            type="secondary",
            use_container_width=True,
            disabled=disabled,
        ):
            queue_pending(prefix, _detected_backfill_all(prefix))
            reset_manual(prefix)
            st.rerun()
        st.warning("以下字段已人工修改，标定不会覆盖：" + "、".join(manual))
    for name in _SLM_FIELD_NAMES:
        state = field_state(prefix, name)
        if state == "detected":
            st.caption(field_hint(prefix, name))
    if detected.get("geometry_correlation") is not None:
        st.caption(
            f"几何相关度 {float(detected['geometry_correlation']):.4f}"
            f"（< 0.50 直接中止，不提交相位）"
        )
    if detected.get("evidence_dir"):
        st.caption(f"证据目录: {detected['evidence_dir']}")
    st.caption(
        "标定 = 入射定位 → 曝光探测 → sweep 几何；相关度 < 0.50 直接中止，"
        "不提交相位。只在换光路 / 换 SLM / 换相机后需要重标。"
    )


def _render_slm_block(prefix: str, slm_num: int) -> None:
    """每台已连接 SLM 的唯一标定入口：光路摘要 + 几何/离焦 + 标定。

    2026-10-24 合并：原先这里只有 D（几何/离焦）与 E（标定）两组，光路摘要与
    第二个主按钮在 multi_slm_controller 的独立折叠块里。两个按钮调的是同一个
    ``run_session_calibration``，差别只在摘要卡，所以独立块是纯重复入口（且在
    「复用」来源下不禁用，点下去只刷新时间戳）。现在摘要卡搬到本块顶部，全块
    只有一个「标定 SLM N 光路」。
    """
    busy = bench_busy()
    disabled = busy
    detected = detected_state(prefix)
    # The camera is an experiment-level choice (block A), but the radius probe
    # lives per-SLM, so read the shared camera back here rather than reaching
    # across blocks.
    cam_type, cam_id = bench_camera()

    with st.expander(
        f"SLM {slm_num} · 光路几何与标定 · {calibration_marker(prefix)}",
        expanded=False,
    ):
        # 光路摘要（未标定时是「○ 未标定 —— …」，已标定时是 ○/● + 实测事实）。
        # 它总至少返回一行，所以不需要空分支。
        for line in bench_summary_lines(prefix):
            st.caption(line)

        if busy:
            st.info(
                f"闭环整形/标定进行中 —— SLM {slm_num} 句柄已交给优化器"
                "（SDK 不可重入），此时显示未连接属正常。"
            )

        _render_slm_geometry_block(prefix, slm_num, disabled, cam_type)
        _render_slm_focus_block(prefix, slm_num, disabled, cam_type)
        _render_slm_calibration_block(prefix, slm_num, disabled, detected)


def bounds_clamp_panel_span(prefix: str, value: float) -> int:
    """把模型覆盖宽度夹到 [32, PANEL_RES[0]] 并落在 10 px 步长上。

    Snap first, then clamp: snapping alone can round *past* the panel edge
    (99999 → 1922 with a 1920-wide panel), and a widget value above ``max_value``
    is a Streamlit error, not a clamp.
    """
    clamped = max(32, min(int(round(value)), int(PANEL_RES[0])))
    snapped = 32 + max(0, round((clamped - 32) / 10)) * 10
    return int(min(snapped, int(PANEL_RES[0])))


def _detected_backfill_all(prefix: str) -> dict[str, Any]:
    """「恢复探测值」用：无视人工表，全部字段回到探测值。"""
    detected = detected_state(prefix)
    fields = bench_fields(prefix)
    backfill: dict[str, Any] = {}
    for name in _SLM_FIELD_NAMES:
        if name not in detected:
            continue
        field = fields[name]
        value = detected[name]
        backfill[field.widget_key] = (
            f"{int(round(value[0]))},{int(round(value[1]))}"
            if field.kind == "center"
            else value
        )
    return backfill


def _geometry_help_text(prefix: str) -> str:
    """几何文件路径的 caption（建议值 + 是否与 ``region`` 匹配）。"""
    from ao_shaping.runners.runner_common import (
        describe_bench_geometry,
        discover_model_in_loop_geometry,
    )

    region = _calibration_widget_values(prefix)["region"]
    raw = str(
        st.session_state.get(_scoped(prefix, "bench_geometry_file"), "") or ""
    ).strip()
    if not raw:
        suggestion = discover_model_in_loop_geometry(int(region))
        head = (
            f"建议值（region={region} 匹配的最新 sweep 产物）：{suggestion}"
            if suggestion is not None
            else f"建议值：data/ 下没有 region={region} 的 sweep 产物，请先跑一次自动标定"
        )
        return f"{head}\n{describe_bench_geometry(None)}"
    path = Path(raw)
    if not path.is_file():
        return f"⚠ 几何文件不存在: {raw}\n{describe_bench_geometry(raw)}"
    mismatch = geometry_region_mismatch(raw, int(region))
    text = describe_bench_geometry(raw)
    if mismatch is not None:
        return f"⚠ 几何文件 region={mismatch} ≠ 当前 region={region}（模型网格本身不同）\n{text}"
    return text


def geometry_region_mismatch(path: str, region: int) -> int | None:
    """几何文件的 ``region`` 与当前值不符时返回文件里的值，否则 ``None``。"""
    import json

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        found = int(payload["region"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return found if found != int(region) else None
