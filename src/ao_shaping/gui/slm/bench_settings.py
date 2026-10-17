"""本次实验光路：侧栏区块 + 「自动探测但允许人工修改」的 session 契约。

为什么单独一个模块
------------------
**一次实验 = 一条光路**。相机型号 / ID、读出窗口、曝光、去零级载波，以及每台
SLM 的入射位置与几何，都在一次实验内固定。把它们留在主区的整形参数里，
就是「每次都要重填一屏数字，换一条光路就直接失效」的根因 —— 尤其
``slm-model-in-loop`` 的 CLI 每次会话本来会自动做
**全板随机相位盘扫描定位入射中心 → sweep 几何标定**，GUI 却绕过这条流程，
只能让操作员手填 ``pupil_center="960,690"``、``panel_span_px=900``。

本模块只管两件事：

1. :func:`render_bench_settings` —— 侧栏「本次实验光路」区块（实验级 + 每 SLM 级）；
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

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import streamlit as st
from loguru import logger

from ao_shaping.drivers.ccd.common import WindowSize
from ao_shaping.drivers.slm.santec.slm200_constants import (
    PANEL_RES,
    PITCH_UM,
    PIXEL_SIZE_UM,
)
from ao_shaping.utils.image.exposure_control import (
    configured_daheng_exposure_limit,
)

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
    "bench_geometry_source",
    "bench_pupil_center",
    "bench_summary_lines",
    "bench_window_side",
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
    "queue_pending",
    "record_detection",
    "render_bench_settings",
    "reset_manual",
    "run_session_calibration",
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

#: 几何来源。``auto`` 每次运行前自动标定；``reuse`` 复用已标定几何文件；
#: ``keep`` 沿用上次结果（本次不再标定，也不改任何值）。
GEOMETRY_SOURCE_OPTIONS: tuple[str, ...] = ("auto", "reuse", "keep")
GEOMETRY_SOURCE_LABELS: dict[str, str] = {
    "auto": "自动标定（推荐）",
    "reuse": "复用已标定几何",
    "keep": "沿用上次结果",
}

#: 相机像元间距默认 / 已知型号表（大恒 MER2-507 = 2.2 μm，本实验系统实测）。
DEFAULT_CAMERA_PIXEL_UM = 2.2
KNOWN_CAMERA_PIXEL_UM: dict[str, float] = {"MER2-507": 2.2}

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
        "camera_pixel_um", CAMERA_PIXEL_KEY, "μm", "rel", CAMERA_PIXEL_REL_TOLERANCE, "相机像元间距"
    ),
    "exposure_ms": BenchField(
        "exposure_ms", "bench_exposure_ms", "ms", "rel", EXPOSURE_REL_TOLERANCE, "会话曝光"
    ),
    "cam_size": BenchField(
        "cam_size", "bench_cam_size", "px", "abs", float(WINDOW_SIDE_TOLERANCE_STEP), "读出窗口边长"
    ),
    "pupil_center": BenchField(
        "pupil_center", "bench_pupil_center", "px", "center", CENTER_TOLERANCE_PX, "入射中心"
    ),
    "panel_span_px": BenchField(
        "panel_span_px", "bench_panel_span_px", "px", "abs", PANEL_SPAN_TOLERANCE_PX, "模型覆盖宽度"
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


def bench_pupil_center(prefix: str = "slm1", fallback: str = PUPIL_CENTER_FALLBACK) -> str:
    """入射中心的字符串形式（``"x,y"``）。

    解析顺序：人工填的文本 → 本次标定的探测值 → ``fallback``。
    ``fallback`` 默认是本实验系统的实测值，只在**从未标定过**时兜底；控件本身
    的默认值是空串，placeholder 写「留空 = 自动定位」。
    """
    raw = str(st.session_state.get(_scoped(prefix, "bench_pupil_center"), "") or "").strip()
    if raw:
        return raw
    detected = detected_state(prefix).get("pupil_center")
    point = _coerce_point(detected)
    if point is not None:
        return f"{int(round(point[0]))},{int(round(point[1]))}"
    return fallback


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
    except Exception as exc:  # noqa: BLE001 - 探测失败不是错误，只是拿不到真值
        logger.debug("相机窗口上限探测失败 ({}): {}", cam_type, exc)
        return None
    finally:
        if cam is not None:
            try:
                cam.close_device()
            except Exception:  # noqa: BLE001 - 关闭失败无可补救
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
# 像元间距探测（自 multi_slm_controller「相机工具」区整体搬来）
# ---------------------------------------------------------------------------


def detect_camera_pixel_um(cam_type: str = "daheng", cam_id: int = 0) -> float | None:
    """从硬件探测相机像元间距（μm）；失败返回 ``None``。

    先读设备 ``PixelSize``，再回落到已知型号表（大恒 MER2-507 = 2.2 μm）。
    """
    if cam_type == "daheng":
        try:
            import gxipy as gx  # type: ignore[import-untyped]
        except (ImportError, OSError):
            return None
        device_manager = gx.DeviceManager()
        dev_count, dev_info_list = device_manager.update_device_list()
        if not dev_count or not dev_info_list or len(dev_info_list) <= cam_id:
            return None
        info = dev_info_list[cam_id]
        sn = info.get("sn")
        model_raw: Any = info.get("model", "") or ""
        if isinstance(model_raw, tuple):
            model_raw = model_raw[0] if model_raw else ""
        model = str(model_raw)
        cam: Any = None
        try:
            cam = device_manager.open_device_by_sn(sn)
            try:
                pixel_size = cam.PixelSize.get()
                if pixel_size is not None and float(pixel_size) > 0:
                    return float(pixel_size)
            except Exception:  # noqa: BLE001 - 特征缺失就落到型号表
                pass
        except Exception as exc:  # noqa: BLE001 - 探测失败不是错误
            logger.debug("大恒探测像元间距失败: {}", exc)
            return None
        finally:
            if cam is not None:
                try:
                    cam.close_device()
                except Exception:  # noqa: BLE001 - 关闭失败无可补救
                    pass
        for known_model, pitch in KNOWN_CAMERA_PIXEL_UM.items():
            if known_model in model:
                return pitch
        return None
    if cam_type == "miicam":
        try:
            from ao_shaping.drivers.ccd.miicam.driver import MIICamera

            cam = MIICamera(cam_id=cam_id)
            cam.open()
            try:
                pixel_size = getattr(cam, "pixel_size_um", None)
                if pixel_size is not None:
                    return float(pixel_size)
            finally:
                cam.close()
        except Exception as exc:  # noqa: BLE001 - 探测失败不是错误
            logger.debug("MiiCam 探测像元间距失败: {}", exc)
        return None
    return None


def auto_check_camera_pixel_um(cam_type: str | None = None, cam_id: int | None = None) -> None:
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
                rel = abs(float(detected) - float(configured)) / abs(float(detected) or 1.0)
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
            st.warning(f"相机像元间距：会话值 {configured} μm ≠ 硬件探测值 {detected} μm")
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
        "region": int(_read(f"{prefix}_shp_model-in-loop_region", MIL_WIDGET_DEFAULTS["region"])),
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
            _read(f"{prefix}_shp_model-in-loop_probe_spread", MIL_WIDGET_DEFAULTS["probe_spread"])
        ),
        "probe_count": int(
            _read(f"{prefix}_shp_model-in-loop_probe_count", MIL_WIDGET_DEFAULTS["probe_count"])
        ),
        "n_rounds": int(_read(f"{prefix}_shp_model-in-loop_n_rounds", MIL_WIDGET_DEFAULTS["n_rounds"])),
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
        exposure_time_ms=float(st.session_state.get("bench_exposure_ms", 0.0) or 0.0),
        max_exposure_ms=float(configured_daheng_exposure_limit()),
        cam_size=int(st.session_state.get("bench_cam_size", DEFAULT_WINDOW_SIDE_PX) or 0),
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
    raw = str(st.session_state.get(_scoped(prefix, "bench_geometry_file"), "") or "").strip()
    return raw or None


def bench_geometry_source(prefix: str) -> str:
    """当前几何来源（非法值回落 ``auto``）。"""
    source = str(st.session_state.get(_scoped(prefix, "bench_geometry_source"), "auto"))
    return source if source in GEOMETRY_SOURCE_OPTIONS else "auto"


def uncalibrated_note(prefix: str) -> str:
    """「还没标定」那一行（状态条与光路摘要共用同一句措辞）。

    措辞里必须出现「运行时先自动标定」和「约 1–3 分钟」：这是操作员在点运行
    **之前**唯一能看到自己将要等多久的地方，而运行本身是同步阻塞的。
    """
    source = bench_geometry_source(prefix)
    return (
        f"几何来源「{GEOMETRY_SOURCE_LABELS.get(source, source)}」"
        " —— 运行时先自动标定（约 1–3 分钟：入射定位 → 曝光探测 → sweep 几何）"
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
    except Exception as exc:  # noqa: BLE001 - 参数/配置不可用，交给操作者改
        message = f"标定参数不可用: {exc}"
        logger.exception("标定参数不可用: {}", exc)
        st.error(message)
        return message

    try:
        with st.spinner(
            "正在标定，请勿触碰设备（入射定位 → 曝光探测 → sweep 几何，约 1–3 分钟）"
        ), with_slm_ownership(slm_num):
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
    except Exception as exc:  # noqa: BLE001 - calibrate_session 抛普通异常
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
    pin_geometry_source: bool = True,
) -> dict[str, Any]:
    """把一次会话标定的结果写进三态表，并排好下一次 flush 的 widget 回填。

    三态的判定只有这一处实现：侧栏按钮 :func:`run_session_calibration` 与运行
    路径 :meth:`~ao_shaping.gui.slm.shaping_controls.ModelInLoopMethod.auto_calibrate`
    共用它，所以「什么算探测值 / 什么算人工值」在两条路径上不可能分歧。

    ``pin_geometry_source=False`` 给运行路径用：操作员选的是「自动标定（每次
    运行前重测）」，一次运行**不应该**把这个选择悄悄改成「复用已标定几何」，
    否则第二次运行就静默地不再测量了。侧栏按钮保留原行为（回填完切到复用，
    因为它本身就是一次性动作）。
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
    backfill = _detected_backfill(
        prefix, resolved_config, pin_source=pin_geometry_source
    )
    queue_pending(prefix, backfill)
    return backfill


def _detected_backfill(
    prefix: str, resolved_config: Any = None, *, pin_source: bool = True
) -> dict[str, Any]:
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
    if pin_source and backfill.get(fields["geometry_file"].widget_key):
        backfill[_scoped(prefix, "bench_geometry_source")] = "reuse"
    return backfill


def _now_text() -> str:
    from ao_shaping.utils.io.file import gen_date_str

    try:
        return str(gen_date_str())
    except Exception:  # noqa: BLE001 - 时间戳不是关键路径
        return ""


# ---------------------------------------------------------------------------
# 光路摘要卡（折叠态也要看得到）
# ---------------------------------------------------------------------------


def bench_summary_lines(prefix: str) -> tuple[str, ...]:
    """光路摘要（``●`` 未标定 / 已标定 + 手动覆盖项）。"""
    detected = detected_state(prefix)
    manual = manual_fields(prefix)
    lines: list[str] = []
    if not detected.get("pupil_center") and not detected.get("geometry_file"):
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
    values["cam_size"] = int(st.session_state.get("bench_cam_size", 0) or 0) if crop else 0
    values["exposure_time_ms"] = float(st.session_state.get("bench_exposure_ms", 0.0) or 0.0)
    values["camera_pixel_um"] = float(
        st.session_state.get(CAMERA_PIXEL_KEY, DEFAULT_CAMERA_PIXEL_UM)
    )
    values["blaze_period_px"] = int(st.session_state.get("bench_blaze_period_px", -1))
    values["blaze_direction"] = str(st.session_state.get("bench_blaze_direction", "vertical"))
    values["max_exposure_ms"] = float(configured_daheng_exposure_limit())

    if method_key == "gs-refine":
        span = float(bench_value(prefix, "panel_span_px", 2 * 450.0))
        values["panel_span_px"] = span
        values["beam_radius_px"] = span / 2.0
    else:
        values["panel_span_px"] = float(bench_value(prefix, "panel_span_px", 900.0))
        values["geometry_file"] = _geometry_file_for_source(prefix) or ""
    values["pupil_center"] = bench_pupil_center(prefix)
    values["bench_geometry_source"] = bench_geometry_source(prefix)
    values["bench_prefix"] = prefix
    values["bench_slm_number"] = slm_num
    return values


# ---------------------------------------------------------------------------
# 侧栏渲染
# ---------------------------------------------------------------------------


def render_bench_settings() -> None:
    """侧栏「本次实验光路」区块入口。

    顺序是硬约束：先 flush pending、再重算人工覆盖表，最后才实例化控件。
    调用方（``render_slm_sidebar``）只需在侧栏早期调用一次。
    """
    flush_pending("")
    sync_manual_states("")
    _render_experiment_block()

    for slm_num in (1, 2):
        prefix = f"slm{slm_num}"
        if not (st.session_state.get(f"{prefix}_connected") and st.session_state.get(prefix)):
            continue
        flush_pending(prefix)
        sync_manual_states(prefix)
        _render_slm_block(prefix, slm_num)


def _render_experiment_block() -> None:
    """实验级共享的光路设置（相机 / 读出窗口 / 载波）+ 光路摘要卡。"""
    busy = bench_busy()
    disabled = busy
    detected = detected_state("")
    manual = manual_fields("")

    with st.expander("本次实验光路", expanded=False):
        if busy:
            st.info(
                "标定或闭环整形进行中 —— SLM 句柄已交给标定器 / 优化器"
                "（SDK 不可重入），此时显示未连接属正常。"
            )

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
                value=int(st.session_state.get("bench_cam_id", 0) or 0),
                step=1,
                key="bench_cam_id",
                disabled=disabled,
            )
        auto_check_camera_pixel_um(cam_type, int(st.session_state.get("bench_cam_id", 0) or 0))

        p1, p2 = st.columns([3, 1])
        with p1:
            st.number_input(
                "像元间距 (μm)",
                min_value=0.1,
                max_value=50.0,
                step=0.1,
                value=float(st.session_state.get(CAMERA_PIXEL_KEY, DEFAULT_CAMERA_PIXEL_UM)),
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

        exposure = st.number_input(
            "会话曝光 (ms)",
            min_value=0.0,
            value=float(st.session_state.get("bench_exposure_ms", 0.0) or 0.0),
            step=0.01,
            format="%.3f",
            key="bench_exposure_ms",
            disabled=disabled,
            help="0 = 自动起曝。**填 0 时大恒驱动会钳到约 0.02 ms，比可用区间暗 "
            "20–75×**；实验前必须按当前激光功率实测曝光。上限取 "
            "AO_DAHENG_MAX_EXPOSURE_MS（本光路饱和保护），缺省/0 时由画面探测。",
        )
        ceiling = configured_daheng_exposure_limit()
        if cam_type == "daheng" and ceiling > 0 and float(exposure) > ceiling:
            st.warning(f"曝光 {exposure} ms 超过本光路饱和保护上限 {ceiling} ms，运行前会被拦下")
        if cam_type == "daheng" and float(exposure) == 0:
            st.warning("曝光 = 0 → 大恒钳到约 0.02 ms（比可用区间暗 20–75×）")

        st.caption("B 读出窗口")
        bounds = window_bounds()
        crop = st.checkbox(
            "启用开窗",
            value=bool(st.session_state.get("bench_cam_crop", True)),
            key="bench_cam_crop",
            disabled=disabled,
            help="关闭则整幅读出；开启后窗口边长必须是设备步长的整数倍",
        )
        window_side = st.number_input(
            "窗口边长 (px)",
            min_value=bounds.min_side,
            max_value=bounds.max_side,
            value=bounds.clamp(
                int(st.session_state.get("bench_cam_size", DEFAULT_WINDOW_SIDE_PX))
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

        st.caption("C 去零级载波")
        d1, d2 = st.columns(2)
        with d1:
            st.selectbox(
                "载波",
                options=list(BLAZE_PERIOD_OPTIONS),
                format_func=lambda p: BLAZE_PERIOD_LABELS[p],
                key="bench_blaze_period_px",
                disabled=disabled,
                help="离轴载波把 0 级移出读出窗；-1 = 关闭，0 = 实测自动选择",
            )
        with d2:
            st.selectbox(
                "光栅方向",
                options=list(BLAZE_DIRECTIONS),
                key="bench_blaze_direction",
                disabled=disabled,
            )

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
        if detected.get("exposure_ms") is not None and field_state("", "exposure_ms") == "detected":
            st.caption("上表曝光为标定时在用的会话曝光（sweep 内部还会按画面自动起曝）")


def _render_slm_block(prefix: str, slm_num: int) -> None:
    """每台已连接 SLM 的面板几何与标定区块。"""
    busy = bench_busy()
    disabled = busy
    detected = detected_state(prefix)

    with st.expander(f"SLM {slm_num} · 光路几何与标定", expanded=False):
        if busy:
            st.info(
                f"闭环整形/标定进行中 —— SLM {slm_num} 句柄已交给优化器"
                "（SDK 不可重入），此时显示未连接属正常。"
            )

        st.caption("D 面板与瞳面")
        st.text_input(
            "入射中心 x,y (面板px)",
            value=str(st.session_state.get(_scoped(prefix, "bench_pupil_center"), "") or ""),
            key=_scoped(prefix, "bench_pupil_center"),
            placeholder="留空 = 自动定位",
            disabled=disabled,
            help="入射光斑在 SLM 面板上的坐标 (x, y)。留空则运行时由 "
            "calibrate_session 的全板随机相位盘扫描定位。",
        )
        st.number_input(
            "模型覆盖宽度 (面板px)",
            min_value=32,
            max_value=int(PANEL_RES[0]),
            value=bounds_clamp_panel_span(
                prefix, float(st.session_state.get(_scoped(prefix, "bench_panel_span_px"), 900.0))
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

        st.caption("E 标定")
        source = st.selectbox(
            "几何来源",
            options=list(GEOMETRY_SOURCE_OPTIONS),
            format_func=lambda s: GEOMETRY_SOURCE_LABELS[s],
            key=_scoped(prefix, "bench_geometry_source"),
            disabled=disabled,
            help="自动标定 = 每次运行前重新测量；复用 = 用下方已标定几何文件；"
            "沿用上次结果 = 本次不改任何光路值",
        )
        if source == "reuse":
            st.text_input(
                "几何文件路径 (bench_geometry.json)",
                value=str(st.session_state.get(_scoped(prefix, "bench_geometry_file"), "") or ""),
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
                else "全板入射定位 → 曝光探测 → sweep 几何（约 1–3 分钟）"
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
    raw = str(st.session_state.get(_scoped(prefix, "bench_geometry_file"), "") or "").strip()
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