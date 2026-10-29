"""多形状整形 campaign 的操作台：形状 + 尺寸是唯一输入，其余由自动辅助层派生。

定位
----
本页是 ``report/shape_trials/multishape_scale/multishape_scale_live_20261028.md``
§3.2 那次实验（4 形状 × 4 阶段 = 16 段 / 112 块）的操作面。那次是逐个 lane 手敲
PowerShell 跑出来的，手敲的两个失败模式（**参数漂移**、**来源丢失**）都是结构性的，
不是"人粗心"：所以本页不摆 38 个参数，只让人选**形状与尺寸**，其余全部由
:mod:`ao_shaping.tools.slm.transfer.campaign` 派生，并在表里逐行标出这个值
是**实测 / 派生 / CLI 默认 / 手工覆盖**的哪一类、为什么。

与 ``diff_opt_controller.py`` 的关系
-----------------------------------
页面骨架（``streamlit run`` 独立入口、``render_slm_sidebar``、``gui/slmshape/connect``
的状态初始化）沿用 ``diff_opt_controller``；进度明细复用
``tools.slm.transfer.campaign_progress``（campaign 口径的唯一读取器）与
``gui.slm.transfer_progress`` 的两件成品（``SHAPE_LABELS``、``render_progress``）。

**不 import ``multi_slm_controller`` 之外任何上层页面，也不被它 import**：
``multi_slm_controller`` 已被 ``diff_opt_controller`` 依赖，反向 import 会闭合成环。

离线可用
--------
计划预览、参数派生、矩阵/预算、检查卡、历史浏览全部**不需要设备**：它们只读
:mod:`campaign` / :mod:`campaign_progress` / :mod:`precheck` 与本地目录。只有
「开始」要求已连接 SLM，并且要在检查卡上把黄项逐条勾掉。

断点续跑
--------
⑥ 的「从断点续跑」开关只在 ``--out`` 里已有 ``journal.json`` 时出现，默认勾上，并列出
本次会跳过的段（口径 = :data:`campaign_progress.COMPLETE_STATUSES`，``stopped`` 不算）。
它落到 runner 的 ``--resume``：``manifest.json`` 已存在的段不再进段、journal 记
``reused``；**不带**它时已存在的段目录会抛 ``FileExistsError``，所以页面写的是
「取消勾选只适用于换一个新的 --out」，而不是「重跑一遍」。

光路无光
--------
:class:`OpticalPathInvalid`（``exit_code = 3``）是实验系统上的**预期**状态，不是缺陷。
后台线程把它整条搬回主线程（``("optical_path", {"reason", "cause", "report"})``），⑥
用 :func:`render_optical_path` 印 ``report()`` 原文 ＋ 按 ``cause`` 的处置方向
（``blocked`` ⇒ 先看光路；``exposure_ceiling`` ⇒ 先调 ``AO_DAHENG_MAX_EXPOSURE_MS``）。
⑤ 的检查卡**不碰** ``verify_light``：那要活相机（``read_frame`` + ``set_exposure_ms``），
塞进去就毁掉"没设备也能先看一眼"这条前提。

线程契约
--------
一次 campaign 要跑几小时，只能放后台线程。``Thread(target=...)`` 的线程体
（:func:`_campaign_worker`）**不读不写** ``st.session_state``（
``tests/ao_shaping/test_conventions.py::test_gui_thread_bodies_never_reach_session_state``
就是这条的红线）：参数快照、``threading.Event`` 与 ``queue.Queue`` 在主线程建好当参数传进去，
结果按 ``("done", summary)`` / ``("error", msg)`` / ``("optical_path", payload)`` 回传，
由 ``@st.fragment(run_every=2.0)`` 在主线程抽干队列。设备交接（``with_slm_ownership``）
也是**主线程**进出的：进入在点「开始」那一刻，退出在队列报完成那一刻 —— 线程里只碰设备，
不碰会话状态。
"""

from __future__ import annotations

import html
import json
import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from loguru import logger

from ao_shaping.gui.slm import bench_settings, precheck
from ao_shaping.gui.slm.multi_slm_controller import render_slm_sidebar
from ao_shaping.gui.slm.shaping_controls import SlmOwnership, with_slm_ownership
from ao_shaping.gui.slm.transfer_progress import SHAPE_LABELS, collect_progress, render_progress
from ao_shaping.gui.slmshape.connect import connected_slm_numbers, ensure_state
from ao_shaping.tools.slm.transfer.campaign import (
    SIZE_WINDOWS,
    FieldSource,
    campaign_matrix,
    derive_settings,
    estimate_disk_gb,
    estimate_seconds,
    matrix_totals,
)
from ao_shaping.tools.slm.transfer.campaign_progress import (
    COMPLETE_STATUSES,
    MISSING,
    collect_campaign_progress,
)
from ao_shaping.tools.slm.transfer.light_guard import (
    CAUSE_BLOCKED,
    CAUSE_EXPOSURE_CEILING,
    EXIT_OPTICAL_PATH_INVALID,
    OpticalPathInvalid,
)

__all__ = ["PAGE_TITLE", "main"]

#: 页面标题 / 侧栏（浏览器标签）里的名字。
PAGE_TITLE = "多形状整形 campaign"

#: 本 campaign 的四个形状，顺序照 ``slm-transfer-shapes --shapes`` 的 click 默认值
#: （``runners/slm/transfer_shapes_runner.py``）。``TRANSFER_SHAPE_CHOICES`` 里还有
#: ``rectangle``，但报告的 16 段里没有它 —— 摆一个从未被这份协议跑过的形状，
#: 等于让操作员跑出一个没有对照的 lane。
CAMPAIGN_SHAPES: tuple[str, ...] = ("square", "triangle", "star", "circle")

#: 尺寸档位（分段控件）。任意其它值走「其他」+ 数字框；窗口半跨由
#: ``campaign.window_for_side`` 就近取档，不在这里另立一张表。
SIDE_PRESETS: tuple[int, ...] = (1200, 600, 200)

#: 「其他」档的文案（同时也是 ``st.segmented_control`` 的一个 option）。
SIDE_CUSTOM = "其他"

#: 高级覆盖面板里可以手填的字段（名字 = runner 的 click 参数名）。
OVERRIDE_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("window_span", "采样窗半跨 window_span（px）", "number"),
    ("period", "载波周期 period（面板 px）", "number"),
    ("capture_grid", "探针采样网格 capture_grid", "int"),
    ("control_grid", "控制网格 control_grid", "int"),
    ("fft_size", "前向 FFT 边长 fft_size", "int"),
    ("frames", "每帧连拍数 frames", "int"),
    ("initial_phase", "初相位构造 initial_phase", "choice"),
    ("replay_exposures_auto", "自动重放曝光 replay_exposures_auto", "flag"),
)

#: ``initial_phase`` 的取值（与 click 的 ``--initial-phase`` 同一张表）。
INITIAL_PHASE_CHOICES: tuple[str, ...] = (
    "auto", "plane", "random", "transport", "backprojection", "gs",
)

#: 来源标签的中文名 + 徽标配色（measured=绿 / derived=蓝 / default=灰 / manual=黄）。
_SOURCE_BADGES: dict[str, tuple[str, str, str]] = {
    FieldSource.MEASURED.value: ("实测", "#e6f4ea", "#137333"),
    FieldSource.DERIVED.value: ("派生", "#e8f0fe", "#1967d2"),
    FieldSource.DEFAULT.value: ("默认", "#f1f3f4", "#5f6368"),
    FieldSource.MANUAL.value: ("手工", "#fef7e0", "#b06000"),
}

#: 运行体在 ``st.session_state`` 里的键（线程本身只在主线程被读写）。
_RUN_KEY = "msc_run"
_SUMMARY_KEY = "msc_summary"
_ERROR_KEY = "msc_error"
_OVERRIDES_KEY = "msc_overrides"
_RESET_KEY = "msc_overrides_reset"
_OUT_SIDE_KEY = "msc_out_side"
_MODEL_KEY = "msc_model"
_BUDGET_KEY = "msc_budget_hours"
_ACK_KEY = "msc_acks"
_RESUME_KEY = "msc_resume"
_RESUME_DIR_KEY = "msc_resume_dir"
_OPTICAL_KEY = "msc_optical_path"


# --------------------------------------------------------------------------- #
# 目录助手（全部只读本地文件，不碰设备）
# --------------------------------------------------------------------------- #
def campaign_root() -> Path:
    """campaign 的输出根目录：``data/debug/slm_transfer``（与 runner 的默认 --out 同级）。"""
    return Path("data") / "debug" / "slm_transfer"


def _journal_mtime(directory: Path) -> float:
    """``journal.json`` 的修改时间（读不到按 0 算 —— 排序不该因为半截产物而崩）。"""
    try:
        return (directory / "journal.json").stat().st_mtime
    except OSError:
        return 0.0


def _campaign_dirs(root: Path | None = None) -> list[Path]:
    """含 ``journal.json`` 的 campaign 目录（深一层 = ``<stamp>/s1200`` 那种布局）。

    按 ``journal.json`` 的 mtime 倒序：目录名是 ``s1200`` / ``s600``，按名字排会把
    三档尺寸混成一组，看不出哪一份是「刚刚跑完的那份」。
    """
    base = campaign_root() if root is None else Path(root)
    if not base.is_dir():
        return []
    found = [p.parent for p in base.glob("*/journal.json")]
    found += [p.parent for p in base.glob("*/*/journal.json")]
    unique = {path.resolve(): path for path in found}
    return sorted(unique.values(), key=_journal_mtime, reverse=True)


def _model_dirs(root: Path | None = None) -> list[Path]:
    """含 ``fit/result.json`` 的目录（= 可以复用的模型），最新在前。"""
    base = campaign_root() if root is None else Path(root)
    if not base.is_dir():
        return []
    found = list(base.glob("*/fit/result.json")) + list(base.glob("*/*/fit/result.json"))
    dirs = {path.parent.parent.resolve(): path.parent.parent for path in found}
    return sorted(dirs.values(), key=lambda p: p.stat().st_mtime, reverse=True)


def _known_zero_xy(campaigns: list[Path]) -> tuple[tuple[float, float] | None, str]:
    """上一份 campaign 实测的零级位置（sensor px）与一句话出处。

    零级是**逐光路实测**的（``transfer_shapes_runner`` 在 fit 之后用
    ``fit_stage.spot(reference)`` 自己测），本页只能沿用历史：读 ``geometry_plan.json``
    的 ``zero_xy``，再退回 journal 的 ``geometry`` 条目。都读不到就诚实地报 ``None``
    —— 几何检查会因此显示「未知」，而不是拿一个编出来的位置算 5.61 %。
    """
    for directory in campaigns:
        plan_file = Path(directory) / "geometry_plan.json"
        try:
            plan = json.loads(plan_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            plan = None
        if isinstance(plan, dict):
            point = plan.get("zero_xy")
            if isinstance(point, (list, tuple)) and len(point) == 2:
                return (float(point[0]), float(point[1])), f"{Path(directory).name}/geometry_plan.json"
        journal_file = Path(directory) / "journal.json"
        try:
            text = journal_file.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            entries = json.loads(text)
        except ValueError:
            continue
        for entry in reversed(entries if isinstance(entries, list) else []):
            if not isinstance(entry, dict) or entry.get("stage") != "geometry":
                continue
            point = entry.get("zero_xy")
            if isinstance(point, (list, tuple)) and len(point) == 2:
                return (float(point[0]), float(point[1])), f"{Path(directory).name}/journal.json"
    return None, "无历史零级位置"


def completed_segments(out_dir: Path) -> list[str]:
    """``--out`` 里已经跑完、``--resume`` 会直接复用的那些段（``形状/阶段``）。

    判据只有一个：``campaign_progress`` 给出的 journal 状态落在
    :data:`COMPLETE_STATUSES`（``complete`` / ``reused``）—— 与 ⑦ 历史面板读的是
    同一份事实，不在这里另立一套"算不算做完"。``stopped`` **不算**：停下的段还有块
    没跑，把它列进「将跳过」正是让操作员误以为收工的那种谎。

    没有 ``journal.json`` 时返回空表：新目录没什么可续的，也就不必问「续跑吗」。
    """
    directory = Path(out_dir)
    if not (directory / "journal.json").is_file():
        return []
    progress = collect_campaign_progress(directory)
    return [f"{stage.shape}/{stage.stage}" for stage in progress.stages
            if stage.status in COMPLETE_STATUSES]


# --------------------------------------------------------------------------- #
# 线程体（不得读写 st.session_state）
# --------------------------------------------------------------------------- #
def _campaign_worker(*, out: Path, kwargs: dict[str, Any],
                     stop_event: threading.Event,
                     out_q: "queue.Queue[tuple[str, Any]]") -> None:
    """后台跑一次 campaign，结果/异常按队列项回传。

    线程边界契约（AGENTS.md「Streamlit background threads」）：参数快照、``stop_event``
    与 ``out_q`` 都是**普通参数**，函数体里没有一处 ``st.session_state`` —— 它是在主线程
    之外运行的，碰到 Streamlit 的 ``ScriptRunContext`` 只会产生竞态而不是进度。

    ``should_stop`` 传 ``stop_event.is_set``（段粒度：一段跑完才停，单块是原子事务）。
    """
    from ao_shaping.runners.slm.transfer_shapes_runner import execute_campaign

    try:
        summary = execute_campaign(out=out, should_stop=stop_event.is_set, **kwargs)
    # 光路无光是**预期**状态而不是缺陷（有人挡光 / 激光没出光 / 光路漂走）：把
    # ``reason`` / ``cause`` / ``report()`` 原样搬回主线程，页面直接印实测数字，
    # 并按 ``cause`` 给处置方向。必须排在 ``Exception`` 之前（它是 RuntimeError）。
    # 不带 traceback 也走 loguru 的 warning 而不是 exception：栈帧只会把一条可诊断
    # 的信息埋掉，这正是 light_guard 存在的理由。
    except OpticalPathInvalid as exc:
        logger.warning("campaign 光路无光: {} (reason={}, cause={})",
                       out, exc.reason, exc.cause)
        out_q.put(("optical_path", {"reason": exc.reason, "cause": exc.cause,
                                    "report": exc.report()}))
    # 线程边界：这里要把 worker 里**任何**失败搬回主线程，所以刻意收全
    # Exception + KeyboardInterrupt + SystemExit（GeneratorExit 之外就没别的了）。
    # 不写 ``except BaseException`` 是因为 src/ 有一条"不许新增宽泛捕获"的棘轮
    # （test_conventions._blind_except_handlers_in_src），而这三项就是这条边界上
    # 需要搬运的全部；``tests/.../test_thread_boundary_exception_capture_keeps_base_exception``
    # 管的是另一类形状（函数内自建 Thread 并把异常存起来重抛），不适用此处。
    except (Exception, KeyboardInterrupt, SystemExit) as exc:
        logger.exception("campaign 失败: {}", out)
        out_q.put(("error", f"{type(exc).__name__}: {exc}"))
    else:
        out_q.put(("done", summary))
    finally:
        stop_event.set()


@dataclass
class _CampaignRun:
    """主线程持有的运行体：线程句柄 + 停止事件 + 结果队列 + 设备交接。

    ``ownership`` 是 :func:`with_slm_ownership` 的**已进入**上下文：进入在点「开始」
    那一刻（主线程，先停相位翻转 → 置 ``bench_busy`` → 释放 SLM 句柄），退出在队列报
    完成那一刻（主线程，重新取回句柄 → 复位 ``bench_busy``）。跨 rerun 持有它，是为了
    让 ``bench_busy`` 在整个 campaign 期间为真 —— 侧栏的光路控件因此**碰不到**正在被
    runner 占用的设备。
    """

    out: Path
    shapes: tuple[str, ...]
    stop_event: threading.Event
    out_q: "queue.Queue[tuple[str, Any]]"
    thread: threading.Thread
    ownership: Any
    started_at: str


# --------------------------------------------------------------------------- #
# ① 目标
# --------------------------------------------------------------------------- #
def render_targets() -> tuple[tuple[str, ...], float]:
    """① 目标：四个形状勾选框 + 尺寸档位。这是全页唯一的必填输入。"""
    st.subheader("① 目标（唯一必填）")
    st.caption("形状与尺寸决定采样窗、目标框几何、零级重叠面积与全部预算；"
               "其余参数在下面由自动辅助层派生并标注来源。")
    columns = st.columns(len(CAMPAIGN_SHAPES))
    shapes = tuple(
        shape for shape, column in zip(CAMPAIGN_SHAPES, columns)
        if column.checkbox(SHAPE_LABELS.get(shape, shape), value=True,
                           key=f"msc_shape_{shape}")
    )
    preset = st.segmented_control(
        "目标边长（CCD 原生 px）", options=[*SIDE_PRESETS, SIDE_CUSTOM],
        default=SIDE_PRESETS[0], key="msc_side_preset",
        help=f"三档是实测过的尺寸；其他值就近取采样窗档位（{SIZE_WINDOWS}）",
    )
    if preset == SIDE_CUSTOM:
        side = float(st.number_input("自定义边长（px）", min_value=16.0, max_value=4000.0,
                                     value=400.0, step=10.0, key="msc_side_custom"))
    else:
        side = float(preset or SIDE_PRESETS[0])
    return shapes, side


# --------------------------------------------------------------------------- #
# ② 自动辅助设置
# --------------------------------------------------------------------------- #
def _badge(source: str) -> str:
    label, background, colour = _SOURCE_BADGES.get(source, (source, "#f1f3f4", "#5f6368"))
    return (f"<span style='background:{background};color:{colour};padding:1px 6px;"
            f"border-radius:8px;font-size:0.85em'>{label}</span>")


def _format_value(value: Any) -> str:
    """把派生值印成一行（``900.0`` 不写成 ``900``：这是给人核对的数字，不是坐标）。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(_format_value(item) for item in value) + ")"
    return str(value)


def _numeric(value: Any) -> Any:
    """表格里的缺值：``"—"``（:data:`MISSING`）→ ``None``。

    ``st.dataframe`` 走 arrow 序列化，同一列里既有 ``0.6574`` 又有 ``"—"`` 时会退化成
    object 列，arrow 接着按 double 转换并抛 ``ArrowInvalid`` —— 整张表**不显示**，栈只进
    日志。触发条件恰好是「某段还没有块读数」（刚开始 / 被 ``stopped`` / 读到半截产物），
    也就是操作员最需要看进度的那一刻。缺值一律给 ``None``：arrow 用 NaN 表示，界面空白。
    """
    return None if value == MISSING else value


def render_settings_table(settings) -> None:
    """② 的参数表：参数 / 值 / 来源徽标 / 说明。

    渲染成 HTML 表而不是 ``st.dataframe``：来源要**配色**（实测绿 / 派生蓝 / 默认灰 /
    手工黄），而 dataframe 的单元格配色要额外引 pandas Styler；这张表只有十几行。
    """
    rows = []
    for name, setting in settings.settings.items():
        rows.append(
            "<tr>"
            f"<td style='white-space:nowrap'><code>{html.escape(name)}</code></td>"
            f"<td style='white-space:nowrap'>{html.escape(_format_value(setting.value))}</td>"
            f"<td>{_badge(setting.source.value)}</td>"
            f"<td style='color:#5f6368'>{html.escape(str(setting.note))}</td>"
            "</tr>"
        )
    st.markdown(
        "<table style='width:100%;border-collapse:collapse;font-size:0.9em'>"
        "<thead><tr style='text-align:left'>"
        "<th>参数</th><th>值</th><th>来源</th><th>说明（为什么是这个值）</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>",
        unsafe_allow_html=True,
    )


def _apply_pending_reset() -> None:
    """清空手工覆盖的**延后写**：在本次运行任何 widget 实例化之前落地。

    为什么不能就地清：点「按尺寸重新推导」时，上一轮的 ``msc_ov_on_*`` /
    ``msc_ov_val_*`` 已经作为 widget 键存在于 ``st.session_state``；只要本轮还没重建
    它们，删除是允许的（Streamlit 只禁止"本次运行里已实例化的 widget 键"），但把
    这次删除放在**函数第一行**，就不必依赖「按钮恰好在面板前面」这个布局前提 ——
    面板挪到按钮上面也不会变成 ``StreamlitAPIException``。
    """
    if not st.session_state.pop(_RESET_KEY, False):
        return
    st.session_state[_OVERRIDES_KEY] = {}
    for name, _, _ in OVERRIDE_FIELDS:
        st.session_state.pop(f"msc_ov_on_{name}", None)
        st.session_state.pop(f"msc_ov_val_{name}", None)


def render_auto_assist(shapes: tuple[str, ...], side: float):
    """② 自动辅助设置：派生 + 高级覆盖 → :class:`CampaignSettings`。

    表格占位上提、填充后置：覆盖面板改过值要**立刻**在表里看到（来源从「派生」变
    「手工」）。若按「先画表、后读面板」的顺序写，表会比面板晚一轮 —— widget 变更
    触发的那次重跑里，表读的还是上一轮的覆盖表，于是人与屏对着看会以为没生效。
    """
    st.subheader("② 自动辅助设置")
    st.caption("每一行都标出这个值从哪来：**实测**（本会话测的）/ **派生**（由形状+尺寸"
               "或报告口径推出）/ **默认**（runner 的 CLI 默认值）/ **手工**（有人改过，"
               "说明里写明被替换掉的派生值）。")
    _apply_pending_reset()
    overrides: dict[str, Any] = dict(st.session_state.get(_OVERRIDES_KEY) or {})
    settings = derive_settings(shapes, side, overrides=overrides or None)

    table_slot = st.empty()
    if st.button("按尺寸重新推导（清空全部手工覆盖）", key="msc_reset_overrides",
                 disabled=not overrides):
        st.session_state[_RESET_KEY] = True
        st.rerun()

    new_overrides = _render_advanced_overrides(settings, overrides)
    if new_overrides != overrides:
        st.session_state[_OVERRIDES_KEY] = new_overrides
        overrides = new_overrides
        settings = derive_settings(shapes, side, overrides=overrides or None)
    with table_slot.container():
        render_settings_table(settings)
    for warning in settings.warnings:
        st.warning(warning)
    for problem in settings.blocking():
        st.error(f"结构性错误：{problem}")
    return settings


def _render_advanced_overrides(settings, overrides: dict[str, Any]) -> dict[str, Any]:
    """高级覆盖面板：每个字段一个「启用覆盖」开关 + 输入框，返回新的覆盖表。

    未启用的字段**不**进覆盖表（于是它仍然是派生/默认值）；启用后才写，被替换掉的
    派生值由 ``derive_settings`` 记进该行的说明里。
    """
    updated = dict(overrides)
    with st.expander("高级覆盖（每一项都会把来源改成「手工」）", expanded=False):
        st.caption("默认全部不覆盖：留空即采用上表的派生值。覆盖用于**换实验变体**，"
                   "不是用来调试 —— 被改过的 lane 只能与同参数的 lane 配对比较。")
        for name, label, kind in OVERRIDE_FIELDS:
            derived = settings.settings.get(name)
            derived_text = _format_value(derived.value) if derived is not None else "—"
            enabled = st.checkbox(label, value=name in updated,
                                  key=f"msc_ov_on_{name}",
                                  help=f"当前派生/默认值：{derived_text}")
            if not enabled:
                updated.pop(name, None)
                st.session_state.pop(f"msc_ov_val_{name}", None)
                continue
            updated[name] = _override_input(name, label, kind, derived)
    return updated


def _override_input(name: str, label: str, kind: str, derived) -> Any:
    """单个覆盖输入框；``derived`` 是它的派生值（作为输入框的初值）。"""
    base = derived.value if derived is not None else None
    widget_key = f"msc_ov_val_{name}"
    if kind == "number":
        return float(st.number_input(label, min_value=1.0, value=float(base),
                                     step=1.0, key=widget_key))
    if kind == "int":
        return int(st.number_input(label, min_value=1, value=int(base), step=1,
                                   key=widget_key))
    if kind == "choice":
        choices = list(INITIAL_PHASE_CHOICES)
        current = str(base) if str(base) in choices else choices[0]
        return st.selectbox(label, options=choices, index=choices.index(current),
                            key=widget_key)
    return bool(st.checkbox(label, value=bool(base), key=widget_key))


# --------------------------------------------------------------------------- #
# ③ 计划预览
# --------------------------------------------------------------------------- #
def default_out_dir(side: float) -> Path:
    """默认 ``--out``：``<root>/multishape_<YYYYMMDD_HHMM>/s<side>``。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return campaign_root() / f"multishape_{stamp}" / f"s{side:g}"


def render_plan(settings) -> Path:
    """③ 计划预览：16 段矩阵 + 合计 + 落盘路径（可在高级里改）。"""
    st.subheader("③ 计划预览")
    rows = campaign_matrix(settings.shapes)
    totals = matrix_totals(rows)
    st.dataframe(
        [{"形状": SHAPE_LABELS.get(row.shape, row.shape), "阶段": row.stage,
          "块数": row.blocks, "位深": row.bit_depth, "updates": row.updates}
         for row in rows],
        hide_index=True, use_container_width=True,
    )
    low, expected, high = estimate_seconds(rows)
    st.caption(
        f"合计 {totals['segments']} 段 / {totals['blocks']} 块 / {totals['updates']} updates"
        f" · 整形墙钟 ≈{expected / 60:.0f} min（{low / 60:.0f}–{high / 60:.0f} min）"
        f" · 预计落盘 ≈{estimate_disk_gb(rows):.1f} GB"
        " · 口径：块数 × 24 s/块与每阶段静态基线（campaign.estimate_seconds / "
        "estimate_disk_gb），capture / fit / 重放不计入"
    )

    default_out = default_out_dir(settings.side)
    # 尺寸变了就换一个默认落盘目录：带 key 的 text_input 只在首次运行采用 ``value``，
    # 不清掉它就会一直写上一档的 ``s1200`` 路径（跑 s200 时悄悄覆盖上一档的产物）。
    # 这里直接写 widget key（在控件实例化**之前**，与 bench_settings.flush_pending 同法），
    # 而不是把会话里的值再当 ``value=`` 传一遍 —— 那正是 Streamlit 会告警、且值被忽略的写法。
    if st.session_state.get(_OUT_SIDE_KEY) != settings.side:
        st.session_state["msc_out_text"] = str(default_out)
        st.session_state[_OUT_SIDE_KEY] = settings.side
    with st.expander("落盘目录（--out）", expanded=False):
        out_text = st.text_input(
            "--out 根目录",
            value=bench_settings.widget_default("msc_out_text", str(default_out)),
            key="msc_out_text",
            help="每个阶段写入其子目录；父目录由 runner 自己建")
        out_dir = Path(out_text).expanduser()
        st.caption(f"解析为：{out_dir.resolve()}（每次点「开始」都按这里写）")
    st.caption(f"落盘目录：`{out_dir.resolve()}`（在「落盘目录」里可改）")
    return out_dir


# --------------------------------------------------------------------------- #
# ④ 模型目录
# --------------------------------------------------------------------------- #
def render_model_choice(models: list[Path], campaigns: list[Path]) -> Path | None:
    """④ 模型目录：默认最新的可用模型，也可以选「从零重训」。"""
    st.subheader("④ 模型目录（自动匹配）")
    options = ["从零重训（不提供模型）"] + [str(path) for path in models] + ["自定义路径"]
    # 上一轮的选项可能已经不在表里（模型目录被删 / 又新出一份）：带 key 的 selectbox
    # 在 state 不属于 options 时会在这里就抛异常，所以先清掉这个 widget 键。
    if st.session_state.get("msc_model_choice") not in (None, *options):
        st.session_state.pop("msc_model_choice", None)
    choice = st.selectbox("复用哪份物理预测模型", options=options,
                          index=1 if models else 0, key="msc_model_choice",
                          help="只列出含 fit/result.json 的目录（其余目录不是可用模型）")
    if choice == options[0]:
        st.session_state[_MODEL_KEY] = None
        st.caption("从零重训：capture → 名义初始化 → 拟合 → 几何重导出，多约 20 min 设备时间。")
        return None
    if choice == options[-1]:
        custom = st.text_input(
            "自定义模型目录",
            value=bench_settings.widget_default("msc_model_custom", ""),
            key="msc_model_custom")
        path = Path(custom).expanduser() if custom.strip() else None
    else:
        path = Path(choice)
    st.session_state[_MODEL_KEY] = str(path) if path is not None else None
    if path is not None:
        progress = collect_campaign_progress(path)
        started = [stage for stage in progress.stages if stage.seed_cv != MISSING]
        if started:
            first, last = started[0], started[-1]
            st.caption(f"新鲜度线索（读自该目录的 manifest）：{first.shape}/{first.stage} "
                       f"段首 CV {first.seed_cv} → {last.shape}/{last.stage} 段末 CV {last.final_cv}")
        else:
            st.caption("该目录里还没有可读的逐块读数（可能只跑完了模型阶段）。")
    return path


# --------------------------------------------------------------------------- #
# ⑤ 检查卡
# --------------------------------------------------------------------------- #
def render_checks(*, settings, out_dir: Path, model_dir: Path | None,
                  campaigns: list[Path], cam_size: int | None):
    """⑤ 运行前检查卡：六项结论 + 黄项勾选，返回卡片与勾选状态。"""
    st.subheader("⑤ 运行前检查")
    zero_xy, zero_source = _known_zero_xy(campaigns)
    if zero_xy is None:
        st.caption(f"零级位置：未知（{zero_source}）—— 几何检查只能报「未知」，"
                   "请先跑光路探针（scripts/slm/probe_optical_path_validity.py）。")
    else:
        st.caption(f"零级位置：({zero_xy[0]:.1f}, {zero_xy[1]:.1f}) sensor px · 出处 {zero_source}")

    budget = float(st.number_input(
        "设备占用预算（小时）", min_value=0.5, max_value=48.0, step=0.5,
        value=bench_settings.widget_default(_BUDGET_KEY, precheck.BUDGET_HOURS),
        key=_BUDGET_KEY,
        help="超了只告警不拦：这条问的是「今天还来不来得及」，不是「跑得对不对」"))
    card = precheck.build_check_card(
        settings=settings, out_dir=out_dir, model_dir=model_dir, zero_xy=zero_xy,
        recent_campaigns=campaigns, cam_size=cam_size, budget_hours=budget)
    flags = precheck.render_check_card(card, key_prefix="msc", flags=st.session_state.get(_ACK_KEY))
    st.session_state[_ACK_KEY] = flags
    return card


# --------------------------------------------------------------------------- #
# ⑥ 运行 / 逐块进度
# --------------------------------------------------------------------------- #
def _runner_kwargs(*, settings, slm_number: int, model_dir: Path | None,
                   resume: bool) -> dict[str, Any]:
    """页面 → ``execute_campaign`` 的关键字快照（点「开始」时在主线程算好）。

    ``CampaignSettings.kwargs()`` **只**含派生表里那些字段（``campaign.RUNNER_CLI_DEFAULTS``
    与形状/尺寸）。``model_dir`` / ``resume`` 不在表里 —— 前者由 ④ 单独选、后者由 ⑥ 的
    断点续跑开关给出 —— 所以必须在这里显式补上；少了它们 ``execute_campaign`` 会安静地
    用 click 默认值（``model_dir=None`` = 从零重训、``resume=False`` = 重跑已完成段），
    两种都是"看起来在续跑、其实白跑几小时"。

    名字照 runner 的 click 参数名写（``--model-dir`` → ``model_dir``、``--resume`` →
    ``resume``）；``tests/ao_shaping/gui/slm/test_multishape_campaign_ui.py`` 把这张快照
    的每个键都对着 ``slm-transfer-shapes`` 的参数表核一遍。
    """
    kwargs = dict(settings.kwargs())
    kwargs["slm_number"] = slm_number
    kwargs["model_dir"] = None if model_dir is None else Path(model_dir)
    kwargs["resume"] = bool(resume)
    return kwargs


def _start_run(*, out_dir: Path, settings, slm_number: int,
               model_dir: Path | None, resume: bool) -> None:
    """主线程：进入设备交接 → 起线程 → 记下运行体。

    这里**没有** ``try/except``：``with_slm_ownership`` 把 ``bench_busy`` 置真之后
    只做两件事 —— ``_stop_background_toggle`` 在它之前，``_release_slm_handle`` 自己
    吞掉 close 失败并只写会话键，所以"置真后抛错"没有实现路径。真要在那里加一步会抛
    的操作，该修的是那个上下文管理器（它的 ``finally`` 才是复位点），不是在这里补一层
    宽泛捕获把 ``bench_busy`` 再设一遍。
    """
    slot = SlmOwnership(slm_num=slm_number)
    ownership = with_slm_ownership(slm_number, slot)
    ownership.__enter__()
    kwargs = _runner_kwargs(settings=settings, slm_number=slm_number,
                            model_dir=model_dir, resume=resume)
    stop_event = threading.Event()
    out_q: queue.Queue = queue.Queue()
    thread = threading.Thread(
        target=_campaign_worker,
        kwargs={"out": out_dir, "kwargs": kwargs, "stop_event": stop_event, "out_q": out_q},
        name="multishape-campaign", daemon=True,
    )
    st.session_state[_RUN_KEY] = _CampaignRun(
        out=out_dir, shapes=tuple(settings.shapes), stop_event=stop_event, out_q=out_q,
        thread=thread, ownership=ownership, started_at=datetime.now().strftime("%H:%M:%S"))
    st.session_state[_SUMMARY_KEY] = None
    st.session_state[_ERROR_KEY] = None
    st.session_state[_OPTICAL_KEY] = None
    thread.start()
    logger.info("campaign 启动: {} ({} 形状, 边长 {} px, 续跑 {}, 模型 {})",
                out_dir, len(settings.shapes), settings.side, bool(resume),
                model_dir if model_dir is not None else "从零重训")


def _finish_run(run: _CampaignRun, kind: str, payload: Any) -> None:
    """主线程：退出设备交接（复位 bench_busy + 取回句柄），清运行体，记结果。

    ``try/finally`` 而不是 ``except``：收尾失败（重新打开 SLM 报错）**不能**吞掉，
    否则操作员只看到一行 log、页面还声称"已完成"。这里的 ``finally`` 保证运行体一定
    被清掉（页面下一轮还能用），异常本身照常浮到 Streamlit 上。
    """
    try:
        run.ownership.__exit__(None, None, None)
    finally:
        st.session_state.pop(_RUN_KEY, None)
        if kind == "done":
            st.session_state[_SUMMARY_KEY] = payload
        elif kind == "optical_path":
            # 光路无光不是"campaign 失败"：它是实验系统的一种可诊断状态，页面上
            # 单独一块（带 report() 原文与按 cause 的处置建议），不混进 error 横幅。
            st.session_state[_OPTICAL_KEY] = payload
        else:
            st.session_state[_ERROR_KEY] = payload


def _render_block_line(stage, planned: int | None = None) -> str:
    """当前段的一行逐块 CV 摘要（与报告 §3.2 同口径：CV 必须带曝光与饱和一起读）。

    ``planned`` 是**该阶段的计划块数**（``campaign_matrix`` 里那一行）。不能拿
    ``len(stage.blocks)`` 当分母：那是「已写下的块」，于是这一行会永远显示
    ``块 4/4``，把「跑到第 4 块（共 12 块）」读成「这一阶段做完了」。
    """
    blocks = stage.blocks
    if not blocks:
        return f"{stage.shape}/{stage.stage}：还没有块读数"
    last = blocks[-1]
    first_cv = blocks[0].cv
    total = len(blocks) if planned is None else planned
    return (f"{stage.shape}/{stage.stage} 块 {len(blocks)}/{total}"
            f" · 首块 CV {first_cv} → 末块 CV {last.cv}"
            f" · 曝光 {last.exposure_ms} ms · 饱和 {last.saturated_px} px"
            f" · 验收 {last.accepted}")


def journal_plan_shapes(progress) -> tuple[str, ...]:
    """这份目录**计划**跑哪些形状：journal 的 ``plan`` 条目（runner 开跑第一行就写）。

    目录里只留下已经跑过的形状，``collect_campaign_progress`` 也只看得到那些；所以
    「计划分母」必须来自别处 —— 开跑时的 ``plan.shapes`` 就是那份计划。没有 plan 条目
    （更早的目录）时返回空元组，由调用方决定退到哪。
    """
    for entry in reversed(progress.journal):
        plan = entry.get("plan") if isinstance(entry, dict) else None
        if isinstance(plan, dict) and plan.get("shapes"):
            return tuple(str(name) for name in plan["shapes"])
    return ()


def plan_shapes(progress) -> tuple[str, ...]:
    """历史浏览时的计划形状：journal 的 ``plan`` 条目，退到目录里出现过的形状。

    ⑦ 没有 ``settings`` 可问，只能这样；⑥ 用的是页面当前选的形状（见
    :func:`render_progress_panel` 的优先级）。
    """
    return journal_plan_shapes(progress) or tuple(
        dict.fromkeys(stage.shape for stage in progress.stages))


def stage_table_rows(progress) -> list[dict[str, Any]]:
    """段状态表的行。

    抽成函数是为了能单测「缺值不给 ``"—"``」这条：``st.dataframe`` 的 arrow 序列化不接受
    同一列混排 float 与字符串（见 :func:`_numeric`），而这条错误只会打到日志、界面直接少
    一张表。
    """
    return [{"形状": SHAPE_LABELS.get(stage.shape, stage.shape), "阶段": stage.stage,
             "状态": stage.status, "块": len(stage.blocks),
             "段首 CV": _numeric(stage.seed_cv), "段末 CV": _numeric(stage.final_cv),
             "核心峰值": _numeric(stage.peak_inside),
             "曝光 ms": _numeric(stage.exposure_ms)}
            for stage in progress.stages]


def block_table_rows(stage) -> list[dict[str, Any]]:
    """当前段的逐块表：``验收`` 来自 ``summary.json``（读取器已 join），其余照抄读数。"""
    return [{"块": block.index, "CV": _numeric(block.cv),
             "饱和 px": _numeric(block.saturated_px),
             "核心峰值": _numeric(block.core_peak),
             "曝光 ms": _numeric(block.exposure_ms),
             "验收": _numeric(block.accepted), "说明": block.note}
            for block in stage.blocks]


def render_progress_panel(out_dir: Path, *, running: bool,
                          shapes: "tuple[str, ...] | list[str] | None" = None) -> None:
    """逐段/逐块进度：总量进度条 + 段状态表 + 当前段逐块 + journal 流水。

    **分母来自计划，不来自读取器。** ``collect_campaign_progress`` 只枚举该目录里
    *已经出现* 的段，所以一份跑到一半的 campaign 会诚实地报「4/4 段」—— 那是本页最
    容易误读的一行（issue #215 的现场就是这个）。计划段数/块数一律由
    ``campaign.campaign_matrix(shapes)`` / ``matrix_totals`` 给出，读取器只提供**分子**。

    分母的优先级：目录 journal 里的 ``plan.shapes``（那是这份目录**实际按什么计划**跑的，
    换过勾选之后也仍然是对的）→ 传入的 ``shapes``（页面当前选择，新目录还没有 plan 时）
    → 目录里出现过的形状。
    """
    progress = collect_campaign_progress(out_dir)
    planned = (journal_plan_shapes(progress) or tuple(shapes or ())
               or plan_shapes(progress))
    rows = campaign_matrix(list(planned))
    plan = matrix_totals(rows)
    planned_blocks = {(row.shape, row.stage): row.blocks for row in rows}
    done_blocks = progress.totals["blocks_done"]
    done_segments = progress.totals["segments_done"]
    total_blocks = plan["blocks"] or progress.totals["blocks_total"]
    total_segments = plan["segments"] or progress.totals["segments_total"]
    st.progress(min(1.0, done_blocks / total_blocks) if total_blocks else 0.0,
                text=f"块 {done_blocks}/{total_blocks} · 段 {done_segments}/{total_segments}"
                     f" · 目录 {out_dir}")
    seen = progress.totals["segments_total"]
    if total_segments and seen < total_segments:
        st.caption(f"本目录里目前读到 {seen} 段（计划 {total_segments} 段）：分母取计划，"
                   "所以跑到一半不会显示成「全部完成」。")
    st.dataframe(stage_table_rows(progress), hide_index=True,
                 use_container_width=True)
    current = next((stage for stage in progress.stages
                    if stage.status not in COMPLETE_STATUSES and stage.blocks), None)
    if current is not None:
        planned_for_stage = planned_blocks.get((current.shape, current.stage))
        st.markdown(f"**当前段** · {_render_block_line(current, planned_for_stage)}")
        # ``验收`` 读的是 ``summary.json`` 的接受判定 —— 读取器已经把 manifest 与
        # summary 两个文件按块号连好了，这里不再自己 join 一遍（两份 join 迟早不一致）。
        st.dataframe(block_table_rows(current), hide_index=True,
                     use_container_width=True)
    with st.expander("阶段流水（journal.json）", expanded=False):
        render_progress(collect_progress(out_dir), running=running)


def _drain_queue(out_q: "queue.Queue[tuple[str, Any]]") -> list[tuple[str, Any]]:
    """非阻塞抽干结果队列（线程→主线程的唯一通道）。"""
    drained: list[tuple[str, Any]] = []
    while True:
        try:
            drained.append(out_q.get_nowait())
        except queue.Empty:
            return drained


@st.fragment(run_every=2.0)
def campaign_progress_fragment() -> None:
    """主线程轮询：抽队列 → 收尾 → 画进度。

    ``run_every=2.0``：一块 24–40 s，2 s 的刷新频率够用且不会把 ``manifest.json``
    读到一半（读取器本身对半截产物也降级，见 ``campaign_progress``）。

    **先抽队列、再看 ``stop_event``**：worker 的 ``finally`` 无论如何都会 set 它，
    反过来先判就会把"已完成"的那次回传跳过 —— 于是 ``_run`` 不被清、``bench_busy``
    不复位、侧栏永久点不动。顺序在这条链上不是风格问题。
    """
    run: _CampaignRun | None = st.session_state.get(_RUN_KEY)
    if run is None:
        return
    for message in _drain_queue(run.out_q):
        _finish_run(run, *message)
        st.rerun(scope="app")
        return
    if not run.thread.is_alive() and run.stop_event.is_set():
        # 队列空 + 线程已结束 = 结果丢了（解释器在 put 之后就中断之类）。worker 的
        # finally 保证 stop_event 已置，所以这个分支不会把正常收尾误判成失败。
        _finish_run(run, "error", "后台线程已结束但没有结果回传（看 GUI 日志）")
        st.rerun(scope="app")
        return

    running_label = "（已请求停止：当前段跑完即停）" if run.stop_event.is_set() else ""
    st.markdown(f"**运行中** · 启动于 {run.started_at} {running_label}")
    render_progress_panel(run.out, running=True, shapes=run.shapes)
    if st.button("停止（当前块结束后）", key="msc_stop",
                 disabled=run.stop_event.is_set()):
        run.stop_event.set()
        st.rerun(scope="fragment")


def _render_resume_control(out_dir: Path) -> bool:
    """⑥ 的「断点续跑」开关：``--out`` 里已经有 journal 时才出现，默认勾上。

    返回值就是稍后要交给 ``_runner_kwargs(resume=...)`` 的那个布尔量。没有
    ``journal.json`` 的目录返回 ``False``：新目录传不传都一样，但写死 ``False`` 比
    "碰巧没影响"更容易在日志里读懂。

    为什么这个开关不是可有可无的：``run_stages`` 里的 resume **只**跳过
    ``manifest.json`` 已存在的段，**不跳过**未完成的段；反过来，**不带** resume 时
    已存在的段目录会令 ``validate.run`` 抛 ``FileExistsError``（"新目录或失败"的旧契约）。
    所以"取消勾选"不是"重跑一遍"，而是"这个目录不能再用" —— 页面必须把这句话写在
    开关旁边，而不是让操作员在 20 分钟后从 traceback 里读出来。
    """
    completed = completed_segments(out_dir)
    if not completed:
        return False
    # 换目录时必须清掉带 key 的 checkbox：``value=`` 只在首次实例化生效，不清就会把
    # 上一档目录的勾选状态带到这里（该续的没续 => 立刻 FileExistsError）。
    if st.session_state.get(_RESUME_DIR_KEY) != str(out_dir):
        st.session_state.pop(_RESUME_KEY, None)
        st.session_state[_RESUME_DIR_KEY] = str(out_dir)
    resume = st.checkbox(
        f"从断点续跑：跳过已完成的 {len(completed)} 段（--resume）", value=True,
        key=_RESUME_KEY,
        help="manifest.json 已存在的段不再进段（journal 记 reused），其余段照跑；"
             "停下的段会从这里接着跑。取消勾选只适用于换一个新的 --out。")
    shown = "、".join(completed[:6]) + ("…" if len(completed) > 6 else "")
    st.caption(f"将复用：{shown}（共 {len(completed)} 段）。"
               "取消勾选不是在原目录重跑 —— runner 会在第一个已存在的段目录上报 "
               "FileExistsError，换目录请改 ③ 的 --out。")
    return bool(resume)


def render_optical_path(payload: dict[str, Any]) -> None:
    """光路无光（退出码 3）：``report()`` 原文 + 按 ``cause`` 的处置方向。

    这不是「campaign 失败」那一类横幅：光路无光是实验系统上的**预期**状态（有人挡光 /
    激光没出光 / 光路漂走了），页面该做的是把实测数字（各档曝光、峰值、该帧噪声底、
    counts/ms）和下一步直接摆在眼前，而不是让人从 traceback 里找。

    ``reason`` / ``cause`` / ``report`` 由 worker 从 :class:`OpticalPathInvalid` 原样搬来：
    库只在 ``run`` 里翻成退出码 3（见 ``transfer_shapes_runner``），所以后台线程拿到的
    仍是完整对象 —— 页面不必也不该自己重算 cause。
    """
    reason = str(payload.get("reason", MISSING))
    cause = str(payload.get("cause", MISSING))
    st.error(f"光路无光（退出码 {EXIT_OPTICAL_PATH_INVALID}）· reason={reason} · cause={cause}")
    st.code(str(payload.get("report", "")), language="text")
    if cause == CAUSE_EXPOSURE_CEILING:
        st.caption("先调 `AO_DAHENG_MAX_EXPOSURE_MS`（或相机曝光范围）：峰值随曝光在涨，"
                   "是会话的曝光上限把阶梯截短了 —— 不是没有光。")
    elif cause == CAUSE_BLOCKED:
        st.caption("先去看光路：峰值不随曝光变化 —— 挡光 / 激光未出光 / 光路漂走了。"
                   "加曝光补不回来。")
    else:
        st.caption(f"cause={cause}：先去看光路，再核曝光上限（两种原因的处置不同）。")


def render_run(*, settings, out_dir: Path, model_dir: Path | None, card,
               connected: bool) -> None:
    """⑥ 运行区：开始按钮（受检查卡/设备/占用三重门禁）+ 断点续跑 + 进度面板。"""
    st.subheader("⑥ 运行 / 逐块进度")
    run: _CampaignRun | None = st.session_state.get(_RUN_KEY)
    flags = st.session_state.get(_ACK_KEY) or {}
    blocked_reasons: list[str] = []
    if card.blocking:
        blocked_reasons.append("检查卡有红停项")
    if not card.acknowledged(flags):
        blocked_reasons.append("还有黄项未勾选")
    if not connected:
        blocked_reasons.append("未连接 SLM")
    if bench_settings.bench_busy():
        blocked_reasons.append("设备正被其它运算占用（bench_busy）")
    if run is not None:
        blocked_reasons.append("本页已有一份 campaign 在跑")
    if settings.blocking():
        blocked_reasons.append("参数结构性错误（见 ②）")

    resume = _render_resume_control(out_dir)
    st.button(
        "开始 campaign", type="primary", key="msc_start",
        disabled=bool(blocked_reasons),
        on_click=_start_run, kwargs={"out_dir": out_dir, "settings": settings,
                                    "slm_number": _active_slm_number(),
                                    "model_dir": model_dir, "resume": resume},
    )
    if blocked_reasons:
        st.caption("开始按钮不可用的原因：" + "；".join(blocked_reasons))
    else:
        st.caption(f"将写入 {out_dir}；运行期间侧栏的光路控件会被禁用（bench_busy）。")

    summary = st.session_state.get(_SUMMARY_KEY)
    error = st.session_state.get(_ERROR_KEY)
    optical = st.session_state.get(_OPTICAL_KEY)
    if optical:
        render_optical_path(optical)
    if error:
        st.error(f"campaign 失败：{error}")
    if summary:
        state = "已按段停下" if summary.get("stopped") else "已完成"
        st.success(f"{state}：{summary.get('stages')} 段 / {summary.get('blocks')} 块的计划"
                   f"（本次停在段边界时会留下 journal 的 stopped 标记）。"
                   f"过程记录：{Path(summary.get('out', out_dir)) / 'records'}"
                   f"（process.csv / process.h5 / process.gif）；逐块读数在 shape/<形状>/<阶段>/。")

    if run is not None:
        campaign_progress_fragment()
    elif out_dir.is_dir():
        # 没有在跑时照样画一遍落盘读数：进度是磁盘上的事实，不是内存里的状态。
        # 断线重开页面 / 上次中断后回来看，都走这一条。
        render_progress_panel(out_dir, running=False, shapes=settings.shapes)


def _active_slm_number() -> int:
    """本页只驱动一台 SLM：取已连接的最小号，没连时用 #1（此时按钮本来就禁用）。"""
    connected = connected_slm_numbers()
    return min(connected) if connected else 1


# --------------------------------------------------------------------------- #
# ⑦ 历史会话
# --------------------------------------------------------------------------- #
def render_history(campaigns: list[Path]) -> None:
    """⑦ 历史会话：挑一个 campaign 目录，离线看它的逐段/逐块读数。"""
    st.subheader("⑦ 历史会话")
    if not campaigns:
        st.caption(f"{campaign_root()} 下还没有 campaign（或都还没写 journal.json）。")
        return
    options = [str(path) for path in campaigns]
    # 目录被删过/挪过就先清 widget 键，否则 selectbox 会因 state 不在 options 里抛异常。
    if st.session_state.get("msc_history_choice") not in (None, *options):
        st.session_state.pop("msc_history_choice", None)
    choice = st.selectbox("选择 campaign 目录", options=options, index=0,
                          key="msc_history_choice")
    directory = Path(choice)
    render_progress_panel(directory, running=False)

# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main() -> None:
    """页面入口：``streamlit run src/ao_shaping/gui/slm/multishape_campaign_ui.py``。"""
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    ensure_state()
    render_slm_sidebar()

    st.title(PAGE_TITLE)
    st.caption("形状 + 尺寸是唯一必填输入；其余参数由自动辅助层派生并标注来源。"
               "计划、检查卡与历史浏览在无设备时同样可用。")
    connected = bool(connected_slm_numbers())
    if not connected:
        st.info("未连接 SLM：本页的实验计划与检查卡照常可用，「开始」按钮需要先在左侧连接 SLM。")

    campaigns = _campaign_dirs()
    shapes, side = render_targets()
    if not shapes:
        st.warning("至少选一个形状：形状 + 尺寸是这份 campaign 的唯一必填输入。")
        st.divider()
        # 不勾形状也要能翻历史：离线读数是本页的独立能力，不该被计划区的空白挡掉。
        render_history(campaigns)
        return

    settings = render_auto_assist(shapes, side)
    out_dir = render_plan(settings)
    model_dir = render_model_choice(_model_dirs(), campaigns)
    cam_size = int(bench_settings.bench_value("", "cam_size", 0) or 0) or None
    card = render_checks(settings=settings, out_dir=out_dir, model_dir=model_dir,
                         campaigns=campaigns, cam_size=cam_size)
    render_run(settings=settings, out_dir=out_dir, model_dir=model_dir, card=card,
               connected=connected)
    st.divider()
    render_history(campaigns)


if __name__ == "__main__":
    main()
