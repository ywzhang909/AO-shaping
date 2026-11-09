"""迁移学习闭环整形的 PySide6 桌面页。

页面形状 = 操作顺序: ① 设备 → ② 目标 → ③ 校验 → ④ 运行 → ⑤ 进度。①–④ 排成左侧
560 px 的可滚动窄列, ⑤ 固定在右侧 —— 会话动辄十几分钟, 操作员要一边盯阶段表一边回头
改参数, 不该为了看进度把表单滚走。

P1 只暴露必设参数
-----------------
表单只给相机类型、cam_id、SLM 编号、目标形状、目标短边/边长、倾斜相位周期与开关。其余字段 (探针数、
拟合迭代、按实测校正块数、验收口径、检查点路径……) 一律用 :class:`TransferFormValues` 的
默认值透传 —— 它们默认值已镜像 Streamlit 的 ``TRANSFER_RECIPE``, 少开一个控件就少一处
两套默认值漂移的机会。

线程: ``run_session`` 同步、动辄几十分钟, 所以会话不在 GUI 线程跑。本页自持一个
``maxThreadCount=1`` 的 :class:`QThreadPool` (同一台设备上并发跑两场只会互相踩设备),
由 :class:`_SessionTask` 在后台线程里**才** import 并调用它 —— 模块级 import 会把 torch
与整条管线拖进 GUI 冷启动。后台线程只经信号回主线程 (阶段 → ``stage``, 结束 →
``finished`` / ``failed``), 绝不碰控件: 跨线程改控件是 Qt 里最常见的崩法。

进度: ``on_stage`` 只说「阶段名 + 三态」, 说不出「这一阶段测出了什么」; 明细只能从会话
目录的 ``journal.json`` 与各阶段产物里读回来, :func:`collect_progress` 已把这层读法封成
「产物缺失也绝不抛异常」的纯函数。信号保证及时, 1 s 的 ``QTimer`` 保证内容。

校验: 控件一变就重跑 :func:`validate_form` (它内部惰性 import torch, 首次约数秒)。有
error 就禁用「开始运行」并把第一条写进 tooltip, warning 只提示。校验结论只来自 params
层, 页面不另写一套判据。

自检: 「设备自检」按钮显式触发一次只读硬件枚举 (相机列表 + 所选 SLM 探针), 结论落在
① 设备区。它**不**挂在 ``_revalidate`` 上 —— 校验是纯参数判断, 每改一个数字就跑一遍
硬件枚举既慢又会让「通过」变成噪音; 反过来, 设备参数改过之后旧结论只会标成「已过期」,
不会自动重跑。自检本身走 :class:`HardwareCheckTask` (后台线程), 与运行中的会话互斥。

自动探测: 页面**加载时**在后台跑一次 :class:`CcdDetectTask` —— 只枚举 Daheng /
MiiCam 两台相机 (不开相机、不探针 SLM), 把 ① 的相机类型与 cam_id 摆到真正在线的
那台上, 结论写进一条中性提示行。操作员已经改过设备字段就不覆盖; 探测失败 / 0 台
只写一行「保留默认」, 绝不阻断后面的完整流程 (``auto_detect_ccd=False`` 可整段关闭,
离屏测试走这条)。

读取零级: 「读取零级」按钮同样只在点击时发生, 走 :class:`_ZeroOrderTask` (后台线程,
相机 SDK 不在 UI 线程里调), 同样与自检 / 运行互斥。「目标边长」的控件上限因此不再
写死 400 px, 而是由实测 0 级到最近帧边的距离决定 (:mod:`~ao_shaping.gui.slm.transfer_desktop.zero_order`);
采样窗半跨随之按边长派生 (``params.effective_window_span``), 所以 ``side=900`` 这类
大目标能通过校验 —— 前提确实是 0 级离帧边够远。

关窗: ``run_session`` 没有取消接口, 所以关窗**不假装能停** —— 停轮询, 最多等
:data:`_CLOSE_GRACE_S` 秒让在飞作业落地 (期间泵事件队列, 让完成信号能送达), 超时就把
窗口关掉; 会话本身仍在后台线程里按计划跑完并写进会话目录。

配色: 状态色 (红 = 阻断 / 失败, 黄 = 提醒, 绿 = 通过 / 完成, 蓝 = 进行中) 按当前调色
板的明暗取两档, 并跟随 ``PaletteChange`` 重刷 —— 写死浅色字会在亮色主题里变成白底白字。
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

from loguru import logger
from PySide6.QtCore import QEvent, QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QColor, QFont, QIcon, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ao_shaping.gui.slm.transfer_desktop import campaign_plan, params
from ao_shaping.gui.slm.transfer_desktop.check_task import (
    CcdDetectTask,
    HardwareCheckTask,
)
from ao_shaping.gui.slm.transfer_desktop.live_view import LiveView
from ao_shaping.gui.slm.transfer_progress import (
    MISSING,
    STATUS_LABELS,
    collect_progress,
)
from ao_shaping.tools.slm.transfer import campaign_progress
from ao_shaping.utils.io.file import ROOT_DIR

__all__ = ["TransferPage"]

#: 表单默认值。单一口径取自 :mod:`params` (那边用字面量钉住了 Streamlit 的同名默认值)。
_DEFAULTS = params.TransferFormValues()

#: 进度轮询间隔 (ms)。``journal.json`` 同步落盘, 1 s 足够跟上分钟级的阶段。
_POLL_MS = 1000

#: 左侧表单列宽度 (px)。够放下「标签 + 控件」, 又不至于把右侧阶段表挤没。
_FORM_WIDTH = 560

#: 关窗时等在飞会话落地的上限 (s)。会话没有取消接口, 无限等会把宿主窗口冻住。
_CLOSE_GRACE_S = 5.0

#: G1 黄色提醒的勾选项最多显示多少字 (含「已确认 ·」前缀)。超出截断并补「…」,
#: 全文留在 tooltip 里; 34 字在 560 px 表单列里不会把一列撑宽。
_ACK_LABEL_MAX = 34

#: ⑦ 计划预览正文的最大高度 (px)。矩阵 + 合计 + 目录 + 16 行成本表自然高约 700 px,
#: 全摊开会把 ⑥ 实时画面挤到最小尺寸 —— 所以正文自己滚动, 只让出这一段高度。
_PLAN_BODY_MAX_HEIGHT = 430

#: ⑦ 成本表的列与列宽 (阶段列由内容定宽, 超宽时靠表格自身横向滚动)。
_COST_COLUMNS: tuple[str, ...] = (
    "形状",
    "阶段",
    "计划块数",
    "计划秒",
    "实测秒",
    "实测块数",
    "偏差",
)
_COST_WIDTHS: tuple[int, ...] = (68, 0, 74, 70, 74, 80, 62)

#: ⑦ 成本表的行高 (px)。16 行是「一览」, 比 ⑤ 的 26 px 更矮一档。
_COST_ROW_HEIGHT = 22

#: ⑦ 偏差列的首字 → 状态色: 快 = 省时间 (绿), 慢 = 要复核 (黄)。「≈计划」的首字不在
#: 表里, 于是保持默认字色 —— 正好踩在计划上不该染成告警色。
_DELTA_COLORS: dict[str, str] = {"快": "ok", "慢": "warn"}

#: ⑥ campaign 逐块页: journal 段状态 → 中文。未知状态原样显示, 绝不抛。
_CAMPAIGN_STATUS_LABELS: dict[str, str] = {
    "complete": "完成",
    "reused": "复用",
    "running": "运行中",
    "stopped": "已停",
}

#: 一棵树两级行: 段行 (缩进 0) 与块行 (缩进的子行) 列义不同, 表头就分两行写:
#: 上一行是段义, 下一行是块义。只写段义会让块行的「验收 / 说明」落在没有表头的列里,
#: 只写块义则「形状 / 阶段」被读成「块号」。曝光 ms 与后两列只属于一级, 单行居中。
_CAMPAIGN_HEADERS = (
    "形状 / 阶段\n块号",
    "状态\nCV",
    "seed CV → final CV\n饱和像素",
    "段内峰值\n核心峰值",
    "曝光 ms",
    "验收",
    "说明",
)

#: 段行 / 块行各列的口径, 挂在表头上 (表头比单元格更有地方说清楚)。
_CAMPAIGN_HEADER_TIPS = (
    "段行: 形状 / 阶段名；块行: 块号（缩进的子行）",
    "段行: journal 状态；块行: 该块 CV",
    "段行: 段首 CV → 段末 CV；块行: 饱和像素数（只印数, 不设阈值）",
    "段行: 核心 · 全幅峰值；块行: 该块核心峰值",
    "该段 / 该块的曝光时间",
    "块行: 数据层的验收判定（✓ 通过 / ✗ 未通过 / — 未写）",
    "块行: summary.json 里的说明；没有就印「—」",
)

_CAMPAIGN_HINT_UNSET = "在 ⑦ 填整轮目录后显示逐块读数"


def _fmt_num(value: object, spec: str) -> str:
    """数值 → 文字; 非数 / 布尔 / 格式不过的一律「—」, 绝不是 0。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return MISSING
    try:
        return spec.format(value)
    except (TypeError, ValueError):
        return MISSING


def _accept_mark(accepted: object) -> tuple[str, str]:
    """数据层的验收判定 → (符号, 配色键)。只照搬判定, 不另立阈值。"""
    if accepted is True:
        return "✓", "ok"
    if accepted is False:
        return "✗", "error"
    return MISSING, "muted"

#: 「目标边长」的保守上限 (px)。没有量过 0 级位置时只能给这个值 —— 采样窗默认
#: 半跨 220 px 扣掉两侧留白后正好 424 px, 400 是留了余量的保守档。
#: 点「读取零级」后按实测 0 级到帧边的距离放宽 (物理上限), 见 ``zero_order``。
_SIDE_MAX_CONSERVATIVE_PX = 400.0

#: 相机类型的中文说明。代码值与 ``CAMERA_CHOICES`` 一一对应。
_CAMERA_LABELS: dict[str, str] = {
    "daheng": "大恒（已验证）",
    "miicam": "MiiCam（已验证）",
    "sim": "仿真源（仅有接口）",
}

#: 状态色: (亮色主题, 暗色主题) 各一档 —— 写死浅色字会在亮色主题里变成「白底白字」。
_STATUS_COLORS: dict[str, tuple[str, str]] = {
    "error": ("#b3261e", "#ff8a80"),
    "failed": ("#b3261e", "#ff8a80"),
    "warn": ("#8a5a00", "#ffcc66"),
    "ok": ("#1a7f37", "#7ee0a1"),
    "complete": ("#1a7f37", "#7ee0a1"),
    "running": ("#0b5cad", "#7cbcff"),
    "pending": ("#5f6368", "#9aa0a6"),
    "skipped": ("#5f6368", "#9aa0a6"),
}

#: ``CheckItem.level`` → 状态色档位。``fail`` 必须显式映射到 ``error``:
#: ``_STATUS_COLORS`` 里没有 ``fail`` 这一档, 直接查表会静默退成 ``pending`` 的灰。
_CHECK_LEVELS: dict[str, str] = {"ok": "ok", "warn": "warn", "fail": "error"}


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _palette_colors(palette: QPalette) -> dict[str, str]:
    """按调色板明暗挑一档状态色。"""
    dark = palette.color(QPalette.ColorRole.Window).lightness() < 128
    return {kind: pair[1 if dark else 0] for kind, pair in _STATUS_COLORS.items()}


def _rich(text: str) -> str:
    """纯文本结论 → 富文本片段 (转义, 保留结论里的换行)。"""
    return escape(text).replace("\n", "<br>")


def _duration_text(seconds: float | None) -> str:
    """阶段耗时 → 表格文字。``None`` 是「没测到」, 不是 0 s。"""
    return MISSING if seconds is None else f"{seconds:.1f} s"


def _stage_summary(stage: Mapping[str, Any]) -> str:
    """阶段表末列: 失败给错误, 否则把关键指标压成一行。"""
    error = stage.get("error")
    if error:
        return " ".join(str(error).split())
    metrics = stage.get("metrics") or ()
    return " · ".join(f"{metric['label']} {metric['text']}" for metric in metrics)


def acceptance_verdict_line(session_dir: Path) -> str:
    """会话目录 → 一行验收阈值结论 (缺失产物绝不抛)。

    核心 lane 的 :func:`~ao_shaping.tools.slm.transfer.workflow.acceptance_verdict`
    只读 ``session.json`` / ``rounds.json`` 等文件得出 ``{metric, lower, upper,
    measured_correlation, measured_relative_l2, accepted, rounds}``；是否开过验收
    循环则看 ``session.json`` 里 ``resolved.acceptance == "looped"``。本函数把
    两者压成给操作员看的一行字。目录不存在、产物缺失或该会话根本没开验收循环时
    返回空串（调用方自行决定何时显示）。
    """
    if not session_dir or not Path(session_dir).is_dir():
        return ""
    from ao_shaping.tools.slm.transfer.workflow import acceptance_verdict

    # 只开过 ABBA 交错测量 或纯模型验收的会话不予显示; 只有 acceptance=="looped" 才算
    # 开了验收阈值循环。
    enabled = False
    try:
        cfg = json.loads(Path(session_dir).joinpath("session.json").read_text(encoding="utf-8"))
        enabled = cfg.get("resolved", {}).get("acceptance") == "looped"
    except Exception:  # policy: 会话产物缺失/损坏时不显示该行，绝不把完成界面打崩
        return ""
    if not enabled:
        return ""
    try:
        verdict = acceptance_verdict(session_dir)
    except Exception:  # 读文件失败也绝不把完成界面打崩
        logger.warning("读取验收阈值结论失败: {}", session_dir)
        return ""
    accepted = bool(verdict.get("accepted"))
    rounds = int(verdict.get("rounds") or 0)
    corr = verdict.get("measured_correlation")
    l2 = verdict.get("measured_relative_l2")
    if not accepted and (corr is None or l2 is None):
        return ""
    tail = "通过" if accepted else "未达标至上限后停止"
    lower = verdict.get("lower")
    upper = verdict.get("upper")
    metrics = []
    for label, value, bound, is_lower in (
        ("相关性", corr, lower, True),
        ("相对 L2", l2, upper, False),
    ):
        if value is None:
            continue
        if bound is None:
            try:
                metrics.append(f"{label} {float(value):.3f}")
            except (TypeError, ValueError):
                metrics.append(f"{label} —")
            continue
        metrics.append(_metric_with_shortfall(label, value, bound, lower_bound=is_lower))
    if not metrics and not accepted:
        return ""
    return "验收阈值: " + " · ".join(metrics + [f"{rounds} 轮"]) + f" · {tail}"


def _metric_with_shortfall(
    label: str, value, bound, *, lower_bound: bool
) -> str:
    """格式化单个验收指标; 未达标时追加「（还差 X · 阈值 Y）」。

    ``lower_bound=True`` 表示指标 >= 阈值才算达标 (相关性下界), False 表示
    <= (相对 L2 上界)。值 / 阈值无法解析时退化为纯值或 ``—``。
    """
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return f"{label} —"
    text = f"{label} {value_f:.3f}"
    try:
        bound_f = float(bound)
    except (TypeError, ValueError):
        return text
    if lower_bound:
        if value_f >= bound_f:
            return text
        deficit = bound_f - value_f
    else:
        if value_f <= bound_f:
            return text
        deficit = value_f - bound_f
    return f"{text}（还差 {deficit:.3f} · 阈值 {bound_f:.3f}）"


def acceptance_progress_line(session_dir: Path) -> str:
    """运行中 (非完成) 的验收阈值实时读数; 未开循环 / 无轮次 / 已达标返回空串。

    取 ``rounds.json`` 最后一轮的实测值, 未达标时给出「当前 X · 还差 Y」,
    便于操作员在循环进行时看见还差多少。绝不抛异常。
    """
    if not session_dir or not Path(session_dir).is_dir():
        return ""
    try:
        cfg = json.loads(Path(session_dir).joinpath("session.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(cfg, dict):
        return ""
    if cfg.get("resolved", {}).get("acceptance") != "looped":
        return ""
    from ao_shaping.tools.slm.transfer.workflow import acceptance_verdict

    try:
        verdict = acceptance_verdict(session_dir)
    except Exception:  # policy: 阈值读数是展示层附加信息, 读不到只隐藏该行, 不打断面板
        logger.warning("读取验收阈值实时读数失败: {}", session_dir)
        return ""
    rounds = int(verdict.get("rounds") or 0)
    if rounds <= 0 or bool(verdict.get("accepted")):
        return ""
    corr = verdict.get("measured_correlation")
    l2 = verdict.get("measured_relative_l2")
    lower = verdict.get("lower")
    upper = verdict.get("upper")
    parts = []
    if corr is not None and lower is not None:
        parts.append(_metric_with_shortfall("相关性", corr, lower, lower_bound=True))
    if l2 is not None and upper is not None:
        parts.append(_metric_with_shortfall("相对 L2", l2, upper, lower_bound=False))
    if not parts:
        return ""
    return f"验收阈值（第 {rounds} 轮）: " + " · ".join(parts)


def reuse_latest_verdict_line(session_dir: Path) -> str:
    """会话目录 → 一行光路复用结论 (缺失产物 / 未开启绝不抛)。

    ``run_session`` 的复用快路径把判定写进 ``<out>/optical_probe.json``（键
    ``serial`` / ``latest`` / ``same_path`` / ``verdict`` / ``current`` /
    ``reference``）。目录不存在、产物缺失或当前会话根本没走复用 (没写该文件)
    时返回空串 (调用方自行决定何时显示)。

    通过时一行简短结论; 未通过时**列出每一条不通过的原因** (0 级零点漂移 /
    相机曝光上界 / 取图格式 / 像素位深 / SLM 序列号 / 无 latest 参照),
    便于操作员定位是哪一项光路变了。
    """
    if not session_dir or not Path(session_dir).is_dir():
        return ""
    probe = Path(session_dir).joinpath("optical_probe.json")
    if not probe.is_file():
        return ""
    try:
        data = json.loads(probe.read_text(encoding="utf-8"))
    except Exception:  # policy: 光路复用结论产物缺失/损坏时降级为空行
        logger.warning("读取光路复用结论失败: {}", session_dir)
        return ""
    same = bool(data.get("same_path"))
    if data.get("skipped"):
        # 光路复用被跳过 (未开启 / 探针失败 / 无序列号 / 无 latest): 说明原因。
        from ao_shaping.tools.slm.transfer.optical_probe import format_probe_skip_line

        reason = data.get("reason")
        return format_probe_skip_line(reason if isinstance(reason, str) else "")
    if same:
        return "光路复用：一致 → 已复用最近一次模型；差异已并入本场"
    # 未采用: 列出每一条不通过的原因 (尽量详细)。
    reason = _reuse_latest_failure_reason(data)
    if reason:
        return f"光路复用：未采用（{reason}） → 走正常全流程"
    # 兜底: 用 workflow 写入的 verdict 字符串 (可能只有粗粒度文案)。
    verdict = data.get("verdict")
    if isinstance(verdict, str) and verdict.strip():
        return f"光路复用：{verdict.strip()}"
    return "光路复用：已变化 → 走正常全流程"


def _reuse_latest_failure_reason(data: dict) -> str:
    """从 ``optical_probe.json`` 的 current/reference 探针里推断不通过的**具体原因**。

    每条 reason 对应 ``same_path`` 的一个判据 (零点 / 曝光 / 格式 / 位深 /
    序列号), 任一项不满足就 listed。绝不抛异常; 读不到字段时给「不可比」。
    """
    try:
        from ao_shaping.tools.slm.transfer.optical_probe import (
            DEFAULT_TOL_EXPOSURE_FACTOR,
            DEFAULT_TOL_ZERO_PX,
        )
    except Exception:  # policy: 探针默认容差是可选依赖, 拿不到就按「不可比」隐藏该行
        return ""
    cur = data.get("current")
    ref = data.get("reference")
    serial = data.get("serial")
    if not isinstance(cur, dict) or not isinstance(ref, dict):
        if ref is None:
            return f"无 latest 参照模型 (SLM 序列号 {serial or '—'} 下没有已保存的 latest，或探针失败)"
        return "当前 / 参照探针数据不完整，无法比对"
    reasons: list[str] = []
    # SLM 序列号
    cur_serial = cur.get("slm_serial")
    if cur_serial is not None and serial is not None and str(cur_serial) != str(serial):
        reasons.append(f"SLM 序列号不同 (当前 {cur_serial} vs 参照 {serial})")
    # 0 级光斑零点
    try:
        cx, cy = cur["zero_xy"]
        rx, ry = ref["zero_xy"]
        dx, dy = abs(float(cx) - float(rx)), abs(float(cy) - float(ry))
        if dx > DEFAULT_TOL_ZERO_PX or dy > DEFAULT_TOL_ZERO_PX:
            reasons.append(
                f"0 级光斑零点漂移 Δx={dx:.1f}px Δy={dy:.1f}px "
                f"(容忍 {DEFAULT_TOL_ZERO_PX}px)"
            )
    except (KeyError, TypeError, ValueError, IndexError):
        reasons.append("0 级光斑零点不可比")
    # 相机曝光上界
    try:
        cur_hi = float(cur["exposure_range_ms"][1])
        ref_hi = float(ref["exposure_range_ms"][1])
        if ref_hi > 0 and not (
            ref_hi / DEFAULT_TOL_EXPOSURE_FACTOR
            <= cur_hi
            <= ref_hi * DEFAULT_TOL_EXPOSURE_FACTOR
        ):
            reasons.append(
                f"相机曝光上界 {cur_hi:.3f}ms vs 参照 {ref_hi:.3f}ms "
                f"(容忍 ×{DEFAULT_TOL_EXPOSURE_FACTOR})"
            )
    except (KeyError, TypeError, ValueError, IndexError):
        reasons.append("相机曝光不可比")
    # 取图格式
    cf_cur = cur.get("camera_format")
    cf_ref = ref.get("camera_format")
    if cf_cur is None or cf_ref is None or cf_cur != cf_ref:
        reasons.append(f"相机取图格式 {cf_cur!r} vs 参照 {cf_ref!r}")
    # 像素位深
    bd_cur = cur.get("bit_depth")
    bd_ref = ref.get("bit_depth")
    if bd_cur is None or bd_ref is None or bd_cur != bd_ref:
        reasons.append(f"像素位深 {bd_cur!r} vs 参照 {bd_ref!r}")
    if not reasons:
        reasons.append("探针参数不可比")
    return "；".join(reasons)


#: 本场会话里模型权重 (冻结参数) 的候选位置, 按优先级排列 (相对会话目录)。
_WEIGHT_GLOBS = (
    "model*/frozen_parameters.json",
    "fit*/frozen_parameters.json",
    "initial/frozen_parameters.json",
)


def _newest_file(paths: Iterable[Path]) -> Path | None:
    """mtime 最新的一个文件; 全读不到 (不存在 / 无权限) 时返回 ``None``。"""
    newest: Path | None = None
    newest_mtime = float("-inf")
    for path in paths:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime > newest_mtime:
            newest, newest_mtime = path, mtime
    return newest


def model_weight_saved_line(session_dir: Path) -> str:
    """会话目录 → 一行「所用模型权重 · 保存于 …」(缺产物返回空串, 绝不抛)。

    日期口径是权重文件在文件系统里的 mtime。复用光路时本场没拟合新模型, 用的是
    ``<models_root>/<serial>/latest`` 里那一份 (见 ``optical_probe.json`` 的
    ``latest`` / ``same_path``), 所以优先读它; 否则取本场 ``model*`` → ``fit*``
    → ``initial`` 里最新的 ``frozen_parameters.json``。
    """
    root = Path(session_dir) if session_dir else None
    if root is None or not root.is_dir():
        return ""
    weight: Path | None = None
    probe = root / "optical_probe.json"
    if probe.is_file():
        try:
            data = json.loads(probe.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        latest = data.get("latest")
        if data.get("same_path") and latest:
            candidate = Path(str(latest)) / "frozen_parameters.json"
            if candidate.is_file():
                weight = candidate
    if weight is None:
        for pattern in _WEIGHT_GLOBS:
            weight = _newest_file(p for p in root.glob(pattern) if p.is_file())
            if weight is not None:
                break
    if weight is None:
        # 兜底: 更深一层子目录 (如 ``model*/run/`` 之类) 里的冻结参数。
        weight = _newest_file(
            p for p in root.glob("**/frozen_parameters.json") if p.is_file()
        )
    if weight is None:
        return ""
    try:
        shown = weight.relative_to(root).as_posix()
    except ValueError:
        # 复用来的 latest 在 models_root 下, 不在本场目录: 用末三级定位。
        parts = list(weight.parts)
        shown = "/".join(parts[-3:]) if len(parts) >= 3 else str(weight)
    # 权重文件在但读不出/不可解析 → 明确的「模型加载失败」而不是空行。
    try:
        mtime = weight.stat().st_mtime
        json.loads(weight.read_text(encoding="utf-8"))
    except OSError:
        return f"模型加载失败：{shown} 读取失败"
    except ValueError:  # JSONDecodeError / UnicodeDecodeError 都是 ValueError
        return f"模型加载失败：{shown} 损坏或不可解析"
    saved = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    return f"模型权重：{shown} · 保存于 {saved}"


def _section(title: str) -> tuple[QGroupBox, QVBoxLayout]:
    """带序号的分组框 (序号就是操作顺序) + 内边距已设好的竖排布局。"""
    box = QGroupBox(title)
    font = box.font()
    font.setWeight(QFont.Weight.DemiBold)
    box.setFont(font)
    layout = QVBoxLayout(box)
    layout.setContentsMargins(10, 8, 10, 10)
    layout.setSpacing(8)
    return box, layout


def _bold(widget: QWidget) -> QWidget:
    """把一个控件刷成强调字重 (返回同一个控件, 便于链式使用)。"""
    font = widget.font()
    font.setWeight(QFont.Weight.DemiBold)
    widget.setFont(font)
    return widget


# ---------------------------------------------------------------------------
# 后台会话
# ---------------------------------------------------------------------------


class _WorkerSignals(QObject):
    """后台会话 → 主线程的单向通道 (``QRunnable`` 不是 ``QObject``, 发不出信号)。"""

    stage = Signal(str, str)
    #: 逐帧的实时画面 (``dict`` 负载, 含 ``frame`` 数组与整数进度); 每次 CCD 读出都发。
    frame = Signal(dict)
    finished = Signal(str)
    failed = Signal(str)


class _SessionTask(QRunnable):
    """在后台线程跑一整场迁移学习会话。"""

    def __init__(
        self,
        values: params.TransferFormValues,
        out: Path,
        should_stop: object | None = None,
    ) -> None:
        super().__init__()
        self._values = values
        self._out = out
        self._should_stop = should_stop
        self.signals = _WorkerSignals()

    def run(self) -> None:
        """线程池入口 (在后台线程执行)。"""
        # 只在后台线程 import: 模块级 import 会拖起 torch 与整条管线。
        from ao_shaping.tools.slm.transfer.workflow import run_session

        try:
            session = params.build_session(self._values, out=self._out)
            run_session(
                session,
                on_stage=self._on_stage,
                on_frame=self._on_frame,
                should_stop=self._should_stop,
            )
        # policy: 后台线程的任何失败都必须变成界面上的结论; 这里只发信号、不吞细节,
        # 也不让异常逃出 QRunnable (Qt 会直接终止线程, 界面就永远停在「运行中」)。
        except Exception as exc:
            logger.warning("迁移学习会话失败: {}", exc)
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.signals.finished.emit(str(self._out))

    def _on_stage(self, name: str, status: str) -> None:
        """``run_session`` 的阶段回调 (后台线程): 只发信号, 不碰控件。"""
        self.signals.stage.emit(name, status)

    def _on_frame(self, payload: dict) -> None:
        """``run_session`` 的帧回调 (后台线程): 只发信号, 不碰控件。"""
        self.signals.frame.emit(payload)


class _ZeroSignals(QObject):
    """「读取零级」后台任务 → 主线程 (``QRunnable`` 自己发不出信号)。"""

    #: ``ZeroOrderReading`` 或异常实例 —— 全暗 / 贴边也是要上屏的结论。
    done = Signal(object)


class _ZeroOrderTask(QRunnable):
    """在后台线程开一次相机读 0 级 (**只读**, 不碰 SLM)。

    打开相机、改曝光、抓帧都是 SDK 调用, 快则几十毫秒、卡住则秒级; 落在 UI
    线程会把窗口冻住。与 :class:`~ao_shaping.gui.slm.transfer_desktop.check_task.HardwareCheckTask`
    同一套做法: 工作线程只发信号。
    """

    def __init__(self, camera: str, cam_id: int) -> None:
        super().__init__()
        self._camera = camera
        self._cam_id = cam_id
        self.signals = _ZeroSignals()

    def run(self) -> None:
        """线程池入口 (在后台线程执行)。"""
        # 惰性 import: ``zero_order`` 会拉起相机 SDK, 页面冷启动不该先付这笔
        # 代价; 也是测试 patch ``zero_order.read_zero_order`` 的接缝。
        from ao_shaping.gui.slm.transfer_desktop.zero_order import read_zero_order

        try:
            reading = read_zero_order(self._camera, self._cam_id)
        # policy: 工作线程不碰界面; 任何失败都变成一条回到主线程的结论
        except Exception as exc:
            logger.opt(exception=exc).warning("读取零级失败")
            self.signals.done.emit(exc)
        else:
            self.signals.done.emit(reading)


#: 在飞会话的强引用。页面先关、线程后完时任务对象不能被回收, 否则后台线程还在
#: 跑的对象会先消失。平时由页面的 ``_task`` 持有, 这里只兜关窗那一段。
_LIVE_TASKS: set[_SessionTask] = set()


def _brief_exception(exc: BaseException, *, limit: int = 160) -> str:
    """一行异常摘要 (类型 + 消息), 压掉换行并限长 —— 只给提示行用。

    放在本模块而不是复用 ``hardware_check._exc_summary``: 那个模块顶层会拉起相机
    SDK, 页面级 import 必须保持惰性, 而这里只需要四行字符串处理。
    """
    text = " ".join(f"{type(exc).__name__}: {exc}".split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ---------------------------------------------------------------------------
# 页面
# ---------------------------------------------------------------------------


class TransferPage(QWidget):
    """迁移学习闭环整形的桌面页 (设备 → 目标 → 校验 → 运行 → 进度)。"""

    def __init__(
        self, parent: QWidget | None = None, *, auto_detect_ccd: bool = True
    ) -> None:
        super().__init__(parent)
        _icon = ROOT_DIR / "camera_rotate.ico"
        if _icon.exists():
            self.setWindowIcon(QIcon(str(_icon)))
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)

        self._task: _SessionTask | None = None
        self._running = False
        self._session_dir: Path | None = None
        # 验收阈值循环的「在两轮间停止」旗标。只被 Stop 按钮置位、每次启动复位;
        # 后台线程通过 ``should_stop=self._acceptance_stop.is_set`` 在两轮之间查询。
        # 它是 threading.Event, 不是 Qt 信号 —— 跨线程「只读布尔」用标准库最稳。
        self._acceptance_stop = threading.Event()
        # 自检只在点按钮时发生。``_check_task`` 是强引用 (与 ``_task`` 同理, 掉引用
        # 会让在飞任务被回收); 结论存成纯数据, 换主题时靠它重上色。
        self._checking = False
        self._check_task: HardwareCheckTask | None = None
        self._check_key: tuple[str, int, int] | None = None
        self._check_at: float = 0.0
        self._check_reused = False
        self._check_heading: tuple[str, str] | None = None
        self._check_rows: list[tuple[str, str, str]] = []
        # 加载时的相机自动探测: ``_ccd_task`` 是强引用 (掉引用会让在飞任务被回收);
        # ``_ccd_preferred`` / ``_ccd_cam_id`` 记下任务启动时控件里的值, 回来时据此
        # 判断操作员有没有自己改过 —— 改过就不覆盖。``_probe_text`` 与提示行同源,
        # 保证 ``ccd_probe_summary()`` 永远返回一个字符串。
        self._ccd_task: CcdDetectTask | None = None
        self._ccd_preferred: str = str(_DEFAULTS.camera)
        self._ccd_cam_id: int = int(_DEFAULTS.cam_id)
        self._probe_text: str = ""
        # 「读取零级」同理只在点按钮时发生; ``_zero_task`` 是强引用, 掉引用会让
        # 在飞任务被回收 (信号永远送不到)。``_zero_reading`` 同时是互斥标志。
        self._zero_reading = False
        self._zero_task: _ZeroOrderTask | None = None
        self._zero_limit: int = int(_SIDE_MAX_CONSERVATIVE_PX)
        #: ② 目标区常驻说明的 (色档, 正文)；``None`` = 还没读过零级。
        self._zero_state: tuple[str, str] | None = None
        self._last_errors: list[str] = []
        self._last_warnings: list[str] = []
        # 第三通道：中性说明（例如采集网格自动升档到多少）。既不是阻塞也不是需要
        # 确认的提醒，所以单独一条颜色，不混进黄字里稀释真提醒的分量。
        self._last_notes: list[str] = []
        # G1: 黄色提醒的逐条「已确认」勾选。控件在 ``_build_checks`` 里才建好,
        # 先给空表 —— 公开取值器可能早于建控件被调用。
        self._ack_checks: list[QCheckBox] = []
        self._status_kind = "pending"
        self._status_message = "尚未运行。参数自检通过后即可启动。"
        self._colors: dict[str, str] = _palette_colors(self.palette())

        # ⑦ 计划预览的状态: 控件在 ``_build`` 里才建好, 所以先摆一份「还没建」的默认值
        # —— 公开取值器与主题重刷都可能早于建控件被调用, 不能靠属性是否存在来兜。
        self._plan_ready = False
        self._plan_matrix_text = ""
        self._plan_summary_text = ""
        self._cost_path: Path | None = None
        self._cost_rows: Sequence[campaign_plan.CostRow] = ()
        self._cost_caption_text = ""
        self._cost_note_text = ""
        # ⑥ campaign 逐块页: 与 ⑦ 共用 ``self._cost_dir`` 同一个目录, 不另开输入框。
        self._campaign_progress: campaign_progress.CampaignProgress | None = None
        self._campaign_status_text = ""
        self._campaign_header_text = ""

        self._poll = QTimer(self)
        self._poll.setInterval(_POLL_MS)
        self._poll.timeout.connect(self._refresh_progress)

        # 控件在 ``_build`` 里逐个创建, 而创建途中的 ``setValue`` 会触发已连好的
        # ``valueChanged → _revalidate``; 此时后面的控件还不存在。在表单搭完前
        # ``_revalidate`` 直接跳过, 搭完后再手动跑一次首次校验。
        self._form_ready = False
        self._build()
        self._form_ready = True
        self._apply_theme()
        # 先把「矩形长边隐藏 / 倾斜相位周期禁用」的初始态摆正, 再跑首次校验。
        self._sync_shape_widgets()
        self._sync_tilt()
        # 首次校验会 import torch (数秒), 但按钮状态必须一开始就是真的:
        # 同步校验, 免得「能点」被读成「参数没问题」。
        self._revalidate()
        # 自动探测放在最后: 线程池与控件都已就绪, 结论回来时页面已是可用状态。
        if auto_detect_ccd:
            self._start_ccd_detection()
        else:
            # 关掉探测时提示行留白并收起 —— 与「探测不可用」是同一套降级, 不影响后面流程。
            self._probe_label.setText("")
            self._probe_label.setVisible(False)

    # ------------------------------------------------------------------ 布局
    def _build(self) -> None:
        """搭出「标题 + 左表单 / 右进度」两栏。"""
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(12)
        root.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setSpacing(16)
        body.addWidget(self._build_form(), 0)
        body.addWidget(self._build_right(), 1)
        root.addLayout(body, 1)

    def _build_right(self) -> QWidget:
        """右栏: ⑥ 实时画面 (CCD / 剖面 / 块进度) + ⑤ 阶段进度表 + ⑦ 计划预览。"""
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 两个页签: 「运行」放 ⑥ 实时画面 + ⑤ 阶段进度 (跑起来看这里, 不被计划预览挤压);
        # 「计划预览」放 ⑦ (开跑前看规模 / 对历史实测, 独立占满整栏)。
        run_host = QWidget()
        run_layout = QVBoxLayout(run_host)
        run_layout.setContentsMargins(0, 8, 0, 0)
        run_layout.setSpacing(12)

        box, live_layout = _section("⑥ 实时画面（每帧刷新）")
        # 两页: 单会话实时画面 (默认, 不自动切换) / campaign 逐块 (只读 ⑦ 的目录)。
        self._live_tabs = QTabWidget()
        self._live_view = LiveView()
        self._live_tabs.addTab(self._live_view, "单会话")
        self._live_tabs.addTab(self._build_campaign_page(), "整轮逐块")
        live_layout.addWidget(self._live_tabs)
        run_layout.addWidget(box, 1)
        run_layout.addWidget(self._build_progress(), 1)

        self._right_tabs = QTabWidget()
        self._right_tabs.addTab(run_host, "运行")
        self._right_tabs.addTab(self._build_plan_preview(), "计划预览")
        layout.addWidget(self._right_tabs, 1)
        return host

    def _build_campaign_page(self) -> QWidget:
        """⑥ 第二页「campaign 逐块」: 页头一行 + 段 / 块两级树。

        只读 ⑦ 的目录 (``self._cost_dir``); 本页自己不开目录框, 也不下任何阈值判断。
        """
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(6)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)
        self._campaign_header = QLabel("")
        self._campaign_header.setWordWrap(True)
        head.addWidget(self._campaign_header, 1)
        self._campaign_refresh = QPushButton("刷新")
        self._campaign_refresh.setToolTip("重读 ⑦ 填的整轮目录（不开设备）")
        self._campaign_refresh.clicked.connect(self.reload_campaign_progress)
        head.addWidget(self._campaign_refresh, 0)
        layout.addLayout(head)

        # 降级文案 (无目录 / 不存在 / 空 …) 单独一行, 树为空时才显示。
        self._campaign_status = QLabel()
        self._campaign_status.setWordWrap(True)
        layout.addWidget(self._campaign_status)

        self._campaign_tree = QTreeWidget()
        self._campaign_tree.setColumnCount(len(_CAMPAIGN_HEADERS))
        self._campaign_tree.setHeaderLabels(list(_CAMPAIGN_HEADERS))
        header = self._campaign_tree.header()
        for column, tip in enumerate(_CAMPAIGN_HEADER_TIPS):
            self._campaign_tree.headerItem().setToolTip(column, tip)
        # 首列 (段名 / 块号) 在铺完行后按内容定宽 (见 _paint_campaign_progress 收尾);
        # 说明列吃掉余宽 (挤窄了由 tooltip 兜全文), 其余列按内容。
        for column, mode in (
            (0, QHeaderView.ResizeMode.Interactive),
            (6, QHeaderView.ResizeMode.Stretch),
        ):
            header.setSectionResizeMode(column, mode)
        for column in range(1, 6):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self._campaign_tree.setRootIsDecorated(True)
        self._campaign_tree.setAlternatingRowColors(True)
        self._campaign_tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._campaign_tree.setUniformRowHeights(True)
        self._campaign_tree.setMinimumHeight(120)
        layout.addWidget(self._campaign_tree, 1)
        self._campaign_ready = True
        self._set_campaign_status(_CAMPAIGN_HINT_UNSET)
        return host

    def _set_campaign_status(self, text: str) -> None:
        """降级一行: 文字 + 次级色; 空串表示树里有内容, 行隐藏。"""
        self._campaign_status_text = text
        self._campaign_status.setText(text)
        self._campaign_status.setStyleSheet(f"color: {self._muted()};")
        self._campaign_status.setVisible(bool(text))

    def reload_campaign_progress(self) -> None:
        """重读 ⑦ 目录里的 campaign 逐块进度 (只读文件, 不开设备、不弹错误框)。

        只在点「刷新」「读取实测」或程序方式 ``set_campaign_dir`` 时发生 —— 与 ⑦ 同一条
        「不按键读盘」规则。任何读盘失败都退成一行说明, 树清空。
        """
        if not getattr(self, "_campaign_ready", False):
            return
        raw = self._cost_dir.text().strip()
        path: Path | None = Path(raw).expanduser() if raw else None
        if path is None:
            self._show_campaign_message(_CAMPAIGN_HINT_UNSET)
            return
        try:
            if not path.exists():
                self._show_campaign_message(f"目录不存在：{raw}")
                return
            if not path.is_dir():
                self._show_campaign_message(f"不是目录：{raw}")
                return
            progress = campaign_progress.collect_campaign_progress(path)
        except (OSError, TypeError, ValueError, AttributeError):  # 数据层承诺不抛; 显示层兜底
            self._show_campaign_message("读取失败：无法读取该目录的逐块进度")
            return
        if not progress.stages:
            self._show_campaign_message(f"目录里还没有段级产物：{path.name}")
            return
        self._campaign_progress = progress
        self._paint_campaign_progress()

    def _show_campaign_message(self, text: str) -> None:
        """清空树并给一行降级说明 (不动 ⑦ 的状态)。"""
        self._campaign_progress = None
        self._campaign_tree.clear()
        self._campaign_header_text = ""
        self._campaign_header.setText("")
        self._set_campaign_status(text)

    def _paint_campaign_progress(self) -> None:
        """把缓存的快照铺进树: 段一行, 块作为子行。缺值一律印「—」。

        不做任何判断 (不分好坏、不上色告警): 饱和 / CV 的阈值属于 G3, 这里只印数。
        """
        progress = self._campaign_progress
        tree = self._campaign_tree
        tree.clear()
        if progress is None:
            self._campaign_header_text = ""
            self._campaign_header.setText("")
            self._set_campaign_status(_CAMPAIGN_HINT_UNSET)
            return
        totals = progress.totals
        name = progress.out.name or str(progress.out)
        header = (
            f"{totals['segments_total']} 段 · 已完成 {totals['segments_done']} 段；"
            f"{totals['blocks_total']} 块 · 已到 {totals['blocks_done']} 块"
            f"　（{name}）"
        )
        self._campaign_header_text = header
        self._campaign_header.setText(header)
        self._set_campaign_status("")

        for stage in progress.stages:
            shape_name = params.SHAPE_LABELS.get(stage.shape, stage.shape)
            stage_name = campaign_plan.CAMPAIGN_STAGE_LABELS.get(stage.stage, stage.stage)
            status = _CAMPAIGN_STATUS_LABELS.get(stage.status, stage.status)
            # 全幅峰值缺 (旧 manifest 没有 peak 列) → 「—」, 不是 0。
            peak_full = _fmt_num(stage.peak_full, "{:.0f}")
            cells = (
                f"{shape_name} / {stage_name}",
                status,
                f"{_fmt_num(stage.seed_cv, '{:.4f}')} → {_fmt_num(stage.final_cv, '{:.4f}')}",
                f"{_fmt_num(stage.peak_inside, '{:.0f}')} · {peak_full}",
                _fmt_num(stage.exposure_ms, "{:.1f}"),
            )
            top = QTreeWidgetItem(tree, list(cells))
            # 段行是「一组块」的抬头: 加粗把两级行分开, 否则缩进很浅时容易读串行。
            stage_font = QFont(top.font(0))
            stage_font.setBold(True)
            for column in range(len(cells)):
                top.setTextAlignment(column, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                top.setFont(column, stage_font)
            top.setTextAlignment(4, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            for block in stage.blocks:
                note = block.note if isinstance(block.note, str) and block.note else MISSING
                child = QTreeWidgetItem(top, [
                    str(block.index),
                    _fmt_num(block.cv, "{:.4f}"),
                    _fmt_num(block.saturated_px, "{:.0f}"),
                    _fmt_num(block.core_peak, "{:.0f}"),
                    _fmt_num(block.exposure_ms, "{:.1f}"),
                    "",
                    note,
                ])
                # 说明列最容易被挤窄: 单元格省略号外的全文挂 tooltip, 不丢信息。
                child.setToolTip(6, note)
                for column in (1, 2, 3, 4):
                    child.setTextAlignment(column, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                child.setTextAlignment(5, Qt.AlignmentFlag.AlignCenter)
                mark, kind = _accept_mark(block.accepted)
                child.setText(5, mark)
                # 未知判定 (—) 用次级色; 配色键只认 ok / error, 不查 ``self._colors`` 里没有的键。
                color = self._colors[kind] if kind in self._colors else self._muted()
                child.setForeground(5, QColor(color))
        tree.expandToDepth(0)
        # 首列是 Interactive: 铺完后按内容定宽, 再留一口气 —— 加粗的长段名紧贴
        # 「状态」列会被读成一行 (ResizeToContents 正好算到像素, 没有呼吸空间)。
        tree.setColumnWidth(0, tree.sizeHintForColumn(0) + 8)

    def _paint_campaign_theme(self) -> None:
        """换主题: 只重上色, 不重读目录。"""
        if not getattr(self, "_campaign_ready", False):
            return
        self._campaign_status.setStyleSheet(f"color: {self._muted()};")
        if self._campaign_progress is not None:
            self._paint_campaign_progress()

    def campaign_tree(self) -> QTreeWidget:
        """⑥ 逐块树 (自动化验证直接读行 / 单元格)。"""
        return self._campaign_tree

    def campaign_header_text(self) -> str:
        """逐块页页头原文 (合计 + 目录名); 没有快照时为 ``""``。"""
        return self._campaign_header_text

    def campaign_status_text(self) -> str:
        """逐块页降级说明; 树已经填上内容时为 ``""``。"""
        return self._campaign_status_text

    def _build_header(self) -> QWidget:
        """标题与一句话说明。"""
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        title = QLabel("迁移学习闭环整形")
        font = title.font()
        font.setPointSize(font.pointSize() + 5)
        font.setWeight(QFont.Weight.DemiBold)
        title.setFont(font)

        subtitle = QLabel(
            "物理预测模型 · 探针采集 · 按 CCD 实测验收：一次会话跑完采集、迁移、整形与审计"
        )
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        return host

    def _build_form(self) -> QWidget:
        """左侧可滚动表单: ① 设备 → ② 目标 → ③ 校验 → ④ 运行。"""
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(self._build_device())
        layout.addWidget(self._build_target())
        layout.addWidget(self._build_checks())
        layout.addWidget(self._build_run())
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(host)
        scroll.setFixedWidth(_FORM_WIDTH)
        return scroll

    def _advanced_fold(self, tooltip: str) -> tuple[QCheckBox, QWidget]:
        """「高级参数」折叠行: 一个勾选框 + 一个可藏容器, 默认收起。

        折叠**纯属展示**: 控件照常构建、照常进 ``form_values()``、照常参与校验
        与 ``_set_form_enabled``, 收起不改变任何行为。**不做「里面有非默认值就
        自动展开」** —— ③ 校验把每个问题都印成文字, 折起来的控件出问题也看得见;
        自动展开反而会把操作者正在看的表单顶走。
        """
        toggle = QCheckBox("高级参数")
        toggle.setToolTip(tooltip)
        container = QWidget()
        toggle.toggled.connect(container.setVisible)
        container.setVisible(False)
        return toggle, container

    def _build_device(self) -> QGroupBox:
        """① 设备: 往哪台硬件下发。"""
        box, layout = _section("① 设备")

        self._camera = self._combo(
            [(kind, _CAMERA_LABELS.get(kind, kind)) for kind in params.CAMERA_CHOICES],
            _DEFAULTS.camera,
        )
        self._cam_id = self._spin(_DEFAULTS.cam_id, 0, 15)
        self._slm_number = self._spin(_DEFAULTS.slm_number, 1, 4)

        # 换相机 / 换 SLM 是装机时一次性的设置, 不是每次运行的旋钮 → 折进「高级
        # 参数」。折叠只藏控件不改行为: form_values() 照读、校验照跑、自检与探测
        # 照常写结论。留在主路径的探测结论行负责回答「这次用哪台相机」。
        self._device_advanced_toggle, self._device_advanced_container = (
            self._advanced_fold(
                "收起不改变任何行为：相机 / SLM 编号照常参与校验与运行；"
                "装机或换机台时才需要展开"
            )
        )
        device_advanced = QFormLayout()
        device_advanced.setContentsMargins(0, 0, 0, 0)
        device_advanced.setSpacing(8)
        device_advanced.addRow("相机类型", self._camera)
        device_advanced.addRow("CCD cam_id", self._cam_id)
        device_advanced.addRow("SLM 编号", self._slm_number)
        self._device_advanced_container.setLayout(device_advanced)

        layout.addWidget(self._device_advanced_toggle)
        layout.addWidget(self._device_advanced_container)

        # 加载时自动探测的结论: 中性提示 (普通文本色, 不是告警), 只说明「这次
        # 用哪台相机」。探测失败也写在这里, 不染色、不阻断。
        self._probe_label = QLabel("正在探测在线相机…")
        self._probe_label.setWordWrap(True)
        self._probe_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self._probe_label)

        # 自检按钮: 只有点它才会碰硬件 (枚举相机 + 探针 SLM)。默认可用 —— 参数
        # 校验没过时, 恰恰更需要知道硬件是不是好的。
        self._check_btn = QPushButton("设备自检")
        self._check_btn.setToolTip(
            "只读枚举相机并探测所选 SLM（会开关一次 SLM，不改显示内容）"
        )
        self._check_btn.clicked.connect(self._run_check)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._check_btn)
        row.addStretch(1)
        layout.addLayout(row)

        self._check_view = QLabel()
        self._check_view.setTextFormat(Qt.TextFormat.RichText)
        self._check_view.setWordWrap(True)
        self._check_view.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._check_view.setVisible(False)
        layout.addWidget(self._check_view)
        return box

    def _build_target(self) -> QGroupBox:
        """② 目标: 跑什么形状、多大、加不加倾斜相位。"""
        box, layout = _section("② 目标与倾斜相位")

        # 目标形状**互斥单选**（用户要求: 4 个形状一次只能选一个）——
        # 用独占 QButtonGroup 包 QRadioButton, 而不是复选, 语义上与「一场会话跑一个
        # 目标」一致; 也顺带消除了「空形状 → 至少选一个」的错误路径。
        self._shapes: dict[str, QRadioButton] = {}
        self._shape_group = QButtonGroup(self)
        self._shape_group.setExclusive(True)
        shape_host = QWidget()
        shape_row = QHBoxLayout(shape_host)
        shape_row.setContentsMargins(0, 0, 0, 0)
        shape_row.setSpacing(12)
        for name in params.SHAPE_CHOICES:
            radio = self._radio(
                params.SHAPE_LABELS.get(name, name), name in _DEFAULTS.shapes
            )
            self._shape_group.addButton(radio)
            self._shapes[name] = radio
            shape_row.addWidget(radio)
            # 勾选 rectangle 时显示「长边」框、把「边长」当短边; 其余形状隐藏长边框。
            radio.toggled.connect(self._sync_shape_widgets)
        shape_row.addStretch(1)

        self._side = self._dspin(
            _DEFAULTS.side, 8.0, _SIDE_MAX_CONSERVATIVE_PX, step=5.0
        )
        # 矩形长边（仅勾选 rectangle 时可见）：默认短边 × 默认长/短比。
        self._side_long = self._dspin(
            _DEFAULTS.side * _DEFAULTS.aspect_ratio,
            8.0,
            _SIDE_MAX_CONSERVATIVE_PX,
            step=5.0,
        )
        self._side_host = QWidget()
        side_qh = QHBoxLayout(self._side_host)
        side_qh.setContentsMargins(0, 0, 0, 0)
        side_qh.setSpacing(8)
        side_qh.addWidget(self._side, 1)
        side_qh.addWidget(self._side_long, 1)
        self._period = self._dspin(_DEFAULTS.period, 4.0, 1024.0, step=1.0)

        # 倾斜相位开关: 默认关。关了之后目标留在 0 级 (zero order)、周期不适用,
        # 于是周期框随之禁用并置灰。
        self._tilt = self._check("添加倾斜相位", _DEFAULTS.tilt)
        self._tilt.setToolTip(
            "开 = 加倾斜相位把目标推离 0 级, 需给出其周期; "
            "关（默认）= 目标留在 0 级, 倾斜相位周期无意义"
        )
        self._tilt.toggled.connect(self._sync_tilt)

        # ABBA 交错测量 配对验收: 默认关 (只按模型损失接受, 跳过 CCD 实测配对与方向核对)。
        self._abba = self._check("ABBA 交错测量 配对验收（CCD 实测）", _DEFAULTS.abba)
        self._abba.setToolTip(
            "开 = 每块测 A1/B1/B2/A2, 由实测指标判定候选是否接受; "
            "关（默认）= 保留按 CCD 实测校正的反馈, 跳过 B1/B2/A2 与方向核对, "
            "候选仅按模型损失接受（模型损失验收，无 CCD 实测配对）, 摘要写 acceptance=model_only（等价 CLI --no-abba）"
        )

        # 验收阈值循环 (CCD 与模型预测比对): 默认关 = 当前单轮行为。开 = 拟合后在
        # 验收行上比对模型预测的相关性 / 相对 L2, 不达标就把验证数据并入训练重训,
        # 直到达标、用户点「停止」或达到安全上限。阈值属于高级口径, 只在开启时
        # 可编辑 (关 → 禁用置灰), 布局跟随本表单其余行。
        self._accept = self._check(
            "验收阈值循环（CCD 与模型预测比对，不达标则重训）",
            _DEFAULTS.acceptance_enabled,
        )
        self._accept.setToolTip(
            "开 = 拟合后比对模型预测与实测 CCD, 不达标则把验证数据并入训练并重训, "
            "直到达标 / 点「停止」/ 到安全上限; 关（默认）= 保持当前单轮行为。"
        )
        self._accept_lower = self._dspin(
            _DEFAULTS.acceptance_lower, 0.0, 1.0, step=0.05
        )
        self._accept_lower.setDecimals(2)
        self._accept_lower.setValue(_DEFAULTS.acceptance_lower)
        self._accept_lower.setToolTip(
            f"相关性（correlation）下界；越界时循环继续重训（算法建议 "
            f"{params.ACCEPTANCE_LOWER_DEFAULT}）"
        )
        self._accept_upper = self._dspin(
            _DEFAULTS.acceptance_upper, 0.0, 1000.0, step=0.05
        )
        self._accept_upper.setDecimals(2)
        self._accept_upper.setValue(_DEFAULTS.acceptance_upper)
        self._accept_upper.setToolTip(
            f"相对 L2（relative_l2）上界；越界时循环继续重训（算法建议 "
            f"{params.ACCEPTANCE_UPPER_DEFAULT}）"
        )
        self._accept_hint = QLabel(
            f"算法建议: 相关性 ≥ {params.ACCEPTANCE_LOWER_DEFAULT}"
            f" · 相对L2 ≤ {params.ACCEPTANCE_UPPER_DEFAULT}"
        )
        self._accept_hint.setWordWrap(True)
        hint_font = self._accept_hint.font()
        hint_font.setPointSize(max(hint_font.pointSize() - 1, 7))
        self._accept_hint.setFont(hint_font)
        self._accept.toggled.connect(self._sync_acceptance)
        self._sync_acceptance()

        # 光路复用: 默认关 = 当前全流程。开 = 先用一次廉价探针量当前光路条件,
        # 与最近一份 latest 模型比对; 一致 (序列号同、光路未变) 就跳过采集与拟合,
        # 直接把那次 CCD 对照重做一遍。各阶段表会把这一步显示成「optical_probe」。
        self._reuse_latest = self._check(
            "光路复用（光路未变则跳过采集/拟合，直接CCD对照）",
            _DEFAULTS.reuse_latest,
        )
        self._reuse_latest.setToolTip(
            "开 = 开设备量一次当前光路探针, 与最近一份 latest 模型比对; 若光路未变就"
            "跳过采集与拟合, 直接复用那份 CCD 对照重做; 关（默认）= 当前全流程重新采集。"
        )

        # 「读取零级」: 让目标边长的上限来自实测光轴 (0 级到最近帧边的距离),
        # 而不是写死的保守值。做法与自检一致 —— 点了才碰硬件, 且只读。
        self._zero_btn = QPushButton("读取零级")
        self._zero_btn.setToolTip(
            "只读：开一次相机找 0 级，由 0 级到帧边的距离算出目标边长上限。"
            "不改光路、不改 SLM 显示"
        )
        self._zero_btn.clicked.connect(self._read_zero_order)
        side_row = QHBoxLayout()
        side_row.setContentsMargins(0, 0, 0, 0)
        side_row.setSpacing(8)
        side_row.addWidget(self._side_host, 1)
        side_row.addWidget(self._zero_btn, 0)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)
        form.addRow("目标形状", shape_host)
        form.addRow("目标短边 / 边长（CCD px）", side_row)
        form.addRow("倾斜相位周期（面板 px）", self._period)
        form.addRow(self._tilt)
        form.addRow("配对验收", self._abba)
        layout.addLayout(form)

        # 验收阈值循环 + 光路复用折进「高级参数」: 都是「转手动」的口径 —— 默认
        # 单轮 / 全流程已经是对的, 需要它们的是调阈值、复跑同一光路这类少数场合。
        # 周期与倾斜开关留在主路径: 那是每次运行都可能动的旋钮。折叠只藏控件
        # 不改行为, ③ 校验照常把这些控件的问题印成文字 (所以不自动展开)。
        self._advanced_toggle, self._advanced_container = self._advanced_fold(
            "收起不改变任何行为：验收阈值循环与光路复用照常参与校验与运行；"
            "③ 校验会把这里的问题照常列出来"
        )
        accept_row = QHBoxLayout()
        accept_row.setContentsMargins(0, 0, 0, 0)
        accept_row.setSpacing(8)
        accept_row.addWidget(self._accept)
        accept_row.addWidget(self._accept_lower)
        accept_row.addWidget(self._accept_upper)
        advanced_form = QFormLayout()
        advanced_form.setContentsMargins(0, 0, 0, 0)
        advanced_form.setSpacing(8)
        advanced_form.addRow(accept_row)
        advanced_form.addRow(self._accept_hint)
        advanced_form.addRow(self._reuse_latest)
        self._advanced_container.setLayout(advanced_form)

        layout.addWidget(self._advanced_toggle)
        layout.addWidget(self._advanced_container)

        # 常驻说明: 上限从哪来、现在是多少 —— 读不到时也要把「仍是保守值」讲清楚。
        self._zero_view = QLabel()
        self._zero_view.setTextFormat(Qt.TextFormat.RichText)
        self._zero_view.setWordWrap(True)
        self._zero_view.setVisible(True)
        hint_font = self._zero_view.font()
        hint_font.setPointSize(max(hint_font.pointSize() - 1, 7))
        self._zero_view.setFont(hint_font)
        layout.addWidget(self._zero_view)
        self._render_zero()
        return box

    def _build_checks(self) -> QGroupBox:
        """③ 校验: 红字阻断, 黄字提醒。"""
        box, layout = _section("③ 校验")
        self._checks = QLabel("正在校验参数…")
        self._checks.setTextFormat(Qt.TextFormat.RichText)
        self._checks.setWordWrap(True)
        self._checks.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self._checks.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._checks.setMinimumHeight(96)
        layout.addWidget(self._checks)

        # G1: 黄字不再只是上色 —— 每条提醒都要逐条勾「已确认」, 否则 ④ 的启动键
        # 一直禁用。红字存在时整块隐藏 (红字本来就拦着, 再确认黄字是多余的)。
        self._ack_box = QWidget()
        ack_layout = QVBoxLayout(self._ack_box)
        ack_layout.setContentsMargins(0, 0, 0, 0)
        ack_layout.setSpacing(2)
        self._ack_hint = QLabel("改任何参数都会作废这些确认，需重新勾选")
        self._ack_hint.setWordWrap(True)
        ack_layout.addWidget(self._ack_hint)
        self._ack_layout = ack_layout
        self._ack_box.setVisible(False)
        layout.addWidget(self._ack_box)
        return box

    def _build_run(self) -> QGroupBox:
        """④ 运行: 一个按钮 + 状态 + 会话目录。"""
        box, layout = _section("④ 运行")

        run_row = QHBoxLayout()
        run_row.setContentsMargins(0, 0, 0, 0)
        run_row.setSpacing(8)
        self._run_btn = _bold(QPushButton("开始运行"))
        self._run_btn.setMinimumHeight(38)
        # 首次校验通过前不允许启动 —— 默认禁用, 免得「能点」被读成「参数没问题」。
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._start_run)
        run_row.addWidget(self._run_btn, 1)
        # 软停止: 只在验收阈值循环跑着的时候有意义 (两轮之间生效), 平时禁用。
        # 它只是置位一个 flag, 让 run_session 在下一轮开始前优雅收尾 —— 不打断
        # 正在进行的单轮, 也不走关窗那套「等落地」。
        self._stop_btn = QPushButton("停止（两轮间生效）")
        self._stop_btn.setToolTip(
            "点击后当前这一轮跑完、下一轮开始前停下；保留已得的最好结果。"
            "只对开启验收阈值循环的会话生效。"
        )
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._request_acceptance_stop)
        run_row.addWidget(self._stop_btn, 0)
        layout.addLayout(run_row)

        self._status = QLabel()
        self._status.setTextFormat(Qt.TextFormat.RichText)
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        planned = params.plan_session_dir()
        self._session_label = QLabel(f"会话目录（规划）: {planned}")
        self._session_label.setWordWrap(True)
        self._session_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._session_label)
        return box

    def _build_progress(self) -> QGroupBox:
        """⑤ 进度: 当前阶段 + 汇总 + 进度条 + 阶段表。"""
        box, layout = _section("⑤ 进度")

        head = QHBoxLayout()
        head.setSpacing(10)
        self._current = _bold(QLabel("尚无阶段计划"))
        self._current.setWordWrap(True)
        self._counts = QLabel("")
        head.addWidget(self._current, 1)
        head.addWidget(self._counts, 0, Qt.AlignmentFlag.AlignRight)
        self._replay_btn = QPushButton("归档回放…")
        self._replay_btn.setToolTip(
            "离线查看已落盘的每块 CCD 帧与 SLM 相位（只读，不碰硬件）"
        )
        self._replay_btn.clicked.connect(self._open_replay)
        head.addWidget(self._replay_btn, 0)
        layout.addLayout(head)

        self._bar = QProgressBar()
        self._bar.setRange(0, 1)
        self._bar.setFormat("已结束 %v / 计划 %m 阶段")
        layout.addWidget(self._bar)

        # 完成后的导出行: 平时隐藏, 只在「会话完成」时替换进度条 / 当前阶段标签。
        self._export_row = QWidget()
        export_layout = QHBoxLayout(self._export_row)
        export_layout.setContentsMargins(0, 0, 0, 0)
        export_layout.setSpacing(8)
        self._btn_export_png = QPushButton("导出 PNG…")
        self._btn_export_png.setToolTip("导出总结对比图（或最后一帧 CCD）")
        self._btn_export_phase = QPushButton("导出相位 CSV…")
        self._btn_export_phase.setToolTip("导出最终 best 相位（弧度）为 CSV")
        self._btn_export_gray = QPushButton("导出 SLM 灰度 CSV…")
        self._btn_export_gray.setToolTip("导出 SLM 实际下发的灰度相位为 CSV")
        self._btn_export_png.clicked.connect(self._export_png)
        self._btn_export_phase.clicked.connect(self._export_phase_csv)
        self._btn_export_gray.clicked.connect(self._export_gray_csv)
        export_layout.addWidget(self._btn_export_png)
        export_layout.addWidget(self._btn_export_phase)
        export_layout.addWidget(self._btn_export_gray)
        self._export_row.setVisible(False)
        layout.addWidget(self._export_row)

        layout.addWidget(self._build_table(), 1)
        return box

    def _build_table(self) -> QTableWidget:
        """阶段表: 序号 / 阶段 / 状态 / 耗时 / 说明。"""
        table = QTableWidget(0, 5)
        table.setHorizontalHeaderLabels(["#", "阶段", "状态", "耗时", "说明"])
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(26)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        table.setShowGrid(False)

        header = table.horizontalHeader()
        for column, width in ((0, 42), (2, 76), (3, 72)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
            header.resizeSection(column, width)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        header.resizeSection(1, 220)
        # 说明列: 内容决定列宽, 超过视口时靠表格自身的横向滚动条滑动查看 ——
        # 用户要求长说明文字支持横向滑动, 而不是被 Stretch 一裁到底。
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setMinimumSectionSize(60)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)

        self._table = table
        return table

    def _build_plan_preview(self) -> QGroupBox:
        """⑦ 计划预览: campaign 4×4 矩阵 + 段级成本表 (只读, 不碰设备)。

        为什么放在右栏最下面: 它是「开跑前看一眼规模 / 拿历史实测对一下计划」的静态
        东西; 跑起来之后视线回到 ⑤ ⑥, 所以它只占自然高度、不抢伸缩份额。
        """
        box, layout = _section("⑦ 计划预览（整轮矩阵 / 段级成本）")

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)
        self._plan_caption = QLabel(
            "只读：一整轮的规模先摆出来，再按历史目录对一下实测偏差"
        )
        self._plan_caption.setWordWrap(True)
        head.addWidget(self._plan_caption, 1)
        self._plan_toggle = QCheckBox("展开")
        self._plan_toggle.setChecked(True)
        self._plan_toggle.setToolTip("收起后右栏只留 ⑥ 实时画面与 ⑤ 阶段进度")
        self._plan_toggle.toggled.connect(self._toggle_plan_body)
        head.addWidget(self._plan_toggle, 0, Qt.AlignmentFlag.AlignRight)
        layout.addLayout(head)

        # 正文放进 QScrollArea: 矩阵 + 合计 + 目录 + 16 行表自然高约 700 px, 直接摊开
        # 会把 ⑥ 实时画面挤到最小尺寸。滚动条只在本区出现, 不牵动左栏表单。
        self._plan_body = QWidget()
        body = QVBoxLayout(self._plan_body)
        body.setContentsMargins(0, 0, 2, 0)
        body.setSpacing(8)

        self._plan_matrix = self._build_plan_matrix()
        body.addWidget(self._plan_matrix)

        # 数据层读不出来时网格还是上一帧: 用一行次级色说明「读取失败」, 不把旧数字
        # 当新数字继续看。平时隐藏, 成功渲染时再藏回去。
        self._plan_matrix_error = QLabel("读取失败")
        self._plan_matrix_error.setWordWrap(True)
        self._plan_matrix_error.setVisible(False)
        body.addWidget(self._plan_matrix_error)

        self._plan_summary = QLabel()
        self._plan_summary.setWordWrap(True)
        self._plan_summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        body.addWidget(self._plan_summary)

        dir_row = QHBoxLayout()
        dir_row.setContentsMargins(0, 0, 0, 0)
        dir_row.setSpacing(8)
        self._cost_dir = QLineEdit()
        self._cost_dir.setPlaceholderText("整轮目录（留空 = 只按计划）")
        self._cost_dir.setClearButtonEnabled(True)
        self._cost_dir.setToolTip(
            "读该目录下各段 cost.json 的实测秒数；空 / 不存在都只显示计划列"
        )
        # 回车与按钮等价; 绝不在每次按键时读盘 (目录一大, 每个字符都扫一遍没有意义)。
        self._cost_dir.returnPressed.connect(self._reload_plan_costs)
        self._cost_btn = QPushButton("读取实测")
        self._cost_btn.setToolTip("只读该目录的 cost.json；不打开相机 / SLM")
        self._cost_btn.clicked.connect(self._reload_plan_costs)
        dir_row.addWidget(self._cost_dir, 1)
        dir_row.addWidget(self._cost_btn, 0)
        body.addLayout(dir_row)

        self._cost_table = self._build_cost_table()
        body.addWidget(self._cost_table)

        # 口径行: 平时只有一句口径; 目录写错时把它那行说明挂在前面, 不弹错误框。
        self._cost_caption = QLabel()
        self._cost_caption.setWordWrap(True)
        body.addWidget(self._cost_caption)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(self._plan_body)
        # 上限定在 430 px: 矩阵 + 合计 + 目录 + 16 行表全摊开要约 700 px, 会让 ⑥ 实时
        # 画面掉到最小尺寸。下限留 0 而不是钉死一个高度: 窗口矮的时候让这一段先缩、
        # 由它自己的滚动条兜住, 不能把整页的最小高度顶上去 (笔记本上会放不下)。
        scroll.setMinimumHeight(0)
        scroll.setMaximumHeight(_PLAN_BODY_MAX_HEIGHT)
        layout.addWidget(scroll)

        # 控件都齐了才允许重画 (``_sync_shape_widgets`` / 换主题都会回到这几个渲染器)。
        self._plan_ready = True
        self._render_plan_matrix()
        self._paint_plan_summary()
        self._reload_plan_costs()
        self._plan_box = box
        return box

    def _toggle_plan_body(self, shown: bool) -> None:
        """收起 / 展开 ⑦ 的正文 (只藏正文, 标题与折叠框留在原处)。"""
        self._plan_body.setVisible(bool(shown))

    def _build_plan_matrix(self) -> QWidget:
        """4 形状 × 4 阶段的矩阵 (行 = 形状, 列 = 阶段)。

        为什么用网格而不是 16 行清单: 1560 宽的宿主窗口里右栏约 900 px, 减去行标题后
        每个阶段列约 170 px; 格内「4 块 · 8-bit · 15 迭代」约 130 px, 最长的阶段表头
        「12-bit 密集（宽脊模型）」折两行也放得下。每个 QLabel 都开 wordWrap —— 窗口
        再窄只会多折一行, 不会把字裁掉, 也不会把整格挤到框外, 所以不会出现「被切一半
        的格子」。真缩到四列都撑不住时仍然是折行 (行变高), 而不是横向滚动或裁切;
        这也是不退回 16 行清单的理由: 清单丢掉「形状 × 阶段」的二维读法, 却省不下高度。
        """
        host = QWidget()
        grid = QGridLayout(host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(2)

        # 形状 / 阶段的顺序一律取数据层的顺序 (campaign 矩阵的行优先展开), 页面不另排。
        # 数据层承诺不抛; 建页这一下再兜一层, 免得一行数据坏了就整页起不来。
        try:
            cells = campaign_plan.matrix_cells()
        except (TypeError, ValueError, AttributeError):
            cells = ()
        stage_order: list[tuple[str, str]] = []
        shape_order: list[tuple[str, str]] = []
        for cell in cells:
            if all(cell.stage != stage for stage, _ in stage_order):
                stage_order.append((cell.stage, cell.stage_label))
            if all(cell.shape != shape for shape, _ in shape_order):
                shape_order.append((cell.shape, cell.shape_label))

        corner = QLabel("形状")
        corner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        corner.setContentsMargins(2, 0, 6, 0)
        grid.addWidget(corner, 0, 0)
        self._plan_corner = corner

        self._plan_stage_headers: list[QLabel] = []
        self._plan_stage_labels: list[str] = []
        self._plan_stage_order: list[str] = []
        for column, (stage, stage_label) in enumerate(stage_order, start=1):
            header = QLabel(stage_label)
            header.setWordWrap(True)
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            header.setContentsMargins(4, 0, 4, 0)
            grid.addWidget(header, 0, column)
            self._plan_stage_headers.append(header)
            self._plan_stage_labels.append(stage_label)
            self._plan_stage_order.append(stage)

        self._plan_row_labels: dict[str, QLabel] = {}
        self._plan_shape_labels: dict[str, str] = {}
        self._plan_cells: dict[tuple[str, str], QLabel] = {}
        for row, (shape, shape_label) in enumerate(shape_order, start=1):
            row_label = QLabel(shape_label)
            row_label.setWordWrap(True)
            row_label.setContentsMargins(2, 2, 6, 2)
            grid.addWidget(row_label, row, 0)
            self._plan_row_labels[shape] = row_label
            self._plan_shape_labels[shape] = shape_label
            for column, (stage, _) in enumerate(stage_order, start=1):
                cell = next(
                    item
                    for item in cells
                    if (item.shape, item.stage) == (shape, stage)
                )
                widget = QLabel(self._plan_cell_text(cell))
                widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
                widget.setWordWrap(True)
                widget.setContentsMargins(4, 2, 4, 2)
                grid.addWidget(widget, row, column)
                self._plan_cells[(shape, stage)] = widget

        for column in range(1, len(stage_order) + 1):
            grid.setColumnStretch(column, 1)
        grid.setColumnStretch(0, 0)
        host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        return host

    @staticmethod
    def _plan_cell_text(cell: campaign_plan.MatrixCell) -> str:
        """一格一行的紧凑写法: 块数 · 位深 · 每块迭代数。"""
        return f"{cell.blocks} 块 · {cell.bit_depth}-bit · {cell.updates} 迭代"

    def _build_cost_table(self) -> QTableWidget:
        """段级成本表: 形状 / 阶段 / 计划块数 / 计划秒 / 实测秒 / 实测块数 / 偏差。

        外观沿用 ⑤ 阶段表那套 (不显示行号、无网格线、隔行底色), 只把行高压矮一档:
        16 行是「一览」, 不该占掉半个右栏。
        """
        table = QTableWidget(0, len(_COST_COLUMNS))
        table.setHorizontalHeaderLabels(list(_COST_COLUMNS))
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(_COST_ROW_HEIGHT)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        table.setShowGrid(False)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)

        header = table.horizontalHeader()
        for column, width in enumerate(_COST_WIDTHS):
            if column == 1:
                # 阶段列由内容定宽: 「12-bit 密集（宽脊模型）」不被截断, 超宽时横向滚动。
                header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
                continue
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
            header.resizeSection(column, width)
        header.setMinimumSectionSize(56)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        return table

    # -------------------------------------------------------------- 控件工厂
    # 每个工厂顺手把「改了 → 重算校验」接上: 新加控件时不可能忘记接线。
    def _spin(self, value: int, low: int, high: int, *, step: int = 1) -> QSpinBox:
        """整数框 (范围与 Streamlit 页逐项一致)。"""
        widget = QSpinBox()
        widget.setRange(low, high)
        widget.setSingleStep(step)
        widget.setValue(value)
        # 不跟键: 输入 "1" → "10" → "100" 的中途不该每一下都报警。
        widget.setKeyboardTracking(False)
        widget.valueChanged.connect(self._revalidate)
        return widget

    def _dspin(
        self, value: float, low: float, high: float, *, step: float = 1.0
    ) -> QDoubleSpinBox:
        """浮点框 (目标边长 / 倾斜相位周期), 一位小数。"""
        widget = QDoubleSpinBox()
        widget.setRange(low, high)
        widget.setDecimals(1)
        widget.setSingleStep(step)
        widget.setValue(value)
        widget.setKeyboardTracking(False)
        widget.valueChanged.connect(self._revalidate)
        return widget

    def _combo(self, options: Sequence[tuple[str, str]], value: str) -> QComboBox:
        """代码值 + 中文说明的下拉框 (``data`` 存代码, 显示存说明)。"""
        widget = QComboBox()
        for code, label in options:
            widget.addItem(label, code)
        widget.setCurrentIndex(max(widget.findData(value), 0))
        widget.currentIndexChanged.connect(self._revalidate)
        return widget

    def _check(self, text: str, checked: bool) -> QCheckBox:
        """复选框。"""
        widget = QCheckBox(text)
        widget.setChecked(checked)
        widget.toggled.connect(self._revalidate)
        return widget

    def _radio(self, text: str, checked: bool) -> QRadioButton:
        """单选钮（形状互斥用）。"""
        widget = QRadioButton(text)
        widget.setChecked(checked)
        widget.toggled.connect(self._revalidate)
        return widget

    # -------------------------------------------------------------- 公开接口
    def form_values(self) -> params.TransferFormValues:
        """控件当前值 → 纯表单快照 (交给 params 层校验与建会话)。"""
        shapes = tuple(
            name for name in params.SHAPE_CHOICES if self._shapes[name].isChecked()
        )
        short_side = float(self._side.value())
        aspect_ratio: float
        if "rectangle" in shapes:
            long_side = float(self._side_long.value())
            aspect_ratio = long_side / short_side if short_side > 0 else 1.0
        else:
            aspect_ratio = 1.0
        return params.TransferFormValues(
            shapes=shapes,
            side=short_side,
            tilt=bool(self._tilt.isChecked()),
            aspect_ratio=aspect_ratio,
            period=float(self._period.value()),
            camera=str(self._camera.currentData()),
            cam_id=int(self._cam_id.value()),
            slm_number=int(self._slm_number.value()),
            abba=bool(self._abba.isChecked()),
            acceptance_enabled=bool(self._accept.isChecked()),
            acceptance_metric=_DEFAULTS.acceptance_metric,
            acceptance_lower=float(self._accept_lower.value()),
            acceptance_upper=float(self._accept_upper.value()),
            acceptance_max_rounds=_DEFAULTS.acceptance_max_rounds,
            reuse_latest=bool(self._reuse_latest.isChecked()),
        )

    def current_problems(self) -> tuple[list[str], list[str]]:
        """最近一次校验的 ``(errors, warnings)`` (副本, 调用方随便改)。"""
        return list(self._last_errors), list(self._last_warnings)

    def current_notes(self) -> list[str]:
        """最近一次校验的**中性说明** (副本)：不算阻塞也不算提醒。"""
        return list(self._last_notes)

    def pending_acks(self) -> list[str]:
        """黄色提醒里还没勾「已确认」的条目 (副本, 顺序同 ③ 的黄字)。"""
        pending: list[str] = []
        for index, warning in enumerate(self._last_warnings):
            box = self._ack_checks[index] if index < len(self._ack_checks) else None
            if box is None or not box.isChecked():
                pending.append(warning)
        return pending

    def set_all_acks(self, checked: bool = True) -> None:
        """程序方式勾 / 取消全部「已确认」(等价于逐个点一下)。"""
        for check in self._ack_checks:
            check.setChecked(bool(checked))
        self._sync_run_button()

    def run_button(self) -> QPushButton:
        """「开始运行」按钮 (自动化验证按它判断可启动性)。"""
        return self._run_btn

    def is_running(self) -> bool:
        """是否有一场会话在跑 (自动化验证用)。"""
        return bool(self._running)

    def session_dir(self) -> Path | None:
        """本次 / 上一次会话目录; 还没开跑过时为 ``None``。"""
        return self._session_dir

    def check_button(self) -> QPushButton:
        """「设备自检」按钮 (自动化验证按它判断自检是否可触发 / 已结束)。"""
        return self._check_btn

    def zero_order_button(self) -> QPushButton:
        """「读取零级」按钮 (自动化验证按它判断读数是否可触发 / 已结束)。"""
        return self._zero_btn

    def stop_button(self) -> QPushButton:
        """「停止（两轮间生效）」按钮 (自动化验证按它判断可停止性)。"""
        return self._stop_btn

    def acceptance_stop_is_set(self) -> bool:
        """验收停止 flag 是否已置位 (自动化验证用; 启动后自动复位)。"""
        return self._acceptance_stop.is_set()

    # ------------------------------------------------------------ 计划预览接口
    def cost_table(self) -> QTableWidget:
        """⑦ 段级成本表 (自动化验证直接读单元格, 不穿多层布局去猜下标)。"""
        return self._cost_table

    def campaign_matrix_text(self) -> str:
        """⑦ 矩阵的纯文本快照: 与屏上同一份字, ``▸`` 标出 ② 选中的形状。"""
        return self._plan_matrix_text

    def plan_summary_text(self) -> str:
        """⑦ 合计行原文 (未上色的纯文本)。"""
        return self._plan_summary_text

    def cost_caption_text(self) -> str:
        """⑦ 成本表脚下那行口径原文 (不含「目录不可读」那句说明)。"""
        return self._cost_caption_text

    def campaign_dir(self) -> Path | None:
        """⑦ 成本表正在读的目录; 空 / 不可读时为 ``None`` (只按计划)。"""
        return self._cost_path

    def set_campaign_dir(self, directory: str | Path | None) -> None:
        """程序方式填 campaign 目录并重读 (等价于手填 + 点「读取实测」)。"""
        self._cost_dir.setText("" if directory is None else str(directory))
        self._reload_plan_costs()

    def _request_acceptance_stop(self) -> None:
        """「停止」按钮: 只是置位 flag, 让 run_session 在两轮之间优雅收尾。

        它是软停止: 不打断正在进行的单轮, 也不走关窗那套「最多等
        :data:`_CLOSE_GRACE_S`」的心跳。置位后按钮立即禁用, 防止连点。
        """
        if not self._running:
            return
        self._acceptance_stop.set()
        self._stop_btn.setEnabled(False)

    def check_report(self) -> str:
        """最近一次自检的纯文本结论 (没自检过为空串; 供自动化断言)。"""
        if self._check_heading is None:
            return ""
        lines = [self._check_heading[1]]
        lines.extend(f"{label}：{detail}" for _, label, detail in self._check_rows)
        if self._check_stale():
            lines.append("设备参数已改，请重新点「设备自检」")
        elif self._check_reused:
            lines.append("复用上次自检（同设备、未过期）")
        return "\n".join(lines)

    def ccd_probe_summary(self) -> str:
        """加载时相机自动探测的结论 (关闭探测时为空串; 供自动化断言, 永不为 None)。"""
        return self._probe_text

    def advanced_toggle(self) -> QCheckBox:
        """② 的「高级参数」勾选行 (自动化: 折 / 展就翻它)。"""
        return self._advanced_toggle

    def advanced_container(self) -> QWidget:
        """② 折叠容器: 收起时 ``isVisible()`` 为 False, 里面的控件照常在。"""
        return self._advanced_container

    def device_advanced_toggle(self) -> QCheckBox:
        """① 的「高级参数」勾选行 (相机 / cam_id / SLM 编号)。"""
        return self._device_advanced_toggle

    def device_advanced_container(self) -> QWidget:
        """① 折叠容器: 同 ②, 收起只藏控件。"""
        return self._device_advanced_container

    def set_camera(self, kind: str) -> None:
        """程序方式选相机后端, 联动与手工点选相同 (未知后端直接报错)。"""
        index = self._camera.findData(str(kind))
        if index < 0:
            choices = ", ".join(params.CAMERA_CHOICES)
            raise ValueError(f"未知相机类型 {kind!r}（可选 {choices}）")
        self._camera.setCurrentIndex(index)
        self._revalidate()

    def set_period(self, value: float) -> None:
        """程序方式填倾斜相位周期 (面板 px), 联动与手工输入相同。"""
        self._period.setValue(float(value))
        self._revalidate()

    def set_shapes(self, shapes: Iterable[str]) -> None:
        """程序方式选择目标形状（互斥单选）; 一次改完只校验一次。

        允许传入空集合以清空（供「至少选一个」错误路径的测试使用）——独占组
        正常不允许清空, 这里临时关掉独占再恢复。
        """
        chosen = {str(name) for name in shapes}
        self._shape_group.setExclusive(False)
        try:
            for name, radio in self._shapes.items():
                radio.blockSignals(True)
                radio.setChecked(name in chosen)
                radio.blockSignals(False)
        finally:
            self._shape_group.setExclusive(True)
        self._sync_shape_widgets()
        self._revalidate()

    def _sync_shape_widgets(self) -> None:
        """矩形勾选时显示「长边」框（此时边长框当短边）；其余形状隐藏长边框。"""
        rect = bool(
            self._shapes.get("rectangle", None)
            and self._shapes["rectangle"].isChecked()
        )
        self._side_long.setVisible(rect)
        if rect:
            self._side.setToolTip("矩形短边（CCD px）")
        else:
            self._side.setToolTip("目标边长（CCD px）")
        # ⑦ 矩阵里「当前形状」那一行跟着 ② 的单选走 (换形状只是重上色)。
        self._render_plan_matrix()

    def _sync_tilt(self) -> None:
        """倾斜相位关（默认）时目标留在 0 级, 周期框禁用置灰; 开时恢复。"""
        enabled = bool(self._tilt.isChecked())
        self._period.setEnabled(enabled)
        self._period.setToolTip(
            None if enabled else "倾斜相位关闭：目标留在 0 级，周期无意义"
        )

    def _sync_acceptance(self) -> None:
        """验收阈值循环关（默认）时上下界框禁用置灰; 开时恢复。"""
        enabled = bool(self._accept.isChecked())
        self._accept_lower.setEnabled(enabled)
        self._accept_upper.setEnabled(enabled)
        hint = (
            None if enabled else "开启验收阈值循环后可设置阈值"
        )
        self._accept_lower.setToolTip(
            hint
            if not enabled
            else f"相关性（correlation）下界；越界时循环继续重训（算法建议 "
            f"{params.ACCEPTANCE_LOWER_DEFAULT}）"
        )
        self._accept_upper.setToolTip(
            hint
            if not enabled
            else f"相对 L2（relative_l2）上界；越界时循环继续重训（算法建议 "
            f"{params.ACCEPTANCE_UPPER_DEFAULT}）"
        )

    # ------------------------------------------------------------------ 校验
    def _revalidate(self) -> None:
        """改动即校验: 红字阻断, 黄字要逐条勾「已确认」才放行。

        只做参数判断 —— 硬件自检是「设备自检」按钮的事, 不在这里触发 (否则每敲
        一个数字都枚举一次设备)。这里顺带重画自检区, 只是把旧结论标成过期。
        """
        if not getattr(self, "_form_ready", True):
            return
        errors, warnings = params.validate_form(self.form_values())
        self._last_errors = errors
        self._last_warnings = warnings
        note = params.capture_grid_note(self.form_values())
        self._last_notes = [note] if note else []
        self._render_checks()
        # 参数一变 → 重建黄字确认行: 旧勾选就此作废 (这就是 G1 的「重新确认」机制)。
        self._render_ack_list()
        self._sync_run_button()
        self._render_check()

    def _sync_run_button(self) -> None:
        """④ 启动键的唯一裁决点: 阻塞 > 设备占用 > 未确认提醒 > 可启动。"""
        if not getattr(self, "_run_btn", None):
            return
        pending = self.pending_acks()
        if self._last_errors:
            enabled, reason = False, self._last_errors[0]
        elif self._running:
            enabled, reason = False, "会话运行中"
        elif self._checking:
            enabled, reason = False, "正在自检设备，稍候"
        elif self._zero_reading:
            enabled, reason = False, "正在读取 0 级，稍候"
        elif pending:
            enabled, reason = False, f"还有 {len(pending)} 项提醒未勾选「已确认」"
        else:
            enabled, reason = True, "启动整场迁移学习会话"
        self._run_btn.setEnabled(enabled)
        self._run_btn.setToolTip(reason)

    def _style_ack_hint(self) -> None:
        """确认区的提示行只换颜色: 它是次级说明, 不该跟黄字抢注意力。"""
        self._ack_hint.setStyleSheet(f"color: {self._muted()};")

    def _render_ack_list(self) -> None:
        """黄字逐条「已确认」: 每次重建行, 参数一变旧勾选就作废。

        红字存在时整块隐藏 —— 启动键已被红字拦住, 再要人确认黄字只是噪音。
        刻意不保存勾选状态: 「改参数 = 重新确认」靠重建这一条机制实现, 不另存副本。
        """
        if not getattr(self, "_form_ready", False):
            return
        if not getattr(self, "_ack_box", None):
            return
        for check in self._ack_checks:
            self._ack_layout.removeWidget(check)
            check.deleteLater()
        self._ack_checks.clear()
        self._style_ack_hint()

        show = bool(self._last_warnings) and not self._last_errors
        self._ack_box.setVisible(show)
        if not show:
            return
        for warning in self._last_warnings:
            first = warning.split("\n", 1)[0].strip()
            text = f"已确认 · {first}"
            if len(text) > _ACK_LABEL_MAX:
                text = text[: _ACK_LABEL_MAX - 1] + "…"
            check = QCheckBox(text)
            check.setToolTip(warning)
            check.toggled.connect(self._sync_run_button)
            self._ack_layout.addWidget(check)
            self._ack_checks.append(check)

    def _render_checks(self) -> None:
        """把最近一次校验结论画成红 / 黄 / 中性三段 (无红无黄 = 绿字通过)。

        中性段只放**本页自己算出来的派生值**（自动升档后的采集网格），不用黄字：
        黄字代表「需要人确认」，说多了就没人看了。红黄都没有时先给一行绿色
        「参数自检通过」，中性段跟在后面 —— 通过本身就是结论，不该被说明挤掉。
        """
        colors = self._colors
        parts: list[str] = []
        if not self._last_errors and not self._last_warnings:
            parts.append(
                f'<p style="margin:0;color:{colors["ok"]}"><b>参数自检通过</b></p>'
            )
            parts.append(
                f'<p style="margin:4px 0 0 0;color:{self._muted()}">'
                "倾斜相位与目标尺寸放得进采样窗，探针数是 6 的倍数。</p>"
            )
        # 黄字段的尾句跟着「此刻有没有勾选框」走: 有红字时确认区整块隐藏, 再说
        # 「逐条勾选后才能启动」就是指着不存在的东西说话。
        warn_tail = (
            "项提醒 · 先修阻塞项，之后逐条勾选「已确认」"
            if self._last_errors
            else "项提醒 · 逐条勾选「已确认」后才能启动"
        )
        for texts, kind, tail, mark in (
            (self._last_errors, "error", "项阻塞 · 修好才能启动", "■"),
            (self._last_warnings, "warn", warn_tail, "!"),
        ):
            if texts:
                parts.append(
                    f'<p style="margin:0 0 4px 0;color:{colors[kind]}">'
                    f"<b>{len(texts)} {tail}</b></p>"
                )
                parts.extend(
                    f'<p style="margin:0 0 6px 0;color:{colors[kind]}">'
                    f"{mark} {_rich(text)}</p>"
                    for text in texts
                )
        if self._last_notes:
            muted = self._muted()
            parts.append(
                f'<p style="margin:0 0 4px 0;color:{muted}">'
                f"<b>{len(self._last_notes)} 项说明 · 页面按这些值执行</b></p>"
            )
            parts.extend(
                f'<p style="margin:0 0 6px 0;color:{muted}">· {_rich(text)}</p>'
                for text in self._last_notes
            )
        self._checks.setText("".join(parts))

    def _set_form_enabled(self, enabled: bool) -> None:
        """运行中锁住必设参数 —— 本次会话用的是启动那一刻的快照。"""
        for widget in (
            self._camera,
            self._cam_id,
            self._slm_number,
            self._side,
            self._side_long,
            self._tilt,
            self._accept,
            self._reuse_latest,
            *self._shapes.values(),
            # G1 的确认行: 会话跑起来后一起置灰 (本次会话的参数快照已定)。
            self._ack_box,
        ):
            widget.setEnabled(enabled)
        # 倾斜相位周期只在开启时才能点; 运行中一律锁。
        self._period.setEnabled(enabled and self._tilt.isChecked())
        # 验收阈值上下界只在循环开启时才能点; 运行中一律锁。
        self._accept_lower.setEnabled(enabled and self._accept.isChecked())
        self._accept_upper.setEnabled(enabled and self._accept.isChecked())
        # 自检同样要锁: 会话正占着 SLM 时再探一次, 只会跟它对抢设备。
        self._check_btn.setEnabled(enabled and not self._checking)
        # 读零级要开相机: 会话跑起来后相机归它用, 关闭时也顺手复位。
        self._zero_btn.setEnabled(
            enabled and not self._checking and not self._zero_reading
        )

    # ---------------------------------------------------------------- 自动探测
    def _start_ccd_detection(self) -> None:
        """页面加载时跑一次的只读相机枚举 (后台线程; 不打开相机、不探针 SLM)。"""
        task = CcdDetectTask(self._ccd_preferred, self._ccd_cam_id)
        # 与自检 / 会话同理: 信号送达前被 Qt 回收就丢掉结论。
        task.setAutoDelete(False)
        self._ccd_task = task
        task.signals.done.connect(self._on_ccd_detected)
        self._pool.start(task)

    def _set_probe_text(self, text: str) -> None:
        """提示行与只读摘要同源更新 (``ccd_probe_summary`` 因此永不为 None)。"""
        self._probe_text = str(text)
        self._probe_label.setText(self._probe_text)
        # 空文案不占行: 探测关掉 (或降级为空) 时 ① 里不留一条空白。
        self._probe_label.setVisible(bool(self._probe_text))

    def _apply_ccd_choice(self, kind: str, cam_id: int) -> None:
        """把探测结果写进控件: 关掉信号再写, 最后只重新校验一次。"""
        changed = False
        index = self._camera.findData(str(kind))
        if index >= 0 and index != self._camera.currentIndex():
            self._camera.blockSignals(True)
            try:
                self._camera.setCurrentIndex(index)
            finally:
                self._camera.blockSignals(False)
            changed = True
        if cam_id != self._cam_id.value():
            self._cam_id.blockSignals(True)
            try:
                self._cam_id.setValue(int(cam_id))
            finally:
                self._cam_id.blockSignals(False)
            changed = True
        if changed:
            self._revalidate()

    def _on_ccd_detected(self, result: object) -> None:
        """加载时探测回到 UI 线程: 只写一行提示, 控件仅在没被改过时才动; 永不抛。

        探测只是提示, 所以这里不 import hardware_check (它顶层拉 SDK), 结论按
        duck typing 读 —— 读不出来就按失败写一行, 后面的完整流程照旧。
        """
        self._ccd_task = None
        if isinstance(result, BaseException):
            self._set_probe_text(
                f"自动探测失败：{_brief_exception(result)}"
                f"（保留 {self._ccd_preferred} cam_id={self._ccd_cam_id}）"
            )
            return

        detail = getattr(result, "detail", "")
        detail_text = detail if isinstance(detail, str) else str(detail)
        kind_raw = getattr(result, "kind", self._ccd_preferred)
        kind = kind_raw if isinstance(kind_raw, str) else str(kind_raw)
        cam_raw = getattr(result, "cam_id", self._ccd_cam_id)
        cam_id = (
            cam_raw
            if isinstance(cam_raw, int) and not isinstance(cam_raw, bool)
            else self._ccd_cam_id
        )

        # 操作员已经自己改过设备字段 (或会话已经跑起来) 就不覆盖: 探测是提示,
        # 不是命令。
        untouched = (
            not self._running
            and self._camera.currentData() == self._ccd_preferred
            and self._cam_id.value() == self._ccd_cam_id
        )
        if untouched:
            self._apply_ccd_choice(kind, cam_id)
        self._set_probe_text(detail_text or f"自动探测：已选 {kind} cam_id={cam_id}")

    # ------------------------------------------------------------------ 自检
    def _device_key(self) -> tuple[str, int, int]:
        """自检唯一关心的三个字段 (其余表单值与硬件无关)。"""
        values = self.form_values()
        return values.camera, values.cam_id, values.slm_number

    def _check_stale(self) -> bool:
        """结论是不是对着另一组设备参数跑的 —— 改过设备字段就算过期。"""
        if self._check_key is None:
            return False
        return self._device_key() != self._check_key

    def _run_check(self) -> None:
        """「设备自检」按钮: 显式触发一次只读硬件枚举, 交给后台线程。

        若已有结论满足「同设备、未过期、且有非空结论」就直接**复用**而不启动线程
        (同一台设备的只读枚举在会话内随时重复点都该立即给出结论)。复用时要带上时间
        印记, 好让界面标出「复用上次自检」; 不改变结论的只读性质。
        """
        if self._checking or self._running or self._zero_reading:
            # 按钮已禁用, 这里是第二道闸 (程序方式触发 clicked 时也拦住)。
            return
        key = self._device_key()
        if (
            key == self._check_key
            and self._check_heading is not None
            and self._check_rows
            and not self._check_stale()
        ):
            # 同一设备、同一份结论还在 (非空), 直接复用, 不碰硬件。
            self._check_at = time.monotonic()
            self._check_reused = True
            self._render_check()
            return
        self._checking = True
        self._check_key = key
        self._check_at = time.monotonic()
        self._check_reused = False
        self._check_heading = ("pending", "正在自检…（枚举相机 / 探测 SLM）")
        self._check_rows = []
        self._check_btn.setEnabled(False)  # 一次只跑一份; 也避免连点排两个任务
        self._zero_btn.setEnabled(False)  # 自检正占着 SLM, 此刻读零级只会抢设备
        self._render_check()
        self._revalidate()

        task = HardwareCheckTask(*key)
        # 与 ``_SessionTask`` 一样不交给 Qt 自动回收: 回收早于信号送达就丢掉结论。
        task.setAutoDelete(False)
        self._check_task = task
        task.signals.done.connect(self._on_check_done)
        self._pool.start(task)

    def _on_check_done(self, payload: object) -> None:
        """自检结论回到 UI 线程: 逐项上屏, 再把两个按钮的决定权交回校验。"""
        self._checking = False
        self._check_task = None
        self._check_at = time.monotonic()
        self._check_reused = False
        if isinstance(payload, BaseException):
            # 自检脚本自己炸了 (SDK 缺失、进程被占用……) 也是一种必须说清的结论。
            self._check_heading = ("error", "自检未完成：检查本身出错")
            self._check_rows = [
                ("error", type(payload).__name__, str(payload) or "（无消息）")
            ]
        else:
            ok = bool(getattr(payload, "ok", False))
            summary = str(getattr(payload, "summary", lambda: "")())
            self._check_heading = (
                "ok" if ok else "error",
                summary or "硬件自检完成",
            )
            self._check_rows = [
                (
                    _CHECK_LEVELS.get(str(getattr(item, "level", "")), "pending"),
                    str(getattr(item, "label", "")),
                    str(getattr(item, "detail", "")),
                )
                for item in getattr(payload, "items", ())
            ]
        self._check_btn.setEnabled(not self._running)
        self._zero_btn.setEnabled(not self._running and not self._zero_reading)
        self._render_check()
        self._revalidate()

    def _render_check(self) -> None:
        """自检区: 结论 + 逐项明细, 颜色按当前主题取; 过期只是加一句提醒。"""
        if self._check_heading is None:
            self._check_view.clear()
            self._check_view.setVisible(False)
            return
        self._check_view.setVisible(True)
        kind, text = self._check_heading
        color = self._colors.get(kind, self._colors["pending"])
        parts = [
            f'<p style="margin:0 0 2px 0"><b style="color:{color}">'
            f"{escape(text)}</b></p>"
        ]
        for level, label, detail in self._check_rows:
            item_color = self._colors.get(level, self._colors["pending"])
            parts.append(
                f'<p style="margin:0"><span style="color:{item_color}">■ '
                f"{escape(label)}</span>{escape('：' + detail)}</p>"
            )
        if self._check_stale():
            parts.append(
                '<p style="margin:2px 0 0 0"><span style="color:'
                f'{self._colors["warn"]}">设备参数已改，请重新点「设备自检」</span></p>'
            )
        # 这次结论来自「直接复用」而不是重跑硬件时, 标出来让操作员知道。
        if self._check_reused and not self._check_stale():
            parts.append(
                '<p style="margin:2px 0 0 0"><span style="color:'
                f'{self._colors["pending"]}">复用上次自检（同设备、未过期）</span></p>'
            )
        self._check_view.setText("".join(parts))

    # -------------------------------------------------------------- 读取零级
    def _read_zero_order(self) -> None:
        """「读取零级」按钮: 只读地开一次相机量 0 级, 交给后台线程。

        与自检 / 运行同一套互斥: 自检正开着 SLM、会话正占着相机时再开一次,
        只会跟它们对抢设备。
        """
        if self._zero_reading or self._checking or self._running:
            # 按钮已禁用, 这里是第二道闸 (程序方式触发 clicked 时也拦住)。
            return
        values = self.form_values()
        self._zero_reading = True
        self._zero_btn.setEnabled(False)  # 一次只跑一份; 也避免连点排两个任务
        self._set_run_state("running", "正在读取 0 级…（开关一次相机）")
        self._revalidate()

        task = _ZeroOrderTask(values.camera, int(values.cam_id))
        # 与 ``_SessionTask`` 一样不交给 Qt 自动回收: 回收早于信号送达就丢掉结论。
        task.setAutoDelete(False)
        self._zero_task = task
        task.signals.done.connect(self._on_zero_done)
        self._pool.start(task)

    def _on_zero_done(self, payload: object) -> None:
        """读数回到 UI 线程: 按实测上限放宽控件, 再把按钮的决定权交回校验。"""
        self._zero_reading = False
        self._zero_task = None
        if isinstance(payload, BaseException):
            # 全暗 / 0 级贴边都是操作性结论: 原因原样上屏, 上限保持原值。
            self._zero_state = (
                "error",
                f"读取零级失败：{payload}。"
                f"目标边长上限仍为 {self._zero_limit} px（保守值）。",
            )
            self._set_run_state("failed", f"读取零级失败：{payload}")
        else:
            max_side = int(getattr(payload, "max_side_px", 0))
            summary = str(getattr(payload, "summary", lambda: "")())
            if max_side > 0:
                self._zero_limit = max_side
                # 上限就是物理依据: 0 级到最近帧边的距离折成的边长。
                # 短边与矩形长边都按同一物理极限放宽 —— 长边是矩形两维中更大者,
                # 任一单边都不能超过帧可容纳的极限 (0 级到帧边距离 - 留白)。
                # 此前只有 _side 更新、_side_long 仍锁保守值, 矩形长边敲不进探测
                # 允许的值 (用户反馈"被限制在 400") —— 一并放宽并夹回越限值。
                for spin in (self._side, self._side_long):
                    spin.setMaximum(float(max_side))
                    if spin.value() > max_side:
                        spin.setValue(float(max_side))
            center = tuple(getattr(payload, "center", (0, 0)))
            shape = tuple(getattr(payload, "frame_shape", (0, 0)))
            self._zero_state = (
                "ok",
                f"已读零级：目标边长上限 {max_side} px"
                f"（0 级 ({center[0]}, {center[1]})，帧 {shape[1]}×{shape[0]}）；"
                "采样窗半跨按边长自动派生。",
            )
            self._set_run_state("complete", summary or "读取零级完成")
        self._zero_btn.setEnabled(not self._running and not self._checking)
        self._render_zero()
        self._revalidate()

    def _render_zero(self) -> None:
        """② 目标的常驻说明: 上限从哪来、现在是多少 (跟随主题换色)。"""
        if self._zero_state is None:
            text = (
                f"未读取零级：目标边长上限 {self._zero_limit} px（保守值）。"
                "点「读取零级」按实际光轴放宽。"
            )
            color = self._muted()
        else:
            kind, text = self._zero_state
            color = self._colors.get(kind, self._muted())
        self._zero_view.setText(f'<span style="color:{color}">{escape(text)}</span>')

    # ------------------------------------------------------------------ 运行
    def _start_run(self) -> None:
        """启动会话: 目录在这一刻确定, 之后改表单不影响本次运行。"""
        if self._running or self._checking or self._zero_reading:
            # 自检正开着 SLM、读零级正开着相机, 此时开跑两边会抢设备
            # (按钮已禁用, 这是第二道闸)。
            return
        values = self.form_values()
        errors, warnings = params.validate_form(values)
        note = params.capture_grid_note(values)
        self._last_notes = [note] if note else []
        # 与 ③ 的最近一版对比: 提醒变了而没走 _revalidate 时, 旧勾选不能算数。
        warnings_changed = warnings != self._last_warnings
        self._last_errors = errors
        self._last_warnings = warnings
        if errors or (warnings and (warnings_changed or self.pending_acks())):
            # 按钮已禁用, 这里是第二道闸 (程序方式触发 clicked 时也拦住):
            # 有阻塞, 或黄字还没逐条勾「已确认」, 都不建目录、不进运行态。
            self._render_checks()
            self._render_ack_list()
            self._sync_run_button()
            return

        out = params.plan_session_dir()
        self._session_dir = out
        self._table.setRowCount(0)
        self._bar.setRange(0, 1)
        self._bar.setValue(0)
        self._current.setText("等待第一个阶段…")
        self._show_exports(False)
        self._counts.setText("")
        self._session_label.setText(f"会话目录: {out}")
        self._live_view.clear()
        # 开跑后视线在 ⑥ / ⑤: 收起 ⑦ 的正文, 把高度让给实时画面与阶段表 (随时可再展开,
        # 它不是「一次性的」, 只是不该和正在刷新的两块抢地方)。
        self._plan_toggle.setChecked(False)

        # 新一轮: 复位验收停止 flag, 并按本次快照决定「停止」按钮可不可用
        # (只有开着验收阈值循环的会话在两轮之间才有得可停)。
        self._acceptance_stop.clear()
        stop_enabled = bool(values.acceptance_enabled)

        self._running = True
        self._set_form_enabled(False)
        self._sync_run_button()
        self._stop_btn.setEnabled(stop_enabled)
        self._stop_btn.setToolTip(
            "点击后当前这一轮跑完、下一轮开始前停下（保留已得的最好结果）"
            if stop_enabled
            else "验收阈值循环未开启：无跨轮可停，此按钮不起作用"
        )
        self._set_run_state("running", "运行中：本次会话的参数已锁定，改表单不影响它。")

        task = _SessionTask(values, out, should_stop=self._acceptance_stop.is_set)
        # 不交给 Qt 自动回收: 回收早于信号送达就会丢掉最后一条结论。
        task.setAutoDelete(False)
        task.signals.stage.connect(self._on_stage)
        task.signals.frame.connect(self._live_view.apply_payload)
        task.signals.finished.connect(self._on_finished)
        task.signals.failed.connect(self._on_failed)
        self._task = task
        _LIVE_TASKS.add(task)
        self._poll.start()
        self._pool.start(task)

    def _on_stage(self, name: str, status: str) -> None:
        """阶段回调 (已排队回主线程): 让这一次刷新提前发生。"""
        logger.debug("迁移学习阶段 {} → {}", name, status)
        self._refresh_progress()

    def _on_finished(self, out: str) -> None:
        """会话跑完: 停轮询, 把最终产物再读一遍, 放回可编辑状态。"""
        self._settle("complete", f"会话完成：{out}")
        self._show_exports(True)
        # 开过验收阈值循环的会话, 补一行「验收阈值: 通过 / 未达标至上限后停止」。
        # 只读文件, 产物缺失时 :func:`acceptance_verdict_line` 返回空串、绝不抛。
        line = acceptance_verdict_line(Path(out))
        if line:
            self._status.setText(f"{self._status.text()}<br>{line}")
        reuse_line = reuse_latest_verdict_line(Path(out))
        if reuse_line:
            self._status.setText(f"{self._status.text()}<br>{reuse_line}")
        weight_line = model_weight_saved_line(Path(out))
        if weight_line:
            self._status.setText(f"{self._status.text()}<br>{weight_line}")

    def _on_failed(self, message: str) -> None:
        """会话失败: 停轮询, 把 journal 里已写下的阶段留在表上。"""
        self._settle("failed", f"会话失败：{message}")
        self._show_exports(False)
        # 边界: 目标被倾斜相位位移推出采样窗边 → 自动**增大**倾斜相位周期
        # (位移 ∝ 1/period), 并回填输入栏; 只回填、不自动重跑 (操作员确认几何
        # 后再手动启动)。只有打开倾斜相位时位移才存在, 关着时不触发。
        if self._tilt.isChecked() and self._auto_bump_period(message):
            self._set_run_state(
                "failed",
                "目标触碰相机窗口边界：已自动增大倾斜相位周期并回填输入栏，"
                "请确认几何（或点「读取零级」核准边长）后再手动运行。",
            )

    def _auto_bump_period(self, message: str) -> bool:
        """识别 ``target intersects camera window boundary`` 失败并回填更大的周期。

        返回 ``True`` 表示本次失败确属边界且已回填大周期; ``False`` 表示无关 / 已在
        倾斜相位周期上限 (不再改, 避免死循环)。纯逻辑、不跑硬件, 供自动化直接调用。
        """
        if "target intersects camera window boundary" not in message:
            return False
        if self._running or self._task is not None:
            return False  # 会话尚未真正收尾, 不改。
        period = self._period.value()
        if period >= self._period.maximum() - 1e-9:
            return False  # 已到上限, 改不动。
        new_period = min(period * 1.5, float(self._period.maximum()))
        self.set_period(new_period)
        return True

    def _settle(self, kind: str, message: str) -> None:
        """运行结束后的共同收尾 (完成 / 失败 / 停止都走这里复位)。"""
        self._poll.stop()
        self._running = False
        task, self._task = self._task, None
        if task is not None:
            _LIVE_TASKS.discard(task)
        # 「停止」只在会话跑着的时候可用; 收尾一律禁用并复位 tooltip。
        self._stop_btn.setEnabled(False)
        self._stop_btn.setToolTip(
            "点击后当前这一轮跑完、下一轮开始前停下；保留已得的最好结果。"
            "只对开启验收阈值循环的会话生效。"
        )
        self._refresh_progress()
        self._set_form_enabled(True)
        self._set_run_state(kind, message)
        self._revalidate()

    def _show_exports(self, shown: bool) -> None:
        """⑤ 进度区: 完成后换成导出按钮行, 下次运行 / 失败时还原进度条。"""
        self._export_row.setVisible(shown)
        self._bar.setVisible(not shown)
        self._current.setVisible(not shown)

    def _best_phase_path(self) -> Path | None:
        """最终 best 相位文件 (首个形状); 缺失时提示并返回 ``None``。"""
        if not self._session_dir:
            return None
        shapes = self.form_values().shapes
        shape = shapes[0] if shapes else "square"
        path = self._session_dir / "anchored" / f"{shape}_best.npy"
        if not path.exists():
            QMessageBox.warning(self, "导出", f"未找到最终相位文件：\n{path}")
            return None
        return path

    def _export_png(self) -> None:
        """导出总结对比图; 没有就回退到实时画面的最后一帧 CCD。"""
        if not self._session_dir:
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "导出 PNG", str(self._session_dir), "PNG (*.png)"
        )
        if not dest:
            return
        src = self._session_dir / "summary" / "ccd_model_comparison.png"
        if src.exists():
            shutil.copyfile(src, dest)
            return
        from ao_shaping.gui.slm.transfer_desktop.archive_replay import export_ccd_png

        payload = self._live_view.last_payload()
        frame = payload.get("frame") if payload else None
        if frame is None:
            QMessageBox.warning(self, "导出 PNG", "没有可导出的 CCD 帧。")
            return
        export_ccd_png(dest, frame)

    def _export_phase_csv(self) -> None:
        """导出最终 best 相位 (弧度) 为 CSV。"""
        best = self._best_phase_path()
        if best is None:
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "导出相位 CSV", str(best), "CSV (*.csv)"
        )
        if not dest:
            return
        from ao_shaping.gui.slm.transfer_desktop.archive_replay import write_phase_csv

        import numpy as np

        write_phase_csv(dest, np.load(best), radians=True)

    def _export_gray_csv(self) -> None:
        """导出 SLM 实际下发的灰度相位为 CSV (0..1023 整数)。"""
        best = self._best_phase_path()
        if best is None:
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "导出 SLM 灰度 CSV", str(best), "CSV (*.csv)"
        )
        if not dest:
            return
        from ao_shaping.gui.slm.transfer_desktop.archive_replay import (
            phase_to_gray,
            write_phase_csv,
        )

        import numpy as np

        write_phase_csv(dest, phase_to_gray(np.load(best)), radians=False)

    # ------------------------------------------------------------------ 进度
    def _open_replay(self) -> None:
        """打开归档回放弹窗: 扫描整个 debug 根下所有历史 pkl (只读, 无硬件)。"""
        from ao_shaping.gui.slm.transfer_desktop.archive_replay import (
            DEBUG_ROOT,
            open_archive_replay,
        )

        open_archive_replay(self, DEBUG_ROOT, preselect=self._session_dir)

    def _refresh_progress(self) -> None:
        """读会话目录, 刷新阶段表 / 进度条 / 当前阶段 / 汇总。"""
        view = collect_progress(self._session_dir)
        stages = view["stages"]
        self._sync_table(stages)

        counts = view["counts"]
        total = max(int(counts.get("total", 0)), 1)
        settled = sum(
            int(counts.get(key, 0)) for key in ("complete", "failed", "skipped")
        )
        self._bar.setRange(0, total)
        self._bar.setValue(min(settled, total))

        current = view["current"]
        if current is not None:
            status = STATUS_LABELS.get(str(current["status"]), str(current["status"]))
            self._current.setText(f"当前：{current['label']} · {status}")
        elif stages:
            self._current.setText("尚未开始")
        else:
            self._current.setText("尚无阶段计划")
        progress_line = acceptance_progress_line(self._session_dir)
        if progress_line:
            self._current.setText(f"{self._current.text()}  |  {progress_line}")

        self._counts.setText(
            " · ".join(
                f"{STATUS_LABELS[key]} {counts.get(key, 0)}"
                for key in ("complete", "running", "pending", "skipped", "failed")
            )
        )

    def _sync_table(self, stages: Sequence[Mapping[str, Any]]) -> None:
        """按计划顺序重画阶段表 (十几个阶段, 重建比就地更新简单且无歧义)。"""
        table = self._table
        table.setRowCount(len(stages))
        for row, stage in enumerate(stages):
            name = str(stage["name"])
            status = str(stage["status"])
            cells = (
                str(stage["index"]),
                params.stage_label(name, str(stage["label"])),
                STATUS_LABELS.get(status, status),
                _duration_text(stage["duration_s"]),
                _stage_summary(stage),
            )
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column != 4:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if column == 2:
                    item.setForeground(
                        QColor(self._colors.get(status, self._colors["pending"]))
                    )
                table.setItem(row, column, item)

    # -------------------------------------------------------------- 计划预览
    def _current_shape_key(self) -> str | None:
        """② 目标里当前勾选的那个形状 (互斥单选; 一个没选就是 ``None``)。"""
        shapes = getattr(self, "_shapes", None) or {}
        for name in campaign_plan.PLAN_SHAPES:
            radio = shapes.get(name)
            if radio is not None and radio.isChecked():
                return name
        return None

    def _highlight_tint(self) -> str:
        """当前形状那一行的底色: 调色板选中色 + 透明度, 跟着主题走。"""
        highlight = self.palette().color(QPalette.ColorRole.Highlight)
        return (
            f"rgba({highlight.red()}, {highlight.green()}, {highlight.blue()}, 48)"
        )

    def _render_plan_matrix(self) -> None:
        """按 ② 的当前形状给矩阵那一行上底色 (只重上色, 不重建控件)。

        ``▸`` 只是给色觉差异留的第二条线索: 底色很淡, 单独靠它不够显眼。
        """
        if not getattr(self, "_plan_ready", False):
            return
        muted = self._muted()
        try:
            cells = campaign_plan.matrix_cells(self._current_shape_key())
        except (TypeError, ValueError, AttributeError):  # 数据层承诺不抛; 显示路径再兜一层, 绝不连累整页
            cells = ()
        if not cells or not self._plan_cells:
            self._plan_matrix_text = "读取失败"
            self._plan_matrix_error.setStyleSheet(f"color: {muted};")
            self._plan_matrix_error.setVisible(True)
            return
        self._plan_matrix_error.setVisible(False)

        tint = f"background-color: {self._highlight_tint()}; border-radius: 4px;"
        for cell in cells:
            widget = self._plan_cells.get((cell.shape, cell.stage))
            if widget is not None:
                widget.setStyleSheet(tint if cell.highlight else "")

        for header in self._plan_stage_headers:
            header.setStyleSheet(f"color: {muted};")
        self._plan_corner.setStyleSheet(f"color: {muted};")

        highlight = {cell.shape for cell in cells if cell.highlight}
        for shape, row_label in self._plan_row_labels.items():
            on = shape in highlight
            row_label.setText(
                ("▸ " if on else "　 ") + self._plan_shape_labels[shape]
            )
            font = row_label.font()
            font.setWeight(
                QFont.Weight.DemiBold if on else QFont.Weight.Normal
            )
            row_label.setFont(font)
            row_label.setStyleSheet(
                f"color: {muted};" if not on else ""
            )

        # 与屏上同一份文字 (行标题 + 每格), 供 ``campaign_matrix_text()`` 取用。
        lines = [" | ".join(["形状", *self._plan_stage_labels])]
        for shape in self._plan_row_labels:
            row = [self._plan_row_labels[shape].text()]
            row.extend(
                self._plan_cells[(shape, stage)].text()
                for stage in self._plan_stage_order
            )
            lines.append(" | ".join(row))
        self._plan_matrix_text = "\n".join(lines)

    def _paint_plan_summary(self) -> None:
        """合计行: 段数 / 块数 / 墙钟 / 落盘 + 口径 (长就折行, 次级色)。"""
        if not getattr(self, "_plan_ready", False):
            return
        try:
            text = campaign_plan.campaign_summary().text
        except (TypeError, ValueError, AttributeError):  # 同上: 显示路径只降级, 不抛
            text = "读取失败：无法计算整轮合计"
        self._plan_summary_text = text
        self._plan_summary.setText(text)
        self._plan_summary.setStyleSheet(f"color: {self._muted()};")

    def _reload_plan_costs(self) -> None:
        """按输入框的目录重读段成本 (只读 cost.json, 不开设备, 不弹错误框)。

        空 / 不存在 / 不是目录 → 数据层本来就把实测列留空、口径给「首次运行」; 这里
        只补一句说明, 让「没历史」与「路径写错」在界面上分得开。
        只在点按钮 / 回车时发生 —— 每个按键都扫一遍目录没有意义。
        """
        if not getattr(self, "_plan_ready", False):
            return
        raw = self._cost_dir.text().strip()
        path: Path | None = Path(raw).expanduser() if raw else None
        note = ""
        try:
            if path is not None and not path.is_dir():
                note = f"目录不可读：{raw}（只显示计划列）"
                path = None
            rows = campaign_plan.cost_rows(path)
            caption = campaign_plan.cost_caption(path)
        except (OSError, TypeError, ValueError):  # 数据层承诺不抛; 读盘兜底, 失败只退成计划列
            note = "读取失败：只显示计划列"
            caption = ""
            path = None
            # 连计划列也读不出来才是空表; 正常失败仍留 16 行计划值, 表格不整块消失。
            try:
                rows = campaign_plan.cost_rows(None)
            except (TypeError, ValueError, AttributeError):
                rows = ()
        self._cost_path = path
        self._cost_rows = rows
        self._cost_caption_text = caption
        self._cost_note_text = note
        self._paint_plan_costs()
        self._paint_cost_caption()
        # 同一个目录、同一次点击: ⑥ 逐块页跟着读 (不另开输入框)。
        self.reload_campaign_progress()

    def _paint_plan_costs(self) -> None:
        """把缓存的段成本行铺进表格 (没实测的格子写「—」, 绝不是 0)。"""
        table = self._cost_table
        rows = self._cost_rows
        table.setRowCount(len(rows))
        for row, cost in enumerate(rows):
            cells = (
                cost.shape_label,
                cost.stage_label,
                str(cost.planned_blocks),
                f"{cost.expected_seconds:.0f} s",
                MISSING
                if cost.measured_seconds is None
                else f"{cost.measured_seconds:.1f} s",
                MISSING
                if cost.measured_blocks is None
                else str(cost.measured_blocks),
                cost.delta_text or MISSING,
            )
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column != 1:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if column == 6 and text != MISSING:
                    # 快 = 省时间 (绿), 慢 = 要复核 (黄); 「≈计划」不表态, 保持默认字色。
                    kind = _DELTA_COLORS.get(text[:1])
                    if kind is not None:
                        item.setForeground(QColor(self._colors[kind]))
                table.setItem(row, column, item)
        # 行数固定 (16) → 高度按内容钉死, 竖向滚动交给外面的正文滚动区: 同一块里套
        # 两层竖向滚动条, 滚轮落在哪层全看指针位置, 操作上很难受。
        header_h = table.horizontalHeader().sizeHint().height()
        table.setFixedHeight(header_h + len(rows) * _COST_ROW_HEIGHT + 8)

    def _paint_cost_caption(self) -> None:
        """口径行: (目录不可读时的说明) + 数据层给的那句成本口径。"""
        if not getattr(self, "_plan_ready", False):
            return
        muted = self._muted()
        parts: list[str] = []
        if self._cost_note_text:
            parts.append(
                f'<span style="color:{self._colors["warn"]}">'
                f"{escape(self._cost_note_text)}</span>"
            )
        if self._cost_caption_text:
            parts.append(
                f'<span style="color:{muted}">'
                f"{escape(self._cost_caption_text)}</span>"
            )
        self._cost_caption.setText("<br>".join(parts))

    # ------------------------------------------------------------------ 外观
    def _muted(self) -> str:
        """次级文字色 (跟随调色板的占位文字色, 不写死灰值)。"""
        return self.palette().color(QPalette.ColorRole.PlaceholderText).name()

    def _set_run_state(self, kind: str, message: str) -> None:
        """运行状态行: 记下原文, 换主题时按新配色重画。"""
        self._status_kind = kind
        self._status_message = message
        color = self._colors.get(kind, self._colors["pending"])
        self._status.setText(f'<span style="color:{color}">{escape(message)}</span>')

    def _apply_theme(self) -> None:
        """按当前调色板重刷状态色 (亮 / 暗主题各一档)。"""
        self._colors = _palette_colors(self.palette())
        self._render_checks()
        self._render_check()
        self._render_zero()
        self._set_run_state(self._status_kind, self._status_message)
        # G1 的提示行用次级色: 换主题只重上色, **不重建**勾选行 —— 系统明暗切换
        # (或日落自动切换) 不是「改了参数」, 顺手作废一遍确认只会让人白勾一次。
        if getattr(self, "_ack_hint", None):
            self._style_ack_hint()
        # ⑦ 的底色 / 次级色都取自调色板: 换主题时一并重上色 (不重读目录)。
        if getattr(self, "_plan_ready", False):
            self._render_plan_matrix()
            self._paint_plan_summary()
            self._paint_plan_costs()
            self._paint_cost_caption()
        # ⑥ 逐块页: 只重上色, 不重读目录。
        self._paint_campaign_theme()

    def changeEvent(self, event: QEvent) -> None:
        """系统主题切换 → 重算状态色 (Qt 覆写点, 驼峰命名由 Qt 决定)。"""
        if event.type() == QEvent.Type.PaletteChange:
            self._apply_theme()
        super().changeEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        """关窗: 停轮询, 有限度地等在飞会话落地 —— ``run_session`` 没有取消接口。"""
        self._poll.stop()
        if self._running or self._zero_reading:
            # 完成信号排在 UI 事件队列里, 干等会拿不到; 必须泵事件驱动回调。
            # 读零级也要一起等: 相机还没关就退池, 下次打开会撞上占用。
            deadline = time.monotonic() + _CLOSE_GRACE_S
            while (self._running or self._zero_reading) and time.monotonic() < deadline:
                QApplication.processEvents()
                time.sleep(0.02)
        super().closeEvent(event)
