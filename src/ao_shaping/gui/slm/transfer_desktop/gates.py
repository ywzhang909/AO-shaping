"""#242 剩下三道安全门的页面侧适配: G2 红停 / G3 逐块饱和 / P4 确认后继续。

本模块只做两件事, 都不碰 Qt
---------------------------
1. **把判定层已经给出的结论翻成人话**: G2 (拟合验证相关性低于阈值 → 红停) 与
   G2b (零级漂移超黄警 → 黄警) 都是 :mod:`~ao_shaping.tools.slm.transfer.workflow`
   与 :mod:`~ao_shaping.tools.slm.transfer.optical_probe` 已经判好的事; G3 的
   逐块饱和同理, 判据在 :mod:`~ao_shaping.tools.slm.transfer.replay_phases`。
   页面不重算阈值, 也不在自己这层"稍微放宽一点"。
2. **把 P4 的「确认后继续」标记写进整轮目录**: 段边界的闸本来就是
   :mod:`~ao_shaping.tools.slm.transfer.campaign_control` 的三个小文件 (挂起 /
   确认 / 否决), runner 在每段末尾用
   :func:`~ao_shaping.tools.slm.transfer.session_record.confirm_at_boundary` 读它们。
   这里只是给页面一套"点按钮 = 写/删一个标记"的函数, 文件名一律从那个模块
   import, 不另抄一份约定。

三个门各自的边界
----------------
- **G2 红停的是「后续形状」, 不是这一场**: 相关性掉了说明本场拟合的模型不可信,
  继续把同一个模型套到下一个形状上只会把错误放大。所以它是一行**结论**, 页面上
  红字说明"别再往下跑", 而不是把「开始运行」按钮锁死 —— 换曝光、重测之后再跑
  是正当操作。
- **G2b 只黄不红**: 20 px 以内照常复用, 20–40 px 是"光轴在走"的黄警, 40 px 以上
  复用判定自己就会否决 (那是 ``assess_reuse`` 的事, 页面不再判一遍)。
- **G3 停的是块, 不是整轮**: 核心峰值越限只让该块的验收结论无效, 其余块照跑;
  段行只当"这里有几块有事"的指针, 所以段行永远最多黄, 不会红。

缺省语义
--------
产物缺失、字段是 ``"—"``、目录不存在: 一律返回"不报警"的空结论。没有证据时报警
与没有证据时放行, 前者会让操作员学会无视黄色 —— 所以这里选后者 (与
``assess_reuse`` 的 fail-open on missing reference 同一取向)。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from ao_shaping.tools.slm.transfer import campaign_control
from ao_shaping.tools.slm.transfer.campaign_progress import BlockRow, StageRow
from ao_shaping.tools.slm.transfer.replay_phases import (
    CORE_SATURATION_COUNTS,
    SATURATION_FRACTION,
    core_saturated,
)

__all__ = [
    "SaturationMark",
    "StageGateState",
    "arm_confirm_gate",
    "answer_confirm_gate",
    "block_saturation",
    "clear_pause_request",
    "correlation_red_stop_line",
    "disarm_confirm_gate",
    "request_pause",
    "stage_gate_state",
    "stage_saturation",
    "zero_drift_warning",
]

#: 页面在挂起「确认后继续」时写进请求文件里的理由 (事后翻目录能看出是谁挂的)。
_CONFIRM_ARM_REASON = "页面：每段结束后等人工确认"
_PAUSE_REASON = "页面：本段结束后暂停"
_CONFIRM_ANSWER_REASON = "页面：段边界人工答复"


def _number(value: object) -> float | None:
    """数值字段 → ``float``; ``"—"`` / 布尔 / 非数 → ``None`` (缺值不是 0)。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _read_json(path: Path) -> dict | None:
    """读一个 JSON dict; 不存在 / 半截 / 不是 dict 一律 ``None`` (绝不抛)。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.debug("安全门读数失败: {}", path)
        return None
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------- G2: 相关性红停


def correlation_red_stop_line(session_dir: str | Path) -> str:
    """G2: 拟合验证相关性低于会话阈值时的一行红停结论; 不该显示时返回 ``""``。

    判定在 :func:`~ao_shaping.tools.slm.transfer.workflow.acceptance_verdict`:
    它把"验证集相关性低于 ``acceptance_lower`` (默认 0.85, 见
    ``TransferSettings.acceptance_lower``) 或压根没测到"判成 ``correlation_red_stop``。
    本函数只读取这个布尔量与会话自己的 ``lower`` 阈值措辞, **不重述 0.85** ——
    操作员在会话里改过阈值时, 这句话跟着改。

    Args:
        session_dir: 会话目录 (读 ``session.json`` / ``rounds.json`` / 拟合产物)。

    Returns:
        红停结论; 相关性达标、会话没跑过拟合、或产物缺失时为 ``""``。
    """
    root = Path(session_dir)
    if not root.is_dir():
        return ""
    from ao_shaping.tools.slm.transfer.workflow import acceptance_verdict

    verdict = acceptance_verdict(root)
    if not verdict.get("correlation_red_stop"):
        return ""
    lower = _number(verdict.get("lower"))
    bound = "—" if lower is None else f"{lower:.2f}"
    correlation = _number(verdict.get("measured_correlation"))
    if correlation is None:
        return f"红停（G2）：本场没有可用的验证相关性，按低于阈值 {bound} 处理"
    return (
        f"红停（G2）：验证相关性 {correlation:.3f} < 阈值 {bound}，"
        "拟合出的模型不可信，后续形状不要再接着跑"
    )


# ------------------------------------------------------------ G2b: 零级漂移黄警


def zero_drift_warning(session_dir: str | Path) -> tuple[str, str]:
    """G2b: 零级漂移的 ``(级别, 一句话)``; 不需要提示时返回 ``("", "")``。

    级别是 ``"warn"`` (20 px 黄警) 或 ``"hard"`` (40 px 硬上限, 复用判定本身已否决)。
    ``"ok"`` 档不给文案: 一次会话大部分时候漂移都在几个像素, 常驻一行"漂移 2.1 px"
    只会把真正要看的那几行挤走。

    参照零点来自 ``optical_probe.json`` 的 ``reference``; 没有参照 (首次跑 / 没走
    复用探针) 时无话可说 —— 与 :func:`optical_probe.zero_drift_verdict` 的
    fail-open 一致。

    Args:
        session_dir: 会话目录 (读 ``<out>/optical_probe.json``)。
    """
    root = Path(session_dir)
    if not root.is_dir():
        return "", ""
    data = _read_json(root / "optical_probe.json")
    if data is None:
        return "", ""
    current = data.get("current")
    reference = data.get("reference")
    if not isinstance(current, dict) or not isinstance(reference, dict):
        return "", ""
    current_zero = current.get("zero_xy")
    reference_zero = reference.get("zero_xy")
    if not isinstance(current_zero, (list, tuple)) or not isinstance(
        reference_zero, (list, tuple)
    ):
        return "", ""

    from ao_shaping.tools.slm.transfer.optical_probe import zero_drift_verdict

    level, text = zero_drift_verdict(current_zero, reference_zero)
    if level == "ok":
        return "", ""
    return level, text


# ------------------------------------------------------- G3: 逐块饱和 (只停本块)


@dataclass(frozen=True)
class SaturationMark:
    """一块 / 一段的饱和结论。

    Attributes:
        level: ``"stop"`` (核心越限, 本块验收作废) / ``"warn"`` (有饱和像素) /
            ``"ok"`` (没有证据说它饱和)。
        note: 追加到「说明」列尾巴的一句话; ``ok`` 时为空串。
        detail: 悬停说明 —— 数 + 判据口径, 免得操作员去翻代码。
    """

    level: str
    note: str
    detail: str


_EMPTY_MARK = SaturationMark("ok", "", "")


def block_saturation(block: BlockRow) -> SaturationMark:
    """G3: 单个块的饱和结论 (核心越限 = 只作废本块, 其余块照跑)。

    两条口径都来自 :mod:`~ao_shaping.tools.slm.transfer.replay_phases`, 与写侧
    ``validate`` 记录 ``saturation_pixels`` 时同一套:

    * 核心峰值 ``>= 4014`` (4095 制下满阱的 250/255) → 本块作废;
    * 全幅还有饱和像素但核心没越限 → 黄警 (目标亮但没削顶)。

    Args:
        block: 整轮逐块视图的一行 (:class:`campaign_progress.BlockRow`)。
    """
    core = _number(block.core_peak)
    saturated = block.saturated_px
    saturated_count = (
        saturated
        if isinstance(saturated, int) and not isinstance(saturated, bool)
        else None
    )
    if core is not None and core_saturated(core):
        return SaturationMark(
            "stop",
            "核心饱和 · 仅停本块",
            f"核心峰值 {core:.0f} ≥ {CORE_SATURATION_COUNTS:.0f}"
            f"（满阱判据 {SATURATION_FRACTION:.3f}）：本块验收作废，不红停整轮",
        )
    if saturated_count is not None and saturated_count > 0:
        return SaturationMark(
            "warn",
            f"饱和 {saturated_count} px · 黄警",
            f"全幅有 {saturated_count} 个像素达满阱判据 {SATURATION_FRACTION:.3f}，"
            f"核心峰值 {core if core is None else round(core)} 未越限：只黄警",
        )
    return _EMPTY_MARK


def stage_saturation(stage: StageRow) -> SaturationMark:
    """G3: 一个段的饱和汇总 —— **指针**, 不是判决 (段行最多黄, 永远不红)。

    G3 的停止点是块: 核心饱和只让那一块的验收结论无效, 其余块照跑。所以段行上写成
    "核心饱和 2 块（仅停本块）"这种指路的话, 用黄字; 用红字会让一眼扫过的人以为
    整段被判死。

    Args:
        stage: 整轮逐块视图的一段 (:class:`campaign_progress.StageRow`)。
    """
    stopped: list[int] = []
    warned: list[int] = []
    for block in stage.blocks:
        mark = block_saturation(block)
        if mark.level == "stop":
            stopped.append(block.index)
        elif mark.level == "warn":
            warned.append(block.index)
    if not stopped and not warned:
        return _EMPTY_MARK
    parts: list[str] = []
    if stopped:
        parts.append(f"核心饱和 {len(stopped)} 块（仅停本块）")
    if warned:
        parts.append(f"饱和黄警 {len(warned)} 块")
    detail: list[str] = []
    if stopped:
        detail.append("核心饱和的块: " + ", ".join(f"#{index}" for index in stopped))
    if warned:
        detail.append("有饱和像素的块: " + ", ".join(f"#{index}" for index in warned))
    detail.append("G3 的停止粒度是块：其余块照常跑完")
    return SaturationMark("warn", " · ".join(parts), "；".join(detail))


# ------------------------------------------------ P4: 段边界「确认后继续」闸


@dataclass(frozen=True)
class StageGateState:
    """整轮目录里两个协作闸的当前状态 (只读, 每次三个 ``stat``)。

    Attributes:
        confirm_pending: 「确认后继续」闸挂着 —— runner 跑到段边界会停下等答复。
        pause_requested: 段间暂停请求还在目录里。
    """

    confirm_pending: bool
    pause_requested: bool


def stage_gate_state(campaign_dir: str | Path) -> StageGateState:
    """读整轮目录的闸状态; 目录不存在时两个都是 ``False``。"""
    return StageGateState(
        confirm_pending=campaign_control.stage_confirm_requested(campaign_dir),
        pause_requested=campaign_control.stage_stop_requested(campaign_dir),
    )


def arm_confirm_gate(campaign_dir: str | Path) -> Path:
    """挂起「确认后继续」: 之后每个段边界都会等页面上的答复。"""
    return campaign_control.request_stage_confirm(
        campaign_dir, reason=_CONFIRM_ARM_REASON, requested_by="transfer_desktop"
    )


def disarm_confirm_gate(campaign_dir: str | Path) -> bool:
    """拆掉闸 (连同可能残留的答复标记); 返回是否删掉了什么。"""
    return campaign_control.clear_stage_confirm(campaign_dir)


def answer_confirm_gate(campaign_dir: str | Path, *, confirmed: bool) -> Path:
    """答复正在等待的段边界: 确认继续 / 否决停下 (否决优先, 超时也算否决)。

    写的是 :mod:`campaign_control` 约定的那两个标记之一, runner 下一次轮询就读到。
    页面**不清理**挂起标记 —— 那是 runner 在
    :func:`~ao_shaping.tools.slm.transfer.session_record.confirm_at_boundary` 里
    读完之后的职责; 在这里顺手删掉, 会让"答复了但从没被消费"这件事无从查证。

    Args:
        campaign_dir: 整轮输出目录 (``--out`` 那一层)。
        confirmed: ``True`` 写确认标记, ``False`` 写否决标记。
    """
    name = (
        campaign_control.CONFIRM_RESUME_FILE
        if confirmed
        else campaign_control.CONFIRM_REJECT_FILE
    )
    path = Path(campaign_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reason": _CONFIRM_ANSWER_REASON,
        "requested_by": "transfer_desktop",
        "confirmed": bool(confirmed),
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return path


def request_pause(campaign_dir: str | Path) -> Path:
    """请求「本段结束后暂停」(与页面上那个内存 stop 旗标同一个停止点)。"""
    return campaign_control.request_stage_stop(
        campaign_dir, reason=_PAUSE_REASON, requested_by="transfer_desktop"
    )


def clear_pause_request(campaign_dir: str | Path) -> bool:
    """撤销段间暂停请求; 返回是否真的删掉了。"""
    return campaign_control.clear_stage_stop(campaign_dir)
