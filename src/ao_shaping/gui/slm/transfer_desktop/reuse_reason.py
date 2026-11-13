"""光路复用的结论读法: 从 ``optical_probe.json`` 还原一行给操作员看的话。

为什么单独一个模块
------------------
:mod:`~ao_shaping.gui.slm.transfer_desktop.page` 是整页的桌面页, Qt 与整条会话
逻辑都在里面; 而「把探针判定读成一行结论」既不需要 Qt, 也不碰设备 —— 它只是读
会话目录里**已经落盘**的 ``optical_probe.json``, 纯 JSON 读法 + 文案拼装。把它
留在页面里会让页面再涨几十行, 也让这条纯逻辑无法在无 Qt 的环境里被单独测; 所以
它搬到这里, 页面只 import 这三个函数, 页面侧的调用路径 (``page.reuse_latest_verdict_line``
/ ``page._reuse_failure_note``) 一字不变。

只读契约
--------
三个函数都**只读**会话目录: 目录不存在、产物缺失或损坏时返回空串或降级文案,
绝不抛 —— 页面拿它填状态行与阶段表的「说明」列, 不能因为一行说明把完成界面打崩。
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

__all__ = [
    "_reuse_failure_note",
    "_reuse_latest_failure_reason",
    "reuse_latest_verdict_line",
]


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

    每条 reason 对应 ``same_path`` 的一个判据 (光斑形状 / 零点 / 曝光 / 格式 /
    位深 / 序列号), 任一项不满足就 listed。绝不抛异常; 读不到字段时给「不可比」。
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
    # 光斑形状 (主判据): 产物里带 ``criteria`` 且形状这条真的**未过**时才追加一行。
    # 「无平场帧, 形状判据不可用」是退路文案 (不是未过), 不追加 —— 没有形状信息
    # 时输出与既往一字不差。
    criteria = data.get("criteria")
    if isinstance(criteria, dict):
        shape = criteria.get("shape")
        if isinstance(shape, str) and (
            shape.startswith("形状判据未过") or shape.startswith("平场帧不可比")
        ):
            reasons.append(f"光斑形状已变 ({shape})")
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


def _reuse_failure_note(session_dir: Path) -> str:
    """光路复用**不通过**时的原因行, 供阶段表「说明」列追加。

    只认真实判为「不通过」的会话 (``same_path`` 为假且未跳过): 复用 ``
    _reuse_latest_failure_reason`` 的逐条原因, 拼成一行。通过 / 被跳过 / 产物
    缺失或损坏时返回空串 (调用方不追加)。绝不抛。

    Args:
        session_dir: 会话目录 (读 ``optical_probe.json``)。
    """
    if not session_dir or not Path(session_dir).is_dir():
        return ""
    probe = Path(session_dir).joinpath("optical_probe.json")
    if not probe.is_file():
        return ""
    try:
        data = json.loads(probe.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("读取光路复用结论失败: {}: {}", session_dir, exc)
        return ""
    if data.get("skipped") or bool(data.get("same_path")):
        return ""
    reason = _reuse_latest_failure_reason(data)
    if reason:
        return f"光路复用未通过：{reason}"
    verdict = data.get("verdict")
    if isinstance(verdict, str) and verdict.strip():
        return f"光路复用未通过：{verdict.strip()}"
    return "光路复用未通过：光路已变化"
