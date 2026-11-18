"""会话目录 → 给操作员看的一行结论: 验收阈值 / 光路复用 / 模型权重。

为什么单独一层
--------------
这四个函数只读**已经落盘的会话产物** (``session.json`` / ``rounds.json`` /
``optical_probe.json`` / ``frozen_parameters.json``), 把它们压成一行中文。它们
不碰 Qt 控件、不碰设备、也不需要页面状态 —— 所以离屏测试只要一个临时目录就能钉住,
不必起 ``QApplication``。

拆出 page.py 的直接理由: page.py 已顶到模块规模棘轮 (``scripts/check_module_size.py``
只允许变小), 而这 350 行与「摆控件」毫无关系; 页面只负责把返回的字符串放进某个
``QLabel``。

缺省语义
--------
产物缺失 / 半截 / 不该显示的会话一律返回 ``""`` (调用方据此隐藏该行), 绝不抛 ——
一行字读不出来, 不该把完成界面打崩。

阈值从哪来
----------
判决本身不在这里: 验收阈值来自
:func:`~ao_shaping.tools.slm.transfer.workflow.acceptance_verdict`, 光路复用来自
``optical_probe.json`` 的写侧。本模块只负责措辞与「该不该显示」的取舍。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from loguru import logger

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
