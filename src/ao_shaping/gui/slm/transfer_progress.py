"""迁移学习闭环的运行进度明细：纯采集一层 + Streamlit 渲染一层。

为什么单独一个模块
------------------
``run_session`` 只发 ``on_stage(name, status)`` 三态信号，进度条能回答「走到第几
阶段」，但说不出「这一阶段测出了什么」。明细只能从**会话目录**里读回来：
``journal.json``（阶段与状态）、``capture/manifest.json``（探针）、
``fit/result.json``（拟合指标）、``anchored/summary.json``（逐块验收）等。所以本
模块分两层：

1. :func:`collect_progress` 是纯函数 —— 读会话目录，返回结构化视图。它**不 import
   streamlit、不 import 设备**，所以没有服务器也能单测；并且一条产物缺失、只写了
   一半、或整个会话目录还不存在时都**不得抛异常**：刚开始的阶段和失败的阶段本来
   就没有产物。
2. :func:`render_progress` 只负责把视图排成几行文字（当前阶段 → 各阶段一行 + 该
   阶段的关键指标）。视觉语言沿用本页邻居：``st.markdown`` 的加粗标签 + 圆点分隔，
   状态用中文词（已完成 / 进行中 / 失败 / 跳过），不用 emoji。

指标口径只有一份
----------------
本模块**不重算**任何指标，只读管线自己写下的数字。唯一的借用是
:func:`~ao_shaping.tools.slm.transfer.adaptation.validation_quality`（验证集上的
相关系数 / 相对 L2 / CV 误差）—— ``fit/adaptation.json`` 的升级判定用的就是它，
面板与判定必须同源，否则操作员会看到两个互相打架的「相关系数」。

「未产出」与「0」是两件事
------------------------
一条指标缺失时 ``value is None``、``text`` 是 :data:`MISSING`（「—」），**永远不是
``0.0``：这条管线的 0 是真实测量值（漂移 0、CV 误差 0），与「没测到」混在一起会
让操作员读错验收结论。

阶段状态里多出的 ``skipped``
----------------------------
``TransferSession.plan()`` 里可能有 ``reuse_capture``（复用 ``--dataset`` 的采集），
而 ``run_session`` 遇到 ``dataset`` 时根本不发这个阶段。计划顺序就是执行顺序，所以
「排在第一个已开始阶段之前、却仍是待运行」的阶段只可能是被跳过 —— 据此标
``skipped``，而不是让它永远显示「待运行」。
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "MAX_DETAIL_LINES",
    "MISSING",
    "SHAPE_LABELS",
    "STAGE_LABELS",
    "STATUS_LABELS",
    "collect_progress",
    "render_progress",
]

#: 一条指标没测到时的显示文本。与 0 严格区分（见模块文档）。
MISSING = "—"

#: 每个阶段最多铺几行细节（逐块验收 / 逐形状汇总）；其余指向会话目录里的产物。
MAX_DETAIL_LINES = 8

#: ``run_session`` 发出的阶段名 → 操作员读得懂的标签。这份映射是**唯一**一份：
#: 采集代码里没有展示知识，``shaping_controls`` 应当从本模块再导出
#: （``from .transfer_progress import STAGE_LABELS as TRANSFER_STAGE_LABELS``），
#: 否则两处各写一遍就会漂移。
STAGE_LABELS: dict[str, str] = {
    "capture": "当前光路采集（探针 + 逐帧自动曝光）",
    "reuse_capture": "复用已有采集",
    "nominal_initialization": "名义模型初始化（冷启动，未标定）",
    "fit": "物理预测模型迁移 / 拟合",
    "CCD_model_blocks": "CCD 模型配对块",
    "exposure_response_capture": "曝光阶梯诊断",
    "exposure_response_analysis": "曝光响应分析",
    "blur_calibration": "有效展宽 / 辐射尺度校准",
    "CCD_anchored_blocks": "CCD 锚定优化（验收：A–B–B–A 或仅模型损失）",
    "GS_and_differentiable_synthesis": "GS + 可微相位合成（离线）",
    "transferred_phase_differentiable_synthesis": "历史接受相位迁移 + 可微合成（离线）",
    "raw_data_audit": "原始数据审计",
    "summary": "指标摘要",
    "direction_check": "方向核对",
}

#: 目标形状 → 中文名。同样是唯一一份（``shaping_controls`` 应从本模块再导出）。
SHAPE_LABELS: dict[str, str] = {
    "square": "正方形",
    "rectangle": "长方形",
    "triangle": "三角形",
    "star": "五角星（已实现，尚无实验结果）",
}

#: 阶段状态 → 中文词。与 ``bench_settings`` 的 ``● / ○`` 同一档文字标识，不用 emoji。
STATUS_LABELS: dict[str, str] = {
    "pending": "待运行",
    "running": "进行中",
    "complete": "已完成",
    "failed": "失败",
    "skipped": "跳过",
}

#: 逐形状划分的固定顺序（只显示实际出现的那些）。
_SPLIT_LABELS: tuple[tuple[str, str], ...] = (
    ("calibration", "校准"),
    ("train", "训练"),
    ("validation", "验证"),
    ("test", "测试"),
)


# ---------------------------------------------------------------------------
# 视图采集（纯函数，无 streamlit、无设备）
# ---------------------------------------------------------------------------


def collect_progress(
    out_dir: str | Path | None,
    *,
    journal: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Read whatever a transfer session has written so far; never raise.

    Args:
        out_dir: 会话目录（``TransferSession.out``）。``None`` 或不存在都返回空视图 ——
            GUI 在 ``build_params`` 之前调用时拿到的就是 ``None``。
        journal: 已经解析好的 ``journal.json`` 内容。省略时从
            ``<out_dir>/journal.json`` 读；给定后不再读文件（调用方持有内存副本时用）。

    Returns:
        ``{"out_dir", "exists", "stages", "current", "next", "counts",
        "finished", "failed"}``；每个 stage 是
        ``{"name", "label", "status", "index", "started", "ended", "duration_s",
        "error", "metrics", "details"}``，每条 metric 是
        ``{"key", "label", "value", "text"}``（``value is None`` 即未产出）。
    """
    out = _as_path(out_dir)
    entries = _journal_entries(out, journal)
    planned = _planned_stages(out)
    latest = _latest_by_stage(entries)
    order = planned + [name for name in latest if name not in planned]
    first_started = min(
        (order.index(name) for name in latest if name in order), default=None
    )

    stages: list[dict[str, Any]] = []
    for position, name in enumerate(order, start=1):
        entry = latest.get(name)
        status = _status_of(entry)
        if status == "pending" and first_started is not None and position - 1 < first_started:
            # 计划顺序即执行顺序：排在已开始阶段之前却仍待运行的，只可能是被跳过。
            status = "skipped"
        stages.append(_stage_view(out, name, position, status, entry))

    current = _current_stage(stages, entries)
    counts = {key: 0 for key in STATUS_LABELS}
    for stage in stages:
        counts[stage["status"]] = counts.get(stage["status"], 0) + 1
    counts["total"] = len(stages)
    return {
        "out_dir": str(out) if out is not None else None,
        "exists": bool(out is not None and out.is_dir()),
        "stages": stages,
        "current": current,
        "next": next((s for s in stages if s["status"] == "pending"), None),
        "counts": counts,
        "finished": _finished(entries, order),
        "failed": any(s["status"] == "failed" for s in stages),
    }


def _as_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    try:
        return Path(value)
    except TypeError:
        return None


def _read_json(path: Path | None) -> Any:
    """Parse a JSON artefact; ``None`` for missing / half-written / unreadable."""
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _journal_entries(
    out: Path | None, journal: Sequence[Mapping[str, Any]] | None
) -> list[Mapping[str, Any]]:
    raw = journal if journal is not None else _read_json(
        out / "journal.json" if out is not None else None
    )
    if not isinstance(raw, list):
        return []
    return [entry for entry in raw if isinstance(entry, Mapping) and entry.get("stage")]


def _planned_stages(out: Path | None) -> list[str]:
    """``session.json["stages"]`` —— ``TransferSession.plan()`` 写下的计划顺序。"""
    session = _read_json(out / "session.json" if out is not None else None)
    stages = session.get("stages") if isinstance(session, Mapping) else None
    if not isinstance(stages, list):
        return []
    return [str(name) for name in stages if isinstance(name, str) and name]


def _latest_by_stage(entries: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """最后一次写入的条目胜出（``journal.json`` 每阶段一条，重跑会追加）。"""
    latest: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        latest[str(entry["stage"])] = entry
    return latest


def _status_of(entry: Mapping[str, Any] | None) -> str:
    status = str(entry.get("status") or "") if entry is not None else ""
    return status if status in {"running", "complete", "failed"} else "pending"


def _current_stage(
    stages: Sequence[dict[str, Any]], entries: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    running = [stage for stage in stages if stage["status"] == "running"]
    if running:
        return running[-1]
    if entries:
        last = str(entries[-1].get("stage"))
        return next((stage for stage in stages if stage["name"] == last), None)
    return None


def _finished(entries: Sequence[Mapping[str, Any]], order: Sequence[str]) -> bool:
    """最后一个计划阶段跑完才算结束 —— 失败与中途停止都不算。"""
    if not entries or _status_of(entries[-1]) != "complete":
        return False
    if not order:
        return True
    return str(entries[-1].get("stage")) == order[-1]


def _stamp(entry: Mapping[str, Any] | None, key: str) -> str | None:
    """Journal timestamps: the start is flat, the end is ``clock()``'s dict."""
    if entry is None:
        return None
    value = entry.get(key)
    if isinstance(value, Mapping):
        value = value.get("timestamp_utc")
    return str(value) if value else None


def _duration(entry: Mapping[str, Any] | None) -> float | None:
    if entry is None:
        return None
    end = entry.get("end")
    if not isinstance(end, Mapping):
        return None
    try:
        seconds = (float(end["monotonic_ns"]) - float(entry["monotonic_ns"])) / 1e9
    except (KeyError, TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None


def _stage_view(
    out: Path | None,
    name: str,
    position: int,
    status: str,
    entry: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metrics, details = _stage_metrics(out, name)
    error = entry.get("error") if entry is not None else None
    if error:
        details.insert(0, f"错误：{error}")
    duration = _duration(entry)
    if duration is not None:
        metrics.append(_num("duration", "耗时", duration, digits=1, suffix=" s"))
    return {
        "name": name,
        "label": STAGE_LABELS.get(name, name),
        "status": status,
        "index": position,
        "started": _stamp(entry, "timestamp_utc"),
        "ended": _stamp(entry, "end"),
        "duration_s": duration,
        "error": str(error) if error else None,
        "metrics": metrics,
        "details": details,
    }


def _stage_metrics(
    out: Path | None, name: str
) -> tuple[list[dict[str, Any]], list[str]]:
    """``(metrics, details)`` for one stage; any failure means "nothing yet"."""
    reader = _READERS.get(name)
    if reader is None or out is None:
        return [], []
    try:
        return reader(out)
    except Exception:  # 产物缺失 / 只写了一半 / 字段改名：一律当作「未产出」
        return [], []


# ---------------------------------------------------------------------------
# 指标行构造器（value 缺失 → text 就是 MISSING，绝不留 0）
# ---------------------------------------------------------------------------


def _metric(key: str, label: str, value: Any, text: Any) -> dict[str, Any]:
    if value is None:
        return {"key": key, "label": label, "value": None, "text": MISSING}
    return {"key": key, "label": label, "value": value, "text": str(text)}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _num(
    key: str, label: str, value: Any, *, digits: int = 4, suffix: str = ""
) -> dict[str, Any]:
    number = _finite(value)
    if number is None:
        return _metric(key, label, None, MISSING)
    return _metric(key, label, number, f"{number:.{digits}f}{suffix}")


def _int(key: str, label: str, value: Any, *, suffix: str = "") -> dict[str, Any]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return _metric(key, label, None, MISSING)
    return _metric(key, label, number, f"{number}{suffix}")


def _text(key: str, label: str, value: Any, text: Any = None) -> dict[str, Any]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return _metric(key, label, None, MISSING)
    return _metric(key, label, value, value if text is None else text)


def _span(
    key: str, label: str, values: Sequence[Any], *, digits: int = 2, suffix: str = ""
) -> dict[str, Any]:
    numbers = [n for n in (_finite(v) for v in values) if n is not None]
    if not numbers:
        return _metric(key, label, None, MISSING)
    low, high = min(numbers), max(numbers)
    body = f"{low:.{digits}f}" if low == high else f"{low:.{digits}f}–{high:.{digits}f}"
    return _metric(key, label, [low, high], f"{body}{suffix}")


def _mean(values: Sequence[Any]) -> float | None:
    numbers = [n for n in (_finite(v) for v in values) if n is not None]
    return sum(numbers) / len(numbers) if numbers else None


def _figure(value: Any, digits: int = 4) -> str:
    number = _finite(value)
    return MISSING if number is None else f"{number:.{digits}f}"


def _shape_name(value: Any) -> str:
    name = str(value or "")
    return SHAPE_LABELS.get(name, name) if name else MISSING


def _scope_text(scopes: Sequence[str], metric: str = "") -> str:
    """验收口径：配对 A–B–B–A / 仅模型损失 / 混合（各块口径不同）。"""
    unique = sorted({scope for scope in scopes if scope})
    if unique == ["model_only"]:
        body = "仅模型损失"
    elif unique in ([], ["paired"]):
        body = "配对 A–B–B–A"
    else:
        body = "混合（逐块见细节）"
    return f"{body}（{metric}）" if metric else body


def _rows(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, Mapping)]


# ---------------------------------------------------------------------------
# 逐阶段读数
# ---------------------------------------------------------------------------


def _capture_metrics(out: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """``capture/manifest.json`` —— 一条一行，边采边写，所以运行中就在长。"""
    rows = _rows(_read_json(out / "capture" / "manifest.json"))
    if not rows:
        return [], []
    splits: dict[str, int] = {}
    for row in rows:
        split = str(row.get("split") or "")
        if split:
            splits[split] = splits.get(split, 0) + 1
    ordered = [key for key, _label in _SPLIT_LABELS]
    parts = [
        f"{label} {splits[key]}"
        for key, label in _SPLIT_LABELS
        if key in splits
    ]
    parts += [f"{key} {count}" for key, count in splits.items() if key not in ordered]
    metrics = [
        _int("probes", "探针", len(rows), suffix=" 条"),
        _text("splits", "划分", " / ".join(parts)),
        _int("usable", "可用", _usable_count(rows), suffix=" 条"),
        _span("exposure", "曝光", [row.get("exposure_ms") for row in rows],
              digits=2, suffix=" ms"),
        _num("peak", "峰值", max(
            (n for n in (_finite(row.get("peak")) for row in rows) if n is not None),
            default=None,
        ), digits=0, suffix="/255"),
    ]
    return metrics, []


def _usable_count(rows: Sequence[Mapping[str, Any]]) -> int | None:
    """``fit.valid`` 的判据就是拟合真正吃进去的探针口径 —— 借它，不重写。"""
    try:
        from ao_shaping.tools.slm.transfer.fit import valid
    except Exception:
        return None
    try:
        return sum(1 for row in rows if valid(row))
    except Exception:
        return None


def _nominal_metrics(out: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """``initial/frozen_parameters.json`` + ``initial/provenance.json``。"""
    params = _read_json(out / "initial" / "frozen_parameters.json")
    if not isinstance(params, Mapping):
        return [], []
    samples = (_read_json(out / "initial" / "provenance.json") or {})
    ramp = samples.get("ramp_samples") if isinstance(samples, Mapping) else None
    axes = len(ramp) if isinstance(ramp, Mapping) else None
    metrics = [
        _int("grid", "网格", params.get("grid_size")),
        _num("defocus", "离焦", params.get("defocus_diopter"), digits=2, suffix=" D"),
        _int("ramp_axes", "斜坡", axes, suffix=" 轴"),
    ]
    return metrics, []


def _fit_metrics(out: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """``fit/result.json``（选中的那次迭代）+ ``fit/adaptation.json``（自动适配）。"""
    root = out / "fit"
    result = _read_json(root / "result.json")
    if not isinstance(result, Mapping):
        return [], []
    quality, scope = _fit_quality(root, result)
    metrics = [
        _int("iteration", "迭代", result.get("iteration")),
        _num("loss", "损失", result.get("loss")),
        _num("correlation", f"相关系数（{scope}）", quality.get("correlation"), digits=3),
        _num("cv_abs_error", "CV 误差", quality.get("cv_abs_error"), digits=4),
        _num("relative_l2", "相对 L2", quality.get("relative_l2"), digits=3),
    ]
    details = []
    adaptation = _read_json(root / "adaptation.json")
    if isinstance(adaptation, Mapping):
        stage = adaptation.get("selected_stage")
        escalated = bool(adaptation.get("escalated"))
        note = f"自动适配：选中 {stage or MISSING}（{'已升级到 full' if escalated else '未升级'}）"
        heuristic = adaptation.get("escalation_heuristic") or {}
        below = _finite(heuristic.get("correlation_below"))
        above = _finite(heuristic.get("relative_l2_above"))
        if below is not None or above is not None:
            note += (
                f"；升级判定阈 相关系数 < {_figure(below, 2)}"
                f" 或 相对 L2 > {_figure(above, 2)}"
            )
        details.append(note)
    return metrics, details


def _fit_quality(
    root: Path, result: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    """验证集口径优先（与 ``adaptation`` 的升级判定同源），否则退回全部探针。"""
    entry_metrics = _rows(result.get("metrics"))
    provenance = _read_json(root / "provenance.json")
    rows = _rows(provenance.get("rows") if isinstance(provenance, Mapping) else None)
    if rows and len(rows) == len(entry_metrics):
        try:
            from ao_shaping.tools.slm.transfer.adaptation import validation_quality

            return dict(validation_quality(result, rows)), "验证"
        except Exception:  # 没有验证集探针 / 长度不齐 / 非有限值
            pass
    return (
        {
            "correlation": _mean([m.get("correlation") for m in entry_metrics]),
            "relative_l2": _mean([m.get("relative_l2") for m in entry_metrics]),
            "cv_abs_error": _mean([m.get("cv_abs_error") for m in entry_metrics]),
        },
        "全部",
    )


def _block_metrics(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """``<run>/summary.json`` —— 逐块 ``paired_decision`` 的结果。

    ``CCD_model_blocks`` 与 ``CCD_anchored_blocks`` 走同一个 reader，只是根目录
    分别是 ``model/`` 与 ``anchored/``（两者都是 ``validate.run`` 的产物）。
    """
    blocks = _rows(_read_json(root / "summary.json"))
    if not blocks:
        return [], []
    paired = [b for b in blocks if "baseline" in b and "candidate" in b]
    scopes = [str(b.get("acceptance") or "paired") for b in blocks]
    metrics = [
        _int("accepted", "接受", sum(1 for b in blocks if b.get("accepted")),
             suffix=f"/{len(blocks)} 块"),
        _text("baseline_candidate", "对比", _transition(paired)),
        _num("drift", "漂移 ≤", max(
            (n for n in (_finite(b.get("drift")) for b in paired) if n is not None),
            default=None,
        ), digits=4),
        _text("scope", "验收口径",
              _scope_text(scopes, _metric_name(blocks) if paired else "")),
    ]
    details = [_block_line(block) for block in blocks[:MAX_DETAIL_LINES]]
    if len(blocks) > MAX_DETAIL_LINES:
        details.append(
            f"…其余 {len(blocks) - MAX_DETAIL_LINES} 块见 {root.name}/summary.json"
        )
    return metrics, details


def _model_block_metrics(out: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """``CCD_model_blocks`` 的配对块写在 ``model/`` 下（锚定块写在 ``anchored/``）。"""
    return _block_metrics(out / "model")


def _anchored_block_metrics(out: Path) -> tuple[list[dict[str, Any]], list[str]]:
    return _block_metrics(out / "anchored")


def _transition(paired: Sequence[Mapping[str, Any]]) -> str | None:
    baseline = _mean([b.get("baseline") for b in paired])
    candidate = _mean([b.get("candidate") for b in paired])
    if baseline is None or candidate is None:
        return None
    return f"基线 {baseline:.4f} → 候选 {candidate:.4f}"


def _metric_name(blocks: Sequence[Mapping[str, Any]]) -> str:
    """``accept_metric`` 的值 → 读得懂的名字（``loss`` / ``cv`` / ``model``）。"""
    names = sorted({str(b.get("accept_metric") or "") for b in blocks} - {""})
    labels = {"loss": "损失", "cv": "CV", "model": "模型损失"}
    if not names:
        return ""
    return " / ".join(labels.get(name, name.upper()) for name in names)


def _block_line(block: Mapping[str, Any]) -> str:
    index = block.get("block")
    position = f"第 {int(index) + 1} 块" if isinstance(index, int) else "整轮"
    verdict = "接受" if block.get("accepted") else "拒绝"
    if "baseline" in block:
        body = (f"基线 {_figure(block.get('baseline'))} → "
                f"候选 {_figure(block.get('candidate'))} · "
                f"漂移 {_figure(block.get('drift'))}")
    else:
        body = (f"模型损失 {_figure(block.get('model_loss_previous'))} → "
                f"{_figure(block.get('model_loss_last'))}")
    return f"{position}（{_shape_name(block.get('shape'))}）：{verdict} · {body}"


def _response_capture_metrics(out: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """``response/manifest.json`` —— 交错曝光阶梯的逐帧记录。"""
    rows = _rows(_read_json(out / "response" / "manifest.json"))
    if not rows:
        return [], []
    phases = sorted({str(row.get("phase")) for row in rows if row.get("phase")})
    levels = {_finite(row.get("exposure_ms")) for row in rows} - {None}
    metrics = [
        _text("phases", "相位", " / ".join(phases)),
        _int("frames", "帧", len(rows)),
        _int("levels", "曝光档", len(levels)),
        _num("peak", "峰值", max(
            (n for n in (_finite(row.get("peak")) for row in rows) if n is not None),
            default=None,
        ), digits=0, suffix="/255"),
    ]
    return metrics, []


def _response_analysis_metrics(out: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """``response_analysis/result.json`` —— 每个相位的线性度拟合与验证残差。"""
    data = _read_json(out / "response_analysis" / "result.json")
    if not isinstance(data, Mapping) or not data:
        return [], []
    results = [value for value in data.values() if isinstance(value, Mapping)]
    metrics = [
        _text("phases", "相位", " / ".join(str(key) for key in data)),
        _int("levels", "曝光档", max(
            (len(value.get("levels_ms") or []) for value in results), default=None
        )),
        _num("validation_rmse", "验证 RMSE ≤", max(
            (n for n in (_finite(v.get("validation_rmse_counts")) for v in results)
             if n is not None), default=None,
        ), digits=2, suffix=" counts"),
        _num("reference_span", "参考帧跨度 ≤", max(
            (n * 100 for n in (_finite(v.get("reference_roi_mean_span_fraction"))
                               for v in results) if n is not None), default=None,
        ), digits=1, suffix="%"),
    ]
    return metrics, []


def _blur_metrics(out: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """``blur/result.json`` —— 选中的有效展宽与辐射尺度。"""
    data = _read_json(out / "blur" / "result.json")
    selected = data.get("selected") if isinstance(data, Mapping) else None
    if not isinstance(selected, Mapping):
        return [], []
    metrics = [
        _num("sigma", "展宽 σ", selected.get("sigma"), digits=2, suffix=" px"),
        _num("gain", "增益", selected.get("gain"), digits=3),
        _num("background", "背景", selected.get("background"), digits=1, suffix=" counts"),
        _num("train_score", "训练评分", selected.get("train_score"), digits=4),
        _num("validation_score", "验证评分", selected.get("validation_score"), digits=4),
    ]
    return metrics, []


def _audit_metrics(out: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """``audit.json`` —— 从原始帧重算指标并校验相位/掩模/时钟契约。"""
    data = _read_json(out / "audit.json")
    if not isinstance(data, Mapping):
        return [], []
    status = str(data.get("status") or "")
    captures = data.get("captures")
    frames = data.get("raw_frames")
    collected = (
        f"{int(captures)} 条 / {int(frames)} 帧"
        if isinstance(captures, int) and isinstance(frames, int)
        else None
    )
    metrics = [
        _text("status", "审计", "通过" if status == "passed" else status or None),
        _text("captures", "采集", collected),
        _num("maximum_peak", "峰值", data.get("maximum_peak"), digits=0, suffix="/255"),
        _int("accepted", "接受", data.get("accepted_blocks"),
             suffix=f"/{data.get('blocks')} 块"),
    ]
    return metrics, []


def _summary_metrics(out: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """``summary/results.json`` —— 逐形状 seed/final 的实测 CV 与模型相关系数。"""
    root = out / "summary"
    data = _read_json(root / "results.json")
    if not isinstance(data, Mapping):
        return [], []
    acceptance = data.get("acceptance") or {}
    scopes = acceptance.get("scopes") if isinstance(acceptance, Mapping) else None
    metrics = [
        _text("acceptance", "验收口径", _scope_text(scopes or [])),
        _int("accepted", "接受", acceptance.get("accepted_blocks")
             if isinstance(acceptance, Mapping) else None, suffix=" 块"),
    ]
    details = []
    for item in _rows(data.get("shapes")):
        seed = (item.get("seed") or {})
        final = (item.get("final") or {})
        seed_cv = (seed.get("measured") or {}).get("cv") if isinstance(seed, Mapping) else None
        final_cv = (final.get("measured") or {}).get("cv") if isinstance(final, Mapping) else None
        correlation = final.get("correlation") if isinstance(final, Mapping) else None
        details.append(
            f"{_shape_name(item.get('shape'))}：CV {_figure(seed_cv, 3)} → "
            f"{_figure(final_cv, 3)} · 模型相关系数 {_figure(correlation, 3)}"
        )
    figure = root / "ccd_model_comparison.png"
    if figure.is_file():
        details.append(f"图 {_relative(figure, out)}")
    return metrics, details


def _relative(path: Path, root: Path) -> str:
    """会话目录内的产物用相对路径 —— 面板只有一栏宽，绝对路径读不完。"""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _direction_metrics(out: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """``direction/result.json`` —— 有限差分与自动求导的方向导数核对。"""
    root = out / "direction"
    data = _read_json(root / "result.json")
    if not isinstance(data, Mapping):
        return [], []
    results = _rows(data.get("results"))
    if not results:
        return [], []
    checks = []
    for item in results:
        entries = _rows(item.get("derivative_checks"))
        # 步长最大的那一档最稳（h = 0.1），其余是 0.03 / 0.01。
        best = max(entries, key=lambda c: _finite(c.get("h")) or 0.0, default=None)
        if best is not None:
            checks.append(best.get("relative_error"))
    metrics = [
        _num("derivative_error", "方向导数相对误差 ≤", max(
            (n for n in (_finite(v) for v in checks) if n is not None), default=None
        ), digits=3),
        _num("measured_loss", "实测 A1 损失 ≤", max(
            (n for n in (_finite(r.get("measured_A1_loss")) for r in results)
             if n is not None), default=None,
        )),
    ]
    details = [
        f"{_shape_name(item.get('shape'))}：实测 A1 {_figure(item.get('measured_A1_loss'))}"
        f" · 模型 {_figure(item.get('anchored_model_loss_plus'))}"
        for item in results[:MAX_DETAIL_LINES]
    ]
    return metrics, details


def _synthesis_metrics(out: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """离线合成：``<shape>_offline_iterations.json`` 里的损失轨迹。"""
    suffix = "_offline_iterations.json"
    files = sorted(out.glob(f"*{suffix}"))
    if not files:
        return [], []
    histories = [(path, _rows(_read_json(path))) for path in files]
    metrics = [
        _int("shapes", "形状", len(histories), suffix=" 个"),
        _int("updates", "更新", max((len(h) for _p, h in histories), default=None),
             suffix=" 次"),
        _num("loss", "末次损失 ≤", max(
            (n for n in (_finite(h[-1].get("loss")) for _p, h in histories if h)
             if n is not None), default=None,
        )),
    ]
    details = []
    for path, history in histories[:MAX_DETAIL_LINES]:
        if not history:
            continue
        details.append(
            f"{_shape_name(path.name[: -len(suffix)])}：损失 "
            f"{_figure(history[0].get('loss'))} → {_figure(history[-1].get('loss'))}"
            f" · EE {_figure(history[-1].get('ee_roi'), 3)}"
        )
    return metrics, details


_READERS: dict[str, Any] = {
    "capture": _capture_metrics,
    "nominal_initialization": _nominal_metrics,
    "fit": _fit_metrics,
    "CCD_model_blocks": _model_block_metrics,
    "exposure_response_capture": _response_capture_metrics,
    "exposure_response_analysis": _response_analysis_metrics,
    "blur_calibration": _blur_metrics,
    "CCD_anchored_blocks": _anchored_block_metrics,
    "raw_data_audit": _audit_metrics,
    "summary": _summary_metrics,
    "direction_check": _direction_metrics,
    "GS_and_differentiable_synthesis": _synthesis_metrics,
    "transferred_phase_differentiable_synthesis": _synthesis_metrics,
}


# ---------------------------------------------------------------------------
# Streamlit 渲染（只有排版，不读产物）
# ---------------------------------------------------------------------------


def render_progress(view: Mapping[str, Any], *, running: bool) -> None:
    """Render the progress detail panel into the active Streamlit container.

    排在进度条下面：一行「现在在做什么」，然后每个已开始阶段一行（状态 + 关键
    指标），逐块验收/逐形状汇总另起缩进子行。``pending`` 阶段不铺 —— 它们没有
    任何指标，只会把面板撑长；当前阶段与「下一步」由头部一行交代。

    Args:
        view: :func:`collect_progress` 的返回值（也接受手工构造的同形 dict）。
        running: 会话是否仍在进行。只有在 journal 无法判断时才有用 —— 已失败 / 已
            跑完最后阶段 / 停在中间三种情况由 ``view`` 自己决定措辞。
    """
    import streamlit as st

    stages = list(view.get("stages") or ())
    if not stages:
        st.caption("迁移学习闭环：还没有阶段记录（尚未开始）。")
        return
    head, counts = _header(view, running)
    st.markdown(head)
    if counts:
        st.caption(counts)
    blocks = [_stage_block(stage) for stage in stages if stage.get("status") != "pending"]
    if blocks:
        st.markdown("\n".join(blocks))
    if view.get("out_dir"):
        st.caption(f"会话目录：{view['out_dir']}")


def _header(view: Mapping[str, Any], running: bool) -> tuple[str, str]:
    counts = view.get("counts") or {}
    total = int(counts.get("total") or 0)
    current = view.get("current") or None
    following = view.get("next") or None
    if view.get("failed") and current:
        head = f"**已结束** · 第 {current['index']} 个阶段失败：{current['label']}"
    elif view.get("finished"):
        head = f"**已完成** · 全部 {total} 个阶段完成"
    elif current and current.get("status") == "running":
        head = f"**运行中** · 当前阶段：{current['label']}（{current['index']}/{total}）"
    elif running and following:
        head = f"**运行中** · 下一步：{following['label']}（{following['index']}/{total}）"
    elif running:
        head = "**运行中** · 正在建立会话目录"
    elif current:
        head = f"**已停止** · 停在 {current['label']}（{current['index']}/{total}）"
    else:
        head = "**未开始** · 还没有阶段记录"
    return head, _counts_line(counts)


def _counts_line(counts: Mapping[str, Any]) -> str:
    if not counts.get("total"):
        return ""
    parts = [f"已完成 {int(counts.get('complete') or 0)}"]
    for key in ("running", "failed", "skipped"):
        value = int(counts.get(key) or 0)
        if value:
            parts.append(f"{STATUS_LABELS[key]} {value}")
    parts.append(f"待运行 {int(counts.get('pending') or 0)}")
    return " · ".join(parts)


def _stage_block(stage: Mapping[str, Any]) -> str:
    status = STATUS_LABELS.get(str(stage.get("status")), str(stage.get("status")))
    texts = [
        f"{metric['label']} {metric['text']}" for metric in stage.get("metrics") or ()
    ]
    body = " · ".join(texts) if texts else "尚未产出指标"
    lines = [f"- **{stage.get('label')}**：{status} —— {body}"]
    lines.extend(f"  - {detail}" for detail in stage.get("details") or ())
    return "\n".join(lines)
