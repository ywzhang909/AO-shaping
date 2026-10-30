"""历史会话浏览器与光路指纹推荐卡：只排版，不拟合、不碰设备、不写产物。

检查点能否续用取决于**光路指纹**是否可比（窗口、零级位置、曝光、相机格式任一变了
就不是同一条光路）。字段名只有一个来源：指纹键取 ``optical_session._FINGERPRINT_KEYS``，
条目键取 ``checkpoint_registry.register_session`` 的落盘记录；字段缺席写
:data:`MISSING`，**不按第二个键名试探**。分数不在这里算：全部来自 ``recommend_load`` /
``compatibility_score``；阈值只有 ``COMPAT_THRESHOLD`` / ``AUTO_APPLY_THRESHOLD`` 两个，
三档结论由 ``compatibility_verdict`` 判，本模块只把结论翻成一句话（不重算、不改权、
不出现第三个阈值）。当前指纹
由调用方放进 ``st.session_state[current_fp_key(key_prefix)]``，没放时开关禁用。缺省根
目录与 ``multishape_campaign_ui.campaign_root`` 同一条；那份入口是整页模块，本层不
import 它（避免页面侧依赖），只对齐路径字面量；未创建时回落 :data:`FALLBACK_ROOT`。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import streamlit as st

from ao_shaping.tools.slm.transfer.checkpoint_registry import scan_checkpoint_index
from ao_shaping.tools.slm.transfer.optical_session import (
    AUTO_APPLY_THRESHOLD,
    COMPAT_THRESHOLD,
    compatibility_score,
    compatibility_verdict,
    load_fingerprint,
    recommend_load,
)
from ao_shaping.utils.image.target.shapes import TARGET_SHAPE_LABELS

__all__ = [
    "COMPAT_THRESHOLD",
    "DEFAULT_ROOT",
    "FALLBACK_ROOT",
    "MISSING",
    "VERDICT_SENTENCES",
    "current_fp_key",
    "render_fingerprint_card",
    "render_history_browser",
    "render_load_panel",
]

#: 「没测到」的唯一写法（与 ``transfer_progress.MISSING`` 同款，全页只此一个占位符）。
MISSING = "—"
#: campaign 输出根目录（与 ``multishape_campaign_ui.campaign_root()`` 同一约定）。
DEFAULT_ROOT = Path("data") / "debug" / "slm_transfer"
#: 旧记录根目录：:data:`DEFAULT_ROOT` 尚未创建时回落到它。
FALLBACK_ROOT = Path("data") / "shape_trials"
#: 兼容分三档 → 推荐行末尾那句结论。三档本身由
#: :func:`ao_shaping.tools.slm.transfer.optical_session.compatibility_verdict` 切
#: （阈值只在那里一份），本表只把 verdict 词翻成人话；键既不在别的分支里再判一次分，
#: 也不在这里写阈值。「没打分」不进这张表 —— 无分数时整句结论都不出现。
VERDICT_SENTENCES: dict[str, str] = {
    "推荐": "推荐（可直接应用）",
    "谨慎": "谨慎（可用，但请核对指纹差异后再应用）",
    "不推荐": "不推荐（低于可用阈值）",
}


def _as_float(value: Any) -> float | None:
    """宽松转 float：布尔、非数值、非有限值 → ``None``。"""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _plain(value: Any) -> str | None:
    """数字 → ``g`` 格式；非数字/非有限值 → ``None``（由调用方决定显示什么）。"""
    number = _as_float(value)
    return None if number is None else f"{number:g}"


def _pair(value: Any) -> list[str] | None:
    """两元数值（list / tuple / ndarray）→ 两个字符串；否则 ``None``。"""
    if isinstance(value, (str, bytes, Mapping)):
        return None
    try:
        numbers = [_plain(item) for item in value]
    except TypeError:
        return None
    if len(numbers) != 2 or any(number is None for number in numbers):
        return None
    return [number for number in numbers if number is not None]  # type: ignore[misc]


def _text(value: Any) -> str:
    """字符串原样（去空白）、数字按 ``g``；缺失或空串写 :data:`MISSING`。"""
    if isinstance(value, str):
        return value.strip() or MISSING
    return _plain(value) or MISSING


def _timestamp(value: Any) -> str:
    """ISO 时间串 → ``YYYY-MM-DD HH:MM``（秒以下对判读没有信息量）。"""
    text = str(value).strip().replace("T", " ") if value is not None else ""
    return text[:16] if text else MISSING


def _score(value: Any, digits: int = 2) -> str:
    """分数 → 定长小数；非数值写 :data:`MISSING`（0.00 是真实测量值，不可混用）。"""
    number = _as_float(value)
    return MISSING if number is None else f"{number:.{digits}f}"


def _size_text(value: Any) -> str:
    """窗口尺寸：两元按 ``x×y px``，标量按 ``v px``。"""
    pair = _pair(value)
    if pair is not None:
        return f"{pair[0]}×{pair[1]} px"
    number = _plain(value)
    return MISSING if number is None else f"{number} px"


def _fingerprint_rows(fingerprint: Mapping[str, Any]) -> list[dict[str, str]]:
    """指纹卡六行（顺序固定，便于对照两份指纹）；键名见模块 docstring。"""
    zero, grid = _pair(fingerprint.get("zero_xy")), _pair(fingerprint.get("control_grid"))
    exposure = _plain(fingerprint.get("exposure_ms"))
    span, step = fingerprint.get("window_span"), fingerprint.get("window_step")
    # 跨度+步长合写一行，避免两行「—」看不出谁缺
    window = MISSING if span is None and step is None else f"{_size_text(span)} / 步长 {_size_text(step)}"
    grid_text = f"{grid[0]} × {grid[1]}" if grid is not None else _size_text(fingerprint.get("control_grid"))
    return [
        {"项目": "指纹时间", "值": _timestamp(fingerprint.get("created_at"))},
        {"项目": "窗口/周期", "值": window},
        {"项目": "零级位置", "值": MISSING if zero is None else f"({zero[0]}, {zero[1]}) px"},
        {"项目": "曝光", "值": MISSING if exposure is None else f"{exposure} ms"},
        {"项目": "相机格式", "值": _text(fingerprint.get("camera_format"))},
        {"项目": "控制分辨率", "值": grid_text},
    ]


def _shape_label(shape: str) -> str:
    """形状键 → 中文名（缺失时原样，空值写 :data:`MISSING`）。"""
    return TARGET_SHAPE_LABELS.get(shape, shape) if shape else MISSING


def _candidate_label(candidate: Mapping[str, Any]) -> str:
    """候选的一行标签：``时间 · 形状键 · CV · 质量``（CV/质量缺失写 ``—``）。"""
    return " · ".join((
        _timestamp(candidate.get("created_at")),
        str(candidate.get("shape") or MISSING),
        f"CV {_score(candidate.get('cv'))}",
        f"质量 {_score(candidate.get('fit_quality'))}",
    ))


def render_fingerprint_card(fingerprint: Mapping[str, Any], *, key_prefix: str = "hb") -> None:
    """把一份光路指纹排成紧凑卡片（六行表格 + 一行用途说明）。

    fingerprint 是 ``optical_session.load_fingerprint`` 的返回值；缺字段只显示
    :data:`MISSING`——一份半截指纹仍然值得看一眼。
    """
    st.dataframe(_fingerprint_rows(fingerprint or {}), hide_index=True,
                 use_container_width=True, key=f"{key_prefix}_fingerprint_card")
    st.caption("本指纹用于与历史检查点比对：窗口、零级位置、曝光、相机格式一致才可比。")


def current_fp_key(key_prefix: str) -> str:
    """页面把「当前光路指纹」放进 session_state 时用的键。"""
    return f"{key_prefix}_current_fp"


def _index_of(candidate: Any, entries: Sequence[Mapping[str, Any]]) -> int | None:
    """``recommend_load`` 回传的候选（输入行的副本）→ 输入里的行号，按 ``path`` 对回。"""
    if not isinstance(candidate, Mapping):
        return None
    path = candidate.get("path")
    return next((i for i, e in enumerate(entries) if e.get("path") == path), None)


def _statuses(entries: Sequence[Mapping[str, Any]], ranked: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """逐候选状态 ``item``/``score``/``rejected``/``reasons``；未被排名者 ``item`` 为 ``None``。"""
    statuses: list[dict[str, Any]] = [{"item": None, "score": None, "rejected": False, "reasons": []} for _ in entries]
    for item in ranked:
        index = _index_of(item.get("candidate"), entries)
        if index is not None:
            statuses[index] = {
                "item": item, "score": _as_float(item.get("score")),
                "rejected": bool(item.get("rejected")),
                "reasons": [str(r) for r in (item.get("reasons") or ()) if str(r).strip()],
            }
    return statuses


def _verdict_clause(score: float | None) -> str:
    """兼容分 → `` —— 推荐（可直接应用）`` 这样的后缀；无分数 ⇒ 空串。

    结论一律问 :func:`compatibility_verdict`：本模块不比较分数、不出现阈值字面量，
    句子的措辞只在 :data:`VERDICT_SENTENCES` 里。verdict 词不在表里（``optical_session``
    改了档名）时也只丢了那句结论，不会在这里补一个判断。
    """
    if score is None:
        return ""
    sentence = VERDICT_SENTENCES.get(compatibility_verdict(score))
    return f" —— {sentence}" if sentence else ""


def _render_recommendation(item: Mapping[str, Any]) -> None:
    """推荐卡本体：一行结论（分数 + 三档结论）+ 维度表 + 推荐依据。"""
    candidate = item.get("candidate")
    label = _candidate_label(candidate) if isinstance(candidate, Mapping) else MISSING
    score = _as_float(item.get("score"))
    st.markdown(
        f"**推荐检查点**：{label}　兼容分数 **{_score(score)}**{_verdict_clause(score)}")
    dimensions = [
        {"维度": str(d.get("name") or MISSING), "权重": _score(d.get("weight")),
         "得分": _score(d.get("value")), "说明": str(d.get("reason") or MISSING)}
        for d in item.get("dimensions") or ()
    ]
    if dimensions:
        st.dataframe(dimensions, hide_index=True, use_container_width=True)
    reasons = [str(r) for r in (item.get("reasons") or ()) if str(r).strip()]
    if reasons:
        st.caption("推荐依据与限制：" + "；".join(reasons))


def _options(entries: Sequence[Mapping[str, Any]], statuses: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[dict[str, str]]]:
    """候选选项文案（被拒注明原因、重名补 ``#n``）与被拒一览（被拒不等于隐藏，只是不可应用）。"""
    seen: dict[str, int] = {}
    labels: list[str] = []
    rejected: list[dict[str, str]] = []
    for entry, status in zip(entries, statuses):
        base = _candidate_label(entry)
        reasons = "；".join(status.get("reasons") or ())
        if status.get("rejected"):
            rejected.append({"候选": base, "原因": reasons or "未给出原因（见候选指纹）"})
            base = f"{base}（不兼容：{reasons or '见推荐卡'}）"
        seen[base] = seen.get(base, 0) + 1
        labels.append(base if seen[base] == 1 else f"{base} · #{seen[base]}")
    return labels, rejected


def render_load_panel(
    *,
    current_fp: Mapping[str, Any] | None,
    candidates: Sequence[Mapping[str, Any]],
    key_prefix: str = "hb",
    on_apply: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any] | None:
    """历史检查点的推荐与装载面板。

    current_fp 可 ``None``（按未提供指纹处理，仍可手选）；candidates 为
    ``scan_checkpoint_index`` 的行；on_apply 在点「应用该检查点」时先回调。
    返回本次点击应用的候选 dict，没点击 / 候选为空 / 被拒 → ``None``。
    """
    entries = [dict(candidate) for candidate in candidates]
    if not entries:
        st.caption("未找到可用的历史检查点")
        return None

    verdict = recommend_load(current_fp, entries)
    statuses = _statuses(entries, list(verdict.get("ranked") or ()))
    recommended = _index_of((verdict.get("recommended") or {}).get("candidate"), entries)
    if recommended is not None and statuses[recommended]["item"] is not None:
        _render_recommendation(statuses[recommended]["item"])
    else:
        st.caption(f"没有候选达到推荐阈值（≥{AUTO_APPLY_THRESHOLD:.2f}）：仍可手选，每个未达阈值的选项都标了原因。")
    labels, rejected = _options(entries, statuses)
    if rejected:
        st.caption("不兼容候选（不可应用）：")
        st.dataframe(rejected, hide_index=True, use_container_width=True)

    choice = st.selectbox("选择历史检查点", options=labels,
                          index=recommended if recommended is not None else 0,
                          key=f"{key_prefix}_load_choice")
    index = labels.index(choice)
    status = statuses[index]
    if status.get("rejected"):
        st.caption(f"该检查点不可应用：{'；'.join(status.get('reasons') or ()) or '见推荐卡'}")
    elif status.get("item") is None:
        st.caption("该候选不在推荐排名内（推荐层未给出评分）：应用前请自行核对指纹。")

    if st.button("应用该检查点", key=f"{key_prefix}_load_apply", disabled=status.get("rejected")):
        payload = dict(entries[index])
        if on_apply is not None:
            on_apply(payload)
        st.session_state[f"{key_prefix}_load_applied"] = str(payload.get("path") or "")
        return payload
    return None


def _resolve_root(root: str | Path | None) -> Path:
    """缺省根目录：新布局优先，未创建时回落到旧记录目录。"""
    if root is not None:
        return Path(root)
    return DEFAULT_ROOT if DEFAULT_ROOT.is_dir() else FALLBACK_ROOT


def _scan(root: Path) -> list[dict[str, Any]] | None:
    """读检查点索引；读不动（权限/格式）返回 ``None`` 并落一行说明。"""
    try:
        entries = scan_checkpoint_index(root)
    except (OSError, ValueError) as exc:
        st.caption(f"读取检查点索引失败：{exc}")
        return None
    return [dict(entry) for entry in entries]


def _entry_fingerprint(entry: Mapping[str, Any]) -> dict[str, Any] | None:
    """条目的指纹：内联 ``fingerprint`` 优先，其次按会话目录读 ``fingerprint.json``。"""
    inline = entry.get("fingerprint")
    if isinstance(inline, Mapping) and inline:
        return dict(inline)
    path = entry.get("path")
    return load_fingerprint(path) if path else None


def _compat_of(current: Mapping[str, Any] | None, fingerprint: Mapping[str, Any] | None) -> float | None:
    """当前指纹对条目指纹的兼容分数；任一缺失 → ``None``。"""
    if current is None or fingerprint is None:
        return None
    score, _ = compatibility_score(current, fingerprint)
    return _as_float(score)


def _history_header(entry: Mapping[str, Any], score: float | None) -> str:
    """一行标题：时间 · 形状 · CV · 质量 ·（兼容）·（已过期）。"""
    bits = [
        _timestamp(entry.get("created_at")), _shape_label(str(entry.get("shape") or "")),
        f"CV {_score(entry.get('cv'))}", f"质量 {_score(entry.get('fit_quality'))}",
    ]
    if score is not None:
        bits.append(f"兼容 {_score(score)}")
    if entry.get("stale"):
        bits.append("已过期")
    return " · ".join(bits)


def _provenance_text(entry: Mapping[str, Any]) -> str:
    """provenance 摘要：这条读数从哪个目录、哪一类产物、哪个哈希来的。"""
    digest = entry.get("sha256")
    status = "已过期（stale）：可查看，但不建议作为默认来源" if entry.get("stale") else "当前"
    return "\n".join((
        "**来源**",
        f"- 会话目录：`{entry.get('path') or MISSING}`",
        f"- 产物类型：{entry.get('kind') or MISSING}",
        f"- 指纹哈希：`{str(digest)[:12] if digest else MISSING}`",
        f"- 生成时间：{_timestamp(entry.get('created_at'))}",
        f"- 状态：{status}",
    ))


def render_history_browser(*, root: str | Path | None = None, key_prefix: str = "hb") -> Path | None:
    """历史检查点浏览器：过滤、逐行展开指纹与 provenance、返回选中的会话目录。

    root 为索引根目录（``None`` ⇒ :data:`DEFAULT_ROOT`，存在时，否则
    :data:`FALLBACK_ROOT`）。返回用户点「选用该会话」的目录，其余情况 ``None``。
    """
    base = _resolve_root(root)
    entries = _scan(base)
    if entries is None:
        return None
    if not entries:
        st.caption(f"{base} 下还没有历史检查点。")
        return None

    stored = st.session_state.get(current_fp_key(key_prefix))
    current = dict(stored) if isinstance(stored, Mapping) else None
    shapes = sorted({str(entry.get("shape") or "") for entry in entries} - {""})
    labels = {shape: _shape_label(shape) for shape in shapes}

    left, right = st.columns(2)
    chosen_shapes = left.multiselect("按形状过滤", options=shapes, default=shapes,
                                     format_func=lambda shape: labels[shape],
                                     key=f"{key_prefix}_hist_shapes")
    only_compat = right.checkbox(f"只看兼容（≥{COMPAT_THRESHOLD:.2f}）", value=False,
                                 key=f"{key_prefix}_hist_compat", disabled=current is None)
    if current is None:
        right.caption(f"未提供当前光路指纹（st.session_state[{current_fp_key(key_prefix)!r}]）：兼容过滤不可用。")

    shown: list[tuple[dict[str, Any], float | None, dict[str, Any] | None]] = []
    for entry in entries:
        if chosen_shapes and str(entry.get("shape") or "") not in chosen_shapes:
            continue
        fingerprint = _entry_fingerprint(entry)
        score = _compat_of(current, fingerprint)
        if only_compat and current is not None and (score is None or score < COMPAT_THRESHOLD):
            continue
        shown.append((entry, score, fingerprint))
    if not shown:
        st.caption("当前过滤条件下没有检查点。")
        return None

    st.caption(f"{len(shown)} / {len(entries)} 个检查点（{base}）")
    picked: Path | None = None
    for index, (entry, score, fingerprint) in enumerate(shown):
        with st.expander(_history_header(entry, score), expanded=False):
            if entry.get("stale"):
                st.badge("已过期", color="orange")
            if fingerprint is None:
                st.caption("该检查点没有光路指纹（早于指纹功能），只能按路径核对。")
            else:
                render_fingerprint_card(fingerprint, key_prefix=f"{key_prefix}_h{index}")
            st.markdown(_provenance_text(entry))
            if st.button("选用该会话", key=f"{key_prefix}_use_{index}"):
                picked = Path(str(entry.get("path")))
                st.session_state[f"{key_prefix}_hist_picked"] = str(picked)
    return picked
