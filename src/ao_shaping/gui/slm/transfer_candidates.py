"""迁移来源候选层（检查点 / 历史接受相位目录）——纯数据，不 import streamlit。

本模块只做「把磁盘上的历史产物变成下拉候选行」这一件事：扫索引、取兼容分、
排文案。没有控件、没有 session_state 访问，所以 CLI 脚本与测试都能在没有
streamlit 的环境里直接 import；streamlit 那一侧（``st.session_state`` 读当前
光路指纹）留在 :mod:`ao_shaping.gui.slm.shaping_controls` 的适配层里。

判据同源：

- 检查点候选走 :func:`ao_shaping.tools.slm.transfer.checkpoint_registry.scan_checkpoint_index`
  与 :func:`ao_shaping.tools.slm.transfer.optical_session.recommend_load`，
  兼容分不在这里另算。
- ``--seed-run`` 候选要求目录里**同时**有 ``manifest.json`` 与 ``config.json``，
  且 manifest 里有 ``<形状>_final`` 接受相位行 —— 后者正是
  :func:`ao_shaping.tools.slm.transfer.validate.replay_seed` 查的那一行
  （:data:`ACCEPTED_PHASE_SUFFIX` 与 ``<形状>_final`` 判据同源，改一处必须改另一处）。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = [
    "TRANSFER_FP_SESSION_KEY",
    "TRANSFER_NO_SOURCE_LABEL",
    "TRANSFER_SHAPE_LABELS",
    "fingerprint_from_state",
    "transfer_checkpoint_candidates",
    "transfer_seed_run_candidates",
]


#: 本页选项集（迁移学习历来跑的四种）→ 中文名，仅用于显示；校验走
#: :data:`ao_shaping.utils.image.target.TRANSFER_SHAPE_CHOICES`，名字的唯一来源是
#: :data:`ao_shaping.utils.image.target.shapes.TARGET_SHAPE_LABELS`。
#: 原先 ``star`` 的文案写着「尚无实验结果」——那条已过期：2026-10-28 的 multishape
#: campaign 里 star 四段全部跑完
#: （``report/shape_trials/multishape_scale/multishape_scale_live_20261028.md`` §3.2）。
TRANSFER_SHAPE_LABELS: dict[str, str] = {
    "square": "正方形",
    "rectangle": "长方形",
    "triangle": "三角形",
    "star": "五角星",
}

#: 迁移来源下拉的第一项：本轮不接历史产物（等价 CLI 不给 ``--checkpoint`` /
#: ``--seed-run``）。两个下拉共用同一条文案，免得操作员要区分两套「空」的措辞。
TRANSFER_NO_SOURCE_LABEL = "（不使用历史产物：本轮冷启动）"

#: 当前光路指纹在 ``st.session_state`` 里的键。放了才显示兼容分 —— 分数归
#: ``optical_session`` 算，本模块只排版，不提供第二个口径。
TRANSFER_FP_SESSION_KEY = "transfer_current_fingerprint"

#: ``--seed-run`` 候选的扫描深度上限（相对候选根）：实测产物在
#: ``<会话>/<run>/``（``live_no_gs_20261024_0926/anchored``），也有会话根本身就是
#: run（``live_no_gs_cv_20261024/config.json``），三层以外的是会话内部的
#: ``shape/<形状>/<阶段>/`` —— 那些是单形状阶段产物，不能当整轮种子。
_SEED_RUN_MAX_DEPTH = 3

#: 接受相位行在 manifest 里的后缀（``validate.replay_seed`` 查的就是 ``<形状>_final``）。
ACCEPTED_PHASE_SUFFIX = "_final"


def _candidate_root(root: str | Path | None) -> Path | None:
    """候选根目录：与 :mod:`ao_shaping.gui.slm.history_browser` 同一约定（只此一份）。

    ``ImportError``（transfer 包缺失，比如只装了 GUI 依赖的机器）⇒ ``None``，
    调用方据此回落到手动输入。
    """
    if root is not None:
        return Path(root)
    try:
        from ao_shaping.gui.slm.history_browser import DEFAULT_ROOT, FALLBACK_ROOT
    except ImportError:
        return None
    return DEFAULT_ROOT if DEFAULT_ROOT.is_dir() else FALLBACK_ROOT


def _short_time(value: Any) -> str:
    """``2026-10-27T18:10:38+00:00`` / ``datetime`` → ``2026-10-27 18:10``；缺失写 ``未记录``。"""
    if hasattr(value, "strftime"):
        return str(value.strftime("%Y-%m-%d %H:%M"))
    text = str(value or "")
    return text[:16].replace("T", " ") if len(text) >= 16 else "未记录"


def _unique_labels(labels: Sequence[str]) -> list[str]:
    """重名候选补 ``#n``（下拉里两项同名时操作员分不出选的是哪一个）。"""
    seen: dict[str, int] = {}
    unique: list[str] = []
    for label in labels:
        count = seen.get(label, 0)
        seen[label] = count + 1
        unique.append(label if count == 0 else f"{label} #{count + 1}")
    return unique


def _shape_label(shape: str) -> str:
    """形状的中文名：本地表优先，缺项查 canonical 词表（不复制第二份）。

    实测（2026-10-29 真实索引）：76 条候选里有 ``circle``，而本地表没有它 ——
    下拉第一行会写英文 ``circle``。canonical 词表
    (:data:`ao_shaping.utils.image.target.shapes.TARGET_SHAPE_LABELS`) 本来就有
    它，所以缺项时借它，而不是在本文件再抄一份形状词表。
    """
    if not shape:
        return "形状未记录"
    label = TRANSFER_SHAPE_LABELS.get(shape)
    if label:
        return label
    from ao_shaping.utils.image.target.shapes import TARGET_SHAPE_LABELS

    return TARGET_SHAPE_LABELS.get(shape, shape)


def _verdict_of(score: float) -> str:
    """兼容分 → 结论词（``推荐`` / ``谨慎`` / ``不推荐``），来自 ``optical_session``。

    transfer 包在模块作用域**不能** import（索引不可用时要让调用方回落到手动输入），
    所以这条 import 放在函数里 —— 与 :func:`_candidate_root` 同款。本模块只转手，
    不比较分数：阈值只有 ``optical_session`` 一份。
    """
    from ao_shaping.tools.slm.transfer.optical_session import compatibility_verdict

    return compatibility_verdict(score)


def _checkpoint_label(entry: Mapping[str, Any], *, score: float | None,
                      verified: bool) -> str:
    """一行候选：``形状 · 注册时间 · 兼容分（结论，若有）· 是否已校验``。

    括号里的结论（``推荐`` / ``谨慎`` / ``不推荐``）由
    :func:`ao_shaping.tools.slm.transfer.optical_session.compatibility_verdict` 给出：
    本模块不打分也不比阈值。无分数时只写 ``兼容 —``、不给结论 ——「没打分」与
    「不推荐」是两个完全不同的读法，混成一个会让操作员跳过核对指纹。
    """
    if score is None:
        tag = "兼容 —"
    else:
        verdict = _verdict_of(score)
        tag = f"兼容 {score:.2f}（{verdict}）" if verdict else f"兼容 {score:.2f}"
    return " · ".join((
        _shape_label(str(entry.get("shape") or "")),
        _short_time(entry.get("created_at")),
        tag,
        "已校验" if verified else "未校验",
    ))


def transfer_checkpoint_candidates(
    root: str | Path | None = None,
    current_fp: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]] | None:
    """检查点候选行（``checkpoint_registry`` 的索引 + 一行展示文案）；不可用 ⇒ ``None``。

    ``None`` 表示「候选索引不可用」（transfer 包 import 失败或索引读不动），
    调用方必须回落到手动文本框 —— 没有索引的机器上不能把功能弄丢。

    分数、拒绝理由全部来自 ``optical_session.recommend_load``：本函数不读指纹、
    不算分，只有排版。``current_fp`` 为 ``None`` 时该项没有分数（``score=None``），
    因为「跟哪条光路比」根本没给出 —— 填一个 0.00 会让操作员以为它不兼容。
    """
    base = _candidate_root(root)
    try:
        from ao_shaping.tools.slm.transfer.checkpoint_registry import scan_checkpoint_index
        from ao_shaping.tools.slm.transfer.optical_session import recommend_load
    except ImportError:
        return None
    if base is None:
        return None
    try:
        entries = scan_checkpoint_index(base)
        ranked = recommend_load(current_fp, entries)["ranked"]
    except (OSError, ValueError):
        return None
    rows: list[dict[str, Any]] = []
    for item in ranked:
        entry = item["candidate"]
        # ``dimensions`` 为空 ⇔ ``recommend_load`` 没能评分（无当前指纹/缺候选指纹）。
        score = float(item["score"]) if item.get("dimensions") else None
        verified = bool(entry.get("sha256_ok")) and not bool(entry.get("stale"))
        rows.append({
            "label": _checkpoint_label(entry, score=score, verified=verified),
            "file": str(entry.get("file") or ""),
            "path": str(entry.get("path") or ""),
            "session": str(entry.get("session") or ""),
            "shape": str(entry.get("shape") or ""),
            "created_at": entry.get("created_at"),
            "score": score,
            "verified": verified,
            "rejected": bool(item.get("rejected")),
            "reasons": [str(r) for r in (item.get("reasons") or ()) if str(r).strip()],
        })
    return rows


def _has_accepted_phase(run_dir: Path) -> bool:
    """目录的 manifest 里是否真有接受相位行。

    判据取自消费者自己：``validate.replay_seed`` 找的是 ``<形状>_final`` 那一行，
    所以「有 manifest.json」不等于「能当种子」—— ``capture/`` 也有 manifest.json，
    但里面只有原始帧记录，选中它会等 ``replay_seed`` 抛 StopIteration。
    """
    try:
        rows = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return any(
        str(row.get("label", "")).endswith(ACCEPTED_PHASE_SUFFIX)
        for row in rows if isinstance(row, Mapping)
    )


def transfer_seed_run_candidates(root: str | Path | None = None
                                 ) -> list[dict[str, Any]] | None:
    """历史接受相位目录候选（``--seed-run`` 的合法输入）；读不动 ⇒ ``None``。

    判据只有两条，且与消费者自己的校验同源：目录里**同时**有 ``manifest.json``
    与 ``config.json``（:class:`~ao_shaping.tools.slm.transfer.workflow.TransferSession`
    运行前就是这么拒的），且 manifest 里真有 ``<形状>_final`` 接受相位行
    （``validate.replay_seed`` 找的就是它）。缺任一条的目录手填进去要等一轮跑起来
    才抛异常，正是要消掉的那类「十分钟后才发现」错误。这里不读指纹、不打分。
    """
    base = _candidate_root(root)
    if base is None:
        return None
    try:
        manifests = sorted(base.rglob("manifest.json"))
    except OSError:
        return None
    rows: list[dict[str, Any]] = []
    for manifest in manifests:
        run_dir = manifest.parent
        try:
            if len(run_dir.relative_to(base).parts) > _SEED_RUN_MAX_DEPTH:
                continue
            config = run_dir / "config.json"
            if not config.is_file():
                continue
            mtime = config.stat().st_mtime
        except OSError:
            continue
        if not _has_accepted_phase(run_dir):
            continue
        rows.append({
            "label": f"{run_dir.relative_to(base)} · {_short_time(datetime.fromtimestamp(mtime))}",
            "path": str(run_dir),
            "mtime": mtime,
        })
    rows.sort(key=lambda row: row["mtime"], reverse=True)
    for row in rows:
        row.pop("mtime", None)
    return rows


def fingerprint_from_state(stored: Any) -> dict[str, Any] | None:
    """``st.session_state`` 里取出的当前光路指纹 → ``dict``；没放/类型不对 ⇒ ``None``。

    纯映射转换：读 session 的那一步（以及它的裸模式兜底）在 GUI 侧。
    """
    return dict(stored) if isinstance(stored, Mapping) else None
