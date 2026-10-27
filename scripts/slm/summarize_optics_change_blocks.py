"""D 块汇总器: 把一轮闭环采集离线汇总成可直接引用的表格、曲线与结构化 JSON。

背景
----
``defocus_differentiable`` campaign 的 D 块 (``stage_d`` / ``stage_d1000``) 每轮
按 A1 → B1 → B2 → A2 顺序各采若干帧, 逐帧记录 ``cv_core`` (冻结窗内对比度)。
本脚本**不碰设备**, 只读落盘 JSON:

* 每轮实测 JSON: ``<run_dir>/stage_d_block1.json`` (文件名固定为 ``stage_d_block1.json``),
  其中 ``records`` 是逐帧的 ``cv_core`` 列表。
* 轮次清单: 优先 ``<stage>/rounds.json`` (新式, 含 ``measured`` 汇总与可选
  ``model_before`` / ``model_after`` 模型侧 CV); 缺失时从
  ``<stage>/run_round{N}`` 或 ``<stage>/run_block{N}`` 目录重建 (旧式 ``stage_d``
  没有 ``rounds.json``, 只有 ``chain.json`` 逐轮判定)。
* 噪声地板: ``--noise-floor`` 指定的 JSON, 其 ``three_sigma_resolution`` 是
  判据 (|Δ| > 3σ 记一轮)。缺省回退 ``<run-root>/stage_a/repeatability/noise_floor.json``;
  对 ``*1000*`` 目标 (1000 px 窗) 的 stage, 缺省优先
  ``logs/target_size_sweep/noise_floor_1000px_after_focus/noise_floor.json``
  (1000 px 窗的 3σ 与 60 px 窗不同, 不能混用)。

每轮量 (全部由 ``cv_core`` 逐帧均值组合, 与采集器口径一致):

* 基线 CV   = mean(A1) 与 mean(A2) 的均值
* 候选 CV   = mean(B1) 与 mean(B2) 的均值
* Δ        = 候选 − 基线
* 漂移      = mean(A2) − mean(A1)
* 重放差    = mean(B2) − mean(B1)

数据仍在落盘时 (缺文件 / 缺标签) 逐轮降级: 能算的算, 算不出的写 ``—``,
绝不用 0 顶替; 每轮与每份来源都逐字记进 ``sources``。

输出 (落 ``--out``, 默认 ``<run-root>/<stage>/summary/``, 不写 ``report/``):

* ``blocks.md``   —— 汇总表 + 一句结论
* ``blocks.png``  —— 上栏模型侧 (仅当 ``rounds.json`` 有 model_before/after) +
                     下栏实测基线/候选折线与 ±3σ 带
* ``blocks.json`` —— 结构化汇总 (noise_floor / rounds / above_3sigma / sources)

用法::

    python scripts/slm/summarize_optics_change_blocks.py --stage stage_d
    python scripts/slm/summarize_optics_change_blocks.py --stage stage_d1000 \
        --noise-floor logs/target_size_sweep/noise_floor_1000px_after_focus/noise_floor.json
"""

from __future__ import annotations

import argparse
from loguru import logger
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
for _candidate in (ROOT, ROOT / "src", ROOT / "libs", ROOT / "libs" / "OOPAO"):
    _text = str(_candidate)
    if _text not in sys.path:
        sys.path.insert(0, _text)

#: 仓库根进了 sys.path 才能导入 scripts._common: 直接以 python scripts/slm/... 运行时只有脚本目录在路径里。
from scripts._common import fmt_general

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

_LABELS = ("A1", "B1", "B2", "A2")
_PER_RUN_JSON_NAME = "stage_d_block1.json"
_RUN_DIR_RE = re.compile(r"^run_(?:round|block)(\d+)$")
_NA = "—"


def _resolve_run_root(raw: str) -> Path:
    """解析 ``--run-root``: 绝对路径直接用, 相对路径先 CWD 后仓库根。"""
    p = Path(raw)
    if p.is_absolute():
        return p
    for base in (Path.cwd(), ROOT):
        cand = base / p
        if cand.is_dir():
            return cand
    return Path.cwd() / p


def _resolve_noise_floor_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    for base in (Path.cwd(), ROOT):
        cand = base / p
        if cand.is_file():
            return cand
    return Path.cwd() / p


def _load_json(path: Path) -> Any | None:
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _label_means(records: list[dict]) -> dict[str, float]:
    """按标签归并 ``cv_core`` 逐帧值, 返回 ``{标签: 均值}``。"""
    by_label: dict[str, list[float]] = {}
    for rec in records:
        label = rec.get("label")
        cv = rec.get("cv_core")
        if label in _LABELS and cv is not None:
            by_label.setdefault(label, []).append(float(cv))
    return {label: _mean(vals) for label, vals in by_label.items()}


def _frames_per_label(records: list[dict]) -> dict[str, int]:
    counts = {label: 0 for label in _LABELS}
    for rec in records:
        label = rec.get("label")
        if label in counts and rec.get("cv_core") is not None:
            counts[label] += 1
    return counts


def _compute_from_records(data: dict) -> dict[str, Any]:
    """从单份 ``stage_d_block1.json`` 的 records 计算一轮的全部量。

    缺标签时降级: 依赖缺失标签的量置 None (渲染时写 ``—``)。
    """
    records = [r for r in data.get("records", []) if isinstance(r, dict)]
    means = _label_means(records)
    m_a1, m_a2 = means.get("A1"), means.get("A2")
    m_b1, m_b2 = means.get("B1"), means.get("B2")

    baseline = (m_a1 + m_a2) / 2.0 if (m_a1 is not None and m_a2 is not None) else None
    candidate = (m_b1 + m_b2) / 2.0 if (m_b1 is not None and m_b2 is not None) else None
    delta = candidate - baseline if (baseline is not None and candidate is not None) else None
    drift = m_a2 - m_a1 if (m_a1 is not None and m_a2 is not None) else None
    replay = m_b2 - m_b1 if (m_b1 is not None and m_b2 is not None) else None
    missing = [label for label in _LABELS if label not in means]
    return {
        "baseline_cv": baseline,
        "candidate_cv": candidate,
        "delta": delta,
        "drift": drift,
        "replay_b2_minus_b1": replay,
        "accepted": data.get("accepted"),
        "frames_per_label": _frames_per_label(records),
        "missing_labels": missing,
    }


def _rel(run_root: Path, path: Path) -> str:
    try:
        return path.relative_to(run_root).as_posix()
    except ValueError:
        return path.as_posix()


def discover_reconstructed_rounds(stage_dir: Path) -> list[int]:
    """无 ``rounds.json`` 时, 从 ``run_round{N}`` / ``run_block{N}`` 目录重建轮次号。"""
    rounds: set[int] = set()
    if not stage_dir.is_dir():
        return []
    for child in stage_dir.iterdir():
        if not child.is_dir():
            continue
        match = _RUN_DIR_RE.match(child.name)
        if match:
            rounds.add(int(match.group(1)))
    return sorted(rounds)


def _per_run_mismatches(computed: dict, entry: dict) -> list[str]:
    """比较逐帧 JSON 算出的量与 ``rounds.json`` 的 ``measured`` 汇总, 返回不一致项。

    新一轮 campaign 复用旧目录时, ``measured.run_dir`` 指向的逐帧 JSON 可能是
    上一轮 (不同窗宽/不同光路) 的残留文件 —— 两份数据对不上即判为陈旧。
    """
    measured = entry.get("measured") if isinstance(entry.get("measured"), dict) else None
    if not isinstance(measured, dict):
        return []
    out: list[str] = []
    for mkey, ckey in (("baseline", "baseline_cv"), ("candidate", "candidate_cv"), ("drift", "drift")):
        ref = measured.get(mkey)
        mine = computed.get(ckey)
        if ref is None or mine is None:
            continue
        if abs(mine - ref) > max(1e-3, 1e-3 * abs(ref)):
            out.append(f"{mkey}: 逐帧 {mine:.6f} vs measured {ref:.6f}")
    return out


def _candidate_per_run_paths(entry: dict, run_root: Path, stage_dir: Path, round_no: int) -> list[Path]:
    """每轮实测 JSON 的候选路径, 按优先级排序。

    优先 stage 自己的 ``run_round{N}`` / ``run_block{N}``, 再是 ``rounds.json``
    记录的 ``measured.run_dir`` (相对 run root, 可能指向别的 stage 目录)。
    实际取哪份由 ``_per_run_mismatches`` 与 measured 汇总的交叉核对裁决。
    """
    candidates = [
        stage_dir / f"run_round{round_no}" / _PER_RUN_JSON_NAME,
        stage_dir / f"run_block{round_no}" / _PER_RUN_JSON_NAME,
    ]
    measured = entry.get("measured") if isinstance(entry.get("measured"), dict) else None
    if isinstance(measured, dict):
        run_dir = measured.get("run_dir")
        if run_dir:
            p = run_root / str(run_dir) / _PER_RUN_JSON_NAME
            if p not in candidates:
                candidates.append(p)
    return candidates


def build_rounds(
    run_root: Path,
    stage_dir: Path,
    chain: dict | None,
) -> tuple[list[dict], dict, list[str]]:
    """汇总所有轮次, 返回 (rounds, sources, warnings)。

    优先级: ``rounds.json`` 清单 → 每轮 ``stage_d_block1.json`` (逐帧) →
    ``rounds.json`` 的 ``measured`` 汇总 (无逐帧时) → 跳过 (全 ``—``)。
    """
    warnings: list[str] = []
    rounds_json_path = stage_dir / "rounds.json"
    rounds_list_raw = _load_json(rounds_json_path)
    rounds_json = None
    if rounds_json_path.is_file():
        if isinstance(rounds_list_raw, list):
            rounds_json = rounds_list_raw
        else:
            warnings.append(f"rounds.json 不是列表, 已按重建路径处理: {_rel(run_root, rounds_json_path)}")

    chain_by_round: dict[int, dict] = {}
    if isinstance(chain, dict):
        for blk in chain.get("blocks", []):
            if isinstance(blk, dict) and isinstance(blk.get("block"), int):
                chain_by_round[blk["block"]] = blk

    sources: dict[str, Any] = {
        "manifest": "rounds.json" if rounds_json is not None else "reconstructed",
        "rounds": {},
        "per_run_json": {},
        "per_run_json_missing": [],
        "stale_per_run_json": [],
        "chain_json": "chain.json" if isinstance(chain, dict) else None,
    }

    if rounds_json is not None:
        raw_rounds = [e for e in rounds_json if isinstance(e, dict)]
        raw_rounds.sort(key=lambda e: (e.get("round") is None, e.get("round", 0)))
        entries: list[tuple[int, dict]] = []
        for idx, entry in enumerate(raw_rounds, start=1):
            round_no = entry.get("round")
            if not isinstance(round_no, int):
                round_no = idx
                warnings.append(f"rounds.json 第 {idx} 条无有效 round 号, 按序号 {idx} 记")
            entries.append((round_no, entry))
    else:
        reconstructed = discover_reconstructed_rounds(stage_dir)
        if not reconstructed:
            warnings.append("未发现任何 run_round{N} / run_block{N} 目录, 本轮无数据")
        entries = [(n, {}) for n in reconstructed]

    rounds: list[dict] = []
    for round_no, entry in entries:
        # 逐帧 JSON: 收集全部候选 (stage 自身目录优先, 再 rounds.json 记的 run_dir),
        # 逐个与 measured 汇总交叉核对, 取第一份一致的; 不一致的判为陈旧文件丢弃
        computed = None
        computed_from: Path | None = None
        for cand in _candidate_per_run_paths(entry, run_root, stage_dir, round_no):
            if not cand.is_file():
                continue
            data = _load_json(cand)
            if not isinstance(data, dict):
                warnings.append(f"第 {round_no} 轮 JSON 解析失败, 跳过: {_rel(run_root, cand)}")
                continue
            trial = _compute_from_records(data)
            mismatches = _per_run_mismatches(trial, entry)
            if not mismatches:
                computed = trial
                computed_from = cand
                break
            warnings.append(
                f"第 {round_no} 轮逐帧 JSON {_rel(run_root, cand)} 与 rounds.json measured 不一致 ({'; '.join(mismatches)}), 判为陈旧文件丢弃"
            )
            sources["stale_per_run_json"].append({"round": round_no, "path": _rel(run_root, cand), "mismatches": mismatches})
        if computed is not None:
            assert computed_from is not None
            rel = _rel(run_root, computed_from)
            source = f"per_run_json:{rel}"
            measured = entry.get("measured") if isinstance(entry.get("measured"), dict) else None
            if isinstance(measured, dict) and measured.get("accepted") is not None:
                # 接受与否以 campaign 清单 (rounds.json) 的判定为准
                computed["accepted"] = measured["accepted"]
            if computed["accepted"] is None:
                blk = chain_by_round.get(round_no)
                if isinstance(blk, dict):
                    computed["accepted"] = blk.get("accepted")
            sources["rounds"][str(round_no)] = f"rounds.json 清单 + 逐帧 {rel}" if rounds_json is not None else f"重建自 {rel} (无 rounds.json)"
            sources["per_run_json"][str(round_no)] = rel
            computed["source"] = source
            rounds.append({"round": round_no, **computed})
            continue

        # 逐帧 JSON 缺失: 退回 rounds.json 的 measured 汇总
        measured = entry.get("measured") if isinstance(entry.get("measured"), dict) else None
        if measured is not None and any(measured.get(k) is not None for k in ("baseline", "candidate", "delta", "drift")):
            baseline = measured.get("baseline")
            candidate = measured.get("candidate")
            delta = measured.get("delta")
            drift = measured.get("drift")
            sources["rounds"][str(round_no)] = "rounds.json:measured 汇总 (逐帧 JSON 缺失, 重放差不可算)"
            sources["per_run_json_missing"].append(round_no)
            rounds.append(
                {
                    "round": round_no,
                    "baseline_cv": baseline,
                    "candidate_cv": candidate,
                    "delta": delta,
                    "drift": drift,
                    "replay_b2_minus_b1": None,
                    "accepted": measured.get("accepted"),
                    "frames_per_label": {label: 0 for label in _LABELS},
                    "missing_labels": list(_LABELS),
                    "source": "rounds.json:measured",
                }
            )
            continue

        sources["rounds"][str(round_no)] = "缺失 (无 rounds.json 条目, 无逐帧 JSON)"
        sources["per_run_json_missing"].append(round_no)
        rounds.append(
            {
                "round": round_no,
                "baseline_cv": None,
                "candidate_cv": None,
                "delta": None,
                "drift": None,
                "replay_b2_minus_b1": None,
                "accepted": None,
                "frames_per_label": {label: 0 for label in _LABELS},
                "missing_labels": list(_LABELS),
                "source": "missing",
            }
        )
    return rounds, sources, warnings


def build_model(rounds_json: list | None) -> dict:
    """模型侧 (上栏) 数据, 只认 ``rounds.json`` 的 model_before / model_after。"""
    note = ""
    before: list[list] = []
    after: list[list] = []
    if rounds_json is not None:
        for entry in rounds_json:
            if not isinstance(entry, dict):
                continue
            round_no = entry.get("round")
            mb = entry.get("model_before")
            ma = entry.get("model_after")
            if isinstance(mb, dict) and mb.get("cv") is not None:
                before.append([round_no, mb["cv"]])
            if isinstance(ma, dict) and ma.get("cv") is not None:
                after.append([round_no, ma["cv"]])
        if not before and not after:
            note = "rounds.json 存在但无 model_before/model_after, blocks.png 上栏 (模型侧) 省略。"
    else:
        note = "本 stage 无 rounds.json (无 model_before/model_after), blocks.png 上栏 (模型侧) 省略。"
    return {
        "available": bool(before or after),
        "before": before or None,
        "after": after or None,
        "note": note,
    }


def _rounds_over_3sigma(rounds: list[dict]) -> list[int]:
    out = []
    for r in rounds:
        if r.get("delta_over_3sigma") is not None and abs(r["delta_over_3sigma"]) > 1.0:
            out.append(r["round"])
    return out


def render_md(
    stage: str,
    run_root: Path,
    noise_floor: dict,
    rounds: list[dict],
    model: dict,
    sources: dict,
    warnings: list[str],
) -> str:
    sigma = noise_floor.get("three_sigma_resolution")
    nf_desc = f"{sigma:.5f}" if sigma is not None else _NA
    nf_src = noise_floor.get("path") or _NA
    side = noise_floor.get("side_px")
    nf_desc_full = f"{nf_desc}" + (f" (口径 {side} px)" if side is not None else "")

    n_total = len(rounds)
    n_missing = sum(1 for r in rounds if r["source"] == "missing")
    n_above = len(_rounds_over_3sigma(rounds))
    drifts = [abs(r["drift"]) for r in rounds if r["drift"] is not None]
    max_drift = max(drifts) if drifts else None

    lines: list[str] = []
    lines.append(f"# D 块汇总 —— {stage}")
    lines.append("")
    lines.append(f"- 运行根: `{run_root}`")
    lines.append(f"- 噪声地板 (3σ): {nf_desc_full}, 来源: `{nf_src}`")
    lines.append(f"- 轮次: {n_total} 轮 (其中 {n_missing} 轮缺数据, 表中以 `{_NA}` 标出)")
    lines.append("")
    lines.append("| 轮 | 基线 CV | 候选 CV | Δ | Δ÷3σ | 漂移 A2−A1 | 重放差 B2−B1 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for r in rounds:
        lines.append(
            "| {round} | {base} | {cand} | {delta} | {dsigma} | {drift} | {replay} |".format(
                round=r["round"],
                base=fmt_general(r["baseline_cv"], ".5f", missing_none=_NA),
                cand=fmt_general(r["candidate_cv"], ".5f", missing_none=_NA),
                delta=fmt_general(r["delta"], ".5f", missing_none=_NA),
                dsigma=fmt_general(r.get("delta_over_3sigma"), ".5f", missing_none=_NA),
                drift=fmt_general(r["drift"], ".5f", missing_none=_NA),
                replay=fmt_general(r["replay_b2_minus_b1"], ".5f", missing_none=_NA),
            )
        )
    lines.append("")
    if sigma is None:
        lines.append(f"**结论**: 噪声地板缺失, 无法判定 |Δ| 是否超 3σ; 共 {n_total} 轮, {n_missing} 轮缺数据。")
    else:
        if max_drift is None:
            drift_desc = "无可用的漂移量"
        elif max_drift < sigma:
            drift_desc = f"最大 |漂移 A2−A1| = {max_drift:.5f}, 小于 3σ = {sigma:.5f}, 漂移可忽略"
        else:
            drift_desc = f"最大 |漂移 A2−A1| = {max_drift:.5f}, 不小于 3σ = {sigma:.5f}, 漂移不可忽略"
        lines.append(
            f"**结论**: {n_total} 轮中 {n_above} 轮 |Δ| 超过 3σ (3σ = {sigma:.5f}); {drift_desc}。"
        )
    if model.get("note"):
        lines.append("")
        lines.append(f"> 模型侧: {model['note']}")
    if warnings:
        lines.append("")
        lines.append("警告:")
        for w in warnings:
            lines.append(f"- {w}")
    lines.append("")
    return "\n".join(lines)


def render_png(
    out_path: Path,
    stage: str,
    rounds: list[dict],
    model: dict,
    sigma: float | None,
) -> None:
    have_model = model["available"]
    nrows = 2 if have_model else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(9.0, 4.2 * nrows), squeeze=False)
    axes = axes.reshape(-1)

    if have_model:
        ax = axes[0]
        for key, color, label in (("before", "tab:blue", "模型前 CV"), ("after", "tab:red", "模型后 CV")):
            series = model.get(key)
            if not series:
                continue
            xs = [p[0] for p in series]
            ys = [p[1] for p in series]
            ax.plot(xs, ys, marker="o", color=color, label=label)
        ax.set_title("模型侧 CV (rounds.json)")
        ax.set_xlabel("轮次")
        ax.set_ylabel("CV")
        ax.grid(alpha=0.3)
        ax.legend()

    ax = axes[-1]
    baseline_pts = [(r["round"], r["baseline_cv"]) for r in rounds if r["baseline_cv"] is not None]
    candidate_pts = [(r["round"], r["candidate_cv"]) for r in rounds if r["candidate_cv"] is not None]
    if baseline_pts:
        ax.plot([p[0] for p in baseline_pts], [p[1] for p in baseline_pts], marker="s", color="tab:blue", label="基线 CV (A1/A2 均值)")
    if candidate_pts:
        ax.plot([p[0] for p in candidate_pts], [p[1] for p in candidate_pts], marker="^", color="tab:red", label="候选 CV (B1/B2 均值)")
    if sigma is not None and baseline_pts:
        low = [v - sigma for _, v in baseline_pts]
        high = [v + sigma for _, v in baseline_pts]
        ax.fill_between([p[0] for p in baseline_pts], low, high, color="tab:blue", alpha=0.15, label="基线 ±3σ")
    if not baseline_pts and not candidate_pts:
        ax.text(0.5, 0.5, "无可用轮次数据", transform=ax.transAxes, ha="center", va="center")
    ax.set_title("实测基线 / 候选 (±3σ 带)")
    ax.set_xlabel("轮次")
    ax.set_ylabel("CV")
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels)

    suptitle = f"D 块汇总 — {stage}"
    if not have_model and model.get("note"):
        suptitle += "  [上栏 (模型侧) 省略: 无 model_before/model_after]"
    fig.suptitle(suptitle)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def resolve_noise_floor(explicit: str | None, run_root: Path, stage: str) -> tuple[dict, list[str]]:
    """按 显式 → 1000px 缺省 → stage_a 缺省 的顺序解析噪声地板。"""
    warnings: list[str] = []
    candidates: list[Path] = []
    if explicit:
        candidates.append(_resolve_noise_floor_path(explicit))
    else:
        if "1000" in stage:
            candidates.append(ROOT / "logs" / "target_size_sweep" / "noise_floor_1000px_after_focus" / "noise_floor.json")
        candidates.append(run_root / "stage_a" / "repeatability" / "noise_floor.json")
    for cand in candidates:
        if cand.is_file():
            data = _load_json(cand)
            if data is None:
                warnings.append(f"噪声地板解析失败, 跳过: {cand}")
                continue
            sigma = data.get("three_sigma_resolution")
            if sigma is None:
                warnings.append(f"噪声地板缺少 three_sigma_resolution: {cand}")
                continue
            return (
                {
                    "path": cand.as_posix(),
                    "three_sigma_resolution": float(sigma),
                    "side_px": data.get("side_px"),
                    "exposure_ms": data.get("exposure_ms"),
                },
                warnings,
            )
    return {"path": None, "three_sigma_resolution": None, "side_px": None, "exposure_ms": None}, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D 块离线汇总 (不碰设备)")
    parser.add_argument(
        "--run-root",
        default="data/debug/defocus_differentiable/20261023_101822",
        help="campaign 根目录 (默认: 20261023_101822)",
    )
    parser.add_argument("--stage", default="stage_d", help="stage 目录名 (stage_d / stage_d1000 / ...)")
    parser.add_argument("--noise-floor", default=None, help="噪声地板 JSON (缺省见模块说明)")
    parser.add_argument("--out", default=None, help="输出目录 (默认 <run-root>/<stage>/summary/)")
    args = parser.parse_args(argv)

    run_root = _resolve_run_root(args.run_root)
    stage_dir = run_root / args.stage
    if not stage_dir.is_dir():
        logger.info(f"错误: stage 目录不存在: {stage_dir}")
        return 1
    out_dir = Path(args.out) if args.out else stage_dir / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    chain_path = stage_dir / "chain.json"
    chain = _load_json(chain_path) if chain_path.is_file() else None

    noise_floor, warnings = resolve_noise_floor(args.noise_floor, run_root, args.stage)
    sigma = noise_floor["three_sigma_resolution"]
    if noise_floor["path"] is None:
        warnings.append("未找到噪声地板 JSON, Δ÷3σ 与 3σ 判据全部置空")

    # 口径核对: 噪声地板 side_px 与逐帧 JSON 的冻结窗 side_px
    if sigma is not None and noise_floor.get("side_px") is not None:
        children = sorted(stage_dir.iterdir()) if stage_dir.is_dir() else []
        for child in children:
            if not child.is_dir() or not _RUN_DIR_RE.match(child.name):
                continue
            data = _load_json(child / _PER_RUN_JSON_NAME)
            if isinstance(data, dict):
                mask = data.get("frozen_mask")
                if isinstance(mask, dict) and mask.get("side_px") not in (None, noise_floor["side_px"]):
                    warnings.append(
                        f"噪声地板口径 {noise_floor['side_px']} px 与 {child.name} 冻结窗 {mask.get('side_px')} px 不一致, 判据可能错配"
                    )
                    break

    rounds, sources, w2 = build_rounds(run_root, stage_dir, chain)
    warnings.extend(w2)
    for r in rounds:
        if r["delta"] is not None and sigma:
            r["delta_over_3sigma"] = r["delta"] / sigma
        else:
            r["delta_over_3sigma"] = None

    rounds_json_path = stage_dir / "rounds.json"
    rounds_json = None
    if rounds_json_path.is_file():
        raw = _load_json(rounds_json_path)
        if isinstance(raw, list):
            rounds_json = raw
    model = build_model(rounds_json)

    above = _rounds_over_3sigma(rounds)
    payload = {
        "stage": args.stage,
        "run_root": run_root.as_posix(),
        "noise_floor": noise_floor,
        "rounds": rounds,
        "above_3sigma": len(above),
        "above_3sigma_rounds": above,
        "missing_rounds": [r["round"] for r in rounds if r["source"] == "missing"],
        "model": model,
        "sources": {**sources, "noise_floor": noise_floor["path"]},
        "warnings": warnings,
    }

    md_path = out_dir / "blocks.md"
    png_path = out_dir / "blocks.png"
    json_path = out_dir / "blocks.json"
    md_path.write_text(render_md(args.stage, run_root, noise_floor, rounds, model, sources, warnings), encoding="utf-8")
    render_png(png_path, args.stage, rounds, model, sigma)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"stage={args.stage} 轮次={len(rounds)} 缺={len(payload['missing_rounds'])} 超3σ={len(above)}")
    logger.info(f"  md : {md_path}")
    logger.info(f"  png: {png_path}")
    logger.info(f"  jsn: {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
