# -*- coding: utf-8 -*-
"""离线比较两个 transfer fit 运行的逐探针指标（不访问任何设备）。

用途
----
``ao_shaping.tools.slm.transfer.fit`` 每次运行都会把全部参与拟合的行（train /
test / validation）的前向预测指标按 ``selected`` 顺序写进 ``result.json`` 的
``metrics``。行顺序等于 manifest 里 ``valid()`` 通过的行顺序，``provenance.json``
的 ``rows`` 与之逐项对应，但不带标签以外的判据信息。

本脚本把两次运行的 ``metrics`` 与 ``provenance.rows`` 对齐，输出：

1. 运行元数据对照（迭代数、train loss、manifest/checkpoint 哈希、ramp 矩阵）；
2. 全体 / train / test 的聚合指标；
3. test 每个探针一行的对照表（含相位 RMS 与误差变化）；
4. test 相位族（``holdout_*_smooth`` / ``holdout_*_zernike``）聚合；
5. train 按「模态 + 幅度」聚合；
6. 退化清单（relative L2 变差或 correlation 下降）；
7. test 上误差变化量与相位 RMS 的秩相关。

对齐可信度由一次硬断言保证：``cv_measured`` 只取决于实测图像，与模型参数无关，
两次运行必须逐位相同。若对齐错位或数据被换过，这一项立刻暴露。

只读离线：不 import 设备驱动，不写数据目录。

用法::

    python scripts/slm/compare_transfer_fit_runs.py ^
        --dataset data/debug/slm_transfer/20261118_transfer_mil_zernike_probe_v1 ^
        --baseline data/debug/slm_transfer/.../fit_m3_baseline ^
        --updated  data/debug/slm_transfer/.../fit_m3_full ^
        --out      report/shape_trials/physical_refine/fit_compare_20261118.md
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

METRICS = ("correlation", "relative_l2", "cv_abs_error", "cv_prediction", "cv_measured")
CV_MEASURED_TOL = 1e-9


def load_run(run: Path) -> dict[str, Any]:
    """读取一次 fit 运行的 result/provenance，并检查条目数一致。"""
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    rows = provenance["rows"]
    metrics = result["metrics"]
    if len(rows) != len(metrics):
        raise ValueError(f"{run}: {len(rows)} provenance rows vs {len(metrics)} metrics")
    return {"path": run, "result": result, "provenance": provenance, "rows": rows,
            "metrics": metrics}


def align(baseline: dict[str, Any], updated: dict[str, Any]) -> list[dict[str, Any]]:
    """按行序对齐两次运行，并对实测侧指标做逐位一致性断言。"""
    left = baseline["rows"]
    right = updated["rows"]
    if [row["label"] for row in left] != [row["label"] for row in right]:
        raise ValueError("provenance row order differs between runs; cannot align metrics")
    entries = []
    for index, (lrow, urow) in enumerate(zip(left, right, strict=True)):
        lm, um = baseline["metrics"][index], updated["metrics"][index]
        if abs(float(lm["cv_measured"]) - float(um["cv_measured"])) > CV_MEASURED_TOL:
            raise ValueError(
                f"row {index} {lrow['label']}: cv_measured differs "
                f"({lm['cv_measured']} vs {um['cv_measured']}); runs are not on the same data")
        entries.append({
            "index": index,
            "label": lrow["label"],
            "split": lrow["split"],
            "file": lrow["file"],
            "exposure_ms": lrow["exposure_ms"],
            "baseline": {name: float(lm[name]) for name in METRICS},
            "updated": {name: float(um[name]) for name in METRICS},
        })
    return entries


def phase_family(label: str) -> str:
    """把标签压成相位族：test 用 holdout 后缀，train 用「模态 + 幅度」。"""
    if label.startswith("holdout_"):
        parts = label.split("_")
        return parts[-1] if len(parts) > 2 else "holdout"
    parts = label.split("_")
    if len(parts) >= 4 and parts[2].upper().startswith("Z"):
        return f"{parts[2].upper()} {parts[3]}"
    return parts[-1]


def phase_rms(dataset: Path, file: str) -> float | None:
    """读 npz 的 ``model_phase``（rad）给出 RMS；缺字段时返回 None。"""
    path = dataset / file
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as payload:
        if "model_phase" not in payload:
            return None
        phase = np.asarray(payload["model_phase"], dtype=float)
    return float(np.sqrt(np.mean((phase - phase.mean()) ** 2)))


def aggregate(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """一组行的均值/中位数/改善计数。"""
    if not entries:
        return {"n": 0}
    summary: dict[str, Any] = {"n": len(entries)}
    for name in METRICS:
        base = np.array([e["baseline"][name] for e in entries], dtype=float)
        upd = np.array([e["updated"][name] for e in entries], dtype=float)
        summary[name] = {
            "baseline_mean": float(base.mean()),
            "updated_mean": float(upd.mean()),
            "baseline_median": float(np.median(base)),
            "updated_median": float(np.median(upd)),
            "delta_mean": float(upd.mean() - base.mean()),
        }
    summary["improved_relative_l2"] = sum(
        1 for e in entries if e["updated"]["relative_l2"] < e["baseline"]["relative_l2"])
    summary["improved_correlation"] = sum(
        1 for e in entries if e["updated"]["correlation"] > e["baseline"]["correlation"])
    summary["improved_cv_abs_error"] = sum(
        1 for e in entries if e["updated"]["cv_abs_error"] < e["baseline"]["cv_abs_error"])
    summary["min_correlation_updated"] = float(
        min(e["updated"]["correlation"] for e in entries))
    summary["min_correlation_baseline"] = float(
        min(e["baseline"]["correlation"] for e in entries))
    return summary


def rank_correlation(x: list[float], y: list[float]) -> float | None:
    """无 scipy 依赖的 Spearman 秩相关（并列取平均秩）。"""
    if len(x) < 3:
        return None

    def ranks(values: list[float]) -> np.ndarray:
        order = np.argsort(np.asarray(values, dtype=float), kind="stable")
        out = np.empty(len(values), dtype=float)
        sorted_values = np.asarray(values, dtype=float)[order]
        i = 0
        while i < len(values):
            j = i
            while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
                j += 1
            out[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return out

    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def load_iterations(run: Path, row_count: int) -> list[dict[str, Any]] | None:
    """读 ``iterations.pkl``（逐迭代的全行 metrics）；缺文件时返回 None。"""
    path = run / "iterations.pkl"
    if not path.exists():
        return None
    with path.open("rb") as stream:
        history = pickle.load(stream)
    for entry in history:
        if len(entry["metrics"]) != row_count:
            raise ValueError(
                f"{path}: iteration {entry['iteration']} has {len(entry['metrics'])} "
                f"metrics vs {row_count} rows")
    return history


def trajectory(history: list[dict[str, Any]], entries: list[dict[str, Any]],
               split: str | None = None) -> list[dict[str, float]]:
    """按相位族给出每轮迭代的 mean relative_l2（``split`` 为空时用全部行）。"""
    indices: dict[str, list[int]] = {}
    for entry in entries:
        if split is not None and entry["split"] != split:
            continue
        indices.setdefault(entry["family"], []).append(int(entry["index"]))
    series: list[dict[str, float]] = []
    for step in history:
        values = {}
        for family, positions in indices.items():
            values[family] = float(np.mean(
                [float(step["metrics"][position]["relative_l2"]) for position in positions]))
        values["iteration"] = float(step["iteration"])
        values["loss"] = float(step["loss"])
        series.append(values)
    return series


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def delta_str(value: float, digits: int = 4) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{digits}f}"


def metric_row(name: str, stats: dict[str, Any], lower_is_better: bool) -> str:
    block = stats[name]
    best = "improved" if (block["delta_mean"] < 0) == lower_is_better else "worse"
    return ("| {name} | {b} | {u} | {d} | {best} |".format(
        name=name, b=fmt(block["baseline_mean"]), u=fmt(block["updated_mean"]),
        d=delta_str(block["delta_mean"]), best=best))


def build_report(dataset: Path, baseline: dict[str, Any], updated: dict[str, Any],
                 entries: list[dict[str, Any]],
                 history: list[dict[str, Any]] | None = None) -> tuple[str, dict[str, Any]]:
    for entry in entries:
        entry["family"] = phase_family(entry["label"])
        entry["phase_rms_rad"] = phase_rms(dataset, entry["file"])
        entry["delta_relative_l2"] = entry["updated"]["relative_l2"] - entry["baseline"]["relative_l2"]
        entry["delta_correlation"] = entry["updated"]["correlation"] - entry["baseline"]["correlation"]

    groups: dict[str, list[dict[str, Any]]] = {
        "all": entries,
        "train": [e for e in entries if e["split"] == "train"],
        "test": [e for e in entries if e["split"] == "test"],
        "validation": [e for e in entries if e["split"] == "validation"],
    }
    by_family: dict[str, dict[str, list[dict[str, Any]]]] = {"train": {}, "test": {}}
    for entry in entries:
        split = entry["split"]
        if split in by_family:
            by_family[split].setdefault(entry["family"], []).append(entry)

    aggregate_all = {name: aggregate(rows) for name, rows in groups.items()}
    aggregate_family = {split: {name: aggregate(rows) for name, rows in families.items()}
                        for split, families in by_family.items()}

    test = groups["test"]
    rms = [e["phase_rms_rad"] for e in test if e["phase_rms_rad"] is not None]
    deltas = [e["delta_relative_l2"] for e in test if e["phase_rms_rad"] is not None]
    rho = rank_correlation(rms, deltas)

    regressions = sorted(entries, key=lambda e: e["delta_relative_l2"], reverse=True)
    regressions = [e for e in regressions if e["delta_relative_l2"] > 0]

    lines: list[str] = []
    lines.append("# Transfer fit 运行对照（offline）")
    lines.append("")
    lines.append(f"- dataset: `{dataset}`")
    lines.append(f"- baseline: `{baseline['path']}` (iteration "
                 f"{baseline['result']['iteration']}, loss {fmt(baseline['result']['loss'], 6)})")
    lines.append(f"- updated: `{updated['path']}` (iteration "
                 f"{updated['result']['iteration']}, loss {fmt(updated['result']['loss'], 6)})")
    lines.append("")
    lines.append("## 1. 运行一致性")
    lines.append("")
    checks = {
        "manifest_sha256": (baseline["provenance"].get("manifest_sha256"),
                            updated["provenance"].get("manifest_sha256")),
        "checkpoint": (baseline["provenance"].get("checkpoint"),
                       updated["provenance"].get("checkpoint")),
        "adaptation": (baseline["provenance"].get("adaptation"),
                       updated["provenance"].get("adaptation")),
        "radiometry": (baseline["provenance"].get("radiometry"),
                       updated["provenance"].get("radiometry")),
    }
    for name, (left, right) in checks.items():
        state = "same" if left == right else "DIFFERS"
        lines.append(f"- {name}: {state}")
    same_ramp = baseline["provenance"].get("ramp_matrix") == updated["provenance"].get("ramp_matrix")
    lines.append(f"- ramp_matrix: {'same' if same_ramp else 'DIFFERS'}")
    lines.append(f"- 行数: {len(entries)}；`cv_measured` 逐行一致（对齐断言通过）")
    lines.append("")

    lines.append("## 2. 聚合指标（均值；correlation 越高越好，其余越低越好）")
    lines.append("")
    for group_name in ("all", "train", "test", "validation"):
        stats = aggregate_all[group_name]
        if not stats.get("n"):
            continue
        lines.append(f"### {group_name} (n={stats['n']})")
        lines.append("")
        lines.append("| metric | baseline | updated | delta | verdict |")
        lines.append("| --- | --- | --- | --- | --- |")
        for name, lower in (("correlation", False), ("relative_l2", True),
                            ("cv_abs_error", True), ("cv_prediction", True)):
            lines.append(metric_row(name, stats, lower))
        lines.append("")
        lines.append(f"改善行数：relative_l2 {stats['improved_relative_l2']}/{stats['n']}，"
                     f"correlation {stats['improved_correlation']}/{stats['n']}，"
                     f"cv_abs_error {stats['improved_cv_abs_error']}/{stats['n']}；"
                     f"最差 correlation baseline {fmt(stats['min_correlation_baseline'])} → "
                     f"updated {fmt(stats['min_correlation_updated'])}")
        lines.append("")

    lines.append("## 3. test 逐探针")
    lines.append("")
    lines.append("| # | label | family | phase RMS (rad) | corr base | corr upd | rL2 base | rL2 upd | ΔrL2 | cvErr base | cvErr upd |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for entry in test:
        rms_value = "n/a" if entry["phase_rms_rad"] is None else fmt(entry["phase_rms_rad"], 3)
        lines.append("| {i} | `{label}` | {family} | {rms} | {cb} | {cu} | {lb} | {lu} | {dl} | {eb} | {eu} |".format(
            i=entry["index"], label=entry["label"], family=entry["family"], rms=rms_value,
            cb=fmt(entry["baseline"]["correlation"]), cu=fmt(entry["updated"]["correlation"]),
            lb=fmt(entry["baseline"]["relative_l2"]), lu=fmt(entry["updated"]["relative_l2"]),
            dl=delta_str(entry["delta_relative_l2"]),
            eb=fmt(entry["baseline"]["cv_abs_error"]), eu=fmt(entry["updated"]["cv_abs_error"])))
    lines.append("")
    lines.append(f"test 上 Δrelative L2 与相位 RMS 的 Spearman ρ = "
                 f"{'n/a' if rho is None else fmt(rho, 3)}（样本 {len(rms)}）")
    lines.append("")

    lines.append("## 4. test 相位族")
    lines.append("")
    lines.append("| family | n | corr base | corr upd | rL2 base | rL2 upd | ΔrL2 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for family, rows in sorted(aggregate_family["test"].items()):
        lines.append("| {name} | {n} | {cb} | {cu} | {lb} | {lu} | {dl} |".format(
            name=family, n=rows["n"], cb=fmt(rows["correlation"]["baseline_mean"]),
            cu=fmt(rows["correlation"]["updated_mean"]),
            lb=fmt(rows["relative_l2"]["baseline_mean"]),
            lu=fmt(rows["relative_l2"]["updated_mean"]),
            dl=delta_str(rows["relative_l2"]["delta_mean"])))
    lines.append("")

    lines.append("## 5. train 模态族（模态 + 幅度）")
    lines.append("")
    lines.append("| family | n | corr base | corr upd | rL2 base | rL2 upd | ΔrL2 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for family, rows in sorted(aggregate_family["train"].items()):
        lines.append("| {name} | {n} | {cb} | {cu} | {lb} | {lu} | {dl} |".format(
            name=family, n=rows["n"], cb=fmt(rows["correlation"]["baseline_mean"]),
            cu=fmt(rows["correlation"]["updated_mean"]),
            lb=fmt(rows["relative_l2"]["baseline_mean"]),
            lu=fmt(rows["relative_l2"]["updated_mean"]),
            dl=delta_str(rows["relative_l2"]["delta_mean"])))
    lines.append("")

    lines.append(f"## 6. 退化行（relative L2 变差，共 {len(regressions)}）")
    lines.append("")
    if regressions:
        lines.append("| label | split | family | ΔrL2 | Δcorr |")
        lines.append("| --- | --- | --- | --- | --- |")
        for entry in regressions:
            lines.append("| `{label}` | {split} | {family} | {dl} | {dc} |".format(
                label=entry["label"], split=entry["split"], family=entry["family"],
                dl=delta_str(entry["delta_relative_l2"]),
                dc=delta_str(entry["delta_correlation"])))
    else:
        lines.append("无。")
    lines.append("")
    lines.append("> 说明：本表只说明同一批实测数据上「预测指标」的变化，不构成光束整形验收。"
                 "目标形状仍需固定曝光/掩膜下的独立 CCD 配对测量与精确相位重放。")
    lines.append("")

    trajectory_payload: dict[str, Any] = {}
    if history:
        lines.append("## 7. 收敛轨迹（updated 运行，逐迭代 mean relative L2）")
        lines.append("")
        for split in ("train", "test"):
            series = trajectory(history, entries, split)
            if not series:
                continue
            families = [name for name in series[0] if name not in {"iteration", "loss"}]
            positions = sorted({0, len(series) // 4, len(series) // 2,
                                len(series) - 1, int(updated["result"]["iteration"])})
            header = "| iteration | loss | " + " | ".join(families) + " |"
            lines.append(f"### {split}")
            lines.append("")
            lines.append(header)
            lines.append("| --- | --- | " + " | ".join("---" for _ in families) + " |")
            for position in sorted(set(positions)):
                step = series[position]
                cells = " | ".join(fmt(step[name]) for name in families)
                lines.append(f"| {int(step['iteration'])} | {fmt(step['loss'], 6)} | {cells} |")
            lines.append("")
            best = {name: int(series[int(np.argmin([step[name] for step in series]))]["iteration"])
                    for name in families}
            lines.append("各族自身最优迭代：" + "，".join(f"{name} → {step}" for name, step in best.items()))
            lines.append("")
            trajectory_payload[split] = series
        lines.append("")

    payload = {
        "dataset": str(dataset),        "baseline": {"path": str(baseline["path"]),
                     "iteration": baseline["result"]["iteration"],
                     "loss": baseline["result"]["loss"]},
        "updated": {"path": str(updated["path"]),
                    "iteration": updated["result"]["iteration"],
                    "loss": updated["result"]["loss"]},
        "consistency": {name: left == right for name, (left, right) in checks.items()}
                       | {"ramp_matrix_same": same_ramp},
        "aggregate": aggregate_all,
        "families": {"train": aggregate_family["train"], "test": aggregate_family["test"]},
        "test_spearman_delta_rl2_vs_phase_rms": rho,
        "rows": entries,
        "trajectory": trajectory_payload,
    }
    return "\n".join(lines), payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="采集目录（含 manifest.json 与 sample_*.npz）")
    parser.add_argument("--baseline", type=Path, required=True, help="baseline fit 输出目录")
    parser.add_argument("--updated", type=Path, required=True, help="updated fit 输出目录")
    parser.add_argument("--out", type=Path, help="Markdown 报告路径；省略则只打印")
    parser.add_argument("--json", type=Path, help="可选：把逐行数据写成 JSON")
    args = parser.parse_args()

    baseline = load_run(args.baseline)
    updated = load_run(args.updated)
    entries = align(baseline, updated)
    history = load_iterations(args.updated, len(updated["rows"]))
    report, payload = build_report(args.dataset, baseline, updated, entries, history)
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"\nwrote {args.out}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
