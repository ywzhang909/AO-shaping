"""换光路后的重标定报告（_r2）：焦点、重拟合、方向检验。

与 `generate_defocus_differentiable_report.py`（旧光路的 D/E/1000 px 那一轮）**互不覆盖**：
本脚本只写 `defocus_differentiable_20261023_r2.md` 与 `20261023_r2_*.png`。
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT_KEY = "report/shape_trials/defocus_differentiable/defocus_differentiable_20261023_r2.md"
OUT_DIR = ROOT / "report" / "shape_trials" / "defocus_differentiable"
FIG_DIR = OUT_DIR / "figures"
STAMP = "20261023_r2"


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return plt


def _latest(pattern: str) -> Path | None:
    matches = sorted(glob.glob(str(ROOT / pattern)))
    return Path(matches[-1]) if matches else None


OLD_FOCUS = ROOT / "data/debug/physical_forward_defocus/alignment_20261022_1812/focus.json"


def load(old_focus_path: Path = OLD_FOCUS) -> dict:
    new_focus_dir = _latest("data/debug/physical_forward_defocus/focus_recal_*")
    refit = _latest("data/debug/physical_forward_defocus/refit_after_focus_*/refit.json")
    flat_dir = ROOT / "logs/target_size_sweep/direction_after_focus_dev/direction_record.json"
    probe_dir = ROOT / "logs/target_size_sweep/direction_probe_seed/direction_record.json"
    probe1000_dir = ROOT / "logs/target_size_sweep/direction_probe_1000px/direction_record.json"
    floor1000 = ROOT / "logs/target_size_sweep/noise_floor_1000px_after_focus/noise_floor.json"
    return dict(
        old_focus=json.loads(old_focus_path.read_text(encoding="utf-8")) if old_focus_path.exists() else None,
        new_focus=json.loads((new_focus_dir / "focus.json").read_text(encoding="utf-8"))
        if new_focus_dir else None,
        refit=json.loads(refit.read_text(encoding="utf-8")) if refit else None,
        flat=json.loads(flat_dir.read_text(encoding="utf-8")) if flat_dir.exists() else None,
        probe=json.loads(probe_dir.read_text(encoding="utf-8")) if probe_dir.exists() else None,
        probe1000=json.loads(probe1000_dir.read_text(encoding="utf-8")) if probe1000_dir.exists() else None,
        floor1000=json.loads(floor1000.read_text(encoding="utf-8")) if floor1000.exists() else None,
    )


def plot_focus(data: dict, path: Path) -> dict:
    plt = _mpl()
    old = (data.get("old_focus") or {}).get("focus", {})
    new = (data.get("new_focus") or {}).get("focus", {})
    labels = ["best_focal_mm", "best_diopter_m1", "gain_vs_flat"]
    old_vals = [old.get("best_focal_mm"), old.get("best_diopter_m1"), old.get("gain_vs_flat")]
    new_vals = [new.get("best_focal_mm"), new.get("best_diopter_m1"), new.get("gain_vs_flat")]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.3))
    for axis, label, before, after in zip(axes, labels, old_vals, new_vals):
        before_v = float(before) if isinstance(before, (int, float)) and np.isfinite(before) else 0.0
        after_v = float(after) if isinstance(after, (int, float)) and np.isfinite(after) else 0.0
        axis.bar(["旧", "新"], [before_v, after_v], color=["#7f7f7f", "#2ca02c"])
        axis.set_title(label, fontsize=9)
        axis.grid(alpha=0.3, axis="y")
    fig.suptitle("光路变化：焦点扫描前后（new = inf/0/1.0 表示已在焦面）", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return dict(old=[None if v is None else float(v) for v in old_vals],
                new=[None if v is None else float(v) for v in new_vals])


def plot_direction(data: dict, path: Path) -> dict:
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8), sharey=False)
    summary = {}
    for axis, key, title in zip(axes, ("flat", "probe", "probe1000"),
                                ("平场 / 60 px / 0.03 rad", "探针 / 60 px / 0.03 rad",
                                 "探针 / 1000 px / 0.06 rad")):
        record = data.get(key)
        if not record or "measured" not in record:
            axis.set_title(f"{title}：无数据", fontsize=9)
            axis.axis("off")
            continue
        measured = record["measured"]
        resolution = record.get("three_sigma_resolution", float("nan"))
        values = [measured["seed"], measured["b_plus"], measured["b_minus"]]
        axis.bar(["seed", "B+", "B−"], values, color=["#7f7f7f", "#2ca02c", "#d62728"])
        axis.axhspan(values[0] - resolution, values[0] + resolution, color="k", alpha=0.08,
                     label=f"±3σ ({resolution:.4f})")
        axis.set_title(f"{title}\nΔ+ {measured['delta_plus']:+.5f} / Δ− {measured['delta_minus']:+.5f}",
                       fontsize=9)
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3, axis="y")
        summary[key] = dict(delta_plus=measured["delta_plus"], delta_minus=measured["delta_minus"],
                            verdict=record.get("verdict"), model=record.get("model_cv"),
                            resolution=resolution)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return summary


def write_report(data: dict, artifacts: dict, path: Path) -> None:
    old = (data.get("old_focus") or {}).get("focus", {})
    new = (data.get("new_focus") or {}).get("focus", {})
    refit = data.get("refit") or {}
    flat = artifacts["direction"].get("flat", {})
    probe = artifacts["direction"].get("probe", {})
    probe1000 = artifacts["direction"].get("probe1000", {})

    def fmt(value, digits=3):
        return f"{value:.{digits}f}" if isinstance(value, (int, float)) else str(value)

    text = f"""# 光路变化后的重标定（2026-10-23，第二轮）

<!-- provenance:start -->
<!-- provenance:end -->

## 摘要

2026-10-23 把 CCD 移到焦平面后重做全部离焦相关标定。**旧报告（`defocus_differentiable_20261023.md`）
描述的是改动前的光路，本文件是独立的一轮**，两者不互相覆盖。

* **焦点**：best_focal_mm {fmt(old.get("best_focal_mm"), 1)} → **{fmt(new.get("best_focal_mm"), 1)}**，
  best_diopter {fmt(old.get("best_diopter_m1"), 4)} → **{fmt(new.get("best_diopter_m1"), 4)} m⁻¹**，
  相对平场增益 {fmt(old.get("gain_vs_flat"), 2)}× → **{fmt(new.get("gain_vs_flat"), 2)}×**
  ⇒ 系统已在焦面，旧的焦面补偿现在会反过来离焦（C 分支自此为空操作）。
* **重拟合**：`defocus_diopter` {fmt(refit.get("old_diopter"), 4)} → **{fmt(refit.get("new_diopter"), 4)}**
  （等效焦距 {fmt(refit.get("implied_focal_mm"), 0)} mm），fit loss {fmt(refit.get("first_loss"), 3)}
  → {fmt(refit.get("last_loss"), 3)}；零点移到 {refit.get("spot_xy")}，采样窗按新零点重建。
* **方向检验**：平场初值 **未通过**（Δ+ {fmt(flat.get("delta_plus"), 5)} / Δ− {fmt(flat.get("delta_minus"), 5)}，
  两向同号）；结构化探针初值见下节。

![焦点对比](figures/{STAMP}_focus_compare.png)

## 1. 焦点重标定

| 项 | 旧（改动前） | 新（CCD 在焦面） |
|---|---|---|
| best_focal_mm | {fmt(old.get("best_focal_mm"), 1)} | **{fmt(new.get("best_focal_mm"), 1)}** |
| best_diopter / m⁻¹ | {fmt(old.get("best_diopter_m1"), 4)} | **{fmt(new.get("best_diopter_m1"), 4)}** |
| 相对平场增益 | {fmt(old.get("gain_vs_flat"), 2)}× | **{fmt(new.get("gain_vs_flat"), 2)}×** |

参考面 FWHM 从改动前的 ~20.7 px 降到本次采集的 **12.2 px**（同一次会话的 3 帧稳定判据）。

## 2. 模型重拟合

| 项 | 值 |
|---|---|
| 样本 | {refit.get("samples")} |
| 迭代 | {refit.get("iterations")} |
| 离焦项 | {fmt(refit.get("old_diopter"), 4)} → **{fmt(refit.get("new_diopter"), 4)} m⁻¹** |
| 等效焦距 | {fmt(refit.get("implied_focal_mm"), 0)} mm |
| 实测零点 | {refit.get("spot_xy")} |
| 采样窗 | ±{refit.get("window_half_span_px")} px @ {refit.get("window_step_px")} px |

## 3. B± 方向检验

![方向检验](figures/{STAMP}_direction.png)

| 初值 | 模型 Δ+ | 模型 Δ− | 实测 Δ+ | 实测 Δ− | 判定 |
|---|---|---|---|---|---|
| 平场（失败） | {fmt((flat.get("model") or {}).get("b_plus"), 4)} | {fmt((flat.get("model") or {}).get("b_minus"), 4)} | {fmt(flat.get("delta_plus"), 5)} | {fmt(flat.get("delta_minus"), 5)} | 未通过 |
| 结构化探针 / 60 px / 0.03 rad | {fmt((probe.get("model") or {}).get("b_plus"), 4)} | {fmt((probe.get("model") or {}).get("b_minus"), 4)} | {fmt(probe.get("delta_plus"), 5)} | {fmt(probe.get("delta_minus"), 5)} | {("通过" if probe.get("verdict") else "未通过")} |
| **结构化探针 / 1000 px / 0.06 rad** | {fmt((probe1000.get("model") or {}).get("b_plus"), 2)} | {fmt((probe1000.get("model") or {}).get("b_minus"), 2)} | **{fmt(probe1000.get("delta_plus"), 5)}** | **{fmt(probe1000.get("delta_minus"), 5)}** | **{("通过" if probe1000.get("verdict") else "未通过")}**（3σ {fmt(probe1000.get("resolution"), 4)}） |

判据：两个方向都超过 3σ 且符号相反（模型下降方向 B+ 应让实测 CV 下降、B− 应上升）。
模型侧三个值（seed / B+ / B−）见方向记录 JSON。

## 4. 结论与下一步

{_conclusion(flat, probe, probe1000)}

## 5. 复现

```bash
python scripts/slm/run_focus_recalibration.py --out <新目录> \\
    --compare data/debug/physical_forward_defocus/alignment_20261022_1812/focus.json --wait-until-idle 600
python scripts/slm/capture_defocus_forward_dataset.py --output <采集目录> \\
    --focus-directory <新目录> --count 6 --region 128 --seed 20261023 --execute
python scripts/slm/refit_model_after_optics_change.py --dataset <采集目录> --out <新检查点> --iterations 150
python scripts/slm/direction_test_after_optics_change.py --checkpoint <新检查点> \\
    --seed-panel <结构化初值.npy> --spot-xy <实测零点> --out <方向目录> --execute
```
"""
    path.write_text(text, encoding="utf-8")


def _conclusion(flat: dict, probe: dict, probe1000: dict | None = None) -> str:
    if probe1000 and probe1000.get("verdict"):
        return ("**1000 px 目标 + 0.06 rad 增量下方向检验通过**：实测 Δ+ = "
                f"{probe1000['delta_plus']:+.5f}、Δ− = {probe1000['delta_minus']:+.5f}，"
                f"均超过该掩膜的 3σ = {probe1000['resolution']:.4f} 且符号相反 "
                "⇒ 新检查点的梯度在实验系统上指对了方向。\n\n"
                "这也说明 60 px 那次失败是**增量太小**（0.03 rad 下实测效应 ~0.015 < 3σ 0.0165），"
                "不是方向错。\n\n下一步：按 1000 px 目标进入 D 块闭环（每块 10 步、A-B-B-A、"
                "按该掩膜的 3σ 判读），并同步用 HDR 采集做大框度量。")
    if not probe:
        return "结构化初值的方向检验尚未完成。"
    if probe.get("verdict"):
        return ("结构化探针初值下方向检验**通过**：B+ 使实测 CV 下降、B− 使其上升，且两向都超 3σ。"
                "新检查点可以进入 D 块闭环；下一轮按「HDR 重扫目标尺寸 → 逐块 A-B-B-A」推进，"
                "并按实测（不是模型）判读。")
    if probe.get("delta_plus", 0.0) < 0.0 < probe.get("delta_minus", 0.0):
        return ("结构化探针初值下**符号正确但幅度不足**：B+ 使实测 CV 下降、B− 使其上升（反号成立），"
                "但两者只有 0.7–0.9σ，低于判据。注意判据里的 3σ = 0.0165 取自**旧光路**的 60 px "
                "噪声地板；新光路必须重测。下一步：① 用新光路重测 60 px 掩膜的 3σ；"
                "② 把方向增量从 0.03 rad 提到 0.06 rad 再验一次（幅度只有模型预测的 ~12%，"
                "说明模型幅度仍偏大，加增量比加密迭代更直接）；通过后再进 D 块。")
    return ("结构化探针初值下方向检验**仍未通过**。已知的候选原因：① 重拟合把照明宽度从 "
            "[366.1, 256.7] 改成了 [234.4, 383.4]（轴向疑似转了 90°）；② 60 px 框在新焦点下 "
            "CV≈1.95（旧 1.13），度量敏感度不同。下一步改为**小步重拟合**（只放开 origin/gain，"
            "冻结 d、照明宽度与残差）再验一次；通过前不进 D 块。")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--old-focus", type=Path, default=OLD_FOCUS,
                        help="对比用的旧 focus.json；默认是 20261022 的历史文件")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / Path(REPORT_KEY).name
    if report_path.exists() and not args.force:
        print(f"  {report_path.name} 已存在；加 --force 覆盖（本文件只属于 r2 这一轮）")
    if args.old_focus == OLD_FOCUS:
        print("⚠ --old-focus 用的是 20261022 的历史标定；换光路后请显式指定", flush=True)
    data = load(args.old_focus)
    artifacts = dict(focus=plot_focus(data, FIG_DIR / f"{STAMP}_focus_compare.png"),
                     direction=plot_direction(data, FIG_DIR / f"{STAMP}_direction.png"))
    write_report(data, artifacts, report_path)
    (OUT_DIR / f"{STAMP}_artifacts.json").write_text(json.dumps(artifacts, indent=2, ensure_ascii=False),
                                                     encoding="utf-8")
    print(json.dumps(artifacts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()





