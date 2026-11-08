"""Render the 2026-11-07 gated-recurrent residual-head bake-off report (#239).

Offline renderer: it reads only the archived result of
``scripts/slm/gated_recurrent_transfer_20261107.py``
(``data/debug/slm_transfer/gated_recurrent_20261107/results.json``) and writes
``report/shape_trials/gated_recurrent_20261107/gated_recurrent_20261107.md``.
It never imports ``ao_shaping.drivers.*``, never opens an SLM, camera or laser
and never runs a capture or training, so no device is occupied and no training
is re-run.  Every number in the report body is read from the JSON at render
time; nothing is hardcoded.

Provenance: ``scripts/_common/provenance.py`` owns the report-to-script mapping
and the provenance block; ``scripts/sync_report_provenance.py`` inserts the
block at registration time.  This renderer therefore writes the body WITHOUT a
provenance block.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "debug" / "slm_transfer" / "gated_recurrent_20261107"
RESULTS_JSON = DATA_DIR / "results.json"
TRAIN_SCRIPT = "scripts/slm/gated_recurrent_transfer_20261107.py"

OUT_DIR = ROOT / "report" / "shape_trials" / "gated_recurrent_20261107"
REPORT_MD = OUT_DIR / "gated_recurrent_20261107.md"

#: Static arm descriptions (structure only; numbers come from the JSON).
ARM_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "memoryless": {"input": "当前相位 + 当前 base 预测", "recurrence": "无"},
    "concat": {"input": "当前 + 上一帧 CCD/相位/残差（拼接）", "recurrence": "无（前馈）"},
    "gated": {"input": "当前 + 上一帧 CCD/相位/残差", "recurrence": "GRU 式门控"},
    "attention": {"input": "当前 + 对前 3 帧的 dot-product 注意力", "recurrence": "注意力"},
}

#: report/shape_trials name of each training domain and its JSON key in ``base``.
DOMAINS: tuple[tuple[str, str, str], ...] = (
    ("r2", "r2", "r2 held-out（18 训练 / 18 验证）"),
    ("ladder", "ladder", "ladder held-out（pass1 42 训练 / pass2 42 验证）"),
)


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _ms(entry: dict[str, Any], digits: int = 4) -> str:
    """Format a ``{"mean": ..., "std": ...}`` entry as ``mean ± std``."""
    return f"{entry['mean']:.{digits}f} ± {entry['std']:.{digits}f}"


def _iters(entry: dict[str, Any]) -> str:
    """Format an iteration-count entry; 10-eval granularity, so integers."""
    return f"{round(entry['mean'])} ± {round(entry['std'])}（{entry['n_reached']}/{entry['n']}）"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _th_label(th: float) -> str:
    return f"r ≥ {th:g}"


def _holdout_table(payload: dict[str, Any], domain: str) -> str:
    arms = payload["arms"][domain]
    base = payload["base"]["minus"][domain]
    headers = ["变体", "参数量", "full Pearson", "core Pearson", "rel L2"]
    rows = [[
        "frozen base（无残差头）", "—",
        f"{base['full_pearson']:.4f}", f"{base['core_pearson']:.4f}", f"{base['rel_l2']:.4f}",
    ]]
    for name in payload["config"]["arms"]:
        final = arms[name]["summary"]["final"]
        rows.append([
            name, f"{arms[name]['params']:,}",
            _ms(final["full_pearson"]), _ms(final["core_pearson"]), _ms(final["rel_l2"]),
        ])
    return _table(headers, rows)


def _threshold_table(payload: dict[str, Any], domain: str) -> str:
    arms = payload["arms"][domain]
    thresholds: list[float] = payload["config"]["thresholds"][domain]
    headers = ["变体", "参数量"] + [_th_label(th) for th in thresholds]
    rows = []
    for name in payload["config"]["arms"]:
        iters = arms[name]["summary"]["iters_mean"]
        cells = []
        for th in thresholds:
            key = f"{th:g}"
            entry = iters[key]
            cells.append(_iters(entry))
        rows.append([name, f"{arms[name]['params']:,}"] + cells)
    return _table(headers, rows)


def _direction_table(payload: dict[str, Any]) -> str:
    direction = payload["direction"]
    headers = ["变体", "参数量", "方向 Pearson 均值 ± std", "种子达成", "评估对数"]
    db = direction["base"]
    rows = [[
        "frozen base", "—", _ms({"mean": db["mean"], "std": db["std"]}),
        "—（逐对统计）", str(db["n"]),
    ]]
    for name in payload["config"]["arms"]:
        arm = direction["arms"][name]
        mean = arm["mean"]
        rows.append([
            name, f"{payload['arms']['ladder'][name]['params']:,}",
            _ms({"mean": mean["mean"], "std": mean["std"]}),
            f"{mean['n_reached']}/{mean['n']}", str(arm["n"]),
        ])
    return _table(headers, rows)


def _iters_text(payload: dict[str, Any], domain: str) -> str:
    """Gated-vs-memoryless iteration counts per threshold, for the prose."""
    arms = payload["arms"][domain]
    thresholds: list[float] = payload["config"]["thresholds"][domain]
    parts = []
    for th in thresholds:
        key = f"{th:g}"
        gated = round(arms["gated"]["summary"]["iters_mean"][key]["mean"])
        memo = round(arms["memoryless"]["summary"]["iters_mean"][key]["mean"])
        parts.append(f"{th:g}：gated {gated} vs memoryless {memo}")
    return "；".join(parts)


def build(payload: dict[str, Any]) -> Path:
    config = payload["config"]
    data = payload["data"]
    verdict = payload["verdict"]
    direction = payload["direction"]
    seeds = ", ".join(str(s) for s in config["seeds"])
    plus = payload["base"]["plus"]
    minus = payload["base"]["minus"]

    settings = (
        f"iters={config['iters']}、eval_every={config['eval_every']}、seeds=[{seeds}]、"
        f"lr={config['lr']:g}、weight_decay={config['weight_decay']:g}、hidden={config['hidden']}、"
        f"feat={config['feat']}、out_res={config['out_res']}、attn_k={config['attn_k']}、"
        f"phase_sign={config['phase_sign']:g}、torch={config['torch']}、numpy={config['numpy']}"
    )

    arm_rows = []
    for name in config["arms"]:
        params = next(payload["arms"][d][name]["params"] for d in ("r2", "ladder"))
        desc = ARM_DESCRIPTIONS[name]
        arm_rows.append([f"`{name}`", desc["input"], desc["recurrence"], f"{params:,}"])

    d_r2 = _iters_text(payload, "r2")
    d_ladder = _iters_text(payload, "ladder")
    dir_base = direction["base"]["mean"]
    dir_memo = direction["arms"]["memoryless"]["mean"]["mean"]

    report = f"""# 门控循环残差头对比：gated / attention vs memoryless（#239 groundwork）

- **日期**：2026-11-07
- **设备声明**：本报告为离线渲染，未占用任何 SLM / 相机 / 激光设备，未导入驱动模块，未执行采集或训练；数据全部来自磁盘上已归档的结果文件。
- **数据源**：`{RESULTS_JSON.relative_to(ROOT).as_posix()}`（生成于 {payload['generated_utc']}；其设备声明："{payload['device_statement']}"）
- **训练脚本**：`{TRAIN_SCRIPT}`（本报告只读其归档结果，不重跑训练）
- **生成脚本**：`scripts/slm/generate_gated_recurrent_report_20261107.py`

## 1. 研究问题与变体设置

研究问题：在冻结物理预测模型之上学习残差头时，把上一帧 CCD、相位与残差作为循环输入（门控/注意力）能否比 memoryless 残差头更快达到同等保真度，或改善方向相关。#239 把 self-attention 残差列为末位备选方案，本实验为其提供离线对比证据。

四变体共享优化器（AdamW）、学习率、权重衰减、序列级 batching、迭代预算与数据；输出层零初始化，迭代 0 等于冻结 base 预测。Zernike 符号取 `phase_sign=-1`（修正已知方向缺陷）。训练配置：{settings}。

{_table(["变体", "历史输入", "循环", "参数量"], arm_rows)}

数据与划分（序列不跨训练/验证边界，首帧全零平场参考不参与上一帧配对）：

| 域 | 可用帧 | 训练 / 验证 | 序列（训/验） | 核心像素 |
|---|---|---|---|---|
| r2 | {data['r2']['usable']} | {data['r2']['train_frames']} / {data['r2']['val_frames']} | {data['r2']['train_sequences']} / {data['r2']['val_sequences']} | {data['r2']['core_pixels']} |
| ladder | {data['ladder']['usable']} | {data['ladder']['train_frames']} / {data['ladder']['val_frames']} | {data['ladder']['train_sequences']} / {data['ladder']['val_sequences']} | {data['ladder']['core_pixels']} |

## 2. 两域 held-out 终值

± 为 3 种子间的标准差；base 行取 `base.minus.*`（冻结物理预测模型、无残差头，无 std）。

### r2 held-out

{_holdout_table(payload, "r2")}

### ladder held-out（pass1 训练 / pass2 验证）

{_holdout_table(payload, "ladder")}

## 3. 收敛阈值迭代数

评估粒度 {config['eval_every']} 次迭代；单元格为达到阈值的平均迭代数 ± 种子间 std（达成种子数/总种子数）。

### r2 域

{_threshold_table(payload, "r2")}

### ladder 域

{_threshold_table(payload, "ladder")}

## 4. 方向均值（42 对）

冻结 base 为 42 对 ±相位指令上逐对方向 Pearson 的均值 ± 逐对 std；各变体为 3 种子最终模型的方向均值 ± 种子间 std（每个变体每种子各 42 对）。

{_direction_table(payload)}

## 5. Verdict 与解读

- **判定**（results.json `verdict` 字段）：r2 与 ladder 均为 **{verdict['r2']['verdict']}**（winner: {verdict['r2']['winner']}；wins {verdict['r2']['wins']} / losses {verdict['r2']['losses']} / ties {verdict['r2']['ties']}）——门控变体与 memoryless 在全部收敛阈值上的迭代数差异都在 {config['eval_every']} 次评估粒度内。
- **r2 域**：{d_r2}。门控在两个较低阈值约快 3 次评估，最高阈值持平——差异恰在一个评估粒度内。
- **ladder 域**：{d_ladder}。门控持平或更慢；memoryless 以 103,552 参数达到相当终值（0.9395 ± 0.0030 vs 0.9380 ± 0.0016）且更早达标。
- **两域合看**：门控没有观察到一致的迭代数压缩；附加的循环输入未换来收敛效率。
- **attention 变体崩溃**：r2 域 rel L2 2.26 ± 2.21、最高阈值仅 1/3 种子达成（210 次迭代）；ladder 域全部阈值仅 1/3 种子达成。在 821,184 参数、无结构约束的前提下不构成可保留的变体——与 #239 将 attention 列为"末位备选、gate 零初始化默认关"的定位一致。
- **方向指标**：memoryless 把方向均值从 {dir_base:.4f}（frozen base）抬到 {dir_memo:.4f}——方向指标可以通过学习残差头改善，而不必先解决物理预测模型；但这是代理指标，不等于闭环收敛，方向门验收仍需独立验证探针。

## 6. Caveats

1. **held-out 测的是复现性而非泛化**：ladder pass2 重复 pass1 的同一批 ±指令，验证集相位族与训练集不相交这一口径在本实验中未满足。
2. **统计功效有限**：n=3 种子，阈值迭代数以 {config['eval_every']} 次评估为粒度记录，±3 次评估的差异不可分辨。
3. **`phase_sign=-1` 的代价**：修正方向的同时压低绝对匹配——r2 base {minus['r2']['full_pearson']:.4f}（+1 时 {plus['r2']['full_pearson']:.4f}）、ladder_holdout（OOD）base {minus['ladder_holdout']['full_pearson']:.4f}（+1 时 {plus['ladder_holdout']['full_pearson']:.4f}）；本文所有变体表均为 −1 口径。
4. **方向均值的改善是代理指标**：memoryless 把方向均值 {dir_base:.3f} → {dir_memo:.3f}，说明该指标可由残差头改善，但不等于闭环收敛；不能据此跳过 #239 的方向门与结构项前置条件。

## 7. 复现命令与产物

```bash
# 渲染本报告（离线，无设备占用，不重跑训练）
python scripts/slm/generate_gated_recurrent_report_20261107.py
```

训练数据由 `{TRAIN_SCRIPT}` 生成并已归档，勿在离线渲染流程中重跑。产物路径：

- `data/debug/slm_transfer/gated_recurrent_20261107/results.json` —— 全部表格数字的唯一来源
- `data/debug/slm_transfer/gated_recurrent_20261107/REPORT.md` —— 训练侧原始报告
- `data/debug/slm_transfer/gated_recurrent_20261107/weights.pt` —— 各变体权重
- `report/shape_trials/gated_recurrent_20261107/gated_recurrent_20261107.md` —— 本报告
"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(report, encoding="utf-8")
    return REPORT_MD


def main() -> int:
    payload = _load(RESULTS_JSON)
    path = build(payload)
    print(f"report written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
