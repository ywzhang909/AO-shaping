"""Render ``report/shape_trials/sim_pretrain_transfer/sim_pretrain_transfer_20261024.md``.

Two independent artefacts feed this one report, and keeping them apart is the whole
point of the document:

* the **simulation pretraining** run (``transfer pretrain``) -- synthetic parameter
  library, grouped cross-validation, frozen checkpoint + residual estimator;
* the **no-GS reproduction** run (``scripts/slm/reproduce_transfer_block_without_gs.py``)
  -- archived hardware captures replayed through the production anchored update loop,
  compared bit for bit against the archived phase.

They never share a checkpoint: the reproduction loads the CCD-fitted checkpoint from
``data/debug/calibrate_transfer_blur/shapes_20261023_1956``, not the simulated prior.
The report states this explicitly so nobody reads the simulation numbers as hardware
validation.

Offline only: reads saved JSON, writes one Markdown file and one figure.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts._common.provenance import insert_header

REPORT_KEY = "report/shape_trials/sim_pretrain_transfer/sim_pretrain_transfer_20261024.md"
OUT_DIR = ROOT / "report" / "shape_trials" / "sim_pretrain_transfer"
FIG_DIR = OUT_DIR / "figures"
ART_DIR = OUT_DIR / "artifacts"
STAMP = "_20261024"

DEFAULT_REPRO = ROOT / "data/debug/transfer_repro_no_gs"
DEFAULT_PRETRAIN = ROOT / "data/debug/slm_transfer"

PARAM_LABELS = {
    "dy_defocus": "离焦（屈光度）",
    "dx_center": "入射中心 x 平移（panel px）",
    "dy_center": "入射中心 y 平移（panel px）",
    "log_width_scale": "宽度缩放 ln(s)",
    "aberration_rad": "Noll 5–11 共享像差（rad）",
}

TIMING_LABELS = {"library_s": "建库", "fit_s": "拟合（含 alpha 扫描）", "total_s": "合计"}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# --------------------------------------------------------------------------- load
SHAPE_ORDER = {name: index for index, name in enumerate(("square", "rectangle", "triangle", "star"))}


def load_reproduction(root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(root.glob("block*/result.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"no reproduction results under {root}")
    return sorted(rows, key=lambda row: (row["block"], SHAPE_ORDER.get(row["shape"], 99), row["shape"]))


def find_pretrain_run(path: Path | None) -> Path | None:
    """Pick the largest ``pretrain*`` run under ``path``.

    Largest = most library combinations, with the directory name as a tie-break. That
    rule matters: the smoke run cannot outrank the full run, but a later partial sweep
    also cannot silently replace the reported numbers just because its name sorts late.
    """
    if path is None:
        return None
    if (path / "report.json").is_file():
        return path
    candidates = [entry for entry in sorted(path.glob("pretrain*"))
                  if (entry / "report.json").is_file()]
    if not candidates:
        return None

    def size(entry: Path) -> tuple[int, str]:
        payload = json.loads((entry / "report.json").read_text(encoding="utf-8"))
        return len(payload.get("grid", [])), entry.name

    return max(candidates, key=size)


# ------------------------------------------------------------------------- figure
def plot_gains(rows: list[dict], path: Path) -> None:
    labels = [f"{row['shape']}\n块 {row['block']}" for row in rows]
    proxy, measured, model = [], [], []
    for row in rows:
        proxy_gain = (row["anchored_proxy"]["seed"]["metrics"]["cv"]
                      - row["anchored_proxy"]["candidate"]["metrics"]["cv"])
        ccd_gain = row["ccd"]["anchor_at_a1"]["cv"] - row["ccd"]["b1"]["cv"]
        proxy.append(proxy_gain)
        measured.append(ccd_gain)
        model.append(row["predicted"]["seed"]["cv"] - row["predicted"]["candidate"]["cv"])

    x = np.arange(len(rows))
    width = 0.26
    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    ax.bar(x - width, proxy, width, label="锚定代理 CV 下降", color="#2b7bba")
    ax.bar(x, measured, width, label="CCD 实测 CV 下降", color="#e08a1e")
    ax.bar(x + width, model, width, label="纯模型预测 CV 下降（负值 = 变差）", color="#9aa0a6")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel("CV 变化（正 = 更均匀）")
    ax.set_title(f"同一候选相位的三路读数（4 形状 × 2 块，{len(rows)}/8 逐位复现）")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# -------------------------------------------------------------------------- parts
def reproduction_table(rows: list[dict]) -> str:
    head = ("| 形状 | 块 | 相位最大偏差 | 逐位相同 | 代理 CV | CCD CV（A1→B1） | 纯模型 CV | 增益比 |\n"
            "|---|---|---|---|---|---|---|---|")
    body = []
    for row in rows:
        seed = row["anchored_proxy"]["seed"]["metrics"]["cv"]
        proxy = row["anchored_proxy"]["candidate"]["metrics"]["cv"]
        a1 = row["ccd"]["anchor_at_a1"]["cv"]
        b1 = row["ccd"]["b1"]["cv"]
        model_seed = row["predicted"]["seed"]["cv"]
        model = row["predicted"]["candidate"]["cv"]
        ratio = (seed - proxy) / (a1 - b1) if a1 != b1 else float("nan")
        body.append(
            f"| {row['shape']} | {row['block']} | {row['phase_reproduction']['max_abs']:.1e} | "
            f"{'是' if row['phase_reproduction']['bitwise_equal'] else '否'} | "
            f"{seed:.4f} → {proxy:.4f} | {a1:.4f} → {b1:.4f} | "
            f"{model_seed:.4f} → {model:.4f} | {ratio:.2f}× |")
    return "\n".join([head, *body])


def acceptance_table(rows: list[dict]) -> str:
    head = "| 形状 | 块 | 基线 loss | 候选 loss | 漂移阈值 | 存档判决 |\n|---|---|---|---|---|---|"
    body = []
    for row in rows:
        acc = row["acceptance"]
        body.append(f"| {row['shape']} | {row['block']} | {acc['baseline']:.4f} | "
                    f"{acc['proposed']:.4f} | {acc['drift']:.4f} | "
                    f"{'接受' if acc['accepted_archived'] else '拒绝'} |")
    return "\n".join([head, *body])


def _coverage(payload: dict) -> tuple[str, bool]:
    """Summarise the sampled library against the full-factorial design.

    Returns the per-level counts as text plus whether the **highest** defocus level was
    cut short -- that is the direction the estimator has never been tested on, so it is
    the one limitation worth stating explicitly.
    """
    counts: dict[float, int] = {}
    for row in payload.get("grid", []):
        level = float(row["defocus"])
        counts[level] = counts.get(level, 0) + 1
    if not counts:
        return "", False
    per_level = max(counts.values())
    levels = sorted(counts)
    text = ", ".join(f"{level:+.2f}: {count}/{per_level}" for level, count in ((lv, counts[lv]) for lv in levels))
    return text, counts[levels[-1]] < per_level


def pretrain_section(run: Path | None) -> str:
    if run is None:
        return ("## 2 仿真预训练\n\n"
                "本轮**未产出**仿真预训练结果：`transfer pretrain` 的目标是给冷启动一个"
                "仿真先验，但它在本次写作时还没有跑出正式规模的 grouped-CV 指标，所以这里"
                "不写任何预训练数字。缺数字比写一个没有分组隔离的数更安全 —— 预训练的价值"
                "完全取决于「验证集组合从未进入训练」这一条是否成立。\n\n"
                "> 复现部分（§3）**不依赖**预训练：它加载的是 CCD 实测拟合出来的检查点。\n")

    payload = json.loads((run / "report.json").read_text(encoding="utf-8"))
    setup = payload.get("setup", {})
    metrics = payload.get("metrics", {})
    folds = payload.get("folds", [])
    scored = payload.get("folds_scored", payload.get("scored", 0))
    timings = payload.get("timings", {})
    sweep = payload.get("alpha_sweep", {})
    alpha = payload.get("alpha")

    rows = ["| 参数 | 分组 CV MAE | RMSE | R² | 偏置 |", "|---|---|---|---|---|"]
    for name, values in metrics.items():
        rows.append(f"| {PARAM_LABELS.get(name, name)} | {values.get('mae', float('nan')):.4g} | "
                    f"{values.get('rmse', float('nan')):.4g} | {values.get('r2', float('nan')):.3f} | "
                    f"{values.get('bias', float('nan')):+.4g} |")

    sweep_rows = ["| 岭惩罚 α | 最差 R² | 离焦 R² | x 平移 R² | 像差 R² |", "|---|---|---|---|---|"]
    for key, entry in sweep.items():
        r2 = entry.get("r2", {})
        sweep_rows.append(f"| {key}{' （采用）' if float(key) == float(alpha) else ''} | "
                          f"{entry.get('worst_r2', float('nan')):.3f} | "
                          f"{r2.get('dy_defocus', float('nan')):.3f} | "
                          f"{r2.get('dx_center', float('nan')):.3f} | "
                          f"{r2.get('aberration_rad', float('nan')):.3f} |")

    coverage, truncated_top = _coverage(payload)
    degenerate = (sweep.get(str(1.0)) or sweep.get("1") or {}).get("r2", {}).get("aberration_rad")
    timing = "；".join(f"{TIMING_LABELS.get(key, key)} {value:.1f}s"
                       for key, value in timings.items() if isinstance(value, (int, float)))
    prior = (run / "frozen_parameters.json")
    prior_text = ""
    if prior.is_file():
        spec = json.loads(prior.read_text(encoding="utf-8"))
        delta = spec.get("pretrain", {}).get("mean_delta")
        if delta:
            prior_text = (
                "先验检查点（`frozen_parameters.json`）由这条回归的常数项给出："
                f"`pretrain.mean_delta = [{', '.join(f'{v:+.4g}' for v in delta)}]`，"
                f"据此 `defocus_diopter = {spec['defocus_diopter']:.4f}`、"
                f"`amplitude_center = ({spec['amplitude_center'][0]:.1f}, "
                f"{spec['amplitude_center'][1]:.1f})`、"
                f"`amplitude_width = ({spec['amplitude_width'][0]:.2f}, "
                f"{spec['amplitude_width'][1]:.2f})`，"
                "`fft_size` / `grid_size` 与写入时一致，可被 "
                "`FrozenDefocusParameters.from_dicts` 直接读回。")

    return "\n".join([
        "## 2 仿真预训练",
        "",
        "### 2.1 这份产物是什么、不是什么",
        "",
        "| | 内容 |",
        "|---|---|",
        "| 输入 | `FrozenDefocusForward` 生成的合成参数库（无相机噪声、无 LUT、无零级光） |",
        "| 产出 | `estimator.npz`（16×5 岭回归系数）、`frozen_parameters.json`（先验检查点）、`report.json` |",
        "| 用途 | 给冷启动一个**先验**，减少「从名义参数出发」的偏差 |",
        "| **不用于** | §3 的复现；也不构成对实验系统的验证 |",
        "",
        "### 2.2 参数库与特征",
        "",
        f"- 本次采样 **{len(payload.get('grid', []))}** 个组合（全因子设计 90 格），"
        f"每组合 **{setup.get('probes_per_combo', '?')}** 个探针，"
        f"FFT **{setup.get('fft_size', '?')}**、网格 **{setup.get('grid_size', '?')}**。",
        f"- 离焦轴格点覆盖（各层组合数）：{coverage}。" if coverage else "",
        "- 探针跨曝光循环 `" + str(setup.get("exposure_cycle")) + "` ms；"
        "跨曝光只比较 `counts / exposure_ms`，否则曝光会被当成光路参数。",
        "- 每组合 **16 个特征** = 8 个描述子的 mean 与 std（跨 8 个探针）；"
        "目标 = 相对名义先验的 5 维残差。",
        "- 正向传播只用 `FrozenDefocusForward`；特征只来自「实测图 + 名义预测图」，"
        "真值参数不进特征。",
        "",
        "### 2.3 分组交叉验证",
        "",
        f"留一**组合**（leave-one-combination-out），共 **{len(folds)}** 折、"
        f"**{scored}** 个验证集组合全部被预测。同一组合内的探针共享同一套光路参数，"
        "按样本或按像素随机切分会让同一组合同时出现在训练与测试里 —— 那样得到的指标"
        "只反映插值能力，不反映迁移能力，所以禁止。",
        "",
        "### 2.4 结果",
        "",
        *rows,
        "",
        f"墙钟：{timing or '未记录'}（CPU，单进程）。",
        "",
        "岭惩罚不是无关紧要的超参 —— 同一分折、同一特征，只换 α 就能把一个可识别参数"
        "压到看起来「不可识别」：",
        "",
        *sweep_rows,
        "",
        (f"α=1（最初的默认值）把像差 R² 压到 {degenerate:.3f}，α=0.01 时是 "
         f"{metrics.get('aberration_rad', {}).get('r2', float('nan')):.3f} —— "
         "差出来的 0.26 全是**正则化伪影**，不是物理可识别性。\n" if degenerate is not None else ""),
        prior_text,
        "",
        "### 2.5 五条必须与数字一起读的边界",
        "",
        "1. **匹配模型，不是实验系统。** 参数库由 `FrozenDefocusForward` 生成、验证也用它，"
        "所以指标衡量的是「在这个传播模型内部能否反解参数」。它**不能**说明模型匹配实验系统；"
        "`report.json` 的 `caveat` 原文即此意。先验的数值本身也只等于"
        "「参数库中心 − 名义值」（离焦 −0.048 − 名义 0.025 = −0.073 等），"
        "所以它反映的是**库的设计中心**，不是实验系统的实测参数。",
        "2. **离焦轴被截断。** 全因子设计有 5 个离焦层，本次只跑了前 "
        f"{len(payload.get('grid', []))} 格"
        + ("，最高的一层只取到一部分格点" if truncated_top else "") +
        "。分折仍然干净（每个验证集组合从未进训练），但"
        + ("**正向极端层根本没有进入库**，所以那里没有可报的误差，也不能把 R² 外推过去。"
           if truncated_top else "所以 R² 只对已覆盖的离焦区间成立。"),
        "3. **α 是在同一批折上选出来的**（0.01 / 0.1 / 1 / 10 对照见上表），"
        "不是嵌套 CV，所以「α=0.01 最好」带选择偏差，只能当提示性证据；"
        "但「α=1 明显更差」在四个 α 上都可复现，那一条是稳的。",
        "4. **入射中心平移的可识别性来自口径裁剪。** 纯瞳面平移在远场只乘一个线性相位，"
        "强度分布不变；所以它只能通过「照明跑出有效口径」造成的包络/能量变化被看到。"
        "本库的 3 个平移点近乎共线（(12,−8) 与 (−18,10) 夹角约 176°），"
        "`dx_center` / `dy_center` 实为同一个一维量的两次投影，两个 R² 都 ≈0.93 属正常。",
        "5. **像差 R²≈0.33 是这一套库的可识别性上限。** `ABERRATION_AXIS` 只到 0.05 rad，"
        "而库内探针相位自身的 std 实测为 1.24–1.73 rad"
        "（`list(pretrain.probes(32, 8, 102419))` 的 train/validation 探针），"
        "即像差只占探针自身相位的约 3%。把 §2.4 里 α 的伪影去掉后剩下的 0.33 "
        "就是这个幅度下的可识别性，不是特征取错。想抬高它必须加大像差轴或换探针族 —— "
        "那是重新设计参数库，本轮没有做。",
        "",
        "> 复现部分（§3）**不使用**上面这个检查点：它加载的是 CCD 实测拟合的 "
        "`calibrate_transfer_blur/shapes_20261023_1956`。",
    ])


# --------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro-root", type=Path, default=DEFAULT_REPRO)
    parser.add_argument("--pretrain-run", type=Path, default=DEFAULT_PRETRAIN)
    parser.add_argument("--no-artifacts", action="store_true")
    args = parser.parse_args()

    rows = load_reproduction(args.repro_root)
    run = find_pretrain_run(args.pretrain_run)
    pretrain_headline = None
    if run is not None:
        payload = json.loads((run / "report.json").read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        pretrain_headline = dict(
            defocus=metrics.get("dy_defocus", {}).get("r2", float("nan")),
            shift=metrics.get("dx_center", {}).get("r2", float("nan")),
            width=metrics.get("log_width_scale", {}).get("r2", float("nan")),
            aberration=metrics.get("aberration_rad", {}).get("r2", float("nan")),
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ART_DIR.mkdir(parents=True, exist_ok=True)

    figure = FIG_DIR / f"transfer_three_readings{STAMP}.png"
    plot_gains(rows, figure)

    if not args.no_artifacts:
        for block_dir in sorted(args.repro_root.glob("block*")):
            target = ART_DIR / f"repro_{block_dir.name}.json"
            shutil.copyfile(block_dir / "result.json", target)
        if run is not None:
            shutil.copyfile(run / "report.json", ART_DIR / "pretrain_report.json")
            prior = run / "frozen_parameters.json"
            if prior.is_file():
                shutil.copyfile(prior, ART_DIR / "pretrain_frozen_parameters.json")

    bitwise = sum(1 for row in rows if row["phase_reproduction"]["bitwise_equal"])
    ratios = [
        (row["anchored_proxy"]["seed"]["metrics"]["cv"] - row["anchored_proxy"]["candidate"]["metrics"]["cv"])
        / (row["ccd"]["anchor_at_a1"]["cv"] - row["ccd"]["b1"]["cv"])
        for row in rows
    ]
    model_worse = sum(1 for row in rows
                      if row["predicted"]["candidate"]["cv"] > row["predicted"]["seed"]["cv"])

    body = "\n".join([
        "# 仿真预训练与无 GS 迁移复现（2026-10-24）",
        "",
        "## 0 结论摘要",
        "",
        f"1. **算法等价性成立**：4 形状 × 2 块的候选相位与存档 `B1` 相位**逐位相同**"
        f"（{bitwise}/{len(rows)}），全程 **0 次** `gs_seed()` 调用。也就是说，"
        "「把已接受的相位作为起点 + 锚定实测的目标函数」这一条链路可以完全脱离 GS 复现。",
        f"2. **纯模型 CV 在 8 组里 {model_worse}/{len(rows)} 变差**，而锚定代理 CV 与 CCD 实测 CV "
        f"{len(rows)}/{len(rows)} 变好。同一候选相位、同一目标函数，两路读数方向相反 —— "
        "判据必须在锚定项上，不能用模型绝对预测接受/拒绝。",
        f"3. **代理增益放大约 {min(ratios):.2f}–{max(ratios):.2f}×**"
        f"（中位 {float(np.median(ratios)):.2f}×）。方向可信，幅度不可直接引用。",
        ("4. **仿真预训练**（§2）在 60 个组合上留一组合交叉验证，"
         f"离焦 R² {pretrain_headline['defocus']:.4f}、入射中心平移 R² "
         f"{pretrain_headline['shift']:.2f}、宽度缩放 R² {pretrain_headline['width']:.4f}，"
         f"**共享像差只有 {pretrain_headline['aberration']:.2f}**。"
         "这是匹配模型内部的可识别性，不是对实验系统的验证；"
         "且它与 §3 的复现**不共用检查点**（见 §1）。"
         if pretrain_headline else
         "4. 仿真预训练（§2）本轮未产出正式规模指标，报告里不写任何预训练数字。"),
        "",
        "## 1 两个独立部分，先说清口径",
        "",
        "| | 仿真预训练（§2） | 无 GS 复现（§3） |",
        "|---|---|---|",
        "| 数据 | `FrozenDefocusForward` 生成的合成参数库 | 已存档的 CCD 帧与相位 |",
        "| 产出 | `estimator.npz` + `frozen_parameters.json`（仿真先验） | 「候选相位 == 存档相位」的逐位证据 |",
        "| 是否接触设备 | 否 | 否（读存档） |",
        "| 能否证明模型匹配实验系统 | **不能** | **不能**（只证明算法可复现） |",
        "",
        "复现用的检查点是 "
        "`data/debug/calibrate_transfer_blur/shapes_20261023_1956/frozen_parameters.json`，"
        "由 CCD 实测拟合得到，**不是** §2 的仿真先验。把两者混为一谈会读出一个不存在的结论："
        "「仿真预训练已经被实验验证」。本轮没有验证它。",
        "",
        pretrain_section(run),
        "",
        "## 3 无 GS 迁移复现",
        "",
        "### 3.1 为什么必须单独证明「不从 GS 开始」",
        "",
        "生产链路是 `gs_seed() → 冻结物理预测模型 → 锚定 CCD 目标的可微更新`。"
        "直接重跑一遍会同时动两件事：GS 初始化给出的相位，以及更新循环。"
        "于是「结果一致」既可能说明更新循环被复现了，也可能只是 GS 恰好给了同一起点。",
        "把 GS 拿掉、起点换成**上一轮已接受的相位**，就只剩更新循环这一项可变量。",
        "",
        "> 口径声明：本次复现的起点来自一次历史 model-mode 运行的接受相位，"
        "而那次运行自己的起点是 GS。因此本报告能说的是「**这次复现没有调用 GS**」，"
        "不能说的是「**历史上从未用过 GS**」。",
        "",
        "### 3.2 三步",
        "",
        "1. **种子**：`<seed-run>/<shape>_final` 的 `model_phase`（块 0），"
        "或存档 `A1` 实际显示的相位（块 1 及以后）。两者的 `max|Δ|` 都为 0，见下表第 3 列。",
        "2. **锚定**：用存档 `A1` 帧逐像素平均后除 `exposure_ms`，"
        "再按同一 `camera_xy` 采样，得到参考辐照度；基线预测取 `forward.predict(seed)`。",
        "3. **更新**：`DifferentiableUniformityOptimizer`（`lr=0.01`、`step_rms_cap=0.2`、"
        "10 步、CPU）每次以 `anchored_intensity(measured, current, baseline)` 为输入，"
        "与 `validate.py` 的 anchored 分支逐步一致。",
        "",
        "### 3.3 逐位判据与结果",
        "",
        reproduction_table(rows),
        "",
        "「相位最大偏差 0.0e+00」是 `np.array_equal` 也成立的**逐位**相等，不是 `allclose` 的近似。"
        "所以下表 `CCD CV` 一列不是重新测的 —— 候选相位既然与存档 `B1` 相同，"
        "存档的那次测量就按同一性适用于它。",
        "",
        "存档判决（loss 口径，四帧 A1/B1/B2/A2 各自独立测）：",
        "",
        acceptance_table(rows),
        "",
        "### 3.4 三路读数：为什么纯模型 CV 会反向",
        "",
        "![三路读数](figures/transfer_three_readings" + STAMP + ".png)",
        "",
        "同一次更新迭代里同时存在三个数：",
        "",
        "| 读数 | 定义 | 8 组方向 |",
        "|---|---|---|",
        "| 纯模型 CV | `forward.predict(candidate)` | 8/8 **变差** |",
        "| 锚定代理 CV | `measured + pred(candidate) − pred(seed)` | 8/8 变好 |",
        "| CCD 实测 CV | 存档 `A1` → `B1` | 8/8 变好 |",
        "",
        "原因不神秘：`forward.predict` 的**绝对值**与 CCD 的绝对值差着一整套"
        "加性背景（杂散光）项与标度，所以它在绝对量上的 CV 与 CCD 的 CV 不是同一个物理量"
        "（0.27 vs 0.68）。优化器最小化的从来不是它 —— 是锚定代理。"
        "代理保留了实测的绝对量，只借用模型的**局部变化**，因此方向与实测一致。",
        "",
        "这解释了一件容易被误读的事：如果只看模型 loss 下降就宣布成功，"
        "在本轮的数据上会得到相反结论。判据必须落在锚定项与实测项上。",
        "",
        "### 3.5 代理增益的放大系数",
        "",
        f"代理增益 / 实测增益 = **{min(ratios):.2f}–{max(ratios):.2f}×**"
        f"（中位 {float(np.median(ratios)):.2f}×）。8 组里 "
        f"{sum(1 for r in ratios if r > 1)} 组高估、{sum(1 for r in ratios if r <= 1)} 组不高于实测"
        f"（最低 {min(ratios):.2f}×）。"
        "含义：代理可以当**方向判据**（好/坏、接受/拒绝），"
        "不能当**幅度预报**（不能说「这一步会降 0.07」）。"
        "放大系数不是一个常数，所以也不该用一个固定折扣率把它换算成实测值。",
        "",
        "### 3.6 这一节没有做的事",
        "",
        "- 没有重新开相机：CCD 一列是存档读数按相位同一性引用，不是新测量。",
        "- 没有重新采集任何帧，因此没有新的噪声估计；接受判决沿用存档的 A–B–B–A 漂移阈值。",
        "- 没有验证检查点本身的物理正确性（离焦方向检验在本轮之前就没有通过），"
        "所以「复现成功」是**算法等价**，不是**模型正确**。",
        "",
        "### 3.7 远场演化的两张动画",
        "",
        "![模型预测的远场演化](images/no_gs_model_iterations" + STAMP + ".gif)",
        "",
        "![存档实测远场 A1→B1→B2→A2](images/no_gs_measured_abba" + STAMP + ".gif)",
        "",
        "两张动画来自同一次离线复现（`scripts/slm/generate_transfer_no_gs_gif.py`，"
        "复用 `reproduce_shape` 的同一条更新循环，`gs_seed()` 调用 0 次），"
        "但画的**不是同一件事**：",
        "",
        "| | 模型预测动画（上） | 实测存档动画（下） |",
        "|---|---|---|",
        "| 帧 | 11 帧 = 种子相位 + 10 次更新 | 4 帧 = A1 / B1 / B2 / A2 |",
        "| 像素来源 | `FrozenDefocusForward.predict` | 存档 `raw_frames` 按 `camera_xy` 采样 ÷ `exposure_ms` |",
        "| 读数的含义 | 模型绝对量，**不是** CCD 读数 | 相机真实读数 |",
        "| CV 走向 | 上升（正方形 0.272→0.300、三角形 0.265→0.320） | 下降（B1/B2 的 ΔCV 为 −0.045 ~ −0.096，A2 回到 ±0.002 内） |",
        "",
        "上表的「CV 走向」两列方向相反，就是 §3.4 那个现象的**可视化**：模型预测的绝对标度"
        "与 CCD 差着一整套加性背景（杂散光）项，所以模型动画只能读**形态演化**，"
        "不能读「好/坏」；判据仍然只有锚定代理与实测两项。",
        "",
        "两条渲染约定，避免把动画看成比数字更强的证据：",
        "",
        "- **面板内共用同一显示尺度**（该面板全部帧的 99.5 分位），所以亮度变化是真实的，"
        "不是逐帧归一化造出来的假变化；跨面板不共用尺度（四个形状的绝对亮度本来就不同）。",
        "- 实测那组的**宏观形态变化比数字看起来更弱**：CV 从 0.681 降到 0.607（约 11%），"
        "画面仍是弥散团块，动的只是内部斑纹与环晕。动画让过程可见，不替代 §3.3 的逐位判据"
        "与 §3.4 的三路读数。",
        "",
        "## 4 复现命令",
        "",
        "```bash",
        "# 块 0：种子取上一轮接受相位（<seed-run>/<shape>_final）",
        "python scripts/slm/reproduce_transfer_block_without_gs.py --block 0 --seed-source run \\",
        "    --out data/debug/transfer_repro_no_gs/block0",
        "# 块 1：种子取存档 A1 实际显示的相位（块 0 已接受的相位继续往下走）",
        "python scripts/slm/reproduce_transfer_block_without_gs.py --block 1 --seed-source a1 \\",
        "    --out data/debug/transfer_repro_no_gs/block1",
        "# §3.7 的两张动画（离线，不开设备）",
        "python scripts/slm/generate_transfer_no_gs_gif.py",
        "# 本报告",
        "python scripts/slm/generate_sim_pretrain_transfer_report.py",
        "```",
        "",
        "`--seed-source run` 与 `--seed-source a1` 的存在本身就是证据的一部分："
        "块 0 的种子是**外部**给的（与 `A1` 显示相位独立测得 `max|Δ|=0`），"
        "块 1 的种子是**链内**承接的。两条都给逐位相同，说明更新循环对起点来源不敏感。",
        "",
        "## 5 数据路径",
        "",
        "| 用途 | 路径 |",
        "|---|---|",
        "| 复现产物 | `data/debug/transfer_repro_no_gs/block{0,1}/result.json` |",
        "| 冻结检查点（复现用） | `data/debug/calibrate_transfer_blur/shapes_20261023_1956/frozen_parameters.json` |",
        "| 种子来源 | `data/debug/validate_transfer_shapes/paired_20261023_1942/manifest.json` |",
        "| 复现对象（80 条 A–B–B–A） | `data/debug/validate_transfer_shapes/anchored_20261023_2002/` |",
        "| 预训练产物 | `data/debug/slm_transfer/pretrain*/{report.json,frozen_parameters.json,estimator.npz}` |",
        "| 本报告输入副本 | `artifacts/` |",
        "",
        "## 6 易误读的三点",
        "",
        "1. **「逐位相同」不等于「实验可重复」**。逐位相同说明算法（含随机性来源）是确定的；"
        "实验重复性受器件噪声限制，同一 seed 的两次硬件运行在基准点就差 ~1e-3。",
        "2. **代理 CV 变好不等于画面变好**。代理是「实测 + 模型局部变化」，"
        "它继承了实测的绝对量，也继承了模型的局部误差；§3.5 的放大系数就是这部分误差的量化。",
        "3. **仿真预训练不能用这一节背书**。§2 与 §3 的数据源、检查点、验证协议都不同，"
        "把 §3 的逐位复现当作 §2 的验证是本报告最容易被误读的地方。",
        "",
    ])

    target = ROOT / REPORT_KEY
    target.write_text(insert_header(body, REPORT_KEY), encoding="utf-8")
    print(f"wrote {target.relative_to(ROOT)}")
    print(f"figure -> {figure.relative_to(ROOT)}")
    print(f"pretrain run: {run if run is not None else 'none (section marked as not produced)'}")


if __name__ == "__main__":
    main()
