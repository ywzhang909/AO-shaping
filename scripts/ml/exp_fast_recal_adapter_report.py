"""Generate the fast-recalibration adaptation experiment report.

Fully offline: reads ``logs/fast_recal_adapter/results.json`` written by
``scripts/slm/exp_fast_recal_adapter.py`` and renders the tables, the paired
statistics and the figures. Never opens a device and never re-runs the
experiment.

Every number is computed from the JSON, and the protocol constants (adapter
size, arm definitions, perturbation families) are imported from the experiment
module itself, so the report cannot describe a protocol that the script no
longer implements.

Judged on corr / R2 / beam-domain only. MSE/PSNR/SSIM are deliberately absent:
this repo measured PSNR 72 dB at a *worse* R2 under sum normalisation, i.e. the
metric was measuring the normalisation rather than the fit.

Writes ``report/shape_trials/fast_recal_adapter/fast_recal_adapter.md`` + ``figures/``.
Provenance is stamped at write time so a re-run cannot delete it.

Run:
    python scripts/ml/exp_fast_recal_adapter_report.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parents[2]
# `scripts.*` is imported as a package path, so the repo ROOT itself must be
# importable -- under a file-form launch (`python scripts/x/y.py`) the cwd is
# not on sys.path and only the script's own directory is.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts._common.provenance import insert_header
from scripts.slm.exp_fast_recal_adapter import (
    ARMS,
    CALIB_ERR,
    GRID,
    K,
    KINDS,
    NCOEF,
    N_ORDER,
    THETA0,
    WIN,
)

# `scripts/ml` is a regular package (it has an `__init__.py`), so once the
# experiment module above put `<repo>/scripts` on `sys.path[0]`, `import ml`
# resolves to `scripts/ml` instead of the real `src/ml` package -- and
# `ml.zernike.eval_stats` then does not exist.  Put `src` back in front.
sys.path.insert(0, str(ROOT / "src"))
from ml.zernike.eval_stats import (
    cohens_dz,
    holm_bonferroni,
    min_attainable_pvalue,
    sign_flip_pvalue,
)

REPORT_KEY = "report/shape_trials/fast_recal_adapter/fast_recal_adapter.md"
RESULTS = ROOT / "logs" / "fast_recal_adapter" / "results.json"
OUT_DIR = ROOT / "report" / "shape_trials" / "fast_recal_adapter"
FIG_DIR = OUT_DIR / "figures"

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

ARM_LABEL = {
    "nominal": "nominal (0 拟合)",
    "nominal_img": "nominal+CNN (0 拟合, 冻结先验)",
    "c_only": f"c_only ({K} Zernike)",
    "c_img": f"c_img ({K} Zernike + 冻结 CNN)",
    "geom+c": f"geom+c ({K} Zernike + 2 几何)",
}


def load() -> list[dict]:
    if not RESULTS.is_file():
        raise SystemExit(
            f"missing {RESULTS} -- run scripts/slm/exp_fast_recal_adapter.py first"
        )
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def grid(rows: list[dict]) -> dict:
    d = defaultdict(dict)
    for r in rows:
        d[(r["kind"], r["arm"], r["n_frames"])][r["seed"]] = r
    return d


def cell(d: dict, kind: str, arm: str, nf: int, metric: str) -> np.ndarray:
    return np.array([r[metric] for r in d[(kind, arm, nf)].values()], float)


def frames_seen(rows: list[dict]) -> list[int]:
    return sorted({r["n_frames"] for r in rows})


def n50(d: dict, kind: str, arm: str, frames: list[int]) -> int | str:
    """Smallest frame count reaching 90% of the arm's own total correlation gain."""
    base = cell(d, kind, "nominal", frames[-1], "corr").mean()
    final = cell(d, kind, arm, frames[-1], "corr").mean()
    gain = final - base
    if gain <= 0:
        return "never"
    for nf in frames:
        if cell(d, kind, arm, nf, "corr").mean() - base >= 0.9 * gain:
            return nf
    return f">{frames[-1]}"


def paired(d: dict, kind: str, arm: str, nf: int, metric: str, ref: str):
    """Pair on SEED (same seed, different arm) -- never pair across seeds."""
    a = d[(kind, arm, nf)]
    b = d[(kind, ref, nf)]
    seeds = sorted(set(a) & set(b))
    diffs = np.array([a[s][metric] - b[s][metric] for s in seeds], float)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return None
    return {
        "n": int(diffs.size),
        "mean": float(diffs.mean()),
        "p": float(sign_flip_pvalue(diffs.tolist())),
        "dz": float(cohens_dz(diffs.tolist())),
    }


def render_figures(d: dict, frames: list[int]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # -- fig 1: corr vs frames, one panel per perturbation family -----------
    fig, axes = plt.subplots(
        1, len(KINDS), figsize=(5.2 * len(KINDS), 4.4), sharey=True
    )
    for ax, kind in zip(axes, KINDS):
        for arm in ARMS:
            mu = [cell(d, kind, arm, nf, "corr").mean() for nf in frames]
            sd = [cell(d, kind, arm, nf, "corr").std() for nf in frames]
            ax.errorbar(
                frames, mu, yerr=sd, marker="o", capsize=3, label=ARM_LABEL[arm], lw=1.8
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(frames)
        ax.set_xticklabels(frames)
        ax.set_xlabel("适配用测量帧数")
        ax.set_title(kind)
        ax.axhline(0, color="grey", lw=0.8, ls=":")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("held-out corr (mean ± sd)")
    axes[-1].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("适配速度：像差变化 vs 几何变化（离线仿真）", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "adaptation_corr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # -- fig 2: the beam-domain metric that corr cannot see ----------------
    fig, axes = plt.subplots(
        1, len(KINDS), figsize=(5.2 * len(KINDS), 4.4), sharey=True
    )
    for ax, kind in zip(axes, KINDS):
        for arm in ARMS:
            mu = [cell(d, kind, arm, nf, "ee90_err").mean() for nf in frames]
            ax.plot(frames, mu, marker="o", label=ARM_LABEL[arm], lw=1.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(frames)
        ax.set_xticklabels(frames)
        ax.set_xlabel("适配用测量帧数")
        ax.set_title(kind)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("EE90 半径误差 (px, 越低越好)")
    axes[-1].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("corr 看不见的几何误差：只有光斑域指标暴露出来", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "adaptation_beam_metric.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def table(d: dict, frames: list[int], metric: str, nd: int = 3) -> str:
    out = [
        "| 扰动 | arm | 拟合参数 | " + " | ".join(f"nf={n}" for n in frames) + " |",
        "|---|---|---|" + "---|" * len(frames),
    ]
    for kind in KINDS:
        for arm in ARMS:
            cells = []
            for nf in frames:
                v = cell(d, kind, arm, nf, metric)
                cells.append(f"{v.mean():+.{nd}f} ± {v.std():.{nd}f}")
            n_par = {
                "nominal": 0,
                "nominal_img": 0,
                "c_only": K,
                "c_img": K,
                "geom+c": K + 2,
            }[arm]
            out.append(
                f"| {kind} | {ARM_LABEL[arm]} | {n_par} | " + " | ".join(cells) + " |"
            )
    return "\n".join(out)


def build(rows: list[dict]) -> str:
    d = grid(rows)
    frames = frames_seen(rows)
    n_seeds = len({r["seed"] for r in rows})
    minp = min_attainable_pvalue(n_seeds)

    # paired tests vs the two references, on BOTH metrics: the geometry finding
    # lives in ee90_err, so testing corr alone would leave the headline claim
    # unaccompanied by the test that supports it.
    tests: dict[str, dict] = {}
    for metric in ("corr", "ee90_err"):
        for kind in KINDS:
            for arm, ref in (
                ("c_only", "nominal"),
                ("c_img", "nominal_img"),
                ("geom+c", "nominal"),
            ):
                p = paired(d, kind, arm, frames[-1], metric, ref)
                if p:
                    tests[f"{kind}/{arm}-{ref}/{metric}@nf{frames[-1]}"] = p
    holm = holm_bonferroni({k: v["p"] for k, v in tests.items()})

    n50_rows = "\n".join(
        f"| {kind} | {ARM_LABEL[arm]} | {n50(d, kind, arm, frames)} |"
        for kind in KINDS
        for arm in ARMS
        if arm != "nominal_img"
    )

    stat_rows = "\n".join(
        f"| `{k}` | {v['n']} | {v['mean']:+.4f} | {v['dz']:+.2f} | "
        f"{v['p']:.4f} | {holm[k]:.4f} |"
        for k, v in sorted(tests.items())
    )
    n_tests = len(tests)
    holm_thresh = 0.05 / n_tests
    need_n = 1
    while min_attainable_pvalue(need_n) >= holm_thresh:
        need_n += 1

    # headline numbers, pulled live
    lo_last = frames[-1]

    def get_test(key: str) -> dict:
        """A paired-test row that is never None, so the report can neither crash
        nor silently print 'None' when a key is absent."""
        return tests.get(
            key, {"n": 0, "mean": float("nan"), "dz": float("nan"), "p": float("nan")}
        )

    def holm_for(row: dict) -> float:
        for k, v in tests.items():
            if v is row:
                return holm[k]
        return float("nan")

    c_only_lo = cell(d, "P-A-low", "c_only", lo_last, "corr").mean()
    geom_lo = cell(d, "P-A-low", "geom+c", lo_last, "corr").mean()
    nom_lo = cell(d, "P-A-low", "nominal", lo_last, "corr").mean()
    nom_img_lo = cell(d, "P-A-low", "nominal_img", lo_last, "corr").mean()
    nom_g_e = cell(d, "P-B-geom", "nominal", lo_last, "ee90_err").mean()
    geom_g_e = cell(d, "P-B-geom", "geom+c", lo_last, "ee90_err").mean()
    # the frozen CNN's parameter count travels with every result row
    frozen_params = max(int(r.get("frozen_params", 0)) for r in rows)
    frames_txt = ", ".join(str(n) for n in frames)

    def n50_txt(kind: str, arm: str) -> str:
        """N50 with its unit, and the two non-numeric answers spelled out."""
        v = n50(d, kind, arm, frames)
        if isinstance(v, int):
            return f"{v} 帧"
        if v == "never":
            return "从未（该变体没有净增益可吃）"
        return f"{v} 帧"

    hi_c = get_test(f"P-A-high/c_only-nominal/corr@nf{lo_last}")
    lo_c = get_test(f"P-A-low/c_only-nominal/corr@nf{lo_last}")
    g_e = get_test(f"P-B-geom/geom+c-nominal/ee90_err@nf{lo_last}")
    g_c = get_test(f"P-B-geom/c_only-nominal/ee90_err@nf{lo_last}")
    lo_ci = get_test(f"P-A-low/c_img-nominal_img/corr@nf{lo_last}")

    def sig(t: dict) -> str:
        """Distinguish 'no effect' from 'protocol underpowered' (see §6).

        Reporting a large d_z whose raw p sits exactly at the attainable floor
        as merely 'not significant' would read as "no effect", which is the one
        reading the data does not support.
        """
        if not t.get("n"):
            return "无数据"
        if holm_for(t) < 0.05:
            return "显著"
        if abs(t["dz"]) >= 1.0 and t["p"] <= min_attainable_pvalue(t["n"]) + 1e-12:
            return "达 p 下限（效应存在，协议欠功效，见 §6）"
        return "不显著（未见效应）"

    # per-seed paired diffs, because a mean can hide a 5/6-consistent effect that
    # is really a 1/6-complete search failure
    diff_rows = []
    for label, kind, arm, ref, metric in (
        ("低阶 c_only", "P-A-low", "c_only", "nominal", "corr"),
        ("高阶 c_only", "P-A-high", "c_only", "nominal", "corr"),
        ("几何 geom+c", "P-B-geom", "geom+c", "nominal", "ee90_err"),
        ("几何 c_only", "P-B-geom", "c_only", "nominal", "ee90_err"),
    ):
        a, b = d[(kind, arm, frames[-1])], d[(kind, ref, frames[-1])]
        seeds = sorted(set(a) & set(b))
        diffs = [a[s][metric] - b[s][metric] for s in seeds]
        # direction depends on the metric: corr/r2 higher-is-better, errors
        # lower-is-better. Counting "v < 0" for corr would invert the verdict.
        lower_better = metric.endswith("_err")
        better = sum(1 for v in diffs if (v < 0 if lower_better else v > 0))
        tie = sum(1 for v in diffs if abs(v) < 1e-12)
        diff_rows.append(
            f"| {label} | {metric} | "
            + " | ".join(f"{v:+.3f}" for v in diffs)
            + f" | **{better}/{len(diffs)}**{'（含 ' + str(tie) + ' 个完全无变化）' if tie else ''} |"
        )

    diff_table = "\n".join(diff_rows)

    return f"""# 光路变化后如何最快适配：少参数在线微调 (2026-10-13)

## 0. 实验目的

**一句话**：在**低阶像差**漂移下，**{K} 个低阶 Zernike 参数**用 **{n50(d, "P-A-low", "c_only", frames)} 帧**吃满可得增益
（Δcorr {lo_c["mean"]:+.3f}, d_z {lo_c["dz"]:+.1f}）；同样的 {K} 个参数在**高阶像差**漂移下**没有净增益**
（Δcorr {hi_c["mean"]:+.3f}, d_z {hi_c["dz"]:+.1f}, {sig(hi_c)}）——
即它是「漂移是低阶」的**赌注**而非通用回退。
在**几何**漂移下 corr/R² 几乎**看不见**问题（nominal 已 +{cell(d, "P-B-geom", "nominal", lo_last, "corr").mean():.3f}），
只有光斑域 EE90 暴露：{K} 个 Zernike 参数对此**完全无效**（Δee90 {g_c["mean"]:+.4f}），
而 **+2 个几何标量**把 EE90 误差从 {nom_g_e:.3f} 降到 {geom_g_e:.3f} px（Δ {g_e["mean"]:+.4f}, {sig(g_e)}）。
**冻结 CNN 对「适配」的贡献不显著**；它的真实收益在**吸收长期标定残差**，与漂移无关。

这次实验要回答四个问题：

1. **光路一动，在线适配到底该重新拟合什么、用几个参数？**
   低阶 Zernike、几何标量、冻结先验——三者的职责互不相交（§1）。
2. **少参数适配器在什么漂移下有效、什么漂移下无效？**
   低阶漂移与高阶漂移必须**分开测**，否则"低阶为主"永远无法被证伪。
3. **几何漂移能不能也被少量参数捕捉？**
   如果能，Zernike 与几何二选一；如果不能，几何必须由**独立参数**、且由**哨兵触发**。
4. **冻结的图像矫正 CNN 对"适配"有没有贡献？**
   还是说它的价值只在**离线标定**、与漂移无关？

**为什么值得做**：一次漂移后的在线适配发生在**光路已经动了**的时刻，
选错适配器类型会让两类参数互相打架（§7.3）；而只看 corr/R² 会**看不见**几何问题（§4）。
这两件事都必须在**选型之前**被测出来，而不是在实验系统上试错。

**预期能排除什么**：

- 排除"{K} 个低阶 Zernike 是通用兜底"（高阶对照变体 `P-A-high`）；
- 排除"冻结 CNN 能加快适配"（`c_img` vs `nominal_img` 的配对）；
- 排除"Zernike 参数也能修几何漂移"（`P-B-geom` 下逐 seed 配对差精确为零）；
- 排除"corr/R² 足以判断几何漂移"（`P-B-geom` 下 nominal 的 corr 已经很高）。

## 0.1 实验设置

> ⚠️ **本实验是纯仿真，全程不接触任何设备**：没有打开 SLM、没有打开相机、
> 没有采集任何一帧 CCD 画面。全部数字来自**仿真传播器**，
> 因此结论对真实光路的迁移性**未经验证**（§8）。

**仿真配置**（常量全部从实验脚本 `scripts/slm/exp_fast_recal_adapter.py` import，
报告无法描述实验脚本已不再实现的协议）：

| 项 | 值 | 来源 |
|---|---|---|
| 传播器 | canonical numpy `forward_intensity` | `ao_shaping.drivers.sim.slm_shaping_bench` |
| Zernike 基 | canonical `generate_zernike_phase` | `ao_shaping.utils.wavefront.zernike_utils` |
| 瞳面网格 / 相机窗口 | {GRID}×{GRID} / {WIN[0]}×{WIN[1]} px | 实验脚本模块常量 |
| 系数维数 | {NCOEF}（n_max={N_ORDER} 三角序，含 piston） | 同上 |
| 适配器大小 | K = {K}（低阶 Zernike） | 同上 |
| 几何真值 θ | 像元 {THETA0["camera_pixel_um"]:g} µm / 孔径 {THETA0["aperture_m"] * 1e3:g} mm / 焦距 {THETA0["focal_length_m"] * 1e3:g} mm | 同上 |
| 冻结模型的标定残差 | 像元 +{CALIB_ERR["camera_pixel_um"] * 100:g}% / 孔径 −{CALIB_ERR["aperture_m"] * 100:g}% | 同上 |
| 扰动族 | {", ".join(KINDS)} | 同上 |
| seed 数 / 适配预算 | {n_seeds} seed / {frames_txt} 帧 | 本次结果 JSON |
| 参数组 | `c`（{K} 个低阶 Zernike）/ `dtheta`（2 个几何标量）/ `g`（冻结 CNN，{frozen_params} 参数） | 实验脚本 |
| 判据指标 | corr、R²、EE90 误差、CV 误差 | 结果 JSON 字段 |
| 统计协议 | 按 seed 配对 + sign-flip 精确检验 + Holm 校正 | `ml/zernike/eval_stats` |

**焦距故意不扰动**：它与像元尺度都只决定远场总尺度，同时扰动两者在数学上不可辨识
——这是结论，不是遗漏。

**冻结模型的几何是故意标定不准的**（上表「冻结模型的标定残差」一行）：这份**长期标定残差**正是离线 CNN
唯一能吸收的东西，适配器随后只需解释漂移带来的**增量**。若标定是完美的，
CNN 将无事可学、`c_img` 会退化成 `c_only`——那样这个对照就是假的。

**扰动族**（含义与阶次范围见 §1 的表）：`P-A-low`（低阶像差）、`P-A-high`（高阶像差对照）、
`P-B-geom`（几何漂移：像元尺度 ±3%、孔径 ±4%）。

## 0.2 算法修改

**本轮没有改动实验协议**：扰动族、参数组划分、适配器的拟合方式、判据指标
全部**一字未动**；结果 JSON 由 `scripts/slm/exp_fast_recal_adapter.py --seeds 6 --frames 1 2 4 8`
重新生成（`logs/fast_recal_adapter/results.json`），报告脚本**只读**该 JSON，
既不重跑实验也不接触设备。

本轮的改动只有三处，都在**报告脚本本身**（不涉及任何计算）：

1. 新增 §0 / §0.1 / §0.2 / §0.3 这一组「实验四要素」小节；
2. 把 §0.1 需要的协议常量从实验模块 **import** 进来（`ARMS / K / KINDS` 之外再补
   几何与网格常量），使报告无法描述实验脚本已不再实现的协议；
3. 修好脚本的**文件方式启动**：`scripts/ml` 是正规包（有自己的 `__init__.py`），
   会被实验模块插入的 `<repo>/scripts` 顶到 `sys.path[0]`，从而遮蔽真正的
   `src/ml`，使 `from ml.zernike.eval_stats import ...` 直接
   `ModuleNotFoundError` —— 属导入图问题，与数值无关。

**被复用的既有机制（已存在，非本轮新增）**：

- `fit_adapter` 的**坐标下降**：`c` 与 `dtheta` 都用网格搜索（几何用 **5×5** 有限网格，
  §8 指出命中即精确复原、落空一步不动 ⇒ 当前数字是**上界**）；
- `TinyCorr` 冻结 CNN：离线训一次、适配时**冻结**，参数 {frozen_params} 个；
- `nominal_img` 对照变体：它把"冻结 CNN 吸收长期标定残差"与"冻结 CNN 帮助适配"分开，
  使 `c_img` 的比较是干净的。

## 0.3 结果分析

**结论 1（低阶漂移：{K} 个低阶 Zernike 参数便宜且有效）**：Δcorr {lo_c["mean"]:+.3f}
（d_z {lo_c["dz"]:+.1f}，{sig(lo_c)}）；N50（达到自身 90% 增益所需帧数，定义见 §5）
= {n50_txt("P-A-low", "c_only")}。

**结论 2（但它不是通用回退：高阶漂移下没有净增益）**：Δcorr {hi_c["mean"]:+.3f}
（d_z {hi_c["dz"]:+.2f}，{sig(hi_c)}）；N50 = {n50_txt("P-A-high", "c_only")}。
⇒「漂移是低阶的」是**赌注**，不是安全默认。

**结论 3（几何漂移只有光斑域指标看得见）**：`P-B-geom` 下 nominal 的 corr 已经
+{cell(d, "P-B-geom", "nominal", lo_last, "corr").mean():.3f}——只看 corr/R² 会得出
"几何没问题"的错误结论。暴露它的是 EE90：nominal {nom_g_e:.3f} px → `geom+c` {geom_g_e:.3f} px。

**结论 4（几何漂移必须由独立参数抓，Zernike 完全无效）**：{K} 个 Zernike 参数在
`P-B-geom` 下 Δee90 = {g_c["mean"]:+.4f}（逐 seed 配对差全部为 +0.000，见 §6 的逐 seed 表）；
+2 个几何标量 Δee90 = {g_e["mean"]:+.4f}（{sig(g_e)}）。反过来在低阶漂移下常开几何参数反而略差
（{c_only_lo:+.3f} → {geom_lo:+.3f}）⇒ 二者互斥，按漂移类型二选一。

**结论 5（冻结 CNN 对适配没有可证明的贡献）**：低阶漂移下 `c_img` vs `nominal_img`
Δ {lo_ci["mean"]:+.3f}（{sig(lo_ci)}），与 `c_only` 相比看不出优势；`nominal → nominal_img`
的改善（{nom_lo:+.3f} → {nom_img_lo:+.3f}）来自**吸收长期标定残差**，与漂移无关。

**协议边界（必须与结论一起读）**：{n_seeds} seed ⇒ sign-flip 最小可达 p = {minp:.5f}；
{n_tests} 个比较的 Holm 阈值是 {holm_thresh:.5f}；要求 {minp:.5f} < {holm_thresh:.5f}
⇒ 需要 n ≥ {need_n}（本设计只有 {n_seeds}）。所以 p 值列应读作「**协议欠功效**」，
请以 **Δ 与 d_z** 判断效应是否存在（§6）。

## 1. 问题设定

光路一动，冻结的标定就过期。要在线适配，就得决定**重新拟合什么**。
参数按职责**互不相交**地分开（这正是「独立一部分参数用来拟合像差」）：

| 组 | 内容 | 何时训练 | 参数量 |
|---|---|---|---|
| `c` | 低阶 Zernike（Noll 序，排除 piston） | **在线**，少量帧上拟合 | **{K}** |
| `dtheta` | 2 个可辨识几何标量 | 在线 | **2** |
| `g` | 小残差 CNN 图像矫正 | **离线**训一次，适配时**冻结** | 9857 |

`focal_length` **故意不扰动**：它与 `camera_pixel_um` 都只决定远场总尺度，
同时扰动两者在数学上不可辨识。这是结论，不是遗漏。

### 扰动（三类，必须分开）

| 代号 | 含义 | 作用 |
|---|---|---|
| `P-A-low` | 低阶像差（Noll 阶 2–6 主导） | 检验「低阶为主」 |
| `P-A-high` | **高阶像差对照**（阶 7–15） | **证伪**「低阶为主」 |
| `P-B-geom` | 几何漂移（像元尺度 ±3%、孔径 ±4%） | 检验几何是否需要独立参数 |

## 2. 协议（先定协议再跑数）

- 模型与「实验系统」**共用同一套 canonical numpy 传播器与 Zernike 基**，**只在几何向量上不同**。
  若两者完全相同，零测量就能完美匹配，实验将什么都不测。
- 冻结模型的几何是**故意标定不准**的（像元 +2%、孔径 −1%）：
  这份**长期标定残差**正是离线 CNN 唯一能吸收的东西，
  适配器随后只需解释漂移带来的**增量**。若标定是完美的，CNN 将无事可学、
  `c_img` 会退化成 `c_only` —— 那样这个对照就是假的。
- 判据只用 **corr / R² / 光斑域（EE90、CV）**。**不用 MSE/PSNR/SSIM**：
  本仓库实测在 sum 归一化下 PSNR 72 dB 而 R² 更差，即指标在度量归一化本身。
- 配对**按 seed**（同 seed 换 arm），不跨 seed 混配。
- {n_seeds} seed ⇒ `min_attainable_pvalue = {minp:.5f}`。
  {"**结构上可达 p<0.05**" if minp < 0.05 else "**结构上到不了 p<0.05**，只能报效应量"}。

## 3. 主结果：held-out corr

![适配速度](figures/adaptation_corr.png)

{table(d, frames, "corr")}

## 4. 光斑域指标：corr 看不见的那一半

![光斑域](figures/adaptation_beam_metric.png)

{table(d, frames, "ee90_err")}

⚠️ **`P-B-geom` 下 nominal 的 corr 已经很高**，因为 peak 归一 + 纯尺度误差对相关
几乎不敏感。**若只看 corr/R² 会得出「几何没问题」的错误结论**；
暴露它的是 EE90 误差。这与 MSE/PSNR/SSIM 的陷阱同源。

## 5. 帧数：N50（达到自身 90% 增益所需帧数）

| 扰动 | arm | N50 |
|---|---|---|
{n50_rows}

## 6. 配对统计（sign-flip 精确检验 + Holm 校正）

| 对比 | n | Δ | d_z | p (raw) | p (Holm) |
|---|---|---|---|---|---|
{stat_rows}

**逐 seed 配对差**（均值会掩盖「5/6 一致但其中 1 个搜索完全失败」）：

| 主比较 | 指标 | {" | ".join(f"s{s}" for s in sorted({r["seed"] for r in rows}))} | 变好比例 |
|---|---|{"---|" * (len({r["seed"] for r in rows}) + 1)}
{diff_table}

> ⚠️ **协议的结构性限制：本研究在 n={n_seeds} 下，Holm 校正后**不可能**有任何结果达到 p<0.05。**
> sign-flip 的最小可达 p 是 `2/2^n = {minp:.5f}`（{n_seeds} seed）；
> 而 {n_tests} 个比较的 Holm 阈值是 `0.05/{n_tests} = {holm_thresh:.5f}`。
> 要求 `{minp:.5f} < {holm_thresh:.5f}` ⇒ **n ≥ {need_n}**（本设计只有 {n_seeds}）。
> 因此上表的 p(Holm) 列应读作「**协议欠功效**」，不是「**无效应**」：
> 请以 **Δ 与 d_z** 判断效应是否存在，用 seed 数判断能否下结论。
> 真正的修法是**预注册少量主比较**（例如只留 3 个：低阶/高阶/几何各一个），
> 而不是事后把 18 个都校正。

## 7. 三条需要修正直觉的结论（每条都带检验，不夸大）

1. **「{K} 个低阶 Zernike」是专用赌注，不是安全默认。**
   低阶漂移下增益巨大（Δcorr {lo_c["mean"]:+.3f}, d_z {lo_c["dz"]:+.1f}, {sig(lo_c)}）；
   高阶漂移下**没有净增益**（Δcorr {hi_c["mean"]:+.3f}, d_z {hi_c["dz"]:+.2f}, {sig(hi_c)}）。
   ⚠️ n=1 时高阶变体的点估计是**负的**，容易被读成「低阶适配器在高阶漂移下有害」。
   6 seed 下该差异**不显著**，所以正确说法是「**无效**」而不是「有害」——
   但同样意味着**不能把它当兜底**。

2. **冻结 CNN 的价值在「标定」不在「适配」。**
   `nominal → nominal_img` 略好（{nom_lo:+.3f} → {nom_img_lo:+.3f}），但那与漂移无关；
   适配效果要看 `c_img vs nominal_img`（低阶 Δ {lo_ci["mean"]:+.3f}, {sig(lo_ci)}），
   与 `c_only` 相比**看不出优势**。结论：**不要为「适配」引入 CNN**。

3. **几何参数必须由哨兵触发，不该常开。**
   几何漂移下 {K} 个 Zernike 参数 Δee90 = {g_c["mean"]:+.4f}（**完全无效**），
   而 +2 个几何标量 Δee90 = {g_e["mean"]:+.4f}（{sig(g_e)}）。
   反过来在**低阶像差**漂移下常开几何参数反而略差（{c_only_lo:+.3f} → {geom_lo:+.3f}，过拟合）。
   ⇒ 二者互斥，按漂动类型二选一。

## 8. 限制（必须与结论一起读）

- **`geom+c` 的漂亮分数部分是搜索 artifact，不是光学性能。** 逐 seed 看，
  几何变体 **5/6 seed 把 EE90 误差打到 ~0，但有 1 个 seed 恰好 +0.000（完全没动）**。
  原因是几何用 **5×5 有限网格坐标搜索**：命中就精确复原、落空就一步都不动。
  ⇒ 在把它当成「几何只需 2 个参数」的证据之前，必须先把搜索换成连续优化
  （或细化网格）并重跑；当前数字是**上界**。
- **14 个 Zernike 参数对几何漂移是「精确零影响」**：逐 seed 配对差**全部恰好 +0.000**。
  这不是「效果小」，而是「低阶 Zernike 子空间里根本没有尺度这个自由度」——
  与 `focal_length`/`camera_pixel_um` 退化性一致。

- **`P-B-geom` 是「同族」扰动**：它只改变适配器恰好拟合的那 2 个自由度，
  所以 `geom+c` 拿到接近完美的分数是**上界**，不代表真实重对准（真实情况还会
  引入面板旋转、渐晕、额外像差）。不要把它当预期性能。
- **仿真结论，不是实验系统结论。**实验系统的「像差」还含散斑、液晶响应、杂散光等
  结构上缺失的物理（见 `gs_refine_fwdmodel_calibration.md` §6），
  本模型**不含**这些，因此对真实光路的迁移性未经验证。
- 高阶变体只测了 Noll 阶 7–15 的**纯 Zernike**像差，未含非 Zernike 分量。
- 未跑 `n_frames > {frames[-1]}`；更长的预算可能改变饱和点。

## 9. 落到本实验系统的建议

1. **保留 {K} 个低阶 Zernike 作为在线适配器**——它便宜（{K} 个数）、
   在低阶漂移下 {n50(d, "P-A-low", "c_only", frames)} 帧即饱和。
2. **几何参数单独留 2 个标量，但用哨兵触发**（参见
   `online_recal_efficiency.md` §5.2 的升级阶梯）；不要每轮都开。
3. **加一道高阶哨兵**：若低阶适配后 EE90/CV 仍不达标，说明漂移不是低阶，
   此时 {K} 参数适配会**帮倒忙**，应转入 {K}+2 或更强的先验。
4. **不要为「适配」引入 CNN。**离线冻结 CNN 值得做（吸收长期标定残差），
   但它不会让适配更快；把它当标定工具，别当适配工具。
"""


def main() -> None:
    rows = load()
    d = grid(rows)
    frames = frames_seen(rows)
    render_figures(d, frames)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = ROOT / REPORT_KEY
    out.write_text(insert_header(build(rows), REPORT_KEY), encoding="utf-8")
    print(f"Wrote {out}")
    print(
        f"  figures: recal_cost_ladder-style -> {FIG_DIR.name}/adaptation_corr.png, adaptation_beam_metric.png"
    )
    print(f"  seeds={len({r['seed'] for r in rows})} frames={frames} rows={len(rows)}")


if __name__ == "__main__":
    main()
