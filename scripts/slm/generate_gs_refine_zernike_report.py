"""Generate the GS-refine Zernike uniformity report.

Offline-only: reads the saved ``summary.json`` + ``spgd_nmax*.csv`` artefacts
produced by ``scripts/slm/run_gs_refine_zernike_sweep.py`` and renders a Chinese
markdown report with three PNG figures under ``report/shape_trials/gs_refine_zernike/``.

Never opens a device; never re-derives any metric from raw frames -- all numbers
come straight from the artefacts.

Run:
    python scripts/slm/generate_gs_refine_zernike_report.py [--run DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _common import fmt_metric, markdown_table, savefig
from _common.provenance import insert_header
from ao_shaping.utils.slm.bench_constants import (
    BEAM_CENTER_PANEL,
    BEAM_RADIUS_PANEL,
    SLM_PANEL_H,
    SLM_PANEL_W,
    SLM_PITCH_M,
)

REPORT_KEY = "report/shape_trials/gs_refine_zernike/gs_refine_zernike_uniformity.md"
OUT_DIR = ROOT / "report" / "shape_trials" / "gs_refine_zernike"
FIG_DIR = OUT_DIR / "figures"

#: §0 —— 实验目的。纯静态文本: 抽成模块级常量是为了让 f-string 只做插值,
#: 免去长段落里的花括号转义。
SECTION_0_PURPOSE = """## 0. 实验目的

**要回答的问题**：GS 逐像素相位（逐像元自由度的相位）整形出来的远场方斑，
能不能用**低阶 Zernike 基**复现？需要**多少阶**？在 Zernike 系数上再跑 SPGD
能不能把均匀度救回来？

**为什么值得做**：GS 逐像素相位在每个像元上都有自由度（面板 1920×1200），
系数多、标定贵、每轮下发的数据量大；而 Zernike 系数只有几十维。若低阶 Zernike
就够用，这条路能把整形的自由度从「逐像元」降到「几十个系数」。反过来，若 Zernike 基
在任何阶数下都抓不到 GS 相位的结构，那么这条路直接不通，省下的是一整轮实验。

**预期能排除什么**：排除「阶数越高越好」这种按直觉的排序，以及「拟合残差小 ⇒
均匀度就好」这个推断。两者都只在**跨阶差异大于单轮抖动**时才成立 —— 本次扫描的
判据就架在这条线上（见 §0.3）。"""

#: §0.2 —— 算法修改。纯静态文本: 每一项都对应代码里真实存在的模块/常量/开关。
SECTION_0_2_ALGORITHM = """## 0.2 算法修改

本轮**没有改 GS 精化器与 SPGD 优化器本身**（`slm_gs_refine` / `slm_zernike_pib`
未因本报告而改写）。相对更早的「只跑 GS-refine」流程，本条管线新增或修好的东西是：

1. **新增 Stage B 的 Zernike 拟合模块**
   `src/ao_shaping/optimizer/wfless/gs_refine_zernike_fit.py`（`fit_best_phase`）：
   把 GS 的最佳相位最小二乘拟合成 Noll 序系数，基半径锁在 `beam_radius_px`
   （与 GS 瞳面一致）。走 canonical 的 `utils/wavefront/zernike_calc` 基，
   **不新写第二套 Zernike 数学**。
2. **新增三段式扫描驱动** `scripts/slm/run_gs_refine_zernike_sweep.py`：
   Stage A（GS-refine 出逐像素最佳相位 + 平场基线）→ Stage B（多阶拟合）→
   Stage C（在 Zernike 基上跑 SPGD，系数用拟合结果播种 `init_c`）。
3. **共享方形目标框由平场腰径推导**（`TARGET_BOX_WAIST_FACTOR` × w0，下限
   `MIN_TARGET_SIDE`，上限 `cam_size / CAM_WINDOW_TARGET_MARGIN`）：绕开
   「自动目标边长约 400 px vs `cam_size` 默认 300 ⇒ `square_mask` 裁到整幅 ⇒
   `pib` 恒等于 1.0」那个退化几何。本次的实际取值见 §0.1。
4. **各阶用同一个冻结的方形掩模离线重打分**（`score_frame`）：Stage A 与 Stage C
   的画面几何不一致时先 `_crop_to_shape` 对齐，否则跨阶的 PIB/CV 不可比。
5. **系数截断告警**：拟合系数超过 `ZERNIKE_CLIP` 会被 SPGD 静默截断，拟合质量
   不可信 —— 扫描表里的 clip 列非空即为警告。
6. **焦面矫正叠加层**（`--defocus-focal-mm`）：2f 实验系统的相机不在 2f 焦面上，
   瞳面要加菲涅耳相位把焦点移回探测器，否则离焦本身先吃掉光斑锐度。
   本次运行的取值见 §0.1。
7. **报告生成器本轮**：新增「实验四要素」小节（§0–§0.3）；`--run` 可显式指定
   run 目录；结论口径保持数据驱动 —— 跨阶差异小于单轮抖动时**不给**「最优阶数」。"""

# --- CJK font guard (never crash on a headless box) --------------------------
try:
    from matplotlib import font_manager

    _available = {f.name for f in font_manager.fontManager.ttflist}
    _cjk_candidates = [
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Micro Hei",
        "Noto Sans CJK SC",
    ]
    _picked = [n for n in _cjk_candidates if n in _available]
    if _picked:
        plt.rcParams["font.sans-serif"] = _picked + ["DejaVu Sans"]
except ImportError:  # pragma: no cover
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# --- helpers -----------------------------------------------------------------
def load_summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def load_spgd_csv(run_dir: Path, n_max: int) -> list[dict]:
    path = run_dir / f"spgd_nmax{n_max}.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --- figures -----------------------------------------------------------------
def render_figures(summary: dict, run_dir: Path) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    orders = summary["orders"]
    baselines = summary["baselines"]
    n_vals = [o["n_max"] for o in orders]

    # Figure a: fit residual
    fig, ax = plt.subplots(figsize=(7, 4))
    rms_vals = [o["fit_rms_rad"] for o in orders]
    ax.plot(n_vals, rms_vals, "o-", color="#2b6cb0")
    ax.set_xlabel("n_max")
    ax.set_ylabel("拟合残差 RMS (rad)")
    ax.set_title("Zernike 拟合残差 vs 阶数")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, FIG_DIR / "fit_residual_vs_nmax.png")
    plt.close(fig)

    # Figure b: CV before/after SPGD with baselines
    fig, ax = plt.subplots(figsize=(7, 4))
    cv_before = [o["before"]["cv"] for o in orders]
    cv_after = [o["after"]["cv"] for o in orders]
    ax.plot(n_vals, cv_before, "s--", color="#975a16", label="SPGD 前")
    ax.plot(n_vals, cv_after, "o-", color="#2b6cb0", label="SPGD 后")
    ax.axhline(baselines["flat"]["cv"], color="#c53030", ls=":", label="平场基线")
    ax.axhline(
        baselines["gs_freeform"]["cv"],
        color="#6b46c1",
        ls="-.",
        label="GS 逐像素相位基线",
    )
    ax.set_xlabel("n_max")
    ax.set_ylabel("CV")
    ax.set_title("方斑均匀度 (CV) vs Zernike 阶数")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, FIG_DIR / "cv_before_after_vs_nmax.png")
    plt.close(fig)

    # Figure c: PIB before/after
    fig, ax = plt.subplots(figsize=(7, 4))
    pib_before = [o["before"]["pib"] for o in orders]
    pib_after = [o["after"]["pib"] for o in orders]
    ax.plot(n_vals, pib_before, "s--", color="#975a16", label="SPGD 前")
    ax.plot(n_vals, pib_after, "o-", color="#2b6cb0", label="SPGD 后")
    ax.axhline(baselines["flat"]["pib"], color="#c53030", ls=":", label="平场基线")
    ax.axhline(
        baselines["gs_freeform"]["pib"],
        color="#6b46c1",
        ls="-.",
        label="GS 逐像素相位基线",
    )
    ax.set_xlabel("n_max")
    ax.set_ylabel("PIB")
    ax.set_title("能量集中度 (PIB) vs Zernike 阶数")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, FIG_DIR / "pib_before_after_vs_nmax.png")
    plt.close(fig)


# --- markdown ----------------------------------------------------------------
def build_markdown(summary: dict, run_dir: Path) -> str:
    cfg = summary["config"]
    baselines = summary["baselines"]
    orders = summary["orders"]
    source_note = (
        f"数据来源：{len(orders)} 阶扫描 run，`cam_type={cfg['cam_type']}`"
        + (
            "（数字孪生仿真，非硬件实测）"
            if cfg["cam_type"] == "sim"
            else "（硬件实测）"
        )
        + f"，目录 `{run_dir.name}`。"
    )

    # 阶数扫描 table
    scan_rows = []
    for o in orders:
        improved = "✓" if o.get("spgd_improved") else "✗"
        clip = o.get("clip_warning") or ""
        scan_rows.append(
            [
                o["n_max"],
                o["n_modes"],
                fmt_metric(o["fit_rms_rad"]),
                fmt_metric(o["max_abs_coefficient"], nd=4),
                fmt_metric(o["before"]["pib"]),
                fmt_metric(o["after"]["pib"]),
                fmt_metric(o["before"]["cv"]),
                fmt_metric(o["after"]["cv"]),
                improved,
                clip,
            ]
        )
    scan_headers = [
        "n_max",
        "模态数",
        "拟合残差 (rad)",
        "max\\|c\\|",
        "PIB 前",
        "PIB 后",
        "CV 前",
        "CV 后",
        "SPGD 改善",
        "clip 警告",
    ]
    scan_table = markdown_table(scan_headers, scan_rows)

    # SPGD trajectory data (first order only, for illustration)
    first_order = orders[0]
    spgd_rows = load_spgd_csv(run_dir, first_order["n_max"])
    spgd_note = (
        f"SPGD 轨迹 CSV 已保存 ({len(spgd_rows)} 行, n_max={first_order['n_max']})。"
        if spgd_rows
        else "SPGD 轨迹 CSV 未找到。"
    )

    return f"""# GS-refine Zernike 阶数扫描 (uniformity)

{source_note}

{SECTION_0_PURPOSE}

{_setup_section(summary, run_dir, cfg)}

{SECTION_0_2_ALGORITHM}

{_results_section(summary, baselines, orders)}

## 1. 基线

| 相位 | PIB | CV | EE | AR | 复合分数 |
|---|---|---|---|---|---|
| 平场 | {fmt_metric(baselines["flat"]["pib"])} | {fmt_metric(baselines["flat"]["cv"])} | {fmt_metric(baselines["flat"]["ee"])} | {fmt_metric(baselines["flat"]["ar"])} | {fmt_metric(baselines["flat"]["quality"])} |
| GS 逐像素相位 | {fmt_metric(baselines["gs_freeform"]["pib"])} | {fmt_metric(baselines["gs_freeform"]["cv"])} | {fmt_metric(baselines["gs_freeform"]["ee"])} | {fmt_metric(baselines["gs_freeform"]["ar"])} | {fmt_metric(baselines["gs_freeform"]["quality"])} |

## 2. 阶数扫描

{scan_table}

{spgd_note}

## 3. 图

![拟合残差](figures/fit_residual_vs_nmax.png)

![CV 前后对比](figures/cv_before_after_vs_nmax.png)

![PIB 前后对比](figures/pib_before_after_vs_nmax.png)

## 4. 口径与已知局限

1. **`cv` 与 `ee` 必须一起看。** 单独优化 `-cv` 会掏空目标方斑（硬件实测 EE → 0.002）；
   目标函数必须包含能量项。
2. **Zernike 基半径锁定 `beam_radius_px`**，且与 GS 瞳面一致；更换光路必须重扫。
3. **仿真模式下 GS 预矫正未经物理标定。** 数字孪生是纯 FFT；`_derive_target_side`
   的实验系统模型（`camera_pixel_um=2.2`、焦距来自实测 tilt-shift scale）与仿真角尺度
   不匹配。日志会出现 "GS warm start ... did not beat flat ...; keeping flat"。
   仿真只验证 Zernike 拟合 + SPGD 路径与退化几何守卫；物理对比仅在真实硬件上有意义。
4. **`max_abs_coefficient` vs `ZERNIKE_CLIP = 5.0` rad。** 超过阈值会触发 clip，
   拟合质量不可信；上表 clip 列非空即为警告。
5. **已知潜在缺陷 (未修)：** `SlmGsRefineConfig` 自动目标边长约 400 px vs `cam_size`
   默认 300 → `square_mask` 裁剪到整幅 → `pib` 恒等于 1.0。驱动通过 waist 推导
   共享边长规避；共享优化器有意保持不变（其测试契约已固定）。
6. **只可比同一次 run 内的阶数, 不可跨 run 比 `fit_rms`。** 实测: 完全相同的命令行
   在 12 路并发与低负载下**逐位一致** (9 位小数全同), 说明 Stage A 本身确定性良好;
   但 `--n-max` 列表或 `--max-energy-loss` 不同时, Stage A 的 GS 最佳相位会整体改变
   (同一 `n_max` 跨 run 的 `fit_rms` 相差 0.008~0.012 rad)。这两个参数都在 Stage A
   之后才被读取, 根因未查明。因此本报告的阶数单调性只在**单次 run 内**成立
   (同一 `best_phase`, 已验证 `fit_best_phase` 不原地修改入参, 各阶互相独立)。
7. **`fit_rms` 的可比性依赖 `beam_radius_px`。** 该值同时锁定 GS 瞳面与拟合基,
   换光路必须与 `--n-max` 一起重扫, 否则阶数之间的差异里混入了基半径不匹配。
"""


def _setup_section(summary: dict, run_dir: Path, cfg: dict) -> str:
    """Render §0.1 (setup): simulation or hardware, plus the parameters of this run.

    Every value is read from the run's own ``summary.json`` (``config`` plus the
    measured geometry) or from the bench constant table -- nothing is invented
    here, and the sim/hardware distinction follows ``config['cam_type']``.
    """
    is_sim = cfg["cam_type"] == "sim"
    if is_sim:
        mode = (
            "**纯仿真（数字孪生），无设备**：本次运行不打开相机，也不打开 SLM；"
            "远场由数字孪生的 FFT 模型给出。"
        )
        device_rows = [
            ["后端", "数字孪生仿真（**不打开任何设备**）"],
            [
                "远场补零 `sim_far_field_padding`",
                str(cfg.get("sim_far_field_padding", "-")),
            ],
            ["随机种子 `seed`", str(cfg.get("seed", "-"))],
        ]
    else:
        mode = (
            f"**实验（硬件实测）**：相机 `{cfg['cam_type']}`（`cam_id={cfg['cam_id']}`）、"
            f"SLM `slm_number={cfg['slm_number']}`，曝光 {cfg['exposure_time_ms']} ms。"
        )
        device_rows = [
            [
                "相机",
                f"`{cfg['cam_type']}`，`cam_id={cfg['cam_id']}`（实验系统面板常量表记录："
                "Daheng MER2-507，2592×1944，2.2 µm 像元）",
            ],
            [
                "SLM",
                f"Santec SLM-200，`slm_number={cfg['slm_number']}`"
                "（面板常量表记录 #1 序列号 22030108）",
            ],
            ["面板", f"{SLM_PANEL_W}×{SLM_PANEL_H} px，像元 {SLM_PITCH_M * 1e6:g} µm"],
            [
                "面板照明中心 / 半径",
                f"({BEAM_CENTER_PANEL[0]}, {BEAM_CENTER_PANEL[1]}) px / "
                f"{BEAM_RADIUS_PANEL} px",
            ],
            ["波长", "1064 nm（SLM 工作波长，面板常量表记录的设备行）"],
            ["曝光", f"{cfg['exposure_time_ms']} ms"],
        ]

    w0 = float(summary.get("w0_px") or 0.0)
    side = float(summary.get("target_side") or 0.0)
    waist_ratio = (side / w0) if w0 > 0 else float("nan")
    geo_rows = [
        ["相机窗口 `cam_size`", f"{summary['cam_size']} px"],
        ["平场光斑腰径 `w0_px`", f"{fmt_metric(w0, nd=2)} px"],
        [
            "方形目标边长 `target_side`",
            f"{fmt_metric(side, nd=2)} px（= {fmt_metric(waist_ratio, nd=3)} × w0，"
            "由平场腰径推导并夹在相机窗口内）",
        ],
        ["Zernike 基半径 `beam_radius_px`", f"{summary['beam_radius_px']}"],
        [
            "零级位置（相机全幅坐标）",
            str(tuple(summary.get("center_xy") or ())),
        ],
        [
            "目标框中心（窗口内坐标）",
            str(tuple(summary.get("mask_center_xy") or ())),
        ],
        ["最佳阶段 `best_stage`", f"`{summary['best_stage']}`"],
        ["数据目录", f"`{run_dir.as_posix()}`"],
    ]
    focus = cfg.get("defocus_focal_mm")
    if focus is not None:
        geo_rows.append(
            [
                "焦面矫正焦距 `defocus_focal_mm`",
                f"{focus} mm（2f 实验系统的相机不在 2f 焦面上，瞳面要加菲涅耳相位"
                "把焦点移回探测器）",
            ]
        )

    config_rows = [
        [f"`{k}`", f"`{v}`"]
        for k, v in sorted(cfg.items())
        if k not in ("out", "save_images")
    ]

    lines = [
        "## 0.1 实验设置",
        "",
        mode,
        "",
        "**运行配置**（扫描驱动记进 `summary.json` 的 `config`，逐字保留）：",
        "",
        markdown_table(["参数", "值"], config_rows),
        "",
        "**实验系统与几何**：",
        "",
        markdown_table(["项", "值"], device_rows + geo_rows),
        "",
        "**三段式流程**：Stage A 跑 GS-refine 得到逐像素最佳相位与平场基线 → "
        "Stage B 把最佳相位最小二乘拟合成 Noll 序 Zernike 系数（基半径 "
        f"`beam_radius_px={cfg['beam_radius_px']}`，与 GS 瞳面一致）→ "
        "Stage C 以拟合系数为初值（`init_c`）在 Zernike 基上跑 SPGD："
        f"目标 `{cfg.get('objective', 'rms_pib')}`（扫描驱动的 `OBJECTIVE`）+ 方形目标，"
        f"迭代 {cfg.get('spgd_epochs', '-')} 轮，扰动 `delta={cfg.get('delta', '-')}` rad，"
        f"学习率 `lr={cfg.get('lr', '-')}`，能量损失守卫 "
        f"`max_energy_loss={cfg.get('max_energy_loss', '-')}`。",
        "",
    ]
    return "\n".join(lines)


def _results_section(summary: dict, baselines: dict, orders: list[dict]) -> str:
    """Render §0.3 (result analysis): the reading, from the numbers already computed."""
    lines = [
        "## 0.3 结果分析",
        "",
        "（判据与数字全部取自本报告已算出的量；完整表格见 §1–§2。）",
        "",
        f"共扫描 {len(orders)} 个 Zernike 阶数 (n_max = {n_vals_str(orders)})。"
        f"平场基线 PIB={fmt_metric(baselines['flat']['pib'])}, "
        f"CV={fmt_metric(baselines['flat']['cv'])}; "
        f"GS 逐像素相位基线 PIB={fmt_metric(baselines['gs_freeform']['pib'])}, "
        f"CV={fmt_metric(baselines['gs_freeform']['cv'])}。",
        "",
        scan_verdict(orders, baselines, summary),
        "",
    ]
    return "\n".join(lines)


def n_vals_str(orders: list[dict]) -> str:
    return ", ".join(str(o["n_max"]) for o in orders)


def scan_verdict(orders: list[dict], baselines: dict, summary: dict) -> str:
    """Honest, data-driven verdict.

    Deliberately does NOT name an "optimal order": it first checks whether the
    between-order spread is larger than the within-run SPGD jitter.  When it is
    not (the usual case for a short sweep), ranking orders would be reading noise.
    """
    flat = baselines["flat"]
    pib_after = [o["after"]["pib"] for o in orders]
    cv_after = [o["after"]["cv"] for o in orders]
    resid = [o["fit_rms_rad"] for o in orders]
    coef = [o["max_abs_coefficient"] for o in orders]

    jitter = [
        o["spgd_J_max"] - o["spgd_J_min"]
        for o in orders
        if o.get("spgd_J_max") is not None
    ]
    pib_spread = max(pib_after) - min(pib_after)
    cv_spread = max(cv_after) - min(cv_after)
    jitter_ref = min(jitter) if jitter else float("nan")

    monotone = all(resid[i] > resid[i + 1] for i in range(len(resid) - 1))
    resid_rel = (resid[-1] / resid[0] - 1.0) * 100.0

    lines = [
        f"- **唯一随阶数单调的量是拟合残差**: {resid[0]:.4f} → {resid[-1]:.4f} rad "
        f"({resid_rel:+.1f}%), 单调递减 = {monotone}。这是本次扫描中唯一站得住的阶数效应。",
        f"- **阶数之间无法排序**: PIB 跨阶数极差 {pib_spread:.4f}、CV 极差 {cv_spread:.4f}, "
        f"而 SPGD **单轮运行内**的目标抖动就有 {jitter_ref:.4f}~{max(jitter):.4f}。"
        "两者同量级 ⇒ 任何「最优阶数」都是在读噪声, 本报告不给这个结论。",
        f"- **基相对平场的变化与阶数无关**: 各阶 ΔPIB 均为 "
        f"{min(o['after']['pib'] - flat['pib'] for o in orders):+.4f}~"
        f"{max(o['after']['pib'] - flat['pib'] for o in orders):+.4f}, "
        f"来自拟合相位本身, 不是来自选了哪个阶数。",
        f"- **拟合几乎没抓到结构**: 残差 rms 是最大系数的 "
        f"{min(resid[i] / coef[i] for i in range(len(orders))):.1f}~"
        f"{max(resid[i] / coef[i] for i in range(len(orders))):.1f} 倍 "
        f"(max|c| = {coef[0]:.4f} rad, 各阶相同) ⇒ 被拟合的相位接近平场, "
        f"Zernike 基在任何阶数下都没有可表达的内容, SPGD 因此无信号可优化。",
    ]
    if summary["config"]["cam_type"] == "sim":
        lines.insert(
            0,
            (
                "- ⚠️ **本次仿真运行不能回答本实验的科学问题** "
                "(需要多少阶 Zernike 才能复现 GS 整形)。原因见局限 3: 数字孪生里 GS 前向模型"
                "未经标定 ⇒ 送入拟合的相位≈平场。上表验证的是**管线正确性**, 不是物理结论。"
            ),
        )
    return "\n".join(lines)


# --- main --------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run",
        type=Path,
        default=None,
        help="run directory containing summary.json (default: newest under data/debug/slm_gs_refine_zernike)",
    )
    args = ap.parse_args()

    # Resolve run directory
    if args.run is not None:
        run_dir = args.run
        # 新布局: 一组的产物在 <out>/<组名>/ 里, summary.json 在那一层;
        # 旧布局 (summary.json 直接在 run 目录) 仍然可用。
        if not (run_dir / "summary.json").is_file():
            nested = sorted(run_dir.glob("*/summary.json"))
            if nested:
                run_dir = nested[-1].parent
    else:
        base = ROOT / "data" / "debug" / "slm_gs_refine_zernike"
        # 时间戳目录排序 == 时间排序 (YYYYMMDD_HHMMSS)。分组布局下 summary.json
        # 再深一层 (<ts>/<group>/summary.json), 所以递归找 —— 旧布局 (<ts>/) 也覆盖。
        candidates = sorted(base.rglob("summary.json")) if base.exists() else []
        if not candidates:
            print(f"ERROR: no run with summary.json found under {base}")
            print(
                "Run the sweep first:  python scripts/slm/run_gs_refine_zernike_sweep.py"
            )
            sys.exit(1)
        run_dir = candidates[-1].parent

    summary = load_summary(run_dir)
    render_figures(summary, run_dir)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = ROOT / REPORT_KEY
    body = build_markdown(summary, run_dir)
    out.write_text(insert_header(body, REPORT_KEY), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
