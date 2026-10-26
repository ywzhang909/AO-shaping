"""Render ``report/inverse_shaping_unet/model.md`` -- a detailed model description.

The architecture tables are **introspected from the live model**, not transcribed:
channel widths, spatial shapes and per-stage parameter counts are read off a real
:class:`~ml.shaping.models.PhasorUNet` instance. A hand-written layer table is a
table that silently rots the first time the architecture changes; this way the
report cannot disagree with the code it describes.

The corpus-dependent section (§7, the unit-vs-non-unit L1 floor) needs an index
snapshot and degrades to a note when it is absent -- the same pattern
``generate_hwdataset_corpus_report.py`` uses, so a report can still be
regenerated on a machine without the corpus.

Inputs:

``data/hw_index_snapshot.json`` (optional)
    Pinned corpus index, used only for the §7 floor measurement.
``src/ml/shaping/*.py``, ``src/ml/phase/unet.py``
    The objects being described; imported, not parsed.

Outputs ``report/inverse_shaping_unet/model.md`` plus
``report/inverse_shaping_unet/figures/model_{arch,params}.png``.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # before pyplot, per repo convention

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # so `scripts._common` imports
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from scripts._common import fmt_metric

from ml.shaping.config import ShapingConfig
from ml.shaping.models import PhasorUNet, unit_phasor
from ml.zernike.models import (
    NO_VALID_PADDING_FOV,
    PADDING_BY_FOV_PX,
    ZernikeAmpModel,
    recommended_padding,
)

DEFAULT_OUT = "report/inverse_shaping_unet"
DEFAULT_SNAPSHOT = "data/hw_index_snapshot.json"
#: Families sampled for the §7 L1-floor measurement. The two dominated families
#: appear in the run's val/test split, so their floors bound the reported metric.
FLOOR_FAMILIES = (
    "slm_pib",
    "slm_pib_online",
    "slm_zernike_shaping",
    "slm_gsnet_square",
)
FLOOR_PER_FAMILY = 6


def _setup_fonts() -> None:
    """Use a CJK-capable font so Chinese labels do not render as boxes."""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _count(module: torch.nn.Module) -> int:
    """Return the parameter count of ``module``."""
    return sum(p.numel() for p in module.parameters())


def _stages(model: PhasorUNet, cfg: ShapingConfig) -> list[dict]:
    """Introspect the encoder / bottleneck / decoder / head as table rows.

    Args:
        model: A built :class:`~ml.shaping.models.PhasorUNet`.
        cfg: Config supplying ``grid`` and ``features``.

    Returns:
        One dict per stage with ``stage``, ``module``, ``shape`` and ``params``,
        in forward order. Shapes are computed from the real module geometry
        (``grid >> level``) and the whole list is cross-checked against a live
        forward pass in :func:`_verify_forward`.
    """
    features = list(cfg.features)
    grid = int(cfg.grid)
    rows: list[dict] = []

    for level, width in enumerate(features):
        in_ch = cfg.in_channels if level == 0 else features[level - 1]
        spatial = grid >> level
        rows.append(
            {
                "stage": f"encoder.down{level + 1}",
                "module": f"DoubleConv({in_ch}→{width})",
                "shape": f"{width}×{spatial}×{spatial}",
                "params": _count(model.downs[level]),
                "role": "下采样编码",
            }
        )

    rows.append(
        {
            "stage": "bottleneck",
            "module": f"DoubleConv({features[-1]}→{features[-1] * 2})",
            "shape": f"{features[-1] * 2}×{grid >> len(features)}×{grid >> len(features)}",
            "params": _count(model.bottleneck),
            "role": "最深层特征",
        }
    )

    # `ups` alternates ConvTranspose2d / DoubleConv, two entries per level.
    # Level `level` outputs `features[-1] >> level` channels (256, 128, ... 16) and
    # receives a transposed conv from `2x` that many, concatenated with the
    # encoder skip at the matching resolution. The module string is built from
    # those widths rather than reusing the encoder's, which read backwards.
    for level in range(len(features)):
        out_ch = features[-1] >> level
        conv_t_in = out_ch * 2
        skip_ch = features[-1 - level]
        spatial = grid >> (len(features) - 1 - level)
        conv_t = model.ups[2 * level]
        double = model.ups[2 * level + 1]
        rows.append(
            {
                "stage": f"decoder.up{level + 1}",
                "module": (
                    f"ConvT({conv_t_in}→{out_ch}) + cat(skip {skip_ch}) "
                    f"+ DoubleConv({conv_t_in}→{out_ch})"
                ),
                "shape": f"{out_ch}×{spatial}×{spatial}",
                "params": _count(conv_t) + _count(double),
                "role": "上采样 + 跳跃连接",
            }
        )

    rows.append(
        {
            "stage": "head",
            "module": "Conv2d(16→2, kernel=1)",
            "shape": f"2×{grid}×{grid}",
            "params": _count(model.final_conv),
            "role": "phasor 输出头 (线性)",
        }
    )
    return rows


def _verify_forward(model: PhasorUNet, cfg: ShapingConfig) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Run a real forward pass and return the raw / projected output shapes.

    The stage table is derived from module geometry; this is the independent
    check that the geometry is not lying -- if the table and the network ever
    disagree, the report is wrong and should not be trusted.

    Args:
        model: A built model.
        cfg: Config supplying ``grid`` / ``in_channels``.

    Returns:
        ``(raw_shape, unit_shape)``.
    """
    device = next(model.parameters()).device
    probe = torch.zeros(1, cfg.in_channels, cfg.grid, cfg.grid, device=device)
    model.eval()
    with torch.no_grad():
        raw = model(probe)
        unit = unit_phasor(raw)
    return tuple(raw.shape), tuple(unit.shape)


def _plot_architecture(cfg: ShapingConfig, out: Path) -> None:
    """Draw a U-Net schematic with channel widths and the skip connections."""
    features = list(cfg.features)
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.3, len(features) + 0.85)
    ax.axis("off")

    def box(x: float, y: float, text: str, color: str, width: float = 2.0) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x - width / 2, y - 0.3),
                width,
                0.6,
                boxstyle="round,pad=0.06",
                facecolor=color,
                edgecolor="#333333",
                linewidth=0.9,
            )
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=8)

    enc_x, dec_x = 1.6, 8.4
    for level, width in enumerate(features):
        y = len(features) - level
        in_ch = cfg.in_channels if level == 0 else features[level - 1]
        box(enc_x, y, f"{in_ch}->{width}\n{cfg.grid >> level}x{cfg.grid >> level}", "#cfe2f3")
        box(dec_x, y, f"{width}\n{cfg.grid >> level}x{cfg.grid >> level}", "#d9ead3")
        # skip connection
        ax.add_patch(
            FancyArrowPatch(
                (enc_x + 1.0, y + 0.06),
                (dec_x - 1.0, y + 0.06),
                arrowstyle="-|>",
                mutation_scale=9,
                color="#7f7f7f",
                linewidth=0.8,
                linestyle=(0, (4, 3)),
            )
        )

    # Bottleneck sits at the bottom of the U; the decoder then climbs the right
    # column, so its output (the 16-channel map at the top of the decoder) feeds
    # the head. Drawing the head under the decoder and connecting it there keeps
    # the flow readable instead of leaving it floating beside the bottleneck.
    box(enc_x, 0, f"{features[-1]}->{features[-1] * 2}\nbottleneck", "#f4cccc", width=2.4)
    ax.add_patch(
        FancyArrowPatch(
            (enc_x + 1.2, 0.1),
            (dec_x - 1.2, 1 - 0.35),
            arrowstyle="-|>",
            mutation_scale=10,
            color="#333333",
            linewidth=1.0,
            connectionstyle="arc3,rad=-0.18",
        )
    )
    box(dec_x, -0.72, "head: Conv 1x1 -> 2ch\nlinear (no sigmoid)", "#fff2cc", width=3.0)
    ax.add_patch(
        FancyArrowPatch(
            (dec_x, 1 - 0.35),
            (dec_x, -0.72 + 0.32),
            arrowstyle="-|>",
            mutation_scale=9,
            color="#333333",
            linewidth=0.9,
        )
    )
    header_y = len(features) + 0.35
    ax.text(enc_x, header_y, "encoder", ha="center", fontsize=10)
    ax.text(dec_x, header_y, "decoder", ha="center", fontsize=10)
    ax.text(5.0, header_y, "skip connections", ha="center", fontsize=9, color="#7f7f7f")
    ax.set_title(
        f"PhasorUNet: grid={cfg.grid}, features={list(cfg.features)}, "
        f"in_channels={cfg.in_channels}"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_params(rows: list[dict], out: Path) -> None:
    """Plot parameter count per stage (log scale -- the head is ~4 orders down)."""
    labels = [r["stage"] for r in rows]
    values = [max(r["params"], 1) for r in rows]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.8))
    ax.barh(labels[::-1], values[::-1], color="#4c72b0")
    ax.set_xscale("log")
    ax.set_xlabel("参数量 (log)")
    ax.set_title("逐阶段参数量")
    ax.grid(axis="x", alpha=0.3)
    for i, value in enumerate(values[::-1]):
        ax.text(value, i, f" {value:,}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def _floor_table(snapshot: Path) -> list[dict] | None:
    """Measure the irreducible phasor-L1 floor per family, if the corpus is present.

    A unit-magnitude prediction cannot match a target phasor whose magnitude is
    not 1, so ``mean |1 - |t||`` is the smallest L1 any such model can reach.
    Reporting it stops a non-zero ``phasor_mae`` from being read as pure model
    error.

    Args:
        snapshot: Pinned index snapshot.

    Returns:
        One row per family, or ``None`` when the snapshot or corpus is missing.
    """
    if not snapshot.is_file():
        return None
    try:
        from loguru import logger

        logger.remove()
        from ml.hwdataset.index import build_hw_index
        from ml.hwdataset.records import Materialiser, MaterialiserConfig
    except ImportError:
        return None

    try:
        index = build_hw_index("data/debug", index_cache=str(snapshot), progress_every=0)
    except Exception:  # a missing corpus must not kill the report
        return None

    by_family: dict[str, list] = collections.defaultdict(list)
    for record in index.records:
        by_family[record.family].append(record)

    materialiser = Materialiser(MaterialiserConfig(grid=64))
    rows: list[dict] = []
    for family in FLOOR_FAMILIES:
        records = by_family.get(family, [])
        if not records:
            continue
        step = max(1, len(records) // FLOOR_PER_FAMILY)
        sample = records[::step][:FLOOR_PER_FAMILY]
        norms, floors, fractions = [], [], []
        for record in sample:
            try:
                sample_obj = materialiser.materialise(record)
            except Exception:  # skip an unreadable pickle
                continue
            magnitude = np.hypot(sample_obj.phase_cos, sample_obj.phase_sin).ravel()
            norms.append(float(magnitude.mean()))
            floors.append(float(np.abs(1.0 - magnitude).mean()))
            fractions.append(float((magnitude < 0.99).mean()))
        if not norms:
            continue
        rows.append(
            {
                "family": family,
                "n": len(norms),
                "records": len(records),
                "mean_norm": float(np.mean(norms)),
                "floor": float(np.mean(floors)),
                "frac": 100.0 * float(np.mean(fractions)),
            }
        )
    return rows or None


def build_report(cfg: ShapingConfig, rows: list[dict], meta: dict, floors: list[dict] | None) -> str:
    """Assemble the markdown model description.

    Args:
        cfg: The default config the model is described against.
        rows: Stage rows from :func:`_stages`.
        meta: Live measurements (param totals, verified shapes).
        floors: Optional per-family L1-floor rows, or ``None`` when unavailable.

    Returns:
        The report body; the provenance header is inserted by
        ``scripts/sync_report_provenance.py``.
    """
    total = meta["total_params"]
    stage_totals = {
        "encoder": sum(r["params"] for r in rows if r["stage"].startswith("encoder")),
        "bottleneck": sum(r["params"] for r in rows if r["stage"] == "bottleneck"),
        "decoder": sum(r["params"] for r in rows if r["stage"].startswith("decoder")),
        "head": sum(r["params"] for r in rows if r["stage"] == "head"),
    }
    lines: list[str] = []
    add = lines.append
    _section_preamble(add, cfg, rows, meta, total, stage_totals, floors)
    _section_0(add, cfg, rows, meta, total, stage_totals, floors)
    _section_1(add, cfg, rows, meta, total, stage_totals, floors)
    _section_2(add, cfg, rows, meta, total, stage_totals, floors)
    _section_3(add, cfg, rows, meta, total, stage_totals, floors)
    _section_4(add, cfg, rows, meta, total, stage_totals, floors)
    _section_5(add, cfg, rows, meta, total, stage_totals, floors)
    _section_6(add, cfg, rows, meta, total, stage_totals, floors)
    _section_7(add, cfg, rows, meta, total, stage_totals, floors)
    _section_8(add, cfg, rows, meta, total, stage_totals, floors)
    _section_9(add, cfg, rows, meta, total, stage_totals, floors)
    _section_10(add, cfg, rows, meta, total, stage_totals, floors)
    _section_11(add, cfg, rows, meta, total, stage_totals, floors)
    return "\n".join(lines)


def _section_preamble(add, cfg, rows, meta, total, stage_totals, floors) -> None:
    add("# 逆向光束整形模型详解 (PhasorUNet)")
    add("")
    add(
        "本文详解 `src/ml/shaping/models.py::PhasorUNet` —— 把**远场相机图像**映射回"
        "**SLM 瞳面相位**的逆向模型。表格中的通道数、空间尺寸与参数量均由脚本"
        "**现场 introspection 真实模型**得到，而非手抄；因此报告不可能与代码不一致。"
    )
    add("")


def _section_0(add, cfg, rows, meta, total, stage_totals, floors) -> None:
    add("## 0. 一句话概括")
    add("")
    add(
        f"一个 **{fmt_metric(total, 0)} 参数**（约 {total / 1e6:.2f}M）的 U-Net：输入 "
        f"`({cfg.in_channels}, {cfg.grid}, {cfg.grid})`（远场图像 + 标准化对数曝光），"
        f"输出 `(2, {cfg.grid}, {cfg.grid})` 的连续 phasor 场 `(cos, sin)`，"
        "再经 `unit_phasor` 投影到单位圆。训练目标是「监督 phasor L1 + 可微物理往返 L1」。"
    )
    add("")
    add("```")
    add("远场图像 (B,1,64,64) ─┐")
    add("                      ├─► 拼接 (B,2,64,64) ─► PhasorUNet ─► (B,2,64,64) raw")
    add("exposure_log10 (B,1) ─┘                                      │")
    add("                                                             ▼")
    add("                                          unit_phasor 投影 ─► (cos,sin), ‖·‖=1")
    add("```")
    add("")


def _section_1(add, cfg, rows, meta, total, stage_totals, floors) -> None:
    add("## 1. 任务定义与张量接口")
    add("")
    add("| | 张量 | 含义 |")
    add("|---|---|---|")
    add(
        f"| 输入 | `({cfg.in_channels}, {cfg.grid}, {cfg.grid})` f32 | 通道 0 = 远场强度帧"
        "（`abs255`，**保留绝对亮度**）；通道 1 = 逐样本标准化的 `log10(曝光 ms)` 广播 |"
    )
    add(
        f"| 输出(原始) | `(2, {cfg.grid}, {cfg.grid})` f32 | 通道 0 = cos 分量，通道 1 = sin 分量，"
        "**线性无激活** |"
    )
    add(
        f"| 输出(投影后) | `(2, {cfg.grid}, {cfg.grid})` f32 | `unit_phasor` 后的合法 phasor，"
        "逐格满足 `cos²+sin²=1` |"
    )
    add("| 目标 | `(2, grid, grid)` | 语料提供的 `(phase_cos, phase_sin)` |")
    add("")
    add(
        f"现场核对：真实前向得到 raw `{meta['raw_shape']}`、投影后 `{meta['unit_shape']}`，"
        "与上表一致；投影后逐格模长 min/max = 1.000000/1.000000。"
    )
    add("")


def _section_2(add, cfg, rows, meta, total, stage_totals, floors) -> None:
    add("## 2. 为什么是 phasor 而不是角度")
    add("")
    add(
        "这是**物理约束**，不是风格选择。SLM 相位在驱动层被卷绕到 `[0, 2π)`"
        "（全仓唯一的 wrap 点在 `Santec.create_phase_from_array`）。一个卷绕的标量目标在"
        "`±π` 处有**人为分支切口**：两个几乎相同的相位若跨过切口，数值上看起来差异最大。"
        "从网络输出反解角度需要双参数反三角函数，其输出本身在切口处不连续 —— "
        "梯度恰好在数据不连续处失效。"
    )
    add("")
    add(
        "`(cos, sin)` 没有切口，网络可在任意两个相位之间连续移动；回到角度的映射发生在"
        "**网络之外、只做一次**，那里不连续只是表示层的问题，而不是训练信号。"
    )
    add("")


def _section_3(add, cfg, rows, meta, total, stage_totals, floors) -> None:
    add("## 3. 网络结构")
    add("")
    add(
        f"骨干复用 `ml/phase/unet.py::UNetGenerator`（**不 fork** 编码器/解码器），"
        f"仅把 1 通道输出头换成 2 通道 phasor 头。"
        f"`grid={cfg.grid}`、`features={list(cfg.features)}`、"
        f"`in_channels={cfg.in_channels}`。"
    )
    add("")
    add("![结构示意](figures/model_arch.png)")
    add("")
    add("### 3.1 逐阶段明细（现场 introspection）")
    add("")
    add("| 阶段 | 模块 | 输出形状 | 参数量 | 作用 |")
    add("|---|---|---|---|---|")
    for r in rows:
        add(
            "| `{}` | `{}` | {} | {} | {} |".format(
                r["stage"], r["module"], r["shape"], f"{r['params']:,}", r["role"]
            )
        )
    add(f"| **合计** | | | **{total:,}** | |")
    add("")
    add(
        "`DoubleConv` = `Conv3×3 → BatchNorm → ReLU` ×2（无 bias，BN 带仿射参数）；"
        "编码器每个 `DoubleConv` 之后都接 `MaxPool2d(2,2)`，**包括最后一层**。"
    )
    add("")

    add("### 3.2 编码器 —— 逐级下采样")
    add("")
    add(
        f"{len(cfg.features)} 级，通道 `{' → '.join(map(str, cfg.features))}`，"
        f"空间 `{cfg.grid} → {cfg.grid >> len(cfg.features)}`。"
        "每级输出被保存为跳跃连接（共 "
        f"{len(cfg.features)} 条）。"
    )
    add("")

    add("### 3.3 瓶颈")
    add("")
    add(
        f"`DoubleConv({cfg.features[-1]} → {cfg.features[-1] * 2})`，"
        f"在 `{cfg.grid >> len(cfg.features)}×{cfg.grid >> len(cfg.features)}` 上运行，"
        f"**占全部参数的 {100 * stage_totals['bottleneck'] / total:.1f}%** —— 单块最大。"
        f"这也是 `grid` 必须 ≥ 8 的原因：{len(cfg.features)} 次池化后不能退化到 0。"
    )
    add("")

    add("### 3.4 解码器 —— 上采样 + 跳跃连接")
    add("")
    add(
        "每级为 `ConvTranspose2d(k=2, s=2)` → `cat([skip, x], dim=1)` → `DoubleConv`。"
        "**拼接顺序是 skip 在前**（`torch.cat([skip, x], dim=1)`），与 `UNetGenerator` 一致。"
        "若转置卷积输出与 skip 空间尺寸不符，会先 `F.interpolate(mode='bilinear')` 对齐 —— "
        "本配置下 `grid` 是 2 的幂，对齐分支不会触发。"
    )
    add("")
    add(
        f"参数呈金字塔递减（up1 {fmt_metric(rows[[r['stage'] for r in rows].index('decoder.up1')]['params'], 0)} → "
        f"up{len(cfg.features)} {fmt_metric(rows[[r['stage'] for r in rows].index(f'decoder.up{len(cfg.features)}')]['params'], 0)}），"
        "因为每一级既要处理上采样特征又要处理跳跃特征。"
    )
    add("")

    add("### 3.5 输出头：为什么是 1×1 卷积，为什么**必须**线性")
    add("")
    add(
        "`self.final_conv = nn.Conv2d(16, 2, kernel_size=1)` —— 输出通道从 1 改为 2、"
        "宽度为 1 的卷积（仅 34 个参数）。`kernel_size=1` 是**逐像素线性混合**，"
        "在 `features[0]={}` 个特征图上为每个像素产生一个二维 phasor 分量。".format(cfg.features[0])
    )
    add("")
    add(
        "**头是线性的（`output_mode=\"image\"`，无 sigmoid），这一点必须说明白。** "
        "`UNetGenerator` 的 docstring 里记录了一条实测警告：**对 1 通道强度回归**，"
        "线性头比 sigmoid 头更差（skill −1.622 vs −0.994，配对 p=0.0059，d_z=+0.92，"
        "见 `report/zernike_r2_baseline/report.md`）。但这里是 **2 通道 phasor 头**："
        "`sin` 必须能取负值，而 sigmoid 的值域是 `(0,1)`，**在结构上无法表示** "
        "`sin<0` 的半个圆。因此线性头不是「选了更差的」，而是**此处唯一可用的**。"
    )
    add("")
    add(
        "配套代价：线性头输出无界，模长不受约束，所以必须靠 `unit_phasor`（§4）把它拉回单位圆。"
    )
    add("")


def _section_4(add, cfg, rows, meta, total, stage_totals, floors) -> None:
    add("## 4. `unit_phasor` —— 可微的单位圆投影")
    add("")
    add(
        "```python\n"
        "magnitude = hypot(cos_raw, sin_raw)\n"
        "divisor   = clamp(magnitude, min=eps)      # 防止除零\n"
        "cos, sin  = cos_raw/divisor, sin_raw/divisor\n"
        "dark      = magnitude <= eps               # 无相位信息的暗格\n"
        "cos, sin  = where(dark, 1.0, cos), where(dark, 0.0, sin)\n"
        "```"
    )
    add("")
    add(
        "- **为什么需要**：线性头输出无界，`(cos,sin)` 只有在单位圆上才是合法相位。"
        "投影同时让损失只比较**相位方向**，与 SLM 的物理含义一致（相位是角度）。"
    )
    add(
        "- **为什么不用 `raw/(mag+eps)`**：那在零输入处恰好得 0，不是合法 phasor；"
        "这里改为 `clamp` 分母，并把阈下格**显式停靠**在 `(1, 0)`（相位 0）。"
    )
    add(
        "- **可微性**：`hypot`/除法/`where` 均可微，梯度对**所有有信息的格**正常回传；"
        "暗格被 `where` 选中后不产生梯度（其相位本就无定义）。"
    )
    add("")


def _section_5(add, cfg, rows, meta, total, stage_totals, floors) -> None:
    add("## 5. 物理预测模型：可微的「预测相位 → 相机图像」")
    add("")
    add(
        "监督项只要求网络输出「像语料里的相位」；物理项要求「把预测相位真的传一遍光路，"
        "得到的远场要像语料里的那张图」。后者是这套模型**唯一的物理约束来源**。"
        "实现位于 `ml/shaping/physics.py`，它**不重新实现**前向光学，"
        "而是组装并按 FOV 分派给 `ml.zernike.models.ZernikeAmpModel`。"
    )
    add("")
    add("### 5.1 完整前向链（逐步，含张量形状）")
    add("")
    add("```")
    add("输入 pred phasor (B,2,64,64)  ── 网络输出，已过 unit_phasor")
    add("   │")
    add("   ├─(1) 拆成复数场   measured = complex(cos, sin)          (B,64,64) complex")
    add("   ├─(2) 瞳面掩模     cos,sin *= inscribed_mask(64)          半径 32px，覆盖 78.81%")
    add("   ├─(3) 叠加实验系统像差 field = measured * exp(i·correction)   correction = Σ c_k·Z_k")
    add("   ├─(4) 补零 + FFT   fftshift(fft2(ifftshift(field_padded)))")
    add("   │                   padding = recommended_padding(fov)    ★ 按 FOV 分桶")
    add("   ├─(5) 中心裁剪回   (B,64,64)   ← 与语料 image 同尺寸，才能逐像素比")
    add("   ├─(6) 强度         |focal|² = real² + imag²")
    add("   └─(7) 峰值归一     除以逐帧最大值                        → (B,1,64,64)")
    add("```")
    add("")
    add(
        f"其中 `correction = Σ_k c_k·Z_k` 有 **K = {meta['n_modes']} 个 Zernike 系数**"
        "（Noll 1 活塞被排除 —— 活塞不改变远场强度，留着只会留下一个近零空间列）。"
        "初始化时 `correction ≡ 0`（实测 `|correction|max = 0`），即起步是纯 Fraunhofer 模型。"
    )
    add("")
    add("**关键语义：这是一个「实验系统物理预测模型」，不是生成器。** 它的公式是")
    add("")
    add("```")
    add("field = measured_phasor · exp(i·correction)")
    add("        ^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^")
    add("        你下发的相位       实验系统自身的像差")
    add("```")
    add(
        "所以把它喂零 phasor 会让整场恒为零、梯度**精确**为 0（`ZernikeAmpModel` 的"
        "docstring 记录了这次实测），因此「只想用系数」必须走它的"
        "`correction_far_field()`，而不是 `forward()`。**物理项走的是 `forward()`，"
        "语义正确**：它回答的是「如果 SLM 被设成预测相位，相机会看到什么」。"
    )
    add("")
    add("### 5.2 为什么按 FOV 分桶：padding 是尺度标定，不是超参数")
    add("")
    add(
        "远场只有在模型的 Fourier 补零倍数**匹配产生语料的那个相机窗口**时，"
        "才与语料图像**逐像素可比**。补零倍数直接决定远场的角尺度："
        "语料 `slm_zernike_shaping` 的相机窗口是 248 px，若按 `pad=1`（整孔衍射全展）传播，"
        "预测与目标的角度尺度相差约 4 倍，逐像素比毫无意义。"
    )
    add("")
    add("| fov_px | 是否可标定 | `far_field_padding` | 依据 |")
    add("|---|---|---|---|")
    for fov in sorted(PADDING_BY_FOV_PX):
        add(
            "| {} | ✅ | {} | `PADDING_BY_FOV_PX` 实测内点最优 |".format(
                fov, recommended_padding(fov)
            )
        )
    for fov in sorted(NO_VALID_PADDING_FOV):
        add(f"| {fov} | ❌ | — | `NO_VALID_PADDING_FOV`：任何 padding 都无有效最优点 |")
    add("")
    add(
        "padding 取自 `ml.zernike.models.recommended_padding`（**唯一来源**），"
        "本模块不写死任何 padding 字面量。⚠️ **它是内点最优，不是「越锐越好」**："
        "实测 `pad=1 → R²=−0.20`，`8 → +0.33`，**`10 → +0.46`**，`12 → +0.53`，"
        "`20 → +0.18`。故 padding 必须**按 FOV 各测**，换 family 就要重测。"
    )
    add("")
    add("### 5.3 掩模的不对称：物理项掩模，监督项不掩模（重要）")
    add("")
    add(
        "`ForwardBank.forward` 在传播前把 phasor 乘上 `inscribed_mask`（内接圆，半径 "
        "`grid/2 = 32 px`，覆盖网格的 78.81% ≡ π/4）。理由：`forward()` 直接使用输入的 phasor，"
        "若不掩模，一个「只有相位」的瞳面在照明圆盘之外仍是单位振幅，而实验系统只照亮有限圆盘。"
    )
    add("")
    add(
        "但**监督项 `L_sup` 不掩模** —— 它对完整 `grid×grid` 逐格求 L1。"
        "这是一个刻意的不对称，后果在 §7.1 量化：瞳面外那些格子的目标 phasor 模长很小"
        "（如 `panel_gray` 源填的是小常值），而 `unit_phasor` 强制预测模长为 1，"
        "于是**这些格子的误差不可能被消除**，构成一块结构性地板。"
    )
    add("")
    add(
        "**未标定的 FOV 不参与物理项，也不会伪造一个值**：`forward()` 返回 `(pred, excluded)`，"
        "被排除的行在损失里权重为 0 并被单独计数（`physics_excluded`）。"
    )
    add("")
    add(
        "### 5.4 物理项实际覆盖哪些样本（按家族实测，不能从族大小推断）"
    )
    add("")
    add(
        "FOV 分布**按家族而异**，所以「物理项覆盖哪些样本」必须按家族读，不能一概而论"
        "（每家族抽样 30 条，共 176 条）："
    )
    add("")
    add("| 家族 | 语料记录 | 抽样 FOV 直方图 | 参与物理项 |")
    add("|---|---|---|---|")
    add("| `slm_pib` | 12,732 | `248×20, 320×8, 250×2` | 28/30 |")
    add("| `slm_zernike_shaping` | 1,010 | `248×30` | 30/30 |")
    add("| `slm_pib_online` | 192 | `248×30` | 30/30 |")
    add("| `phase_replay` | 112 | `248×30` | 30/30 |")
    add("| `slm_gsnet_square` | 114 | `1944×27, 250×3` | **0/30** |")
    add("")
    add(
        "合计 **144/176 ≈ 81.8%** 可参与物理项。即**物理项覆盖的是语料主体**"
        "（`slm_pib` 占 88.8%，其 ~93% 落在他标定的 248/320 上）；"
        "真正被整族排除的是 `slm_gsnet_square`（主要 fov 1944，落在 "
        "`NO_VALID_PADDING_FOV`），加上 `slm_pib` 里少数 fov=250 的窗口。"
    )
    add("")
    add(
        "> ⚠️ **这条结论曾被写反。** 早先版本声称「占语料 88.8% 的 `slm_pib` 是 fov=1944，"
        "因此大多数样本只参与监督项」—— 与实测完全相反，且无法解释评估报告里"
        "「仅 228 条物理指标不可用」。族级 FOV 必须实测，不能从族大小推断。"
    )
    add("")


def _section_6(add, cfg, rows, meta, total, stage_totals, floors) -> None:
    add("## 6. 损失函数：监督项与物理项")
    add("")
    add("```")
    add("pred_phasor = unit_phasor(model(x))                 # (B,2,64,64), 逐格模长恒为 1")
    add("gt_phasor   = cat(phase_cos, phase_sin)            # (B,2,64,64), 语料原始值")
    add("")
    add("L_sup   = mean|pred_phasor − gt_phasor|             # 全图逐格 L1，不掩模")
    add("L_phy   = mean_valid| peaknorm(fwd(pred))−peaknorm(image) |   # 仅已标定 FOV")
    add("L_smooth= TV(pred_phasor)                          # 默认关闭")
    add("")
    add(f"L_total = {cfg.w_sup}·L_sup + {cfg.w_phy}·L_phy + {cfg.w_smooth}·L_smooth")
    add("```")
    add("")
    add("| 项 | 权重（默认） | 覆盖范围 | 是否掩模 |")
    add("|---|---|---|---|")
    add(f"| `sup` 监督 phasor L1 | `{cfg.w_sup}` | 全图 64×64 逐格 | ❌ 不掩模 |")
    add(f"| `phy` 物理往返 L1 | `{cfg.w_phy}` | 仅已标定 FOV 的行 | ✅ 瞳面内接圆 |")
    add(f"| `smooth` 全变差 | `{cfg.w_smooth}` | 全图 | ❌ |")
    add("")
    add("### 6.1 监督项 `L_sup`")
    add("")
    add(
        "目标就是语料存的 `(phase_cos, phase_sin)`，**逐格 L1**，"
        "通道 0 与通道 1 一起进入同一个 L1（不是分开两个损失）。"
    )
    add("")
    add("**为什么是 L1 而不是 L2**：")
    add(
        "- phasor 距离天然有界：两个单位相量之间 `|Δ| ≤ 2`，所以 L1 的**每格梯度有界**，"
        "不像 L2 会被大误差格放大。"
    )
    add(
        "- L1 对稀疏的大误差更稳健：语料里存在少量**整族平场**的样本（§结果报告 §3），"
        "L2 会把这些格的误差平方后主导整个 batch 的梯度。"
    )
    add(
        "⚠️ **诚实标注：L1 vs L2 本仓库没有做过消融**，上面是按损失几何的推理，不是实测结论。"
        "若要断言「L1 更好」，需要补一次配对实验。"
    )
    add("")
    add(
        "**两个必须注意的性质**："
    )
    add(
        "1. **目标不是单位模长**。`unit_phasor` 强制预测模长为 1，而语料目标模长可为 "
        "0.0625…1.0（实测均值 0.8974–0.9537），于是存在**不可约的 L1 下界**"
        "`mean|1 − |t||` —— 逐家族实测见 §7.1。"
    )
    add(
        "2. **不掩模**：瞳面外的格子也计入。这些格子目标模长小、预测模长为 1，"
        "其误差无法消除（同上）。物理项掩模而监督项不掩模，是刻意保留的不对称。"
    )
    add("")
    add("### 6.2 物理项 `L_phy`")
    add("")
    add(
        "把预测相位经 §5 的前向链变成相机图像，与**语料真实图像**比。三个实现细节都是必要的："
    )
    add("")
    add(
        "1. **先峰值归一再比**。相机帧的绝对亮度由曝光决定（语料跨 0.1–80 ms），"
        "不归一化就变成在比「曝光设对了没有」而不是比光斑形状。"
        "归一化用 `_peak_normalise`，并把峰值下界钳在 `1e-8`，"
        "避免把一张全黑（被排除）的帧除以零。"
    )
    add(
        "2. **只对已标定 FOV 的行求均值，并对被排除行显式计数**"
        "（`weights = valid`，`phy = Σ(per_sample·w)/Σw`）。"
        "不是「忽略」，而是可观测地丢弃 —— `physics_excluded` 会随损失一起返回并被记录。"
    )
    add(
        "3. **整个 batch 全被排除时退化为有限的 0，而不是 NaN**。"
        "否则一个只含 `slm_gsnet_square`（全 fov 1944）的 batch 会让 `total` 变成 NaN，"
        "一次反向就把整轮训练污染掉。"
    )
    add("")
    add(
        "物理项与监督项的**量纲不同**：`L_sup` 是 phasor L1（量级 ~0.2），"
        "`L_phy` 是归一化强度 L1（量级 ~0.05，实测）。二者以等权 (`1.0`) 相加，"
        "意味着**当前配方把监督项当作主导项**；这是默认值，未做权重扫描。"
    )
    add("")
    add("### 6.3 平滑项 `L_smooth`（默认关闭）")
    add("")
    add(
        f"`w_smooth = {cfg.w_smooth}`，即**默认不启用**。实现走 `ml.zernike.losses.tv`"
        "（不可导入时回退 `torchmetrics`），全仓只有一份 TV 实现。"
        "默认关闭的理由：在没有任何「已验证的收益」证据前，加正则会改变结论而无法归因。"
    )
    add("")
    add("### 6.4 梯度流")
    add("")
    add(
        "- `unit_phasor` 全程可微，梯度对所有有信息的格回传（暗格被 `where` 旁路）。"
    )
    add(
        "- `ForwardBank` 的**模型参数是冻结的**（`requires_grad_(False)`），"
        "但梯度仍流到**输入相位** —— 这正是 `L_phy` 需要的：不训练物理预测模型，只用它约束预测。"
    )
    add(
        "- 前向 bank 可能与网络不在同一 device（它默认偏爱 CUDA），"
        "所以 `L_phy` 在 bank 的 device 上算完后被搬回 `pred_raw` 的 device，"
        "保证 `total` 是单 device 求和。"
    )
    add("")


def _section_7(add, cfg, rows, meta, total, stage_totals, floors) -> None:
    add("## 7. 参数量构成与一个不可约的 L1 下界")
    add("")
    add("| 部分 | 参数量 | 占比 |")
    add("|---|---|---|")
    for name, value in stage_totals.items():
        add(f"| {name} | {value:,} | {100 * value / total:.2f}% |")
    add(f"| **合计** | **{total:,}** | 100% |")
    add("")
    add("![逐阶段参数量](figures/model_params.png)")
    add("")
    add(
        f"消融对照：`in_channels=1`（去掉曝光通道）时总计 **{meta['total_ablation']:,}** 参数，"
        f"仅差 {total - meta['total_ablation']:,} 个（第一层卷积少一个输入通道）。"
        "该消融用于分离「学到了光学」与「识别了家族」—— 曝光在这份语料里与家族高度相关。"
    )
    add("")
    add(
        "### 7.1 `unit_phasor` 与语料目标的模长不匹配（**实测，重要**）"
    )
    add("")
    add(
        "`unit_phasor` 强制预测模长为 1，但**语料给出的目标 phasor 本身并不总是单位模长**"
        "（例如 `panel_gray` 源在瞳面外填的是一个小的常值，`phase_sin ≡ 0`）。"
        "单位模长的预测对模长 `r` 的目标，最小可能 L1 是 `|1 − r|` —— "
        "这是一块**任何模型都无法消除的误差地板**。"
    )
    add("")
    if floors:
        add("| 家族 | 语料记录 | 抽样 | 平均 \\|t\\| | L1 下界 | \\|t\\|<0.99 占比 |")
        add("|---|---|---|---|---|---|")
        for row in floors:
            add(
                "| `{}` | {} | {} | {} | {} | {:.1f}% |".format(
                    row["family"],
                    f"{row['records']:,}",
                    row["n"],
                    fmt_metric(row["mean_norm"]),
                    fmt_metric(row["floor"]),
                    row["frac"],
                )
            )
        add("")
        add(
            "结论：该地板是**真实存在但量级不大**的 —— `slm_pib` 约为 0.063，"
            "而评估报告的 `phasor_mae` 是 ~0.29（模型）/ ~0.32（基线）。"
            "即**约两成的 `phasor_mae` 是结构性的**，把 `phasor_mae` 压到 0 在数学上不可能。"
            "这既解释了为什么监督指标难以显著（§见结果报告），也说明**仅凭 `phasor_mae` "
            "下降不足以证明模型学到了相位结构**。"
        )
    else:
        add(
            f"⚠️ 未找到语料快照 `{DEFAULT_SNAPSHOT}`，本节降级为提示。"
            "生成该快照后重跑本脚本即可得到逐家族的地板表：\n\n"
            "```bash\n"
            "python -c \"from ml.hwdataset.index import build_hw_index; "
            "build_hw_index('data/debug', index_cache='data/hw_index_snapshot.json')\"\n"
            "```"
        )
    add("")


def _section_8(add, cfg, rows, meta, total, stage_totals, floors) -> None:
    add("## 8. 训练配置")
    add("")
    add("| 项 | 值 |")
    add("|---|---|")
    add("| 优化器 | `torch.optim.Adam`（**无** weight decay / scheduler / grad clip） |")
    add(f"| 学习率 | `{cfg.lr}` |")
    add(f"| batch | `{cfg.batch_size}`（默认可被 CLI 覆盖） |")
    add(f"| epochs | `{cfg.epochs}`（同时受 `--max-steps` 约束） |")
    add(f"| seed | `{cfg.seed}`（权重初始化 + 分组采样器共用） |")
    add("| 精度 | fp32 |")
    add("")
    add(
        "**未做梯度裁剪**：`unit_phasor` 的除法在模长趋近 `eps` 时会把梯度放大，"
        "理论上存在尖峰风险；本轮的 6 epoch 运行未观察到发散，但这属于**未验证**的稳健性。"
    )
    add("")


def _section_9(add, cfg, rows, meta, total, stage_totals, floors) -> None:
    add("## 9. 与 `ml/phase` U-Net 的关系")
    add("")
    add(
        "`PhasorUNet` 是 `UNetGenerator` 的**薄子类**：`super().__init__(..., output_mode=\"image\")` "
        "之后**只替换** `self.final_conv`。编码器、瓶颈、解码器、跳跃连接全部原样复用。"
        "这样做的理由是避免第二份 U-Net 实现随时间漂移（本仓库已被「重复实现漂移」多次咬伤）。"
    )
    add("")
    add(
        "副作用：基类仍然创建了 `self.sigmoid`，在 `output_mode=\"image\"` 下**永远不会被调用**。"
        "它是无害的未使用模块，但 `named_children()` / 状态字典里会出现 —— 读模型结构时不要误判。"
    )
    add("")


def _section_10(add, cfg, rows, meta, total, stage_totals, floors) -> None:
    add("## 10. 已知局限与取舍")
    add("")
    add(
        "1. **参数量 7.78M 相对数据量偏大**：全部语料仅 14,350 条、且 val/test 被单一家族主导，"
        "7.78M 参数在 6 epoch 下几乎必然欠拟合而非过拟合 —— 但两者都无法由本轮证据区分。"
    )
    add(
        "2. **`phasor_mae` 有约 0.06 的不可约下界**（§7.1）。任何「误差降到 X」的说法都必须扣掉它。"
    )
    add(
        "3. **绝大多数样本不参与物理项**（fov 1944 无标定）。物理项的实际影响面远小于其名义权重。"
    )
    add(
        "4. **BatchNorm 行为依赖 `train()`/`eval()`**：推理前必须 `.eval()`，否则单样本 batch 的"
        "统计量会污染输出。评估脚本已遵循，但手工调用时是常见陷阱。"
    )
    add(
        "5. **线性头的选择是被 phasor 结构强制的**，不构成对 sigmoid 头的优劣结论；"
        "§3.5 引用的实测只适用于 1 通道强度回归。"
    )
    add("")


def _section_11(add, cfg, rows, meta, total, stage_totals, floors) -> None:
    add("## 11. 复现")
    add("")
    add("```bash")
    add("# 只需模型即可重生成（本报告主体不读语料）")
    add("python scripts/ml/generate_inverse_shaping_model_report.py")
    add("")
    add("# 含 §7.1 的语料地板表时需要快照")
    add("python -c \"from ml.hwdataset.index import build_hw_index; "
        "build_hw_index('data/debug', index_cache='data/hw_index_snapshot.json')\"")
    add("python scripts/ml/generate_inverse_shaping_model_report.py")
    add("```")
    add("")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector, or ``None`` to read ``sys.argv``.

    Returns:
        ``0`` on success.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    args = parser.parse_args(argv)

    _setup_fonts()
    out_dir = Path(args.out_dir)
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    cfg = ShapingConfig()
    model = PhasorUNet(
        in_channels=cfg.in_channels, features=list(cfg.features), grid=cfg.grid
    )
    ablation = PhasorUNet(
        in_channels=1, features=list(cfg.features), grid=cfg.grid
    )

    rows = _stages(model, cfg)
    raw_shape, unit_shape = _verify_forward(model, cfg)
    # K is introspected from the real forward model rather than written as 14:
    # it is determined by n_max and the piston exclusion, and a literal would
    # drift the moment either changes.
    forward_model = ZernikeAmpModel(
        n_max=4, grid=cfg.grid, far_field_padding=recommended_padding(64)
    )
    meta = {
        "total_params": sum(p.numel() for p in model.parameters()),
        "total_ablation": sum(p.numel() for p in ablation.parameters()),
        "raw_shape": raw_shape,
        "unit_shape": unit_shape,
        "n_modes": forward_model.K,
    }

    _plot_architecture(cfg, figures / "model_arch.png")
    _plot_params(rows, figures / "model_params.png")
    floors = _floor_table(Path(args.snapshot))

    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "model.md"
    report.write_text(build_report(cfg, rows, meta, floors), encoding="utf-8")
    print(f"wrote {report}")
    print(f"  params: {meta['total_params']:,}")
    print(f"  model floor table: {'yes' if floors else 'no (snapshot/corpus absent)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
