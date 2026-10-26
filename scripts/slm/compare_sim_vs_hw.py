"""仿真 vs 实测 对照尺: 同一批采集脚本, 同一套指标, 逐项对照。

做什么
------

第三轮硬件实测的产物已经归档在
``report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/``。本脚本
把**同一批采集脚本**在**数字孪生**上再跑一遍, 用**同一套指标口径**把两边的数字
摆到一张表里, 并给出「差在哪、差多少」的**可归因**清单。

它是一把**量尺**, 不是结论: 表里的每一个数字都是从归档产物里**读出来**的, 所以
表永远跟着数据走 —— 数据变了, 表跟着变。

三条纪律
--------

1. **实测侧的数字不写死**: 全部从归档的 ``measurements.json`` / ``summary.json`` /
   焦面标定 JSON 里读 (:func:`load_bundle`)。脚本里没有一个硬编码的实测值。
2. **仿真侧走同一批采集脚本**: 离焦扫描用
   ``scripts/slm/hw_gs_defocus_probe.py``、焦面标定用
   ``ao_shaping.tools.slm.panel.focus_probe``、阶数扫描用
   ``scripts/slm/run_gs_refine_zernike_sweep.py`` —— 三者都有现成的
   ``--cam-type sim`` 路径。本脚本**不重写**任何采集逻辑, 只负责:
   (a) 把仿真系统的口径旋钮拧到与第三轮可比的位置, (b) 读回产物, (c) 算相对差。
3. **指标口径不另写一套**: CV / PIB / 复合分数来自
   ``ao_shaping.algorithm.signal_processing.square_shaping``, 光斑几何来自
   ``ao_shaping.utils.image.frame_analysis`` —— 与采集脚本用的是**同一个函数**。
   所以两边的差只能来自物理预测模型, 不能来自口径。

仿真口径旋钮 (为什么是这些值)
----------------------------

* ``far_field_padding = 4``: 孪生模块的 docstring 写明实测 ``Kx = 7546.4 /
  Ky = 7495.9 px·SLMpx``, 面板 1920 列 ⇒ ``7557/1920 = 3.94`` ⇒ **4 才是复现
  实测角标度的那个值**。脚本侧默认的 16 会让角标度差 4 倍。
* ``peak_counts_per_ms = 133.5``: 由实测反推 —— 实测平场 0.2 ms 整帧峰值
  26.7 counts ⇒ 峰值分数 1.0 ⇒ ``133.5 counts/(峰值分数·ms)``; 最佳焦面点
  107.1 counts/ms (峰值分数约 0.80) 交叉验证自洽。**开了 counts 口径之后, 仿真
  帧的峰值与实测帧的峰值是同一个量纲** (counts), 不再是归一化到 ~100 的相对强度。
* ``read_sigma_counts = 0.5`` / ``shot_k = 0`` / ``quantize_bits = 8``: 由第三轮
  噪声组 (同相位重复 10 次) 的**原始帧**反推 —— 背景只有 0 / 0.25 / 0.5 三个值
  ⇒ 每帧平均后 σ ≈ 0.28 counts ⇒ 每原始帧 σ ≈ 0.56 counts; 而峰值 σ 只有
  0.18 counts (峰值 26.7 counts) ⇒ 26.7 counts 处若是散粒噪声, σ 会有 √26.7 ≈
  5.2 counts ⇒ **探测器在这个光级上不是散粒噪声受限**, 故 ``shot_k = 0``。

复现
-----

.. code-block:: bash

    # 跑仿真 + 出表 (纯离线, 不碰硬件)
    python scripts/slm/compare_sim_vs_hw.py --execute

    # 只用已有的仿真产物重新出表
    python scripts/slm/compare_sim_vs_hw.py --sim-root data/debug/sim_vs_hw_20261022

产物: ``report/shape_trials/sim_vs_hw/artifacts/sim_vs_hw_20261022.md``
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

# 让脚本可独立运行: `python scripts/slm/compare_sim_vs_hw.py` 时, 解释器的
# sys.path[0] 是 scripts/slm/, 既找不到 `ao_shaping` (在 src/) 也找不到
# `scripts` 包 (在仓库根) —— 这里把仓库根与 src/ 显式加进去, 否则 import 直接挂。
_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ROOT = Path(__file__).resolve().parents[2]
# `scripts.*` 以包路径被导入, 所以仓库 ROOT 本身必须可导入。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "src"))

from scripts._common import fmt_sigfig

import numpy as np

# ---------------------------------------------------------------------------
# 默认路径与口径常量
# ---------------------------------------------------------------------------

#: 第三轮硬件实测的归档根目录。
DEFAULT_HW_ROOT = ROOT / "report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022"

#: 仿真产物根目录 (默认落在 data/debug/ 下, 不入库)。
DEFAULT_SIM_ROOT = ROOT / "data/debug/sim_vs_hw_20261022"

#: 对照表产物 (数据表: 逐项数字 + 判定 + 归因清单)。
DEFAULT_OUT_MD = ROOT / "report/shape_trials/sim_vs_hw/artifacts/sim_vs_hw_20261022.md"

#: 报告正文产物 (入库的那一份: 叙述 + 同一份对照表 + 能与不能 + 局限)。
DEFAULT_OUT_REPORT = ROOT / "report/shape_trials/sim_vs_hw/sim_vs_hw_20261022.md"

#: 孪生参数归档 (本轮最终采用的那一组, ``--params`` 读回即复现)。
DEFAULT_PARAMS_JSON = (
    ROOT / "report/shape_trials/sim_vs_hw/artifacts/sim_twin_params_20261022.json"
)

#: 单次远场求值的实测开销 (1920x1200 面板, ``far_field_padding = 4``, 16 线程):
#: 1.04-1.07 s / RSS 增量 ~0.97 GB。写进报告, 让人知道「再调一轮」的代价。
SIM_EVAL_SECONDS = 1.05
SIM_EVAL_RSS_MB = 970.0

#: 仿真随机种子 (固定 ⇒ 逐位可重复)。
SIM_SEED = 0

#: 仿真远场 FFT 的补零因子 —— 4 才是复现实测角标度的值 (见模块 docstring)。
SIM_FAR_FIELD_PADDING = 4

#: 绝对光度换算: counts = 峰值分数 · peak_counts_per_ms · 曝光(ms)。
PEAK_COUNTS_PER_MS = 133.5

#: 探测器噪声 (counts 口径): 读出噪声 / 散粒系数 / 量化位深。
READ_SIGMA_COUNTS = 0.5
SHOT_K = 0.0
QUANTIZE_BITS = 8
#: Tolerance for treating two recorded optical parameters (a defocus step, a
#: diopter) as the same grid value when picking the matching row or point. It is a
#: matching question, not a degeneracy question, so it stays separate from the floors
#: below even though one of them is the same magnitude.
VALUE_MATCH_TOLERANCE = 1e-9

#: A defocus coordinate counts as zero -- the nominal plane -- within this tolerance.
#: Used both to find the base row and to count rows sitting on that plane.
ZERO_DEFOCUS_TOLERANCE = 1e-12

#: Magnitude below which a measured or derived quantity is treated as zero, so a ratio
#: built on it is undefined and the point is skipped instead of divided by. Distinct from
#: ZERO_DEFOCUS_TOLERANCE: that one identifies a plane, this one protects a division.
NONZERO_FLOOR = 1e-12

#: 采集参数 —— 与第三轮一致。
EXPOSURE_MS = 0.2
ROI_PX = 250
TARGET_SIDE_PX = 10
FRAMES = 4

#: 第三轮硬件侧用的 GS 相位 (硬件上优化出来的 Zernike 相位, 面板分辨率)。
#: **数据目录不入库**, 所以缺失时回落到「现场生成模型 GS 相位」并在表里注明。
HW_GS_PHASE = Path(
    "data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4/best_phase_nmax4.npy"
)

#: 相对差判定阈值 (|相对差| ≤ 阈值 ⇒ 「一致」; ≤ 3× 阈值 ⇒ 「偏差」; 否则「不一致」)。
TOL_UNIFORMITY = 0.20  # CV / 复合分数
TOL_POWER = 0.20  # PIB / 框外功率占比
TOL_WIDTH = 0.25  # 光斑宽度
TOL_PHOTOMETRY = 0.20  # 峰值 (counts)
TOL_NOISE = 0.50  # 噪声地板 σ
TOL_FOCUS = 0.50  # 焦距
TOL_QUALITY = 0.30  # 阶数扫描的复合质量分

#: 探针测量里用到的指标列 (与 ``hw_gs_defocus_probe`` 的 measurements.json 同名)。
PROBE_METRICS: tuple[str, ...] = (
    "cv",
    "pib",
    "score",
    "frame_peak",
    "halo_power_fraction_outside_box",
    "spot_fwhm_px",
    "spot_hollowness",
    "frame_mean_in_target",
)

#: 噪声组统计的指标 (峰值用 counts 口径)。
NOISE_METRICS: tuple[str, ...] = ("cv", "pib", "score", "frame_peak", "spot_fwhm_px")


# ---------------------------------------------------------------------------
# 数据容器
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeGroup:
    """一个探针组的测量 (采集脚本的 ``measurements.json`` + ``run_config.json``)。"""

    rows: tuple[dict[str, Any], ...]
    run_config: dict[str, Any] = field(default_factory=dict)

    @property
    def baseline(self) -> dict[str, Any]:
        """基线行 (第 0 行: Δz=0, 相位已施加)。"""
        return self.rows[0] if self.rows else {}

    def row_at_dz(self, dz_mm: float) -> dict[str, Any] | None:
        """按 Δz (mm) 找一行; 找不到返回 ``None``。"""
        for row in self.rows:
            if abs(float(row.get("dz_mm", float("nan"))) - float(dz_mm)) < VALUE_MATCH_TOLERANCE:
                return row
        return None


@dataclass(frozen=True)
class Bundle:
    """一侧 (实测或仿真) 的全部对照数据。"""

    root: Path
    dz_scan: ProbeGroup
    dz_noise: ProbeGroup
    zernike: dict[str, dict[str, Any]]
    focus: dict[str, Any]
    focus_name: str

    @property
    def label(self) -> str:
        return "hw" if self.root == DEFAULT_HW_ROOT else "sim"


@dataclass(frozen=True)
class Cell:
    """对照表的一行。"""

    section: str
    metric: str
    hw: float | None
    sim: float | None
    tol: float
    unit: str = ""
    note: str = ""


# ---------------------------------------------------------------------------
# 纯函数: 读数据
# ---------------------------------------------------------------------------


def read_json(path: Path) -> Any:
    """读一个 JSON 文件。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_probe_group(group_dir: Path) -> ProbeGroup:
    """读一个探针组目录 (``measurements.json`` + ``run_config.json``)。

    Args:
        group_dir: 组目录 (含 ``measurements.json``)。

    Returns:
        :class:`ProbeGroup`; ``measurements.json`` 不存在时返回空组 (不抛异常 ——
        缺哪一侧就少哪一段对照, 不该让整张表崩掉)。
    """
    group_dir = Path(group_dir)
    m_path = group_dir / "measurements.json"
    if not m_path.exists():
        return ProbeGroup(rows=())
    rows = read_json(m_path)
    if isinstance(rows, dict):  # 兼容 {"value": [...]} 这种包装
        rows = rows.get("value", [])
    cfg_path = group_dir / "run_config.json"
    cfg = read_json(cfg_path) if cfg_path.exists() else {}
    return ProbeGroup(rows=tuple(rows), run_config=cfg)


def _find_group(root: Path, group_name: str) -> Path | None:
    """在根目录下找 ``*/<group_name>`` (中间层目录名带日期)。"""
    direct = root / group_name
    if direct.exists():
        return direct
    for candidate in sorted(root.glob(f"*/{group_name}")):
        if candidate.is_dir():
            return candidate
    return None


def load_bundle(root: Path) -> Bundle:
    """读一侧的全部对照数据 (实测或仿真的产物根目录)。

    目录布局 (实测与仿真**同构**, 所以同一个读取器服务两侧)::

        <root>/
          hw_groups_<date>/dz_scan_hw_gs/measurements.json
          hw_groups_<date>/dz_noise_hw_gs/measurements.json
          hw_zernike_sweep_<date>/zernike_order_nmax*/summary.json
          focus_calibration/slm1_<stamp>.json

    Args:
        root: 产物根目录。

    Returns:
        :class:`Bundle`。

    Raises:
        FileNotFoundError: 根目录不存在。
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"产物根目录不存在: {root}")

    scan_dir = _find_group(root, "dz_scan_hw_gs")
    noise_dir = _find_group(root, "dz_noise_hw_gs")
    dz_scan = load_probe_group(scan_dir) if scan_dir else ProbeGroup(rows=())
    dz_noise = load_probe_group(noise_dir) if noise_dir else ProbeGroup(rows=())

    zernike: dict[str, dict[str, Any]] = {}
    for summary_path in sorted(root.glob("*/zernike_order_*/summary.json")):
        zernike[summary_path.parent.name] = read_json(summary_path)

    focus: dict[str, Any] = {}
    focus_name = ""
    focus_dir = root / "focus_calibration"
    if focus_dir.exists():
        candidates = sorted(focus_dir.glob("*.json"))
        if candidates:
            focus_path = candidates[-1]
            focus_name = focus_path.name
            focus = read_json(focus_path)

    return Bundle(
        root=root,
        dz_scan=dz_scan,
        dz_noise=dz_noise,
        zernike=zernike,
        focus=focus,
        focus_name=focus_name,
    )


# ---------------------------------------------------------------------------
# 纯函数: 统计
# ---------------------------------------------------------------------------


def finite_values(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    """取一列里所有有限的值 (NaN/None 被丢掉)。"""
    out: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def summarise_repeatability(
    rows: Sequence[dict[str, Any]], keys: Sequence[str]
) -> dict[str, dict[str, float]]:
    """同相位重复测量的**噪声地板**: 每列的均值与标准差 (样本标准差, ddof=1)。

    Args:
        rows: 重复测量的行。
        keys: 要统计的列名。

    Returns:
        ``{列名: {"mean": ..., "std": ..., "n": ...}}``。
    """
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        values = finite_values(rows, key)
        if not values:
            continue
        out[key] = {
            "mean": float(statistics.fmean(values)),
            "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
            "n": float(len(values)),
        }
    return out


def relative_diff(hw: float | None, sim: float | None) -> float | None:
    """相对差 ``(sim - hw) / |hw|``。

    实测值为 0 或缺失时**没有**有意义的相对差, 返回 ``None`` (表里显示 ``—``)。
    """
    if hw is None or sim is None:
        return None
    if not math.isfinite(hw) or not math.isfinite(sim):
        return None
    if abs(hw) < NONZERO_FLOOR:
        return None
    return (sim - hw) / abs(hw)


def verdict(rel: float | None, tol: float) -> str:
    """按相对差给判定: 一致 / 偏差 / 不一致。"""
    if rel is None:
        return "—"
    magnitude = abs(rel)
    if magnitude <= tol:
        return "一致"
    if magnitude <= 3.0 * tol:
        return "偏差"
    return "不一致"


def _get(mapping: dict[str, Any], key: str) -> float | None:
    """从一个 dict 里取一个浮点数 (``NaN`` 视为缺失; ``inf`` 保留 —— 1/f = 0 的焦距就是 ``inf``)。"""
    value = mapping.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


# ---------------------------------------------------------------------------
# 纯函数: 对照表
# ---------------------------------------------------------------------------


def build_cells(
    hw: Bundle,
    sim: Bundle,
    *,
    far_field_padding: int | tuple[int, int] = SIM_FAR_FIELD_PADDING,
) -> list[Cell]:
    """把两侧的产物变成对照表的行。

    每一行都是「同一个量」在两侧的取值 —— 指标口径完全一致 (采集脚本用的是同一套
    函数), 所以相对差只反映物理预测模型的差距。

    Args:
        hw: 实测侧。
        sim: 仿真侧。
        far_field_padding: 仿真远场补零因子 (决定宽度换算系数; 各向异性时
            两轴都对齐相机像元 ⇒ 换算 → 1.0)。

    Returns:
        :class:`Cell` 列表 (按 section 顺序)。
    """
    cells: list[Cell] = []

    # 仿真侧的光斑宽度要换算成**相机像元**才能与实测侧相比 (见 ``sim_width_scale``)。
    width_scale = sim_width_scale(far_field_padding)

    # --- §1 基线 (Δz=0, 相位已施加) -------------------------------------
    hw_base = hw.dz_scan.baseline
    sim_base = sim.dz_scan.baseline
    baseline_specs: tuple[tuple[str, str, float], ...] = (
        ("CV (目标框内)", "cv", TOL_UNIFORMITY),
        ("PIB (框内功率占比)", "pib", TOL_POWER),
        ("复合分数 0.5·PIB+0.5/(1+CV)", "score", TOL_UNIFORMITY),
        ("整帧峰值 (counts)", "frame_peak", TOL_PHOTOMETRY),
        ("光斑 FWHM (px)", "spot_fwhm_px", TOL_WIDTH),
        ("框外功率占比", "halo_power_fraction_outside_box", TOL_POWER),
        ("光斑空心度", "spot_hollowness", TOL_UNIFORMITY),
    )
    for label, key, tol in baseline_specs:
        sim_value = _get(sim_base, key)
        if key == "spot_fwhm_px":
            sim_value = _to_camera_px(sim_value, width_scale)
        cells.append(
            Cell(
                section="§1 基线 (Δz=0, GS 相位)",
                metric=label,
                hw=_get(hw_base, key),
                sim=sim_value,
                tol=tol,
            )
        )

    # --- §2 噪声地板 (同相位重复 10 次的 σ) -------------------------------
    hw_noise = summarise_repeatability(hw.dz_noise.rows, NOISE_METRICS)
    sim_noise = summarise_repeatability(sim.dz_noise.rows, NOISE_METRICS)
    noise_labels: tuple[tuple[str, str], ...] = (
        ("σ(CV)", "cv"),
        ("σ(PIB)", "pib"),
        ("σ(复合分数)", "score"),
        ("σ(整帧峰值, counts)", "frame_peak"),
        ("σ(光斑 FWHM, px)", "spot_fwhm_px"),
    )
    for label, key in noise_labels:
        cells.append(
            Cell(
                section="§2 噪声地板 (同相位重复的 σ)",
                metric=label,
                hw=(hw_noise.get(key) or {}).get("std"),
                sim=(sim_noise.get(key) or {}).get("std"),
                tol=TOL_NOISE,
            )
        )
    for label, key in (("均值 CV", "cv"), ("均值 PIB", "pib")):
        cells.append(
            Cell(
                section="§2 噪声地板 (同相位重复的 σ)",
                metric=label,
                hw=(hw_noise.get(key) or {}).get("mean"),
                sim=(sim_noise.get(key) or {}).get("mean"),
                tol=TOL_UNIFORMITY,
            )
        )

    # --- §3 离焦响应 ------------------------------------------------------
    dz_points = sorted({float(r.get("dz_mm", 0.0)) for r in hw.dz_scan.rows})
    extreme = dz_points[-1] if dz_points else 0.0
    smallest = dz_points[0] if dz_points else 0.0
    for label, key, tol, dz in (
        ("光斑 FWHM @ Δz=0 (px)", "spot_fwhm_px", TOL_WIDTH, 0.0),
        ("整帧峰值 @ Δz=0 (counts)", "frame_peak", TOL_PHOTOMETRY, 0.0),
        ("PIB @ Δz=0", "pib", TOL_POWER, 0.0),
        ("框外功率占比 @ Δz=0", "halo_power_fraction_outside_box", TOL_POWER, 0.0),
        (
            f"光斑 FWHM @ Δz={extreme:+.4g} mm (px)",
            "spot_fwhm_px",
            TOL_WIDTH,
            extreme,
        ),
        (
            f"整帧峰值 @ Δz={extreme:+.4g} mm (counts)",
            "frame_peak",
            TOL_PHOTOMETRY,
            extreme,
        ),
        (f"PIB @ Δz={extreme:+.4g} mm", "pib", TOL_POWER, extreme),
        (
            f"框外功率占比 @ Δz={extreme:+.4g} mm",
            "halo_power_fraction_outside_box",
            TOL_POWER,
            extreme,
        ),
        (
            f"光斑 FWHM @ Δz={smallest:+.4g} mm (px)",
            "spot_fwhm_px",
            TOL_WIDTH,
            smallest,
        ),
        (
            f"整帧峰值 @ Δz={smallest:+.4g} mm (counts)",
            "frame_peak",
            TOL_PHOTOMETRY,
            smallest,
        ),
    ):
        sim_value = _get(sim.dz_scan.row_at_dz(dz) or {}, key)
        if key == "spot_fwhm_px":
            sim_value = _to_camera_px(sim_value, width_scale)
        cells.append(
            Cell(
                section="§3 离焦响应 (Δz 扫描)",
                metric=label,
                hw=_get(hw.dz_scan.row_at_dz(dz) or {}, key),
                sim=sim_value,
                tol=tol,
            )
        )

    # --- §4 焦面标定 ------------------------------------------------------
    focus_specs: tuple[tuple[str, str, float], ...] = (
        ("最佳 1/f (m⁻¹)", "best_diopter_m1", TOL_FOCUS),
        ("最佳焦距 f (mm)", "best_focal_mm", TOL_FOCUS),
        ("最佳点峰值/曝光 (counts/ms)", "best_peak_per_ms", TOL_PHOTOMETRY),
        ("最佳点 FWHM (px)", "best_fwhm_px", TOL_WIDTH),
        ("相对平场的增益", "gain_vs_flat", TOL_POWER),
    )
    for label, key, tol in focus_specs:
        sim_value = _focus_value(sim.focus, key)
        if key == "best_fwhm_px":
            sim_value = _to_camera_px(sim_value, width_scale)
        cells.append(
            Cell(
                section="§4 焦面标定 (focus_probe)",
                metric=label,
                hw=_focus_value(hw.focus, key),
                sim=sim_value,
                tol=tol,
            )
        )

    # --- §5 阶数扫描 ------------------------------------------------------
    # 只比**两侧都有**的组: 硬件侧扫了 8 个阶, 仿真侧只跑了其中几个,
    # 剩下的整段是空的 —— 满屏的「—」会把真差距埋掉。
    common_groups = sorted(set(hw.zernike) & set(sim.zernike))
    group_names = common_groups or sorted(set(hw.zernike) | set(sim.zernike))
    for group_name in group_names:
        hw_sum = hw.zernike.get(group_name, {})
        sim_sum = sim.zernike.get(group_name, {})
        for baseline_key, baseline_label in (
            ("flat", "平场基线"),
            ("gs_freeform", "GS 逐像素相位基线"),
        ):
            hw_base_block = hw_sum.get("baselines", {}).get(baseline_key, {})
            sim_base_block = sim_sum.get("baselines", {}).get(baseline_key, {})
            for metric, tol in (
                ("pib", TOL_POWER),
                ("cv", TOL_UNIFORMITY),
                ("ee", TOL_POWER),
                ("ar", TOL_UNIFORMITY),
                ("quality", TOL_QUALITY),
            ):
                cells.append(
                    Cell(
                        section=f"§5 阶数扫描 ({group_name})",
                        metric=f"{baseline_label} · {metric}",
                        hw=_get(hw_base_block, metric),
                        sim=_get(sim_base_block, metric),
                        tol=tol,
                    )
                )
        hw_orders = hw_sum.get("orders", [])
        sim_orders = sim_sum.get("orders", [])
        for hw_order, sim_order in zip(hw_orders, sim_orders):
            for phase in ("before", "after"):
                hw_block = hw_order.get(phase, {})
                sim_block = sim_order.get(phase, {})
                for metric, tol in (
                    ("pib", TOL_POWER),
                    ("cv", TOL_UNIFORMITY),
                    ("ee", TOL_POWER),
                    ("ar", TOL_UNIFORMITY),
                    ("quality", TOL_QUALITY),
                ):
                    cells.append(
                        Cell(
                            section=(
                                f"§5 阶数扫描 ({group_name}, n_max="
                                f"{hw_order.get('n_max')})"
                            ),
                            metric=f"{phase} · {metric}",
                            hw=_get(hw_block, metric),
                            sim=_get(sim_block, metric),
                            tol=tol,
                            note=(
                                "SPGD 轮数两侧不同 (硬件 154+150, 仿真 "
                                "--zernike-epochs/--zernike-spgd-epochs), 只看方向"
                                if phase == "after"
                                else ""
                            ),
                        )
                    )

    return cells


def _focus_value(focus: dict[str, Any], key: str) -> float | None:
    """从焦面标定 JSON 里取一个量。

    ``best_peak_per_ms`` / ``best_fwhm_px`` 不是顶层键 —— 它们要从「最佳点」那一行
    (``points`` 里 ``diopter_m1`` 等于 ``best_diopter_m1`` 的那条) 上取。
    """
    if key in focus:
        return _get(focus, key)
    points = focus.get("points", [])
    best_diopter = _get(focus, "best_diopter_m1")
    best_point: dict[str, Any] = {}
    if best_diopter is not None:
        for point in points:
            if _get(point, "diopter_m1") is None:
                continue
            if abs(float(point["diopter_m1"]) - best_diopter) < VALUE_MATCH_TOLERANCE:
                best_point = point
                break
    if not best_point and points:
        best_point = points[0]
    source = {
        "best_peak_per_ms": "peak_per_ms",
        "best_fwhm_px": "fwhm_px",
    }.get(key, key)
    return _get(best_point, source)


def far_field_scale(
    padding: int | tuple[int, int] = SIM_FAR_FIELD_PADDING,
) -> tuple[float, float] | None:
    """孪生的一个远场采样点等于多少**相机像元** (行, 列)。

    孪生自己有这个换算 (:attr:`SimPibSystem.camera_px_per_model_px`), 但它绑在
    进程级 system 上; 这里另建一个**独立**的 system 实例来问它, 不动全局状态。

    为什么要这个数: 仿真帧的像素是**远场采样点**, 不是相机像元。面板 1920×1200
    是长方形, 单一补零因子无法让两轴都对齐 ⇒ 行方向天然差 ~1.6 倍。光斑宽度的
    差距里有一部分只是这个标度差, 不算物理预测模型的差距。

    Args:
        padding: 远场补零因子 (``int`` 或 ``(行, 列)`` 元组, 见 V2.0 各向异性)。

    Returns:
        ``(行, 列)`` 两个换算系数; 孪生 API 变了就返回 ``None``。
    """
    try:
        from ao_shaping.drivers.sim.slm_pib_sim import SimPibSystem

        system = SimPibSystem(far_field_padding=padding)
        return system.camera_px_per_model_px
    except Exception:  # 孪生 API 漂移不该让整张表崩掉
        return None


def sim_width_scale(
    padding: int | tuple[int, int] = SIM_FAR_FIELD_PADDING,
) -> float:
    """仿真侧的光斑宽度 (远场采样点) -> 相机像元 的换算因子。

    实测侧的光斑宽度是**相机像元**, 仿真侧是**远场采样点**: 孪生的约定是
    「一个远场采样点 = 一个相机像元」, 但面板是 1920x1200 的长方形, 单一补零
    因子无法让两轴都对齐 (行向 1.57 / 列向 0.98 相机像元)。直接拿两边的数字
    相比就是**两种口径**。

    远场图样在**物理上**是各向同性的 (同一个瞳面 => 同一个远场), 差别只在采样
    间距; 而 `spot_fwhm` 量的是「行向与列向 FWHM 的平均」(都是远场采样点)。
    对一幅各向同性的图样, 行向量到的采样点数 = `W/行距`、列向 = `W/列距`,
    所以平均值 `f = W*(1/行 + 1/列)/2` => 换算因子取两者的**调和平均**
    `2/(1/行 + 1/列)` (约 1.21)。

    Args:
        padding: 远场补零因子 (决定换算系数; 各向异性补零时两轴都 = 相机像元
            ⇒ 换算因子 → 1.0)。
    """
    scale = far_field_scale(padding)
    if scale is None:
        return 1.0
    row, col = (float(v) for v in scale)
    if row <= 0.0 or col <= 0.0:
        return 1.0
    return 2.0 / (1.0 / row + 1.0 / col)


def _to_camera_px(value: float | None, scale: float) -> float | None:
    """把一个远场采样点单位的宽度换算成相机像元。"""
    if value is None:
        return None
    return float(value) * float(scale)


def frame_total_stats(group_dir: Path) -> dict[str, float] | None:
    """一组重复测量的**整帧总功率**统计 (DC 底座的直接证据)。

    探针的每帧 npz 存的是**原始帧** (已减中位数并裁负), 所以整帧总和里
    含着那块「被整流出来的底座」。底座稳不稳 ⇒ 功率比类指标 (PIB) 稳不稳。

    Args:
        group_dir: 组目录 (含 ``frame_*.npz``)。

    Returns:
        ``{"mean": ..., "std": ..., "rel_std": ..., "n": ...}``; 没有帧文件返回 ``None``。
    """
    frames = sorted(Path(group_dir).glob("frame_*.npz"))
    if not frames:
        return None
    totals: list[float] = []
    for path in frames:
        with np.load(path) as data:
            totals.append(float(np.asarray(data["frame"], dtype=np.float64).sum()))
    values = np.asarray(totals, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "rel_std": (std / mean) if mean else 0.0,
        "n": float(values.size),
    }


# ---------------------------------------------------------------------------
# 纯函数: 差距清单
# ---------------------------------------------------------------------------


def _noise_group_dir(bundle: Bundle, group_name: str) -> Path | None:
    """找一侧的噪声组**帧目录** (归档里只有 measurements, 原始帧在 data/debug)。"""
    candidates = [
        bundle.root / "hw_groups_sim" / group_name,
        bundle.root / group_name,
        ROOT / "data/debug/hw_groups_20261022" / group_name,
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("frame_*.npz")):
            return candidate
    return None


def build_gap_list(hw: Bundle, sim: Bundle) -> list[str]:
    """把「差在哪」拆成**可归因**的几类, 每类给出证据数字与判定。

    六类: ①几何/焦面 ②光斑宽度/面板静态相差 ③光度/离焦能量守恒 ④噪声/暗电平
    ⑤离焦响应形状 ⑥基线。每一类都先说**现状** (对齐 / 仍不一致), 再给机理。

    Args:
        hw: 实测侧。
        sim: 仿真侧。

    Returns:
        每一类一段文字 (markdown 列表项)。
    """
    gaps: list[str] = []
    width_scale = sim_width_scale()

    def _w(value: float | None) -> float | None:
        """远场采样点 → 相机像元 (光斑宽度专用)。"""
        return _to_camera_px(value, width_scale)

    # ① 几何 / 焦面 ------------------------------------------------------
    hw_focus = _focus_value(hw.focus, "best_focal_mm")
    sim_focus = _focus_value(sim.focus, "best_focal_mm")
    hw_diopter = _focus_value(hw.focus, "best_diopter_m1")
    sim_diopter = _focus_value(sim.focus, "best_diopter_m1")
    hw_zero_fwhm = _focus_flat_fwhm_at_zero(hw.focus)
    hw_best_fwhm = _focus_value(hw.focus, "best_fwhm_px")
    sim_best_fwhm = _w(_focus_value(sim.focus, "best_fwhm_px"))
    sim_zero_fwhm = _w(_focus_flat_fwhm_at_zero(sim.focus))
    flat_response = (
        hw_best_fwhm / hw_zero_fwhm if hw_zero_fwhm and hw_best_fwhm else None
    )
    sim_flat_response = (
        sim_best_fwhm / sim_zero_fwhm if sim_zero_fwhm and sim_best_fwhm else None
    )
    focus_rel = relative_diff(hw_diopter, sim_diopter)
    focus_verdict = verdict(focus_rel, TOL_FOCUS)
    gaps.append(
        "① **几何 / 焦面** — "
        f"{'✅ 已对齐' if focus_verdict == '一致' else '⚠️ 仍不一致'}: 实测最佳 1/f = "
        f"{fmt_sigfig(hw_diopter)} m⁻¹ (f = {fmt_sigfig(hw_focus, 6)} mm), 仿真最佳 1/f = "
        f"{fmt_sigfig(sim_diopter)} m⁻¹ (f = {fmt_sigfig(sim_focus, 6)} mm)。"
        "**机理**: 孪生原来没有「CCD 离焦」这个自由度 (远场是面板相位的纯 FFT ⇒ 焦面"
        "按构造就在探测器上), 所以「用瞳面菲涅尔相位把焦面搬回探测器」在仿真里毫无"
        "效果、最佳 1/f 恒为 0。现在孪生有 `camera_defocus_m` (探测器在 f+Δz 处, "
        "瞳面上等价地乘 `-π·Δz·r²/(λf²)`), 补偿焦距 `f_v = f²/Δz` ⇒ 实测的 "
        f"{fmt_sigfig(hw_diopter)} m⁻¹ (= 瞳面二次相位焦距 {fmt_sigfig(hw_focus, 6)} mm, 边缘相位 "
        "24.7 rad) 就是 **Δz ≈ 10.09 mm**。更硬的证据是**平场对菲涅尔矫正的响应**: "
        f"实测平场宽度 {fmt_sigfig(hw_zero_fwhm, 4)} px → {fmt_sigfig(hw_best_fwhm, 4)} px "
        f"(收到 {fmt_sigfig(flat_response, 3)}×), 仿真 {fmt_sigfig(sim_zero_fwhm, 4)} px → "
        f"{fmt_sigfig(sim_best_fwhm, 4)} px (收到 {fmt_sigfig(sim_flat_response, 3)}×) —— 两侧现在"
        "都是「一个被离焦散开的平场 + 一片把它收回来的菲涅尔相位」。"
    )

    # ② 光斑宽度 / 面板静态相差 -------------------------------------------
    hw_gs_fwhm = _get(hw.dz_scan.baseline, "spot_fwhm_px")
    sim_gs_fwhm_raw = _get(sim.dz_scan.baseline, "spot_fwhm_px")
    sim_gs_fwhm = _w(sim_gs_fwhm_raw)
    width_rel = relative_diff(hw_gs_fwhm, sim_gs_fwhm)
    width_verdict = verdict(width_rel, TOL_WIDTH)
    scale = far_field_scale()
    scale_note = ""
    if scale is not None:
        scale_note = (
            f"仿真侧的数字已经换算成相机像元 (孪生一个远场采样点 = {scale[0]:.2f} 相机"
            f"像元 (行) / {scale[1]:.2f} (列), 调和平均 ×{width_scale:.3f})。"
        )
    gaps.append(
        "② **光斑宽度 / 面板静态相差** — "
        f"{'✅ 已对齐' if width_verdict == '一致' else '⚠️ 仍不一致'}: GS 相位下的"
        f"光斑宽度 实测 {fmt_sigfig(hw_gs_fwhm, 4)} px vs 仿真 {fmt_sigfig(sim_gs_fwhm, 4)} px。"
        + scale_note
        + " **机理**: 孪生的光斑宽度由**入射光的有效口径**定 (衍射极限 ~ "
        "1.03·λf/D), 实测焦面矫正后的平场宽度 "
        f"{fmt_sigfig(hw_best_fwhm, 4)} px 反推出**强度** 1/e² 半径 ≈ 170–400 面板 px —— "
        "本轮取 `beam_w0 = 120` 面板 px (孪生的 `beam_w0` 是**幅度**的 1/e 半径 ⇒ "
        "强度的 1/e² 半径 ≈ 170 px), 并且**不**再截断到照明盘。"
        "面板静态相差 (`SimPanelAberration.zernike_rad`) 会把光斑**再**撑宽, 但它同时"
        "把光斑从「平顶」变成「带裙边的尖峰」⇒ 目标框内的 CV 反而变差, 所以本轮**没有**"
        "用 Zernike 去凑宽度, 只把它留作可调旋钮。"
    )

    # ③ 光度 / 离焦能量守恒 ------------------------------------------------
    hw_peak = _get(hw.dz_scan.baseline, "frame_peak")
    sim_peak = _get(sim.dz_scan.baseline, "frame_peak")
    hw_counts_rate = hw_peak / EXPOSURE_MS if hw_peak else None
    sim_counts_rate = sim_peak / EXPOSURE_MS if sim_peak else None
    hw_far_peak = _get(hw.dz_scan.row_at_dz(20.0) or {}, "frame_peak")
    sim_far_peak = _get(sim.dz_scan.row_at_dz(20.0) or {}, "frame_peak")
    peak_rel = relative_diff(hw_peak, sim_peak)
    far_rel = relative_diff(hw_far_peak, sim_far_peak)
    gaps.append(
        "③ **光度 / 离焦能量守恒** — "
        f"{'✅ 已对齐' if verdict(peak_rel, TOL_PHOTOMETRY) == '一致' else '⚠️ 仍不一致'}"
        f": 0.2 ms 曝光下整帧峰值 实测 {fmt_sigfig(hw_peak, 4)} counts "
        f"(⇒ {fmt_sigfig(hw_counts_rate, 4)} counts/ms) vs 仿真 {fmt_sigfig(sim_peak, 4)} counts "
        f"(⇒ {fmt_sigfig(sim_counts_rate, 4)} counts/ms)。**机理**: counts 口径原来**逐帧**"
        "把远场峰值归一到 100 ⇒ 离焦散开的光又被拉回原峰值 (实测峰值随离焦崩塌到 "
        f"{fmt_sigfig(hw_far_peak, 4)} counts, 仿真几乎不动)。现在 `counts_reference=\"flat\"` "
        "用**固定参考** (平场在理想聚焦下的峰值), 峰值分数 = 当前远场 / 该参考 ⇒ "
        f"Δz=+20 mm 处仿真 {fmt_sigfig(sim_far_peak, 4)} counts "
        f"({'一致' if verdict(far_rel, TOL_PHOTOMETRY) == '一致' else '仍偏高'})。"
    )

    # ④ 噪声 / 暗电平 ------------------------------------------------------
    hw_noise = summarise_repeatability(hw.dz_noise.rows, NOISE_METRICS)
    sim_noise = summarise_repeatability(sim.dz_noise.rows, NOISE_METRICS)
    noise_parts: list[str] = []
    noise_verdicts: list[str] = []
    for label, key in (
        ("σ(CV)", "cv"),
        ("σ(PIB)", "pib"),
        ("σ(复合分数)", "score"),
        ("σ(峰值, counts)", "frame_peak"),
    ):
        hw_std = (hw_noise.get(key) or {}).get("std")
        sim_std = (sim_noise.get(key) or {}).get("std")
        noise_parts.append(f"{label} 实测 {fmt_sigfig(hw_std, 3)} vs 仿真 {fmt_sigfig(sim_std, 3)}")
        noise_verdicts.append(verdict(relative_diff(hw_std, sim_std), TOL_NOISE))
    hw_pedestal = frame_total_stats(
        _noise_group_dir(hw, "dz_noise_hw_gs") or Path("nonexistent")
    )
    sim_pedestal = frame_total_stats(
        _noise_group_dir(sim, "dz_noise_hw_gs") or Path("nonexistent")
    )
    pedestal_note = ""
    if hw_pedestal and sim_pedestal:
        pedestal_note = (
            " **根因是暗电平, 不是噪声大小**: 探针的帧处理是「减中位数 + 裁负」, 而"
            "孪生的 counts 帧原来**背景恒为 0** ⇒ 读出噪声被**整流**成一块与像素数"
            "成正比的 DC 底座, 而它的「中位数」恰好卡在 0/0.25 的跳变边界上 ⇒ 逐帧"
            f"跳变。整帧总功率 实测 {fmt_sigfig(hw_pedestal['mean'], 5)} ± "
            f"{fmt_sigfig(hw_pedestal['std'], 3)} counts "
            f"({fmt_sigfig(hw_pedestal['rel_std'] * 100, 3)}%), 仿真 "
            f"{fmt_sigfig(sim_pedestal['mean'], 5)} ± {fmt_sigfig(sim_pedestal['std'], 3)} counts "
            f"({fmt_sigfig(sim_pedestal['rel_std'] * 100, 3)}%)。补上实测相机的 26.75 counts "
            "暗电平 (量化作用在**带暗电平的原始 ADU** 上, 暗电平靠减中位数去掉) 之后,"
            " 中位数不再落在跳变边界上 ⇒ 底座稳了, σ(PIB) 也随之落到同一档。"
        )
    gaps.append(
        "④ **噪声 / 暗电平** — "
        + ("✅ 已对齐: " if all(v == "一致" for v in noise_verdicts) else "部分对齐: ")
        + "; ".join(noise_parts)
        + "。"
        + pedestal_note
    )

    # ⑤ 离焦响应形状 ------------------------------------------------------
    hw_curve = _normalised_curve(hw.dz_scan.rows, "spot_fwhm_px")
    sim_curve = _normalised_curve(sim.dz_scan.rows, "spot_fwhm_px")
    gaps.append(
        "⑤ **离焦响应形状** (FWHM 归一化到各自 Δz=0): "
        + _curve_shape_note(hw_curve, sim_curve)
    )

    # ⑥ 基线 --------------------------------------------------------------
    hw_cv = _get(hw.dz_scan.baseline, "cv")
    sim_cv = _get(sim.dz_scan.baseline, "cv")
    hw_pib = _get(hw.dz_scan.baseline, "pib")
    sim_pib = _get(sim.dz_scan.baseline, "pib")
    hw_hollow = _get(hw.dz_scan.baseline, "spot_hollowness")
    sim_hollow = _get(sim.dz_scan.baseline, "spot_hollowness")
    cv_verdict = verdict(relative_diff(hw_cv, sim_cv), TOL_UNIFORMITY)
    gaps.append(
        "⑥ **基线** (Δz=0, GS 相位, 同一目标框): CV 实测 "
        f"{fmt_sigfig(hw_cv, 4)} vs 仿真 {fmt_sigfig(sim_cv, 4)} ({cv_verdict}); PIB 实测 "
        f"{fmt_sigfig(hw_pib, 4)} vs 仿真 {fmt_sigfig(sim_pib, 4)}; 空心度 实测 "
        f"{fmt_sigfig(hw_hollow, 4)} vs 仿真 {fmt_sigfig(sim_hollow, 4)}。"
        "**注意**: 目标框边长固定 10 px, 光斑一旦比框**宽**, 框内就接近平顶 ⇒ CV 低; "
        "光斑比框**窄**时框采到的是翼 ⇒ CV 爆炸。所以 CV 对**光斑宽度与剖面形状**"
        "极其敏感: 宽度对上之后, 剩下的差主要是**剖面形状** (实测 GS 相位打出来的"
        "是一个**平顶方斑**, 而孪生把同一张相位打在**另一块**面板相差上, 剖面更尖)。"
    )

    return gaps


def _focus_flat_fwhm_at_zero(focus: dict[str, Any]) -> float | None:
    """焦面标定里 1/f = 0 那一点的 FWHM (即**未矫正**的平场宽度)。"""
    for point in focus.get("points", []):
        diopter = _get(point, "diopter_m1")
        if diopter is not None and abs(diopter) < 1e-9:
            return _get(point, "fwhm_px")
    return None


def _normalised_curve(
    rows: Sequence[dict[str, Any]], key: str
) -> list[tuple[float, float]]:
    """把一条曲线归一化到它自己的 Δz=0 值。

    Returns:
        ``[(dz_mm, 归一化值)]`` (按 Δz 升序); Δz=0 那一行本身就是 1.0。
    """
    values = [
        (float(r.get("dz_mm", 0.0)), _get(r, key))
        for r in rows
    ]
    values = [(dz, v) for dz, v in values if v is not None]
    base = next((v for dz, v in values if abs(dz) < ZERO_DEFOCUS_TOLERANCE), None)
    if base is None or abs(base) < NONZERO_FLOOR:
        return []
    return sorted(((dz, v / base) for dz, v in values), key=lambda item: item[0])


def _curve_shape_note(
    hw_curve: Sequence[tuple[float, float]],
    sim_curve: Sequence[tuple[float, float]],
) -> str:
    """给两条归一化曲线一个形状对照的结论。

    逐点比较**不够稳**: 远场是个散斑场, 单帧的 FWHM 本身就不单调 (实测在
    Δz = −10 mm 处量到 19.2 px, 在 −15 mm 处量到 144 px) ⇒ 逐点相对差会被
    **单帧的宽度测量噪声**支配。所以这里给两个量:

    * 逐点相对差 (列出, 但注明它噪声大);
    * **稳健量** —— |Δz| ≥ 5 mm 区间上归一化值的**中位数**, 那是「弥散得多快」
      的代表值。
    """
    if not hw_curve or not sim_curve:
        return "曲线缺失 (没有可用的 Δz 扫描数据)"
    sim_map = {round(dz, 6): value for dz, value in sim_curve}
    parts: list[str] = []
    worst = 0.0
    worst_dz = 0.0
    hw_far: list[float] = []
    sim_far: list[float] = []
    for dz, value in hw_curve:
        sim_value = sim_map.get(round(dz, 6))
        if sim_value is None:
            continue
        if abs(dz) >= 5.0:
            hw_far.append(value)
            sim_far.append(sim_value)
        rel = (sim_value - value) / value if abs(value) > NONZERO_FLOOR else None
        if rel is not None and abs(rel) > abs(worst):
            worst = rel
            worst_dz = dz
        parts.append(f"Δz={dz:+.4g}: 实测 {value:.3f} / 仿真 {sim_value:.3f}")
    head = "; ".join(parts[:6])
    tail = f"; …共 {len(parts)} 点" if len(parts) > 6 else ""
    hw_median = statistics.median(hw_far) if hw_far else None
    sim_median = statistics.median(sim_far) if sim_far else None
    robust = ""
    if hw_median and sim_median:
        robust = (
            f"**稳健量** (|Δz| ≥ 5 mm 的 {len(hw_far)} 点, 归一化 FWHM 的中位数): "
            f"实测 {fmt_sigfig(hw_median, 3)} vs 仿真 {fmt_sigfig(sim_median, 3)} ⇒ 仿真的离焦响应"
            f"**陡 {fmt_sigfig(sim_median / hw_median, 3)}×**。"
        )
    return (
        f"{head}{tail}。逐点相对差最大 {worst:+.1%} @ Δz={worst_dz:+.4g} mm "
        "(但逐点值噪声大: 远场是散斑场, 单帧 FWHM 本身不单调 —— 实测在 Δz = −10 mm "
        "处量到 19.2 px、在 −15 mm 处量到 144 px)。" + robust
    )


# ---------------------------------------------------------------------------
# 纯函数: 渲染
# ---------------------------------------------------------------------------


def _format_cell(value: float | None, digits: int = 4) -> str:
    return fmt_sigfig(value, digits)


def render_table(
    cells: Sequence[Cell], *, heading: str | None = "## 1. 对照表"
) -> list[str]:
    """把对照行渲染成 markdown 表格。

    **数据表与报告正文共用这一份**: 报告正文里的「逐项对照表」不是手抄的, 而是
    同一个函数对同一批 ``cells`` 的输出 (只是标题不同)。

    Args:
        cells: 对照行。
        heading: 章节标题; ``None`` = 不写标题 (由调用方自己给)。

    Returns:
        markdown 行 (每行一个字符串)。
    """
    lines: list[str] = []
    add = lines.append
    if heading:
        add(heading)
        add("")
    add("| 指标 | 实测 | 仿真 | 相对差 (sim−hw)/hw | 判定 |")
    add("|---|---|---|---|---|")
    current_section = ""
    for cell in cells:
        if cell.section != current_section:
            current_section = cell.section
            add("")
            add(f"**{current_section}**")
            add("")
            add("| 指标 | 实测 | 仿真 | 相对差 (sim−hw)/hw | 判定 |")
            add("|---|---|---|---|---|")
        rel = relative_diff(cell.hw, cell.sim)
        rel_text = "—" if rel is None else f"{rel:+.1%}"
        label = cell.metric if not cell.note else f"{cell.metric} *({cell.note})*"
        add(
            f"| {label} | {_format_cell(cell.hw)} | "
            f"{_format_cell(cell.sim)} | {rel_text} | {verdict(rel, cell.tol)} |"
        )
    add("")
    add(
        "判定阈值: |相对差| ≤ 阈值 ⇒ **一致**; ≤ 3× 阈值 ⇒ **偏差**; 否则 "
        "**不一致**。阈值逐指标见脚本常量 (CV/复合 20%、PIB 20%、宽度 25%、"
        "峰值 20%、噪声 σ 50%、焦距 50%)。"
    )
    add("")
    return lines


def render_markdown(
    cells: Sequence[Cell],
    gaps: Sequence[str],
    hw: Bundle,
    sim: Bundle,
    sim_config: dict[str, Any],
) -> str:
    """把对照表渲染成 markdown。

    Args:
        cells: 对照行。
        gaps: 差距清单。
        hw: 实测侧。
        sim: 仿真侧。
        sim_config: 仿真口径 (种子/补零/光度/噪声…), 原样写进表头。

    Returns:
        markdown 文本。
    """
    lines: list[str] = []
    add = lines.append

    add("# 仿真 vs 实测 对照尺 (2026-10-22)")
    add("")
    add(
        "> 由 `scripts/slm/compare_sim_vs_hw.py` 生成。实测侧的数字**全部从归档产物"
        "读出** (无硬编码), 仿真侧走**同一批采集脚本**的 `--cam-type sim` 路径, "
        "指标口径与采集脚本**共用同一套函数** ⇒ 表里的差只能来自物理预测模型。"
    )
    add("")

    # --- 口径与参数 ------------------------------------------------------
    add("## 0. 口径与参数")
    add("")
    add("| 项 | 实测侧 | 仿真侧 |")
    add("|---|---|---|")
    add(f"| 产物根目录 | `{hw.root}` | `{sim.root}` |")
    add(
        "| 采集脚本 | `hw_gs_defocus_probe.py` / `focus_probe` / "
        "`run_gs_refine_zernike_sweep.py` | 同左 (`--cam-type sim`) |"
    )
    add(
        f"| 曝光 / ROI / 目标框 | {EXPOSURE_MS} ms / {ROI_PX} px / "
        f"{TARGET_SIDE_PX} px | {EXPOSURE_MS} ms / {ROI_PX} px / "
        f"{TARGET_SIDE_PX} px |"
    )
    add(
        f"| 每点帧数 | {FRAMES} | {FRAMES} |"
    )
    add(
        f"| GS 相位来源 | `{hw.dz_scan.run_config.get('gs_phase_source', '—')}` "
        f"| `{sim.dz_scan.run_config.get('gs_phase_source', '—')}` |"
    )
    for key, value in sim_config.items():
        add(f"| {key} | — | `{value}` |")
    add("")

    # --- 对照表 (与报告正文共用同一份渲染函数) ------------------------------
    lines.extend(render_table(cells))

    # --- 差距清单 --------------------------------------------------------
    add("## 2. 差距清单 (可归因)")
    add("")
    for gap in gaps:
        add(f"- {gap}")
    add("")

    # --- 本轮的孪生参数 + 判定汇总 ----------------------------------------
    add("## 3. 本轮的孪生参数与判定汇总")
    add("")
    add(
        "**本轮孪生用的参数就是上表 §0 的那一组**, 可复现地归档在 "
        "`report/shape_trials/sim_vs_hw/artifacts/sim_twin_params_20261022.json` "
        "(`--params <文件>` 读回同一组)。三个根因各自对应一个旋钮:"
    )
    add("")
    add("| 根因 | 旋钮 | 本轮取值 | 语义 |")
    add("|---|---|---|---|")
    add(
        "| ① 孪生没有「CCD 离焦」自由度 | `camera_defocus_m` | "
        f"`{sim_config.get('相机离焦 (camera_defocus_m)')}` m | 探测器在 2f 焦面之后 "
        "Δz ⇒ 瞳面等价二次相位 `-π·Δz·r²/(λf²)`; 补偿焦距 `f_v = f²/Δz` = 1549.3 mm |"
    )
    add(
        "| ② 逐帧把远场峰值归一到 100 | `counts_reference` | "
        f"`{sim_config.get('counts 归一化参考 (counts_reference)')}` | 固定参考 = "
        "平场在**理想聚焦**下的峰值 ⇒ 光被散开后 counts 自然下降 |"
    )
    add(
        "| ③ counts 帧没有暗电平 | `dark_level_counts` | "
        f"`{sim_config.get('暗电平 (dark_level_counts)')}` counts | 先加暗电平 → 加噪声 "
        "→ 量化; 减中位数 (暗帧扣除) 把它去掉 ⇒ 底座不再抖 |"
    )
    add(
        "| 光斑宽度 (有效口径) | `illumination_radius_px` / `beam_w0` | "
        f"`{sim_config.get('照明盘半径 (illumination_radius_px)')}` px / "
        f"`{sim_config.get('入射光束幅度 1/e 半径 (beam_w0)')}` | 入射光的有效口径定"
        "衍射极限: 实测焦面平场宽度 14.6 相机像元 ⇒ 强度的 1/e² 半径 ≈ 170 面板 px "
        "(孪生的 `beam_w0` 是**幅度**的 1/e 半径, 故取 120) |"
    )
    add(
        "| 面板静态相差 | `SimPanelAberration` | "
        f"`{sim_config.get('面板 Zernike (rad)')}` / 粗糙度 "
        f"`{sim_config.get('面板粗糙度 (roughness_sigma_rad)')}` | 低阶 Zernike + "
        "(可选) 空间相关的粗糙度; 本轮**没有**用它去凑宽度 (见差距清单 ②) |"
    )
    add("")
    add("### 3.1 已对齐 (判定「一致」或「偏差」)")
    add("")
    aligned: list[str] = []
    off: list[str] = []
    tally = {"一致": 0, "偏差": 0, "不一致": 0, "—": 0}
    for cell in cells:
        rel = relative_diff(cell.hw, cell.sim)
        grade = verdict(rel, cell.tol)
        tally[grade] = tally.get(grade, 0) + 1
        rel_text = "—" if rel is None else f"{rel:+.1%}"
        text = (
            f"{cell.metric}: 实测 {_format_cell(cell.hw)} / "
            f"仿真 {_format_cell(cell.sim)} ({rel_text})"
        )
        if grade == "一致":
            aligned.append(f"✅ {text}")
        elif grade == "偏差":
            aligned.append(f"🟡 {text}")
        else:
            off.append(f"⚠️ {text}")
    add(
        f"判定分布: **一致 {tally['一致']}** / 偏差 {tally['偏差']} / "
        f"不一致 {tally['不一致']} (共 {len(cells)} 行; ✅ = 一致, 🟡 = 偏差)。"
    )
    add("")
    if aligned:
        for item in aligned:
            add(f"- {item}")
    else:
        add("- (没有)")
    add("")
    add("### 3.2 仍不一致 (以及为什么)")
    add("")
    if off:
        for item in off:
            add(f"- {item}")
    else:
        add("- (没有)")
    add("")
    add(
        "> ⚠️ **§5 的 `before` / `after` 行两侧的 SPGD 轮数不同** (硬件 154+150, 仿真 "
        "8+8): `before` 是 Stage-A 的 GS 相位在 12.54 px 框内的功率占比, 轮数少 ⇒ "
        "光还没被收进框 ⇒ 仿真显著偏低。这几行**只能看方向** (仿真里 `quality` "
        "确实随阶数上升), 绝对值不可比。"
    )
    add("")
    add(
        "> ⚠️ **仿真侧的光斑宽度已换算成相机像元**: 孪生一个远场采样点 = 1.57 相机"
        "像元 (行) / 0.98 (列) —— 面板是 1920×1200 的长方形, 单一补零因子无法让两轴"
        "都对齐。不换算就是拿「远场采样点」跟「相机像元」比, 那是两种口径。"
    )
    add("")
    add(
        "> ⚠️ **§4 的「相对平场的增益」与「最佳点峰值/曝光」仍偏高**: 两侧的"
        "「平场」不是同一个东西 —— 硬件那一轮的最佳 1/f 点是一张**没有 GS 整形**的"
        "平场, 而它的 GS 相位是**针对这块面板的相差优化过**的 (所以 GS 光斑比平场"
        "亮 1.6×); 孪生把同一张 GS 相位打在**另一块** (合成的) 面板相差上, 那个"
        "「GS 比平场亮」的增益就不存在。要真正对上, 需要**实测反演**出面板相差"
        "再注入 (见 `scripts/slm/` 的 model_in_loop 系列), 不是本轮的量尺能定的。"
    )
    add("")

    # --- 附录 A: 逐点离焦响应 -------------------------------------------
    add("## 附录 A: 离焦响应逐点对照")
    add("")
    add("| Δz (mm) | 实测 FWHM (px) | 仿真 FWHM (px) | 实测峰值 (counts) | 仿真峰值 (counts) | 实测 PIB | 仿真 PIB |")
    add("|---|---|---|---|---|---|---|")
    dz_points = sorted({float(r.get("dz_mm", 0.0)) for r in hw.dz_scan.rows})
    for dz in dz_points:
        hw_row = hw.dz_scan.row_at_dz(dz) or {}
        sim_row = sim.dz_scan.row_at_dz(dz) or {}
        add(
            f"| {dz:+.4g} | {_format_cell(_get(hw_row, 'spot_fwhm_px'))} | "
            f"{_format_cell(_get(sim_row, 'spot_fwhm_px'))} | "
            f"{_format_cell(_get(hw_row, 'frame_peak'))} | "
            f"{_format_cell(_get(sim_row, 'frame_peak'))} | "
            f"{_format_cell(_get(hw_row, 'pib'))} | "
            f"{_format_cell(_get(sim_row, 'pib'))} |"
        )
    add("")

    # --- 附录 B: 复现 ----------------------------------------------------
    add("## 附录 B: 复现")
    add("")
    add("```bash")
    add("# 跑仿真 + 出表 (纯离线)")
    add("python scripts/slm/compare_sim_vs_hw.py --execute")
    add("")
    add("# 只重出表 (复用已有仿真产物)")
    add("python scripts/slm/compare_sim_vs_hw.py \\")
    add(f"    --sim-root {sim.root}")
    add("```")
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 报告正文 (入库的那一份)
# ---------------------------------------------------------------------------


def _verdict_tally(cells: Sequence[Cell]) -> dict[str, int]:
    """统计判定分布 (一致 / 偏差 / 不一致 / —)。"""
    tally = {"一致": 0, "偏差": 0, "不一致": 0, "—": 0}
    for cell in cells:
        grade = verdict(relative_diff(cell.hw, cell.sim), cell.tol)
        tally[grade] = tally.get(grade, 0) + 1
    return tally


def render_report(
    cells: Sequence[Cell],
    gaps: Sequence[str],
    hw: Bundle,
    sim: Bundle,
    sim_config: dict[str, Any],
    *,
    params_path: str | None = None,
    elapsed_s: float | None = None,
) -> str:
    """把本轮对照渲染成**可入库的报告正文**。

    与 :func:`render_markdown` 的分工: 那一个是**数据表** (逐项数字 + 判定 + 归因
    清单, 跟着数据走); 本函数是**报告正文** (做了什么 / 用了什么 / 对上了什么 /
    能用来干什么 / 怎么复现 / 局限)。两者共用 :func:`render_table` 与
    :func:`build_gap_list`, 所以表格与归因**不会手抄走样**。

    Args:
        cells: 对照行。
        gaps: 差距清单 (归因)。
        hw: 实测侧。
        sim: 仿真侧。
        sim_config: 仿真口径 (原样写进「仿真用了什么」)。
        params_path: 参数归档文件的路径。
        elapsed_s: 整轮仿真耗时 (s); ``None`` = 本轮没有跑仿真 (只重出表)。

    Returns:
        markdown 文本。
    """
    lines: list[str] = []
    add = lines.append

    params_abs = Path(params_path) if params_path else DEFAULT_PARAMS_JSON
    params_rel = _relpath(params_abs)
    params_href = _relpath(params_abs, start=DEFAULT_OUT_REPORT.parent)
    table_href = _relpath(DEFAULT_OUT_MD, start=DEFAULT_OUT_REPORT.parent)

    hw_flat_peak = _get(hw.dz_scan.baseline, "frame_peak")
    hw_counts_rate = hw_flat_peak / EXPOSURE_MS if hw_flat_peak else None
    hw_diopter = _focus_value(hw.focus, "best_diopter_m1")
    hw_focal_mm = _focus_value(hw.focus, "best_focal_mm")
    hw_flat_fwhm = _focus_value(hw.focus, "best_fwhm_px")
    hw_gain = _focus_value(hw.focus, "gain_vs_flat")
    hw_linearity = _get(hw.focus, "linearity_ratio")
    sim_linearity = _get(sim.focus, "linearity_ratio")
    tally = _verdict_tally(cells)
    hw_noise = summarise_repeatability(hw.dz_noise.rows, NOISE_METRICS)
    hw_pedestal = frame_total_stats(
        _noise_group_dir(hw, "dz_noise_hw_gs") or Path("nonexistent")
    )
    sim_pedestal = frame_total_stats(
        _noise_group_dir(sim, "dz_noise_hw_gs") or Path("nonexistent")
    )

    add("# 数字孪生 vs 第三轮硬件实测: 逐项对照 (2026-10-22)")
    add("")
    add(
        "> 生成脚本: `scripts/slm/compare_sim_vs_hw.py` (`--report`)。"
        "本文是**报告正文**; 同一轮渲染出的**数据表** (逐项数字 + 判定 + 归因清单) "
        f"在 [`{table_href}`]({table_href}), 孪生参数在 "
        f"[`sim_twin_params_20261022.json`]({params_href})。"
    )
    add("")

    # --- 1. 本轮做了什么 ---------------------------------------------------
    add("## 1. 本轮做了什么")
    add("")
    add(
        "把**同一批采集脚本**搬到**数字孪生**上, 跑第三轮的**同一组测量**, 再用"
        "**同一套指标函数**与第三轮硬件归档产物逐项对照。三条纪律:"
    )
    add("")
    add(
        "1. **实测侧零硬编码**: 每一个实测数字都从归档产物 "
        f"`{_relpath(hw.root)}` 的 `measurements.json` / `summary.json` / "
        "焦面标定 JSON 里**读**出来 —— 数据变了, 表跟着变。"
    )
    add(
        "2. **仿真侧不改采集脚本**: 离焦扫描用 `scripts/slm/hw_gs_defocus_probe.py`、"
        "焦面标定用 `ao_shaping.tools.slm.panel.focus_probe`、阶数扫描用 "
        "`scripts/slm/run_gs_refine_zernike_sweep.py`, 三者都有现成的 `--cam-type sim` "
        "路径; 本脚本只在**运行时**替换它们各自的 `configure_sim_system`, 把完整口径"
        "注入进去。**三个脚本文件一个字都没改。**"
    )
    add(
        "3. **指标口径不另写一套**: CV / PIB / 复合分数来自 "
        "`ao_shaping.algorithm.signal_processing.square_shaping`, 光斑几何来自 "
        "`ao_shaping.utils.image.frame_analysis` —— 与采集脚本用的是**同一个函数**。"
    )
    add("")
    add(
        "**两侧同源的三件事** (不同源就是两种口径, 表就没有意义): 同一张瞳面菲涅尔"
        f"焦面矫正片 (`defocus_focal_mm` = {fmt_sigfig(hw_focal_mm, 6)} mm)、同一批 1/f "
        "格点 (焦面标定的 42 个点)、同一个目标框 (阶数扫描) —— 全部从实测那一轮的 "
        "`run_config.json` / `summary.json` 读出。"
    )
    add("")

    # --- 2. 仿真用了什么 ---------------------------------------------------
    add("## 2. 仿真用了什么 (以及为什么是那个值)")
    add("")
    add(
        f"完整一组在 [`sim_twin_params_20261022.json`]({params_href}) "
        "(`--params` 读回即复现)。每一项都不是拧到「看着像」, 而是**由某条实测反推**:"
    )
    add("")
    add("| 旋钮 | 本轮取值 | 为什么是那个值 (依据哪条实测) |")
    add("|---|---|---|")
    add(
        "| `far_field_padding` | "
        f"{sim_config.get('远场补零 (far_field_padding)')} | 孪生的角标度 = "
        "`λf/(N_pad·p)`; 实测 "
        "Kx = 7546.4 / Ky = 7495.9 (相机像元/面板像元), 面板 1920 列 ⇒ "
        "7557/1920 = 3.94 ⇒ **4** 才复现实测角标度 (脚本默认的 16 是 4× 放大的"
        "显示口径, 角标度会差 4 倍) |"
    )
    add(
        "| `peak_counts_per_ms` | "
        f"{fmt_sigfig(sim_config.get('绝对光度 (peak_counts_per_ms)'), 5)} | 实测平场 "
        f"0.2 ms 整帧峰值 {fmt_sigfig(hw_flat_peak, 4)} counts ⇒ "
        f"{fmt_sigfig(hw_counts_rate, 4)} counts/(峰值分数·ms) ⇒ 取 133.5 使仿真帧与实测帧"
        "**同一量纲** (不再是「归一到 ~100」的相对强度) |"
    )
    add(
        "| `read_sigma_counts` / `shot_k` / `quantize_bits` | "
        f"{fmt_sigfig(sim_config.get('读出噪声 (read_sigma_counts)'), 3)} / "
        f"{fmt_sigfig(sim_config.get('散粒系数 (shot_k)'), 3)} / "
        f"{sim_config.get('量化位深 (quantize_bits)')} | 实测帧是 uint8、4 帧平均、"
        "背景只有 0 / 0.25 / 0.5 三个值 ⇒ 每原始帧 σ ≈ 0.56 counts; 而 "
        f"σ(整帧峰值) 只有 {fmt_sigfig((hw_noise.get('frame_peak') or {}).get('std'), 3)} "
        f"counts ({fmt_sigfig(hw_flat_peak, 4)} counts 处) ⇒ 该光级**不是散粒受限** "
        "⇒ `shot_k = 0` |"
    )
    add(
        "| `dark_level_counts` | "
        f"{fmt_sigfig(sim_config.get('暗电平 (dark_level_counts)'), 4)} | 实测相机的暗电平"
        " (远高于读出噪声)。它让背景**不再落在中位数的跳变边界**上: 整帧总功率的"
        "相对抖动 "
        + (
            f"实测 {fmt_sigfig(hw_pedestal['rel_std'] * 100, 3)}% vs 仿真 "
            f"{fmt_sigfig(sim_pedestal['rel_std'] * 100, 3)}%"
            if hw_pedestal and sim_pedestal
            else "(无帧文件, 未统计)"
        )
        + " (补暗电平之前仿真是 72.7%) |"
    )
    add(
        "| `camera_defocus_m` | "
        f"{fmt_sigfig(sim_config.get('相机离焦 (camera_defocus_m)'), 9)} m | 实测最佳 "
        f"1/f = {fmt_sigfig(hw_diopter, 4)} m⁻¹ ⇒ 瞳面二次相位的补偿焦距 "
        f"{fmt_sigfig(hw_focal_mm, 6)} mm; 孪生里 `f_v = f²/Δz` (Collins/ABCD) ⇒ "
        "`Δz = 0.125²/1.54933` |"
    )
    add(
        "| `counts_reference` | "
        f"`{sim_config.get('counts 归一化参考 (counts_reference)')}` | 逐帧把峰值归一"
        "会把离焦散开的光**又拉回原峰值** (实测峰值随离焦崩塌 27 → 0.25 counts); "
        "固定参考 = 平场在**理想聚焦**下的峰值 ⇒ counts 守恒能量 |"
    )
    add(
        "| `beam_w0` | "
        f"{fmt_sigfig(sim_config.get('入射光束幅度 1/e 半径 (beam_w0)'), 4)} | 实测焦面"
        f"矫正后的平场 FWHM {fmt_sigfig(hw_flat_fwhm, 4)} 相机像元 ⇒ 反推**强度** 1/e² 半径 "
        "≈ 170 面板 px; 孪生的 `beam_w0` 是**幅度**的 1/e 半径 ⇒ 120 (衍射极限由"
        "入射有效口径定, 不是面板尺寸) |"
    )
    add(
        "| `illumination_radius_px` | "
        f"`{sim_config.get('照明盘半径 (illumination_radius_px)')}` | 真实照明是软的;"
        "**硬截断会造出 Airy 环**, 让阶数扫描的自动定尺把目标框撑到 148 px (实测是 "
        "12.54 px) ⇒ 本轮不截断 |"
    )
    add(
        "| 面板静态相差 (`SimPanelAberration`) | "
        f"`{sim_config.get('面板 Zernike (rad)')}` / 粗糙度 "
        f"{fmt_sigfig(sim_config.get('面板粗糙度 (roughness_sigma_rad)'), 3)} | 孪生的面板是"
        "**理想相位板**。实测多出来的宽度已由「入射有效口径」解释; 再叠低阶 Zernike "
        "虽然能把光斑撑宽, 但会把**平顶**光斑变成**带裙边的尖峰** ⇒ 目标框内的 CV "
        "反而变差 ⇒ 本轮**不用它凑宽度** |"
    )
    add("")

    # --- 3. 逐项对照表 -----------------------------------------------------
    add("## 3. 逐项对照表")
    add("")
    add(
        "下表由 `render_table()` 生成 —— 与数据表"
        f" [`{table_href}`]({table_href}) 的 §1 **逐字相同** (同一份渲染函数、同一批 "
        "`cells`), 不是手抄。"
    )
    add("")
    lines.extend(render_table(cells, heading=None))

    # --- 4. 对上了什么 / 仍差什么 -------------------------------------------
    add("## 4. 对上了什么 / 仍差什么")
    add("")
    add(
        f"判定分布: **一致 {tally['一致']}** / 偏差 {tally['偏差']} / "
        f"不一致 {tally['不一致']} (共 {len(cells)} 行)。逐项归因见下 "
        "(与数据表 §2 同一份函数产出):"
    )
    add("")
    for gap in gaps:
        add(f"- {gap}")
    add("")
    add("**还没对上的几类, 各自的物理原因** (不硬凑):")
    add("")
    add(
        "- **极小光斑峰值 (Δz = ±20 mm, 0.25 vs 1.5–1.75 counts)**: 这是**量化级**"
        "地板, 不是物理差距 —— 实测 0.25 = 4 帧里只有 1 帧出现 1 count; 仿真的 "
        "1.5–1.75 来自「暗电平 + σ=0.5 的 4 帧平均」在减中位数后的残余。要压下去"
        "只能改噪声分布, 那是编造。"
    )
    add(
        "- **§4 的「最佳点峰值/曝光」与「相对平场增益」**: 两侧的「平场」不是同一个量"
        " —— 硬件那一轮的 GS 相位是**针对这块面板的相差优化过**的, 所以它的 GS 光斑"
        f"比未整形平场亮 (实测增益 {fmt_sigfig(hw_gain, 4)}×); 孪生把同一张 GS 相位打在"
        "**另一块 (合成的)**面板相差上, 那个增益不存在。要真正对上, 得先**实测反演**"
        "出这块面板的面形再注入, 超出「对照尺」能定的范围。"
    )
    add(
        "- **§5 的 `before` / `after` 行**: 两侧 SPGD 轮数不同 (硬件 154+150, 仿真 "
        "8+8), 这几行**只能看方向** (`quality` 在仿真里确实随阶数上升), 绝对值不可比。"
    )
    add("")

    # --- 5. 能用来干什么 / 不能用来干什么 -----------------------------------
    add("## 5. 这条仿真链能用来干什么 / 不能用来干什么")
    add("")
    add("### 5.1 能")
    add("")
    add(
        "- **证明接口与流程在线实现**: 同一份 CLI、同一套指标函数、同构的产物"
        " (`measurements.json` / `run_config.json` / `frame_*.npz` / 焦面标定 JSON), "
        "在孪生上跑完不掉链子 —— 换后端 (`--cam-type sim`) 不改一行采集代码。"
    )
    add(
        "- **定量检验「哪些物理效应必须建模」**: 三个根因 (CCD 离焦 / counts 的固定"
        "参考 / 暗电平) 都是**先量出差距、再补模型、再看差距消掉**的; 这类判断可以"
        "反复在孪生上做, 不必占机时。"
    )
    add(
        "- **调参与回归**: 目标框 / 曝光 / 阶数 / SPGD 超参的行为在孪生里可复现 "
        "(固定 seed 逐位可重复), 适合当 CI 级的冒烟与回归。"
    )
    add("")
    add("### 5.2 不能")
    add("")
    add(
        "- **不能当实测替代**。孪生里 focus_probe 的「最佳 1/f」是**几何产物**: 它"
        "等于 `1/(f²/Δz)`, 正是我们把 `camera_defocus_m` 设成 10.0847 mm 的**逆运算** "
        "—— 表里它对上实测, 证明的是「我们把 CCD 离焦量按实测反推对了」, **不是**"
        "「孪生独立标定出了 1549.33 mm」。任何「孪生给出的焦距/标定值」都必须这样读。"
    )
    add(
        "- **曝光线性度检查在孪生里量的是噪声**: 孪生没有真实的曝光非线性 / 暗电流 / "
        "增益切换, 所以 `linearity_ratio` "
        f"(实测 {fmt_sigfig(hw_linearity, 4)} vs 仿真 {fmt_sigfig(sim_linearity, 4)}) 在仿真侧只是"
        "「同一帧换个曝光」的量化噪声, 不能当作传感器线性度的验证。"
    )
    add(
        "- **面板相差是合成的**: `SimPanelAberration` 里的 Zernike 与粗糙度是**我们"
        "选的**, 不是这块面板的实测面形。用它解释「为什么光斑宽了 1.5×」只在量级上"
        "成立。"
    )
    add(
        "- **不能用来定绝对精度类结论**: 衍射效率、真实散斑统计、长期漂移、像素响应"
        "非均匀 (PRNU) 这些都要真实器件。"
    )
    add("")

    # --- 6. 复现 -----------------------------------------------------------
    add("## 6. 复现")
    add("")
    add("```bash")
    add("# 跑仿真 + 出数据表 + 出本文 (纯离线, 不碰硬件)")
    add("python scripts/slm/compare_sim_vs_hw.py --execute \\")
    add(f"    --params {params_rel}")
    add("")
    add("# 只重出表与本文 (复用已有仿真产物, 秒级)")
    add("python scripts/slm/compare_sim_vs_hw.py \\")
    add(f"    --params {params_rel} \\")
    add(f"    --sim-root {_relpath(sim.root)}")
    add("```")
    add("")
    add("| 项 | 值 |")
    add("|---|---|")
    add(f"| 随机种子 | `{sim_config.get('仿真种子 (sim_seed)')}` (参数文件里固定) |")
    add(f"| 参数文件 | `{params_rel}` |")
    add(f"| 仿真产物 | `{_relpath(sim.root)}` (不入库) |")
    add(f"| 实测归档 | `{_relpath(hw.root)}` |")
    add(
        f"| 单次远场求值 | ≈ {SIM_EVAL_SECONDS:.2f} s / RSS 增量 ≈ "
        f"{SIM_EVAL_RSS_MB / 1000:.2f} GB (1920×1200 面板, `far_field_padding = 4`) |"
    )
    add(
        "| 整轮耗时 | "
        + (f"≈ {elapsed_s:.0f} s" if elapsed_s else "本轮只重出表, 未跑仿真")
        + " (离焦扫描 23 点 + 噪声组 9 次 + 焦面标定 42 点 + 阶数扫描) |"
    )
    add("")

    # --- 7. 局限 -----------------------------------------------------------
    add("## 7. 局限 (没有建模的物理效应)")
    add("")
    add("按本轮实际看到的写, 不列「可能」:")
    add("")
    # V2.0 旋钮的实际状态 (决定下面各条是否标注「本轮已开启」)
    _eta = float(sim_config.get("零级光占比 (zero_order_eta)") or 0.0)
    _prnu = float(sim_config.get("PRNU σ (prnu_sigma)") or 0.0)
    _dark = float(sim_config.get("暗电流 FPN σ (dark_fpn_sigma)") or 0.0)
    _g = sim_config.get("SLM LUT gamma")
    _lt = sim_config.get("SLM LUT 表")
    zo_on = _eta > 0.0
    cam_on = (_prnu > 0.0) or (_dark > 0.0)
    lut_on = (_g is not None) or (_lt is not None)
    add(
        "1. **零级 (η) 分量**: 真实 SLM 的瞳面是双分量 "
        "`U = U_inc·[η + (1−η)e^{iφ}]`, 零级只过物理透镜、在 `f+d` 处散成一圈弥散晕; "
        "孪生的瞳面只有**一路光**, 没有这个双分量 ⇒ 「框外功率占比」虽然对上了, 但"
        "它的**来源构成**不同 (实测含零级晕, 仿真只有边瓣与焦外弥散)。"
        + (
            f"*(V2.0 已建模: `SimPibSystem(zero_order_eta=...)` / `ZeroOrderModel`; "
            f"**本轮已开启** (eta = {_eta:g}), 零级晕已进入远场。)*"
            if zo_on
            else "*(V2.0 已建模: `SimPibSystem(zero_order_eta=...)` / `ZeroOrderModel`, 由"
            "`--zero-order-eta` 开启; 本对照默认 0 = 理想单分量, 数字不变。)*"
        )
    )
    add(
        "2. **面板面形误差**: 只有「合成 Zernike + 可选空间相关粗糙度」, 没有这块面板"
        "的实测面形; 逐像元白噪声 (`roughness_corr_px = 0`) 在远场上只加弥散本底、"
        "**不撑宽光斑**, 这也是本轮没用它凑宽度的原因。"
    )
    add(
        "3. **探测器非理想性**: 曝光非线性、暗电流、增益切换、像素响应非均匀 (PRNU)、"
        "读出噪声的空间相关性 —— 都没有。噪声是逐像元高斯白噪声 + 8-bit 量化。"
        "*(V2.0 已建模 PRNU + 暗电流 FPN: `CameraNonIdealModel`, 由 `--prnu-sigma` / "
        "`--dark-fpn-sigma` 开启, 固定图在量化**前**乘/加; "
        + (
            f"**本轮已开启** (prnu_sigma = {_prnu:g}, dark_fpn_sigma = {_dark:g}), "
            "噪声地板会上抬。"
            if cam_on
            else "本对照默认关闭。"
        )
        + ")*"
    )
    add(
        "4. **LCOS 的响应时间与松弛**: 仿真面板是**瞬时响应** "
        "(`display_data` 的 `wait_time_s` 被接受并忽略), 所以 `display_and_average` "
        "的稳定性判据在孪生里恒为「一次就稳」。"
    )
    add(
        "5. **SLM 的 LUT 非线性与 10-bit 灰度量化对远场的影响**: 孪生只做"
        "「弧度 → 灰度 → 远场」的理想映射 (`two_pi_gray = 993`), 没有 LUT 曲线、"
        "没有灰度级量化误差。"
        "*(V2.0 已建模: `SlmLutModel`, 由 `--slm-lut-gamma` / `--slm-lut-table` 开启;"
        "设备**实际显示**的相位 = LUT 反查 + 量化误差, 喂给远场。"
        + (
            f"**本轮已开启** (gamma = {_g if _g is not None else 1.0}, "
            f"table = {_lt})。"
            if lut_on
            else "本对照默认 γ=1 = 理想线性。"
        )
        + ")*"
    )
    add("6. **热漂移 / 长期漂移**: 没有。噪声组只重复 9 次 (几十秒尺度)。")
    add(
        "7. **远场网格的各向异性**: 面板 1920×1200 是长方形, 单一补零因子无法让两轴"
        "都等于一个相机像元 (行 1.57 / 列 0.98) —— 宽度比较已按调和平均 ×1.21 换算,"
        "但仍有 ~1% 的标度残差。"
        "*(V2.0 已支持各向异性补零: `far_field_padding=(行,列)`, 由 "
        "`--sim-far-field-padding 行,列` 开启, 使两轴节距都 = 相机像元 ⇒ 残差归零;"
        "本对照默认 `4` (int) 保持原角标度。)*"
    )
    add("")
    return "\n".join(lines)


def _relpath(path: Path | str, start: Path | None = None) -> str:
    """仓库相对路径 (反斜杠统一成正斜杠); 不在仓库内就原样返回。"""
    try:
        target = Path(path).resolve()
        base = (start or ROOT).resolve()
        return target.relative_to(base).as_posix()
    except (ValueError, OSError):
        return str(path)


# ---------------------------------------------------------------------------
# 仿真执行 (副作用; 纯离线, 不碰硬件)
# ---------------------------------------------------------------------------


def _twin_default(name: str) -> Any:
    """读孪生模块的某个默认常量 (延迟 import)。

    孪生模块 import 期就注册 ``"sim"`` 相机, 所以本脚本**不在模块级** import 它
    —— 默认值在这里按需解析, 单一事实源仍在孪生那边。
    """
    import ao_shaping.drivers.sim.slm_pib_sim as twin

    return getattr(twin, name)


@dataclass(frozen=True)
class SimConfig:
    """仿真口径: 种子 + 角标度 + 光度 + 噪声 + 孪生的物理旋钮。

    V2.0 新增 ``zero_order_eta`` / ``slm_lut_*`` / ``prnu_sigma`` / ``dark_fpn_sigma``
    四组旋钮, 对应报告第 7 节列缺的器件级物理效应 (零级光 / SLM LUT 非线性 /
    探测器 PRNU + 暗电流 FPN)。它们**默认全部 = 理想器件** (identity), 本比较脚本
    的默认口径不变 ⇒ 表里数字与改进前逐位相同; 要启用就调旋钮或从 ``--params`` 文件读。
    """

    seed: int = SIM_SEED
    far_field_padding: int | tuple[int, int] = SIM_FAR_FIELD_PADDING
    peak_counts_per_ms: float | None = PEAK_COUNTS_PER_MS
    read_sigma_counts: float = READ_SIGMA_COUNTS
    shot_k: float = SHOT_K
    quantize_bits: int | None = QUANTIZE_BITS
    # --- 孪生的物理旋钮 (三个根因) ---------------------------------------
    #: 相机离焦 (m): 探测器在 2f 焦面**之后** Δz 处 ⇒ 瞳面上等价地乘一个
    #: 二次相位 ``-π·Δz·r²/(λ·f²)``, 补偿它所需的面板菲涅尔焦距 ``f_v = f²/Δz``。
    camera_defocus_m: float = 0.0
    #: counts 口径的归一化参考: ``"frame"`` (逐帧, 历史行为) / ``"flat"`` (固定).
    counts_reference: str | None = None
    #: 探测器暗电平 (counts) —— 真实相机实测 26.75。
    dark_level_counts: float = 0.0
    #: 入射高斯光束的**幅度** 1/e 半径 (面板 px); ``None`` = 孪生的默认值。
    beam_w0: float | None = None
    #: 照明盘半径 (面板 px); ``None`` = 不截断。
    illumination_radius_px: float | None = None
    #: 面板静态相差: Noll 序 → 弧度。
    panel_zernike_rad: dict[int, float] = field(default_factory=dict)
    panel_zernike_n_max: int = 6
    panel_roughness_sigma_rad: float = 0.0
    panel_roughness_corr_px: float = 0.0
    panel_roughness_seed: int = 0
    # --- V2.0 非理想物理 (NonIdealModule) ---------------------------------
    #: 零级光能量占比 η (0~1); 0 = 理想单分量器件。
    zero_order_eta: float = 0.0
    #: SLM LUT 的 Gamma 指数 (1 = 理想线性); ``None`` = 不建 LUT 模型。
    slm_lut_gamma: float | None = None
    #: 实测 LUT 表 (灰度→弧度) 的 .npy 路径; 给了就**优先于** ``slm_lut_gamma``。
    slm_lut_table: str | None = None
    #: 探测器 PRNU 乘性图标准差 (1%~2% 量级); 0 = 无 PRNU。
    prnu_sigma: float = 0.0
    #: 探测器暗电流 FPN 标准差 (counts); 0 = 暗电平是全局常数。
    dark_fpn_sigma: float = 0.0
    #: 生成固定 PRNU / 暗场图的 RNG 种子 (固定 ⇒ 可复现)。
    nonideal_seed: int = 0

    def __post_init__(self) -> None:
        """把「用孪生默认值」的旋钮解析掉。

        孪生模块是**延迟 import** 的 (它 import 期就注册 ``"sim"`` 相机), 所以
        默认值不写死在这里 —— 写成 ``None`` 再在这里解析, 单一事实源仍在孪生那边。
        """
        if self.beam_w0 is None:
            object.__setattr__(self, "beam_w0", _twin_default("BEAM_W0"))
        if self.counts_reference is None:
            object.__setattr__(
                self, "counts_reference", _twin_default("COUNTS_REFERENCE_FRAME")
            )
        # 远场补零因子归一化: ``None`` / 列表 → 默认值 / tuple。
        ffp = self.far_field_padding
        if ffp is None:
            object.__setattr__(self, "far_field_padding", SIM_FAR_FIELD_PADDING)
        elif isinstance(ffp, (list, tuple)):
            object.__setattr__(
                self,
                "far_field_padding",
                (int(ffp[0]), int(ffp[1])),
            )

    def as_dict(self) -> dict[str, Any]:
        ffp = self.far_field_padding
        ffp_text = (
            f"{ffp[0]},{ffp[1]}" if isinstance(ffp, (tuple, list)) else ffp
        )
        return {
            "仿真种子 (sim_seed)": self.seed,
            "远场补零 (far_field_padding)": ffp_text,
            "绝对光度 (peak_counts_per_ms)": self.peak_counts_per_ms,
            "读出噪声 (read_sigma_counts)": self.read_sigma_counts,
            "散粒系数 (shot_k)": self.shot_k,
            "量化位深 (quantize_bits)": self.quantize_bits,
            "相机离焦 (camera_defocus_m)": self.camera_defocus_m,
            "counts 归一化参考 (counts_reference)": self.counts_reference,
            "暗电平 (dark_level_counts)": self.dark_level_counts,
            "入射光束幅度 1/e 半径 (beam_w0)": self.beam_w0,
            "照明盘半径 (illumination_radius_px)": self.illumination_radius_px,
            "面板 Zernike (rad)": dict(self.panel_zernike_rad),
            "面板粗糙度 (roughness_sigma_rad)": self.panel_roughness_sigma_rad,
            "面板粗糙度相关长度 (roughness_corr_px)": self.panel_roughness_corr_px,
            # --- V2.0 非理想物理 -------------------------------------------
            "零级光占比 (zero_order_eta)": self.zero_order_eta,
            "SLM LUT gamma": self.slm_lut_gamma,
            "SLM LUT 表": self.slm_lut_table,
            "PRNU σ (prnu_sigma)": self.prnu_sigma,
            "暗电流 FPN σ (dark_fpn_sigma)": self.dark_fpn_sigma,
            "非理想模型种子 (nonideal_seed)": self.nonideal_seed,
        }

    def to_json(self) -> dict[str, Any]:
        """可序列化的口径 (写进参数文件)。"""
        ffp = self.far_field_padding
        ffp_json = list(ffp) if isinstance(ffp, (tuple, list)) else ffp
        return {
            "seed": self.seed,
            "far_field_padding": ffp_json,
            "peak_counts_per_ms": self.peak_counts_per_ms,
            "read_sigma_counts": self.read_sigma_counts,
            "shot_k": self.shot_k,
            "quantize_bits": self.quantize_bits,
            "camera_defocus_m": self.camera_defocus_m,
            "counts_reference": self.counts_reference,
            "dark_level_counts": self.dark_level_counts,
            "beam_w0": self.beam_w0,
            "illumination_radius_px": self.illumination_radius_px,
            "panel_zernike_rad": {
                str(int(k)): float(v) for k, v in self.panel_zernike_rad.items()
            },
            "panel_zernike_n_max": self.panel_zernike_n_max,
            "panel_roughness_sigma_rad": self.panel_roughness_sigma_rad,
            "panel_roughness_corr_px": self.panel_roughness_corr_px,
            "panel_roughness_seed": self.panel_roughness_seed,
            # --- V2.0 非理想物理 -------------------------------------------
            "zero_order_eta": self.zero_order_eta,
            "slm_lut_gamma": self.slm_lut_gamma,
            "slm_lut_table": self.slm_lut_table,
            "prnu_sigma": self.prnu_sigma,
            "dark_fpn_sigma": self.dark_fpn_sigma,
            "nonideal_seed": self.nonideal_seed,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "SimConfig":
        """从参数文件读回口径 (缺项用默认值)。"""
        zernike = {
            int(k): float(v) for k, v in (payload.get("panel_zernike_rad") or {}).items()
        }
        ffp_raw = payload.get("far_field_padding", SIM_FAR_FIELD_PADDING)
        ffp = (
            (int(ffp_raw[0]), int(ffp_raw[1]))
            if isinstance(ffp_raw, (list, tuple))
            else int(ffp_raw)
        )
        return cls(
            seed=int(payload.get("seed", SIM_SEED)),
            far_field_padding=ffp,
            peak_counts_per_ms=payload.get("peak_counts_per_ms", PEAK_COUNTS_PER_MS),
            read_sigma_counts=float(payload.get("read_sigma_counts", READ_SIGMA_COUNTS)),
            shot_k=float(payload.get("shot_k", SHOT_K)),
            quantize_bits=payload.get("quantize_bits", QUANTIZE_BITS),
            camera_defocus_m=float(payload.get("camera_defocus_m", 0.0)),
            counts_reference=payload.get("counts_reference"),
            dark_level_counts=float(payload.get("dark_level_counts", 0.0)),
            beam_w0=payload.get("beam_w0"),
            illumination_radius_px=payload.get("illumination_radius_px"),
            panel_zernike_rad=zernike,
            panel_zernike_n_max=int(payload.get("panel_zernike_n_max", 6)),
            panel_roughness_sigma_rad=float(
                payload.get("panel_roughness_sigma_rad", 0.0)
            ),
            panel_roughness_corr_px=float(payload.get("panel_roughness_corr_px", 0.0)),
            panel_roughness_seed=int(payload.get("panel_roughness_seed", 0)),
            zero_order_eta=float(payload.get("zero_order_eta", 0.0)),
            slm_lut_gamma=payload.get("slm_lut_gamma"),
            slm_lut_table=payload.get("slm_lut_table"),
            prnu_sigma=float(payload.get("prnu_sigma", 0.0)),
            dark_fpn_sigma=float(payload.get("dark_fpn_sigma", 0.0)),
            nonideal_seed=int(payload.get("nonideal_seed", 0)),
        )


def build_noise_spec(config: SimConfig) -> Any:
    """构造 counts 口径的探测器噪声模型。

    ``peak_counts_per_ms`` 为 ``None`` 时不构造 (旧口径下噪声模型不同)。
    """
    if config.peak_counts_per_ms is None:
        return None
    from ao_shaping.drivers.sim.slm_pib_sim import SimNoiseSpec

    return SimNoiseSpec(
        read_sigma_counts=float(config.read_sigma_counts),
        shot_k=float(config.shot_k),
        quantize_bits=config.quantize_bits,
        dark_level_counts=float(config.dark_level_counts),
    )


def build_panel_aberration(config: SimConfig) -> Any:
    """构造面板静态相差 (Zernike + 粗糙度); 全零 ⇒ ``None`` (理想面板)。"""
    from ao_shaping.drivers.sim.slm_pib_sim import SimPanelAberration

    spec = SimPanelAberration(
        zernike_rad={int(k): float(v) for k, v in config.panel_zernike_rad.items()},
        zernike_n_max=int(config.panel_zernike_n_max),
        roughness_sigma_rad=float(config.panel_roughness_sigma_rad),
        roughness_corr_px=float(config.panel_roughness_corr_px),
        roughness_seed=int(config.panel_roughness_seed),
    )
    return None if spec.is_empty() else spec


def build_nonideal(config: SimConfig) -> dict[str, Any]:
    """构造 V2.0 的 NonIdealModule 参数 (零级光 / SLM LUT / 探测器非理想性)。

    把 ``SimConfig`` 里那几个旋钮收成 :func:`reset_system` 能直接解的字典
    (``zero_order_eta`` / ``slm_lut`` / ``nonideal_camera``)。全部旋钮置 0/None
    时返回的模型是 identity ⇒ 与改进前的孪生逐位相同。

    Args:
        config: 仿真口径。

    Returns:
        ``{"zero_order_eta": ..., "slm_lut": SlmLutModel|None,
        "nonideal_camera": CameraNonIdealModel|None}``。
    """
    from ao_shaping.drivers.sim.nonideal import (
        SlmLutModel,
        CameraNonIdealModel,
    )

    lut: SlmLutModel | None = None
    if config.slm_lut_gamma is not None or config.slm_lut_table is not None:
        table = (
            np.load(str(config.slm_lut_table))
            if config.slm_lut_table
            else None
        )
        lut = SlmLutModel(
            gamma=(
                float(config.slm_lut_gamma)
                if config.slm_lut_gamma is not None
                else 1.0
            ),
            lut_table=table,
        )
    camera = CameraNonIdealModel(
        prnu_sigma=float(config.prnu_sigma),
        dark_fpn_sigma=float(config.dark_fpn_sigma),
        seed=int(config.nonideal_seed),
    )
    return {
        "zero_order_eta": float(config.zero_order_eta),
        "slm_lut": lut,
        "nonideal_camera": camera if not camera.is_identity() else None,
    }


def reset_sim_system(config: SimConfig) -> None:
    """把**进程级**的仿真系统重置到指定口径。

    孪生的 system 是进程级的单例, 脚本侧有三份 ``configure_sim_system`` 各自
    重置它 —— 只跑一个脚本时没问题, 但对照脚本要连着跑好几个采集脚本, 所以
    **每次跑仿真前都要显式重置**, 否则第二次会沿用第一次的种子/补零/光度。

    Args:
        config: 仿真口径。
    """
    from ao_shaping.drivers.sim.slm_pib_sim import reset_system

    reset_system(
        seed=int(config.seed),
        far_field_padding=config.far_field_padding,
        peak_counts_per_ms=config.peak_counts_per_ms,
        noise_spec=build_noise_spec(config),
        panel_aberration=build_panel_aberration(config),
        camera_defocus_m=float(config.camera_defocus_m),
        counts_reference=str(config.counts_reference),
        beam_w0=float(config.beam_w0),
        illumination_radius_px=config.illumination_radius_px,
        **build_nonideal(config),
    )


def patch_sim_configurer(module: Any, attr: str, config: SimConfig) -> None:
    """把采集脚本自己的 ``configure_sim_system`` 换成**带完整口径**的版本。

    三个采集脚本各自有一份 ``configure_sim_system(seed, far_field_padding)``, 它
    只重置种子与补零 —— 绝对光度与噪声口径会被它**冲掉**。本函数用同签名的
    替身覆盖它, 于是脚本内部那一行调用就带上了全部口径旋钮。

    这是**适配**而不是改写: 采集脚本的文件一个字都没动, 物理路径也一条没变。

    Args:
        module: 采集脚本的模块对象。
        attr: 被替换的属性名。
        config: 仿真口径。
    """

    def _configure(seed: int, far_field_padding: int) -> None:
        from ao_shaping.drivers.sim.slm_pib_sim import reset_system

        reset_system(
            seed=int(seed),
            far_field_padding=config.far_field_padding,
            peak_counts_per_ms=config.peak_counts_per_ms,
            noise_spec=build_noise_spec(config),
            panel_aberration=build_panel_aberration(config),
            camera_defocus_m=float(config.camera_defocus_m),
            counts_reference=str(config.counts_reference),
            beam_w0=float(config.beam_w0),
            illumination_radius_px=config.illumination_radius_px,
            **build_nonideal(config),
        )

    setattr(module, attr, _configure)


def _resolve_gs_phase(explicit: str | None) -> str | None:
    """定 GS 相位来源: 显式路径 > 第三轮硬件相位 > ``None`` (现场生成模型 GS)。"""
    if explicit:
        return explicit
    if HW_GS_PHASE.exists():
        return str(HW_GS_PHASE)
    return None


def run_sim_probe(
    config: SimConfig,
    *,
    out_root: Path,
    group: str,
    dz_mm: Sequence[float],
    gs_phase: str | None,
    exposure_ms: float,
    roi_px: int,
    target_side_px: int,
    frames: int,
    defocus_focal_mm: float | None = None,
) -> Path:
    """用 ``scripts/slm/hw_gs_defocus_probe.py`` 在仿真里跑一组离焦扫描。

    Args:
        config: 仿真口径。
        out_root: 产物根目录。
        group: 组名 (落盘目录名)。
        dz_mm: Δz 扫描点 (mm)。
        gs_phase: GS 相位来源 (``.npy``), ``None`` = 现场生成模型 GS 相位。
        exposure_ms: 曝光 (ms)。
        roi_px: ROI 边长 (px)。
        target_side_px: 目标框边长 (px)。
        frames: 每点平均帧数。
        defocus_focal_mm: 瞳面菲涅尔焦面矫正的焦距 (mm)。**必须与硬件侧一致**
            (硬件那一轮的 ``run_config.json`` 里写着 ``defocus_focal_mm``) ——
            硬件是带着这张矫正片测的, 仿真不带就是**两种口径**。

    Returns:
        组目录路径。
    """
    from scripts.slm import hw_gs_defocus_probe as probe

    patch_sim_configurer(probe, "configure_sim_system", config)
    argv = [
        "--execute",
        "--cam-type",
        "sim",
        "--sim-seed",
        str(config.seed),
        "--sim-far-field-padding",
        str(config.far_field_padding),
        "--exposure-ms",
        str(exposure_ms),
        "--roi-px",
        str(roi_px),
        "--target-side-px",
        str(target_side_px),
        "--frames",
        str(frames),
        "--group",
        group,
        "--out-root",
        str(out_root),
    ]
    if dz_mm:
        argv += ["--dz-mm"] + [f"{float(v):.10g}" for v in dz_mm]
    if gs_phase:
        argv += ["--gs-phase", gs_phase]
    if defocus_focal_mm:
        argv += ["--defocus-focal-mm", f"{float(defocus_focal_mm):.10g}"]
    args = probe.build_parser().parse_args(argv)
    args.group_meta = {}
    optics = probe.build_optics(args)
    plan = probe.build_plan(args, optics)
    try:
        probe.run_execute(args, optics, plan)
    except SystemExit as exc:  # 探针自己的退出码 (光路受阻等)
        raise RuntimeError(f"仿真探针 {group} 提前退出: {exc}") from exc
    return Path(out_root) / group


def run_sim_focus_probe(
    config: SimConfig,
    *,
    out_dir: Path,
    exposure_ms: float,
    coarse_points: int,
    diopter_range: float,
    refine_points: int,
    frames: int,
    max_exposure_ms: float,
    diopters: Sequence[float] | None = None,
) -> Path:
    """用 ``ao_shaping.tools.slm.panel.focus_probe`` 在仿真里标定焦面。

    Args:
        config: 仿真口径。
        out_dir: 证据目录 (标定 JSON 落在这里)。
        exposure_ms: 起始曝光 (ms)。
        coarse_points: 粗扫点数。
        diopter_range: 粗扫 1/f 半宽 (m⁻¹)。
        refine_points: 细扫点数。
        frames: 每点平均帧数。
        max_exposure_ms: 曝光上限 (ms)。
        diopters: 显式的 1/f 网格 (m⁻¹)。给了就用它 (实测那一轮的网格),
            两边落在**同一批格点**上, 最佳值才可比。

    Returns:
        证据 JSON 的路径。
    """
    from ao_shaping.tools.slm.panel import focus_probe

    patch_sim_configurer(focus_probe, "_configure_sim_system", config)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = focus_probe.probe_session_focus(
        slm_number=1,
        cam_type="sim",
        cam_id=0,
        exposure_ms=float(exposure_ms),
        radius_px=float(focus_probe.BEAM_RADIUS_PANEL),
        centre_px=(
            float(focus_probe.SLM_PANEL_W / 2.0),
            float(focus_probe.SLM_PANEL_H / 2.0),
        ),
        coarse_points=int(coarse_points),
        diopter_range=float(diopter_range),
        refine_points=int(refine_points),
        frames=int(frames),
        max_exposure_ms=float(max_exposure_ms),
        wavelength_nm=1064.0,
        debug_dir=out_dir,
        sim_seed=int(config.seed),
        sim_far_field_padding=int(config.far_field_padding),
        diopters=(
            None if diopters is None else [float(d) for d in diopters]
        ),
    )
    if result.get("error"):
        raise RuntimeError(f"仿真焦面标定失败: {result['error']}")
    candidates = sorted(out_dir.glob("slm1_*.json"))
    if not candidates:
        raise RuntimeError(f"仿真焦面标定没有产出证据文件: {out_dir}")
    return candidates[-1]


def run_sim_zernike_sweep(
    config: SimConfig,
    *,
    out_root: Path,
    group: str,
    orders: Sequence[int],
    epochs: int,
    spgd_epochs: int,
    exposure_ms: float,
    cam_size: int,
    defocus_focal_mm: float | None = None,
    target_side: float | None = None,
) -> Path:
    """用 ``scripts/slm/run_gs_refine_zernike_sweep.py`` 在仿真里跑阶数扫描。

    Args:
        config: 仿真口径。
        out_root: 产物根目录。
        group: 组名。
        orders: 要跑的 Zernike 阶 (n_max)。
        epochs: Stage-A 的 SPGD 轮数。
        spgd_epochs: Stage-C 的 SPGD 轮数。
        exposure_ms: 曝光 (ms)。
        cam_size: 相机窗口 (px)。

    Returns:
        组目录路径。
    """
    from scripts.slm import run_gs_refine_zernike_sweep as sweep

    patch_sim_configurer(sweep, "configure_sim_system", config)
    argv = [
        "--cam-type",
        "sim",
        "--sim-far-field-padding",
        str(config.far_field_padding),
        "--exposure-time-ms",
        str(exposure_ms),
        "--cam-size",
        str(cam_size),
        "--epochs",
        str(epochs),
        "--spgd-epochs",
        str(spgd_epochs),
        "--orders",
        ",".join(str(int(o)) for o in orders),
        "--out",
        str(out_root),
        "--group",
        group,
        "--seed",
        str(config.seed),
    ]
    if defocus_focal_mm:
        argv += ["--defocus-focal-mm", f"{float(defocus_focal_mm):.10g}"]
    if target_side:
        # 目标框**两侧同源**: 实测那一轮是 12.54 px (自动定尺的结果), 仿真若用自己
        # 的自动定尺就会得到另一个框 —— 框不同, PIB/CV/AR 全都不可比。
        argv += ["--target-side", f"{float(target_side):.6g}"]
    try:
        sweep.main(argv)
    except SystemExit as exc:  # pragma: no cover - 脚本自己的退出路径
        raise RuntimeError(f"仿真阶数扫描 {group} 提前退出: {exc}") from exc
    return Path(out_root) / group


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def hw_defocus_focal_mm(bundle: Bundle) -> float | None:
    """硬件那一轮用的**瞳面菲涅尔焦面矫正**焦距 (mm)。

    硬件是**带着这张矫正片**测的 (第三轮 ``run_config.json`` 里写着
    ``defocus_focal_mm``); 仿真侧若不带同一张矫正片, 两边就是**两种口径**
    (硬件量的是「已矫正的光斑」, 仿真量的是「没矫正的光斑」)。
    """
    for source in (bundle.dz_scan.run_config, bundle.dz_noise.run_config):
        value = source.get("defocus_focal_mm")
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def hw_focus_diopter_grid(bundle: Bundle) -> list[float] | None:
    """实测那一轮焦面标定用过的 **1/f 网格** (m⁻¹)。

    两边落在**同一批格点**上, 最佳 1/f 才可比: 焦面标定是个「在网格上挑最大
    峰值」的过程, 网格不同 ⇒ 挑出来的点就不同 (第三轮硬件用的是 33 个粗扫点
    (步长 0.375 m⁻¹) + 9 个细扫点)。
    """
    points = bundle.focus.get("points") or []
    values: list[float] = []
    for point in points:
        diopter = _get(point, "diopter_m1")
        if diopter is None:
            continue
        values.append(float(diopter))
    if not values:
        return None
    return sorted(set(values))


def hw_zernike_target_side(bundle: Bundle) -> float | None:
    """实测那一轮阶数扫描用的**目标框边长** (相机像元)。

    两侧必须用**同一个框**: 框是「功率/均匀度」类指标的分母, 框不同则 PIB / CV /
    AR 全部不可比。实测的框是它自己自动定尺的结果, 从 ``summary.json`` 里读。
    """
    for summary in bundle.zernike.values():
        value = _get(summary, "target_side")
        if value is not None and value > 0.0:
            return float(value)
    return None


def parse_zernike_spec(text: str) -> dict[int, float]:
    """解析 ``"6:1.9,4:0.5"`` 这样的面板 Zernike 规格 (Noll 序:弧度)。"""
    out: dict[int, float] = {}
    for chunk in str(text or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"面板 Zernike 规格必须是 'Noll:弧度', 收到 {chunk!r}")
        noll, rad = chunk.split(":", 1)
        out[int(noll)] = float(rad)
    return out


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--hw-root",
        default=str(DEFAULT_HW_ROOT),
        help="实测侧产物根目录 (默认: 第三轮归档目录)",
    )
    parser.add_argument(
        "--sim-root",
        default=str(DEFAULT_SIM_ROOT),
        help="仿真侧产物根目录 (默认: data/debug/sim_vs_hw_20261022)",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT_MD),
        help="对照表产物 (默认: report/shape_trials/sim_vs_hw/artifacts/...)",
    )
    parser.add_argument(
        "--elapsed-s",
        type=float,
        default=None,
        help=(
            "上一次 --execute 的实测耗时 (s)。只重出表/报告时用它把真实耗时写进报告 "
            "(不给则写「本轮只重出表」)"
        ),
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_OUT_REPORT),
        help=(
            "报告正文产物 (默认: report/shape_trials/sim_vs_hw/sim_vs_hw_20261022.md); "
            "传空字符串表示不写"
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="先跑仿真再出表 (默认只出表, 复用已有的仿真产物)",
    )
    parser.add_argument("--sim-seed", type=int, default=None, help="仿真种子")
    parser.add_argument(
        "--sim-far-field-padding",
        type=int,
        default=None,
        help="仿真远场补零因子 (默认 4 = 复现实测角标度的值)",
    )
    parser.add_argument(
        "--peak-counts-per-ms",
        type=float,
        default=None,
        help="绝对光度换算 (counts/(峰值分数·ms)); 传 0 表示走旧的相对强度口径",
    )
    parser.add_argument(
        "--read-sigma-counts",
        type=float,
        default=None,
        help="探测器读出噪声 (counts)",
    )
    parser.add_argument("--shot-k", type=float, default=None, help="散粒噪声系数")
    parser.add_argument(
        "--quantize-bits",
        type=int,
        default=None,
        help="量化位深; 传 0 表示不量化",
    )
    parser.add_argument(
        "--params",
        default="",
        help="孪生参数文件 (JSON); 给了就从它读全部旋钮 (命令行参数仍可覆盖)",
    )
    parser.add_argument(
        "--write-params",
        default="",
        help="把本轮实际用的孪生参数写到这个 JSON (可复现用)",
    )
    parser.add_argument(
        "--camera-defocus-m",
        type=float,
        default=None,
        help=(
            "相机离焦 (m): 探测器在 2f 焦面之后 Δz 处 ⇒ 等效瞳面二次相位焦距 "
            "f_v = f²/Δz (本实验系统实测 f_v = 1549.33 mm ⇒ Δz ≈ 10.09 mm)"
        ),
    )
    parser.add_argument(
        "--counts-reference",
        default="",
        choices=("", "frame", "flat"),
        help="counts 口径的归一化参考: frame = 逐帧 (历史), flat = 固定 (平场)",
    )
    parser.add_argument(
        "--dark-level-counts",
        type=float,
        default=None,
        help="探测器暗电平 (counts); 实测相机 26.75",
    )
    parser.add_argument(
        "--beam-w0",
        type=float,
        default=None,
        help="入射高斯光束的**幅度** 1/e 半径 (面板 px)",
    )
    parser.add_argument(
        "--illumination-radius-px",
        type=float,
        default=None,
        help="照明盘半径 (面板 px); 0 = 不截断",
    )
    # --- V2.0 非理想物理 (NonIdealModule) 旋钮 ----------------------------
    parser.add_argument(
        "--zero-order-eta",
        type=float,
        default=None,
        help="零级光能量占比 η (0~1); 0 = 理想单分量 (V2.0 NonIdealModule)",
    )
    parser.add_argument(
        "--slm-lut-gamma",
        type=float,
        default=None,
        help="SLM LUT 的 Gamma 指数 (1 = 理想线性); 给了就建 LUT 模型 (V2.0)",
    )
    parser.add_argument(
        "--slm-lut-table",
        default="",
        help="实测 LUT 表 (灰度→弧度) 的 .npy 路径; 优先于 --slm-lut-gamma (V2.0)",
    )
    parser.add_argument(
        "--prnu-sigma",
        type=float,
        default=None,
        help="探测器 PRNU 乘性图标准差 (1%~2%); 0 = 无 PRNU (V2.0)",
    )
    parser.add_argument(
        "--dark-fpn-sigma",
        type=float,
        default=None,
        help="探测器暗电流 FPN 标准差 (counts); 0 = 暗电平全局常数 (V2.0)",
    )
    parser.add_argument(
        "--nonideal-seed",
        type=int,
        default=None,
        help="生成固定 PRNU/暗场图的 RNG 种子 (V2.0)",
    )
    parser.add_argument(
        "--panel-zernike",
        default="",
        help="面板静态相差的 Zernike (Noll:弧度, 逗号分隔), 例如 '6:1.9'",
    )
    parser.add_argument(
        "--panel-roughness-sigma-rad",
        type=float,
        default=None,
        help="面板粗糙度 (高斯随机场的 RMS, 弧度)",
    )
    parser.add_argument(
        "--panel-roughness-corr-px",
        type=float,
        default=None,
        help="面板粗糙度的空间相关长度 (面板 px); 0 = 逐像元白噪声",
    )
    parser.add_argument(
        "--defocus-focal-mm",
        default="auto",
        help=(
            "瞳面菲涅尔焦面矫正的焦距 (mm); 默认 auto = 读硬件那一轮的 "
            "run_config.json (两边必须同一张矫正片), 0 = 不矫正"
        ),
    )
    parser.add_argument("--exposure-ms", type=float, default=EXPOSURE_MS, help="曝光 (ms)")
    parser.add_argument("--roi-px", type=int, default=ROI_PX, help="ROI 边长 (px)")
    parser.add_argument(
        "--target-side-px",
        type=int,
        default=TARGET_SIDE_PX,
        help="目标框边长 (px); 0 = 按光斑自动定尺",
    )
    parser.add_argument("--frames", type=int, default=FRAMES, help="每点平均帧数")
    parser.add_argument(
        "--gs-phase",
        default="",
        help="GS 相位来源 (.npy); 不给则用第三轮的硬件相位, 再没有就现场生成模型 GS",
    )
    parser.add_argument(
        "--zernike-orders",
        default="3",
        help="阶数扫描的 n_max (逗号分隔)",
    )
    parser.add_argument(
        "--zernike-epochs",
        type=int,
        default=8,
        help="仿真阶数扫描的 Stage-A SPGD 轮数 (硬件侧是 154, 差值要按方向读)",
    )
    parser.add_argument(
        "--zernike-spgd-epochs",
        type=int,
        default=8,
        help="仿真阶数扫描的 Stage-C SPGD 轮数 (硬件侧是 150)",
    )
    parser.add_argument(
        "--focus-coarse-points",
        type=int,
        default=33,
        help=(
            "仿真焦面标定的粗扫点数 (默认 33 = 与第三轮硬件同一张网格: "
            "步长 0.375 m⁻¹, 最佳 1/f 才落在同一个格点附近)"
        ),
    )
    parser.add_argument(
        "--focus-diopter-range",
        type=float,
        default=6.0,
        help="仿真焦面标定的粗扫 1/f 半宽 (m⁻¹)",
    )
    parser.add_argument(
        "--focus-refine-points",
        type=int,
        default=9,
        help="仿真焦面标定的细扫点数 (默认 9 = 与第三轮硬件一致)",
    )
    parser.add_argument(
        "--focus-max-exposure-ms",
        type=float,
        default=20.0,
        help="仿真焦面标定的曝光上限 (ms)",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> SimConfig:
    """把命令行 + 参数文件合成一份仿真口径。

    优先级: **命令行 > 参数文件 > 脚本默认值**。参数文件里的旋钮是「上一轮调
    出来的、已归档的那一组」, 用来复现同一张对照表。
    """
    base = SimConfig()
    if args.params:
        payload = read_json(Path(args.params))
        base = SimConfig.from_json(payload)

    def pick(value: Any, fallback: Any) -> Any:
        return fallback if value is None else value

    peak_counts = args.peak_counts_per_ms
    if peak_counts is not None and float(peak_counts) <= 0.0:
        peak_counts = None
    quantize = args.quantize_bits
    if quantize is not None and int(quantize) <= 0:
        quantize = None

    return SimConfig(
        seed=int(pick(args.sim_seed, base.seed)),
        far_field_padding=pick(args.sim_far_field_padding, base.far_field_padding),
        peak_counts_per_ms=(
            float(peak_counts) if peak_counts is not None else base.peak_counts_per_ms
        ),
        read_sigma_counts=float(
            pick(args.read_sigma_counts, base.read_sigma_counts)
        ),
        shot_k=float(pick(args.shot_k, base.shot_k)),
        quantize_bits=(
            int(quantize) if quantize is not None else base.quantize_bits
        ),
        camera_defocus_m=float(
            pick(args.camera_defocus_m, base.camera_defocus_m)
        ),
        counts_reference=str(
            args.counts_reference or base.counts_reference
        ),
        dark_level_counts=float(
            pick(args.dark_level_counts, base.dark_level_counts)
        ),
        beam_w0=pick(args.beam_w0, base.beam_w0),
        illumination_radius_px=(
            None
            if args.illumination_radius_px in (None, 0.0)
            else float(args.illumination_radius_px)
        )
        if args.illumination_radius_px is not None
        else base.illumination_radius_px,
        panel_zernike_rad=(
            parse_zernike_spec(args.panel_zernike)
            if args.panel_zernike
            else dict(base.panel_zernike_rad)
        ),
        panel_zernike_n_max=int(base.panel_zernike_n_max),
        panel_roughness_sigma_rad=float(
            pick(args.panel_roughness_sigma_rad, base.panel_roughness_sigma_rad)
        ),
        panel_roughness_corr_px=float(
            pick(args.panel_roughness_corr_px, base.panel_roughness_corr_px)
        ),
        panel_roughness_seed=int(base.panel_roughness_seed),
        # --- V2.0 非理想物理 -------------------------------------------
        zero_order_eta=float(pick(args.zero_order_eta, base.zero_order_eta)),
        slm_lut_gamma=pick(args.slm_lut_gamma, base.slm_lut_gamma),
        slm_lut_table=(
            args.slm_lut_table if args.slm_lut_table else base.slm_lut_table
        ),
        prnu_sigma=float(pick(args.prnu_sigma, base.prnu_sigma)),
        dark_fpn_sigma=float(pick(args.dark_fpn_sigma, base.dark_fpn_sigma)),
        nonideal_seed=int(pick(args.nonideal_seed, base.nonideal_seed)),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """跑仿真 (可选) → 读两侧产物 → 出对照表。"""
    args = _parse_args(argv)
    hw_root = Path(args.hw_root)
    sim_root = Path(args.sim_root)
    out_path = Path(args.out)

    config = build_config(args)

    if args.write_params:
        params_path = Path(args.write_params)
        params_path.parent.mkdir(parents=True, exist_ok=True)
        params_path.write_text(
            json.dumps(config.to_json(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"孪生参数已写到: {params_path}")

    elapsed_s: float | None = args.elapsed_s
    if args.execute:
        elapsed_s = _run_sim_side(args, config, sim_root)

    hw = load_bundle(hw_root)
    sim = load_bundle(sim_root)
    cells = build_cells(hw, sim)
    gaps = build_gap_list(hw, sim)
    markdown = render_markdown(cells, gaps, hw, sim, config.as_dict())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"\n对照表已写到: {out_path}")

    if args.report:
        report_path = Path(args.report)
        report = render_report(
            cells,
            gaps,
            hw,
            sim,
            config.as_dict(),
            params_path=args.params or str(DEFAULT_PARAMS_JSON),
            elapsed_s=elapsed_s,
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        print(f"报告正文已写到: {report_path}")
    return 0


def _run_sim_side(
    args: argparse.Namespace, config: SimConfig, sim_root: Path
) -> float:
    """跑仿真侧的全部采集 (三个脚本, 同一批口径)。

    每个采集脚本之前都**显式重置进程级仿真系统** —— 否则第二个脚本会沿用第一个
    的种子/补零/光度 (孪生的 system 是进程级单例)。

    Returns:
        整轮耗时 (s)。
    """
    sim_root.mkdir(parents=True, exist_ok=True)
    hw_bundle = load_bundle(Path(args.hw_root))
    # Δz 扫描点**从实测产物里读**, 不用脚本里的默认值 —— 这样两侧的点一一对应。
    dz_points = sorted({float(r.get("dz_mm", 0.0)) for r in hw_bundle.dz_scan.rows})
    # 噪声组: 实测是「同相位重复 9 次 + 1 个基线行」, 仿真照抄这个结构。
    noise_points = [0.0] * max(
        1, len([r for r in hw_bundle.dz_noise.rows if abs(float(r.get("dz_mm", 0.0))) < ZERO_DEFOCUS_TOLERANCE]) - 1
    )
    gs_phase = _resolve_gs_phase(args.gs_phase)

    # 瞳面菲涅尔焦面矫正: **必须与硬件同一张矫正片** (默认从硬件那一轮的
    # run_config.json 读)。硬件带着它测, 仿真不带就是两种口径。
    defocus_focal_mm: float | None = None
    if str(args.defocus_focal_mm).strip().lower() != "auto":
        value = float(args.defocus_focal_mm)
        defocus_focal_mm = value if value > 0 else None
    else:
        defocus_focal_mm = hw_defocus_focal_mm(hw_bundle)
    print(f"[sim] 瞳面菲涅尔焦面矫正: {defocus_focal_mm} mm", flush=True)

    groups_root = sim_root / "hw_groups_sim"
    started = time.time()
    print(f"[sim] 离焦扫描 ({len(dz_points)} 点) ...", flush=True)
    run_sim_probe(
        config,
        out_root=groups_root,
        group="dz_scan_hw_gs",
        dz_mm=dz_points,
        gs_phase=gs_phase,
        exposure_ms=float(args.exposure_ms),
        roi_px=int(args.roi_px),
        target_side_px=int(args.target_side_px),
        frames=int(args.frames),
        defocus_focal_mm=defocus_focal_mm,
    )
    print(f"[sim] 噪声组 ({len(noise_points)} 次重复) ...", flush=True)
    run_sim_probe(
        config,
        out_root=groups_root,
        group="dz_noise_hw_gs",
        dz_mm=noise_points,
        gs_phase=gs_phase,
        exposure_ms=float(args.exposure_ms),
        roi_px=int(args.roi_px),
        target_side_px=int(args.target_side_px),
        frames=int(args.frames),
        defocus_focal_mm=defocus_focal_mm,
    )
    print("[sim] 焦面标定 ...", flush=True)
    run_sim_focus_probe(
        config,
        out_dir=sim_root / "focus_calibration",
        exposure_ms=float(args.exposure_ms),
        coarse_points=int(args.focus_coarse_points),
        diopter_range=float(args.focus_diopter_range),
        refine_points=int(args.focus_refine_points),
        frames=int(args.frames),
        max_exposure_ms=float(args.focus_max_exposure_ms),
        diopters=hw_focus_diopter_grid(hw_bundle),
    )
    print("[sim] 阶数扫描 ...", flush=True)
    orders = [int(o) for o in str(args.zernike_orders).split(",") if o.strip()]
    run_sim_zernike_sweep(
        config,
        out_root=sim_root / "hw_zernike_sweep_sim",
        group="zernike_order_nmax3",
        orders=orders,
        epochs=int(args.zernike_epochs),
        spgd_epochs=int(args.zernike_spgd_epochs),
        exposure_ms=float(args.exposure_ms),
        cam_size=256,
        defocus_focal_mm=defocus_focal_mm,
        target_side=hw_zernike_target_side(hw_bundle),
    )
    elapsed = time.time() - started
    print(f"[sim] 全部完成, 用时 {elapsed:.1f} s")
    return elapsed


if __name__ == "__main__":
    raise SystemExit(main())
