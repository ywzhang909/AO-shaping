"""GS 方扫报告的**数据模型与常量**。

从 ``generate_hw_gs_square_scans_report.py`` 抽出的自包含单元：本模块只依赖
标准库/numpy/matplotlib，**不 import 仓库内任何模块**，因此不会被反向依赖
（抽出的常数与 dataclass 是整份报告的公共底座）。

抽出理由：它们被报告里几乎所有函数引用，留在原文件会让任何一次"按主题拆函数"
的尝试都撞上循环导入 —— 先把公共底座搬走，上面的函数层才切得开。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np

# 与 facade 同在 scripts/slm/ 下，parents[2] 算出的仓库根一致；
# 这里自带一份，使本模块不依赖 facade（否则会成环）。
_ROOT = Path(__file__).resolve().parents[2]
ROOT = _ROOT

REPORT_KEY = "report/shape_trials/hw_gs_square_scans/hw_gs_square_scans_20261023.md"

OUT_DIR = ROOT / "report" / "shape_trials" / "hw_gs_square_scans"

DEFAULT_RUN_ROOT = ROOT / "data" / "debug" / "hw_gs_square_scans" / "20261023_gs_square"

DEFAULT_LOG = ROOT / "logs" / "gs_square_scan.log"

FIG_TAG = "_20261023"

STAGES: tuple[str, ...] = ("sides", "zernike", "blaze")

VARIABLE: dict[str, str] = {
    "sides": "side_px",
    "zernike": "n_max",
    "blaze": "blaze_period_px",
}

STAGE_CN: dict[str, str] = {
    "sides": "边长扫描",
    "zernike": "Zernike 阶数扫描",
    "blaze": "闪耀光栅扫描",
}

SATURATION_COUNTS = 255.0

DARK_FRAME_FRACTION = 0.10

OUTLIER_CV_RATIO = 1.5

ROW_MATCH_ATOL = 1e-9

PARAM_NPZ_CSV_ATOL = 1e-6

SIDE_PX_NPZ_CSV_ATOL = 1e-9

SCORE_IDENTITY_ATOL = 1e-6

AREA_IDENTITY_ATOL = 1e-3

EXPOSURE_MATCH_ATOL = 1e-9

PEAK_MATCH_ATOL = 1e-3

DEFAULT_RETEST_ROOT = ROOT / "data" / "debug" / "hw_gs_square_scans" / "20261023_retest_frozen"

DEFAULT_RETEST_PRIOR_ROOT = (
    ROOT / "data" / "debug" / "hw_gs_square_scans" / "20261023_retest_power_changed"
)

DEFAULT_RETEST_LOG_DIR = ROOT / "logs"

DEFAULT_ATOMIC_ROOT = ROOT / "data" / "debug" / "hw_gs_square_scans" / "20261023_atomic_full"

RETEST_STAGES: tuple[str, ...] = ("blaze", "zernike")

RETEST_STAGE_CN: dict[str, str] = {
    "blaze": "闪耀光栅",
    "zernike": "Zernike 阶数",
}

@dataclass(frozen=True)
class FieldRecord:
    """**现场记录** —— 不在这两批产物里的外部读数。

    存在的唯一理由: 说明「激光功率变了多少」。它**不参与任何 CSV 解析**,
    只出现在 §11.1 一张表头写着「现场记录（非本批产物）」的表里, 并且
    :func:`retest_power_ratio` 的入参就是它 —— 想改数就改这里, 并同时改 ``source``。

    ``readings`` 是 ``(曝光 ms, 峰值 counts)`` 的序列。**斜率只用最大曝光那个点**:
    低曝光读数被噪声地板抬高 (0.05 ms 只有 3 counts), 拿它定斜率会把功率比算歪。
    """

    label: str
    readings: tuple[tuple[float, float], ...]
    source: str

    def anchor(self) -> tuple[float, float]:
        """返回 ``(曝光 ms, 峰值 counts)`` 里曝光最大的那个点。"""
        return max(self.readings, key=lambda pair: pair[0])

    def counts_per_ms(self) -> float:
        exposure_ms, peak = self.anchor()
        return peak / exposure_ms

FIELD_BASELINE_FLAT = FieldRecord(
    label="基线同期 平场 0 级",
    readings=((0.5, 163.0),),
    source="logs/probe_light_exposures.py 现场读数（2026-10-23，基线同一光路状态）",
)

FIELD_RETEST_FLAT = FieldRecord(
    label="复测前 平场 0 级",
    readings=((0.05, 3.0), (0.2, 8.0), (1.0, 38.0), (5.0, 180.0)),
    source="logs/probe_light_exposures.py 现场读数（2026-10-23，复测开始前）",
)

@dataclass(frozen=True)
class Cell:
    """一个表格单元。

    ``value`` 是**数值本身** (来自 CSV 原文或现场推导), ``nd`` 是显示精度。
    回读校验的容差就是 ``0.5 · 10^-nd`` —— 即"显示精度内的半个末位", 所以只要
    渲染时多丢一位小数, 校验立刻失败。
    """

    value: float
    nd: int = 4
    suffix: str = ""

    def text(self) -> str:
        if not math.isfinite(self.value):
            return "-"
        return f"{self.value:.{self.nd}f}{self.suffix}"

    def accepts(self, parsed: float) -> bool:
        """``parsed`` (从渲染后的 Markdown 解析回来) 是否就是这个数。"""
        if not math.isfinite(self.value):
            return not math.isfinite(parsed)
        return abs(parsed - self.value) <= 0.5 * 10.0 ** (-self.nd) + 1e-12

@dataclass
class Table:
    """一张数值表 + 它自己的回读校验。

    ``rows`` 里既有 :class:`Cell` 也有纯文本 (标记列, 如 ``正常`` / ``饱和``)。
    """

    headers: tuple[str, ...]
    rows: list[tuple[Cell | str, ...]] = field(default_factory=list)

    def render(self) -> str:
        lines = ["| " + " | ".join(self.headers) + " |"]
        lines.append("|" + "|".join(["---"] * len(self.headers)) + "|")
        for row in self.rows:
            lines.append(
                "| " + " | ".join(c.text() if isinstance(c, Cell) else str(c) for c in row) + " |"
            )
        return "\n".join(lines)

    @property
    def numeric_cells(self) -> int:
        return sum(1 for row in self.rows for c in row if isinstance(c, Cell))

@dataclass
class Stage:
    """一段扫描的全部产物 + 解析出来的派生量。"""

    name: str
    headers: list[str]
    rows: list[dict[str, str]]
    run_config: dict[str, Any]
    manifest_entry: dict[str, Any]

    @property
    def variable(self) -> str:
        return VARIABLE[self.name]

    @property
    def x(self) -> list[float]:
        return [float(r[self.variable]) for r in self.rows]

    def col(self, key: str) -> list[float]:
        return [float(r[key]) for r in self.rows]

    def median(self, key: str) -> float:
        vals = [v for v in self.col(key) if math.isfinite(v)]
        return float(np.median(vals)) if vals else float("nan")

@dataclass(frozen=True)
class SettleEvent:
    """日志里的一条 ``settle criterion not met`` 记录。"""

    stage: str
    index: int
    waited_s: float
    probes: int
    probe_peak: float
    probe_fwhm: float
    raw_fragment: str = ""
    row_key: str = ""

    def describe(self) -> str:
        return (
            f"settle 判据超时 {self.waited_s:.2f} s / {self.probes} 次探测 / "
            f"探测峰值 {self.probe_peak:g}"
        )

@dataclass
class Run:
    """一次运行的全部输入。"""

    root: Path
    stages: dict[str, Stage]
    manifest: list[dict[str, Any]]
    plan: dict[str, Any]
    stage_plans: dict[str, dict[str, Any]]
    settle_events: list[SettleEvent]
    log_path: Path | None
    log_crosscheck: list[tuple[str, str, str, bool]]

    # -- 常用参数, 全部来自 run_config / manifest ---------------------------
    @property
    def exposure_ms(self) -> float:
        return float(self.stages["sides"].run_config["exposure_ms_working"])

    @property
    def exposure_requested_ms(self) -> float:
        return float(self.stages["sides"].run_config["exposure_ms_requested"])

    @property
    def exposure_ceiling_ms(self) -> float:
        return float(self.stages["sides"].run_config["exposure_ceiling_ms"])

    @property
    def zero_order_xy(self) -> tuple[float, float]:
        xy = self.stages["sides"].run_config["zero_order_xy_frozen"]
        return float(xy[0]), float(xy[1])

    @property
    def frame_shape_hw(self) -> tuple[int, int]:
        h, w = self.stages["sides"].run_config["frame_shape_hw"]
        return int(h), int(w)

    @property
    def settle_policy(self) -> dict[str, Any]:
        return dict(self.stages["sides"].run_config.get("settle") or {})

    @property
    def spot_assessment(self) -> dict[str, Any]:
        return dict(self.stages["sides"].run_config.get("spot_assessment") or {})

    def settle_for(self, stage: str, index: int) -> SettleEvent | None:
        for ev in self.settle_events:
            if ev.stage == stage and ev.index == index:
                return ev
        return None

@dataclass
class Findings:
    """报告要讲的全部结论 —— 每一个都在这里算出来, 正文只做排版。"""

    sides_x: list[float]
    pib: list[float]
    cv: list[float]
    score: list[float]
    score_peak_index: int
    score_runner_up_index: int
    score_peak_gap: float
    pib_violations: list[int]
    cv_violations: list[int]
    cv_spearman: float
    score_descend_ok: bool
    terms_at_peak: tuple[float, float]
    terms_at_last: tuple[float, float]
    terms_at_small: tuple[float, float]
    near_window: tuple[float, float]
    near_best_index: int
    alt_window: tuple[float, float]
    alt_best_index: int
    selected_side: int
    zern_explained: list[float]
    zern_residual: list[float]
    zern_cv: list[float]
    zern_score: list[float]
    zern_clean_cv: list[float]
    zern_clean_cv_spread: float
    zern_cv_pearson: float
    zern_explained_violations: list[int]
    blaze_score: list[float]
    blaze_score_violations_clean: list[int]
    blaze_score_violations_all: list[int]
    blaze_shift_ratio: list[float]
    model_pib: list[float]
    model_delta: list[float]
    side_max_raw: float
    side_max: float
    step: float
    edge_margin: float
    binding_edge: str
    dropped_configs: list[str]
    csv_cells: int

@dataclass(frozen=True)
class SettleRecord:
    """单配置日志里的一条 settle 告警 (复测批次每个配置一个日志文件)。"""

    waited_s: float
    probes: int
    probe_peak: float
    probe_fwhm: float
    raw_fragment: str

    def describe(self) -> str:
        return (
            f"settle 判据超时 {self.waited_s:.2f} s / {self.probes} 次探测 / "
            f"探测峰值 {self.probe_peak:g}"
        )

@dataclass
class RetestConfig:
    """复测批次里的一个配置 (原子化调用 ⇒ 一个配置 = 一个进程 = 一份 metrics.json + 一个日志)。"""

    stage: str
    config: str
    value: float
    row: dict[str, str]
    metrics: dict[str, Any]
    log_path: Path | None
    settle: list[SettleRecord]

    @property
    def preflight(self) -> dict[str, Any]:
        return dict(self.metrics.get("preflight") or {})

    @property
    def spot(self) -> dict[str, Any]:
        return dict(self.preflight.get("spot_assessment") or {})

    @property
    def zero_order(self) -> tuple[float, float]:
        xy = self.preflight.get("zero_order_xy_frozen") or (float("nan"), float("nan"))
        return float(xy[0]), float(xy[1])

    @property
    def spot_flagged(self) -> bool:
        """平场评估自己报「没有可评分的局域光斑」(饱和平台让 ``argmax`` 失义)。"""
        return str(self.spot.get("reason", "")).startswith("光路有光但没有可评分的局域光斑")

@dataclass
class RetestStage:
    name: str
    headers: list[str]
    rows: list[dict[str, str]]
    configs: list[RetestConfig]
    #: 段级 ``run_config.json`` (v1 那批没有 ⇒ 空 dict)。**没有 `plan`** 是正常的。
    run_config: dict[str, Any] = field(default_factory=dict)

    @property
    def variable(self) -> str:
        return VARIABLE[self.name]

    @property
    def x(self) -> list[float]:
        """自变量列 (与 :class:`Stage` 同名, 让 ``compute_findings`` 能直接吃这一批)。"""
        return [float(r[self.variable]) for r in self.rows]

    def row_of(self, value: float) -> dict[str, str]:
        for row in self.rows:
            if abs(float(row[self.variable]) - float(value)) < ROW_MATCH_ATOL:
                return row
        raise KeyError(f"复测 {self.name} 段没有取值 {value} 的行")

    def col(self, key: str) -> list[float]:
        return [float(r[key]) for r in self.rows]

    def median(self, key: str) -> float:
        """与 :meth:`Stage.median` 同名同义。"""
        vals = [v for v in self.col(key) if math.isfinite(v)]
        return float(np.median(vals)) if vals else float("nan")

@dataclass
class RetestRun:
    root: Path
    manifest: list[dict[str, Any]]
    stages: dict[str, RetestStage]
    log_dir: Path

    @property
    def exposure_ms(self) -> float:
        """生效曝光 —— 从**每配置的 preflight** 读, 并强制它们一致。"""
        values = {float(cfg.preflight.get("exposure_ms_working", float("nan")))
                  for st in self.stages.values() for cfg in st.configs}
        if len(values) != 1:
            raise ValueError(f"复测各配置的生效曝光不一致: {sorted(values)}")
        return values.pop()

    @property
    def exposure_ceiling_ms(self) -> float:
        values = {float(cfg.preflight.get("exposure_ceiling_ms", float("nan")))
                  for st in self.stages.values() for cfg in st.configs}
        if len(values) != 1:
            raise ValueError(f"复测各配置的曝光上限不一致: {sorted(values)}")
        return values.pop()

    @property
    def stage_plans(self) -> dict[str, Any]:
        """原子化批次没有 ``plan`` ⇒ 空 dict (``compute_findings`` 只用它算"计划里被砍掉的配置")。"""
        return {}

    def _sides_cfg(self) -> dict[str, Any]:
        stage = self.stages.get("sides")
        return dict(stage.run_config) if stage is not None else {}

    @property
    def zero_order_xy(self) -> tuple[float, float]:
        """段级冻结 0 级 (只有带 sides 段的整轮批次才有)。"""
        return run_zero_order(self._sides_cfg())

    @property
    def frame_shape_hw(self) -> tuple[int, int]:
        shape = self._sides_cfg().get("frame_shape_hw") or ()
        if len(shape) != 2:
            return 0, 0
        return int(shape[0]), int(shape[1])

    @property
    def all_configs(self) -> list[RetestConfig]:
        return [cfg for name in RETEST_STAGES for cfg in self.stages[name].configs]

    @property
    def settle_count(self) -> int:
        return sum(len(cfg.settle) for cfg in self.all_configs)

    def settle_for(self, stage: str, index: int) -> SettleRecord | None:
        """该配置自己的第一条 settle 告警 (没有就 ``None``)。

        与 :meth:`Run.settle_for` 同名同义 —— 复测/整轮批次的日志是**每配置一个文件**,
        所以按配置下标取, 不需要像基线那样按"下一行打印"归属。
        """
        st = self.stages.get(stage)
        if st is None or not (0 <= index < len(st.configs)):
            return None
        events = st.configs[index].settle
        return events[0] if events else None

    @property
    def selected_source(self) -> str:
        values = {str(r.get("selected_source", "?")) for st in self.stages.values() for r in st.rows}
        return "/".join(sorted(values))

    @property
    def mode(self) -> str:
        modes = {str((e.get("summary") or {}).get("mode", "?")) for e in self.manifest}
        return "/".join(sorted(modes))

@dataclass
class RetestFindings:
    """§11 要用的全部派生量 —— 每一个都现场算, 不复用任何字面量。"""

    power_ratio: float
    baseline_counts_per_ms: float
    retest_counts_per_ms: float
    equivalent_exposure_ms: float
    baseline_exposure_ms: float
    retest_exposure_ms: float
    # -- 批内邻居: 40 / 18 相对左右邻居中位数的相对偏差 --------------------
    blaze_cv: list[float]
    blaze_cv_monotone: bool
    blaze40_dev: float
    baseline_blaze40_dev: float
    zern_cv: list[float]
    zern_cv_monotone: bool
    zern18_dev: float
    baseline_zern18_dev: float
    # -- 位移: 方向 / 标度 / 1/P 形状 --------------------------------------
    offset_identity_residual: float
    ratio_retest: list[float]
    ratio_baseline: list[float]
    alpha_retest: float
    alpha_baseline: float
    alpha_predicted: float
    measured_ends_retest: tuple[float, float, float]
    measured_ends_baseline: tuple[float, float, float]
    predicted_ends_retest: tuple[float, float, float]
    predicted_ends_baseline: tuple[float, float, float]
    offset_y_small_period: list[tuple[float, float]]
    # -- 原点抖动 ----------------------------------------------------------
    zero_order_x_span: float
    zero_order_y_span: float
    zero_order_x_span_zern: float
    zero_order_y_span_zern: float
    centroid_x_span: float
    centroid_y_span: float
    measured_x_span: float
    measured_y_span: float
    offset_y_vs_baseline: list[tuple[float, float, float]]
    # -- 拟合一致 / 标记 ---------------------------------------------------
    fit_bit_identical: list[float]
    fit_differs: list[float]
    flagged_configs: list[str]
    settle_configs: list[str]
    dark_probe_configs: list[str]
    retest_rows: int
    # -- 批内一致性 (单批, 不跨批): 相对左右邻居中位数的最大偏差 --------------
    batch_dev_max: float
    prior_root: Path | None

CROP_FACTOR = 1.4

SIDES_PANEL_PX: tuple[float, ...] = (200.0, 600.0, 1000.0, 1500.0, 1800.0)

ZERNIKE_PANEL_ORDER: tuple[float, ...] = (30.0, 19.0, 12.0, 7.0)

BLAZE_PANEL_PERIOD: tuple[float, ...] = (32.0, 64.0, 128.0)

BLAZE_PANEL_OPTIONAL: float = 16.0

FULLFRAME_TARGET_PX = 1000.0

@dataclass(frozen=True)
class MosaicSpec:
    """一张拼图。"""

    key: str          # sides / zernike / blaze / fullframe
    stem: str         # 文件名主干 (不含 batch 后缀)
    title: str
    ncols: int
    cell_inch: float
    crop: bool

MOSAIC_CN: dict[str, str] = {
    "sides": "1 · 边长",
    "zernike": "2 · Zernike",
    "blaze": "3 · 光栅",
    "fullframe": "4 · 整帧",
}

MOSAICS: tuple[MosaicSpec, ...] = (
    MosaicSpec("sides", "fig6_farfield_sides", "边长段: 方斑随边长变化", 3, 3.4, True),
    MosaicSpec("zernike", "fig7_farfield_zernike", "Zernike 段: 低阶重构后的远场", 3, 3.4, True),
    MosaicSpec("blaze", "fig8_farfield_blaze", "闪耀光栅段: 光栅把方斑搬走", 2, 3.6, True),
    MosaicSpec("fullframe", "fig9_farfield_fullframe", "整帧: 0 级不在画面中心", 1, 7.2, False),
)

FLAG_LEGEND = (
    "`†` = 该行采集时 **settle 判据超时** (日志原文 `settle criterion not met`); "
    "`‡` = 该行整帧电平低于同段中位数的 "
    f"{DARK_FRAME_FRACTION:.0%} (帧近乎全黑); "
    f"`※` = 该行 CV 超过同段中位数的 {OUTLIER_CV_RATIO:.1f}×; "
    "`§` = 该行方框有像素落在画面外 (`box_clipped_px > 0`)。"
)

@dataclass
class AtomicFindings:
    """§13 要用的派生量。全部现场算, 不内联任何数字。"""

    exposure_ms: float
    ceiling_ms: float
    mode: str
    zero_order: tuple[float, float]
    side_max: float
    side_max_raw: float
    step: float
    rejected: tuple[float, ...]
    n_sides: int
    selected_side: float
    selected_score: float
    score_max: float
    score_argmax_side: float
    plateau: list[tuple[float, float]]
    plateau_gap: float
    tail: list[tuple[float, float]]
    box_peak_min: float
    box_peak_max: float
    frame_peak_values: list[float]
    ar_median: float
    ar_outliers: list[tuple[float, float]]
    zern_cv_min: float
    zern_cv_max: float
    zern_pib_min: float
    zern_pib_max: float
    zern_explained_min: float
    zern_explained_max: float
    zern_modes_max: float
    tier_split: float | None
    tier_low: list[tuple[float, float, float]]
    tier_high: list[tuple[float, float, float]]
    centroid_x_span: float
    blaze: list[tuple[float, float, float, float, float, float, float, float]]
    blaze_meas_min: tuple[float, float]
    blaze_meas_max: tuple[float, float]
    blaze_ratio_min: float
    blaze_ratio_max: float
    blaze_flat_ratio: float
    blaze_predicted_ratio: float
    blaze_clipped: list[float]
    cross: list[tuple[str, float, float, float, float, float, float, float]]
    cross_cv_max: float
    cross_score_max: float

def run_zero_order(cfg: dict[str, Any]) -> tuple[float, float]:
    """段级 ``zero_order_xy_frozen`` (取不到就给 nan)。"""
    xy = cfg.get("zero_order_xy_frozen")
    if not xy:
        return float("nan"), float("nan")
    return float(xy[0]), float(xy[1])

RESIDUAL_PANEL_ORDER: tuple[int, ...] = (30, 19, 12, 7)

RESIDUAL_VLIM = math.pi

RESIDUAL_CMAP = "RdBu_r"

BASE_CMAP = "twilight"

RESIDUAL_RMS_RTOL = 0.02

RESIDUAL_RECON_ATOL = 1e-9

RESIDUAL_HIGH_FREQ = 0.25

@dataclass
class ResidualPanel:
    """fig12 的一列: 一个 ``n_max`` 的残差相位 + 它的结构描述量。"""

    n_max: int
    config: str
    row: dict[str, str]
    modes: float
    #: `metrics.json` / CSV 的记录值 (§13.8 表格与图标题用的就是它们)。
    rms_recorded: float
    explained: float
    #: 现场重算值 (与记录值对拍, 容差见 :data:`RESIDUAL_RMS_RTOL`)。
    rms_recomputed: float
    recon_delta: float
    #: 残差相位 (fit_grid², 已卷到 ±π, 单位圆外 0) 与它的统计掩码。
    residual: np.ndarray
    mask: np.ndarray
    #: 结构描述量 (全部现场算, 见 :meth:`structure_note`)。
    mean_freq: float
    high_share: float
    radial_ratio: float
    m4_share: float

    def structure_note(self) -> str:
        """「残差里的可见结构」一句话 (纯数据派生, 不含任何手写判断)。"""
        return (
            f"细颗粒高频（平均 {self.mean_freq:.2f} cyc/px，"
            f"{self.high_share:.0%} 能量在 >{RESIDUAL_HIGH_FREQ:g} cyc/px）、"
            f"径向近均匀（外缘/中心 RMS {self.radial_ratio:.2f}）、"
            f"m=4 对称仅 {self.m4_share:.1%}"
        )

@dataclass
class ResidualSet:
    """§13.8 的全部输入: 一张基准相位 + 每列一个 :class:`ResidualPanel`。"""

    base: np.ndarray
    base_path: Path
    fit_grid: int
    panels: list[ResidualPanel]


__all__ = [
    "AREA_IDENTITY_ATOL",
    "AtomicFindings",
    "BASE_CMAP",
    "BLAZE_PANEL_OPTIONAL",
    "BLAZE_PANEL_PERIOD",
    "CROP_FACTOR",
    "Cell",
    "DARK_FRAME_FRACTION",
    "DEFAULT_ATOMIC_ROOT",
    "DEFAULT_LOG",
    "DEFAULT_RETEST_LOG_DIR",
    "DEFAULT_RETEST_PRIOR_ROOT",
    "DEFAULT_RETEST_ROOT",
    "DEFAULT_RUN_ROOT",
    "EXPOSURE_MATCH_ATOL",
    "FIELD_BASELINE_FLAT",
    "FIELD_RETEST_FLAT",
    "FIG_TAG",
    "FLAG_LEGEND",
    "FULLFRAME_TARGET_PX",
    "FieldRecord",
    "Findings",
    "MOSAICS",
    "MOSAIC_CN",
    "MosaicSpec",
    "OUTLIER_CV_RATIO",
    "OUT_DIR",
    "PARAM_NPZ_CSV_ATOL",
    "PEAK_MATCH_ATOL",
    "REPORT_KEY",
    "RESIDUAL_CMAP",
    "RESIDUAL_HIGH_FREQ",
    "RESIDUAL_PANEL_ORDER",
    "RESIDUAL_RECON_ATOL",
    "RESIDUAL_RMS_RTOL",
    "RESIDUAL_VLIM",
    "RETEST_STAGES",
    "RETEST_STAGE_CN",
    "ROW_MATCH_ATOL",
    "ResidualPanel",
    "ResidualSet",
    "RetestConfig",
    "RetestFindings",
    "RetestRun",
    "RetestStage",
    "Run",
    "SATURATION_COUNTS",
    "SCORE_IDENTITY_ATOL",
    "SIDES_PANEL_PX",
    "SIDE_PX_NPZ_CSV_ATOL",
    "STAGES",
    "STAGE_CN",
    "SettleEvent",
    "SettleRecord",
    "Stage",
    "Table",
    "VARIABLE",
    "ZERNIKE_PANEL_ORDER",
    "run_zero_order",
]
