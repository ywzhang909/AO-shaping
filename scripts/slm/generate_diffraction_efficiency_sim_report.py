"""衍射效率标定工具的非理想因素仿真报告 (2026-10-22)。

**纯离线、纯仿真**: 全程不打开任何设备 (SLM / 相机 / 光学平台), 只跑仓库里
已经落地的**同一条**分析链路 ——

    ao_shaping.tools.slm.calibration.diffraction_efficiency.core
        .run_sweep / .fit_phase_response / .diagnose

硬件后端 (`measure.py`) 与仿真后端 (`simulate.py`) 共用的就是这条链路, 所以本
报告验的不是"另一条平行实现", 而是实验系统将来要走的那条路。

本轮要回答两个问题
------------------

**Q1**: 仿真能否正确识别不同 SLM 填充因子、LUT 下的衍射效率?
**Q2**: 相关参数 + 面板相差对衍射效率的影响有多大?

四个研究 (每个研究里的每个工况都走同一条 ``run_sweep`` → ``fit_phase_response``
→ ``diagnose`` 链路):

* **A 填充因子**: ``fill_factor ∈ {1.0, 0.9, 0.8, 0.7, 0.6}``, 外加
  ``fill_factor=0.8 且 fill_background_rad=π`` (死区相位翻转) 一个对照工况。
* **B 相位 LUT**: 四种器件律, 真值 ``depth_max_rad = 2π``:
  幂律 γ=1.0 / γ=1.4 / S 形 LUT / 带死区的 LUT。
* **C 面板静态相差**: Noll 序 Zernike (倾斜 (1,1) / 离焦 (2,0) / 像散 (2,2) /
  彗差 (3,1)) 各 0.5 rad, 外加表面粗糙度 σ ∈ {0.1, 0.3} rad。
* **D 光学离焦**: ``defocus_coeff ∈ {0, 1e-5, 3e-5, 6e-5}`` rad/panel px²。

**分析侧拿不到真值**: 器件真值 (相位深度 / γ / LUT) 只写在 ``SimConfig`` 里,
拟合结果出来之后才与真值对账 —— 这样验的才是拟合链路本身。

产物::

    report/slm/diffraction_efficiency/sim_nonideality_20261022.md
    report/slm/diffraction_efficiency/figures/*.png
    report/slm/diffraction_efficiency/artifacts/*.csv

运行::

    python scripts/slm/generate_diffraction_efficiency_sim_report.py            # 全量
    python scripts/slm/generate_diffraction_efficiency_sim_report.py --quick    # 冒烟

术语: 中文叙述遵守仓库术语表: 实验系统, 瞳面 (pupil), 加性背景 (杂散光) 项。
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _common.provenance import insert_header

from ao_shaping.tools.slm.calibration.diffraction_efficiency.core import (
    OVER_MODULATION_RAD,
    UNDER_MODULATION_RAD,
    BlazeSpec,
    FwhmGate,
    PhaseResponseFit,
    SweepPoint,
    blaze_gray,
    diagnose,
    evaluate_fwhm_gate,
    fit_phase_response,
    run_sweep,
)
from ao_shaping.tools.slm.calibration.diffraction_efficiency.simulate import (
    SimConfig,
    SimulatedBench,
)
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# 路径常量 (写死, 见 report/AGENTS.md §2)
# ---------------------------------------------------------------------------
REPORT_KEY = "report/slm/diffraction_efficiency/sim_nonideality_20261022.md"
OUT_DIR = ROOT / "report" / "slm" / "diffraction_efficiency"
FIG_DIR = OUT_DIR / "figures"
ARTE_DIR = OUT_DIR / "artifacts"

# ---------------------------------------------------------------------------
# 仿真口径常量
# ---------------------------------------------------------------------------
#: 闪耀光栅周期 (面板像元)。周期 32 → ±1 级落在 kx/32 = 235.8 CCD px 处。
PERIOD_PX: int = 32
#: 器件满量程灰度 (数字孪生的 ``gray_levels``), 同时是 2π 相位对应的灰度。
GRAY_MAX: int = 1023
#: 几何标定 K (CCD px / panel px), 与实验系统的几何标定同源。
K_CCD_PER_PANEL_PX: float = 7546.4
#: 预期间距 (CCD px)。
PREDICTED_SEPARATION_PX: float = K_CCD_PER_PANEL_PX / float(PERIOD_PX)
#: 全量模式的扫描点数 (与硬件后端默认 ``n_steps=21`` 同口径)。
N_STEPS_FULL: int = 21
#: 冒烟模式的扫描点数。
N_STEPS_QUICK: int = 5
#: 一级衍射的**可探测下限** (归一化命令 u)。低于它时一级衍射峰值低于零级
#: 光斑的 Airy 旁瓣环 (实测 ≈ 零级峰值的 8%), 工具记 "窗口内没有一级衍射光斑"
#: 并把该点的一级效率记为 0 —— 那是**探测下限**, 不是物理零。见 §5。
DETECT_U_MIN: float = 0.30
#: 光斑宽度门禁的倍数 (与工具的 `--fwhm-gate-ratio` 默认值一致)。
GATE_RATIO: float = 3.0
#: 图名戳 (report/AGENTS.md §2.1: 同一系列的每张图都要带报告日期戳)。
FIGURE_STAMP: str = "20261022"
#: 拟合相位深度相对偏差的告警阈值 (5%)。
DEPTH_BIAS_WARN: float = 0.05
#: 拟合 γ 相对偏差的告警阈值 (10%)。
GAMMA_BIAS_WARN: float = 0.10
#: 器件真实满量程相位深度 (所有工况的真值都是 2π)。
TRUTH_DEPTH_RAD: float = 2.0 * math.pi

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 工况定义
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Case:
    """一个仿真工况。

    Attributes:
        key: 唯一标识 (CSV / 图例用)。
        study: 所属研究 ("A" / "B" / "C" / "D")。
        label: 中文标签。
        short: 短标签 (汇总图 x 轴用)。
        config: 数字孪生配置 (器件真值只在这里)。
        truth_gamma: 器件真实幂律指数; ``None`` = 用 LUT 描述 (幂律不适用)。
        lut: 显式的 灰度→相位 查找表 (弧度); ``None`` = 走幂律。
    """

    key: str
    study: str
    label: str
    short: str
    config: SimConfig
    truth_gamma: float | None = None
    lut: tuple[float, ...] | None = None
    note: str = ""

    @property
    def truth_depth_rad(self) -> float:
        """器件真值: 满量程相位深度 (弧度)。"""
        return TRUTH_DEPTH_RAD

    def truth_phase(self, u: np.ndarray) -> np.ndarray:
        """器件真值的 相位(弧度) vs 归一化命令 ``u``。"""
        x = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
        if self.lut is not None:
            index = np.rint(x * (len(self.lut) - 1)).astype(int)
            table = np.asarray(self.lut, dtype=np.float64)
            return table[index]
        gamma = 1.0 if self.truth_gamma is None else float(self.truth_gamma)
        return self.truth_depth_rad * np.power(x, gamma)


@dataclass
class CaseResult:
    """一个工况的完整结果 (扫描点 + 拟合 + 诊断 + 对账指标)。"""

    case: Case
    points: list[SweepPoint]
    fit: PhaseResponseFit
    fit_detectable: PhaseResponseFit
    diagnosis: dict[str, Any]
    peak_eta: float
    peak_eta_command: float
    max_residual_px: float
    max_residual_significant_px: float
    fwhm_ref_px: float
    fwhm_ref_command: float
    max_ellipticity: float
    max_fwhm_px: float
    n_wide_alarm: int
    n_wide_alarm_bright: int
    n_bright: int
    n_points: int
    gate_rel: FwhmGate
    fit_rel: PhaseResponseFit | None
    rel_error: str | None
    gate_abs: FwhmGate
    fit_abs: PhaseResponseFit | None
    abs_error: str | None
    abs_verdict_trustworthy: bool
    phase_dev_max_rad: float
    phase_dev_rms_rad: float
    n_detectable: int

    # -- 对账便捷量 ---------------------------------------------------------
    @property
    def depth_pi(self) -> float:
        return self.fit.depth_max_pi

    @property
    def depth_bias(self) -> float:
        """拟合相位深度的相对偏差 (对真值)。"""
        truth = self.case.truth_depth_rad
        return (self.fit.depth_max_rad - truth) / truth

    @property
    def gamma_bias(self) -> float | None:
        """拟合 γ 的相对偏差; LUT 工况无 γ 真值 → ``None``。"""
        if self.case.truth_gamma is None:
            return None
        return (self.fit.gamma - self.case.truth_gamma) / self.case.truth_gamma

    @property
    def verdict(self) -> str:
        return str(self.diagnosis.get("modulation", "?"))

    @property
    def truth_verdict(self) -> str:
        """按真值深度 + 工具自身阈值推出的"应当"判定。"""
        depth = self.case.truth_depth_rad
        if depth < UNDER_MODULATION_RAD:
            return "under"
        if depth > OVER_MODULATION_RAD:
            return "over"
        return "nominal"

    @property
    def verdict_ok(self) -> bool:
        return self.verdict == self.truth_verdict

    @property
    def depth_pi_detectable(self) -> float:
        """只在"可探测段"上重拟合得到的相位深度 (π) —— 稳健性对照。"""
        return self.fit_detectable.depth_max_pi

    @property
    def depth_pi_gated(self) -> float:
        """**相对门禁**之后的相位深度 (π); 门禁没剔除任何点时与未门禁的一致。"""
        return (self.fit_rel or self.fit).depth_max_pi

    @property
    def gate_rel_cell(self) -> str:
        """表格里的一格: 违例/评估 (max 比) → 深度是否变化。"""
        gate = self.gate_rel
        head = f"{len(gate.violations)}/{gate.evaluated_points} (max {gate.max_ratio:.2f}×)"
        if self.fit_rel is None:
            return f"{head} → **拟合失败**"
        if abs(self.fit_rel.depth_max_pi - self.depth_pi) < 1e-9:
            return f"{head} → 深度不变"
        return (
            f"{head} → 深度 {self.depth_pi:.4f}→{self.fit_rel.depth_max_pi:.4f}π"
        )

    @property
    def gate_abs_cell(self) -> str:
        """表格/正文里的一格: **绝对参照**门禁的结果。

        门禁一旦触发, 被剔除的点**加上**被跳过 (零级被熄灭) 的点一起退出拟合,
        所以离焦工况这里要么给出"有效点 0 → 拟合失败", 要么给出被门禁改过的深度。
        """
        gate = self.gate_abs
        head = f"{len(gate.violations)}/{gate.evaluated_points}"
        if self.fit_abs is None:
            remaining = self.n_points - len(gate.violations) - len(gate.skipped_indices)
            return f"{head} → **拟合失败** (有效点 {remaining})"
        if abs(self.fit_abs.depth_max_pi - self.depth_pi) < 1e-9:
            return f"{head} → 深度不变"
        return f"{head} → 深度 {self.depth_pi:.4f}→{self.fit_abs.depth_max_pi:.4f}π"


# ---------------------------------------------------------------------------
# 器件律 (LUT) 构造
# ---------------------------------------------------------------------------
def smoothstep_lut(gray_levels: int = GRAY_MAX) -> tuple[float, ...]:
    """S 形查找表: 前 15 % 灰度段相位保持 0, 之后按 smoothstep 升到 2π。

    单调不减, 末值恰为 2π; 长度必须是 ``gray_levels + 1`` (数字孪生的硬要求)。

    Args:
        gray_levels: 器件灰度级数 (10 bit = 1023)。

    Returns:
        逐灰度的相位表 (弧度)。
    """
    x = np.arange(gray_levels + 1, dtype=np.float64) / float(gray_levels)
    phase = np.zeros_like(x)
    rising = x > 0.15
    t = (x[rising] - 0.15) / 0.85
    phase[rising] = 2.0 * math.pi * (3.0 * t * t - 2.0 * t * t * t)
    return tuple(float(v) for v in phase)


def dead_zone_lut(gray_levels: int = GRAY_MAX) -> tuple[float, ...]:
    """带死区的查找表: 灰度低于满量程 20 % 时相位为 0, 之后线性升到 2π。

    单调不减, 末值恰为 2π; 长度必须是 ``gray_levels + 1``。

    Args:
        gray_levels: 器件灰度级数。

    Returns:
        逐灰度的相位表 (弧度)。
    """
    x = np.arange(gray_levels + 1, dtype=np.float64) / float(gray_levels)
    phase = np.zeros_like(x)
    rising = x > 0.20
    phase[rising] = 2.0 * math.pi * (x[rising] - 0.20) / 0.80
    return tuple(float(v) for v in phase)


# ---------------------------------------------------------------------------
# 工况清单
# ---------------------------------------------------------------------------
def build_cases() -> list[Case]:
    """构造四个研究的全部工况。器件真值只写在各自的 ``SimConfig`` 里。"""
    cases: list[Case] = []

    # -- 研究 A: 填充因子 ---------------------------------------------------
    for ff in (1.0, 0.9, 0.8, 0.7, 0.6):
        cases.append(
            Case(
                key=f"A_ff{ff:g}",
                study="A",
                label=f"填充因子 {ff:.1f}",
                short=f"ff={ff:g}",
                config=SimConfig(fill_factor=ff),
                truth_gamma=1.0,
            )
        )
    cases.append(
        Case(
            key="A_ff0.8_bgpi",
            study="A",
            label="填充因子 0.8, 死区相位 π",
            short="ff=0.8, bg=π",
            config=SimConfig(fill_factor=0.8, fill_background_rad=math.pi),
            truth_gamma=1.0,
            note="死区相位与调制区反相 —— 死区的光与零级相干干涉",
        )
    )

    # -- 研究 B: 相位 LUT ---------------------------------------------------
    cases.append(
        Case(
            key="B_gamma1.0",
            study="B",
            label="幂律 γ=1.0 (参照)",
            short="γ=1.0",
            config=SimConfig(gamma=1.0),
            truth_gamma=1.0,
        )
    )
    cases.append(
        Case(
            key="B_gamma1.4",
            study="B",
            label="幂律 γ=1.4",
            short="γ=1.4",
            config=SimConfig(gamma=1.4),
            truth_gamma=1.4,
        )
    )
    cases.append(
        Case(
            key="B_lut_scurve",
            study="B",
            label="S 形 LUT (前 15 % 平坦, 后 smoothstep 升到 2π)",
            short="S 形 LUT",
            config=SimConfig(lut_table=smoothstep_lut()),
            lut=smoothstep_lut(),
        )
    )
    cases.append(
        Case(
            key="B_lut_deadzone",
            study="B",
            label="死区 LUT (前 20 % 相位为 0, 后线性升到 2π)",
            short="死区 LUT",
            config=SimConfig(lut_table=dead_zone_lut()),
            lut=dead_zone_lut(),
        )
    )

    # -- 研究 C: 面板静态相差 -----------------------------------------------
    cases.append(
        Case(
            key="C_none",
            study="C",
            label="无相差 (参照)",
            short="无相差",
            config=SimConfig(),
            truth_gamma=1.0,
        )
    )
    for noll, name, nm in (
        (2, "倾斜 (1,1)", "(1,1)"),
        (4, "离焦 (2,0)", "(2,0)"),
        (6, "像散 (2,2)", "(2,2)"),
        (8, "彗差 (3,1)", "(3,1)"),
    ):
        cases.append(
            Case(
                key=f"C_noll{noll}",
                study="C",
                label=f"Zernike Noll {noll} {name}, 0.5 rad",
                short=f"Noll {noll}",
                config=SimConfig(panel_zernike={noll: 0.5}),
                truth_gamma=1.0,
            )
        )
    for sigma in (0.1, 0.3):
        cases.append(
            Case(
                key=f"C_rough{sigma:g}",
                study="C",
                label=f"表面粗糙度 σ={sigma:g} rad",
                short=f"σ={sigma:g}",
                config=SimConfig(roughness_sigma_rad=sigma),
                truth_gamma=1.0,
            )
        )

    # -- 研究 D: 光学离焦 ---------------------------------------------------
    # 前四个是规格里给的档位; 后两个 (1e-4 / 2e-4) 是为了让工具**新增的**
    # 「零级光斑过宽 → 阶次不可分辨, 读数不可信」告警真的能响: 告警门槛是
    # FWHM > 0.3×阶次间距 = 70.7 px, 而 6e-5 的离焦只把光斑撑到 ~59 px。
    for coeff in (0.0, 1e-5, 3e-5, 6e-5, 1e-4, 2e-4):
        cases.append(
            Case(
                key=f"D_defocus{coeff:g}",
                study="D",
                label=f"光学离焦 {coeff:g} rad/panel px²",
                short=f"{coeff:g}",
                config=SimConfig(defocus_coeff=coeff),
                truth_gamma=1.0,
            )
        )
    return cases


# ---------------------------------------------------------------------------
# 跑一个工况
# ---------------------------------------------------------------------------
def _verdict_from_depth(depth_rad: float) -> str:
    """按工具自身的阈值给出 欠调制/标称/过调制 判定。"""
    if depth_rad < UNDER_MODULATION_RAD:
        return "under"
    if depth_rad > OVER_MODULATION_RAD:
        return "over"
    return "nominal"


def _fit_with_gate(
    points: list[SweepPoint], gate: FwhmGate
) -> tuple[PhaseResponseFit | None, str | None]:
    """带门禁地拟合; 点被剔得太狠时**让错误冒出来** (而不是静默降级)。"""
    try:
        fit = fit_phase_response(
            [p.command for p in points],
            [p.eta_plus_total for p in points],
            command_max=float(GRAY_MAX),
            period_px=PERIOD_PX,
            gate=gate,
        )
        return fit, None
    except ValueError as exc:
        return None, str(exc)


def run_case(
    case: Case,
    *,
    n_steps: int,
    abs_reference_fwhm_px: float | None = None,
) -> CaseResult:
    """跑完一个工况: 扫描 → 拟合 → 诊断 → 与真值对账。

    Args:
        case: 工况。
        n_steps: 扫描点数。
        abs_reference_fwhm_px: **绝对**在焦参照值 (px)。给了就用它算门禁
            (跨次运行可比的模式, 对应工具的 ``--reference-fwhm-px``);
            None = 只算相对门禁 (参照值取本次扫描命令最小的那一点)。

    Returns:
        :class:`CaseResult`。
    """
    cfg = case.config
    bench = SimulatedBench(cfg)
    panel_shape = bench.panel_shape
    pupil_center = (panel_shape[1] // 2, panel_shape[0] // 2)
    frame_center = (bench.frame_shape[0] // 2, bench.frame_shape[1] // 2)
    holder: dict[str, np.ndarray] = {}

    def pattern_for(command: float) -> np.ndarray:
        """命令 (灰度) → 面板灰度图。绕开驱动的 弧度→灰度 管线, 测器件本征。"""
        spec = BlazeSpec(
            period_px=PERIOD_PX,
            axis="x",
            depth_rad=2.0 * math.pi * float(command) / float(GRAY_MAX),
            pupil_center=pupil_center,
            pupil_radius=int(cfg.pupil_radius_px),
        )
        return blaze_gray(spec, panel_shape, GRAY_MAX)

    def display(gray: np.ndarray) -> None:
        holder["gray"] = gray

    def capture() -> np.ndarray:
        return bench.capture(holder["gray"])

    commands = np.linspace(0.0, float(GRAY_MAX), int(n_steps))
    points = run_sweep(
        pattern_for,
        display,
        capture,
        commands,
        expected_center=frame_center,
        predicted_separation_px=PREDICTED_SEPARATION_PX,
    )

    fit = fit_phase_response(
        [p.command for p in points],
        [p.eta_plus_total for p in points],
        command_max=float(GRAY_MAX),
        period_px=PERIOD_PX,
    )
    #: **相对门禁**: 参照值 = 本次扫描命令最小那一点的 FWHM (本次运行开始时的
    #: 光斑宽度) —— 抓的是"扫描过程中光斑变糊"。
    gate_rel = evaluate_fwhm_gate(points, gate_ratio=GATE_RATIO)
    fit_rel, rel_error = _fit_with_gate(points, gate_rel)
    #: **绝对门禁**: 参照值由调用方给 (在焦那天量到的 FWHM) —— 抓的是
    #: "这条光路一开始就不在焦"。
    gate_abs = evaluate_fwhm_gate(
        points,
        reference_fwhm_px=abs_reference_fwhm_px,
        gate_ratio=GATE_RATIO,
    )
    fit_abs, abs_error = _fit_with_gate(points, gate_abs)
    diagnosis = diagnose(
        points,
        fit,
        full_scale=float(cfg.full_scale),
        predicted_separation_px=PREDICTED_SEPARATION_PX,
        gate=gate_rel,
    )
    #: 绝对参照门禁下的诊断 (只用它的"判定是否可信"标志)。
    if fit_abs is not None:
        diagnosis_abs = diagnose(
            points,
            fit_abs,
            full_scale=float(cfg.full_scale),
            predicted_separation_px=PREDICTED_SEPARATION_PX,
            gate=gate_abs,
        )
        abs_trustworthy = bool(
            diagnosis_abs.get("modulation_verdict_trustworthy", True)
        )
    else:
        abs_trustworthy = False

    # 稳健性对照: 只在"一级衍射可探测"的命令段上重拟合 (分析侧仍然拿不到真值)。
    detectable = [
        p
        for p in points
        if p.eta_plus_total is not None
        and p.command / float(GRAY_MAX) >= DETECT_U_MIN
    ]
    if len(detectable) >= 3:
        fit_detectable = fit_phase_response(
            [p.command for p in detectable],
            [p.eta_plus_total for p in detectable],
            command_max=float(GRAY_MAX),
            period_px=PERIOD_PX,
        )
    else:
        fit_detectable = fit

    etas = [
        (float(p.command), float(p.eta_plus_total))
        for p in points
        if p.eta_plus_total is not None
    ]
    peak_cmd = max(etas, key=lambda item: item[1])[0] if etas else float("nan")
    #: 峰值效率取工具自己 `diagnose` 给出的值 (它对全部扫描点取 max)。
    peak_eta = float(diagnosis.get("peak_eta_plus_total", float("nan")))

    residuals = [p.residual_px for p in points if p.residual_px is not None]
    max_residual = float(max(residuals)) if residuals else float("nan")
    #: 只看"信噪比够"的点 (一级效率 ≥ 5 %): 探测下限附近的边际点会把几何残差
    #: 放大到几十 px, 但那是检出问题, 不是几何标定漂移。
    sig_residuals = [
        p.residual_px
        for p in points
        if p.residual_px is not None and (p.eta_plus_total or 0.0) >= 0.05
    ]
    max_residual_sig = float(max(sig_residuals)) if sig_residuals else float("nan")

    # 零级光斑的 FWHM 取"零级最亮"的那一点 (低调制端), 高调制端零级被熄灭,
    # 此时的 FWHM 是噪声底上的读数, 没有意义。
    bright_zero = max(
        (p for p in points if p.eta_zero is not None),
        key=lambda p: float(p.eta_zero or 0.0),
    )
    fwhm_ref = float(bright_zero.fwhm_px or 0.0)

    ells = [p.ellipticity for p in points if p.ellipticity is not None]
    max_ell = float(max(ells)) if ells else float("nan")

    #: 零级光斑宽度的**最坏值** (工具的"过宽"告警就是按它判的)。
    fwhms = [p.fwhm_px for p in points if p.fwhm_px is not None]
    max_fwhm = float(max(fwhms)) if fwhms else float("nan")
    #: 工具主动喊"阶次不可分辨, 读数不可信"的点数。
    n_alarm = sum(
        1 for p in points if any("过宽" in note for note in p.notes)
    )
    #: 其中"零级还亮着"的点 (零级被熄灭的那几点的宽度是噪声底上的读数,
    #: 报警没有诊断价值) —— 这才是与离焦相关的报警数。
    bright = [p for p in points if (p.eta_zero or 0.0) >= 0.5]
    n_bright = len(bright)
    n_alarm_bright = sum(
        1 for p in bright if any("过宽" in note for note in p.notes)
    )

    u = np.asarray(fit.commands, dtype=np.float64) / float(GRAY_MAX)
    truth_phase = case.truth_phase(u)
    model_phase = fit.depth_max_rad * np.power(np.clip(u, 0.0, None), fit.gamma)
    deviation = np.abs(truth_phase - model_phase)

    return CaseResult(
        case=case,
        points=points,
        fit=fit,
        fit_detectable=fit_detectable,
        diagnosis=diagnosis,
        peak_eta=peak_eta,
        peak_eta_command=peak_cmd,
        max_residual_px=max_residual,
        max_residual_significant_px=max_residual_sig,
        fwhm_ref_px=fwhm_ref,
        fwhm_ref_command=float(bright_zero.command),
        max_ellipticity=max_ell,
        max_fwhm_px=max_fwhm,
        n_wide_alarm=n_alarm,
        n_wide_alarm_bright=n_alarm_bright,
        n_bright=n_bright,
        n_points=len(points),
        gate_rel=gate_rel,
        fit_rel=fit_rel,
        rel_error=rel_error,
        gate_abs=gate_abs,
        fit_abs=fit_abs,
        abs_error=abs_error,
        abs_verdict_trustworthy=abs_trustworthy,
        phase_dev_max_rad=float(deviation.max()) if deviation.size else float("nan"),
        phase_dev_rms_rad=(
            float(np.sqrt(np.mean(deviation**2))) if deviation.size else float("nan")
        ),
        n_detectable=len(detectable),
    )


# ---------------------------------------------------------------------------
# 表格与图
# ---------------------------------------------------------------------------
def _study_cases(results: list[CaseResult], study: str) -> list[CaseResult]:
    return [r for r in results if r.case.study == study]


def _table_rows(rows: list[CaseResult]) -> str:
    """一个研究的主表 (逐工况一行)。"""
    lines = [
        "| 工况 | 真值 Δφ_max (π) | 拟合 Δφ_max (π) | 相对偏差 | 拟合 γ (真值) | 泄漏 ℓ "
        "| 峰值 η±1 (对应命令) | 判定 (真值) | 最大几何残差 (px) | 零级 FWHM (px) | 椭圆率 "
        "| 工具报警 (零级仍亮的点) | 相对门禁 (3×): 违例/评估 (max×) → 深度 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        gamma_truth = (
            "—" if r.case.truth_gamma is None else f"{r.case.truth_gamma:.2f}"
        )
        bias = r.depth_bias * 100.0
        mark = " **(!)**" if not r.verdict_ok else ""
        alarm = f"{r.n_wide_alarm_bright}/{r.n_bright}"
        if r.n_wide_alarm_bright:
            alarm = f"**{alarm}**"
        lines.append(
            f"| {r.case.label} | {r.case.truth_depth_rad / math.pi:.2f} | "
            f"{r.depth_pi:.4f} | {bias:+.1f} %{mark} | "
            f"{r.fit.gamma:.3f} ({gamma_truth}) | {r.fit.leakage:.4f} | "
            f"{r.peak_eta:.4f} ({r.peak_eta_command:.0f}) | "
            f"{r.verdict}{mark} ({r.truth_verdict}) | "
            f"{r.max_residual_px:.2f} | {r.fwhm_ref_px:.1f} | {r.max_ellipticity:.3f} | "
            f"{alarm} | {r.gate_rel_cell} |"
        )
    return "\n".join(lines)


def _curve_panel(
    ax: plt.Axes,
    rows: list[CaseResult],
    *,
    title: str,
    xlabel: str = "命令 (灰度, 满量程 1023)",
) -> None:
    """η±1 vs 命令 + 拟合曲线叠加。"""
    for r in rows:
        xs = [p.command for p in r.points]
        ys = [float(p.eta_plus_total or 0.0) for p in r.points]
        ax.plot(xs, ys, "o", ms=3.4, alpha=0.85, label=f"{r.case.short} 实测")
        model = r.fit.model_eta_plus_total
        ax.plot(r.fit.commands, model, "-", lw=1.6, alpha=0.9, label=f"{r.case.short} 拟合")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("±1 级相对衍射效率 η±1")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.6, ncol=2)


def fig_fill_factor(results: list[CaseResult], out_dir: Path) -> Path:
    """研究 A: 填充因子扫描。"""
    rows = _study_cases(results, "A")
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2))
    _curve_panel(axes[0], rows, title="① 填充因子: η±1 vs 命令 (实测 + 拟合)")
    xs = np.arange(len(rows))
    truth = [r.case.truth_depth_rad / math.pi for r in rows]
    fitted = [r.depth_pi for r in rows]
    axes[1].plot(xs, truth, "k--", lw=1.6, marker="s", ms=6, label="真值 Δφ_max")
    axes[1].plot(xs, fitted, "o-", color="#c53030", ms=6, label="拟合 Δφ_max")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([r.case.short for r in rows], fontsize=8)
    axes[1].set_ylabel("满命令相位深度 Δφ_max (π)")
    axes[1].set_title("② 拟合 vs 真值: 填充因子直接偏置相位深度", fontsize=10)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle("研究 A: 填充因子对衍射效率测量的影响 (数字孪生, 无硬件)")
    fig.tight_layout()
    out = out_dir / f"fig_de_fill_factor_{FIGURE_STAMP}.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


def fig_lut(results: list[CaseResult], out_dir: Path) -> Path:
    """研究 B: 相位 LUT。"""
    rows = _study_cases(results, "B")
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2))
    _curve_panel(axes[0], rows, title="① 四种器件律: η±1 vs 命令 (实测 + 拟合)")
    u = np.linspace(0.0, 1.0, 201)
    for r in rows:
        truth = r.case.truth_phase(u)
        axes[1].plot(u, truth, "-", lw=2.0, label=f"{r.case.short} 器件真值律")
        model = r.fit.depth_max_rad * np.power(u, r.fit.gamma)
        axes[1].plot(u, model, "--", lw=1.4, alpha=0.85, label=f"{r.case.short} 拟合假设的幂律")
    axes[1].set_xlabel("归一化命令 u")
    axes[1].set_ylabel("器件相位 (rad)")
    axes[1].set_title("② 器件真值律 vs 拟合假设的幂律 Δφ=Δφ_max·u^γ", fontsize=10)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=6.4, ncol=2)
    fig.suptitle("研究 B: 相位查找表对衍射效率测量的影响 (数字孪生, 无硬件)")
    fig.tight_layout()
    out = out_dir / f"fig_de_lut_{FIGURE_STAMP}.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


def fig_aberration(results: list[CaseResult], out_dir: Path) -> Path:
    """研究 C: 面板静态相差。"""
    rows = _study_cases(results, "C")
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2))
    _curve_panel(axes[0], rows, title="① 面板静态相差: η±1 vs 命令 (实测 + 拟合)")
    xs = np.arange(len(rows))
    axes[1].plot(xs, [r.depth_pi for r in rows], "o-", color="#c53030", ms=6, label="拟合 Δφ_max (π)")
    axes[1].axhline(2.0, color="k", ls="--", lw=1.4, label="真值 2π")
    axes[1].plot(
        xs,
        [r.peak_eta for r in rows],
        "s-",
        color="#2b6cb0",
        ms=5,
        label="峰值 η±1",
    )
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([r.case.short for r in rows], fontsize=8)
    axes[1].set_ylabel("Δφ_max (π) / 峰值 η±1")
    axes[1].set_title("② 相差下的相位深度与一级效率峰值", fontsize=10)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle("研究 C: 面板静态相差 (Zernike / 粗糙度) 的影响 (数字孪生, 无硬件)")
    fig.tight_layout()
    out = out_dir / f"fig_de_aberration_{FIGURE_STAMP}.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


def fig_summary(results: list[CaseResult], out_dir: Path) -> Path:
    """四个研究的汇总: 拟合 Δφ_max vs 真值。"""
    fig, ax = plt.subplots(figsize=(14.4, 5.6))
    xs = np.arange(len(results))
    truth = [r.case.truth_depth_rad / math.pi for r in results]
    fitted = [r.depth_pi for r in results]
    colors = ["#2196F3" if r.verdict_ok else "#c53030" for r in results]
    ax.bar(xs - 0.2, truth, width=0.4, color="#9E9E9E", label="真值 Δφ_max (π)")
    ax.bar(xs + 0.2, fitted, width=0.4, color=colors, label="拟合 Δφ_max (π)")
    ax.axhline(1.8, color="#975a16", ls="--", lw=1.2, label="欠调制阈值 1.8π")
    ax.axhline(2.2, color="#975a16", ls=":", lw=1.2, label="过调制阈值 2.2π")
    ax.axhline(2.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{r.case.study}:{r.case.short}" for r in results],
        rotation=60,
        ha="right",
        fontsize=7.5,
    )
    ax.set_ylabel("满命令相位深度 Δφ_max (π)")
    ax.set_title(
        "四个研究的汇总: 拟合相位深度 vs 真值\n"
        "红色柱 = 工具给出的欠调制/过调制判定与真值不符",
        fontsize=10,
    )
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = out_dir / f"fig_de_summary_{FIGURE_STAMP}.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


def fig_defocus(results: list[CaseResult], out_dir: Path) -> Path:
    """研究 D: 光学离焦。"""
    rows = _study_cases(results, "D")
    coeffs = [float(r.case.config.defocus_coeff) for r in rows]
    ref_fwhm = min(r.fwhm_ref_px for r in rows)
    alarm_px = 0.3 * PREDICTED_SEPARATION_PX
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2))
    axes[0].plot(coeffs, [r.fwhm_ref_px for r in rows], "o-", color="#2b6cb0", ms=6)
    axes[0].axhline(
        alarm_px,
        color="#c53030",
        ls="--",
        lw=1.5,
        label=f"工具告警门槛 0.3×间距 = {alarm_px:.0f} px",
    )
    axes[0].axhline(
        3.0 * ref_fwhm,
        color="#975a16",
        ls=":",
        lw=1.5,
        label=f"建议门禁 3×在焦参照 = {3.0 * ref_fwhm:.0f} px",
    )
    axes[0].axhline(
        ref_fwhm,
        color="#4CAF50",
        ls="-",
        lw=1.0,
        alpha=0.8,
        label=f"在焦参照值 = {ref_fwhm:.0f} px",
    )
    axes[0].set_xlabel("光学离焦系数 (rad / panel px²)")
    axes[0].set_ylabel("零级光斑 FWHM (px)")
    axes[0].set_title("① 离焦把零级光斑弥散 (FWHM 涨到 19 倍)", fontsize=10)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7.5, loc="upper left")
    axes[1].plot(coeffs, [r.depth_pi for r in rows], "o-", color="#c53030", ms=6, label="拟合 Δφ_max (π)")
    axes[1].plot(coeffs, [r.peak_eta for r in rows], "s-", color="#4CAF50", ms=5, label="峰值 η±1")
    axes[1].axhline(2.0, color="k", ls="--", lw=1.4, label="真值 2π")
    axes[1].axhline(2.2, color="#975a16", ls=":", lw=1.2, label="过调制阈值 2.2π")
    axes[1].set_xlabel("光学离焦系数 (rad / panel px²)")
    axes[1].set_ylabel("Δφ_max (π) / 峰值 η±1")
    axes[1].set_title("② 离焦对拟合相位深度与效率峰值的偏置", fontsize=10)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle("研究 D: 光学离焦的影响 (数字孪生, 无硬件)")
    fig.tight_layout()
    out = out_dir / f"fig_de_defocus_{FIGURE_STAMP}.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# CSV 产物
# ---------------------------------------------------------------------------
CSV_FIELDS: tuple[str, ...] = (
    "case_key",
    "case_label",
    "study",
    "command",
    "u",
    "e_zero",
    "e_plus",
    "e_minus",
    "eta_zero",
    "eta_plus_total",
    "separation_px",
    "residual_px",
    "peak_adu",
    "fwhm_px",
    "ellipticity",
    "model_eta_plus_total",
    "model_depth_rad",
    "fit_depth_pi",
    "fit_gamma",
    "fit_leakage",
    "fit_depth_pi_detectable",
    "truth_depth_pi",
    "truth_gamma",
)


def write_study_csv(results: list[CaseResult], study: str, out_dir: Path) -> Path:
    """把一个研究的逐点扫描数据写成 CSV (自描述: 每行带该工况的拟合与真值)。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"de_{study.lower()}_sweep.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        handle.write(
            f"# diffraction efficiency non-ideality study {study} | "
            f"period={PERIOD_PX}px gray_max={GRAY_MAX} K={K_CCD_PER_PANEL_PX} "
            f"predicted_separation={PREDICTED_SEPARATION_PX:.3f}px\n"
        )
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for r in _study_cases(results, study):
            models = {
                float(cmd): (float(eta), float(depth))
                for cmd, eta, depth in zip(
                    r.fit.commands,
                    r.fit.model_eta_plus_total,
                    r.fit.model_depth_rad,
                    strict=False,
                )
            }
            for p in r.points:
                model = models.get(float(p.command), (None, None))
                writer.writerow(
                    {
                        "case_key": r.case.key,
                        "case_label": r.case.label,
                        "study": r.case.study,
                        "command": f"{p.command:.6g}",
                        "u": f"{p.command / float(GRAY_MAX):.6g}",
                        "e_zero": f"{p.e_zero:.8g}",
                        "e_plus": f"{p.e_plus:.8g}",
                        "e_minus": f"{p.e_minus:.8g}",
                        "eta_zero": "" if p.eta_zero is None else f"{p.eta_zero:.8g}",
                        "eta_plus_total": (
                            "" if p.eta_plus_total is None else f"{p.eta_plus_total:.8g}"
                        ),
                        "separation_px": (
                            "" if p.separation_px is None else f"{p.separation_px:.8g}"
                        ),
                        "residual_px": (
                            "" if p.residual_px is None else f"{p.residual_px:.8g}"
                        ),
                        "peak_adu": f"{p.peak_adu:.8g}",
                        "fwhm_px": "" if p.fwhm_px is None else f"{p.fwhm_px:.8g}",
                        "ellipticity": (
                            "" if p.ellipticity is None else f"{p.ellipticity:.8g}"
                        ),
                        "model_eta_plus_total": (
                            "" if model[0] is None else f"{model[0]:.8g}"
                        ),
                        "model_depth_rad": (
                            "" if model[1] is None else f"{model[1]:.8g}"
                        ),
                        "fit_depth_pi": f"{r.depth_pi:.8g}",
                        "fit_gamma": f"{r.fit.gamma:.8g}",
                        "fit_leakage": f"{r.fit.leakage:.8g}",
                        "fit_depth_pi_detectable": f"{r.depth_pi_detectable:.8g}",
                        "truth_depth_pi": f"{r.case.truth_depth_rad / math.pi:.8g}",
                        "truth_gamma": (
                            "" if r.case.truth_gamma is None else f"{r.case.truth_gamma:.8g}"
                        ),
                    }
                )
    logger.info("研究 {} 的逐点 CSV 已写出: {}", study, path)
    return path


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def _result_map(results: list[CaseResult]) -> dict[str, CaseResult]:
    return {r.case.key: r for r in results}


def _md_extend(base: SimpleNamespace, extra: SimpleNamespace) -> SimpleNamespace:
    """Merge a section namespace; section names must stay unique."""
    clash = set(vars(extra)) & set(vars(base))
    if clash:
        raise AssertionError(f'section-name collision: {sorted(clash)}')
    return SimpleNamespace(**vars(base), **vars(extra))

def _md_metrics(results: list[CaseResult]) -> SimpleNamespace:
    """#205 batch-2: metrics block of the former 596-line build_markdown; body moved line-exact, free names re-bound from v."""
    by_key = _result_map(results)
    a_rows = _study_cases(results, "A")
    b_rows = _study_cases(results, "B")
    c_rows = _study_cases(results, "C")
    d_rows = _study_cases(results, "D")

    wrong = [r for r in results if not r.verdict_ok]
    depth_bad = [r for r in results if abs(r.depth_bias) > DEPTH_BIAS_WARN]
    gamma_bad = [
        r
        for r in results
        if r.gamma_bias is not None and abs(r.gamma_bias) > GAMMA_BIAS_WARN
    ]
    max_resid = max(r.max_residual_px for r in results)
    max_resid_sig = max(r.max_residual_significant_px for r in results)
    #: "正常工况" = 排除两个极端: 死区相位 π 的假阳性工况, 与离焦 ≥ 1e-4 的
    #: 真·不可分辨工况 (研究 D 里光斑已经和 ±1 级同宽)。
    extreme_keys = {"A_ff0.8_bgpi"} | {
        r.case.key for r in d_rows if float(r.case.config.defocus_coeff) >= 1e-4
    }
    max_resid_normal = max(
        r.max_residual_significant_px for r in results if r.case.key not in extreme_keys
    )
    max_ell = max(r.max_ellipticity for r in results)
    ref = by_key["C_none"]

    ff08 = by_key["A_ff0.8"]
    ff08bg = by_key["A_ff0.8_bgpi"]
    ff06 = by_key["A_ff0.6"]
    g14 = by_key["B_gamma1.4"]
    scurve = by_key["B_lut_scurve"]
    dead = by_key["B_lut_deadzone"]
    coma = by_key["C_noll8"]
    rough = by_key["C_rough0.3"]
    d3 = by_key["D_defocus3e-05"]
    d6 = by_key["D_defocus6e-05"]

    def bias_pct(r: CaseResult) -> str:
        return f"{r.depth_bias * 100.0:+.1f} %"
    return SimpleNamespace(by_key=by_key, a_rows=a_rows, b_rows=b_rows, c_rows=c_rows, d_rows=d_rows, wrong=wrong, depth_bad=depth_bad, gamma_bad=gamma_bad, max_resid=max_resid, max_resid_sig=max_resid_sig, extreme_keys=extreme_keys, max_resid_normal=max_resid_normal, max_ell=max_ell, ref=ref, ff08=ff08, ff08bg=ff08bg, ff06=ff06, g14=g14, scurve=scurve, dead=dead, coma=coma, rough=rough, d3=d3, d6=d6, bias_pct=bias_pct)


def _md_sections_core(v: SimpleNamespace, results: list[CaseResult], quick: bool) -> SimpleNamespace:
    """#205 batch-2: core block of the former 596-line build_markdown; body moved line-exact, free names re-bound from v."""
    bias_pct = v.bias_pct
    c_rows = v.c_rows
    d3 = v.d3
    d6 = v.d6
    d_rows = v.d_rows
    dead = v.dead
    depth_bad = v.depth_bad
    ff06 = v.ff06
    ff08 = v.ff08
    ff08bg = v.ff08bg
    g14 = v.g14
    gamma_bad = v.gamma_bad
    ref = v.ref
    scurve = v.scurve
    wrong = v.wrong
    # 稳健性表 (检出下限的影响)
    robust_rows = "\n".join(
        f"| {r.case.study}: {r.case.short} | {r.depth_pi:.4f} | {r.depth_pi_detectable:.4f} | "
        f"{(r.depth_pi_detectable - r.depth_pi) * math.pi:+.4f}π | {r.n_detectable} |"
        for r in results
    )

    detect_table = "\n".join(
        [
            "| 工况 | 全部扫描点拟合 Δφ_max (π) | 仅可探测段拟合 Δφ_max (π) | 差 | 可探测点数 |",
            "|---|---|---|---|---|",
            robust_rows,
        ]
    )

    summary = f"""**结论摘要 (全部为仿真, 无任何实验系统验证)**

**Q1 (仿真能否正确识别不同填充因子 / LUT 下的衍射效率?): 部分能, 但"相位深度"这一栏不可信。**
在本轮 {len(results)} 个工况里, 工具对**一级衍射效率峰值**的读数本身是干净的
(全部工况的峰值 η±1 落在 {min(r.peak_eta for r in results):.4f}–{max(r.peak_eta for r in results):.4f},
参照工况 {ref.peak_eta:.4f}); 但**满命令相位深度的拟合值**被显著偏置:
参照工况 (填充因子 1.0、幂律 γ=1.0、无相差、无离焦) 拟合出 {ref.depth_pi:.4f}π
(真值 2π, 偏差 {bias_pct(ref)}), 而填充因子 0.8 的工况拟合出 {ff08.depth_pi:.4f}π
(偏差 {bias_pct(ff08)}), 填充因子 0.6 拟合出 {ff06.depth_pi:.4f}π (偏差 {bias_pct(ff06)})。
**填充因子越小, 拟合出的相位深度越大** —— 工具把"死区损失"误读成了"相位调制更深"。
后果是判定直接翻转: 参照工况给出 {ref.verdict} (正确), 而填充因子 0.8 给出
{ff08.verdict} (真值应为 {ff08.truth_verdict}) —— **一个标称器件被报成过调制**。
更糟的是偏置的**方向本身不稳定**: 同样填充因子 0.8, 只把死区相位翻到 π,
偏差就变成 {bias_pct(ff08bg)} (判定变成 {ff08bg.verdict}) ⇒ 偏置既不可预测也不可校正。

**Q2 (相关参数 + 相差的影响有多大?): 相差本身几乎无害, 光学离焦才是主要威胁。**
面板静态相差 (Zernike 0.5 rad / 粗糙度 0.3 rad) 对拟合深度的影响在
{min(abs(r.depth_bias) for r in c_rows if r.case.key != "C_none") * 100.0:.1f} %–{max(abs(r.depth_bias) for r in c_rows if r.case.key != "C_none") * 100.0:.1f} %
量级, 判定全部正确; 真正把结果毁掉的是**光学离焦**: 离焦系数
{d3.case.config.defocus_coeff:g} rad/panel px² 就把零级光斑的 FWHM 从
{ref.fwhm_ref_px:.1f} px 抬到 {d3.fwhm_ref_px:.1f} px, 并把拟合深度推到
{d3.depth_pi:.4f}π (偏差 {bias_pct(d3)}), 判定同样翻转成 {d3.verdict};
离焦系数 {d6.case.config.defocus_coeff:g} 时深度偏差达 {bias_pct(d6)}。
好在**工具现在会主动喊出来**, 但它**来得太晚**: 告警门槛是 FWHM > 0.3×间距 =
{0.3 * PREDICTED_SEPARATION_PX:.0f} px, 而离焦 {d3.case.config.defocus_coeff:g}
(FWHM {d3.fwhm_ref_px:.0f} px) 与 {d6.case.config.defocus_coeff:g}
(FWHM {d6.fwhm_ref_px:.0f} px) 这两档的判定**已经翻掉了**, "零级仍亮"的扫描点上
却一次都没报警 ({d3.n_wide_alarm_bright}/{d3.n_bright}、
{d6.n_wide_alarm_bright}/{d6.n_bright}); 直到离焦 {d_rows[-2].case.config.defocus_coeff:g}
(FWHM {d_rows[-2].fwhm_ref_px:.0f} px) 才 {d_rows[-2].n_wide_alarm_bright}/{d_rows[-2].n_bright} 全报警。
⇒ **告警是症状, 不是门禁**。

**工具新加的光斑宽度门禁 (`--fwhm-gate-ratio` / `--reference-fwhm-px`) 也一样有边界:
默认的相对门禁抓不到"一直不在焦"** —— 离焦工况里所有点的 FWHM 都差不多, 比值 ≈ 1,
门禁全过 (违例 {len(d3.gate_rel.violations)}/{d3.gate_rel.evaluated_points}, max 比
{d3.gate_rel.max_ratio:.2f}×), 深度一点没变 (见 §光斑宽度门禁)。**只有绝对参照
(`--reference-fwhm-px {ref.fwhm_ref_px:.1f}`, 在焦那天量到的 FWHM) 才会让离焦工况
的每个点都越限 ⇒ 拟合**显式失败** (有效点 0), 那个偏了 {bias_pct(d3)} 的深度
**根本报不出来**。⇒ 门禁的真正价值是「把静默的偏置变成显式的失败」。
反过来, **幂律 γ 本身是可以被正确测出来的**: γ=1.4 的工况拟合出 γ={g14.fit.gamma:.4f}
(偏差 {g14.gamma_bias * 100.0:+.1f} %), 深度 {g14.depth_pi:.4f}π
(偏差 {bias_pct(g14)}) —— 因为拟合用的**离散器件模型**本来就带 γ 参数。
但换成显式 LUT 就全崩: S 形 LUT 拟合出深度 {scurve.depth_pi:.4f}π
(偏差 {bias_pct(scurve)})、γ={scurve.fit.gamma:.3f}, 死区 LUT 拟合出
{dead.depth_pi:.4f}π (偏差 {bias_pct(dead)}) —— **单一幂律根本无法描述非幂律器件**,
拟合出的 γ 只是"凑出一条曲线"的等效参数, 没有物理意义。

**汇总**: {len(results)} 个工况中, 工具的 欠调制/标称/过调制 判定与真值不符的有
**{len(wrong)} 个** ({', '.join(f'{r.case.study}:{r.case.short}' for r in wrong) if wrong else '无'});
拟合相位深度相对偏差超过 {DEPTH_BIAS_WARN * 100:.0f} % 的有 **{len(depth_bad)} 个**;
γ 相对偏差超过 {GAMMA_BIAS_WARN * 100:.0f} % 的有 **{len(gamma_bad)} 个**
(LUT 工况无 γ 真值, 不计入)。"""

    scope = f"""## 口径 (模拟了什么 / 没模拟什么)

**跑了什么**: 数字孪生 (`SimulatedBench`) + **与实验系统同一条**分析链路
(`core.run_sweep` → `core.fit_phase_response` → `core.diagnose`)。
器件真值 (相位深度 / γ / LUT) 只写在 `SimConfig` 里, 分析侧拿不到;
拟合结果出来之后才与真值对账 —— 验的是拟合链路本身, 不是"喂了答案"的自证。

| 量 | 值 |
|---|---|
| 瞳面 (pupil) 采样 | {SimConfig().region}×{SimConfig().region} 采样点, 瞳面半径 {SimConfig().pupil_radius_px} 面板像元 |
| 远场 | 零填充 FFT, padding = {SimConfig().padding}, 输出画幅 {SimConfig().crop_ccd_px:.0f}×{SimConfig().crop_ccd_px:.0f} CCD px |
| 几何标定 K | {K_CCD_PER_PANEL_PX} CCD px / panel px ⇒ ±1 级预期间距 {PREDICTED_SEPARATION_PX:.2f} px |
| 光栅周期 | {PERIOD_PX} 面板像元 (轴向 = 面板 x) |
| 扫描 | 灰度 0→{GRAY_MAX} 共 {N_STEPS_FULL if not quick else N_STEPS_QUICK} 点, 器件 2π 灰度 = 满量程 {GRAY_MAX} |
| 器件真值 | 全部工况的满量程相位深度都是 2π (标称器件) |
| 噪声 / 探测器 | 暗电平 {SimConfig().dark_level_adu:.0f} ADU (加性背景 (杂散光) 项) + 读出噪声 {SimConfig().read_noise_adu:.0f} ADU, 满量程 {SimConfig().full_scale:.0f} ADU |

**必须写明的近似与缺口**:

1. **填充因子是"相干场混合"的零阶近似, 不是真实器件的分区模型。** 数字孪生把
   每个瞳面采样点的复场写成 `ff·e^{{iφ}} + (1−ff)·e^{{iφ_bg}}`: 死区那一份是**常量**
   (没有空间调制), 因此只进零级, 而 ±1 级被整体乘以 `ff` (振幅) / `ff²` (能量)。
   真实器件的硬边矩形填充因子还会给每一阶乘一个 `sinc(m·p/P)` 包络, 并把死区的光
   送进像元衍射包络 —— 本模型把两者的零级贡献合并进混合项, **不包含**这一层。
   ⇒ 本报告的填充因子数字只能当作"工具在这种近似下的行为", 不能当作器件的绝对效率。
2. **LUT 研究是"用查找表取代幂律"**, 不是给幂律加噪声。`SimConfig.lut_table`
   一旦给出就完全替代 `depth_max_rad·u^γ`; 拟合侧仍然只假设幂律 ⇒ 这一栏测的
   正是"幂律假设在非幂律器件上失效多少"。
3. **面板静态相差加在瞳面网格上, 不是逐像元的面板缺陷。** 数字孪生把它作为
   瞳面相位 (240×240 瞳面采样网格, 半径 120 采样点) 乘上去, 即它模拟的是
   "光路/面板引入的波前相差", 不是"某个像元坏了"。Zernike 振幅按 Noll 归一化
   (单位振幅 = 该模式 RMS = 1 rad), 所以 0.5 rad = 该模式的 RMS 相差 0.5 rad。
4. **光学离焦是瞳面二次相位** `exp(i·c·r²)` (`c` = `defocus_coeff`, rad/panel px²),
   模拟"相机不在后焦面"; 数字孪生**没有**做焦面重采样, 所以离焦表现为光斑变糊
   而不是"重新对焦后的更小光斑"。
5. **没有模拟**: 波前矫正图、驱动的 弧度→灰度 管线、LC 稳定时间、相机帧缓存、
   偏振分量的相干叠加 (泄漏项只能作为"未调制的光"整体进入, 不能区分偏振与填充因子)、
   温度漂移、面板本身的像元级缺陷。**"填充因子" 与 "偏振不纯" 在本链路里是同一个
   可观测 (峰值 η±1 偏低), 原理上无法区分** —— 见 §结论。
6. **子像元采样带来的系统偏差 (数字孪生特有, 实验系统没有)**: 瞳面按
   {2 * SimConfig().pupil_radius_px // SimConfig().region}×{2 * SimConfig().pupil_radius_px // SimConfig().region}
   面板像元做复相量块平均, 对周期 {PERIOD_PX} 的光栅相当于给每一阶乘一个
   `sinc(4/{PERIOD_PX})` 的像元孔径包络, 且该包络随命令变化。参照工况的残差
   (拟合 rms = {ref.fit.residual_rms:.4f}) 与 {bias_pct(ref)} 的深度偏差主要来自这里;
   实验系统的面板像元孔径包络是 `sinc(1/{PERIOD_PX})` ≈ 0.999, 不存在这一层。
7. **零级光斑宽度的测量口径**: FWHM 与椭圆率在半径 **0.45×阶次间距**
   ({0.45 * PREDICTED_SEPARATION_PX:.0f} px) 的大窗口里测 (不碰邻居阶次),
   **不是**在 0.08×间距 ({0.08 * PREDICTED_SEPARATION_PX:.0f} px) 的能量积分 ROI 里测 ——
   在能量 ROI 里测会被截顶, 离焦时永远报一个"看起来正常"的小值。
   工具的"过宽"告警门槛是 FWHM > 0.3×间距 ({0.3 * PREDICTED_SEPARATION_PX:.0f} px)。
   **在焦参照值 = {ref.fwhm_ref_px:.1f} px** (本轮数字孪生, 预期间距
   {PREDICTED_SEPARATION_PX:.1f} px); 实验系统上必须用自己光路的平场帧重新量这个参照值。
   ⚠️ 表里的"零级 FWHM"取**零级最亮**的那个扫描点 (平场端); 高调制端零级被熄灭,
   那几点的 FWHM 是噪声底上的读数, 没有意义。
8. **光斑宽度门禁跑了两种模式**: ① **相对门禁** (工具默认) —— 参照值取本次扫描
   命令最小那一点的 FWHM, 抓"扫描过程中光斑变糊"; ② **绝对门禁** —— 参照值取
   **在焦那天量到的 {ref.fwhm_ref_px:.1f} px** (本轮数字孪生里由在焦参照工况给出),
   对应工具的 `--reference-fwhm-px`, 抓"这条光路一开始就不在焦"。
   两者都**跳过**零级被熄灭的点 (`min_eta_zero = 0.05`, 那些点的宽度是噪声底读数)
   ⇒ **任何按点数统计的告警/门禁都必须先剔除熄灭点**, 否则高调制端的噪声读数会把
   结论淹没 (见 §光斑宽度门禁)。门禁一旦**触发**, 那些被跳过的点也会跟着退出拟合
   (`fit_phase_response` 在 `gate.violations` 非空时把 `gate.skipped_indices` 一并剔除)
   ⇒ 离焦工况的有效点直接归零、拟合显式失败; 而**干净的扫描里被跳过的点照旧参与拟合**
   (它们信息量最大)。"""

    fig_lines = "\n".join(
        [
            f"![研究 A: 填充因子](figures/fig_de_fill_factor_{FIGURE_STAMP}.png)",
            f"![研究 B: 相位 LUT](figures/fig_de_lut_{FIGURE_STAMP}.png)",
            f"![研究 C: 面板静态相差](figures/fig_de_aberration_{FIGURE_STAMP}.png)",
            f"![四个研究的汇总](figures/fig_de_summary_{FIGURE_STAMP}.png)",
            f"![研究 D: 光学离焦](figures/fig_de_defocus_{FIGURE_STAMP}.png)",
        ]
    )
    return SimpleNamespace(robust_rows=robust_rows, detect_table=detect_table, summary=summary, scope=scope, fig_lines=fig_lines)


def _md_sections_studies(v: SimpleNamespace, results: list[CaseResult]) -> SimpleNamespace:
    """#205 batch-2: studies block of the former 596-line build_markdown; body moved line-exact, free names re-bound from v."""
    a_rows = v.a_rows
    b_rows = v.b_rows
    bias_pct = v.bias_pct
    by_key = v.by_key
    c_rows = v.c_rows
    d3 = v.d3
    d6 = v.d6
    d_rows = v.d_rows
    dead = v.dead
    detect_table = v.detect_table
    ff06 = v.ff06
    ff08 = v.ff08
    ff08bg = v.ff08bg
    g14 = v.g14
    ref = v.ref
    rough = v.rough
    scurve = v.scurve
    study_a = f"""## 研究 A: 填充因子

![填充因子](figures/fig_de_fill_factor_{FIGURE_STAMP}.png)

{_table_rows(a_rows)}

读数:

* 填充因子从 1.0 降到 0.6, 拟合相位深度从 {by_key['A_ff1'].depth_pi:.4f}π 单调升到
  {ff06.depth_pi:.4f}π —— **方向是反的**: 填充因子低意味着"更多光没被调制",
  工具却报出"相位调得更深"。
* 拟合的泄漏项 ℓ 随填充因子单调抬升: 0.9 → {by_key['A_ff0.9'].fit.leakage:.4f},
  0.8 → {ff08.fit.leakage:.4f}, 0.7 → {by_key['A_ff0.7'].fit.leakage:.4f},
  0.6 → {ff06.fit.leakage:.4f}。也就是说 **填充因子的损失大部分被记成了"相位更深"
  而不是"泄漏"** (填充因子 0.8 时 ℓ 只有 {ff08.fit.leakage:.4f}, 而深度偏了 {bias_pct(ff08)})。
  原因在口径 §1: 数字孪生的填充因子把 ±1 级乘以 `ff` (能量 ff²), 而拟合模型的泄漏项是
  "把 (1−ℓ) 份额的光原样留在零级" —— 两个模型结构不同, 拟合只能用"更深的相位 + 更陡的
  幂律"去补偿, 于是深度被抬高、泄漏被压低。
* 峰值 η±1 随填充因子从 {ref.peak_eta:.4f} 降到 {ff06.peak_eta:.4f}
  (填充因子 0.6) —— 有信号, 但**远小于同工况的深度偏置 ({bias_pct(ff06)})**,
  而且单调平缓 ⇒ 靠"峰值效率偏低"去判填充因子, 灵敏度只有靠相位深度的一半,
  且两者会给出互相矛盾的结论 (效率说"只差一点", 深度说"偏了三成")。
* 死区相位翻到 π 的对照工况 (填充因子 0.8) 给出深度 {ff08bg.depth_pi:.4f}π
  (偏差 {bias_pct(ff08bg)})、峰值 η±1 {ff08bg.peak_eta:.4f} —— 与同填充因子、零相位
  的工况 ({ff08.depth_pi:.4f}π / {ff08.peak_eta:.4f}) 相比, **偏置从 {bias_pct(ff08)} 翻到 {bias_pct(ff08bg)}**:
  死区那一份光与零级**相干干涉**, 干涉的符号取决于死区相位, 于是拟合出的深度
  既可以偏大也可以偏小 ⇒ **偏置连方向都不稳定, 没有任何事后校正的可能**。
  这也解释了峰值 η±1 为什么反而更高 ({ff08bg.peak_eta:.4f} vs {ff08.peak_eta:.4f}):
  死区相位翻转让零级多收到一份干涉光。
* 该工况还有一个附带发现: 它是全部 {len(results)} 个工况里**唯一**触发工具几何告警的 ——
  最大几何残差 {ff08bg.max_residual_px:.2f} px (预期间距的 {ff08bg.max_residual_px / PREDICTED_SEPARATION_PX * 100:.1f} %),
  远超工具 10 % 的门槛; 但那**只来自一个点** (u = 0.10, 该点 η±1 = 0.016,
  正好卡在检出下限上), 其余 20 个点的残差都 < 0.4 px
  (只看 η±1 ≥ 5 % 的点, 最大残差 {ff08bg.max_residual_significant_px:.2f} px)
  ⇒ **几何告警会被"检出下限附近的边际点"触发假阳性**, 不能直接当作"光路动过了"。"""

    study_b = f"""## 研究 B: 相位查找表 (LUT)

![相位 LUT](figures/fig_de_lut_{FIGURE_STAMP}.png)

{_table_rows(b_rows)}

读数:

* **幂律器件可以测准**: γ=1.0 → 深度 {by_key['B_gamma1.0'].depth_pi:.4f}π (偏差 {bias_pct(by_key['B_gamma1.0'])}),
  γ=1.4 → 深度 {g14.depth_pi:.4f}π (偏差 {bias_pct(g14)})、γ={g14.fit.gamma:.4f}
  (偏差 {g14.gamma_bias * 100.0:+.1f} %)。拟合模型带 γ 参数, 所以这一栏是**可用的**。
* **非幂律器件直接崩掉**: S 形 LUT 拟合出深度 {scurve.depth_pi:.4f}π (偏差 {bias_pct(scurve)})、
  γ={scurve.fit.gamma:.3f}; 死区 LUT 拟合出 {dead.depth_pi:.4f}π (偏差 {bias_pct(dead)})、
  γ={dead.fit.gamma:.3f}。两者的 γ 都不是"器件的 γ" —— 器件根本没有 γ。
  图 ② 里能看到拟合假设的幂律与器件真值律的形状差得多: S 形 LUT 的最大偏差
  {scurve.phase_dev_max_rad:.3f} rad (rms {scurve.phase_dev_rms_rad:.3f} rad),
  死区 LUT {dead.phase_dev_max_rad:.3f} rad (rms {dead.phase_dev_rms_rad:.3f} rad),
  而幂律工况只有 {max(by_key['B_gamma1.0'].phase_dev_max_rad, g14.phase_dev_max_rad):.3f} rad 量级。
* 判定同样不可信: S 形 LUT 判 {scurve.verdict}、死区 LUT 判 {dead.verdict},
  而两者的真值都是标称 (2π 落在 {UNDER_MODULATION_RAD / math.pi:.1f}π–{OVER_MODULATION_RAD / math.pi:.1f}π 之间)。
* 工程含义: **"用幂律拟合出的 γ" 只有在器件确实是幂律时才有意义**。
  真实器件出厂标定出来的是查找表, 所以本工具在真实器件上给出的 γ 很可能是
  "拟合出来的等效参数"而不是物理量 —— 用来做线性化查表 (remap) 尚可, 用来报告
  "器件的 γ" 不可信。"""

    study_c = f"""## 研究 C: 面板静态相差

![面板静态相差](figures/fig_de_aberration_{FIGURE_STAMP}.png)

{_table_rows(c_rows)}

读数:

* 四个 0.5 rad 的 Zernike (倾斜 (1,1) / 离焦 (2,0) / 像散 (2,2) / 彗差 (3,1))
  与两种粗糙度 (σ = 0.1 / 0.3 rad) 对拟合深度的影响都在
  {min(abs(r.depth_bias) for r in c_rows if r.case.key != 'C_none') * 100.0:.1f} %–{max(abs(r.depth_bias) for r in c_rows if r.case.key != 'C_none') * 100.0:.1f} % 量级,
  判定全部为 {c_rows[0].verdict} (与真值一致) ⇒ **低阶面板相差不是本工具的威胁**。
* 物理原因: 相差是**空间缓变**的, 它主要把光斑的能量从零级挪到邻近区域 (零级光斑
  变宽、椭圆率略升), 而 ±1 级的**总能量**几乎不变 ⇒ 效率比值 η±1 动得很少。
* 粗糙度 σ=0.3 rad (逐像元随机) 的工况拟合深度 {rough.depth_pi:.4f}π
  (偏差 {bias_pct(rough)})、峰值 η±1 {rough.peak_eta:.4f} —— 逐像元随机相位会把
  一级能量**摊到相邻阶次**, 但在总能量归一化 (E0+E+1+E−1) 的口径下, 摊走的能量
  里有一部分仍落在 ±1 的积分圆内, 所以效率读数几乎不动。
* 几何残差最大 {max(r.max_residual_px for r in c_rows):.2f} px
  (预期间距 {PREDICTED_SEPARATION_PX:.1f} px 的 {max(r.max_residual_px for r in c_rows) / PREDICTED_SEPARATION_PX * 100:.2f} %),
  远低于工具 10 % 的告警门槛 ⇒ **相差不会破坏几何标定的核对**。"""

    study_d = f"""## 研究 D: 光学离焦

![光学离焦](figures/fig_de_defocus_{FIGURE_STAMP}.png)

{_table_rows(d_rows)}

读数:

* 零级光斑的 FWHM (现在在半径 0.45×阶次间距 = {0.45 * PREDICTED_SEPARATION_PX:.0f} px 的大窗口里测,
  不再被 0.08×间距 = {0.08 * PREDICTED_SEPARATION_PX:.0f} px 的能量积分 ROI 截顶) 随离焦系数
  从 {d_rows[0].fwhm_ref_px:.1f} px 涨到 {d_rows[-1].fwhm_ref_px:.1f} px
  (约 {d_rows[-1].fwhm_ref_px / max(d_rows[0].fwhm_ref_px, 1e-9):.1f} 倍)。
  **阶次间距只有 {PREDICTED_SEPARATION_PX:.1f} px**, 所以最大的那个工况里零级光斑
  已经和 ±1 级同宽 ⇒ 那几帧根本不是"可分辨的阶次", 效率读数没有意义。
  (在焦参照值: {ref.fwhm_ref_px:.1f} px —— 这就是后面"参照值"的来历。)
* 拟合相位深度随之从 {d_rows[0].depth_pi:.4f}π 涨到 {d_rows[-1].depth_pi:.4f}π
  (偏差 {bias_pct(d_rows[-1])}), 判定在离焦系数 {d3.case.config.defocus_coeff:g} 处
  就已经翻成 {d3.verdict} —— **一个完全标称的器件被报成过调制**。
* 但**峰值 η±1 反而更高** ({d_rows[0].peak_eta:.4f} → {d_rows[-1].peak_eta:.4f}) ⇒
  离焦的坏处**不在效率读数上, 而在相位深度上**。机制: 离焦把零级与 ±1 级**同时**抹开,
  而能量积分圆的半径是按几何间距取的**固定值** ⇒ 每个光斑有多少能量落在圆内随光斑
  宽度而变, 这正是 η±1(u) 曲线的**形状**被改变的原因, 而拟合吃的正是形状
  (不是峰值) ⇒ 它把这条被改过的曲线读成"相位更深"。
  **实践含义: 离焦在效率读数上几乎不留痕迹** —— 只能靠光斑宽度暴露。
* **工具现在会主动报警, 但它来得太晚**: 告警门槛是 FWHM > 0.3×间距 =
  {0.3 * PREDICTED_SEPARATION_PX:.0f} px。本轮离焦 {d3.case.config.defocus_coeff:g}
  (FWHM {d3.fwhm_ref_px:.0f} px) 与 {d6.case.config.defocus_coeff:g}
  (FWHM {d6.fwhm_ref_px:.0f} px) 的工况里, **零级仍亮着的扫描点一个都没报警**
  ({d3.n_wide_alarm_bright}/{d3.n_bright} 与 {d6.n_wide_alarm_bright}/{d6.n_bright}),
  而这两个工况的判定**已经翻掉了**; 直到离焦 {d_rows[-2].case.config.defocus_coeff:g}
  (FWHM {d_rows[-2].fwhm_ref_px:.0f} px) 才开始 {d_rows[-2].n_wide_alarm_bright}/{d_rows[-2].n_bright} 全报警。
  ⇒ **告警是症状, 不是门禁**: 它只在"已经不可信"之后才喊。
  ⇒ 真正的门禁必须用 FWHM 相对**在焦参照值** ({ref.fwhm_ref_px:.1f} px) 的倍数自己定
  (见 §结论 3(a)): 本轮 {d3.case.config.defocus_coeff:g} 那一档是
  {d3.fwhm_ref_px / max(ref.fwhm_ref_px, 1e-9):.1f} 倍参照值, 判定已经错了。
* ⚠️ 报警是**逐点**的, 而高调制端零级被熄灭 ⇒ 那几点的"宽度"是噪声底上的读数,
  所以**参照工况里也有 {ref.n_wide_alarm} 个点报警** (u = 0.95 与 1.0, 与离焦无关;
  填充因子低的工况连这两个点都不报警, 因为零级始终很强)。
  表里给的是**零级仍亮着的点**上的报警数 (分母 = 零级仍亮的点数) —— 那才是与离焦相关的。
* 几何残差: 离焦 ≤ {d6.case.config.defocus_coeff:g} 时残差只有
  {max(r.max_residual_px for r in d_rows if r.case.config.defocus_coeff <= 6e-5):.2f} px
  (预期间距的 {max(r.max_residual_px for r in d_rows if r.case.config.defocus_coeff <= 6e-5) / PREDICTED_SEPARATION_PX * 100:.2f} %),
  **抓不到离焦**; 只有到 {d_rows[-1].case.config.defocus_coeff:g} (光斑已经和 ±1 级同宽)
  残差才炸到 {d_rows[-1].max_residual_px:.2f} px
  (预期间距的 {d_rows[-1].max_residual_px / PREDICTED_SEPARATION_PX * 100:.0f} %),
  那时工具的"几何标定可能已过期"告警才响 —— 但那时读数早就没意义了。
* 椭圆率最大 {max(r.max_ellipticity for r in d_rows):.3f} (1.0 = 圆) —— 离焦是
  对称的, 主要表现为变糊而不是变扁, 所以椭圆率不是可靠的离焦指示器。"""

    detect = f"""## 检出下限对拟合的影响 (稳健性对照)

一级衍射并不是从 u=0 就可见: 零级光斑自带一圈 Airy 旁瓣环 (数字孪生上实测旁瓣峰值
≈ 零级峰值的 8 %), 而一级衍射的峰值在低调制端只有几个百分点 —— 低于旁瓣环时,
工具找不到一级光斑, 把该点的一级效率**记为 0** 并在 `notes` 里写明
"几何预测窗口内没有一级衍射光斑"。**那是探测下限, 不是物理零**。

下表是同一批扫描点、只把命令段限制在 u ≥ {DETECT_U_MIN:.2f} 之后重拟合的结果
(分析侧仍然拿不到真值):

{detect_table}

读数: 参照工况在两种口径下分别给出 {ref.depth_pi:.4f}π 与 {ref.depth_pi_detectable:.4f}π,
差别不大; 但填充因子 0.6 从 {ff06.depth_pi:.4f}π 变成 {ff06.depth_pi_detectable:.4f}π,
离焦 {d6.case.config.defocus_coeff:g} 从 {d6.depth_pi:.4f}π 变成 {d6.depth_pi_detectable:.4f}π
⇒ **低端的"记 0"点确实会把深度进一步推高**, 但把低位剔除之后, 偏置**并没有消失** ——
说明主因是模型失配 (填充因子 / 离焦改变了 η±1(u) 的形状), 而不是检出下限。
**工程结论: 扫描点数不必多, 但命令段必须覆盖到 η±1 峰值附近; 低位"记 0"的点
会系统性地把相位深度推高。**"""
    return SimpleNamespace(study_a=study_a, study_b=study_b, study_c=study_c, study_d=study_d, detect=detect)


def _md_sections_tail(v: SimpleNamespace, results: list[CaseResult]) -> SimpleNamespace:
    """#205 batch-2: tail block of the former 596-line build_markdown; body moved line-exact, free names re-bound from v."""
    bias_pct = v.bias_pct
    d3 = v.d3
    d6 = v.d6
    d_rows = v.d_rows
    ff08 = v.ff08
    ff08bg = v.ff08bg
    g14 = v.g14
    max_resid_normal = v.max_resid_normal
    ref = v.ref
    scurve = v.scurve
    wrong = v.wrong
    conclusions = f"""## 结论与对实验系统操作的影响

1. **工具能可靠给出的**: 一级衍射效率的**峰值** (±1 级相对能量的最大值) 与它出现的
   命令; ±1 级与零级的**几何位置** —— 在正常工况里 (填充因子/相差/离焦 ≤ 6e-5),
   只看 η±1 ≥ 5 % 的点, 残差全程 ≤ {max_resid_normal:.2f} px
   (预期间距的 {max_resid_normal / PREDICTED_SEPARATION_PX * 100:.2f} %)
   ⇒ 几何标定的核对是可用的; 幂律器件的 **γ** (γ=1.4 测得 {g14.fit.gamma:.4f},
   偏差 {g14.gamma_bias * 100.0:+.1f} %)。
   ⚠️ 两个例外必须分开看:
   ① **假阳性** —— 填充因子 0.8 + 死区相位 π 的工况最大残差 {ff08bg.max_residual_px:.2f} px
   (预期间距的 {ff08bg.max_residual_px / PREDICTED_SEPARATION_PX * 100:.1f} %),
   但那**只来自一个点** (u = 0.10, 该点 η±1 = 0.016, 卡在检出下限上), 其余点的残差
   < 0.4 px; 工具会因此报"几何标定可能已过期" ⇒ 实验系统上应当**先剔除一级效率过低的点**
   再核对间距。
   ② **真信号** —— 离焦 {d_rows[-1].case.config.defocus_coeff:g} 的工况残差
   {d_rows[-1].max_residual_px:.2f} px (预期间距的
   {d_rows[-1].max_residual_px / PREDICTED_SEPARATION_PX * 100:.0f} %), 那时零级已经
   和 ±1 级同宽 ⇒ 几何核对与"过宽"告警这时是一致的, 是真·不可分辨。
2. **工具不能可靠给出的**: **满命令相位深度**。在填充因子 ≤ 0.8、离焦
   ≥ {d3.case.config.defocus_coeff:g} rad/panel px²、非幂律器件这三种情形里, 拟合深度
   分别偏 {bias_pct(ff08)}、{bias_pct(d3)}、{bias_pct(scurve)}; 而把死区相位翻到 π
   的对照工况偏 {bias_pct(ff08bg)} —— **偏置的量级是十几个到几十个百分点, 连方向都不稳定**
   ⇒ 也就是说 **"欠调制/过调制" 这个判定在有填充因子损失或有离焦的实验系统上会假阳性,
   而且无法靠一个固定改正量救回来**。本轮 {len(results)} 个工况里判定错误的就有 {len(wrong)} 个。
3. **欠调制/过调制判定能不能信? 在默认配置下不能信, 必须加两个前置条件**:
   (a) **零级光斑必须够窄** —— 这件事**工具现在自己会做** (光斑宽度门禁
   `evaluate_fwhm_gate` / `fit_phase_response(gate=...)` / `diagnose(gate=...)`,
   CLI `--fwhm-gate-ratio` 默认 3.0)。但**必须用绝对参照**:
   用 `--reference-fwhm-px` 给一条**在焦那天量到的**零级 FWHM
   (本轮数字孪生里是 {ref.fwhm_ref_px:.1f} px; 实验系统上要用自己光路的平场帧量出来),
   门限 = {GATE_RATIO:.0f}×参照值 = {GATE_RATIO * ref.fwhm_ref_px:.0f} px。
   **只给相对参照值 (工具默认) 是不够的**: 它抓的是"扫描过程中光斑变糊",
   抓不到"这条光路一开始就不在焦" —— 本轮离焦 {d3.case.config.defocus_coeff:g}
   的工况在相对门禁下违例 {len(d3.gate_rel.violations)}/{d3.gate_rel.evaluated_points}
   (max 比 {d3.gate_rel.max_ratio:.2f}×, 深度不变), 换成绝对参照就
   {len(d3.gate_abs.violations)}/{d3.gate_abs.evaluated_points} 全越限、拟合**直接失败**。
   ⇒ 门禁的价值是"**把静默的偏置变成显式的失败**", 不是"把深度算得更准":
   本轮离焦 {d3.case.config.defocus_coeff:g} 的工况在**相对**门禁下
   违例 {len(d3.gate_rel.violations)}/{d3.gate_rel.evaluated_points}
   (max 比 {d3.gate_rel.max_ratio:.2f}×, 深度不变), 换成**绝对**参照就是
   {len(d3.gate_abs.violations)}/{d3.gate_abs.evaluated_points} 全越限 ——
   **拟合显式失败** (有效点 0, 抛「{d3.abs_error}」), 那个偏了 {bias_pct(d3)} 的深度
   再也报不出来, `modulation_verdict_trustworthy` 变成 False。
   ⇒ 报告相位深度时**仍要同时报出零级 FWHM 与在焦参照值的比值**。
   (b) **必须先用同一个器件把"相位深度 vs 灰度"标定过一遍**, 或者至少用一只已知
   2π 灰度的参考件确认峰值 η±1 的量级 —— 因为"峰值 η±1 偏低"与"相位深度偏大"在
   本链路里是**同一个效应的两种表述**。
4. **离焦补偿是不是必须的? 在本仿真口径下是必须的** —— 而且补偿应当做在**扫描之前**
   (把离焦当成被测光路的一部分先消掉), 不是事后修正数字。理由: 离焦改变的是
   η±1(u) 曲线的**形状**, 拟合用的是形状, 事后只能靠"再调 Δφ_max"去补, 而那正是
   被离焦污染过的量。注意本工具的灰度直写模式 (绕开驱动管线) **无法**叠加菲涅尔
   离焦补偿 (补偿必须由驱动做), 所以实验系统上要么改用命令弧度模式, 要么先物理对焦。
   工具新增的「阶次不可分辨, 本点读数不可信 (先做离焦补偿)」告警把这条纪律自动化了:
   离焦大的实验系统上, 扫描本身就会在 `notes` 里报警, 不需要人再去猜 FWHM。
5. **填充因子的损失能不能与偏振损失区分开? 原理上不能。** 二者在测量链路里都是
   "未调制的光留在零级", 只表现为**峰值 η±1 偏低** (工具自己的诊断也这么写:
   "填充因子偏低, 或入射光含未调制的正交偏振分量")。本轮进一步给出更糟的一条:
   在数字孪生的相干混合近似下, 这两种损失甚至**不会**被记成泄漏 (拟合的 ℓ 全程
   ≤ {max(r.fit.leakage for r in results):.4f}), 而是被误读成"相位更深"。
   ⇒ 想要区分, 必须**独立于本工具**去测: 例如旋转偏振片看一级效率是否随之变化
   (偏振损失会变, 填充因子不会), 或用显微/显微干涉直接测像元反射率。
6. **给实验系统的具体建议**: ① **给工具一条在焦的绝对参照值**:
   扫描之前在同一块面板、同一光路下量一张平场帧, 把零级光斑的 FWHM 记下来,
   每次扫描都用 `--reference-fwhm-px <那个值>` 传进去 (本轮数字孪生里是
   {ref.fwhm_ref_px:.1f} px; 门限 = {GATE_RATIO:.0f}× = {GATE_RATIO * ref.fwhm_ref_px:.0f} px)。
   **只用默认的相对参照值是不够的** —— 它只抓"扫描过程中光斑变糊", 抓不到
   "这条光路一开始就不在焦" (本轮数据: 离焦工况相对门禁 0 违例、深度不变,
   绝对门禁全部越限 ⇒ **拟合显式失败 (有效点 0)**, 偏掉的深度报不出来)。
   **门禁的价值是"把静默的偏置变成显式的失败"**;
   ② 把 ±1 级间距残差 (> 预期间距的 10 %) 当硬门槛, 它能抓几何漂移但**抓不到**离焦;
   ③ 报告相位深度时**同时报告零级 FWHM、峰值 η±1、门禁违例数与 `modulation_verdict_trustworthy`**,
   四者一起才可解释;
   ④ 不要相信"过调制"结论, 除非绝对门禁通过 (违例 0) 且峰值 η±1 明显低于 1−ℓ 时才另行判断;
   ⑤ 若怀疑器件不是幂律 (真实器件常常是), 直接上厂商的相位查找表做线性化, 不要用
   本工具拟合出的 γ 去构建 remap 表 (它只是等效参数)。"""

    falsified = f"""## 被推翻 / 待确认

**被本轮数据推翻的说法**:

1. ~~"衍射效率测量对器件非理想因素不敏感"~~ —— 填充因子 0.8 就把相位深度推出
   {bias_pct(ff08)}, 离焦 {d3.case.config.defocus_coeff:g} 就推出 {bias_pct(d3)};
   两者的影响量级 (十几个百分点) 远大于面板相差 (几个百分点)。
2. ~~"一级衍射效率读数可以反推出器件的相位响应"~~ —— 只有在器件确实是幂律、
   光路在焦、填充因子接近 1 这三个条件同时成立时才成立; 本报告的四个研究里有三个
   破坏了这些条件。
3. ~~"残差小 = 拟合可信"~~ —— 参照工况的拟合 rms 是 {ref.fit.residual_rms:.4f},
   而 S 形 LUT 工况是 {scurve.fit.residual_rms:.4f} ({scurve.fit.residual_rms / ref.fit.residual_rms:.1f} 倍),
   离焦 {d3.case.config.defocus_coeff:g} 是 {d3.fit.residual_rms:.4f}
   ({d3.fit.residual_rms / ref.fit.residual_rms:.1f} 倍) —— **残差确实变大了, 但只大了
   2–5 倍, 而相位深度偏了十几个到几十个百分点**; 而且工具的 `diagnose` 并不会因为残差大
   而报警 (它只报几何残差、椭圆率、饱和、峰值效率)。⇒ **残差 rms 只能当烟雾报警器,
   当不了判决书**; 真正的判决必须拿真值对账。
4. ~~"有了光斑宽度门禁, 离焦造成的偏置就被挡住了"~~ —— **只对一半**。
   **相对门禁 (工具默认) 挡不住**: 离焦工况里所有点的 FWHM 都差不多, 比值 ≈ 1,
   违例 {len(d3.gate_rel.violations)}/{d3.gate_rel.evaluated_points}
   (max 比 {d3.gate_rel.max_ratio:.2f}×), 深度**一点没变** ({d3.depth_pi:.4f}π,
   偏差 {bias_pct(d3)})。**绝对参照 (在焦那天的 {ref.fwhm_ref_px:.1f} px) 才挡得住**:
   违例 {len(d3.gate_abs.violations)}/{d3.gate_abs.evaluated_points}, **拟合显式失败**
   (有效点 0, 抛「{d3.abs_error}」) ⇒ 门禁不是"算得更准", 是"**不该出的数出不来**"。

**因核心算法变更而复核过的说法**:

* ~~"离焦在效率读数上几乎不留痕迹"~~ —— **保留**。`core.measure_orders` 改成在
  0.45×阶次间距的大窗口里测 FWHM/椭圆率之后, **能量积分 ROI 没有变**, 所以 η±1 的
  读数与变更前逐位相同 (峰值 η±1 仍随离焦系数从 {d_rows[0].peak_eta:.4f} 略升到
  {d_rows[-1].peak_eta:.4f}); 变的是**零级光斑宽度不再被截顶**
  ({d_rows[0].fwhm_ref_px:.1f} px → {d_rows[-1].fwhm_ref_px:.1f} px), 以及工具现在会
  **主动喊**「阶次不可分辨, 本点读数不可信 (先做离焦补偿)」。
  ⇒ 这条说法应该改写成: **离焦在效率读数上不留痕迹, 但工具会主动报警;
  报警是症状 (只说"已经不可信"), 门禁由工具的 `--reference-fwhm-px`
  (在焦那天的 {ref.fwhm_ref_px:.1f} px) 来做 ⇒ 偏置变成显式的失败。**
* ⚠️ 顺带记一条**工具侧的口径**: 报警是逐点的, 而高调制端零级被熄灭 ⇒ 那几点的
  "宽度"是噪声底上的读数, 于是**参照工况里也有 {ref.n_wide_alarm} 个点报警**
  (u = 0.95 与 1.0), 与离焦无关。⇒ 用报警点数做诊断时必须**剔除零级被熄灭的点**,
  否则离焦的影响会被噪声淹没。

**待确认 (本轮无法回答)**:

1. **数字孪生的填充因子模型是相干混合的零阶近似** (口径 §1)。真实硬边矩形填充因子的
   像元包络 `sinc(m·p/P)` 会改变 ±1 级的绝对效率, 本轮**没有**模拟。因此
   "填充因子 0.8 ⇒ 深度偏 {bias_pct(ff08)}" 这个数字**只在数字孪生口径下成立**,
   不能直接搬到真实器件。
2. **填充因子损失与偏振损失不可区分** 的结论是基于"两者都表现为未调制光进入零级"
   这一建模假设; 本轮没有仿真偏振分量 (数字孪生里没有偏振自由度)。
3. **离焦系数的绝对标定**: `defocus_coeff` 是数字孪生自己的参数, 与实验系统的
   "相机离焦多少毫米" 没有换算关系; 能搬到实验系统的是**定性结论**
   (离焦 → 光斑变糊 → 深度偏大 → 判定假阳性), 不是系数本身。
4. **只测了周期 {PERIOD_PX} 像元的一维闪耀光栅**。周期变化会改变像元孔径包络
   (`sinc(4/P)`) 与一级/零级的能量分配, 结论的量级可能随周期变。
5. **本轮没有做重复性/噪声地板评估**: 每个工况只跑了一条噪声序列 (seed 0),
   因此"偏差"的数值里混着单次噪声。残差 rms 已经在表里, 但没有多次重复的置信区间。
6. **一级衍射在 u ≲ 0.25 不可探测** (检出下限, 见 §检出下限)。真实器件的零级旁瓣
   结构、相机噪声、离焦杂散光都会移动这个下限, 本轮的 {DETECT_U_MIN:.2f} 阈值
   只对数字孪生成立。"""

    repro = f"""## 复现

```bash
python scripts/slm/generate_diffraction_efficiency_sim_report.py          # 全量
python scripts/slm/generate_diffraction_efficiency_sim_report.py --quick  # 冒烟 (不写 report/)
```

逐点扫描数据 (CSV):

| 研究 | 文件 |
|---|---|
| A 填充因子 | `artifacts/de_a_sweep.csv` |
| B 相位 LUT | `artifacts/de_b_sweep.csv` |
| C 面板静态相差 | `artifacts/de_c_sweep.csv` |
| D 光学离焦 | `artifacts/de_d_sweep.csv` |

图:

| 图 | 文件 |
|---|---|
| 填充因子 | `figures/fig_de_fill_factor_{FIGURE_STAMP}.png` |
| 相位 LUT | `figures/fig_de_lut_{FIGURE_STAMP}.png` |
| 面板静态相差 | `figures/fig_de_aberration_{FIGURE_STAMP}.png` |
| 四研究汇总 | `figures/fig_de_summary_{FIGURE_STAMP}.png` |
| 光学离焦 | `figures/fig_de_defocus_{FIGURE_STAMP}.png` |
"""

    gate_rows = "\n".join(
        f"| {r.case.study}: {r.case.short} | {r.gate_rel.reference_fwhm_px:.1f} px | "
        f"{r.gate_rel_cell} | {r.gate_abs_cell} | "
        f"{('**' + r.abs_error + '**') if r.abs_error else '—'} | "
        f"{'是' if r.abs_verdict_trustworthy else '**否**'} |"
        for r in results
    )

    gate_section = f"""## 光斑宽度门禁: 相对门禁抓不到"一直不在焦", 绝对参照才是实验系统要用的模式

工具新增了两件事 (本报告全程走的就是这条新链路):

* `core.evaluate_fwhm_gate(points, reference_fwhm_px=None, gate_ratio=3.0, min_eta_zero=0.05)`
  —— 参照值默认取**命令最小**那一点 (平相位) 的零级 FWHM; 门限 = 倍数 × 参照值;
  **零级被熄灭 (`eta_zero < 0.05`) 的点被跳过、不判违规** (那里的"宽度"是噪声底读数)。
* `core.fit_phase_response(..., gate=...)` 把越限的点从拟合里剔除;
  `core.diagnose(..., gate=...)` 额外给出 `fwhm_gate` 与 `modulation_verdict_trustworthy`。
  CLI: `--fwhm-gate-ratio` (默认 3.0) 与 `--reference-fwhm-px`。

本轮把**两条门禁**各跑了一遍 (相对门禁 = 工具默认; 绝对门禁 = 参照值取
**在焦那天的 {ref.fwhm_ref_px:.1f} px**, 对应 `--reference-fwhm-px {ref.fwhm_ref_px:.1f}`):

| 工况 | 相对门禁参照值 | 相对门禁 → 深度 (违例/评估, max 比) | 绝对门禁 → 深度 | 绝对门禁的失败信息 | 绝对门禁下判定可信? |
|---|---|---|---|---|---|
{gate_rows}

### 结论 1: 相对门禁抓不到"一直不在焦"

研究 D 里每个工况的**所有**点的 FWHM 都差不多 (离焦是**均匀**的, 不是扫描过程中变的),
所以"当前 FWHM / 本次扫描自己的平场 FWHM"这个比值 ≈ 1, 门禁**全过**:
离焦 {d3.case.config.defocus_coeff:g} 的工况相对门禁参照值 = {d3.gate_rel.reference_fwhm_px:.1f} px,
违例 {len(d3.gate_rel.violations)}/{d3.gate_rel.evaluated_points}
(max 比 {d3.gate_rel.max_ratio:.2f}×), 深度**一点没变**
({d3.depth_pi:.4f}π, 偏差 {bias_pct(d3)}) —— 偏置原封不动地留了下来。

⇒ **相对门禁抓的是"扫描过程中光斑变糊"** (机械漂移 / 热漂移 / 焦面移动),
**抓不到"这条光路一开始就不在焦"**。对后者它完全无能为力。

### 结论 2: 绝对参照才是实验系统要用的模式 —— 偏置的深浅本身不重要, "出不出来"才重要

把在焦那天量到的 FWHM ({ref.fwhm_ref_px:.1f} px) 传给 `--reference-fwhm-px`,
门限就变成 {GATE_RATIO:.0f}×{ref.fwhm_ref_px:.1f} = {GATE_RATIO * ref.fwhm_ref_px:.0f} px,
于是离焦工况的**每一个**被评估的点都越限:

* 离焦 {d3.case.config.defocus_coeff:g} (FWHM {d3.fwhm_ref_px:.0f} px):
  违例 {len(d3.gate_abs.violations)}/{d3.gate_abs.evaluated_points}
  → **拟合显式失败** (有效点 0): 「{d3.abs_error}」
* 离焦 {d6.case.config.defocus_coeff:g} (FWHM {d6.fwhm_ref_px:.0f} px):
  违例 {len(d6.gate_abs.violations)}/{d6.gate_abs.evaluated_points} → 「{d6.abs_error}」
* 离焦 {d_rows[-1].case.config.defocus_coeff:g} (FWHM {d_rows[-1].fwhm_ref_px:.0f} px):
  违例 {len(d_rows[-1].gate_abs.violations)}/{d_rows[-1].gate_abs.evaluated_points}
  → 「{d_rows[-1].abs_error}」
* 而**在焦/其余工况** (研究 A/B/C + 离焦 0 与 1e-05) 的绝对门禁违例数都是 0,
  深度不变 ⇒ 绝对参照不会误伤正常测量。

⇒ **绝对参照下, 离焦工况的拟合显式失败 (有效点 0), 而不是给出一个偏了的深度** ——
那个偏了 {bias_pct(d3)}…{bias_pct(d_rows[-1])} 的深度**根本不会被报出来**;
`diagnose` 也会把 `modulation_verdict_trustworthy` 置为 False。
⇒ **门禁的真正价值是「把静默的偏置变成显式的失败」**, 而不是"把深度算得更准"
(它并不改数, 它只是让不该有的数**出不来**)。

**为什么能失败得这么干净**: 门禁一旦触发, 整轮扫描就不可信了 ⇒ 被剔除的点
**加上**那些"因为零级被熄灭而跳过宽度评估"的点会**一起**退出拟合
(`fit_phase_response` 在 `gate.violations` 非空时把 `gate.skipped_indices` 也剔掉)。
离焦工况于是 {len(d3.gate_abs.violations)} 个越限点 + {len(d3.gate_abs.skipped_indices)} 个被跳过的点
= {d3.n_points} 个点**全部**退出, 有效点 0 ⇒ 拟合抛错。
⇒ 反过来, **干净的扫描里被跳过的点仍然参与拟合** (它们信息量最大),
所以这道"全丢"只在门禁真的触发时才发生。"""
    return SimpleNamespace(conclusions=conclusions, falsified=falsified, repro=repro, gate_rows=gate_rows, gate_section=gate_section)


def build_markdown(results: list[CaseResult], *, quick: bool) -> str:
    """拼装报告正文 (数字全部来自本次运行的计算结果)。"""
    v = _md_metrics(results)
    v = _md_extend(v, _md_sections_core(v, results, quick))
    v = _md_extend(v, _md_sections_studies(v, results))
    v = _md_extend(v, _md_sections_tail(v, results))
    conclusions = v.conclusions
    detect = v.detect
    falsified = v.falsified
    fig_lines = v.fig_lines
    gate_section = v.gate_section
    repro = v.repro
    scope = v.scope
    study_a = v.study_a
    study_b = v.study_b
    study_c = v.study_c
    study_d = v.study_d
    summary = v.summary
    body = f"""# 衍射效率标定工具的非理想因素仿真（2026-10-22）

{summary}

{fig_lines}

{scope}

{study_a}

{study_b}

{study_c}

{study_d}

{gate_section}

{detect}

{conclusions}

{falsified}

{repro}
"""
    return insert_header(body, REPORT_KEY)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_all(*, n_steps: int, studies: tuple[str, ...]) -> list[CaseResult]:
    """跑完指定研究的所有工况。

    **在焦参照工况先跑**: 它的平场 FWHM 就是"这条光路在焦时的光斑宽度",
    拿它当**绝对**门禁的参照值 (对应工具的 ``--reference-fwhm-px``)。
    """
    results: list[CaseResult] = []
    cases = [c for c in build_cases() if c.study in studies]
    #: 在焦参照工况: 填充因子 1.0 / 幂律 γ=1.0 / 无相差 / 无离焦。
    reference_case = next((c for c in cases if c.key == "A_ff1"), None)
    abs_reference: float | None = None
    if reference_case is not None:
        logger.info("先跑在焦参照工况 {} (绝对门禁的参照值来源)", reference_case.key)
        ref_result = run_case(reference_case, n_steps=n_steps)
        results.append(ref_result)
        abs_reference = ref_result.fwhm_ref_px
        logger.info(
            "在焦参照值 (平场零级 FWHM) = {:.1f} px ⇒ 绝对门禁限值 = {:.1f} px "
            "({:.0f}× 参照)",
            abs_reference,
            GATE_RATIO * abs_reference,
            GATE_RATIO,
        )
    for index, case in enumerate(cases, start=1):
        if case is reference_case:
            continue
        logger.info(
            "[{}/{}] 研究 {} 工况 {}: 开始 ({} 个扫描点)",
            index,
            len(cases),
            case.study,
            case.key,
            n_steps,
        )
        result = run_case(case, n_steps=n_steps, abs_reference_fwhm_px=abs_reference)
        results.append(result)
        logger.info(
            "    拟合 Δφ_max={:.4f}π (真值 2.00π, 偏差 {:+.1f} %) γ={:.4f} 泄漏={:.4f} "
            "判定={} (真值 {}) 峰值 η±1={:.4f} 最大残差={:.2f}px",
            result.depth_pi,
            result.depth_bias * 100.0,
            result.fit.gamma,
            result.fit.leakage,
            result.verdict,
            result.truth_verdict,
            result.peak_eta,
            result.max_residual_px,
        )
        logger.info(
            "    相对门禁 {} | 绝对门禁 {}",
            result.gate_rel_cell,
            result.gate_abs_cell,
        )
    return results


def print_summary(results: list[CaseResult]) -> None:
    """打印一行一个工况的汇总 (冒烟模式的主要输出)。"""
    logger.info()
    logger.info("=" * 100)
    logger.info(
        f"{'研究':<4} {'工况':<18} {'拟合Δφ(π)':>10} {'偏差':>9} "
        f"{'拟合γ':>8} {'泄漏':>8} {'峰值η±1':>9} {'拟合rms':>9} {'判定':>8} {'真值判定':>8} "
        f"{'残差px':>8} {'FWHM px':>8} {'相对门禁':>18} {'绝对门禁':>20}"
    )
    logger.info("-" * 150)
    for r in results:
        flag = "" if r.verdict_ok else "  <== 判定与真值不符"
        logger.info(
            f"{r.case.study:<4} {r.case.short:<18} {r.depth_pi:>10.4f} "
            f"{r.depth_bias * 100.0:>8.1f}% {r.fit.gamma:>8.4f} "
            f"{r.fit.leakage:>8.4f} {r.peak_eta:>9.4f} {r.fit.residual_rms:>9.4f} "
            f"{r.verdict:>8} {r.truth_verdict:>8} {r.max_residual_px:>8.2f} "
            f"{r.fwhm_ref_px:>8.1f} {r.gate_rel_cell:>18} {r.gate_abs_cell:>20}{flag}"
        )
    logger.info("-" * 100)
    wrong = [r for r in results if not r.verdict_ok]
    logger.info(
        f"工况数 {len(results)} | 判定与真值不符 {len(wrong)} 个 "
        f"({', '.join(f'{r.case.study}:{r.case.short}' for r in wrong) if wrong else '无'}) | "
        f"|Δφ 偏差| > {DEPTH_BIAS_WARN * 100:.0f}% 的 "
        f"{sum(1 for r in results if abs(r.depth_bias) > DEPTH_BIAS_WARN)} 个 | "
        f"最大几何残差 {max(r.max_residual_px for r in results):.2f} px"
    )
    logger.info("=" * 100)
    logger.info()


#: 旧版 (无日期戳) 的图名 —— 重跑后必须删掉, 否则 report/ 里留下孤儿资产
#: (report/AGENTS.md §4)。新图名一律带 :data:`FIGURE_STAMP`。
LEGACY_FIGURE_NAMES: tuple[str, ...] = (
    "fig_de_fill_factor.png",
    "fig_de_lut.png",
    "fig_de_aberration.png",
    "fig_de_summary.png",
    "fig_de_defocus.png",
)


def _remove_legacy_figures(fig_dir: Path) -> list[Path]:
    """删掉无日期戳的旧图 (如果有)。"""
    removed: list[Path] = []
    for name in LEGACY_FIGURE_NAMES:
        path = fig_dir / name
        if path.is_file():
            path.unlink()
            removed.append(path)
            logger.info("已删除无日期戳的旧图: {}", path)
    return removed


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(
        description="衍射效率标定工具的非理想因素仿真报告 (2026-10-22)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="冒烟模式: 每工况 5 个扫描点, 只跑研究 A + D, 写到临时目录 (不写 report/)",
    )
    args = parser.parse_args(argv)

    if args.quick:
        n_steps = N_STEPS_QUICK
        studies = ("A", "D")
        out_dir = Path(tempfile.mkdtemp(prefix="de_sim_quick_"))
        fig_dir = out_dir / "figures"
        arte_dir = out_dir / "artifacts"
        logger.info("冒烟模式: {} 个扫描点, 研究 {}, 输出到 {}", n_steps, studies, out_dir)
    else:
        n_steps = N_STEPS_FULL
        studies = ("A", "B", "C", "D")
        out_dir = OUT_DIR
        fig_dir = FIG_DIR
        arte_dir = ARTE_DIR
        logger.info("全量模式: {} 个扫描点, 研究 {}, 输出到 {}", n_steps, studies, out_dir)

    fig_dir.mkdir(parents=True, exist_ok=True)
    arte_dir.mkdir(parents=True, exist_ok=True)
    if not args.quick:
        _remove_legacy_figures(fig_dir)

    results = run_all(n_steps=n_steps, studies=studies)

    for study in studies:
        write_study_csv(results, study, arte_dir)

    if "A" in studies:
        fig_fill_factor(results, fig_dir)
    if "B" in studies:
        fig_lut(results, fig_dir)
    if "C" in studies:
        fig_aberration(results, fig_dir)
    if "D" in studies:
        fig_defocus(results, fig_dir)
    if set(studies) == {"A", "B", "C", "D"}:
        fig_summary(results, fig_dir)

    print_summary(results)

    if args.quick:
        logger.info("冒烟模式完成: 产物在 {} (不写 report/)", out_dir)
        return 0

    markdown = build_markdown(results, quick=False)
    report_path = OUT_DIR / Path(REPORT_KEY).name
    report_path.write_text(markdown, encoding="utf-8")
    logger.info("报告已写出: {}", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
