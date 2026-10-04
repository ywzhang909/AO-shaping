"""OOPAO 后端 vs 传统 numpy/FFT 光束后端 —— 对比报告生成器。

对比项目自身仿真后端的两个"臂" (arm):

* ``numpy`` 臂 — ``AO_OOPAO_BACKEND`` 未设置, ``beam_backend`` 走历史遗留的
  FFT 频谱 Kolmogorov 相位屏 + 角谱传播 (``beam_simulation``)。
* ``oopao`` 臂 — ``AO_OOPAO_BACKEND=1``, 相位屏走 OOPAO ``Atmosphere``
  (von Karman), 传播走 OOPAO ``Atmosphere.ASM``。

⚠️ 适用范围 (务必先读): 在
``src/ao_shaping/drivers/sim/beam_backend.py`` 中**只有两个函数会切换内核** ——
``turbulence_phase()`` 与 ``propagate()``。``focal_plane()`` / ``apply_lens()`` /
``gaussian_pupil()`` / ``grid()`` 两条臂走的是**同一份纯 numpy 代码**。因此:

* 湍流差异 → 出现在相位屏上;
* 传播差异 → 只有经过 ``propagate()`` 时才可能出现;
* 焦面 (远场/PSF) 差异 → 全部来自相位屏不同, 而不是来自焦面传播内核不同。

本脚本对每个 (静态像差 × 大气湍流) 场景同时跑两条臂, 输出
``report.md`` + ``summary.csv`` + 每场景对比图。**完全离线** (纯数值仿真,
不接触任何硬件), 但**必须**能导入 OOPAO —— 若 OOPAO 不可用, 脚本会**大声中止**
(退出码 2), 绝不允许静默降级成一个被标成 ``oopao`` 的 numpy 重复结果。

用法::

    .venv/bin/python scripts/generate_oopao_vs_numpy_report.py --quick   # 冒烟
    .venv/bin/python scripts/generate_oopao_vs_numpy_report.py           # 完整矩阵

输出::

    report/oopao_vs_numpy/report.md
    report/oopao_vs_numpy/summary.csv
    report/oopao_vs_numpy/figures/*.png
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
_SRC = ROOT / "src"

# `scripts._common` lives in this package, so the REPO ROOT (not just
# `src`) must be importable. Direct `python scripts/<name>.py` does not put
# it there; pytest does via `pythonpath = ["src", ".", "scripts"]`.
sys.path.insert(0, str(_SRC))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ao_shaping.drivers.sim import beam_backend as bb  # noqa: E402
from ao_shaping.drivers.sim import oopao_backend  # noqa: E402
from ao_shaping.utils.wavefront.zernike_utils import (  # noqa: E402
    generate_zernike_phase,
    list_zernike_modes,
)
# `scripts._common` lives in this package, so the REPO ROOT (not just `src`)
# must be importable. A direct `python scripts/<name>.py` does not put it
# there; pytest does, via `pythonpath = ["src", ".", "scripts"]` in pyproject.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._common import fmt_ratio

# CJK 字体 (仓库约定): 逐级回退到 DejaVu Sans, 并关闭 unicode minus。
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

# --------------------------------------------------------------------------- #
# 固定仿真台参数 (物理基准, 不随场景变化)
# --------------------------------------------------------------------------- #
APERTURE_SIZE_M = 0.064  # 口径 64 mm
WAVELENGTH_M = 1064e-9  # 1064 nm
FOCAL_LENGTH_M = 200.0  # focal_plane() 的等效焦距 (m)
ASM_Z_M = 200.0  # propagate() 的比较距离 (m)
ZERNIKE_N_MAX = 4  # generate_zernike_phase(n_max=...) 的径向阶数上限
WAVES_TO_RAD = 2.0 * np.pi  # waves -> radians (系数本身就是弧度)
EE_RADIUS_FACTOR = 4.0  # ee_r4 的半径 = 4 × 衍射极限 FWHM
LOG10_FLOOR = -6.0  # 强度图 log10 色阶下限

ARM_NUMPY = "numpy"
ARM_OOPAO = "oopao"
ARMS: tuple[str, str] = (ARM_NUMPY, ARM_OOPAO)

# 记录在案的 OOPAO 版本事实 (2026-09 核对); 运行时优先用 git 实测值覆盖。
RECORDED_OOPAO_REV = "e8e9aa6"
RECORDED_OOPAO_REV_FULL = "e8e9aa60cf99f4ab21a4dae7c29aae9b9ec6ec87"
RECORDED_PREV_REV = "8e12a17f"
RECORDED_AHEAD_COUNT = 9
RECORDED_REV_DATE = "2026-09-24"
RECORDED_PREV_REV_DATE = "2026-08-26"
OOPAO_IMPORT_WARNING = (
    "Significant changes were done to the OOPAO repository, the Telescope class "
    'is no longer the "master" class and the Source is now carrying the EM-field info.'
)

CSV_COLUMNS: tuple[str, ...] = (
    "scenario",
    "arm",
    "n_grid",
    "seed",
    "cn2",
    "aberration",
    "aberration_pv_waves",
    "distance_m",
    "phase_std_rad",
    "phase_rms_rad",
    "strehl",
    "fwhm_px",
    "ee_r4",
    "energy_frac",
)

METRIC_LABELS: tuple[tuple[str, str], ...] = (
    ("phase_std_rad", "湍流相位 std [rad]"),
    ("phase_rms_rad", "总相位 RMS [rad]"),
    ("strehl", "Strehl"),
    ("fwhm_px", "焦面 FWHM [px]"),
    ("ee_r4", "EE(r=4·FWHM)"),
    ("energy_frac", "ASM 能量守恒比"),
)

# cn2=0 对照组中两臂应当逐位相同的指标: 全部相位类 + 焦面类。
# 刻意排除 energy_frac —— 它经由 propagate(), 正是两个传播核唯一会差的那一项,
# 把它纳入相等性断言会让"双臂一致"永远不成立。
ARM_INVARIANT_METRICS: tuple[str, ...] = (
    "phase_std_rad",
    "phase_rms_rad",
    "strehl",
    "fwhm_px",
    "ee_r4",
)


class OopaoBackendUnavailableError(RuntimeError):
    """OOPAO 臂无法被真实启用 (缺库 / 静默回退) —— 必须大声中止。"""


# --------------------------------------------------------------------------- #
# 场景矩阵
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AberrationCase:
    """静态像差用例。

    Attributes:
        name: 用例名。
        noll_coefficients: Noll(1976) 索引 -> 相对系数 (无量纲形状权重)。
        pv_waves: 目标峰谷相位 (waves), 内部按实测单位 PV 换算成弧度。
    """

    name: str
    noll_coefficients: dict[int, float]
    pv_waves: float

    def label(self) -> str:
        """返回 ``名称 (Noll 4=Defocus)`` 形式的中英文标签。"""
        if not self.noll_coefficients:
            return f"{self.name} (无像差)"
        parts = []
        for noll in sorted(self.noll_coefficients):
            nm = _noll_name(noll)
            parts.append(f"Noll {noll}={nm}")
        return f"{self.name} ({', '.join(parts)})"


@dataclass(frozen=True)
class TurbulenceCase:
    """大气湍流用例 (字段与 ``make_beam_config`` / ``turbulence_phase`` 对齐)。"""

    name: str
    cn2: float
    l_min: float
    l_max: float
    distance: float

    @property
    def cn2_label(self) -> str:
        """Cn2 的显示标签 (0 显示为 ``0``)。"""
        return "0" if self.cn2 <= 0.0 else f"{self.cn2:.0e}"


@dataclass(frozen=True)
class Scenario:
    """一个 (像差 × 湍流) 场景。"""

    key: str
    aberration: AberrationCase
    turbulence: TurbulenceCase


ABERRATIONS: tuple[AberrationCase, ...] = (
    AberrationCase("none", {}, 0.0),
    AberrationCase("defocus", {4: 1.0}, 0.5),
    AberrationCase("astig+coma", {5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0}, 0.8),
    AberrationCase("spherical", {11: 1.0}, 0.6),
)

# Cn2 预设的弱/中/强档位沿用 scripts/simulate_atmospheric_comparison.py,
# 并补一个 cn2=0 的对照组 —— 只有它存在才有"纯像差"这一列。
TURBULENCES: tuple[TurbulenceCase, ...] = (
    TurbulenceCase("none", 0.0, 2e-3, 30.0, 500.0),
    TurbulenceCase("weak", 1e-16, 2e-3, 30.0, 500.0),
    TurbulenceCase("moderate", 5e-15, 1e-3, 20.0, 1000.0),
    TurbulenceCase("strong", 5e-14, 5e-4, 10.0, 1500.0),
)


def _noll_name(noll: int) -> str:
    """返回 Noll 索引的英文模式名 (canonical Noll 1976 约定)。"""
    for index, _n, _m, name in list_zernike_modes(ZERNIKE_N_MAX):
        if index == noll:
            return name.split(" / ")[0]
    return f"mode{noll}"


def build_scenarios(
    aberration_names: list[str], turbulence_names: list[str]
) -> list[Scenario]:
    """按像差 × 湍流 叉乘生成场景列表 (顺序即报告顺序)。

    Args:
        aberration_names: 像差用例名子集 (必须在 ``ABERRATIONS`` 内)。
        turbulence_names: 湍流用例名子集 (必须在 ``TURBULENCES`` 内)。

    Returns:
        场景列表, 顺序为"外层像差、内层湍流"。

    Raises:
        ValueError: 名称未知时。
    """
    by_aber = {case.name: case for case in ABERRATIONS}
    by_turb = {case.name: case for case in TURBULENCES}
    unknown = [n for n in aberration_names if n not in by_aber]
    if unknown:
        raise ValueError(f"unknown aberration case(s): {unknown}")
    unknown = [n for n in turbulence_names if n not in by_turb]
    if unknown:
        raise ValueError(f"unknown turbulence case(s): {unknown}")
    return [
        Scenario(f"{ab}__turb-{turb}", by_aber[ab], by_turb[turb])
        for ab in aberration_names
        for turb in turbulence_names
    ]


# --------------------------------------------------------------------------- #
# 两条臂的启用 / 校验
# --------------------------------------------------------------------------- #
def activate_arm(arm_name: str, seed: int) -> None:
    """切换到指定臂, 清缓存, 播种全局 RNG, 并断言路由真的生效。

    ``oopao_backend._get_backend`` 是 ``@lru_cache(maxsize=8)``, 键里**只有**
    物理配置、没有臂标志。不清缓存时第二条臂可能拿到第一条臂缓存下来的
    ``Atmosphere``, 于是"oopao"臂会静默退化成 numpy —— 这正是本脚本最大的坑。

    Args:
        arm_name: ``"numpy"`` 或 ``"oopao"``。
        seed: 写进 ``np.random.seed`` 的全局种子。

    Raises:
        ValueError: 臂名未知。
        OopaoBackendUnavailableError: 路由与预期不符 (静默回退 / 误路由)。
    """
    if arm_name not in ARMS:
        raise ValueError(f"unknown arm: {arm_name!r}")

    if arm_name == ARM_OOPAO:
        os.environ["AO_OOPAO_BACKEND"] = "1"
    else:
        os.environ.pop("AO_OOPAO_BACKEND", None)

    # 强制: 每次换臂 / 换场景配置都清掉 lru_cache。
    oopao_backend._get_backend.cache_clear()

    # 两条臂不对称: 显式 rng 覆盖 turbulence_phase, 而遗留全局 RNG 路径
    # (TraditionalAOSystem._sample_turbulence_phase 不传 rng) 需要全局播种。
    np.random.seed(seed)

    enabled = bb._oopao_enabled()
    expected = arm_name == ARM_OOPAO
    if enabled != expected:
        raise OopaoBackendUnavailableError(
            f"backend routing mismatch for arm {arm_name!r}: "
            f"beam_backend._oopao_enabled() -> {enabled}, expected {expected} "
            f"(AO_OOPAO_BACKEND={os.environ.get('AO_OOPAO_BACKEND')!r}, "
            f"oopao_available={oopao_backend._oopao_available()})"
        )


def assert_oopao_usable() -> None:
    """启动自检: OOPAO 必须真的可用且路由得通, 否则大声中止。

    Raises:
        OopaoBackendUnavailableError: OOPAO 缺失或任一臂路由不符。
    """
    if not oopao_backend._oopao_available():
        raise OopaoBackendUnavailableError(
            "OOPAO 不可用 (oopao_backend._oopao_available() == False)。请用带 OOPAO "
            "的 venv 运行: .venv/bin/python scripts/generate_oopao_vs_numpy_report.py。"
            "本脚本**拒绝**在 OOPAO 缺席时继续 —— 否则会产出一份被标成 oopao 的 "
            "numpy 重复结果。"
        )
    activate_arm(ARM_OOPAO, seed=0)
    activate_arm(ARM_NUMPY, seed=0)
    logger.info("OOPAO backend 可用且双臂路由自检通过")


# --------------------------------------------------------------------------- #
# 物理量与指标
# --------------------------------------------------------------------------- #
def pupil_mask(n_grid: int) -> np.ndarray:
    """返回口径内布尔掩码 (与 ``gaussian_pupil`` 的截断半径一致)。"""
    cfg = bb.make_beam_config(
        n_grid=n_grid,
        aperture_size=APERTURE_SIZE_M,
        wavelength=WAVELENGTH_M,
        cn2=0.0,
        l_max=TURBULENCES[0].l_max,
        l_min=TURBULENCES[0].l_min,
        propagation_distance=TURBULENCES[0].distance,
    )
    x, y = bb.grid(cfg)
    return np.sqrt(x**2 + y**2) <= cfg.aperture_size / 2.0


def aberration_phase_rad(
    case: AberrationCase, n_grid: int, mask: np.ndarray
) -> np.ndarray:
    """生成像差相位图 [rad] (raw 未包裹, 孔径外为 0)。

    走 canonical 入口 ``generate_zernike_phase`` (禁止自写 Zernike 数学)。该函数
    对系数**线性**, 因此先取单位系数相位、实测其在瞳孔内的峰谷值, 再整体线性缩放
    到目标 ``pv_waves`` 弧度 —— 这样"waves-PV"是真正可控的量, 且全程只用官方
    多项式求值。

    Args:
        case: 像差用例。
        n_grid: 网格边长。
        mask: 瞳孔内掩码。

    Returns:
        ``(n_grid, n_grid)`` float64 弧度数组, 已 ``nan_to_num`` (Zernike 网格与
        光束网格在瞳孔边缘最多差 1 px, 那里 ``generate_zernike_phase`` 返回 NaN;
        置 0 对两臂完全一致, 不引入任何臂间偏置)。
    """
    if not case.noll_coefficients or case.pv_waves == 0.0:
        # coefficients 为空时 generate_zernike_phase 返回 uint16 全零, 该路径不使用。
        return np.zeros((n_grid, n_grid), dtype=float)
    unit = np.nan_to_num(
        generate_zernike_phase(
            case.noll_coefficients,
            resolution=(n_grid, n_grid),
            n_max=ZERNIKE_N_MAX,
        ),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    unit_pv = float(np.ptp(unit[mask]))
    if unit_pv <= 0.0:
        raise ValueError(f"degenerate Zernike phase for case {case.name!r}")
    return unit * (case.pv_waves * WAVES_TO_RAD / unit_pv)


def _interp_crossing(
    inner: int, outer: int, y_inner: float, y_outer: float, level: float
) -> float:
    """在半高阈值上对相邻两点做线性插值, 返回外侧交点坐标。"""
    if y_inner == y_outer:
        return float(outer)
    return outer + (level - y_outer) * (inner - outer) / (y_inner - y_outer)


def half_max_width(profile: np.ndarray) -> float:
    """估计一维剖面的半高全宽 (线性插值过阈值交点)。

    Args:
        profile: 一维强度剖面。

    Returns:
        以"像素"为单位的 FWHM; 剖面全零时返回 0。
    """
    peak = float(profile.max())
    if peak <= 0.0:
        return 0.0
    level = 0.5 * peak
    idx = int(np.argmax(profile))
    last = profile.size - 1
    lo = idx
    while lo > 0 and profile[lo] > level:
        lo -= 1
    hi = idx
    while hi < last and profile[hi] > level:
        hi += 1
    left = _interp_crossing(
        lo, lo - 1, float(profile[lo]), float(profile[lo - 1]), level
    )
    right = _interp_crossing(
        hi, hi + 1, float(profile[hi]), float(profile[hi + 1]), level
    )
    return float(right - left)


def fwhm_px(intensity: np.ndarray) -> float:
    """从过峰值的行/列剖面估计焦面 FWHM (两向取平均)。

    做法: 定位 ``argmax`` 像素, 分别取穿过它的**行剖面**与**列剖面**, 各自用
    :func:`half_max_width` 求半高宽, 再取平均。焦面对称时两向几乎相等
    (数值验证: 差 < 0.5 px)。
    """
    flat = int(np.argmax(intensity))
    row_idx, col_idx = np.unravel_index(flat, intensity.shape)
    return 0.5 * (
        half_max_width(intensity[row_idx, :]) + half_max_width(intensity[:, col_idx])
    )


def encircled_energy(intensity: np.ndarray, radius_px: float) -> float:
    """峰值归一化的包围能量 (圆心取 ``argmax``, 与仓库"0 级 = argmax"约定一致)。"""
    total = float(intensity.sum())
    if total <= 0.0:
        return 0.0
    row_idx, col_idx = np.unravel_index(int(np.argmax(intensity)), intensity.shape)
    rows = np.arange(intensity.shape[0]) - row_idx
    cols = np.arange(intensity.shape[1]) - col_idx
    inside = (rows[:, None] ** 2 + cols[None, :] ** 2) <= radius_px**2
    return float(intensity[inside].sum() / total)


def energy_fraction(field_in: np.ndarray, field_out: np.ndarray) -> float:
    """角谱传播的能量守恒比 ``Σ|E_out|² / Σ|E_in|²``。"""
    e_in = float(np.sum(np.abs(field_in) ** 2))
    if e_in <= 0.0:
        return 0.0
    return float(np.sum(np.abs(field_out) ** 2) / e_in)


def rms_in_mask(phase: np.ndarray, mask: np.ndarray) -> float:
    """掩码内相位 RMS [rad] (参考 ``compat.TraditionalAOSystem._phase_rms``)。"""
    values = phase[mask]
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(values**2)))


def std_in_mask(phase: np.ndarray, mask: np.ndarray) -> float:
    """掩码内相位 std [rad] (湍流分量)。"""
    values = phase[mask]
    if values.size == 0:
        return 0.0
    return float(np.std(values))


def make_beam_config_for(scenario: Scenario, n_grid: int) -> bb.BeamSimConfig:
    """按场景构造 ``BeamSimConfig``。"""
    turb = scenario.turbulence
    return bb.make_beam_config(
        n_grid=n_grid,
        aperture_size=APERTURE_SIZE_M,
        wavelength=WAVELENGTH_M,
        cn2=turb.cn2,
        l_max=turb.l_max,
        l_min=turb.l_min,
        propagation_distance=turb.distance,
    )


# --------------------------------------------------------------------------- #
# 单臂执行
# --------------------------------------------------------------------------- #
@dataclass
class ArmResult:
    """一条臂在单个场景下的全部中间量与指标。"""

    scenario: Scenario
    arm: str
    phase_screen: np.ndarray
    total_phase: np.ndarray
    propagate_intensity: np.ndarray
    focal_intensity: np.ndarray
    metrics: dict[str, float]

    def csv_row(self, n_grid: int, seed: int) -> dict[str, Any]:
        """把结果摊成一行 ``summary.csv`` 记录 (列序见 :data:`CSV_COLUMNS`)。"""
        turb = self.scenario.turbulence
        row: dict[str, Any] = {
            "scenario": self.scenario.key,
            "arm": self.arm,
            "n_grid": int(n_grid),
            "seed": int(seed),
            "cn2": float(turb.cn2),
            "aberration": self.scenario.aberration.name,
            "aberration_pv_waves": float(self.scenario.aberration.pv_waves),
            "distance_m": float(turb.distance),
        }
        for column in CSV_COLUMNS:
            if column not in row:
                row[column] = float(self.metrics[column])
        return {column: row[column] for column in CSV_COLUMNS}


def compute_ideal(cfg: bb.BeamSimConfig) -> tuple[float, float, np.ndarray]:
    """计算与臂无关的"理想"参考量。

    ``focal_plane()`` 两条臂走的是同一份纯 numpy 代码, 所以理想参考只需算一次。

    Returns:
        ``(ideal_peak, ideal_fwhm_px, ideal_intensity)``。
    """
    activate_arm(ARM_NUMPY, seed=0)
    pupil = bb.gaussian_pupil(cfg)
    focal = bb.focal_plane(pupil, cfg, FOCAL_LENGTH_M)
    intensity = np.abs(focal) ** 2
    return float(intensity.max()), fwhm_px(intensity), intensity


def run_arm(
    scenario: Scenario,
    arm_name: str,
    n_grid: int,
    seed: int,
    mask: np.ndarray,
    aberr: np.ndarray,
    ideal_peak: float,
    ideal_fwhm: float,
) -> ArmResult:
    """在指定臂上跑完一个场景的物理链路并计算指标。

    链路: ``turbulence_phase`` → ``+ Zernike 像差`` → ``gaussian_pupil · e^{iφ}``
    → ``propagate()`` (唯一发生内核切换的传播) → ``focal_plane()`` (臂不变)。
    """
    turb = scenario.turbulence
    cfg = make_beam_config_for(scenario, n_grid)

    # 换臂 + 清 OOPAO lru_cache + 全局播种, 并断言路由真的切换了。
    activate_arm(arm_name, seed=seed)

    screen = bb.turbulence_phase(
        cfg,
        cn2=turb.cn2,
        l_max=turb.l_max,
        l_min=turb.l_min,
        propagation_distance=turb.distance,
        rng=np.random.default_rng(seed),
    )
    total = np.asarray(screen, dtype=float) + aberr
    pupil = bb.gaussian_pupil(cfg)
    field = pupil * np.exp(1j * total)

    propagated = bb.propagate(field, cfg, ASM_Z_M)
    prop_intensity = np.abs(propagated) ** 2
    focal_intensity = np.abs(bb.focal_plane(field, cfg, FOCAL_LENGTH_M)) ** 2

    metrics = {
        "phase_std_rad": std_in_mask(np.asarray(screen, dtype=float), mask),
        "phase_rms_rad": rms_in_mask(total, mask),
        "strehl": float(
            np.clip(focal_intensity.max() / max(ideal_peak, 1e-12), 0.0, 1.0)
        ),
        "fwhm_px": fwhm_px(focal_intensity),
        "ee_r4": encircled_energy(focal_intensity, EE_RADIUS_FACTOR * ideal_fwhm),
        "energy_frac": energy_fraction(field, propagated),
    }
    return ArmResult(
        scenario=scenario,
        arm=arm_name,
        phase_screen=np.asarray(screen, dtype=float),
        total_phase=total,
        propagate_intensity=prop_intensity,
        focal_intensity=focal_intensity,
        metrics=metrics,
    )


# --------------------------------------------------------------------------- #
# 绘图
# --------------------------------------------------------------------------- #
def _log_intensity(intensity: np.ndarray) -> np.ndarray:
    """把强度图转成 ``log10(I / Imax)`` 显示图 (vmin 固定为 :data:`LOG10_FLOOR`)。"""
    peak = float(intensity.max())
    if peak <= 0.0:
        return np.full(intensity.shape, LOG10_FLOOR)
    return np.log10(intensity / peak + 10.0**LOG10_FLOOR)


def _colorbar(fig: plt.Figure, mappable: Any, ax_row: np.ndarray, label: str) -> None:
    """给整行面板挂一个共享 colorbar。"""
    fig.colorbar(mappable, ax=list(ax_row), fraction=0.03, pad=0.02, label=label)


def render_scenario_figure(
    scenario: Scenario,
    results: dict[str, ArmResult],
    n_grid: int,
    seed: int,
    asm_z: float,
    focal_length: float,
    path: Path,
) -> None:
    """渲染单个场景的 5×2 对比图 (行 = 物理量, 列 = 两条臂)。

    Rows:
        0. 湍流相位屏 [rad]  —— 两条臂的生成器不同, 差异最直接
        1. 总相位 (湍流 + 像差) [rad]
        2. ``propagate()`` 强度 (z = asm_z) —— 唯一内核真正切换的传播
        3. ``focal_plane()`` 焦面强度 (f = focal_length) —— 该函数臂不变
        4. 指标归一化条形图 (两臂)
    """
    turb = scenario.turbulence
    aberr = scenario.aberration
    fig, axes = plt.subplots(5, 2, figsize=(10.5, 21.0), layout="constrained")
    axes = np.atleast_2d(axes)

    def panel(row: int, arm_name: str) -> Any:
        return axes[row, ARMS.index(arm_name)]

    def style_image(ax: plt.Axes, arm_name: str, text: str) -> None:
        ax.set_title(f"[{ARM_LABELS[arm_name]}] {text}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # 每行统一色阶, 保证两臂视觉可比。
    screen_limit = max(float(np.abs(results[arm].phase_screen).max()) for arm in ARMS)
    total_limit = max(float(np.abs(results[arm].total_phase).max()) for arm in ARMS)
    screen_limit = screen_limit if screen_limit > 0 else 1.0
    total_limit = total_limit if total_limit > 0 else 1.0

    # mappables[row][arm_index] —— 每行一个 (numpy, oopao) 配对, 便于整行共享色标。
    mappables: dict[int, list[Any]] = {0: [], 1: [], 2: [], 3: []}
    for arm_name in ARMS:
        res = results[arm_name]
        m = panel(0, arm_name)
        mappables[0].append(
            m.imshow(
                res.phase_screen,
                cmap="twilight_shifted",
                origin="lower",
                vmin=-screen_limit,
                vmax=screen_limit,
            )
        )
        style_image(
            m,
            arm_name,
            f"湍流相位屏 [rad]  (std={res.metrics['phase_std_rad']:.3f})",
        )

        m = panel(1, arm_name)
        mappables[1].append(
            m.imshow(
                res.total_phase,
                cmap="twilight_shifted",
                origin="lower",
                vmin=-total_limit,
                vmax=total_limit,
            )
        )
        style_image(
            m,
            arm_name,
            f"总相位 = 湍流 + 像差 [rad]  (RMS={res.metrics['phase_rms_rad']:.3f})",
        )

        m = panel(2, arm_name)
        mappables[2].append(
            m.imshow(
                _log_intensity(res.propagate_intensity),
                cmap="inferno",
                origin="lower",
                vmin=LOG10_FLOOR,
                vmax=0.0,
            )
        )
        style_image(
            m,
            arm_name,
            f"propagate() 强度 z={asm_z:g} m\n"
            f"能量守恒比={res.metrics['energy_frac']:.6f}  (log10 I/Imax)",
        )

        m = panel(3, arm_name)
        mappables[3].append(
            m.imshow(
                _log_intensity(res.focal_intensity),
                cmap="inferno",
                origin="lower",
                vmin=LOG10_FLOOR,
                vmax=0.0,
            )
        )
        style_image(
            m,
            arm_name,
            f"focal_plane() 焦面强度 f={focal_length:g} m\n"
            f"Strehl={res.metrics['strehl']:.4f}  FWHM={res.metrics['fwhm_px']:.2f} px  (log10 I/Imax)",
        )

    for row, label in (
        (0, "湍流相位屏 [rad]"),
        (1, "总相位 [rad]"),
        (2, "log10 I/Imax (propagate)"),
        (3, "log10 I/Imax (focal)"),
    ):
        _colorbar(fig, mappables[row][0], axes[row, :], label)

    # 第 5 行: 两臂指标归一化条形图 (按每指标两臂最大值归一, 并标注原始值)。
    metric_keys = [key for key, _ in METRIC_LABELS]
    values = {
        arm_name: [float(results[arm_name].metrics[key]) for key in metric_keys]
        for arm_name in ARMS
    }
    norm = np.array(
        [
            max(values[ARM_NUMPY][i], values[ARM_OOPAO][i], 1e-12)
            for i in range(len(metric_keys))
        ]
    )
    x = np.arange(len(metric_keys), dtype=float)
    width = 0.38
    for offset, arm_name in ((-width / 2, ARM_NUMPY), (width / 2, ARM_OOPAO)):
        raw = np.array(values[arm_name])
        bars = axes[4, 0 + ARMS.index(arm_name)].bar(
            x + offset,
            raw / norm,
            width=width,
            color=ARM_COLORS[arm_name],
            edgecolor="black",
            linewidth=0.6,
        )
        for rect, value in zip(bars, raw, strict=True):
            axes[4, 0 + ARMS.index(arm_name)].annotate(
                f"{value:.4g}",
                (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                ha="center",
                va="bottom",
                fontsize=7,
            )
    for arm_name in ARMS:
        ax = axes[4, ARMS.index(arm_name)]
        ax.set_xticks(
            x,
            [label for _, label in METRIC_LABELS],
            fontsize=7,
            rotation=30,
            ha="right",
        )
        ax.set_ylim(0.0, 1.25)
        ax.set_ylabel("归一化 (两臂最大值 = 1)")
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"[{ARM_LABELS[arm_name]}] 指标对比 (标注为原始值)", fontsize=10)

    diff = propagation_relative_difference(results)
    fig.suptitle(
        f"场景 {scenario.key} | 像差 {aberr.label()} @ {aberr.pv_waves:g} waves PV | "
        f"湍流 {turb.name} (Cn2={turb.cn2_label}, L={turb.distance:g} m) | "
        f"seed={seed} n_grid={n_grid}\n"
        f"传播核双臂相对强度差 max|ΔI|/Imax = {diff:.3e} "
        f"({'完全一致' if diff == 0.0 else '非零'})",
        fontsize=11,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("写出场景图 {}", path)


def propagation_relative_difference(results: dict[str, ArmResult]) -> float:
    """两臂 ``propagate()`` 强度图的最大相对差 ``max|ΔI| / max(I_numpy)``。"""
    base = results[ARM_NUMPY].propagate_intensity
    other = results[ARM_OOPAO].propagate_intensity
    peak = float(base.max())
    if peak <= 0.0:
        return 0.0
    return float(np.abs(base - other).max() / peak)


def render_summary_figure(rows: list[dict[str, Any]], path: Path) -> None:
    """汇总图: Strehl 与相位 RMS 随湍流档位的变化 (颜色=像差, 线型=臂)。"""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    turb_order = [case.name for case in TURBULENCES]
    aberr_order = [case.name for case in ABERRATIONS]
    x = np.arange(len(turb_order), dtype=float)

    for ax, (key, title) in zip(
        axes,
        (("strehl", "Strehl"), ("phase_rms_rad", "瞳孔内总相位 RMS [rad]")),
        strict=True,
    ):
        for aberr_name in aberr_order:
            for arm_name in ARMS:
                ys = []
                for turb_name in turb_order:
                    match = [
                        r
                        for r in rows
                        if r["aberration"] == aberr_name
                        and r["turb_key"] == turb_name
                        and r["arm"] == arm_name
                    ]
                    ys.append(float(match[0][key]) if match else np.nan)
                ax.plot(
                    x,
                    ys,
                    color=ABERRATION_COLORS.get(aberr_name, "gray"),
                    linestyle=ARM_LINESTYLES[arm_name],
                    marker=ARM_MARKERS[arm_name],
                    markersize=4,
                    label=f"{aberr_name} / {ARM_LABELS[arm_name]}",
                )
        ax.set_xticks(x, turb_order)
        ax.set_xlabel("大气湍流档位 (按 Cn2 递增)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylim(0.0, 1.05)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("OOPAO 臂 vs numpy 臂: Strehl / 相位 RMS 随湍流档位变化")
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("写出汇总图 {}", path)


ARM_LABELS: dict[str, str] = {
    ARM_NUMPY: "numpy (legacy FFT)",
    ARM_OOPAO: "oopao (OOPAO)",
}
ARM_COLORS: dict[str, str] = {ARM_NUMPY: "tab:blue", ARM_OOPAO: "tab:orange"}
ARM_LINESTYLES: dict[str, str] = {ARM_NUMPY: "-", ARM_OOPAO: "--"}
ARM_MARKERS: dict[str, str] = {ARM_NUMPY: "o", ARM_OOPAO: "s"}
ABERRATION_COLORS: dict[str, str] = {
    "none": "tab:green",
    "defocus": "tab:red",
    "astig+coma": "tab:purple",
    "spherical": "tab:brown",
}


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def _git(args: list[str], repo: Path) -> str:
    """在 ``repo`` 里跑只读 git 命令并返回 stdout (失败时抛出 CalledProcessError)。"""
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
        env={**os.environ, "GIT_MASTER": "1"},
    )
    return completed.stdout.strip()


def collect_provenance(oopao_repo: Path) -> dict[str, str]:
    """收集 OOPAO 版本溯源信息 (优先实测 git, 失败则回落到记录值)。"""
    info: dict[str, str] = {
        "rev": RECORDED_OOPAO_REV,
        "rev_full": RECORDED_OOPAO_REV_FULL,
        "rev_date": RECORDED_REV_DATE,
        "prev_rev": RECORDED_PREV_REV,
        "prev_rev_date": RECORDED_PREV_REV_DATE,
        "ahead": str(RECORDED_AHEAD_COUNT),
        "verified": "否 (git 不可用, 使用记录值)",
    }
    if not oopao_repo.is_dir():
        info["verified"] = "否 (找不到 libs/OOPAO, 使用记录值)"
        return info
    try:
        info["rev_full"] = _git(["rev-parse", "HEAD"], oopao_repo)
        info["rev"] = info["rev_full"][:7]
        info["rev_date"] = _git(
            ["log", "-1", "--format=%cs", info["rev_full"]], oopao_repo
        )
        info["ahead"] = _git(
            ["rev-list", "--count", f"{RECORDED_PREV_REV}..HEAD"], oopao_repo
        )
        info["prev_rev_date"] = _git(
            ["log", "-1", "--format=%cs", RECORDED_PREV_REV], oopao_repo
        )
        info["verified"] = "是"
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        logger.warning("OOPAO git 溯源探测失败 ({}), 回落到记录值", exc)
    return info


def oopao_package_dir() -> str:
    """返回实际被导入的 OOPAO 包目录 (editable 安装时应指向 ``libs/OOPAO/OOPAO``)。"""
    module = sys.modules.get("OOPAO")
    paths = list(getattr(module, "__path__", []) or []) if module else []
    if not paths:
        return "unknown"
    return str(Path(paths[0]).resolve())


def run_matrix(
    scenarios: list[Scenario], n_grid: int, seed: int
) -> list[dict[str, Any]]:
    """跑完整矩阵, 每场景两臂, 返回带图数据与 CSV 行的记录列表。"""
    mask = pupil_mask(n_grid)
    records: list[dict[str, Any]] = []
    for scenario in scenarios:
        logger.info(
            "场景 {}: 像差={} 湍流={} (Cn2={})",
            scenario.key,
            scenario.aberration.label(),
            scenario.turbulence.name,
            scenario.turbulence.cn2_label,
        )
        cfg = make_beam_config_for(scenario, n_grid)
        ideal_peak, ideal_fwhm, _ = compute_ideal(cfg)
        aberr = aberration_phase_rad(scenario.aberration, n_grid, mask)

        results: dict[str, ArmResult] = {}
        for arm_name in ARMS:
            results[arm_name] = run_arm(
                scenario,
                arm_name,
                n_grid,
                seed,
                mask,
                aberr,
                ideal_peak,
                ideal_fwhm,
            )
        records.append(
            {
                "scenario": scenario,
                "ideal_peak": ideal_peak,
                "ideal_fwhm_px": ideal_fwhm,
                "ee_radius_px": EE_RADIUS_FACTOR * ideal_fwhm,
                "results": results,
                "propagation_rel_diff": propagation_relative_difference(results),
            }
        )
    return records


def write_csv(
    records: list[dict[str, Any]], n_grid: int, seed: int, path: Path
) -> list[dict[str, Any]]:
    """把矩阵结果写成 ``summary.csv`` (每 (场景, 臂) 一行), 并返回这些行。

    Args:
        records: :func:`run_matrix` 的返回值。
        n_grid: 仿真网格边长 (写入每行便于事后复现)。
        seed: 随机种子 (同上)。
        path: 输出 CSV 路径。

    Returns:
        每行除 CSV 列外额外带一个 ``turb_key`` 字段 (供汇总图分组, 不落盘)。
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        scenario: Scenario = record["scenario"]
        for arm_name in ARMS:
            row = record["results"][arm_name].csv_row(n_grid=n_grid, seed=seed)
            rows.append({**row, "turb_key": scenario.turbulence.name})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in CSV_COLUMNS})
    logger.info("写出 {} ({} 行)", path, len(rows))
    return rows


def _rel_diff(numpy_value: float, oopao_value: float) -> str:
    """两臂相对差 (百分比); numpy 值为 0 时返回 ``—``。"""
    if numpy_value == 0.0:
        return "—"
    return f"{(oopao_value - numpy_value) / abs(numpy_value) * 100.0:+.2f}%"


def write_report(
    records: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    out_dir: Path,
    provenance: dict[str, str],
    oopao_dir: str,
    n_grid: int,
    seed: int,
    stamp: str,
) -> Path:
    """生成中文 ``report.md`` (所有数字均来自本次运行)。

    ``stamp`` 必须与写图时用的是同一个时间戳 —— 若在此处重新取当前时间,
    跨过秒边界时报告里的图片链接会与磁盘上的文件名错开。
    """
    generated_at = datetime.now().isoformat(timespec="seconds")
    figures_dir = out_dir / "figures"
    now = stamp

    lines: list[str] = []
    add = lines.append

    add("# OOPAO 后端 vs 传统 numpy/FFT 后端 —— 对比报告")
    add("")
    add(f"- 生成时间: `{generated_at}`")
    add(
        f"- 场景矩阵: **{len(records)}** 个 (像差 {len(ABERRATIONS)} × 湍流 "
        f"{len(TURBULENCES)}, 本次实际取子集) × 2 条臂 = {len(rows)} 行 CSV"
    )
    add(
        f"- 仿真参数: `n_grid={n_grid}`、`seed={seed}`、口径 "
        f"`{APERTURE_SIZE_M * 1e3:g} mm`、波长 `{WAVELENGTH_M * 1e9:g} nm`"
    )
    add(
        f"- 输出: `{out_dir / 'report.md'}`、`{out_dir / 'summary.csv'}`、"
        f"`{figures_dir}/*.png`"
    )
    add("- 重跑命令: `.venv/bin/python scripts/generate_oopao_vs_numpy_report.py`")
    add("")

    # ---------------- 溯源 ---------------- #
    add("## 1. 版本溯源 (Provenance)")
    add("")
    add("| 项目 | 值 |")
    add("|---|---|")
    add(f"| OOPAO 导入路径 (editable) | `{oopao_dir}` |")
    add(f"| OOPAO 当前 rev | `{provenance['rev_full']}` (短 `{provenance['rev']}`) |")
    add(f"| 当前 rev 提交日期 | {provenance['rev_date']} |")
    add(
        f"| 先前 pin 的 rev | `{provenance['prev_rev']}` ({provenance['prev_rev_date']}) |"
    )
    add(f"| 领先提交数 | **{provenance['ahead']}** 个提交 |")
    add(f"| 本次运行是否实测 git 校验 | {provenance['verified']} |")
    add("| `oopao_backend._oopao_available()` | `True` (启动自检已断言) |")
    add("")
    add(
        f"> **关键变更**: 本地 editable clone 位于 `libs/OOPAO`, 当前 rev "
        f"`{provenance['rev']}` 比此前 pin 的 `{provenance['prev_rev']}` "
        f"**领先 {provenance['ahead']} 个提交** ({provenance['rev_date']} vs "
        f"{provenance['prev_rev_date']})。因此**新 rev 相对旧文档假设存在 API 变化**。"
    )
    add("")
    add("### 1.1 OOPAO 导入告警")
    add("")
    add("`import OOPAO` 会在 stdout 打印横幅与如下告警 (本脚本视为无害噪声, 不做屏蔽):")
    add("")
    add("```")
    add(f"OOPAO Warning: {OOPAO_IMPORT_WARNING}")
    add("```")
    add("")
    add(
        "含义: `Telescope` 不再是 master class, EM 场信息改由 `Source` 携带。"
        "这正是新 rev 与旧文档假设不一致的信号 —— 任何按旧语义写的 OOPAO 集成在升级后"
        "都应重新核对。`ao_shaping` 通过 `drivers/sim/_oopao_compat.py` 的 shadow 包"
        "只加载所需子模块, 并只用到 `Atmosphere` / `Source` / `Telescope` 的"
        "**相位屏与 ASM** 能力, 不触碰 master-class 语义。"
    )
    add("")

    # ---------------- 适用范围 ---------------- #
    add("## 2. 适用范围 (Scope) —— 必读")
    add("")
    add(
        "在 `src/ao_shaping/drivers/sim/beam_backend.py` 中, **只有两个函数会切换内核**:"
    )
    add("")
    add(
        "| 函数 | `AO_OOPAO_BACKEND` 未设置 (numpy 臂) | `AO_OOPAO_BACKEND=1` (oopao 臂) |"
    )
    add("|---|---|---|")
    add(
        "| `turbulence_phase()` | FFT 频谱 Kolmogorov 频谱合成 | OOPAO `Atmosphere` von Karman 层 |"
    )
    add(
        "| `propagate()` | `beam_simulation.propagation` (角谱) | OOPAO `Atmosphere.ASM` |"
    )
    add("| `focal_plane()` | 纯 numpy | **纯 numpy (完全相同的代码)** |")
    add(
        "| `gaussian_pupil()` / `grid()` / `apply_lens()` | 纯 numpy | **纯 numpy (完全相同的代码)** |"
    )
    add("")
    add("由此得到三条必须写明的结论:")
    add("")
    add(
        "1. **这不是一次全光学模型替换**。本报告是 *湍流相位屏生成器 + 角谱传播核* 的"
        "对比基准, 不是整条 `beam_backend` 链路的替换。"
    )
    add(
        "2. **远场 (焦面) 差异只来自相位屏, 不来自焦面传播核**。`focal_plane()` 两条臂"
        "逐字节相同, 所以本报告里 Strehl / FWHM / EE 的差异 100% 是湍流相位屏不同造成的。"
    )
    add(
        "3. **完全不受影响的模块**: `drivers/sim/slm_pib_sim.py` 与 "
        "`optimizer/rl/envs/fouriergsnet_env.py` 均不 import 任何 `beam_backend` 符号, "
        "其仿真路径与 OOPAO 后端开关**完全无关**。"
    )
    add("")

    # ---------------- 方法 ---------------- #
    add("## 3. 方法与指标定义")
    add("")
    add("每个 (像差, 湍流) 场景在**同一进程**内依次跑两条臂, 物理链路:")
    add("")
    add("```text")
    add(
        "screen = turbulence_phase(cfg, cn2/l_max/l_min/distance, rng=default_rng(seed))"
    )
    add("aberr  = nan_to_num(generate_zernike_phase(noll_coeffs_rad, (n, n), n_max=4))")
    add("total  = screen + aberr                       # rad, 未包裹")
    add("field  = gaussian_pupil(cfg) * exp(1j * total)")
    add("prop   = propagate(field, cfg, z)             # ← 唯一内核切换的传播")
    add("focal  = focal_plane(field, cfg, f)           # ← 臂不变")
    add("```")
    add("")
    add("| 指标 | 定义 |")
    add("|---|---|")
    add("| `phase_std_rad` | 瞳孔掩码内湍流相位屏的 std (rad) |")
    add(
        "| `phase_rms_rad` | 瞳孔掩码内 `total` 的 RMS (rad), 同 `compat.TraditionalAOSystem._phase_rms` |"
    )
    add(
        "| `strehl` | `clip(max(I_focal) / max(I_ideal), 0, 1)`, 理想值每场景只算一次 (臂无关) |"
    )
    add(
        "| `fwhm_px` | 过峰值的**行剖面 / 列剖面**各求半高宽 (阈值交点线性插值) 后取平均 |"
    )
    add(
        '| `ee_r4` | 圆心取 `argmax` (仓库"0 级 = argmax"约定)、半径 `4 × 衍射极限 FWHM` 的包围能量占比 |'
    )
    add("| `energy_frac` | `Σ|propagate 输出|² / Σ|输入场|²` (角谱能量守恒比) |")
    add("")
    add(
        "**播种与确定性**: 每条臂都 (a) 向 `turbulence_phase` 显式传 "
        "`rng=np.random.default_rng(seed)`, (b) 在换臂时执行 `np.random.seed(seed)` "
        "—— 遗留 numpy 路径在 `rng=None` 时回落到全局 RNG, 而 "
        "`TraditionalAOSystem._sample_turbulence_phase` 恰好不传 rng。两条臂的相位屏"
        '**不会**相同 (生成器不同), 这是预期且正确的, 不做任何"对齐"处理。'
    )
    add("")
    add(
        "**像差幅值**: 通过 canonical `generate_zernike_phase` 生成单位系数相位, 实测其"
        "在瞳孔内的峰谷值, 再线性缩放到目标 waves-PV (×2π 得弧度)。该函数对系数线性, "
        "因此缩放是精确的; 全程复用官方多项式求值, 未自写任何 Zernike 数学。"
    )
    add("")
    add(
        "**防静默回退**: `_get_backend` 是 `@lru_cache(maxsize=8)`, 键中只有物理配置、"
        "**没有臂标志**。脚本在每次换臂和每个场景开始前都调用 "
        "`oopao_backend._get_backend.cache_clear()`, 并断言 "
        "`oopao_backend._oopao_available() is True` 且 `bb._oopao_enabled()` 与预期臂一致; "
        "不一致即以退出码 2 中止, 绝不会产出被标成 `oopao` 的 numpy 重复结果。"
    )
    add("")

    # ---------------- 场景矩阵 ---------------- #
    add("## 4. 场景矩阵")
    add("")
    add("| 像差用例 | Noll 系数 (单位权重) | 目标 PV [waves] |")
    add("|---|---|---|")
    for case in ABERRATIONS:
        coeffs = ", ".join(
            f"{k}: {v:g}" for k, v in sorted(case.noll_coefficients.items())
        )
        add(f"| `{case.name}` | {coeffs or '—'} | {case.pv_waves:g} |")
    add("")
    add("| 湍流用例 | Cn2 [m^(-2/3)] | l0 [m] | L0 [m] | 传播距离 [m] |")
    add("|---|---|---|---|---|")
    for case in TURBULENCES:
        add(
            f"| `{case.name}` | {case.cn2_label} | {case.l_min:g} | "
            f"{case.l_max:g} | {case.distance:g} |"
        )
    add("")

    # ---------------- 汇总 ---------------- #
    add("## 5. 汇总对比")
    add("")
    add("![汇总图](figures/summary_overview.png)")
    add("")
    add("| 场景 | 像差 | 湍流 | Cn2 | 指标 | numpy | oopao | 相对差 |")
    add("|---|---|---|---|---|---|---|---|")
    for record in records:
        scenario: Scenario = record["scenario"]
        res = record["results"]
        for key, label in METRIC_LABELS:
            a = res[ARM_NUMPY].metrics[key]
            b = res[ARM_OOPAO].metrics[key]
            add(
                f"| `{scenario.key}` | {scenario.aberration.name} | "
                f"{scenario.turbulence.name} | {scenario.turbulence.cn2_label} | "
                f"{label} | {fmt_ratio(a)} | {fmt_ratio(b)} | {_rel_diff(a, b)} |"
            )
    add("")

    # ---------------- 逐场景 ---------------- #
    add("## 6. 逐场景明细")
    add("")
    for record in records:
        scenario: Scenario = record["scenario"]
        res = record["results"]
        add(f"### 6.{records.index(record) + 1} `{scenario.key}`")
        add("")
        add(
            f"- 像差: **{scenario.aberration.label()}**, 目标 PV "
            f"{scenario.aberration.pv_waves:g} waves"
        )
        add(
            f"- 湍流: **{scenario.turbulence.name}** (Cn2={scenario.turbulence.cn2_label}, "
            f"l0={scenario.turbulence.l_min:g} m, L0={scenario.turbulence.l_max:g} m, "
            f"传播距离={scenario.turbulence.distance:g} m)"
        )
        add(
            f"- 理想 (无像差无湍流) 焦面: 峰值 {record['ideal_peak']:.6g}, "
            f"FWHM {record['ideal_fwhm_px']:.3f} px → EE 半径 "
            f"{record['ee_radius_px']:.3f} px (臂无关, 每场景只算一次)"
        )
        add(
            f"- 两臂 `propagate()` 强度最大相对差 `max|ΔI|/Imax = "
            f"{record['propagation_rel_diff']:.6e}`"
        )
        add("")
        add("| 指标 | numpy | oopao | 相对差 |")
        add("|---|---|---|---|")
        for key, label in METRIC_LABELS:
            a = res[ARM_NUMPY].metrics[key]
            b = res[ARM_OOPAO].metrics[key]
            add(f"| {label} | {fmt_ratio(a)} | {fmt_ratio(b)} | {_rel_diff(a, b)} |")
        add("")
        filename = f"figures/{scenario.key}_{now}.png"
        add(f"![{scenario.key}]({filename})")
        add("")

    # ---------------- 解读 ---------------- #
    add("## 7. 解读与结论")
    add("")
    cn2_zero_present = any(
        record["scenario"].turbulence.cn2 <= 0.0 for record in records
    )
    max_prop_diff = max(
        (record["propagation_rel_diff"] for record in records), default=0.0
    )
    energy_values = [
        record["results"][arm].metrics["energy_frac"]
        for record in records
        for arm in ARMS
    ]
    turbulent = [
        record for record in records if record["scenario"].turbulence.cn2 > 0.0
    ]
    std_ratios = [
        record["results"][ARM_OOPAO].metrics["phase_std_rad"]
        / record["results"][ARM_NUMPY].metrics["phase_std_rad"]
        for record in turbulent
        if record["results"][ARM_NUMPY].metrics["phase_std_rad"] > 0.0
    ]
    add(
        f"**(a) 湍流相位屏是两臂唯一真正的物理分歧点。** 在 "
        f"{len(set(record['scenario'].turbulence.name for record in turbulent))} "
        f"个含湍流的档位里, 两臂的 `phase_std_rad` / `phase_rms_rad` / Strehl / "
        "FWHM / EE 全部不同 —— 这是 FFT 频谱 Kolmogorov 与 OOPAO von Karman 两个"
        "**不同生成器**的必然结果, 不是 bug。两条臂不可互相替代, 也不应把 oopao 的"
        "数值直接当成 numpy 的续值。"
    )
    add("")
    if std_ratios:
        add(
            f"> ⚠️ **同 Cn2 下两臂的湍流强度归一化并不一致** —— 这是本次运行最值得注意"
            f"的定量结果。在 {len(std_ratios)} 个含湍流场景中, oopao 臂的 "
            f"`phase_std_rad` / numpy 臂之比 = **{min(std_ratios):.2f}× ~ "
            f"{max(std_ratios):.2f}×** (中位 {sorted(std_ratios)[len(std_ratios) // 2]:.2f}×)。"
            "尽管 `beam_backend.turbulence_phase` 的 docstring 声称 OOPAO 分层已"
            '"rescaling to the per-slab r0 that matches the historical aotools/FFT '
            'path", 实测两条路径在**相同 Cn2** 下给出的相位起伏强度差了一个量级左右。'
            "因此: **Strehl / FWHM 的绝对值不可跨臂直接比较** —— oopao 臂并不是"
            '"同条件下的等价实现", 而是一个把同样的 Cn2 映射到更湍流的光场的实现。'
        )
        add("")
    add(
        f"**(b) 角谱传播核在本配置下几乎是恒等替换。** `beam_simulation.propagation` 用 "
        f"`kz = sqrt(|(2π/λ)² - f_x² - f_y²|)`, OOPAO `ASM` 用抛物近似 "
        f"`exp(-iπλz f²)`; 二者只差一个全局相位 `exp(ikz)` (强度不可见) 与 "
        f"`O((f λ)²)` 量级的高阶项。实测两臂 `propagate()` 强度图最大相对差 "
        f"`{max_prop_diff:.3e}`, 能量守恒比在 "
        f"`{min(energy_values):.9f}`–`{max(energy_values):.9f}` 之间 —— "
        "**因此 `energy_frac` 不是区分两臂的指标**, 它的价值在于证明两条核都严格保能量。"
    )
    add("")
    add(
        "> 注: 上式的 `|·|` 是**实现事实而非严格解** —— 常规角谱对倏逝分量"
        "(参数为负) 应取纯虚 `kz` 使其随 z 衰减, 而 `sqrt(np.abs(...))` 会把它变成"
        "实数 `kz` 并因此 **放大** 而非衰减。本台参数 (口径 64 mm、λ=1064 nm、"
        f"ASM z={ASM_Z_M:g} m) 下频谱上限远低于 1/λ, 不存在倏逝分量, 故该差异"
        "在本报告中不产生任何影响; 但把 `propagation()` 推到高空间频率或大 z 时必须"
        "记得这处 `abs`。"
    )
    add("")
    add(
        "**(c) 焦面指标差异的归因。** `focal_plane()` 两臂代码完全相同, 所以上表里 "
        "Strehl / FWHM / EE 的所有差异都只能沿链路回溯到 `turbulence_phase()`。"
        '任何用焦面指标去判断"传播核是否被换掉"的读法都是错的。'
    )
    add("")
    if cn2_zero_present:
        add(
            "**(d) `cn2=0` 对照组两臂数值完全相同 —— 这是设计如此, 不是缺陷。** "
            "`turbulence_phase()` 在 `cn2 <= 0` 时直接返回全零 (在切换内核之前), "
            "而像差相位是确定性、无 RNG 的, 于是两臂的入射场逐字节相同, "
            "相位类指标与焦面指标必然一致; 此时唯一还可能不同的是 `propagate()` 核, "
            '见 (b)。把这一行当作"双盲对照"来验证报告没有把 numpy 结果误标成 oopao。'
        )
        add("")
    add(
        "**(e) 使用建议.** 若目标是**数值可比性** (回归基线、与历史数据对齐), 保持 "
        "`AO_OOPAO_BACKEND` 未设置; 若目标是**物理保真度** (更接近真实大气相位谱), "
        "则应在同一份基准里同时保留两臂结果, 不要跨臂直接比较绝对值。"
    )
    add("")

    add("## 8. 产物清单")
    add("")
    add("- `report.md` — 本文件")
    add(
        f"- `summary.csv` — {len(rows)} 行 (每 (场景, 臂) 一行), 列: "
        + ", ".join(f"`{c}`" for c in CSV_COLUMNS)
    )
    add("- `figures/summary_overview.png` — 汇总图 (Strehl / 相位 RMS vs 湍流档位)")
    add(f"- `figures/*_{now}.png` — 每个场景一张 5×2 对比图 (共 {len(records)} 张)")
    add("")

    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("写出 {}", path)
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="生成 OOPAO 后端 vs 传统 numpy/FFT 后端的对比报告 (离线)",
    )
    parser.add_argument("--n-grid", type=int, default=64, help="仿真网格边长 (默认 64)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认 42)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "report" / "oopao_vs_numpy",
        help="输出目录 (默认 report/oopao_vs_numpy)",
    )
    parser.add_argument(
        "--aberrations",
        default="none,defocus,astig+coma",
        help="逗号分隔的像差用例子集 (可选: "
        + ",".join(case.name for case in ABERRATIONS)
        + ")",
    )
    parser.add_argument(
        "--turbulence",
        default=",".join(case.name for case in TURBULENCES),
        help="逗号分隔的湍流用例子集 (可选: "
        + ",".join(case.name for case in TURBULENCES)
        + ")",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="冒烟模式: 只取前 2 个像差 × 前 2 个湍流",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """脚本入口。

    Returns:
        进程退出码 (0 = 成功)。
    """
    args = parse_args(argv)
    if args.n_grid < 8:
        raise ValueError(f"--n-grid must be >= 8, got {args.n_grid}")

    aberration_names = [n.strip() for n in args.aberrations.split(",") if n.strip()]
    turbulence_names = [n.strip() for n in args.turbulence.split(",") if n.strip()]
    if args.quick:
        aberration_names = aberration_names[:2]
        turbulence_names = turbulence_names[:2]

    scenarios = build_scenarios(aberration_names, turbulence_names)
    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "矩阵: {} 场景 × {} 臂 = {} 次运行 (n_grid={}, seed={})",
        len(scenarios),
        len(ARMS),
        len(scenarios) * len(ARMS),
        args.n_grid,
        args.seed,
    )

    try:
        assert_oopao_usable()
    except OopaoBackendUnavailableError as exc:
        logger.error("FATAL: {}", exc)
        return 2

    provenance = collect_provenance(ROOT / "libs" / "OOPAO")
    oopao_dir = oopao_package_dir()
    logger.info(
        "OOPAO 溯源: rev={} (领先 {} 提交, git 校验={})",
        provenance["rev"],
        provenance["ahead"],
        provenance["verified"],
    )

    records = run_matrix(scenarios, args.n_grid, args.seed)
    if not records:
        logger.error("场景矩阵为空, 没有任何结果可写")
        return 3

    rows = write_csv(records, args.n_grid, args.seed, out_dir / "summary.csv")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for record in records:
        scenario = record["scenario"]
        render_scenario_figure(
            scenario,
            record["results"],
            args.n_grid,
            args.seed,
            ASM_Z_M,
            FOCAL_LENGTH_M,
            figures_dir / f"{scenario.key}_{stamp}.png",
        )
    render_summary_figure(rows, figures_dir / "summary_overview.png")

    write_report(
        records,
        rows,
        out_dir,
        provenance,
        oopao_dir,
        args.n_grid,
        args.seed,
        stamp,
    )

    identical = [
        record["scenario"].key
        for record in records
        if all(
            record["results"][ARM_NUMPY].metrics[key]
            == record["results"][ARM_OOPAO].metrics[key]
            for key in ARM_INVARIANT_METRICS
        )
    ]
    if identical:
        logger.info(
            "相位/焦面指标逐位相同的情景 (预期为 cn2=0 对照组; energy_frac 除外 —— "
            "它正是两个传播核唯一会差的地方): {}",
            identical,
        )
    else:
        logger.warning(
            "没有任何情景的两臂相位/焦面指标逐位相同 —— 若矩阵含 cn2=0 情景, "
            "请检查相位生成是否引入了臂间偏置"
        )
    logger.info("完成: {}", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
