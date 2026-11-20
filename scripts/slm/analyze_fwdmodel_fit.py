"""离线分析: 物理预测模型缺了什么, 才能对上 M3 的实测远场。

要回答的问题
------------
``absolute_fidelity_20261116`` 系列采集给了 12 张「命令相位 → 冻结模型预测」与
「同一相位上屏后的 CCD 实测远场」。冻结模型在这批数据上的失配, 究竟来自哪几个
**物理预测模型缺陷**? 候选 (按任务书给的顺序):

* **离焦** (defocus dioptre) —— 冻结检查点里的 ``defocus_diopter`` 是否还成立;
* **横向尺度** (scale_x / scale_y) —— 面板↔相机映射 (ramp_matrix) 的 K 标定是否有尺度偏差;
* **横向中心** (dx, dy) —— 面板零点 / 相机零点是否还对中 (跨会话零点会漂);
* **灰阶** 993 vs 1023 —— 面板物理 2π 是 993 灰度, 但编码按 1023 满量程走, 等效
  **+3.02% 的整体相位过调制** (``1023/993``), 等价于把命令相位乘 ``1.0302``;
* **模糊** (blur sigma) —— 冻结检查点的 ``sigma_ccd_px = 0``, 光学链路的实际 PSF 展宽
  在模型里完全缺失;
* **背景 / 增益** —— 曝光口径 (本批 0.034 ms) 与检查点标定口径不同, 绝对 counts 不可比。

这些参数**能不能从已落盘的硬件数据里拟合出来**, 以及**能不能与相位一起联合优化**,
是本脚本要给的判决。

只读离线
--------
脚本只读已经落盘的 ``plan.json`` / ``summary.json`` / ``phase/*.npy`` /
``frames/*.npy`` / ``model/*.npy`` 与冻结检查点, **全程不 import 任何设备驱动**,
收尾断言 ``ao_shaping.drivers`` 不在 ``sys.modules`` 里, 违反就 exit 3, 因此可以在
没有任何硬件的机器上跑。复算模型用的是仓库自己的
``FrozenDefocusForward`` (纯 numpy/torch, 无设备依赖)。

口径
----
* 采样约定与采集脚本一致: ``map_coordinates(frame, [y-0.5, x-0.5], order=1)``;
* 覆盖掩膜: 采样点 (及其双线性邻居) 落在全帧内的才算数, 窗外一律排除, 避免把
  零填充当信号;
* 暗电平: 用全帧里远离光斑的稳健地板估计, 从实测里减掉; 模型侧的 ``background``
  单独报告;
* 两套掩膜:
  * ``fit_mask`` —— 目标函数用的**大核** (窗心圆盘, 默认半径 240 CCD px)。它必须
    同时装下「实测零级」与「模型零级」, 否则拟合只会看到两个互不相关的斑点;
  * ``core_mask`` —— 指标用的**小核** (实测内容峰周围的圆盘, 默认半径 100 CCD px),
    逐图样对比用;
* 目标函数: **每个图样各自**解一次仿射 ``m ≈ a·P + b`` (吸收曝光口径的绝对增益/偏置),
  再算形状残差 ``√(Σ(m̂-M)²/ΣM²)``。这样绝对 counts 不可比的问题被隔离掉, 剩下的
  才是「形状差多少」。

几何参数的语义 (拟合与合成真值**完全一致**)
------------------------------------------
实测窗的第 ``i`` 个采样点对应相机坐标 ``spot + i·step``; 模型侧采样点改为
``spot − (dx, dy) + (s_x, s_y)·i·step``。于是

* ``dx > 0`` ⇒ 模型相机网格整体**向右**挪 ``dx`` 像素, 于是模型内容相对实测窗口
  **向左**偏 ``dx`` 像素 (内容在相机坐标里是固定的, 网格往哪挪, 内容就往反方向跑);
* ``s_x > 1`` ⇒ 模型内容相对实测窗口在 x 方向**放大** ``s_x`` 倍;
* ``dx = dy = 0, s_x = s_y = 1`` ⇒ 与实测窗严格重合 (冻结模型的原始口径)。

``(dx, dy)`` 与模型自带的 ``registration_shift_ccd_px`` (issue #237) 是同一件事:
后者把采样点往 ``−(dx, dy)`` 挪, 正好让内容往 ``+(dx, dy)`` 挪。

用法::

    .venv/Scripts/python.exe scripts/slm/analyze_fwdmodel_fit.py --dry-run
    .venv/Scripts/python.exe scripts/slm/analyze_fwdmodel_fit.py \
        --out report/slm/differentiable_opt/artifacts/fwdmodel_fit_20261116

产物 (默认输出目录见 ``DEFAULT_OUT_DIR_NAME``): ``fwdmodel_fit.json``、
``fwdmodel_fit.md``、``run.log``、``fwdfit_*.png`` 以及本脚本自身的源码副本。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
# 以文件形式启动时只有脚本自己的目录在 ``sys.path`` 上, 仓库根与 ``src``/``scripts``
# 都要自己补 (与 ``scripts/`` 下其它脚本同一约定)。
for _entry in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import matplotlib

matplotlib.use("Agg")  # 必须在 pyplot 之前

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import minimize
from scipy.stats import spearmanr

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    FrozenDefocusForward,
    FrozenDefocusParameters,
    load_frozen_parameters,
)
from ao_shaping.utils.wavefront.zernike_calc import generate_noll_polynomial
from scripts._common import fmt_general
from scripts._common.plot_style import setup_cjk_font

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
#: 退出码: 0 正常; 1 没有可分析的数据; 3 离线契约被破坏 (import 了设备驱动);
#: 4 合成自检没通过 (拟合没把真值捞回来)。
EXIT_OK = 0
EXIT_NO_DATA = 1
EXIT_OFFLINE_VIOLATION = 3
EXIT_SELFCHECK_FAILED = 4

SCRIPT_REL = "scripts/slm/analyze_fwdmodel_fit.py"
DEFAULT_OUT_DIR_NAME = "fwdmodel_fit_20261116"
REPORT_DIR = ROOT / "report" / "slm" / "differentiable_opt"
FIGURES_DIR = REPORT_DIR / "figures"

#: 冻结检查点 (M3 照明的 refit 冻结参数)。
DEFAULT_CHECKPOINT = (
    ROOT
    / "data/debug/slm_transfer/defocus_forward_20261107_focus_r2_fit"
    / "checkpoints/M3_illumination/frozen_parameters.json"
)

#: 已落盘的采集目录。``absolute_fidelity_12bit_fixed_20261116`` 是**主数据集**:
#: 曝光固定 0.034 ms、峰值 2383/4095 不饱和、定心落在 (1224, 1105), 且冻结前向在
#: 该窗上的复算与落盘 ``model/*.npy`` 逐点一致 (max|d| = 0)。
#: ``absolute_fidelity_20261116`` 是同日的对照: 曝光 1000 ms、峰值顶到 4095 (饱和),
#: 且定心取到了 (1220, 606) —— 那个零点不是本次光斑 (实测光斑其实在 y≈1105),
#: 采样窗整体偏了约 500 px, 所以它只能当**佐证**, 不能当主证据。
DEFAULT_DATASET_DIRS: tuple[Path, ...] = (
    ROOT / "data/debug/slm_transfer/absolute_fidelity_12bit_fixed_20261116",
    ROOT / "data/debug/slm_transfer/absolute_fidelity_20261116",
)

#: 拟合集 / 验证集。拟合集只用「模型已知该怎么解释」的方向性图样 (零相位、纯倾斜、
#: 离焦型), 验证集用随机相位与锚定 M3 最优网格 —— 后者的相位**不是**模型能自洽解释的,
#: 因此验证集上的数字才是「这套缺陷参数能不能外推」的判决。
DEFAULT_FIT_PATTERNS: tuple[str, ...] = (
    "zero",
    "tilt_x_1rad",
    "tilt_x_3rad",
    "tilt_y_1rad",
    "tilt_y_3rad",
    "defocus_z4_1rad",
)
DEFAULT_HOLDOUT_PATTERNS: tuple[str, ...] = (
    "rand_rms0p5_seed0",
    "rand_rms0p5_seed1",
    "rand_rms1p5_seed0",
    "rand_rms1p5_seed1",
    "anchored_m3_best",
    "anchored_m3_rs15_best",
)

#: 全局缺陷参数的搜索边界 (含物理含义的注释)。
#:
#: * ``defocus_delta_diopter``: 冻结 ``defocus_diopter`` 的增量。冻结值 0.0309 D 已经
#:   很小, ±0.02 D 足以覆盖「完全没对焦」到「差很多」; 再大没有硬件意义。
#: * ``lateral_scale_x/y``: 模型内容相对实测窗的放大倍率。1.0 = 与实测同一比例。
#: * ``centre_dx/dy_px``: 模型相机网格相对实测窗的平移 (CCD px), 正号 = 网格向右挪。
#:   上界取到窗的半跨度 260 px: 跨会话零点漂移实测有 ~180 px, 界给小了根本够不着。
#: * ``phase_scale``: 命令相位 → 有效相位的乘子。1.0 = 模型原样; 1.0302 = 灰阶
#:   过调制 (1023/993); 0.9707 = 反向 (993/1023)。
#: * ``blur_sigma_samples``: 模型窗格上的高斯展宽 (采样点), 乘 ``step_px`` 得 CCD px。
GLOBAL_BOUNDS: dict[str, tuple[float, float]] = {
    "defocus_delta_diopter": (-0.02, 0.02),
    "lateral_scale_x": (0.70, 1.80),
    "lateral_scale_y": (0.70, 1.80),
    "centre_dx_px": (-260.0, 260.0),
    "centre_dy_px": (-260.0, 260.0),
    "phase_scale": (0.85, 1.15),
    "blur_sigma_samples": (0.0, 6.0),
}
GLOBAL_NAMES: tuple[str, ...] = tuple(GLOBAL_BOUNDS)

#: 冻结 (无缺陷) 的基准值。
FROZEN_GLOBALS: dict[str, float] = {
    "defocus_delta_diopter": 0.0,
    "lateral_scale_x": 1.0,
    "lateral_scale_y": 1.0,
    "centre_dx_px": 0.0,
    "centre_dy_px": 0.0,
    "phase_scale": 1.0,
    "blur_sigma_samples": 0.0,
}

#: 灰阶口径的两个具名假设 (面板物理 2π = 993 灰度, 但编码按 1023 满量程走)。
GRAY_HYPOTHESES: dict[str, float] = {
    "frozen_no_scale": 1.0,
    "over_modulation_1023_over_993": 1023.0 / 993.0,
    "under_scaling_993_over_1023": 993.0 / 1023.0,
}

#: 指标用的核心核半径 (CCD px) 与目标函数用的大核半径 (CCD px)。
DEFAULT_CORE_RADIUS_CCD_PX = 100.0
DEFAULT_FIT_RADIUS_CCD_PX = 240.0

#: 径向剖面的最大半径 (CCD px) 与 bin 宽度 (CCD px)。
RADIAL_MAX_CCD_PX = 160.0
RADIAL_BIN_CCD_PX = 4.0

#: 找内容峰时的平滑半径 (采样点): 让 argmax 不被散斑骗走。
PEAK_SMOOTH_SIGMA = 3.0

#: 复算模型与落盘 ``model/*.npy`` 的对账容差 (绝对 counts)。
REPRO_TOL = 1e-6

#: 数值上的小量。
EPS = 1e-12

#: 目标函数退化时的兜底值 (Powell 不接受 inf)。
BIG_COST = 1.0e6


# ---------------------------------------------------------------------------
# 离线契约
# ---------------------------------------------------------------------------
def _leaked_driver_modules() -> list[str]:
    """``sys.modules`` 里所有 ``ao_shaping.drivers*`` (应为空)。"""
    return sorted(name for name in sys.modules if name.startswith("ao_shaping.drivers"))


def assert_no_drivers() -> None:
    """离线契约的硬保证: 收尾断言没 import 任何设备驱动。"""
    leaked = _leaked_driver_modules()
    if leaked:
        sys.stderr.write(
            "离线契约违约: 分析路径上导入了设备驱动包:\n  " + "\n  ".join(leaked) + "\n"
        )
        sys.stderr.flush()
        raise SystemExit(EXIT_OFFLINE_VIOLATION)
    logger.info("离线契约通过: sys.modules 里没有任何 ao_shaping.drivers.* (未打开任何设备)")


# ---------------------------------------------------------------------------
# 逐像元统计原语
# ---------------------------------------------------------------------------
def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """展平后的 Pearson 相关系数; 空 / 常量数组返回 NaN。"""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.size == 0 or x.size != y.size:
        return float("nan")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        x = np.nan_to_num(x)
        y = np.nan_to_num(y)
    if x.std() <= 0 or y.std() <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """展平后的 Spearman 秩相关系数; 退化输入返回 NaN。"""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.size == 0 or x.size != y.size or x.size < 3:
        return float("nan")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        x = np.nan_to_num(x)
        y = np.nan_to_num(y)
    if np.ptp(x) <= 0 or np.ptp(y) <= 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def nrmse(reference: np.ndarray, test: np.ndarray) -> float:
    """峰值归一化后的 RMS 误差 ``‖test̂-ref̂‖₂ / ‖ref̂‖₂``。"""
    ref = np.asarray(reference, dtype=np.float64).ravel()
    tst = np.asarray(test, dtype=np.float64).ravel()
    if ref.size == 0 or ref.size != tst.size:
        return float("nan")
    sr = float(np.max(ref))
    st = float(np.max(tst))
    if not (math.isfinite(sr) and math.isfinite(st)) or sr <= 0.0 or st <= 0.0:
        return float("nan")
    ref = ref / sr
    tst = tst / st
    denom = float(np.linalg.norm(ref))
    if denom <= EPS:
        return float("nan")
    return float(np.linalg.norm(tst - ref) / denom)


def affine_fit(model: np.ndarray, measured: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """在掩膜内最小二乘解 ``m ≈ a·P + b``, 返回 ``(a, b)``。

    模型在掩膜里几乎没有信号时 (例如零级错位 180 px、而小核只罩住实测零级),
    最小二乘会给出荒唐的系数 (1e6 量级), 所以这里按**相对标准差**判退化: 模型
    的起伏不到实测的千分之一就返回 NaN。
    """
    p = np.asarray(model, dtype=np.float64)[mask]
    y = np.asarray(measured, dtype=np.float64)[mask]
    if p.size < 2:
        return float("nan"), float("nan")
    if p.std() <= 1e-3 * max(y.std(), EPS):
        return float("nan"), float("nan")
    design = np.stack([p, np.ones_like(p)], axis=1)
    try:
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    return float(coef[0]), float(coef[1])


def shape_residual(
    model: np.ndarray, measured: np.ndarray, mask: np.ndarray
) -> tuple[float, float, float]:
    """``(残差, a, b)``: 仿射对齐后 ``√(Σ(m̂-M)²/ΣM²)`` (掩膜内)。

    残差是**无量纲**的形状差: 已吸收每个图样各自的绝对增益与偏置, 所以它只反映
    「形状差多少」, 不受曝光口径影响。
    """
    a, b = affine_fit(model, measured, mask)
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("nan"), a, b
    y = np.asarray(measured, dtype=np.float64)[mask]
    pred = a * np.asarray(model, dtype=np.float64)[mask] + b
    denom = float(np.sqrt(np.sum(y**2)))
    if denom <= EPS:
        return float("nan"), a, b
    return float(np.sqrt(np.sum((pred - y) ** 2)) / denom), a, b


def content_peak(image: np.ndarray, *, smooth_sigma: float = PEAK_SMOOTH_SIGMA) -> tuple[float, float]:
    """内容峰 (x, y) 坐标 (列, 行); 先平滑再取 argmax, 免得被散斑骗走。"""
    arr = np.asarray(image, dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).any() or float(np.max(arr)) <= 0:
        return float("nan"), float("nan")
    smoothed = gaussian_filter(arr, sigma=float(smooth_sigma)) if smooth_sigma > 0 else arr
    iy, ix = np.unravel_index(int(np.argmax(smoothed)), smoothed.shape)
    return float(ix), float(iy)


def fwhm_profile(profile: np.ndarray, step: float) -> float:
    """一维剖面的半高全宽 (CCD px); 线性插值取值。"""
    arr = np.asarray(profile, dtype=np.float64)
    if arr.size < 3 or not np.isfinite(arr).all():
        return float("nan")
    peak = float(np.max(arr))
    if peak <= 0:
        return float("nan")
    half = 0.5 * peak
    above = np.flatnonzero(arr >= half)
    if above.size < 2:
        return float("nan")
    lo = int(above[0])
    hi = int(above[-1])
    left = float(lo)
    if lo > 0 and arr[lo - 1] < half:
        denom = arr[lo] - arr[lo - 1]
        frac = 0.0 if abs(denom) <= EPS else (half - arr[lo - 1]) / denom
        left = lo - 1 + frac
    right = float(hi)
    if hi + 1 < arr.size and arr[hi + 1] < half:
        # 右半高点落在 [hi, hi+1] 这一段里: arr[hi] 已经在半高处时, 交点就在 hi。
        denom = arr[hi + 1] - arr[hi]
        frac = 0.0 if abs(denom) <= EPS else (half - arr[hi]) / denom
        right = hi + frac
    return float((right - left) * step)


def radial_profile(
    image: np.ndarray, centre_index_xy: tuple[float, float], *, step: float,
    max_radius_ccd: float = RADIAL_MAX_CCD_PX, bin_ccd: float = RADIAL_BIN_CCD_PX,
) -> tuple[np.ndarray, np.ndarray]:
    """以 ``centre_index_xy`` (列, 行, **窗格下标**) 为心的径向平均剖面。

    注意窗格是下采样的: 半径要乘 ``step_px`` 才是 CCD px; 圆心给的是**下标**,
    不是相机坐标。

    Returns:
        ``(半径中心 (CCD px), 剖面值)``; 半径中心从 ``bin_ccd/2`` 起。
    """
    arr = np.asarray(image, dtype=np.float64)
    height, width = arr.shape
    ys, xs = np.mgrid[0:height, 0:width]
    dx = (xs - float(centre_index_xy[0])) * step
    dy = (ys - float(centre_index_xy[1])) * step
    radius = np.hypot(dx, dy)
    n_bins = max(1, int(round(max_radius_ccd / bin_ccd)))
    edges = np.arange(n_bins + 1) * bin_ccd
    index = np.digitize(radius.ravel(), edges) - 1
    valid = (index >= 0) & (index < n_bins)
    sums = np.bincount(index[valid], weights=arr.ravel()[valid], minlength=n_bins)
    counts = np.bincount(index[valid], minlength=n_bins)
    profile = np.divide(sums, counts, out=np.zeros(n_bins), where=counts > 0)
    centres = (edges[:-1] + edges[1:]) / 2.0
    return centres, profile


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------
@dataclass
class PatternPair:
    """一个图样的三件套: 命令相位 / 实测全帧 / 落盘模型窗。"""

    pattern_id: str
    phase: np.ndarray
    frame: np.ndarray | None
    model_saved: np.ndarray | None
    plan_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Dataset:
    """一个采集目录的离线读入结果。"""

    name: str
    root: Path | None
    source: str
    pattern_ids: list[str]
    pairs: dict[str, PatternPair]
    spot_xy: tuple[float, float]
    exposure_ms: float
    half_span_px: float
    step_px: float
    centre_model: tuple[int, int]
    core_shape: str
    core_side_px: float
    plan_meta: dict[str, Any] = field(default_factory=dict)
    summary_meta: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def window_shape(self) -> tuple[int, int]:
        count = int(round(self.half_span_px / self.step_px))
        return (2 * count + 1, 2 * count + 1)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_real_dataset(data_dir: Path) -> Dataset:
    """读一个 ``absolute_fidelity_*`` 采集目录 (只读)。"""
    summary = _load_json(data_dir / "summary.json")
    plan_path = data_dir / "plan.json"
    plan = _load_json(plan_path) if plan_path.is_file() else {}

    spot = summary.get("spot_xy") or plan.get("spot_xy")
    if not spot:
        raise ValueError(f"{data_dir}: summary/plan 里没有 spot_xy")
    exposure = float(summary.get("exposure_ms", plan.get("exposure_ms", 1.0)))
    half = float(summary.get("window_half_span_px", plan.get("window_half_span_px", 260.0)))
    step = float(summary.get("window_step_px", plan.get("window_step_px", 2.0)))
    core = summary.get("core") or plan.get("core") or {}

    pattern_ids = list(summary.get("pattern_ids") or plan.get("pattern_ids") or [])
    plan_patterns = {str(item["pattern_id"]): item for item in plan.get("patterns", [])}

    pairs: dict[str, PatternPair] = {}
    for pid in pattern_ids:
        phase_path = data_dir / "phase" / f"{pid}.npy"
        frame_path = data_dir / "frames" / f"{pid}.npy"
        model_path = data_dir / "model" / f"{pid}.npy"
        if not phase_path.is_file():
            logger.warning(f"{data_dir.name}: 缺少相位 {phase_path.name}, 跳过")
            continue
        phase = np.load(phase_path).astype(np.float32)
        frame = np.load(frame_path).astype(np.float32) if frame_path.is_file() else None
        model = np.load(model_path).astype(np.float32) if model_path.is_file() else None
        pairs[pid] = PatternPair(
            pattern_id=pid,
            phase=phase,
            frame=frame,
            model_saved=model,
            plan_meta=plan_patterns.get(pid, {}),
        )

    count = int(round(half / step))
    centre = summary.get("centre_model") or plan.get("centre_model") or [count, count]
    return Dataset(
        name=data_dir.name,
        root=data_dir,
        source="real",
        pattern_ids=pattern_ids,
        pairs=pairs,
        spot_xy=(float(spot[0]), float(spot[1])),
        exposure_ms=exposure,
        half_span_px=half,
        step_px=step,
        centre_model=(int(centre[0]), int(centre[1])),
        core_shape=str(core.get("shape", "square")),
        core_side_px=float(core.get("side", 400.0)),
        plan_meta=plan,
        summary_meta=summary,
    )


# ---------------------------------------------------------------------------
# 几何: 采样窗 / 覆盖掩膜 / 掩膜
# ---------------------------------------------------------------------------
def window_offsets(dataset: Dataset) -> np.ndarray:
    """采样窗相对窗心的偏移 (x 与 y 同), 单位 CCD px。"""
    count = int(round(dataset.half_span_px / dataset.step_px))
    return np.arange(-count, count + 1, dtype=np.float64) * dataset.step_px


def camera_grid(
    dataset: Dataset, *, dx_px: float = 0.0, dy_px: float = 0.0,
    scale_x: float = 1.0, scale_y: float = 1.0,
) -> np.ndarray:
    """模型侧采样窗 ``(H, W, 2)`` (x, y), 单位 CCD px。

    口径见模块 docstring: 实测窗第 ``i`` 个采样点对应相机坐标 ``spot + i·step``,
    模型侧改为 ``spot − (dx, dy) + (s_x, s_y)·i·step``。
    """
    offsets = window_offsets(dataset)
    height = width = offsets.size
    xs = dataset.spot_xy[0] - dx_px + scale_x * offsets
    ys = dataset.spot_xy[1] - dy_px + scale_y * offsets
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([xx, yy], axis=-1)


def coverage_mask(grid: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
    """采样窗里哪些点被全帧真的覆盖 (双线性邻居都在帧内)。"""
    height, width = int(frame_shape[0]), int(frame_shape[1])
    xs = grid[..., 0]
    ys = grid[..., 1]
    return (xs >= 1.0) & (xs <= width - 1.0) & (ys >= 1.0) & (ys <= height - 1.0)


def disk_mask(
    grid: np.ndarray, centre_xy: tuple[float, float], *, radius_ccd_px: float, step_px: float
) -> np.ndarray:
    """以 ``centre_xy`` (x, y, **相机坐标**) 为心、半径 ``radius_ccd_px`` 的圆盘。

    注意窗格是**下采样**的: 窗格下标 ``i`` 对应相机坐标 ``grid[0, i, 0]``, 与相机
    坐标不是一回事。所以圆心必须先换算成**下标**, 半径再乘 ``step_px`` 变成 CCD px。
    """
    height, width = grid.shape[:2]
    col = int(np.argmin(np.abs(grid[0, :, 0] - float(centre_xy[0]))))
    row = int(np.argmin(np.abs(grid[:, 0, 1] - float(centre_xy[1]))))
    ys, xs = np.mgrid[0:height, 0:width]
    radius = np.hypot((xs - col) * step_px, (ys - row) * step_px)
    return radius <= float(radius_ccd_px)


def grid_centre_xy(grid: np.ndarray) -> tuple[float, float]:
    """采样窗的相机坐标中心 ``(x, y)``。"""
    return float(np.mean(grid[0, :, 0])), float(np.mean(grid[:, 0, 1]))


def index_to_camera(grid: np.ndarray, col: float, row: float) -> tuple[float, float]:
    """窗格下标 → 相机坐标 (x, y)。"""
    height, width = grid.shape[:2]
    col = float(min(max(col, 0.0), width - 1.0))
    row = float(min(max(row, 0.0), height - 1.0))
    c0 = int(np.floor(col))
    r0 = int(np.floor(row))
    c1 = min(c0 + 1, width - 1)
    r1 = min(r0 + 1, height - 1)
    fx = col - c0
    fy = row - r0
    x = (1 - fx) * grid[r0, c0, 0] + fx * grid[r0, c1, 0]
    y = (1 - fy) * grid[r0, c0, 1] + fy * grid[r1, c0, 1]
    return float(x), float(y)


def sample_frame_to_window(frame: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """把全帧 CCD 帧采样到窗网格 (与采集脚本同一约定)。"""
    coords = np.stack([grid[..., 1] - 0.5, grid[..., 0] - 0.5], axis=0).reshape(2, -1)
    return map_coordinates(
        np.asarray(frame, dtype=np.float64), coords, order=1, mode="constant"
    ).reshape(grid.shape[:2])


def dark_level(
    frame: np.ndarray, spot_xy: tuple[float, float], *, exclude_radius_px: float = 260.0
) -> float:
    """全帧里远离光斑的稳健地板 (暗电平 + 杂散光)。"""
    arr = np.asarray(frame, dtype=np.float64)
    height, width = arr.shape
    ys, xs = np.mgrid[0:height, 0:width]
    radius = np.hypot(xs - float(spot_xy[0]), ys - float(spot_xy[1]))
    far = radius > float(exclude_radius_px)
    if not far.any():
        return float(np.percentile(arr, 10.0))
    return float(np.median(arr[far]))


# ---------------------------------------------------------------------------
# 模型窗
# ---------------------------------------------------------------------------
class ModelFactory:
    """按全局缺陷参数造模型窗; 同一采样网格的前向会被缓存。

    快路径: ``slm_axial_offset_m == 0`` 时 ``axial_plane_correction`` 的 scale 恒为 1,
    所以**离焦只改** ``effective_defocus``, 采样网格不动 —— 于是只换离焦时不必重建
    前向 (重建一次要 ~70 ms, 全在 1920² 的 Zernike 基上)。脚本自带的「模型复算对账」
    会把这条快路径钉住 (复算必须与落盘 ``model/*.npy`` 逐点一致)。
    """

    def __init__(self, params: FrozenDefocusParameters, dataset: Dataset, *, device: str = "cpu"):
        self.params = params
        self.dataset = dataset
        self.device = device
        self._fast_defocus = float(params.slm_axial_offset_m) == 0.0
        self._cache: dict[tuple, FrozenDefocusForward] = {}

    def forward(
        self, *, defocus_delta: float, dx_px: float, dy_px: float,
        scale_x: float, scale_y: float,
    ) -> FrozenDefocusForward:
        grid_key = (
            round(dx_px, 9), round(dy_px, 9), round(scale_x, 9), round(scale_y, 9),
        )
        cached = self._cache.get(grid_key)
        if cached is None:
            grid = camera_grid(
                self.dataset, dx_px=dx_px, dy_px=dy_px, scale_x=scale_x, scale_y=scale_y
            )
            params = replace(
                self.params, defocus_diopter=self.params.defocus_diopter + float(defocus_delta)
            )
            fwd = FrozenDefocusForward(params, grid, device=self.device)
            if len(self._cache) > 512:
                self._cache.clear()
            self._cache[grid_key] = fwd
            return fwd
        if not self._fast_defocus:
            grid = camera_grid(
                self.dataset, dx_px=dx_px, dy_px=dy_px, scale_x=scale_x, scale_y=scale_y
            )
            params = replace(
                self.params, defocus_diopter=self.params.defocus_diopter + float(defocus_delta)
            )
            return FrozenDefocusForward(params, grid, device=self.device)
        clone = copy.copy(cached)
        clone.effective_defocus = float(cached.params.defocus_diopter) + float(defocus_delta)
        return clone

    def window(
        self,
        phase: np.ndarray,
        *,
        defocus_delta: float = 0.0,
        dx_px: float = 0.0,
        dy_px: float = 0.0,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        phase_scale: float = 1.0,
        blur_sigma_samples: float = 0.0,
        exposure_ms: float | None = None,
    ) -> np.ndarray:
        """模型窗 (强度, 单位与冻结检查点一致)。"""
        fwd = self.forward(
            defocus_delta=defocus_delta, dx_px=dx_px, dy_px=dy_px,
            scale_x=scale_x, scale_y=scale_y,
        )
        commanded = np.asarray(phase, dtype=np.float32) * float(phase_scale)
        with torch.no_grad():
            out = fwd.predict(
                torch.as_tensor(commanded[None]),
                exposure_ms=None if exposure_ms is None else float(exposure_ms),
            )
        image = out[0].detach().cpu().numpy().astype(np.float64)
        if blur_sigma_samples and blur_sigma_samples > 0:
            image = gaussian_filter(image, sigma=float(blur_sigma_samples))
        return image


# ---------------------------------------------------------------------------
# 目标函数
# ---------------------------------------------------------------------------
@dataclass
class PreparedPattern:
    """一个图样的实测侧准备结果 (窗口 / 掩膜 / 暗电平)。"""

    pattern_id: str
    measured: np.ndarray
    fit_mask: np.ndarray
    core_mask: np.ndarray
    dark: float
    phase: np.ndarray
    model_baseline: np.ndarray
    plan_meta: dict[str, Any]


def prepare_patterns(
    dataset: Dataset,
    factory: ModelFactory,
    *,
    core_radius_ccd_px: float = DEFAULT_CORE_RADIUS_CCD_PX,
    fit_radius_ccd_px: float = DEFAULT_FIT_RADIUS_CCD_PX,
) -> list[PreparedPattern]:
    """配对 + 暗电平扣除 + 覆盖掩膜 + 大核 (拟合) / 小核 (指标)。"""
    prepared: list[PreparedPattern] = []
    true_grid = camera_grid(dataset)
    centre = grid_centre_xy(true_grid)
    for pid in dataset.pattern_ids:
        pair = dataset.pairs.get(pid)
        if pair is None or pair.frame is None:
            continue
        dark = dark_level(pair.frame, dataset.spot_xy)
        raw = sample_frame_to_window(pair.frame, true_grid)
        measured = raw - dark
        coverage = coverage_mask(true_grid, pair.frame.shape)
        fit_mask = coverage & disk_mask(
            true_grid, centre, radius_ccd_px=fit_radius_ccd_px, step_px=dataset.step_px
        )
        # 小核以**实测内容峰**为心 (实测零级所在), 不以窗心为心。
        peak_col, peak_row = content_peak(measured)
        peak_xy_cam = index_to_camera(true_grid, peak_col, peak_row)
        core_mask = coverage & disk_mask(
            true_grid, peak_xy_cam, radius_ccd_px=core_radius_ccd_px, step_px=dataset.step_px
        )
        if not fit_mask.any() or not core_mask.any():
            logger.warning(f"{dataset.name}/{pid}: 掩膜为空, 跳过")
            continue
        baseline = (
            np.asarray(pair.model_saved, dtype=np.float64)
            if pair.model_saved is not None
            else factory.window(pair.phase, exposure_ms=dataset.exposure_ms)
        )
        prepared.append(
            PreparedPattern(
                pattern_id=pid,
                measured=measured,
                fit_mask=fit_mask,
                core_mask=core_mask,
                dark=dark,
                phase=pair.phase,
                model_baseline=baseline,
                plan_meta=pair.plan_meta,
            )
        )
    return prepared


def global_cost(
    prepared: Sequence[PreparedPattern],
    factory: ModelFactory,
    dataset: Dataset,
    params: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """全局缺陷参数下的总形状残差 (越小越好) 与逐图样的仿射系数。

    目标函数刻意**不含**绝对增益/偏置: 每个图样各自解一次 ``m ≈ a·P + b``, 于是
    曝光口径的差异被完全隔离, 剩下的残差只反映形状 (焦斑尺寸 / 位置 / 旁瓣)。
    """
    total = 0.0
    affine: dict[str, float] = {}
    for item in prepared:
        model = factory.window(
            item.phase,
            defocus_delta=params["defocus_delta_diopter"],
            dx_px=params["centre_dx_px"],
            dy_px=params["centre_dy_px"],
            scale_x=params["lateral_scale_x"],
            scale_y=params["lateral_scale_y"],
            phase_scale=params["phase_scale"],
            blur_sigma_samples=params["blur_sigma_samples"],
            exposure_ms=dataset.exposure_ms,
        )
        residual, a, b = shape_residual(model, item.measured, item.fit_mask)
        if not math.isfinite(residual):
            return BIG_COST, {}
        total += residual
        affine[f"{item.pattern_id}:a"] = a
        affine[f"{item.pattern_id}:b"] = b
    return float(total), affine


def estimate_align_shift(model: np.ndarray, measured: np.ndarray) -> tuple[float, float]:
    """把 ``model`` 对齐到 ``measured`` 所需的平移 ``(dx, dy)`` (采样点)。

    用相位相关 (FFT): 去均值后求互功率谱, 峰值位置就是平移。返回值是**把模型内容
    挪到实测内容处**所需的采样点数, 正号 = 向右/向下。

    约定 (由 ``tests/ao_shaping/scripts/test_analyze_fwdmodel_fit.py`` 钉住):
    若 ``measured[i] == model[i - s]`` (即实测是模型右移 ``s`` 个采样点), 返回
    ``(s, 0)``。
    """
    a = np.asarray(model, dtype=np.float64)
    b = np.asarray(measured, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        return float("nan"), float("nan")
    a = a - float(a.mean())
    b = b - float(b.mean())
    if float(np.linalg.norm(a)) <= EPS or float(np.linalg.norm(b)) <= EPS:
        return float("nan"), float("nan")
    corr = np.fft.ifft2(np.conj(np.fft.fft2(a)) * np.fft.fft2(b)).real
    height, width = corr.shape
    peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
    dx = _parabolic_offset(corr[peak[0], (peak[1] - 1) % width],
                           corr[peak[0], peak[1]],
                           corr[peak[0], (peak[1] + 1) % width])
    dy = _parabolic_offset(corr[(peak[0] - 1) % height, peak[1]],
                           corr[peak[0], peak[1]],
                           corr[(peak[0] + 1) % height, peak[1]])
    # 互相关在 k 处达到峰值 ⇒ 实测是模型右移 k 个采样点 ⇒ 模型要往回挪 k。
    shift_x = float(peak[1])
    shift_y = float(peak[0])
    if shift_x > width / 2:
        shift_x -= width
    if shift_y > height / 2:
        shift_y -= height
    return float(shift_x + dx), float(shift_y + dy)


def _parabolic_offset(left: float, centre: float, right: float) -> float:
    """三点抛物线插值的亚采样峰偏移。"""
    denom = left - 2.0 * centre + right
    if abs(denom) <= EPS:
        return 0.0
    return float(0.5 * (left - right) / denom)


def _clip_to_bounds(params: dict[str, float]) -> dict[str, float]:
    return {
        name: float(min(max(float(params[name]), lo), hi))
        for name, (lo, hi) in GLOBAL_BOUNDS.items()
    }


def _to_vector(params: dict[str, float]) -> np.ndarray:
    return np.array([float(params[name]) for name in GLOBAL_NAMES], dtype=np.float64)


def _from_vector(vector: Sequence[float]) -> dict[str, float]:
    return {name: float(v) for name, v in zip(GLOBAL_NAMES, vector)}


def fit_globals(
    prepared: Sequence[PreparedPattern],
    factory: ModelFactory,
    dataset: Dataset,
    *,
    max_evals: int = 400,
) -> dict[str, Any]:
    """全局缺陷参数的拟合: 互相关给中心初值 → 坐标下降 → Powell 精修。

    为什么要互相关打底: 中心错位是这批数据里**最大**的一项 (冻结模型的光轴零点
    与采集当天的实测零点差了上百像素), 而中心误差会让其余参数 (尺度/离焦/模糊) 的
    代价面变得多峰 —— 直接上 Powell 很容易掉进「中心没对准」的坑里, 后面所有参数
    都会被中心误差污染。
    """
    base_cost, _ = global_cost(prepared, factory, dataset, FROZEN_GLOBALS)

    # --- 0) 互相关给中心初值 ---
    shifts: list[tuple[float, float]] = []
    for item in prepared:
        model = factory.window(item.phase, exposure_ms=dataset.exposure_ms)
        shifts.append(estimate_align_shift(model, item.measured))
    good = [(x, y) for x, y in shifts if math.isfinite(x) and math.isfinite(y)]
    best = dict(FROZEN_GLOBALS)
    if good:
        best["centre_dx_px"] = float(np.median([x for x, _ in good])) * dataset.step_px
        best["centre_dy_px"] = float(np.median([y for _, y in good])) * dataset.step_px
    seeded_cost, _ = global_cost(prepared, factory, dataset, best)
    if seeded_cost > base_cost:
        best = dict(FROZEN_GLOBALS)
        best_cost = base_cost
    else:
        best_cost = seeded_cost

    # --- 1) 坐标下降: 逐参数扫描 (先粗后细), 两轮 ---
    for round_index in range(2):
        for name, values in _coordinate_grid(best, wide=round_index == 0).items():
            for value in values:
                trial = _clip_to_bounds({**best, name: float(value)})
                cost, _ = global_cost(prepared, factory, dataset, trial)
                if cost < best_cost:
                    best_cost = cost
                    best = trial

    # --- 2) Powell 精修 ---
    def objective(vector: np.ndarray) -> float:
        params = _clip_to_bounds(_from_vector(vector))
        cost, _ = global_cost(prepared, factory, dataset, params)
        return cost

    result = minimize(
        objective,
        _to_vector(best),
        method="Powell",
        options={"maxfev": int(max_evals), "xtol": 1e-4, "ftol": 1e-6},
    )
    refined = _clip_to_bounds(_from_vector(result.x))
    refined_cost, affine = global_cost(prepared, factory, dataset, refined)
    if refined_cost > best_cost:
        refined = best
        refined_cost = best_cost
        affine = global_cost(prepared, factory, dataset, best)[1]

    at_bounds = [
        name for name, (lo, hi) in GLOBAL_BOUNDS.items()
        if min(abs(refined[name] - lo), abs(refined[name] - hi))
        <= 1e-6 * max(abs(hi - lo), 1.0)
    ]
    return {
        "params": refined,
        "cost": float(refined_cost),
        "baseline_cost": float(base_cost),
        "grid_params": best,
        "grid_cost": float(best_cost),
        "at_bounds": at_bounds,
        "optimizer": {
            "method": "phase-correlation + coordinate descent + Powell",
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
        },
        "affine": affine,
    }


def _coordinate_grid(seed: dict[str, float], *, wide: bool) -> dict[str, np.ndarray]:
    """坐标下降用的逐参数扫描点 (以当前值为中心)。

    ``wide=True`` 的第一轮扫**全区间** (尺度/离焦/相位标度/模糊的真实修正量可能
    很大, 局部窗口够不着); 第二轮才收到当前值附近做细化。
    """
    if wide:
        return {
            "defocus_delta_diopter": np.linspace(-0.015, 0.015, 9),
            "lateral_scale_x": np.linspace(0.70, 1.80, 9),
            "lateral_scale_y": np.linspace(0.70, 1.80, 9),
            "centre_dx_px": np.linspace(
                max(-260.0, seed["centre_dx_px"] - 40.0),
                min(260.0, seed["centre_dx_px"] + 40.0), 7,
            ),
            "centre_dy_px": np.linspace(
                max(-260.0, seed["centre_dy_px"] - 40.0),
                min(260.0, seed["centre_dy_px"] + 40.0), 7,
            ),
            "phase_scale": np.linspace(0.85, 1.15, 9),
            "blur_sigma_samples": np.linspace(0.0, 6.0, 7),
        }
    return {
        "defocus_delta_diopter": np.linspace(
            max(-0.02, seed["defocus_delta_diopter"] - 0.004),
            min(0.02, seed["defocus_delta_diopter"] + 0.004), 7,
        ),
        "lateral_scale_x": np.linspace(
            max(0.70, seed["lateral_scale_x"] - 0.05),
            min(1.80, seed["lateral_scale_x"] + 0.05), 7,
        ),
        "lateral_scale_y": np.linspace(
            max(0.70, seed["lateral_scale_y"] - 0.05),
            min(1.80, seed["lateral_scale_y"] + 0.05), 7,
        ),
        "centre_dx_px": np.linspace(
            max(-260.0, seed["centre_dx_px"] - 12.0),
            min(260.0, seed["centre_dx_px"] + 12.0), 7,
        ),
        "centre_dy_px": np.linspace(
            max(-260.0, seed["centre_dy_px"] - 12.0),
            min(260.0, seed["centre_dy_px"] + 12.0), 7,
        ),
        "phase_scale": np.linspace(
            max(0.85, seed["phase_scale"] - 0.015),
            min(1.15, seed["phase_scale"] + 0.015), 7,
        ),
        "blur_sigma_samples": np.linspace(
            max(0.0, seed["blur_sigma_samples"] - 0.5),
            min(6.0, seed["blur_sigma_samples"] + 0.5), 7,
        ),
    }


# ---------------------------------------------------------------------------
# 逐图样指标
# ---------------------------------------------------------------------------
def pattern_metrics(
    model: np.ndarray,
    measured: np.ndarray,
    core_mask: np.ndarray,
    *,
    fit_mask: np.ndarray,
    step_px: float,
    grid: np.ndarray,
) -> dict[str, Any]:
    """单个图样的失配指标 (相关系数 / 归一化残差 / 径向剖面 / FWHM / 峰值偏移)。

    * 相关性与残差给**两套**: 小核 (共用的, 含平移信息) 与大核 (拟合核, 装得下
      零级 —— 冻结模型的零级错位 ~180 px, 小核里根本没有模型的信号, 小核残差会是
      NaN, 所以「冻结 vs 拟合」的主表用大核残差);
    * FWHM 与径向剖面各自以**自己的**内容峰为心 (只量形状, 不量平移);
    * 峰值偏移单独给出 (量平移)。
    """
    m = np.asarray(model, dtype=np.float64)
    y = np.asarray(measured, dtype=np.float64)
    residual, a, b = shape_residual(m, y, core_mask)
    residual_wide, a_wide, b_wide = shape_residual(m, y, fit_mask)

    m_col, m_row = content_peak(m)
    y_col, y_row = content_peak(y)
    m_cam = index_to_camera(grid, m_col, m_row)
    y_cam = index_to_camera(grid, y_col, y_row)
    # 径向剖面比的是**形状**, 所以模型侧先用大核仿射归一化 (增益/偏置), 否则
    # 模型与实测的绝对 counts 不同, 比出来的全是单位差。
    if math.isfinite(a_wide) and math.isfinite(b_wide):
        m_shape = a_wide * m + b_wide
    else:
        m_shape = m
    radii, prof_m = radial_profile(m_shape, (m_col, m_row), step=step_px)
    _, prof_y = radial_profile(y, (y_col, y_row), step=step_px)
    profile_corr = pearson(prof_m, prof_y)
    denom = float(np.linalg.norm(prof_y))
    profile_l1 = (
        float(np.sum(np.abs(prof_m - prof_y)) / denom) if denom > EPS else float("nan")
    )

    peak_dx = m_cam[0] - y_cam[0]
    peak_dy = m_cam[1] - y_cam[1]

    height, width = m.shape
    row_m = int(round(m_row)) if math.isfinite(m_row) else height // 2
    col_m = int(round(m_col)) if math.isfinite(m_col) else width // 2
    row_y = int(round(y_row)) if math.isfinite(y_row) else height // 2
    col_y = int(round(y_col)) if math.isfinite(y_col) else width // 2
    row_m = min(max(row_m, 0), height - 1)
    col_m = min(max(col_m, 0), width - 1)
    row_y = min(max(row_y, 0), height - 1)
    col_y = min(max(col_y, 0), width - 1)
    fwhm_x_model = fwhm_profile(m[row_m, :], step_px)
    fwhm_x_meas = fwhm_profile(y[row_y, :], step_px)
    fwhm_y_model = fwhm_profile(m[:, col_m], step_px)
    fwhm_y_meas = fwhm_profile(y[:, col_y], step_px)

    def _ratio(a_: float, b_: float) -> float:
        if math.isfinite(a_) and math.isfinite(b_) and b_ > 0:
            return float(a_ / b_)
        return float("nan")

    return {
        "spearman": spearman(m[core_mask], y[core_mask]),
        "pearson": pearson(m[core_mask], y[core_mask]),
        "shape_residual": residual,
        "spearman_wide": spearman(m[fit_mask], y[fit_mask]),
        "pearson_wide": pearson(m[fit_mask], y[fit_mask]),
        "shape_residual_wide": residual_wide,
        "nrmse_peak_normalised": nrmse(m[core_mask], y[core_mask]),
        "radial_profile_correlation": profile_corr,
        "radial_profile_l1": profile_l1,
        "fwhm_model_x_ccd_px": fwhm_x_model,
        "fwhm_measured_x_ccd_px": fwhm_x_meas,
        "fwhm_ratio_x": _ratio(fwhm_x_model, fwhm_x_meas),
        "fwhm_model_y_ccd_px": fwhm_y_model,
        "fwhm_measured_y_ccd_px": fwhm_y_meas,
        "fwhm_ratio_y": _ratio(fwhm_y_model, fwhm_y_meas),
        "peak_offset_dx_ccd_px": peak_dx,
        "peak_offset_dy_ccd_px": peak_dy,
        "peak_offset_norm_ccd_px": float(math.hypot(peak_dx, peak_dy)),
        "model_peak_camera_xy": list(m_cam),
        "measured_peak_camera_xy": list(y_cam),
        "model_peak": float(np.max(m)),
        "measured_peak": float(np.max(y)),
        "affine_a": a,
        "affine_b": b,
        "affine_a_wide": a_wide,
        "affine_b_wide": b_wide,
        "radial_radii_ccd_px": [float(v) for v in radii],
        "radial_profile_model": [float(v) for v in prof_m],
        "radial_profile_measured": [float(v) for v in prof_y],
    }


# ---------------------------------------------------------------------------
# 边际曲线 / 二维图 / 雅可比
# ---------------------------------------------------------------------------
def marginal_curves(
    prepared: Sequence[PreparedPattern],
    factory: ModelFactory,
    dataset: Dataset,
    optimum: dict[str, float],
    *,
    n_points: int = 13,
) -> dict[str, Any]:
    """每个全局参数在最优值附近的边际代价曲线 (其余参数固定在最优值)。"""
    curves: dict[str, Any] = {}
    for name in GLOBAL_NAMES:
        lo, hi = GLOBAL_BOUNDS[name]
        span = hi - lo
        values = np.linspace(
            max(lo, optimum[name] - 0.35 * span),
            min(hi, optimum[name] + 0.35 * span),
            int(n_points),
        )
        costs = []
        for value in values:
            cost, _ = global_cost(
                prepared, factory, dataset, {**optimum, name: float(value)}
            )
            costs.append(cost)
        frozen_cost, _ = global_cost(
            prepared, factory, dataset, {**optimum, name: FROZEN_GLOBALS[name]}
        )
        curves[name] = {
            "values": [float(v) for v in values],
            "costs": [float(c) for c in costs],
            "frozen_value_cost": float(frozen_cost),
            "best_cost": float(min(costs)),
        }
    return curves


def map_2d(
    prepared: Sequence[PreparedPattern],
    factory: ModelFactory,
    dataset: Dataset,
    optimum: dict[str, float],
    name_x: str,
    name_y: str,
    *,
    n_points: int = 11,
) -> dict[str, Any]:
    """两个全局参数的二维代价图 (其余固定在最优值)。"""
    lo_x, hi_x = GLOBAL_BOUNDS[name_x]
    lo_y, hi_y = GLOBAL_BOUNDS[name_y]
    xs = np.linspace(lo_x, hi_x, int(n_points))
    ys = np.linspace(lo_y, hi_y, int(n_points))
    grid = np.zeros((len(ys), len(xs)))
    for iy, yv in enumerate(ys):
        for ix, xv in enumerate(xs):
            params = dict(optimum)
            params[name_x] = float(xv)
            params[name_y] = float(yv)
            cost, _ = global_cost(prepared, factory, dataset, params)
            grid[iy, ix] = cost
    return {
        "name_x": name_x,
        "name_y": name_y,
        "x": [float(v) for v in xs],
        "y": [float(v) for v in ys],
        "cost": [[float(v) for v in row] for row in grid],
    }


def jacobian_spectrum(
    prepared: Sequence[PreparedPattern],
    factory: ModelFactory,
    dataset: Dataset,
    optimum: dict[str, float],
    *,
    rel_step: float = 0.01,
) -> dict[str, Any]:
    """最优点的代价梯度 (有限差分) 与逐参数的**敏感度**。

    目标函数是标量, 所以「条件数」在 1×N 的雅可比上没有意义 (奇异值只有一个)。
    有意义的是**每个参数在它的整个取值范围里能把代价改变多少**:
    ``sensitivity_i = |∂cost/∂x_i| · (hi_i − lo_i)`` —— 敏感度接近 0 的参数, 这批
    数据**根本约束不住**它 (不可辨识); 敏感度大的参数才谈得上「拟合出来了」。
    """
    base, _ = global_cost(prepared, factory, dataset, optimum)
    grad = np.zeros(len(GLOBAL_NAMES))
    for i, name in enumerate(GLOBAL_NAMES):
        lo, hi = GLOBAL_BOUNDS[name]
        step = max(abs(optimum[name]) * rel_step, 0.02 * (hi - lo))
        plus = dict(optimum)
        minus = dict(optimum)
        plus[name] = min(hi, optimum[name] + step)
        minus[name] = max(lo, optimum[name] - step)
        c_plus, _ = global_cost(prepared, factory, dataset, plus)
        c_minus, _ = global_cost(prepared, factory, dataset, minus)
        denom = plus[name] - minus[name]
        grad[i] = (c_plus - c_minus) / denom if abs(denom) > EPS else 0.0
    spans = np.array([GLOBAL_BOUNDS[name][1] - GLOBAL_BOUNDS[name][0] for name in GLOBAL_NAMES])
    sensitivity = np.abs(grad) * spans
    reference = float(np.max(sensitivity)) if sensitivity.size else 0.0
    return {
        "base_cost": float(base),
        "names": list(GLOBAL_NAMES),
        "gradient": [float(v) for v in grad],
        "sensitivity": [float(v) for v in sensitivity],
        "sensitivity_relative": (
            [float(v / reference) for v in sensitivity] if reference > EPS else []
        ),
        "identifiable": [
            name for name, value in zip(GLOBAL_NAMES, sensitivity)
            if value >= 0.05 * reference
        ],
        "unidentifiable": [
            name for name, value in zip(GLOBAL_NAMES, sensitivity)
            if value < 0.05 * reference
        ],
    }


# ---------------------------------------------------------------------------
# 联合优化 (相位 + 全局参数) 可辨识性
# ---------------------------------------------------------------------------
def zernike_basis(grid: int, max_n: int) -> np.ndarray:
    """低阶 Zernike 基 (Noll 归一化, 圆外置 0), 形状 ``(n_modes, grid, grid)``。"""
    modes: list[np.ndarray] = []
    for n in range(1, int(max_n) + 1):
        for m in range(-n, n + 1, 2):
            raw = generate_noll_polynomial(n, m, (int(grid), int(grid)), 1.0)
            arr = np.nan_to_num(
                np.asarray(raw, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
            )
            modes.append(arr)
    return np.stack(modes) if modes else np.zeros((0, int(grid), int(grid)), dtype=np.float32)


def joint_fit(
    prepared: Sequence[PreparedPattern],
    factory: ModelFactory,
    dataset: Dataset,
    *,
    start_globals: dict[str, float],
    max_noll: int = 3,
    max_evals: int = 600,
) -> dict[str, Any]:
    """相位 (低阶 Zernike 残差) 与全局缺陷参数**联合**优化。

    目的不是拿到最优解, 而是看两件事:

    1. 全频随机相位能把残差压到多低 (即「形状失配」里有多少是相位本身能解释的);
    2. 联合优化把全局参数拖到了哪里 (若与「只拟合全局参数」的解差很多, 说明两者
       互为替代, 全局参数**不可辨识**)。
    """
    grid = int(prepared[0].phase.shape[0])
    basis = zernike_basis(grid, max_noll)
    n_modes = basis.shape[0]
    basis_t = torch.as_tensor(basis, dtype=torch.float32)

    def evaluate(vector: np.ndarray) -> tuple[float, list[float]]:
        globals_vec = vector[: len(GLOBAL_NAMES)]
        coeffs = vector[len(GLOBAL_NAMES):]
        params = _clip_to_bounds(_from_vector(globals_vec))
        coeff_tensor = torch.as_tensor(coeffs, dtype=torch.float32)
        total = 0.0
        detail: list[float] = []
        for item in prepared:
            extra = (basis_t * coeff_tensor[:, None, None]).sum(0)
            phase = torch.as_tensor(item.phase, dtype=torch.float32) + extra
            model = factory.window(
                phase.numpy(),
                defocus_delta=params["defocus_delta_diopter"],
                dx_px=params["centre_dx_px"],
                dy_px=params["centre_dy_px"],
                scale_x=params["lateral_scale_x"],
                scale_y=params["lateral_scale_y"],
                phase_scale=params["phase_scale"],
                blur_sigma_samples=params["blur_sigma_samples"],
                exposure_ms=dataset.exposure_ms,
            )
            residual, _, _ = shape_residual(model, item.measured, item.fit_mask)
            if not math.isfinite(residual):
                return BIG_COST, []
            total += residual
            detail.append(float(residual))
        return float(total), detail

    x0 = np.concatenate([_to_vector(start_globals), np.zeros(n_modes, dtype=np.float64)])

    def objective(vector: np.ndarray) -> float:
        cost, _ = evaluate(vector)
        return cost

    result = minimize(
        objective,
        x0,
        method="Powell",
        options={"maxfev": int(max_evals), "xtol": 1e-3, "ftol": 1e-6},
    )
    cost, detail = evaluate(result.x)
    globals_final = _clip_to_bounds(_from_vector(result.x[: len(GLOBAL_NAMES)]))
    coeffs = result.x[len(GLOBAL_NAMES):]
    return {
        "n_modes": int(n_modes),
        "max_noll": int(max_noll),
        "cost": float(cost),
        "globals": globals_final,
        "coefficients": [float(c) for c in coeffs],
        "coefficient_rms_rad": (
            float(np.sqrt(np.mean(np.square(coeffs)))) if coeffs.size else 0.0
        ),
        "per_pattern_residual": detail,
        "optimizer": {
            "method": "Powell",
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
        },
    }


# ---------------------------------------------------------------------------
# 灰阶材料性
# ---------------------------------------------------------------------------
def gray_materiality(
    prepared: Sequence[PreparedPattern],
    factory: ModelFactory,
    dataset: Dataset,
    optimum: dict[str, float],
) -> dict[str, Any]:
    """灰阶 (993 vs 1023) 与离焦的材料性对比。

    做法: 在**其它参数最优**的前提下, 把单个参数换成一个具名假设, 看形状残差涨多少。
    「涨得多」= 该缺陷在这批数据上**材料性**强。
    """

    def cost_with(**overrides: float) -> float:
        params = dict(optimum)
        params.update(overrides)
        cost, _ = global_cost(prepared, factory, dataset, params)
        return float(cost)

    best = cost_with()
    baseline = float(global_cost(prepared, factory, dataset, FROZEN_GLOBALS)[0])

    hypotheses = {
        "gray_1023_over_993": cost_with(
            phase_scale=GRAY_HYPOTHESES["over_modulation_1023_over_993"]
        ),
        "gray_993_over_1023": cost_with(
            phase_scale=GRAY_HYPOTHESES["under_scaling_993_over_1023"]
        ),
        "gray_frozen_1p0": cost_with(phase_scale=1.0),
        "defocus_plus_0p002D": cost_with(
            defocus_delta_diopter=optimum["defocus_delta_diopter"] + 0.002
        ),
        "defocus_minus_0p002D": cost_with(
            defocus_delta_diopter=optimum["defocus_delta_diopter"] - 0.002
        ),
        "scale_x_plus_1pct": cost_with(lateral_scale_x=optimum["lateral_scale_x"] * 1.01),
        "scale_x_minus_1pct": cost_with(lateral_scale_x=optimum["lateral_scale_x"] * 0.99),
        "scale_y_plus_1pct": cost_with(lateral_scale_y=optimum["lateral_scale_y"] * 1.01),
        "scale_y_minus_1pct": cost_with(lateral_scale_y=optimum["lateral_scale_y"] * 0.99),
        "blur_plus_0p5_sample": cost_with(
            blur_sigma_samples=optimum["blur_sigma_samples"] + 0.5
        ),
        "blur_minus_0p5_sample": cost_with(
            blur_sigma_samples=max(0.0, optimum["blur_sigma_samples"] - 0.5)
        ),
    }
    delta = {name: float(value - best) for name, value in hypotheses.items()}

    # 灰阶假设对**远场本身**的影响: 3% 的相位标度差到底把远场改成什么样。
    sensitivity: dict[str, Any] = {}
    for label, scale in GRAY_HYPOTHESES.items():
        corrs: list[float] = []
        nrmses: list[float] = []
        for item in prepared:
            common = dict(
                defocus_delta=optimum["defocus_delta_diopter"],
                dx_px=optimum["centre_dx_px"],
                dy_px=optimum["centre_dy_px"],
                scale_x=optimum["lateral_scale_x"],
                scale_y=optimum["lateral_scale_y"],
                blur_sigma_samples=optimum["blur_sigma_samples"],
                exposure_ms=dataset.exposure_ms,
            )
            reference = factory.window(item.phase, phase_scale=1.0, **common)
            other = factory.window(item.phase, phase_scale=scale, **common)
            corrs.append(pearson(reference[item.fit_mask], other[item.fit_mask]))
            nrmses.append(nrmse(reference[item.fit_mask], other[item.fit_mask]))
        sensitivity[label] = {
            "phase_scale": scale,
            "mean_correlation_vs_frozen_phase_scale": float(np.mean(corrs)),
            "mean_nrmse_vs_frozen_phase_scale": float(np.mean(nrmses)),
            "per_pattern_correlation": [float(c) for c in corrs],
        }

    equivalent: dict[str, float] = {}
    for rms in (0.5, 1.5, 3.0):
        equivalent[f"rms_{rms}rad_phase_error"] = float(rms * (1023.0 / 993.0 - 1.0))
    return {
        "best_cost": best,
        "frozen_cost": baseline,
        "hypothesis_costs": hypotheses,
        "delta_cost_vs_best": delta,
        "phase_scale_sensitivity": sensitivity,
        "gray_over_modulation_factor": GRAY_HYPOTHESES["over_modulation_1023_over_993"],
        "equivalent_phase_error_rad": equivalent,
    }


# ---------------------------------------------------------------------------
# 合成数据 (dry-run 自检)
# ---------------------------------------------------------------------------
def _noll_phase(grid: int, nm: tuple[int, int], amplitude: float) -> np.ndarray:
    raw = generate_noll_polynomial(
        int(nm[0]), int(nm[1]), (int(grid), int(grid)), float(amplitude)
    )
    return np.nan_to_num(
        np.asarray(raw, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )


def synthetic_frozen_parameters() -> FrozenDefocusParameters:
    """dry-run 基线冻结参数 (检查点缺失时用, 不碰真实数据)。

    形状沿 ``differentiable_defocus_uniformity`` 单测先例, 尺度对齐
    :func:`make_synthetic_dataset` 默认窗 (grid=128, step=2 px, 光斑 1224/1105);
    合成与拟合走**同一个前向**, 真值回收不依赖参数物理真实性。
    增益/模糊取值 (gain=250, sigma_ccd_px=1) 让残差降幅稳过 50% 自检门槛
    (gain=25/sigma=2 时基线失配主导不足, 降幅仅 25.6%)。
    """
    return FrozenDefocusParameters(
        defocus_diopter=-0.1,
        origin_xy=(1224.0, 1105.0),
        amplitude_center_xy=(960.0, 600.0),
        amplitude_width_xy=(400.0, 300.0),
        residual_noll_radians={5: 0.3},
        gain=250.0,
        background=1.0,
        sigma_ccd_px=1.0,
        ramp_matrix=((8000.0, 0.0), (0.0, -8000.0)),
        fft_size=256,
        pixel_samples=1,
        grid_size=128,
    )


def _synthetic_build(
    params: FrozenDefocusParameters,
    *,
    truth: dict[str, float],
    grid: int,
    half_span_px: float,
    step_px: float,
    spot_xy: tuple[float, float],
    frame_shape: tuple[int, int],
    exposure_ms: float,
    dark: float,
    noise: float,
    pattern_ids: Sequence[str],
    seed: int,
) -> Dataset:
    """按真值造一批合成采集 (见 :func:`make_synthetic_dataset`)。"""
    rng = np.random.default_rng(seed)
    dataset = Dataset(
        name="synthetic_smoke",
        root=None,
        source="synthetic",
        pattern_ids=list(pattern_ids),
        pairs={},
        spot_xy=spot_xy,
        exposure_ms=exposure_ms,
        half_span_px=half_span_px,
        step_px=step_px,
        centre_model=(
            int(round(half_span_px / step_px)),
            int(round(half_span_px / step_px)),
        ),
        core_shape="square",
        core_side_px=400.0,
    )
    factory = ModelFactory(params, dataset)
    # 窗格下标 i ↔ 相机坐标 spot + (i − half_count)·step, 所以下标 i 的相机坐标是
    # (camera − spot)/step + half_count。half_count 是**半**跨度, 不是数组长度。
    half_count = int(round(dataset.half_span_px / dataset.step_px))

    for pid in pattern_ids:
        if pid == "zero":
            phase = np.zeros((grid, grid), dtype=np.float32)
        elif pid.startswith("tilt_x"):
            phase = _noll_phase(grid, (2, 1), 1.0 if pid.endswith("1rad") else 3.0)
        elif pid.startswith("defocus"):
            phase = _noll_phase(grid, (4, 0), 1.0)
        else:
            raw = rng.standard_normal((grid, grid))
            phase = (raw * (1.5 / float(raw.std()))).astype(np.float32)

        measured_window = factory.window(
            phase,
            defocus_delta=truth["defocus_delta_diopter"],
            dx_px=truth["centre_dx_px"],
            dy_px=truth["centre_dy_px"],
            scale_x=truth["lateral_scale_x"],
            scale_y=truth["lateral_scale_y"],
            phase_scale=truth["phase_scale"],
            blur_sigma_samples=truth["blur_sigma_samples"],
            exposure_ms=exposure_ms,
        )
        measured_window = measured_window + rng.normal(0.0, noise, measured_window.shape)

        # 铺回全帧 (逆映射): 相机坐标 → 实测窗下标。
        height, width = frame_shape
        ys, xs = np.mgrid[0:height, 0:width]
        frame = map_coordinates(
            measured_window,
            [
                (ys - spot_xy[1]) / step_px + half_count,
                (xs - spot_xy[0]) / step_px + half_count,
            ],
            order=1,
            mode="constant",
        )
        frame = frame + dark + rng.normal(0.0, noise, frame.shape)
        dataset.pairs[pid] = PatternPair(
            pattern_id=pid,
            phase=phase,
            frame=frame.astype(np.float32),
            model_saved=None,
            plan_meta={"pattern_id": pid},
        )
    return dataset


def make_synthetic_dataset(
    params: FrozenDefocusParameters,
    *,
    truth: dict[str, float],
    grid: int = 128,
    half_span_px: float = 260.0,
    step_px: float = 2.0,
    spot_xy: tuple[float, float] = (1224.0, 1105.0),
    frame_shape: tuple[int, int] = (1900, 1600),
    exposure_ms: float = 0.034,
    dark: float = 12.0,
    noise: float = 1.5,
    pattern_ids: Sequence[str] = ("zero", "tilt_x_1rad", "rand_rms1p5_seed0"),
    seed: int = 20261117,
    centre_on_model_axis: bool = True,
) -> Dataset:
    """造一批**已知真值**的合成采集, 用来验证分析机器 (而不是物理) 是对的。

    「实测」= 用**同一套代码路径**在真值参数下算出的模型窗, 再加暗电平与噪声,
    最后铺回全帧 —— 于是拟合必须把真值原样捞回来。检的是: 掩膜逻辑、CCD 采样
    往返、暗电平/偏置处理、几何参数的**符号约定**、以及优化器本身。

    真值语义与真数据、与拟合口径**完全一致** (见模块 docstring 的几何约定)。

    ``centre_on_model_axis=True`` 时, 采样窗心取**模型自己的光轴** (前向预测的
    零级所在相机坐标), 于是「窗口中心 ↔ 模型光轴」的常量错位被消掉, 拟合面对的
    只有真值里的那几项缺陷 —— 这是真数据里**不成立**的理想化, 但正是自检想要的
    可分性 (真数据里 (dx, dy) 与尺度本来就部分退化, 那是结论不是缺陷)。
    """
    kwargs = dict(
        truth=truth,
        grid=grid,
        half_span_px=half_span_px,
        step_px=step_px,
        frame_shape=frame_shape,
        exposure_ms=exposure_ms,
        dark=dark,
        noise=noise,
        pattern_ids=pattern_ids,
        seed=seed,
    )
    dataset = _synthetic_build(params, spot_xy=spot_xy, **kwargs)
    if centre_on_model_axis:
        factory = ModelFactory(params, dataset)
        zero_pair = dataset.pairs.get("zero")
        if zero_pair is not None:
            content = factory.window(zero_pair.phase, exposure_ms=dataset.exposure_ms)
            col, row = content_peak(content)
            mx, my = index_to_camera(camera_grid(dataset), col, row)
            dataset = _synthetic_build(params, spot_xy=(mx, my), **kwargs)
            logger.info(f"合成数据集: 采样窗心已对齐到模型光轴 ({mx:.1f}, {my:.1f})")
    logger.info(f"合成数据集已生成: {len(dataset.pairs)} 个图样, 真值 {truth}")
    return dataset


#: 合成自检的容差 (拟合值 vs 真值)。离焦 / 模糊 / 相位标度 / 尺度之间本来就互相
#: 退化 (都改 PSF), 所以这几项给宽容差并且**只作参考**; 中心平移是唯一**锐利**的
#: 可辨识项, 必须过关。
SYNTHETIC_TOLERANCES: dict[str, float] = {
    "defocus_delta_diopter": 3.0e-3,
    "lateral_scale_x": 0.06,
    "lateral_scale_y": 0.06,
    "centre_dx_px": 6.0,
    "centre_dy_px": 6.0,
    "phase_scale": 0.02,
    "blur_sigma_samples": 0.3,
}

#: 合成自检里**必须**过关的参数 (其余只报参考值)。
SYNTHETIC_MUST_PASS: tuple[str, ...] = ("centre_dx_px", "centre_dy_px")


def check_synthetic_recovery(
    truth: dict[str, float], fitted: dict[str, float],
    *, baseline_cost: float = float("nan"), fitted_cost: float = float("nan"),
    min_reduction: float = 0.5,
) -> dict[str, Any]:
    """比较合成真值与拟合值, 给出逐项容差判定。

    只有 ``SYNTHETIC_MUST_PASS`` 里的参数算「必须过关」; 其余 (离焦/模糊/相位标度/
    尺度) 互相退化, 合成自检只作参考 —— 退化本身是结论的一部分。另外要求残差有
    实质下降 (否则拟合根本没起作用)。
    """
    rows: list[dict[str, Any]] = []
    for name, tol in SYNTHETIC_TOLERANCES.items():
        got = float(fitted[name])
        want = float(truth[name])
        rows.append(
            {
                "parameter": name,
                "truth": want,
                "fitted": got,
                "error": got - want,
                "tolerance": tol,
                "must_pass": name in SYNTHETIC_MUST_PASS,
                "passed": bool(abs(got - want) <= tol),
            }
        )
    must_pass_rows = [row for row in rows if row["must_pass"]]
    reduction = (
        float((baseline_cost - fitted_cost) / baseline_cost)
        if math.isfinite(baseline_cost) and baseline_cost > EPS else float("nan")
    )
    reduction_ok = bool(math.isfinite(reduction) and reduction >= min_reduction)
    return {
        "rows": rows,
        "passed": bool(all(row["passed"] for row in must_pass_rows) and reduction_ok),
        "all_within_tolerance": bool(all(row["passed"] for row in rows)),
        "residual_reduction": reduction,
        "residual_reduction_required": min_reduction,
        "residual_reduction_ok": reduction_ok,
    }


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    path = out_dir / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"图已写出: {path}")
    return path


def figure_radial(payloads: list[dict[str, Any]], out_dir: Path) -> Path:
    """各图样的径向剖面: 实测 vs 冻结模型 vs 拟合模型。"""
    setup_cjk_font()
    rows: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []
    for payload in payloads:
        for pid, metrics in payload["baseline"].items():
            fitted = payload.get("fitted_metrics", {}).get(pid)
            if fitted is None:
                continue
            rows.append((payload["name"], pid, metrics, fitted))
    if not rows:
        raise ValueError("没有可画的径向剖面数据")
    ncol = 3
    nrow = int(math.ceil(len(rows) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
    for index, (name, pid, base, fit) in enumerate(rows):
        ax = axes[index // ncol][index % ncol]
        radii = np.asarray(base["radial_radii_ccd_px"], dtype=np.float64)
        meas = np.asarray(base["radial_profile_measured"], dtype=np.float64)
        model = np.asarray(base["radial_profile_model"], dtype=np.float64)
        scale = float(np.max(meas)) if meas.size and np.max(meas) > 0 else 1.0
        ax.plot(radii, meas / scale, "o-", ms=3, label="实测")
        ax.plot(radii, model / scale, "s--", ms=3, label="冻结模型")
        fit_prof = np.asarray(fit.get("radial_profile_model", []), dtype=np.float64)
        if fit_prof.size == radii.size:
            ax.plot(radii, fit_prof / scale, "^:", ms=3, label="拟合模型")
        ax.set_title(f"{pid}\n{name}", fontsize=8)
        ax.set_xlabel("半径 (CCD px)", fontsize=8)
        ax.set_ylabel("归一化强度", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
    for index in range(len(rows), nrow * ncol):
        axes[index // ncol][index % ncol].axis("off")
    fig.tight_layout()
    return _save(fig, out_dir, "fwdfit_radial_profiles.png")


def figure_sweeps(curves: dict[str, Any], out_dir: Path) -> Path:
    """全局参数的边际代价曲线。"""
    setup_cjk_font()
    names = list(curves)
    ncol = 3
    nrow = int(math.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow), squeeze=False)
    for index, name in enumerate(names):
        ax = axes[index // ncol][index % ncol]
        values = np.asarray(curves[name]["values"], dtype=np.float64)
        costs = np.asarray(curves[name]["costs"], dtype=np.float64)
        ax.plot(values, costs, "o-", ms=3)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("参数值", fontsize=8)
        ax.set_ylabel("形状残差 (Σ)", fontsize=8)
        ax.tick_params(labelsize=7)
    for index in range(len(names), nrow * ncol):
        axes[index // ncol][index % ncol].axis("off")
    fig.tight_layout()
    return _save(fig, out_dir, "fwdfit_marginal_curves.png")


def figure_maps(maps: list[dict[str, Any]], out_dir: Path) -> Path:
    """两两参数的二维代价图 (退化岭的直观证据)。"""
    setup_cjk_font()
    if not maps:
        raise ValueError("没有可画的代价图")
    ncol = 2
    nrow = int(math.ceil(len(maps) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 4.4 * nrow), squeeze=False)
    for index, payload in enumerate(maps):
        ax = axes[index // ncol][index % ncol]
        grid = np.asarray(payload["cost"], dtype=np.float64)
        xs = np.asarray(payload["x"], dtype=np.float64)
        ys = np.asarray(payload["y"], dtype=np.float64)
        mesh = ax.pcolormesh(xs, ys, grid, shading="auto", cmap="viridis")
        fig.colorbar(mesh, ax=ax)
        ax.set_xlabel(payload["name_x"], fontsize=8)
        ax.set_ylabel(payload["name_y"], fontsize=8)
        ax.tick_params(labelsize=7)
    for index in range(len(maps), nrow * ncol):
        axes[index // ncol][index % ncol].axis("off")
    fig.tight_layout()
    return _save(fig, out_dir, "fwdfit_cost_maps.png")


def figure_holdout(
    payload: dict[str, Any],
    images: dict[str, dict[str, np.ndarray]],
    out_dir: Path,
    *,
    limit: int = 3,
) -> Path:
    """验证集图样: 实测 / 冻结模型 / 拟合模型 (强度图)。"""
    setup_cjk_font()
    holdout = [row for row in payload.get("holdout_table", []) if row["set"] == "holdout"]
    picked = holdout[:limit]
    if not picked:
        raise ValueError("没有可画的验证集图样")
    ncol = 3
    nrow = len(picked)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.4 * nrow), squeeze=False)
    for row, item in enumerate(picked):
        pid = item["pattern_id"]
        panels = [
            ("实测", images.get(pid, {}).get("measured")),
            ("冻结模型", images.get(pid, {}).get("model_baseline")),
            ("拟合模型", images.get(pid, {}).get("model_fitted")),
        ]
        finite = [float(np.max(p[1])) for p in panels if p[1] is not None]
        vmax = max(finite) if finite else 1.0
        for col, (title, image) in enumerate(panels):
            ax = axes[row][col]
            if image is None:
                ax.axis("off")
                continue
            ax.imshow(np.asarray(image, dtype=np.float64), cmap="inferno", vmin=0.0, vmax=vmax)
            ax.set_title(f"{pid} · {title}", fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    return _save(fig, out_dir, "fwdfit_holdout_images.png")


def figure_gray(materiality: dict[str, Any], out_dir: Path) -> Path:
    """各具名缺陷假设带来的残差增量 (材料性)。"""
    setup_cjk_font()
    delta = materiality["delta_cost_vs_best"]
    names = list(delta)
    values = [float(delta[name]) for name in names]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar(range(len(names)), values, color="#4C72B0")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("形状残差增量 (相对最优)", fontsize=9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return _save(fig, out_dir, "fwdfit_gray_materiality.png")


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def _cell(value: Any) -> str:
    """报告表格单元的渲染 (用仓库共享的格式化器, 不自己再造一个)。"""
    return fmt_general(value, ".4g", bool_labels=("是", "否"), verbatim_int=True)


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(v) for v in row) + " |")
    return "\n".join(lines)


def render_markdown(payload: dict[str, Any]) -> str:
    """把结果渲染成人读的 Markdown 报告。"""
    parts: list[str] = []
    parts.append("# 物理预测模型缺陷拟合分析 (M3 绝对保真度数据)\n")
    parts.append(f"- 生成时间 (UTC): `{payload['generated_utc']}`")
    parts.append(f"- 脚本: `{payload['script']}`")
    parts.append(
        f"- 离线契约: `sys.modules` 里的设备驱动包 = "
        f"{payload['offline']['driver_modules_imported'] or '无'}"
    )
    parts.append("")

    verdict = payload.get("verdicts", {})
    if verdict:
        parts.append("## 结论速览\n")
        for key, value in verdict.items():
            parts.append(f"- **{key}**: {value}")
        parts.append("")

    for dataset in payload["datasets"]:
        name = dataset["name"]
        parts.append(f"## 数据集 `{name}`\n")
        meta = dataset["metadata"]
        parts.append(
            _table(
                ["项", "值"],
                [
                    ["来源", dataset["source"]],
                    ["根目录", dataset.get("root") or "(内存合成)"],
                    ["零点 spot_xy (x, y)", meta["spot_xy"]],
                    ["曝光 (ms)", meta["exposure_ms"]],
                    ["采样窗", f"{meta['window_shape']} @ step {meta['step_px']} px"],
                    ["图样数", len(dataset["pattern_ids"])],
                    [
                        "模型复算最大偏差 (counts)",
                        dataset["pairing"]["model_reproduction_max_abs_diff"],
                    ],
                    ["配对问题", dataset["pairing"]["issues"] or "无"],
                ],
            )
        )
        parts.append("")

        fit = dataset["fit"]
        parts.append("### 拟合到的全局缺陷参数\n")
        parts.append(
            _table(
                ["参数", "冻结值", "拟合值", "物理含义"],
                [
                    [
                        "defocus_delta_diopter",
                        0.0,
                        fit["params"]["defocus_delta_diopter"],
                        f"模型总离焦 {dataset['frozen_model']['defocus_diopter']} + Δ D (屈光度)",
                    ],
                    [
                        "lateral_scale_x",
                        1.0,
                        fit["params"]["lateral_scale_x"],
                        "模型内容相对实测窗的 x 向放大倍率",
                    ],
                    [
                        "lateral_scale_y",
                        1.0,
                        fit["params"]["lateral_scale_y"],
                        "模型内容相对实测窗的 y 向放大倍率",
                    ],
                    [
                        "centre_dx_px",
                        0.0,
                        fit["params"]["centre_dx_px"],
                        "模型相机网格相对实测窗的 x 向平移 (CCD px, 正 = 网格右挪)",
                    ],
                    [
                        "centre_dy_px",
                        0.0,
                        fit["params"]["centre_dy_px"],
                        "模型相机网格相对实测窗的 y 向平移 (CCD px, 正 = 网格下挪)",
                    ],
                    [
                        "phase_scale",
                        1.0,
                        fit["params"]["phase_scale"],
                        "命令相位 → 有效相位的乘子 (1.0302 = 993/1023 过调制)",
                    ],
                    [
                        "blur_sigma_samples",
                        0.0,
                        fit["params"]["blur_sigma_samples"],
                        f"模型窗上的高斯展宽 (采样点; = "
                        f"{fit['params']['blur_sigma_samples'] * meta['step_px']:.3g} CCD px)",
                    ],
                ],
            )
        )
        parts.append("")
        parts.append(
            f"- 拟合集形状残差合计: 冻结 **{_cell(fit['baseline_cost'])}** → 拟合 "
            f"**{_cell(fit['cost'])}** (下降 "
            f"{_cell(100.0 * (fit['baseline_cost'] - fit['cost']) / max(fit['baseline_cost'], EPS))} %)"
        )
        if fit.get("at_bounds"):
            parts.append(
                f"- ⚠ 拟合值顶到搜索边界的参数: `{fit['at_bounds']}` "
                "(这些参数的数值只是边界值, 不是真值)"
            )
        parts.append("")

        parts.append("### 逐图样: 冻结模型 vs 拟合模型\n")
        rows = []
        for item in dataset["holdout_table"]:
            rows.append(
                [
                    item["pattern_id"],
                    item["set"],
                    item["baseline"]["shape_residual_wide"],
                    item["fitted"]["shape_residual_wide"],
                    item["baseline"]["pearson_wide"],
                    item["fitted"]["pearson_wide"],
                    item["baseline"]["fwhm_ratio_x"],
                    item["fitted"]["fwhm_ratio_x"],
                    item["baseline"]["fwhm_ratio_y"],
                    item["fitted"]["fwhm_ratio_y"],
                    item["baseline"]["peak_offset_norm_ccd_px"],
                    item["fitted"]["peak_offset_norm_ccd_px"],
                ]
            )
        parts.append(
            _table(
                [
                    "图样",
                    "集合",
                    "残差(冻结)",
                    "残差(拟合)",
                    "Pearson(冻结)",
                    "Pearson(拟合)",
                    "FWHM比x(冻结)",
                    "FWHM比x(拟合)",
                    "FWHM比y(冻结)",
                    "FWHM比y(拟合)",
                    "峰值偏移px(冻结)",
                    "峰值偏移px(拟合)",
                ],
                rows,
            )
        )
        parts.append("")
        parts.append(
            "> 残差与相关性在**大核** (拟合核, 半径 "
            f"{dataset['metadata']['fit_radius_ccd_px']:.0f} CCD px) 上算; 小核 "
            f"(半径 {dataset['metadata']['core_radius_ccd_px']:.0f} CCD px, 罩住实测零级) "
            "在冻结模型上是空的 —— 冻结模型的零级根本不在小核里, 那一列会全是 NaN。"
        )
        parts.append("")

        jac = dataset.get("jacobian")
        if jac:
            parts.append("### 可辨识性 (最优点的代价敏感度)\n")
            parts.append(
                "- 敏感度 (代价能被她整个取值范围改变多少): "
                + str(dict(zip(jac["names"], [_cell(v) for v in jac["sensitivity"]])))
            )
            parts.append(f"- 目标函数约束得住: `{jac['identifiable']}`")
            parts.append(f"- 目标函数约束不住 (不可辨识): `{jac['unidentifiable']}`")
            parts.append("")

        joint = dataset.get("joint")
        if joint:
            parts.append("### 联合优化 (低阶相位 + 全局参数)\n")
            parts.append(
                _table(
                    ["项", "值"],
                    [
                        ["相位基 (Noll)", f"n ≤ {joint['max_noll']} ({joint['n_modes']} 个模态)"],
                        ["仅全局参数 (拟合集) 残差", joint["globals_only_cost"]],
                        ["仅全频随机相位 (全局冻结) 残差", joint["phase_only_cost"]],
                        ["相位 + 全局 联合 残差", joint["cost"]],
                        ["联合后 defocus_delta_diopter", joint["globals"]["defocus_delta_diopter"]],
                        ["联合后 lateral_scale_x", joint["globals"]["lateral_scale_x"]],
                        ["联合后 lateral_scale_y", joint["globals"]["lateral_scale_y"]],
                        ["联合后 centre_dx_px", joint["globals"]["centre_dx_px"]],
                        ["联合后 centre_dy_px", joint["globals"]["centre_dy_px"]],
                        ["联合后 phase_scale", joint["globals"]["phase_scale"]],
                        ["联合后 blur_sigma_samples", joint["globals"]["blur_sigma_samples"]],
                        ["残差相位系数 RMS (rad)", joint["coefficient_rms_rad"]],
                    ],
                )
            )
            parts.append("")

        gray = dataset.get("gray_materiality")
        if gray:
            parts.append("### 灰阶 / 离焦材料性\n")
            rows = [
                [name_, value, gray["delta_cost_vs_best"][name_]]
                for name_, value in gray["hypothesis_costs"].items()
            ]
            parts.append(_table(["具名假设", "形状残差合计", "相对最优的增量"], rows))
            parts.append("")
            parts.append(
                f"- 灰阶过调制因子 1023/993 = {gray['gray_over_modulation_factor']:.6f}"
            )
            parts.append(f"- 等效相位误差: {gray['equivalent_phase_error_rad']}")
            parts.append("")

    parts.append("## 最小额外数据需求 (要把剩下的失配也吃掉)\n")
    parts.append(
        "1. **同会话的零级位置 / 模型原点**: 冻结检查点的 `origin_xy` 来自 2026-11-07 的\n"
        "   M3 照明拟合, 而采集是 2026-11-16, 期间零级漂了 ~180 px。只要在**同一次会话**\n"
        "   里重跑一次零级定心 (probe 已经用同帧定心), 或把零级像素坐标一起存进检查点,\n"
        "   这项失配就完全消失 —— 不需要任何新硬件数据, 只需要**同会话**的零点坐标。\n"
        "2. **离焦**: 这一批数据对离焦**不敏感** (代价对 ΔD 几乎无梯度), 所以拟合不出\n"
        "   来。离焦必须靠**独立的焦点扫描** (改变 z 位移, 量 FWHM vs 焦点) 或者\n"
        "   `phys_20261116_01/focus_sweep.csv` 那类数据来定, 不能靠远场强度形状反演。\n"
        "3. **照明形状**: 冻结模型假设面板上的照明是高斯 (width 383×142 panel px, 各向异性\n"
        "   2.7:1), 而实测零级是**圆**的 (13.8×13.6 px) 而模型是 **20×10 px 的椭圆**。\n"
        "   要解释这个, 需要同会话量一次**面板上的光斑形状** (或者干脆把照明振幅从\n"
        "   「各向异性高斯」换成实测形状)。\n"
        "4. **灰阶**: 3% 的相位标度差在远场强度分布上只有 ~1% (NRMSE), 而零级错位是\n"
        "   100% (Pearson 从 0 到 0.97)。所以灰阶**不需要**额外数据, 但也**测不出来** ——\n"
        "   要测它得做**斑点级**的对比 (同一张图上零级之外的散斑), 而不是零级轮廓。"
    )
    parts.append("")
    parts.append("## 产物\n")
    for item in payload.get("artifacts", []):
        parts.append(f"- `{item}`")
    parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def analyse_dataset(
    dataset: Dataset,
    params: FrozenDefocusParameters,
    *,
    fit_patterns: Sequence[str],
    holdout_patterns: Sequence[str],
    core_radius_ccd_px: float,
    fit_radius_ccd_px: float,
    max_fit_evals: int,
    max_joint_evals: int,
    joint_max_noll: int,
    do_joint: bool,
    do_sweeps: bool,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """一个数据集的完整分析, 返回 (可 JSON 序列化的结果, 图像字典)。"""
    started = time.monotonic()
    factory = ModelFactory(params, dataset)
    prepared = prepare_patterns(
        dataset,
        factory,
        core_radius_ccd_px=core_radius_ccd_px,
        fit_radius_ccd_px=fit_radius_ccd_px,
    )
    prepared_by_id = {item.pattern_id: item for item in prepared}
    logger.info(f"{dataset.name}: 配对到 {len(prepared)} 个图样")

    true_grid = camera_grid(dataset)

    # --- 配对自检: 落盘 model 与前向复算是否一致 (冻结参数是否真的能复现) ---
    repro: list[float] = []
    pairing_issues: list[str] = []
    for pid, item in prepared_by_id.items():
        saved = dataset.pairs[pid].model_saved
        if saved is None:
            continue
        recomputed = factory.window(item.phase, exposure_ms=dataset.exposure_ms)
        if recomputed.shape != saved.shape:
            pairing_issues.append(
                f"{pid}: 模型形状不符 {recomputed.shape} vs {saved.shape}"
            )
            continue
        diff = float(np.max(np.abs(recomputed - np.asarray(saved, dtype=np.float64))))
        repro.append(diff)
        if diff > REPRO_TOL:
            pairing_issues.append(f"{pid}: 模型复算偏差 {diff:.3g} > {REPRO_TOL:g}")
        plan_rms = item.plan_meta.get("phase_rms_rad")
        if plan_rms is not None:
            actual = float(np.std(item.phase))
            if abs(actual - float(plan_rms)) > 1e-3:
                pairing_issues.append(
                    f"{pid}: 相位 RMS {actual:.5f} 与 plan 记录 {plan_rms} 不符"
                )
    repro_max = max(repro) if repro else float("nan")

    # --- 基线指标 ---
    baseline_metrics: dict[str, dict[str, Any]] = {}
    for pid, item in prepared_by_id.items():
        baseline_metrics[pid] = pattern_metrics(
            item.model_baseline,
            item.measured,
            item.core_mask,
            fit_mask=item.fit_mask,
            step_px=dataset.step_px,
            grid=true_grid,
        )

    # --- 拟合 ---
    fit_set = [prepared_by_id[pid] for pid in fit_patterns if pid in prepared_by_id]
    holdout_set = [prepared_by_id[pid] for pid in holdout_patterns if pid in prepared_by_id]
    if not fit_set:
        raise ValueError(f"{dataset.name}: 拟合集为空")
    fit_ids = {item.pattern_id for item in fit_set}
    logger.info(
        f"{dataset.name}: 拟合集 {len(fit_set)} 个 {sorted(fit_ids)}, "
        f"验证集 {len(holdout_set)} 个"
    )

    fit = fit_globals(fit_set, factory, dataset, max_evals=max_fit_evals)
    optimum = fit["params"]
    logger.info(
        f"{dataset.name}: 拟合残差 {fit['baseline_cost']:.5f} → {fit['cost']:.5f}"
    )

    # --- 拟合后的指标 (拟合集 + 验证集) ---
    fitted_metrics: dict[str, dict[str, Any]] = {}
    images: dict[str, dict[str, np.ndarray]] = {}
    for pid, item in prepared_by_id.items():
        model = factory.window(
            item.phase,
            defocus_delta=optimum["defocus_delta_diopter"],
            dx_px=optimum["centre_dx_px"],
            dy_px=optimum["centre_dy_px"],
            scale_x=optimum["lateral_scale_x"],
            scale_y=optimum["lateral_scale_y"],
            phase_scale=optimum["phase_scale"],
            blur_sigma_samples=optimum["blur_sigma_samples"],
            exposure_ms=dataset.exposure_ms,
        )
        images[pid] = {
            "measured": item.measured,
            "model_baseline": item.model_baseline,
            "model_fitted": model,
        }
        fitted_metrics[pid] = pattern_metrics(
            model,
            item.measured,
            item.core_mask,
            fit_mask=item.fit_mask,
            step_px=dataset.step_px,
            grid=true_grid,
        )

    holdout_table = [
        {
            "pattern_id": pid,
            "set": "fit" if pid in fit_ids else "holdout",
            "baseline": baseline_metrics[pid],
            "fitted": fitted_metrics[pid],
        }
        for pid in prepared_by_id
    ]

    result: dict[str, Any] = {
        "name": dataset.name,
        "source": dataset.source,
        "root": str(dataset.root) if dataset.root else None,
        "metadata": {
            "spot_xy": list(dataset.spot_xy),
            "exposure_ms": dataset.exposure_ms,
            "half_span_px": dataset.half_span_px,
            "step_px": dataset.step_px,
            "window_shape": list(dataset.window_shape),
            "core_radius_ccd_px": core_radius_ccd_px,
            "fit_radius_ccd_px": fit_radius_ccd_px,
            "core_shape": dataset.core_shape,
            "core_side_px": dataset.core_side_px,
            "dark_levels": {pid: float(item.dark) for pid, item in prepared_by_id.items()},
        },
        "pattern_ids": list(dataset.pattern_ids),
        "pairing": {
            "matched": sorted(prepared_by_id),
            "issues": pairing_issues,
            "model_reproduction_max_abs_diff": repro_max,
            "reproduction_tolerance": REPRO_TOL,
        },
        "frozen_model": {
            "defocus_diopter": params.defocus_diopter,
            "gain": params.gain,
            "background": params.background,
            "sigma_ccd_px": params.sigma_ccd_px,
            "phase_sign": params.phase_sign,
            "fft_size": params.fft_size,
            "grid_size": params.grid_size,
            "origin_xy": list(params.origin_xy),
        },
        "baseline": baseline_metrics,
        "fitted_metrics": fitted_metrics,
        "fit": {
            "params": optimum,
            "cost": fit["cost"],
            "baseline_cost": fit["baseline_cost"],
            "grid_params": fit["grid_params"],
            "grid_cost": fit["grid_cost"],
            "at_bounds": fit["at_bounds"],
            "optimizer": fit["optimizer"],
            "affine": fit["affine"],
            "fit_patterns": sorted(fit_ids),
            "holdout_patterns": sorted(item.pattern_id for item in holdout_set),
        },
        "holdout_table": holdout_table,
    }

    if do_sweeps:
        logger.info(f"{dataset.name}: 扫边际曲线与代价图")
        result["marginal_curves"] = marginal_curves(fit_set, factory, dataset, optimum)
        result["cost_maps"] = [
            map_2d(
                fit_set, factory, dataset, optimum, "defocus_delta_diopter", "phase_scale"
            ),
            map_2d(
                fit_set, factory, dataset, optimum,
                "defocus_delta_diopter", "blur_sigma_samples",
            ),
            map_2d(fit_set, factory, dataset, optimum, "phase_scale", "lateral_scale_x"),
        ]
        result["jacobian"] = jacobian_spectrum(fit_set, factory, dataset, optimum)

    if do_joint:
        logger.info(f"{dataset.name}: 联合优化 (相位 + 全局参数)")
        # 「仅相位」: 全局参数冻结在基准值, 只放开相位。
        phase_only = joint_fit(
            fit_set,
            factory,
            dataset,
            start_globals=FROZEN_GLOBALS,
            max_noll=joint_max_noll,
            max_evals=max_joint_evals,
        )
        # 「相位 + 全局」联合。
        joint = joint_fit(
            fit_set,
            factory,
            dataset,
            start_globals=optimum,
            max_noll=joint_max_noll,
            max_evals=max_joint_evals,
        )
        result["joint"] = {
            "globals_only_cost": float(fit["cost"]),
            "phase_only_cost": float(phase_only["cost"]),
            "cost": float(joint["cost"]),
            "globals": joint["globals"],
            "coefficients": joint["coefficients"],
            "coefficient_rms_rad": joint["coefficient_rms_rad"],
            "n_modes": joint["n_modes"],
            "max_noll": joint["max_noll"],
            "optimizer": joint["optimizer"],
            "per_pattern_residual": joint["per_pattern_residual"],
            "phase_only_globals": phase_only["globals"],
        }

    logger.info(f"{dataset.name}: 灰阶材料性")
    result["gray_materiality"] = gray_materiality(fit_set, factory, dataset, optimum)
    result["seconds"] = time.monotonic() - started
    return result, images


def build_verdicts(payload: dict[str, Any]) -> dict[str, str]:
    """把数字翻译成判决 (人话)。"""
    verdicts: dict[str, str] = {}
    for dataset in payload["datasets"]:
        name = dataset["name"]
        fit = dataset["fit"]
        baseline = fit["baseline_cost"]
        fitted = fit["cost"]
        degenerate = baseline >= BIG_COST * 0.5
        if degenerate:
            verdicts[f"{name}: 拟合集形状残差"] = (
                f"冻结模型在拟合核里**没有信号** (代价退化 = {baseline:.3g}); "
                f"拟合后 {fitted:.4f}"
            )
        else:
            reduction = 100.0 * (baseline - fitted) / max(baseline, EPS)
            verdicts[f"{name}: 拟合集形状残差"] = (
                f"冻结 {baseline:.4f} → 拟合 {fitted:.4f} (降 {reduction:.1f} %)"
            )

        holdout_rows = [row for row in dataset["holdout_table"] if row["set"] == "holdout"]
        if holdout_rows:
            hb = sum(row["baseline"]["shape_residual_wide"] for row in holdout_rows) / len(holdout_rows)
            hf = sum(row["fitted"]["shape_residual_wide"] for row in holdout_rows) / len(holdout_rows)
            holdout_gain = 100.0 * (hb - hf) / max(hb, EPS)
            verdicts[f"{name}: 验证集形状残差 (大核)"] = (
                f"冻结 {hb:.4f} → 拟合 {hf:.4f} (降 {holdout_gain:.1f} %)"
            )
        else:
            verdicts[f"{name}: 验证集形状残差 (大核)"] = "无验证集"

        offsets = [item["baseline"]["peak_offset_norm_ccd_px"] for item in dataset["holdout_table"]]
        if offsets:
            verdicts[f"{name}: 冻结模型零级错位"] = (
                f"模型内容峰相对实测零级的偏移 (CCD px): "
                f"均值 {sum(offsets) / len(offsets):.1f}, 最大 {max(offsets):.1f}"
            )

        jac = dataset.get("jacobian")
        if jac:
            verdicts[f"{name}: 参数可辨识性"] = (
                f"目标函数约束得住: {jac['identifiable']}; "
                f"约束不住 (不可辨识): {jac['unidentifiable']}"
            )

        joint = dataset.get("joint")
        if joint:
            drift = {
                key: abs(joint["globals"][key] - fit["params"][key])
                / (GLOBAL_BOUNDS[key][1] - GLOBAL_BOUNDS[key][0])
                for key in GLOBAL_NAMES
            }
            stable = [key for key, value in drift.items() if value <= 0.05]
            verdicts[f"{name}: 联合可辨识性"] = (
                f"仅全局 {joint['globals_only_cost']:.4f} / 仅全频随机相位 {joint['phase_only_cost']:.4f} / "
                f"联合 {joint['cost']:.4f}; 联合后仍然稳定的全局参数: {stable}; "
                f"漂移最大的是 {max(drift, key=lambda k: drift[k])} "
                f"({max(drift.values()):.3f} 个取值区间)"
            )
        gray = dataset.get("gray_materiality")
        if gray:
            delta = gray["delta_cost_vs_best"]
            strongest = max(delta.items(), key=lambda kv: kv[1])
            sens = gray.get("phase_scale_sensitivity", {})
            over = sens.get("over_modulation_1023_over_993", {})
            bound_note = (
                " (⚠ 拟合的 phase_scale 顶在搜索边界上, 所以这些增量是**边界值之间的差**, "
                "不是「灰阶比离焦更重要」的证据)"
                if "phase_scale" in (fit.get("at_bounds") or []) else ""
            )
            verdicts[f"{name}: 灰阶材料性"] = (
                f"1023/993 过调制让远场形状的相关性只掉到 "
                f"{over.get('mean_correlation_vs_frozen_phase_scale', float('nan')):.5f} "
                f"(NRMSE {over.get('mean_nrmse_vs_frozen_phase_scale', float('nan')):.4f}) ⇒ "
                f"**灰阶在远场强度分布上几乎不材料**; "
                f"相对最优的残差增量 {delta.get('gray_1023_over_993', float('nan')):.4g} "
                f"(最强具名缺陷 {strongest[0]}, 增量 {strongest[1]:.4g}){bound_note}"
            )
    return verdicts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="离线分析物理预测模型缺陷 (离焦 / 横向尺度 / 中心 / 灰阶 / 模糊)",
    )
    parser.add_argument(
        "--dataset-dir",
        action="append",
        dest="dataset_dirs",
        default=None,
        help="采集目录 (可重复); 缺省用两个 absolute_fidelity 目录",
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="冻结参数 json")
    parser.add_argument(
        "--out", default=str(REPORT_DIR / "artifacts" / DEFAULT_OUT_DIR_NAME), help="产物目录"
    )
    parser.add_argument("--dry-run", action="store_true", help="合成数据自检 (不碰真实数据)")
    parser.add_argument("--fit-patterns", default=",".join(DEFAULT_FIT_PATTERNS))
    parser.add_argument("--holdout-patterns", default=",".join(DEFAULT_HOLDOUT_PATTERNS))
    parser.add_argument("--core-radius-ccd-px", type=float, default=DEFAULT_CORE_RADIUS_CCD_PX)
    parser.add_argument("--fit-radius-ccd-px", type=float, default=DEFAULT_FIT_RADIUS_CCD_PX)
    parser.add_argument("--max-fit-evals", type=int, default=400)
    parser.add_argument("--max-joint-evals", type=int, default=500)
    parser.add_argument("--joint-max-noll", type=int, default=3)
    parser.add_argument("--no-joint", action="store_true", help="跳过联合可辨识性分析")
    parser.add_argument("--no-sweeps", action="store_true", help="跳过边际曲线与代价图")
    parser.add_argument("--no-figures", action="store_true", help="不出图")
    parser.add_argument(
        "--joint-all",
        action="store_true",
        help="对所有数据集都做联合优化 (默认只做主数据集)",
    )
    return parser.parse_args(argv)


def _provenance_header(text: str, out_dir: Path) -> str:
    """给报告正文插入溯源块 (``report/AGENTS.md`` 的硬性要求)。

    优先用仓库的 ``scripts._common.provenance.insert_header`` (键已在 registry 里时
    它给出规范块); 还没登记时手写同样格式的块 —— 等登记之后, ``sync_report_provenance``
    会把旧块剥掉换成规范块 (``insert_header`` 先 strip 再插入), 不会叠两个。
    """
    try:
        key = (out_dir / "fwdmodel_fit.md").relative_to(ROOT).as_posix()
    except ValueError:
        key = ""
    if key:
        try:
            from scripts._common.provenance import insert_header

            updated = insert_header(text, key)
            if updated != text:
                return updated
        except Exception as exc:  # pragma: no cover - registry 缺失时走手写兜底
            logger.warning(f"provenance 渲染失败, 改用手写块: {exc}")
    try:
        link = Path(os.path.relpath(ROOT / "scripts" / "slm" / "analyze_fwdmodel_fit.py", out_dir))
        script_cell = f"[`scripts/slm/analyze_fwdmodel_fit.py`]({link.as_posix()})"
    except ValueError:
        script_cell = "`scripts/slm/analyze_fwdmodel_fit.py`"
    try:
        command_out = Path(os.path.relpath(out_dir, ROOT)).as_posix()
    except ValueError:
        command_out = out_dir.as_posix()
    block = (
        "<!-- provenance:start -->\n"
        f"**生成脚本**: {script_cell}\n\n"
        "**生成命令**: `.venv/Scripts/python.exe scripts/slm/analyze_fwdmodel_fit.py "
        f"--out {command_out}`\n\n"
        "**运行环境**: 离线 (只读已落盘的采集目录与冻结检查点; 不 import 任何设备驱动)\n\n"
        "<!-- provenance:end -->"
    )
    lines = text.splitlines()
    anchor = next((i for i, ln in enumerate(lines) if ln.startswith("# ")), None)
    if anchor is None:
        return f"{block}\n\n{text}".rstrip("\n") + "\n"
    rest = lines[anchor + 1:]
    while rest and not rest[0].strip():
        rest.pop(0)
    return "\n".join(lines[: anchor + 1] + ["", block, ""] + rest).rstrip("\n") + "\n"


def _write_run_log(out_dir: Path, lines: Iterable[str]) -> Path:
    path = out_dir / "run.log"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"无法 JSON 序列化: {type(value)!r}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    log_lines: list[str] = []

    def log(message: str) -> None:
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        line = f"[{stamp}] {message}"
        log_lines.append(line)
        logger.info(message)

    log("开始: " + " ".join(sys.argv))
    if args.dry_run and not Path(args.checkpoint).exists():
        # dry-run 的语义就是"合成数据自检 (不碰真实数据)": CI 容器没有 data/,
        # 检查点缺失时用合成基线参数 (issue #205 第 132 批修复, 第 139 批重申)。
        params = synthetic_frozen_parameters()
        log("dry-run: 冻结参数存档缺失, 使用合成基线参数")
    else:
        params = load_frozen_parameters(args.checkpoint)
    log(
        f"冻结参数: defocus={params.defocus_diopter:.6g} D, gain={params.gain:.6g}, "
        f"background={params.background:.6g}, sigma_ccd_px={params.sigma_ccd_px:g}, "
        f"phase_sign={params.phase_sign:g}, fft={params.fft_size}, grid={params.grid_size}, "
        f"origin={params.origin_xy}"
    )

    fit_patterns = [p.strip() for p in args.fit_patterns.split(",") if p.strip()]
    holdout_patterns = [p.strip() for p in args.holdout_patterns.split(",") if p.strip()]

    datasets: list[Dataset] = []
    truth: dict[str, float] | None = None
    if args.dry_run:
        truth = {
            "defocus_delta_diopter": 0.006,
            "lateral_scale_x": 1.05,
            "lateral_scale_y": 0.95,
            "centre_dx_px": 8.0,
            "centre_dy_px": -6.0,
            "phase_scale": 1023.0 / 993.0,
            "blur_sigma_samples": 0.8,
        }
        log(f"dry-run: 合成真值 {truth}")
        datasets.append(make_synthetic_dataset(params, truth=truth))
    else:
        dirs = [
            Path(d) for d in (args.dataset_dirs or [str(p) for p in DEFAULT_DATASET_DIRS])
        ]
        for data_dir in dirs:
            if not data_dir.is_dir():
                logger.warning(f"数据目录不存在, 跳过: {data_dir}")
                continue
            datasets.append(load_real_dataset(data_dir))
    if not datasets:
        logger.error("没有可分析的数据")
        _write_run_log(out_dir, log_lines)
        return EXIT_NO_DATA

    payloads: list[dict[str, Any]] = []
    image_sets: list[dict[str, dict[str, np.ndarray]]] = []
    for index, dataset in enumerate(datasets):
        log(f"--- 分析数据集 {dataset.name} ---")
        do_joint = (not args.no_joint) and (args.joint_all or index == 0)
        payload, images = analyse_dataset(
            dataset,
            params,
            fit_patterns=fit_patterns,
            holdout_patterns=holdout_patterns,
            core_radius_ccd_px=args.core_radius_ccd_px,
            fit_radius_ccd_px=args.fit_radius_ccd_px,
            max_fit_evals=args.max_fit_evals,
            max_joint_evals=args.max_joint_evals,
            joint_max_noll=args.joint_max_noll,
            do_joint=do_joint,
            do_sweeps=not args.no_sweeps,
        )
        payloads.append(payload)
        image_sets.append(images)
        log(f"数据集 {dataset.name} 完成, 用时 {payload['seconds']:.1f} s")

    synthetic_check: dict[str, Any] | None = None
    if truth is not None:
        check = check_synthetic_recovery(
            truth,
            payloads[0]["fit"]["params"],
            baseline_cost=payloads[0]["fit"]["baseline_cost"],
            fitted_cost=payloads[0]["fit"]["cost"],
        )
        payloads[0]["synthetic_recovery"] = check
        payloads[0]["synthetic_truth"] = truth
        synthetic_check = check
        for row in check["rows"]:
            log(
                f"dry-run 自检 {row['parameter']}: 真值 {row['truth']:.6g} → 拟合 "
                f"{row['fitted']:.6g} (误差 {row['error']:+.4g}, 容差 {row['tolerance']:g}) "
                f"{'PASS' if row['passed'] else 'FAIL'}"
                f"{' [必须项]' if row['must_pass'] else ' [参考项: 与离焦/模糊/尺度互相退化]'}"
            )
        log(
            f"dry-run 自检: 残差 {payloads[0]['fit']['baseline_cost']:.4f} → "
            f"{payloads[0]['fit']['cost']:.4f} (降 "
            f"{100.0 * check['residual_reduction']:.1f} %)"
        )

    result: dict[str, Any] = {
        "schema": "fwdmodel_fit/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": SCRIPT_REL,
        "offline": {"driver_modules_imported": _leaked_driver_modules()},
        "checkpoint": str(Path(args.checkpoint)),
        "dry_run": bool(args.dry_run),
        "datasets": payloads,
    }
    result["verdicts"] = build_verdicts(result)
    if synthetic_check is not None:
        result["synthetic_recovery_passed"] = bool(synthetic_check["passed"])
        if synthetic_check["passed"]:
            log(
                "dry-run 自检通过: 中心平移在容差内, 且残差有实质下降 "
                "(PSF 族参数的偏差属已知退化, 只作参考)"
            )
        else:
            logger.error("dry-run 自检未通过: 拟合没有把合成真值捞回来")

    # --- 产物 ---
    json_path = out_dir / "fwdmodel_fit.json"
    md_path = out_dir / "fwdmodel_fit.md"
    artifacts = [json_path.name, md_path.name, "run.log", Path(__file__).name]
    if not args.no_figures:
        figure_paths: list[Path] = []
        try:
            figure_paths.append(figure_radial(payloads, out_dir))
            primary = payloads[0]
            if primary.get("marginal_curves"):
                figure_paths.append(figure_sweeps(primary["marginal_curves"], out_dir))
            if primary.get("cost_maps"):
                figure_paths.append(figure_maps(primary["cost_maps"], out_dir))
            figure_paths.append(figure_holdout(primary, image_sets[0], out_dir))
            if primary.get("gray_materiality"):
                figure_paths.append(figure_gray(primary["gray_materiality"], out_dir))
        except Exception as exc:  # 出图失败不应该让整轮分析作废
            logger.warning(f"出图失败 (分析结果仍会写出): {type(exc).__name__}: {exc}")
        for path in figure_paths:
            artifacts.append(path.name)
            if args.dry_run:
                # dry-run 的图是**合成数据**的, 不能盖掉共享 figures 目录里的真数据图。
                continue
            try:
                FIGURES_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, FIGURES_DIR / path.name)
            except Exception as exc:
                logger.warning(f"图复制到 {FIGURES_DIR} 失败: {exc}")

    result["artifacts"] = artifacts
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    md_path.write_text(_provenance_header(render_markdown(result), out_dir), encoding="utf-8")

    # 自身源码副本, 保证报告可离线复现。
    source_copy = out_dir / Path(__file__).name
    shutil.copy2(Path(__file__).resolve(), source_copy)
    log(f"产物已写出: {out_dir}")

    assert_no_drivers()
    log(f"总用时 {time.monotonic() - started:.1f} s")
    _write_run_log(out_dir, log_lines)
    print(json.dumps(result["verdicts"], ensure_ascii=False, indent=2))
    if synthetic_check is not None and not synthetic_check["passed"]:
        return EXIT_SELFCHECK_FAILED
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
