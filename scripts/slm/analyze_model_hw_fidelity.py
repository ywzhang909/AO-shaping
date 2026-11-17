"""离线核对: 可微模型的预测远场与上屏实测远场到底有多接近。

要回答的问题
------------
把可微优化出来的相位真的送到 SLM 上之后, **模型预测的远场**和**实测远场**是不是
几乎一致? 本脚本只读已经落盘的运行目录 (``model_farfield.npy`` +
``measured_model_grid.npy`` + ``run_config.json`` + 历史 CSV), 全程 **不 import 任何
设备驱动**, 收尾断言 ``ao_shaping.drivers`` 不在 ``sys.modules`` 里, 违反就 exit 3,
因此可以在没有任何硬件的机器上跑。

每个运行给出
------------
* (A) 模型网格上的逐像元一致性: 线性 / sqrt / log 强度相关系数, 目标区与圆盘 ROI
  相关, 归一化 RMS 误差 (全网格与 ROI), 质心偏移与二阶矩宽度比, FWHM 比, 包围能量
  半径; 其中线性相关系数要与 ``run_config.json`` 记录的
  ``comparison.best.intensity_correlation`` 对账, 对不上就大声报出来;
* (B) 尺度扫描: 把实测按自身质心重采样 (``scipy.ndimage.zoom`` order=1), 在
  s ∈ [0.40, 2.50] 上找相关系数峰值 ``s*`` —— 若 ``s*`` 顶在区间端点上、且与
  「模型/实测 FWHM 比」相符, 说明失配主要是**几何尺度 (K 标定)**, 而不是形状;
* (C) 每个 ``history_hw.csv`` 行配对 ``history.csv`` 同迭代号的模型损失 / 相位 RMS,
  看「模型损失下降」是不是伴随「实测峰值改善」;
* (D) 零相位参考 (``comparison.flat_reference``) 的相关系数与峰值比;
* (E) 按仓库口径 (≥0.8 高度一致 / 0.4–0.8 中度一致 / <0.4 不一致) 给判定, 并回答
  「几乎一致? 是/否」, 同时点名主导偏差。

跨运行表按日期排序, 覆盖所有能分析的运行, 用来回答「一致性是不是变过、哪个记录在案的
配置差异跟着它一起变」。相关系数在 5–7 个运行之间比大小是**弱证据**, 报告里明确写出。

在本实验系统之外**不做**任何假设: 面板↔相机的轴向口径 (转置 / 镜像) 沿用
``hw_differentiable_phase_opt.py`` 的 8 个候选择优, 择优结果与 ``run_config.json``
里记录的一致时才算对账通过。

用法::

    .venv/Scripts/python.exe scripts/slm/analyze_model_hw_fidelity.py \
        --all-differentiable-opt \
        --run-dir report/slm/differentiable_opt/artifacts/20261023_015046 \
        --out report/slm/differentiable_opt/artifacts/model_hw_fidelity_20261116

产物 (默认输出目录见 ``DEFAULT_OUT_DIR``): ``model_hw_fidelity.json``、
``model_hw_fidelity.md``、``fidelity_pred_vs_meas.png``、``fidelity_scale_scan.png``;
两张图同时复制到 ``report/slm/differentiable_opt/figures/``。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from functools import partial
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

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
from loguru import logger
from scipy.ndimage import zoom as _ndi_zoom

from ao_shaping.utils.image.frame_analysis import measure_spot
from scripts._common.plot_style import setup_cjk_font
from scripts._common import fmt_spec_or_text

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
#: 退出码: 0 正常; 1 没有可分析的运行; 3 离线契约被破坏 (import 了设备驱动)。
EXIT_OK = 0
EXIT_NO_RUNS = 1
EXIT_OFFLINE_VIOLATION = 3

SCRIPT_REL = "scripts/slm/analyze_model_hw_fidelity.py"
DEFAULT_OUT_DIR_NAME = "model_hw_fidelity_20261116"

#: ``--all-differentiable-opt`` 扫描的目录 (data/debug 下的一次性运行产物)。
DIFFERENTIABLE_OPT_DIR = ROOT / "data" / "debug" / "differentiable_opt"
REPORT_DIR = ROOT / "report" / "slm" / "differentiable_opt"
FIGURES_DIR = REPORT_DIR / "figures"

#: 没有把 ``run_config.json`` 落在运行目录里的老运行, 配置可能落在这里
#: (按 ``timestamp_utc`` 字段与运行目录名匹配, 见 :func:`_resolve_config`)。
EXTERNAL_CONFIG_DIRS: tuple[Path, ...] = (
    ROOT / "report" / "shape_trials" / "sim_vs_hw" / "artifacts",
)

#: 参考实现 (只读 import, 绝不修改)。
REFERENCE_SCRIPT = ROOT / "scripts" / "slm" / "hw_differentiable_phase_opt.py"
REFERENCE_MODULE_NAME = "_hw_differentiable_phase_opt_ref"

#: 面板↔相机轴向口径的候选; 与参考实现的 ``_AXIS_CANDIDATES`` 逐字同序。
#: (名字, 是否转置, 是否翻行, 是否翻列)。
AXIS_CANDIDATES: tuple[tuple[str, bool, bool, bool], ...] = (
    ("identity", False, False, False),
    ("transpose", True, False, False),
    ("flip_rows", False, True, False),
    ("flip_cols", False, False, True),
    ("flip_both", False, True, True),
    ("transpose_flip_rows", True, True, False),
    ("transpose_flip_cols", True, False, True),
    ("transpose_flip_both", True, True, True),
)

#: 尺度扫描区间与步长。上限特意放到 4.00 (任务书的区间是 [0.40, 2.50]): 只有越过
#: 2.50 才能把「s* 真的落在 3 附近 (几何 K 标定)」与「s* 只是顶在 2.50 端点上」分开。
#: 报告同时给出全区间 s* 与限制在 2.50 以内的 s*。
SCALE_MIN = 0.40
SCALE_MAX = 4.00
SCALE_MAX_REQUESTED = 2.50
SCALE_STEP = 0.05

#: 判定口径 (与仓库报告一致)。
CORR_HIGH = 0.8
CORR_MID = 0.4
#: 「几乎一致」的附加条件: FWHM 比落在该区间、光斑**峰值**偏移不超过该采样数。
#: 用峰值偏移而不是全网格质心偏移: 后者被 0 级与背景拖住, 量的不是光斑位置。
FWHM_RATIO_TOL = 1.25
PEAK_OFFSET_TOL_SAMPLES = 3.0

EPS = 1e-12
#: 对账容差: 重算的线性相关系数与记录值之差超过它就算「对不上」。
REPRO_TOL = 1e-6

_NUMERIC_HISTORY_KEYS = (
    "iteration",
    "loss",
    "grad_norm",
    "phase_rms_rad",
    "measured_peak",
    "exposure_ms",
)
_ENCIRELE_FRACTIONS = (0.5, 0.8, 0.9)


# ---------------------------------------------------------------------------
# 离线契约
# ---------------------------------------------------------------------------
def _leaked_driver_modules() -> list[str]:
    """``sys.modules`` 里所有 ``ao_shaping.drivers.*`` (应为空)。"""
    return sorted(name for name in sys.modules if name.startswith("ao_shaping.drivers"))


def assert_no_drivers() -> None:
    """离线契约的硬保证: 收尾断言没 import 任何设备驱动。

    与参考实现的 ``assert_no_drivers`` 同一口径, 但**本脚本更严**: 连
    ``ao_shaping.drivers`` 顶包出现都算违约 —— 本脚本没有任何理由碰设备。
    """
    leaked = _leaked_driver_modules()
    if leaked:
        sys.stderr.write(
            "离线契约违约: 分析路径上导入了设备驱动包:\n  " + "\n  ".join(leaked) + "\n"
        )
        sys.stderr.flush()
        raise SystemExit(EXIT_OFFLINE_VIOLATION)
    logger.info(
        "离线契约通过: sys.modules 里没有任何 ao_shaping.drivers.* (未打开任何设备)"
    )


# ---------------------------------------------------------------------------
# 参考实现 (只读 import) 与本地镜像
# ---------------------------------------------------------------------------
def load_reference_module() -> tuple[Any | None, str]:
    """按路径 import 参考实现; 失败时返回 ``(None, 原因)``, 由本地镜像接手。

    关键点: ``sys.modules`` 必须先登记再 ``exec_module`` —— 参考实现里有
    ``@dataclass`` 定义, 数据类在解析字段类型时回查 ``sys.modules``, 不预登记会
    在 ``exec_module`` 阶段炸掉。
    """
    if not REFERENCE_SCRIPT.is_file():
        return None, f"参考实现不存在: {REFERENCE_SCRIPT}"
    spec = importlib.util.spec_from_file_location(REFERENCE_MODULE_NAME, REFERENCE_SCRIPT)
    if spec is None or spec.loader is None:
        return None, "spec_from_file_location 失败"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # 任何失败都退回本地镜像
        sys.modules.pop(spec.name, None)
        return None, f"exec_module 失败: {type(exc).__name__}: {exc}"
    return module, "loaded"


def _peak_xy(arr: np.ndarray) -> tuple[float, float]:
    """峰值 (x, y) 坐标; 退化画面返回 NaN。"""
    if arr.size == 0 or not np.isfinite(arr).any() or float(arr.max()) <= 0:
        return float("nan"), float("nan")
    iy, ix = np.unravel_index(int(np.argmax(arr)), arr.shape)
    return float(ix), float(iy)


def mirror_compare_farfields(model: np.ndarray, measured: np.ndarray) -> dict[str, Any]:
    """本地镜像: 参考实现 ``compare_farfields`` 的同一算式 (L569–612)。

    仅在参考实现 import 失败时使用。线性相关系数用 ``np.corrcoef`` 作用于展平后的
    ``float64`` 强度, FWHM 用 ``measure_spot`` —— 与参考实现逐字一致, 所以本地镜像与
    参考实现对同一对数组必须给出**位级相同**的相关系数。
    """
    mod = np.asarray(model, dtype=np.float64)
    meas = np.asarray(measured, dtype=np.float64)
    mx, my = _peak_xy(mod)
    px, py = _peak_xy(meas)
    model_spot = measure_spot(mod)
    meas_spot = measure_spot(meas)
    a = mod.ravel().astype(np.float64)
    b = meas.ravel().astype(np.float64)
    corr = float(np.corrcoef(a, b)[0, 1]) if (a.std() > 0 and b.std() > 0) else float("nan")
    model_peak = float(mod.max())
    meas_peak = float(meas.max())
    return {
        "model_peak_xy": [mx, my],
        "measured_peak_xy": [px, py],
        "peak_offset_samples": [
            float(px - mx) if np.isfinite(px - mx) else float("nan"),
            float(py - my) if np.isfinite(py - my) else float("nan"),
        ],
        "model_fwhm_px": float(model_spot.fwhm_px),
        "measured_fwhm_px": float(meas_spot.fwhm_px),
        "model_peak": model_peak,
        "measured_peak": meas_peak,
        "peak_ratio_raw": (
            float(meas_peak / model_peak) if model_peak > 0 else float("nan")
        ),
        "intensity_correlation": corr,
    }


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


def nrmse(
    reference: np.ndarray,
    test: np.ndarray,
    *,
    scale_ref: float | None = None,
    scale_test: float | None = None,
) -> float:
    """峰值归一化后的 RMS 误差 ``‖test̂-ref̂‖₂ / ‖ref̂‖₂``。

    为什么必须先按**各自的峰值**归一化: 模型远场是归一化强度 (峰值 ~1e-3), 实测网格
    是 CCD counts (峰值 ~1e2–1e4), 直接相减得到的"误差"只是单位换算的产物 (实测里
    读到过 4.6e5), 没有任何物理含义。归一化之后这个数才是"形状差多少"。

    ``scale_ref``/``scale_test`` 允许传入**全局**峰值, 这样 ROI 上的 NRMSE 与全网格
    的 NRMSE 用的是同一把尺子 (否则 ROI 内的小峰值会把误差人为压小)。缺省用各自
    数组自身的峰值。
    """
    ref = np.asarray(reference, dtype=np.float64).ravel()
    tst = np.asarray(test, dtype=np.float64).ravel()
    if ref.size == 0 or ref.size != tst.size:
        return float("nan")
    sr = float(scale_ref) if scale_ref is not None else float(np.max(ref))
    st = float(scale_test) if scale_test is not None else float(np.max(tst))
    if not (math.isfinite(sr) and math.isfinite(st)) or sr <= 0.0 or st <= 0.0:
        return float("nan")
    ref_hat = ref / sr
    tst_hat = tst / st
    denom = float(np.sqrt(np.mean(ref_hat**2)))
    if not math.isfinite(denom) or denom <= 0.0:
        return float("nan")
    return float(np.sqrt(np.mean((tst_hat - ref_hat) ** 2)) / denom)


def log10_intensity(arr: np.ndarray) -> np.ndarray:
    """log10(I + 1e-12); 只用于相关系数, 不改变数组本身。"""
    return np.log10(np.maximum(np.asarray(arr, dtype=np.float64), EPS))


def sqrt_intensity(arr: np.ndarray) -> np.ndarray:
    """sqrt(max(I, 0))。"""
    return np.sqrt(np.clip(np.asarray(arr, dtype=np.float64), 0.0, None))


def apply_convention(
    arr: np.ndarray, transpose: bool, flip_rows: bool, flip_cols: bool
) -> np.ndarray:
    """按候选口径变换 (顺序与参考实现逐字一致: 转置 → 翻行 → 翻列)。"""
    out = np.asarray(arr, dtype=np.float64)
    if transpose:
        out = out.T
    if flip_rows:
        out = out[::-1, :]
    if flip_cols:
        out = out[:, ::-1]
    return out


def intensity_centroid_rms(arr: np.ndarray) -> tuple[float, float, float]:
    """强度加权质心 ``(cx, cy)`` 与各向同性二阶矩宽度 RMS。

    权重取 ``max(I, 0)`` (散斑旁瓣可以是负的基线扣除值, 负权重会拖歪质心)。
    宽度 = ``sqrt(Σ w·r² / Σ w)``, 其中 ``r`` 是到质心的距离 —— 这是「光斑有多宽」
    的逐像元度量, 与 FWHM 是两种口径, 报告里分开写。
    """
    img = np.asarray(arr, dtype=np.float64)
    w = np.clip(img, 0.0, None)
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0.0:
        return float("nan"), float("nan"), float("nan")
    ys, xs = np.indices(img.shape, dtype=np.float64)
    cx = float((w * xs).sum() / total)
    cy = float((w * ys).sum() / total)
    var = float((w * ((xs - cx) ** 2 + (ys - cy) ** 2)).sum() / total)
    return cx, cy, float(np.sqrt(max(var, 0.0)))


def target_roi_mask(
    shape: tuple[int, int], peak_xy: tuple[float, float], side_px: float
) -> np.ndarray:
    """以实测峰值为中心、边长 ``side_px`` 的目标区掩码。"""
    mask = np.zeros(shape, dtype=bool)
    if not (np.isfinite(peak_xy[0]) and np.isfinite(peak_xy[1])):
        return mask
    half = max(1, int(round(float(side_px) / 2.0)))
    py, px = int(round(peak_xy[1])), int(round(peak_xy[0]))
    y0, y1 = max(0, py - half), min(shape[0], py + half)
    x0, x1 = max(0, px - half), min(shape[1], px + half)
    mask[y0:y1, x0:x1] = True
    return mask


def disc_roi_mask(
    shape: tuple[int, int], peak_xy: tuple[float, float], radius_px: float
) -> np.ndarray:
    """以实测峰值为中心、半径 ``radius_px`` 的圆盘掩码。"""
    mask = np.zeros(shape, dtype=bool)
    if not (np.isfinite(peak_xy[0]) and np.isfinite(peak_xy[1])):
        return mask
    radius = max(1.0, float(radius_px))
    py, px = int(round(peak_xy[1])), int(round(peak_xy[0]))
    half = int(math.ceil(radius))
    y0, y1 = max(0, py - half), min(shape[0], py + half + 1)
    x0, x1 = max(0, px - half), min(shape[1], px + half + 1)
    if y1 <= y0 or x1 <= x0:
        return mask
    local_y, local_x = np.ogrid[y0 - py : y1 - py, x0 - px : x1 - px]
    mask[y0:y1, x0:x1] = (local_x**2 + local_y**2) <= radius**2
    return mask


def encircled_radii(
    arr: np.ndarray, fractions: Sequence[float] = _ENCIRELE_FRACTIONS
) -> dict[str, float]:
    """包围能量半径: 从自身质心起累计能量达到给定比例所需的半径 (远场采样)。

    半径按整数 bin 累计 (``bincount``), 再在相邻 bin 之间线性插值 —— 分辨率好于
    整数采样, 又不必做二维距离排序。非有限值一律返回 NaN。
    """
    img = np.clip(np.asarray(arr, dtype=np.float64), 0.0, None)
    out = {f"r{int(round(f * 100))}": float("nan") for f in fractions}
    total = float(img.sum())
    if not np.isfinite(total) or total <= 0.0:
        return out
    cx, cy, _ = intensity_centroid_rms(img)
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return out
    ys, xs = np.indices(img.shape, dtype=np.float64)
    radii = np.rint(np.hypot(xs - cx, ys - cy)).astype(np.int64)
    counts = np.bincount(radii.ravel(), weights=img.ravel())
    cumulative = np.cumsum(counts) / total
    for frac in fractions:
        idx = int(np.searchsorted(cumulative, frac))
        if idx >= cumulative.size:
            out[f"r{int(round(frac * 100))}"] = float("nan")
            continue
        if idx == 0:
            out[f"r{int(round(frac * 100))}"] = 0.0
            continue
        lo, hi = cumulative[idx - 1], cumulative[idx]
        if hi <= lo:
            out[f"r{int(round(frac * 100))}"] = float(idx)
        else:
            out[f"r{int(round(frac * 100))}"] = float(idx - 1 + (frac - lo) / (hi - lo))
    return out


# ---------------------------------------------------------------------------
# 尺度扫描
# ---------------------------------------------------------------------------
def centre_crop_or_pad(arr: np.ndarray, size: int) -> np.ndarray:
    """把 ``arr`` 居中裁到 / 补零到 ``size×size``。"""
    out = np.zeros((size, size), dtype=np.float64)
    src = np.asarray(arr, dtype=np.float64)
    h, w = src.shape
    n_rows, n_cols = min(h, size), min(w, size)
    src_y0 = (h - n_rows) // 2
    src_x0 = (w - n_cols) // 2
    dst_y0 = (size - n_rows) // 2
    dst_x0 = (size - n_cols) // 2
    out[dst_y0 : dst_y0 + n_rows, dst_x0 : dst_x0 + n_cols] = src[
        src_y0 : src_y0 + n_rows, src_x0 : src_x0 + n_cols
    ]
    return out


def rescale_centred(arr: np.ndarray, scale: float) -> np.ndarray:
    """把画面绕**阵列中心**放大/缩小 ``scale`` 倍, 再裁/补回原尺寸。

    阵列中心就是光轴: 模型远场与实测网格的光斑峰值都落在阵列中心附近 (本仓库的
    ``model_farfield.npy`` 峰值在 ~(512, 512)), 所以这是一个两侧共用的、无自由参数的
    枢轴。**不用**强度加权质心: 实测网格里有一大片弱背景 (二阶矩宽度 ~325 采样, 是
    光斑本身的十几倍), 它把质心拖走几十个采样, 绕质心重采样会把中心光斑整体挪位,
    反而把相关系数从 0.6 打到 0.04 —— 详见 ``scale_scan`` 的实现说明。
    """
    img = np.asarray(arr, dtype=np.float64)
    size = img.shape[0]
    if abs(float(scale) - 1.0) < 1e-12:
        return centre_crop_or_pad(img, size)
    zoomed = _ndi_zoom(
        img, float(scale), order=1, mode="constant", cval=0.0, prefilter=False
    )
    return centre_crop_or_pad(zoomed, size)


def scale_scan(
    model: np.ndarray,
    measured: np.ndarray,
    scales: np.ndarray,
    *,
    fwhm_ratio: float,
) -> dict[str, Any]:
    """在 ``scales`` 上扫「实测缩放后与模型的线性相关系数」。

    两侧都绕**阵列中心** (光轴) 重采样, 不做任何平移对齐, 所以 ``s = 1`` 恰好还原
    未缩放的实测与模型的相关系数 —— 这是尺度失配的**零假设基线**。``s*`` 明显大于 1
    且与「模型/实测 FWHM 比」相符 ⇒ 实测光斑比模型窄, 失配是几何尺度
    (K 标定 / 焦距补偿一类), 不是形状失配。

    为什么枢轴必须是阵列中心而不是强度质心: 实测网格的强度质心被大片弱背景拖到偏离
    光轴几十个采样处 (例如 20261116_114424 的质心 y ≈ 474 而峰值在 512), 若各自绕
    质心重采样, 中心光斑会被整体挪位, s = 1 的相关系数会从 0.60 塌到 0.04, 扫描读数
    失去意义 (这在动手修之前实测过)。

    Returns:
        含 ``scales``/``corr`` 两条等长序列、全区间峰值 ``s_star``、``corr(s*)``、
        ``corr(s = 1)``、未缩放时的全网格相关系数、以及限制在任务书区间
        ``[0.40, 2.50]`` 内的 ``s_star`` 的字典。
    """
    model_arr = np.asarray(model, dtype=np.float64)
    measured_arr = np.asarray(measured, dtype=np.float64)

    scale_values = np.asarray(scales, dtype=np.float64)
    corr: list[float] = []
    for scale in scale_values:
        corr.append(pearson(model_arr, rescale_centred(measured_arr, float(scale))))
    values = np.asarray(corr, dtype=np.float64)
    finite = np.isfinite(values)

    def _peak(mask: np.ndarray) -> tuple[float, float]:
        if not (finite & mask).any():
            return float("nan"), float("nan")
        masked = np.where(finite & mask, values, -np.inf)
        idx = int(np.nanargmax(masked))
        return float(scale_values[idx]), float(values[idx])

    s_star, corr_star = _peak(np.ones_like(finite))
    within = scale_values <= SCALE_MAX_REQUESTED + 1e-9
    s_star_requested, corr_star_requested = _peak(within)
    at_one = int(np.argmin(np.abs(scale_values - 1.0)))
    return {
        "scales": [float(s) for s in scale_values],
        "corr": [float(c) if math.isfinite(c) else float("nan") for c in values],
        "s_star": s_star,
        "corr_at_s_star": corr_star,
        "s_star_within_requested_range": s_star_requested,
        "corr_at_s_star_within_requested_range": corr_star_requested,
        "corr_at_1": float(values[at_one]) if math.isfinite(values[at_one]) else None,
        "corr_full_grid_unaligned": pearson(model_arr, measured_arr),
        "boundary_hit": bool(
            math.isfinite(s_star)
            and (s_star <= scale_values[0] + 1e-9 or s_star >= scale_values[-1] - 1e-9)
        ),
        "fwhm_ratio_implied_scale": float(fwhm_ratio),
        "pivot": "绕阵列中心 (光轴) 缩放实测; 不额外做平移对齐, s=1 即未缩放基线",
    }


def scale_scan_reading(scan: dict[str, Any]) -> str:
    """把尺度扫描结果写成一句话。"""
    s_star = scan.get("s_star")
    corr_star = scan.get("corr_at_s_star")
    corr_one = scan.get("corr_at_1")
    ratio = scan.get("fwhm_ratio_implied_scale")
    if not (isinstance(s_star, float) and math.isfinite(s_star)):
        return "尺度扫描没有有限值, 无法读数。"
    parts = [
        f"最佳尺度 s* = {s_star:.2f} (相关系数 {corr_star:.3f})",
        f"s = 1.00 处 {('%.3f' % corr_one) if corr_one is not None else 'n/a'}",
        "两侧均绕阵列中心 (光轴) 缩放, s = 1 即未缩放基线",
    ]
    if scan.get("boundary_hit"):
        parts.append(
            f"s* 顶在扫描区间端点上 (区间 [{SCALE_MIN:.2f}, {SCALE_MAX:.2f}])"
        )
    if isinstance(ratio, float) and math.isfinite(ratio) and ratio > 0:
        near_ratio = abs(s_star - ratio) <= max(0.1, 0.15 * ratio)
        near_inverse = abs(s_star - 1.0 / ratio) <= max(0.1, 0.15 / ratio)
        near_three = abs(s_star - 3.0) <= 0.35 or abs(s_star - 1.0 / 3.0) <= 0.05
        if near_ratio or near_inverse:
            parts.append(
                f"与「模型/实测 FWHM 比」{ratio:.2f} 相符 ⇒ 失配以几何尺度为主"
                " (K 标定 / 焦距补偿一类), 不是形状"
            )
        elif near_three:
            parts.append(
                f"s* ≈ 3 (或 1/3) 而不等于 FWHM 比 {ratio:.2f} ⇒ 指向几何 K 标定的"
                "整倍/分数关系"
            )
        else:
            parts.append(f"与「模型/实测 FWHM 比」{ratio:.2f} 不一致 ⇒ 不只是尺度问题")
    return "; ".join(parts) + "。"


# ---------------------------------------------------------------------------
# 历史 CSV
# ---------------------------------------------------------------------------
def read_history(path: Path) -> list[dict[str, Any]]:
    """读历史 CSV; 数值列转 float, 其余转 str。

    运行目录里的 ``history*.csv`` 都带 UTF-8 BOM (开头是 ``\\ufeffiteration``),
    所以按 ``utf-8-sig`` 解码并再兜底剥一次 BOM —— 否则首个列名永远对不上
    ``_NUMERIC_HISTORY_KEYS``, ``iteration`` 会退化成字符串, 块 C 的配对全成 None。
    """
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if key is None:
                    continue
                name = key.strip().lstrip("\ufeff")
                text = (value or "").strip()
                if name in _NUMERIC_HISTORY_KEYS:
                    try:
                        row[name] = float(text)
                    except ValueError:
                        row[name] = float("nan")
                else:
                    row[name] = text
            rows.append(row)
    return rows


def pair_history(
    model_rows: Sequence[dict[str, Any]], hw_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """把每个 ``history_hw`` 行按迭代码配对 ``history`` 同迭代码的模型损失 / 相位 RMS。

    注意 (报告里也写): ``history_hw.csv`` 的 ``iteration`` 与 ``history.csv`` 的
    ``iteration`` 只是**同一套标号**, 不是同一个时钟 —— 前者是测量锚定步, 后者是
    纯模型步。这里按标号配对是任务指定的口径; 结论只看**趋势方向**, 不把两列当成
    同一时刻的同一量。
    """
    by_iter: dict[int, dict[str, Any]] = {}
    for row in model_rows:
        it = row.get("iteration")
        if isinstance(it, float) and math.isfinite(it):
            by_iter[int(it)] = row
    paired: list[dict[str, Any]] = []
    for hw in hw_rows:
        it = hw.get("iteration")
        key = int(it) if isinstance(it, float) and math.isfinite(it) else None
        model_row = by_iter.get(key) if key is not None else None
        hw_loss = hw.get("loss")
        model_loss = model_row.get("loss") if model_row else None
        hw_rms = hw.get("phase_rms_rad")
        model_rms = model_row.get("phase_rms_rad") if model_row else None
        paired.append(
            {
                "iteration": key,
                "hw_loss_measured": hw_loss,
                "hw_phase_rms_rad": hw_rms,
                "hw_measured_peak": hw.get("measured_peak"),
                "hw_exposure_ms": hw.get("exposure_ms"),
                "model_loss_same_iter": model_loss,
                "model_phase_rms_same_iter": model_rms,
                "loss_ratio_hw_over_model": (
                    float(hw_loss / model_loss)
                    if isinstance(hw_loss, float)
                    and isinstance(model_loss, float)
                    and model_loss not in (0.0,)
                    and math.isfinite(hw_loss)
                    and math.isfinite(model_loss)
                    else None
                ),
                "phase_rms_diff_hw_minus_model": (
                    float(hw_rms - model_rms)
                    if isinstance(hw_rms, float)
                    and isinstance(model_rms, float)
                    and math.isfinite(hw_rms)
                    and math.isfinite(model_rms)
                    else None
                ),
            }
        )

    # 只用**成功配对**的行: ``history.csv`` 与 ``history_hw.csv`` 的迭代号不是同一个
    # 时钟 (报告 caveat 里写了), 配对取的是两张表迭代码的交集。因此趋势只能就这些
    # 配对行来说 —— 若拿 ``history.csv`` 的首尾去比 ``history_hw.csv`` 的首尾, 两段区间
    # 并不对应, 会跟上面那张配对表自相矛盾 (实测里真出现过)。
    joint = [
        (r["iteration"], float(r["model_loss_same_iter"]), float(r["hw_measured_peak"]))
        for r in paired
        if isinstance(r.get("model_loss_same_iter"), float)
        and math.isfinite(r["model_loss_same_iter"])
        and isinstance(r.get("hw_measured_peak"), float)
        and math.isfinite(r["hw_measured_peak"])
    ]
    statement = "历史数据不足, 无法判断。"
    if len(joint) >= 2:
        it_first, model_first, peak_first = joint[0]
        it_last, model_last, peak_last = joint[-1]
        loss_drop = model_last < model_first
        peak_rise = peak_last > peak_first
        statement = (
            f"配对迭代 {it_first} → {it_last} 区间: "
            f"模型损失 {model_first:.4e} → {model_last:.4e} "
            f"({'下降' if loss_drop else '未下降'}), "
            f"实测峰值 {peak_first:.1f} → {peak_last:.1f} "
            f"({'上升' if peak_rise else '未上升'})。"
        )
        if loss_drop and not peak_rise:
            statement += (
                " ⇒ 模型损失下降**没有**伴随实测峰值改善: 降低模型损失不等于提高"
                "上屏光斑质量。"
            )
        elif loss_drop and peak_rise:
            statement += " ⇒ 模型损失下降与实测峰值改善同向, 但样本只有几步, 不作强结论。"
        elif not loss_drop and peak_rise:
            statement += " ⇒ 模型损失这一步没降, 实测峰值反而升了, 两者不同向, 不作强结论。"
        else:
            statement += " ⇒ 模型损失与实测峰值都没往有利方向走。"
    return {"rows": paired, "statement": statement}


# ---------------------------------------------------------------------------
# 判定
# ---------------------------------------------------------------------------
def verdict_band(corr: float | None) -> str:
    """仓库口径的相关系数区间判定。"""
    if corr is None or not isinstance(corr, float) or not math.isfinite(corr):
        return "无法判定 (相关系数非有限)"
    if corr >= CORR_HIGH:
        return "高度一致 (强度相关系数 ≥ 0.8)"
    if corr >= CORR_MID:
        return "中度一致 (0.4 ≤ 强度相关系数 < 0.8)"
    return "不一致 (强度相关系数 < 0.4)"


def nearly_identical(
    corr: float | None, fwhm_ratio: float | None, peak_offset: float | None
) -> tuple[bool, list[str]]:
    """「几乎一致」的判定: 相关系数高档 + 尺度接近 + 光斑峰值基本重合。"""
    reasons: list[str] = []
    if corr is None or not math.isfinite(corr) or corr < CORR_HIGH:
        reasons.append(
            f"线性相关系数 {corr:.3f} < {CORR_HIGH}"
            if isinstance(corr, float) and math.isfinite(corr)
            else "线性相关系数不可用"
        )
    if (
        fwhm_ratio is None
        or not math.isfinite(fwhm_ratio)
        or not (1.0 / FWHM_RATIO_TOL <= fwhm_ratio <= FWHM_RATIO_TOL)
    ):
        reasons.append(
            f"模型/实测 FWHM 比 {fwhm_ratio:.2f} 超出 [1/{FWHM_RATIO_TOL:.2f}, {FWHM_RATIO_TOL:.2f}]"
            if isinstance(fwhm_ratio, float) and math.isfinite(fwhm_ratio)
            else "FWHM 比不可用"
        )
    if (
        peak_offset is None
        or not math.isfinite(peak_offset)
        or peak_offset > PEAK_OFFSET_TOL_SAMPLES
    ):
        reasons.append(
            f"光斑峰值偏移 {peak_offset:.2f} 个远场采样 > {PEAK_OFFSET_TOL_SAMPLES:.1f}"
            if isinstance(peak_offset, float) and math.isfinite(peak_offset)
            else "峰值偏移不可用"
        )
    return (len(reasons) == 0), reasons


def dominant_discrepancy(
    corr: float | None, fwhm_ratio: float | None, peak_offset: float | None
) -> str:
    """点名主导偏差: 把三项各自超出其容限的倍数拿来比, 取最大的那项。

    用**峰值**偏移而不是全网格质心偏移: 后者被 0 级与背景拖住, 量的是能量分布,
    不是光斑位置 (它单独作为一项指标列在表里)。启发式, 只用来在报告里给出**一个**
    最该先查的方向, 不替代 Judge 判断。
    """
    candidates: list[tuple[str, float]] = []
    if isinstance(fwhm_ratio, float) and math.isfinite(fwhm_ratio) and fwhm_ratio > 0:
        candidates.append(
            (
                f"几何尺度失配 (模型/实测 FWHM 比 {fwhm_ratio:.2f})",
                abs(math.log(fwhm_ratio)) / math.log(FWHM_RATIO_TOL),
            )
        )
    if isinstance(peak_offset, float) and math.isfinite(peak_offset):
        candidates.append(
            (
                f"光斑峰值偏移 ({peak_offset:.2f} 个远场采样)",
                peak_offset / PEAK_OFFSET_TOL_SAMPLES,
            )
        )
    if isinstance(corr, float) and math.isfinite(corr):
        candidates.append(
            (f"整体形状/能量分布 (线性相关系数 {corr:.3f})", (1.0 - corr) / (1.0 - CORR_HIGH))
        )
    if not candidates:
        return "无法判定"
    return max(candidates, key=lambda item: item[1])[0]


# ---------------------------------------------------------------------------
# 运行目录发现与配置解析
# ---------------------------------------------------------------------------
def _has_farfield_pair(run_dir: Path) -> bool:
    return (run_dir / "model_farfield.npy").is_file() and (
        run_dir / "measured_model_grid.npy"
    ).is_file()


def discover_differentiable_opt(root: Path) -> list[Path]:
    """``--all-differentiable-opt``: 扫出同时有模型/实测远场的运行目录。"""
    if not root.is_dir():
        return []
    found = [p for p in sorted(root.iterdir()) if p.is_dir() and _has_farfield_pair(p)]
    return found


def _timestamp_of(config: dict[str, Any], path: Path) -> str | None:
    stamp = config.get("timestamp_utc")
    if isinstance(stamp, str) and stamp:
        return stamp
    name = path.stem
    if name.startswith("run_config_"):
        return name[len("run_config_") :]
    return None


def resolve_config(
    run_dir: Path, overrides: dict[str, Path]
) -> tuple[dict[str, Any] | None, str]:
    """找出运行目录对应的 ``run_config.json``。

    顺序: 命令行 ``--config RUN_DIR=PATH`` 覆盖 → 运行目录自带 → 已知外部目录里按
    ``timestamp_utc`` 与运行目录名匹配 (老运行把配置留在了别处)。
    """
    override = overrides.get(str(run_dir.resolve()))
    candidates: list[tuple[Path, str]] = []
    if override is not None:
        candidates.append((override, "命令行 --config"))
    candidates.append((run_dir / "run_config.json", "运行目录"))
    for external in EXTERNAL_CONFIG_DIRS:
        if not external.is_dir():
            continue
        for path in sorted(external.glob("run_config_*.json")):
            candidates.append((path, f"外部目录 {external.name}"))
    for path, source in candidates:
        if not path.is_file():
            continue
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"读配置失败 {path}: {type(exc).__name__}: {exc}")
            continue
        if source != "运行目录" and source != "命令行 --config":
            stamp = _timestamp_of(config, path)
            if stamp != run_dir.name:
                continue
        return config, f"{source}: {_rel(path)}"
    return None, "未找到 run_config.json"


def recorded_orientation(best: dict[str, Any]) -> tuple[bool, bool, bool] | None:
    """从记录的 ``comparison.best`` 里取轴向口径; 缺任何一个键就返回 None。"""
    keys = ("transpose", "flip_rows", "flip_cols")
    if not all(isinstance(best.get(k), bool) for k in keys):
        return None
    return (bool(best["transpose"]), bool(best["flip_rows"]), bool(best["flip_cols"]))


def focus_compensation_flags(value: Any) -> dict[str, Any]:
    """``focus_compensation_focal_mm`` 的读法。

    屈光度按仓库口径取 ``1000 / focal_mm`` (m⁻¹; 仓库里 ``focal_mm = 1000 / diopter_m1``,
    见 ``scripts/slm/generate_focus_plane_report.py``)。
    ``inf`` 或 |1/f| 极小 (焦距几十米开外) 都意味着记录在案的焦距补偿面板**实质为零**;
    而 ~1600 mm 这种有限焦距对应 ~0.6 m⁻¹, 是**真实**的补偿, 不能算零。
    """
    if isinstance(value, float) and math.isinf(value):
        return {
            "focus_compensation_focal_mm": None,
            "focus_compensation_focal_mm_repr": "inf",
            "focus_compensation_diopter_m1": 0.0,
            "focus_compensation_effective_zero": True,
        }
    if isinstance(value, float) and math.isfinite(value) and value != 0.0:
        diopter = 1000.0 / value
        return {
            "focus_compensation_focal_mm": float(value),
            "focus_compensation_focal_mm_repr": f"{value:.4g}",
            "focus_compensation_diopter_m1": diopter,
            "focus_compensation_effective_zero": abs(diopter) < 0.05,
        }
    if value is None:
        return {
            "focus_compensation_focal_mm": None,
            "focus_compensation_focal_mm_repr": "n/a",
            "focus_compensation_diopter_m1": None,
            "focus_compensation_effective_zero": None,
        }
    return {
        "focus_compensation_focal_mm": float(value),
        "focus_compensation_focal_mm_repr": str(value),
        "focus_compensation_diopter_m1": None,
        "focus_compensation_effective_zero": None,
    }


# ---------------------------------------------------------------------------
# 单个运行的分析
# ---------------------------------------------------------------------------
def _rel(path: Path) -> str:
    """仓库内路径的相对形式; 仓库外的输出目录退回绝对路径 (仅用于日志)。"""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_summary(config: dict[str, Any] | None, path: str) -> dict[str, Any]:
    """跨运行表要的配置列。"""
    if config is None:
        return {
            "source": path,
            "iters": None,
            "lr": None,
            "init_phase": None,
            "gs_iters": None,
            "hw_iters": None,
            "seed": None,
            "target": None,
            "target_side_px": None,
            "exposure_ms_requested": None,
            "m_grid": None,
            "padding": None,
            "n_pupil": None,
            "k_mean": None,
            "kx": None,
            "ky": None,
            "rotation_deg": None,
        }
    params = config.get("params") or {}
    model = config.get("model") or {}
    geometry = config.get("geometry") or {}
    return {
        "source": path,
        "iters": params.get("iters"),
        "lr": params.get("lr"),
        "init_phase": params.get("init_phase") or "random",
        "gs_iters": params.get("gs_iters"),
        "hw_iters": params.get("hw_iters"),
        "seed": params.get("seed"),
        "target": params.get("target"),
        "target_side_px": params.get("target_side_px"),
        "exposure_ms_requested": params.get("exposure_ms"),
        "m_grid": model.get("m_grid"),
        "padding": model.get("padding"),
        "n_pupil": model.get("n_pupil"),
        "k_mean": geometry.get("k_mean"),
        "kx": geometry.get("kx"),
        "ky": geometry.get("ky"),
        "rotation_deg": geometry.get("rotation_deg"),
    }


def _recorded_summary(best: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    """记录在案的对账列 (full / recorded_only 两种运行都有)。"""
    flat = comparison.get("flat_reference") or {}
    return {
        "intensity_correlation": best.get("intensity_correlation"),
        "model_fwhm_px": best.get("model_fwhm_px"),
        "measured_fwhm_px": best.get("measured_fwhm_px"),
        "peak_offset_samples": best.get("peak_offset_samples"),
        "strehl_like_peak_ratio": best.get("strehl_like_peak_ratio"),
        "ccd_peak": best.get("ccd_peak"),
        "ccd_fwhm_px": best.get("ccd_fwhm_px"),
        "exposure_ms": best.get("exposure_ms"),
        "orientation_match": best.get("orientation_match"),
        "verdict": best.get("verdict"),
        "flat_reference_correlation": flat.get("intensity_correlation"),
        "flat_reference_peak_ratio_raw": flat.get("peak_ratio_raw"),
    }


def _fwhm_ratio(metrics: dict[str, Any]) -> float | None:
    model_fwhm = metrics.get("fwhm_model_px")
    meas_fwhm = metrics.get("fwhm_measured_px")
    if (
        isinstance(model_fwhm, float)
        and isinstance(meas_fwhm, float)
        and math.isfinite(model_fwhm)
        and math.isfinite(meas_fwhm)
        and meas_fwhm > 0
    ):
        return float(model_fwhm / meas_fwhm)
    return None


def _metrics_series(
    model: np.ndarray,
    measured: np.ndarray,
    *,
    target_side_px: float,
    core: dict[str, Any],
) -> dict[str, Any]:
    """块 (A): 模型网格上的逐像元一致性。``measured`` 已应用择优轴向口径。"""
    peak_xy = tuple(float(v) for v in core["measured_peak_xy"])
    fwhm_model = float(core["model_fwhm_px"])
    fwhm_measured = float(core["measured_fwhm_px"])
    target_mask = target_roi_mask(model.shape, peak_xy, target_side_px)
    disc_mask = disc_roi_mask(model.shape, peak_xy, fwhm_measured)

    cx_m, cy_m, rms_m = intensity_centroid_rms(model)
    cx_h, cy_h, rms_h = intensity_centroid_rms(measured)
    offset_x = cx_h - cx_m if (math.isfinite(cx_h) and math.isfinite(cx_m)) else float("nan")
    offset_y = cy_h - cy_m if (math.isfinite(cy_h) and math.isfinite(cy_m)) else float("nan")
    offset_mag = (
        float(math.hypot(offset_x, offset_y))
        if math.isfinite(offset_x) and math.isfinite(offset_y)
        else float("nan")
    )
    # 光斑**峰值**偏移 (实测-模型) 的模长 —— 这才是「光斑位置对不对」的度量;
    # 上面的全网格质心偏移被 0 级与背景拖住, 量的是能量分布, 单独列。
    peak_dx, peak_dy = (core["peak_offset_samples"] + [float("nan")])[:2]
    peak_offset_mag = (
        float(math.hypot(float(peak_dx), float(peak_dy)))
        if math.isfinite(float(peak_dx)) and math.isfinite(float(peak_dy))
        else float("nan")
    )
    # NRMSE 的公共尺子: 两个数组各自的全局峰值 (见 :func:`nrmse` 的说明)。
    peak_model = float(np.max(model))
    peak_measured = float(np.max(measured))

    enc_model = encircled_radii(model)
    enc_measured = encircled_radii(measured)
    enc_ratio = {
        key: (
            float(enc_model[key] / enc_measured[key])
            if isinstance(enc_model[key], float)
            and isinstance(enc_measured[key], float)
            and math.isfinite(enc_model[key])
            and math.isfinite(enc_measured[key])
            and enc_measured[key] > 0
            else None
        )
        for key in enc_model
    }

    return {
        "corr_linear": pearson(model, measured),
        "corr_sqrt": pearson(sqrt_intensity(model), sqrt_intensity(measured)),
        "corr_log10": pearson(log10_intensity(model), log10_intensity(measured)),
        "corr_linear_target_roi": (
            pearson(model[target_mask], measured[target_mask]) if target_mask.any() else None
        ),
        "corr_linear_disc_roi": (
            pearson(model[disc_mask], measured[disc_mask]) if disc_mask.any() else None
        ),
        "nrmse_full": nrmse(measured, model, scale_ref=peak_measured, scale_test=peak_model),
        "nrmse_target_roi": (
            nrmse(
                measured[target_mask],
                model[target_mask],
                scale_ref=peak_measured,
                scale_test=peak_model,
            )
            if target_mask.any()
            else None
        ),
        "nrmse_disc_roi": (
            nrmse(
                measured[disc_mask],
                model[disc_mask],
                scale_ref=peak_measured,
                scale_test=peak_model,
            )
            if disc_mask.any()
            else None
        ),
        "nrmse_definition": "峰值归一化后的 ‖test̂-参考̂‖₂/‖参考̂‖₂ (各自除以全局峰值)",
        "grid_centroid_model_xy": [float(cx_m), float(cy_m)],
        "grid_centroid_measured_xy": [float(cx_h), float(cy_h)],
        "grid_centroid_offset_xy": [float(offset_x), float(offset_y)],
        "grid_centroid_offset_samples": offset_mag,
        "peak_offset_samples": list(core["peak_offset_samples"]),
        "peak_offset_magnitude_samples": peak_offset_mag,
        "rms_width_model": float(rms_m),
        "rms_width_measured": float(rms_h),
        "rms_width_ratio_model_over_measured": (
            float(rms_m / rms_h)
            if math.isfinite(rms_m) and math.isfinite(rms_h) and rms_h > 0
            else None
        ),
        "fwhm_model_px": fwhm_model,
        "fwhm_measured_px": fwhm_measured,
        "fwhm_ratio_model_over_measured": _fwhm_ratio(
            {"fwhm_model_px": fwhm_model, "fwhm_measured_px": fwhm_measured}
        ),
        "encircled_model": enc_model,
        "encircled_measured": enc_measured,
        "encircled_ratio_model_over_measured": enc_ratio,
        "roi": {
            "target_side_px": float(target_side_px),
            "target_pixels": int(target_mask.sum()),
            "disc_radius_px": float(fwhm_measured),
            "disc_pixels": int(disc_mask.sum()),
        },
    }


def analyse_full_run(
    run_dir: Path,
    config: dict[str, Any] | None,
    config_source: str,
    scales: np.ndarray,
    *,
    reference: Any | None,
    do_scale_scan: bool,
) -> dict[str, Any]:
    """分析一个「模型/实测远场都在」的运行。"""
    model = np.load(run_dir / "model_farfield.npy")
    measured_raw = np.load(run_dir / "measured_model_grid.npy")
    comparison = (config or {}).get("comparison") or {}
    best = comparison.get("best") or {}
    recorded = _recorded_summary(best, comparison)

    compare = getattr(reference, "compare_farfields", None) if reference else None

    # ---- 轴向口径 ---------------------------------------------------------
    candidate_corr: dict[str, float] = {}
    for name, transpose, flip_rows, flip_cols in AXIS_CANDIDATES:
        variant = apply_convention(measured_raw, transpose, flip_rows, flip_cols)
        candidate_corr[name] = float(mirror_compare_farfields(model, variant)["intensity_correlation"])
    best_name = max(
        candidate_corr,
        key=lambda key: candidate_corr[key] if math.isfinite(candidate_corr[key]) else -math.inf,
    )
    recorded_flags = recorded_orientation(best)
    chosen_flags: tuple[bool, bool, bool] | None = recorded_flags
    chosen_name = next(
        (n for n, t, fr, fc in AXIS_CANDIDATES if (t, fr, fc) == recorded_flags),
        None,
    )
    if chosen_flags is None:
        chosen_flags = next(
            (t, fr, fc) for n, t, fr, fc in AXIS_CANDIDATES if n == best_name
        )
        chosen_name = best_name
    measured = apply_convention(measured_raw, *chosen_flags)

    # ---- 对账 -------------------------------------------------------------
    core_mirror = mirror_compare_farfields(model, measured)
    core = core_mirror
    engine_note = "local_mirror"
    if callable(compare):
        core_ref = compare(model, measured, transpose_measured=False)
        core = core_ref
        engine_note = "reference_module"
        delta_engine = abs(
            float(core_ref["intensity_correlation"])
            - float(core_mirror["intensity_correlation"])
        )
        if delta_engine > REPRO_TOL:
            logger.warning(
                f"[{run_dir.name}] 参考实现与本地镜像的相关系数不一致 "
                f"(Δ={delta_engine:.3e}), 以参考实现为准"
            )
    repro_computed = float(core["intensity_correlation"])
    repro_recorded = recorded.get("intensity_correlation")
    repro_delta = (
        repro_computed - float(repro_recorded)
        if isinstance(repro_recorded, (int, float))
        else None
    )
    repro_ok = repro_delta is not None and abs(repro_delta) <= REPRO_TOL
    if repro_delta is not None and not repro_ok:
        logger.warning(
            f"[{run_dir.name}] 对账失败: 记录 {float(repro_recorded):.6f} vs 重算 "
            f"{repro_computed:.6f} (Δ={repro_delta:+.3e}); 记录的轴向口径 "
            f"{chosen_name} 不能复现, 8 个候选里的最佳是 {best_name} "
            f"({candidate_corr[best_name]:.6f})"
        )

    metrics = _metrics_series(
        model,
        measured,
        target_side_px=float((config or {}).get("params", {}).get("target_side_px") or 60.0),
        core=core,
    )
    ratio = metrics["fwhm_ratio_model_over_measured"]

    scan = {"enabled": False}
    if do_scale_scan:
        scan = scale_scan(model, measured, scales, fwhm_ratio=ratio if ratio else float("nan"))
        scan["enabled"] = True
        scan["reading"] = scale_scan_reading(scan)

    history = read_history(run_dir / "history.csv")
    history_hw = read_history(run_dir / "history_hw.csv")
    pairing = pair_history(history, history_hw)
    pairing["model_rows"] = len(history)
    pairing["hw_rows"] = len(history_hw)

    verdict = {
        "band": verdict_band(metrics["corr_linear"]),
        "nearly_identical": None,
        "nearly_identical_reasons": [],
        "dominant_discrepancy": dominant_discrepancy(
            metrics["corr_linear"], ratio, metrics["peak_offset_magnitude_samples"]
        ),
    }
    flag, reasons = nearly_identical(
        metrics["corr_linear"], ratio, metrics["peak_offset_magnitude_samples"]
    )
    verdict["nearly_identical"] = flag
    verdict["nearly_identical_reasons"] = reasons
    if callable(getattr(reference, "_verdict_text", None)) and best:
        verdict["repo_verdict_text"] = reference._verdict_text({**best, **core})

    flat = {
        "intensity_correlation": recorded.get("flat_reference_correlation"),
        "peak_ratio_raw": recorded.get("flat_reference_peak_ratio_raw"),
        "source": "run_config.json: comparison.flat_reference (零相位参考的实测落盘值)",
    }

    return {
        "run_id": run_dir.name,
        "run_dir": run_dir.relative_to(ROOT).as_posix(),
        "mode": "full",
        "config_source": config_source,
        "config": _config_summary(config, config_source),
        "focus": focus_compensation_flags(
            (config or {}).get("focus_compensation_focal_mm")
            if config
            else None
        ),
        "recorded": recorded,
        "engine": engine_note,
        "orientation": {
            "used": chosen_name,
            "recorded": recorded.get("orientation_match"),
            "searched_best": best_name,
            "candidate_correlations": candidate_corr,
        },
        "reproduction": {
            "recorded_corr": repro_recorded,
            "recomputed_corr": repro_computed,
            "delta": repro_delta,
            "within_tolerance": repro_ok,
            "tolerance": REPRO_TOL,
        },
        "metrics": metrics,
        "scale_scan": scan,
        "history_pairing": pairing,
        "flat_reference": flat,
        "verdict": verdict,
        "hashes": {
            "model_farfield_sha256": _sha256(run_dir / "model_farfield.npy")[:16],
            "measured_model_grid_sha256": _sha256(run_dir / "measured_model_grid.npy")[:16],
        },
        "grid_shape": list(np.asarray(model).shape),
    }


def analyse_recorded_only_run(
    run_dir: Path, config: dict[str, Any], config_source: str
) -> dict[str, Any]:
    """没有原始网格、只有记录值的运行 (2026-10-22 等)。"""
    comparison = config.get("comparison") or {}
    best = comparison.get("best") or {}
    recorded = _recorded_summary(best, comparison)
    model_fwhm = recorded.get("model_fwhm_px")
    meas_fwhm = recorded.get("measured_fwhm_px")
    ratio = (
        float(model_fwhm / meas_fwhm)
        if isinstance(model_fwhm, float)
        and isinstance(meas_fwhm, float)
        and math.isfinite(model_fwhm)
        and math.isfinite(meas_fwhm)
        and meas_fwhm > 0
        else None
    )
    offset = recorded.get("peak_offset_samples") or [None, None]
    offset_mag = (
        float(math.hypot(float(offset[0]), float(offset[1])))
        if len(offset) == 2
        and all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in offset)
        else None
    )
    corr = recorded.get("intensity_correlation")
    return {
        "run_id": run_dir.name,
        "run_dir": run_dir.relative_to(ROOT).as_posix(),
        "mode": "recorded_only",
        "config_source": config_source,
        "config": _config_summary(config, config_source),
        "focus": focus_compensation_flags(config.get("focus_compensation_focal_mm")),
        "recorded": recorded,
        "metrics": {
            "corr_linear": corr,
            "fwhm_model_px": model_fwhm,
            "fwhm_measured_px": meas_fwhm,
            "fwhm_ratio_model_over_measured": ratio,
            "peak_offset_samples": list(offset),
            "peak_offset_magnitude_samples": offset_mag,
        },
        "flat_reference": {
            "intensity_correlation": recorded.get("flat_reference_correlation"),
            "peak_ratio_raw": recorded.get("flat_reference_peak_ratio_raw"),
            "source": "run_config.json: comparison.flat_reference",
        },
        "verdict": {
            "band": verdict_band(corr if isinstance(corr, float) else None),
            "nearly_identical": False,
            "nearly_identical_reasons": ["没有原始网格, 只能读记录值"],
            "dominant_discrepancy": dominant_discrepancy(
                corr if isinstance(corr, float) else None, ratio, offset_mag
            ),
        },
        "note": "运行目录里没有 model_farfield.npy / measured_model_grid.npy, 只读记录值。",
    }


def analyse_run(
    run_dir: Path,
    overrides: dict[str, Path],
    scales: np.ndarray,
    *,
    reference: Any | None,
    do_scale_scan: bool,
) -> dict[str, Any] | None:
    """派发: 有网格走全量分析, 没网格但配置里有 ``comparison.best`` 就走记录值。"""
    config, source = resolve_config(run_dir, overrides)
    if _has_farfield_pair(run_dir):
        return analyse_full_run(
            run_dir, config, source, scales, reference=reference, do_scale_scan=do_scale_scan
        )
    if config is not None and (config.get("comparison") or {}).get("best"):
        return analyse_recorded_only_run(run_dir, config, source)
    logger.warning(f"[{run_dir.name}] 既没有远场网格也没有 comparison.best, 跳过")
    return None


# ---------------------------------------------------------------------------
# 跨运行表与趋势
# ---------------------------------------------------------------------------
def _sort_key(run: dict[str, Any]) -> tuple[str, str]:
    """按日期排序 (运行目录名里的日期前缀), 同日期按运行 id。"""
    name = run["run_id"]
    for token in name.split("_"):
        if len(token) >= 8 and token[:8].isdigit():
            return token[:8], name
    return "99999999", name


def cross_run_rows(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """跨运行表: 一行一个运行, 列见报告说明。"""
    rows: list[dict[str, Any]] = []
    for run in sorted(runs, key=_sort_key):
        config = run["config"]
        metrics = run["metrics"]
        recorded = run["recorded"]
        scan = run.get("scale_scan") or {}
        rows.append(
            {
                "run_id": run["run_id"],
                "date": _sort_key(run)[0],
                "mode": run["mode"],
                "iters": config.get("iters"),
                "lr": config.get("lr"),
                "init_phase": config.get("init_phase"),
                "gs_iters": config.get("gs_iters"),
                "hw_iters": config.get("hw_iters"),
                "seed": config.get("seed"),
                "focus_compensation_focal_mm": run["focus"].get(
                    "focus_compensation_focal_mm_repr"
                ),
                "focus_compensation_effective_zero": run["focus"].get(
                    "focus_compensation_effective_zero"
                ),
                "k_mean": config.get("k_mean"),
                "rotation_deg": config.get("rotation_deg"),
                "exposure_ms_optimized": (
                    recorded.get("exposure_ms", {}) or {}
                ).get("optimized")
                if isinstance(recorded.get("exposure_ms"), dict)
                else recorded.get("exposure_ms"),
                "corr_linear": metrics.get("corr_linear"),
                "corr_sqrt": metrics.get("corr_sqrt"),
                "corr_log10": metrics.get("corr_log10"),
                "corr_linear_target_roi": metrics.get("corr_linear_target_roi"),
                "corr_linear_disc_roi": metrics.get("corr_linear_disc_roi"),
                "nrmse_full": metrics.get("nrmse_full"),
                "fwhm_ratio_model_over_measured": metrics.get(
                    "fwhm_ratio_model_over_measured"
                ),
                "grid_centroid_offset_samples": metrics.get("grid_centroid_offset_samples"),
                "peak_offset_samples": metrics.get("peak_offset_samples"),
                "peak_offset_magnitude_samples": metrics.get("peak_offset_magnitude_samples"),
                "s_star": scan.get("s_star"),
                "corr_at_s_star": scan.get("corr_at_s_star"),
                "s_star_within_requested_range": scan.get("s_star_within_requested_range"),
                "corr_at_1": scan.get("corr_at_1"),
                "corr_full_grid_unaligned": scan.get("corr_full_grid_unaligned"),
                "ccd_peak": recorded.get("ccd_peak"),
                "ccd_fwhm_px": recorded.get("ccd_fwhm_px"),
                "flat_reference_correlation": recorded.get("flat_reference_correlation"),
                "verdict_band": run["verdict"]["band"],
                "nearly_identical": run["verdict"].get("nearly_identical"),
                "dominant_discrepancy": run["verdict"].get("dominant_discrepancy"),
                "hashes": run.get("hashes"),
                "duplicate_of": None,
            }
        )
    by_measured: dict[str, str] = {}
    for row in rows:
        hashes = row.get("hashes") or {}
        key = hashes.get("measured_model_grid_sha256")
        if key is None:
            continue
        if key in by_measured:
            row["duplicate_of"] = by_measured[key]
        else:
            by_measured[key] = row["run_id"]
    return rows


def trend_paragraph(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """回答「一致性在 2026-10-23 与 2026-11-16 之间变过吗、哪个配置差异跟着变」。

    5–7 个运行之间比相关系数是**弱证据**; 这里只报事实与候选差异, 并明确写出哪些候选
    用现有数据**分不开**。
    """
    def _by_date(date: str) -> list[dict[str, Any]]:
        return [r for r in rows if r["date"] == date]

    old = _by_date("20261023")
    new = _by_date("20261116")
    old_corr = [r["corr_linear"] for r in old if isinstance(r["corr_linear"], float)]
    new_corr = [r["corr_linear"] for r in new if isinstance(r["corr_linear"], float)]

    def _fmt(values: Sequence[float]) -> str:
        return "[" + ", ".join(f"{v:.3f}" for v in values) + "]" if values else "n/a"

    def _mean(values: Sequence[float]) -> float | None:
        return float(sum(values) / len(values)) if values else None

    old_ratio = [
        r["fwhm_ratio_model_over_measured"]
        for r in old
        if isinstance(r["fwhm_ratio_model_over_measured"], float)
    ]
    new_ratio = [
        r["fwhm_ratio_model_over_measured"]
        for r in new
        if isinstance(r["fwhm_ratio_model_over_measured"], float)
    ]
    old_focus = sorted({r["focus_compensation_focal_mm"] for r in old})
    new_focus = sorted({r["focus_compensation_focal_mm"] for r in new}, key=str)
    old_iters = sorted({r["iters"] for r in old if r["iters"] is not None})
    new_iters = sorted({r["iters"] for r in new if r["iters"] is not None})
    old_gs = sorted({r["gs_iters"] for r in old if r["gs_iters"] is not None})
    new_gs = sorted({r["gs_iters"] for r in new if r["gs_iters"] is not None})
    old_k = [r["k_mean"] for r in old if isinstance(r["k_mean"], float)]
    new_k = [r["k_mean"] for r in new if isinstance(r["k_mean"], float)]
    old_exp = sorted(
        {r["exposure_ms_optimized"] for r in old if isinstance(r["exposure_ms_optimized"], float)}
    )
    new_exp = sorted(
        {r["exposure_ms_optimized"] for r in new if isinstance(r["exposure_ms_optimized"], float)}
    )

    numbers = {
        "corr_20261023": old_corr,
        "corr_20261023_mean": _mean(old_corr),
        "corr_20261116": new_corr,
        "corr_20261116_mean": _mean(new_corr),
        "fwhm_ratio_20261023": old_ratio,
        "fwhm_ratio_20261116": new_ratio,
        "focus_compensation_20261023": old_focus,
        "focus_compensation_20261116": new_focus,
        "iters_20261023": old_iters,
        "iters_20261116": new_iters,
        "gs_iters_20261023": old_gs,
        "gs_iters_20261116": new_gs,
        "k_mean_20261023": old_k,
        "k_mean_20261116": new_k,
        "exposure_ms_optimized_20261023": old_exp,
        "exposure_ms_optimized_20261116": new_exp,
    }

    statement = "样本不足, 无法比较两个日期。"
    if old_corr and new_corr:
        dh = _mean(old_corr) or 0.0
        dn = _mean(new_corr) or 0.0
        statement = (
            f"变了。2026-10-23 的运行相关系数 {_fmt(old_corr)} (均值 {dh:.3f}), "
            f"2026-11-16 的运行 {_fmt(new_corr)} (均值 {dn:.3f}), 降幅约 "
            f"{abs(dh - dn):.3f}; 模型/实测 FWHM 比同时从 {_fmt(old_ratio)} 变成 "
            f"{_fmt(new_ratio)} —— 模型远场几乎没变 (同一 GS 初值, 模型 FWHM 都是 ~31 "
            f"个采样), 变的是实测在模型网格上的尺度 (实测 FWHM 从 ~19 掉到 ~10 个采样), "
            f"即几何尺度失配在 11-16 明显变大。记录在案的配置差异里, "
            f"**焦距补偿**跟着一起变: 10-23 是有限焦距 ({old_focus}), 11-16 是 "
            f"{new_focus} (inf / 极大焦距 ⇒ 补偿面板实质为零); 同时 "
            f"iters {old_iters}→{new_iters}、gs_iters {old_gs}→{new_gs}、"
            f"k_mean {_fmt(old_k)}→{_fmt(new_k)}、优化后曝光 {_fmt(old_exp)}→{_fmt(new_exp)} ms "
            f"也都变了。"
        )
        statement += (
            " 这些候选**分不开**: 只有 5–7 个运行、且它们几乎同时变了 (焦距补偿、iters、"
            "gs_iters、k_mean、曝光), 没有哪个运行单独只改了其中一项, 所以现有数据只能"
            "指出「焦距补偿从有限变成实质为零」是与这次变化最同向的记录差异, 不能证明它"
            "是原因; 6 个样本的相关系数比较本身也是弱证据。"
        )
    return {"statement": statement, **numbers}


# ---------------------------------------------------------------------------
# 图形
# ---------------------------------------------------------------------------
def _log_display(arr: np.ndarray) -> np.ndarray:
    """对数显示: log10(I), 下限取峰值下 4 个数量级。"""
    a = np.asarray(arr, dtype=np.float64)
    peak = float(a.max()) if a.size else 1.0
    floor = max(peak * 1e-4, EPS)
    return np.log10(np.maximum(a, floor))


def _norm01(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    peak = float(a.max()) if a.size else 0.0
    return a / peak if peak > 0 else a


def figure_pred_vs_meas(runs: Sequence[dict[str, Any]], path: Path) -> str | None:
    """每行一个运行: 模型远场 (log) / 实测远场 (log) / 峰值归一化差异。"""
    full = [r for r in runs if r["mode"] == "full"]
    if not full:
        return None
    setup_cjk_font()
    n_rows = len(full)
    fig, axes = plt.subplots(n_rows, 3, figsize=(11.0, 3.1 * n_rows), squeeze=False)
    for i, run in enumerate(full):
        model = np.load(ROOT / run["run_dir"] / "model_farfield.npy")
        measured_raw = np.load(ROOT / run["run_dir"] / "measured_model_grid.npy")
        flags = next(
            (t, fr, fc)
            for n, t, fr, fc in AXIS_CANDIDATES
            if n == run["orientation"]["used"]
        )
        measured = apply_convention(measured_raw, *flags)
        diff = np.abs(_norm01(model) - _norm01(measured))
        panels = (
            (_log_display(model), "inferno", "模型远场 log10(I)"),
            (_log_display(measured), "inferno", "实测远场 log10(I)"),
            (diff, "magma", "‖峰值归一化之差‖"),
        )
        for j, (data, cmap, title) in enumerate(panels):
            ax = axes[i][j]
            if j < 2:
                vmax = float(np.percentile(data, 99.5))
                vmin = float(data.min())
            else:
                vmax = float(max(np.percentile(data, 99.5), 1e-6))
                vmin = 0.0
            im = ax.imshow(data, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)
            if i == 0:
                ax.set_title(title, fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046)
        corr = run["metrics"]["corr_linear"]
        axes[i][0].set_ylabel(
            f"{run['run_id']}\ncorr={corr:.3f}", fontsize=8
        )
    fig.suptitle("模型预测远场 vs 上屏实测远场 (逐运行)", fontsize=12)
    fig.tight_layout(rect=(0.045, 0.0, 1.0, 0.985))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path.as_posix()


def figure_scale_scan(runs: Sequence[dict[str, Any]], path: Path) -> str | None:
    """尺度扫描: 每个运行一条 corr(s) 曲线, 标出 s* 与 s = 1。"""
    scanned = [r for r in runs if (r.get("scale_scan") or {}).get("enabled")]
    if not scanned:
        return None
    setup_cjk_font()
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    for run in scanned:
        scan = run["scale_scan"]
        ax.plot(scan["scales"], scan["corr"], label=f"{run['run_id']}", linewidth=1.2)
        if isinstance(scan.get("s_star"), float) and math.isfinite(scan["s_star"]):
            ax.plot([scan["s_star"]], [scan["corr_at_s_star"]], marker="o", markersize=4)
    ax.axvline(1.0, color="0.4", linestyle="--", linewidth=1.0, label="s = 1")
    ax.axvline(3.0, color="tab:red", linestyle=":", linewidth=1.0, label="s = 3 (几何 K 标定候选)")
    ax.axvline(
        SCALE_MAX_REQUESTED,
        color="0.6",
        linestyle="-.",
        linewidth=0.8,
        label=f"任务书上限 {SCALE_MAX_REQUESTED:.2f}",
    )
    ax.set_xlabel("实测重采样尺度 s (绕阵列中心 / 光轴)")
    ax.set_ylabel("模型 vs 实测 线性相关系数")
    ax.set_title("尺度扫描: 相关系数峰值 s* 指向几何 K 标定")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.as_posix()


# ---------------------------------------------------------------------------
# 清理与写盘
# ---------------------------------------------------------------------------
def json_safe(value: Any) -> Any:
    """把 NumPy 标量 / 非有限浮点 / Path 变成可序列化的值 (非有限 → null)。"""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    return value


# R-25 反漂移 (issue #205 第 132 批): 本地 _fmt 收口为规范 formatter 的直接别名
# (66 输入电池逐字节全等; 守卫只查顶层 def, 别名不重复实现)。
_fmt = fmt_spec_or_text


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def write_markdown(
    payload: dict[str, Any], runs: Sequence[dict[str, Any]], path: Path
) -> None:
    """写人读的报告。"""
    lines: list[str] = []
    lines.append("# 模型预测 vs 上屏实测: 离线一致性核对")
    lines.append("")
    lines.append(f"生成时间 (UTC): {payload['generated_utc']}")
    lines.append(f"脚本: `{SCRIPT_REL}` (纯离线: 未 import 任何设备驱动)")
    lines.append("")
    lines.append("## 0. 范围与方法")
    lines.append("")
    lines.append(
        "* 数据来源: 已落盘的运行目录 (`model_farfield.npy` + `measured_model_grid.npy` "
        "+ `run_config.json` + `history*.csv`), 全程离线, 未占用 SLM / CCD / 激光。"
    )
    lines.append(
        "* 轴向口径 (转置 / 镜像) 不假设, 按 8 个候选择优, 并与 `run_config.json` 里"
        "记录的口径对账; 线性相关系数要与记录的 `comparison.best.intensity_correlation` "
        f"对到 {REPRO_TOL:g} 以内。"
    )
    lines.append(
        "* 尺度扫描: 把实测绕**阵列中心 (光轴)** 用 `scipy.ndimage.zoom` (order=1) 重采样, "
        f"s ∈ [{SCALE_MIN:.2f}, {SCALE_MAX:.2f}], 步长 {SCALE_STEP:.2f}; 不做任何平移对齐, "
        "所以 s = 1 就是未缩放基线 (`corr(s=1)` 恒等于全网格相关系数)。不用强度质心做枢轴: "
        "实测网格的弱背景会把质心拖偏几十个采样, 绕质心重采样会把中心光斑整体挪位。"
        "区间上限特意超过任务书的 "
        f"{SCALE_MAX_REQUESTED:.2f}: 只有越过它, 才能把「s* 真落在 3 附近 (几何 K 标定)」"
        "与「s* 只是顶在端点上」分开; 报告同时给出限制在 "
        f"{SCALE_MAX_REQUESTED:.2f} 以内的 s*。"
    )
    lines.append(
        "* 判定口径 (与仓库一致): ≥0.8 高度一致 / 0.4–0.8 中度一致 / <0.4 不一致;"
        f"「几乎一致」再要求 FWHM 比 ∈ [1/{FWHM_RATIO_TOL:.2f}, {FWHM_RATIO_TOL:.2f}] "
        f"且**光斑峰值**偏移 ≤ {PEAK_OFFSET_TOL_SAMPLES:.1f} 个远场采样。"
    )
    lines.append(
        "* 归一化 RMS 误差 (NRMSE) 先把模型(归一化强度)与实测(counts)各自除以**全局峰值**, "
        "再算 `‖test̂-参考̂‖₂/‖参考̂‖₂` —— 不这样做, 单位换算本身就会把误差顶到 1e5 量级。"
    )
    lines.append("")
    engine = payload["reference_module"]
    lines.append(
        f"参考实现: {'已按路径 import' if engine['loaded'] else '未 import (' + engine['reason'] + ')'}"
        f" ⇒ 相关系数引擎: {payload['engine_used']}。"
    )
    lines.append("")

    # ---- 逐运行 -----------------------------------------------------------
    lines.append("## 1. 逐运行一致性 (块 A)")
    lines.append("")
    for run in runs:
        if run["mode"] != "full":
            continue
        metrics = run["metrics"]
        repro = run["reproduction"]
        lines.append(f"### {run['run_id']} ({run['run_dir']})")
        lines.append("")
        lines.append(
            f"* 配置: iters={run['config']['iters']}, lr={run['config']['lr']}, "
            f"init={run['config']['init_phase']}, gs_iters={run['config']['gs_iters']}, "
            f"hw_iters={run['config']['hw_iters']}, 焦距补偿="
            f"{run['focus']['focus_compensation_focal_mm_repr']} mm"
            f" (实质为零={run['focus']['focus_compensation_effective_zero']})"
        )
        lines.append(
            f"* 对账: 记录 {_fmt(repro['recorded_corr'], '.6f')} vs 重算 "
            f"{_fmt(repro['recomputed_corr'], '.6f')} "
            f"(Δ={_fmt(repro['delta'], '+.3e')}), "
            f"{'通过' if repro['within_tolerance'] else '**不通过, 已大声记录**'}; "
            f"口径 记录={run['orientation']['recorded']} 使用={run['orientation']['used']} "
            f"候选最佳={run['orientation']['searched_best']}"
        )
        lines.append("")
        rows = [
            ["线性相关系数 (全网格)", _fmt(metrics["corr_linear"], ".4f")],
            ["sqrt 强度相关系数", _fmt(metrics["corr_sqrt"], ".4f")],
            ["log10 强度相关系数", _fmt(metrics["corr_log10"], ".4f")],
            [
                "目标区相关系数 (边长 "
                f"{_fmt(metrics['roi']['target_side_px'], '.0f')} 采样)",
                _fmt(metrics["corr_linear_target_roi"], ".4f"),
            ],
            [
                "圆盘 ROI 相关系数 (半径 "
                f"{_fmt(metrics['roi']['disc_radius_px'], '.2f')} 采样)",
                _fmt(metrics["corr_linear_disc_roi"], ".4f"),
            ],
            ["归一化 RMS 误差 (全网格)", _fmt(metrics["nrmse_full"], ".4f")],
            ["归一化 RMS 误差 (目标区)", _fmt(metrics["nrmse_target_roi"], ".4f")],
            ["归一化 RMS 误差 (圆盘 ROI)", _fmt(metrics["nrmse_disc_roi"], ".4f")],
            [
                "全网格强度质心偏移 (实测-模型) / 模长",
                f"({_fmt(metrics['grid_centroid_offset_xy'][0], '.2f')}, "
                f"{_fmt(metrics['grid_centroid_offset_xy'][1], '.2f')}) / "
                f"{_fmt(metrics['grid_centroid_offset_samples'], '.2f')} 采样",
            ],
            [
                "峰值偏移 (实测-模型)",
                f"({_fmt((metrics['peak_offset_samples'] or [None])[0], '.2f')}, "
                f"{_fmt((metrics['peak_offset_samples'] or [None, None])[1], '.2f')}) / "
                f"模长 {_fmt(metrics['peak_offset_magnitude_samples'], '.2f')} 采样",
            ],
            [
                "二阶矩宽度 (模型 / 实测 / 比)",
                f"{_fmt(metrics['rms_width_model'], '.3f')} / "
                f"{_fmt(metrics['rms_width_measured'], '.3f')} / "
                f"{_fmt(metrics['rms_width_ratio_model_over_measured'], '.3f')}",
            ],
            [
                "FWHM (模型 / 实测 / 比)",
                f"{_fmt(metrics['fwhm_model_px'], '.3f')} / "
                f"{_fmt(metrics['fwhm_measured_px'], '.3f')} / "
                f"{_fmt(metrics['fwhm_ratio_model_over_measured'], '.3f')}",
            ],
            [
                "包围能量半径 50/80/90% (模型)",
                " / ".join(
                    _fmt(metrics["encircled_model"][k], ".2f")
                    for k in ("r50", "r80", "r90")
                ),
            ],
            [
                "包围能量半径 50/80/90% (实测)",
                " / ".join(
                    _fmt(metrics["encircled_measured"][k], ".2f")
                    for k in ("r50", "r80", "r90")
                ),
            ],
            [
                "包围能量半径比 (模型/实测)",
                " / ".join(
                    _fmt(metrics["encircled_ratio_model_over_measured"][k], ".3f")
                    for k in ("r50", "r80", "r90")
                ),
            ],
        ]
        lines.extend(_table(["指标", "值"], rows))
        lines.append("")

    # ---- 尺度扫描 ---------------------------------------------------------
    lines.append("## 2. 尺度扫描 (块 B)")
    lines.append("")
    for run in runs:
        scan = run.get("scale_scan") or {}
        if not scan.get("enabled"):
            continue
        lines.append(f"* **{run['run_id']}**: {scan['reading']}")
        lines.append(
            f"  - 全区间 s* = {_fmt(scan.get('s_star'), '.2f')} "
            f"(corr {_fmt(scan.get('corr_at_s_star'), '.3f')}); "
            f"限制在 ≤{SCALE_MAX_REQUESTED:.2f} 时 s* = "
            f"{_fmt(scan.get('s_star_within_requested_range'), '.2f')} "
            f"(corr {_fmt(scan.get('corr_at_s_star_within_requested_range'), '.3f')}); "
            f"s = 1 基线 {_fmt(scan.get('corr_at_1'), '.3f')}; "
            f"FWHM 比隐含尺度 {_fmt(scan.get('fwhm_ratio_implied_scale'), '.2f')}"
        )
    lines.append("")
    lines.append(
        "读法: `s*` 是让实测绕**阵列中心 (光轴)** 放大/缩小后与模型相关系数最大的尺度"
        "(不做平移对齐, 所以 `corr(s=1)` 就是全网格相关系数)。`s*` 明显偏离 1、且与"
        "「模型/实测 FWHM 比」相符时, 说明两者差的**不是形状而是几何尺度** —— "
        "即模型远场采样尺度与实验系统实际尺度对不上 (K 标定 / 焦距补偿一类), "
        "这时候再怎么调相位也换不回相关系数。`s*` 明显大于 1 表示实测光斑比模型窄。"
    )
    lines.append("")

    # ---- 历史配对 ---------------------------------------------------------
    lines.append("## 3. 模型损失 vs 实测峰值 (块 C)")
    lines.append("")
    for run in runs:
        pairing = run.get("history_pairing")
        if not pairing or not pairing.get("rows"):
            continue
        lines.append(f"### {run['run_id']}")
        lines.append("")
        lines.append(f"* {pairing['statement']}")
        lines.append("")
        rows = [
            [
                str(row["iteration"]),
                _fmt(row["model_loss_same_iter"], ".4e"),
                _fmt(row["hw_loss_measured"], ".4e"),
                _fmt(row["model_phase_rms_same_iter"], ".4f"),
                _fmt(row["hw_phase_rms_rad"], ".4f"),
                _fmt(row["hw_measured_peak"], ".1f"),
                _fmt(row["hw_exposure_ms"], ".4f"),
            ]
            for row in pairing["rows"]
        ]
        lines.extend(
            _table(
                [
                    "迭代",
                    "模型损失(history)",
                    "实测损失(history_hw)",
                    "模型相位 RMS",
                    "实测相位 RMS",
                    "实测峰值",
                    "曝光 ms",
                ],
                rows,
            )
        )
        lines.append("")
    lines.append(
        "注意: `history_hw.csv` 与 `history.csv` 的迭代号只是同一套标号, 不是同一个时钟"
        "(前者是测量锚定步, 后者是纯模型步); 这里按标号配对是任务指定的口径, 结论只看"
        "趋势方向。"
    )
    lines.append("")

    # ---- 零相位参考 -------------------------------------------------------
    lines.append("## 4. 零相位参考 (块 D)")
    lines.append("")
    rows = []
    for run in runs:
        flat = run.get("flat_reference") or {}
        rows.append(
            [
                run["run_id"],
                _fmt(flat.get("intensity_correlation"), ".4f"),
                _fmt(flat.get("peak_ratio_raw"), ".4f"),
            ]
        )
    lines.extend(_table(["运行", "零相位相关系数", "原始峰值比"], rows))
    lines.append("")
    lines.append(
        "零相位参考是「相位全零/补偿面板单独上屏」时的实测, 与模型的相关系数量级给出一"
        "条地板线: 优化后的相关系数离它多远, 才是相位优化真正带来的东西。"
    )
    lines.append("")

    # ---- 判定 -------------------------------------------------------------
    lines.append("## 5. 判定 (块 E)")
    lines.append("")
    rows = []
    for run in runs:
        verdict = run["verdict"]
        rows.append(
            [
                run["run_id"],
                verdict["band"],
                "是" if verdict.get("nearly_identical") else "否",
                str(verdict.get("dominant_discrepancy")),
            ]
        )
    lines.extend(_table(["运行", "仓库口径判定", "几乎一致?", "主导偏差"], rows))
    lines.append("")
    for run in runs:
        reasons = run["verdict"].get("nearly_identical_reasons") or []
        if reasons:
            lines.append(f"* {run['run_id']} 判「否」的理由: " + "; ".join(reasons))
    lines.append("")

    # ---- 跨运行 -----------------------------------------------------------
    lines.append("## 6. 跨运行表 (按日期)")
    lines.append("")
    lines.append(
        "`init_phase` 为 `random` 表示没有 GS 初值; `focus_compensation_focal_mm` 为 `inf` "
        "表示记录的焦距补偿面板实质为零。`duplicate_of` 不为空表示实测网格与另一运行逐字节"
        "相同 (副本)。"
    )
    lines.append("")
    rows = []
    for row in payload["cross_run_table"]:
        rows.append(
            [
                row["run_id"],
                row["date"],
                str(row["mode"]),
                _fmt(row["iters"], ".0f"),
                _fmt(row["lr"], ".3g"),
                str(row["init_phase"]),
                _fmt(row["gs_iters"], ".0f"),
                _fmt(row["corr_linear"], ".3f"),
                _fmt(row["corr_sqrt"], ".3f"),
                _fmt(row["corr_log10"], ".3f"),
                _fmt(row["corr_linear_target_roi"], ".3f"),
                _fmt(row["corr_linear_disc_roi"], ".3f"),
                _fmt(row["fwhm_ratio_model_over_measured"], ".2f"),
                _fmt(row["peak_offset_magnitude_samples"], ".2f"),
                _fmt(row["grid_centroid_offset_samples"], ".2f"),
                _fmt(row["nrmse_full"], ".3f"),
                _fmt(row["s_star"], ".2f"),
                _fmt(row["corr_at_s_star"], ".3f"),
                _fmt(row["corr_at_1"], ".3f"),
                _fmt(row["ccd_peak"], ".1f"),
                _fmt(row["ccd_fwhm_px"], ".2f"),
                str(row["focus_compensation_focal_mm"]),
                str(row["verdict_band"]),
            ]
        )
    lines.extend(
        _table(
            [
                "运行",
                "日期",
                "模式",
                "iters",
                "lr",
                "init",
                "gs_iters",
                "线性 corr",
                "sqrt corr",
                "log corr",
                "ROI(目标区) corr",
                "ROI(圆盘) corr",
                "FWHM 比",
                "峰值偏移",
                "全网格质心偏移",
                "NRMSE",
                "s*",
                "corr(s*)",
                "corr(s=1)",
                "ccd_peak",
                "ccd_fwhm",
                "焦距补偿 mm",
                "判定段",
            ],
            rows,
        )
    )
    lines.append("")

    # ---- 趋势 -------------------------------------------------------------
    lines.append("## 7. 一致性变过吗 (跨日期)")
    lines.append("")
    lines.append(payload["trend"]["statement"])
    lines.append("")

    lines.append("## 8. 未定的部分")
    lines.append("")
    for note in payload["caveats"]:
        lines.append(f"* {note}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_caveats(runs: Sequence[dict[str, Any]], payload: dict[str, Any]) -> list[str]:
    """把「没做到 / 做不到」的话如实写出来。"""
    caveats: list[str] = [
        "`data/debug/differentiable_opt/` 下只有 3 个运行目录 "
        "(phys_20261116_01 / phys_20261116_02 / 20261116_114424); 更早的 2026-10-22、"
        "2026-10-23 运行只在 `report/slm/differentiable_opt/artifacts/` 下, 其中 "
        "2026-10-22 两个运行以及 20261023_012505 没有原始远场网格, 只能读记录值。",
        "20261023_020423 没有匹配到任何 `run_config.json`, 它的配置列 (iters/lr/init/"
        "焦距补偿) 为空, 相关系数由 8 个轴向候选自行择优得到。",
        "`history_hw.csv` 与 `history.csv` 的迭代号不是同一个时钟, 块 C 的配对只按标号, "
        "结论只看趋势方向。",
        "跨运行比相关系数只有 5–7 个样本, 是弱证据; 焦距补偿 / iters / gs_iters / k_mean "
        "/ 曝光几乎同时变化, 现有数据分不开是哪一项造成差异。",
    ]
    duplicated = [
        row["run_id"] for row in payload["cross_run_table"] if row.get("duplicate_of")
    ]
    if duplicated:
        caveats.append(
            "实测网格逐字节相同的运行 (副本) 已标记: " + ", ".join(duplicated) + "。"
        )
    nonrepro = [
        r["run_id"]
        for r in runs
        if r["mode"] == "full" and not r["reproduction"].get("within_tolerance")
    ]
    if nonrepro:
        caveats.append(
            "对账超出容差的运行 (记录口径不能复现, 已按 8 候选重新择优并大声记录): "
            + ", ".join(nonrepro)
            + "。"
        )
    return caveats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_config_overrides(values: Sequence[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"--config 需要 RUN_DIR=PATH 形式, 收到: {item}")
        run_dir, config_path = item.split("=", 1)
        overrides[str(Path(run_dir).resolve())] = Path(config_path)
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="离线核对模型预测远场与上屏实测远场的一致性 (不碰设备)。"
    )
    parser.add_argument("--run-dir", action="append", default=[], help="运行目录, 可重复")
    parser.add_argument(
        "--all-differentiable-opt",
        action="store_true",
        help=(
            f"扫描 {DIFFERENTIABLE_OPT_DIR.relative_to(ROOT).as_posix()} 与 "
            f"{REPORT_DIR.relative_to(ROOT).as_posix()}/artifacts 下所有运行目录"
        ),
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="RUN_DIR=PATH",
        help="给没有自带 run_config.json 的运行目录外部指定配置 (可重复)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=f"输出目录 (默认 {DEFAULT_OUT_DIR_NAME})",
    )
    parser.add_argument(
        "--fig",
        dest="fig",
        action="store_true",
        default=True,
        help="画图 (默认开)",
    )
    parser.add_argument("--no-fig", dest="fig", action="store_false", help="不画图")
    parser.add_argument(
        "--scale-scan",
        dest="scale_scan",
        action="store_true",
        default=True,
        help="做尺度扫描 (默认开)",
    )
    parser.add_argument(
        "--no-scale-scan", dest="scale_scan", action="store_false", help="不做尺度扫描"
    )
    parser.add_argument(
        "--scale-step",
        type=float,
        default=SCALE_STEP,
        help=f"尺度扫描步长 (默认 {SCALE_STEP:.2f})",
    )
    parser.add_argument(
        "--log-level", default="INFO", help="loguru 级别 (默认 INFO)"
    )
    return parser


def collect_run_dirs(args: argparse.Namespace) -> list[Path]:
    run_dirs: list[Path] = []
    seen: set[str] = set()
    if args.all_differentiable_opt:
        for root in (DIFFERENTIABLE_OPT_DIR, REPORT_DIR / "artifacts"):
            for path in discover_differentiable_opt(root):
                key = str(path.resolve())
                if key not in seen:
                    seen.add(key)
                    run_dirs.append(path)
    for raw in args.run_dir:
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_dir():
            logger.warning(f"--run-dir 不存在或不是目录: {raw}")
            continue
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            run_dirs.append(path)
    return run_dirs


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    started = time.perf_counter()

    out_dir = Path(args.out) if args.out else REPORT_DIR / "artifacts" / DEFAULT_OUT_DIR_NAME
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    run_dirs = collect_run_dirs(args)
    if not run_dirs:
        logger.error("没有可分析的运行目录 (检查 --run-dir / --all-differentiable-opt)")
        return EXIT_NO_RUNS
    overrides = _parse_config_overrides(args.config)

    reference, reference_reason = load_reference_module()
    if reference is None:
        logger.warning(f"参考实现未能 import ({reference_reason}), 改用本地镜像")
    else:
        logger.info(f"参考实现已按路径 import: {REFERENCE_SCRIPT.name}")

    step = float(args.scale_step) if args.scale_step and args.scale_step > 0 else SCALE_STEP
    scales = np.arange(SCALE_MIN, SCALE_MAX + 1e-9, step)

    logger.info(f"待分析运行 {len(run_dirs)} 个; 尺度扫描步长 {step:.2f} ({len(scales)} 个点)")

    runs: list[dict[str, Any]] = []
    for index, run_dir in enumerate(run_dirs, start=1):
        logger.info(f"[{index}/{len(run_dirs)}] {run_dir.name}")
        result = analyse_run(
            run_dir,
            overrides,
            scales,
            reference=reference,
            do_scale_scan=bool(args.scale_scan),
        )
        if result is not None:
            runs.append(result)

    if not runs:
        logger.error("所有运行目录都被跳过")
        return EXIT_NO_RUNS

    rows = cross_run_rows(runs)
    payload: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": SCRIPT_REL,
        "reference_module": {
            "path": REFERENCE_SCRIPT.relative_to(ROOT).as_posix(),
            "loaded": reference is not None,
            "reason": reference_reason,
        },
        "engine_used": "reference_module" if reference is not None else "local_mirror",
        "scale_scan": {
            "min": SCALE_MIN,
            "max": SCALE_MAX,
            "step": step,
            "points": int(scales.size),
        },
        "thresholds": {
            "corr_high": CORR_HIGH,
            "corr_mid": CORR_MID,
            "fwhm_ratio_tol": FWHM_RATIO_TOL,
            "peak_offset_tol_samples": PEAK_OFFSET_TOL_SAMPLES,
            "repro_tolerance": REPRO_TOL,
        },
        "runs": runs,
        "cross_run_table": rows,
        "trend": trend_paragraph(rows),
    }
    payload["caveats"] = build_caveats(runs, payload)

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.add(out_dir / "run.log", level=args.log_level)

    fig_paths: list[str] = []
    if args.fig:
        pred_png = out_dir / "fidelity_pred_vs_meas.png"
        scan_png = out_dir / "fidelity_scale_scan.png"
        if figure_pred_vs_meas(runs, pred_png):
            fig_paths.append(pred_png.name)
        if figure_scale_scan(runs, scan_png):
            fig_paths.append(scan_png.name)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        for name in list(fig_paths):
            src = out_dir / name
            if src.is_file():
                dst = FIGURES_DIR / name
                dst.write_bytes(src.read_bytes())
                logger.info(f"图已复制到 {dst.relative_to(ROOT).as_posix()}")
    payload["figures"] = fig_paths

    payload["offline_guarantee"] = {
        "drivers_imported": _leaked_driver_modules(),
        "pass": not _leaked_driver_modules(),
    }
    payload["wall_time_s"] = time.perf_counter() - started

    (out_dir / "model_hw_fidelity.json").write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(payload, runs, out_dir / "model_hw_fidelity.md")

    full = [r for r in runs if r["mode"] == "full"]
    logger.info(
        f"完成: {len(runs)} 个运行 (其中 {len(full)} 个有原始网格) ⇒ {_rel(out_dir)}"
    )
    for run in full:
        logger.info(
            f"  {run['run_id']}: corr={_fmt(run['metrics']['corr_linear'], '.3f')} "
            f"({run['verdict']['band']}), 几乎一致="
            f"{'是' if run['verdict']['nearly_identical'] else '否'}, "
            f"主导偏差={run['verdict']['dominant_discrepancy']}"
        )
    logger.info(payload["trend"]["statement"])
    logger.info(f"墙钟时间 {payload['wall_time_s']:.1f} s")

    assert_no_drivers()
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
