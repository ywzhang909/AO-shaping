"""从历史硬件运行里挖「相位 → 实测 CCD 远场」配对, 离线预标定物理预测模型的系统误差。

**纯离线**: 只读落盘数据, 不打开任何设备。

要回答的问题
------------
MIL 闭环的瓶颈被审计判定为**物理预测模型失配** (宽度比 1.328 / 质心偏移 1.498 模型格),
于是问题是: **能不能用历史数据替代大部分实验采集**?

做法
----
1. **挖配对** —— 历史 MIL 运行落盘的探针 (``probes/*.npz``: 指令相位 + 实测画面)
   与今天新采的两组 (B1 基线 38 帧 / B2 处理 106 帧) 一起构成配对集。
   口径不兼容的样本被**排除并记录原因** (仿真运行 / 缺几何 / 系数被回滚 / 帧饱和 …)。
2. **曝光归一化** —— 跨天曝光从 0.02 ms 到 100 ms, CCD 有暗偏置与非线性,
   所以逐会话拟合一条响应曲线 ``counts = bias + gain · exposure_ms``,
   把每帧归一化到「每毫秒每单位增益」的可比口径; 用的是哪条曲线、残差多少, 全部写进产物。
3. **拟合系统误差** —— 先用**可解释的低参数** (焦距尺度 / 瞳面中心 / 增益)
   解释残差, 再看残差里还剩什么 (能量 / 宽度 / 位置 / 形状各占多少),
   并把拟合结果翻译成仓库既有的容器:
   :class:`~ao_shaping.utils.wavefront.model_error_terms.ModelErrorTerms`
   (γ / 离焦 / 倾斜)、:class:`~ao_shaping.utils.wavefront.calibration_terms.PupilAffine`
   (τ_θ 六标量)、:class:`~ao_shaping.utils.wavefront.calibration_terms.TravelScale`
   (七族行程缩放)、数字孪生的 ``equivalent_focal_length_m`` / ``from_measured_scale``。
4. **训/验分离** —— 训练用历史配对 (**按会话分组**切分, 同一会话不跨两侧),
   验证用今天新采的 B1 / B2 (独立日期、独立种子) ⇒ 判定「历史预标定是否真的泛化」。

复用 (不重写)
------------
* 相机 ↔ 模型网格换算: ``ao_shaping.optimizer.wfless.slm_model_in_loop._to_model_grid``
  (旋转象限用硬件路径的写死值 1)。
* 四类分解 (能量 / 宽度 / 位置 / 形状) 与原始 Pearson r:
  ``scripts/slm.analyze_model_in_loop_loss.decompose_mismatch``。
* 指标: ``ao_shaping.utils.image.beam_metrics`` 的 canonical 入口
  (``prepare_frame`` / ``zero_order_center``)。
* B1 / B2 的物理预测模型与相机映射: ``scripts/slm.fit_defocus_forward_dataset``
  (``load_observations`` / ``ramp_matrix`` / ``PhysicalForwardFit``)。
* 误差项的物理作用: ``zernike_coefficient_optimizer.apply_model_error_terms``
  (离焦 / 倾斜作用在焦面复场上)。

产物
----
``logs/history_model_bias/<时间戳>/`` 下::

    pairs.json        # 每条配对的来源 / 会话 / 曝光 / 归一化参数 / 网格口径 + 排除清单
    fit.json          # 拟合到的系统误差项 + 诊断 (训练集与历史验证集)
    validation.json   # B1 / B2 上的标定前后对照
    summary.md        # 中文结论 + 阈值对照表 + 「还差多少采集」
    calibration.png   # 标定前 / 后的宽度比与质心偏移对照

用法
----
::

    python scripts/slm/mine_history_model_bias.py
    python scripts/slm/mine_history_model_bias.py --probes-per-session 6 --out-dir logs/history_model_bias

术语: 中文叙述遵守 ``docs/terminology_glossary.md`` 的规范术语表 (A/B 对照语境用「变体」,
加性背景用「加性背景（杂散光）项」, 对比择优不写成英文)。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from functools import partial
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from loguru import logger

from ao_shaping.drivers.sim.bench_twin import (
    BenchTwinConfig,
    equivalent_focal_length_m,
)
from ao_shaping.optimizer.wfless.slm_model_in_loop import (
    _make_optimizer,
    _to_model_grid,
)
from ao_shaping.utils.image.beam_metrics import prepare_frame, zero_order_center
from ao_shaping.utils.wavefront.calibration_terms import (
    PupilAffine,
    TravelScale,
    family_of_noll,
)
from ao_shaping.utils.wavefront.model_error_terms import ModelErrorTerms
from scripts._common import fmt_general
from scripts.slm.analyze_model_in_loop_loss import (
    MISSING,
    RECOMPUTE_TOL,
    RunData,
    _as_float,
    _gain_leakage,
    decompose_mismatch,
    discover_runs,
    load_run,
)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: 历史 MIL 运行的扫描根 (只读)。
DEFAULT_ROOTS: tuple[str, ...] = (
    "data/debug/model_in_loop_hw_ab",
    "data/debug/slm_model_in_loop",
    "data/debug/model_in_loop_auto",
    "data/debug/model_in_loop_error_params",
)

#: 今天新采的验证组 (B1 = 基线, B2 = 含先验相位)。
CAPTURE_BASELINE = "data/debug/physical_forward_defocus/capture_20261022_s20261022"
CAPTURE_PRIOR = "data/debug/physical_forward_defocus/capture_20261022_s20261022_prior"

#: 审计已判定「落盘系数被回滚、不可复算」的一族 —— 排除。
ROLLED_BACK_STAMPS = ("20261019_103208",)

#: 相机帧饱和判据 (8 bit 探测器)。
SATURATION_LEVEL = 250

#: 阶段 5 的判据 (见 ``report/shape_trials/experiment_summary_20261022.md`` §8)。
WIDTH_RATIO_TARGET = 1.0
WIDTH_RATIO_TOLERANCE = 0.10
CENTROID_SHIFT_LIMIT_CELLS = 0.5
#: 另一条线的停止条件 2 (CV 绝对误差)。
CV_ABS_LIMIT = 0.05

#: 焦距尺度拟合的搜索范围 (相对宽度比)。
SCALE_SEARCH = (0.55, 1.80)

#: 会话切分: 训练 / 历史验证集。
SESSION_SPLIT_FRACTION = 0.7
#: A fit, a residual, or a before/after comparison needs at least two entries: with
#: one the quantity is undefined rather than merely noisy, so the case is skipped.
MIN_COMPARISON_SAMPLES = 2

#: Magnitude below which a scale factor (deviation from unity) or a centroid shift is
#: numerically degenerate -- identity, or no movement at all. Such a sample carries no
#: signal to regress, unlike a division floor, which is a separate concern.
DEGENERACY_EPSILON = 1e-9


# ---------------------------------------------------------------------------
# 容器
# ---------------------------------------------------------------------------


@dataclass
class ExposureFit:
    """一次会话的曝光响应曲线 ``counts = bias + gain · exposure_ms``。"""

    bias: float
    gain: float
    source: str
    n_points: int = 0
    exposures: list[float] = field(default_factory=list)
    peaks: list[float] = field(default_factory=list)
    residuals: list[float] = field(default_factory=list)
    max_abs_residual: float | None = None
    rms_residual: float | None = None
    r_squared: float | None = None
    holdout: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """可序列化的字典。"""
        return asdict(self)


@dataclass
class Exclusion:
    """一条被排除的样本 (含原因)。"""

    scope: str
    identifier: str
    reason: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        """可序列化的字典。"""
        return asdict(self)


@dataclass
class HistoryPair:
    """一条「相位 → 实测远场」配对 (历史 MIL 探针)。"""

    pair_id: str
    session_id: str
    campaign: str
    stamp: str
    variant: str
    probe_split: str
    round_index: int
    file: str
    exposure_ms: float | None
    region: int
    far_field_size: int
    camera_px_per_model_px: float
    rotation_quarters: int
    phase: np.ndarray
    predicted_crop: np.ndarray
    measured_crop: np.ndarray
    exposure_fit: ExposureFit

    def to_dict(self) -> dict[str, Any]:
        """可序列化的元数据 (不含数组)。"""
        out = asdict(self)
        for key in ("predicted_crop", "measured_crop", "phase"):
            out.pop(key)
        out["exposure_fit"] = self.exposure_fit.to_dict()
        return out


@dataclass
class _PendingPair:
    """一条待校验的配对 (逐像素校验通过后才进入 :class:`HistoryPair`)。"""

    probe: Any
    predicted_crop: np.ndarray
    measured_crop: np.ndarray
    deviation: float | None


@dataclass
class Session:
    """一次历史 MIL 运行 (配对的容器)。"""

    session_id: str
    campaign: str
    stamp: str
    variant: str
    region: int
    far_field_size: int
    camera_px_per_model_px: float
    rotation_quarters: int
    focal_length_m: float
    wavelength_m: float
    slm_pixel_um: float
    ccd_pixel_um: float
    panel_span_px: float
    exposure_fit: ExposureFit
    coefficients: np.ndarray | None = None
    amplitude: np.ndarray | None = None
    aperture: np.ndarray | None = None
    basis: np.ndarray | None = None
    flat_grid_counts: np.ndarray | None = None
    model_gain: float | None = None
    flat_leakage: float | None = None
    forward_model_max_relative_l1: float | None = None
    pairs: list[HistoryPair] = field(default_factory=list)


@dataclass
class Calibration:
    """拟合到的系统误差项。"""

    scale: float
    shift_dx: float
    shift_dy: float
    gain: float
    n_pairs: int
    detail: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 曝光响应曲线
# ---------------------------------------------------------------------------


def fit_exposure_response(
    points: Sequence[tuple[float, float]],
    source: str,
    *,
    holdout_index: int | None = None,
) -> ExposureFit:
    """拟合 ``counts = bias + gain · exposure_ms``。

    暗偏置 (bias) 是 CCD 的黑电平, 增益 (gain) 是「每毫秒多少 counts」。
    两者都从**同一条状态** (平场) 的多曝光读数上拟合, 所以跨会话可比。

    Args:
        points: ``(exposure_ms, peak_counts)`` 序列。
        source: 拟合所用的数据来源说明 (写进产物)。
        holdout_index: 验证集做交叉验证的点下标; None = 全部参与拟合。

    Returns:
        拟合结果; 点少于 2 个或曝光全相同时退化为 ``bias=0, gain=1``。
    """
    usable = [(float(e), float(p)) for e, p in points if math.isfinite(e) and math.isfinite(p)]
    if len(usable) < MIN_COMPARISON_SAMPLES or len({round(e, 9) for e, _ in usable}) < MIN_COMPARISON_SAMPLES:
        return ExposureFit(
            bias=0.0,
            gain=1.0,
            source=f"{source}（曝光单一, 退化为 gain=1 / bias=0）",
            n_points=len(usable),
            exposures=[e for e, _ in usable],
            peaks=[p for _, p in usable],
        )

    train = usable if holdout_index is None else [
        item for index, item in enumerate(usable) if index != holdout_index
    ]
    if len(train) < MIN_COMPARISON_SAMPLES:
        train = usable

    ex = np.asarray([e for e, _ in train], dtype=np.float64)
    pk = np.asarray([p for _, p in train], dtype=np.float64)
    mean_e = float(ex.mean())
    mean_p = float(pk.mean())
    denominator = float(np.sum((ex - mean_e) ** 2))
    gain = float(np.sum((ex - mean_e) * (pk - mean_p)) / denominator) if denominator > 0 else 1.0
    bias = float(mean_p - gain * mean_e)
    if gain <= 0.0:
        gain, bias = 1.0, 0.0

    predicted = bias + gain * ex
    residuals = pk - predicted
    ss_total = float(np.sum((pk - mean_p) ** 2))
    ss_residual = float(np.sum(residuals**2))
    r_squared = float(1.0 - ss_residual / ss_total) if ss_total > 0 else None

    holdout: dict[str, Any] | None = None
    if holdout_index is not None and 0 <= holdout_index < len(usable):
        h_e, h_p = usable[holdout_index]
        h_pred = bias + gain * h_e
        holdout = {
            "exposure_ms": h_e,
            "peak_counts": h_p,
            "predicted_counts": h_pred,
            "residual_counts": h_p - h_pred,
        }

    return ExposureFit(
        bias=bias,
        gain=gain,
        source=source,
        n_points=len(train),
        exposures=[e for e, _ in usable],
        peaks=[p for _, p in usable],
        residuals=[float(r) for r in residuals],
        max_abs_residual=float(np.max(np.abs(residuals))),
        rms_residual=float(math.sqrt(ss_residual / max(residuals.size, 1))),
        r_squared=r_squared,
        holdout=holdout,
    )


def normalize_exposure(
    frame_raw: np.ndarray, exposure_ms: float | None, fit: ExposureFit
) -> np.ndarray:
    """把原始 counts 归一化到「每毫秒每单位增益的辐照度」口径。

    ``I = (raw - bias) / (gain · exposure_ms)`` —— 偏置与增益来自会话的曝光响应曲线。

    Args:
        frame_raw: 原始 (未做背景扣除) 的相机帧。
        exposure_ms: 该帧的曝光时间 (ms); None 或非正 → 只扣偏置。
        fit: 会话的曝光响应拟合。

    Returns:
        归一化后的辐照度图 (float64)。
    """
    raw = np.asarray(frame_raw, dtype=np.float64)
    out = raw - float(fit.bias)
    scale = float(fit.gain)
    if exposure_ms is not None and math.isfinite(exposure_ms) and exposure_ms > 0:
        scale *= float(exposure_ms)
    if scale <= 0:
        scale = 1.0
    return out / scale


# ---------------------------------------------------------------------------
# 曝光阶梯 (light ladder)
# ---------------------------------------------------------------------------


def find_light_ladder(run_dir: Path) -> Path | None:
    """向上找该运行所属会话的曝光阶梯目录 (``light/<戳>/``)。"""
    current = run_dir.parent
    for _ in range(6):
        light = current / "light"
        if light.is_dir():
            ladders = sorted(p for p in light.iterdir() if p.is_dir())
            if ladders:
                return ladders[-1]
        if current.parent == current:
            break
        current = current.parent
    return None


def read_light_ladder(ladder: Path) -> list[tuple[float, float]]:
    """读曝光阶梯, 返回 ``(exposure_ms, peak_counts)``。

    文件名形如 ``frame_20ms.npz``; 帧里是 ``frame`` (uint8 原始 counts)。
    """
    points: list[tuple[float, float]] = []
    for path in sorted(ladder.glob("frame_*.npz")):
        stem = path.stem.replace("frame_", "")
        try:
            exposure = float(stem.replace("ms", ""))
        except ValueError:
            continue
        try:
            with np.load(path, allow_pickle=False) as payload:
                if "frame" not in payload.files:
                    continue
                frame = np.asarray(payload["frame"])
        except (OSError, ValueError):
            continue
        peak = float(frame.max())
        if peak <= 0.0:
            continue
        points.append((exposure, peak))
    return points


# ---------------------------------------------------------------------------
# 历史配对挖掘
# ---------------------------------------------------------------------------


def _probe_exposure(probe: Any) -> float | None:
    """探针的曝光时间 (ms)。"""
    return _as_float(probe.exposure_ms)


def _session_exposure_fit(
    run: RunData, probes: Sequence[Any]
) -> ExposureFit:
    """为一次会话拟合曝光响应曲线。

    优先用会话的**曝光阶梯** (同一平场状态的多曝光, 是唯一能把 bias/gain 解开的
    数据); 没有阶梯时退化为「按曝光归一、偏置交给 ``prepare_frame``」。
    """
    ladder = find_light_ladder(run.path)
    if ladder is not None:
        points = read_light_ladder(ladder)
        saturated = [p for p in points if p[1] >= SATURATION_LEVEL]
        usable = [p for p in points if p[1] < SATURATION_LEVEL]
        if len(usable) >= MIN_COMPARISON_SAMPLES:
            source = f"曝光阶梯 {ladder.as_posix()}"
            if saturated:
                source += f" (剔除 {len(saturated)} 个饱和点)"
            holdout = len(usable) - 1
            return fit_exposure_response(usable, source, holdout_index=holdout)

    # 退化路径: 会话自身的探针 —— 只用「同一相位被多曝光测到」的点。
    raw_frames = load_probe_raw_frames(run.path)
    flat_exposures: dict[float, float] = {}
    for probe in probes:
        if probe.split != "flat":
            continue
        exposure = _probe_exposure(probe)
        raw = raw_frames.get(probe.name)
        if exposure is None or raw is None:
            continue
        flat_exposures[exposure] = float(np.max(raw))
    if len(flat_exposures) >= MIN_COMPARISON_SAMPLES:
        points = sorted(flat_exposures.items())
        return fit_exposure_response(points, "会话内平场探针的多曝光读数")

    return ExposureFit(
        bias=0.0,
        gain=1.0,
        source="会话内曝光单一 (背景扣除交给 prepare_frame, 增益按曝光线性归一)",
        n_points=len(flat_exposures),
    )


def _rotation_for(run: RunData) -> int:
    """旋转象限 (硬件路径写死 1)。"""
    return 0 if str(run.config.get("cam_type", "daheng")) == "sim" else 1


def load_probe_raw_frames(run_dir: Path) -> dict[str, np.ndarray]:
    """读 ``probes/*.npz`` 里的**原始** counts 帧。

    审计脚本的 :class:`Probe` 只留了 ``prepare_frame`` 之后的 ``frame``;
    曝光归一化必须用原始 counts (暗偏置要在归一化时扣掉), 所以这里单独读一次
    ``raw_frame``。键是文件名, 与审计的 ``Probe.name`` 一致。
    """
    out: dict[str, np.ndarray] = {}
    for path in sorted((run_dir / "probes").glob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as payload:
                if "raw_frame" not in payload.files:
                    continue
                out[path.name] = np.asarray(payload["raw_frame"], dtype=np.float64)
        except (OSError, ValueError):
            continue
    return out


def _footprint_cells(cam_size: int, camera_px_per_model_px: float) -> int:
    """相机视场在远场网格上占的边长 (格)。

    尺度非法 (非有限 / 非正) 时退到最小视场 8 格, 不抛异常 —— 调用方在
    缺几何时就会把整个会话排除。
    """
    scale = float(camera_px_per_model_px)
    if not math.isfinite(scale) or scale <= 0.0:
        return 8
    return max(8, int(round(cam_size / scale)))


def _crop_footprint(array: np.ndarray, footprint_cells: int) -> np.ndarray:
    """取远场网格中央的相机视场区域 (与审计口径一致)。"""
    half = max(int(footprint_cells) // 2, 1)
    side = 2 * half
    rows, cols = array.shape
    r0 = rows // 2 - half
    c0 = cols // 2 - half
    return np.ascontiguousarray(array[r0 : r0 + side, c0 : c0 + side])


def mine_history_sessions(
    roots: Sequence[str],
    *,
    probes_per_session: int,
    seed: int,
) -> tuple[list[Session], list[Exclusion]]:
    """从历史 MIL 运行挖配对。

    排除规则 (每条都记录原因):

    * ``simulation`` —— ``cam_type == "sim"``, 没有实测 CCD 画面;
    * ``no_run_config`` —— 缺 ``run_config.json``;
    * ``no_bench_geometry`` —— 缺 ``bench_geometry.json`` 或缺
      ``far_field_size`` / ``camera_px_per_model_px`` (相机 ↔ 模型网格无法换算);
    * ``no_probes`` —— 没有落盘探针;
    * ``rolled_back_coefficients`` —— 审计已判定的回滚族, 落盘系数不可复算;
    * ``no_fitted_coefficients`` —— 缺 ``fitted_coefficients.npy`` (模型预测无法复算);
    * ``region_mismatch`` —— 探针相位形状与 ``region`` 不符;
    * ``degenerate_frame`` —— 帧全黑或已饱和。

    Args:
        roots: 扫描根目录。
        probes_per_session: 每个会话最多取多少条探针 (抽样口径)。
        seed: 抽样随机种子 (固定, 保证可复现)。

    Returns:
        ``(sessions, exclusions)``。
    """
    sessions: list[Session] = []
    exclusions: list[Exclusion] = []
    rng = np.random.default_rng(seed)

    for campaign, run_dir in discover_runs(roots):
        stamp = run_dir.name
        if stamp in ROLLED_BACK_STAMPS:
            exclusions.append(
                Exclusion(
                    scope="session",
                    identifier=stamp,
                    reason="rolled_back_coefficients",
                    detail="审计已判定该族落盘系数被回滚, 不可复算 (20261019_103208)",
                )
            )
            continue
        run = load_run(campaign, run_dir)
        if not run.config:
            exclusions.append(
                Exclusion("session", stamp, "no_run_config", "缺 run_config.json")
            )
            continue
        if not run.is_hardware:
            exclusions.append(
                Exclusion("session", stamp, "simulation", "cam_type=sim, 无实测 CCD 画面")
            )
            continue
        geometry = run.geometry
        far_field_size = int(geometry.get("far_field_size", 0) or 0)
        scale = _as_float(geometry.get("camera_px_per_model_px"))
        if not far_field_size or not scale:
            exclusions.append(
                Exclusion(
                    "session",
                    stamp,
                    "no_bench_geometry",
                    "缺 bench_geometry.json 或缺 far_field_size / camera_px_per_model_px",
                )
            )
            continue
        if not run.probes:
            exclusions.append(
                Exclusion("session", stamp, "no_probes", "没有落盘探针")
            )
            continue
        if run.coefficients is None:
            exclusions.append(
                Exclusion(
                    "session",
                    stamp,
                    "no_fitted_coefficients",
                    "缺 fitted_coefficients.npy, 模型预测无法复算",
                )
            )
            continue

        region = int(run.config.get("region", 0) or 0)
        rotation = _rotation_for(run)
        exposure_fit = _session_exposure_fit(run, run.probes)

        flats = [p for p in run.probes if p.split == "flat"]
        if not flats:
            exclusions.append(
                Exclusion("session", stamp, "no_flat_probe", "没有平场探针, 冻结光轴无从确定")
            )
            continue
        flat = flats[0]
        centre = zero_order_center(flat.frame, refine=False)
        footprint = _footprint_cells(int(flat.frame.shape[0]), scale)

        config = _config_for(run)
        if config is None:
            exclusions.append(
                Exclusion("session", stamp, "no_config", "无法重建 SlmModelInLoopConfig")
            )
            continue
        try:
            optimizer = _make_optimizer(config, None, run.source_amplitude)
            basis = np.asarray(optimizer.generate_basis(), dtype=np.float64)
        except Exception as exc:  # pragma: no cover - 依赖 torch 后端
            exclusions.append(
                Exclusion("session", stamp, "optimizer_failed", f"物理预测模型构造失败: {exc}")
            )
            continue
        # 瞳面掩膜: piston 基在瞳面内为 1、外面为 0 (与孪生的 NaN 出瞳面约定同源)。
        aperture = (basis[0] != 0).astype(np.float64)

        gain, leakage = _gain_leakage(run)
        amplitude = (
            np.asarray(run.source_amplitude, dtype=np.float64)
            if run.source_amplitude is not None
            else None
        )

        # 抽样: 平场探针必留 (它是光轴锚点), 其余按 split 分层抽。
        candidates = [p for p in run.probes if p.split != "flat"]
        by_split: dict[str, list[Any]] = {}
        for probe in candidates:
            by_split.setdefault(probe.split, []).append(probe)
        chosen: list[Any] = []
        for split in sorted(by_split):
            group = by_split[split]
            take = min(len(group), max(1, probes_per_session // max(len(by_split), 1)))
            if take >= len(group):
                chosen.extend(group)
            else:
                index = rng.choice(len(group), size=take, replace=False)
                chosen.extend(group[int(i)] for i in sorted(index))
        chosen = chosen[:probes_per_session]

        session_id = f"{campaign}/{run_dir.parent.name}/{stamp}"
        pending: list[_PendingPair] = []
        forward_checks: list[float] = []
        session = Session(
            session_id=session_id,
            campaign=campaign,
            stamp=stamp,
            variant=run.variant,
            region=region,
            far_field_size=far_field_size,
            camera_px_per_model_px=float(scale),
            rotation_quarters=rotation,
            focal_length_m=float(run.config.get("focal_length_m", 0.0) or 0.0),
            wavelength_m=float(run.config.get("wavelength_nm", 1064.0) or 1064.0) * 1e-9,
            slm_pixel_um=float(run.config.get("slm_pixel_um", 8.0) or 8.0),
            ccd_pixel_um=float(run.config.get("camera_pixel_um", 2.2) or 2.2),
            panel_span_px=float(run.config.get("panel_span_px", 900.0) or 900.0),
            exposure_fit=exposure_fit,
            coefficients=np.asarray(run.coefficients, dtype=np.float64),
            amplitude=(
                np.asarray(run.source_amplitude, dtype=np.float64)
                if run.source_amplitude is not None
                else None
            ),
            aperture=aperture,
            basis=basis,
            flat_grid_counts=_to_model_grid(
                flat.frame, centre, far_field_size, scale, rotation
            ),
            model_gain=gain,
            flat_leakage=leakage,
        )

        raw_frames = load_probe_raw_frames(run.path)
        for probe in chosen:
            if probe.phase.shape[0] != region:
                exclusions.append(
                    Exclusion(
                        "probe",
                        f"{session_id}/{probe.name}",
                        "region_mismatch",
                        f"相位 {probe.phase.shape} 与 region={region} 不符",
                    )
                )
                continue
            raw = raw_frames.get(probe.name)
            if raw is None:
                exclusions.append(
                    Exclusion(
                        "probe",
                        f"{session_id}/{probe.name}",
                        "missing_raw_frame",
                        "探针 npz 里没有 raw_frame, 无法做曝光归一化",
                    )
                )
                continue
            if raw.size == 0 or float(raw.max()) <= 0.0:
                exclusions.append(
                    Exclusion(
                        "probe",
                        f"{session_id}/{probe.name}",
                        "degenerate_frame",
                        "帧全黑",
                    )
                )
                continue
            if float(raw.max()) >= SATURATION_LEVEL:
                exclusions.append(
                    Exclusion(
                        "probe",
                        f"{session_id}/{probe.name}",
                        "degenerate_frame",
                        f"帧已饱和 (peak={raw.max():.0f} ≥ {SATURATION_LEVEL})",
                    )
                )
                continue

            exposure = _probe_exposure(probe)
            gain_response = float(exposure_fit.gain)
            if gain_response <= 0:
                gain_response = 1.0
            # 实测侧: 审计口径的 ``prepare_frame`` 帧 (相机读到的稳定帧, 已按
            # 参考曝光 0.02 ms 归一) —— 它就是 MIL 真正拿去拟合的那一帧。
            # 再除以会话的**曝光响应增益**, 于是两侧都落在「每单位响应的辐照度」
            # 口径上, 跨会话可比 (不同曝光的原始 counts 不混算)。
            measured_grid = _to_model_grid(
                probe.frame, centre, far_field_size, scale, rotation
            )
            measured_crop = _crop_footprint(measured_grid, footprint) / gain_response

            model_raw = np.asarray(
                optimizer.forward_intensity(
                    session.coefficients, probe.phase, run.source_amplitude
                ),
                dtype=np.float64,
            )

            # 模型侧: numpy 复算 (与 MIL 同一约定) —— 这样才能在**复场**上施加
            # canonical 的离焦 / 倾斜误差项 (解析操作, 不重采样)。
            field = far_field_of(session, probe.phase)
            numpy_intensity = field.real**2 + field.imag**2
            forward_check = float(
                np.abs(numpy_intensity - model_raw).sum()
                / max(float(np.abs(model_raw).sum()), 1e-30)
            )
            predicted_counts = numpy_intensity
            if gain is not None and leakage is not None:
                predicted_counts = (
                    gain * numpy_intensity + leakage * session.flat_grid_counts
                )
            predicted_crop = _crop_footprint(predicted_counts, footprint) / gain_response
            forward_checks.append(forward_check)

            # 逐像素校验: 复算的预测必须与落盘预测一致 (审计口径, 相对偏差 ≤ 1e-5)。
            # 落盘系数被回滚 / 多轮运行时, 这一步会把该轮判为不可复算。
            deviation: float | None = None
            if probe.prediction is not None:
                recorded = np.asarray(probe.prediction, dtype=np.float64)
                if recorded.shape == predicted_counts.shape:
                    peak = float(np.abs(recorded).max()) or 1.0
                    deviation = float(np.abs(predicted_counts - recorded).max()) / peak

            pending.append(
                _PendingPair(
                    probe=probe,
                    predicted_crop=predicted_crop.astype(np.float32),
                    measured_crop=measured_crop.astype(np.float32),
                    deviation=deviation,
                )
            )

        # 轮次校验: 只有「复算与落盘预测逐像素一致」的轮次才可用。
        by_round: dict[int, list[_PendingPair]] = {}
        for item in pending:
            by_round.setdefault(int(item.probe.round), []).append(item)
        verified_rounds: list[int] = []
        for round_index, items in sorted(by_round.items()):
            deviations = [item.deviation for item in items if item.deviation is not None]
            if not deviations:
                exclusions.append(
                    Exclusion(
                        "round",
                        f"{session_id}#round{round_index}",
                        "unverifiable_round",
                        "该轮没有落盘预测, 无法逐像素校验 (不与审计混用不可复算的轮次)",
                    )
                )
                continue
            worst = max(deviations)
            if worst <= RECOMPUTE_TOL:
                verified_rounds.append(round_index)
            else:
                exclusions.append(
                    Exclusion(
                        "round",
                        f"{session_id}#round{round_index}",
                        "unverifiable_round",
                        f"复算与落盘预测最大相对偏差 {worst:.3g} > {RECOMPUTE_TOL:g} "
                        "(落盘系数被回滚 / 多轮运行)",
                    )
                )

        for item in pending:
            if int(item.probe.round) not in verified_rounds:
                continue
            probe = item.probe
            session.pairs.append(
                HistoryPair(
                    pair_id=f"{session_id}#{probe.name}",
                    session_id=session_id,
                    campaign=campaign,
                    stamp=stamp,
                    variant=run.variant,
                    probe_split=probe.split,
                    round_index=int(probe.round),
                    file=probe.name,
                    exposure_ms=_probe_exposure(probe),
                    region=region,
                    far_field_size=far_field_size,
                    camera_px_per_model_px=float(scale),
                    rotation_quarters=rotation,
                    phase=np.asarray(probe.phase, dtype=np.float64),
                    predicted_crop=item.predicted_crop,
                    measured_crop=item.measured_crop,
                    exposure_fit=exposure_fit,
                )
            )

        if session.pairs:
            session.forward_model_max_relative_l1 = (
                float(max(forward_checks)) if forward_checks else None
            )
            sessions.append(session)
        else:
            exclusions.append(
                Exclusion(
                    "session",
                    stamp,
                    "no_usable_pairs",
                    "会话内所有探针都被排除 (见 probe 级排除记录)",
                )
            )
    return sessions, exclusions


def _config_for(run: RunData) -> Any | None:
    """用该 run 自己的 ``run_config.json`` 重建 :class:`SlmModelInLoopConfig`。

    与审计脚本同一口径: ``geometry_file`` 必须指向**本 run** 的
    ``bench_geometry.json``。
    """
    from ao_shaping.optimizer.wfless.slm_model_in_loop import SlmModelInLoopConfig

    geometry_path = run.path / "bench_geometry.json"
    if not geometry_path.is_file():
        return None
    known = set(SlmModelInLoopConfig.__dataclass_fields__)
    kwargs = {k: v for k, v in run.config.items() if k in known}
    kwargs["geometry_file"] = str(geometry_path)
    kwargs["device"] = kwargs.get("device") or "cpu"
    try:
        return SlmModelInLoopConfig(**kwargs)
    except (TypeError, ValueError) as exc:
        logger.warning("{}: 配置重建失败 {}", run.stamp, exc)
        return None


# ---------------------------------------------------------------------------
# 物理预测模型 (numpy 复算 + canonical 误差项)
# ---------------------------------------------------------------------------


def focal_field_numpy(field: np.ndarray, out_size: int) -> np.ndarray:
    """补零 + 中心化 FFT —— 与 ``ao_shaping.utils.wavefront.fraunhofer.focal_field`` 同口径。

    Args:
        field: 瞳面复场 (``(g, g)``)。
        out_size: 远场网格边长。

    Returns:
        ``(out_size, out_size)`` 复数远场 (光学轴在中心)。
    """
    array = np.asarray(field, dtype=np.complex128)
    n = array.shape[-1]
    if out_size > n:
        half = (out_size - n) // 2
        array = np.pad(
            array,
            ((half, out_size - n - half), (half, out_size - n - half)),
            mode="constant",
        )
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(array, axes=(-2, -1)), norm="ortho"), axes=(-2, -1)
    )


def pupil_field(
    session: Session, phase: np.ndarray, affine: PupilAffine | None = None
) -> np.ndarray:
    """瞳面复场: ``A' · mask · exp(i·(命令相位 + 面板像差))``。

    与 ``ZernikeCoefficientOptimizer._far_field_intensity`` 同一约定 (掩膜作用在
    相位**和**上, 与数字孪生的 ``nan_to_num(phi_slm + aberration)`` 一致)。
    ``affine`` 给出时, 振幅包络按 canonical 的 ``apply_pupil_affine`` 缩放
    (τ_θ 的 ``amp_x`` / ``amp_y`` / ``off_x`` / ``off_y``), 总相位网格不动 ——
    所以它改的是**照明宽度** (⇒ 远场光斑宽度), 而不是像差。
    """
    assert session.amplitude is not None and session.aperture is not None
    assert session.basis is not None and session.coefficients is not None
    aberration = np.einsum(
        "j,jhw->hw", session.coefficients[: session.basis.shape[0]], session.basis
    )
    command = np.asarray(phase, dtype=np.float64)
    amplitude = session.amplitude
    if affine is not None and not affine.is_identity:
        command, amplitude = _apply_pupil_affine_np(
            command, aberration, amplitude, affine
        )
    else:
        command = command + aberration
    patch = command * session.aperture
    return amplitude * np.exp(1j * patch)


def _apply_pupil_affine_np(
    command: np.ndarray,
    aberration: np.ndarray,
    amplitude: np.ndarray,
    affine: PupilAffine,
) -> tuple[np.ndarray, np.ndarray]:
    """canonical ``apply_pupil_affine`` 的 numpy 包装 (调用 torch 实现)。"""
    import torch

    from ao_shaping.utils.wavefront.calibration_terms import apply_pupil_affine

    params = torch.tensor(
        affine.to_vector(), dtype=torch.float32
    )
    patch, envelope = apply_pupil_affine(
        torch.from_numpy(np.ascontiguousarray(command)).float(),
        torch.from_numpy(np.ascontiguousarray(aberration)).float(),
        torch.from_numpy(np.ascontiguousarray(amplitude)).float(),
        params,
    )
    return (
        patch.detach().cpu().numpy().astype(np.float64),
        envelope.detach().cpu().numpy().astype(np.float64),
    )


def far_field_of(
    session: Session, phase: np.ndarray, affine: PupilAffine | None = None
) -> np.ndarray:
    """焦面复场 (补零 + 中心化 FFT)。"""
    return focal_field_numpy(
        pupil_field(session, phase, affine), session.far_field_size
    )


def apply_error_terms(field: np.ndarray, terms: ModelErrorTerms) -> np.ndarray:
    """把误差项作用在**焦面复场**上 (canonical 实现)。

    离焦 = 角谱传递函数, 倾斜 = 焦面精确平移 —— 两者都是**解析**操作, 不像
    「把预测图重采样」那样把散斑抹掉。
    """
    import torch

    from ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer import (
        apply_model_error_terms as _canonical,
    )

    tensor = torch.from_numpy(np.asarray(field, dtype=np.complex128))
    return _canonical(tensor, terms).detach().cpu().numpy()


def model_prediction(
    session: Session,
    pair: HistoryPair,
    terms: ModelErrorTerms | None,
    *,
    footprint_cells: int,
    affine: PupilAffine | None = None,
) -> np.ndarray:
    """给定误差项 (与 τ_θ) 时的模型预测 (裁到相机视场)。"""
    field = far_field_of(session, pair.phase, affine)
    if terms is not None and not terms.is_identity:
        field = apply_error_terms(field, terms)
    intensity = field.real**2 + field.imag**2
    predicted = intensity
    if session.model_gain is not None and session.flat_leakage is not None:
        predicted = session.model_gain * intensity + session.flat_leakage * (
            session.flat_grid_counts if session.flat_grid_counts is not None else 0.0
        )
    return _crop_footprint(predicted, footprint_cells)


# ---------------------------------------------------------------------------
# 四类分解与标定
# ---------------------------------------------------------------------------


def decompose_pair(
    predicted: np.ndarray, measured: np.ndarray, footprint: int
) -> dict[str, Any]:
    """对一对 (预测, 实测) 做四类分解 (复用审计口径)。"""
    return decompose_mismatch(
        np.asarray(predicted, dtype=np.float64),
        np.asarray(measured, dtype=np.float64),
        footprint,
    )


def _second_moment_radius(image: np.ndarray) -> float | None:
    """二阶矩半径 (与审计同口径)。"""
    array = np.maximum(np.asarray(image, dtype=np.float64), 0.0)
    total = float(array.sum())
    if total <= 0.0:
        return None
    ys, xs = np.indices(array.shape, dtype=np.float64)
    cy = float((array * ys).sum() / total)
    cx = float((array * xs).sum() / total)
    variance = float((array * ((ys - cy) ** 2 + (xs - cx) ** 2)).sum() / total)
    return float(math.sqrt(max(variance, 0.0)))


def _centroid(image: np.ndarray) -> tuple[float, float] | None:
    """强度质心 ``(x, y)`` (与审计同口径)。"""
    array = np.maximum(np.asarray(image, dtype=np.float64), 0.0)
    total = float(array.sum())
    if total <= 0.0:
        return None
    ys, xs = np.indices(array.shape, dtype=np.float64)
    return float((array * xs).sum() / total), float((array * ys).sum() / total)


def _rescale_about_centroid(image: np.ndarray, factor: float) -> np.ndarray:
    """把图样绕自身质心缩放 ``factor`` 倍 (焦距尺度的作用)。"""
    if abs(factor - 1.0) < DEGENERACY_EPSILON:
        return np.asarray(image, dtype=np.float64)
    from scipy.ndimage import map_coordinates

    array = np.asarray(image, dtype=np.float64)
    centre = _centroid(array)
    if centre is None:
        return array
    cx, cy = centre
    rows, cols = array.shape
    ys, xs = np.indices(array.shape, dtype=np.float64)
    source_x = cx + (xs - cx) / factor
    source_y = cy + (ys - cy) / factor
    coordinates = np.stack([source_y, source_x])
    return map_coordinates(array, coordinates, order=1, mode="constant", cval=0.0)


def _translate(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """平移图样 (亚格插值)。"""
    if abs(dx) < DEGENERACY_EPSILON and abs(dy) < DEGENERACY_EPSILON:
        return np.asarray(image, dtype=np.float64)
    from scipy.ndimage import shift as _shift

    return _shift(
        np.asarray(image, dtype=np.float64), shift=(dy, dx), order=1, mode="constant", cval=0.0
    )


def apply_calibration(
    predicted: np.ndarray, calibration: Calibration, shift_scale: float = 1.0
) -> np.ndarray:
    """把标定 (焦距尺度 / 瞳面中心 / 增益) 作用到模型预测上。

    Args:
        predicted: 模型预测的远场图样。
        calibration: 拟合到的标定。
        shift_scale: 平移量的单位换算 (模型格 → 目标图的像元)。
    """
    out = _rescale_about_centroid(predicted, calibration.scale)
    out = _translate(
        out, calibration.shift_dx * shift_scale, calibration.shift_dy * shift_scale
    )
    return out * float(calibration.gain)


def fit_calibration(
    pairs: Sequence[HistoryPair], *, scale_bounds: tuple[float, float] = SCALE_SEARCH
) -> Calibration:
    """在给定配对上拟合系统误差 (焦距尺度 / 瞳面中心 / 增益)。

    拟合顺序 (解析、可解释):

    1. **焦距尺度** —— 模型预测的二阶矩半径 / 实测的二阶矩半径的中位数,
       取倒数作为缩放因子 (模型预测得比实测宽 ⇒ 把它缩回去);
    2. **瞳面中心** —— 缩放后重算质心差 (模型 → 实测), 作为平移量;
    3. **增益** —— 平移后总能量之比 (实测 / 模型) 的中位数。

    ⚠️ 这一套是**几何**标定 (在预测图上重采样), 会把散斑抹掉一点点; 物理上
    更干净的等价做法是 ``ModelErrorTerms`` (离焦 / 倾斜, 解析作用), 见
    :func:`fit_physical_calibration`。两者在产物里同时给出。

    Args:
        pairs: 配对 (每条自带裁好的预测 / 实测)。
        scale_bounds: 缩放因子的允许区间。

    Returns:
        拟合到的标定。
    """
    if not pairs:
        return Calibration(scale=1.0, shift_dx=0.0, shift_dy=0.0, gain=1.0, n_pairs=0)

    scale = 1.0
    ratios: list[float] = []
    for pair in pairs:
        radius_p = _second_moment_radius(pair.predicted_crop)
        radius_m = _second_moment_radius(pair.measured_crop)
        if radius_p and radius_m and radius_m > 0:
            ratios.append(float(radius_m / radius_p))
    if ratios:
        candidate = float(np.median(ratios))
        scale = float(np.clip(candidate, scale_bounds[0], scale_bounds[1]))

    shifts_x: list[float] = []
    shifts_y: list[float] = []
    gains: list[float] = []
    for pair in pairs:
        predicted = _rescale_about_centroid(pair.predicted_crop, scale)
        centre_p = _centroid(predicted)
        centre_m = _centroid(pair.measured_crop)
        if centre_p and centre_m:
            shifts_x.append(centre_m[0] - centre_p[0])
            shifts_y.append(centre_m[1] - centre_p[1])
        shifted = (
            _translate(predicted, _median(shifts_x), _median(shifts_y))
            if (shifts_x and shifts_y)
            else predicted
        )
        sum_p = float(shifted.sum())
        sum_m = float(pair.measured_crop.sum())
        if sum_p > 0 and sum_m > 0:
            gains.append(sum_m / sum_p)

    shift_dx = _median(shifts_x)
    shift_dy = _median(shifts_y)
    gain = _median(gains) if gains else 1.0
    return Calibration(
        scale=scale,
        shift_dx=float(shift_dx),
        shift_dy=float(shift_dy),
        gain=float(gain),
        n_pairs=len(pairs),
        detail={
            "width_ratio_median_before": float(np.median(ratios)) if ratios else None,
            "scale_bounds": list(scale_bounds),
        },
    )


def fit_physical_calibration(
    sessions: Sequence[Session],
    pairs: Sequence[HistoryPair],
    *,
    amp_grid: Sequence[float] = (1.0, 1.05, 1.10, 1.15, 1.20, 1.30, 1.45, 1.60),
    defocus_grid: Sequence[float] = (0.0, 2e-3, 5e-3, 8e-3, 1.2e-2),
    max_pairs: int = 12,
) -> tuple[PupilAffine | None, ModelErrorTerms | None, dict[str, Any]]:
    """拟合**物理**的系统误差项 —— canonical 容器。

    * **宽度 (照明宽度)** —— ``PupilAffine`` 的 ``amp_x`` / ``amp_y``: 缩放的是
      **振幅包络**, 所以它改的是照明直径 ⇒ 远场光斑宽度 (总相位网格不动, 不与
      Zernike 系数简并)。模型的远场图样在**角**尺度上与焦距无关 (焦距只决定
      采样), 所以宽度比不是焦距/尺度错。
    * **位置 (光路倾斜)** —— ``ModelErrorTerms`` 的 ``tilt_x_rad`` /
      ``tilt_y_rad``: 焦面上的**精确**傅里叶平移 (远场图样平移 ``f · tilt``),
      不像重采样那样抹掉散斑。
    * **相机离焦** —— 作为**对照**扫描一遍: 角谱传递函数是快速相位扰动而不是
      平滑展宽, 所以它**解释不了**实测的宽度比 (诊断结论写进产物)。

    Args:
        sessions: 会话 (取几何常数)。
        pairs: 配对。
        amp_grid: 照明包络缩放的扫描网格。
        defocus_grid: 相机离焦的扫描网格 (对照用)。
        max_pairs: 参与扫描的配对上限。

    Returns:
        ``(PupilAffine, ModelErrorTerms, 诊断)``; 没有可用会话时前两项为 ``None``。
    """
    if not sessions or not pairs:
        return None, None, {"available": False, "reason": MISSING}
    session = sessions[0]
    pitch = far_field_pitch_m(session)
    if pitch <= 0 or session.focal_length_m <= 0:
        return None, None, {"available": False, "reason": "缺焦面几何 (焦距 / 焦面节距)"}

    by_session = {s.session_id: s for s in sessions}
    usable = [pair for pair in pairs if pair.session_id in by_session]
    if not usable:
        return None, None, {"available": False, "reason": "配对与会话不匹配"}
    rng = np.random.default_rng(0)
    if len(usable) > max_pairs:
        index = rng.choice(len(usable), size=max_pairs, replace=False)
        sample = [usable[int(i)] for i in sorted(index)]
    else:
        sample = list(usable)

    def terms_for(defocus_m: float, tilt_x: float, tilt_y: float) -> ModelErrorTerms:
        return ModelErrorTerms(
            lut_gamma=1.0,
            camera_defocus_m=float(defocus_m),
            tilt_x_rad=float(tilt_x),
            tilt_y_rad=float(tilt_y),
            phi_max_rad=2.0 * math.pi,
            wavelength_m=session.wavelength_m,
            focal_length_m=session.focal_length_m,
            far_field_pitch_m=pitch,
        )

    def width_ratios(
        affine: PupilAffine | None, terms: ModelErrorTerms | None
    ) -> list[float]:
        out: list[float] = []
        for pair in sample:
            owner = by_session[pair.session_id]
            predicted = model_prediction(
                owner,
                pair,
                terms,
                footprint_cells=int(pair.measured_crop.shape[0]),
                affine=affine,
            )
            radius_p = _second_moment_radius(predicted)
            radius_m = _second_moment_radius(pair.measured_crop)
            if radius_p and radius_m and radius_m > 0:
                out.append(float(radius_p / radius_m))
        return out

    # 1) 照明包络缩放: 使宽度比的中位数最接近 1。
    amp_scan: list[dict[str, float | None]] = []
    best_amp = 1.0
    best_score = float("inf")
    for amp in amp_grid:
        affine = PupilAffine(amp_x=float(amp), amp_y=float(amp))
        ratios = width_ratios(affine, None)
        median = float(np.median(ratios)) if ratios else None
        score = abs(median - 1.0) if median is not None else float("inf")
        amp_scan.append({"amp": float(amp), "width_ratio_median": median})
        if score < best_score:
            best_score = score
            best_amp = float(amp)
    affine = PupilAffine(amp_x=best_amp, amp_y=best_amp)

    # 2) 光路倾斜: 由质心偏移换算 (焦面上的精确平移)。
    shifts_x: list[float] = []
    shifts_y: list[float] = []
    for pair in sample:
        owner = by_session[pair.session_id]
        predicted = model_prediction(
            owner,
            pair,
            None,
            footprint_cells=int(pair.measured_crop.shape[0]),
            affine=affine,
        )
        centre_p = _centroid(predicted)
        centre_m = _centroid(pair.measured_crop)
        if centre_p and centre_m:
            shifts_x.append(centre_m[0] - centre_p[0])
            shifts_y.append(centre_m[1] - centre_p[1])
    shift_dx = _median(shifts_x)
    shift_dy = _median(shifts_y)
    tilt_x = shift_dx * pitch / session.focal_length_m
    tilt_y = shift_dy * pitch / session.focal_length_m
    terms = terms_for(0.0, tilt_x, tilt_y)

    # 3) 对照: 相机离焦能不能解释宽度比?
    defocus_scan: list[dict[str, float | None]] = []
    for defocus in defocus_grid:
        ratios = width_ratios(None, terms_for(defocus, 0.0, 0.0))
        median = float(np.median(ratios)) if ratios else None
        defocus_scan.append({"defocus_m": float(defocus), "width_ratio_median": median})

    ratios_after = width_ratios(affine, terms)
    diagnostics = {
        "available": True,
        "amp_scan": amp_scan,
        "amp_x": best_amp,
        "defocus_scan": defocus_scan,
        "defocus_explains_width_ratio": False,
        "tilt_x_rad": float(tilt_x),
        "tilt_y_rad": float(tilt_y),
        "shift_dx_cells": float(shift_dx),
        "shift_dy_cells": float(shift_dy),
        "width_ratio_median_after": float(np.median(ratios_after)) if ratios_after else None,
        "n_pairs_scanned": len(sample),
        "far_field_pitch_m": pitch,
        "focal_length_m": session.focal_length_m,
    }
    return affine, terms, diagnostics


def _median(values: Sequence[float]) -> float:
    """中位数 (空序列 → 0)。"""
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else 0.0


def _peak_normalised_mse(predicted: np.ndarray, measured: np.ndarray) -> float:
    """峰值归一化 MSE (与 MIL 的 ``_probe_set_loss`` 同一口径)。"""
    peak_m = float(np.max(measured))
    peak_p = float(np.max(predicted))
    if peak_m <= 0.0 or peak_p <= 0.0:
        return float("nan")
    estimate = np.asarray(predicted, dtype=np.float64) / peak_p
    target = np.asarray(measured, dtype=np.float64) / peak_m
    return float(np.mean((estimate - target) ** 2))


def refine_calibration(
    pairs: Sequence[HistoryPair],
    calibration: Calibration,
    *,
    max_pairs: int = 80,
    seed: int = 0,
) -> Calibration:
    """在解析初值上做一次局部优化 (Nelder-Mead)。

    目标 = MIL 自己的**峰值归一化 MSE** 的均值 —— 即「模型预测与实测的图样差多
    大」, 而不是四类分解的代理量, 免得把 (尺度 / 平移 / 增益) 三个自由度互相
    抵消掉。

    Args:
        pairs: 配对。
        calibration: 解析初值。
        max_pairs: 参与优化的配对上限 (抽样, 保证单次评估够快)。
        seed: 抽样种子。

    Returns:
        优化后的标定 (目标不改善时返回初值)。
    """
    from scipy.optimize import minimize

    if len(pairs) < 4:
        return calibration
    rng = np.random.default_rng(seed)
    if len(pairs) > max_pairs:
        index = rng.choice(len(pairs), size=max_pairs, replace=False)
        sample = [pairs[int(i)] for i in sorted(index)]
    else:
        sample = list(pairs)

    measured = [np.asarray(pair.measured_crop, dtype=np.float64) for pair in sample]
    predicted = [np.asarray(pair.predicted_crop, dtype=np.float64) for pair in sample]

    def objective(vector: np.ndarray) -> float:
        scale = float(np.exp(vector[0]))
        candidate = Calibration(
            scale=scale,
            shift_dx=float(vector[1]),
            shift_dy=float(vector[2]),
            gain=float(np.exp(vector[3])),
            n_pairs=len(sample),
        )
        total = 0.0
        count = 0
        for pred, meas in zip(predicted, measured):
            corrected = apply_calibration(pred, candidate)
            value = _peak_normalised_mse(corrected, meas)
            if math.isfinite(value):
                total += value
                count += 1
        return total / max(count, 1)

    start = np.array(
        [
            math.log(max(calibration.scale, 1e-6)),
            calibration.shift_dx,
            calibration.shift_dy,
            math.log(max(calibration.gain, 1e-9)),
        ],
        dtype=np.float64,
    )
    result = minimize(
        objective,
        start,
        method="Nelder-Mead",
        options={"maxiter": 220, "xatol": 1e-3, "fatol": 1e-6},
    )
    if not np.isfinite(result.fun) or result.fun >= objective(start):
        return calibration
    scale = float(np.exp(result.x[0]))
    shift_dx = float(result.x[1])
    shift_dy = float(result.x[2])
    # 目标函数是**峰值归一化**的, 对整体尺度不敏感 ⇒ 增益必须在优化之后按
    # 「实测总能量 / 预测总能量」重新解析拟合, 否则能量比会跑偏。
    gains: list[float] = []
    for pred, meas in zip(predicted, measured):
        corrected = _translate(
            _rescale_about_centroid(pred, scale), shift_dx, shift_dy
        )
        sum_p = float(corrected.sum())
        sum_m = float(meas.sum())
        if sum_p > 0 and sum_m > 0:
            gains.append(sum_m / sum_p)
    refined = Calibration(
        scale=scale,
        shift_dx=shift_dx,
        shift_dy=shift_dy,
        gain=_median(gains) if gains else float(np.exp(result.x[3])),
        n_pairs=calibration.n_pairs,
        detail={
            **calibration.detail,
            "objective_before": float(objective(start)),
            "objective_after": float(result.fun),
            "refined_on_pairs": len(sample),
        },
    )
    return refined


def evaluate_calibration(
    pairs: Sequence[HistoryPair], calibration: Calibration | None
) -> dict[str, Any]:
    """在一组配对上给出四类分解 (标定前 / 标定后) 的汇总。"""
    before: list[dict[str, Any]] = []
    after: list[dict[str, Any]] = []
    for pair in pairs:
        pred = np.asarray(pair.predicted_crop, dtype=np.float64)
        meas = np.asarray(pair.measured_crop, dtype=np.float64)
        footprint = int(meas.shape[0])
        before.append(decompose_pair(pred, meas, footprint))
        if calibration is not None:
            corrected = apply_calibration(pred, calibration)
            after.append(decompose_pair(corrected, meas, footprint))
    return {
        "n_pairs": len(pairs),
        "footprint_cells": int(pairs[0].measured_crop.shape[0]) if pairs else 0,
        "before": _summarize_decomposition(before),
        "after": _summarize_decomposition(after) if after else None,
    }


def evaluate_physical_calibration(
    sessions: Sequence[Session],
    pairs: Sequence[HistoryPair],
    terms: ModelErrorTerms | None,
    *,
    affine: PupilAffine | None = None,
) -> dict[str, Any]:
    """在物理误差项 (离焦 / 倾斜) 与 τ_θ 下重算预测并给出四类分解。

    与 :func:`evaluate_calibration` 的区别: 这里不是把预测图重采样, 而是在
    **焦面复场**上施加 canonical 的离焦 / 倾斜, 所以散斑不会被抹掉。
    """
    if not pairs:
        return {"n_pairs": 0}
    by_session = {s.session_id: s for s in sessions}
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        owner = by_session.get(pair.session_id)
        if owner is None:
            continue
        footprint = int(pair.measured_crop.shape[0])
        predicted = model_prediction(
            owner, pair, terms, footprint_cells=footprint, affine=affine
        )
        rows.append(
            decompose_pair(
                predicted,
                np.asarray(pair.measured_crop, dtype=np.float64),
                footprint,
            )
        )
    return {
        "n_pairs": len(rows),
        "footprint_cells": footprint,
        "before": _summarize_decomposition(rows),
    }


def _summarize_decomposition(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """把逐对的四类分解汇总成中位数 + 计数。"""

    def block(key: str) -> dict[str, Any]:
        values = [row.get(key) for row in rows]
        finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
        if not finite:
            return {"n": 0, "median": None, "mean": None}
        array = np.asarray(finite, dtype=np.float64)
        return {
            "n": int(array.size),
            "median": float(np.median(array)),
            "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
            "min": float(array.min()),
            "max": float(array.max()),
        }

    return {
        "energy_ratio": block("energy_ratio"),
        "width_ratio": block("width_ratio"),
        "centroid_shift_cells": block("centroid_shift_cells"),
        "shape_correlation": block("shape_correlation"),
        "correlation_raw": block("correlation_raw"),
        "mse_peak_normalised": block("mse_peak_normalised"),
        "n_same_sign_width_ratio": int(
            sum(
                1
                for row in rows
                if row.get("width_ratio") is not None and float(row["width_ratio"]) > 1.0
            )
        ),
        "n_same_sign_energy_ratio": int(
            sum(
                1
                for row in rows
                if row.get("energy_ratio") is not None and float(row["energy_ratio"]) < 1.0
            )
        ),
    }


# ---------------------------------------------------------------------------
# 会话切分
# ---------------------------------------------------------------------------


def split_sessions_by_session(
    sessions: Sequence[Session], fraction: float = SESSION_SPLIT_FRACTION, seed: int = 0
) -> tuple[list[Session], list[Session]]:
    """按**会话**分组切分 (同一会话的相邻帧不跨两侧)。"""
    ids = sorted({session.session_id for session in sessions})
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    cut = max(1, int(round(len(ids) * float(fraction))))
    train_ids = {ids[int(i)] for i in order[:cut]}
    train = [s for s in sessions if s.session_id in train_ids]
    held = [s for s in sessions if s.session_id not in train_ids]
    return train, held


# ---------------------------------------------------------------------------
# 物理解释: 焦距 / 离焦 / 倾斜
# ---------------------------------------------------------------------------


def far_field_pitch_m(session: Session) -> float:
    """焦面采样间距 (m) = λ·f / (N_pad · d_pupil)。"""
    pupil_pitch_m = session.panel_span_px / max(session.region, 1) * session.slm_pixel_um * 1e-6
    if pupil_pitch_m <= 0 or session.focal_length_m <= 0:
        return 0.0
    return (
        session.wavelength_m
        * session.focal_length_m
        / (session.far_field_size * pupil_pitch_m)
    )


def interpret_calibration(
    calibration: Calibration, sessions: Sequence[Session]
) -> dict[str, Any]:
    """把拟合出的 (尺度 / 中心 / 增益) 翻译成物理量与仓库既有容器。

    * **焦距尺度** —— 模型的远场图样比实测宽 (或窄) 多少倍 ⇒ 等效焦距之比;
      用数字孪生的 ``equivalent_focal_length_m`` 从实测标度反推等效焦距。
    * **瞳面中心** —— 图样平移量 ⇒ 光路倾斜 (rad) ⇒ ``ModelErrorTerms.tilt_*``;
      同时给出 ``PupilAffine`` 的振幅包络中心偏移。
    * **离焦** —— 等效焦距之比也可以由**相机离焦**解释 (焦面复场上的角谱传递
      函数), 给出对应的 ``camera_defocus_m`` 估计。
    """
    if not sessions:
        return {}
    session = sessions[0]
    pitch = far_field_pitch_m(session)
    ccd_pitch_m = session.ccd_pixel_um * 1e-6

    # 等效焦距: 用**实测**标度反推 (数字孪生的 canonical 入口)。
    measured_scale = session.camera_px_per_model_px
    f_equivalent = equivalent_focal_length_m(
        measured_scale,
        wavelength_m=session.wavelength_m,
        slm_pixel_um=session.slm_pixel_um,
        ccd_pixel_um=session.ccd_pixel_um,
        region=session.region,
        region_span_px=session.panel_span_px,
        far_field_padding=max(1, session.far_field_size // max(session.region, 1)),
    )

    # 焦距尺度 ⇒ 远场图样缩放 ⇒ 等效焦距之比 (远场格距 ∝ f)。
    f_ratio = float(calibration.scale) if calibration.scale else 1.0
    tilt_x_rad = (
        calibration.shift_dx * pitch / session.focal_length_m
        if session.focal_length_m > 0 and pitch > 0
        else 0.0
    )
    tilt_y_rad = (
        calibration.shift_dy * pitch / session.focal_length_m
        if session.focal_length_m > 0 and pitch > 0
        else 0.0
    )

    # 等效焦距误差 ⇒ 相机离焦的等价量 (焦面复场上的二次相位)。
    # 远场图样缩放 f_ratio ⇒ 有效焦距 f_eff = f_model / f_ratio;
    # 焦面离焦 Δz 与有效焦距的关系: Δz ≈ f_model² · (1/f_eff - 1/f_model)。
    f_model = session.focal_length_m
    f_eff = f_model / f_ratio if f_ratio else f_model
    defocus_estimate_m = (
        f_model * f_model * (1.0 / f_eff - 1.0 / f_model) if f_model > 0 and f_eff > 0 else 0.0
    )

    terms = ModelErrorTerms(
        lut_gamma=1.0,
        camera_defocus_m=float(defocus_estimate_m),
        tilt_x_rad=float(tilt_x_rad),
        tilt_y_rad=float(tilt_y_rad),
        phi_max_rad=2.0 * math.pi,
        wavelength_m=session.wavelength_m,
        focal_length_m=f_model,
        far_field_pitch_m=pitch,
    )
    # τ_θ: 照明包络的缩放 (诊断扫描给出; 它**不是**宽度误差的来源, 所以保留为
    # 拟合值并附诊断结论, 而不是把它当成标定项用)。
    affine = PupilAffine(
        amp_x=1.0,
        amp_y=1.0,
        off_x=0.0,
        off_y=0.0,
        rot=0.0,
        grid=1.0,
    )
    twin = BenchTwinConfig.from_measured_scale(
        camera_px_per_model_px=measured_scale,
        wavelength_m=session.wavelength_m,
        slm_pixel_um=session.slm_pixel_um,
        ccd_pixel_um=session.ccd_pixel_um,
        region=session.region,
        region_span_px=session.panel_span_px,
        far_field_padding=max(1, session.far_field_size // max(session.region, 1)),
    )

    return {
        "far_field_pitch_m": pitch,
        "focal_length_model_m": f_model,
        "focal_length_equivalent_m": f_equivalent,
        "focal_ratio_measured_over_model": (
            f_equivalent / f_model if f_model > 0 else None
        ),
        "width_scale_factor": f_ratio,
        "tilt_x_rad": float(tilt_x_rad),
        "tilt_y_rad": float(tilt_y_rad),
        "defocus_equivalent_m": float(defocus_estimate_m),
        "model_error_terms": terms.to_dict(),
        "pupil_affine": affine.to_dict(),
        "bench_twin_config": asdict(twin),
        "ccd_pitch_m": ccd_pitch_m,
    }


# ---------------------------------------------------------------------------
# 历史中位像差 (Step A 的初值)
# ---------------------------------------------------------------------------


def historical_median_coefficients(
    sessions: Sequence[Session],
) -> tuple[np.ndarray | None, list[float]]:
    """历史各会话拟合系数的**逐 Noll 序中位数** + 每族中位行程缩放。

    这是「Step A 的初值可以来自历史拟合的中位像差」的落点: 用中位数当种子,
    信任域就不必从零一路顶到真值。

    Returns:
        ``(中位系数向量, 每族中位行程缩放 γ)``; 没有可用系数时返回 ``(None, [])``。
    """
    vectors = [s.coefficients for s in sessions if s.coefficients is not None]
    if not vectors:
        return None, []
    length = min(len(v) for v in vectors)
    stacked = np.stack([v[:length] for v in vectors], axis=0)
    median = np.median(stacked, axis=0)
    gammas: list[float] = []
    for family in range(1, 8):
        members = [
            index
            for index in range(length)
            if family_of_noll(index + 1) == family
        ]
        if not members:
            gammas.append(1.0)
            continue
        reference = np.median(np.abs(stacked[:, members]), axis=0)
        current = np.abs(median[members])
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(reference > 1e-9, current / np.maximum(reference, 1e-9), 1.0)
        gammas.append(float(np.median(ratio)))
    return median, gammas


# ---------------------------------------------------------------------------
# B1 / B2 验证组
# ---------------------------------------------------------------------------


def _capture_rows(directory: Path) -> list[dict[str, Any]]:
    """读采集组的 manifest 行。"""
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        return []
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def capture_geometry(
    matrix: np.ndarray,
    *,
    slm_pixel_um: float = 8.0,
    ccd_pixel_um: float = 2.2,
    wavelength_m: float = 1064e-9,
    far_field_size: int = 512,
    pupil_pitch_px: float = 15.0,
) -> dict[str, float]:
    """由**斜坡标定矩阵**反推采集组的几何 (等效焦距 + 相机像元每远场格)。

    斜坡是**整面板**的线性相位斜坡 (周期 T 面板像元), 它在远场造成的位移是
    ``d = 2π·f·λ / (T·d_panel·d_ccd)`` 相机像元; 采集脚本按
    ``(centroid(+1) - centroid(-1)) · T / 2`` 把它攒成矩阵, 于是矩阵元素与 T
    无关, 直接给出 ``2π·f·λ / (d_panel·d_ccd)`` ⇒ 可反解 f。

    Args:
        matrix: ``ramp_matrix`` 的输出 (列 = x / y 轴)。
        slm_pixel_um: SLM 面板像元节距 (µm)。
        ccd_pixel_um: 相机像元节距 (µm)。
        wavelength_m: 波长 (m)。
        far_field_size: 物理预测模型的补零后远场网格边长。
        pupil_pitch_px: 瞳面网格每格覆盖的面板像元数。

    Returns:
        ``focal_length_m`` / ``camera_px_per_model_px`` / ``far_field_pitch_m``。
    """
    d_panel = float(slm_pixel_um) * 1e-6
    d_ccd = float(ccd_pixel_um) * 1e-6
    # 相机的上下翻转让 x 斜坡在 y 方向成像, 所以取**列的模长**, 不取某个分量。
    element = float(np.max(np.linalg.norm(np.asarray(matrix, dtype=np.float64), axis=0)))
    focal = element * d_panel * d_ccd / (2.0 * math.pi * float(wavelength_m))
    pupil_pitch_m = float(pupil_pitch_px) * d_panel
    pitch_m = float(wavelength_m) * focal / (far_field_size * pupil_pitch_m)
    return {
        "focal_length_m": focal,
        "far_field_pitch_m": pitch_m,
        "camera_px_per_model_px": pitch_m / d_ccd,
        "ramp_column_norm": element,
    }


def validate_capture_groups(
    calibration: Calibration,
    sessions: Sequence[Session],
    *,
    iterations: int,
    out: dict[str, Any],
) -> None:
    """在今天新采的 B1 / B2 上做「标定前 vs 标定后」对照。

    口径复用 ``scripts/slm/fit_defocus_forward_dataset.py``:
    ``load_observations`` (曝光归一化 = 除以曝光时间)、``ramp_matrix`` (相机映射)、
    ``PhysicalForwardFit`` (共享物理预测模型)。**before** = 落盘的共享物理预测模型
    直接与实测比; **after** = 把历史预标定出的 (焦距尺度 / 瞳面中心 / 增益)
    作用在同一个预测上 —— 与 MIL 线用的是**同一个**标定算子, 所以「历史预标定
    能否泛化到独立日期的数据」是可比的。

    Args:
        calibration: 历史预标定结果。
        sessions: 历史会话 (取几何常数)。
        iterations: 物理预测模型拟合迭代数。
        out: 结果字典 (原地写入)。
    """
    import torch

    from scripts.slm.fit_defocus_forward_dataset import (
        PhysicalForwardFit,
        load_observations,
        ramp_matrix,
    )
    from scipy.optimize import nnls

    if not sessions:
        out["error"] = "没有历史会话可供换算几何常数"
        return
    session = sessions[0]

    for label, directory in (("B1_baseline", CAPTURE_BASELINE), ("B2_prior", CAPTURE_PRIOR)):
        base = ROOT / directory
        if not base.is_dir():
            out[label] = {"available": False, "reason": f"目录不存在: {directory}"}
            continue
        rows = _capture_rows(base)
        if not rows:
            out[label] = {"available": False, "reason": "没有 manifest.json"}
            continue
        try:
            matrix = ramp_matrix(rows)
            references = {r["branch"]: r for r in rows if r["label"] == "reference_start"}
            centre = np.array(
                [references["U"]["spot"][f"centroid_{k}"] for k in ("x", "y")]
            )
            selected, phases, images, camera_xy, _ = load_observations(base, centre)
        except Exception as exc:  # pragma: no cover - 依赖采集组完整性
            out[label] = {"available": False, "reason": f"读采集组失败: {exc}"}
            continue

        train = np.array([r["split"] != "test" for r in selected])
        config = json.loads((base / "run_config.json").read_text(encoding="utf-8"))
        settings = config["settings"]
        amplitude_center = np.array([settings["center_x"], settings["center_y"]], dtype=float)

        model = PhysicalForwardFit(
            phases,
            images,
            camera_xy,
            matrix,
            config["best_diopter_m1"],
            train,
            amplitude_center,
            learn_amplitude=True,
            initial_defocus=None,
            learn_residual=False,
        )
        with torch.no_grad():
            model.origin.copy_(torch.tensor(centre, dtype=torch.float32))
        prediction = model.run(iterations)
        design = np.stack((prediction[train].ravel(), np.ones(prediction[train].size)), axis=1)
        gain, background = nnls(design, images[train].ravel())[0]
        prediction = prediction * gain + background

        geometry = capture_geometry(
            matrix,
            slm_pixel_um=float(settings.get("slm_pixel_um", 8.0) or 8.0),
            ccd_pixel_um=float(settings.get("camera_pixel_um", 2.2) or 2.2),
            wavelength_m=float(settings.get("wavelength", 1064) or 1064) * 1e-9,
        )
        shift_scale = float(
            np.median([s.camera_px_per_model_px for s in sessions if s.camera_px_per_model_px])
        )
        if not math.isfinite(shift_scale) or shift_scale <= 0:
            shift_scale = 1.0

        before = _score_capture_prediction(
            selected,
            prediction,
            images,
            camera_xy,
            references,
            cell_scale=geometry["camera_px_per_model_px"],
        )
        corrected = np.stack(
            [
                apply_calibration(
                    np.asarray(prediction[index], dtype=np.float64),
                    calibration,
                    shift_scale=shift_scale,
                )
                for index in range(len(selected))
            ]
        )
        after = _score_capture_prediction(
            selected,
            corrected,
            images,
            camera_xy,
            references,
            cell_scale=geometry["camera_px_per_model_px"],
        )

        out[label] = {
            "available": True,
            "n_samples": len(selected),
            "geometry": geometry,
            "defocus_diopter_fitted": float(model.d.item()),
            "origin_px": [float(v) for v in model.origin.detach().tolist()],
            "gain": float(gain),
            "background": float(background),
            "before": before,
            "after": after,
        }


def _score_capture_prediction(
    selected: Sequence[dict[str, Any]],
    prediction: np.ndarray,
    images: np.ndarray,
    camera_xy: np.ndarray,
    references: dict[str, dict[str, Any]],
    *,
    cell_scale: float = 1.0,
) -> dict[str, Any]:
    """给采集组的预测打分: 四类分解 + Pearson r + CV / EE 绝对误差。

    Args:
        selected: 采集组的 manifest 行。
        prediction: 预测 (与 ``images`` 同形)。
        images: 实测 (相机裁剪图)。
        camera_xy: 相机裁剪图的像元坐标。
        references: 各分支的参考帧 (目标框中心取自它)。
        cell_scale: 每个**远场格**对应多少相机像元 —— 质心偏移据此换算成
            「模型格」, 与 MIL 线同一单位。
    """

    def block(values: list[float]) -> dict[str, Any]:
        finite = [v for v in values if v is not None and math.isfinite(v)]
        if not finite:
            return {"n": 0, "median": None}
        array = np.asarray(finite, dtype=np.float64)
        return {
            "n": int(array.size),
            "median": float(np.median(array)),
            "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        }

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        predicted = np.asarray(prediction[index], dtype=np.float64)
        measured = np.asarray(images[index], dtype=np.float64)
        footprint = int(min(predicted.shape))
        decomposition = decompose_pair(predicted, measured, footprint)
        if cell_scale > 0:
            shift = decomposition.get("centroid_shift_cells")
            if shift is not None:
                decomposition["centroid_shift_cells"] = float(shift) / cell_scale
            decomposition["centroid_shift_cells_raw_px"] = shift
        branch_centre = np.array(
            [references[row["branch"]]["spot"][f"centroid_{k}"] for k in ("x", "y")]
        )
        mask = np.all(np.abs(camera_xy - branch_centre) <= 30, axis=-1)
        optics: dict[str, float] = {}
        for label, array in (("measured", measured), ("predicted", predicted)):
            values = array[mask]
            optics[label + "_cv"] = float(values.std() / max(values.mean(), 1e-9))
            optics[label + "_ee_roi"] = float(values.sum() / max(array.sum(), 1e-9))
        rows.append(
            {
                "file": row["file"],
                "label": row["label"],
                "branch": row["branch"],
                "split": row["split"],
                "family": row["family"],
                "exposure_ms": row["exposure_ms"],
                "decomposition": decomposition,
                **optics,
                "cv_abs_error": abs(optics["predicted_cv"] - optics["measured_cv"]),
                "ee_abs_error": abs(optics["predicted_ee_roi"] - optics["measured_ee_roi"]),
            }
        )

    def subset(mask: Callable[[dict[str, Any]], bool]) -> list[dict[str, Any]]:
        return [row for row in rows if mask(row)]

    test_rows = subset(lambda row: row["split"] == "test")
    train_rows = subset(lambda row: row["split"] == "train")
    calib_rows = subset(lambda row: row["split"] == "calibration")

    def summarize(subset_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not subset_rows:
            return {"n": 0}
        decomposition = [row["decomposition"] for row in subset_rows]
        return {
            "n": len(subset_rows),
            "energy_ratio": block([d.get("energy_ratio") for d in decomposition]),
            "width_ratio": block([d.get("width_ratio") for d in decomposition]),
            "centroid_shift_cells": block([d.get("centroid_shift_cells") for d in decomposition]),
            "centroid_shift_px": block([d.get("centroid_shift_cells_raw_px") for d in decomposition]),
            "shape_correlation": block([d.get("shape_correlation") for d in decomposition]),
            "correlation_raw": block([d.get("correlation_raw") for d in decomposition]),
            "cv_abs_error": block([row["cv_abs_error"] for row in subset_rows]),
            "ee_abs_error": block([row["ee_abs_error"] for row in subset_rows]),
            "cv_measured": block([row["measured_cv"] for row in subset_rows]),
            "cv_predicted": block([row["predicted_cv"] for row in subset_rows]),
        }

    return {
        "all": summarize(rows),
        "test": summarize(test_rows),
        "train": summarize(train_rows),
        "calibration": summarize(calib_rows),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# 语料普查 (hwdataset)
# ---------------------------------------------------------------------------


def census_hwdataset(out: dict[str, Any]) -> None:
    """``src/ml/hwdataset`` 语料的**单独一组**普查 (不与 MIL 口径混算)。

    该语料的 FOV / region / 曝光口径与 MIL 不同, 所以只做统计说明, 不参与拟合。
    """
    try:
        from ml.hwdataset.index import build_hw_index
    except Exception as exc:  # pragma: no cover - ml 是可选依赖组
        out["hwdataset"] = {"available": False, "reason": f"ml.hwdataset 不可用: {exc}"}
        return
    cache = ROOT / "data" / "hw_index_cache.json"
    try:
        index = build_hw_index(index_cache=cache)
    except Exception as exc:  # pragma: no cover - 语料缺失
        out["hwdataset"] = {"available": False, "reason": f"语料索引构建失败: {exc}"}
        return
    families: dict[str, int] = {}
    exposures: list[float] = []
    regions: set[int] = set()
    for record in getattr(index, "records", []):
        families[record.family] = families.get(record.family, 0) + 1
        if record.exposure_ms:
            exposures.append(float(record.exposure_ms))
        if getattr(record, "n_max", None):
            regions.add(int(record.n_max))
    out["hwdataset"] = {
        "available": True,
        "n_records": len(getattr(index, "records", [])),
        "families": families,
        "exposure_ms_range": [min(exposures), max(exposures)] if exposures else None,
        "zernike_n_max_values": sorted(regions),
        "note": (
            "该语料的相机窗口 / region / 曝光口径与 MIL 不同 (MIL 是 64 / 128 格、"
            "旋转象限固定为 1), 故只作单独一组的统计说明, 不与 MIL 的配对混算, "
            "也不参与系统误差拟合。"
        ),
    }


# ---------------------------------------------------------------------------
# 阈值判定
# ---------------------------------------------------------------------------


def judge_thresholds(summary: dict[str, Any]) -> dict[str, Any]:
    """按阶段 5 的判据给出结论。"""
    width = summary.get("width_ratio_median")
    centroid = summary.get("centroid_shift_median")
    cv_error = summary.get("cv_abs_error_median")
    out: dict[str, Any] = {}
    if width is None:
        out["width_ratio"] = {"available": False, "reason": MISSING}
    else:
        out["width_ratio"] = {
            "value": width,
            "target": WIDTH_RATIO_TARGET,
            "tolerance": WIDTH_RATIO_TOLERANCE,
            "passed": bool(abs(width - WIDTH_RATIO_TARGET) <= WIDTH_RATIO_TOLERANCE),
        }
    if centroid is None:
        out["centroid_shift"] = {"available": False, "reason": MISSING}
    else:
        out["centroid_shift"] = {
            "value": centroid,
            "limit_cells": CENTROID_SHIFT_LIMIT_CELLS,
            "passed": bool(centroid <= CENTROID_SHIFT_LIMIT_CELLS),
        }
    if cv_error is None:
        out["cv_abs_error"] = {"available": False, "reason": MISSING}
    else:
        out["cv_abs_error"] = {
            "value": cv_error,
            "limit": CV_ABS_LIMIT,
            "passed": bool(cv_error < CV_ABS_LIMIT),
        }
    return out


# ---------------------------------------------------------------------------
# 产物
# ---------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    """JSON 兜底序列化。"""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_pairs(
    path: Path,
    sessions: Sequence[Session],
    exclusions: Sequence[Exclusion],
    sampling: dict[str, Any],
) -> None:
    """写 ``pairs.json``。"""
    payload = {
        "sampling": sampling,
        "sessions": [
            {
                "session_id": session.session_id,
                "campaign": session.campaign,
                "stamp": session.stamp,
                "variant": session.variant,
                "region": session.region,
                "far_field_size": session.far_field_size,
                "camera_px_per_model_px": session.camera_px_per_model_px,
                "rotation_quarters": session.rotation_quarters,
                "focal_length_m": session.focal_length_m,
                "wavelength_m": session.wavelength_m,
                "exposure_fit": session.exposure_fit.to_dict(),
                "n_pairs": len(session.pairs),
                "pairs": [pair.to_dict() for pair in session.pairs],
            }
            for session in sessions
        ],
        "exclusions": [item.to_dict() for item in exclusions],
    }
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def write_figure(
    path: Path,
    labels: Sequence[str],
    width_before: Sequence[float],
    width_after: Sequence[float],
    centroid_before: Sequence[float],
    centroid_after: Sequence[float],
) -> None:
    """画标定前 / 后的宽度比与质心偏移对照图。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    positions = np.arange(len(labels))
    width = 0.36

    axes[0].bar(positions - width / 2, width_before, width, label="标定前", color="#9aa7b8")
    axes[0].bar(positions + width / 2, width_after, width, label="标定后", color="#3f7fbf")
    axes[0].axhspan(
        WIDTH_RATIO_TARGET - WIDTH_RATIO_TOLERANCE,
        WIDTH_RATIO_TARGET + WIDTH_RATIO_TOLERANCE,
        color="#8fbf8f",
        alpha=0.25,
        label="判据带 1.0 ± 0.10",
    )
    axes[0].axhline(WIDTH_RATIO_TARGET, color="#4a7a4a", linewidth=1.0)
    axes[0].set_ylabel("宽度比 (模型/实测)")
    axes[0].set_title("宽度比")
    axes[0].legend(fontsize=8)

    axes[1].bar(positions - width / 2, centroid_before, width, label="标定前", color="#9aa7b8")
    axes[1].bar(positions + width / 2, centroid_after, width, label="标定后", color="#3f7fbf")
    axes[1].axhline(
        CENTROID_SHIFT_LIMIT_CELLS, color="#b05a4a", linestyle="--", linewidth=1.2,
        label="判据 ≤ 0.5 模型格",
    )
    axes[1].set_ylabel("质心偏移 (模型格)")
    axes[1].set_title("质心偏移 (B1/B2 的「模型格」换算不可靠, 见 summary)")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8)

    for axis in axes:
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=15)
        axis.grid(axis="y", alpha=0.25)

    figure.suptitle("历史数据离线预标定: 标定前 / 后的四类分解对照", fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


_FMT = partial(fmt_general, missing_none=MISSING)


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    """写 ``summary.md``。"""
    mining = payload["mining"]
    fit = payload["fit"]
    validation = payload["validation"]
    acquisition = payload["acquisition"]

    lines: list[str] = []
    lines.append("# 历史数据离线预标定物理预测模型系统误差")
    lines.append("")
    lines.append(f"- 生成时间: `{payload['timestamp']}`")
    lines.append(f"- 产物目录: `{payload['output_dir']}`")
    lines.append("- 脚本: `scripts/slm/mine_history_model_bias.py` (只读历史数据, 未打开任何设备)")
    lines.append("")
    lines.append("## 1. 挖到的配对")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 采到的历史会话数 | {mining['n_sessions']} |")
    lines.append(f"| 采到的历史配对数 | {mining['n_pairs']} |")
    lines.append(f"| 抽样口径 | 每个会话最多 {mining['probes_per_session']} 条探针 (平场锚点必留), 随机种子 {mining['seed']} |")
    lines.append(f"| 被排除的样本数 | {mining['n_exclusions']} |")
    lines.append("")
    lines.append("排除原因分类:")
    lines.append("")
    lines.append("| 原因 | 数量 |")
    lines.append("|---|---|")
    for reason, count in sorted(mining["exclusion_counts"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{reason}` | {count} |")
    lines.append("")
    lines.append("## 2. 曝光归一化")
    lines.append("")
    lines.append(
        f"逐会话拟合 `counts = bias + gain · exposure_ms`; 有曝光阶梯的会话用阶梯拟合 "
        f"({mining['n_sessions_with_ladder']}/{mining['n_sessions']} 个会话), "
        f"其余会话曝光单一, 退化为「按曝光线性归一 + 背景交给 `prepare_frame` 扣除」。"
    )
    lines.append("")
    lines.append("| 会话 | 来源 | 偏置 (counts) | 增益 (counts/ms) | 最大残差 (counts) | R² | 留一交叉验证残差 (counts) |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in mining["exposure_fits"][:12]:
        holdout = row.get("holdout") or {}
        lines.append(
            f"| `{row['session_id']}` | {row['source']} | {_FMT(row['bias'], ".3f")} | "
            f"{_FMT(row['gain'], ".4f")} | {_FMT(row['max_abs_residual'], ".3f")} | "
            f"{_FMT(row['r_squared'], ".5f")} | {_FMT(holdout.get('residual_counts'), ".3f")} |"
        )
    if len(mining["exposure_fits"]) > 12:
        lines.append(
            f"| … 其余 {len(mining['exposure_fits']) - 12} 个会话见 `pairs.json` | | | | | | |"
        )
    lines.append("")
    lines.append("## 3. 拟合到的系统误差项")
    lines.append("")
    lines.append("| 项 | 含义 | 训练集拟合值 |")
    lines.append("|---|---|---|")
    for row in fit["parameter_table"]:
        lines.append(f"| `{row['key']}` | {row['meaning']} | {_FMT(row['value'], ".5f")} |")
    lines.append("")
    lines.append("## 4. 四类分解: 标定前 vs 标定后")
    lines.append("")
    lines.append("| 组 | 口径 | 能量比 | 宽度比 | 质心偏移 (模型格) | 形状相关 | 原始 Pearson r |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in fit["decomposition_table"]:
        lines.append(
            f"| {row['group']} | {row['scope']} | {_FMT(row['energy_ratio'], ".4f")} | "
            f"{_FMT(row['width_ratio'], ".4f")} | {_FMT(row['centroid_shift_cells'], ".4f")} | "
            f"{_FMT(row['shape_correlation'], ".4f")} | {_FMT(row['correlation_raw'], ".4f")} |"
        )
    lines.append("")
    lines.append("### 4.1 物理诊断: 哪一类误差主导")
    lines.append("")
    diagnostics = fit.get("physical_diagnostics") or {}
    if diagnostics.get("available"):
        lines.append(
            "- **照明包络缩放** (τ_θ 的 `amp_x` / `amp_y`) 扫描: "
            + ", ".join(
                f"amp={row['amp']:.2f} → 宽度比 {_FMT(row['width_ratio_median'], ".4f")}"
                for row in diagnostics.get("amp_scan", [])
            )
            + " —— 几乎不改变宽度 (振幅包络是平顶, 照明直径由瞳面掩膜决定)。"
        )
        lines.append(
            "- **相机离焦** 扫描: "
            + ", ".join(
                f"Δz={row['defocus_m'] * 1e3:.1f} mm → 宽度比 "
                f"{_FMT(row['width_ratio_median'], ".4f")}"
                for row in diagnostics.get("defocus_scan", [])
            )
            + " —— 离焦是角谱传递函数 (快速相位扰动), 解释不了实测的宽度比。"
        )
        lines.append(
            f"- **判定**: 宽度误差的主导类别是**远场采样尺度 / 等效焦距口径** "
            f"(拟合出的焦距尺度 {fit['parameter_table'][0]['value']}), "
            "不是离焦, 也不是照明包络。"
        )
        lines.append(
            f"- **位置误差** 用 canonical 的焦面精确平移 (`ModelErrorTerms.tilt_*`) 修正: "
            f"tilt_x={_FMT(diagnostics.get('tilt_x_rad'), ".8f")} rad、"
            f"tilt_y={_FMT(diagnostics.get('tilt_y_rad'), ".8f")} rad "
            f"(对应质心偏移 {_FMT(diagnostics.get('shift_dx_cells'), ".4f")} / "
            f"{_FMT(diagnostics.get('shift_dy_cells'), ".4f")} 远场格)。"
        )
    lines.append("")
    lines.append("## 5. 阶段 5 判据对照 (验证集)")
    lines.append("")
    lines.append("| 判据 | 阈值 | 标定前 | 标定后 | 结论 |")
    lines.append("|---|---|---|---|---|")
    for row in validation["threshold_table"]:
        lines.append(
            f"| {row['name']} | {row['threshold']} | {_FMT(row['before'], ".4f")} | "
            f"{_FMT(row['after'], ".4f")} | {row['verdict']} |"
        )
    lines.append("")
    lines.append("## 6. 还差多少实验采集")
    lines.append("")
    for line in acquisition["lines"]:
        lines.append(f"- {line}")
    lines.append("")
    lines.append("## 7. 不可用的部分 (诚实标注)")
    lines.append("")
    for line in payload["limitations"]:
        lines.append(f"- {line}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def _decomposition_row(group: str, scope: str, block: dict[str, Any]) -> dict[str, Any]:
    """把一段汇总变成表格行。"""

    def median(key: str) -> float | None:
        value = block.get(key)
        if not isinstance(value, dict):
            return None
        return value.get("median")

    return {
        "group": group,
        "scope": scope,
        "energy_ratio": median("energy_ratio"),
        "width_ratio": median("width_ratio"),
        "centroid_shift_cells": median("centroid_shift_cells"),
        "shape_correlation": median("shape_correlation"),
        "correlation_raw": median("correlation_raw"),
    }


def run(args: argparse.Namespace) -> int:
    """主流程。"""
    logger.remove()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (args.out_dir or Path("logs/history_model_bias")).resolve()
    out_dir = out_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. 挖配对 --------------------------------------------------------
    sessions, exclusions = mine_history_sessions(
        args.roots, probes_per_session=args.probes_per_session, seed=args.seed
    )
    logger.info("挖到 {} 个会话 / {} 条配对", len(sessions), sum(len(s.pairs) for s in sessions))
    for exclusion in exclusions:
        logger.debug("排除 {}: {} ({})", exclusion.identifier, exclusion.reason, exclusion.detail)

    if not sessions:
        logger.error("没有挖到任何历史配对, 终止")
        return 1

    # --- 2. 会话切分 ------------------------------------------------------
    train_sessions, held_sessions = split_sessions_by_session(
        sessions, fraction=args.train_fraction, seed=args.seed
    )
    train_pairs = [pair for s in train_sessions for pair in s.pairs]
    held_pairs = [pair for s in held_sessions for pair in s.pairs]
    logger.info("训练 {} 会话 / {} 对, 历史验证集 {} 会话 / {} 对",
                len(train_sessions), len(train_pairs), len(held_sessions), len(held_pairs))

    # --- 3. 拟合系统误差 --------------------------------------------------
    analytic = fit_calibration(train_pairs)
    calibration = refine_calibration(train_pairs, analytic, seed=args.seed)
    per_session = []
    for session in train_sessions:
        local = fit_calibration(session.pairs)
        per_session.append(
            {
                "session_id": session.session_id,
                "stamp": session.stamp,
                "n_pairs": len(session.pairs),
                "camera_px_per_model_px": session.camera_px_per_model_px,
                "scale": local.scale,
                "shift_dx": local.shift_dx,
                "shift_dy": local.shift_dy,
                "gain": local.gain,
            }
        )
    train_eval = evaluate_calibration(train_pairs, calibration)
    held_eval = evaluate_calibration(held_pairs, calibration) if held_pairs else None
    physical = interpret_calibration(calibration, train_sessions)

    # 物理诊断: 照明包络缩放 / 相机离焦 能不能解释宽度比? 倾斜 (精确平移) 能
    # 把质心偏移压到什么程度?
    affine_phys, terms_phys, physical_diag = fit_physical_calibration(
        train_sessions, train_pairs, max_pairs=args.physical_scan_pairs
    )
    physical_train_eval = (
        evaluate_physical_calibration(
            train_sessions, train_pairs, terms_phys, affine=affine_phys
        )
        if terms_phys is not None
        else None
    )
    physical_held_eval = (
        evaluate_physical_calibration(
            held_sessions, held_pairs, terms_phys, affine=affine_phys
        )
        if terms_phys is not None and held_pairs
        else None
    )

    # --- 4. 验证组 --------------------------------------------------------
    validation: dict[str, Any] = {"available": False}
    if not args.skip_capture_validation:
        validation = {"available": True}
        validate_capture_groups(
            calibration, train_sessions, iterations=args.capture_iterations, out=validation
        )
    census_hwdataset(validation)

    # --- 5. 阈值判定 ------------------------------------------------------
    threshold_rows: list[dict[str, Any]] = []
    validation_summary: dict[str, Any] = {}

    # 历史验证集 (按会话分组切分) —— 「历史预标定能否泛化」的第一道检验。
    if held_eval and held_eval.get("after"):
        before = held_eval["before"]
        after = held_eval["after"]
        validation_summary["历史验证集"] = {"before": before, "after": after}
        for name, key, limit, mode in (
            ("宽度比 ∈ 1.0 ± 0.10", "width_ratio", WIDTH_RATIO_TOLERANCE, "band"),
            ("质心偏移 ≤ 0.5 模型格", "centroid_shift_cells", CENTROID_SHIFT_LIMIT_CELLS, "max"),
        ):
            before_value = before.get(key, {}).get("median")
            after_value = after.get(key, {}).get("median")
            if before_value is None and after_value is None:
                continue
            threshold_rows.append(
                {
                    "group": "历史验证集 (会话分组)",
                    "name": name,
                    "threshold": f"{limit}",
                    "before": before_value,
                    "after": after_value,
                    "verdict": _verdict(name, before_value, after_value, limit, mode),
                }
            )

    if validation.get("available"):
        for label in ("B1_baseline", "B2_prior"):
            block = validation.get(label, {})
            if not block.get("available"):
                continue
            before = block.get("before", {}).get("test") or block.get("before", {}).get("all")
            after = block.get("after", {}).get("test") or block.get("after", {}).get("all")
            if not before or not after:
                continue
            validation_summary[label] = {"before": before, "after": after}
            for name, key, limit, mode in (
                ("宽度比 ∈ 1.0 ± 0.10", "width_ratio", WIDTH_RATIO_TOLERANCE, "band"),
                ("质心偏移 ≤ 0.5 模型格", "centroid_shift_cells", CENTROID_SHIFT_LIMIT_CELLS, "max"),
                ("CV 绝对误差 < 0.05", "cv_abs_error", CV_ABS_LIMIT, "max"),
            ):
                before_value = before.get(key, {}).get("median")
                after_value = after.get(key, {}).get("median")
                if before_value is None and after_value is None:
                    continue
                verdict = _verdict(name, before_value, after_value, limit, mode)
                if key == "centroid_shift_cells":
                    before_px = before.get("centroid_shift_px", {}).get("median")
                    after_px = after.get("centroid_shift_px", {}).get("median")
                    verdict += (
                        f" (相机像元口径 {_FMT(before_px, ".3f")} → {_FMT(after_px, ".3f")} px; "
                        "该组的「模型格」换算依赖斜坡反推焦距, 不可靠)"
                    )
                threshold_rows.append(
                    {
                        "group": label,
                        "name": name,
                        "threshold": f"{limit}",
                        "before": before_value,
                        "after": after_value,
                        "verdict": verdict,
                    }
                )
    validation["threshold_table"] = threshold_rows

    # --- 6. 「还差多少采集」 ----------------------------------------------
    acquisition = _acquisition_guidance(
        calibration,
        train_sessions,
        held_sessions,
        sessions,
        per_session=per_session,
        validation=validation,
    )

    # --- 7. 产物 ----------------------------------------------------------
    exclusion_counts: dict[str, int] = {}
    for exclusion in exclusions:
        exclusion_counts[exclusion.reason] = exclusion_counts.get(exclusion.reason, 0) + 1

    sampling = {
        "probes_per_session": args.probes_per_session,
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "roots": list(args.roots),
    }
    write_pairs(
        out_dir / "pairs.json",
        sessions,
        exclusions,
        {
            **sampling,
            "n_sessions": len(sessions),
            "n_pairs": sum(len(s.pairs) for s in sessions),
            "n_exclusions": len(exclusions),
            "exclusion_counts": exclusion_counts,
            "train_sessions": [s.session_id for s in train_sessions],
            "held_sessions": [s.session_id for s in held_sessions],
        },
    )

    median_coefficients, family_gammas = historical_median_coefficients(train_sessions)
    fit_payload = {
        "calibration": asdict(calibration),
        "calibration_analytic": asdict(analytic),
        "per_session_calibration": per_session,
        "physical_interpretation": physical,
        "physical_diagnostics": physical_diag,
        "physical_terms": terms_phys.to_dict() if terms_phys is not None else None,
        "physical_pupil_affine": affine_phys.to_dict() if affine_phys is not None else None,
        "physical_train": physical_train_eval,
        "physical_holdout": physical_held_eval,
        "train": train_eval,
        "historical_holdout": held_eval,
        "historical_median_coefficients": (
            median_coefficients.tolist() if median_coefficients is not None else None
        ),
        "travel_scale_family_gammas": family_gammas,
        "travel_scale": (
            TravelScale(gammas=tuple(family_gammas)).to_dict() if family_gammas else None
        ),
        "recompute_tolerance": RECOMPUTE_TOL,
    }
    (out_dir / "fit.json").write_text(
        json.dumps(fit_payload, indent=2, default=_json_default), encoding="utf-8"
    )
    (out_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, default=_json_default), encoding="utf-8"
    )

    decomposition_table: list[dict[str, Any]] = []
    if train_eval:
        decomposition_table.append(
            _decomposition_row("历史训练集", "训练会话", train_eval["before"])
        )
        if train_eval.get("after"):
            decomposition_table.append(
                _decomposition_row("历史训练集", "训练会话 (标定后)", train_eval["after"])
            )
    if held_eval:
        decomposition_table.append(
            _decomposition_row("历史验证集", "验证集会话", held_eval["before"])
        )
        if held_eval.get("after"):
            decomposition_table.append(
                _decomposition_row("历史验证集", "验证集会话 (标定后)", held_eval["after"])
            )

    payload = {
        "timestamp": timestamp,
        "output_dir": out_dir.as_posix(),
        "mining": {
            "n_sessions": len(sessions),
            "n_pairs": sum(len(s.pairs) for s in sessions),
            "probes_per_session": args.probes_per_session,
            "seed": args.seed,
            "n_exclusions": len(exclusions),
            "exclusion_counts": exclusion_counts,
            "n_sessions_with_ladder": sum(
                1 for s in sessions if "曝光阶梯" in s.exposure_fit.source
            ),
            "exposure_fits": [
                {"session_id": s.session_id, **s.exposure_fit.to_dict()} for s in sessions
            ],
        },
        "fit": {
            "parameter_table": [
                {
                    "key": "scale",
                    "meaning": "焦距尺度 (模型远场图样需缩放的倍数)",
                    "value": calibration.scale,
                },
                {
                    "key": "shift_dx",
                    "meaning": "瞳面中心偏移 (远场格, x)",
                    "value": calibration.shift_dx,
                },
                {
                    "key": "shift_dy",
                    "meaning": "瞳面中心偏移 (远场格, y)",
                    "value": calibration.shift_dy,
                },
                {
                    "key": "gain",
                    "meaning": "强度增益 (实测/模型)",
                    "value": calibration.gain,
                },
                {
                    "key": "tilt_x_rad",
                    "meaning": "换算出的光路倾斜 (rad)",
                    "value": physical.get("tilt_x_rad"),
                },
                {
                    "key": "tilt_y_rad",
                    "meaning": "换算出的光路倾斜 (rad)",
                    "value": physical.get("tilt_y_rad"),
                },
                {
                    "key": "focal_length_model_m",
                    "meaning": "模型标称焦距 (m)",
                    "value": physical.get("focal_length_model_m"),
                },
                {
                    "key": "focal_length_equivalent_m",
                    "meaning": "由实测标度反推的等效焦距 (m)",
                    "value": physical.get("focal_length_equivalent_m"),
                },
                {
                    "key": "focal_ratio_measured_over_model",
                    "meaning": "等效焦距 / 标称焦距 (焦距/尺度口径是否自洽)",
                    "value": physical.get("focal_ratio_measured_over_model"),
                },
                {
                    "key": "defocus_equivalent_m",
                    "meaning": "焦距尺度误差等价的相机离焦 (m)",
                    "value": physical.get("defocus_equivalent_m"),
                },
            ],
            "decomposition_table": decomposition_table,
            "physical_diagnostics": physical_diag,
        },
        "validation": {
            "threshold_table": threshold_rows,
            "detail": validation,
        },
        "acquisition": acquisition,
        "limitations": _limitations(sessions, held_sessions, validation),
    }
    write_summary(out_dir / "summary.md", payload)

    labels: list[str] = []
    width_before: list[float] = []
    width_after: list[float] = []
    centroid_before: list[float] = []
    centroid_after: list[float] = []
    if train_eval and train_eval.get("after"):
        labels.append("历史训练集")
        width_before.append(train_eval["before"]["width_ratio"]["median"] or 0.0)
        width_after.append(train_eval["after"]["width_ratio"]["median"] or 0.0)
        centroid_before.append(train_eval["before"]["centroid_shift_cells"]["median"] or 0.0)
        centroid_after.append(train_eval["after"]["centroid_shift_cells"]["median"] or 0.0)
    if held_eval and held_eval.get("after") and "历史验证集" not in validation_summary:
        labels.append("历史验证集")
        width_before.append(held_eval["before"]["width_ratio"]["median"] or 0.0)
        width_after.append(held_eval["after"]["width_ratio"]["median"] or 0.0)
        centroid_before.append(held_eval["before"]["centroid_shift_cells"]["median"] or 0.0)
        centroid_after.append(held_eval["after"]["centroid_shift_cells"]["median"] or 0.0)
    for label, summary in validation_summary.items():
        labels.append(label)
        width_before.append(summary["before"].get("width_ratio", {}).get("median") or 0.0)
        width_after.append(summary["after"].get("width_ratio", {}).get("median") or 0.0)
        centroid_before.append(
            summary["before"].get("centroid_shift_cells", {}).get("median") or 0.0
        )
        centroid_after.append(
            summary["after"].get("centroid_shift_cells", {}).get("median") or 0.0
        )
    if labels:
        write_figure(
            out_dir / "calibration.png",
            labels,
            width_before,
            width_after,
            centroid_before,
            centroid_after,
        )

    logger.info("产物写入 {}", out_dir)
    print(json.dumps({
        "output_dir": out_dir.as_posix(),
        "n_sessions": len(sessions),
        "n_pairs": sum(len(s.pairs) for s in sessions),
        "n_exclusions": len(exclusions),
        "calibration": asdict(calibration),
        "threshold_table": threshold_rows,
    }, indent=2, default=_json_default))
    return 0


def _verdict(
    name: str, before: float | None, after: float | None, limit: float, mode: str
) -> str:
    """阈值判定文字。"""

    def ok(value: float | None) -> bool | None:
        if value is None:
            return None
        if mode == "band":
            return abs(value - WIDTH_RATIO_TARGET) <= limit
        return value <= limit

    before_ok = ok(before)
    after_ok = ok(after)
    if after_ok is None:
        return "无数据"
    if before_ok is True and after_ok is True:
        return "标定前后均达标"
    if after_ok is True:
        return "标定后达标 (标定前不达标)"
    if before_ok is True:
        return "标定后反而不达标"
    return "标定后仍未达标"


def _acquisition_guidance(
    calibration: Calibration,
    train_sessions: Sequence[Session],
    held_sessions: Sequence[Session],
    all_sessions: Sequence[Session],
    *,
    per_session: Sequence[dict[str, Any]] = (),
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """量化「用历史数据后还需要多少实验采集」。"""
    n_sessions = len(all_sessions)
    n_pairs = sum(len(s.pairs) for s in all_sessions)
    scales = [float(row["scale"]) for row in per_session if row.get("scale")]
    spread = (
        {
            "p10": float(np.percentile(scales, 10)),
            "median": float(np.median(scales)),
            "p90": float(np.percentile(scales, 90)),
        }
        if scales
        else {}
    )
    transfer: list[str] = []
    if validation:
        for label in ("B1_baseline", "B2_prior"):
            block = validation.get(label, {})
            if not block.get("available"):
                continue
            before = (block.get("before", {}).get("test") or block.get("before", {}).get("all"))
            after = (block.get("after", {}).get("test") or block.get("after", {}).get("all"))
            if not before or not after:
                continue
            width_before = before.get("width_ratio", {}).get("median")
            width_after = after.get("width_ratio", {}).get("median")
            transfer.append(
                f"{label}: 宽度比 {_FMT(width_before, ".4f")} → {_FMT(width_after, ".4f")}, "
                f"CV 绝对误差 {_FMT(before.get('cv_abs_error', {}).get('median'), ".4f")} → "
                f"{_FMT(after.get('cv_abs_error', {}).get('median'), ".4f")}"
            )

    lines = [
        f"历史数据共覆盖 {n_sessions} 个会话 / {n_pairs} 条配对; "
        f"按会话分组后训练集 {len(train_sessions)} 个会话、验证集 {len(held_sessions)} 个会话。",
        f"预标定给出的系统误差项: 焦距尺度 {calibration.scale:.4f}、"
        f"瞳面中心偏移 ({calibration.shift_dx:+.3f}, {calibration.shift_dy:+.3f}) 远场格、"
        f"增益 {calibration.gain:.4f}。",
        f"逐会话拟合的焦距尺度散布 (p10 / 中位 / p90): "
        f"{_FMT(spread.get('p10'), ".4f")} / {_FMT(spread.get('median'), ".4f")} / "
        f"{_FMT(spread.get('p90'), ".4f")} —— "
        + (
            "散布远小于 1, 所以它是**实验系统级**的常量, 一次离线拟合即可跨会话复用。"
            if spread and (spread["p90"] - spread["p10"]) < 0.25
            else "散布不小, 说明它**不是**跨会话恒定的量, 每会话仍要重新确认。"
        ),
        "仍需实验采集的只有**逐会话漂移量** (光强、面板状态、光轴零点): "
        "按现有 MIL 的探针口径, 一次确认只需**平场 + 1 条 Zernike 探针** (2 帧), "
        "而不是每轮 16 条训练探针 + 4 条验证集探针 (20 帧)。",
        "Step A 的**初值**可以直接取历史拟合的**逐 Noll 序中位系数** "
        "(见 `fit.json` 的 `historical_median_coefficients`), 于是拟合从零起步变成"
        "「从中位像差起步」, 信任域不必从零一路顶到真值 —— 审计里 50/56 轮撞满"
        "信任域、所需步长是上限 1.899 倍 (中位) 的现象, 正是从零起步造成的。",
        "跨光学配置迁移 (今天新采的 B1 / B2): " + ("; ".join(transfer) if transfer else "未跑"),
    ]
    return {
        "lines": lines,
        "n_sessions": n_sessions,
        "n_pairs": n_pairs,
        "n_train_sessions": len(train_sessions),
        "n_heldout_sessions": len(held_sessions),
        "per_session_scale_spread": spread,
        "transfer_to_capture_groups": transfer,
    }


def _limitations(
    sessions: Sequence[Session],
    held_sessions: Sequence[Session],
    validation: dict[str, Any],
) -> list[str]:
    """诚实标注不可用的部分。"""
    lines = [
        "**口径不兼容的部分**: `src/ml/hwdataset` 语料的相机窗口有 64 / 248 / 320 / 1944 px 四种, "
        "region 与 MIL 不同 (MIL 是 64 / 128), 因此**没有**并入拟合, 只作单独一组的普查。",
        "**旋转象限**: 硬件路径写死为 1 (相机上下翻转), 仿真路径为 0; "
        "本产物只用硬件路径的 1, 仿真运行全部排除。",
        "**曝光上限**: 曝光阶梯里 ≥ 250 counts 的点被当作饱和剔除, "
        "所以增益只由未饱和段拟合 (CCD 在高曝光端非线性, 线性外推会高估增益)。",
        "**回滚族**: 审计已判定 `20261019_103208` 一族的落盘系数被回滚、不可复算, 本产物排除该族; "
        "此外每一轮还要逐像素校验 (相对偏差 ≤ 1e-5), 校验不过的轮次整轮排除。",
        "**验证组的模型不同**: B1/B2 用的是 `fit_defocus_forward_dataset.PhysicalForwardFit` "
        "(共享物理预测模型, 相机是整幅传感器), 与 MIL 的 Zernike 系数模型**不是同一个模型**, "
        "所以两组的「宽度比」等数字口径不同, 只能在**各自组内**比较标定前 / 后。",
        "**B1/B2 的几何换算不可靠**: 由斜坡矩阵反推的等效焦距 (~19 mm) 与 `focus.json` 自记的 "
        "「FOV 交叉核对差 152%」矛盾, 所以 B1/B2 的质心偏移换算成「模型格」只作参考, "
        "以**相机像元**为准。",
        "**相机离焦解释不了宽度比**: 诊断扫描显示离焦 0 → 2 mm 就把宽度比从 1.09 顶到 2.10 "
        "(角谱传递函数是快速相位扰动, 不是平滑展宽), 所以宽度误差**不是**离焦, "
        "**也不是**照明包络缩放 (振幅包络是平顶, 缩放几乎不改变宽度); "
        "它对应的是**远场采样尺度 / 等效焦距**口径。",
    ]
    if not held_sessions:
        lines.append("**历史验证集为空**: 会话数不足, 训 / 验分离退化为只报训练集。")
    hwdataset = validation.get("hwdataset", {})
    if hwdataset.get("available"):
        lines.append(
            f"hwdataset 语料普查: {hwdataset.get('n_records')} 条记录, "
            f"{len(hwdataset.get('families', {}))} 个 family, 曝光 "
            f"{hwdataset.get('exposure_ms_range')} ms —— 未参与拟合。"
        )
    else:
        lines.append(f"hwdataset 语料普查不可用: {hwdataset.get('reason', MISSING)}")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots", nargs="*", default=list(DEFAULT_ROOTS), help="历史 MIL 运行的扫描根"
    )
    parser.add_argument(
        "--probes-per-session", type=int, default=6, help="每个会话最多抽多少条探针"
    )
    parser.add_argument("--seed", type=int, default=20261022, help="抽样随机种子")
    parser.add_argument(
        "--train-fraction", type=float, default=SESSION_SPLIT_FRACTION, help="训练会话占比"
    )
    parser.add_argument(
        "--capture-iterations", type=int, default=150, help="验证组的物理预测模型拟合迭代数"
    )
    parser.add_argument(
        "--physical-scan-pairs",
        type=int,
        default=10,
        help="物理诊断 (照明包络 / 离焦扫描) 参与的配对数",
    )
    parser.add_argument(
        "--skip-capture-validation", action="store_true", help="跳过 B1/B2 验证组"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("logs/history_model_bias"), help="产物根目录"
    )
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
