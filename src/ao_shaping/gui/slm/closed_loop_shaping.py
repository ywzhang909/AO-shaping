"""实测闭环动态整形：单线程串行事务、门禁、候选生成与在线适配。

与 ``dynamic_shaping.py``（开环时间轴播放）和 ``static_shaping_page.py``
（单帧下发）完全隔离：本模块只描述「实测闭环」自己的轮次语义，不读写它们的
计划、相位或状态。

一轮是**不可分割的事务**，顺序固定，不允许交错::

    写相位 → 稳定 → 取新帧 → 门禁 → 实测指标 → (可选) 在线适配 → 生成候选
    → 人工/自动接受或回滚

三条硬规矩：

* **配对**：帧必须晚于本次相位写入，并带自己的帧号；旧帧、门禁失败的帧既不得
  进入指标比较，也不得进入适配数据。``Bench.capture`` 的语义就是"取一帧新帧"。
* **实测优先**：候选是否接受只看实测指标（外加噪声余量），预测改善不等于实测
  改善；预测分数与实测分数在记录里分开保存，页面上也必须分开显示。
* **在线适配是批量递增，不是逐帧梯度**：累计合规样本 → 从已晋级检查点
  warm-start 重拟合 → 用独立验证相位比较后晋级。样本不足时保持旧检查点并如实
  报告 ``insufficient``，不得降低验证门槛。

本模块顶层只导入 numpy/stdlib；torch 与 transfer 拟合链在函数内部惰性导入，
因此导入本模块不会拉起 torch 或设备 SDK。
"""
from __future__ import annotations

import json
import threading
import time

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from loguru import logger

if TYPE_CHECKING:  # 只在类型检查期导入，运行时仍保持轻量顶层（numpy/loguru/stdlib）
    from ao_shaping.gui.slm.target_lock import LockConfig

__all__ = [
    "AdaptationPolicy",
    "AdaptationResult",
    "Bench",
    "CapturedFrame",
    "ClosedLoopConfig",
    "ClosedLoopRunner",
    "FrameGate",
    "GateThresholds",
    "LoopEvent",
    "LoopState",
    "LoopSummary",
    "OnlineTransferAdapter",
    "PhaseProposal",
    "RoundRecord",
    "analysis_grid",
    "check_frame",
    "compare_to_target",
    "describe_gate",
    "flat_zero_transfer_fraction",
    "grid_spacing_px",
    "load_slm_diffraction_efficiency",
    "promotion_decision",
    "refine_phase",
    "sample_on_grid",
    "score_from_metrics",
    "shaping_efficiency_report",
    "target_field",
    "window_fraction",
]


class LoopState(str, Enum):
    """闭环状态。转换只由 :class:`ClosedLoopRunner` 自己写。"""

    IDLE = "idle"
    PREFLIGHT = "preflight"
    MEASURING = "measuring"
    UPDATING = "updating"
    AWAITING_DECISION = "awaiting_decision"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAULT = "fault"


#: 门禁原因码 → 面向操作员的中文说明。页面显示说明，测试断言原因码。
_GATE_TEXT: dict[str, str] = {
    "empty_frame": "帧为空",
    "nonfinite": "帧内存在非有限值",
    "saturated": "有像素触顶（饱和），强度剖面不可比",
    "low_signal": "峰值低于有效信号下限",
    "centroid_jump": "光斑质心跳变超过上限",
    "exposure_mismatch": "曝光回读与设定不一致",
    "exposure_invalid": "曝光不是正的有限值",
    "grid_out_of_bounds": "分析窗口超出传感器范围",
}

#: 漂移反馈 Jacobian 的奇异判定上限（条件数）。tilt 两轴近似相关（如一轴无响应）
#: 时条件数骤增，反演会把噪声放大成离谱的反馈倾斜，宁可跳过本轮。
_DRIFT_COND_LIMIT: float = 1.0e6


@dataclass(frozen=True)
class GateThresholds:
    """单帧门禁阈值。历史安全档位不是永久上限，每次运行按当前光路重定。"""

    saturation_fraction: float = 250.0 / 255.0
    min_peak_fraction: float = 0.02
    min_peak_counts: float = 32.0
    max_centroid_jump_px: float = 40.0
    exposure_tolerance_ms: float = 0.5


@dataclass(frozen=True)
class FrameGate:
    """一帧的门禁结论。``ok`` 为假时该帧不得参与指标比较或适配。"""

    ok: bool
    reasons: tuple[str, ...]
    peak: float
    full_scale: float
    saturation_pixels: int
    centroid_xy: tuple[float, float] | None
    exposure_ms: float

    def describe(self) -> str:
        return describe_gate(self)


def describe_gate(gate: FrameGate) -> str:
    """把门禁结论写成一句中文说明（通过时也给数值，便于记录）。"""
    if gate.ok:
        return (
            f"通过：峰值 {gate.peak:.0f}/{gate.full_scale:.0f}，"
            f"饱和 0 px，曝光 {gate.exposure_ms:.2f} ms"
        )
    parts = [_GATE_TEXT.get(code, code) for code in gate.reasons]
    return (
        f"未通过：{'；'.join(parts)}"
        f"（峰值 {gate.peak:.0f}/{gate.full_scale:.0f}，"
        f"饱和 {gate.saturation_pixels} px，曝光 {gate.exposure_ms:.2f} ms）"
    )


def _centroid(frame: np.ndarray) -> tuple[float, float] | None:
    """强度加权质心，仅用作门禁的跳变判据，不作为分析用质心。

    分析侧的质心是 SSoT（``utils.image.centroid``），这里刻意不重写那套窗口
    语义：门禁只需要一个稳定、廉价、可复现的"光斑大致还在原处"判据。
    """
    total = float(frame.sum())
    if not np.isfinite(total) or total <= 0:
        return None
    ys, xs = np.indices(frame.shape, dtype=np.float64)
    return float((frame * xs).sum() / total), float((frame * ys).sum() / total)


def check_frame(
    frame: np.ndarray,
    *,
    full_scale: float,
    exposure_ms: float,
    expected_exposure_ms: float | None = None,
    previous_centroid: tuple[float, float] | None = None,
    thresholds: GateThresholds = GateThresholds(),
) -> FrameGate:
    """判定单帧能否成为本轮的有效测量。

    任何一条不满足即为未通过；调用方不得把未通过的帧用于指标比较、模型适配或
    候选生成，只能记录并重测。
    """
    data = np.asarray(frame)
    reasons: list[str] = []
    peak = 0.0
    saturated = 0
    centroid: tuple[float, float] | None = None

    if data.size == 0:
        reasons.append("empty_frame")
    if not np.all(np.isfinite(data)):
        reasons.append("nonfinite")
    exposure = float(exposure_ms)
    if not np.isfinite(exposure) or exposure <= 0:
        reasons.append("exposure_invalid")
    elif expected_exposure_ms is not None and abs(exposure - float(expected_exposure_ms)) > thresholds.exposure_tolerance_ms:
        reasons.append("exposure_mismatch")

    if data.size and np.all(np.isfinite(data)):
        peak = float(data.max())
        saturated = int(np.count_nonzero(data >= thresholds.saturation_fraction * float(full_scale)))
        if saturated > 0:
            reasons.append("saturated")
        floor = max(thresholds.min_peak_counts, thresholds.min_peak_fraction * float(full_scale))
        if peak < floor:
            reasons.append("low_signal")
        centroid = _centroid(data.astype(np.float64, copy=False))
        if previous_centroid is not None and centroid is not None:
            jump = float(np.hypot(centroid[0] - previous_centroid[0], centroid[1] - previous_centroid[1]))
            if jump > thresholds.max_centroid_jump_px:
                reasons.append("centroid_jump")

    return FrameGate(
        ok=not reasons,
        reasons=tuple(reasons),
        peak=peak,
        full_scale=float(full_scale),
        saturation_pixels=saturated,
        centroid_xy=centroid,
        exposure_ms=exposure,
    )


def sample_on_grid(frame: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """把 CCD 帧采样到模型/指标共用的固定网格 ``xy`` (H, W, 2)。"""
    grid = np.asarray(xy)
    if grid.ndim != 3 or grid.shape[-1] != 2:
        raise ValueError(f"xy must be (H, W, 2), got {grid.shape}")
    ys = np.rint(grid[..., 1]).astype(int)
    xs = np.rint(grid[..., 0]).astype(int)
    height, width = frame.shape[:2]
    if ys.min() < 0 or xs.min() < 0 or ys.max() >= height or xs.max() >= width:
        raise ValueError("fixed analysis window extends beyond sensor")
    return np.asarray(frame, dtype=np.float64)[ys, xs]


def analysis_grid(centre_xy: tuple[float, float], span: int) -> np.ndarray:
    """闭环固定的分析窗口：与 ``transfer.fit.load_dataset`` 同式。

    ``np.mgrid[-span:span+1:2]`` 以光斑为原点、步长 2 px，因此同一个 ``span``
    得到的网格与离线拟合/验证逐点对齐 —— 闭环分数与 transfer 的验证分数量的
    是同一块像素，而不是各自裁一块。窗口一旦开跑即锁定。
    """
    span = int(span)
    if span < 1:
        raise ValueError("span must be >= 1")
    yy, xx = np.mgrid[-span:span + 1:2, -span:span + 1:2]
    return np.stack((xx + round(float(centre_xy[0])), yy + round(float(centre_xy[1]))), axis=-1)


def grid_spacing_px(xy: np.ndarray) -> float | None:
    """分析网格相邻采样点的 CCD 像素步长；无法确定时返回 ``None``。

    页面标定与 runner 共用这一定义，避免"页面按一种步长换算、运行按另一种"造成
    位置容差口径不一致。
    """
    grid = np.asarray(xy, dtype=np.float64)
    if grid.ndim != 3 or grid.shape[-1] != 2:
        return None
    dx = np.abs(np.diff(grid[..., 0], axis=1))
    dy = np.abs(np.diff(grid[..., 1], axis=0))
    steps = np.concatenate((dx[dx > 0.0], dy[dy > 0.0]))
    if steps.size == 0:
        return None
    spacing = float(np.median(steps))
    if not np.isfinite(spacing) or spacing <= 0.0:
        return None
    return spacing


def _tilt_ramp(shape: tuple[int, ...], axis: int) -> np.ndarray:
    """峰谷恰为 1 rad 的线性斜坡（与 ``DriftBench.tilt()`` 同形）。

    ``axis=1`` 得到沿列（x）的斜坡，``axis=0`` 得到沿行（y）的斜坡。长度为 1 的
    轴上峰谷为 0（J 该列必为 0，由奇异判定接管）。
    """
    n = int(shape[axis])
    ramp = np.arange(n, dtype=np.float64) / float(max(n - 1, 1))
    if axis == 1:
        return np.broadcast_to(ramp[None, :], shape).astype(np.float64).copy()
    return np.broadcast_to(ramp[:, None], shape).astype(np.float64).copy()


def _camera_centroid(intensity: np.ndarray, xy: np.ndarray) -> tuple[float, float] | None:
    """相机像素坐标系（与 ``analysis_grid`` 同一坐标表）下的强度加权质心。

    闭环门禁里的 ``_centroid`` 用的是**整帧像素索引**；这里拿预测强度在分析窗口
    的实际 CCD 坐标（``xy``）加权，两者落在同一套相机像素上，Jacobian 的
    px/rad 才能直接作用到实测残差 ``record.gate.centroid_xy`` 上。
    """
    weights = np.asarray(intensity, dtype=np.float64)
    grid = np.asarray(xy, dtype=np.float64)
    if weights.shape != grid.shape[:2]:
        raise ValueError(
            f"predicted intensity shape {weights.shape} does not match the "
            f"analysis grid {grid.shape[:2]}"
        )
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        return None
    return (
        float((weights * grid[..., 0]).sum() / total),
        float((weights * grid[..., 1]).sum() / total),
    )


def compare_to_target(measured: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """在固定网格上比较实测与目标的**形状**：复用 transfer 的 ``fit.metrics``。

    两侧先除以各自的均值再比较。理由不是"让数字好看"，而是两侧量纲本来就不同:
    相机侧是 counts（或 counts/ms），目标侧是无量纲的 0/1 掩膜；不归一化时
    ``relative_l2`` 只反映绝对能量差，同一光路换曝光就整体平移，轮次之间不可比。
    闭环要判的是**目标窗口内的形状是否更像目标**，绝对光效另由
    :func:`window_fraction` 单独报告，绝不混进接受判据。

    返回 ``correlation`` / ``relative_l2`` / ``cv_*``，定义与离线拟合、验证一致。
    """
    from ao_shaping.tools.slm.transfer.fit import metrics

    measured = np.asarray(measured, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if measured.shape != target.shape:
        raise ValueError(f"measured {measured.shape} and target {target.shape} must share the grid")
    if not (np.all(np.isfinite(measured)) and np.all(np.isfinite(target))):
        raise ValueError("non-finite intensity in measured frame or target")
    if float(measured.min()) < 0 or float(target.min()) < 0:
        raise ValueError("intensity must be nonnegative")
    measured_mean = float(measured.mean())
    target_mean = float(target.mean())
    if measured_mean <= 0 or target_mean <= 0:
        # 全零帧/空目标：不能给"平场"一个看起来还行的分数。
        raise ValueError("measured frame or target has zero mean intensity")
    if float(measured.std()) <= 0 or float(target.std()) <= 0:
        # np.corrcoef 在常值输入上给 nan；闭环里"平场"必须是可判定的低分。
        return {
            "correlation": 0.0,
            "cv_prediction": 0.0,
            "cv_measured": 0.0,
            "cv_abs_error": 0.0,
            "relative_l2": float(
                np.linalg.norm(measured - target)
                / max(np.linalg.norm(target), 1e-9)
            ),
        }
    return metrics([measured / measured_mean], [target / target_mean])[0]


def window_fraction(frame: np.ndarray, xy: np.ndarray) -> float | None:
    """目标窗口内的能量占全帧的比例（诊断量，不参与接受判据）。

    形状分数会奖励"窗口内很像、窗口外漏光"；把窗口内占比单独报出来，操作员
    才看得见"为了形状牺牲了多少光"。全帧总能量非正时返回 ``None``。
    """
    data = np.asarray(frame, dtype=np.float64)
    total = float(data.sum())
    if not np.isfinite(total) or total <= 0:
        return None
    return float(sample_on_grid(data, xy).sum() / total)



def flat_zero_transfer_fraction(energy: float, flat_zero_energy: float) -> float:
    """能量占平相位 0 级基准的比例 ``η = E / F0``。

    定义来自效率匹配实验（``docs/dev/efficiency_match_experiment_plan_20261121.md``）：
    * 倾斜（闪耀光栅）``η_d = G1 / F0`` —— 光栅把平相位 0 级的能量送进
      +1 级的份额，即 SLM 衍射效率；
    * 整形 ``η_s = (S - S0) / F0`` —— 目标窗口里来自可调制分量的净能量份额。

    ``energy`` 允许为负（``S - S0`` 可以为负，表示不可调制 0 级的泄漏比
    可调制分量还多）；分母必须是正的有限数。两口径分母都是**平相位 0 级窗口
    积分**，不是 CCD 全帧能量，也不是闪耀后三级之和。
    """
    numerator = float(energy)
    denominator = float(flat_zero_energy)
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        raise ValueError(
            f"平相位 0 级基准必须为正有限数，能量必须有限 ({energy}, {flat_zero_energy})"
        )
    return numerator / denominator


def load_slm_diffraction_efficiency(path: str | Path) -> dict[str, Any]:
    """从衍射效率探针的 ``summary.json`` 读 SLM 衍射效率 ``η = E_blaze / F0``。

    定义（2026-11-26）：**SLM 衍射效率 = 从平相位 0 级能转移走的能量份额**，
    公式 ``η = E_blaze / F0``：

    * ``F0`` = ``relative_efficiency.baseline_zero_window_counts`` —— 平相位帧
      零级窗口的暗减积分，即“可转移走的能量池”（与闪耀帧同一窗口、同一曝光）；
    * ``E_blaze`` = 闪耀帧闪耀级（``--gray-max`` 那一侧）窗口能量。

    探针自己算的 ``relative_efficiency.blaze_energy_share`` 就是该式，存在时
    直接采信（与探针文档同一口径）；缺失时按上式从 ``lobe.energies`` 复算。
    **不用** ``lobe.ratios[blaze_order]``：它的分母是闪耀后三窗之和
    (E₀+E₊₁+E₋₁)，会把效率抬高（同一探针上 0.3978 vs 0.1927）。
    ``lobe_share`` 仅作诊断保留，不参与效率计算。

    兼容两版 schema 的闪耀级标注：``relative_efficiency.blaze_order``（新）、
    ``lobe.orders.blaze_order``（旧）；都缺时取 ±1 级中较强者。缺键、效率非正
    或 > 1（窗口口径不一致）直接报错，不静默给个 0。
    """
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    lobe = payload["lobe"]
    energies = lobe["energies"]
    relative = payload["relative_efficiency"]
    flat_zero = float(relative["baseline_zero_window_counts"])
    blaze_order = relative.get("blaze_order") or lobe.get("orders", {}).get("blaze_order")
    if blaze_order is None:
        first_orders = [o for o in ("plus_one", "minus_one") if o in energies]
        if not first_orders:
            raise KeyError(f"探针没有 ±1 级能量，无法确定闪耀级: {source}")
        blaze_order = max(first_orders, key=lambda o: float(energies[o]))
    blaze_energy = float(energies[str(blaze_order)])
    from_energies = flat_zero_transfer_fraction(blaze_energy, flat_zero)
    reported_share = relative.get("blaze_energy_share")
    efficiency = from_energies if reported_share is None else float(reported_share)
    if not 0.0 < efficiency <= 1.0:
        raise ValueError(f"SLM 衍射效率非正或无效 ({efficiency}): {source}")
    return {
        "efficiency": efficiency,
        "blaze_order": str(blaze_order),
        "blaze_energy": blaze_energy,
        "flat_zero_energy": float(flat_zero),
        "lobe_share": float(lobe["ratios"][str(blaze_order)]),
        "gray_depth": payload.get("config", {}).get("gray_depth"),
        "timestamp_utc": payload.get("timestamp_utc"),
    }


def shaping_efficiency_report(
    summary: LoopSummary,
    *,
    slm_diffraction_efficiency: float | None = None,
) -> dict[str, Any]:
    """整形完成后的能量利用率报告：当前（最优）结果 vs SLM 衍射效率。

    * **能量利用率** = 最优轮 ``window_fraction``：目标窗口能量 / **CCD 全帧**
      能量。直接复用该轮已落账的诊断量，**不重新采集**，不在完成态再动设备。
    * **SLM 衍射效率** = 探针实测的 ``η = E_blaze / F0``（从平相位 0 级转移走
      的能量份额，分母是**平相位零级窗口**），见
      :func:`load_slm_diffraction_efficiency`。
    * **相对 SLM 效率** = 利用率 ÷ 衍射效率。

    ⚠ 两个分母不同（整帧 vs 平相位零级窗口），所以 ``relative_to_slm`` 只是
    粗对照，不是严格的“同一池子里拿到了几成”。严格同分母的对比需要同轮平相位
    帧（``η_s = (S - S0) / F0``），完成态不采集，因此只报数值、不设硬判定，
    并在页面文案里注明口径。

    最优轮按 ``summary.best_score`` 匹配（它对应 best_phase）；缺 best_score
    时退化为实测分数最低且带 window_fraction 的轮次；没有任何有效轮次时
    ``round_index`` / ``window_fraction`` 为 ``None``。
    """
    best: RoundRecord | None = None
    if summary.best_score is not None:
        best = next(
            (r for r in summary.rounds
             if r.measured_score is not None
             and abs(r.measured_score - summary.best_score) < 1e-12
             and r.window_fraction is not None),
            None,
        )
    if best is None:
        candidates = [r for r in summary.rounds
                      if r.measured_score is not None and r.window_fraction is not None]
        if candidates:
            best = min(candidates, key=lambda r: r.measured_score)
    utilization = best.window_fraction if best is not None else None
    relative = None
    if utilization is not None and slm_diffraction_efficiency is not None:
        efficiency = float(slm_diffraction_efficiency)
        if not np.isfinite(efficiency) or efficiency <= 0:
            raise ValueError("slm_diffraction_efficiency 必须为正的有限数")
        relative = utilization / efficiency
    return {
        "round_index": best.round_index if best is not None else None,
        "window_fraction": utilization,
        "slm_diffraction_efficiency": slm_diffraction_efficiency,
        "relative_to_slm": relative,
    }



def score_from_metrics(entry: Mapping[str, float]) -> float:
    """闭环分数，与 transfer 验证质量同式：``1 - correlation + relative_l2``。"""
    return 1.0 - float(entry["correlation"]) + float(entry["relative_l2"])


def target_field(
    xy: np.ndarray,
    *,
    centre_xy: tuple[float, float] | None = None,
    half_width: float,
    half_height: float,
    shape: str = "rect",
) -> np.ndarray:
    """在模型网格上生成目标强度场（1 在目标内，0 在外）。

    ``centre_xy`` 用**相机像素坐标**（与 ``xy`` 同一坐标系）。``None`` 取网格点的
    坐标均值 —— 即 ``analysis_grid`` 的构造中心。注意不能用 ``(width-1)/2`` 这种
    *索引*中心：``analysis_grid`` 的坐标是 6..14 这类绝对像素，拿索引中心当坐标会
    把目标凭空放到 (2, 2)，一个采样点都不覆盖。
    """
    grid = np.asarray(xy, dtype=np.float64)
    if grid.ndim != 3 or grid.shape[-1] != 2:
        raise ValueError(f"xy must be (H, W, 2), got {grid.shape}")
    if centre_xy is not None:
        cx, cy = float(centre_xy[0]), float(centre_xy[1])
    else:
        cx = float(grid[..., 0].mean())
        cy = float(grid[..., 1].mean())
    gx = grid[..., 0] - cx
    gy = grid[..., 1] - cy
    if shape == "rect":
        inside = (np.abs(gx) <= float(half_width)) & (np.abs(gy) <= float(half_height))
    elif shape == "ring":
        radius = np.hypot(gx, gy)
        inner = min(float(half_width), float(half_height)) * 0.6
        outer = max(float(half_width), float(half_height))
        inside = (radius <= outer) & (radius >= inner)
    else:
        raise ValueError("shape must be 'rect' or 'ring'")
    if not inside.any():
        raise ValueError("target covers no model-grid sample; widen the target")
    return inside.astype(np.float64)


# --------------------------------------------------------------------------- #
# 候选相位：MIL 式 Step B 更新，物理预测模型换成 transfer 的 FrozenDefocusForward
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PhaseProposal:
    """一次候选相位及其“模型估计”分数（不是实测）。"""

    phase: np.ndarray
    predicted_score: float
    max_step_rad: float
    rms_rad: float
    iterations: int
    history: tuple[float, ...] = ()


def _pearson(p: Any, t: Any, torch: Any) -> Any:
    pc = p - p.mean()
    tc = t - t.mean()
    denom = torch.sqrt((pc * pc).sum() * (tc * tc).sum())
    if float(denom.detach()) <= 0:
        return torch.zeros((), dtype=p.dtype, device=p.device)
    return (pc * tc).sum() / denom


def refine_phase(
    forward: Any,
    target: np.ndarray,
    initial_phase: np.ndarray,
    *,
    iterations: int = 60,
    lr: float = 0.02,
    max_step_rad: float = 0.35,
    phase_rms_cap: float | None = 1.2,
    device: str | None = None,
    exposure_ms: float | None = None,
    on_iteration: Callable[[int, float], None] | None = None,
) -> PhaseProposal:
    """在冻结的 transfer 物理预测模型上做有界相位更新（MIL Step B 的闭环版）。

    只有相位可训练；模型参数是晋级过的检查点，本轮不更新。步长通过
    ``max_step_rad`` 的 ``tanh`` 重参数化硬限制为逐像素信赖域，``phase_rms_cap``
    再限制整体相位 RMS，避免一次候选把相位推到离当前工作点很远的地方。

    两个形状必须分清, 弄反了会静默跑在错网格上:

    * ``initial_phase`` 是 **SLM 控制网格**上的相位（``grid_size`` 见检查点），
      也就是 ``FrozenDefocusForward.predict`` 吃进去的东西；
    * ``target`` 是 **相机采样网格**上的目标强度（``analysis_grid`` 的结果）。

    ``exposure_ms`` 只在检查点带 ``bias_counts`` 时才必需（模型按 counts 预测），
    不给就沿用 ``forward.exposure_ms``。
    """
    import torch

    init = torch.as_tensor(np.asarray(initial_phase, dtype=np.float32), device=device)
    target_t = torch.as_tensor(np.asarray(target, dtype=np.float32), device=device)
    if init.ndim != 2:
        raise ValueError(f"initial phase must be a 2-D control grid, got {tuple(init.shape)}")
    expected = tuple(target_t.shape)
    raw = torch.zeros_like(init, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=float(lr))
    history: list[float] = []
    best_loss = float("inf")
    best_phase = init.detach().clone()

    def _predict(phase: Any) -> Any:
        prediction = (
            forward.predict(phase, exposure_ms=exposure_ms)
            if exposure_ms is not None
            else forward.predict(phase)
        )
        while getattr(prediction, "ndim", 0) > 2:
            prediction = prediction[0]
        if tuple(prediction.shape) != expected:
            raise ValueError(
                f"forward returned {tuple(prediction.shape)} but target is {expected}; "
                "the analysis grid must be built from the same span as the checkpoint"
            )
        return prediction

    for step in range(int(iterations)):
        delta = float(max_step_rad) * torch.tanh(raw / float(max_step_rad))
        if phase_rms_cap is not None:
            # sqrt 在 0 处导数无穷；raw=0 起步时 delta=0、rms=0，若不夹紧会
            # 让 `0 * inf = nan` 顺着 clamp 反传污染第一步（实测过）。
            rms = torch.sqrt((delta * delta).mean().clamp_min(1e-12))
            cap = torch.as_tensor(float(phase_rms_cap), dtype=delta.dtype, device=delta.device)
            delta = delta * torch.clamp(cap / rms, max=1.0)
        phase = init + delta
        predicted = _predict(phase)
        correlation = _pearson(predicted, target_t, torch)
        rel_l2 = torch.linalg.norm(predicted - target_t) / torch.clamp(torch.linalg.norm(target_t), min=1e-9)
        loss = (1.0 - correlation) + rel_l2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        history.append(value)
        if on_iteration is not None:
            on_iteration(step, value)
        if value < best_loss:
            best_loss = value
            best_phase = (init + delta).detach().clone()

    final = best_phase.detach().to("cpu").numpy()
    delta_np = final - np.asarray(initial_phase, dtype=np.float32)
    return PhaseProposal(
        phase=np.asarray(final, dtype=np.float64),
        predicted_score=float(best_loss),
        max_step_rad=float(np.abs(delta_np).max()) if delta_np.size else 0.0,
        rms_rad=float(np.sqrt(np.mean(np.square(delta_np)))) if delta_np.size else 0.0,
        iterations=int(iterations),
        history=tuple(history),
    )


# --------------------------------------------------------------------------- #
# 在线适配：累计样本 → warm-start 批量重拟合 → 独立验证后晋级
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AdaptationPolicy:
    """何时重拟合、何时晋级。默认值是有意保守的起点，不是光路通用精度保证。"""

    min_train_rows: int = 4
    min_validation_rows: int = 2
    refit_every: int = 2
    promotion_margin: float = 0.01
    iterations: int = 40
    lr: float = 0.01
    adaptation: str = "alignment"


@dataclass(frozen=True)
class AdaptationResult:
    """一次适配尝试的结论。``status`` 只取下列值之一。"""

    status: str  # promoted | rejected | insufficient | failed
    detail: str
    validation_score: float | None = None
    incumbent_score: float | None = None
    train_rows: int = 0
    validation_rows: int = 0
    checkpoint: str | None = None
    quality: dict[str, float] | None = None

    @property
    def promoted(self) -> bool:
        return self.status == "promoted"


def promotion_decision(
    candidate: float | None,
    incumbent: float | None,
    *,
    margin: float,
) -> tuple[bool, str]:
    """晋级判据：候选分数必须比在位检查点至少好 ``margin``，且两者都有限。

    分数越低越好（``1 - correlation + relative_l2``）。相等或更差一律不晋级：
    "没有变坏"不足以证明适配有效。
    """
    if candidate is None or incumbent is None:
        return False, "缺少可比较的验证分数"
    if not (np.isfinite(candidate) and np.isfinite(incumbent)):
        return False, "验证分数不是有限值"
    if candidate <= incumbent - float(margin):
        return True, f"验证分数 {candidate:.4f} 优于在位 {incumbent:.4f}"
    return False, f"验证分数 {candidate:.4f} 未超过在位 {incumbent:.4f} - {float(margin):.4f}"


class OnlineTransferAdapter:
    """用累计的合规样本对 transfer 物理预测模型做增量适配。

    这里"在线"指 **采集 → 追加合规样本 → warm-start 累计重拟合 → 独立验证 →
    晋级**，与硬件轮次同步推进；不是每帧一次梯度更新。每次重拟合都从**在位检查
    点**出发、用全部累计训练行，所以旧样本不会因重建模型而被遗忘，优化器动量也
    不假装能跨轮恢复。

    会话目录就是一个**真正的 transfer 数据集**：``manifest.json`` + 每行一个
    ``sample_XXXX.npz``（``model_phase`` / ``raw_frames``）。每轮多写约 5 MB
    （一帧 1944x2592 uint8），换来三件事：

    * **可重放**：整场闭环能被离线 ``fit --dataset`` 复算，而不必信任内存状态；
    * **契约单一**：``build_fit_model`` 只认磁盘 manifest，不引入第二套内存分支；
    * **可审计**：晋级检查点的训练数据就在旁边，模型变差时能追到具体轮次。

    几何与原点仍然必须来自一次**开场标定采集**（``reference_start`` + 两轴
    ``±ramp``）。没给 ``source_dataset`` 时适配不可用 —— 这不是可以放宽的门槛：
    ``load_dataset`` 缺 reference 会直接抛 ``valid reference_start required``，
    缺负 ramp 会抛 ``missing negative ramp``。
    """

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        session_dir: str | Path,
        span: int,
        device: str | None = None,
        policy: AdaptationPolicy = AdaptationPolicy(),
        radiometry: str = "exposure",
        source_dataset: str | Path | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.session_dir = Path(session_dir)
        self.span = int(span)
        self.device = device
        self.policy = policy
        self.radiometry = radiometry
        self.rows: list[dict[str, Any]] = []
        self.seeded = False
        self.seed_error: str | None = None
        self._pending_since_refit = 0
        self._next_index = 0
        self._active_forward: Any | None = None
        self._active_xy: np.ndarray | None = None
        if source_dataset is not None:
            try:
                self.seed_from_source(source_dataset)
            except Exception as exc:  # 标定缺失是可报告的结论
                # 开场标定失败不该让整页崩掉：闭环仍可只做实测，只是不训练模型。
                self.seed_error = f"开场标定不可用：{exc}"

    # -- 开场标定 ---------------------------------------------------------- #

    def seed_from_source(self, source: str | Path) -> int:
        """把既有采集数据集里的标定行与探针行复制进会话目录。

        复制而不是引用：``load_dataset`` 只按 ``directory / row["file"]`` 取载荷，
        指向外部目录会让数据集在源目录被移走或清理后**静默失效**（缺文件但不缺
        manifest 行）。复制量是 9 个标定文件 + 少量探针，远小于整场采集。

        返回新复制的载荷文件数。
        """
        import json
        import shutil

        source = Path(source)
        rows = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        usable = [
            row
            for row in rows
            if row.get("label") in {"reference_start", "reference_end"}
            or str(row.get("label", "")).startswith(("ramp_", "probe_"))
        ]
        if not any(row.get("label") == "reference_start" for row in usable):
            raise ValueError("source dataset has no reference_start row")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for row in usable:
            name = row["file"]
            target = self.session_dir / name
            if not target.is_file():
                shutil.copy2(source / name, target)
                copied += 1
            self.rows.append(dict(row))
        self.seeded = True
        self.write_manifest()
        return copied

    # -- 数据登记 ---------------------------------------------------------- #

    def add_round(
        self,
        *,
        phase: np.ndarray,
        frame: np.ndarray,
        exposure_ms: float,
        spot_xy: tuple[float, float],
        label: str,
        split: str = "train",
        full_scale: float = 255.0,
    ) -> str:
        """登记一轮合规实测样本，返回载荷文件名。

        ``split`` 由**事先声明的规则**决定（闭环里是"每 N 轮留一轮做验证族"），
        不能事后按结果挑：用训练过的轮次当验证会系统性高估适配效果。
        """
        import json

        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")
        data = np.asarray(frame)
        if data.ndim != 2:
            raise ValueError(f"frame must be 2-D, got {data.shape}")
        scale = float(full_scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("full_scale must be finite and positive")
        exposure = float(exposure_ms)
        if not np.isfinite(exposure) or exposure <= 0:
            raise ValueError("exposure_ms must be finite and positive")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        name = f"sample_{self._next_index:04d}.npz"
        self._next_index += 1
        pixels = np.clip(data, 0, scale)
        pixels = pixels.astype(np.uint16 if scale > 255 else np.uint8)
        np.savez_compressed(
            self.session_dir / name,
            model_phase=np.asarray(phase, dtype=np.float64),
            raw_frames=pixels[None, ...],
        )
        row = {
            "file": name,
            "label": label,
            "split": split,
            "exposure_ms": exposure,
            "peak": float(pixels.max()),
            "full_scale": scale,
            "settled": {"settled": True},
            "exposure_status": "ok",
            "spot_xy": [float(spot_xy[0]), float(spot_xy[1])],
            "frames": {"frames": 1},
            "source": "closed_loop",
        }
        self.rows.append(row)
        self._pending_since_refit += 1
        # 立即落 manifest: 进程被杀也不会留下"有载荷没登记"的半轮。
        path = self.write_manifest()
        (path.parent / f"{Path(name).stem}.row.json").write_text(
            json.dumps(row, indent=2), encoding="utf-8"
        )
        return name

    # -- 就绪与重拟合 ------------------------------------------------------ #

    def counts(self) -> tuple[int, int]:
        train = sum(1 for row in self.rows if row.get("split") == "train")
        validation = sum(1 for row in self.rows if row.get("split") == "validation")
        return train, validation

    def ready(self) -> bool:
        """是否够一次适配尝试（行数 + 自上次重拟合以来的新样本）。"""
        if not self.seeded:
            return False
        train, validation = self.counts()
        return (
            train >= self.policy.min_train_rows
            and validation >= self.policy.min_validation_rows
            and self._pending_since_refit >= self.policy.refit_every
        )

    def write_manifest(self) -> Path:
        import json

        self.session_dir.mkdir(parents=True, exist_ok=True)
        path = self.session_dir / "manifest.json"
        path.write_text(json.dumps(self.rows, indent=2), encoding="utf-8")
        return path

    def refit(self) -> AdaptationResult:
        """累计重拟合一次；**只有独立验证确认更好**才替换在位检查点。"""
        train, validation = self.counts()
        if not self.seeded:
            return AdaptationResult(
                status="insufficient",
                detail=self.seed_error or "未提供开场标定（reference_start + ramp_*），不改模型",
                train_rows=train,
                validation_rows=validation,
            )
        if train < self.policy.min_train_rows or validation < self.policy.min_validation_rows:
            return AdaptationResult(
                status="insufficient",
                detail=f"训练 {train}/验证 {validation} 行不足",
                train_rows=train,
                validation_rows=validation,
            )

        import torch

        from ao_shaping.tools.slm.transfer.adaptation import validation_quality
        from ao_shaping.tools.slm.transfer.fit import build_fit_model, fit_loss

        try:
            self.write_manifest()
            candidate = build_fit_model(
                self.session_dir,
                self.checkpoint,
                span=self.span,
                device=self.device,
                adaptation=self.policy.adaptation,
                radiometry=self.radiometry,
            )
            optimizer = candidate.model.optimizer
            best_entry: dict[str, Any] | None = None
            best_loss = float("inf")
            for step in range(int(self.policy.iterations)):
                loss, entry, _ = fit_loss(candidate.model, candidate, iteration=step)
                # 迭代选择只看**训练**损失；验证集留到晋级判据里用一次。
                if entry["loss"] < best_loss:
                    best_loss = float(entry["loss"])
                    best_entry = entry
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    candidate.model.d.clamp_(-2, 2)
                    candidate.model.center.clamp_(0, 1920)
                    candidate.model.log_width.clamp_(
                        float(np.log(80)), float(np.log(800))
                    )
            if best_entry is None:
                raise RuntimeError("no training iteration completed")
            quality = validation_quality(best_entry, candidate.rows)
            candidate_score = float(quality["score"])

            incumbent = build_fit_model(
                self.session_dir,
                self.checkpoint,
                span=self.span,
                device=self.device,
                adaptation=self.policy.adaptation,
                radiometry=self.radiometry,
            )
            with torch.no_grad():
                _, incumbent_entry, _ = fit_loss(incumbent.model, incumbent)
            incumbent_quality = validation_quality(incumbent_entry, incumbent.rows)
            incumbent_score = float(incumbent_quality["score"])
        except Exception as exc:  # 适配失败必须是可报告的结论
            return AdaptationResult(
                status="failed",
                detail=f"重拟合失败：{exc}",
                train_rows=train,
                validation_rows=validation,
            )

        promote, detail = promotion_decision(
            candidate_score, incumbent_score, margin=self.policy.promotion_margin
        )
        if not promote:
            return AdaptationResult(
                status="rejected",
                detail=detail,
                validation_score=candidate_score,
                incumbent_score=incumbent_score,
                train_rows=train,
                validation_rows=validation,
                quality=quality,
            )

        checkpoint = self._export_checkpoint(best_entry, candidate)
        self.checkpoint = checkpoint
        self._pending_since_refit = 0
        self._active_forward = None
        self._active_xy = None
        return AdaptationResult(
            status="promoted",
            detail=detail,
            validation_score=candidate_score,
            incumbent_score=incumbent_score,
            train_rows=train,
            validation_rows=validation,
            checkpoint=str(checkpoint),
            quality=quality,
        )

    def _export_checkpoint(self, entry: Mapping[str, Any], comps: Any) -> Path:
        """按 ``fit`` 的同一 schema 写出自包含检查点（可被 ``load_frozen_parameters`` 直读）。

        字段必须补齐 ``ramp_matrix`` / ``grid_size`` / ``fft_size`` /
        ``pixel_samples`` / 配准位移，否则重载会退回 ``config.json`` 上游链 ——
        那条链指向的是**原始采集**目录，闭环会话里并不存在。
        """
        import json

        target = self.session_dir / f"promoted_{len(self.rows):04d}"
        target.mkdir(parents=True, exist_ok=True)
        parameters = dict(
            **dict(entry["parameters"]),
            ramp_matrix=comps.matrix.tolist(),
            grid_size=int(comps.old.grid_size),
            fft_size=int(comps.old.fft_size),
            pixel_samples=int(comps.old.pixel_samples),
            registration_shift_ccd_px=float(comps.model.registration_value),
        )
        (target / "frozen_parameters.json").write_text(
            json.dumps(parameters, indent=2), encoding="utf-8"
        )
        return target

    # -- 供候选生成用的物理预测模型 -------------------------------------------- #

    def forward(self, xy: np.ndarray) -> Any:
        """返回当前在位检查点对应的 ``FrozenDefocusForward``（惰性、按网格缓存）。

        ``xy`` 一旦变化就重建：物理预测模型把相机采样坐标固化在构造里，拿旧实例喂新
        网格会静默给出错位的预测。
        """
        from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
            FrozenDefocusForward,
            load_frozen_parameters,
        )

        if (
            self._active_forward is None
            or self._active_xy is None
            or not np.array_equal(self._active_xy, xy)
        ):
            params = load_frozen_parameters(self.checkpoint)
            self._active_forward = FrozenDefocusForward(
                params, np.asarray(xy), device=self.device
            )
            self._active_xy = np.asarray(xy)
        return self._active_forward


# --------------------------------------------------------------------------- #
# 设备事务与轮次编排
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CapturedFrame:
    """一次采集的结果。``frame_id`` 用于证明它晚于本次相位写入。"""

    frame: np.ndarray
    frame_id: int
    exposure_ms: float
    full_scale: float
    monotonic_ns: int


class Bench(Protocol):
    """单线程串行的设备事务。实现方负责打开/关闭设备与稳定等待。

    ``open`` 由 :meth:`ClosedLoopRunner.run` 在 worker 线程里、任何其他调用之前
    执行一次（打开设备、设曝光、丢过渡帧都在这里）；``close`` 在 ``finally`` 里
    调用，且必须可重复调用。``apply_phase`` 必须在返回前完成"写入确认 + 稳定"，
    否则闭环会拿旧图案配新帧。``capture`` 只取**新**帧（必要时丢弃过渡帧），
    不得返回缓冲旧帧。
    """

    def open(self) -> None: ...

    def describe(self) -> dict[str, Any]: ...

    def apply_phase(self, phase: np.ndarray) -> str: ...

    def capture(self) -> CapturedFrame: ...

    def close(self) -> None: ...


@dataclass
class RoundRecord:
    """一轮的完整记录。页面与日志都从这里取数，不各自重算。"""

    round_index: int
    phase_id: str
    frame_id: int
    exposure_ms: float
    gate: FrameGate
    metrics: dict[str, float] | None = None
    measured_score: float | None = None
    predicted_score: float | None = None
    phase_step_rms: float | None = None
    window_fraction: float | None = None
    decision: str | None = None  # baseline | accept | reject | auto | gate_fail | hold | rollback | invalid | fault
    adaptation: str | None = None
    note: str = ""
    #: 采集时刻（``time.monotonic()`` 秒），hold 模式用来和 SLM 写入时刻配对。
    captured_at_s: float | None = None
    #: 本轮漂移反馈倾斜的峰值（rad，相位网格上的 max|tilt|）；未启用反馈时为 ``None``。
    drift_feedback_tilt_rad: float | None = None
    #: 本轮反馈据以计算的质心残差 (x, y)，单位是 **相机像素**（相对基线帧）。
    drift_residual_px: tuple[float, float] | None = None


@dataclass(frozen=True)
class LoopEvent:
    """控制器 → 页面的事件。页面只渲染，不推断状态。"""

    kind: str
    payload: Any = None
    state: LoopState | None = None


@dataclass
class LoopSummary:
    state: LoopState
    rounds: list[RoundRecord] = field(default_factory=list)
    applied_phase: np.ndarray | None = None
    best_phase: np.ndarray | None = None
    best_score: float | None = None
    applied_score: float | None = None
    reason: str = ""
    adaptation: AdaptationResult | None = None


@dataclass(frozen=True)
class ClosedLoopConfig:
    """一次闭环运行的固定契约。开跑后不可变。"""

    xy: np.ndarray
    target: np.ndarray
    start_phase: np.ndarray
    max_rounds: int = 8
    #: ``optimize``：逐轮优化，收敛后即停（旧行为）。
    #: ``hold``：目标保持，收敛后继续锁定同一目标与固定 CCD 像素位置。
    mode: str = "optimize"
    #: ``hold`` 模式的位置容差，单位是 **CCD 像素**（由监督器换算到分析网格）。
    #: 只有在 ``lock_config`` 为 ``None``（未标定）时才用它推导网格单位容差。
    hold_target_ccd_px: float = 5.0
    #: 标定后的监督器阈值。``None`` 表示沿用默认 LockConfig + 位置容差换算。
    #: 提供方负责把位置容差换算成网格单位；这里不二次改写，避免两处口径打架。
    lock_config: LockConfig | None = None
    candidate_policy: str = "confirm"  # confirm | auto
    auto_accept_margin: float = 0.0
    max_consecutive_gate_failures: int = 3
    max_rounds_without_improvement: int = 5
    step_iterations: int = 60
    step_lr: float = 0.02
    max_step_rad: float = 0.35
    phase_rms_cap: float | None = 1.2
    adapt: bool = True
    exposure_ms: float | None = None
    validation_every: int = 3
    thresholds: GateThresholds = GateThresholds()
    #: 漂移反馈：hold 模式下用 Jacobian + P 控制器把实测质心拉回基线位置。
    #: 默认关闭 —— 关闭时既不计算 Jacobian，也不改写下发相位，旧行为逐位不变。
    drift_feedback: bool = False
    #: P 控制器增益（无量纲）。反馈项每轮按最新残差重算，不累加到已下发相位。
    drift_feedback_gain: float = 0.5
    #: 单轮反馈倾斜的峰值上限（rad）；超过即判故障，宁可停机也不下发离谱相位。
    drift_feedback_max_rad: float = 10.0

    def __post_init__(self) -> None:
        if self.candidate_policy not in {"confirm", "auto"}:
            raise ValueError("candidate_policy must be 'confirm' or 'auto'")
        if self.mode not in {"optimize", "hold"}:
            raise ValueError("mode must be 'optimize' or 'hold'")
        if self.mode == "hold":
            if not np.isfinite(self.hold_target_ccd_px) or self.hold_target_ccd_px <= 0:
                raise ValueError("hold_target_ccd_px must be finite and positive")
            if self.candidate_policy != "auto":
                raise ValueError("hold mode runs unattended; candidate_policy must be 'auto'")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        xy = np.asarray(self.xy)
        target = np.asarray(self.target)
        phase = np.asarray(self.start_phase)
        if xy.ndim != 3 or xy.shape[-1] != 2:
            raise ValueError("xy must be (H, W, 2)")
        if target.shape != xy.shape[:2]:
            raise ValueError("target must match the analysis grid")
        # 相位在 **SLM 控制网格**上，与目标/相机网格尺度不同，不能要求同形。
        if phase.ndim != 2:
            raise ValueError("start_phase must be a 2-D control-grid phase")
        if self.exposure_ms is not None:
            if not np.isfinite(self.exposure_ms) or self.exposure_ms <= 0:
                raise ValueError("exposure_ms must be finite and positive")
        if self.validation_every < 0:
            raise ValueError("validation_every must be >= 0")
        if self.drift_feedback:
            if not np.isfinite(self.drift_feedback_gain) or self.drift_feedback_gain <= 0:
                raise ValueError("drift_feedback_gain must be finite and positive")
            if (
                not np.isfinite(self.drift_feedback_max_rad)
                or self.drift_feedback_max_rad <= 0
            ):
                raise ValueError("drift_feedback_max_rad must be finite and positive")


class ClosedLoopRunner:
    """闭环绕编排：**阻塞式**跑完整场，由工作线程调用。

    只在轮次边界响应暂停/停止/接受/拒绝；一轮内部的"写相位—采集—配对"不会被
    打断，因此不会出现半完成的事务。
    """

    def __init__(
        self,
        bench: Bench,
        *,
        config: ClosedLoopConfig,
        forward: Any | None = None,
        adapter: OnlineTransferAdapter | None = None,
        on_event: Callable[[LoopEvent], None] | None = None,
        on_frame: Callable[[CapturedFrame, RoundRecord | None], None] | None = None,
    ) -> None:
        self.bench = bench
        self.config = config
        self.forward = forward
        self.adapter = adapter
        self._on_event = on_event
        self._on_frame = on_frame
        self.state = LoopState.IDLE
        self.summary = LoopSummary(state=LoopState.IDLE)

        self._lock = threading.Lock()
        self._decision: str | None = None
        self._decision_ready = threading.Event()
        self._pause = threading.Event()
        self._pause.set()
        self._stop = threading.Event()
        # 漂移反馈：Jacobian 按 forward 实例（id）缓存；适配晋级换实例后会重新测。
        self._drift_jac: dict[int, np.ndarray | None] = {}
        self._drift_warned: set[int] = set()

    # -- 外部控制（GUI 线程） ---------------------------------------------- #

    def request_accept(self) -> None:
        self._set_decision("accept")

    def request_reject(self) -> None:
        self._set_decision("reject")

    def request_pause(self) -> None:
        if self.state in {LoopState.STOPPED, LoopState.FAULT}:
            return
        self._pause.clear()
        self._emit(LoopEvent("pause_requested"), state=LoopState.PAUSED)

    def request_resume(self) -> None:
        self._pause.set()

    def request_stop(self) -> None:
        self._stop.set()
        self._pause.set()
        self._set_decision("stop")
        self._emit(LoopEvent("stop_requested"), state=LoopState.STOPPING)

    def _set_decision(self, value: str) -> None:
        with self._lock:
            self._decision = value
        self._decision_ready.set()

    # -- 事件 -------------------------------------------------------------- #

    def _emit(self, event: LoopEvent, state: LoopState | None = None) -> None:
        if state is not None:
            self.state = state
            self.summary.state = state
            event = LoopEvent(event.kind, event.payload, state)
        if self._on_event is not None:
            self._on_event(event)

    def _set_state(self, state: LoopState, note: str = "") -> None:
        self._emit(LoopEvent("state", note), state=state)

    # -- 主流程 ------------------------------------------------------------ #

    def run(self) -> LoopSummary:
        """按 :attr:`ClosedLoopConfig.mode` 分派到同一设备事务下的两种语义。

        ``optimize`` 逐轮优化到收敛/上限；``hold`` 保持同一目标与固定 CCD
        像素位置。两条路径共用门禁与故障收尾，差别只在停止条件谁说了算。
        """
        if self.config.mode == "hold":
            return self._run_hold()
        return self._run_optimize()

    def _run_optimize(self) -> LoopSummary:
        """跑完整场并返回摘要；任何故障都落到 ``FAULT`` 并带上原因。

        一条不可妥协的顺序：**帧的门禁先于一切判断**。门禁未通过的帧没有资格被
        接受、被比较或被喂给适配 —— 否则"饱和帧 = 分数 0 = 看起来最好"会把闭环
        直接推向过曝。
        """
        config = self.config
        self._set_state(LoopState.PREFLIGHT)
        grid = np.asarray(config.xy)
        target = np.asarray(config.target, dtype=np.float64)
        applied_phase = np.asarray(config.start_phase, dtype=np.float64)
        opened = False

        try:
            # 设备事务的唯一入口: 打开并配置 SLM/CCD（含曝光与过渡帧丢弃）。
            # 缺了这一步, describe/apply_phase/capture 都无从谈起 —— 真实的
            # HardwareBench 会直接抛 "open() must be called first"。
            self.bench.open()
            opened = True
            info = self.bench.describe()
            self._emit(LoopEvent("bench", info))

            baseline = self._measure(applied_phase, "r00", 0, None)
            if not baseline.gate.ok:
                baseline.decision = "baseline_fail"
                baseline.note = describe_gate(baseline.gate)
                self._emit(LoopEvent("round", baseline))
                return self._finish(
                    LoopState.FAULT,
                    f"起点相位未取得有效帧，未开始闭环：{describe_gate(baseline.gate)}",
                )
            previous_centroid = baseline.gate.centroid_xy
            baseline.decision = "baseline"
            self.summary.applied_phase = applied_phase
            self.summary.best_phase = applied_phase
            self.summary.applied_score = baseline.measured_score
            self.summary.best_score = baseline.measured_score
            self.summary.rounds.append(baseline)
            self._emit(LoopEvent("round", baseline), state=LoopState.MEASURING)

            consecutive_failures = 0
            since_improvement = 0
            for index in range(1, int(config.max_rounds) + 1):
                boundary = self._await_boundary()
                if boundary == "stop":
                    return self._finish(LoopState.STOPPED, "运行中停止：当前相位保持")

                adaptation: AdaptationResult | None = None
                if config.adapt and self.adapter is not None and self.adapter.ready():
                    self._set_state(LoopState.UPDATING, "适配重拟合")
                    adaptation = self.adapter.refit()
                    self._emit(LoopEvent("adaptation", adaptation))
                    if adaptation.promoted:
                        self.forward = self.adapter.forward(grid)

                forward = self.forward if self.forward is not None else self._adapter_forward(grid)
                if forward is None:
                    return self._finish(LoopState.FAULT, "没有可用的物理预测模型，无法生成候选")

                self._set_state(LoopState.UPDATING, f"生成第 {index} 轮候选")
                proposal = refine_phase(
                    forward,
                    target,
                    self.summary.applied_phase,
                    iterations=config.step_iterations,
                    lr=config.step_lr,
                    max_step_rad=config.max_step_rad,
                    phase_rms_cap=config.phase_rms_cap,
                    exposure_ms=config.exposure_ms,
                )

                self._set_state(LoopState.MEASURING, f"下发第 {index} 轮候选")
                record = self._measure(proposal.phase, f"r{index:02d}", index, previous_centroid)
                record.predicted_score = proposal.predicted_score
                record.phase_step_rms = proposal.rms_rad
                record.adaptation = adaptation.status if adaptation is not None else None

                if not record.gate.ok:
                    # 门禁失败 = 本轮没有可用证据。回滚到已确认相位，绝不"因为
                    # 分数看起来是 0 而接受"。
                    record.decision = "gate_fail"
                    record.note = describe_gate(record.gate)
                    consecutive_failures += 1
                    self.bench.apply_phase(self.summary.applied_phase)
                    self.summary.rounds.append(record)
                    self._emit(LoopEvent("round", record), state=LoopState.MEASURING)
                    if consecutive_failures >= config.max_consecutive_gate_failures:
                        return self._finish(
                            LoopState.FAULT,
                            f"连续 {consecutive_failures} 轮帧门禁未通过，已停止",
                        )
                    continue

                consecutive_failures = 0
                previous_centroid = record.gate.centroid_xy

                applied = self.summary.applied_score
                measured = record.measured_score
                if applied is None or measured is None:
                    record.decision = "gate_fail"
                    record.note = "缺少可比较的实测分数"
                    self.bench.apply_phase(self.summary.applied_phase)
                    self.summary.rounds.append(record)
                    self._emit(LoopEvent("round", record), state=LoopState.MEASURING)
                    continue

                if config.candidate_policy == "auto":
                    accepted = measured <= applied - config.auto_accept_margin
                    record.decision = "auto" if accepted else "reject"
                    if not accepted:
                        record.note = "实测未改善，自动拒绝"
                else:
                    self._set_state(LoopState.AWAITING_DECISION, f"第 {index} 轮候选待确认")
                    decision = self._await_decision(record)
                    if decision == "stop":
                        self.bench.apply_phase(self.summary.applied_phase)
                        return self._finish(LoopState.STOPPED, "候选待确认时停止：当前相位保持")
                    accepted = decision == "accept"
                    record.decision = "accept" if accepted else "reject"

                if accepted:
                    self.summary.applied_phase = proposal.phase
                    self.summary.applied_score = measured
                    if self.summary.best_score is None or measured < self.summary.best_score:
                        self.summary.best_score = measured
                        self.summary.best_phase = proposal.phase
                    since_improvement = 0
                    if config.adapt and self.adapter is not None:
                        # 验证族按**事先声明的规则**划分（每 N 个被接受的轮次留一
                        # 轮），不能事后按结果挑：用训练过的轮次做验证会系统性高估
                        # 适配效果。
                        holdout = (
                            config.validation_every > 0
                            and index % int(config.validation_every) == 0
                        )
                        try:
                            self.adapter.add_round(
                                phase=proposal.phase,
                                frame=self._last_frame,
                                exposure_ms=record.exposure_ms,
                                spot_xy=record.gate.centroid_xy or (0.0, 0.0),
                                label=f"round_{index:02d}",
                                split="validation" if holdout else "train",
                                full_scale=record.gate.full_scale,
                            )
                        except Exception as exc:  # 落盘失败不该毁掉闭环
                            record.note = f"{record.note}（样本登记失败：{exc}）".strip()
                else:
                    # 回滚 = 重新下发最后确认过的相位；候选从未成为"当前相位"。
                    self.bench.apply_phase(self.summary.applied_phase)
                    since_improvement += 1
                    self.summary.rounds.append(record)
                    self._emit(LoopEvent("round", record), state=LoopState.MEASURING)
                    if since_improvement >= config.max_rounds_without_improvement:
                        return self._finish(
                            LoopState.STOPPED,
                            f"连续 {since_improvement} 轮候选未被接受，已停止",
                        )
                    continue

                self.summary.rounds.append(record)
                self._emit(LoopEvent("round", record), state=LoopState.MEASURING)

            return self._finish(LoopState.STOPPED, "达到最大轮数，当前相位保持")
        except Exception as exc:  # 设备异常必须变成可报告状态
            # 故障时也要尽量把 SLM 留在最后一个已确认相位上；回滚失败只记录。
            # 设备没打开成功时不能下发（否则又是一条 "must be called first"）。
            try:
                if opened and self.summary.applied_phase is not None:
                    self.bench.apply_phase(self.summary.applied_phase)
            except Exception as exc:  # 收尾回滚是尽力而为
                logger.warning("故障回滚相位失败，SLM 可能停在未确认相位: {}", exc)
            return self._finish(LoopState.FAULT, f"运行异常：{exc}")
        finally:
            try:
                self.bench.close()
            except Exception as exc:  # 关闭失败不该掩盖主结论
                logger.warning("收尾关闭设备失败（结论照常落盘）: {}", exc)

    def _run_hold(self) -> LoopSummary:
        """固定目标、固定 CCD 像素位置的保持模式。

        与 :meth:`_run_optimize` 共用设备事务与门禁，但停止条件归
        :class:`TargetLockSupervisor` 所有：位置漂移、连续无效帧与振荡由监督器
        判定，本方法只执行它给出的 ``accept`` / ``propose`` / ``rollback``。目标在
        ``arm()`` 时固定，容差由 ``hold_target_ccd_px`` 换算到分析网格，因此每轮
        比较的都是同一张图的**绝对坐标**，不是"重新居中后的形状"。
        """
        # 惰性导入：target_lock 只依赖 numpy/stdlib；本模块顶层仍保持
        # "只导入 numpy/stdlib" 的既有承诺。
        from ao_shaping.gui.slm.target_lock import (
            FixedTarget,
            LockAction,
            LockConfig,
            MeasuredFrame,
            TargetLockSupervisor,
        )

        config = self.config
        self._set_state(LoopState.PREFLIGHT)
        grid = np.asarray(config.xy)
        target = np.asarray(config.target, dtype=np.float64)
        applied_phase = np.asarray(config.start_phase, dtype=np.float64)
        opened = False

        def measured_frame(record: RoundRecord, write_seq: int) -> MeasuredFrame:
            captured_at_s = record.captured_at_s
            return MeasuredFrame(
                image=sample_on_grid(self._last_frame, config.xy),
                write_seq=int(write_seq),
                capture_seq=int(record.frame_id),
                # 缺失采集时刻只能当成"早于写入"：宁可判旧帧，也不放宽配对。
                captured_at_s=(
                    float(captured_at_s) if captured_at_s is not None else float("-inf")
                ),
                exposure_ms=float(record.exposure_ms),
            )

        try:
            self.bench.open()
            opened = True
            info = self.bench.describe()
            self._emit(LoopEvent("bench", info))

            spacing_ccd_px = self._grid_spacing_px()
            if spacing_ccd_px is None:
                return self._finish(
                    LoopState.FAULT,
                    "分析网格没有正的有限步长，无法把保持容差换算成 CCD 像素",
                )

            lock_config = config.lock_config
            if lock_config is None:
                # 未标定：沿用原来的默认阈值，位置容差仍由页面输入换算。
                lock_config = LockConfig(
                    max_centroid_error_px=float(config.hold_target_ccd_px) / spacing_ccd_px
                )
            threshold_source = "标定" if config.lock_config else "默认"
            self._emit(LoopEvent("state", f"监督器阈值（{threshold_source}）：{lock_config}"))
            supervisor = TargetLockSupervisor(lock_config)
            supervisor.arm(FixedTarget(target))

            # 基线写入同样要登记：序号 0，写入时刻先于本次下发。
            supervisor.expect_write(0, written_at_s=time.monotonic())
            baseline = self._measure(applied_phase, "r00", 0, None)
            if not baseline.gate.ok:
                baseline.decision = "baseline_fail"
                baseline.note = describe_gate(baseline.gate)
                self._emit(LoopEvent("round", baseline))
                return self._finish(
                    LoopState.FAULT,
                    f"起点相位未取得有效帧，未开始保持：{describe_gate(baseline.gate)}",
                )

            previous_centroid = baseline.gate.centroid_xy
            self.summary.applied_phase = applied_phase
            self.summary.best_phase = applied_phase
            self.summary.rounds.append(baseline)
            baseline.decision = "baseline"
            self._emit(LoopEvent("round", baseline), state=LoopState.MEASURING)

            decision = supervisor.observe(measured_frame(baseline, 0))
            if decision is not None:
                self._emit_lock(decision, 0, spacing_ccd_px, supervisor.target_centroid)
            if decision is None or decision.action is LockAction.FAULT:
                return self._finish(LoopState.FAULT, self._lock_fault_reason(supervisor, decision))
            self.summary.applied_score = decision.score
            self.summary.best_score = decision.best_score

            consecutive_failures = 0
            for index in range(1, int(config.max_rounds) + 1):
                boundary = self._await_boundary()
                if boundary == "stop":
                    return self._finish(LoopState.STOPPED, "运行中停止：当前相位保持")
                # hold 恒为 auto 策略：accept/reject 在本模式没有意义，忽略即可。

                adaptation: AdaptationResult | None = None
                if config.adapt and self.adapter is not None and self.adapter.ready():
                    self._set_state(LoopState.UPDATING, "适配重拟合")
                    adaptation = self.adapter.refit()
                    self._emit(LoopEvent("adaptation", adaptation))
                    if adaptation.promoted:
                        self.forward = self.adapter.forward(grid)

                forward = self.forward if self.forward is not None else self._adapter_forward(grid)
                if forward is None:
                    return self._finish(LoopState.FAULT, "没有可用的物理预测模型，无法生成候选")

                self._set_state(LoopState.UPDATING, f"生成第 {index} 轮候选")
                proposal = refine_phase(
                    forward,
                    target,
                    self.summary.applied_phase,
                    iterations=config.step_iterations,
                    lr=config.step_lr,
                    max_step_rad=config.max_step_rad,
                    phase_rms_cap=config.phase_rms_cap,
                    exposure_ms=config.exposure_ms,
                )

                # 漂移反馈（hold 模式可选）：残差 = 上一有效帧（首轮即基线帧）相对
                # 基线质心的位移，每轮按最新残差重算，不累加到 applied_phase。
                feedback_tilt: np.ndarray | None = None
                feedback_peak_rad: float | None = None
                residual_px: tuple[float, float] | None = None
                if config.drift_feedback:
                    baseline_xy = baseline.gate.centroid_xy
                    current_xy = previous_centroid
                    if baseline_xy is not None and current_xy is not None:
                        residual_px = (
                            float(current_xy[0]) - float(baseline_xy[0]),
                            float(current_xy[1]) - float(baseline_xy[1]),
                        )
                        feedback_tilt, feedback_peak_rad = self._drift_feedback(
                            forward, residual_px
                        )
                        if (
                            feedback_peak_rad is not None
                            and feedback_peak_rad > float(config.drift_feedback_max_rad)
                        ):
                            # 超限不走门禁，直接走既有 FAULT 收尾（回滚到纯形状）。
                            note = (
                                f"第 {index} 轮漂移反馈倾斜 {feedback_peak_rad:.3f} rad "
                                f"超过上限 {float(config.drift_feedback_max_rad):.3f} rad，已停止"
                            )
                            self.bench.apply_phase(self.summary.applied_phase)
                            return self._finish(LoopState.FAULT, note)

                self._set_state(LoopState.MEASURING, f"下发第 {index} 轮候选")
                # 写入时刻必须在写之前登记：监督器据此判定"帧是否晚于写入"。
                supervisor.expect_write(index, written_at_s=time.monotonic())
                command = (
                    proposal.phase
                    if feedback_tilt is None
                    else proposal.phase + feedback_tilt
                )
                record = self._measure(command, f"r{index:02d}", index, previous_centroid)
                record.predicted_score = proposal.predicted_score
                record.phase_step_rms = proposal.rms_rad
                record.adaptation = adaptation.status if adaptation is not None else None
                record.drift_feedback_tilt_rad = feedback_peak_rad
                record.drift_residual_px = residual_px

                if not record.gate.ok:
                    # 门禁失败 = 本轮没有可用证据；既不能评分，也不进适配。
                    record.decision = "gate_fail"
                    record.note = describe_gate(record.gate)
                    consecutive_failures += 1
                    self.bench.apply_phase(self.summary.applied_phase)
                    self.summary.rounds.append(record)
                    self._emit(LoopEvent("round", record), state=LoopState.MEASURING)
                    if consecutive_failures >= config.max_consecutive_gate_failures:
                        return self._finish(
                            LoopState.FAULT,
                            f"连续 {consecutive_failures} 轮帧门禁未通过，已停止",
                        )
                    continue

                consecutive_failures = 0
                previous_centroid = record.gate.centroid_xy

                decision = supervisor.observe(measured_frame(record, index))
                if decision is not None:
                    self._emit_lock(
                        decision,
                        index,
                        spacing_ccd_px,
                        supervisor.target_centroid,
                        drift_feedback_tilt_rad=feedback_peak_rad,
                        drift_residual_px=residual_px,
                    )

                if decision is None:
                    # 无效帧且监督器尚未冻结：本轮没有证据，回滚到最佳并计数；
                    # 何时冻结由监督器决定，这里不自行判死。
                    reasons = supervisor.last_gate_reasons
                    why = ",".join(reasons) if reasons else "unknown"
                    record.decision = "invalid"
                    record.note = f"监督器未通过：{why}"
                    consecutive_failures += 1
                    self.bench.apply_phase(self.summary.applied_phase)
                    self.summary.rounds.append(record)
                    self._emit(LoopEvent("round", record), state=LoopState.MEASURING)
                    if consecutive_failures >= config.max_consecutive_gate_failures:
                        return self._finish(
                            LoopState.FAULT,
                            f"连续 {consecutive_failures} 轮监督器未通过（{why}），已停止",
                        )
                    continue

                if decision.action is LockAction.FAULT:
                    record.decision = "fault"
                    record.note = decision.reason
                    self.bench.apply_phase(self.summary.applied_phase)
                    self.summary.rounds.append(record)
                    self._emit(LoopEvent("round", record), state=LoopState.MEASURING)
                    return self._finish(
                        LoopState.FAULT, self._lock_fault_reason(supervisor, decision)
                    )

                if decision.action is LockAction.ACCEPT:
                    # 监督器只在实测改善超过余量时 ACCEPT；最佳值以它持有的为准。
                    self.summary.applied_phase = proposal.phase
                    self.summary.best_phase = proposal.phase
                    self.summary.applied_score = decision.score
                    self.summary.best_score = decision.best_score
                    record.decision = "accept"
                elif decision.action is LockAction.PROPOSE:
                    # 噪声内没有改善证据：设备留在最佳相位上，不把候选当成结论。
                    record.decision = "hold"
                    self.bench.apply_phase(self.summary.applied_phase)
                else:  # ROLLBACK
                    record.decision = "rollback"
                    self.bench.apply_phase(self.summary.applied_phase)
                self.summary.rounds.append(record)
                self._emit(LoopEvent("round", record), state=LoopState.MEASURING)

            # 保持模式不因 max_rounds_without_improvement 停止：冻结由监督器负责。
            return self._finish(LoopState.STOPPED, "达到运行轮数上限：保持当前相位")
        except Exception as exc:  # 设备异常必须变成可报告状态
            try:
                if opened and self.summary.applied_phase is not None:
                    self.bench.apply_phase(self.summary.applied_phase)
            except Exception as exc:  # 收尾回滚是尽力而为
                logger.warning("故障回滚相位失败，SLM 可能停在未确认相位: {}", exc)
            return self._finish(LoopState.FAULT, f"运行异常：{exc}")
        finally:
            try:
                self.bench.close()
            except Exception as exc:  # 关闭失败不该掩盖主结论
                logger.warning("收尾关闭设备失败（结论照常落盘）: {}", exc)

    def _grid_spacing_px(self) -> float | None:
        """分析网格相邻采样点的 CCD 像素步长；与页面标定共用同一定义。"""
        return grid_spacing_px(self.config.xy)

    def _predicted_centroid(
        self, forward: Any, phase: np.ndarray, xy: np.ndarray
    ) -> tuple[float, float] | None:
        """在相机像素坐标上预测相位对应的质心；失败时返回 ``None``。"""
        exposure_ms = self.config.exposure_ms
        prediction = (
            forward.predict(phase, exposure_ms=exposure_ms)
            if exposure_ms is not None
            else forward.predict(phase)
        )
        while getattr(prediction, "ndim", 0) > 2:
            prediction = prediction[0]
        if hasattr(prediction, "detach"):
            prediction = prediction.detach().cpu().numpy()
        return _camera_centroid(np.asarray(prediction, dtype=np.float64), xy)

    def _drift_jacobian(
        self,
        forward: Any,
        phase_shape: tuple[int, ...],
        xy: np.ndarray,
    ) -> np.ndarray | None:
        """``∂相机像素质心/∂tilt`` 的 2×2 Jacobian（px/rad），按 forward 实例缓存。

        用零相位与 1 rad 峰谷 tilt-x / tilt-y 斜坡做前向数值微分：列 0 是 tilt-x
        的响应（沿列斜坡），列 1 是 tilt-y 的响应（沿行斜坡）。预测质心用
        ``analysis_grid`` 的相机像素坐标加权，与实测 ``centroid_xy`` 同一口径。
        """
        key = id(forward)
        if key in self._drift_jac:
            return self._drift_jac[key]
        jac: np.ndarray | None = None
        zero = np.zeros(phase_shape, dtype=np.float64)
        base = self._predicted_centroid(forward, zero, xy)
        if base is not None:
            jac = np.zeros((2, 2), dtype=np.float64)
            for column, axis in ((0, 1), (1, 0)):
                forecast = self._predicted_centroid(
                    forward, _tilt_ramp(phase_shape, axis), xy
                )
                if forecast is None:
                    jac = None
                    break
                jac[0, column] = forecast[0] - base[0]
                jac[1, column] = forecast[1] - base[1]
        self._drift_jac[key] = jac
        return jac

    def _drift_feedback(
        self,
        forward: Any,
        residual_px: tuple[float, float],
    ) -> tuple[np.ndarray | None, float | None]:
        """按质心残差给出本轮反馈倾斜相位与峰值；不可用时返回 ``(None, None)``。

        符号约定：``J = ∂质心/∂tilt``，把质心拉回基线需要 ``tilt = -g·J⁻¹·e``；
        正号只会把残差越推越大（P 控制器对常量漂移的发散方向）。
        """
        config = self.config
        phase_shape = tuple(np.asarray(config.start_phase).shape)
        jac = self._drift_jacobian(forward, phase_shape, np.asarray(config.xy))
        key = id(forward)
        if jac is None or not np.all(np.isfinite(jac)):
            self._warn_drift_skip(key)
            return None, None
        if float(np.linalg.cond(jac)) > _DRIFT_COND_LIMIT:
            self._warn_drift_skip(key)
            return None, None
        try:
            coeff = np.linalg.solve(jac, np.asarray(residual_px, dtype=np.float64))
        except np.linalg.LinAlgError:
            self._warn_drift_skip(key)
            return None, None
        coeff = -float(config.drift_feedback_gain) * coeff
        tilt = coeff[0] * _tilt_ramp(phase_shape, 1) + coeff[1] * _tilt_ramp(phase_shape, 0)
        return tilt, float(np.max(np.abs(tilt)))

    def _warn_drift_skip(self, key: int) -> None:
        """Jacobian 不可用时只在同一 forward 实例首次出现时记一条 warning。"""
        if key in self._drift_warned:
            return
        self._drift_warned.add(key)
        self._emit(
            LoopEvent("state", "漂移反馈：Jacobian 不可用或近奇异，跳过本轮反馈")
        )

    def _emit_lock(
        self,
        decision: Any,
        round_index: int,
        spacing_ccd_px: float,
        target_centroid: tuple[float, float] | None,
        drift_feedback_tilt_rad: float | None = None,
        drift_residual_px: tuple[float, float] | None = None,
    ) -> None:
        """把监督器决策翻成页面可渲染的 ``lock`` 事件（键固定，缺值用 nan）。"""

        def number(value: float | None) -> float:
            return float("nan") if value is None else float(value)

        error_px = decision.centroid_error_px
        centroid = decision.centroid_xy
        payload = {
            "round_index": int(round_index),
            "lock_state": decision.state.value,
            "action": decision.action.value,
            "score": number(decision.score),
            "best_score": number(decision.best_score),
            "centroid_error_ccd_px": (
                float("nan") if error_px is None else float(error_px) * float(spacing_ccd_px)
            ),
            "target_centroid_xy": (
                [float("nan"), float("nan")]
                if target_centroid is None
                else [float(target_centroid[0]), float(target_centroid[1])]
            ),
            "centroid_xy": (
                [float("nan"), float("nan")]
                if centroid is None
                else [float(centroid[0]), float(centroid[1])]
            ),
        }
        # 漂移反馈字段只在启用时进 payload：关闭时保持既有键集合逐位不变。
        if self.config.drift_feedback:
            payload["drift_feedback_tilt_rad"] = number(drift_feedback_tilt_rad)
            payload["drift_residual_px"] = (
                [float("nan"), float("nan")]
                if drift_residual_px is None
                else [float(drift_residual_px[0]), float(drift_residual_px[1])]
            )
        self._emit(LoopEvent("lock", payload))

    @staticmethod
    def _lock_fault_reason(supervisor: Any, decision: Any | None) -> str:
        """把监督器状态、帧原因与评分历史写成一句可追责的故障原因。"""
        state = getattr(supervisor, "state", None)
        state_name = getattr(state, "value", str(state))
        gate = tuple(getattr(supervisor, "last_gate_reasons", ()) or ())
        gate_text = f"，帧原因：{','.join(gate)}" if gate else ""
        if decision is None:
            detail = "监督器未给出决策（已冻结或帧未通过门禁）"
        else:
            detail = f"{getattr(decision, 'reason', '') or '未知原因'}"
        history = tuple(getattr(supervisor, "score_history", ()) or ())
        recent = (
            "尚无有效评分"
            if not history
            else "最近评分：" + "、".join(f"{float(s):.4f}" for s in history[-3:])
        )
        return f"监督器状态 {state_name}：{detail}{gate_text}；{recent}"

    # -- 内部 -------------------------------------------------------------- #

    _last_frame: np.ndarray

    def _adapter_forward(self, grid: np.ndarray) -> Any | None:
        if self.adapter is None:
            return None
        return self.adapter.forward(grid)

    def _measure(
        self,
        phase: np.ndarray,
        phase_id: str,
        round_index: int,
        previous_centroid: tuple[float, float] | None,
    ) -> RoundRecord:
        """写相位 → 取新帧 → 门禁 → 实测指标。**永远返回记录**（含失败原因）。

        返回 ``None`` 会让调用方被迫用"没有分数"当分数，正是门禁被绕过的入口；
        所以这里改成"记录 + ``gate.ok``"，由调用方显式处理失败分支。
        """
        self.bench.apply_phase(phase)
        captured = self.bench.capture()
        self._last_frame = captured.frame
        gate = check_frame(
            captured.frame,
            full_scale=captured.full_scale,
            exposure_ms=captured.exposure_ms,
            expected_exposure_ms=self.config.exposure_ms,
            thresholds=self.config.thresholds,
            previous_centroid=previous_centroid,
        )
        record = RoundRecord(
            round_index=int(round_index),
            phase_id=phase_id,
            frame_id=captured.frame_id,
            exposure_ms=captured.exposure_ms,
            gate=gate,
            captured_at_s=float(captured.monotonic_ns) / 1e9,
        )
        if self._on_frame is not None:
            self._on_frame(captured, record)
        if not gate.ok:
            return record
        try:
            measured = sample_on_grid(captured.frame, self.config.xy)
            fraction = window_fraction(captured.frame, self.config.xy)
        except ValueError as exc:
            record.gate = FrameGate(
                ok=False,
                reasons=("grid_out_of_bounds",),
                peak=gate.peak,
                full_scale=gate.full_scale,
                saturation_pixels=gate.saturation_pixels,
                centroid_xy=gate.centroid_xy,
                exposure_ms=gate.exposure_ms,
            )
            record.note = str(exc)
            return record
        record.window_fraction = fraction
        try:
            entry = compare_to_target(measured, self.config.target)
        except ValueError as exc:
            record.gate = FrameGate(
                ok=False,
                reasons=("nonfinite",),
                peak=gate.peak,
                full_scale=gate.full_scale,
                saturation_pixels=gate.saturation_pixels,
                centroid_xy=gate.centroid_xy,
                exposure_ms=gate.exposure_ms,
            )
            record.note = str(exc)
            return record
        record.metrics = entry
        record.measured_score = score_from_metrics(entry)
        return record

    def _await_boundary(self) -> str:
        """轮次边界：等待暂停解除；返回 ``stop`` 时调用方须收尾。"""
        while not self._pause.is_set():
            if self._stop.is_set():
                return "stop"
            self._pause.wait(timeout=0.05)
        return "stop" if self._stop.is_set() else "run"

    def _await_decision(self, record: RoundRecord) -> str:
        with self._lock:
            if self._decision is not None:
                decision = self._decision
                self._decision = None
                self._decision_ready.clear()
                return decision
        self._emit(LoopEvent("awaiting_decision", record))
        while not self._decision_ready.wait(timeout=0.2):
            # 停止请求在"等待人工确认"期间也要生效，否则操作员按了停止却卡住。
            if self._stop.is_set():
                return "stop"
        with self._lock:
            decision = self._decision or "reject"
            self._decision = None
            self._decision_ready.clear()
        return decision

    def _finish(self, state: LoopState, reason: str) -> LoopSummary:
        self.summary.reason = reason
        self._set_state(state, reason)
        return self.summary
