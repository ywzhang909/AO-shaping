"""SLM 远场光束整形算法的仿真基准 (Unit B)。

在一套*纯仿真*的环境里比较三种远场整形策略 —— 不需要 SLM / CCD / DM 硬件:

* ``"gs"`` —— :func:`~ao_shaping.algorithm.gerchberg_saxton.gerchberg_saxton`
* ``"backprop"`` —— :func:`~ao_shaping.algorithm.differentiable_shaping.train_beam_shaping`
* ``"spgd-sim"`` —— 一个紧凑的自包含 SPGD 循环, 跑在另外两者所用的同一个
  夫琅禾费 (FFT) 正向模型上, 使三个优化器看到完全相同的传播物理。

每个算法产出一个 SLM 相位图, 随后用**同一个** FFT 传播子把它转回仿真远场
强度, 所以这个基准是公平的正面对比: 各列之间唯一的差别是优化策略,
绝不是正向模型。

曝光 / 亮度不变量指标遵循项目规则 —— 所有强度比较都做归一化, 使绝对尺度
无关紧要。

公开 API:
    - :func:`run_benchmark` (一个算法 × 一个目标形状 → 结果字典)
    - :func:`run_benchmark_suite` (算法 × 形状的网格 →
      ``(list[dict], DataFrame)``)
    - :func:`measure_shaped_area` / :func:`check_area_requirement`
      (固定形状面积门槛)
    - GIF + 指标 CSV/MD 写出器 (PIL / stdlib csv)

本模块面向 Python 3.12+, 无硬件依赖且完全离线;
测试位于 ``tests/ao_shaping/algorithm/``。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from PIL import Image

# 复用 canonical 的算法与指标实现, 使基准测的正是 runner 交付的那套东西,
# 而不是某个再实现。
from ao_shaping.utils.image.beam_metrics import compute_shaping_metrics
from ao_shaping.utils.image.targets import create_target_shape
from ao_shaping.optimizer import train_beam_shaping
from ao_shaping.algorithm.signal_processing.gerchberg_saxton import gerchberg_saxton

# ---------------------------------------------------------------------------
# 常量 / 默认值
# ---------------------------------------------------------------------------
DEFAULT_GRID: tuple[int, int] = (128, 128)
DEFAULT_CELL_SPACING: float = 8e-6
DEFAULT_DISTANCE: float = 0.1
DEFAULT_WAVELENGTH: float = 1064e-9
DEFAULT_TARGET_AREA: int = 256
DEFAULT_ITERATIONS: int = 100
DEFAULT_MAX_FRAMES: int = 40

_SHAPES: set[str] = {"square", "circle", "gaussian"}
_ALGORITHMS: set[str] = {"gs", "backprop", "spgd-sim"}

# 公开别名 (slm_shaping_runner CLI 引用): 已排序的元组, 保证确定性输出顺序.
SUITE_SHAPES: tuple[str, ...] = tuple(sorted(_SHAPES))
SUITE_ALGORITHMS: tuple[str, ...] = tuple(sorted(_ALGORITHMS))


# ---------------------------------------------------------------------------
# 正向模型 (所有算法共用)
# ---------------------------------------------------------------------------
def _propagate_far_field(
    phase: npt.NDArray[np.floating],
    *,
    cell_spacing: float = DEFAULT_CELL_SPACING,
    distance: float = DEFAULT_DISTANCE,
    wavelength: float = DEFAULT_WAVELENGTH,
) -> npt.NDArray[np.floating]:
    """``exp(i*phase)`` 的夫琅禾费远场强度 (FFT 焦面模型)。

    SLM 面光场 ``exp(j*phi)`` (均匀照明, 平坦相位 = 中心处的 0 级) 用单次 FFT
    传播到远场 —— 即 ``gerchberg_saxton(..., propagation="fft")`` 所用的
    同一个焦面模型。返回归一化的*强度* ``|FFT(exp(j*phi))|^2``,
    使所有算法都在同一套物理上比较。

    Args:
        phase: 二维相位图, 单位弧度。
        cell_spacing: 像素间距 (米)。FFT 不用它, 但为对称性保留。
        distance: 传播距离 (米)。FFT 不用它, 但为与 ASM 路径对称而保留。
        wavelength: 波长 (米)。FFT 不用它, 但为对称性保留。

    Returns:
        float64 的二维远场强度, 归一化到总和 == 1。
    """
    field = np.exp(1j * np.asarray(phase, dtype=np.float64))
    ff = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2
    total = float(ff.sum())
    if total > 0:
        ff = ff / total
    return ff


# ---------------------------------------------------------------------------
# 面积辅助函数 (固定形状门槛)
# ---------------------------------------------------------------------------
def measure_shaped_area(intensity: npt.NDArray[np.floating], threshold_ratio: float = 0.5) -> int:
    """统计强度不低于 ``threshold_ratio × 峰值`` 的像素数。

    这是项目标准的 "整形面积" 定义, 方斑 / diff runner 也用它:
    亮区是那些高于整幅画面*峰值*强度一半的像素集合 (阈值是相对的,
    绝不用绝对值, 从而让该指标对曝光不敏感)。

    Args:
        intensity: 二维强度图。
        threshold_ratio: 定义 "亮" 的峰值强度占比。
            必须在 ``(0, 1]`` 内。

    Returns:
        亮像素数量 (空 / 全零画面时为 ``0``)。

    Raises:
        ValueError: 若 ``threshold_ratio`` 不在 ``(0, 1]`` 内。
    """
    if not 0.0 < float(threshold_ratio) <= 1.0:
        raise ValueError(f"threshold_ratio must be in (0, 1], got {threshold_ratio}")
    arr = np.asarray(intensity, dtype=np.float64)
    peak = float(np.max(arr)) if arr.size else 0.0
    if peak <= 0:
        return 0
    return int(np.count_nonzero(arr >= threshold_ratio * peak))


def check_area_requirement(
    measured: int,
    requested: int,
    tolerance: float = 0.20,
) -> dict[str, Any]:
    """检查实测整形面积是否满足要求的面积。

    当 ``measured >= (1 - tolerance) * requested`` 时视为满足 (我们容忍
    欠量, 但绝不把超尺寸的图形判为失败 —— 这个门槛问的是
    "有没有把目标方框填满")。

    Args:
        measured: 实测亮像素数。
        requested: 要求的目标方框像素数。
        tolerance: 允许的相对欠量 ``(0, 1)``。

    Returns:
        含 ``"met"`` (bool)、``"measured_area"``、``"requested_area"``、
        ``"tolerance"``、``"fill_ratio"`` (``measured / requested``) 以及
        ``"shortfall"`` (``max(0, requested - measured)``) 的字典。

    Raises:
        ValueError: 若任一面积为负, 或 ``tolerance`` 不在 ``(0, 1)`` 内。
    """
    if measured < 0 or requested < 0:
        raise ValueError(f"Areas must be non-negative, got {measured}, {requested}")
    if not 0.0 < float(tolerance) < 1.0:
        raise ValueError(f"tolerance must be in (0, 1), got {tolerance}")
    fill_ratio = (float(measured) / float(requested)) if requested > 0 else 1.0
    required_min = (1.0 - float(tolerance)) * float(requested)
    return {
        "met": bool(float(measured) >= required_min),
        "measured_area": int(measured),
        "requested_area": int(requested),
        "tolerance": float(tolerance),
        "fill_ratio": float(fill_ratio),
        "shortfall": int(max(0, requested - measured)),
    }


# ---------------------------------------------------------------------------
# 目标生成
# ---------------------------------------------------------------------------
def create_benchmark_target(
    shape: str,
    grid_size: tuple[int, int],
    *,
    target_area: int = DEFAULT_TARGET_AREA,
    aspect_ratio: float = 1.0,
) -> tuple[npt.NDArray[np.floating], dict[str, Any]]:
    """为一次基准运行构建归一化的目标形状。

    包装 :func:`~ao_shaping.utils.image.targets.create_target_shape`,
    使基准使用与 runner 完全相同的目标工厂。

    Args:
        shape: ``"square"``、``"circle"`` 或 ``"gaussian"``。
        grid_size: 输出的 ``(高, 宽)`` 网格。
        target_area: 要求的亮像素数。对 ``"square"`` 它通过
            ``side = round(sqrt(area))`` 确定边长; 对 ``"circle"``
            它定出半径使**填充**圆面积逼近该要求; 对 ``"gaussian"``
            则被忽略 (sigma 取默认值)。
        aspect_ratio: ``"square"`` 的宽:高; ``>1`` 表示矩形
            (长边水平)。其余情况忽略。

    Returns:
        ``(target_intensity, info)``, 其中 ``target_intensity`` 是
        ``[0, 1]`` 内的归一化二维强度, ``info`` 含
        ``"requested_area"``、``"shape"``、``"grid_size"`` 和
        ``"side_px"`` (以网格像素计的方斑边长)。

    Raises:
        ValueError: 若 ``shape`` 不受支持。
    """
    if shape not in _SHAPES:
        raise ValueError(f"Unsupported shape {shape!r}; choose from {sorted(_SHAPES)}")

    if shape == "square":
        side = max(1, int(round(float(target_area) ** 0.5)))
        # aspect_ratio > 1 时长边水平: 边长 × 比例。
        long_side = max(side, int(round(side * float(aspect_ratio))))
        target = create_target_shape(
            "rectangle" if aspect_ratio > 1.0 else "square",
            grid_size,
            side=long_side if aspect_ratio > 1.0 else side,
            aspect_ratio=float(aspect_ratio),
        )
        info_side = side
    elif shape == "circle":
        target = create_target_shape("circle", grid_size, radius_ratio=0.3)
        info_side = 0
    else:  # gaussian
        target = create_target_shape("gaussian", grid_size, radius_ratio=0.3)
        info_side = 0

    total = float(target.sum())
    if total > 0:
        target = target.astype(np.float64) / total

    requested = measure_shaped_area(target, threshold_ratio=0.5)
    info = {
        "shape": shape,
        "grid_size": grid_size,
        "requested_area": requested,
        "side_px": info_side,
    }
    return target, info


# ---------------------------------------------------------------------------
# 各算法的执行
# ---------------------------------------------------------------------------
def _run_gerchberg_saxton(
    target_intensity: npt.NDArray[np.floating],
    grid_size: tuple[int, int],
    iterations: int,
    seed: int,
) -> npt.NDArray[np.floating]:
    """GS 相位恢复; 返回相位图 (弧度)。"""
    target_amp = np.sqrt(np.maximum(target_intensity, 0.0))
    result = gerchberg_saxton(
        source_amplitude=np.ones(grid_size, dtype=np.float64),
        target_amplitude=target_amp,
        iterations=int(iterations),
        propagation="fft",
    )
    logger.debug(
        f"GS done: {result.iterations} iters, converged={result.converged}, "
        f"final error={result.error_history[-1]:.4f}"
    )
    return np.asarray(result.phase, dtype=np.float64)


def _run_backprop(
    target_intensity: npt.NDArray[np.floating],
    grid_size: tuple[int, int],
    iterations: int,
    seed: int,
    device: str | None,
) -> npt.NDArray[np.floating]:
    """可微梯度下降整形; 返回相位图。"""
    result = train_beam_shaping(
        target=target_intensity,
        grid_size=grid_size,
        propagation="fft",
        optimizer="adam",
        iterations=int(iterations),
        lr=3e-2,
        w_uniformity=0.4,
        w_efficiency=0.6,
        device=device,
        seed=seed,
    )
    logger.debug(
        f"Backprop done: {getattr(result, 'iterations', '?')} iters, "
        f"converged={getattr(result, 'converged', '?')}"
    )
    return np.asarray(result.phase, dtype=np.float64)


def _run_spgd_sim(
    target_intensity: npt.NDArray[np.floating],
    grid_size: tuple[int, int],
    iterations: int,
    seed: int,
) -> npt.NDArray[np.floating]:
    """跑在同一个 FFT 正向模型上的自包含 SPGD 循环。

    SPGD (Stochastic Parallel Gradient Descent) 用与 GS/backprop 相同的
    :func:`_propagate_far_field` 模型, 直接对目标强度优化 SLM 相位图。
    代价函数结合均匀度 (亮掩码内的 CV) 与围栏能量, 与项目的整形目标一致,
    因此这个对比是同类可比的。
    """
    rng = np.random.default_rng(seed)
    phase = rng.normal(0.0, 0.05, size=grid_size).astype(np.float64)
    mask = target_intensity > 0.5 * float(np.max(target_intensity))

    best_cost = np.inf
    best_phase = phase.copy()
    for it in range(int(iterations)):
        # 随机扰动 (每次迭代统计量相同)。
        delta = rng.normal(0.0, 0.08, size=grid_size)
        plus = _propagate_far_field(phase + delta)
        minus = _propagate_far_field(phase - delta)
        cost_plus, cost_minus = ( _cost(plus, mask), _cost(minus, mask))
        grad = (cost_plus - cost_minus) / (2.0 * 0.08)
        lr = 0.10 / (1.0 + 0.01 * it)
        phase = phase - lr * grad * delta
        c = _cost(_propagate_far_field(phase), mask)
        if c < best_cost:
            best_cost, best_phase = c, phase.copy()

    logger.debug(f"SPGD-sim done: {iterations} iters, best cost={best_cost:.4f}")
    return best_phase


def _cost(intensity: npt.NDArray[np.floating], mask: npt.NDArray[np.bool_]) -> float:
    """由均匀度 CV + 围栏能量缺口构成的整形代价。"""
    metrics = compute_shaping_metrics(intensity, mask)
    cv = float(metrics.get("uniformity_cv", 0.0))
    ee = float(metrics.get("encircled_energy", 0.0))
    return cv + (1.0 - ee)


# ---------------------------------------------------------------------------
# 公开的基准入口
# ---------------------------------------------------------------------------
def run_benchmark(
    algorithm: str,
    shape: str,
    grid_size: tuple[int, int] = DEFAULT_GRID,
    target_area: int = DEFAULT_TARGET_AREA,
    aspect_ratio: float = 1.0,
    iterations: int | None = None,
    seed: int = 42,
    max_frames: int = DEFAULT_MAX_FRAMES,
    device: str | None = None,
) -> dict[str, Any]:
    """在一个仿真目标形状上跑一个整形算法。

    Args:
        algorithm: ``"gs"``、``"backprop"`` 或 ``"spgd-sim"``。
        shape: ``"square"``、``"circle"`` 或 ``"gaussian"``。
        grid_size: ``(高, 宽)`` 网格。
        target_area: 要求的目标方框像素数 (方斑边长由其平方根推出)。
        aspect_ratio: 方斑目标的宽:高 (``>1`` → 矩形)。
        iterations: 优化迭代次数; 默认 100。
        seed: 随机种子 (可复现的运行)。
        max_frames: 所记录演化过程的长度上限 (GIF 帧预算)。
        device: Backprop 的计算设备 (``"cuda"``/``"cpu"``/None=自动)。

    Returns:
        含算法标识、目标 / 要求面积、仿真强度、整形面积测量、面积要求检查、
        整形指标 (``uniformity_cv``、``encircled_energy``、
        ``uniformity_cv``) 以及耗时的字典。
    """
    if algorithm not in _ALGORITHMS:
        raise ValueError(f"Unsupported algorithm {algorithm!r}; choose {sorted(_ALGORITHMS)}")
    if shape not in _SHAPES:
        raise ValueError(f"Unsupported shape {shape!r}; choose {sorted(_SHAPES)}")

    import time

    iterations = int(iterations) if iterations is not None else DEFAULT_ITERATIONS

    # 在唯一的权威入口处把标量 grid_size (``32``) 归一化成二维网格
    # (``(32, 32)``)。所有下游消费方 (``create_benchmark_target``、
    # ``gerchberg_saxton``/``backprop``/``spgd-sim`` 各相位执行器) 都需要
    # 二维数组; 标量会把 ``np.ones(grid_size)`` 压成一维源振幅并抛出
    # ``ValueError: Input amplitudes must be 2D arrays``。
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    target_intensity, info = create_benchmark_target(
        shape, grid_size, target_area=target_area, aspect_ratio=aspect_ratio
    )
    requested = int(info["requested_area"])

    t0 = time.perf_counter()
    if algorithm == "gs":
        phase = _run_gerchberg_saxton(target_intensity, grid_size, iterations, seed)
    elif algorithm == "backprop":
        phase = _run_backprop(target_intensity, grid_size, iterations, seed, device)
    else:
        phase = _run_spgd_sim(target_intensity, grid_size, iterations, seed)
    elapsed = time.perf_counter() - t0

    simulated = _propagate_far_field(phase)
    measured_area = measure_shaped_area(simulated)
    area_check = check_area_requirement(measured_area, requested)

    # 为 (对称的) 指标调用归一化目标。
    target_norm = target_intensity / float(target_intensity.sum()) if target_intensity.sum() > 0 else target_intensity
    mask = target_norm > 0.5 * float(np.max(target_norm))
    metrics = compute_shaping_metrics(simulated, mask)

    result: dict[str, Any] = {
        "algorithm": algorithm,
        "shape": shape,
        "grid_size": grid_size,
        "target_area": target_area,
        "aspect_ratio": float(aspect_ratio),
        "iterations": iterations,
        "seed": seed,
        "requested_area": requested,
        "measured_area": measured_area,
        "area_met": area_check["met"],
        "fill_ratio": area_check["fill_ratio"],
        "uniformity_cv": float(metrics.get("uniformity_cv", 0.0)),
        "encircled_energy": float(metrics.get("encircled_energy", 0.0)),
        "intensity_peak": float(metrics.get("peak", 0.0)),
        "elapsed_s": float(elapsed),
        "simulated": simulated,
        "target": target_norm,
        "phase": phase,
    }

    return result


def run_benchmark_suite(
    algorithms: list[str] | tuple[str, ...] | None = None,
    shapes: list[str] | tuple[str, ...] | None = None,
    *,
    grid_size: tuple[int, int] = DEFAULT_GRID,
    target_area: int = DEFAULT_TARGET_AREA,
    aspect_ratio: float = 1.0,
    iterations: int | None = None,
    seed: int = 42,
    max_frames: int = DEFAULT_MAX_FRAMES,
    device: str | None = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """跑遍所有算法 × 所有选定形状 (穷举网格)。

    Args:
        algorithms: ``{"gs","backprop","spgd-sim"}`` 的子集;
            为 *None* 时取全部三个。
        shapes: ``{"square","circle","gaussian"}`` 的子集;
            为 *None* 时取全部三个。
        grid_size: 网格尺寸。
        target_area: 要求的目标方框面积 (像素)。
        aspect_ratio: 方斑的宽高比。
        iterations: 优化迭代次数。
        seed: 随机种子。
        max_frames: GIF 帧预算 (每格)。
        device: Backprop 设备。

    Returns:
        ``(rows, dataframe)``, 其中每一行是 :func:`run_benchmark` 的标量结果
        (已剥掉 phase/simulated 数组), ``dataframe`` 是每格一行的表格视图。
    """
    algos = list(algorithms or sorted(_ALGORITHMS))
    shps = list(shapes or sorted(_SHAPES))

    rows: list[dict[str, Any]] = []
    for alg in algos:
        for shp in shps:
            logger.info(f"Benchmark: algorithm={alg} shape={shp}")
            result = run_benchmark(
                algorithm=alg,
                shape=shp,
                grid_size=grid_size,
                target_area=target_area,
                aspect_ratio=aspect_ratio,
                iterations=iterations,
                seed=seed,
                max_frames=max_frames,
                device=device,
            )
            rows.append(result)

    df = to_dataframe(rows)

    return rows, df


# ---------------------------------------------------------------------------
# 序列化辅助 (DF / CSV / MD / GIF)
# ---------------------------------------------------------------------------
HPRINT_KEYS: tuple[str, ...] = (
    "algorithm",
    "shape",
    "requested_area",
    "measured_area",
    "area_met",
    "fill_ratio",
    "uniformity_cv",
    "encircled_energy",
    "elapsed_s",
)


def to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """把各行投影到标量 (非数组) 字段, 得到一个 DataFrame。"""
    scalar_rows = []
    for row in rows:
        scalar_rows.append({k: row[k] for k in HPRINT_KEYS if k in row})
    df = pd.DataFrame(scalar_rows)
    if not df.empty:
        df = df.sort_values(["algorithm", "shape"]).reset_index(drop=True)
    return df



def build_gif_frames(
    target: npt.NDArray[np.floating],
    simulated: npt.NDArray[np.floating],
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> list[Image.Image]:
    """渲染一小段 PIL 帧序列: target → simulated 堆叠。

    使用感知上经过缩放的灰度调色板, 使强度动态在低位深 GIF 里也可见。
    至少返回一帧。

    Args:
        target: 归一化的目标强度 ``(H, W)``。
        simulated: 归一化的仿真强度 ``(H, W)``。
        max_frames: 实用意义上的最大帧数 (GIF 超过 40 帧没有收益)。

    Returns:
        PIL ``Image`` 列表 (模式 ``"P"``, 8 位调色板)。
    """
    n_frames = max(1, min(max_frames, 24))
    stack = np.stack([target, simulated], axis=-1)  # (H, W, 2)
    stack = stack / max(float(stack.max()), 1e-12)
    frames: list[Image.Image] = []
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        frame = stack[..., 0] * (1 - t) + stack[..., 1] * t
        gray = (255 * frame).astype(np.uint8)
        frames.append(Image.fromarray(gray, mode="L").convert("P"))
    return frames


__all__ = [
    "DEFAULT_CELL_SPACING",
    "DEFAULT_DISTANCE",
    "DEFAULT_GRID",
    "DEFAULT_ITERATIONS",
    "DEFAULT_MAX_FRAMES",
    "DEFAULT_TARGET_AREA",
    "DEFAULT_WAVELENGTH",
    "SUITE_ALGORITHMS",
    "SUITE_SHAPES",
    "check_area_requirement",
    "create_benchmark_target",
    "measure_shaped_area",
    "run_benchmark",
    "run_benchmark_suite",
    "build_gif_frames",
    "to_dataframe",
    "HPRINT_KEYS",
]
