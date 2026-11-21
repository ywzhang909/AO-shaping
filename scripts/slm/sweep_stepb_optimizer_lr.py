"""Step B 优化器 / 学习率 / 调度 扫描（纯离线仿真, 不碰任何硬件设备）。

要回答什么
----------
MIL 闭环的 Step B（可微整形迭代, 见
``ao_shaping.optimizer.wfless.model_in_loop_shaping`` 与
``ao_shaping.algorithm.signal_processing.differentiable_shaping``）每步都花
真实墙钟时间。在**同等物理光斑质量**下, 能否用下面三招砍掉迭代预算：

1. 提高 Adam 学习率（``lr`` 阶梯）；
2. 动态学习率调度（余弦 / 阶梯 / 预热+余弦, 经
   ``train_beam_shaping(lr_schedule=...)``）；
3. 换 L-BFGS。

以及离线赢了的方案, 经过 SLM 面板显示链路（块状映射 + 灰度量化）之后
还能不能活下来 —— 这是接硬件验证前的必经桥。

三层设计
--------
* **Level 1**：完整 MIL 仿真（``simulate_iterative_shaping``, 6 轮, Step A
  500 步）, 只扫 ``StepBConfig.lr`` ∈ {0.02, 0.05, 0.1, 0.2}（若 0.2 不发散
  再加 0.4）。口径与
  ``report/slm/differentiable_opt/artifacts/stepb_budget_sweep_20261116/sweep_stepb_budget.py``
  完全一致, 数字可直接互比。
* **Level 2**：单次 ``train_beam_shaping`` 调用（同一固定问题: 64×64 网格、
  同一组像差、同一种子、方斑边长 6、权重 0.4/0.6/0/0）, 扫
  优化器 × 学习率 × 调度 × 预算。
* **Level 3**：把每个 Level 2 终态相位送进真实显示链路的**离线镜像**
  （``phase_model_to_panel`` 的最近邻块映射 + ``phase_to_slm_grayscale``
  灰度量化, 2π 灰度取 1023 与 993 两种）, 去量化回模型网格后用同一前向
  重新传播, 报告 CV/EE/FF 的存活情况。

物理指标一律用远场方斑指标（CV / EE / FF）, 不是模型 loss —— loss 降了
不等于光斑变好。

用法::

    .venv/Scripts/python.exe scripts/slm/sweep_stepb_optimizer_lr.py
    .venv/Scripts/python.exe scripts/slm/sweep_stepb_optimizer_lr.py --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from loguru import logger

from ao_shaping.algorithm.signal_processing.differentiable_shaping import (
    _build_forward,
    create_target_mask,
    train_beam_shaping,
)
from ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer import (
    ZernikeCoefficientOptimizer,
)
from ao_shaping.optimizer.wfless.model_in_loop_shaping import (
    ModelInLoopConfig,
    SimulationEnvParams,
    StepAConfig,
    StepBConfig,
    coefficients_from_noll,
    simulate_iterative_shaping,
    square_metrics_at_zero_order,
)
from ao_shaping.utils.slm.bench_constants import (
    LASER_WAVELENGTH_M,
    SLM_PITCH_M,
)
from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale

# `scripts._common` 在仓库根下, 所以裸跑 `python scripts/slm/<name>.py` 时
# 必须让仓库根（而不只是本包目录）可导入。
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._common.plot_style import setup_cjk_font

DEFAULT_OUT = (
    ROOT
    / "report"
    / "slm"
    / "differentiable_opt"
    / "artifacts"
    / "stepb_optimizer_lr_20261116"
)
FIGURES_DIR = ROOT / "report" / "slm" / "differentiable_opt" / "figures"

#: 与 stepb_budget_sweep 完全一致的外围条件（可互比的口径）。
N_ROUNDS = 6
STEP_A_ITERS = 500
ABERRATIONS: dict[int, float] = {4: 0.35, 5: 0.30, 8: 0.20}
SEED = 0
REGION = 64
TARGET_SIDE = 6
W_UNIFORMITY = 0.4
W_EFFICIENCY = 0.6

#: Level 1 学习率阶梯。
L1_LRS: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2)
#: 0.2 不发散时追加的探针。
L1_LR_PROBE = 0.4

#: Level 2 个变体: (名称, 优化器, lr, 调度, 迭代数)。
L2_ARMS: tuple[tuple[str, str, float, str | None, int], ...] = (
    ("adam_0.05", "adam", 0.05, None, 60),
    ("adam_0.05", "adam", 0.05, None, 100),
    ("adam_0.05", "adam", 0.05, None, 150),
    ("adam_0.05", "adam", 0.05, None, 300),
    ("adam_0.05", "adam", 0.05, None, 600),
    ("adam_0.1", "adam", 0.1, None, 60),
    ("adam_0.1", "adam", 0.1, None, 150),
    ("adam_0.2", "adam", 0.2, None, 60),
    ("adam_0.2", "adam", 0.2, None, 150),
    ("adam_0.1_cosine", "adam", 0.1, "cosine", 150),
    ("adam_0.2_warmup_cosine", "adam", 0.2, "warmup-cosine", 150),
    ("lbfgs_60", "lbfgs", 1.0, None, 60),
    ("lbfgs_30", "lbfgs", 1.0, None, 30),
)

#: Level 3 面板参数 —— 取自
#: ``data/debug/differentiable_opt/phys_20261116_01/run_config.json``
#: （model_pupil_samples=256, model_padding=4, pupil_center=[960, 600],
#: pupil_radius=480 ⇒ dp = 960/256 = 3.75, m_grid = 1024）。
PANEL_N_PUPIL = 256
PANEL_M_GRID = 1024
PANEL_DP = 960.0 / 256.0
PANEL_PUPIL_CENTER: tuple[float, float] = (960.0, 600.0)
#: phase_model_to_panel 的默认面板形状 (SLM_PANEL_H, SLM_PANEL_W)。
PANEL_SHAPE: tuple[int, int] = (1200, 1920)
#: 2π 灰度标度: 配置默认 1023; SLM #1 在 1064 nm 的实测 2π 响应为 993。
GRAYSCALES: tuple[int, ...] = (1023, 993)
#: ``ModelSpec.centre_index`` —— 模型网格的直流格下标 (``m_grid // 2``),
#: 也是 fftshift 后远场零级所在格 / 远场目标区的中心锚点。
PANEL_CENTRE_INDEX: int = PANEL_M_GRID // 2
#: ``ModelSpec.pupil_start`` —— 瞳面块在模型网格里的起始下标 (行=列)。
PANEL_PUPIL_START: int = PANEL_CENTRE_INDEX - PANEL_N_PUPIL // 2
#: ``ModelSpec.block_centre`` —— 瞳面块的几何中心下标 (半整数),
#: 模型网格与面板之间的对位锚点。
PANEL_BLOCK_CENTRE: float = PANEL_PUPIL_START + (PANEL_N_PUPIL - 1) / 2.0
#: Level-2 区域网格 → 模型网格瞳面块的整数复制倍数 (256/64)。
REGION_TO_PUPIL_FACTOR = PANEL_N_PUPIL // REGION


# ---------------------------------------------------------------------------
# Level 1: MIL 仿真学习率阶梯
# ---------------------------------------------------------------------------


def _run_level1_arm(lr: float) -> dict[str, object]:
    """跑一条 Level 1 变体（完整 MIL 仿真）, 返回汇总记录。"""
    config = ModelInLoopConfig(
        env=SimulationEnvParams(
            region=REGION, noise_enabled=False, n_frames=1, seed=SEED
        ),
        step_a=StepAConfig(iterations=STEP_A_ITERS),
        step_b=StepBConfig(iterations=150, lr=lr),
        aberrations=dict(ABERRATIONS),
        n_rounds=N_ROUNDS,
        seed=SEED,
    )
    started = time.perf_counter()
    result = simulate_iterative_shaping(config)
    wall_s = time.perf_counter() - started

    cv_trace: list[float] = []
    ee_trace: list[float] = []
    ff_trace: list[float] = []
    for round_metrics in result.rounds:
        after = round_metrics.metrics_after
        cv_trace.append(float(after["uniformity_cv"]))
        ee_trace.append(float(after["encircled_energy"]))
        ff_trace.append(float(after["flatness_factor"]))

    finite = all(math.isfinite(v) for v in cv_trace)
    diverged = (not finite) or max(cv_trace) > 2.0
    oscillated = finite and cv_trace[-1] > 1.2 * min(cv_trace)
    status = "diverged" if diverged else ("oscillated" if oscillated else "ok")
    best_idx = int(np.argmin(cv_trace)) if finite else -1
    record: dict[str, object] = {
        "arm": f"l1_lr{lr}",
        "level": 1,
        "optimizer": "adam(MIL)",
        "lr": lr,
        "lr_schedule": None,
        "iterations": 150,
        "wall_s": round(wall_s, 3),
        "status": status,
        "cv_trace": [round(v, 6) for v in cv_trace],
        "ee_trace": [round(v, 6) for v in ee_trace],
        "ff_trace": [round(v, 6) for v in ff_trace],
        "cv_final": round(cv_trace[-1], 6),
        "ee_final": round(ee_trace[-1], 6),
        "ff_final": round(ff_trace[-1], 6),
    }
    if best_idx >= 0:
        record["cv_best"] = round(cv_trace[best_idx], 6)
        record["ee_best"] = round(ee_trace[best_idx], 6)
        record["ff_best"] = round(ff_trace[best_idx], 6)
        record["best_round"] = best_idx
    return record


# ---------------------------------------------------------------------------
# Level 2: 单次 train_beam_shaping（固定问题）
# ---------------------------------------------------------------------------


def _level2_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造 Level 2 的固定问题: (目标掩码, 源振幅, 初始相位)。

    同一组像差（Zernike Noll 4/5/8, 与 Level 1 相同）被写进初始相位,
    源振幅取数字孪生的原生高斯（``ZernikeCoefficientOptimizer.native_amplitude``）,
    与 MIL 口径一致。
    """
    optimizer = ZernikeCoefficientOptimizer(
        n_orders=8, region=REGION, initial_coefficients=np.zeros(45), seed=SEED
    )
    basis = optimizer.generate_basis()
    coeffs = coefficients_from_noll(45, ABERRATIONS)
    aberr_phase = np.tensordot(coeffs, basis, axes=(0, 0))
    amplitude = ZernikeCoefficientOptimizer.native_amplitude(REGION)
    target = create_target_mask("square", (REGION, REGION), TARGET_SIDE)
    rng = np.random.default_rng(SEED)
    initial_phase = aberr_phase + 0.1 * rng.standard_normal((REGION, REGION))
    return target, amplitude, initial_phase


def _measure(phase: np.ndarray, amplitude: np.ndarray) -> dict[str, float]:
    """用与 Level 2 相同的 fft 前向, 测方斑指标。"""
    torch = _require_torch()
    forward = _build_forward("fft", SLM_PITCH_M, 0.1, LASER_WAVELENGTH_M)
    intensity = forward(
        torch.from_numpy(np.ascontiguousarray(phase, dtype=np.float64)),
        torch.from_numpy(np.ascontiguousarray(amplitude, dtype=np.float64)),
    )
    metrics, _ = square_metrics_at_zero_order(
        intensity.detach().cpu().numpy(), TARGET_SIDE
    )
    return {
        "cv": float(metrics["uniformity_cv"]),
        "ee": float(metrics["encircled_energy"]),
        "ff": float(metrics["flatness_factor"]),
    }


def _require_torch():
    from ao_shaping.algorithm.signal_processing.differentiable_shaping import (
        _torch,
    )

    return _torch()


def _laplacian_rms(phase: np.ndarray) -> float:
    """相位高频度量: 离散拉普拉斯算子的 RMS（弧度）。

    定义: ``rms(Δ²phase)``, 其中 ``Δ²`` 为二维中心二阶差分
    （``p[i+1,j]+p[i-1,j]+p[i,j+1]+p[i,j-1]-4p[i,j]``, 边缘用单侧差分）。
    面板块状映射的截止空间频率很低, 该值越大说明相位越依赖面板表达不了的
    高频成分。
    """
    p = np.asarray(phase, dtype=np.float64)
    lap = (
        np.roll(p, -1, axis=0) + np.roll(p, 1, axis=0)
        + np.roll(p, -1, axis=1) + np.roll(p, 1, axis=1)
        - 4.0 * p
    )
    # 去掉环绕伪影: 把首尾行/列的 roll 贡献替换为单侧差分。
    lap[0, :] = p[1, :] - p[0, :]
    lap[-1, :] = p[-1, :] - p[-2, :]
    lap[:, 0] = p[:, 1] - p[:, 0]
    lap[:, -1] = p[:, -1] - p[:, -2]
    return float(np.sqrt(np.mean(lap**2)))


def _wrap_fraction(snapshots: list[np.ndarray]) -> float:
    """相位包裹事件率: 逐步 (包裹后) 相位轨迹里发生 2π 跳变的步占比。

    对每个像素的时间序列做 ``np.unwrap``, 某一步发生包裹 ⇔ 该像素在该步
    的 unwrap 偏移量发生变化; 一步里只要有任一像素发生包裹即记为一次
    「包裹步」。返回 ``包裹步数 / 总步数``。
    """
    if len(snapshots) < 2:
        return 0.0
    stack = np.stack(snapshots, axis=0)  # (T, H, W)
    unwrapped = np.unwrap(stack.reshape(stack.shape[0], -1), axis=0)
    offset = unwrapped - stack.reshape(stack.shape[0], -1)
    step_wrapped = np.any(np.diff(offset, axis=0) != 0.0, axis=1)
    return float(np.count_nonzero(step_wrapped) / (stack.shape[0] - 1))


def _run_level2_arm(
    name: str,
    optimizer: str,
    lr: float,
    lr_schedule: str | None,
    iterations: int,
    problem: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, object]:
    """跑一条 Level 2 个变体, 返回汇总记录（含 Level 3 的原始侧指标）。"""
    target, amplitude, initial_phase = problem
    snapshots: list[np.ndarray] = []

    def _snap(phase_tensor) -> None:
        snapshots.append(
            phase_tensor.detach().cpu().numpy().astype(np.float64, copy=False)
        )

    started = time.perf_counter()
    result = train_beam_shaping(
        target,
        (REGION, REGION),
        source_amplitude=amplitude,
        initial_phase=initial_phase,
        propagation="fft",
        optimizer=optimizer,
        iterations=iterations,
        lr=lr,
        lr_schedule=lr_schedule,
        w_uniformity=W_UNIFORMITY,
        w_efficiency=W_EFFICIENCY,
        w_zero_order=0.0,
        w_smoothness=0.0,
        seed=SEED,
        device="cpu",
        phase_callback=_snap,
    )
    wall_s = time.perf_counter() - started

    phase = np.asarray(result.phase, dtype=np.float64)
    metrics = _measure(phase, amplitude)
    loss_hist = [float(v) for v in result.loss_history]
    record: dict[str, object] = {
        "arm": name,
        "level": 2,
        "optimizer": optimizer,
        "lr": lr,
        "lr_schedule": lr_schedule,
        "iterations": iterations,
        "wall_s": round(wall_s, 3),
        "loss_first": round(loss_hist[0], 6),
        "loss_final": round(loss_hist[-1], 6),
        "cv_final": round(metrics["cv"], 6),
        "ee_final": round(metrics["ee"], 6),
        "ff_final": round(metrics["ff"], 6),
        "phase_rms_rad": round(float(np.sqrt(np.mean(phase**2))), 6),
        "wrap_fraction": round(_wrap_fraction(snapshots), 6),
        "laplacian_rms_rad": round(_laplacian_rms(phase), 6),
        "loss_history": [round(v, 6) for v in loss_hist],
        # 终态相位: 留给 main 里的 Level 3 面板量化评估, 之后被 pop 掉,
        # 不进 JSON/CSV（大数组）。
        "phase": phase,  # 不放 JSON, 仅本地用
    }
    return record


# ---------------------------------------------------------------------------
# Level 3: 面板显示链路离线镜像
# ---------------------------------------------------------------------------


def _model_to_panel(phase_model: np.ndarray, panel_shape: tuple[int, int]) -> np.ndarray:
    """``phase_model_to_panel`` 的离线镜像（最近邻块映射, 逐行同逻辑）。

    模型采样间距 ``dp=3.75`` 不是整数面板像元, 硬件看到的相位是**块状**的;
    对位锚点是瞳面块的几何中心 ``block_centre = 384 + (256-1)/2 = 511.5``。
    """
    height, width = int(panel_shape[0]), int(panel_shape[1])
    out = np.zeros((height, width), dtype=np.float64)
    cx, cy = PANEL_PUPIL_CENTER
    half = (PANEL_N_PUPIL - 1) / 2.0
    x0 = int(round(cx - half * PANEL_DP))
    x1 = int(round(cx + half * PANEL_DP)) + 1
    y0 = int(round(cy - half * PANEL_DP))
    y1 = int(round(cy + half * PANEL_DP)) + 1
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x1, width), min(y1, height)
    if x1c <= x0c or y1c <= y0c:
        return out
    xs = np.arange(x0c, x1c, dtype=np.float64)
    ys = np.arange(y0c, y1c, dtype=np.float64)
    mj = np.rint(PANEL_BLOCK_CENTRE + (xs - cx) / PANEL_DP).astype(np.int64)
    mi = np.rint(PANEL_BLOCK_CENTRE + (ys - cy) / PANEL_DP).astype(np.int64)
    mj = np.clip(mj, 0, PANEL_M_GRID - 1)
    mi = np.clip(mi, 0, PANEL_M_GRID - 1)
    out[y0c:y1c, x0c:x1c] = np.asarray(phase_model)[np.ix_(mi, mj)]
    return out


def _panel_to_model(panel_phase: np.ndarray, panel_shape: tuple[int, int]) -> np.ndarray:
    """``_model_to_panel`` 的逆: 面板相位 → 模型网格瞳面块。

    正向映射把模型格 ``i``（瞳面块内, ``PANEL_PUPIL_START ≤ i <
    PANEL_PUPIL_START + PANEL_N_PUPIL``）写到面板像元
    ``round(cx + (i - PANEL_BLOCK_CENTRE) · dp)``; 这里用同一组下标取回, 于是
    逆映射是正向块映射的精确采样逆（每个模型格取回它自己那块里的像元）。
    """
    out = np.zeros((PANEL_M_GRID, PANEL_M_GRID), dtype=np.float64)
    cx, cy = PANEL_PUPIL_CENTER
    idx = np.arange(
        PANEL_PUPIL_START, PANEL_PUPIL_START + PANEL_N_PUPIL, dtype=np.float64
    )
    xs = np.rint(cx + (idx - PANEL_BLOCK_CENTRE) * PANEL_DP).astype(np.int64)
    ys = np.rint(cy + (idx - PANEL_BLOCK_CENTRE) * PANEL_DP).astype(np.int64)
    xs = np.clip(xs, 0, int(panel_shape[1]) - 1)
    ys = np.clip(ys, 0, int(panel_shape[0]) - 1)
    sl = slice(PANEL_PUPIL_START, PANEL_PUPIL_START + PANEL_N_PUPIL)
    out[sl, sl] = panel_phase[np.ix_(ys, xs)]
    return out


def _region_from_model(model_phase: np.ndarray) -> np.ndarray:
    """模型网格瞳面块 → Level-2 区域网格（4×4 块的块心采样）。"""
    pupil_start = PANEL_PUPIL_START
    block = model_phase[
        pupil_start : pupil_start + PANEL_N_PUPIL,
        pupil_start : pupil_start + PANEL_N_PUPIL,
    ]
    return block[2::REGION_TO_PUPIL_FACTOR, 2::REGION_TO_PUPIL_FACTOR]


def _region_to_model(region_phase: np.ndarray) -> np.ndarray:
    """Level-2 区域网格 → 模型网格瞳面块（4×4 整数复制, 瞳外为零）。"""
    model = np.zeros((PANEL_M_GRID, PANEL_M_GRID), dtype=np.float64)
    pupil_start = PANEL_PUPIL_START
    model[pupil_start : pupil_start + PANEL_N_PUPIL,
          pupil_start : pupil_start + PANEL_N_PUPIL] = np.kron(
        region_phase, np.ones((REGION_TO_PUPIL_FACTOR, REGION_TO_PUPIL_FACTOR))
    )
    return model


def _degrade_and_measure(
    region_phase: np.ndarray,
    amplitude: np.ndarray,
    grayscale: int | None,
) -> dict[str, float]:
    """把区域相位送进显示链路再测指标。

    ``grayscale=None`` 时只做块映射（不量化, 验证块映射无损）；否则经
    ``phase_to_slm_grayscale`` 以给定 2π 灰度标度量化, 去量化回弧度后重传播。
    """
    model_phase = _region_to_model(region_phase)
    panel_phase = _model_to_panel(model_phase, PANEL_SHAPE)
    if grayscale is None:
        back = _panel_to_model(panel_phase, PANEL_SHAPE)
    else:
        gray = phase_to_slm_grayscale(panel_phase, max_grayscale=int(grayscale))
        dequant = gray.astype(np.float64) / float(grayscale) * (2.0 * math.pi)
        back = _panel_to_model(dequant, PANEL_SHAPE)
    return _measure(_region_from_model(back), amplitude)


# ---------------------------------------------------------------------------
# 产物
# ---------------------------------------------------------------------------


def _write_csv(path: Path, records: list[dict[str, object]]) -> None:
    """所有变体一行一条写 CSV（Level 1/2 共用一套列, 未涉及的列为空）。"""
    fields = [
        "level", "arm", "optimizer", "lr", "lr_schedule", "iterations",
        "wall_s", "status", "cv_final", "ee_final", "ff_final",
        "cv_best", "ee_best", "ff_best", "loss_first", "loss_final",
        "phase_rms_rad", "wrap_fraction", "laplacian_rms_rad",
        "cv_block", "ee_block", "ff_block",
        "cv_gray1023", "ee_gray1023", "ff_gray1023",
        "cv_gray993", "ee_gray993", "ff_gray993",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row: dict[str, object] = {}
            for field in fields:
                value = record.get(field)
                row[field] = "" if value is None else value
            writer.writerow(row)


def _fmt_cell(value: object, nd: int = 4) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.{nd}f}"
    return str(value)


def _write_md(
    path: Path,
    level1: list[dict[str, object]],
    level2: list[dict[str, object]],
    meta: dict[str, object],
) -> None:
    """写可直接摘录的 md: 三层表格 + 数据驱动的要点。"""
    lines: list[str] = []
    lines.append("# Step B 优化器 / 学习率 / 调度扫描（MIL Step B, 纯离线仿真）")
    lines.append("")
    lines.append(
        "口径: Level 1 与 `stepb_budget_sweep_20261116` 完全一致"
        "（region 64、无噪声、1 帧、Step A 500 步、6 轮、像差 {4:0.35, 5:0.30, 8:0.20}、"
        "seed 0）。Level 2 为单次 `train_beam_shaping` 调用（fft 前向, 64×64, "
        "像差写入初始相位, 原生高斯振幅, 方斑边长 6, 权重 0.4/0.6/0/0, seed 0, CPU）。"
    )
    lines.append("")
    lines.append("## Level 1 — MIL 仿真学习率阶梯（150 步 / 轮）")
    lines.append("")
    lines.append("| 变体 | lr | 状态 | 墙钟 s | 末轮 CV | 末轮 EE | 末轮 FF | 最佳 CV |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for record in level1:
        lines.append(
            "| {arm} | {lr} | {status} | {wall} | {cv} | {ee} | {ff} | {best} |".format(
                arm=record["arm"],
                lr=record["lr"],
                status=record["status"],
                wall=_fmt_cell(record["wall_s"], 1),
                cv=_fmt_cell(record.get("cv_final"), 4),
                ee=_fmt_cell(record.get("ee_final"), 4),
                ff=_fmt_cell(record.get("ff_final"), 4),
                best=_fmt_cell(record.get("cv_best"), 4),
            )
        )
    lines.append("")
    lines.append("各变体逐轮 CV 轨迹（JSON `level1[].cv_trace`）用于判断收敛/震荡/发散。")
    lines.append("")
    lines.append("## Level 2 — 预算 × 优化器 × 调度（单次 train_beam_shaping）")
    lines.append("")
    lines.append(
        "| 变体 | 优化器 | lr | 调度 | 迭代 | 墙钟 s | 终态 CV | 终态 EE | 终态 FF | "
        "相位 RMS (rad) | 包裹步占比 | 拉普拉斯 RMS (rad) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for record in level2:
        lines.append(
            "| {arm} | {opt} | {lr} | {sched} | {it} | {wall} | {cv} | {ee} | {ff} | "
            "{rms} | {wrap} | {lap} |".format(
                arm=record["arm"],
                opt=record["optimizer"],
                lr=record["lr"],
                sched=record["lr_schedule"] or "—",
                it=record["iterations"],
                wall=_fmt_cell(record["wall_s"], 2),
                cv=_fmt_cell(record.get("cv_final"), 4),
                ee=_fmt_cell(record.get("ee_final"), 4),
                ff=_fmt_cell(record.get("ff_final"), 4),
                rms=_fmt_cell(record.get("phase_rms_rad"), 3),
                wrap=_fmt_cell(record.get("wrap_fraction"), 3),
                lap=_fmt_cell(record.get("laplacian_rms_rad"), 4),
            )
        )
    lines.append("")
    lines.append("## Level 3 — 面板显示链路存活（量化前后）")
    lines.append("")
    lines.append(
        "显示链路镜像: 区域相位 → 4×4 块复制到 1024² 模型网格瞳面块 → "
        "`phase_model_to_panel` 最近邻块映射（dp=3.75, 块心 511.5, 面板 1200×1920, "
        "参数取自 `data/debug/differentiable_opt/phys_20261116_01/run_config.json`）→ "
        "灰度量化（2π 灰度 1023 / 993）→ 去量化回弧度 → 逆映射回区域网格 → 同一 fft "
        "前向重传播。块映射本身无损（`block` 列应与原始列一致）；真正的退化在灰度量化。"
    )
    lines.append("")
    lines.append(
        "| 变体 | CV 原始 | CV 块映射 | CV 灰度1023 | CV 灰度993 | "
        "EE 原始 | EE 灰度1023 | EE 灰度993 | FF 原始 | FF 灰度1023 | FF 灰度993 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for record in level2:
        lines.append(
            "| {arm} | {cv0} | {cvb} | {cvg} | {cvg2} | {ee0} | {eeg} | {eeg2} | "
            "{ff0} | {ffg} | {ffg2} |".format(
                arm=record["arm"],
                cv0=_fmt_cell(record.get("cv_final"), 4),
                cvb=_fmt_cell(record.get("cv_block"), 4),
                cvg=_fmt_cell(record.get("cv_gray1023"), 4),
                cvg2=_fmt_cell(record.get("cv_gray993"), 4),
                ee0=_fmt_cell(record.get("ee_final"), 4),
                eeg=_fmt_cell(record.get("ee_gray1023"), 4),
                eeg2=_fmt_cell(record.get("ee_gray993"), 4),
                ff0=_fmt_cell(record.get("ff_final"), 4),
                ffg=_fmt_cell(record.get("ff_gray1023"), 4),
                ffg2=_fmt_cell(record.get("ff_gray993"), 4),
            )
        )
    lines.append("")
    lines.append("## 要点（数据驱动）")
    lines.append("")
    base = next(
        (r for r in level2 if r["arm"] == "adam_0.05" and r["iterations"] == 150),
        None,
    )
    if base is not None:
        lines.append(
            f"- 基线 adam/0.05@150: 墙钟 {_fmt_cell(base['wall_s'], 2)} s, "
            f"CV {_fmt_cell(base.get('cv_final'), 4)}, EE {_fmt_cell(base.get('ee_final'), 4)}。"
        )
    winners = sorted(
        level2,
        key=lambda r: float(r.get("cv_gray1023") or math.inf),
    )
    if winners:
        best = winners[0]
        lines.append(
            f"- 灰度 1023 存活后 CV 最低的变体: {best['arm']} "
            f"(CV {_fmt_cell(best.get('cv_gray1023'), 4)})。"
        )
    l1_ok = [r for r in level1 if r["status"] == "ok"]
    if l1_ok:
        fastest = min(l1_ok, key=lambda r: float(r["wall_s"]))
        lines.append(
            f"- Level 1 中不发散且墙钟最短: {fastest['arm']} "
            f"({_fmt_cell(fastest['wall_s'], 1)} s)。"
        )
    lines.append("")
    lines.append(
        "详细逐迭代 loss / CV 轨迹与全部数字见 `stepb_optimizer_lr.json`;"
        " 图表见 `stepblr_quality_vs_budget.png` 与 `stepblr_quantization_survival.png`。"
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_figures(
    out_dir: Path,
    level1: list[dict[str, object]],
    level2: list[dict[str, object]],
) -> list[Path]:
    """两张图: 质量-预算曲线 + 量化存活。返回写出的路径。"""
    setup_cjk_font()
    written: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    for record in level2:
        iters = int(record["iterations"])
        cv = float(record["cv_final"])
        ee = float(record["ee_final"])
        label = f"{record['arm']} (lr={record['lr']}" + (
            f", {record['lr_schedule']}" if record["lr_schedule"] else ""
        ) + ")"
        ax.plot(iters, cv, "o", label=label)
        ax2 = ax.twinx()
        ax2.plot(iters, ee, "x", color="gray", alpha=0.5, markersize=5)
    ax.set_xlabel("迭代数")
    ax.set_ylabel("终态 CV（越低越好）", color="C0")
    ax2.set_ylabel("终态 EE", color="gray")
    ax.set_title("Level 2: 终态质量 vs 预算")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_yscale("log")
    ax2.set_yscale("log")
    ax = axes[1]
    for record in level1:
        cv_trace = record["cv_trace"]
        ax.plot(range(len(cv_trace)), cv_trace, "-o", markersize=3,
                label=f"lr={record['lr']} ({record['status']})")
    ax.set_xlabel("轮次")
    ax.set_ylabel("逐轮 CV")
    ax.set_title("Level 1: MIL 仿真逐轮 CV 轨迹")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p1 = out_dir / "stepblr_quality_vs_budget.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    written.append(p1)

    names = [str(r["arm"]) for r in level2]
    cv_raw = [float(r["cv_final"]) for r in level2]
    cv_block = [float(r.get("cv_block") or math.nan) for r in level2]
    cv_g1023 = [float(r.get("cv_gray1023") or math.nan) for r in level2]
    cv_g993 = [float(r.get("cv_gray993") or math.nan) for r in level2]
    x = np.arange(len(names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(x - 1.5 * width, cv_raw, width, label="原始")
    ax.bar(x - 0.5 * width, cv_block, width, label="块映射")
    ax.bar(x + 0.5 * width, cv_g1023, width, label="灰度 1023")
    ax.bar(x + 1.5 * width, cv_g993, width, label="灰度 993")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("CV（越低越好）")
    ax.set_title("Level 3: 面板显示链路前后的 CV 存活")
    ax.legend()
    fig.tight_layout()
    p2 = out_dir / "stepblr_quantization_survival.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    written.append(p2)
    return written


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="产物目录（默认 report/slm/differentiable_opt/artifacts/stepb_optimizer_lr_20261116）",
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Step B 优化器/学习率/调度扫描开始, 产物目录 {}", out_dir)
    budget_cuts: list[str] = []
    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    # -- Level 1 -----------------------------------------------------
    level1: list[dict[str, object]] = []
    lrs = list(L1_LRS)
    for lr in lrs:
        logger.info("Level 1: lr={}", lr)
        record = _run_level1_arm(lr)
        level1.append(record)
        timings[f"l1_{lr}"] = float(record["wall_s"])
        logger.info(
            "  -> {} | CV {} -> {} (逐轮见 JSON)",
            record["status"],
            record["cv_trace"][0],
            record["cv_final"],
        )
    l1_02 = next(r for r in level1 if float(r["lr"]) == 0.2)
    if l1_02["status"] != "diverged":
        record = _run_level1_arm(L1_LR_PROBE)
        level1.append(record)
        timings[f"l1_{L1_LR_PROBE}"] = float(record["wall_s"])
        logger.info("Level 1: 0.2 未发散, 追加探针 lr={}", L1_LR_PROBE)
    else:
        logger.info("Level 1: lr=0.2 发散, 跳过 lr={} 探针", L1_LR_PROBE)
        budget_cuts.append(f"L1 lr={L1_LR_PROBE} 未跑（0.2 已发散）")

    # -- Level 2 + Level 3 --------------------------------------------
    problem = _level2_problem()
    target, amplitude, _ = problem
    level2: list[dict[str, object]] = []
    for name, optimizer, lr, schedule, iterations in L2_ARMS:
        logger.info("Level 2: {} ({}, lr={}, {}, iters={})",
                    name, optimizer, lr, schedule or "固定", iterations)
        record = _run_level2_arm(name, optimizer, lr, schedule, iterations, problem)
        level2.append(record)
        timings[name] = float(record["wall_s"])
        logger.info(
            "  -> CV {:.4f} EE {:.4f} 墙钟 {:.2f}s",
            record["cv_final"], record["ee_final"], record["wall_s"],
        )

    # Level 3: 把每条变体的终态相位送进显示链路的离线镜像再测一遍。
    # 相位由 _run_level2_arm 以私有键 "phase" 带出, 用完即 pop（不进 JSON/CSV）。
    for record in level2:
        phase = record.pop("phase", None)
        if phase is None:
            continue
        for tag, grayscale in (("block", None), ("gray1023", 1023), ("gray993", 993)):
            metrics = _degrade_and_measure(phase, amplitude, grayscale)
            record[f"cv_{tag}"] = round(metrics["cv"], 6)
            record[f"ee_{tag}"] = round(metrics["ee"], 6)
            record[f"ff_{tag}"] = round(metrics["ff"], 6)
        record.pop("phase_snapshots", None)

    total_wall = time.perf_counter() - total_start

    # -- 产物 --------------------------------------------------------
    payload = {
        "meta": {
            "question": "MIL Step B 能否用更大 lr / 动态调度 / L-BFGS 在同等物理质量下省迭代, 且面板量化后存活",
            "harness": "stepb_budget_sweep_20261116 同口径 (region 64, Step A 500, 6 轮, 像差 {4:0.35,5:0.30,8:0.20}, seed 0)",
            "level2_problem": "train_beam_shaping 单次调用, fft 前向, 64x64, 像差入初始相位, 原生高斯振幅, 方斑边长 6, 权重 0.4/0.6/0/0, seed 0, CPU",
            "panel_params": {
                "n_pupil": PANEL_N_PUPIL,
                "m_grid": PANEL_M_GRID,
                "dp": PANEL_DP,
                "pupil_center": list(PANEL_PUPIL_CENTER),
                "panel_shape_hw": list(PANEL_SHAPE),
                "source": "data/debug/differentiable_opt/phys_20261116_01/run_config.json",
            },
            "grayscales": list(GRAYSCALES),
        },
        "budget": {
            "target_wall_s": 25 * 60.0,
            "total_wall_s": round(total_wall, 2),
            "cuts_applied": budget_cuts,
            "timings_s": timings,
        },
        "level1": {"arms": level1},
        "level2": {"arms": level2},
    }
    (out_dir / "stepb_optimizer_lr.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(out_dir / "stepb_optimizer_lr.csv", level1 + level2)
    _write_md(out_dir / "stepb_optimizer_lr.md", level1, level2, payload["meta"])
    figures = _make_figures(out_dir, level1, level2)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for figure in figures:
        # 复制（不是移动）: 图要同时留在产物目录和 figures/ 两处。
        target_path = FIGURES_DIR / figure.name
        shutil.copy2(figure, target_path)
        logger.info("图已复制到 {}（原件留在 {}）", target_path, figure)

    logger.info(
        "完成: 总墙钟 {:.1f}s, Level 1 {} 变体, Level 2 {} 变体",
        total_wall, len(level1), len(level2),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
