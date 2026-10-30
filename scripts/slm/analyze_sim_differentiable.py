"""离线分析: 可微分优化在仿真里是否真的有效。

本脚本是 ``report/simulation/sim_credibility/`` 的第二部分产出器 —— 第一部分
(``scripts/ml/analyze_sim2f_direction_corpus.py``) 回答"仿真与实测的绝对画面对不对得上、
变化方向有没有信息量"; 这一部分回答"既然仿真不完全可信, 那么在仿真里用梯度下降做
整形, 结论还剩多少"。

四组测量, 全部离线, 只用 CPU:

1. **梯度正确性**: 把 ``total_loss ∘ _build_forward`` 的 autograd 梯度与中心差分逐像素
   比对 (``fft`` 与 ``asm`` 两个传播模型)。附带一阶下降检查: 沿负梯度方向走一步,
   损失必须下降。这是"可微分优化可信"的唯一硬前提。
2. **损失函数是否真的在优化**: 逐形状跑 canonical 基准
   ``beam_shaping_benchmark.run_benchmark``, 并单独复算损失的两个分量, 确认它们对
   相位的响应是否与 canonical 指标一致。**逐形状报, 不跨形状平均**。
3. **物理预测模型选择**: 同一形状在 ``fft`` 与 ``asm`` 下各自优化, 再做交叉评估。
4. **参数敏感性**: 固定 ``asm`` 最优相位, 只扰动传播参数 (距离 / 波长 / 像素间距),
   看整形质量掉多少 —— 直接对应仿真标定误差, 即可微分优化的可信边界。

指标一律来自 canonical 实现 (``compute_shaping_metrics`` / ``total_loss`` /
``_build_forward``), 本脚本不复制任何传播或指标代码。

用法::

    python scripts/slm/analyze_sim_differentiable.py
    python scripts/slm/analyze_sim_differentiable.py --quick   # 小网格冒烟
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
# 先 ROOT 再 ROOT/src: 绝不能把 scripts/ 放进来, 否则 scripts/ml 会遮蔽 src/ml。
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

ARTIFACT_DIR = ROOT / "report" / "simulation" / "sim_credibility" / "artifacts"

# 全部测量共用的固定口径 —— 这些是判据, 不是调参旋钮。
GRID: tuple[int, int] = (128, 128)
ITERATIONS: int = 300
SEED: int = 42
# 中心差分步长: 步长太大会被三阶项截断, 太小会被 float64 舍入吃掉, 1e-5 是二者折中。
FD_EPS: float = 1e-5
N_GRAD_SAMPLES: int = 8
# 逐像素相对误差的通过阈值。中心差分本身的分辨极限在 1e-5 量级, 所以不能要求 1e-9;
# 但梯度反号会给出 2.0, 差 3 个数量级, 这个阈值仍有判别力 (见 selftest)。
GRAD_REL_TOL: float = 1e-3
# 一阶下降检查: 沿负梯度走一步, 实际下降幅度至少是线性预测的这个比例。
DESCENT_STEP_RAD: float = 1e-3
DESCENT_MIN_FRACTION: float = 0.2

# 参数扰动档位 (相对名义值)。
PERTURBATIONS: tuple[float, ...] = (-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10)


def _git_rev() -> str:
    """当前提交, 写进产物以便报告可追溯。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover - 只在无 git 环境退化
        return "unknown"


def _torch():
    """惰性导入 torch, 缺失时给出可读的失败信息。"""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - 环境依赖
        raise SystemExit(
            "本脚本需要 torch (纯 CPU 即可)。请在项目虚拟环境里运行。"
        ) from exc
    return torch


def _targets(grid: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """返回 (canonical 基准目标, 0/1 掩膜目标)。

    canonical 基准 ``create_benchmark_target`` 返回的目标是**按总和归一化**的
    (整幅求和 = 1), 而损失函数 ``uniformity_loss`` 用 ``target > 0.5`` 取目标区 ——
    两者的约定不一致, 见 :func:`loss_degeneracy`。这里同时给出两种约定。
    """
    from ao_shaping.algorithm.signal_processing.beam_shaping_benchmark import (
        DEFAULT_TARGET_AREA,
        create_benchmark_target,
    )

    target, _info = create_benchmark_target(
        "square", grid, target_area=DEFAULT_TARGET_AREA
    )
    mask01 = (target > 0.5 * float(np.max(target))).astype(np.float64)
    return target, mask01


# ---------------------------------------------------------------------------
# 1. 梯度正确性
# ---------------------------------------------------------------------------
def _loss_of(
    torch,
    forward,
    phase_np: np.ndarray,
    target_t,
    w_uniformity: float,
    w_efficiency: float,
) -> float:
    """无梯度地求一次损失值 (供中心差分使用)。"""
    from ao_shaping.algorithm.signal_processing.differentiable_shaping import total_loss

    with torch.no_grad():
        phase = torch.as_tensor(phase_np, dtype=torch.float64)
        intensity = forward(phase)
        loss = total_loss(
            intensity,
            phase,
            target_t,
            w_uniformity=w_uniformity,
            w_efficiency=w_efficiency,
        )
    return float(loss.detach())


def gradient_check(
    propagation: str,
    grid: tuple[int, int] = GRID,
    n_samples: int = N_GRAD_SAMPLES,
    seed: int = SEED,
) -> dict[str, Any]:
    """把 autograd 梯度与中心差分逐像素比对。

    用 0/1 掩膜目标 (而非按总和归一化的目标), 否则损失对相位不敏感, 差分本身无意义。

    Args:
        propagation: ``"fft"`` 或 ``"asm"``。
        grid: 网格尺寸。
        n_samples: 抽样比对的像素数。
        seed: 相位与抽样点的随机种子。

    Returns:
        含逐点相对误差、余弦相似度、一阶下降检查结果的字典。
    """
    from ao_shaping.algorithm.signal_processing.beam_shaping_benchmark import (
        DEFAULT_WAVELENGTH,
    )
    from ao_shaping.algorithm.signal_processing.differentiable_shaping import (
        _build_forward,
        total_loss,
    )
    from ao_shaping.utils.slm.bench_constants import SLM_PITCH_M

    torch = _torch()
    w_uniformity, w_efficiency = 0.4, 0.6
    _sum1, mask01 = _targets(grid)
    target_t = torch.as_tensor(mask01, dtype=torch.float64)

    forward = _build_forward(propagation, SLM_PITCH_M, 0.1, DEFAULT_WAVELENGTH)

    rng = np.random.default_rng(seed)
    phase0 = rng.normal(0.0, 0.3, size=grid)

    phase = torch.tensor(phase0, dtype=torch.float64, requires_grad=True)
    loss = total_loss(
        forward(phase),
        phase,
        target_t,
        w_uniformity=w_uniformity,
        w_efficiency=w_efficiency,
    )
    loss.backward()
    grad = phase.grad.detach().numpy().copy()
    base_loss = float(loss.detach())

    flat_idx = rng.choice(grid[0] * grid[1], size=n_samples, replace=False)
    rel_errors: list[float] = []
    fd_vals: list[float] = []
    ag_vals: list[float] = []
    per_pixel: list[dict[str, Any]] = []
    scale = float(np.median(np.abs(grad))) or 1.0

    for k in flat_idx:
        i, j = divmod(int(k), grid[1])
        plus = phase0.copy()
        plus[i, j] += FD_EPS
        minus = phase0.copy()
        minus[i, j] -= FD_EPS
        fd = (
            _loss_of(torch, forward, plus, target_t, w_uniformity, w_efficiency)
            - _loss_of(torch, forward, minus, target_t, w_uniformity, w_efficiency)
        ) / (2.0 * FD_EPS)
        ag = float(grad[i, j])
        rel = abs(ag - fd) / max(abs(fd), scale * 1e-3)
        rel_errors.append(rel)
        fd_vals.append(fd)
        ag_vals.append(ag)
        per_pixel.append(
            {"row": i, "col": j, "autograd": ag, "finite_diff": fd, "rel_err": rel}
        )

    fd_arr = np.asarray(fd_vals)
    ag_arr = np.asarray(ag_vals)
    cosine = float(
        np.dot(fd_arr, ag_arr)
        / (np.linalg.norm(fd_arr) * np.linalg.norm(ag_arr) + 1e-300)
    )

    # 一阶下降: 沿负梯度方向走一小步, 损失必须下降。
    # 梯度绝对值很小 (损失对强度做了归一化, 单像素贡献约为 1/N), 所以步长按
    # 最大分量归一化到固定的相位幅度, 否则位移会落进 float64 的舍入噪声里。
    grad_scale = float(np.max(np.abs(grad))) or 1.0
    direction = grad / grad_scale
    descent_phase = phase0 - DESCENT_STEP_RAD * direction
    descent_loss = _loss_of(
        torch, forward, descent_phase, target_t, w_uniformity, w_efficiency
    )
    predicted_drop = DESCENT_STEP_RAD * float(np.sum(grad * direction))
    actual_drop = base_loss - descent_loss
    descent_fraction = actual_drop / predicted_drop if predicted_drop > 0 else float("nan")

    return {
        "propagation": propagation,
        "grid": list(grid),
        "n_samples": n_samples,
        "fd_eps": FD_EPS,
        "loss": base_loss,
        "grad_abs_median": scale,
        "rel_err_median": float(np.median(rel_errors)),
        "rel_err_max": float(np.max(rel_errors)),
        "rel_err_tol": GRAD_REL_TOL,
        "cosine": cosine,
        "gradient_ok": bool(
            np.max(rel_errors) < GRAD_REL_TOL and cosine > 0.9999
        ),
        "descent_predicted_drop": predicted_drop,
        "descent_actual_drop": actual_drop,
        "descent_fraction": descent_fraction,
        "descent_ok": bool(
            actual_drop > 0 and descent_fraction > DESCENT_MIN_FRACTION
        ),
        "per_pixel": per_pixel,
    }


def selftest(propagation: str = "fft", grid: tuple[int, int] = GRID) -> dict[str, Any]:
    """自检: 证明上面的比对能区分对与错。

    正确梯度必须通过; 把梯度整体反号后, 相对误差应接近 2.0 而必须不通过。
    只有两条都成立, 这个检查才不是恒真的。
    """
    good = gradient_check(propagation, grid, n_samples=4)

    flipped = [
        abs(-p["autograd"] - p["finite_diff"])
        / max(abs(p["finite_diff"]), good["grad_abs_median"] * 1e-3)
        for p in good["per_pixel"]
    ]
    flipped_max = float(np.max(flipped))

    return {
        "correct_gradient_ok": good["gradient_ok"],
        "flipped_rel_err_max": flipped_max,
        "flipped_rejected": bool(flipped_max > 1.5),
        "discriminating": bool(good["gradient_ok"] and flipped_max > 1.5),
    }


# ---------------------------------------------------------------------------
# 2. 损失函数对相位的响应
# ---------------------------------------------------------------------------
def loss_degeneracy(grid: tuple[int, int] = GRID) -> dict[str, Any]:
    """确认损失的两个分量是否对相位真的有响应。

    这一项是整份报告的关键取证: 若 ``uniformity_loss`` 在 canonical 目标下恒为 0,
    则"均匀度"根本没进入优化目标; 若 ``efficiency_loss`` 读到的围栏能量与 canonical
    指标 ``compute_shaping_metrics`` 相差一个常数因子, 则效率项也是错的。
    两者同时发生, 优化器就会停在未整形的画面上, 而基准仍把它当"可微优化结果"报出去。
    """
    from ao_shaping.algorithm.signal_processing.beam_shaping_benchmark import (
        DEFAULT_WAVELENGTH,
        _propagate_far_field,
    )
    from ao_shaping.algorithm.signal_processing.differentiable_shaping import (
        _build_forward,
        efficiency_loss,
        total_loss,
        uniformity_loss,
    )
    from ao_shaping.utils.image.beam_metrics import compute_shaping_metrics
    from ao_shaping.utils.slm.bench_constants import SLM_PITCH_M

    torch = _torch()
    sum1, mask01 = _targets(grid)
    forward = _build_forward("fft", SLM_PITCH_M, 0.1, DEFAULT_WAVELENGTH)

    rng = np.random.default_rng(SEED)
    cases = {"flat": np.zeros(grid), "random": rng.normal(0.0, 0.1, size=grid)}

    mask_bool = mask01 > 0
    rows: list[dict[str, Any]] = []
    for convention, target in (("sum1", sum1), ("mask01", mask01)):
        target_t = torch.as_tensor(target, dtype=torch.float64)
        for name, phase_np in cases.items():
            with torch.no_grad():
                intensity = forward(torch.as_tensor(phase_np, dtype=torch.float64))
                uni = float(uniformity_loss(intensity, target_t).detach())
                eff = float(efficiency_loss(intensity, target_t).detach())
                loss = float(
                    total_loss(
                        intensity,
                        torch.as_tensor(phase_np, dtype=torch.float64),
                        target_t,
                        w_uniformity=0.4,
                        w_efficiency=0.6,
                    ).detach()
                )
            arr = intensity.detach().numpy()
            metrics = compute_shaping_metrics(
                _propagate_far_field(phase_np), mask_bool
            )
            rows.append(
                {
                    "target_convention": convention,
                    "phase": name,
                    "uniformity_loss": uni,
                    "efficiency_loss": eff,
                    "loss": loss,
                    "metric_uniformity_cv": float(metrics["uniformity_cv"]),
                    "metric_encircled_energy": float(metrics["encircled_energy"]),
                    "efficiency_loss_implied_ee": 1.0 - eff,
                    "target_mask_px": int(
                        int((target > 0.5 * float(np.max(target))).sum())
                    ),
                    "target_max": float(np.max(target)),
                }
            )
            del arr

    return {
        "rows": rows,
        "uniformity_term_dead_on_sum1": bool(
            all(
                r["uniformity_loss"] == 0.0
                for r in rows
                if r["target_convention"] == "sum1"
            )
        ),
        "efficiency_term_mismatch_on_sum1": bool(
            all(
                abs(r["efficiency_loss_implied_ee"] - r["metric_encircled_energy"]) > 0.1
                for r in rows
                if r["target_convention"] == "sum1"
            )
        ),
    }


# ---------------------------------------------------------------------------
# 3. 收敛 / 效果
# ---------------------------------------------------------------------------
def benchmark_as_shipped(
    grid: tuple[int, int] = GRID,
    iterations: int = ITERATIONS,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """canonical 基准今天实际报出的数字 (三种算法 × 四种形状)。

    这里**不改** ``run_benchmark`` 的任何口径 —— 它把按总和归一化的目标直接交给
    ``train_beam_shaping``, 正是 :func:`loss_degeneracy` 取证的那个缺陷现场。
    """
    from ao_shaping.algorithm.signal_processing.beam_shaping_benchmark import (
        DEFAULT_TARGET_AREA,
        run_benchmark,
    )

    cases = [
        ("square", 1.0, "正方形"),
        ("square", 2.0, "长方形 (2:1)"),
        ("circle", 1.0, "圆形"),
        ("gaussian", 1.0, "高斯"),
    ]
    rows: list[dict[str, Any]] = []
    for shape, aspect, label in cases:
        for algorithm in ("gs", "backprop", "spgd-sim"):
            t0 = time.perf_counter()
            res = run_benchmark(
                algorithm,
                shape,
                grid_size=grid,
                target_area=DEFAULT_TARGET_AREA,
                aspect_ratio=aspect,
                iterations=iterations,
                seed=seed,
                device="cpu",
            )
            wall = time.perf_counter() - t0
            rows.append(
                {
                    "shape": shape,
                    "shape_label": label,
                    "aspect_ratio": aspect,
                    "algorithm": algorithm,
                    "iterations": iterations,
                    "uniformity_cv": float(res["uniformity_cv"]),
                    "encircled_energy": float(res["encircled_energy"]),
                    "intensity_peak": float(res["intensity_peak"]),
                    "measured_area_px": int(res["measured_area"]),
                    "requested_area_px": int(res["requested_area"]),
                    "fill_ratio": float(res["fill_ratio"]),
                    "area_met": bool(res["area_met"]),
                    "elapsed_s": float(res["elapsed_s"]),
                    "wall_s": wall,
                }
            )
            print(
                f"  {label:<14s} {algorithm:<9s} "
                f"CV={rows[-1]['uniformity_cv']:.4f} "
                f"EE={rows[-1]['encircled_energy']:.4f} "
                f"{rows[-1]['elapsed_s']:.2f}s",
                flush=True,
            )
    return rows


def benchmark_mask_fixed(
    grid: tuple[int, int] = GRID,
    iterations: int = ITERATIONS,
    seed: int = SEED,
) -> dict[str, Any]:
    """把目标换成 0/1 掩膜后重跑, 看可微分优化到底能到什么水平。

    只走 canonical 入口: ``train_beam_shaping`` (可微) 与 ``gerchberg_saxton`` (GS)。
    基准里的 SPGD 循环绑定在它自己的代价函数上, 不参与这一节 —— 它的数值已在上表,
    而它的代价函数用的是**相对**阈值掩膜, 不受这个归一化缺陷影响。
    """
    from ao_shaping.algorithm.signal_processing.beam_shaping_benchmark import (
        DEFAULT_WAVELENGTH,
    )
    from ao_shaping.algorithm.signal_processing.differentiable_shaping import (
        train_beam_shaping,
    )
    from ao_shaping.algorithm.signal_processing.gerchberg_saxton import (
        gerchberg_saxton,
    )
    from ao_shaping.utils.slm.bench_constants import SLM_PITCH_M

    nominal = {
        "cell_spacing": SLM_PITCH_M,
        "distance": 0.1,
        "wavelength": DEFAULT_WAVELENGTH,
    }

    cases = [
        ("square", 1.0, "正方形"),
        ("square", 2.0, "长方形 (2:1)"),
        ("circle", 1.0, "圆形"),
        ("gaussian", 1.0, "高斯"),
    ]
    from ao_shaping.algorithm.signal_processing.beam_shaping_benchmark import (
        create_benchmark_target,
        DEFAULT_TARGET_AREA,
    )

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for shape, aspect, label in cases:
        target, _info = create_benchmark_target(
            shape, grid, target_area=DEFAULT_TARGET_AREA, aspect_ratio=aspect
        )
        mask01 = (target > 0.5 * float(np.max(target))).astype(np.float64)

        # 未优化基线: 全零相位。有它才能判定"优化后"到底是真整形还是停在起点,
        # 而不是靠与另一个算法互相接近来推断 (见 as_shipped 那节两者都停在起点)。
        m_flat = _evaluate_phase(np.zeros(grid), mask01, "fft", **nominal)
        rows.append(
            {
                "shape": shape,
                "shape_label": label,
                "aspect_ratio": aspect,
                "algorithm": "flat-baseline",
                "iterations": 0,
                "uniformity_cv": m_flat["uniformity_cv"],
                "encircled_energy": m_flat["encircled_energy"],
                "loss_first": float("nan"),
                "loss_last": float("nan"),
                "elapsed_s": 0.0,
            }
        )
        print(
            f"  {label:<14s} flat-baseline    "
            f"CV={m_flat['uniformity_cv']:.4f} EE={m_flat['encircled_energy']:.4f}",
            flush=True,
        )

        # 可微: 两种初相位 (全零 / 小随机), 以说明初相位是否是鞍点。
        for init_name, init_phase in (
            ("flat", None),
            ("random", rng.normal(0.0, 0.1, size=grid)),
        ):
            t0 = time.perf_counter()
            result = train_beam_shaping(
                mask01,
                grid,
                propagation="fft",
                optimizer="adam",
                iterations=iterations,
                device="cpu",
                initial_phase=init_phase,
                seed=seed,
                cell_spacing=nominal["cell_spacing"],
                distance=nominal["distance"],
                wavelength=nominal["wavelength"],
            )
            wall = time.perf_counter() - t0
            m = _evaluate_phase(
                np.asarray(result.phase, dtype=np.float64),
                mask01,
                "fft",
                **nominal,
            )
            rows.append(
                {
                    "shape": shape,
                    "shape_label": label,
                    "aspect_ratio": aspect,
                    "algorithm": f"backprop-{init_name}",
                    "iterations": iterations,
                    "uniformity_cv": m["uniformity_cv"],
                    "encircled_energy": m["encircled_energy"],
                    "loss_first": float(result.loss_history[0]),
                    "loss_last": float(result.loss_history[-1]),
                    "elapsed_s": wall,
                }
            )
            print(
                f"  {label:<14s} backprop-{init_name:<6s} "
                f"CV={m['uniformity_cv']:.4f} EE={m['encircled_energy']:.4f} "
                f"{wall:.2f}s",
                flush=True,
            )

        # GS: 目标振幅取掩膜的平方根。
        t0 = time.perf_counter()
        gs = gerchberg_saxton(
            source_amplitude=np.ones(grid, dtype=np.float64),
            target_amplitude=np.sqrt(mask01),
            iterations=iterations,
            propagation="fft",
        )
        wall = time.perf_counter() - t0
        m = _evaluate_phase(
            np.asarray(gs.phase, dtype=np.float64), mask01, "fft", **nominal
        )
        rows.append(
            {
                "shape": shape,
                "shape_label": label,
                "aspect_ratio": aspect,
                "algorithm": "gs",
                "iterations": iterations,
                "uniformity_cv": m["uniformity_cv"],
                "encircled_energy": m["encircled_energy"],
                "loss_first": float("nan"),
                "loss_last": float("nan"),
                "elapsed_s": wall,
            }
        )
        print(
            f"  {label:<14s} gs               "
            f"CV={m['uniformity_cv']:.4f} EE={m['encircled_energy']:.4f} {wall:.2f}s",
            flush=True,
        )

    return {"nominal": nominal, "rows": rows}


# ---------------------------------------------------------------------------
# 4. 物理预测模型选择 / 5. 参数敏感性
# ---------------------------------------------------------------------------
def _metrics_of(intensity: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """canonical 整形指标 (掩膜取目标非零区)。"""
    from ao_shaping.utils.image.beam_metrics import compute_shaping_metrics

    mask = target > 0
    m = compute_shaping_metrics(intensity, mask)
    return {
        "uniformity_cv": float(m.get("uniformity_cv", float("nan"))),
        "encircled_energy": float(m.get("encircled_energy", float("nan"))),
    }


def _evaluate_phase(
    phase: np.ndarray,
    target: np.ndarray,
    propagation: str,
    *,
    cell_spacing: float,
    distance: float,
    wavelength: float,
) -> dict[str, float]:
    """用 canonical 物理预测模型把固定相位重新评估一次。"""
    from ao_shaping.algorithm.signal_processing.differentiable_shaping import (
        _build_forward,
    )

    torch = _torch()
    forward = _build_forward(propagation, cell_spacing, distance, wavelength)
    with torch.no_grad():
        intensity = forward(torch.as_tensor(phase, dtype=torch.float64))
    arr = intensity.detach().numpy()
    total = float(arr.sum())
    if total > 0:
        arr = arr / total
    return _metrics_of(arr, target)


def model_and_sensitivity(
    grid: tuple[int, int] = GRID,
    iterations: int = ITERATIONS,
    seed: int = SEED,
) -> dict[str, Any]:
    """同一形状在 fft / asm 两个模型下各自优化, 再做交叉评估与参数扰动。

    目标用 0/1 掩膜 (与 :func:`benchmark_mask_fixed` 同口径), 否则两个模型都会
    停在未整形的画面上, 交叉评估没有意义。
    """
    from ao_shaping.algorithm.signal_processing.beam_shaping_benchmark import (
        DEFAULT_WAVELENGTH,
    )
    from ao_shaping.algorithm.signal_processing.differentiable_shaping import (
        train_beam_shaping,
    )
    from ao_shaping.utils.slm.bench_constants import SLM_PITCH_M

    _sum1, mask01 = _targets(grid)
    nominal = {
        "cell_spacing": SLM_PITCH_M,
        "distance": 0.1,
        "wavelength": DEFAULT_WAVELENGTH,
    }

    phases: dict[str, np.ndarray] = {}
    for propagation in ("fft", "asm"):
        t0 = time.perf_counter()
        result = train_beam_shaping(
            mask01,
            grid,
            propagation=propagation,
            optimizer="adam",
            iterations=iterations,
            device="cpu",
            seed=seed,
            cell_spacing=nominal["cell_spacing"],
            distance=nominal["distance"],
            wavelength=nominal["wavelength"],
        )
        wall = time.perf_counter() - t0
        phases[propagation] = np.asarray(result.phase, dtype=np.float64)
        print(
            f"  优化 ({propagation}): loss {result.loss_history[0]:.4f} -> "
            f"{result.loss_history[-1]:.4f}, {wall:.2f}s",
            flush=True,
        )

    cross: list[dict[str, Any]] = []
    for opt_model, phase in phases.items():
        for eval_model in ("fft", "asm"):
            m = _evaluate_phase(phase, mask01, eval_model, **nominal)
            cross.append(
                {
                    "optimized_on": opt_model,
                    "evaluated_on": eval_model,
                    "same_model": opt_model == eval_model,
                    **m,
                }
            )

    # 参数敏感性: 固定 asm 最优相位, 只动传播参数。
    asm_phase = phases["asm"]
    sensitivity: list[dict[str, Any]] = []
    for param in ("distance", "wavelength", "cell_spacing"):
        for rel in PERTURBATIONS:
            kwargs = dict(nominal)
            kwargs[param] = nominal[param] * (1.0 + rel)
            m = _evaluate_phase(asm_phase, mask01, "asm", **kwargs)
            sensitivity.append(
                {
                    "parameter": param,
                    "rel_change": rel,
                    "value": kwargs[param],
                    "uniformity_cv": m["uniformity_cv"],
                    "encircled_energy": m["encircled_energy"],
                }
            )

    return {
        "nominal": nominal,
        "cross_model": cross,
        "sensitivity": sensitivity,
        "sensitivity_phase_source": "asm",
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="小网格 / 少迭代冒烟, 不写正式产物"
    )
    parser.add_argument("--grid", type=int, default=None, help="方形网格边长")
    parser.add_argument("--iterations", type=int, default=None)
    args = parser.parse_args(argv)

    grid = (args.grid, args.grid) if args.grid else GRID
    iterations = args.iterations or ITERATIONS
    if args.quick:
        grid, iterations = (32, 32), 20

    torch = _torch()
    t_start = time.perf_counter()

    print("[1/5] 梯度正确性 (autograd vs 中心差分)", flush=True)
    checks = [gradient_check("fft", grid), gradient_check("asm", grid)]
    for c in checks:
        print(
            f"  {c['propagation']:<4s} 中位相对误差 {c['rel_err_median']:.3e} "
            f"最大 {c['rel_err_max']:.3e} 余弦 {c['cosine']:.6f} -> "
            f"{'通过' if c['gradient_ok'] else '不通过'}; "
            f"下降比例 {c['descent_fraction']:.3f}",
            flush=True,
        )
    st = selftest("fft", grid)
    print(
        f"  自检: 正确梯度通过={st['correct_gradient_ok']} "
        f"反号被拒={st['flipped_rejected']} "
        f"(反号相对误差 {st['flipped_rel_err_max']:.3f})",
        flush=True,
    )

    print("[2/5] 损失分量对相位的响应", flush=True)
    deg = loss_degeneracy(grid)
    for r in deg["rows"]:
        print(
            f"  {r['target_convention']:<7s} {r['phase']:<7s} "
            f"uni={r['uniformity_loss']:.4f} eff={r['efficiency_loss']:.4f} "
            f"(效率项隐含 EE={r['efficiency_loss_implied_ee']:.4f}, "
            f"指标 EE={r['metric_encircled_energy']:.4f})",
            flush=True,
        )

    print("[3/5] canonical 基准今天报出的数字 (逐形状)", flush=True)
    shipped = benchmark_as_shipped(grid, iterations)

    print("[4/5] 目标换成 0/1 掩膜后重跑 (逐形状)", flush=True)
    fixed = benchmark_mask_fixed(grid, iterations)

    print("[5/5] 物理预测模型选择 + 参数敏感性", flush=True)
    ms = model_and_sensitivity(grid, iterations)

    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_rev": _git_rev(),
            "grid": list(grid),
            "iterations": iterations,
            "seed": SEED,
            "fd_eps": FD_EPS,
            "grad_rel_tol": GRAD_REL_TOL,
            "n_grad_samples": N_GRAD_SAMPLES,
            "torch": torch.__version__,
            "device": "cpu",
            "quick": bool(args.quick),
            "wall_s": None,
        },
        "gradient_check": checks,
        "selftest": st,
        "loss_degeneracy": deg,
        "benchmark_as_shipped": shipped,
        "benchmark_mask_fixed": fixed,
        "model_sensitivity": ms,
    }
    payload["meta"]["wall_s"] = round(time.perf_counter() - t_start, 2)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    stem = "differentiable_quick" if args.quick else "differentiable_20261030"
    out = ARTIFACT_DIR / f"{stem}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"写出 {out.relative_to(ROOT)}  (总耗时 {payload['meta']['wall_s']}s)",
        flush=True,
    )

    # 硬门禁: 梯度错 = 后面全部结论作废, 直接非零退出。
    if not all(c["gradient_ok"] and c["descent_ok"] for c in checks):
        print("梯度检查未通过 —— 报告不应引用收敛结论。", file=sys.stderr)
        return 2
    if not st["discriminating"]:
        print("自检未能区分对错梯度 —— 检查本身无效。", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
