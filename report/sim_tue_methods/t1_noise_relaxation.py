"""T1 系统松弛 A/B：Step B 逐步高斯噪声注入 (issue #149 / R-83)。

在 model-in-the-loop 仿真 (SimFourierGSNetEnv, 64x64, CPU, 无硬件) 中复现
TU/e 可微仿真的"系统松弛"：Step B 冻结一个与真值不一致的 Zernike 像差
(错模型侧) 或一致的 (对模型侧) 向量，逐像素相位朝该模型优化；每 N 步向
演化中的 SLM 相位注入零均值高斯瞳面噪声 (sigma rad)，对比注入 vs 不注入在
"真实像差下实测最终形状" 上的 CV/EE/复合分数。反向验证：sigma 过大必劣化。

全部 sim-only, 不动硬件写入路径。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ao_shaping.algorithm.signal_processing.differentiable_shaping import (
    create_target_mask,
)
from ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer import (
    ZernikeCoefficientOptimizer,
)
from ao_shaping.optimizer.wfless.model_in_loop_shaping import (
    StepBConfig,
    shape_phase_with_frozen_aberration,
)
from ao_shaping.utils.image.beam_metrics import (
    compute_square_metrics,
    zero_order_center,
)

REGION = 64
N_ORDERS = 10
TARGET_SIDE = 12
ITERS = 200
LR = 0.05
W_UNIF = 0.4
W_EFF = 0.6
# 真值像差：Noll 4..15 (12 个模式, 跳过 piston/tilt), 0.5 rad RMS
TRUE_NOLL = list(range(4, 16))
TRUE_RMS = 0.5


def _build_optimizer() -> ZernikeCoefficientOptimizer:
    return ZernikeCoefficientOptimizer(
        n_orders=N_ORDERS,
        region=REGION,
        initial_coefficients=None,
        lr=LR,
        max_iterations=10,
        device="cpu",
        seed=0,
        dtype="float64",
    )


def _true_aberration(rng: np.random.Generator) -> np.ndarray:
    c = np.zeros(_build_optimizer().n_coefficients)
    vals = rng.normal(0.0, 1.0, len(TRUE_NOLL))
    vals = vals * (TRUE_RMS / np.sqrt(np.mean(vals**2)))
    for n, v in zip(TRUE_NOLL, vals):
        c[n - 1] = v  # Noll 1-indexed -> 0-indexed array
    return c


def _mismatched_model(rng: np.random.Generator, c_true: np.ndarray) -> np.ndarray:
    """一个与真值故意不一致、但同量级 (0.5 rad RMS) 的像差估计。

    保留部分真值 (前 6 个模式 ×0.5) + 独立的残差 (后 6 个模式 + 扰动),
    模拟 Step A "估了但估错" 的情形 (对 #124 的"故意不一致"的物理解读)。
    """
    c = np.zeros_like(c_true)
    half = len(TRUE_NOLL) // 2
    for i, n in enumerate(TRUE_NOLL):
        if i < half:
            c[n - 1] = 0.5 * c_true[n - 1]
        else:
            c[n - 1] = rng.normal(0.0, 0.3)
    # 归一到 0.5 rad RMS 量级 (保留"量级一致、模式不一致")
    vec = c[np.array([n - 1 for n in TRUE_NOLL])]
    rms = float(np.sqrt(np.mean(vec**2)))
    if rms > 0:
        c *= (TRUE_RMS / rms)
    return c


def _twin_forward(opt, c: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """在真值像差 c 下测量 phase 的远场 (数字孪生, 无噪声)。"""
    return opt.forward_intensity(c, phase)


def _square_metrics(frame: np.ndarray) -> dict[str, float]:
    m = compute_square_metrics(frame, TARGET_SIDE, zero_order_center(frame, refine=False))
    return {
        "cv": float(m["uniformity_cv"]),
        "ee": float(m["encircled_energy"]),
        "composite": 0.5 * float(m["encircled_energy"]) + 0.5 / (1.0 + float(m["uniformity_cv"])),
    }


def _phase_highfreq_rms(phase: np.ndarray) -> float:
    """相位的高频分量 RMS (低通 5px 后的残差), 衡量松弛带来的"破碎"代价。"""
    from scipy.ndimage import uniform_filter

    smooth = uniform_filter(phase, size=5)
    return float(np.sqrt(np.mean((phase - smooth) ** 2)))


def run_arm(
    c_true: np.ndarray,
    c_model: np.ndarray,
    sigma: float,
    inject_every: int,
    seed: int,
) -> dict[str, float]:
    opt = _build_optimizer()
    target = create_target_mask("square", (REGION, REGION), TARGET_SIDE)
    cfg = StepBConfig(
        iterations=ITERS,
        lr=LR,
        target_shape="square",
        target_side=TARGET_SIDE,
        w_uniformity=W_UNIF,
        w_efficiency=W_EFF,
        noise_inject_every=inject_every,
        noise_sigma_rad=sigma,
    )
    t0 = time.time()
    phase, _ = shape_phase_with_frozen_aberration(
        opt,
        c_model,
        target,
        np.zeros((REGION, REGION), dtype=np.float64),
        cfg,
        dtype="float64",
        device="cpu",
        seed=seed,
    )
    dt = time.time() - t0
    frame = _twin_forward(opt, c_true, phase)  # 在真值像差下实测
    m = _square_metrics(frame)
    m["highfreq_rms"] = _phase_highfreq_rms(phase)
    m["wall_s"] = dt
    return m


def main() -> None:
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    seeds = [0, 1, 2, 3, 4]
    sigmas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    inject_every = 5

    blocks = {}
    for arm in ("mismatch", "match"):
        rows = {}
        for sigma in sigmas:
            acc = []
            for s in seeds:
                rng_true = np.random.default_rng(1000 + s)
                c_true = _true_aberration(rng_true)
                if arm == "match":
                    c_model = c_true.copy()
                else:
                    c_model = _mismatched_model(np.random.default_rng(2000 + s), c_true)
                acc.append(run_arm(c_true, c_model, sigma, inject_every if sigma > 0 else 0, s))
            cv = [r["cv"] for r in acc]
            ee = [r["ee"] for r in acc]
            comp = [r["composite"] for r in acc]
            hf = [r["highfreq_rms"] for r in acc]
            rows[f"sigma={sigma}"] = {
                "cv_mean": float(np.mean(cv)),
                "cv_std": float(np.std(cv)),
                "ee_mean": float(np.mean(ee)),
                "ee_std": float(np.std(ee)),
                "composite_mean": float(np.mean(comp)),
                "composite_std": float(np.std(comp)),
                "highfreq_rms_mean": float(np.mean(hf)),
                "highfreq_rms_std": float(np.std(hf)),
                "n_seeds": len(seeds),
            }
        blocks[arm] = rows

    payload = {
        "region": REGION,
        "n_orders": N_ORDERS,
        "target_side": TARGET_SIDE,
        "iters": ITERS,
        "lr": LR,
        "w_uniformity": W_UNIF,
        "w_efficiency": W_EFF,
        "true_rms_rad": TRUE_RMS,
        "true_noll": TRUE_NOLL,
        "inject_every": inject_every,
        "seeds": seeds,
        "arms": blocks,
    }
    with open(out / "t1_noise_relaxation.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 控制台摘要
    print(f"\nT1 系统松弛 A/B (region={REGION}, iters={ITERS}, inject_every={inject_every}, seeds={seeds})")
    for arm in ("mismatch", "match"):
        print(f"\n[{arm} 模型]")
        print(f"{'sigma':>7} | {'CV mean±std':>14} | {'EE mean±std':>14} | {'composite':>12} | {'HF-rms':>8}")
        for sigma in sigmas:
            r = blocks[arm][f"sigma={sigma}"]
            print(
                f"{sigma:>7.2f} | {r['cv_mean']:>7.4f}±{r['cv_std']:<6.4f} | "
                f"{r['ee_mean']:>7.4f}±{r['ee_std']:<6.4f} | "
                f"{r['composite_mean']:>6.4f}±{r['composite_std']:<5.4f} | "
                f"{r['highfreq_rms_mean']:>8.4f}"
            )
    print(f"\nJSON -> {out / 't1_noise_relaxation.json'}")


if __name__ == "__main__":
    main()
