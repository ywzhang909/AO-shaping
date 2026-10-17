"""T2 混合拟合：Zernike(Noll≤15) + 逐像素 freeform 联合优化 (issue #149 / R-84)。

在 model-in-the-loop 仿真 (64x64, CPU, 无硬件) 中复现 TU/e 的"混合拟合"：
参数空间 = Zernike(15 项, Noll≤15) 低阶结构 + 逐像素 freeform，线性叠加进
可微前向，与目标形状 (三角形 / 五角星) 做损失 (w·CV + (1-w)·(1-EE))，
梯度下降同时更新两组参数。

三臂对照 (同一真值像差 A, 同一目标, 同预算):
  1. zernike_only   : φ_slm = 15 项 Zernike (验证"15 项 Zernike 做不出三角形"的表达力缺口)
  2. perpixel_only  : φ_slm = 逐像素 (现有 Step B 架构, 冻结式)
  3. joint          : φ_slm = 15 项 Zernike + 逐像素 (联合, 同时更新; 记录全程最佳 epoch)

最终在真值像差 A 下实测远场, 报 CV / EE / 复合分数 + 全程最佳 epoch。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ao_shaping.algorithm.signal_processing.differentiable_shaping import (
    create_target_mask,
)
from ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer import (
    ZernikeCoefficientOptimizer,
)
from ao_shaping.optimizer.wfless.model_in_loop_shaping import _aperture_from_basis
from ao_shaping.utils.image.beam_metrics import compute_square_metrics, zero_order_center
from ao_shaping.utils.wavefront.fraunhofer import interpolated_pupil_intensity

REGION = 64
N_J = 4          # Noll ≤ 15 -> 15 项 Zernike (order 0-4 的 15 个)
TARGET_SIDE = 16
ITERS = 300
LR = 0.05
W_UNIF = 0.4
W_EFF = 0.6
TRUE_NOLL = list(range(4, 16))  # 12 模式, 0.5 rad RMS
TRUE_RMS = 0.5


def _zernike_basis(n_orders: int, region: int):
    opt = ZernikeCoefficientOptimizer(
        n_orders=n_orders, region=region, initial_coefficients=None, lr=0.05,
        max_iterations=5, device="cpu", seed=0, dtype="float64",
    )
    basis = opt.generate_basis()
    aperture = _aperture_from_basis(basis)
    return opt, basis, aperture


def _true_aberration(rng):
    n_total = 66  # 用 66 项承载
    c = np.zeros(n_total)
    vals = rng.normal(0.0, 1.0, len(TRUE_NOLL))
    vals = vals * (TRUE_RMS / np.sqrt(np.mean(vals**2)))
    for n, v in zip(TRUE_NOLL, vals):
        c[n - 1] = v
    return c


def _square(frame):
    m = compute_square_metrics(frame, TARGET_SIDE, zero_order_center(frame, refine=False))
    return {
        "cv": float(m["uniformity_cv"]),
        "ee": float(m["encircled_energy"]),
        "composite": 0.5 * float(m["encircled_energy"]) + 0.5 / (1.0 + float(m["uniformity_cv"])),
    }


def _make_forward(basis15, aperture, amp, target, aberr15):
    """返回一个闭包 (c_15, f) -> loss。

    总瞳面相位 = (c_15·Z_15 + f + A_15)。A_15 是真值像差 (Noll≤15 部分, 15 维
    系数向量), 优化时必须在场里, 各臂 (含 perpixel-only) 才会学到去补偿它,
    与真实 model-in-loop (Step A 拟合 A、Step B 在 A 下优化相位) 一致。
    """
    b15 = torch.from_numpy(np.ascontiguousarray(np.nan_to_num(basis15))).to("cpu", dtype=torch.float64)
    ap = torch.from_numpy(np.ascontiguousarray(aperture)).to("cpu", dtype=torch.float64)
    am = torch.from_numpy(np.ascontiguousarray(amp)).to("cpu", dtype=torch.float64)
    tgt = torch.from_numpy(np.ascontiguousarray(target)).to("cpu", dtype=torch.float64)
    a15 = torch.from_numpy(np.ascontiguousarray(aberr15)).to("cpu", dtype=torch.float64)

    def total_patch(c15_t, f_t):
        cZ = torch.einsum("j,jhw->hw", c15_t, b15)
        return torch.nan_to_num(cZ + f_t + torch.einsum("j,jhw->hw", a15, b15)) * ap

    def intensity(c15_t, f_t):
        patch = total_patch(c15_t, f_t)
        return interpolated_pupil_intensity(patch, am, REGION, 1)

    def loss(c15_t, f_t):
        I = intensity(c15_t, f_t)
        from ao_shaping.algorithm.signal_processing.differentiable_shaping import (
            efficiency_loss,
            uniformity_loss,
        )
        return W_UNIF * uniformity_loss(I, tgt) + W_EFF * efficiency_loss(I, tgt)

    return intensity, loss


def _shape(zernike_only: bool, perpixel: bool, seed: int, basis15, aperture, amp, target, aberr15, track_best=False):
    """优化 φ_slm。返回 (c_15, f, best_epoch)。"""
    _, loss = _make_forward(basis15, aperture, amp, target, aberr15)
    c15 = None
    f = None
    if zernike_only:
        c15 = torch.nn.Parameter(torch.zeros(15, dtype=torch.float64))
        f = torch.zeros((REGION, REGION), dtype=torch.float64)
    if perpixel:
        f = torch.nn.Parameter(torch.zeros((REGION, REGION), dtype=torch.float64))
    params = []
    if c15 is not None:
        params.append(c15)
    if isinstance(f, torch.nn.Parameter):
        params.append(f)
    if not params:
        raise ValueError("nothing to optimize")
    # 小随机起点逃离退化零点
    torch.manual_seed(seed)
    for p in params:
        with torch.no_grad():
            p += 0.1 * torch.randn_like(p)
    optm = torch.optim.Adam(params, lr=LR)
    best_loss, best_epoch = float("inf"), 0
    for i in range(ITERS):
        optm.zero_grad()
        l = loss(c15 if c15 is not None else torch.zeros(15, dtype=torch.float64),
                 f)
        l.backward()
        optm.step()
        lv = float(l.detach())
        if lv < best_loss:
            best_loss, best_epoch = lv, i
    c_final = None if c15 is None else c15.detach().numpy()
    f_final = None if not isinstance(f, torch.nn.Parameter) else f.detach().numpy()
    return c_final, f_final, (best_epoch if track_best else None)


def main():
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    seeds = [0, 1, 2]
    shapes = ["triangle", "star"]

    opt66, basis10, aperture10 = _zernike_basis(10, REGION)  # 用于最终 66 项 forward
    amp = ZernikeCoefficientOptimizer.native_amplitude(REGION)
    _, basis15, aperture15 = _zernike_basis(N_J, REGION)

    results = {"region": REGION, "n_j": N_J, "target_side": TARGET_SIDE, "iters": ITERS,
               "lr": LR, "true_rms": TRUE_RMS, "seeds": seeds, "shapes": {}}
    for shape in shapes:
        target = create_target_mask(shape, (REGION, REGION), TARGET_SIDE)
        shape_res = {}
        for arm in ("zernike_only", "perpixel_only", "joint"):
            rows = []
            for s in seeds:
                rng_true = np.random.default_rng(3000 + s)
                A = _true_aberration(rng_true)
                c15, f, best_epoch = _shape(
                    zernike_only=(arm == "zernike_only"),
                    perpixel=(arm in ("perpixel_only", "joint")),
                    seed=s, basis15=basis15, aperture=aperture15, amp=amp, target=target,
                    aberr15=A[:15],
                    track_best=(arm == "joint"),
                )
                # 最终在真值像差 A 下实测: 总相位 = (c15·Z + f) + A
                # 用 66 项 optimizer: coefficients = (c15_padded + A), phase_slm = f
                c15_padded = np.zeros(66)
                if c15 is not None:
                    c15_padded[:15] = c15
                c_total = c15_padded + A
                phase_slm = f if f is not None else np.zeros((REGION, REGION))
                frame = opt66.forward_intensity(c_total, phase_slm)
                m = _square(frame)
                m["best_epoch"] = None if best_epoch is None else float(best_epoch)
                rows.append(m)
            agg = {
                "cv_mean": float(np.mean([r["cv"] for r in rows])),
                "ee_mean": float(np.mean([r["ee"] for r in rows])),
                "composite_mean": float(np.mean([r["composite"] for r in rows])),
                "cv_std": float(np.std([r["cv"] for r in rows])),
                "ee_std": float(np.std([r["ee"] for r in rows])),
            }
            if arm == "joint":
                agg["best_epoch_mean"] = float(np.mean([r["best_epoch"] for r in rows]))
            shape_res[arm] = agg
        results["shapes"][shape] = shape_res
        print(f"\n[{shape}] target_side={TARGET_SIDE} iters={ITERS} seeds={seeds}")
        for arm in ("zernike_only", "perpixel_only", "joint"):
            r = shape_res[arm]
            extra = f"  best_epoch={r['best_epoch_mean']:.0f}" if arm == "joint" else ""
            print(f"  {arm:<14} CV={r['cv_mean']:.3f}±{r['cv_std']:.3f}  "
                  f"EE={r['ee_mean']:.3f}±{r['ee_std']:.3f}  "
                  f"composite={r['composite_mean']:.3f}{extra}")

    with open(out / "t2_mixed_fitting.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nJSON -> {out / 't2_mixed_fitting.json'}")


if __name__ == "__main__":
    main()
