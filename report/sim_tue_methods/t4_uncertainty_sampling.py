"""T4 样本查询 / 不确定性采样（主动学习）：issue #149 补充 / 论文 7.7 T4。

在 model-in-the-loop 可微仿真 (Zernike Noll 4..15 12 维参数域, 64x64 前向,
CPU, 无硬件) 上复现 TU/e 可微仿真的第四类能力「样本查询 / 不确定性采样」:

- 可微仿真: 12 维 Zernike 系数 -> 远场强度 (interpolated_pupil_intensity,
  端到端可微, 前向 ~27ms)。
- 冻结的逆模型: 3 个独立训练的小 CNN 集成 (远场 32x32 -> 12 维系数),
  认知不确定性 U(c) = 集成预测方差 (全程可微)。
- 主动学习环: 用梯度下降直接优化仿真输入 c, 最大化 U(c) (即"一口气反向
  搜索出网络最不确定的样本")。约束实验:
    random                  —— 域内均匀盲采样 (TU/e 所称传统做法);
    uncertainty_boxed       —— 梯度搜索 + 幅值盒约束 |c_i| <= BOX;
    uncertainty_unconstrained —— 梯度搜索不加幅值约束 (TU/e 的塌缩演示:
                                  参数被推向极端, 数据集塌缩)。
- 验证: 每轮把选出的样本并入逆模型训练集、重训、在固定验证集上测系数
  RMSE, 得三条"查询数 vs 测试 RMSE"学习曲线; 反向验证: 最小化 U(c) 应收敛
  到低幅值 (分布内) 样本, 对应 TU/e "最小化不确定度会让物体消失"。

全部 sim-only, 不动硬件写入路径。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer import (
    ZernikeCoefficientOptimizer,
)
from ao_shaping.optimizer.wfless.model_in_loop_shaping import _aperture_from_basis
from ao_shaping.utils.wavefront.fraunhofer import interpolated_pupil_intensity

# ---- 配置 -----------------------------------------------------------------
REGION = 64
N_ACT = 12                      # Noll 4..15 (1-indexed)
ACT_IDX = np.array(list(range(3, 15)))  # 0-indexed 位置
BOX = 0.8                       # 有约束变体的幅值盒 (rad)
UNCONST_RADIUS = 3.0            # 无约束变体允许搜索到的半径上限 (仅停止条件)
NET_SIZE = 32                   # 逆模型输入 (远场降采样)
N_MEMBERS = 3                   # 集成成员数
INIT_N = 300                    # 初始 (分布内) 训练集大小
INIT_SCALE = 0.3                # 初始集采样半径 (rad)
ROUNDS = 3
PER_ROUND = 8
SEARCH_INITS = 8                # 每轮梯度搜索的随机起点数
SEARCH_ITERS = 40
SEARCH_LR = 0.02
RANDOM_POOL = 1000              # 随机变体每轮盲采样池 (TU/e 传统做法: 盲仿一大批再挑最不确定的)
UNCONST_RADIUS = 6.0            # 无约束变体允许搜索到的范数上限 (仅停止条件, 让塌缩可见)
HOLDOUT_N = 64
FULL_RADIUS = 0.8               # 验证集/域定义半径 (rad, 与 BOX 同域)


def _build_forward() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    opt = ZernikeCoefficientOptimizer(
        n_orders=12, region=REGION, initial_coefficients=None, lr=0.05,
        max_iterations=3, device="cpu", seed=0, dtype="float64",
    )
    basis = opt.generate_basis()
    aperture = _aperture_from_basis(basis)
    amp = ZernikeCoefficientOptimizer.native_amplitude(REGION)
    b_t = torch.from_numpy(np.ascontiguousarray(basis)).to("cpu", dtype=torch.float64)
    ap_t = torch.from_numpy(np.ascontiguousarray(aperture)).to("cpu", dtype=torch.float64)
    am_t = torch.from_numpy(np.ascontiguousarray(amp)).to("cpu", dtype=torch.float64)
    return b_t, ap_t, am_t


_B, _AP, _AM = _build_forward()


def far_field(c12: torch.Tensor) -> torch.Tensor:
    """12 维 Zernike 系数 (Noll 4..15) -> 64x64 远场强度 (可微)。"""
    c91 = torch.zeros(_B.shape[0], dtype=torch.float64)
    idx = torch.from_numpy(ACT_IDX).to(torch.int64)
    c91.index_copy_(0, idx, c12)
    Z = torch.einsum("j,jhw->hw", c91, _B)
    ph = torch.nan_to_num(Z) * _AP
    return interpolated_pupil_intensity(ph, _AM, REGION, 1)


def far_field_down(c12: torch.Tensor) -> torch.Tensor:
    I = far_field(c12)
    return torch.nn.functional.interpolate(
        I[None, None], size=(NET_SIZE, NET_SIZE), mode="bilinear", align_corners=True
    )[0, 0]


def triu_indices(n: int) -> np.ndarray:
    return np.triu_indices(n, k=1)


# ---- 逆模型 (小 CNN) --------------------------------------------------------
class InverseNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1)
        self.r1 = nn.ReLU()
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.r2 = nn.ReLU()
        self.fc = nn.Linear(32 * NET_SIZE * NET_SIZE, N_ACT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.r1(self.c1(x))
        x = self.r2(self.c2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


def make_ensemble(seeds: tuple[int, ...]) -> list[InverseNet]:
    nets = []
    for s in seeds:
        torch.manual_seed(s)
        nets.append(InverseNet())
    return nets


def sample_domain(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    c = rng.normal(0.0, 1.0, size=(n, N_ACT))
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    r = np.abs(rng.normal(0.0, radius / 2.0, size=n)) + radius / 4.0
    r = np.clip(r, 0.0, radius)
    return (c * r[:, None]).astype(np.float64)


_FAR_CACHE: dict[int, torch.Tensor] = {}
_FORWARD_CALLS: int = 0


def far_field_count() -> int:
    return _FORWARD_CALLS


def far_field_down_cached(c: np.ndarray) -> torch.Tensor:
    global _FORWARD_CALLS
    key = int(np.abs(np.ascontiguousarray(c).view(np.uint64)).sum())
    v = _FAR_CACHE.get(key)
    if v is None:
        v = far_field_down(torch.from_numpy(np.ascontiguousarray(c)))
        _FAR_CACHE[key] = v
        _FORWARD_CALLS += 1
    return v


def train_ensemble(nets: list[InverseNet], c_train: np.ndarray, epochs: int = 12, lr: float = 1e-3) -> None:
    """在给定系数集上训练全部集成成员 (输入=各自前向远场, 标签=系数)。"""
    # 远场与系数都按构造确定, 只需算一次 (带缓存)
    with torch.no_grad():
        X = torch.stack([far_field_down_cached(c) for c in c_train])
    Xf = X.unsqueeze(1).float()
    Y = torch.from_numpy(c_train).float()
    lossf = nn.MSELoss()
    for net in nets:
        net.train()
        optm = torch.optim.Adam(net.parameters(), lr=lr)
        for _ in range(epochs):
            optm.zero_grad()
            l = lossf(net(Xf), Y)
            l.backward()
            optm.step()
        net.eval()


def ensemble_pred(nets: list[InverseNet], x32: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x4 = x32[None, None]  # (1,1,H,W)
    preds = []
    for net in nets:
        with torch.no_grad():
            p = net(x4.float())
        preds.append(p)
    stack = torch.stack(preds, dim=0)  # (K, 1, 12)
    mean = stack.mean(0)[0]
    var = stack.var(0, unbiased=False)[0]
    return mean, var


def uncertainty_diff(nets: list[InverseNet], c12: torch.Tensor) -> torch.Tensor:
    """可微认知不确定性: 集成预测方差 (对 c 可微)。far-field 只算一次, 各成员共享。"""
    x4 = far_field_down(c12)[None, None]  # (1,1,H,W)
    preds = [net(x4.float()) for net in nets]
    stack = torch.stack(preds, dim=0)  # (K, 1, 12)
    var = stack.var(0, unbiased=False)
    return var.mean()

def grad_search(
    nets: list[InverseNet],
    c0: np.ndarray,
    iters: int,
    maximize: bool,
    box: float | None,
) -> tuple[np.ndarray, float]:
    """从 c0 出发用梯度下降 最大化/最小化 U(c)。box 给定时每步做投影回域。"""
    c = torch.from_numpy(np.ascontiguousarray(np.array(c0, copy=True))).to("cpu", dtype=torch.float64)
    c = torch.nn.Parameter(c)
    optm = torch.optim.Adam([c], lr=SEARCH_LR)
    best = c.detach().clone()
    best_u = float("inf") if maximize else float("-inf")
    for _ in range(iters):
        optm.zero_grad()
        u = uncertainty_diff(nets, c)
        if maximize:
            u.backward()
        else:
            (-u).backward()
        optm.step()
        with torch.no_grad():
            if box is not None:
                c.clamp_(-box, box)
            else:
                nrm = c.norm()
                if nrm > UNCONST_RADIUS:
                    c *= UNCONST_RADIUS / nrm
        u_now = float(uncertainty_diff(nets, c.detach()))
        if (maximize and u_now < best_u) or (not maximize and u_now > best_u):
            best = c.detach().clone()
            best_u = u_now
    return best.numpy(), best_u


def test_rmse(nets: list[InverseNet], c_test: np.ndarray) -> float:
    errs = []
    for c in c_test:
        with torch.no_grad():
            x = far_field_down(torch.from_numpy(np.ascontiguousarray(c)))[None, None].float()
        preds = torch.stack([net(x) for net in nets])
        mean = preds.mean(0)[0].detach().numpy()
        errs.append(float(np.mean((mean - c) ** 2)))
    return float(np.sqrt(np.mean(errs)))


def main() -> None:
    t_start = time.time()
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    print("== T4 不确定性采样 / 主动学习 ==")
    # 域与验证集 (固定)
    c_init = sample_domain(rng, INIT_N, INIT_SCALE)
    c_holdout = sample_domain(rng, HOLDOUT_N, FULL_RADIUS)

    # 初始逆模型
    nets = make_ensemble((11, 22, 33))
    t0 = time.time()
    train_ensemble(nets, c_init)
    print(f"初始集成训练 {time.time() - t0:.1f}s")
    rmse0 = test_rmse(nets, c_holdout)
    print(f"初始 (仅 {INIT_N} 个分布内样本) 验证集 RMSE = {rmse0:.4f}")

    results = {
        "region": REGION, "n_act": N_ACT, "box": BOX, "unconst_radius": UNCONST_RADIUS,
        "init_n": INIT_N, "init_scale": INIT_SCALE, "rounds": ROUNDS,
        "per_round": PER_ROUND, "holdout_n": HOLDOUT_N,
        "net_params": int(sum(p.numel() for p in nets[0].parameters())),
        "init_rmse": rmse0,
        "arms": {}, "sanity_min_confidence": None,
    }

    # ---- 反向验证: 最小化 U(c) 应收敛到分布内 (低幅值) 样本 ----
    t0 = time.time()
    c_lo, u_lo = grad_search(nets, c_init[0], SEARCH_ITERS, maximize=False, box=None)
    # 对照: 从域边界点出发
    c_lo2, u_lo2 = grad_search(
        nets, np.full(N_ACT, 0.75), SEARCH_ITERS, maximize=False, box=None
    )
    results["sanity_min_confidence"] = {
        "from_interior": {"norm": float(np.linalg.norm(c_lo)), "U": u_lo},
        "from_boundary": {"norm": float(np.linalg.norm(c_lo2)), "U": u_lo2},
        "time_s": round(time.time() - t0, 1),
    }
    print(f"反向验证 (最小化 U): 内部起点 norm {np.linalg.norm(c_lo):.3f}, "
          f"边界起点 norm {np.linalg.norm(c_lo2):.3f}, {results['sanity_min_confidence']['time_s']}s")

    arms = {
        "random": "域内均匀盲采样 (传统做法: 每轮盲仿 200 个再挑最不确定的 8 个)",
        "uncertainty_boxed": "梯度搜索 + 幅值盒 |c_i|<=BOX",
        "uncertainty_unconstrained": "梯度搜索, 无幅值约束 (塌缩演示)",
    }
    # 每个变体从同一组 (同种子) 初始逆模型出发, 避免跨变体污染。
    for arm, desc in arms.items():
        arm_nets = make_ensemble((11, 22, 33))
        train_ensemble(arm_nets, c_init)
        domain_limit = {"random": FULL_RADIUS, "uncertainty_boxed": BOX,
                        "uncertainty_unconstrained": UNCONST_RADIUS}[arm]
        selected: list[np.ndarray] = []
        curve = []
        t0 = time.time()
        for r in range(ROUNDS):
            cand_pool: list[np.ndarray] = []
            if arm == "random":
                cand_pool = list(sample_domain(rng, RANDOM_POOL, FULL_RADIUS))
            else:
                box = BOX if arm == "uncertainty_boxed" else None
                inits = sample_domain(rng, SEARCH_INITS, FULL_RADIUS if box is None else BOX)
                for c0 in inits:
                    c_found, _ = grad_search(arm_nets, c0, SEARCH_ITERS, maximize=True, box=box)
                    cand_pool.append(c_found)
            # 按当前逆模型的认知不确定性 U 降序, 取 PER_ROUND 个进入训练集
            us = []
            for c in cand_pool:
                _, var = ensemble_pred(arm_nets, far_field_down(torch.from_numpy(np.ascontiguousarray(c))))
                us.append(float(var.mean()))
            order = np.argsort(-np.array(us))[:PER_ROUND]
            round_sel = [cand_pool[i] for i in order]
            norms = [float(np.linalg.norm(c)) for c in round_sel]
            arr = np.array(round_sel)
            d = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=2)
            pair_dist = float(d[triu_indices(d.shape[0])].mean()) if len(round_sel) > 1 else 0.0
            selected.extend(round_sel)
            c_train = np.vstack([c_init, np.array(selected)])
            train_ensemble(arm_nets, c_train, epochs=12)
            rmse = test_rmse(arm_nets, c_holdout)
            curve.append({
                "round": r + 1,
                "n_queries": len(selected),
                "test_rmse": round(rmse, 5),
                "sel_norm_mean": round(float(np.mean(norms)), 4),
                "sel_norm_max": round(float(max(norms)), 4),
                "sel_pair_dist": round(pair_dist, 4),
                "forward_calls_cum": far_field_count(),
                "boundary_hits": int(sum(1 for n in norms if n >= 0.98 * domain_limit)),
                "elapsed_s": round(time.time() - t0, 1),
            })
            print(f"[{arm}] round {r + 1}: RMSE={rmse:.4f} norm={np.mean(norms):.3f}/{max(norms):.3f} pair_dist={pair_dist:.3f}")
        results["arms"][arm] = {"desc": desc, "curve": curve,
                                "total_time_s": round(time.time() - t0, 1),
                                "forward_calls_total": far_field_count()}

    with open(out / "t4_uncertainty_sampling.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n完成, 共 {time.time() - t_start:.0f}s。结果: {out / 't4_uncertainty_sampling.json'}")
    print("=== 学习曲线 (验证集 RMSE) ===")
    print(f"{'round':>5} {'n_q':>4} | " + " | ".join(results["arms"]))
    for r in range(ROUNDS):
        row = f"{r+1:>5} {results['arms']['random']['curve'][r]['n_queries']:>4} | "
        row += " | ".join(
            f"{results['arms'][a]['curve'][r]['test_rmse']:.4f}" for a in results["arms"]
        )
        print(row)


if __name__ == "__main__":
    main()
