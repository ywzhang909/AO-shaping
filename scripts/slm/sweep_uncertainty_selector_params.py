"""参数扫描: commit 080e0ac9 (R-86) 的不确定性探针选择器能否靠调参救回来。

在 :mod:`scripts.slm.verify_uncertainty_probes_sim` 里已经测到: 选出的探针其远场
预测方差低于盲随机。但那一次对照有两个混杂因子 (幅度、带限), 补上归一之后
ratio 仍是 0.513。这个脚本回答一个更窄、也更可执行的问题:

    **在同一个被约束的探针族里, 梯度上升是否优于随机抽样?**

"同一个族" = 同样的带限 sigma、同样的幅度带。族内的随机抽样是这套机制**唯一
公平的对手**: 如果连"从自己定义的分布里随手抽一张"都赢不了, 那它作为探针
选择器就没有价值, 无论参数怎么调。

每个配置跑 ``n_seeds`` 个独立种子 (各自建选择器、共享同一个参考后验), 报告
ratio 的均值/中位数与"赢的种子比例"。**判据是赢的比例, 不是均值** —— 均值会被
个别极端种子拉走, 而"多数种子上赢"才是机制成立的必要条件。

纯离线: 正向算子是解析 FFT, 不碰硬件。用法::

    python scripts/slm/sweep_uncertainty_selector_params.py
    python scripts/slm/sweep_uncertainty_selector_params.py --seeds 12
"""

from __future__ import annotations

import sys
from loguru import logger
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch
from scipy import ndimage

from ao_shaping.algorithm.signal_processing.uncertainty_probe_selector import (
    UncertaintyProbeSelector,
)
from ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer import (
    ZernikeCoefficientOptimizer,
)


def _log(msg: str) -> None:
    logger.info(msg)


def _u(selector: UncertaintyProbeSelector, probe: np.ndarray) -> float:
    """选择器自己的公共方法度量 (单一真源, 不在这里重写公式)。"""
    leaf = torch.from_numpy(np.ascontiguousarray(probe, dtype=np.float64))
    leaf.requires_grad_(True)
    with torch.no_grad():
        return float(selector.predictive_variance(leaf))


def _same_family_random(
    rng: np.random.Generator, *, region: int, bandlimit_sigma: float,
    std: float,
) -> np.ndarray:
    """从**同一被约束族**里随机抽一张: 同样的带限 sigma, 再归一到同样的 std。

    这是本脚本的核心对照。与"循环里的盲随机探针"相比, 它排除了带限与幅度的
    差异 —— 剩下的差异才只可能来自梯度上升。
    """
    p = rng.standard_normal((region, region))
    if bandlimit_sigma > 0:
        p = ndimage.gaussian_filter(p, sigma=bandlimit_sigma)
    s = float(p.std())
    if s <= 1e-12:
        p = rng.standard_normal((region, region))
        s = float(p.std())
    return p * (std / s)


#: 扫描的配置。每项只改一到两个维度, 且都对着上一轮诊断出的可疑点:
#: 带限 (会压低 U)、步数/步长 (可能根本没收敛)、幅度下限 (会被推到边界)、
#: 样本数与岭 (后验质量)、多样性 (可能把探针推离最优)。
CONFIGS: tuple[dict, ...] = (
    {"tag": "baseline (线上默认)", "bandlimit_sigma": 1.5, "lr": 0.3,
     "steps": 30, "n_samples": 6, "ridge": 1e-3, "diversity_weight": 0.5,
     "spread_floor_frac": 0.75},
    {"tag": "无带限 sigma=0", "bandlimit_sigma": 0.0, "lr": 0.3,
     "steps": 30, "n_samples": 6, "ridge": 1e-3, "diversity_weight": 0.5,
     "spread_floor_frac": 0.75},
    {"tag": "带限放宽 sigma=3", "bandlimit_sigma": 3.0, "lr": 0.3,
     "steps": 30, "n_samples": 6, "ridge": 1e-3, "diversity_weight": 0.5,
     "spread_floor_frac": 0.75},
    {"tag": "步数 300", "bandlimit_sigma": 1.5, "lr": 0.3,
     "steps": 300, "n_samples": 6, "ridge": 1e-3, "diversity_weight": 0.5,
     "spread_floor_frac": 0.75},
    {"tag": "步长 1.0 + 步数 300", "bandlimit_sigma": 1.5, "lr": 1.0,
     "steps": 300, "n_samples": 6, "ridge": 1e-3, "diversity_weight": 0.5,
     "spread_floor_frac": 0.75},
    {"tag": "无带限 + 步数 300", "bandlimit_sigma": 0.0, "lr": 0.3,
     "steps": 300, "n_samples": 6, "ridge": 1e-3, "diversity_weight": 0.5,
     "spread_floor_frac": 0.75},
    {"tag": "幅度下限=上限 (钉住)", "bandlimit_sigma": 1.5, "lr": 0.3,
     "steps": 30, "n_samples": 6, "ridge": 1e-3, "diversity_weight": 0.5,
     "spread_floor_frac": 1.0},
    {"tag": "多样性关 (w=0)", "bandlimit_sigma": 1.5, "lr": 0.3,
     "steps": 30, "n_samples": 6, "ridge": 1e-3, "diversity_weight": 0.0,
     "spread_floor_frac": 0.75},
    {"tag": "样本 20", "bandlimit_sigma": 1.5, "lr": 0.3,
     "steps": 30, "n_samples": 20, "ridge": 1e-3, "diversity_weight": 0.5,
     "spread_floor_frac": 0.75},
    {"tag": "岭 1e-1 (弱正则)", "bandlimit_sigma": 1.5, "lr": 0.3,
     "steps": 30, "n_samples": 6, "ridge": 1e-1, "diversity_weight": 0.5,
     "spread_floor_frac": 0.75},
)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--region", type=int, default=32)
    ap.add_argument("--n-orders", type=int, default=4)
    ap.add_argument("--spread", type=float, default=4.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else (
        ROOT / "data" / "debug" / "uncertainty_selector_sweep" / stamp
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    source = np.zeros((args.region, args.region), dtype=np.float64)
    yy, xx = np.mgrid[0:args.region, 0:args.region]
    c = (args.region - 1) / 2.0
    source[(yy - c) ** 2 + (xx - c) ** 2 <= c ** 2] = 1.0
    opt = ZernikeCoefficientOptimizer(
        n_orders=args.n_orders, region=args.region, radius=args.region // 2,
        lr=0.05, device="cpu", dtype="float32", seed=None,
        far_field_size=args.region * 4, frozen_modes=(1, 2, 3),
    )
    opt.set_source_amplitude(source)

    # 所有配置共享同一个参考后验: 由固定的一组随机探针给出。否则每个配置在
    # 不同的尺子上量, 跨配置就不可比了。
    ref_rng = np.random.default_rng(12345)
    reference = [
        ndimage.gaussian_filter(ref_rng.standard_normal((args.region,) * 2), 1.5)
        for _ in range(4)
    ]
    reference = [r * (args.spread / float(r.std())) for r in reference]

    _log("=" * 92)
    _log("不确定性选择器参数扫描  (commit 080e0ac9 / #134 R-86)")
    _log(f"region={args.region} n_orders={args.n_orders} spread={args.spread} "
         f"seeds={args.seeds}")
    _log("对照 = 同一被约束族内的随机抽样 (同带限、同幅度)")
    _log("=" * 92)

    rows = []
    logger.info(f"{'配置':<26}{'U_unc':>11}{'U_rand':>11}{'ratio':>8}"
          f"{'中位ratio':>11}{'赢':>7}{'std(选中)':>11}{'秒':>7}")
    for cfg in CONFIGS:
        ratios, u_uncs, u_rands, stds, secs = [], [], [], [], []
        for s in range(args.seeds):
            seed = 1000 + s
            sel = UncertaintyProbeSelector(
                opt, seed=seed,
                n_samples=int(cfg["n_samples"]),
                ridge=float(cfg["ridge"]),
                spread_cap=args.spread,
                spread_floor=float(cfg["spread_floor_frac"]) * args.spread,
                bandlimit_sigma=float(cfg["bandlimit_sigma"]),
                diversity_weight=float(cfg["diversity_weight"]),
                lr=float(cfg["lr"]),
                mode="maximize",
            )
            sel.update_covariance(reference)      # 共享尺子
            t0 = datetime.now()
            for _ in range(int(cfg["steps"])):
                sel.update()
            secs.append((datetime.now() - t0).total_seconds())
            probe = sel.current_phase()
            std = float(probe.std())
            rand = _same_family_random(
                np.random.default_rng(90000 + seed), region=args.region,
                bandlimit_sigma=float(cfg["bandlimit_sigma"]), std=std,
            )
            uu, ur = _u(sel, probe), _u(sel, rand)
            ratios.append(uu / ur if ur > 0 else np.nan)
            u_uncs.append(uu)
            u_rands.append(ur)
            stds.append(std)

        arr = np.asarray(ratios, dtype=np.float64)
        wins = int(np.sum(arr > 1.0))
        rows.append({
            "config": {k: v for k, v in cfg.items()},
            "u_uncertainty_mean": float(np.mean(u_uncs)),
            "u_random_mean": float(np.mean(u_rands)),
            "ratio_mean": float(np.nanmean(arr)),
            "ratio_median": float(np.nanmedian(arr)),
            "ratio_all": [float(x) for x in arr],
            "wins": wins,
            "n_seeds": args.seeds,
            "win_fraction": wins / args.seeds,
            "probe_std_mean": float(np.mean(stds)),
            "seconds_mean": float(np.mean(secs)),
        })
        logger.info(f"{cfg['tag']:<26}{np.mean(u_uncs):>11.4e}{np.mean(u_rands):>11.4e}"
              f"{np.nanmean(arr):>8.3f}{np.nanmedian(arr):>11.3f}"
              f"{wins:>4}/{args.seeds:<2}{np.mean(stds):>11.3f}"
              f"{np.mean(secs):>7.1f}")

    best = max(rows, key=lambda r: (r["win_fraction"], r["ratio_median"]))
    _log("-" * 92)
    _log(f"最佳配置: {best['config']['tag']}  "
         f"赢 {best['wins']}/{args.seeds} ({best['win_fraction']:.0%}), "
         f"中位 ratio {best['ratio_median']:.3f}")
    if best["win_fraction"] >= 0.8 and best["ratio_median"] > 1.0:
        _log("=> 存在多数种子上稳定优于同族随机的配置; 值得进一步做闭环 A/B。")
    else:
        _log("=> 没有任何配置在多数种子上稳定优于同族随机。")
        _log("   调参救不回这个机制: 族内随机抽样已经是它的最强对手。")

    (out_dir / "sweep.json").write_text(
        json.dumps({
            "commit": "080e0ac9de2d7d453676d174ae4a1845acb77ee8",
            "issue": "#134 (R-86)",
            "timestamp_utc": stamp,
            "args": vars(args),
            "rows": rows,
            "best_tag": best["config"]["tag"],
        }, indent=2, ensure_ascii=False),
        encoding="utf-8")
    _log(f"保存: {out_dir / 'sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())