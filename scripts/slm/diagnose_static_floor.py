# -*- coding: utf-8 -*-
"""离线诊断：加性静态项能否移动 smooth/bandlimited 收敛下界（§9.2）。

用 fit 已抽取并对齐的 measured / prediction（161x161），不重拟合：
  1. 从 train measured 的高通分量构建**共享静态图**（test 不参与建图）；
  2. 把静态图作为加性项从所有 measured 中扣除（估计"观测模型补全后"的 target）；
  3. 用同一 predict 重算 test 的 correlation / relative L2；
  4. 与未扣静态项的基线对照，判断收敛下界是否可重复下降。

这是"加性静态项能解释多少收敛下界"的**上界诊断**：静态图幅度用 train 最小
二乘投影估计（仅 train），test 作为验证集完全不参与。若 test 上 smooth/bandlimited 相对
L2 明显下降，则支持"补观测模型"方向；若不下降，则加性静态项不足以解释收敛
下界，需考虑与相位交互的静态项或展宽。

用法：
  python3 diagnose_static_floor.py --dataset <dir> --fit <name> \
    --json <path>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


def highpass(img: np.ndarray, sigma: float) -> np.ndarray:
    return img - gaussian_filter(img, sigma)


def rl2(pred: np.ndarray, target: np.ndarray) -> float:
    p, t = pred.ravel(), target.ravel()
    return float(np.linalg.norm(p - t) / max(np.linalg.norm(t), 1e-9))


def corr(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.corrcoef(pred.ravel(), target.ravel())[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--fit", default="fit_m3_full")
    ap.add_argument("--json", required=True, type=Path)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--flat", action="store_true",
                    help="用 _optical_probe_flat.npz 的高频作静态模板（独立于相位），"
                         "替代 train measured 平均；幅度仍只用 train 最小二乘估计")
    ap.add_argument("--iteration", type=int, default=None,
                    help="默认取 result.json 的 best iteration")
    args = ap.parse_args()

    ds = args.dataset
    fitdir = ds / args.fit
    inputs = np.load(fitdir / "inputs.npz")
    measured = np.asarray(inputs["measured"], dtype=float)  # (68,161,161)
    train_mask = np.asarray(inputs["train"], dtype=bool)
    result = json.load(open(fitdir / "result.json"))
    it = args.iteration if args.iteration is not None else int(result["iteration"])
    pred = np.load(fitdir / f"iteration_{it:04d}.npz")["prediction"].astype(float)

    # 逐行标签 / 相位族（从 compare json 或 probe_plan + capture_records）。
    cmp = json.load(open(ds / "fit_compare_m3_z11_vs_z21.json"))
    rows = cmp["rows"]  # 与 measured 同顺序（index 字段）
    labels = [r["label"] for r in rows]
    split = [r["split"] for r in rows]
    families = [r["family"] for r in rows]
    # 把 "Z4 +0.10" 之类归为单模，smooth/bandlimited 归为其族名。
    def fam(f: str) -> str:
        if "smooth" in f.lower():
            return "smooth"
        if "bandlimited" in f.lower():
            return "bandlimited"
        return "zernike"
    fams = [fam(f) for f in families]

    # 高频静态图：
    #   - 默认：只用 train measured 的高通分量均值构建；
    #   - --flat：用 _optical_probe_flat.npz（零相位参考）的高频，完全独立于相位。
    if args.flat:
        flat = np.load(ds / "_optical_probe_flat.npz")
        frame = np.asarray(flat["frame"], dtype=float) / float(flat["exposure_ms"])
        cxy = np.asarray(inputs["camera_xy"], dtype=int)  # (161,161,2), [...,0]=x(col), [...,1]=y(row)
        patch = frame[cxy[:, :, 1], cxy[:, :, 0]]
        static = highpass(patch, args.sigma)
    else:
        train_hp = np.stack([highpass(measured[i], args.sigma)
                             for i in range(len(measured)) if train_mask[i]])
        static = train_hp.mean(axis=0)
    # 符号：使 train 高通投影系数均值为正。
    coeffs_train = [float((highpass(measured[i], args.sigma) * static).sum()
                          / (static * static).sum())
                    for i in range(len(measured)) if train_mask[i]]
    if np.mean(coeffs_train) < 0:
        static = -static

    # 对每个探针，用 train 估计的静态图，按最小二乘投影求其幅度，从 measured 扣除。
    target_corr = np.empty_like(measured)
    for i in range(len(measured)):
        hi = highpass(measured[i], args.sigma)
        c = float((hi * static).sum() / (static * static).sum())
        target_corr[i] = measured[i] - c * static

    # 逐行指标：基线 vs 扣静态项。
    recs = []
    for i in range(len(measured)):
        recs.append({
            "index": i,
            "label": labels[i],
            "split": split[i],
            "family": fams[i],
            "corr_base": corr(pred[i], measured[i]),
            "corr_corr": corr(pred[i], target_corr[i]),
            "rl2_base": rl2(pred[i], measured[i]),
            "rl2_corr": rl2(pred[i], target_corr[i]),
        })

    def aggregate(keys, name):
        sub = [r for r in recs if r["split"] in keys]
        if not sub:
            return None
        return {
            "n": len(sub),
            "corr_base": float(np.mean([r["corr_base"] for r in sub])),
            "corr_corr": float(np.mean([r["corr_corr"] for r in sub])),
            "rl2_base": float(np.mean([r["rl2_base"] for r in sub])),
            "rl2_corr": float(np.mean([r["rl2_corr"] for r in sub])),
            "rl2_delta": float(np.mean([r["rl2_corr"] - r["rl2_base"] for r in sub])),
        }

    summary = {
        "dataset": str(ds),
        "fit": args.fit,
        "iteration": it,
        "sigma": args.sigma,
        "static_norm": float(np.linalg.norm(static)),
        "static_coeff_train_mean": float(np.mean(coeffs_train)),
        "all": aggregate(["train", "test"], "all"),
        "train": aggregate(["train"], "train"),
        "test": aggregate(["test"], "test"),
    }
    # 按族细分 test / train。
    for sp in ("train", "test"):
        for f in ("zernike", "smooth", "bandlimited"):
            sub = [r for r in recs if r["split"] == sp and r["family"] == f]
            if sub:
                summary[f"{sp}_{f}"] = {
                    "n": len(sub),
                    "rl2_base": float(np.mean([r["rl2_base"] for r in sub])),
                    "rl2_corr": float(np.mean([r["rl2_corr"] for r in sub])),
                    "rl2_delta": float(np.mean([r["rl2_corr"] - r["rl2_base"] for r in sub])),
                    "corr_base": float(np.mean([r["corr_base"] for r in sub])),
                    "corr_corr": float(np.mean([r["corr_corr"] for r in sub])),
                }
    out = {"summary": summary, "rows": recs}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
