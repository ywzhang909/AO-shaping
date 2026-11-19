# -*- coding: utf-8 -*-
"""离线表征 transfer 数据集中的共享静态结构（§8.5 假设的稳健验证）。

复现并扩展现有 §8.5 的"远场静态结构检验"：从 12 探针扩到全部 68 探针。
目标：判断实测远场高频内容是否主要为共享静态光学结构（PSF 环 / 散斑 /
固定图案），并给出可入观测模型的形式（加性远场 vs 瞳面交互）。

方法（与 §8.5 一致）：
  - 每个探针取 400x400 spot 窗口（中心 = capture_records 的 spot_xy）；
  - 除以 exposure_ms（每探针标量）、按窗口均值归一；
  - 高斯高通分离（sigma=6 px），得到高通图 h_i；
  - 两两 Pearson 相关（非对角均值）；跨探针均值图作为"静态图" g；
  - 每探针 g 的解释能量（投影能量 / 高通能量）；
  - SVD rank-1 解释能量（检验是否真为单张共享图）。

输出：stdout 摘要 + --json <path> 写逐探针/汇总 JSON。

用法：
  python3 characterize_transfer_static_structure.py \
    --dataset data/debug/slm_transfer/20261118_transfer_mil_zernike_probe_v1 \
    --json out/static_structure.json
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


def highpass(img: np.ndarray, sigma: float = 6.0) -> np.ndarray:
    """高斯高通：img - lowpass(img)。"""
    return img - gaussian_filter(img, sigma)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel()
    b = b.ravel()
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = np.sqrt((a0 * a0).sum() * (b0 * b0).sum())
    if denom <= 0:
        return float("nan")
    return float((a0 * b0).sum() / denom)


def explained_energy(signal: np.ndarray, basis: np.ndarray) -> float:
    """signal 在 basis 上的投影能量 / signal 自身能量。"""
    sig = signal.ravel()
    b = basis.ravel()
    b0 = b - b.mean()
    num = (sig * b0).sum()
    den = (b0 * b0).sum()
    if den <= 0:
        return float("nan")
    coeff = num / den
    resid = sig - coeff * b0
    sig_energy = (sig * sig).sum()
    if sig_energy <= 0:
        return float("nan")
    return float(((sig * sig).sum() - (resid * resid).sum()) / sig_energy)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--window", type=int, default=400)
    ap.add_argument("--sigma", type=float, default=6.0)
    args = ap.parse_args()

    ds = args.dataset
    records = pickle.load(open(ds / "capture_records.pkl", "rb"))
    plan = json.load(open(ds / "probe_plan.json", "rb"))["plan"]
    # record label 与 plan label 命名不一致：zernike 单模探针 record label == plan
    # label（无后缀）；random/holdout 探针 record label = plan label + "_" + 相位族
    # （后缀为 zernike/smooth/bandlimited）。plan 的 kind 是探针大类
    # （zernike/random/holdout），与后缀无关。record label 前缀匹配 plan label。
    plan_by_label = {p["label"]: p for p in plan}
    suffixes = {"zernike", "smooth", "bandlimited"}

    def match_plan(record_label: str):
        if record_label in plan_by_label:
            return plan_by_label[record_label]
        for suf in suffixes:
            if record_label.endswith("_" + suf):
                base_label = record_label[: -(len(suf) + 1)]
                if base_label in plan_by_label:
                    return plan_by_label[base_label]
        return None

    # 探针 = 能匹配到 plan 的 sample（跳过 reference/ramp 校准）。
    probes = []
    for rec in records:
        if match_plan(rec["label"]) is None:
            continue
        probes.append(rec)
    probes.sort(key=lambda r: r["file"])

    hp_images: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}
    for rec in probes:
        label = rec["label"]
        p = match_plan(label)
        if p is None:
            continue
        # 族用 record label 的相位族后缀（zernike/smooth/bandlimited，见 §8.3）；
        # 无后缀的单模探针归入 zernike。
        phase_kind = next((s for s in suffixes if label.endswith("_" + s)), "zernike")
        z = np.load(ds / rec["file"])
        frame = np.asarray(z["frame"], dtype=float)
        sy, sx = frame.shape
        cx, cy = rec["spot_xy"]
        w = args.window
        x0 = int(round(cx - w / 2))
        y0 = int(round(cy - w / 2))
        x0 = max(0, min(x0, sx - w))
        y0 = max(0, min(y0, sy - w))
        win = frame[y0 : y0 + w, x0 : x0 + w]
        exposure = rec.get("exposure_ms", 1.0) or 1.0
        win = win / exposure
        win = win / win.mean()
        hp = highpass(win, args.sigma)
        hp_images[label] = hp
        meta[label] = {
            "file": rec["file"],
            "split": p["split"],
            "kind": phase_kind,
            "label": label,
            "window_origin": [x0, y0],
            "exposure_ms": rec.get("exposure_ms"),
        }

    labels = sorted(hp_images)
    n = len(labels)

    # 两两相关（非对角均值）。
    corr = np.full((n, n), np.nan)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i != j:
                corr[i, j] = pearson(hp_images[a], hp_images[b])
    offdiag = corr[~np.eye(n, dtype=bool)]

    # 静态图 = 跨探针均值（符号约定：使与首个探针相关为正）。
    stack = np.stack([hp_images[l] for l in labels])
    mean_img = stack.mean(axis=0)
    if pearson(mean_img, hp_images[labels[0]]) < 0:
        mean_img = -mean_img

    # 每探针解释能量（用均值图作基）。
    expl = {l: explained_energy(hp_images[l], mean_img) for l in labels}

    # train→test 验证集检验：静态图只用 train 探针平均构建，test 不参与建图。
    train_labels = [l for l in labels if meta[l]["split"] == "train"]
    test_labels = [l for l in labels if meta[l]["split"] == "test"]
    holdout_static = np.stack([hp_images[l] for l in train_labels]).mean(axis=0)
    if pearson(holdout_static, hp_images[train_labels[0]]) < 0:
        holdout_static = -holdout_static
    expl_holdout = {l: explained_energy(hp_images[l], holdout_static) for l in labels}
    holdout_report = {}
    for sp, ls in (("train", train_labels), ("test", test_labels)):
        if ls:
            vals = [expl_holdout[l] for l in ls]
            holdout_report[sp] = {
                "n": len(ls),
                "explained_energy_mean": float(np.mean(vals)),
                "explained_energy_min": float(np.min(vals)),
            }
            # 按相位族细分。
            for kind in sorted({meta[l]["kind"] for l in ls}):
                sub = [expl_holdout[l] for l in ls if meta[l]["kind"] == kind]
                holdout_report[f"{sp}_{kind}"] = {
                    "n": len(sub),
                    "explained_energy_mean": float(np.mean(sub)),
                }

    # SVD rank-1 解释能量（检验单张共享图是否充分）。
    flat = np.stack([hp_images[l].ravel() for l in labels])
    flat = flat - flat.mean(axis=1, keepdims=True)
    _, s, _ = np.linalg.svd(flat, full_matrices=False)
    energy_total = float((flat * flat).sum())
    rank1 = float(s[0] ** 2) if energy_total > 0 else float("nan")
    rank1_frac = rank1 / energy_total if energy_total > 0 else float("nan")

    # 按族分组。
    def group(kind):
        return [l for l in labels if meta[l]["kind"] == kind]

    fam_report = {}
    for kind in sorted({meta[l]["kind"] for l in labels}):
        ls = group(kind)
        if not ls:
            continue
        vals = [expl[l] for l in ls]
        fam_report[kind] = {
            "n": len(ls),
            "explained_energy_mean": float(np.mean(vals)),
            "explained_energy_min": float(np.min(vals)),
            "explained_energy_max": float(np.max(vals)),
        }

    # 按族 / split 的相关均值。
    def corr_group(ls):
        idx = [labels.index(l) for l in ls]
        sub = corr[np.ix_(idx, idx)]
        m = sub[~np.eye(len(idx), dtype=bool)]
        return float(np.nanmean(m)) if m.size else float("nan")

    split_corr = {}
    for sp in ("train", "test"):
        ls = [l for l in labels if meta[l]["split"] == sp]
        if ls:
            split_corr[sp] = corr_group(ls)

    summary = {
        "dataset": str(ds),
        "n_probes": n,
        "window": args.window,
        "sigma": args.sigma,
        "cross_corr_non_diag_mean": float(np.nanmean(offdiag)),
        "cross_corr_non_diag_min": float(np.nanmin(offdiag)),
        "cross_corr_non_diag_max": float(np.nanmax(offdiag)),
        "split_corr": split_corr,
        "family": fam_report,
        "heldout_static": holdout_report,
        "svd_rank1_frac": rank1_frac,
        "svd_top5_frac": float(sum(s[i] ** 2 for i in range(min(5, len(s)))) / energy_total)
        if energy_total > 0
        else float("nan"),
    }

    if args.json:
        out = {
            "summary": summary,
            "per_probe": {
                l: {
                    "split": meta[l]["split"],
                    "kind": meta[l]["kind"],
                    "explained_energy": expl[l],
                    "explained_energy_holdout_static": expl_holdout[l],
                    "cross_corr_mean_to_others": float(np.nanmean(
                        [corr[i, j] for j in range(n) if j != i])),
                }
                for i, l in enumerate(labels)
            },
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nper-probe explained energy (selected):")
    for l in labels:
        print(f"  {l:34s} split={meta[l]['split']:5s} kind={meta[l]['kind']:12s} "
              f"expl={expl[l]:.3f}")


if __name__ == "__main__":
    main()
