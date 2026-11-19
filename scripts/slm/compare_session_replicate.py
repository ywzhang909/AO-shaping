# -*- coding: utf-8 -*-
"""跨会话复现对照：同一批相位重拍一次，静态结构与观测算子还能不能复用。

输入是两个 run dir：original（有 fit 与 iteration 预测）和 replicate（同 seed 重拍）。
逐行按 ``label`` 配对，窗口 / 曝光归一化 / 高通全部沿用
``scripts/slm/transfer_residual_budget.py``（其内部又复用
``scripts/slm/diagnose_static_floor.py`` 的 ``highpass`` / ``corr`` / ``rl2``），
本脚本不新造指标。

五个诚实的检查：
  1. 相位同一性：配对行的 ``model_phase`` 必须 np.array_equal，否则直接停下；
  2. 跨会话图像差：两会话帧平均 + 同一 fit 窗 + 各自曝光归一化 + 高通，
     给出 xcorr 与相对 L2；
  3. 噪声参照：各自会话内的 3 帧 std，给出 cross_session_diff_rms / 会话内噪声；
     ``diff_over_expected``（除 sqrt(n_o**2+n_r**2)）才是"纯噪声下应 ≈1"的判据，
     ``diff_over_noise``（除两会话噪声均值）按字面口径一并给出；
  4. 静态图稳定性：original train 建图，投影到 original test 与 replicate 全行；
     再反过来用 replicate train 建图投影 original 行；
  5. 算子迁移：``fit_m3_full/iteration_XXXX`` 的预测（只在 original 上拟合）分别对
     original 与 replicate measured 打分，给 corr / rl2 及差值。

用法：
  python3 scripts/slm/compare_session_replicate.py \\
    --original data/debug/slm_transfer/<orig> \\
    --replicate data/debug/slm_transfer/<replicate> \\
    --fit fit_m3_full --iteration 17 --out out/replicate_compare.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.slm.diagnose_static_floor import corr, highpass, rl2
from scripts.slm.transfer_residual_budget import (
    build_static,
    family,
    load_reference,
    measured_window,
    noise_map,
    rms,
)

REFERENCE_ATOL = 1e-6


def manifest_index(dataset: Path) -> dict[str, dict]:
    rows = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    index: dict[str, dict] = {}
    for row in rows:
        label = str(row["label"])
        if label in index:
            raise ValueError(f"{dataset}: label {label!r} 在 manifest 中重复")
        index[label] = row
    return index


def phase_mismatches(pairs: list[tuple[dict, dict]], original: Path,
                     replicate: Path) -> list[dict]:
    bad: list[dict] = []
    for orig_row, repl_row in pairs:
        a = np.load(original / orig_row["file"])["model_phase"]
        b = np.load(replicate / repl_row["file"])["model_phase"]
        if not np.array_equal(a, b):
            bad.append({
                "label": orig_row["label"],
                "original_file": orig_row["file"],
                "replicate_file": repl_row["file"],
                "max_abs_diff": float(np.abs(np.asarray(a, dtype=float)
                                             - np.asarray(b, dtype=float)).max()),
            })
    return bad


def mean_metrics(rows: list[dict], keys: tuple[str, ...]) -> dict:
    return {"n": len(rows), **{key: float(np.mean([r[key] for r in rows])) for key in keys}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", required=True, type=Path, help="原始 run dir（含 fit）")
    ap.add_argument("--replicate", required=True, type=Path, help="同相位重拍 run dir")
    ap.add_argument("--fit", default="fit_m3_full", help="original 下的 fit 目录名")
    ap.add_argument("--iteration", type=int, default=None,
                    help="默认取 result.json 的 best iteration")
    ap.add_argument("--out", required=True, type=Path, help="输出 JSON 路径")
    ap.add_argument("--sigma", type=float, default=2.0, help="高斯高通 sigma（同参考脚本）")
    ap.add_argument("--noise-ddof", type=int, default=1,
                    help="帧间 std 的自由度修正（默认 1）")
    args = ap.parse_args()

    original: Path = args.original
    replicate: Path = args.replicate
    fitdir = original / args.fit
    result = json.loads((fitdir / "result.json").read_text(encoding="utf-8"))
    iteration = args.iteration if args.iteration is not None else int(result["iteration"])

    inputs = np.load(fitdir / "inputs.npz")
    measured = np.asarray(inputs["measured"], dtype=float)
    train_mask = np.asarray(inputs["train"], dtype=bool)
    camera_xy = np.asarray(inputs["camera_xy"], dtype=int)
    pred = np.load(fitdir / f"iteration_{iteration:04d}.npz")["prediction"].astype(float)

    compare_rows = sorted(json.loads((original / "fit_compare_m3_z11_vs_z21.json")
                                     .read_text(encoding="utf-8"))["rows"],
                          key=lambda row: int(row["index"]))
    if len(compare_rows) != len(measured):
        raise ValueError(f"compare rows {len(compare_rows)} != inputs.measured {len(measured)}")

    orig_index = manifest_index(original)
    repl_index = manifest_index(replicate)
    missing = [row["label"] for row in compare_rows if row["label"] not in repl_index]
    if missing:
        raise RuntimeError(f"replicate 缺少 {len(missing)} 个配对的 label：{missing[:10]}")
    pairs = [(orig_index[row["label"]], repl_index[row["label"]]) for row in compare_rows]

    bad = phase_mismatches(pairs, original, replicate)
    if bad:
        for item in bad[:10]:
            logger.error("相位不一致 {}: max |diff| = {:.3e}", item["label"], item["max_abs_diff"])
        raise RuntimeError(
            f"{len(bad)}/{len(pairs)} 行的 model_phase 与 original 不同；"
            f"replicate 没有复现同一批探针，后续比较无意义。")

    # 静态图：两个方向各建一张（同一构造、同一符号约定）。
    static_ns = SimpleNamespace(flat=False, sigma=args.sigma)
    repl_measured = np.stack([
        measured_window(replicate / repl_row["file"], camera_xy, float(repl_row["exposure_ms"]))
        for _, repl_row in pairs
    ])
    repl_train_mask = np.asarray([row["split"] == "train" for row in compare_rows], dtype=bool)
    static_orig = build_static(static_ns, original, measured, train_mask, camera_xy)
    static_repl = build_static(static_ns, replicate, repl_measured, repl_train_mask, camera_xy)

    reference = None
    ref_path = original / "static_floor_diagnosis.json"
    if ref_path.exists():
        reference = load_reference(ref_path)

    rows: list[dict] = []
    max_window_deviation = 0.0
    max_reference_deviation = 0.0
    for position, ((orig_row, repl_row), meta) in enumerate(zip(pairs, compare_rows)):
        exp_orig = float(orig_row["exposure_ms"])
        exp_repl = float(repl_row["exposure_ms"])
        orig_rate = measured_window(original / orig_row["file"], camera_xy, exp_orig)
        repl_rate = measured_window(replicate / repl_row["file"], camera_xy, exp_repl)

        deviation = float(np.abs(orig_rate - measured[position]).max())
        max_window_deviation = max(max_window_deviation, deviation)
        if deviation > 1e-9 * max(1.0, float(np.abs(measured[position]).max())):
            raise RuntimeError(
                f"行 {position} ({meta['label']}): 从 raw_frames 重建的 measured 与 "
                f"inputs.npz 不符（最大差 {deviation:.3e}）；窗口或曝光逻辑已偏")

        orig_hp = highpass(orig_rate, args.sigma)
        repl_hp = highpass(repl_rate, args.sigma)
        diff = orig_hp - repl_hp

        noise_orig_rms = rms(noise_map(original / orig_row["file"], exp_orig,
                                       args.sigma, args.noise_ddof))
        noise_repl_rms = rms(noise_map(replicate / repl_row["file"], exp_repl,
                                       args.sigma, args.noise_ddof))
        noise_rms = 0.5 * (noise_orig_rms + noise_repl_rms)
        expected_diff_rms = float(np.hypot(noise_orig_rms, noise_repl_rms))
        diff_rms = rms(diff)

        def static_ratio(residual: np.ndarray, static: np.ndarray) -> tuple[float, float]:
            """同一口径：coeff 投影后 static_rms / residual_rms（denominator 是残差，不是原图）。"""
            coeff = float((residual * static).sum() / (static * static).sum())
            static_rms = rms(coeff * static)
            return float(static_rms / rms(residual)), coeff

        pred_hp = highpass(pred[position], args.sigma)
        res_orig = orig_hp - pred_hp
        res_repl = repl_hp - pred_hp

        ratio_orig_map_orig, coeff_orig_map_orig = static_ratio(res_orig, static_orig)
        ratio_orig_map_repl, coeff_orig_map_repl = static_ratio(res_repl, static_orig)
        ratio_repl_map_orig, coeff_repl_map_orig = static_ratio(res_orig, static_repl)
        ratio_repl_map_repl, coeff_repl_map_repl = static_ratio(res_repl, static_repl)

        corr_orig = corr(pred[position], orig_rate)
        rl2_orig = rl2(pred[position], orig_rate)
        corr_repl = corr(pred[position], repl_rate)
        rl2_repl = rl2(pred[position], repl_rate)
        # 诊断（不改判定口径）：rl2 对幅度敏感。给原预测乘一个最佳标量再打分，
        # 可区分"算子形状没迁移"与"只是整幅增益换了"。增益用 LS：<pred, meas>/<pred, pred>。
        pred_energy = float((pred[position] * pred[position]).sum())
        gain_orig = float((pred[position] * orig_rate).sum() / pred_energy)
        gain_repl = float((pred[position] * repl_rate).sum() / pred_energy)
        rl2_repl_gain_corrected = rl2(gain_repl * pred[position], repl_rate)

        if reference is not None:
            ref = reference[position]
            for key, value in (("corr_base", corr_orig), ("rl2_base", rl2_orig)):
                deviation = abs(float(ref[key]) - value)
                max_reference_deviation = max(max_reference_deviation, deviation)
                if deviation > REFERENCE_ATOL:
                    raise RuntimeError(
                        f"行 {position} ({meta['label']}): original 侧 {key} 与 "
                        f"static_floor_diagnosis.json 不符（{ref[key]!r} vs {value!r}）")

        orig_shift = np.asarray(orig_row["spot_xy"], dtype=float)
        repl_shift = np.asarray(repl_row["spot_xy"], dtype=float)
        shift = repl_shift - orig_shift

        rows.append({
            "index": position,
            "label": meta["label"],
            "split": meta["split"],
            "family": family(str(meta["family"])),
            "original_file": orig_row["file"],
            "replicate_file": repl_row["file"],
            "exposure_ms_original": exp_orig,
            "exposure_ms_replicate": exp_repl,
            "spot_shift_px": [float(shift[0]), float(shift[1])],
            "spot_shift_magnitude_px": float(np.hypot(*shift)),
            "xcorr_orig_repl": corr(orig_hp, repl_hp),
            "rl2_orig_repl": rl2(repl_hp, orig_hp),
            "noise_orig_rms": noise_orig_rms,
            "noise_repl_rms": noise_repl_rms,
            "noise_rms": noise_rms,
            "expected_diff_rms": expected_diff_rms,
            "diff_rms": diff_rms,
            "diff_over_noise": diff_rms / noise_rms,
            "diff_over_expected": diff_rms / expected_diff_rms,
            "static_orig_map_on_original": ratio_orig_map_orig,
            "static_orig_map_on_replicate": ratio_orig_map_repl,
            "static_repl_map_on_original": ratio_repl_map_orig,
            "static_repl_map_on_replicate": ratio_repl_map_repl,
            "static_coeff_orig_map_on_original": coeff_orig_map_orig,
            "static_coeff_orig_map_on_replicate": coeff_orig_map_repl,
            "static_coeff_repl_map_on_original": coeff_repl_map_orig,
            "static_coeff_repl_map_on_replicate": coeff_repl_map_repl,
            "corr_orig": corr_orig,
            "rl2_orig": rl2_orig,
            "corr_repl": corr_repl,
            "rl2_repl": rl2_repl,
            "dcorr": corr_repl - corr_orig,
            "drl2": rl2_repl - rl2_orig,
            "operator_gain_orig": gain_orig,
            "operator_gain_repl": gain_repl,
            "operator_gain_ratio": gain_repl / gain_orig,
            "rl2_repl_gain_corrected": rl2_repl_gain_corrected,
        })

    mean_keys = ("xcorr_orig_repl", "rl2_orig_repl", "noise_orig_rms", "noise_repl_rms",
                 "noise_rms", "expected_diff_rms", "diff_rms", "diff_over_noise",
                 "diff_over_expected", "static_orig_map_on_original",
                 "static_orig_map_on_replicate", "static_repl_map_on_original",
                 "static_repl_map_on_replicate", "corr_orig", "rl2_orig", "corr_repl",
                 "rl2_repl", "dcorr", "drl2", "operator_gain_orig", "operator_gain_repl",
                 "operator_gain_ratio", "rl2_repl_gain_corrected")

    summary: dict[str, dict] = {"all": mean_metrics(rows, mean_keys)}
    for split in ("train", "test"):
        subset = [r for r in rows if r["split"] == split]
        if subset:
            summary[split] = mean_metrics(subset, mean_keys)
    for name in ("zernike", "smooth", "bandlimited"):
        subset = [r for r in rows if r["family"] == name]
        if subset:
            summary[name] = mean_metrics(subset, mean_keys)
    for split in ("train", "test"):
        for name in ("zernike", "smooth", "bandlimited"):
            subset = [r for r in rows if r["split"] == split and r["family"] == name]
            if subset:
                summary[f"{split}_{name}"] = mean_metrics(subset, mean_keys)

    stability = {}
    cells = {
        ("original_map", "original"): "static_orig_map_on_original",
        ("original_map", "replicate"): "static_orig_map_on_replicate",
        ("replicate_map", "original"): "static_repl_map_on_original",
        ("replicate_map", "replicate"): "static_repl_map_on_replicate",
    }
    for (map_name, session), key in cells.items():
        for split in ("all", "train", "test"):
            subset = rows if split == "all" else [r for r in rows if r["split"] == split]
            if subset:
                stability[f"{map_name}_on_{session}_{split}"] = {
                    "n": len(subset),
                    "static_over_residual": float(np.mean([r[key] for r in subset])),
                }

    shifts = np.asarray([r["spot_shift_px"] for r in rows], dtype=float)
    payload = {
        "original": str(original),
        "replicate": str(replicate),
        "fit": args.fit,
        "iteration": iteration,
        "sigma": args.sigma,
        "noise_ddof": args.noise_ddof,
        "window": "original fit inputs.npz camera_xy (161x161, step 2 px)",
        "metric_source": ("scripts/slm/diagnose_static_floor.highpass/corr/rl2 via "
                          "scripts/slm/transfer_residual_budget helpers"),
        "static_source": "train-split measured high-pass mean (same-sign convention)",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "consistency": {
            "n_matched": len(rows),
            "phases_equal": True,
            "n_phase_mismatch": 0,
            "phase_check": "np.array_equal(model_phase_original, model_phase_replicate)",
            "max_window_reconstruction_deviation": float(max_window_deviation),
            "reference_corr_rl2_check": (
                {"reference": str(ref_path), "atol": REFERENCE_ATOL,
                 "max_abs_deviation": float(max_reference_deviation), "passed": True}
                if reference is not None else "absent"),
            "spot_shift_px_mean": [float(v) for v in shifts.mean(axis=0)],
            "spot_shift_magnitude_px_mean": float(np.hypot(shifts[:, 0], shifts[:, 1]).mean()),
            "spot_shift_magnitude_px_max": float(np.hypot(shifts[:, 0], shifts[:, 1]).max()),
        },
        "summary": summary,
        "static_stability": stability,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    keys = ("xcorr_orig_repl", "rl2_orig_repl", "diff_over_noise", "diff_over_expected",
            "static_orig_map_on_replicate", "static_repl_map_on_original", "corr_orig",
            "corr_repl", "dcorr", "rl2_orig", "rl2_repl", "drl2")
    header = f"{'group':<18}{'n':>4}" + "".join(f"{k[:12]:>13}" for k in keys)
    print(header)
    print("-" * len(header))
    for name, stats in summary.items():
        print(f"{name:<18}{stats['n']:>4}"
              + "".join(f"{stats[k]:>13.4f}" for k in keys))
    print("-" * len(header))
    print("static map stability (mean static_rms / residual_rms)")
    for name, stats in stability.items():
        print(f"  {name:<40}n={stats['n']:>3}  ratio={stats['static_over_residual']:.4f}")
    overall = summary["all"]
    print(f"\noperator gain (diagnostic, LS scalar on pred): "
          f"orig {overall['operator_gain_orig']:.4f}  repl {overall['operator_gain_repl']:.4f}  "
          f"ratio {overall['operator_gain_ratio']:.4f}  |  "
          f"rl2_repl raw {overall['rl2_repl']:.4f} -> gain-corrected "
          f"{overall['rl2_repl_gain_corrected']:.4f}")
    print("\nspot shift (replicate - original): mean "
          f"({shifts[:, 0].mean():+.2f}, {shifts[:, 1].mean():+.2f}) px, "
          f"|shift| mean {np.hypot(shifts[:, 0], shifts[:, 1]).mean():.2f} px, "
          f"max {np.hypot(shifts[:, 0], shifts[:, 1]).max():.2f} px")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
