# -*- coding: utf-8 -*-
"""残差预算：把 measured − predicted 的 CCD 残差拆成噪声 / 共享静态结构 / 剩余项。

这是"三条候选残差模型"日后对照的参考线。指标定义与
``scripts/slm/diagnose_static_floor.py`` 完全一致（同一 ``highpass`` / ``corr`` /
``rl2``、同一静态图构建与逐行最小二乘投影），本脚本不重拟合，只做分解：

  (a) 帧间噪声：每行 ``sample_*.npz`` 的 ``raw_frames`` ``(3, H, W)`` uint8，逐像素在
      3 帧上的标准差（ddof=1），按 fit 的归一化除以曝光（counts/ms），再过同一高通；
  (b) 共享静态结构：train measured 高通均值（``--flat`` 时改用
      ``_optical_probe_flat.npz`` 的高频）作模板，投影到该行残差上的分量；
  (c) 剩余项：残差减 (b)。投影是最小二乘，故 <剩余项, 静态图> = 0，
      ``static_rms**2 + leftover_rms**2 == residual_rms**2``（脚本内断言闭合）。

``corr_base`` / ``rl2_base`` 必须与 ``<dataset>/static_floor_diagnosis.json`` 中同一
行、同一 iteration 的值逐行一致（atol 1e-6）；对不上就直接失败，不调指标。

用法：
  python3 scripts/slm/transfer_residual_budget.py \\
    --dataset data/debug/slm_transfer/<run> --fit fit_m3_full \\
    --iteration 17 --out out/residual_budget.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 参考定义（唯一真源）：不要在本脚本里重写数值行为。
from scripts.slm.diagnose_static_floor import corr, highpass, rl2

CONSISTENCY_ATOL = 1e-6
CLOSURE_RTOL = 1e-9


def family(field: str) -> str:
    """把 "Z4 +0.10" 归为 zernike，smooth/bandlimited 归其族名（同参考脚本）。"""
    low = field.lower()
    if "smooth" in low:
        return "smooth"
    if "bandlimited" in low:
        return "bandlimited"
    return "zernike"


def build_static(
    args: argparse.Namespace,
    dataset: Path,
    measured: np.ndarray,
    train_mask: np.ndarray,
    camera_xy: np.ndarray,
) -> np.ndarray:
    """共享静态图，与 ``diagnose_static_floor.build`` 同一路径（含符号约定）。"""
    if args.flat:
        flat = np.load(dataset / "_optical_probe_flat.npz")
        frame = np.asarray(flat["frame"], dtype=float) / float(flat["exposure_ms"])
        patch = frame[camera_xy[:, :, 1], camera_xy[:, :, 0]]
        static = highpass(patch, args.sigma)
    else:
        train_hp = np.stack([highpass(measured[i], args.sigma)
                             for i in range(len(measured)) if train_mask[i]])
        static = train_hp.mean(axis=0)
    coeffs_train = [float((highpass(measured[i], args.sigma) * static).sum()
                          / (static * static).sum())
                    for i in range(len(measured)) if train_mask[i]]
    if float(np.mean(coeffs_train)) < 0:
        static = -static
    return static


def load_reference(path: Path) -> dict[int, dict]:
    if not path.exists():
        raise RuntimeError(
            f"缺少参考诊断 {path}；先跑 scripts/slm/diagnose_static_floor.py "
            f"（同一 --fit / --iteration）再运行本脚本")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["index"]): row for row in payload["rows"]}


def load_frames(payload_path: Path, exposure_ms: float) -> np.ndarray:
    """raw_frames → counts/ms：与 fit 的 ``count_rate`` 同一归一化。"""
    payload = np.load(payload_path)
    raw = np.asarray(payload["raw_frames"], dtype=float)
    if raw.ndim != 3:
        raise ValueError(f"{payload_path.name}: raw_frames 形状 {raw.shape}，应为 (frames, H, W)")
    if not np.isfinite(exposure_ms) or exposure_ms <= 0:
        raise ValueError(f"{payload_path.name}: 需要正的曝光，得到 {exposure_ms!r}")
    return raw / exposure_ms


def measured_window(payload_path: Path, camera_xy: np.ndarray,
                    exposure_ms: float) -> np.ndarray:
    """帧平均 + fit 窗提取（``camera_xy``）→ counts/ms；与 ``inputs.npz`` 的 measured 同源。"""
    rate = load_frames(payload_path, exposure_ms)
    return rate.mean(axis=0)[camera_xy[..., 1], camera_xy[..., 0]]


def noise_map(payload_path: Path, exposure_ms: float, sigma: float, ddof: int) -> np.ndarray:
    """帧间噪声图：raw_frames 逐像素 std(ddof) → counts/ms → 同一高通。"""
    rate = load_frames(payload_path, exposure_ms)
    return highpass(rate.std(axis=0, ddof=ddof), sigma)


def rms(image: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(image))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--fit", default="fit_m3_full")
    ap.add_argument("--iteration", type=int, default=None,
                    help="默认取 result.json 的 best iteration")
    ap.add_argument("--out", required=True, type=Path, help="输出 JSON 路径")
    ap.add_argument("--sigma", type=float, default=2.0, help="高斯高通 sigma（同参考脚本）")
    ap.add_argument("--flat", action="store_true",
                    help="用 _optical_probe_flat.npz 的高频作静态模板（同参考脚本）")
    ap.add_argument("--noise-ddof", type=int, default=1,
                    help="帧间 std 的自由度修正（默认 1，与 3 帧样本方差一致）")
    args = ap.parse_args()

    dataset: Path = args.dataset
    fitdir = dataset / args.fit
    result = json.loads((fitdir / "result.json").read_text(encoding="utf-8"))
    iteration = args.iteration if args.iteration is not None else int(result["iteration"])

    inputs = np.load(fitdir / "inputs.npz")
    measured = np.asarray(inputs["measured"], dtype=float)
    train_mask = np.asarray(inputs["train"], dtype=bool)
    camera_xy = np.asarray(inputs["camera_xy"], dtype=int)
    pred = np.load(fitdir / f"iteration_{iteration:04d}.npz")["prediction"].astype(float)
    if pred.shape != measured.shape:
        raise ValueError(f"prediction {pred.shape} 与 measured {measured.shape} 不一致")

    compare_path = dataset / "fit_compare_m3_z11_vs_z21.json"
    compare_rows = json.loads(compare_path.read_text(encoding="utf-8"))["rows"]
    compare_rows = sorted(compare_rows, key=lambda row: int(row["index"]))
    if len(compare_rows) != len(measured):
        raise ValueError(f"compare rows {len(compare_rows)} != measured {len(measured)}")
    for position, row in enumerate(compare_rows):
        if int(row["index"]) != position:
            raise ValueError(f"compare index 不连续：位置 {position} 是 {row['index']}")

    reference = load_reference(dataset / "static_floor_diagnosis.json")
    if len(reference) != len(measured):
        raise ValueError(f"参考诊断 rows {len(reference)} != measured {len(measured)}")

    static = build_static(args, dataset, measured, train_mask, camera_xy)

    rows: list[dict] = []
    deviations: list[float] = []
    for position, meta in enumerate(compare_rows):
        measured_hp = highpass(measured[position], args.sigma)
        residual = measured_hp - highpass(pred[position], args.sigma)

        coeff = float((residual * static).sum() / (static * static).sum())
        static_part = coeff * static
        leftover = residual - static_part

        noise = noise_map(dataset / meta["file"], float(meta["exposure_ms"]),
                          args.sigma, args.noise_ddof)
        noise_rms = rms(noise)
        residual_rms = rms(residual)
        static_rms = rms(static_part)
        leftover_rms = rms(leftover)
        closure = abs(static_rms ** 2 + leftover_rms ** 2 - residual_rms ** 2) / residual_rms ** 2
        if closure > CLOSURE_RTOL:
            raise RuntimeError(f"行 {position}: 能量闭合失败 {closure:.3e}")

        corr_base = corr(pred[position], measured[position])
        rl2_base = rl2(pred[position], measured[position])

        ref = reference[position]
        if str(ref["label"]) != str(meta["label"]):
            raise RuntimeError(
                f"行 {position}: label 不一致 参考 {ref['label']!r} vs compare {meta['label']!r}")
        for key, value in (("corr_base", corr_base), ("rl2_base", rl2_base)):
            deviation = abs(float(ref[key]) - value)
            deviations.append(deviation)
            if deviation > CONSISTENCY_ATOL:
                raise RuntimeError(
                    f"行 {position} ({meta['label']}): {key} 复现失败 "
                    f"参考 {ref[key]!r} vs 本脚本 {value!r}（差 {deviation:.3e} > "
                    f"{CONSISTENCY_ATOL:g}）。参考定义未变，检查 --iteration / --fit / "
                    f"inputs.npz 是否对应同一诊断。")

        rows.append({
            "index": position,
            "label": meta["label"],
            "split": meta["split"],
            "family": family(str(meta["family"])),
            "file": meta["file"],
            "exposure_ms": float(meta["exposure_ms"]),
            "noise_rms": noise_rms,
            "residual_rms": residual_rms,
            "noise_over_residual": noise_rms / residual_rms,
            "static_rms": static_rms,
            "static_over_residual": static_rms / residual_rms,
            "leftover_rms": leftover_rms,
            "leftover_over_residual": leftover_rms / residual_rms,
            "static_coeff": coeff,
            "corr_base": corr_base,
            "rl2_base": rl2_base,
        })

    mean_keys = ("noise_rms", "residual_rms", "noise_over_residual", "static_rms",
                 "static_over_residual", "leftover_rms", "leftover_over_residual",
                 "corr_base", "rl2_base")

    def aggregate(subset: list[dict]) -> dict:
        return {"n": len(subset), **{key: float(np.mean([r[key] for r in subset]))
                                     for key in mean_keys}}

    summary: dict[str, dict] = {"all": aggregate(rows)}
    for split in ("train", "test"):
        subset = [r for r in rows if r["split"] == split]
        if subset:
            summary[split] = aggregate(subset)
    for name in ("zernike", "smooth", "bandlimited"):
        subset = [r for r in rows if r["family"] == name]
        if subset:
            summary[name] = aggregate(subset)
    for split in ("train", "test"):
        for name in ("zernike", "smooth", "bandlimited"):
            subset = [r for r in rows if r["split"] == split and r["family"] == name]
            if subset:
                summary[f"{split}_{name}"] = aggregate(subset)
    # 噪声与残差都是 counts/ms：曝光不同的行不可直接比较（噪声 ∝ 1/曝光）。
    # 本数据集里 smooth/bandlimited 恰好都是 0.027 ms，只按族平均会把它读成噪声更大。
    for exposure in sorted({r["exposure_ms"] for r in rows}):
        subset = [r for r in rows if r["exposure_ms"] == exposure]
        summary[f"exposure_{exposure:g}ms"] = aggregate(subset)

    payload = {
        "dataset": str(dataset),
        "fit": args.fit,
        "iteration": iteration,
        "sigma": args.sigma,
        "noise_ddof": args.noise_ddof,
        "noise_normalization": "raw_frames / exposure_ms (counts/ms, same as fit count_rate)",
        "noise_scaling_note": ("noise_rms 与 residual_rms 都是 counts/ms，随 1/曝光放大；"
                              "跨曝光比较请看 exposure_<ms>ms 分组"),
        "static_source": "flat" if args.flat else "train_measured_mean",
        "static_norm": float(np.linalg.norm(static)),
        "consistency": {
            "reference": str(dataset / "static_floor_diagnosis.json"),
            "keys": ["corr_base", "rl2_base"],
            "atol": CONSISTENCY_ATOL,
            "n_rows": len(rows),
            "max_abs_deviation": float(max(deviations)),
            "passed": True,
        },
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("一致性通过：{} 行，corr_base/rl2_base 最大偏差 {:.3e}",
                len(rows), max(deviations))
    header = (f"{'group':<20}{'n':>4}{'noise':>9}{'resid':>9}{'n/r':>7}"
              f"{'static':>9}{'s/r':>7}{'left':>9}{'corr':>8}{'rl2':>8}")
    print(header)
    print("-" * len(header))
    for name, stats in summary.items():
        print(f"{name:<20}{stats['n']:>4}{stats['noise_rms']:>9.3f}"
              f"{stats['residual_rms']:>9.3f}{stats['noise_over_residual']:>7.3f}"
              f"{stats['static_rms']:>9.3f}{stats['static_over_residual']:>7.3f}"
              f"{stats['leftover_rms']:>9.3f}{stats['corr_base']:>8.4f}"
              f"{stats['rl2_base']:>8.4f}")
    print("-" * len(header))
    print(f"{'row':<20}{'split':>6}{'noise':>9}{'resid':>9}{'n/r':>7}"
          f"{'static':>9}{'s/r':>7}{'left':>9}{'corr':>8}{'rl2':>8}")
    for row in rows:
        print(f"{row['label'][:19]:<20}{row['split']:>6}{row['noise_rms']:>9.3f}"
              f"{row['residual_rms']:>9.3f}{row['noise_over_residual']:>7.3f}"
              f"{row['static_rms']:>9.3f}{row['static_over_residual']:>7.3f}"
              f"{row['leftover_rms']:>9.3f}{row['corr_base']:>8.4f}"
              f"{row['rl2_base']:>8.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
