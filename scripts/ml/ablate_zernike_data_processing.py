"""Data-processing ablation for BOTH trainers in ``src/ml/zernike/``.

Question
--------
Does any dataset-level preprocessing improve **validation** performance of
``train_amp.py`` (physics ``ZernikeAmpModel``) or ``train_coeff.py`` (learned
``ZernikeCoeffConvNet``)?

Protocol (per AGENTS.md: never judge on MSE/PSNR/SSIM; pair on the axis you varied)
-----------------------------------------------------------------------------------
* The varied axis is **image_mode** -- the one data-processing knob both trainers
  expose, applied by the shared ``MaterialiserConfig``. Modes: abs255 / peak / robust.
  (``raw`` is excluded: it keeps physical ADU units, which no peak-based metric in
  these trainers is defined against.)
* **Paired on seed.** For a fixed seed both arms get the SAME weight-init stream and
  the SAME fold assignment (seed is consumed only by the split / init), so the seed's
  contribution cancels in the per-seed difference. Seeds vary; folds do not.
* Judged on **val R^2** only. MSE/PSNR/SSIM are recorded but never used to rank, because
  a sum-normalised variant makes them degenerate (measured elsewhere in this repo:
  PSNR 72 dB / SSIM 0.9996 with R^2 *worse* than a constant predictor).
* Statistics come from ``ml.zernike.eval_stats`` (the repo's single source): exact
  sign-flip paired test + Cohen's d_z. ``min_attainable_pvalue`` is reported too, so an
  underpowered protocol cannot masquerade as a null result.

Usage
-----
    python scripts/ml/ablate_zernike_data_processing.py --trainer amp
    python scripts/ml/ablate_zernike_data_processing.py --trainer coeff
    python scripts/ml/ablate_zernike_data_processing.py --trainer both --epochs 30

删除条件 (AGENTS.md 「scripts/ 目录」规则 3): 产物写到 ``logs/`` (默认
``logs/ablate_zernike_data_processing``, 不入库), 且 ``report/zernike_r2_baseline/
pod_ridge/report.md``「最终判决」表里 image_mode abs255/peak/robust 一行虽记录了
对应结论 (skill 全为负), 但该报告的 provenance 条目没有登记本脚本 —— 结论归属
``diagnose_pod_ridge.py`` / ``sweep_target_transform.py`` 等。当本消融被明确归属
到某 report/ 条目 (pod_ridge 报告或 issue)、且不再需要重跑两个 trainer 的
image_mode 配对对比时, 可删。
"""
from __future__ import annotations

import argparse
from loguru import logger
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: Modes under test. ``raw`` is deliberately excluded (physical ADU units).
IMAGE_MODES = ("abs255", "peak", "robust")
FAMILY = "slm_zernike_shaping"


def run_amp(image_mode: str, seed: int, epochs: int, out_root: Path) -> dict:
    """Train the physics forward model and return its best val metrics."""
    import torch

    from ml.zernike.train_amp import AmpTrainConfig, train

    torch.manual_seed(seed)
    cfg = AmpTrainConfig(
        families=(FAMILY,),
        seed=seed,
        epochs=epochs,
        use_wandb=False,
        save_checkpoint=False,
        image_every=0,
        image_mode=image_mode,
        out_dir=str(out_root / f"amp_{image_mode}_s{seed}"),
    )
    res = train(cfg)
    return {
        "trainer": "amp",
        "image_mode": image_mode,
        "seed": seed,
        "best_epoch": res.best_epoch,
        "val_r2": res.best_val_r2,
        "val_mse": res.best_val_mse,
        "val_psnr": res.best_val_psnr,
        "final_train_mse": res.final_train_mse,
        "n_modes": res.n_modes,
    }


def run_coeff(image_mode: str, seed: int, epochs: int, out_root: Path,
              fold: int = 8) -> dict:
    """Train the learned forward model on one fold and return its val metrics.

    ``image_mode`` is the varied axis. The corpus is the ZERNIKE-source subset (the
    same ``slm_zernike_shaping`` records the physics trainer sees, plus 192
    ``slm_pib_online`` ones), so both arms describe the same family.

    The default fold is 8, not 0: folds 0-7 are ``slm_pib_online`` pickles with only
    17-31 validation records each, while folds 8-17 are the 101-record
    ``slm_zernike_shaping`` pickles. Holding out a 101-record fold keeps this arm's
    statistical power comparable to the physics arm's.
    """
    import torch

    from ml.zernike.train_coeff import CoeffTrainConfig, train

    torch.manual_seed(seed)
    cfg = CoeffTrainConfig(
        image_mode=image_mode,
        seed=seed,
        epochs=epochs,
        # The fold protocol varies the DATA axis; here the varied axis is image_mode,
        # so a single fixed fold is used and pairing happens on the seed.
        protocol="file",
        fold=fold,
        use_wandb=False,
        save_checkpoint=False,
        out_dir=str(out_root / f"coeff_{image_mode}_s{seed}"),
    )
    res = train(cfg)
    # `CoeffTrainResult` exposes scalars directly (best_val_r2 / best_epoch) plus a
    # flat `final_metrics` dict whose val metrics are `val_`-prefixed. Both are read
    # here; the best-epoch R^2 is the selection metric, matching train_amp.
    fm = dict(getattr(res, "final_metrics", {}) or {})
    return {
        "trainer": "coeff",
        "image_mode": image_mode,
        "seed": seed,
        "fold": cfg.fold,
        "held_out": str(getattr(res, "held_out", "") or ""),
        "n_train": getattr(res, "n_train", None),
        "n_val": getattr(res, "n_val", None),
        "val_r2": res.best_val_r2,
        "val_r2_last": fm.get("val_r2"),
        "val_mse": fm.get("val_mse"),
        "val_psnr": fm.get("val_psnr"),
        "val_ssim": fm.get("val_ssim"),
    }


def _paired_report(rows: list[dict], baseline: str) -> dict:
    """Rank modes by val R^2 and test each against ``baseline`` on the seed axis."""
    import numpy as np

    from ml.zernike.eval_stats import (
        cohens_dz,
        min_attainable_pvalue,
        sign_flip_pvalue,
    )

    by_mode: dict[str, dict[int, float]] = {}
    for r in rows:
        by_mode.setdefault(r["image_mode"], {})[r["seed"]] = r["val_r2"]

    out: dict = {"baseline": baseline, "modes": {}, "ranking": []}
    for mode, per_seed in sorted(by_mode.items()):
        vals = [per_seed[s] for s in sorted(per_seed)]
        entry = {
            "n_seeds": len(vals),
            "val_r2_mean": float(np.mean(vals)),
            "val_r2_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "val_r2_per_seed": {str(s): v for s, v in sorted(per_seed.items())},
        }
        if mode != baseline:
            common = sorted(set(per_seed) & set(by_mode[baseline]))
            diffs = [per_seed[s] - by_mode[baseline][s] for s in common]
            if diffs:
                entry["paired_vs_baseline"] = {
                    "n_pairs": len(diffs),
                    "mean_diff": float(np.mean(diffs)),
                    "p_signflip": sign_flip_pvalue(diffs),
                    "cohens_dz": cohens_dz(diffs),
                    "min_attainable_p": min_attainable_pvalue(len(diffs)),
                    "per_seed_diff": {
                        str(s): d for s, d in zip(common, diffs, strict=True)
                    },
                }
        out["modes"][mode] = entry

    out["ranking"] = [
        m for m, _ in sorted(
            ((m, v["val_r2_mean"]) for m, v in out["modes"].items()),
            key=lambda t: -t[1],
        )
    ]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trainer", choices=["amp", "coeff", "both"], default="amp")
    ap.add_argument("--modes", nargs="*", default=list(IMAGE_MODES))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument(
        "--fold",
        type=int,
        default=8,
        help="CoeffTrainConfig fold. 8-17 are the 101-record slm_zernike_shaping "
             "pickles; 0-7 are 17-31-record slm_pib_online ones.",
    )
    ap.add_argument("--baseline", default="abs255")
    ap.add_argument("--out", default="logs/ablate_zernike_data_processing")
    args = ap.parse_args(argv)

    out_root = ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    trainers = ["amp", "coeff"] if args.trainer == "both" else [args.trainer]
    all_rows: list[dict] = []
    for t in trainers:
        fn = run_amp if t == "amp" else run_coeff
        for mode in args.modes:
            for seed in args.seeds:
                started = time.perf_counter()
                row = fn(mode, seed, args.epochs, out_root) if t == "amp" else fn(
                    mode, seed, args.epochs, out_root, args.fold
                )
                row["seconds"] = round(time.perf_counter() - started, 1)
                all_rows.append(row)
                mse_txt = ("n/a" if row["val_mse"] is None
                           else f"{row['val_mse']:.5f}")
                logger.info(
                    f"[{t}] mode={mode:7s} seed={seed} "
                    f"val_r2={row['val_r2']:+.4f} val_mse={mse_txt} "
                    f"({row['seconds']:.0f}s)",

                )

    report = {
        "design": {
            "varied_axis": "image_mode (dataset-level, via MaterialiserConfig)",
            "pairing": "paired on seed; fold held fixed",
            "judged_on": "val_r2 (never MSE/PSNR/SSIM)",
            "family": FAMILY,
            "seeds": args.seeds,
            "epochs": args.epochs,
            "modes": args.modes,
            "stats_source": "ml.zernike.eval_stats (sign_flip_pvalue, cohens_dz)",
        },
        "rows": all_rows,
        "reports": {t: _paired_report([r for r in all_rows if r["trainer"] == t],
                                      args.baseline) for t in trainers},
    }
    path = out_root / f"ablation_{'_'.join(trainers)}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info("\n=== summary (judged on val R^2) ===")
    for t in trainers:
        rep = report["reports"][t]
        logger.info(f"\n[{t}] baseline={rep['baseline']}  ranking: {' > '.join(rep['ranking'])}")
        for mode, e in sorted(rep["modes"].items(), key=lambda kv: -kv[1]["val_r2_mean"]):
            line = (f"  {mode:7s} R2={e['val_r2_mean']:+.4f} +/- {e['val_r2_sd']:.4f} "
                    f"(n={e['n_seeds']})")
            pv = e.get("paired_vs_baseline")
            if pv:
                line += (f" | vs base: d={pv['mean_diff']:+.4f} "
                         f"p={pv['p_signflip']:.4f} (min {pv['min_attainable_p']:.4f}) "
                         f"dz={pv['cohens_dz']:+.2f}")
            logger.info(line)
    logger.info(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())