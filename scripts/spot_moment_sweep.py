"""Does the anchored second-moment term reduce the forward model's error?

Paired across seeds, because the repo's documented noise floor makes a single run
meaningless: the *same* config gave val R2 between 0.780 and 0.923 depending only on the
split seed. A single-seed comparison here would invent a winner.

Protocol: one training run per (config, seed) on the real corpus, the incumbent included,
sharing split + budget + init. Reported as the paired per-seed difference against the
incumbent, because the fold/split difficulty then cancels instead of polluting the delta.

Ranked on R2 and on the moment gap. NEVER on MSE/PSNR/SSIM -- this repo has already been
burned by that: `normalization="sum"` once reported PSNR 72 dB and SSIM 0.9996 while its
R2 was worse than a constant predictor.

The incumbent's val curve plateaus by epoch 5 and then oscillates inside the noise band,
so the budget is raised well past convergence: extra epochs cannot help, but they make
the comparison fair by giving every config the same chance.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"D:\Projects\TIFO\AO-shaping")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from ml.zernike.train_amp import AmpTrainConfig, train  # noqa: E402
from ml.zernike.losses import LossConfig  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 60

# The incumbent, plus the new term at four weights. n_max is included because the repo's
# own (n_max, lr) sweep found it the dominant lever, and a moment term cannot be judged
# at one capacity.
CONFIGS: dict[str, dict] = {
    "incumbent (mse)":            {"w_spot_moment": 0.0, "n_max": 15},
    "moment 0.25":                {"w_spot_moment": 0.25, "n_max": 15},
    "moment 0.5":                 {"w_spot_moment": 0.5,  "n_max": 15},
    "moment 1.0":                 {"w_spot_moment": 1.0,  "n_max": 15},
    "moment 2.0":                 {"w_spot_moment": 2.0,  "n_max": 15},
    "incumbent n_max=20":         {"w_spot_moment": 0.0,  "n_max": 20},
    "moment 0.5 n_max=20":        {"w_spot_moment": 0.5,  "n_max": 20},
    "moment 1.0 n_max=20":        {"w_spot_moment": 1.0,  "n_max": 20},
}


def run_one(name: str, cfg_kwargs: dict, seed: int) -> dict:
    cfg = AmpTrainConfig(
        seed=seed,
        epochs=EPOCHS,
        n_max=cfg_kwargs["n_max"],
        loss="mse",
        loss_weights=LossConfig(w_mse=1.0, w_shape_gap=1.0,
                               w_spot_moment=cfg_kwargs["w_spot_moment"]),
        num_workers=0,
        use_wandb=False,
        out_dir=ROOT / "logs" / "moment_sweep" / f"{name.replace(' ', '_')}_s{seed}",
    )
    summary = train(cfg)
    gaps = [
        r["val_spot_moment_gap"]
        for r in summary.history
        if "val_spot_moment_gap" in r and r["val_spot_moment_gap"] == r["val_spot_moment_gap"]
    ]
    return {
        "config": name, "seed": seed, "n_max": cfg_kwargs["n_max"],
        "w_spot_moment": cfg_kwargs["w_spot_moment"],
        "val_r2": float(summary.best_val_r2),
        "val_mse": float(summary.best_val_mse),
        "val_psnr": float(summary.best_val_psnr),
        "coef_abs_max": float(max(abs(c) for c in summary.coefficients)),
        "dead_modes": int(summary.dead_modes),
        "val_moment_gap": float(min(gaps)) if gaps else float("nan"),
    }


def main() -> None:
    print("=" * 104)
    print("ANCHORED SECOND MOMENT vs FORWARD ERROR - paired over seeds, ranked on R2")
    print("=" * 104)
    print(f"seeds={SEEDS} epochs={EPOCHS} corpus=slm_zernike_shaping")
    print("Never rank on MSE/PSNR/SSIM: this repo measured a 72 dB PSNR with worse-than-constant R2.")

    rows: list[dict] = []
    for name, kwargs in CONFIGS.items():
        for seed in SEEDS:
            r = run_one(name, kwargs, seed)
            rows.append(r)
            print(f"  {name:<24} seed={seed}  val_r2={r['val_r2']:+.4f}  "
                  f"psnr={r['val_psnr']:5.2f}  |c|max={r['coef_abs_max']:.3f}")

    def agg(name: str, key: str) -> np.ndarray:
        return np.array([r[key] for r in rows if r["config"] == name])

    print(f"\n{'config':<24}{'R2 mean':>10}{'sd':>8}{'PSNR':>8}{'dR2 vs inc':>13}"
          f"{'paired sd':>11}{'pos':>7}{'moment gap':>12}")

    base = {seed: r["val_r2"] for seed, r in
            zip(SEEDS, [r for r in rows if r["config"] == "incumbent (mse)"])}
    table = []
    for name in CONFIGS:
        r2 = agg(name, "val_r2")
        deltas = np.array([r["val_r2"] - base[r["seed"]]
                           for r in rows if r["config"] == name])
        table.append({
            "config": name, "r2_mean": float(r2.mean()), "r2_sd": float(r2.std(ddof=1)),
            "psnr_mean": float(agg(name, "val_psnr").mean()),
            "d_r2": float(deltas.mean()), "d_r2_se": float(deltas.std(ddof=1) / np.sqrt(len(deltas))),
            "positives": int((deltas > 0).sum()),
        })
        t = table[-1]
        mg = agg(name, "val_moment_gap")
        mgm = float(np.nanmean(mg)) if np.isfinite(mg).any() else float("nan")
        t["moment_gap_mean"] = mgm
        print(f"{name:<24}{t['r2_mean']:>+10.4f}{t['r2_sd']:>8.4f}{t['psnr_mean']:>8.2f}"
              f"{t['d_r2']:>+13.4f}{t['d_r2_se']:>11.4f}{t['positives']:>4}/{len(deltas)}"
              f"{mgm:>12.4f}")

    best = max(table, key=lambda r: r["r2_mean"])
    print(f"\nbest mean R2: {best['config']} ({best['r2_mean']:+.4f})")
    incumbent = next(r for r in table if r["config"] == "incumbent (mse)")
    verdict = (
        "REDUCES the error" if best["d_r2"] > 0 and best["positives"] == len(SEEDS)
        else "does NOT reduce the error on this corpus"
    )
    print(f"verdict: the anchored moment term {verdict} "
          f"(best paired delta {best['d_r2']:+.4f} +/- {best['d_r2_se']:.4f}, "
          f"{best['positives']}/{len(SEEDS)})")
    print(f"incumbent reference: R2 {incumbent['r2_mean']:+.4f} +/- {incumbent['r2_sd']:.4f}")

    out = ROOT / "report" / "loss_defects" / "moment_sweep.json"
    out.write_text(json.dumps({"rows": rows, "table": table}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
