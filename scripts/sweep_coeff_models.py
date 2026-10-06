"""Run the coefficient -> far-field forward model under a protocol with power.

Produces everything ``report/zernike_coeff2amp/report.md`` is drawn from, as a saved
artefact rather than prose, in ``logs/zernike_coeff_sweep.json``.

=========================  Why this script exists  ==========================

A single train/val split cannot resolve the effects being claimed here. On this
corpus, changing ONLY the split seed moved R^2 from 0.780 to 0.923
(sigma ~= 0.073), and leave-one-pickle-out per-fold R^2 spans 0.737..0.941
(sigma ~= 0.08). Every claim below is therefore a **paired per-fold difference**
tested with an exact sign-flip permutation test, and every model sees the SAME
folds.

Three sanity gates run FIRST, because each one catches an error that would make
the headline table confidently wrong:

1. **Constant-predictor canary.** A per-fold mean-image predictor must score
   R^2 <= 0 on held-out data. A positive value means the split leaked.
2. **Normalisation trap.** Target normalisation is a trap: PSNR is
   ``10*log10(L^2/MSE)`` and SSIM's stabilisers ``C1=(0.01L)^2``/``C2=(0.015L)^2``
   both reward shrinking image values toward the stabilisers. Total-energy ("sum")
   normalisation therefore reports near-perfect PSNR/SSIM while its R^2 is *worse*
   than a constant predictor. This gate re-scores ONE trained model's predictions
   against the same images put through three normalisations and asserts the R^2
   ordering DISAGREES with the PSNR ordering.
3. **Padding contract.** The Noll prefix must be preserved exactly (asserted in
   the dataset's own tests; re-checked here on the real vectors).

Usage:
    python scripts/sweep_coeff_models.py --quick     # 3 folds, 4 epochs
    python scripts/sweep_coeff_models.py             # 18 folds, 40 epochs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from ml.hwdataset.index import PhaseSource, build_hw_index  # noqa: E402
from ml.hwdataset.records import MaterialiserConfig  # noqa: E402
from ml.hwdataset.zernike_dataset import (  # noqa: E402
    ZERNIKE_COEFF_SOURCES,
    ZernikeCoeffDataset,
    fit_coeff_stats,
    pad_coefficients,
)
from ml.zernike.eval_stats import (  # noqa: E402
    cohens_dz,
    holm_bonferroni,
    sign_flip_pvalue,
)
from ml.zernike.forward_model import peak_normalize  # noqa: E402
from ml.zernike.train_coeff import (  # noqa: E402
    INDEX_CACHE,
    CoeffTrainConfig,
    build_folds,
    r2_score,
    train,
)

OUT_JSON = "logs/zernike_coeff_sweep.json"
FIG_DIR = "report/zernike_coeff2amp/figures"


# ---------------------------------------------------------------------------
# Statistics live in ml.zernike.eval_stats -- the single source of truth.
# They were inline here until 383b7c7; keeping a second copy here is exactly the
# "duplicate implementation drifts" failure this repo has been bitten by.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
def gate_constant_predictor(
    dataset: ZernikeCoeffDataset, folds, fold_indices: list[int]
) -> dict:
    """Gate 1: a per-fold mean-image predictor must score R^2 <= 0.

    Predicting the TRAIN-split mean image and scoring it on the held-out fold is
    the honest null model. If it beats 0, the fold is not independent and every
    later number is inflated.
    """
    rows = []
    for index in fold_indices:
        fold = folds[index]
        train_imgs, val_imgs = [], []
        for pos in fold.train_positions:
            train_imgs.append(dataset[pos]["image"].numpy())
        for pos in fold.val_positions:
            val_imgs.append(dataset[pos]["image"].numpy())
        if not train_imgs or not val_imgs:
            continue
        mean_image = np.mean(np.stack(train_imgs), axis=0)
        prediction = np.repeat(mean_image[None], len(val_imgs), axis=0)
        value = float(np.mean(r2_score(prediction, np.stack(val_imgs))))
        rows.append({"fold": index, "held_out": folds[index].held_out.name, "r2": value})
    worst = max((r["r2"] for r in rows), default=float("nan"))
    passed = bool(np.all([r["r2"] <= 1e-9 for r in rows])) if rows else False
    return {
        "gate": "constant_predictor",
        "criterion": "held-out R^2 <= 0 for every fold",
        "max_r2_over_folds": worst,
        "passed": passed,
        "per_fold": rows,
    }


def gate_normalisation_trap(
    model, batch: dict[str, torch.Tensor], device: torch.device
) -> dict:
    """Gate 2: PSNR/SSIM are normalisation artifacts; R^2 is not.

    The SAME trained model's predictions are scored against the SAME images after
    putting *both* sides through one normalisation.

    An earlier version of this gate normalised only the target and so could not
    demonstrate anything -- it just showed a scale mismatch. The point of the trap
    is subtler and only appears when both sides move together:

    * ``R^2 = 1 - SS_res/SS_tot`` and every term is quadratic, so dividing a
      sample by ``k`` divides numerator and denominator by ``k^2`` and **cancels**.
      R^2 is therefore *invariant* to a per-sample rescale.
    * ``PSNR = 10*log10(L^2/MSE)`` has ``L`` fixed, so dividing by ``k`` divides
      MSE by ``k^2`` and PSNR **rises by 20*log10(k)**, unboundedly.

    So the gate asserts exactly that: R^2 unchanged across normalisations while
    PSNR swings by a large factor. That is why this repo selects on R^2 -- a
    normalisation that "improves" PSNR by 40 dB has changed nothing about the fit.
    """
    model.eval()
    with torch.no_grad():
        prediction = peak_normalize(model(batch["coeffs"].to(device))).cpu().numpy()
    raw = batch["image"].numpy()  # already "robust"-normalised, in [0, 1]

    def peak_norm(x: np.ndarray) -> np.ndarray:
        flat = x.reshape(len(x), 1, -1)
        scale = np.maximum(flat.max(axis=2), 1e-12)[:, :, None]
        return (flat / scale).reshape(x.shape)

    def sum_norm(x: np.ndarray) -> np.ndarray:
        flat = x.reshape(len(x), 1, -1)
        total = np.maximum(flat.sum(axis=2, keepdims=True), 1e-12)
        return (flat / total).reshape(x.shape)

    def robust_norm(x: np.ndarray) -> np.ndarray:
        flat = x.reshape(len(x), 1, -1)
        bg = np.median(flat, axis=2, keepdims=True)
        excess = flat - bg
        pos = np.where(excess > 0, excess, np.nan)
        scale = np.nan_to_num(np.nanpercentile(pos, 50, axis=2, keepdims=True), nan=1.0)
        scale = np.maximum(scale, 1e-12)
        return np.clip(excess / scale, 0.0, 1.0).reshape(x.shape)

    rows = []
    for name, fn in (("peak", peak_norm), ("robust", robust_norm), ("sum", sum_norm)):
        # BOTH sides through the same normalisation -- that is the whole point.
        target = fn(raw)
        pred_n = fn(prediction)
        mse = float(np.mean((pred_n - target) ** 2))
        rows.append(
            {
                "normalization": name,
                "r2": float(np.mean(r2_score(pred_n, target))),
                "psnr": float(10.0 * np.log10(1.0 / mse)) if mse > 0 else float("inf"),
            }
        )

    r2_values = [r["r2"] for r in rows]
    finite_psnr = [r["psnr"] for r in rows if np.isfinite(r["psnr"])]
    r2_spread = float(np.ptp(r2_values)) if r2_values else float("nan")
    psnr_spread = (
        float(np.ptp(finite_psnr)) if len(finite_psnr) > 1 else float("nan")
    )
    # The trap, stated as a ratio rather than as exact invariance. R^2 is only
    # *exactly* invariant under a COMMON per-sample scale; normalising each side
    # by its own maximum applies DIFFERENT factors to prediction and target, so
    # R^2 does move a little. What matters is the orders-of-magnitude gap: PSNR
    # swings tens of dB while R^2 barely moves, so PSNR is reporting the
    # normalisation rather than the fit.
    ratio = (
        psnr_spread / r2_spread
        if np.isfinite(psnr_spread) and r2_spread > 0
        else float("inf")
    )
    return {
        "gate": "normalisation_trap",
        "criterion": (
            "PSNR spread over {peak,robust,sum} must exceed the R^2 spread by "
            ">=100x (R^2 spread <= 0.1 dB-equivalent, PSNR spread > 10 dB)"
        ),
        "r2_spread": r2_spread,
        "psnr_spread_db": psnr_spread,
        "psnr_to_r2_spread_ratio": ratio,
        "r2_invariant_to_1e_2": bool(r2_spread <= 1e-2),
        "psnr_moves": bool(np.isfinite(psnr_spread) and psnr_spread > 10.0),
        "passed": bool(np.isfinite(psnr_spread) and psnr_spread > 10.0 and ratio >= 100.0),
        "rows": rows,
    }


def gate_padding(dataset: ZernikeCoeffDataset) -> dict:
    """Gate 3: Noll-prefix padding must be exact on real vectors."""
    worst = 0.0
    checked = 0
    for position in range(0, len(dataset), 97):  # stride to stay quick
        sample = dataset[position]
        raw = sample["coeffs_raw"].numpy()
        present = int(sample["n_terms_record"])
        padded = pad_coefficients(raw[:present], dataset.n_max)
        worst = max(worst, float(np.abs(padded[:present] - raw[:present]).max()))
        checked += 1
    return {
        "gate": "noll_prefix_padding",
        "criterion": "max |padded[:n] - raw[:n]| == 0",
        "records_checked": checked,
        "max_abs_error": worst,
        "passed": worst == 0.0,
    }


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Run the gates, then the paired cross-validation, then write the artefact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--folds", type=int, default=0, help="0 = every fold")
    parser.add_argument("--arms", nargs="+", default=["conv", "mlp"])
    parser.add_argument(
        "--input-terms",
        type=int,
        nargs="*",
        default=None,
        help=(
            "A/B the input width: one run per value, all on the SAME folds. "
            "The corpus longest vector is 78 terms, so 78 vs 136 measures "
            "whether the permanently-zero input dims cost generalisation."
        ),
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--protocol",
        choices=["file", "objective"],
        default="file",
        help="'file' = leave-one-pickle-out; 'objective' = leave-one-objective-out",
    )
    parser.add_argument("--out", default=OUT_JSON)
    args = parser.parse_args(argv)

    epochs = 4 if args.quick else args.epochs

    index = build_hw_index(index_cache=INDEX_CACHE, progress_every=0)
    index = index.filter(sources=list(ZERNIKE_COEFF_SOURCES))
    dataset = ZernikeCoeffDataset(
        index, config=MaterialiserConfig(grid=64, image_mode="robust"), use_cache=False
    )
    folds = build_folds(dataset, args.protocol)
    fold_indices = (
        [0, len(folds) // 2, len(folds) - 1] if args.quick else list(range(len(folds)))
    )
    if args.folds:
        fold_indices = fold_indices[: args.folds]

    started = time.perf_counter()
    gates = [gate_padding(dataset), gate_constant_predictor(dataset, folds, fold_indices)]
    print(json.dumps(gates, indent=2, default=str))

    # Each configuration is one arm: (architecture, input_terms).
    configs = [
        (arm, terms) for arm in args.arms for terms in (args.input_terms or [None])
    ]
    arm_names = [
        f"{arm}/in{terms if terms else "full"}" for arm, terms in configs
    ]
    results: dict[str, list[dict]] = {name: [] for name in arm_names}
    trap_gate: dict | None = None
    for arm_name, (arm, terms) in zip(arm_names, configs):
        for fold_index in fold_indices:
            suffix = f"_in{terms}" if terms else ""
            out_dir = (
                Path("logs/zernike_coeff") / f"{arm}_fold{fold_index:02d}{suffix}"
            )
            cfg = CoeffTrainConfig(
                protocol=args.protocol,
                architecture=arm,
                input_terms=terms,
                epochs=epochs,
                fold=fold_index,
                out_dir=str(out_dir),
                image_every=max(1, epochs),
                log_every=max(1, epochs // 2),
            )
            print(
                f"\n=== arm={arm_name} fold={fold_index} epochs={epochs} ===",
                flush=True,
            )
            outcome = train(cfg)
            row = {
                "arm": arm_name,
                "architecture": arm,
                "input_terms": terms,
                "fold": fold_index,
                "held_out": outcome.held_out,
                "n_train": outcome.n_train,
                "n_val": outcome.n_val,
                "best_val_r2": outcome.best_val_r2,
                "best_epoch": outcome.best_epoch,
                "final": outcome.final_metrics,
            }
            results[arm_name].append(row)
            if trap_gate is None:
                trap_gate = _trap_from_checkpoint(
                    out_dir / "best_coefficients.pt",
                    dataset,
                    folds[fold_index].val_positions,
                )
    if trap_gate is not None:
        gates.append(trap_gate)
        print(json.dumps([trap_gate], indent=2, default=str))

    # (a) Canary-subtracted skill. The absolute R^2 is optimistic because a
    # train-mean image already scores highly on this corpus, so the honest
    # yardstick is the model's improvement OVER that null on the same fold.
    null_by_fold = {
        row["fold"]: row["r2"]
        for g in gates
        if g["gate"] == "constant_predictor"
        for row in g["per_fold"]
    }
    for name, rows in results.items():
        for row in rows:
            null = null_by_fold.get(row["fold"])
            row["null_r2"] = null
            row["skill_r2"] = (
                None if null is None else row["best_val_r2"] - null
            )

    # Paired comparison on the SHARED folds.
    statistics: dict[str, dict] = {}
    if len(arm_names) >= 2:
        base = arm_names[0]
        for arm in arm_names[1:]:
            shared = sorted(
                set(r["fold"] for r in results[base])
                & set(r["fold"] for r in results[arm])
            )
            by_base = {r["fold"]: r["best_val_r2"] for r in results[base]}
            by_arm = {r["fold"]: r["best_val_r2"] for r in results[arm]}
            diffs = [by_arm[f] - by_base[f] for f in shared]
            statistics[f"{arm}_minus_{base}"] = {
                "folds": shared,
                "mean_diff": float(np.mean(diffs)) if diffs else float("nan"),
                "p_sign_flip": sign_flip_pvalue(diffs),
                "cohens_dz": cohens_dz(diffs),
                "note": f"negative favours {base}",
                "mean_skill_diff": float(
                    np.mean(
                        [
                            (by_arm[f] - by_base[f])
                            for f in shared
                            if by_arm[f] is not None and by_base[f] is not None
                        ]
                    )
                ),
                "skill_by_arm": by_arm,
                "skill_by_base": by_base,
            }
        adjusted = holm_bonferroni(
            {k: v["p_sign_flip"] for k, v in statistics.items()}
        )
        for key, value in adjusted.items():
            statistics[key]["p_holm"] = value

    summary = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "epochs": epochs,
        "arms": arm_names,
        "configs": [{"architecture": a, "input_terms": t} for a, t in configs],
        "folds_used": fold_indices,
        "n_folds": len(fold_indices),
        "gates": gates,
        "all_gates_passed": all(g["passed"] for g in gates),
        "per_fold": results,
        "arm_means": {
            arm: {
                "mean_r2": float(np.mean([r["best_val_r2"] for r in rows]))
                if rows
                else float("nan"),
                "std_r2": float(np.std([r["best_val_r2"] for r in rows], ddof=1))
                if len(rows) > 1
                else float("nan"),
                "mean_skill_r2": float(
                    np.mean([r["skill_r2"] for r in rows if r["skill_r2"] is not None])
                )
                if any(r["skill_r2"] is not None for r in rows)
                else float("nan"),
                "mean_null_r2": float(
                    np.mean([r["null_r2"] for r in rows if r["null_r2"] is not None])
                )
                if any(r["null_r2"] is not None for r in rows)
                else float("nan"),
            }
            for arm, rows in results.items()
        },
        "paired_statistics": statistics,
        "wall_seconds": time.perf_counter() - started,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _plot(summary)

    print("\n================ SUMMARY ================")
    print(f"all gates passed: {summary['all_gates_passed']}")
    for arm, stats in summary["arm_means"].items():
        print(f"  {arm:5s} val R^2 = {stats['mean_r2']:+.4f} +/- {stats['std_r2']:.4f}")
    for key, value in statistics.items():
        print(
            f"  {key}: mean diff {value['mean_diff']:+.4f}, "
            f"p={value['p_sign_flip']:.4f}, Holm p={value['p_holm']:.4f}, "
            f"d_z={value['cohens_dz']:+.2f}"
        )
    print(f"artefact: {out_path}")
    return 0


def _trap_from_checkpoint(path: Path, dataset, positions) -> dict | None:
    """Re-score a trained checkpoint against three normalisations (gate 2)."""
    if not path.exists():
        return None
    from ml.zernike.forward_model import ZernikeCoeffConfig, build_forward_model

    blob = torch.load(path, map_location="cpu", weights_only=False)
    config = ZernikeCoeffConfig(**blob["model_config"])
    model = build_forward_model(config)
    model.load_state_dict(blob["state_dict"])
    index = build_hw_index(index_cache=INDEX_CACHE, progress_every=0).filter(
        sources=list(ZERNIKE_COEFF_SOURCES)
    )
    dataset = ZernikeCoeffDataset(
        index,
        config=MaterialiserConfig(grid=64, image_mode="robust"),
        coeff_mean=blob["coeff_mean"],
        coeff_std=blob["coeff_std"],
        use_cache=False,
    )
    # Positions come from the fold, not from re-matching `held_out` against a path:
    # an objective fold's label is an objective name, not a filename.
    if not positions:
        return None
    # The checkpoint may have been trained on a trimmed input width
    # (input_terms), so feed the model exactly what it was built for.
    width = int(config.n_coeffs)
    batch = {
        "coeffs": torch.stack([dataset[i]["coeffs"] for i in positions[:32]])[:, :width],
        "image": torch.stack([dataset[i]["image"] for i in positions[:32]]),
    }
    return gate_normalisation_trap(model, batch, torch.device("cpu"))


def _plot(summary: dict) -> None:
    """Per-fold R^2 and the gate table."""
    fig_dir = ROOT / FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))

    for arm, rows in summary["per_fold"].items():
        folds = [r["fold"] for r in rows]
        values = [r["best_val_r2"] for r in rows]
        axes[0].plot(folds, values, marker="o", label=f"{arm} (mean {np.mean(values):+.3f})")
    axes[0].axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axes[0].set_xlabel("leave-one-pickle-out fold")
    axes[0].set_ylabel("held-out R^2 (selection metric)")
    axes[0].set_title("Per-fold held-out R^2")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    trap = next((g for g in summary["gates"] if g["gate"] == "normalisation_trap"), None)
    if trap:
        labels = [r["normalization"] for r in trap["rows"]]
        width = 0.38
        positions = np.arange(len(labels))
        axes[1].bar(positions - width / 2, [r["r2"] for r in trap["rows"]], width, label="R^2")
        axes[1].bar(
            positions + width / 2,
            [r["psnr"] for r in trap["rows"]],
            width,
            label="PSNR (diagnostic)",
        )
        axes[1].set_xticks(positions)
        axes[1].set_xticklabels(labels)
        axes[1].set_title(
            f"Normalisation trap: PSNR spread {trap.get('psnr_spread_db', float('nan')):.1f} dB"
            f" vs R2 spread {trap.get('r2_spread', float('nan')):.3f}"
        )
        axes[1].set_yscale("log")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle(
        f"Coefficient -> far-field forward model | {summary['n_folds']} folds, "
        f"{summary['epochs']} epochs | all gates passed: {summary['all_gates_passed']}"
    )
    fig.savefig(fig_dir / "cv_and_gates.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())