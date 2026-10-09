"""Grouped cross-validation for the physics model vs the U-Net baseline.

Why this script exists
----------------------
The earlier comparisons used ``_select_records``, a single split that puts
``val_fraction`` (0.25) of the *files* into validation and then head-truncates
validation to ``max_val=128`` records. Two consequences, both measured:

* every record outside the 2 validation files is **never** validated, and
* R^2 swings between +0.78 and +0.92 across split seeds (sigma ~ 0.06).

That noise floor was large enough that no model comparison could be resolved.

The corpus is **not i.i.d.**, and inspecting it says why. The 1010 usable records
of ``slm_zernike_shaping`` sit in 10 pickles of exactly 101 records each, but
those pickles are 2-4 timestamps of only **four** distinct optimisation
objectives::

    rms_pib  4 files  404 records
    rmse_out 3 files  303 records
    shape    2 files  202 records
    roi_pib  1 file   101 records

and the objectives have genuinely different image distributions. Scoring one
objective's mean image against another's targets gives::

    R^2(rms_pib -> roi_pib) = -0.670      # worse than predicting a constant
    R^2(shape  -> roi_pib) = +0.040
    R^2(rms_pib -> shape)  = +0.710

So a random validation fold gets a *random objective mixture*, and since
``roi_pib`` is 10 % of the corpus and nearly orthogonal to the rest, the pooled
R^2 depends on the mixture. That -- not the fold size -- is the dominant noise
source. A split-seed sweep was measuring the mixture lottery, not model quality.

Note the repo has **no sklearn dependency at all** (verified: zero references to
``sklearn``/``model_selection`` anywhere), so the folds are built by hand here
rather than via ``GroupKFold``. ``_select_records`` already uses the right group
key -- ``str(record.path)`` -- and this script keeps that convention.

Two protocols:

``objective``
    Leave-one-objective-out (4 folds). The strict test: fit on three objectives,
    validate on the fourth, i.e. generalisation to an objective never seen. This
    is the question that matters on the bench, where each run uses a different
    objective, and every fold's validation set is one homogeneous distribution.

``file``
    Leave-one-pickle-out (10 folds). Sibling pickles of the same objective stay
    in train, so this is mildly optimistic -- but it validates **every one of the
    1010 records exactly once**, which averages the mixture lottery away and
    yields 10 folds, enough power for a paired test.

Run it::

    python scripts/compare_models_cv.py --protocol both
    python scripts/compare_models_cv.py --protocol objective --models physics unet

Every model sees the *same* folds, inputs, target, loss, optimiser, schedule and
epoch budget, so per-fold differences form a valid paired sample.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger

from ml.hwdataset import HwPhaseImageDataset, MaterialiserConfig, build_hw_index
from ml.zernike.metrics import batch_image_metrics, constant_baseline_metrics
from ml.zernike.eval_stats import holm_bonferroni as eval_holm_bonferroni
from ml.zernike.eval_stats import sign_flip_pvalue
from ml.zernike.train_amp import AmpTrainConfig, collect_split

from compare_unet_baseline import (
    _inputs,
    _peak_normalise,
    build_model,
    fit,
    forward_for,
)

# `slm_zernike_shaping_rms_pib_20260926_162917_20260926_162917.pkl` -> `rms_pib`
_OBJECTIVE_RE = re.compile(r"^slm_zernike_shaping_(?P<objective>.+?)_\d{8}_\d{6}")

# ``skill`` / ``r2_const`` are the honest headline on this corpus: a *constant*
# predictor reaches ``r2`` ~= +0.910, so ``r2`` alone cannot separate models.
# ``skill = 1 - mse_model / mse_const`` has no such floor. See
# ``report/zernike_r2_baseline/report.md``.
METRICS = ("mse", "r2", "ssim", "psnr", "nrmse", "skill", "r2_const", "mse_const")


def objective_of(path: Path) -> str:
    """Extract the optimisation objective from a corpus pickle filename.

    The four objectives are very unevenly represented (``roi_pib`` has a single
    file), which is exactly why folds must be built per objective rather than by
    shuffling records.
    """
    match = _OBJECTIVE_RE.match(path.stem)
    if match is None:
        raise ValueError(f"cannot parse an objective out of {path.name!r}")
    return match.group("objective")


@dataclass
class Fold:
    """One grouped split, as position lists into ``dataset.records``."""

    index: int
    train: list[int]
    val: list[int]
    label: str


def build_folds(records: list, protocol: str) -> list[Fold]:
    """Construct the grouped folds.

    Grouping is by objective first in both protocols, so an objective is never
    split across the train/val boundary. ``file`` refines that to the individual
    pickle.
    """
    by_objective: dict[str, list[int]] = defaultdict(list)
    by_file: dict[Path, list[int]] = defaultdict(list)
    for position, record in enumerate(records):
        by_objective[objective_of(record.path)].append(position)
        by_file[record.path].append(position)

    if protocol == "objective":
        folds = [
            Fold(
                index=i,
                train=[p for name, ps in by_objective.items() if name != held for p in ps],
                val=ps,
                label=f"hold out {held} (n={len(ps)})",
            )
            for i, (held, ps) in enumerate(sorted(by_objective.items()))
        ]
    elif protocol == "file":
        folds = [
            Fold(
                index=i,
                train=[p for path, ps in by_file.items() if path != held_path for p in ps],
                val=ps,
                label=f"hold out {held_path.stem[:34]} (n={len(ps)})",
            )
            for i, (held_path, ps) in enumerate(sorted(by_file.items(), key=lambda kv: str(kv[0])))
        ]
    else:  # pragma: no cover - argparse restricts the choices
        raise ValueError(f"unknown protocol {protocol!r}")

    for fold in folds:
        if not fold.train or not fold.val:
            raise ValueError(
                f"fold {fold.index} is degenerate: {len(fold.train)} train / {len(fold.val)} val"
            )
    return folds


def _predict(model, stacked: torch.Tensor, forward) -> torch.Tensor:
    chunks = []
    with torch.no_grad():
        for start in range(0, stacked.shape[0], 256):
            stop = min(start + 256, stacked.shape[0])
            chunks.append(forward(model, stacked[start:stop]))
    return torch.cat(chunks)


def _metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Image metrics **plus** the constant-predictor floor.

    ``r2`` alone is not a scoreboard here: on this corpus a *constant* image
    (the split's own mean) reaches R^2 ~= +0.910, because peak-normalised
    far-field frames are nearly static (pooled target variance ~0.014). So every
    model in a comparison table has to be read against ``r2_const``, and
    ``skill = 1 - mse_model / mse_const`` is the floor-free number. Without it a
    model that does nothing scores 0.91. See
    ``report/zernike_r2_baseline/report.md``.
    """
    values = batch_image_metrics(prediction, target)
    values.update(constant_baseline_metrics(prediction, target))
    return {key: float(values[key]) for key in METRICS if key in values}


def score_by_objective(
    prediction: torch.Tensor, target: torch.Tensor, groups: list[str]
) -> dict[str, dict[str, float]]:
    """Metric per objective inside one fold.

    A single pooled R^2 over a mixture of objectives is exactly the number that
    made the first comparison unreadable, so the per-objective breakdown is the
    primary output and the pooled value is secondary.
    """
    out: dict[str, dict[str, float]] = {}
    for name in sorted(set(groups)):
        mask = torch.tensor([g == name for g in groups])
        if not bool(mask.any()):
            continue
        out[name] = _metrics(prediction[mask], target[mask])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--protocol", choices=["objective", "file", "both"], default="both")
    parser.add_argument("--models", nargs="*", default=["physics", "hybrid", "unet"])
    parser.add_argument("--residual-width", type=int, default=32)
    parser.add_argument("--n-max", type=int, default=15)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--unet-features", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    parser.add_argument(
        "--unet-output-mode",
        choices=["phase", "image"],
        default="phase",
        help=(
            "U-Net head. 'phase' ends in a sigmoid (right for an SLM phase map, "
            "wrong for an intensity target -- see report/zernike_r2_baseline). "
            "'image' is the linear head."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="logs/models_cv.json")
    parser.add_argument(
        "--analyse",
        nargs="*",
        default=None,
        help="recompute the paired statistics from saved run JSON(s) and exit",
    )
    args = parser.parse_args()

    if args.analyse:
        for path in args.analyse:
            _reanalyse(Path(path))
        return 0

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    index = build_hw_index(index_cache="data/hw_index_cache.json").filter(
        families=["slm_zernike_shaping"]
    )
    records = list(index.records)
    dataset = HwPhaseImageDataset(
        index, config=MaterialiserConfig(grid=args.grid), use_cache=True
    )
    logger.info(
        "corpus: {} records in {} pickles, {} objectives",
        len(records), len({r.path for r in records}),
        len({objective_of(r.path) for r in records}),
    )

    protocols = ["objective", "file"] if args.protocol == "both" else [args.protocol]
    report: dict = {"config": vars(args), "device": str(device), "protocols": {}}

    for protocol in protocols:
        folds = build_folds(records, protocol)
        logger.info(
            "\n=== protocol={} folds={} (train {} / val {} records per fold) ===",
            protocol, len(folds), len(folds[0].train), len(folds[0].val),
        )
        rows: list[dict] = []
        for fold in folds:
            train_t = collect_split(dataset, fold.train, device)
            val_t = collect_split(dataset, fold.val, device)
            val_groups = [objective_of(records[p].path) for p in fold.val]
            cfg = AmpTrainConfig(
                families=("slm_zernike_shaping",),
                n_max=args.n_max,
                grid=args.grid,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                use_wandb=False,
                save_checkpoint=False,
                seed=args.seed,
                out_dir="logs/cv",
            )
            for name in args.models:
                torch.manual_seed(args.seed)
                model = build_model(
                    name, cfg,
                    residual_width=args.residual_width,
                    unet_features=args.unet_features,
                    unet_output_mode=args.unet_output_mode,
                ).to(device)
                started = time.perf_counter()
                model, seconds, _ = fit(model, train_t, cfg, device, forward_for(name))
                prediction = _predict(model, _inputs(val_t), forward_for(name))
                target = _peak_normalise(val_t["target"].clone())
                pooled = _metrics(prediction, target)
                per_objective = score_by_objective(prediction, target, val_groups)
                logger.info(
                    "  fold {} [{}] {:<7} r2={:+.4f} ssim={:.4f} psnr={:.2f} ({:.1f}s) {}",
                    fold.index, fold.label, name, pooled["r2"], pooled["ssim"],
                    pooled["psnr"], seconds,
                    " ".join(f"{k}:{v['r2']:+.3f}" for k, v in per_objective.items()),
                )
                rows.append(
                    {
                        "fold": fold.index,
                        "label": fold.label,
                        "model": name,
                        "n_train": len(fold.train),
                        "n_val": len(fold.val),
                        "seconds": seconds,
                        "pooled": pooled,
                        "by_objective": per_objective,
                    }
                )
                # Free the fitted model and its predictions before the next model
                # reuses the device; the split tensors outlive the inner loop and
                # are released at the end of the fold instead.
                del model, prediction, target
            del train_t, val_t
            if device.type == "cuda":
                torch.cuda.empty_cache()
        report["protocols"][protocol] = _summarise(rows, args.models)
        _log_summary(protocol, rows, args.models)
        rows.clear()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("\nwrote {}", out)
    return 0


def exact_sign_flip_p(diffs: list[float]) -> float:
    """Exact two-sided sign-flip permutation p-value for a paired sample.

    Delegates to :func:`ml.zernike.eval_stats.sign_flip_pvalue`, which is the
    repo's single implementation. This wrapper used to carry its own copy of the
    enumeration, which is precisely the "second copy drifts" failure the
    statistics module exists to prevent -- two scripts computing the same test
    is how a report ends up quoting a p-value no code can reproduce. At 10 folds
    the null distribution is enumerable (2**10 = 1024 sign assignments), so no
    normality assumption is needed, and the smallest attainable two-sided p is
    2/1024 = 0.00195.

    Note that k-fold differences are *not* independent (fold i's training set
    overlaps fold j's), which makes even this test mildly anti-conservative --
    the effect size is the honest headline, the p-value only brackets it.
    """
    return sign_flip_pvalue(diffs)


def holm_bonferroni(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values, preserving input keys.

    Delegates to :func:`ml.zernike.eval_stats.holm_bonferroni`; see
    :func:`exact_sign_flip_p` for why this wrapper exists.
    """
    return eval_holm_bonferroni(p_values)


def _reanalyse(path: Path) -> None:
    """Recompute the paired statistics from a saved run file, no retraining.

    Lets the significance machinery be revised without paying for the folds
    again -- the per-fold scores are the expensive artefact, not the test.
    """
    blob = json.loads(path.read_text(encoding="utf-8"))
    for protocol, saved in blob["protocols"].items():
        models = list(saved["models"])
        rows = [
            {
                "fold": fold,
                "model": name,
                "pooled": {m: saved["models"][name][m]["per_fold"][fold] for m in METRICS},
            }
            for name in models
            for fold in range(len(saved["models"][name]["r2"]["per_fold"]))
        ]
        _log_summary(f"{protocol} (re-analysed)", rows, models)
        reference = models[0]
        for other in models[1:]:
            print(f"  {reference} minus {other}")
            for metric in METRICS:
                a = saved["models"][reference][metric]["per_fold"]
                b = saved["models"][other][metric]["per_fold"]
                diff = [x - y for x, y in zip(a, b)]
                sd = statistics.stdev(diff)
                d_z = statistics.fmean(diff) / sd if sd > 0 else float("inf")
                print(
                    f"    {metric:<5} diff={statistics.fmean(diff):+.4f}+-{sd:.4f}"
                    f"  d_z={d_z:+.2f}  p_ttest={_safe_ttest(a, b):.4f}"
                    f"  p_signflip={exact_sign_flip_p(diff):.4f}"
                )


def _safe_ttest(a: list[float], b: list[float]) -> float:
    from scipy import stats

    return float(stats.ttest_rel(a, b).pvalue)


def _summarise(rows: list[dict], models: list[str]) -> dict:
    """Aggregate per-fold scores and run paired comparisons between models."""
    out: dict = {"folds": len({r["fold"] for r in rows}), "models": {}}
    for name in models:
        mine = [r for r in rows if r["model"] == name]
        out["models"][name] = {
            metric: {
                "mean": statistics.fmean(r["pooled"][metric] for r in mine),
                "std": statistics.stdev(r["pooled"][metric] for r in mine) if len(mine) > 1 else 0.0,
                "per_fold": [r["pooled"][metric] for r in mine],
            }
            for metric in METRICS
        }
    # Per-objective means. Pooled R^2 over a mixture of objectives is the number
    # that made the first comparison unreadable, so the breakdown is kept in the
    # artefact rather than only in the log line.
    by_objective: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in mine:
        for objective, metrics in row.get("by_objective", {}).items():
            for metric, value in metrics.items():
                by_objective[objective][metric].append(value)
    out["by_objective"] = {
        objective: {
            metric: {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "per_fold": values,
            }
            for metric, values in metrics.items()
        }
        for objective, metrics in by_objective.items()
    }
    out["paired"] = _paired(rows, models[0], models[1:])
    return out


def _paired(rows: list[dict], reference: str, others: list[str]) -> dict:
    """Paired per-fold differences against ``reference``.

    The same folds are used for every model, so the per-fold differences are a
    paired sample and a paired t-test is valid. An unpaired test would inflate
    the variance by the fold-to-fold spread that both models share -- which is
    precisely the spread that made the first comparison unreadable.
    """
    from scipy import stats

    def per_fold(name: str, metric: str) -> dict[int, float]:
        return {r["fold"]: r["pooled"][metric] for r in rows if r["model"] == name}

    out: dict = {}
    raw_p: dict[str, float] = {}
    for other in others:
        entry: dict = {}
        for metric in METRICS:
            a, b = per_fold(reference, metric), per_fold(other, metric)
            folds = sorted(set(a) & set(b))
            diff = [a[f] - b[f] for f in folds]
            test = stats.ttest_rel([a[f] for f in folds], [b[f] for f in folds])
            sd = statistics.stdev(diff) if len(diff) > 1 else 0.0
            # Cohen's d_z for paired samples: mean difference over its own SD.
            # This is the effect size that survives a 10-fold sample; the raw
            # difference alone would be misleading at this n.
            d_z = statistics.fmean(diff) / sd if sd > 0 else float("inf")
            key = f"{reference}_minus_{other}:{metric}"
            entry[metric] = {
                "mean_diff": statistics.fmean(diff),
                "std_diff": sd,
                "cohens_dz": d_z,
                "t": float(test.statistic),
                "p_ttest": float(test.pvalue),
                "p_signflip": exact_sign_flip_p(diff),
                "per_fold": diff,
            }
            raw_p[key] = float(test.pvalue)
        out[f"{reference}_minus_{other}"] = entry
    adjusted = holm_bonferroni(raw_p)
    for pair, entry in out.items():
        other = pair.split("_minus_")[-1]
        for metric in METRICS:
            entry[metric]["p_holm"] = adjusted[f"{reference}_minus_{other}:{metric}"]
    return out


def _log_summary(protocol: str, rows: list[dict], models: list[str]) -> None:
    logger.info("\n--- {} : pooled over folds ---", protocol)
    for metric in METRICS:
        line = f"  {metric:<5}"
        for name in models:
            values = [r["pooled"][metric] for r in rows if r["model"] == name]
            line += f" | {name}: {statistics.fmean(values):+.4f} +- {statistics.stdev(values):.4f}"
        logger.info(line)


if __name__ == "__main__":
    raise SystemExit(main())
