"""Is the input_terms effect real, or is it one shared seed's init luck?

The 18-fold paired test reported in136 - in78 = -0.098, p=0.0144. But every fold
used ``seed=0``, so all 18 "independent" paired differences came from ONE weight
initialisation per configuration. This script isolates that: it holds ONE FOLD
fixed and varies the seed, which is the axis the previous design confounded with
the fold axis.

If the sign of (in136 - in78) flips as the seed changes, the earlier p-value was
measuring one particular init pair, not a property of trimming the input.

Writes logs/seed_sensitivity.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.zernike.eval_stats import seed_agreement  # noqa: E402
from ml.zernike.train_coeff import CoeffTrainConfig, train  # noqa: E402

OUT = "logs/seed_sensitivity.json"


def main(argv: list[str] | None = None) -> int:
    """Run both input widths across several seeds on one fold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--input-terms", type=int, nargs="+", default=[78, 136])
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args(argv)

    rows: list[dict] = []
    started = time.perf_counter()
    for seed in args.seeds:
        for terms in args.input_terms:
            out_dir = Path("logs/zernike_coeff") / f"seedseed{seed}_in{terms}"
            cfg = CoeffTrainConfig(
                protocol="file",
                fold=args.fold,
                epochs=args.epochs,
                seed=seed,
                input_terms=terms,
                use_wandb=False,
                log_every=max(1, args.epochs),
                image_every=10**9,  # figures are irrelevant here; skip the work
                out_dir=str(out_dir),
            )
            print(
                f"\n=== fold {args.fold} seed {seed} input_terms {terms} ===",
                flush=True,
            )
            result = train(cfg)
            rows.append(
                {
                    "fold": args.fold,
                    "seed": seed,
                    "input_terms": terms,
                    "best_val_r2": result.best_val_r2,
                }
            )
            print(
                f"    -> best_val_r2 {result.best_val_r2:+.4f}", flush=True
            )

    by_seed: dict[int, dict[int, float]] = {}
    for row in rows:
        by_seed.setdefault(row["seed"], {})[row["input_terms"]] = row["best_val_r2"]
    diffs: list[float] = []
    table: list[dict] = []
    for seed, values in sorted(by_seed.items()):
        if len(values) < 2:
            continue
        keys = sorted(values)
        lo, hi = keys[0], keys[-1]
        diff = values[hi] - values[lo]
        diffs.append(diff)
        table.append(
            {
                "seed": seed,
                f"in{lo}": values[lo],
                f"in{hi}": values[hi],
                f"in{hi}_minus_in{lo}": diff,
            }
        )

    # {seed: high_width - low_width} for the seed-agreement diagnosis.
    seed_diffs: dict[int, float] = {}
    for seed, values in sorted(by_seed.items()):
        if len(values) < 2:
            continue
        keys = sorted(values)
        seed_diffs[seed] = values[keys[-1]] - values[keys[0]]
    agreement = seed_agreement(seed_diffs)
    summary = {
        "fold": args.fold,
        "epochs": args.epochs,
        "input_terms": args.input_terms,
        "seeds": args.seeds,
        "per_seed": table,
        "mean_diff": float(np.mean(diffs)) if diffs else float("nan"),
        "std_diff": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else float("nan"),
        "signs_observed": agreement["signs_observed"],
        "sign_flips": agreement["sign_flips"],
        "per_term_means": {
            f"in{t}": float(
                np.mean([r["best_val_r2"] for r in rows if r["input_terms"] == t])
            )
            for t in args.input_terms
        },
        "wall_seconds": time.perf_counter() - started,
        "interpretation": agreement["interpretation"],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n================ SEED SENSITIVITY ================")
    print(f"fold {args.fold}, {args.epochs} epochs, seeds {args.seeds}")
    for entry in table:
        parts = "  ".join(f"{k}={v:+.4f}" for k, v in entry.items() if k != "seed")
        print(f"  seed {entry['seed']}: {parts}")
    print(f"mean diff {summary['mean_diff']:+.4f} +/- {summary['std_diff']:.4f}")
    print(f"signs observed: {summary['signs_observed']}")
    print(f"SIGN FLIPS: {summary['sign_flips']}")
    print(f"=> {summary['interpretation']}")
    for term, mean in summary["per_term_means"].items():
        print(f"   mean {term}: {mean:+.4f}")
    print(f"artefact: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())