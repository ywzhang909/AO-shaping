"""Paired bootstrap CI for the deswivel three-arm comparison (offline, no devices).

``compare_deswivel_forward_inverse_20261118.py`` writes per-phase forward metrics
for the three geometry arms but only the mean deltas. The pre-registered decision
rule in ``docs/dev/axis_mismatch_independent_capture_plan_20261118.md`` §5 asks for
a **bootstrap 95% CI on the per-phase paired delta**, so this script recomputes it
from the run's ``report.json`` without touching any hardware or re-training.

Usage::

    python scripts/slm/paired_bootstrap_deswivel_20261118.py \
        --run-dir data/debug/deswivel_forward_inverse_20261118/independent_20261118

It writes ``<run-dir>/paired_bootstrap_heldout.json`` and prints one line per
comparison. Validation phases (``index % 6 >= 3``) are the only paired unit; the
training split is never used for the verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SCHEMA = "deswivel_paired_bootstrap/1"
METRICS = ("correlation", "r2", "relative_l2")
#: metrics where a larger value favors the variant arm
HIGHER_IS_BETTER = {"correlation": True, "r2": True, "relative_l2": False}


def _per_phase(report: dict, arm: str, split: str, metric: str) -> np.ndarray:
    rows = report["forward"][arm][split]["per_row"]
    return np.asarray([float(row[metric]) for row in rows], dtype=np.float64)


def paired_bootstrap(
    report: dict,
    *,
    split: str = "heldout",
    seed: int = 20261118,
    n_boot: int = 20000,
) -> dict:
    """Bootstrap the mean of ``variant - baseline`` per phase on one split."""
    baseline = {
        metric: _per_phase(report, "baseline", split, metric) for metric in METRICS
    }
    n = len(baseline["correlation"])
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, n, size=(n_boot, n))

    comparisons: dict[str, dict] = {}
    for arm in ("corrected", "axis"):
        for metric in METRICS:
            variant = _per_phase(report, arm, split, metric)
            if variant.shape != baseline[metric].shape:
                raise ValueError(
                    f"{arm}/{metric} has {variant.shape} rows, baseline has "
                    f"{baseline[metric].shape}; splits must be aligned"
                )
            delta = variant - baseline[metric]
            boot = delta[draw].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            excludes_zero = bool(lo > 0.0 or hi < 0.0)
            better = (
                delta.mean() > 0.0 if HIGHER_IS_BETTER[metric] else delta.mean() < 0.0
            )
            comparisons[f"{arm}_minus_baseline_{metric}"] = {
                "mean_delta": float(delta.mean()),
                "ci95": [float(lo), float(hi)],
                "ci_excludes_zero": excludes_zero,
                "mean_favors": arm if better else "baseline",
                "verdict": (
                    f"{arm} ({'better' if better else 'worse'})"
                    if excludes_zero
                    else "inconclusive"
                ),
                "n_phases": int(n),
                "n_positive": int((delta > 0).sum()),
                "per_phase_delta": [float(value) for value in delta],
            }
    return {
        "schema": SCHEMA,
        "split": split,
        "unit": "phase (paired)",
        "seed": seed,
        "n_boot": n_boot,
        "higher_is_better": HIGHER_IS_BETTER,
        "comparisons": comparisons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", default="heldout", choices=("heldout", "train"))
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--n-boot", type=int, default=20000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    report_path = args.run_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result = paired_bootstrap(
        report, split=args.split, seed=args.seed, n_boot=args.n_boot
    )
    result["source"] = str(report_path)

    out = args.out or args.run_dir / f"paired_bootstrap_{args.split}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    for name, item in result["comparisons"].items():
        lo, hi = item["ci95"]
        print(
            f"{name}: mean {item['mean_delta']:+.5f} "
            f"CI95 [{lo:+.5f}, {hi:+.5f}] "
            f"excludes0={item['ci_excludes_zero']} "
            f"favor={item['mean_favors']} n={item['n_phases']}"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
