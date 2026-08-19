"""Upload the real dual-spot experiment results (this session) to wandb.

One consolidated run per experiment family in project ``ao-zernike``:

  - real-per-coeff : 65 single-output coefficient models (run-split)
  - real-joint     : joint low-order(10) models — run-split + 0414 sample
                     unfiltered / dpeak>=100 / dpeak>=200
  - real-pib       : PIB scalar regression (r=15/25/40) — the positive result

Each run logs the aggregated scalar metrics, the per-coefficient table as a
wandb Table, and uploads the key artifacts (summary CSV, R2 comparison PNG,
test predictions). No per-epoch curves (trainers ran use_wandb=False), so the
runs are result-summaries, consistent with how the session archived results.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/push_results_wandb.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

import wandb  # noqa: E402

CKPT = Path("data/zernike_pred/checkpoints")
PROJECT = "ao-zernike"


def _per_coeff_table(summ: dict) -> list[list]:
    rows = []
    for name, m in summ.get("per_coeff", {}).items():
        rows.append([name, m["r2"], m["mae"], m["corr"]])
    return rows


def run_real_per_coeff() -> None:
    rows = []
    for f in sorted((CKPT / "real_per_coeff").glob("results_*.json")):
        rows += json.loads(f.read_text())
    names = sorted({r["name"] for r in rows})
    table = wandb.Table(
        columns=["index", "name", "best_val_mae", "best_epoch", "test_mae", "test_r2"],
        data=[
            [r["index"], r["name"], r["best_val_mae"], r["best_epoch"], r["test_mae"], r["test_r2"]]
            for r in sorted(rows, key=lambda r: r["index"])
        ],
    )
    with wandb.init(project=PROJECT, job_type="results", name="real-per-coeff-65") as run:
        r2 = [r["test_r2"] for r in rows if r["test_r2"] is not None]
        mae = [r["test_mae"] for r in rows]
        run.summary.update(
            {
                "n_models": len(rows),
                "test_mae_mean": float(np.mean(mae)),
                "test_mae_min": float(np.min(mae)),
                "test_mae_max": float(np.max(mae)),
                "test_r2_mean": float(np.mean(r2)),
                "test_r2_pos_count": int(sum(1 for v in r2 if v > 0)),
            }
        )
        run.log({"per_coeff_summary": table})
        run.log({"per_coeff_test_mae": wandb.Image(str(CKPT / "real_per_coeff" / "per_coeff_test_mae.png"))})
        run.log({"r2_comparison": wandb.Image(str(CKPT / "real_experiments_r2.png"))})
        print("wandb run real-per-coeff-65:", run.url)


def run_real_joint() -> None:
    experiments = {
        "joint10-run-split": "real_joint10_run",
        "joint10-0414-sample": "real_joint10_0414",
        "joint10-0414-dpeak100": "real_joint10_0414f100",
        "joint10-0414-dpeak200": "real_joint10_0414f",
    }
    with wandb.init(project=PROJECT, job_type="results", name="real-joint-loworder") as run:
        for label, d in experiments.items():
            p = CKPT / d / "eval_summary.json"
            if not p.exists():
                print(f"  skip missing {p}")
                continue
            s = json.loads(p.read_text())
            run.summary[f"{label}_test_mae"] = s["test_mae"]
            run.summary[f"{label}_test_r2"] = s["test_r2"]
            run.log(
                {
                    label: wandb.Table(
                        columns=["coeff", "r2", "mae", "corr"],
                        data=_per_coeff_table(s),
                    )
                }
            )
        run.log({"r2_comparison": wandb.Image(str(CKPT / "real_experiments_r2.png"))})
        print("wandb run real-joint-loworder:", run.url)


def run_real_pib() -> None:
    with wandb.init(project=PROJECT, job_type="results", name="real-pib-regression") as run:
        table_rows, scatter_rows = [], []
        for r in (15, 25, 40):
            p = CKPT / f"real_pib_r{r}" / "eval_summary.json"
            if not p.exists():
                print(f"  skip missing {p}")
                continue
            s = json.loads(p.read_text())
            table_rows.append([r, s["test_mae"], s["test_r2"], s.get("test_corr"), s["best_epoch"]])
            pred = np.load(CKPT / f"real_pib_r{r}" / "test_pred.npy").ravel()
            true = np.load(CKPT / f"real_pib_r{r}" / "test_true.npy").ravel()
            for x, y in zip(true, pred):
                scatter_rows.append([r, float(x), float(y)])
            run.summary[f"pib_r{r}_test_r2"] = s["test_r2"]
            run.summary[f"pib_r{r}_test_corr"] = s.get("test_corr")
        run.log({"pib_metrics": wandb.Table(columns=["bucket_r", "test_mae", "test_r2", "test_corr", "best_epoch"], data=table_rows)})
        run.log({"pib_scatter": wandb.Table(columns=["bucket_r", "true", "pred"], data=scatter_rows)})
        run.log({"r2_comparison": wandb.Image(str(CKPT / "real_experiments_r2.png"))})
        print("wandb run real-pib-regression:", run.url)


def main() -> None:
    run_real_per_coeff()
    run_real_joint()
    run_real_pib()
    print("done")


if __name__ == "__main__":
    main()