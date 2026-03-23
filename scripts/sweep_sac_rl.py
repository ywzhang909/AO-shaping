from __future__ import annotations

from datetime import datetime
from pathlib import Path
import csv
import json
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
PYTHONPATH = "src;.venv\\Lib\\site-packages"


STATIC_SWEEP = [
    {
        "name": "static_baseline",
        "script": "src/ao_shaping/optimizer/rl/sac_train.py",
        "args": {
            "total-timesteps": 512,
            "max-steps": 24,
            "n-grid": 32,
            "n-actuators": 4,
            "n-subapertures": 4,
            "history-len": 4,
            "n-zernike-modes": 6,
            "zernike-coeff-std": 0.16,
            "zernike-coeff-clip": 0.35,
            "learning-rate": 3e-4,
            "buffer-size": 20000,
            "batch-size": 64,
            "learning-starts": 64,
            "gamma": 0.99,
            "tau": 0.01,
            "train-freq": 1,
            "gradient-steps": 1,
            "eval-freq": 128,
            "eval-episodes": 4,
            "goal-gain": 0.18,
            "hold-target-steps": 3,
            "action-scale": 0.025,
            "time-penalty": 0.008,
            "action-penalty": 0.0008,
            "saturation-penalty": 0.01,
            "seed": 11,
        },
    },
    {
        "name": "static_conservative",
        "script": "src/ao_shaping/optimizer/rl/sac_train.py",
        "args": {
            "total-timesteps": 512,
            "max-steps": 24,
            "n-grid": 32,
            "n-actuators": 4,
            "n-subapertures": 4,
            "history-len": 4,
            "n-zernike-modes": 6,
            "zernike-coeff-std": 0.14,
            "zernike-coeff-clip": 0.30,
            "learning-rate": 1e-4,
            "buffer-size": 30000,
            "batch-size": 64,
            "learning-starts": 96,
            "gamma": 0.995,
            "tau": 0.008,
            "train-freq": 1,
            "gradient-steps": 1,
            "eval-freq": 128,
            "eval-episodes": 4,
            "goal-gain": 0.15,
            "hold-target-steps": 4,
            "action-scale": 0.02,
            "time-penalty": 0.006,
            "action-penalty": 0.0005,
            "saturation-penalty": 0.008,
            "seed": 13,
        },
    },
    {
        "name": "static_fast",
        "script": "src/ao_shaping/optimizer/rl/sac_train.py",
        "args": {
            "total-timesteps": 512,
            "max-steps": 24,
            "n-grid": 32,
            "n-actuators": 4,
            "n-subapertures": 4,
            "history-len": 4,
            "n-zernike-modes": 6,
            "zernike-coeff-std": 0.18,
            "zernike-coeff-clip": 0.40,
            "learning-rate": 3e-4,
            "buffer-size": 20000,
            "batch-size": 128,
            "learning-starts": 64,
            "gamma": 0.985,
            "tau": 0.012,
            "train-freq": 1,
            "gradient-steps": 2,
            "eval-freq": 128,
            "eval-episodes": 4,
            "goal-gain": 0.20,
            "hold-target-steps": 3,
            "action-scale": 0.03,
            "time-penalty": 0.010,
            "action-penalty": 0.0010,
            "saturation-penalty": 0.012,
            "seed": 17,
        },
    },
]


TURB_SWEEP = [
    {
        "name": "turb_baseline",
        "script": "src/ao_shaping/optimizer/rl/sac_turbulence_train.py",
        "args": {
            "total-timesteps": 512,
            "max-steps": 32,
            "n-grid": 32,
            "n-actuators": 4,
            "n-subapertures": 4,
            "history-len": 4,
            "cn2": 5e-15,
            "screen-step-px": 1,
            "screen-margin-steps": 12,
            "learning-rate": 1e-4,
            "buffer-size": 30000,
            "batch-size": 64,
            "learning-starts": 96,
            "gamma": 0.985,
            "tau": 0.01,
            "train-freq": 1,
            "gradient-steps": 1,
            "eval-freq": 128,
            "eval-episodes": 4,
            "goal-gain": 0.10,
            "hold-target-steps": 5,
            "action-scale": 0.02,
            "time-penalty": 0.012,
            "action-penalty": 0.0006,
            "saturation-penalty": 0.008,
            "seed": 23,
        },
    },
    {
        "name": "turb_conservative",
        "script": "src/ao_shaping/optimizer/rl/sac_turbulence_train.py",
        "args": {
            "total-timesteps": 512,
            "max-steps": 32,
            "n-grid": 32,
            "n-actuators": 4,
            "n-subapertures": 4,
            "history-len": 4,
            "cn2": 3e-15,
            "screen-step-px": 1,
            "screen-margin-steps": 16,
            "learning-rate": 8e-5,
            "buffer-size": 30000,
            "batch-size": 64,
            "learning-starts": 128,
            "gamma": 0.99,
            "tau": 0.008,
            "train-freq": 1,
            "gradient-steps": 1,
            "eval-freq": 128,
            "eval-episodes": 4,
            "goal-gain": 0.08,
            "hold-target-steps": 6,
            "action-scale": 0.015,
            "time-penalty": 0.010,
            "action-penalty": 0.0004,
            "saturation-penalty": 0.006,
            "seed": 29,
        },
    },
    {
        "name": "turb_reactive",
        "script": "src/ao_shaping/optimizer/rl/sac_turbulence_train.py",
        "args": {
            "total-timesteps": 512,
            "max-steps": 32,
            "n-grid": 32,
            "n-actuators": 4,
            "n-subapertures": 4,
            "history-len": 4,
            "cn2": 7e-15,
            "screen-step-px": 2,
            "screen-margin-steps": 12,
            "learning-rate": 1.5e-4,
            "buffer-size": 30000,
            "batch-size": 64,
            "learning-starts": 96,
            "gamma": 0.98,
            "tau": 0.012,
            "train-freq": 1,
            "gradient-steps": 2,
            "eval-freq": 128,
            "eval-episodes": 4,
            "goal-gain": 0.12,
            "hold-target-steps": 4,
            "action-scale": 0.025,
            "time-penalty": 0.014,
            "action-penalty": 0.0008,
            "saturation-penalty": 0.010,
            "seed": 31,
        },
    },
]


def _run_case(case: dict, out_root: Path) -> dict[str, object]:
    log_dir = out_root / "logs" / case["name"]
    model_dir = out_root / "models" / case["name"]
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    cmd = [PYTHON, case["script"]]
    for key, value in case["args"].items():
        cmd.extend([f"--{key}", str(value)])
    cmd.extend(["--log-dir", str(log_dir), "--model-dir", str(model_dir)])

    env = dict(os.environ)
    env["PYTHONPATH"] = PYTHONPATH
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    summary_path = log_dir / "summary.json"
    summary: dict[str, object] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    result = {
        "name": case["name"],
        "returncode": proc.returncode,
        "log_dir": str(log_dir),
        "model_dir": str(model_dir),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
    }
    result.update(summary)
    return result


def _score(row: dict[str, object], mode: str) -> float:
    if row.get("returncode", 1) != 0:
        return -1e9
    reward = float(row.get("mean_reward", -1e6))
    strehl = float(row.get("mean_final_strehl", 0.0))
    pib = float(row.get("mean_best_pib", 0.0))
    if mode == "static":
        return reward + 4.0 * strehl + pib / 1e6
    return reward + 3.0 * strehl + pib / 2e6


def _save_tables(rows: list[dict[str, object]], out_root: Path, prefix: str) -> None:
    csv_path = out_root / f"{prefix}_summary.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (out_root / f"{prefix}_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)


def _plot_scores(rows: list[dict[str, object]], out_root: Path, prefix: str) -> None:
    names = [row["name"] for row in rows]
    rewards = [float(row.get("mean_reward", np.nan)) for row in rows]
    strehls = [float(row.get("mean_final_strehl", np.nan)) for row in rows]
    pibs = [float(row.get("mean_best_pib", np.nan)) / 1e6 for row in rows]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].bar(names, rewards)
    axes[0].set_ylabel("Mean Reward")
    axes[1].bar(names, strehls)
    axes[1].set_ylabel("Final Strehl")
    axes[2].bar(names, pibs)
    axes[2].set_ylabel("Best PIB (x1e6)")
    axes[2].tick_params(axis="x", rotation=20)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_root / f"{prefix}_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / "logs" / f"sac_sweep_{timestamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    static_rows = [_run_case(case, out_root) for case in STATIC_SWEEP]
    turb_rows = [_run_case(case, out_root) for case in TURB_SWEEP]

    for row in static_rows:
        row["score"] = _score(row, "static")
    for row in turb_rows:
        row["score"] = _score(row, "turbulence")

    static_best = max(static_rows, key=lambda row: float(row["score"]))
    turb_best = max(turb_rows, key=lambda row: float(row["score"]))

    _save_tables(static_rows, out_root, "static")
    _save_tables(turb_rows, out_root, "turbulence")
    _plot_scores(static_rows, out_root, "static")
    _plot_scores(turb_rows, out_root, "turbulence")

    recommendation = {
        "generated_at": timestamp,
        "static_best": static_best,
        "turbulence_best": turb_best,
        "recommended_static_params": next(
            case["args"] for case in STATIC_SWEEP if case["name"] == static_best["name"]
        ),
        "recommended_turbulence_params": next(
            case["args"] for case in TURB_SWEEP if case["name"] == turb_best["name"]
        ),
    }
    with (out_root / "recommendation.json").open("w", encoding="utf-8") as fh:
        json.dump(recommendation, fh, indent=2, ensure_ascii=False)

    report_lines = [
        "# SAC Hyperparameter Sweep",
        "",
        f"- Output: `{out_root}`",
        f"- Best static run: `{static_best['name']}`",
        f"- Best turbulence run: `{turb_best['name']}`",
        "",
        "## Recommended Static Params",
    ]
    for key, value in recommendation["recommended_static_params"].items():
        report_lines.append(f"- `{key}`: `{value}`")
    report_lines.append("")
    report_lines.append("## Recommended Turbulence Params")
    for key, value in recommendation["recommended_turbulence_params"].items():
        report_lines.append(f"- `{key}`: `{value}`")
    (out_root / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(out_root)


if __name__ == "__main__":
    main()
