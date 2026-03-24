from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
ENV = dict(os.environ)
ENV["PYTHONPATH"] = "src;.venv\\Lib\\site-packages"


CURRICULUM = [
    {
        "name": "stage1_easy",
        "total-timesteps": 1024,
        "max-steps": 32,
        "n-grid": 32,
        "n-actuators": 4,
        "n-subapertures": 4,
        "history-len": 4,
        "cn2": 5e-16,
        "screen-step-px": 1,
        "screen-margin-steps": 12,
        "goal-gain": 0.02,
        "hold-target-steps": 3,
        "action-scale": 0.01,
        "time-penalty": 0.006,
        "action-penalty": 0.00015,
        "saturation-penalty": 0.003,
        "learning-rate": 5e-5,
        "buffer-size": 150000,
        "batch-size": 64,
        "learning-starts": 256,
        "tau": 0.006,
        "gamma": 0.995,
        "train-freq": 1,
        "gradient-steps": 1,
        "eval-freq": 256,
        "eval-episodes": 6,
        "seed": 401,
    },
    {
        "name": "stage2_medium",
        "total-timesteps": 1024,
        "max-steps": 32,
        "n-grid": 32,
        "n-actuators": 4,
        "n-subapertures": 4,
        "history-len": 4,
        "cn2": 8e-16,
        "screen-step-px": 1,
        "screen-margin-steps": 12,
        "goal-gain": 0.025,
        "hold-target-steps": 3,
        "action-scale": 0.01,
        "time-penalty": 0.007,
        "action-penalty": 0.0002,
        "saturation-penalty": 0.0035,
        "learning-rate": 5e-5,
        "buffer-size": 150000,
        "batch-size": 64,
        "learning-starts": 0,
        "tau": 0.006,
        "gamma": 0.995,
        "train-freq": 1,
        "gradient-steps": 1,
        "eval-freq": 256,
        "eval-episodes": 6,
        "seed": 401,
    },
    {
        "name": "stage3_target",
        "total-timesteps": 6144,
        "max-steps": 36,
        "n-grid": 32,
        "n-actuators": 4,
        "n-subapertures": 4,
        "history-len": 4,
        "cn2": 1e-15,
        "screen-step-px": 1,
        "screen-margin-steps": 14,
        "goal-gain": 0.026,
        "hold-target-steps": 5,
        "action-scale": 0.01,
        "time-penalty": 0.0075,
        "action-penalty": 0.00025,
        "saturation-penalty": 0.0045,
        "learning-rate": 2e-5,
        "buffer-size": 150000,
        "batch-size": 64,
        "learning-starts": 0,
        "tau": 0.004,
        "gamma": 0.997,
        "train-freq": 1,
        "gradient-steps": 1,
        "eval-freq": 256,
        "eval-episodes": 6,
        "seed": 401,
    },
]


def _run_stage(stage: dict[str, object], log_dir: Path, model_dir: Path, init_model: Path | None) -> Path:
    cmd = [PYTHON, "src/ao_shaping/optimizer/rl/sac_turbulence_train.py"]
    for key, value in stage.items():
        if key == "name":
            continue
        cmd.extend([f"--{key}", str(value)])
    cmd.extend(["--log-dir", str(log_dir), "--model-dir", str(model_dir)])
    if init_model is not None:
        cmd.extend(["--init-model", str(init_model)])
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=ENV,
        capture_output=True,
        text=True,
        check=False,
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (log_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Curriculum stage failed: {stage['name']}")
    return model_dir / "sac_turbulence_final.zip"


def _load_eval_curve(log_dir: Path) -> np.ndarray:
    data = np.load(log_dir / "eval" / "evaluations.npz")
    return data["results"].mean(axis=1)


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / "logs" / f"mamba_curriculum_{timestamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    stage_rows: list[dict[str, object]] = []
    init_model: Path | None = None
    for stage in CURRICULUM:
        log_dir = ROOT / "logs" / f"{stage['name']}_{timestamp}"
        model_dir = ROOT / "models" / f"{stage['name']}_{timestamp}"
        final_model = _run_stage(stage, log_dir, model_dir, init_model)
        summary = json.loads((log_dir / "summary.json").read_text(encoding="utf-8"))
        row = {
            "stage": stage["name"],
            "log_dir": str(log_dir),
            "model_dir": str(model_dir),
            **{k: v for k, v in stage.items() if k != "name"},
            **summary,
        }
        stage_rows.append(row)
        init_model = final_model

    pd.DataFrame(stage_rows).to_csv(out_root / "curriculum_summary.csv", index=False)
    with (out_root / "curriculum_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(stage_rows, fh, indent=2, ensure_ascii=False)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    stage_names = [row["stage"] for row in stage_rows]
    axes[0].bar(stage_names, [float(row["mean_reward"]) for row in stage_rows])
    axes[0].set_ylabel("Mean Reward")
    axes[1].bar(stage_names, [float(row["mean_final_strehl"]) for row in stage_rows])
    axes[1].set_ylabel("Final Strehl")
    axes[2].bar(stage_names, [float(row["mean_best_pib"]) / 1e6 for row in stage_rows])
    axes[2].set_ylabel("Best PIB (x1e6)")
    axes[2].tick_params(axis="x", rotation=15)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_root / "curriculum_stage_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for row in stage_rows:
        rewards = _load_eval_curve(Path(row["log_dir"]))
        ax.plot(rewards, marker="o", label=row["stage"])
    ax.set_title("Curriculum Eval Reward Curves")
    ax.set_ylabel("Mean Eval Reward")
    ax.set_xlabel("Eval Index")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_root / "curriculum_eval_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    final = stage_rows[-1]
    converged = float(final["mean_final_strehl"]) >= 0.45 and float(final["std_reward"]) < 20.0
    report_lines = [
        "# Mamba Curriculum Report",
        "",
        f"- Output: `{out_root}`",
        f"- Final stage: `{final['stage']}`",
        f"- Converged: `{converged}`",
        "",
        "## Stage Results",
    ]
    for row in stage_rows:
        report_lines.append(
            f"- `{row['stage']}`: mean_reward={float(row['mean_reward']):.3f}, "
            f"mean_final_strehl={float(row['mean_final_strehl']):.3f}, "
            f"mean_best_pib={float(row['mean_best_pib']):.1f}"
        )
    report_lines.extend(
        [
            "",
            "## One-Click",
            "`python scripts/run_curriculum_mamba_turbulence.py`",
        ]
    )
    (out_root / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(out_root)


if __name__ == "__main__":
    main()
