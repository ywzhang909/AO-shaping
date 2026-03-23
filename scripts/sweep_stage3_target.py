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


STAGE1 = {
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
}

STAGE2 = {
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
}

STAGE3_CANDIDATES = [
    {
        "name": "stage3_baseline_long",
        "total-timesteps": 4096,
        "max-steps": 32,
        "n-grid": 32,
        "n-actuators": 4,
        "n-subapertures": 4,
        "history-len": 4,
        "cn2": 1e-15,
        "screen-step-px": 1,
        "screen-margin-steps": 12,
        "goal-gain": 0.03,
        "hold-target-steps": 3,
        "action-scale": 0.01,
        "time-penalty": 0.008,
        "action-penalty": 0.0002,
        "saturation-penalty": 0.004,
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
        "name": "stage3_low_lr",
        "total-timesteps": 4096,
        "max-steps": 32,
        "n-grid": 32,
        "n-actuators": 4,
        "n-subapertures": 4,
        "history-len": 4,
        "cn2": 1e-15,
        "screen-step-px": 1,
        "screen-margin-steps": 12,
        "goal-gain": 0.028,
        "hold-target-steps": 4,
        "action-scale": 0.01,
        "time-penalty": 0.0085,
        "action-penalty": 0.00025,
        "saturation-penalty": 0.0045,
        "learning-rate": 3e-5,
        "buffer-size": 150000,
        "batch-size": 64,
        "learning-starts": 0,
        "tau": 0.005,
        "gamma": 0.995,
        "train-freq": 1,
        "gradient-steps": 1,
        "eval-freq": 256,
        "eval-episodes": 6,
        "seed": 401,
    },
    {
        "name": "stage3_steady_control",
        "total-timesteps": 4096,
        "max-steps": 36,
        "n-grid": 32,
        "n-actuators": 4,
        "n-subapertures": 4,
        "history-len": 4,
        "cn2": 1e-15,
        "screen-step-px": 1,
        "screen-margin-steps": 14,
        "goal-gain": 0.025,
        "hold-target-steps": 5,
        "action-scale": 0.01,
        "time-penalty": 0.007,
        "action-penalty": 0.0003,
        "saturation-penalty": 0.005,
        "learning-rate": 3e-5,
        "buffer-size": 150000,
        "batch-size": 64,
        "learning-starts": 0,
        "tau": 0.005,
        "gamma": 0.997,
        "train-freq": 1,
        "gradient-steps": 1,
        "eval-freq": 256,
        "eval-episodes": 6,
        "seed": 401,
    },
    {
        "name": "stage3_long_horizon",
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
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    final_model = model_dir / "sac_turbulence_final.zip"
    if (log_dir / "summary.json").exists() and final_model.exists():
        return final_model
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
        text=True,
        capture_output=True,
        check=False,
    )
    (log_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (log_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Stage failed: {stage['name']}")
    return final_model


def _load_eval(log_dir: Path) -> np.ndarray:
    data = np.load(log_dir / "eval" / "evaluations.npz")
    return data["results"].mean(axis=1)


def _convergence_metrics(rewards: np.ndarray, mean_final_strehl: float) -> dict[str, float | bool]:
    window = rewards[-3:] if rewards.size >= 3 else rewards
    slope = float(window[-1] - window[0]) / max(len(window) - 1, 1)
    std = float(np.std(window))
    is_plateau = abs(slope) < 20.0 and std < 20.0 and mean_final_strehl >= 0.35
    return {
        "late_reward_mean": float(np.mean(window)),
        "late_reward_std": std,
        "late_reward_slope": slope,
        "converged": is_plateau,
    }


def _ranking_key(row: dict[str, object]) -> tuple[float, float, float, float]:
    converged = 1.0 if bool(row["converged"]) else 0.0
    return (
        converged,
        float(row["mean_final_strehl"]),
        -abs(float(row["late_reward_slope"])),
        -float(row["late_reward_std"]),
    )


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "logs" / f"stage3_target_sweep_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    stage1_log = ROOT / "logs" / f"stage1_easy_sweep_{timestamp}"
    stage1_model = ROOT / "models" / f"stage1_easy_sweep_{timestamp}"
    stage1_final = _run_stage(STAGE1, stage1_log, stage1_model, None)

    stage2_log = ROOT / "logs" / f"stage2_medium_sweep_{timestamp}"
    stage2_model = ROOT / "models" / f"stage2_medium_sweep_{timestamp}"
    stage2_final = _run_stage(STAGE2, stage2_log, stage2_model, stage1_final)

    rows: list[dict[str, object]] = []
    fig, ax = plt.subplots(figsize=(10, 6))
    for stage in STAGE3_CANDIDATES:
        log_dir = ROOT / "logs" / f"{stage['name']}_{timestamp}"
        model_dir = ROOT / "models" / f"{stage['name']}_{timestamp}"
        _run_stage(stage, log_dir, model_dir, stage2_final)
        summary = json.loads((log_dir / "summary.json").read_text(encoding="utf-8"))
        rewards = _load_eval(log_dir)
        ax.plot(rewards, marker="o", label=stage["name"])
        row = {
            "stage": stage["name"],
            "log_dir": str(log_dir),
            "model_dir": str(model_dir),
            **{k: v for k, v in stage.items() if k != "name"},
            **summary,
            **_convergence_metrics(rewards, float(summary["mean_final_strehl"])),
        }
        rows.append(row)

    ax.set_title("Stage3 Target Sweep Eval Reward")
    ax.set_xlabel("Eval Index")
    ax.set_ylabel("Mean Eval Reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "stage3_eval_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    rows.sort(key=_ranking_key, reverse=True)
    best = rows[0]

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "stage3_sweep_summary.csv", index=False)
    with (out_dir / "stage3_sweep_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].bar(df["stage"], df["mean_final_strehl"])
    axes[0, 0].set_title("Mean Final Strehl")
    axes[0, 1].bar(df["stage"], df["late_reward_std"])
    axes[0, 1].set_title("Late Reward Std")
    axes[1, 0].bar(df["stage"], df["late_reward_slope"])
    axes[1, 0].set_title("Late Reward Slope")
    axes[1, 1].bar(df["stage"], df["mean_reward"])
    axes[1, 1].set_title("Mean Reward")
    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=15)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "stage3_sweep_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    report_lines = [
        "# Stage3 Target Sweep Report",
        "",
        f"- Output: `{out_dir}`",
        f"- Best candidate: `{best['stage']}`",
        f"- Converged: `{best['converged']}`",
        f"- mean_reward: `{float(best['mean_reward']):.3f}`",
        f"- mean_final_strehl: `{float(best['mean_final_strehl']):.3f}`",
        f"- late_reward_std: `{float(best['late_reward_std']):.3f}`",
        f"- late_reward_slope: `{float(best['late_reward_slope']):.3f}`",
        "",
        "## Best Params",
    ]
    for key in STAGE3_CANDIDATES[0]:
        if key == "name":
            continue
        report_lines.append(f"- {key}: `{best[key]}`")
    (out_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
