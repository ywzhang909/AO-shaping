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
from stable_baselines3 import SAC
from tensorboard.backend.event_processing import event_accumulator


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ao_shaping.optimizer.rl.sac_train import build_env as build_static_env
from ao_shaping.optimizer.rl.sac_turbulence_train import build_env as build_turb_env


PYTHON = sys.executable
ENV = dict(os.environ)
ENV["PYTHONPATH"] = "src;.venv\\Lib\\site-packages"


STATIC_ARGS = {
    "total-timesteps": 2048,
    "max-steps": 24,
    "n-grid": 32,
    "n-actuators": 4,
    "n-subapertures": 4,
    "history-len": 4,
    "n-zernike-modes": 6,
    "zernike-coeff-std": 0.14,
    "zernike-coeff-clip": 0.30,
    "goal-gain": 0.15,
    "hold-target-steps": 4,
    "action-scale": 0.02,
    "time-penalty": 0.006,
    "action-penalty": 0.0005,
    "saturation-penalty": 0.008,
    "learning-rate": 1e-4,
    "buffer-size": 100000,
    "batch-size": 64,
    "learning-starts": 256,
    "tau": 0.008,
    "gamma": 0.995,
    "train-freq": 1,
    "gradient-steps": 1,
    "eval-freq": 256,
    "eval-episodes": 6,
    "seed": 101,
}


TURB_BASE = {
    "max-steps": 32,
    "n-grid": 32,
    "n-actuators": 4,
    "n-subapertures": 4,
    "history-len": 4,
    "screen-step-px": 1,
    "screen-margin-steps": 12,
    "train-freq": 1,
    "gradient-steps": 1,
    "buffer-size": 150000,
    "batch-size": 64,
    "eval-freq": 256,
    "eval-episodes": 6,
}


TURB_SWEEP = [
    {
        "name": "mamba_gentle",
        **TURB_BASE,
        "total-timesteps": 1024,
        "cn2": 5e-16,
        "goal-gain": 0.04,
        "hold-target-steps": 4,
        "action-scale": 0.010,
        "time-penalty": 0.008,
        "action-penalty": 0.0002,
        "saturation-penalty": 0.004,
        "learning-rate": 5e-5,
        "learning-starts": 512,
        "tau": 0.006,
        "gamma": 0.995,
        "seed": 211,
    },
    {
        "name": "mamba_balanced",
        **TURB_BASE,
        "total-timesteps": 1024,
        "cn2": 8e-16,
        "goal-gain": 0.05,
        "hold-target-steps": 4,
        "action-scale": 0.012,
        "time-penalty": 0.009,
        "action-penalty": 0.0003,
        "saturation-penalty": 0.005,
        "learning-rate": 5e-5,
        "learning-starts": 512,
        "tau": 0.006,
        "gamma": 0.995,
        "seed": 223,
    },
    {
        "name": "mamba_tracking",
        **TURB_BASE,
        "total-timesteps": 1024,
        "cn2": 1e-15,
        "goal-gain": 0.05,
        "hold-target-steps": 4,
        "action-scale": 0.012,
        "time-penalty": 0.010,
        "action-penalty": 0.0003,
        "saturation-penalty": 0.005,
        "learning-rate": 8e-5,
        "learning-starts": 512,
        "tau": 0.008,
        "gamma": 0.992,
        "seed": 227,
    },
    {
        "name": "mamba_low_gain",
        **TURB_BASE,
        "total-timesteps": 1024,
        "cn2": 1e-15,
        "goal-gain": 0.03,
        "hold-target-steps": 3,
        "action-scale": 0.010,
        "time-penalty": 0.008,
        "action-penalty": 0.0002,
        "saturation-penalty": 0.004,
        "learning-rate": 5e-5,
        "learning-starts": 512,
        "tau": 0.006,
        "gamma": 0.995,
        "seed": 229,
    },
]


def _run_training(script: str, args: dict[str, object], log_dir: Path, model_dir: Path) -> None:
    cmd = [PYTHON, script]
    for key, value in args.items():
        cmd.extend([f"--{key}", str(value)])
    cmd.extend(["--log-dir", str(log_dir), "--model-dir", str(model_dir)])
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
        raise RuntimeError(f"Training failed for {log_dir.name}. See stderr.log")


def _load_summary(log_dir: Path) -> dict[str, float]:
    return json.loads((log_dir / "summary.json").read_text(encoding="utf-8"))


def _score(summary: dict[str, float]) -> float:
    return (
        float(summary["mean_reward"])
        + 30.0 * float(summary["mean_final_strehl"])
        + float(summary["mean_best_pib"]) / 1e6
        - 0.5 * float(summary["std_reward"])
    )


def _plot_sweep(rows: list[dict[str, object]], out_dir: Path) -> None:
    names = [row["name"] for row in rows]
    rewards = [float(row["mean_reward"]) for row in rows]
    strehl = [float(row["mean_final_strehl"]) for row in rows]
    score = [float(row["score"]) for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].bar(names, rewards)
    axes[0].set_ylabel("Mean Reward")
    axes[1].bar(names, strehl)
    axes[1].set_ylabel("Final Strehl")
    axes[2].bar(names, score)
    axes[2].set_ylabel("Composite Score")
    axes[2].tick_params(axis="x", rotation=20)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "turbulence_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _read_scalar(log_dir: Path, tag: str) -> tuple[np.ndarray, np.ndarray] | None:
    run_dirs = [path for path in log_dir.iterdir() if path.is_dir()]
    if not run_dirs:
        return None
    ea = event_accumulator.EventAccumulator(str(run_dirs[0]))
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None
    events = ea.Scalars(tag)
    return (
        np.array([event.step for event in events], dtype=float),
        np.array([event.value for event in events], dtype=float),
    )


def _load_eval_rewards(log_dir: Path) -> np.ndarray:
    data = np.load(log_dir / "eval" / "evaluations.npz")
    return data["results"].mean(axis=1)


def _build_env_from_config(log_dir: Path):
    config = json.loads((log_dir / "config.json").read_text(encoding="utf-8"))
    common = {
        "max_steps": config["max_steps"],
        "n_grid": config["n_grid"],
        "n_actuators": config["n_actuators"],
        "n_subapertures": config["n_subapertures"],
        "history_len": config["history_len"],
        "goal_gain": config["goal_gain"],
        "hold_target_steps": config["hold_target_steps"],
        "action_scale": config["action_scale"],
        "time_penalty": config["time_penalty"],
        "action_penalty": config["action_penalty"],
        "saturation_penalty": config["saturation_penalty"],
    }
    if "n_zernike_modes" in config:
        env = build_static_env(
            **common,
            n_zernike_modes=config["n_zernike_modes"],
            zernike_coeff_std=config["zernike_coeff_std"],
            zernike_coeff_clip=config["zernike_coeff_clip"],
        )
        model_path = ROOT / "models" / log_dir.name / "sac_static_final.zip"
        return env, model_path, "static"
    env = build_turb_env(
        **common,
        cn2=config["cn2"],
        screen_step_px=config["screen_step_px"],
        screen_margin_steps=config["screen_margin_steps"],
    )
    model_path = ROOT / "models" / log_dir.name / "sac_turbulence_final.zip"
    return env, model_path, "turbulence"


def _rollout(log_dir: Path, episodes: int = 3) -> pd.DataFrame:
    config = json.loads((log_dir / "config.json").read_text(encoding="utf-8"))
    env, model_path, _ = _build_env_from_config(log_dir)
    model = SAC.load(model_path)
    rows: list[dict[str, float | int]] = []
    for seed_offset in range(episodes):
        obs, info = env.reset(seed=int(config["seed"]) + seed_offset)
        rows.append(
            {
                "episode": seed_offset,
                "step": 0,
                "reward": 0.0,
                "strehl": float(info.get("strehl", 0.0)),
                "pib": float(info.get("pib", 0.0)),
                "rms": float(info.get("rms", 0.0)),
            }
        )
        done = False
        step = 0
        while not done and step < int(config["max_steps"]):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            rows.append(
                {
                    "episode": seed_offset,
                    "step": step,
                    "reward": float(reward),
                    "strehl": float(info.get("strehl", 0.0)),
                    "pib": float(info.get("pib", 0.0)),
                    "rms": float(info.get("rms", 0.0)),
                }
            )
            done = terminated or truncated
    env.close()
    return pd.DataFrame(rows)


def _plot_final_training(log_dir: Path, out_dir: Path, prefix: str) -> dict[str, float | bool]:
    eval_rewards = _load_eval_rewards(log_dir)
    strehl = _read_scalar(log_dir, "ao/strehl_mavg200")
    pib = _read_scalar(log_dir, "ao/pib_mavg200")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(eval_rewards, marker="o")
    axes[0, 0].set_title("Eval Mean Reward")
    if strehl is not None:
        axes[0, 1].plot(strehl[0], strehl[1])
    axes[0, 1].set_title("Strehl Moving Average")
    if pib is not None:
        axes[1, 0].plot(pib[0], pib[1])
    axes[1, 0].set_title("PIB Moving Average")
    axes[1, 1].plot(eval_rewards[-min(5, len(eval_rewards)):], marker="o")
    axes[1, 1].set_title("Late Eval Window")
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_training.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    window = eval_rewards[-min(4, len(eval_rewards)):]
    slope = float(window[-1] - window[0]) / max(len(window) - 1, 1)
    std = float(np.std(window))
    return {
        "late_reward_mean": float(np.mean(window)),
        "late_reward_std": std,
        "late_reward_slope": slope,
        "is_plateau": abs(slope) < 20.0 and std < 20.0,
    }


def _plot_rollout(df: pd.DataFrame, out_dir: Path, prefix: str) -> dict[str, float]:
    grouped = df.groupby("episode")
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for _, group in grouped:
        axes[0].plot(group["step"], group["strehl"], alpha=0.8)
        axes[1].plot(group["step"], group["pib"], alpha=0.8)
        axes[2].plot(group["step"], group["rms"], alpha=0.8)
    axes[0].set_ylabel("Strehl")
    axes[1].set_ylabel("PIB")
    axes[2].set_ylabel("RMS")
    axes[2].set_xlabel("Step")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_rollout_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    df.to_csv(out_dir / f"{prefix}_rollout.csv", index=False)
    return {
        "rollout_final_strehl_mean": float(grouped["strehl"].last().mean()),
        "rollout_best_pib_mean": float(grouped["pib"].max().mean()),
        "rollout_final_rms_mean": float(grouped["rms"].last().mean()),
    }


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / "logs" / f"mamba_turbulence_sweep_{timestamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    sweep_log_root = out_root / "sweep_logs"
    sweep_model_root = out_root / "sweep_models"
    sweep_log_root.mkdir(parents=True, exist_ok=True)
    sweep_model_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for case in TURB_SWEEP:
        log_dir = sweep_log_root / case["name"]
        model_dir = sweep_model_root / case["name"]
        args = {key: value for key, value in case.items() if key != "name"}
        _run_training("src/ao_shaping/optimizer/rl/sac_turbulence_train.py", args, log_dir, model_dir)
        summary = _load_summary(log_dir)
        row = {"name": case["name"], **args, **summary}
        row["score"] = _score(summary)
        rows.append(row)

    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    pd.DataFrame(rows).to_csv(out_root / "sweep_summary.csv", index=False)
    with (out_root / "sweep_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    _plot_sweep(rows, out_root)

    best = rows[0]
    best_args = {key: best[key] for key in TURB_BASE.keys() | {
        "total-timesteps",
        "cn2",
        "goal-gain",
        "hold-target-steps",
        "action-scale",
        "time-penalty",
        "action-penalty",
        "saturation-penalty",
        "learning-rate",
        "learning-starts",
        "tau",
        "gamma",
        "seed",
    }}
    best_args["total-timesteps"] = 2048
    best_args["eval-episodes"] = 6
    best_args["eval-freq"] = 256

    static_log = ROOT / "logs" / f"static_long_{timestamp}"
    static_model = ROOT / "models" / static_log.name
    turb_log = ROOT / "logs" / f"turbulence_mamba_best_{timestamp}"
    turb_model = ROOT / "models" / turb_log.name
    _run_training("src/ao_shaping/optimizer/rl/sac_train.py", STATIC_ARGS, static_log, static_model)
    _run_training("src/ao_shaping/optimizer/rl/sac_turbulence_train.py", best_args, turb_log, turb_model)

    report_dir = ROOT / "logs" / f"sac_mamba_converged_report_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    static_summary = _load_summary(static_log)
    turb_summary = _load_summary(turb_log)
    static_conv = _plot_final_training(static_log, report_dir, "static_long")
    turb_conv = _plot_final_training(turb_log, report_dir, "turbulence_mamba_best")
    static_rollout = _plot_rollout(_rollout(static_log), report_dir, "static_long")
    turb_rollout = _plot_rollout(_rollout(turb_log), report_dir, "turbulence_mamba_best")

    combined = {
        "best_turbulence_config": best_args,
        "sweep_top3": rows[:3],
        "static": {
            **static_summary,
            **static_conv,
            **static_rollout,
            "converged": bool(static_conv["is_plateau"]),
            "log_dir": str(static_log),
            "model_dir": str(static_model),
        },
        "turbulence": {
            **turb_summary,
            **turb_conv,
            **turb_rollout,
            "converged": bool(turb_conv["is_plateau"] and float(turb_summary["mean_final_strehl"]) >= 0.35),
            "log_dir": str(turb_log),
            "model_dir": str(turb_model),
        },
        "one_click": "python scripts/sweep_mamba_turbulence_report.py",
    }
    with (report_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(combined, fh, indent=2, ensure_ascii=False)
    pd.DataFrame(
        [
            {"task": "static", **combined["static"]},
            {"task": "turbulence", **combined["turbulence"]},
        ]
    ).to_csv(report_dir / "summary.csv", index=False)

    lines = [
        "# Mamba Turbulence Sweep Report",
        "",
        f"- Sweep dir: `{out_root}`",
        f"- Report dir: `{report_dir}`",
        f"- Best turbulence config: `{rows[0]['name']}`",
        f"- Static converged: `{combined['static']['converged']}`",
        f"- Turbulence converged: `{combined['turbulence']['converged']}`",
        "",
        "## Best Turbulence Args",
    ]
    for key, value in best_args.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Static",
            f"- mean_reward: `{combined['static']['mean_reward']:.3f}`",
            f"- mean_final_strehl: `{combined['static']['mean_final_strehl']:.3f}`",
            "",
            "## Turbulence",
            f"- mean_reward: `{combined['turbulence']['mean_reward']:.3f}`",
            f"- mean_final_strehl: `{combined['turbulence']['mean_final_strehl']:.3f}`",
            f"- late_reward_std: `{combined['turbulence']['late_reward_std']:.3f}`",
            f"- late_reward_slope: `{combined['turbulence']['late_reward_slope']:.3f}`",
            "",
            "## One-Click",
            f"`{combined['one_click']}`",
        ]
    )
    (report_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(report_dir)


if __name__ == "__main__":
    main()
