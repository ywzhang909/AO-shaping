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

TURB_ARGS = {
    "total-timesteps": 2048,
    "max-steps": 32,
    "n-grid": 32,
    "n-actuators": 4,
    "n-subapertures": 4,
    "history-len": 4,
    "cn2": 1e-15,
    "screen-step-px": 1,
    "screen-margin-steps": 12,
    "goal-gain": 0.06,
    "hold-target-steps": 4,
    "action-scale": 0.015,
    "time-penalty": 0.010,
    "action-penalty": 0.0004,
    "saturation-penalty": 0.006,
    "learning-rate": 8e-5,
    "buffer-size": 150000,
    "batch-size": 64,
    "learning-starts": 256,
    "tau": 0.008,
    "gamma": 0.99,
    "train-freq": 1,
    "gradient-steps": 1,
    "eval-freq": 256,
    "eval-episodes": 6,
    "seed": 103,
}


def _run_training(script: str, log_dir: Path, model_dir: Path, args: dict[str, object]) -> None:
    cmd = [PYTHON, script]
    for key, value in args.items():
        cmd.extend([f"--{key}", str(value)])
    cmd.extend(["--log-dir", str(log_dir), "--model-dir", str(model_dir)])
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
        raise RuntimeError(f"Training failed for {script}. See {log_dir / 'stderr.log'}")


def _format_command(script: str, args: dict[str, object], log_dir: Path, model_dir: Path) -> str:
    parts = ["python", script]
    for key, value in args.items():
        parts.extend([f"--{key}", str(value)])
    parts.extend(["--log-dir", str(log_dir), "--model-dir", str(model_dir)])
    return " ".join(parts)


def _load_eval(log_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(log_dir / "eval" / "evaluations.npz")
    rewards = data["results"].mean(axis=1)
    return data["timesteps"], rewards


def _read_scalar(log_dir: Path, tag: str) -> tuple[np.ndarray, np.ndarray] | None:
    run_dirs = [p for p in log_dir.iterdir() if p.is_dir()]
    if not run_dirs:
        return None
    event_dir = run_dirs[0]
    ea = event_accumulator.EventAccumulator(str(event_dir))
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None
    events = ea.Scalars(tag)
    return (
        np.array([event.step for event in events], dtype=float),
        np.array([event.value for event in events], dtype=float),
    )


def _build_env(log_dir: Path):
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


def _rollout(log_dir: Path, episodes: int = 3) -> dict[str, object]:
    config = json.loads((log_dir / "config.json").read_text(encoding="utf-8"))
    env, model_path, mode = _build_env(log_dir)
    model = SAC.load(model_path)
    trajectories: list[dict[str, list[float]]] = []
    frames: list[np.ndarray] = []
    for seed_offset in range(episodes):
        obs, info = env.reset(seed=int(config["seed"]) + seed_offset)
        frames.append(obs["ccd"][-1])
        trace = {
            "step": [0],
            "reward": [0.0],
            "strehl": [float(info.get("strehl", 0.0))],
            "pib": [float(info.get("pib", 0.0))],
            "rms": [float(info.get("rms", 0.0))],
        }
        done = False
        step = 0
        while not done and step < int(config["max_steps"]):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            if seed_offset == 0:
                frames.append(obs["ccd"][-1])
            trace["step"].append(step)
            trace["reward"].append(float(reward))
            trace["strehl"].append(float(info.get("strehl", 0.0)))
            trace["pib"].append(float(info.get("pib", 0.0)))
            trace["rms"].append(float(info.get("rms", 0.0)))
            done = terminated or truncated
        trajectories.append(trace)
    env.close()
    return {"mode": mode, "trajectories": trajectories, "frames": frames}


def _convergence_metrics(
    rewards: np.ndarray,
    *,
    mode: str,
    mean_final_strehl: float,
) -> dict[str, float | bool]:
    window = rewards[-3:] if rewards.size >= 3 else rewards
    slope = float(window[-1] - window[0]) / max(len(window) - 1, 1)
    std = float(np.std(window))
    if mode == "static":
        is_plateau = abs(slope) < 8.0 and std < 35.0
    else:
        is_plateau = abs(slope) < 20.0 and std < 20.0 and mean_final_strehl >= 0.35
    return {
        "late_reward_mean": float(np.mean(window)),
        "late_reward_std": std,
        "late_reward_slope": slope,
        "is_plateau": is_plateau,
    }


def _plot_training(
    log_dir: Path,
    out_dir: Path,
    label: str,
    *,
    mode: str,
    mean_final_strehl: float,
) -> dict[str, float | bool]:
    timesteps, rewards = _load_eval(log_dir)
    strehl = _read_scalar(log_dir, "ao/strehl_mavg200")
    pib = _read_scalar(log_dir, "ao/pib_mavg200")
    actor = _read_scalar(log_dir, "train/actor_loss")
    critic = _read_scalar(log_dir, "train/critic_loss")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(timesteps, rewards, marker="o")
    axes[0, 0].set_title("Eval Mean Reward")
    if strehl is not None:
        axes[0, 1].plot(strehl[0], strehl[1])
    axes[0, 1].set_title("AO Strehl Moving Average")
    if pib is not None:
        axes[1, 0].plot(pib[0], pib[1])
    axes[1, 0].set_title("AO PIB Moving Average")
    if actor is not None and critic is not None:
        axes[1, 1].plot(actor[0], actor[1], label="actor")
        axes[1, 1].plot(critic[0], critic[1], label="critic")
        axes[1, 1].legend()
    axes[1, 1].set_title("Loss")
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{label}_training.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return _convergence_metrics(rewards, mode=mode, mean_final_strehl=mean_final_strehl)


def _plot_rollout(rollout: dict[str, object], out_dir: Path, label: str) -> pd.DataFrame:
    trajectories = rollout["trajectories"]
    frames = rollout["frames"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    rows: list[dict[str, float | int]] = []
    for idx, trace in enumerate(trajectories):
        axes[0].plot(trace["step"], trace["strehl"], alpha=0.8, label=f"ep{idx}")
        axes[1].plot(trace["step"], trace["pib"], alpha=0.8)
        axes[2].plot(trace["step"], trace["rms"], alpha=0.8)
        for step, reward, strehl, pib, rms in zip(
            trace["step"], trace["reward"], trace["strehl"], trace["pib"], trace["rms"]
        ):
            rows.append(
                {
                    "episode": idx,
                    "step": step,
                    "reward": reward,
                    "strehl": strehl,
                    "pib": pib,
                    "rms": rms,
                }
            )
    axes[0].set_ylabel("Strehl")
    axes[1].set_ylabel("PIB")
    axes[2].set_ylabel("Residual RMS")
    axes[2].set_xlabel("Step")
    axes[0].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{label}_rollout_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    idx = np.linspace(0, len(frames) - 1, min(6, len(frames)), dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for ax, frame_idx in zip(axes.flat, idx):
        ax.imshow(frames[frame_idx], cmap="inferno")
        ax.set_title(f"frame={frame_idx}")
        ax.axis("off")
    for ax in axes.flat[len(idx):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"{label}_rollout_frames.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{label}_rollout.csv", index=False)
    return df


def _summarize_rollout(df: pd.DataFrame) -> dict[str, float]:
    grouped = df.groupby("episode")
    return {
        "rollout_final_strehl_mean": float(grouped["strehl"].last().mean()),
        "rollout_best_pib_mean": float(grouped["pib"].max().mean()),
        "rollout_final_rms_mean": float(grouped["rms"].last().mean()),
    }


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = ROOT / "logs" / f"sac_converged_report_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    static_log = ROOT / "logs" / f"static_long_{timestamp}"
    turb_log = ROOT / "logs" / f"turbulence_long_{timestamp}"
    static_model = ROOT / "models" / static_log.name
    turb_model = ROOT / "models" / turb_log.name
    static_log.mkdir(parents=True, exist_ok=True)
    turb_log.mkdir(parents=True, exist_ok=True)
    static_model.mkdir(parents=True, exist_ok=True)
    turb_model.mkdir(parents=True, exist_ok=True)

    _run_training("src/ao_shaping/optimizer/rl/sac_train.py", static_log, static_model, STATIC_ARGS)
    _run_training("src/ao_shaping/optimizer/rl/sac_turbulence_train.py", turb_log, turb_model, TURB_ARGS)

    static_summary = json.loads((static_log / "summary.json").read_text(encoding="utf-8"))
    turb_summary = json.loads((turb_log / "summary.json").read_text(encoding="utf-8"))
    static_conv = _plot_training(
        static_log,
        report_dir,
        "static_long",
        mode="static",
        mean_final_strehl=float(static_summary["mean_final_strehl"]),
    )
    turb_conv = _plot_training(
        turb_log,
        report_dir,
        "turbulence_long",
        mode="turbulence",
        mean_final_strehl=float(turb_summary["mean_final_strehl"]),
    )

    static_rollout = _rollout(static_log)
    turb_rollout = _rollout(turb_log)
    static_rollout_df = _plot_rollout(static_rollout, report_dir, "static_long")
    turb_rollout_df = _plot_rollout(turb_rollout, report_dir, "turbulence_long")
    static_rollout_summary = _summarize_rollout(static_rollout_df)
    turb_rollout_summary = _summarize_rollout(turb_rollout_df)

    combined = {
        "static": {
            **static_summary,
            **static_conv,
            **static_rollout_summary,
            "log_dir": str(static_log),
            "model_dir": str(static_model),
            "converged": bool(static_conv["is_plateau"]),
        },
        "turbulence": {
            **turb_summary,
            **turb_conv,
            **turb_rollout_summary,
            "log_dir": str(turb_log),
            "model_dir": str(turb_model),
            "converged": bool(turb_conv["is_plateau"]),
        },
        "commands": {
            "static": "python scripts/run_long_sac_experiments.py",
            "direct_static": _format_command("src/ao_shaping/optimizer/rl/sac_train.py", STATIC_ARGS, static_log, static_model),
            "direct_turbulence": _format_command("src/ao_shaping/optimizer/rl/sac_turbulence_train.py", TURB_ARGS, turb_log, turb_model),
        },
    }

    with (report_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(combined, fh, indent=2, ensure_ascii=False)

    summary_rows = pd.DataFrame(
        [
            {"task": "static", **combined["static"]},
            {"task": "turbulence", **combined["turbulence"]},
        ]
    )
    summary_rows.to_csv(report_dir / "summary.csv", index=False)

    report_lines = [
        "# SAC Convergence Report",
        "",
        f"- Report dir: `{report_dir}`",
        f"- Static converged: `{combined['static']['converged']}`",
        f"- Turbulence converged: `{combined['turbulence']['converged']}`",
        "",
        "## Static",
        f"- mean_reward: `{combined['static']['mean_reward']:.3f}`",
        f"- mean_final_strehl: `{combined['static']['mean_final_strehl']:.3f}`",
        f"- mean_best_pib: `{combined['static']['mean_best_pib']:.1f}`",
        f"- rollout_final_strehl_mean: `{combined['static']['rollout_final_strehl_mean']:.3f}`",
        f"- late_reward_mean: `{combined['static']['late_reward_mean']:.3f}`",
        f"- late_reward_std: `{combined['static']['late_reward_std']:.3f}`",
        f"- late_reward_slope: `{combined['static']['late_reward_slope']:.3f}`",
        "",
        "## Turbulence",
        f"- mean_reward: `{combined['turbulence']['mean_reward']:.3f}`",
        f"- mean_final_strehl: `{combined['turbulence']['mean_final_strehl']:.3f}`",
        f"- mean_best_pib: `{combined['turbulence']['mean_best_pib']:.1f}`",
        f"- rollout_final_strehl_mean: `{combined['turbulence']['rollout_final_strehl_mean']:.3f}`",
        f"- late_reward_mean: `{combined['turbulence']['late_reward_mean']:.3f}`",
        f"- late_reward_std: `{combined['turbulence']['late_reward_std']:.3f}`",
        f"- late_reward_slope: `{combined['turbulence']['late_reward_slope']:.3f}`",
        "",
        "## One-Click",
        f"`{combined['commands']['static']}`",
    ]
    (report_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(report_dir)


if __name__ == "__main__":
    main()
