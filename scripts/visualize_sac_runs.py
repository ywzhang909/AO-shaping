from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from tensorboard.backend.event_processing import event_accumulator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.optimizer.rl.sac_train import build_env as build_static_env
from ao_shaping.optimizer.rl.sac_turbulence_train import build_env as build_turb_env


def _find_event_dir(log_dir: Path) -> Path | None:
    candidates = [path for path in log_dir.iterdir() if path.is_dir() and any(child.name.startswith("events.out.tfevents") for child in path.iterdir())]
    return candidates[0] if candidates else None


def _read_scalars(log_dir: Path, tags: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    event_dir = _find_event_dir(log_dir)
    if event_dir is None:
        return {}
    ea = event_accumulator.EventAccumulator(str(event_dir))
    ea.Reload()
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tag in tags:
        if tag not in ea.Tags().get("scalars", []):
            continue
        events = ea.Scalars(tag)
        steps = np.array([event.step for event in events], dtype=float)
        values = np.array([event.value for event in events], dtype=float)
        out[tag] = (steps, values)
    return out


def _load_eval(log_dir: Path) -> dict[str, np.ndarray] | None:
    path = log_dir / "eval" / "evaluations.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return {key: data[key] for key in data.files}


def _infer_model_path(log_dir: Path) -> Path | None:
    model_root = ROOT / "models" / log_dir.name
    for name in ["sac_static_final.zip", "sac_turbulence_final.zip"]:
        path = model_root / name
        if path.exists():
            return path
    return None


def _build_env_from_config(config: dict) -> tuple[object, str]:
    def cfg(name: str, default):
        return config.get(name, default)

    common = {
        "max_steps": config["max_steps"],
        "n_grid": config["n_grid"],
        "n_actuators": config["n_actuators"],
        "n_subapertures": config["n_subapertures"],
        "history_len": config["history_len"],
        "goal_gain": cfg("goal_gain", 0.15 if "n_zernike_modes" in config else 0.10),
        "hold_target_steps": cfg("hold_target_steps", 4 if "n_zernike_modes" in config else 5),
        "action_scale": cfg("action_scale", 0.02),
        "time_penalty": cfg("time_penalty", 0.006 if "n_zernike_modes" in config else 0.012),
        "action_penalty": cfg("action_penalty", 0.0005 if "n_zernike_modes" in config else 0.0006),
        "saturation_penalty": cfg("saturation_penalty", 0.008),
    }
    if "n_zernike_modes" in config:
        env = build_static_env(
            **common,
            n_zernike_modes=config["n_zernike_modes"],
            zernike_coeff_std=config["zernike_coeff_std"],
            zernike_coeff_clip=config["zernike_coeff_clip"],
        )
        return env, "static"
    env = build_turb_env(
        **common,
        cn2=config["cn2"],
        screen_step_px=config["screen_step_px"],
        screen_margin_steps=config["screen_margin_steps"],
    )
    return env, "turbulence"


def _rollout(log_dir: Path, steps_limit: int = 64) -> dict[str, object] | None:
    config_path = log_dir / "config.json"
    model_path = _infer_model_path(log_dir)
    if not config_path.exists() or model_path is None:
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    env, mode = _build_env_from_config(config)
    model = SAC.load(model_path)
    obs, info = env.reset(seed=config.get("seed", 42))
    frames = [obs["ccd"][-1]]
    trace = {
        "step": [0],
        "reward": [0.0],
        "strehl": [float(info.get("strehl", 0.0))],
        "pib": [float(info.get("pib", 0.0))],
        "rms": [float(info.get("rms", 0.0))],
    }
    done = False
    step = 0
    while not done and step < steps_limit:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        frames.append(obs["ccd"][-1])
        trace["step"].append(step)
        trace["reward"].append(float(reward))
        trace["strehl"].append(float(info.get("strehl", 0.0)))
        trace["pib"].append(float(info.get("pib", 0.0)))
        trace["rms"].append(float(info.get("rms", 0.0)))
        done = terminated or truncated
    env.close()
    return {"mode": mode, "trace": trace, "frames": frames}


def _plot_training(run_name: str, scalars: dict[str, tuple[np.ndarray, np.ndarray]], out_dir: Path) -> None:
    tags = [
        "eval/mean_reward",
        "ao/strehl_mavg200",
        "ao/pib_mavg200",
        "train/actor_loss",
        "train/critic_loss",
        "train/ent_coef",
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for ax, tag in zip(axes.flat, tags):
        if tag in scalars:
            steps, values = scalars[tag]
            ax.plot(steps, values, linewidth=1.5)
        ax.set_title(tag)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Training Curves - {run_name}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_training.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_eval(run_name: str, eval_data: dict[str, np.ndarray] | None, out_dir: Path) -> None:
    if eval_data is None:
        return
    timesteps = eval_data["timesteps"]
    rewards = eval_data["results"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(timesteps, rewards.mean(axis=1), marker="o")
    axes[0].fill_between(
        timesteps,
        rewards.mean(axis=1) - rewards.std(axis=1),
        rewards.mean(axis=1) + rewards.std(axis=1),
        alpha=0.2,
    )
    axes[0].set_ylabel("Eval Reward")
    axes[1].plot(timesteps, eval_data["ep_lengths"].mean(axis=1), marker="o")
    axes[1].set_ylabel("Episode Length")
    axes[1].set_xlabel("Timesteps")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"Eval Curves - {run_name}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_eval.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_rollout(run_name: str, rollout: dict[str, object] | None, out_dir: Path) -> None:
    if rollout is None:
        return
    trace = rollout["trace"]
    frames = rollout["frames"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(trace["step"], trace["strehl"], label="Strehl")
    axes[1].plot(trace["step"], trace["pib"], label="PIB")
    axes[2].plot(trace["step"], trace["rms"], label="RMS")
    axes[2].plot(trace["step"], trace["reward"], label="Reward", alpha=0.6)
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[2].set_xlabel("Step")
    fig.suptitle(f"Rollout Metrics - {run_name}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_rollout_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    idx = np.linspace(0, len(frames) - 1, min(6, len(frames)), dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for ax, frame_idx in zip(axes.flat, idx):
        ax.imshow(frames[frame_idx], cmap="inferno")
        ax.set_title(f"step={frame_idx}")
        ax.axis("off")
    for ax in axes.flat[len(idx):]:
        ax.axis("off")
    fig.suptitle(f"Rollout Frames - {run_name}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_rollout_frames.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(trace).to_csv(out_dir / f"{run_name}_rollout.csv", index=False)


def main() -> None:
    log_roots = [
        ROOT / "logs" / "sac_static_smoke",
        ROOT / "logs" / "sac_turbulence_smoke",
    ]
    log_roots.extend(sorted((ROOT / "logs").glob("static_focus_*")))
    log_roots.extend(sorted((ROOT / "logs").glob("turb_focus_*")))
    sweep_dirs = sorted((ROOT / "logs").glob("sac_sweep_*"))
    if sweep_dirs:
        newest = sweep_dirs[-1]
        log_roots.extend(sorted((newest / "logs").iterdir()))

    out_dir = ROOT / "logs" / f"sac_visual_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for log_dir in log_roots:
        if not log_dir.exists() or not log_dir.is_dir():
            continue
        run_name = log_dir.name
        scalars = _read_scalars(
            log_dir,
            tags=[
                "eval/mean_reward",
                "ao/strehl_mavg200",
                "ao/pib_mavg200",
                "train/actor_loss",
                "train/critic_loss",
                "train/ent_coef",
            ],
        )
        eval_data = _load_eval(log_dir)
        rollout = _rollout(log_dir)
        summary = {}
        summary_path = log_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        _plot_training(run_name, scalars, out_dir)
        _plot_eval(run_name, eval_data, out_dir)
        _plot_rollout(run_name, rollout, out_dir)

        row = {"run_name": run_name, "log_dir": str(log_dir)}
        row.update(summary)
        if rollout is not None:
            row["rollout_final_strehl"] = rollout["trace"]["strehl"][-1]
            row["rollout_best_pib"] = max(rollout["trace"]["pib"])
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary_rows, fh, indent=2, ensure_ascii=False)

    report_lines = [
        "# SAC Visualization Report",
        "",
        f"- Output: `{out_dir}`",
        "",
        "## Runs",
    ]
    for row in summary_rows:
        report_lines.append(
            f"- `{row['run_name']}`: mean_reward={row.get('mean_reward', 'n/a')}, "
            f"mean_final_strehl={row.get('mean_final_strehl', 'n/a')}, "
            f"mean_best_pib={row.get('mean_best_pib', 'n/a')}"
        )
    (out_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONPATH", "src;.venv\\Lib\\site-packages")
    main()
