"""Train SAC for closed-loop AO correction under slowly drifting turbulence."""

from __future__ import annotations

from pathlib import Path
import json
import sys

import click
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

from ao_shaping.optimizer.rl.device_registry import build_default_registry
from ao_shaping.optimizer.rl.envs import SimTurbulenceAOEnv
from ao_shaping.optimizer.rl.sac_common import (
    AOTrainingCallback,
    TemporalAOExtractor,
    evaluate_model,
    save_evaluation_artifacts,
    validate_env,
    wrap_monitor,
)


def _warn_if_physical_selected(dm_device: str, ccd_device: str, wfs_device: str) -> None:
    registry = build_default_registry()
    specs = [
        registry.get("dm", dm_device),
        registry.get("ccd", ccd_device),
        registry.get("wfs", wfs_device),
    ]
    physical = [spec for spec in specs if not spec.is_virtual]
    if physical:
        names = ", ".join(f"{spec.device_type}:{spec.name}" for spec in physical)
        click.echo(
            f"[warning] selected physical devices ({names}), but this trainer uses simulation only.",
            err=True,
        )


def build_env(
    *,
    max_steps: int,
    n_grid: int,
    n_actuators: int,
    n_subapertures: int,
    history_len: int,
    cn2: float,
    screen_step_px: int,
    screen_margin_steps: int,
    goal_gain: float,
    hold_target_steps: int,
    action_scale: float,
    time_penalty: float,
    action_penalty: float,
    saturation_penalty: float,
):
    env = SimTurbulenceAOEnv(
        max_steps=max_steps,
        n_grid=n_grid,
        n_actuators=n_actuators,
        n_subapertures=n_subapertures,
        history_len=history_len,
        cn2=cn2,
        screen_step_px=screen_step_px,
        screen_margin_steps=screen_margin_steps,
        goal_gain=goal_gain,
        hold_target_steps=hold_target_steps,
        action_scale=action_scale,
        time_penalty=time_penalty,
        action_penalty=action_penalty,
        saturation_penalty=saturation_penalty,
    )
    return wrap_monitor(env)


@click.command()
@click.option("--total-timesteps", type=int, default=120_000, show_default=True)
@click.option("--max-steps", type=int, default=80, show_default=True)
@click.option("--n-grid", type=int, default=64, show_default=True)
@click.option("--n-actuators", type=int, default=6, show_default=True)
@click.option("--n-subapertures", type=int, default=6, show_default=True)
@click.option("--history-len", type=int, default=8, show_default=True)
@click.option("--cn2", type=float, default=1e-15, show_default=True)
@click.option("--screen-step-px", type=int, default=1, show_default=True)
@click.option("--screen-margin-steps", type=int, default=12, show_default=True)
@click.option("--goal-gain", type=float, default=0.06, show_default=True)
@click.option("--hold-target-steps", type=int, default=4, show_default=True)
@click.option("--action-scale", type=float, default=0.015, show_default=True)
@click.option("--time-penalty", type=float, default=0.010, show_default=True)
@click.option("--action-penalty", type=float, default=0.0004, show_default=True)
@click.option("--saturation-penalty", type=float, default=0.006, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--learning-rate", type=float, default=8e-5, show_default=True)
@click.option("--buffer-size", type=int, default=150_000, show_default=True)
@click.option("--batch-size", type=int, default=64, show_default=True)
@click.option("--learning-starts", type=int, default=6_000, show_default=True)
@click.option("--tau", type=float, default=0.008, show_default=True)
@click.option("--gamma", type=float, default=0.99, show_default=True)
@click.option("--train-freq", type=int, default=4, show_default=True)
@click.option("--gradient-steps", type=int, default=4, show_default=True)
@click.option("--eval-freq", type=int, default=4_000, show_default=True)
@click.option("--eval-episodes", type=int, default=5, show_default=True)
@click.option("--log-dir", type=str, default="logs/sac_turbulence", show_default=True)
@click.option("--model-dir", type=str, default="models/sac_turbulence", show_default=True)
@click.option(
    "--dm-device",
    type=click.Choice(build_default_registry().names("dm"), case_sensitive=False),
    default="sim_dm",
    show_default=True,
)
@click.option(
    "--ccd-device",
    type=click.Choice(build_default_registry().names("ccd"), case_sensitive=False),
    default="sim_ccd",
    show_default=True,
)
@click.option(
    "--wfs-device",
    type=click.Choice(build_default_registry().names("wfs"), case_sensitive=False),
    default="sim_wfs",
    show_default=True,
)
def main(
    total_timesteps: int,
    max_steps: int,
    n_grid: int,
    n_actuators: int,
    n_subapertures: int,
    history_len: int,
    cn2: float,
    screen_step_px: int,
    screen_margin_steps: int,
    goal_gain: float,
    hold_target_steps: int,
    action_scale: float,
    time_penalty: float,
    action_penalty: float,
    saturation_penalty: float,
    seed: int,
    learning_rate: float,
    buffer_size: int,
    batch_size: int,
    learning_starts: int,
    tau: float,
    gamma: float,
    train_freq: int,
    gradient_steps: int,
    eval_freq: int,
    eval_episodes: int,
    log_dir: str,
    model_dir: str,
    dm_device: str,
    ccd_device: str,
    wfs_device: str,
) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    _warn_if_physical_selected(dm_device=dm_device, ccd_device=ccd_device, wfs_device=wfs_device)

    log_path = Path(log_dir)
    model_path = Path(model_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)

    config = {
        "total_timesteps": total_timesteps,
        "max_steps": max_steps,
        "n_grid": n_grid,
        "n_actuators": n_actuators,
        "n_subapertures": n_subapertures,
        "history_len": history_len,
        "cn2": cn2,
        "screen_step_px": screen_step_px,
        "screen_margin_steps": screen_margin_steps,
        "goal_gain": goal_gain,
        "hold_target_steps": hold_target_steps,
        "action_scale": action_scale,
        "time_penalty": time_penalty,
        "action_penalty": action_penalty,
        "saturation_penalty": saturation_penalty,
        "seed": seed,
        "learning_rate": learning_rate,
        "buffer_size": buffer_size,
        "batch_size": batch_size,
        "learning_starts": learning_starts,
        "tau": tau,
        "gamma": gamma,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "eval_freq": eval_freq,
        "eval_episodes": eval_episodes,
    }
    with (log_path / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)

    env = build_env(
        max_steps=max_steps,
        n_grid=n_grid,
        n_actuators=n_actuators,
        n_subapertures=n_subapertures,
        history_len=history_len,
        cn2=cn2,
        screen_step_px=screen_step_px,
        screen_margin_steps=screen_margin_steps,
        goal_gain=goal_gain,
        hold_target_steps=hold_target_steps,
        action_scale=action_scale,
        time_penalty=time_penalty,
        action_penalty=action_penalty,
        saturation_penalty=saturation_penalty,
    )
    validate_env(env.env, seed=seed)
    eval_env = build_env(
        max_steps=max_steps,
        n_grid=n_grid,
        n_actuators=n_actuators,
        n_subapertures=n_subapertures,
        history_len=history_len,
        cn2=cn2,
        screen_step_px=screen_step_px,
        screen_margin_steps=screen_margin_steps,
        goal_gain=goal_gain,
        hold_target_steps=hold_target_steps,
        action_scale=action_scale,
        time_penalty=time_penalty,
        action_penalty=action_penalty,
        saturation_penalty=saturation_penalty,
    )

    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        learning_starts=learning_starts,
        train_freq=(train_freq, "step"),
        gradient_steps=gradient_steps,
        gamma=gamma,
        tau=tau,
        ent_coef="auto_0.05",
        target_update_interval=1,
        use_sde=True,
        target_entropy="auto",
        tensorboard_log=str(log_path),
        seed=seed,
        verbose=1,
        policy_kwargs={
            "features_extractor_class": TemporalAOExtractor,
            "features_extractor_kwargs": {
                "features_dim": 256,
                "cnn_dim": 96,
                "recurrent_dim": 128,
            },
            "net_arch": {"pi": [256, 256], "qf": [256, 256]},
        },
    )

    callbacks = CallbackList([
        AOTrainingCallback(log_every=50, image_every=500),
        EvalCallback(
            eval_env,
            best_model_save_path=str(model_path / "best"),
            log_path=str(log_path / "eval"),
            eval_freq=max(eval_freq, 1),
            n_eval_episodes=eval_episodes,
            deterministic=True,
            render=False,
        ),
    ])

    model.learn(total_timesteps=total_timesteps, callback=callbacks, tb_log_name="moving_turbulence")
    model.save(model_path / "sac_turbulence_final")

    results = evaluate_model(model, eval_env, episodes=eval_episodes, deterministic=True)
    save_evaluation_artifacts(results, log_path, prefix="moving_turbulence")
    with (log_path / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in results.items() if k != "trajectories"}, fh, indent=2, ensure_ascii=False)

    click.echo(
        "Moving-turbulence SAC finished: "
        f"mean_reward={results['mean_reward']:.3f}, "
        f"mean_final_strehl={results['mean_final_strehl']:.3f}, "
        f"mean_best_pib={results['mean_best_pib']:.3f}"
    )

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
