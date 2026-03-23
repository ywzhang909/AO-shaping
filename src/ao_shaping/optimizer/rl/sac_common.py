"""Shared SAC utilities for AO simulation training."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import json

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure, Image
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TemporalAOExtractor(BaseFeaturesExtractor):
    """Compact temporal encoder for AO dict observations."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        cnn_dim: int = 96,
        recurrent_dim: int = 128,
    ) -> None:
        super().__init__(observation_space, features_dim=features_dim)
        slopes_dim = observation_space["hartmann_slopes"].shape[-1]
        dm_dim = observation_space["dm_signal"].shape[-1]
        metrics_dim = observation_space["metrics"].shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, cnn_dim),
            nn.LayerNorm(cnn_dim),
            nn.SiLU(),
        )
        self.slopes_encoder = nn.Sequential(
            nn.Linear(slopes_dim, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
        )
        self.dm_encoder = nn.Sequential(
            nn.Linear(dm_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
        )
        self.metrics_encoder = nn.Sequential(
            nn.Linear(metrics_dim, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
        )
        self.temporal_gru = nn.GRU(
            input_size=cnn_dim + 96 + 64,
            hidden_size=recurrent_dim,
            batch_first=True,
        )
        self.output_head = nn.Sequential(
            nn.Linear(recurrent_dim + 32, features_dim),
            nn.LayerNorm(features_dim),
            nn.SiLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        ccd = observations["ccd"]
        slopes = observations["hartmann_slopes"]
        dm_signal = observations["dm_signal"]
        metrics = observations["metrics"]

        batch, time_steps, height, width = ccd.shape
        ccd_feat = self.cnn(ccd.reshape(batch * time_steps, 1, height, width)).view(
            batch, time_steps, -1
        )
        slopes_feat = self.slopes_encoder(slopes)
        dm_feat = self.dm_encoder(dm_signal)
        temporal_input = torch.cat([ccd_feat, slopes_feat, dm_feat], dim=-1)
        _, hidden = self.temporal_gru(temporal_input)
        metrics_feat = self.metrics_encoder(metrics)
        return self.output_head(torch.cat([hidden[-1], metrics_feat], dim=-1))


class AOTrainingCallback(BaseCallback):
    """TensorBoard logger for AO-specific metrics and frames."""

    def __init__(self, log_every: int = 50, image_every: int = 500, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_every = log_every
        self.image_every = image_every
        self.reward_window: deque[float] = deque(maxlen=200)
        self.pib_window: deque[float] = deque(maxlen=200)
        self.strehl_window: deque[float] = deque(maxlen=200)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        actions = self.locals.get("actions", [])
        new_obs = self.locals.get("new_obs", {})

        if not infos:
            return True

        info = infos[-1]
        reward = float(rewards[-1]) if len(rewards) else 0.0
        self.reward_window.append(reward)
        self.pib_window.append(float(info.get("pib", 0.0)))
        self.strehl_window.append(float(info.get("strehl", 0.0)))

        if self.num_timesteps % self.log_every == 0:
            self.logger.record("ao/reward_mavg200", float(np.mean(self.reward_window)))
            self.logger.record("ao/pib", float(info.get("pib", 0.0)))
            self.logger.record("ao/pib_mavg200", float(np.mean(self.pib_window)))
            self.logger.record("ao/best_pib", float(info.get("best_pib", 0.0)))
            self.logger.record("ao/pib_target", float(info.get("pib_target", 0.0)))
            self.logger.record("ao/strehl", float(info.get("strehl", 0.0)))
            self.logger.record("ao/strehl_mavg200", float(np.mean(self.strehl_window)))
            self.logger.record("ao/best_strehl", float(info.get("best_strehl", 0.0)))
            self.logger.record("ao/rms", float(info.get("rms", 0.0)))
            self.logger.record("ao/disturbance_rms", float(info.get("disturbance_rms", 0.0)))
            self.logger.record("ao/success_streak", float(info.get("success_streak", 0)))

            if len(actions):
                act = np.asarray(actions[-1], dtype=float)
                self.logger.record("ao/action_l2", float(np.linalg.norm(act)))
                self.logger.record("ao/action_abs_mean", float(np.mean(np.abs(act))))

        if self.num_timesteps % self.image_every == 0 and isinstance(new_obs, dict):
            ccd = new_obs.get("ccd")
            slopes = new_obs.get("hartmann_slopes")

            if ccd is not None:
                ccd_arr = np.asarray(ccd, dtype=np.float32)
                if ccd_arr.ndim == 4:
                    ccd_img = ccd_arr[-1, -1]
                elif ccd_arr.ndim == 3:
                    ccd_img = ccd_arr[-1]
                else:
                    ccd_img = ccd_arr
                ccd_uint8 = np.clip(ccd_img * 255.0, 0, 255).astype(np.uint8)
                self.logger.record("ao/ccd", Image(ccd_uint8, "HW"), exclude=("stdout", "log", "json", "csv"))

            if slopes is not None:
                slopes_arr = np.asarray(slopes, dtype=np.float32)
                if slopes_arr.ndim == 3:
                    slopes_vec = slopes_arr[-1, -1]
                elif slopes_arr.ndim == 2:
                    slopes_vec = slopes_arr[-1]
                else:
                    slopes_vec = slopes_arr
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.plot(slopes_vec, linewidth=1.0)
                ax.set_title("Hartmann Slopes")
                ax.grid(alpha=0.3)
                self.logger.record("ao/slopes", Figure(fig, close=True), exclude=("stdout", "log", "json", "csv"))
        return True


def wrap_monitor(env: gym.Env) -> Monitor:
    return Monitor(env)


def validate_env(env: gym.Env, seed: int) -> None:
    obs, _ = env.reset(seed=seed)
    if not env.observation_space.contains(obs):
        raise ValueError("Reset observation is outside the declared observation space.")
    obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    if not env.observation_space.contains(obs):
        raise ValueError("Step observation is outside the declared observation space.")
    if not np.isfinite(reward):
        raise ValueError("Environment returned a non-finite reward.")
    if not isinstance(terminated, bool) or not isinstance(truncated, bool):
        raise TypeError("Environment terminated/truncated flags must be bool.")


def evaluate_model(model, env: gym.Env, episodes: int, deterministic: bool = True) -> dict[str, object]:
    trajectories: list[dict[str, list[float]]] = []
    rewards: list[float] = []
    final_strehls: list[float] = []
    best_pibs: list[float] = []

    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        traj = {
            "reward": [],
            "strehl": [float(info.get("strehl", 0.0))],
            "pib": [float(info.get("pib", 0.0))],
            "rms": [float(info.get("rms", 0.0))],
        }

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            traj["reward"].append(float(reward))
            traj["strehl"].append(float(info.get("strehl", 0.0)))
            traj["pib"].append(float(info.get("pib", 0.0)))
            traj["rms"].append(float(info.get("rms", 0.0)))
            done = terminated or truncated

        trajectories.append(traj)
        rewards.append(ep_reward)
        final_strehls.append(traj["strehl"][-1])
        best_pibs.append(max(traj["pib"]))

    return {
        "episodes": episodes,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "mean_final_strehl": float(np.mean(final_strehls)) if final_strehls else 0.0,
        "mean_best_pib": float(np.mean(best_pibs)) if best_pibs else 0.0,
        "trajectories": trajectories,
    }


def save_evaluation_artifacts(results: dict[str, object], output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories = results.get("trajectories", [])
    if not trajectories:
        return

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for traj in trajectories:
        axes[0].plot(traj["strehl"], alpha=0.7)
        axes[1].plot(traj["pib"], alpha=0.7)
        axes[2].plot(traj["rms"], alpha=0.7)
    axes[0].set_ylabel("Strehl")
    axes[1].set_ylabel("PIB")
    axes[2].set_ylabel("Residual RMS")
    axes[2].set_xlabel("Step")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {k: v for k, v in results.items() if k != "trajectories"}
    with (output_dir / f"{prefix}_evaluation.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
