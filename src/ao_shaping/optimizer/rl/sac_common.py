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
from torch.nn import functional as F
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


class SelectiveStateSpaceBlock(nn.Module):
    """Lightweight Mamba-style selective scan block for temporal fusion."""

    def __init__(
        self,
        d_model: int,
        expand: int = 2,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        inner_dim = d_model * expand
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, inner_dim * 2)
        self.depthwise_conv = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=inner_dim,
        )
        self.dt_proj = nn.Linear(inner_dim, inner_dim)
        self.b_proj = nn.Linear(inner_dim, inner_dim)
        self.c_proj = nn.Linear(inner_dim, inner_dim)
        self.out_proj = nn.Linear(inner_dim, d_model)
        self.skip = nn.Parameter(torch.ones(inner_dim))
        self.a_log = nn.Parameter(torch.zeros(inner_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = self.depthwise_conv(u.transpose(1, 2)).transpose(1, 2)
        u = F.silu(u[:, : x.shape[1], :])
        dt = F.softplus(self.dt_proj(u)) + 1e-4
        b = torch.sigmoid(self.b_proj(u))
        c = torch.tanh(self.c_proj(u))
        a = -torch.exp(self.a_log).unsqueeze(0)

        state = torch.zeros(u.shape[0], u.shape[-1], device=u.device, dtype=u.dtype)
        outputs: list[torch.Tensor] = []
        for step in range(u.shape[1]):
            dt_t = dt[:, step, :]
            u_t = u[:, step, :]
            decay = torch.exp(a * dt_t)
            state = decay * state + dt_t * b[:, step, :] * u_t
            y_t = c[:, step, :] * state + self.skip * u_t
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        y = y * torch.sigmoid(gate)
        return residual + self.out_proj(y)


class MultiScalePatchTokenizer(nn.Module):
    """Encode each CCD frame into multi-scale patch tokens plus an attention summary."""

    def __init__(
        self,
        token_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, token_dim, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )
        self.local_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.global_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.local_norm = nn.LayerNorm(token_dim)
        self.global_norm = nn.LayerNorm(token_dim)
        self.summary_query = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)
        self.summary_attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.stem(frames)
        local_tokens = self.local_pool(features).flatten(2).transpose(1, 2)
        global_tokens = self.global_pool(features).flatten(2).transpose(1, 2)
        local_tokens = self.local_norm(local_tokens)
        global_tokens = self.global_norm(global_tokens)
        patch_tokens = torch.cat([local_tokens, global_tokens], dim=1)
        query = self.summary_query.expand(frames.shape[0], -1, -1)
        summary, _ = self.summary_attention(query=query, key=patch_tokens, value=patch_tokens, need_weights=False)
        return patch_tokens, summary[:, 0]


class MambaCrossAttentionTemporalAOExtractor(BaseFeaturesExtractor):
    """Cross-attention + Mamba-style temporal fusion for turbulence SAC."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 384,
        token_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(observation_space, features_dim=features_dim)
        slopes_dim = observation_space["hartmann_slopes"].shape[-1]
        dm_dim = observation_space["dm_signal"].shape[-1]
        metrics_dim = observation_space["metrics"].shape[0]

        self.ccd_tokenizer = MultiScalePatchTokenizer(token_dim=token_dim, num_heads=num_heads)
        self.slopes_encoder = nn.Sequential(
            nn.Linear(slopes_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.SiLU(),
        )
        self.dm_encoder = nn.Sequential(
            nn.Linear(dm_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.SiLU(),
        )
        self.metrics_encoder = nn.Sequential(
            nn.Linear(metrics_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.SiLU(),
        )
        self.time_embedding = nn.Parameter(
            torch.randn(1, observation_space["ccd"].shape[0], token_dim) * 0.02
        )
        self.visual_to_control_attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_encoder = nn.ModuleList(
            [SelectiveStateSpaceBlock(token_dim, expand=2, kernel_size=3) for _ in range(num_layers)]
        )
        self.context_proj = nn.Linear(token_dim * 4, token_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.Sigmoid(),
        )
        self.output_head = nn.Sequential(
            nn.Linear(token_dim * 3, features_dim),
            nn.LayerNorm(features_dim),
            nn.SiLU(),
            nn.Linear(features_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.SiLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        ccd = observations["ccd"]
        slopes = observations["hartmann_slopes"]
        dm_signal = observations["dm_signal"]
        metrics = observations["metrics"]

        batch, time_steps, height, width = ccd.shape
        patch_tokens, ccd_tokens = self.ccd_tokenizer(
            ccd.reshape(batch * time_steps, 1, height, width)
        )
        ccd_tokens = ccd_tokens.view(batch, time_steps, -1)
        ccd_tokens = ccd_tokens + self.time_embedding[:, :time_steps]

        slopes_tokens = self.slopes_encoder(slopes)
        dm_tokens = self.dm_encoder(dm_signal)
        control_tokens = 0.5 * (slopes_tokens + dm_tokens)
        cross_tokens, _ = self.visual_to_control_attention(
            query=control_tokens.reshape(batch * time_steps, 1, -1),
            key=patch_tokens,
            value=patch_tokens,
            need_weights=False,
        )
        cross_tokens = cross_tokens[:, 0].view(batch, time_steps, -1)
        temporal_tokens = ccd_tokens + cross_tokens + control_tokens
        for block in self.temporal_encoder:
            temporal_tokens = block(temporal_tokens)

        metrics_token = self.metrics_encoder(metrics).unsqueeze(1)
        context_token = self.context_proj(
            torch.cat(
                [
                    ccd_tokens[:, -1],
                    slopes_tokens[:, -1],
                    dm_tokens[:, -1],
                    metrics_token[:, 0],
                ],
                dim=-1,
            )
        ).unsqueeze(1)

        fused_token, _ = self.cross_attention(
            query=context_token,
            key=temporal_tokens,
            value=temporal_tokens,
            need_weights=False,
        )
        fused_token = fused_token[:, 0]
        temporal_summary = temporal_tokens[:, -1]
        gate = self.gate(torch.cat([fused_token, metrics_token[:, 0]], dim=-1))
        gated_fusion = gate * fused_token + (1.0 - gate) * temporal_summary
        return self.output_head(
            torch.cat([gated_fusion, temporal_summary, metrics_token[:, 0]], dim=-1)
        )


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
