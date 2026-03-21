"""SAC trainer for SimTurbulenceAOEnv with rich TensorBoard visualization."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import click
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import Figure, Image
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ao_shaping.optimizer.rl.device_registry import build_default_registry
from ao_shaping.optimizer.rl.envs import SimTurbulenceAOEnv


class MambaTemporalBlock(nn.Module):
    """Lightweight Mamba-style temporal mixer with selective scan behavior."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            groups=d_model,
        )
        self.a_logit = nn.Parameter(torch.zeros(d_model))
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        residual = x
        x = self.norm(x)
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        value = self.depthwise_conv(value.transpose(1, 2)).transpose(1, 2)
        gate = torch.sigmoid(gate)

        a = torch.sigmoid(self.a_logit).view(1, 1, -1)
        state = torch.zeros_like(value[:, 0, :])
        outputs = []
        for t in range(value.size(1)):
            state = a.squeeze(1) * state + (1 - a.squeeze(1)) * value[:, t, :]
            outputs.append(state * gate[:, t, :])
        y = torch.stack(outputs, dim=1)
        return residual + self.out_proj(y)


class CrossAttentionMambaExtractor(BaseFeaturesExtractor):
    """Fuse CCD, Hartmann slopes and DM signals via cross-attention + Mamba."""

    def __init__(self, observation_space: gym.spaces.Dict, d_model: int = 128) -> None:
        self.d_model = d_model
        super().__init__(observation_space, features_dim=d_model + 32)

        ccd_shape = observation_space["ccd"].shape
        slopes_dim = observation_space["hartmann_slopes"].shape[-1]
        dm_dim = observation_space["dm_signal"].shape[-1]
        metrics_dim = observation_space["metrics"].shape[0]
        self.ccd_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, d_model),
            nn.LayerNorm(d_model),
        )
        self.slopes_encoder = nn.Sequential(
            nn.Linear(slopes_dim, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
        )
        self.dm_encoder = nn.Sequential(
            nn.Linear(dm_dim, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
        )
        self.metrics_encoder = nn.Sequential(
            nn.Linear(metrics_dim, 32),
            nn.SiLU(),
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=4, batch_first=True
        )
        self.temporal_mamba = MambaTemporalBlock(d_model)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        ccd = observations["ccd"]  # [B, T, H, W]
        slopes = observations["hartmann_slopes"]  # [B, T, S]
        dm_signal = observations["dm_signal"]  # [B, T, A]
        metrics = observations["metrics"]  # [B, 3]

        batch, time_steps, h, w = ccd.shape
        ccd_feat = self.ccd_encoder(ccd.reshape(batch * time_steps, 1, h, w)).view(
            batch, time_steps, self.d_model
        )
        slopes_feat = self.slopes_encoder(slopes)
        dm_feat = self.dm_encoder(dm_signal)

        kv = torch.stack([slopes_feat, dm_feat], dim=2).reshape(
            batch * time_steps, 2, self.d_model
        )
        q = ccd_feat.reshape(batch * time_steps, 1, self.d_model)
        fused, _ = self.cross_attn(query=q, key=kv, value=kv, need_weights=False)
        fused = fused.reshape(batch, time_steps, self.d_model)

        temporal_feat = self.temporal_mamba(fused)[:, -1, :]
        metrics_feat = self.metrics_encoder(metrics)
        return torch.cat([temporal_feat, metrics_feat], dim=-1)


class RichAOTensorboardCallback(BaseCallback):
    """Record AO-specific scalars and images for TensorBoard."""

    def __init__(self, log_every: int = 50, image_every: int = 500, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_every = log_every
        self.image_every = image_every
        self.pib_window: deque[float] = deque(maxlen=200)
        self.reward_window: deque[float] = deque(maxlen=200)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        actions = self.locals.get("actions", [])
        new_obs = self.locals.get("new_obs", {})

        if not infos:
            return True

        info = infos[-1]
        reward = float(rewards[-1]) if len(rewards) else 0.0
        self.pib_window.append(float(info.get("pib", 0.0)))
        self.reward_window.append(reward)

        if self.num_timesteps % self.log_every == 0:
            self.logger.record("ao/pib", float(info.get("pib", 0.0)))
            self.logger.record("ao/best_pib", float(info.get("best_pib", 0.0)))
            self.logger.record("ao/pib_target", float(info.get("pib_target", 1.0)))
            self.logger.record("ao/strehl", float(info.get("strehl", 0.0)))
            self.logger.record("ao/rms", float(info.get("rms", 0.0)))
            self.logger.record("ao/pib_mavg200", float(np.mean(self.pib_window)))
            self.logger.record("ao/reward_mavg200", float(np.mean(self.reward_window)))

            if len(actions):
                act = np.asarray(actions[-1], dtype=float)
                self.logger.record("ao/action_l2", float(np.linalg.norm(act)))
                self.logger.record("ao/action_abs_mean", float(np.mean(np.abs(act))))

        if self.num_timesteps % self.image_every == 0 and isinstance(new_obs, dict):
            ccd = new_obs.get("ccd")
            slopes = new_obs.get("hartmann_slopes")
            metrics = new_obs.get("metrics")

            if ccd is not None:
                ccd_arr = np.asarray(ccd, dtype=np.float32)
                if ccd_arr.ndim == 4:
                    ccd_img = ccd_arr[-1, -1]
                elif ccd_arr.ndim == 3:
                    ccd_img = ccd_arr[-1]
                else:
                    ccd_img = ccd_arr
                ccd_uint8 = np.clip(ccd_img * 255.0, 0, 255).astype(np.uint8)
                self.logger.record("ao/ccd_image", Image(ccd_uint8, "HW"), exclude=("stdout", "log", "json", "csv"))

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
                ax.set_xlabel("Index")
                ax.set_ylabel("Slope")
                ax.grid(alpha=0.3)
                self.logger.record("ao/hartmann_slopes", Figure(fig, close=True), exclude=("stdout", "log", "json", "csv"))

            if metrics is not None:
                m = np.asarray(metrics[-1] if metrics.ndim == 2 else metrics, dtype=np.float32)
                self.logger.record("ao/obs_strehl", float(m[0]))
                self.logger.record("ao/obs_rms", float(m[1]))
                self.logger.record("ao/obs_pib", float(m[2]))

        return True


def build_env(max_steps: int, n_grid: int, n_actuators: int, n_subapertures: int, cn2: float, history_len: int) -> gym.Env:
    env = SimTurbulenceAOEnv(
        max_steps=max_steps,
        n_grid=n_grid,
        n_actuators=n_actuators,
        n_subapertures=n_subapertures,
        cn2=cn2,
        history_len=history_len,
    )
    return Monitor(env)


def _warn_if_physical_selected(dm_device: str, ccd_device: str, wfs_device: str) -> None:
    registry = build_default_registry()
    specs = [
        registry.get("dm", dm_device),
        registry.get("ccd", ccd_device),
        registry.get("wfs", wfs_device),
    ]
    physical = [s for s in specs if not s.is_virtual]
    if physical:
        names = ", ".join([f"{s.device_type}:{s.name}" for s in physical])
        click.echo(
            f"[warning] selected physical devices ({names}), "
            "but this SAC script currently uses simulation backend only.",
            err=True,
        )


@click.command()
@click.option("--total-timesteps", type=int, default=100_000, show_default=True)
@click.option("--max-steps", type=int, default=80, show_default=True)
@click.option("--n-grid", type=int, default=64, show_default=True)
@click.option("--n-actuators", type=int, default=6, show_default=True)
@click.option("--n-subapertures", type=int, default=6, show_default=True)
@click.option("--cn2", type=float, default=1e-14, show_default=True)
@click.option("--history-len", type=int, default=8, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--log-dir", type=str, default="logs/sac_turbulence", show_default=True)
@click.option("--model-dir", type=str, default="models/sac_turbulence", show_default=True)
@click.option(
    "--dm-device",
    type=click.Choice(build_default_registry().names("dm"), case_sensitive=False),
    default="sim_dm",
    show_default=True,
    help="Registered DM device name.",
)
@click.option(
    "--ccd-device",
    type=click.Choice(build_default_registry().names("ccd"), case_sensitive=False),
    default="sim_ccd",
    show_default=True,
    help="Registered CCD device name.",
)
@click.option(
    "--wfs-device",
    type=click.Choice(build_default_registry().names("wfs"), case_sensitive=False),
    default="sim_wfs",
    show_default=True,
    help="Registered WFS device name.",
)
def main(
    total_timesteps: int,
    max_steps: int,
    n_grid: int,
    n_actuators: int,
    n_subapertures: int,
    cn2: float,
    history_len: int,
    seed: int,
    log_dir: str,
    model_dir: str,
    dm_device: str,
    ccd_device: str,
    wfs_device: str,
) -> None:
    _warn_if_physical_selected(dm_device=dm_device, ccd_device=ccd_device, wfs_device=wfs_device)
    click.echo(f"Selected devices -> DM:{dm_device} CCD:{ccd_device} WFS:{wfs_device}")

    log_dir = Path(log_dir)
    model_dir = Path(model_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(
        max_steps=max_steps,
        n_grid=n_grid,
        n_actuators=n_actuators,
        n_subapertures=n_subapertures,
        cn2=cn2,
        history_len=history_len,
    )

    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        learning_starts=2000,
        train_freq=1,
        gradient_steps=1,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        tensorboard_log=str(log_dir),
        seed=seed,
        verbose=1,
        policy_kwargs={
            "features_extractor_class": CrossAttentionMambaExtractor,
            "features_extractor_kwargs": {"d_model": 128},
        },
    )

    callbacks = CallbackList([
        RichAOTensorboardCallback(log_every=50, image_every=500),
    ])

    model.learn(total_timesteps=total_timesteps, callback=callbacks, tb_log_name="sac_run")

    model.save(model_dir / "sac_turbulence_final")
    env.close()


if __name__ == "__main__":
    main()
