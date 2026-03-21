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
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.logger import Figure, Image
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ao_shaping.optimizer.rl.device_registry import build_default_registry
from ao_shaping.optimizer.rl.envs import SimTurbulenceAOEnv


class TemporalAOExtractor(BaseFeaturesExtractor):
    """Compact temporal encoder tuned for stable SAC training."""

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
            self.logger.record("ao/initial_pib", float(info.get("initial_pib", 0.0)))
            self.logger.record("ao/pib_target", float(info.get("pib_target", 1.0)))
            self.logger.record("ao/pib_ratio", float(info.get("pib_ratio", 0.0)))
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


def validate_env(env: gym.Env, seed: int) -> None:
    """Lightweight preflight validation for the custom dict environment."""
    obs, _ = env.reset(seed=seed)
    if not env.observation_space.contains(obs):
        raise ValueError("Reset observation is outside the declared observation space.")

    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    if not env.observation_space.contains(obs):
        raise ValueError("Step observation is outside the declared observation space.")
    if not np.isfinite(reward):
        raise ValueError("Environment returned a non-finite reward.")
    if not isinstance(terminated, bool) or not isinstance(truncated, bool):
        raise TypeError("Environment terminated/truncated flags must be bool.")


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
@click.option("--learning-rate", type=float, default=1e-4, show_default=True)
@click.option("--buffer-size", type=int, default=100_000, show_default=True)
@click.option("--batch-size", type=int, default=128, show_default=True)
@click.option("--learning-starts", type=int, default=1_000, show_default=True)
@click.option("--tau", type=float, default=0.01, show_default=True)
@click.option("--gamma", type=float, default=0.98, show_default=True)
@click.option("--train-freq", type=int, default=4, show_default=True)
@click.option("--gradient-steps", type=int, default=4, show_default=True)
@click.option("--eval-freq", type=int, default=2_000, show_default=True)
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
    learning_rate: float,
    buffer_size: int,
    batch_size: int,
    learning_starts: int,
    tau: float,
    gamma: float,
    train_freq: int,
    gradient_steps: int,
    eval_freq: int,
    log_dir: Path | str,
    model_dir: Path | str,
    dm_device: str,
    ccd_device: str,
    wfs_device: str,
) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    _warn_if_physical_selected(dm_device=dm_device, ccd_device=ccd_device, wfs_device=wfs_device)
    click.echo(f"Selected devices -> DM:{dm_device} CCD:{ccd_device} WFS:{wfs_device}")

    log_dir = Path(log_dir) if isinstance(log_dir, str) else log_dir
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
    validate_env(env.env, seed=seed)
    eval_env = build_env(
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
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        learning_starts=learning_starts,
        train_freq=(train_freq, "step"),
        gradient_steps=gradient_steps,
        gamma=gamma,
        tau=tau,
        ent_coef="auto_0.1",
        target_update_interval=1,
        use_sde=True,
        target_entropy="auto",
        tensorboard_log=str(log_dir),
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
        RichAOTensorboardCallback(log_every=50, image_every=500),
        EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir / "best"),
            log_path=str(log_dir / "eval"),
            eval_freq=max(eval_freq, 1),
            n_eval_episodes=5,
            deterministic=True,
            render=False,
        ),
    ])

    model.learn(total_timesteps=total_timesteps, callback=callbacks, tb_log_name="sac_run")

    model.save(model_dir / "sac_turbulence_final")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
