import numpy as np

from ao_shaping.optimizer.rl.envs import SimTurbulenceAOEnv


def test_sim_turbulence_env_camera_mode_reset_and_step() -> None:
    env = SimTurbulenceAOEnv(
        n_grid=32,
        n_actuators=4,
        n_subapertures=4,
        max_steps=5,
        cn2=1e-15,
    )

    obs, info = env.reset(seed=123)
    assert obs["ccd"].shape == (8, 32, 32)
    assert obs["hartmann_slopes"].shape == (8, 32)
    assert obs["dm_signal"].shape == (8, 16)
    assert obs["metrics"].shape == (3,)
    assert "pib" in info

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    next_obs, reward, terminated, truncated, _ = env.step(action)
    assert next_obs["ccd"].shape == (8, 32, 32)
    assert next_obs["hartmann_slopes"].shape == (8, 32)
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False


def test_sim_turbulence_env_hartmann_mode_reset_and_step() -> None:
    env = SimTurbulenceAOEnv(
        n_grid=32,
        n_actuators=4,
        n_subapertures=4,
        max_steps=3,
        cn2=1e-15,
    )

    obs, info = env.reset(seed=123)
    assert obs["ccd"].shape == (8, 32, 32)
    assert obs["hartmann_slopes"].shape == (8, 32)
    assert obs["dm_signal"].shape == (8, 16)
    assert obs["metrics"].shape == (3,)
    assert info["pib_target"] > 0

    action = np.ones(env.action_space.shape, dtype=np.float32) * 0.05
    _, _, terminated, truncated, _ = env.step(action)
    assert terminated is False
    assert truncated is False
