"""TDD tests for time-varying turbulence in ``SimFourierGSNetEnv``.

The env models the low-order aberration state as ``self.aberrations`` — a
``dict`` mapping Noll index -> coefficient in **radians**. Time-varying
turbulence is an Ornstein-Uhlenbeck (OU) mean-reverting random walk applied to
a configurable subset of those Noll coefficients, one OU step per closed-loop
frame::

    x <- x - (x / tau) * dt + sigma * sqrt(2 * dt / tau) * N(0, 1)

with ``sigma`` the stationary standard deviation in radians, ``tau`` the
relaxation time in frames and ``dt`` the frame step. All tests are offline,
deterministic (fixed seeds) and use small ``K_px`` grids so they stay fast.
"""
from __future__ import annotations

import numpy as np

from ao_shaping.drivers.sim.fouriergsnet_env import SimFourierGSNetEnv

K_SMALL = 256
PANEL_H, PANEL_W = 1920, 1200
NOLLS = (4, 5, 6, 11, 13)


def _small_env(seed: int = 0) -> SimFourierGSNetEnv:
    return SimFourierGSNetEnv(K_px=K_SMALL, noise_enabled=False, seed=seed)


def test_turbulence_inactive_by_default() -> None:
    """Fresh env: inactive, empty noll set, advance_time is a no-op + cache hit."""
    env = _small_env(seed=0)
    env.slm.display_phase(np.zeros((PANEL_H, PANEL_W)))

    assert env.turbulence_active is False
    assert env.turbulence_nolls == ()

    before = env.render_intensity()
    snapshot = dict(env.aberrations)
    env.advance_time()

    assert env.aberrations == snapshot
    after = env.render_intensity()
    assert after is before  # unchanged render key -> cache hit
    assert np.array_equal(before, after)


def test_turbulence_bounded_ou() -> None:
    """OU coefficients stay bounded (< 4*sigma) and actually move."""
    env = _small_env(seed=0)
    env.configure_turbulence(seed=42, sigma=0.5, tau=50.0, dt=1.0, nolls=NOLLS)

    assert env.turbulence_active is True
    assert env.turbulence_nolls == NOLLS

    trajectory: list[float] = []
    for _ in range(500):
        env.advance_time()
        trajectory.append(env.aberrations[NOLLS[0]])

    for noll in NOLLS:
        assert abs(env.aberrations[noll]) < 4 * 0.5

    # After warmup, every step must respect the bound.
    assert max(abs(x) for x in trajectory[50:]) < 4 * 0.5
    # Not frozen: at least one mode moved off its zero initial condition.
    assert any(env.aberrations[noll] != 0.0 for noll in NOLLS)


def test_turbulence_stationary_std() -> None:
    """Empirical stationary std of one noll matches sigma within loose bounds."""
    sigma = 0.5
    env = _small_env(seed=0)
    env.configure_turbulence(seed=42, sigma=sigma, tau=50.0, dt=1.0, nolls=(4,))

    warmup = 200
    n_steps = 2000
    for _ in range(warmup):
        env.advance_time()

    samples = np.empty(n_steps, dtype=np.float64)
    for i in range(n_steps):
        env.advance_time()
        samples[i] = env.aberrations[4]

    empirical_std = float(np.std(samples))
    assert 0.35 <= empirical_std <= 0.65, f"empirical std={empirical_std:.4f}"


def test_turbulence_drifts_render_changes() -> None:
    """OU drift on Noll 4 changes the rendered far field between frames."""
    env = _small_env(seed=0)
    env.slm.display_phase(np.zeros((PANEL_H, PANEL_W)))
    env.configure_turbulence(seed=42, sigma=1.0, tau=50.0, dt=1.0, nolls=(4,))

    I0 = env.render_intensity()
    for _ in range(30):
        env.advance_time()
    I1 = env.render_intensity()

    assert not np.array_equal(I0, I1)

    peak0 = np.unravel_index(int(np.argmax(I0)), I0.shape)
    peak1 = np.unravel_index(int(np.argmax(I1)), I1.shape)
    rms = float(np.sqrt(np.mean((I1 - I0) ** 2)))
    assert peak0 != peak1 or rms > 1e-6, f"peak0={peak0}, peak1={peak1}, rms={rms}"


def test_turbulence_seeded_reproducible() -> None:
    """Same seed and params -> identical OU coefficient trajectory."""
    env_a = _small_env(seed=0)
    env_b = _small_env(seed=0)
    env_a.configure_turbulence(seed=42, sigma=0.5, tau=50.0, dt=1.0, nolls=NOLLS)
    env_b.configure_turbulence(seed=42, sigma=0.5, tau=50.0, dt=1.0, nolls=NOLLS)

    traj_a: list[tuple[float, ...]] = []
    traj_b: list[tuple[float, ...]] = []
    for _ in range(100):
        env_a.advance_time()
        env_b.advance_time()
        traj_a.append(tuple(env_a.aberrations[n] for n in NOLLS))
        traj_b.append(tuple(env_b.aberrations[n] for n in NOLLS))

    assert np.array_equal(np.asarray(traj_a), np.asarray(traj_b))


def test_turbulence_preserves_manual_aberrations() -> None:
    """OU only evolves configured nolls; other manual aberrations are untouched."""
    env = _small_env(seed=0)
    env.aberrations = {4: 1.0, 11: 0.5, 7: 0.3}
    env.configure_turbulence(seed=42, sigma=0.5, tau=50.0, dt=1.0, nolls=(4, 11))

    for _ in range(20):
        env.advance_time()

    assert env.aberrations[4] != 1.0
    assert env.aberrations[11] != 0.5
    assert env.aberrations[7] == 0.3
