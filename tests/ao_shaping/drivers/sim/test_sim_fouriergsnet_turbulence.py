"""``SimFourierGSNetEnv`` 时变湍流的 TDD 测试。

环境把低阶像差状态建模为 ``self.aberrations`` —— 一个 Noll 索引 → **弧度**系数的
字典。时变湍流是对其中可配置子集施加的 Ornstein-Uhlenbeck (OU) 均值回归随机游走,
每个闭环帧推进一步::

    x <- x - (x / tau) * dt + sigma * sqrt(2 * dt / tau) * N(0, 1)

其中 ``sigma`` 是以弧度为单位的平稳标准差, ``tau`` 是以帧为单位的弛豫时间, ``dt``
是帧步长。全部测试离线、确定性 (固定种子), 且用小 ``K_px`` 网格保持快速。

与其他模块的关系: 被测对象是 ``drivers/sim/fouriergsnet_env.py`` 的
``configure_turbulence`` / ``advance_time``; 其静态像差端口 (弧度、Noll 索引) 与
``optimizer/wfless/model_in_loop_shaping.py`` 注入真值时用的是同一套约定, 故这里
锁定的是那条注入-拟合闭环的可信前提。
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
    """新建环境: 未激活、noll 集为空、advance_time 是空操作且渲染命中缓存。"""
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
    """OU 系数有界 (< 4σ) 且确实在动。"""
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

    # 预热之后, 每一步都必须满足该界。
    assert max(abs(x) for x in trajectory[50:]) < 4 * 0.5
    # 不是冻结: 至少有一个模式离开了零初值。
    assert any(env.aberrations[noll] != 0.0 for noll in NOLLS)


def test_turbulence_stationary_std() -> None:
    """单个 noll 的经验平稳标准差在宽松容差内与 sigma 相符。"""
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
    """Noll 4 上的 OU 漂移使相邻两帧的渲染远场不同。"""
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
    """同种子同参数 -> 完全相同的 OU 系数轨迹。"""
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
    """OU 只演化被配置的 noll; 手动设置的其它像差不受影响。"""
    env = _small_env(seed=0)
    env.aberrations = {4: 1.0, 11: 0.5, 7: 0.3}
    env.configure_turbulence(seed=42, sigma=0.5, tau=50.0, dt=1.0, nolls=(4, 11))

    for _ in range(20):
        env.advance_time()

    assert env.aberrations[4] != 1.0
    assert env.aberrations[11] != 0.5
    assert env.aberrations[7] == 0.3
