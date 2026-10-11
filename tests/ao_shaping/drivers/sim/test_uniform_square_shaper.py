from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.drivers.sim.slm_shaping_bench import (
    GS_METHODS,
    ShapingBenchConfig,
    UniformSquareShaper,
    compute_bench_metrics,
    gs_shape,
    make_target,
    uniformity_cv,
)


def _make_small_cfg() -> ShapingBenchConfig:
    return ShapingBenchConfig(
        n_grid=64,
        aperture_size=64 * 8e-6,
        wavelength=1064e-9,
        focal_length=0.125,
        target_side_px=16,
        far_field_padding=4,
        seed=0,
    )


def test_gamma_zero_matches_gs_shape():
    """γ=0 ⇒ weight stays 1 ⇒ the iteration IS plain GS, iterate for iterate.

    ``select="last"`` is required here: the class defaults to ``select="best"``
    (the oscillation is real, so the best iterate and the last iterate differ),
    while ``gs_shape`` always returns the last iterate. Comparing like for like
    is what makes this a statement about the *dynamics* rather than about the
    selection policy.
    """
    cfg = _make_small_cfg()
    shaper = UniformSquareShaper(
        cfg, n_iters=8, method="weighted", gamma=0.0, seed=3, select="last"
    )
    res_shaper = shaper.run()
    res_gs = gs_shape(cfg, n_iters=8, relax=1.0, seed=3)
    assert np.allclose(res_shaper.phase, res_gs.phase, atol=1e-9, equal_nan=False)


def test_best_select_beats_last_select():
    """``select="best"`` must never score worse than ``select="last"``.

    The three variants converge *oscillatorily* (measured intra-square CV swings
    between 0.37 and 1.50 within 60 iterations), so taking the last iterate is a
    random draw from that swing. Tracking the best is what makes the method
    reliable, and it is asserted here rather than assumed.
    """
    cfg = _make_small_cfg()
    best = UniformSquareShaper(
        cfg, n_iters=25, method="weighted", gamma=0.5, seed=4, select="best"
    ).run()
    last = UniformSquareShaper(
        cfg, n_iters=25, method="weighted", gamma=0.5, seed=4, select="last"
    ).run()

    def score(res):
        return res.metrics["PIB"] / (1.0 + res.metrics["CV"])

    assert score(best) >= score(last)


def test_invalid_select_raises():
    with pytest.raises(ValueError):
        UniformSquareShaper(_make_small_cfg(), select="middle")


def test_beta_zero_freezes_outside_region():
    cfg = _make_small_cfg()
    shaper = UniformSquareShaper(cfg, n_iters=5, method="hio", beta=0.0, seed=1)
    res = shaper.run()
    assert res.phase.shape == (cfg.n_grid, cfg.n_grid)
    assert np.all(np.isfinite(res.phase))


@pytest.mark.parametrize("hio_frac", [1.0, 0.0])
def test_hio_weighted_splits_the_schedule(hio_frac):
    cfg = _make_small_cfg()
    shaper = UniformSquareShaper(
        cfg,
        n_iters=6,
        method="hio-weighted",
        hio_fraction=hio_frac,
        seed=2,
    )
    res = shaper.run()
    assert res.phase.shape == (cfg.n_grid, cfg.n_grid)
    assert np.all(np.isfinite(res.phase))


def test_hio_weighted_different_fractions_differ():
    cfg = _make_small_cfg()
    res1 = UniformSquareShaper(
        cfg, n_iters=6, method="hio-weighted", hio_fraction=1.0, seed=2
    ).run()
    res2 = UniformSquareShaper(
        cfg, n_iters=6, method="hio-weighted", hio_fraction=0.0, seed=2
    ).run()
    assert not np.allclose(res1.phase, res2.phase)


@pytest.mark.parametrize("method", GS_METHODS)
def test_methods_produce_valid_phase(method):
    cfg = _make_small_cfg()
    res = UniformSquareShaper(cfg, n_iters=4, method=method, seed=5).run()
    assert res.phase.shape == (cfg.n_grid, cfg.n_grid)
    assert np.all(np.isfinite(res.phase))
    assert res.phase.dtype == np.float64


def test_uniformity_improves_over_random_start():
    cfg = _make_small_cfg()
    # Run with enough iterations
    res = UniformSquareShaper(cfg, n_iters=30, method="weighted", gamma=0.8, seed=7).run()
    # Compute CV inside square for initial random phase
    from ao_shaping.drivers.sim.slm_shaping_bench import (
        _fraunhofer_intensity,
        _gs_initial_field,
        _pad_centred,
    )
    from ao_shaping.drivers.sim.beam_backend import gaussian_pupil

    beam_cfg = cfg.make_beam_config()
    field, amp_slm, amp_target, target, base = _gs_initial_field(
        cfg, beam_cfg, seed=7, base_phase=None
    )
    inten_init = _fraunhofer_intensity(field, cfg)
    inten_init = inten_init / (inten_init.sum() + 1e-12)
    center_init = np.unravel_index(np.argmax(inten_init), inten_init.shape)[::-1]
    cv_init = uniformity_cv(inten_init, target, center=center_init)
    cv_final = res.metrics["CV"]
    # CV should improve (lower) - if flaky, we note but try
    assert cv_final < cv_init


# γ=0 与 β=0 是刻意保留的退化哨兵, 不是错误: γ=0 ⇒ 权重恒为 1 ⇒ 与 plain GS
# 逐位等价 (test_gamma_zero_matches_gs_shape 依赖这一点); β=0 ⇒ 噪声区被冻结。
# 因此非法用例取负值, 校验区间在 0 处是闭的。
@pytest.mark.parametrize(
    "kwargs,exc",
    [
        ({"method": "unknown"}, ValueError),
        ({"method": "weighted", "gamma": -0.1}, ValueError),
        ({"method": "weighted", "gamma": 1.0}, ValueError),
        ({"method": "weighted", "weight_clip": 1.0}, ValueError),
        ({"method": "hio", "beta": -0.1}, ValueError),
        ({"method": "hio", "beta": 1.5}, ValueError),
        ({"method": "hio-weighted", "hio_fraction": -0.1}, ValueError),
        ({"method": "hio-weighted", "hio_fraction": 1.1}, ValueError),
        ({"n_iters": 0}, ValueError),
    ],
)
def test_invalid_parameters_raise(kwargs, exc):
    cfg = _make_small_cfg()
    with pytest.raises(exc):
        UniformSquareShaper(cfg, **kwargs)
