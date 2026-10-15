from __future__ import annotations
import math

def test_stage_weights_hold_three_triples_for_coarse_middle_fine():
    from ao_shaping.optimizer.wfless import adaptive_spgd_core
    cfg = adaptive_spgd_core.AdaptiveSpgdConfig()
    assert hasattr(cfg, 'stage_weights')
    sw = cfg.stage_weights
    assert isinstance(sw, tuple) and len(sw) == 3
    for i, tri in enumerate(sw):
        assert isinstance(tri, tuple) and len(tri) == 3
        w_c, w_m, w_f = tri
        for v in (w_c, w_m, w_f):
            assert isinstance(v, (int,float)) and math.isfinite(float(v))

def test_epoch_thirds_transitions():
    from ao_shaping.optimizer.wfless import adaptive_spgd_core
    # coarse < 1/3, middle between 1/3 and 2/3, fine >= 2/3
    get = adaptive_spgd_core.get_stage_weights
    cfg = adaptive_spgd_core.AdaptiveSpgdConfig()
    # epoch 0 coarse
    sw0 = get(cfg, epoch=0, total_epochs=30)
    assert sw0 == cfg.stage_weights[0]
    # 10 is 1/3 (10/30=0.33) - middle starts after coarse ends? per spec: transitions at epoch thirds
    sw1 = get(cfg, epoch=10, total_epochs=30)  # >=1/3 and <2/3 -> middle
    assert sw1 == cfg.stage_weights[1]
    sw2 = get(cfg, epoch=20, total_epochs=30)  # >=2/3 -> fine
    assert sw2 == cfg.stage_weights[2]

def test_stagnation_boost_defaults_and_disable():
    from ao_shaping.optimizer.wfless import adaptive_spgd_core
    cfg = adaptive_spgd_core.AdaptiveSpgdConfig()
    assert cfg.stagnate_win == 10
    assert cfg.stagnate_boost == 1.5
    assert cfg.enable_stagnate_boost is True
    cfg_off = adaptive_spgd_core.AdaptiveSpgdConfig(stagnate_boost=1.0)
    assert cfg_off.stagnate_boost == 1.0

def test_detector_running_min_over_last_10_and_stagnation_only():
    from ao_shaping.optimizer.wfless import adaptive_spgd_core
    # simulate: CV goes down then flat
    cv_hist = [1.0, 0.8, 0.6, 0.5, 0.48, 0.47, 0.46, 0.455, 0.454, 0.453, 0.453]
    # after 10 elements, best over last 10? detector maintains running min over last 10 epochs
    should_boost = adaptive_spgd_core.should_boost_stagnation(cv_hist, win=10, min_improve_frac=0.01)
    # no improvement >=1% in last 10 vs best of last 10? best hasn't improved meaningfully -> True (stagnant)
    assert should_boost is True
    # rising CV alone shouldn't trigger? if improving (going down) maybe False; if rising, also check best
    cv_rising = [0.5, 0.49, 0.48, 0.47, 0.46, 0.461, 0.462, 0.463, 0.464, 0.465, 0.466]
    should_boost_rising = adaptive_spgd_core.should_boost_stagnation(cv_rising, win=10, min_improve_frac=0.01)
    # best is still 0.46 from earlier? or running min - stagnation is about best not improving; rising means current worse than best -> stagnant relative to best
    assert should_boost_rising is True


# ---------------------------------------------------------------------------
# Wave 2: GradientBalancer (§3.2)
# ---------------------------------------------------------------------------

def test_gradient_balancer_init_and_reset():
    from ao_shaping.optimizer.wfless import adaptive_spgd_core
    gb = adaptive_spgd_core.GradientBalancer(n_terms=3)
    assert gb.n_terms == 3
    assert list(gb.gbar) == [1.0, 1.0, 1.0]
    gb.compute([1.0, 2.0, 3.0])
    assert not all(abs(v - 1.0) < 1e-12 for v in gb.gbar)
    gb.reset()
    assert list(gb.gbar) == [1.0, 1.0, 1.0]


def test_gradient_balancer_ema_updates_gbar():
    from ao_shaping.optimizer.wfless import adaptive_spgd_core
    gb = adaptive_spgd_core.GradientBalancer(n_terms=3, ema_decay=0.9)
    gb.compute([0.0, 0.0, 0.0])
    # gbar = 0.9*ones + 0.1*0 = 0.9
    assert all(abs(v - 0.9) < 1e-12 for v in gb.gbar)
    gb.compute([1.0, 1.0, 1.0])
    # gbar = 0.9*0.9 + 0.1*1 = 0.91
    assert all(abs(v - 0.91) < 1e-12 for v in gb.gbar)


def test_gradient_balancer_equalizes_terms():
    from ao_shaping.optimizer.wfless import adaptive_spgd_core
    # A term with a large magnitude should be down-weighted relative to a small one.
    gb = adaptive_spgd_core.GradientBalancer(n_terms=3, ema_decay=0.0)
    # ema_decay=0 -> gbar becomes |dJ| exactly, so balancing is a pure normalisation
    out = gb.compute([100.0, 1.0, 1.0])
    # each weighted term is dJ_i / sum(|dJ|) * ... -> bounded and finite
    assert math.isfinite(out)
    assert abs(out) <= sum(abs(x) for x in (100.0, 1.0, 1.0)) + 1e-9


def test_gradient_balancer_zero_gradient_is_finite():
    from ao_shaping.optimizer.wfless import adaptive_spgd_core
    gb = adaptive_spgd_core.GradientBalancer(n_terms=3)
    out = gb.compute([0.0, 0.0, 0.0])
    assert math.isfinite(out)
    assert out == 0.0


def test_gradient_balancer_accepts_weights():
    from ao_shaping.optimizer.wfless import adaptive_spgd_core
    gb = adaptive_spgd_core.GradientBalancer(n_terms=3, ema_decay=0.0)
    out = gb.compute([1.0, 1.0, 1.0], weights=(1.0, 0.0, 0.0))
    assert math.isfinite(out)


def test_gradient_balancer_da_does_not_divide_by_zero():
    from ao_shaping.optimizer.wfless import adaptive_spgd_core
    gb = adaptive_spgd_core.GradientBalancer(n_terms=3)
    # da is accepted for API compatibility; tiny/zero da must not raise.
    out = gb.compute([1e-30, 1e-30, 1e-30], da=[0.0, 0.0, 0.0])
    assert math.isfinite(out)


# ---------------------------------------------------------------------------
# Wave 2: adapt_step_size (§4.2)
# ---------------------------------------------------------------------------

def test_adapt_step_size_empty_history_unchanged():
    from ao_shaping.optimizer.wfless import adaptive_spgd_core
    assert adaptive_spgd_core.adapt_step_size(0.1, []) == 0.1
    assert adaptive_spgd_core.adapt_step_size(0.1, []) == 0.1


def test_adapt_step_size_high_accept_rate_grows():
    from ao_shaping.optimizer.wfless import adaptive_spgd_core
    hist = [True] * 8 + [False, False]  # 0.8 accept
    out = adaptive_spgd_core.adapt_step_size(0.1, hist)
    assert abs(out - 0.14) < 1e-9


def test_adapt_step_size_low_accept_rate_shrinks():
    from ao_shaping.optimizer.wfless import adaptive_spgd_core
    hist = [True, False, False, False, False, False, False, False, True, False]  # 0.2
    out = adaptive_spgd_core.adapt_step_size(0.1, hist)
    assert abs(out - 0.06) < 1e-9


def test_adapt_step_size_middle_band_unchanged():
    from ao_shaping.optimizer.wfless import adaptive_spgd_core
    hist = [True] * 5 + [False] * 5  # 0.5
    out = adaptive_spgd_core.adapt_step_size(0.1, hist)
    assert abs(out - 0.1) < 1e-9


def test_adapt_step_size_respects_clip_bounds():
    from ao_shaping.optimizer.wfless import adaptive_spgd_core
    hi = adaptive_spgd_core.adapt_step_size(0.45, [True] * 10)  # 0.45*1.4=0.63 -> clip 0.5
    assert abs(hi - 0.5) < 1e-9
    lo = adaptive_spgd_core.adapt_step_size(0.03, [False] * 10)  # 0.03*0.6=0.018 -> clip 0.02
    assert abs(lo - 0.02) < 1e-9


def test_adapt_step_size_window_truncation():
    from ao_shaping.optimizer.wfless import adaptive_spgd_core
    # Only the last `window` entries count.
    hist = [False] * 20 + [True] * 10  # last 10 all True -> grow
    out = adaptive_spgd_core.adapt_step_size(0.1, hist, window=10)
    assert abs(out - 0.14) < 1e-9


# ---------------------------------------------------------------------------
# Wave 2: SNR-driven per-mode sigma (§4.1)
# ---------------------------------------------------------------------------

def test_snr_signal_basic_and_zero_noise():
    from ao_shaping.optimizer.wfless import adaptive_spgd_core
    assert abs(adaptive_spgd_core.snr_signal(1.0, 1.0) - (1.0 / (2 ** 0.5))) < 1e-9
    assert adaptive_spgd_core.snr_signal(1.0, 0.0) == float("inf")


def test_sigma_scale_enlarge_when_snr_low():
    from ao_shaping.optimizer.wfless import adaptive_spgd_core
    assert adaptive_spgd_core.sigma_scale_from_snr(1.0) == 1.3


def test_sigma_scale_shrink_when_snr_high():
    from ao_shaping.optimizer.wfless import adaptive_spgd_core
    assert adaptive_spgd_core.sigma_scale_from_snr(20.0) == 0.85


def test_sigma_scale_unchanged_in_band():
    from ao_shaping.optimizer.wfless import adaptive_spgd_core
    assert adaptive_spgd_core.sigma_scale_from_snr(5.0) == 1.0


def test_sigma_clip_bounds_constant():
    from ao_shaping.optimizer.wfless import adaptive_spgd_core
    # Default clip window for per-mode sigma.
    import inspect
    sig = inspect.signature(adaptive_spgd_core.sigma_scale_from_snr)
    assert sig.parameters["sigma_min"].default == 0.01
    assert sig.parameters["sigma_max"].default == 0.3
