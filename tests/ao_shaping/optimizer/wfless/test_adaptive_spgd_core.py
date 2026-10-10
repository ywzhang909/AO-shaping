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
