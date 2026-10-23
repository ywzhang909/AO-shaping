"""块级策略的离线验证：方向序列、A-B-B-A、接受/拒绝、停止与零信号判定。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ao_shaping.optimizer.wfless.defocus_uniformity_shaping import (
    DefocusShapingConfig,
    DefocusUniformityShaper,
    load_frozen_checkpoint,
)

ROOT = Path(__file__).resolve().parents[4]
CHECKPOINT = ROOT / "data/debug/physical_forward_defocus/uniformity_refit_20261022_r1"


def _shape(size: int = 8) -> np.ndarray:
    return np.arange(size * size, dtype=float).reshape(size, size) * 1e-3


def test_direction_test_sequence_is_a_bplus_a_bminus_a():
    shaper = DefocusUniformityShaper(DefocusShapingConfig(branch="U"))
    seed, step = _shape(), np.full((8, 8), 0.01)
    plan = shaper.plan_direction_test(seed, step)
    assert [s.label for s in plan] == ["A", "B+", "A", "B-", "A"]
    assert np.allclose(plan[0].phase, seed) and np.allclose(plan[4].phase, seed)
    assert np.allclose(plan[1].phase - seed, step)
    assert np.allclose(plan[3].phase - seed, -step)


def test_block_sequence_is_a_b_b_a():
    shaper = DefocusUniformityShaper(DefocusShapingConfig(branch="U"))
    plan = shaper.plan_block(_shape(), _shape() + 0.5)
    assert [s.label for s in plan] == ["A1", "B1", "B2", "A2"]
    assert np.allclose(plan[0].phase, plan[3].phase)


def test_accept_only_when_improvement_exceeds_three_sigma():
    shaper = DefocusUniformityShaper(DefocusShapingConfig(branch="U"))
    baseline = [1.10, 1.11, 1.09, 1.10]
    noise = [1.10, 1.101, 1.099, 1.10]
    accepted = shaper.evaluate_candidate(1.05, baseline, noise)
    rejected = shaper.evaluate_candidate(1.099, baseline, noise)
    assert accepted.accepted and accepted.delta < 0
    assert not rejected.accepted
    assert accepted.resolution > 0


def test_reject_halves_cap_and_two_consecutive_rejects_stop():
    config = DefocusShapingConfig(branch="U", block_rms_cap=0.1)
    shaper = DefocusUniformityShaper(config)
    noisy = [1.10, 1.101, 1.099, 1.10]
    verdict = shaper.evaluate_candidate(1.10, noisy, noisy)
    assert not verdict.accepted
    shaper.record_block(verdict, block=0)
    assert config.block_rms_cap == pytest.approx(0.05)
    shaper.record_block(shaper.evaluate_candidate(1.10, noisy, noisy), block=1)
    assert shaper.should_stop
    assert "连续 2 块" in shaper.state.stop_reason
    assert config.block_rms_cap == pytest.approx(0.025)


def test_accept_resets_consecutive_rejects():
    noisy = [1.10, 1.101, 1.099, 1.10]
    shaper = DefocusUniformityShaper(DefocusShapingConfig(branch="U"))
    shaper.record_block(shaper.evaluate_candidate(1.10, noisy, noisy), block=0)
    assert shaper.state.consecutive_rejects == 1
    good = shaper.evaluate_candidate(1.0, noisy, noisy)
    assert good.accepted
    shaper.record_block(good, block=1)
    assert shaper.state.consecutive_rejects == 0
    assert not shaper.should_stop


def test_zero_signal_verdict():
    shaper = DefocusUniformityShaper(DefocusShapingConfig(branch="U"))
    invalid, reason = shaper.zero_signal_verdict(2.0, 5.0)
    assert invalid and "≤" in reason
    valid, reason = shaper.zero_signal_verdict(50.0, 5.0)
    assert not valid and reason == ""


def test_config_requires_focus_dir_for_c_branch():
    with pytest.raises(ValueError, match="focus_dir"):
        DefocusShapingConfig(branch="C")


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="need the fitted checkpoint on this machine")
def test_checkpoint_loader_records_sha256_and_parameters():
    record = load_frozen_checkpoint(CHECKPOINT, selection_rule="training score minimum")
    described = record.describe()
    assert len(described["result_sha256"]) == 64
    assert len(described["upstream_sha256"]) == 64
    assert described["selection_rule"] == "training score minimum"
    assert described["parameters"]["defocus_diopter"] == pytest.approx(-0.645165741443634, rel=1e-9)
    assert described["parameters"]["fft_size"] == 1024


def test_checkpoint_loader_rejects_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_frozen_checkpoint(tmp_path)
