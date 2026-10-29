"""相位增量的频域带限正则 (issue #213 质量项 1b)。"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.algorithm.signal_processing.spectral_regulariser import (
    control_grid_cutoff,
    mtf_matched_mask,
    spectral_penalty,
)


def test_mask_is_a_radial_passband():
    size = 64
    cutoff = 0.2
    mask = mtf_matched_mask(size, cutoff=cutoff, rolloff=0.05)
    assert mask.shape == (size, size)
    freq = np.fft.fftfreq(size)
    fy, fx = np.meshgrid(freq, freq, indexing="ij")
    radius = np.hypot(fx, fy)
    assert np.all(mask[radius <= cutoff] == pytest.approx(1.0))
    assert np.all(mask[radius >= cutoff + 0.05] == pytest.approx(0.0))
    # Monotone fall-off inside the transition band.
    band = (radius > cutoff) & (radius < cutoff + 0.05)
    assert np.all(mask[band] <= 1.0) and np.all(mask[band] >= 0.0)
    assert np.all(np.diff(mask[band][np.argsort(radius[band])]) <= 1e-6)


def test_zero_rolloff_is_a_hard_cut():
    mask = mtf_matched_mask(32, cutoff=0.25, rolloff=0.0)
    assert set(np.unique(mask)) <= {0.0, 1.0}


def test_smooth_phase_has_near_zero_penalty():
    torch = pytest.importorskip("torch")
    size = 64
    mask = mtf_matched_mask(size, cutoff=0.2, rolloff=0.05)
    yy, xx = np.mgrid[:size, :size]
    smooth = torch.tensor(0.1 * np.sin(2 * np.pi * xx / size), dtype=torch.float32)
    penalty = float(spectral_penalty(smooth, mask))
    assert penalty < 1e-3


def test_nyquist_checkerboard_is_fully_out_of_band():
    torch = pytest.importorskip("torch")
    size = 64
    mask = mtf_matched_mask(size, cutoff=0.2, rolloff=0.05)
    yy, xx = np.mgrid[:size, :size]
    checker = torch.tensor(((-1.0) ** (xx + yy)), dtype=torch.float32)
    penalty = float(spectral_penalty(checker, mask))
    assert penalty == pytest.approx(1.0, abs=1e-5)


def test_penalty_is_scale_invariant():
    torch = pytest.importorskip("torch")
    size = 64
    mask = mtf_matched_mask(size, cutoff=0.2, rolloff=0.05)
    yy, xx = np.mgrid[:size, :size]
    pattern = np.cos(2 * np.pi * (3 * xx + 5 * yy) / size)
    small = float(spectral_penalty(torch.tensor(0.01 * pattern, dtype=torch.float32), mask))
    large = float(spectral_penalty(torch.tensor(0.4 * pattern, dtype=torch.float32), mask))
    assert small == pytest.approx(large, rel=1e-4)


def test_penalty_backpropagates_to_the_phase():
    torch = pytest.importorskip("torch")
    size = 32
    mask = mtf_matched_mask(size, cutoff=0.2, rolloff=0.05)
    yy, xx = np.mgrid[:size, :size]
    # Mixed band content: the saturated checkerboard alone has zero gradient
    # (its out-of-band fraction is 1 whatever its amplitude), so combine the
    # Nyquist mode with a mid-band one to exercise the derivative.
    pattern = 0.05 * (((-1.0) ** (xx + yy)) + 0.5 * np.cos(2 * np.pi * 2 * xx / size))
    phase = torch.tensor(pattern, dtype=torch.float32, requires_grad=True)
    spectral_penalty(phase, mask).backward()
    assert phase.grad is not None
    assert float(phase.grad.abs().sum()) > 0


def test_mask_shape_mismatch_raises():
    torch = pytest.importorskip("torch")
    mask = mtf_matched_mask(32, cutoff=0.2, rolloff=0.05)
    with pytest.raises(ValueError):
        spectral_penalty(torch.zeros(16, 16), mask)


def test_cutoff_defaults_to_seventy_percent_of_nyquist():
    assert control_grid_cutoff(640) == pytest.approx(0.35)
    assert control_grid_cutoff(640, nyquist_fraction=1.0) == pytest.approx(0.5)


def test_cutoff_validates_its_input():
    with pytest.raises(ValueError):
        control_grid_cutoff(1)
    with pytest.raises(ValueError):
        control_grid_cutoff(640, nyquist_fraction=0.0)
    with pytest.raises(ValueError):
        mtf_matched_mask(32, cutoff=0.0)
    with pytest.raises(ValueError):
        mtf_matched_mask(32, cutoff=0.45, rolloff=0.1)
