"""Waveform engine tests (pure math, no IO).

Covers WaveformEngine.compute/clip_all for every WaveformType and the
hardware-voltage clipping contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.gui.streamlit_helper.r50_controller.r50_control_service import (
    WaveformConfig,
    WaveformEngine,
    WaveformType,
)


def _cfg(**kw) -> WaveformConfig:
    defaults = dict(
        type=WaveformType.DC,
        voltage=10.0,
        amp=10.0,
        offset=5.0,
        freq=2.0,
        voltage_a=10.0,
        voltage_b=-5.0,
    )
    defaults.update(kw)
    return WaveformConfig(**defaults)


class TestCompute:
    def test_dc_constant(self) -> None:
        cfg = _cfg(type=WaveformType.DC, voltage=7.5)
        for t in (0.0, 0.25, 1.7):
            assert WaveformEngine.compute(cfg, t) == pytest.approx(7.5)

    def test_hold_same_as_dc(self) -> None:
        cfg = _cfg(type=WaveformType.HOLD, voltage=-3.0)
        assert WaveformEngine.compute(cfg, 0.9) == pytest.approx(-3.0)

    def test_sine_phase(self) -> None:
        cfg = _cfg(type=WaveformType.SINE, amp=10.0, offset=5.0, freq=2.0)
        assert WaveformEngine.compute(cfg, 0.0) == pytest.approx(5.0)
        assert WaveformEngine.compute(cfg, 0.125) == pytest.approx(15.0)  # t=1/(4f)

    def test_square_alternates(self) -> None:
        cfg = _cfg(type=WaveformType.SQUARE, voltage_a=10.0, voltage_b=-5.0, freq=1.0)
        assert WaveformEngine.compute(cfg, 0.1) == pytest.approx(10.0)
        assert WaveformEngine.compute(cfg, 0.6) == pytest.approx(-5.0)

    def test_alt_goes_zero(self) -> None:
        cfg = _cfg(type=WaveformType.ALT, voltage=8.0, freq=1.0)
        assert WaveformEngine.compute(cfg, 0.1) == pytest.approx(8.0)
        assert WaveformEngine.compute(cfg, 0.6) == pytest.approx(0.0)


class TestClipping:
    def test_clips_high(self) -> None:
        cfg = WaveformConfig(vmin=-20.0, vmax=120.0)
        assert WaveformEngine.clip_all(999.0, cfg) == 120.0

    def test_clips_low(self) -> None:
        cfg = WaveformConfig(vmin=-20.0, vmax=120.0)
        assert WaveformEngine.clip_all(-999.0, cfg) == -20.0

    def test_compute_respects_vmax(self) -> None:
        cfg = _cfg(type=WaveformType.DC, voltage=500.0, vmax=120.0)
        assert WaveformEngine.compute(cfg, 0.0) == pytest.approx(120.0)

    def test_sine_output_within_range(self) -> None:
        cfg = _cfg(type=WaveformType.SINE, amp=200.0, offset=0.0, vmin=-20.0, vmax=120.0)
        ts = np.linspace(0.0, 1.0, 100)
        out = np.array([WaveformEngine.compute(cfg, t) for t in ts])
        assert out.min() >= -20.0 and out.max() <= 120.0
