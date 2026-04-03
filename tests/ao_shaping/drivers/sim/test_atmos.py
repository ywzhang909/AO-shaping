from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.drivers.sim.atmos import (
    SimulatedATP,
    SimulatedThermalScreen,
    SimulatedTurbulentScreen,
)
from ao_shaping.drivers.sim.wave import SimWave


def _make_wave(npix: int = 64) -> SimWave:
    return SimWave(
        wavefront=np.ones((npix, npix), dtype=np.complex128),
        wavelength=1.064e-6,
        dpix=2e-5,
    )


def test_turbulent_screen_requires_open() -> None:
    screen = SimulatedTurbulentScreen(dist=1000.0, Cn2=1e-14)
    with pytest.raises(RuntimeError, match="not connected"):
        screen.process(_make_wave())


def test_zero_turbulence_returns_zero_opd() -> None:
    screen = SimulatedTurbulentScreen(dist=1000.0, Cn2=0.0, L0=20.0, l0=1e-3)
    screen.open()
    screen.set_seed(7)
    wave = _make_wave()
    baseline = wave.wavefront.copy()

    screen._apply_turbulence_fallback(wave)
    opd = screen.get_opd()

    assert opd is not None
    assert np.allclose(opd, 0.0)
    assert np.allclose(wave.wavefront, baseline)


def test_turbulence_strength_increases_phase_std() -> None:
    stds: list[float] = []

    for cn2 in (1e-15, 1e-14, 5e-14):
        screen = SimulatedTurbulentScreen(dist=1000.0, Cn2=cn2, L0=20.0, l0=1e-3)
        screen.open()
        screen.set_seed(11)
        wave = _make_wave()
        screen._apply_turbulence_fallback(wave)
        opd = screen.get_opd()
        assert opd is not None
        stds.append(float(np.std(opd)))

    assert stds[0] < stds[1] < stds[2]


def test_thermal_screen_process_keeps_wave_shape() -> None:
    screen = SimulatedThermalScreen(dist=100.0, absorb=1e-5)
    screen.open()
    wave = _make_wave()

    out = screen.process(wave)

    assert out is wave
    assert out.wavefront.shape == (64, 64)


def test_atp_process_keeps_wave_shape() -> None:
    atp = SimulatedATP(prop_dist=500.0, layers=4, Cn2=1e-14, Thermal=False, Turbulent=True)
    atp.open()
    wave = _make_wave()

    out = atp.propagate(wave)

    assert out is wave
    assert out.wavefront.shape == (64, 64)
