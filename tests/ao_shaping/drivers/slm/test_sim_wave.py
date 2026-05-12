from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.drivers.sim.atmos.screens import SimulatedTurbulentScreen
from ao_shaping.drivers.sim.wave import (
    ApertureApplier,
    SimWave,
    WaveGenerator,
    WaveMetric,
    WavePropagator,
)


def test_wave_generator_requires_open() -> None:
    generator = WaveGenerator(npix=32, dpix=1e-5, wavelength=1.064e-6)
    with pytest.raises(RuntimeError, match="not connected"):
        generator.compute()


def test_sim_wave_pipeline_keeps_driver_contract() -> None:
    generator = WaveGenerator(npix=64, dpix=1e-5, wavelength=1.064e-6, beam_type="gaussian", aperture=3e-4)
    aperture = ApertureApplier(radius=1.2e-4, wavelength=1.064e-6, npix=64, dpix=1e-5)
    propagator = WavePropagator(prop_dist=0.2)

    generator.open()
    aperture.open()
    propagator.open()

    wave = generator.compute()
    assert isinstance(wave, SimWave)

    initial_power = float(np.sum(wave.intensity))
    wave = aperture.compute(wave)
    wave = propagator.compute(wave)

    assert wave.wavefront.shape == (64, 64)
    assert np.iscomplexobj(wave.wavefront)
    assert float(np.sum(wave.intensity)) <= initial_power


@pytest.mark.parametrize("center", ["origin", "centroid", "peak"])
def test_wave_metrics(center: str) -> None:
    npix = 32
    dpix = 1e-5
    x = (np.arange(npix) - npix // 2) * dpix
    xx, yy = np.meshgrid(x, x)
    intensity = np.exp(-((xx - 1.5e-5) ** 2 + (yy + 0.5e-5) ** 2) / (2 * (3e-5) ** 2))

    bucket = WaveMetric.power_bucket(
        SimWave(np.sqrt(intensity).astype(np.complex128), wavelength=1.064e-6, dpix=dpix),
        r_bucket=8e-5,
        center=center,
    )
    assert bucket > 0.0


def test_turbulent_screen_fallback_generates_opd() -> None:
    screen = SimulatedTurbulentScreen(dist=10.0, Cn2=1e-15, L0=20.0, l0=1e-3)
    screen.open()

    wave = SimWave(
        wavefront=np.ones((64, 64), dtype=np.complex128),
        wavelength=1.064e-6,
        dpix=2e-5,
    )
    wave_before = wave.wavefront.copy()

    wave_after = screen._apply_turbulence_fallback(wave)
    opd = screen.get_opd()

    assert wave_after is wave
    assert opd is not None
    assert opd.shape == (64, 64)
    assert not np.allclose(wave.wavefront, wave_before)
