from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.drivers.sim.wave import (
    ApertureApplier,
    LensApplier,
    SimWave,
    WaveGenerator,
    WaveMetric,
    WavePropagator,
    radius_metric,
)


def test_wave_generator_gaussian_beam_has_center_weighting() -> None:
    generator = WaveGenerator(
        npix=64,
        dpix=1e-5,
        wavelength=1.064e-6,
        aperture=4e-4,
        beam_type="gaussian",
    )
    generator.open()

    wave = generator.generate()
    center = wave.intensity[wave.npix // 2, wave.npix // 2]
    corner = wave.intensity[0, 0]

    assert isinstance(wave, SimWave)
    assert wave.wavefront.shape == (64, 64)
    assert center > corner


def test_aperture_reduces_total_power() -> None:
    wave = SimWave(
        wavefront=np.ones((64, 64), dtype=np.complex128),
        wavelength=1.064e-6,
        dpix=1e-5,
    )
    aperture = ApertureApplier(radius=1.5e-4, wavelength=1.064e-6, npix=64, dpix=1e-5)
    aperture.open()

    initial_power = float(np.sum(wave.intensity))
    aperture.apply(wave)

    assert float(np.sum(wave.intensity)) < initial_power


def test_lens_then_propagation_preserves_total_power() -> None:
    wave = SimWave(
        wavefront=np.ones((64, 64), dtype=np.complex128),
        wavelength=1.064e-6,
        dpix=1e-5,
    )
    lens = LensApplier(focal_length=0.2, wavelength=1.064e-6, npix=64, dpix=1e-5)
    propagator = WavePropagator(prop_dist=0.2)
    lens.open()
    propagator.open()

    initial_power = float(np.sum(wave.intensity))
    lens.apply(wave)
    propagator.propagate(wave)

    assert np.iscomplexobj(wave.wavefront)
    assert float(np.sum(wave.intensity)) == pytest.approx(initial_power, rel=1e-6)


def test_radius_metric_grows_with_energy_threshold() -> None:
    npix = 64
    dpix = 1e-5
    coord = (np.arange(npix) - npix // 2) * dpix
    xx, yy = np.meshgrid(coord, coord)
    intensity = np.exp(-(xx**2 + yy**2) / (2 * (4e-5) ** 2))

    r50 = radius_metric(intensity, xx, yy, center="centroid", energy=0.5)
    r90 = radius_metric(intensity, xx, yy, center="centroid", energy=0.9)

    assert r50 > 0
    assert r90 > r50


def test_wave_metric_centroid_tracks_shifted_spot() -> None:
    npix = 64
    dpix = 1e-5
    coord = (np.arange(npix) - npix // 2) * dpix
    xx, yy = np.meshgrid(coord, coord)
    intensity = np.exp(-((xx - 2e-5) ** 2 + (yy + 3e-5) ** 2) / (2 * (5e-5) ** 2))

    cx, cy = WaveMetric.centroid(intensity, xx, yy)

    assert cx == pytest.approx(2e-5, abs=5e-6)
    assert cy == pytest.approx(-3e-5, abs=5e-6)
