from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.drivers.sim.ccd import SimulatedCCD
from ao_shaping.drivers.sim.laser import SimulatedLaser
from ao_shaping.drivers.sim.optics import SimulatedAperture, SimulatedLens, SimulatedSLM
from ao_shaping.drivers.sim.wave import SimWave


def test_simulated_ccd_repeatable_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ao_shaping.drivers.sim.ccd.simulated_ccd.time.sleep", lambda _: None)

    cam_a = SimulatedCCD(resolution=(32, 32), noise_level=0.0, random_seed=3)
    cam_b = SimulatedCCD(resolution=(32, 32), noise_level=0.0, random_seed=3)
    cam_a.open()
    cam_b.open()

    img_a = cam_a.get_numpy_image()
    img_b = cam_b.get_numpy_image()

    assert img_a.dtype == np.uint16
    assert img_a.shape == (32, 32)
    assert np.array_equal(img_a, img_b)


def test_simulated_ccd_exposure_changes_brightness(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ao_shaping.drivers.sim.ccd.simulated_ccd.time.sleep", lambda _: None)

    cam = SimulatedCCD(resolution=(32, 32), noise_level=0.0, random_seed=9)
    cam.open()
    short_img = cam.get_numpy_image()
    cam.reset_exposure_time(40)

    bright_img = cam.get_numpy_image()

    assert float(bright_img.mean()) > float(short_img.mean())


def test_simulated_laser_generates_center_weighted_fallback_wave() -> None:
    laser = SimulatedLaser(power=25.0, wavelength=1064.0, aperture=0.08, random_seed=5)
    laser.open()

    wave = laser.generate(npix=64, dpix=1e-3)

    assert isinstance(wave, dict)
    assert wave["wavefront"].shape == (64, 64)
    assert wave["power"] == pytest.approx(25.0)
    assert np.abs(wave["wavefront"][32, 32]) > np.abs(wave["wavefront"][0, 0])


def test_simulated_slm_applies_phase_without_changing_intensity() -> None:
    slm = SimulatedSLM(resolution=(64, 64), phase_range=2 * np.pi)
    slm.open()
    slm.set_phase(np.full((64, 64), np.pi / 2))
    wave = SimWave(
        wavefront=np.ones((64, 64), dtype=np.complex128),
        wavelength=1.064e-6,
        dpix=8e-6,
    )
    intensity_before = wave.intensity.copy()

    out = slm.process(wave)

    assert out is wave
    assert np.allclose(wave.intensity, intensity_before)
    assert np.allclose(np.angle(wave.wavefront)[32, 32], np.pi / 2, atol=1e-6)


def test_simulated_slm_rejects_wrong_phase_shape() -> None:
    slm = SimulatedSLM(resolution=(64, 64))
    with pytest.raises(ValueError, match="doesn't match"):
        slm.set_phase(np.zeros((32, 32)))


def test_simulated_lens_and_aperture_modify_wavefront() -> None:
    wave = SimWave(
        wavefront=np.ones((64, 64), dtype=np.complex128),
        wavelength=1.064e-6,
        dpix=1e-5,
    )
    lens = SimulatedLens(focus_length=0.2, wavelength=1064.0)
    aperture = SimulatedAperture(radius=1.5e-4)
    lens.open()
    aperture.open()

    lens.process(wave)
    after_lens = wave.wavefront.copy()
    aperture.process(wave)

    assert not np.allclose(np.angle(after_lens), 0.0)
    assert float(np.sum(np.abs(wave.wavefront) > 0)) < wave.wavefront.size
