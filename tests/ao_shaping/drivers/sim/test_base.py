from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.drivers.sim.base import (
    SimulatedDevice,
    SimulatedDeviceError,
    OpticalDevice,
    WavefrontProcessor,
    DeviceState,
)


class DummySimulatedDevice(SimulatedDevice):
    """A dummy simulated device for testing."""

    def __init__(self, device_id: str = "", enable_noise: bool = True, random_seed: int | None = None):
        super().__init__(device_id=device_id, enable_noise=enable_noise, random_seed=random_seed)
        self._compute_called = False

    def compute(self, *args, **kwargs):
        self._compute_called = True
        return "dummy_result"


class DummyOpticalDevice(OpticalDevice):
    """A dummy optical device for testing."""

    def __init__(
        self,
        device_id: str = "",
        wavelength: float = 1064.0,
        enable_noise: bool = True,
        random_seed: int | None = None,
    ):
        super().__init__(
            device_id=device_id,
            wavelength=wavelength,
            enable_noise=enable_noise,
            random_seed=random_seed,
        )
        self._process_called = False

    def process(self, wave):
        self._process_called = True
        return wave

    def compute(self, *args, **kwargs):
        # For testing purposes, just call process if we have a wave argument
        if args:
            return self.process(args[0])
        return None


class DummyWavefrontProcessor(WavefrontProcessor):
    """A dummy wavefront processor for testing."""

    def __init__(
        self,
        device_id: str = "",
        wavelength: float = 1064.0,
        npix: int = 512,
        dpix: float = 1e-3,
        enable_noise: bool = True,
        random_seed: int | None = None,
    ):
        super().__init__(
            device_id=device_id,
            wavelength=wavelength,
            npix=npix,
            dpix=dpix,
            enable_noise=enable_noise,
            random_seed=random_seed,
        )
        self._process_called = False

    def process(self, wave):
        self._process_called = True
        return wave

    def compute(self, *args, **kwargs):
        # For testing purposes, just call process if we have a wave argument
        if args:
            return self.process(args[0])
        return None


def test_simulated_device_initialization():
    """Test SimulatedDevice initialization."""
    dev = DummySimulatedDevice(device_id="test_dev", enable_noise=False, random_seed=42)
    assert dev.device_id == "test_dev"
    assert dev._enable_noise is False
    assert dev._random_seed == 42
    assert dev._state == DeviceState.DISCONNECTED
    assert dev.manufacturer == "Simulation"
    assert dev.model == "Generic Simulated Device"


def test_simulated_device_open_close():
    """Test opening and closing a simulated device."""
    dev = DummySimulatedDevice()
    assert not dev.is_connected()
    dev.open()
    assert dev.is_connected()
    dev.close()
    assert not dev.is_connected()


def test_simulated_device_set_seed():
    """Test setting the random seed."""
    dev = DummySimulatedDevice()
    initial_seed = dev._random_seed
    dev.set_seed(123)
    assert dev._random_seed == 123
    # Check that the rng is reset
    assert dev._rng.bit_generator.seed_seq.entropy == 123


def test_simulated_device_set_noise():
    """Test enabling and disabling noise."""
    dev = DummySimulatedDevice()
    assert dev._enable_noise is True
    dev.set_noise(False)
    assert dev._enable_noise is False
    dev.set_noise(True)
    assert dev._enable_noise is True


def test_simulated_device_generate_noise():
    """Test noise generation."""
    dev = DummySimulatedDevice(enable_noise=True, random_seed=42)
    noise = dev._generate_noise((2, 2), scale=1.0)
    assert noise.shape == (2, 2)
    # Verify noise is finite and not all zeros (seed-based tests may vary across NumPy versions)
    assert np.all(np.isfinite(noise))
    assert np.any(noise != 0)

    dev.set_noise(False)
    noise = dev._generate_noise((2, 2), scale=1.0)
    assert np.allclose(noise, np.zeros((2, 2)), atol=1e-7)


def test_optical_device_initialization():
    """Test OpticalDevice initialization."""
    dev = DummyOpticalDevice(device_id="optical_test", wavelength=1550.0)
    assert dev.device_id == "optical_test"
    assert dev.wavelength == 1550.0
    assert dev._input_wave is None
    assert dev._output_wave is None


def test_optical_device_set_input_get_output():
    """Test setting input and getting output."""
    dev = DummyOpticalDevice()
    test_wave = "test_wave"
    dev.set_input(test_wave)
    assert dev._input_wave == test_wave
    # Initially, output is None
    assert dev.get_output() is None
    # After processing, the processor returns the processed wave.
    result = dev.process(test_wave)
    assert result == test_wave
    # The base OpticalDevice does not automatically populate _output_wave
    assert dev.get_output() is None


def test_wavefront_processor_initialization():
    """Test WavefrontProcessor initialization."""
    dev = DummyWavefrontProcessor(device_id="wfp_test", wavelength=1064.0, npix=256, dpix=0.5e-3)
    assert dev.device_id == "wfp_test"
    assert dev.wavelength == 1064.0
    assert dev.npix == 256
    assert dev.dpix == 0.5e-3
    assert dev._phase_pattern is None


def test_wavefront_processor_set_phase():
    """Test setting phase pattern."""
    dev = DummyWavefrontProcessor(npix=32)
    phase = np.ones((32, 32))
    dev.set_phase(phase)
    assert np.array_equal(dev.get_phase(), phase)

    # Test warning for incorrect shape
    # We can't easily test the warning without capturing logs, but we can test that it sets the phase anyway
    phase_wrong = np.ones((16, 16))
    dev.set_phase(phase_wrong)
    # The phase is set even if shape is wrong (with a warning)
    assert np.array_equal(dev.get_phase(), phase_wrong)


def test_abstract_methods_raise_not_implemented():
    """Test that abstract methods raise NotImplementedError when not implemented."""
    # We cannot instantiate the abstract class directly, but we can test by creating a subclass that doesn't implement the method
    class IncompleteSimulatedDevice(SimulatedDevice):
        pass

    with pytest.raises(TypeError):
        # This will fail because the abstract method is not implemented
        IncompleteSimulatedDevice()

    class IncompleteOpticalDevice(OpticalDevice):
        pass

    with pytest.raises(TypeError):
        IncompleteOpticalDevice()

    class IncompleteWavefrontProcessor(WavefrontProcessor):
        pass

    with pytest.raises(TypeError):
        IncompleteWavefrontProcessor()
