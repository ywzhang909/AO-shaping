"""Unit tests for ADC driver and MockADC.

Tests the MockADC simulation class thoroughly, and verifies the
NidaqADC real-driver API (imports and error paths) without requiring
actual NI DAQ hardware.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.drivers.device_base import DeviceState
from ao_shaping.drivers.mock_devices import MockADC, MockADCError


# =============================================================================
# MockADC Tests
# =============================================================================


class TestMockADCInitialization:
    def test_default_parameters(self):
        adc = MockADC()
        assert adc._device_name == "Dev1"
        assert adc._channel == "ai0"
        assert adc._sample_rate == 5000
        assert adc._samples_per_channel == 10
        assert adc._base_voltage == 0.0
        assert adc._noise_std == 0.01
        assert adc.state == DeviceState.DISCONNECTED

    def test_custom_parameters(self):
        adc = MockADC(
            device_id="my_adc",
            device_name="Dev2",
            channel="ai3",
            sample_rate=10000,
            samples_per_channel=50,
            base_voltage=1.5,
            noise_std=0.05,
            random_seed=42,
        )
        assert adc._device_id == "my_adc"
        assert adc._device_name == "Dev2"
        assert adc._channel == "ai3"
        assert adc._sample_rate == 10000
        assert adc._samples_per_channel == 50
        assert adc._base_voltage == 1.5
        assert adc._noise_std == 0.05

    def test_auto_device_id(self):
        adc1 = MockADC()
        adc2 = MockADC()
        assert adc1._device_id != ""
        assert adc2._device_id != ""
        assert adc1._device_id != adc2._device_id

    def test_parameters_registered(self):
        adc = MockADC()
        param_names = adc.list_parameters()
        assert "device_name" in param_names
        assert "channel" in param_names
        assert "sample_rate" in param_names
        assert "samples_per_channel" in param_names
        assert "base_voltage" in param_names
        assert "noise_std" in param_names

    def test_parameter_values_accessible(self):
        adc = MockADC(base_voltage=2.0, noise_std=0.1)
        assert adc.get_parameter_value("base_voltage") == 2.0
        assert adc.get_parameter_value("noise_std") == 0.1
        assert adc.get_parameter_value("sample_rate") == 5000

    def test_class_attributes(self):
        assert MockADC.device_type is not None
        assert MockADC.manufacturer == "Mock"
        assert MockADC.model == "Simulated ADC"


class TestMockADCLifecycle:
    def test_open_closes(self):
        adc = MockADC()
        assert not adc.is_connected()
        assert adc.state == DeviceState.DISCONNECTED

        adc.open()
        assert adc.is_connected()
        assert adc.state == DeviceState.READY

        adc.close()
        assert not adc.is_connected()
        assert adc.state == DeviceState.DISCONNECTED

    def test_context_manager(self):
        with MockADC() as adc:
            assert adc.is_connected()
            assert adc.state == DeviceState.READY
        assert not adc.is_connected()
        assert adc.state == DeviceState.DISCONNECTED

    def test_open_idempotent(self):
        adc = MockADC()
        adc.open()
        adc.open()
        assert adc.is_connected()

    def test_close_idempotent(self):
        adc = MockADC()
        adc.open()
        adc.close()
        adc.close()
        assert not adc.is_connected()

    def test_health_check_when_disconnected(self):
        adc = MockADC()
        healthy, msg = adc.health_check()
        assert not healthy
        assert "not connected" in msg.lower()

    def test_health_check_when_connected(self):
        with MockADC() as adc:
            healthy, msg = adc.health_check()
            assert healthy
            assert msg == "OK"


class TestMockADCRead:
    def test_read_returns_ndarray(self):
        with MockADC() as adc:
            data = adc.read()
            assert isinstance(data, np.ndarray)

    def test_read_default_samples(self):
        with MockADC(samples_per_channel=20) as adc:
            data = adc.read()
            assert data.shape == (20,)

    def test_read_custom_samples(self):
        with MockADC() as adc:
            data = adc.read(samples=100)
            assert data.shape == (100,)

    def test_read_dtype_float64(self):
        with MockADC() as adc:
            data = adc.read()
            assert data.dtype == np.float64

    def test_read_returns_base_voltage_approximately(self):
        with MockADC(base_voltage=3.0, noise_std=0.01, random_seed=42) as adc:
            data = adc.read(samples=1000)
            assert abs(float(np.mean(data)) - 3.0) < 0.01

    def test_read_before_open_raises(self):
        adc = MockADC()
        with pytest.raises(RuntimeError, match="not connected"):
            adc.read()

    def test_read_after_close_raises(self):
        adc = MockADC()
        adc.open()
        adc.close()
        with pytest.raises(RuntimeError, match="not connected"):
            adc.read()

    def test_read_zero_noise(self):
        with MockADC(base_voltage=1.0, noise_std=0.0) as adc:
            data = adc.read(samples=10)
            np.testing.assert_array_equal(data, np.full(10, 1.0))

    def test_read_with_noise(self):
        with MockADC(base_voltage=0.0, noise_std=0.05, random_seed=123) as adc:
            data = adc.read(samples=500)
            std = float(np.std(data))
            assert 0.03 < std < 0.07


class TestMockADCReadMean:
    def test_read_mean_returns_float(self):
        with MockADC() as adc:
            result = adc.read_mean()
            assert isinstance(result, float)

    def test_read_mean_approximates_base_voltage(self):
        with MockADC(base_voltage=2.5, noise_std=0.01, random_seed=42) as adc:
            mean = adc.read_mean(samples=1000)
            assert mean == pytest.approx(2.5, abs=0.01)

    def test_read_mean_default_samples(self):
        with MockADC(samples_per_channel=30, noise_std=0.0) as adc:
            mean = adc.read_mean()
            assert mean == pytest.approx(0.0, abs=1e-10)

    def test_read_mean_custom_samples(self):
        with MockADC(base_voltage=1.0, noise_std=0.0) as adc:
            mean = adc.read_mean(samples=5)
            assert mean == 1.0


class TestMockADCErrorHandling:
    def test_get_parameter_unknown(self):
        adc = MockADC()
        with pytest.raises(KeyError):
            adc.get_parameter_value("nonexistent")

    def test_set_parameter_unknown(self):
        adc = MockADC()
        with pytest.raises(KeyError):
            adc.set_parameter_value("nonexistent", 42)

    def test_set_parameter_invalid_value(self):
        adc = MockADC()
        result = adc.set_parameter_value("sample_rate", "not a number")
        assert result is False

    def test_mock_adc_error(self):
        with pytest.raises(MockADCError):
            raise MockADCError("Test error")


class TestMockADCHardwareInfo:
    def test_get_hardware_info_keys(self):
        with MockADC(
            device_id="info_test",
            device_name="Dev3",
            channel="ai1",
            sample_rate=20000,
            samples_per_channel=25,
        ) as adc:
            info = adc.get_hardware_info()
            assert "serial_number" in info
            assert "firmware_version" in info
            assert "device_name" in info
            assert "channel" in info
            assert "sample_rate" in info
            assert "samples_per_channel" in info

    def test_get_hardware_info_values(self):
        with MockADC(
            device_id="hw_test",
            device_name="Dev5",
            channel="ai2",
            sample_rate=15000,
            samples_per_channel=40,
        ) as adc:
            info = adc.get_hardware_info()
            assert info["device_name"] == "Dev5"
            assert info["channel"] == "ai2"
            assert info["sample_rate"] == 15000
            assert info["samples_per_channel"] == 40
            assert info["serial_number"].startswith("MOCK_ADC_hw_test")
            assert info["firmware_version"] == "1.0.0-mock"

    def test_get_hardware_info_serial_number_derived_from_id(self):
        adc = MockADC(device_id="unique_abc")
        info = adc.get_hardware_info()
        assert info["serial_number"] == "MOCK_ADC_unique_a"

    def test_get_hardware_info_returns_dict(self):
        with MockADC() as adc:
            info = adc.get_hardware_info()
            assert isinstance(info, dict)


class TestMockADCReproducibility:
    def test_same_seed_produces_same_results(self):
        with MockADC(base_voltage=0.0, noise_std=0.1, random_seed=999) as adc1:
            data1 = adc1.read(samples=100)

        with MockADC(base_voltage=0.0, noise_std=0.1, random_seed=999) as adc2:
            data2 = adc2.read(samples=100)

        np.testing.assert_array_equal(data1, data2)

    def test_different_seed_produces_different_results(self):
        with MockADC(base_voltage=0.0, noise_std=0.1, random_seed=100) as adc1:
            data1 = adc1.read(samples=50)

        with MockADC(base_voltage=0.0, noise_std=0.1, random_seed=200) as adc2:
            data2 = adc2.read(samples=50)

        assert not np.array_equal(data1, data2)


class TestMockADCDeviceBaseInheritance:
    def test_has_standard_capabilities(self):
        adc = MockADC()
        assert adc.has_capability("connect")
        assert adc.has_capability("disconnect")
        assert adc.has_capability("get_status")

    def test_get_status(self):
        with MockADC(device_id="status_test") as adc:
            status = adc.get_status()
            assert status["device_id"] == "status_test"
            assert status["connected"] is True
            assert status["state"] == "READY"

    def test_repr(self):
        adc = MockADC(device_id="repr_test")
        rep = repr(adc)
        assert "MockADC" in rep
        assert "repr_tes" in rep

    def test_get_twin_state_includes_hardware(self):
        with MockADC(device_name="TwinDev") as adc:
            state = adc.get_twin_state()
            assert "hardware" in state
            assert state["hardware"]["device_name"] == "TwinDev"
            assert state["hardware"]["base_voltage"] == 0.0

    def test_set_parameter_updates_read(self):
        with MockADC(base_voltage=0.0, noise_std=0.0) as adc:
            data = adc.read(samples=5)
            np.testing.assert_array_equal(data, np.zeros(5))

            adc.set_parameter_value("base_voltage", 3.0)
            data = adc.read(samples=5)
            np.testing.assert_array_equal(data, np.full(5, 3.0))


# =============================================================================
# NidaqADC Import & Error Tests (no real hardware required)
# =============================================================================


class TestNidaqADC:
    def test_class_importable(self):
        from ao_shaping.drivers.adc import NidaqADC, NidaqADCError, NidaqADCNotFoundError

        assert NidaqADC is not None
        assert NidaqADCError is not None
        assert NidaqADCNotFoundError is not None

    def test_class_exists_in_package_init(self):
        from ao_shaping.drivers import NidaqADC, NidaqADCError

        assert NidaqADC is not None
        assert NidaqADCError is not None

    def test_class_attributes(self):
        from ao_shaping.drivers.adc import NidaqADC

        assert NidaqADC.manufacturer == "National Instruments"
        assert NidaqADC.model == "NI DAQ (nidaqmx)"

    def test_open_raises_when_nidaqmx_not_installed(self):
        from ao_shaping.drivers.adc import NidaqADC, NidaqADCError

        adc = NidaqADC()
        with pytest.raises(NidaqADCError, match="nidaqmx is not installed"):
            adc.open()

    def test_read_before_open_raises(self):
        from ao_shaping.drivers.adc import NidaqADC, NidaqADCError

        adc = NidaqADC()
        with pytest.raises(NidaqADCError, match="not open"):
            adc.read()

    def test_read_mean_before_open_raises(self):
        from ao_shaping.drivers.adc import NidaqADC, NidaqADCError

        adc = NidaqADC()
        with pytest.raises(NidaqADCError):
            adc.read_mean()

    def test_get_hardware_info_returns_dict(self):
        from ao_shaping.drivers.adc import NidaqADC

        adc = NidaqADC(device_name="Dev1", channel="ai0")
        info = adc.get_hardware_info()
        assert isinstance(info, dict)
        assert info["device_name"] == "Dev1"
        assert info["channel"] == "ai0"
        assert info["virtual_channel"] == "Dev1/ai0"
        assert "nidaqmx_available" in info

    def test_parameters_registered(self):
        from ao_shaping.drivers.adc import NidaqADC

        adc = NidaqADC()
        param_names = adc.list_parameters()
        assert "device_name" in param_names
        assert "channel" in param_names
        assert "sample_rate" in param_names
        assert "samples_per_channel" in param_names

    def test_close_is_safe_when_not_opened(self):
        from ao_shaping.drivers.adc import NidaqADC

        adc = NidaqADC()
        adc.close()
        assert not adc.is_connected()