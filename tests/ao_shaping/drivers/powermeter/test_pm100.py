"""Tests for the Thorlabs PM100 power meter driver.

Mock-based tests run without hardware and without PyVISA (injecting
scripted test doubles through the ``instrument`` / ``resource_manager``
constructor arguments). Hardware-dependent tests are skipped when no
PM100 is reachable on the VISA bus.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.drivers.device_base import DeviceState, DeviceType
from ao_shaping.drivers.powermeter import (
    PM100Error,
    PM100NotFoundError,
    PM100NotConnectedError,
    ThorlabsPM100,
)
from ao_shaping.drivers.visa_base import is_pyvisa_available


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeVisaInstrument:
    """Scripted fake VISA instrument.

    Records all ``write()`` / ``query()`` calls and serves canned responses.
    ``SENS:POW:DC:UNIT`` / ``SENS:CORR:WAV`` writes update the query
    responses so set-then-get round trips behave like the real device.
    """

    def __init__(self, idn: str = "Thorlabs,PM100D,P1001234,1.6.0"):
        self.idn = idn
        self.writes: list[str] = []
        self.queries: list[str] = []
        self.closed = False
        self.responses: dict[str, str] = {
            "*IDN?": idn,
            "SENS:CORR:WAV?": "1064",
            "SENS:POW:DC:UNIT?": "W",
            "*OPC?": "1",
            "FETCH?": "0.001234",
        }

    def write(self, command: str) -> None:
        self.writes.append(command)
        if command.startswith("SENS:POW:DC:UNIT "):
            self.responses["SENS:POW:DC:UNIT?"] = command.split()[-1]
        elif command.startswith("SENS:CORR:WAV "):
            self.responses["SENS:CORR:WAV?"] = str(int(command.split()[-1]))

    def query(self, command: str) -> str:
        self.queries.append(command)
        if command not in self.responses:
            raise RuntimeError(f"Unexpected query: {command!r}")
        return self.responses[command]

    def close(self) -> None:
        self.closed = True


class FakeResource:
    """Minimal VISA resource double returned by FakeResourceManager."""

    def __init__(self, idn: str):
        self.idn = idn
        self.closed = False

    def query(self, command: str) -> str:
        if command != "*IDN?":
            raise RuntimeError(f"Unexpected query: {command!r}")
        return self.idn

    def close(self) -> None:
        self.closed = True


class FakeResourceManager:
    """Fake VISA resource manager with a fixed address -> IDN table."""

    def __init__(self, resources: dict[str, str]):
        self.resources = resources
        self.opened: list[tuple[str, dict]] = []

    def list_resources(self, query: str = "?*::INSTR"):
        return tuple(self.resources.keys())

    def open_resource(self, addr: str, **kwargs) -> FakeResource:
        if addr not in self.resources:
            raise RuntimeError(f"Unknown resource: {addr}")
        self.opened.append((addr, kwargs))
        return FakeResource(self.resources[addr])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pm100():
    """Unopened ThorlabsPM100 with an injected fake instrument."""
    fake = FakeVisaInstrument()
    return ThorlabsPM100(instrument=fake), fake


@pytest.fixture
def open_pm100(pm100):
    """Opened ThorlabsPM100 with an injected fake instrument."""
    pm, fake = pm100
    pm.open()
    return pm, fake


# ---------------------------------------------------------------------------
# Constants and class attributes
# ---------------------------------------------------------------------------


class TestPM100Constants:
    def test_valid_units(self):
        assert ThorlabsPM100.VALID_UNITS == ("W", "DBM")

    def test_trace_size(self):
        assert ThorlabsPM100.TRACE_SIZE == 100

    def test_scpi_constants(self):
        assert ThorlabsPM100.SCPI_CONFIGURE_POWER == "CONF:POW"
        assert ThorlabsPM100.SCPI_READ_POWER == "FETCH?"
        assert ThorlabsPM100.SCPI_WAVELENGTH == "SENS:CORR:WAV"
        assert ThorlabsPM100.SCPI_UNIT == "SENS:POW:DC:UNIT"
        assert ThorlabsPM100.SCPI_IDN == "*IDN?"

    def test_device_type(self):
        assert ThorlabsPM100.device_type is DeviceType.POWER_METER
        assert DeviceType.POWER_METER.name == "POWER_METER"

    def test_class_identification(self):
        assert ThorlabsPM100.manufacturer == "Thorlabs"
        assert ThorlabsPM100.model == "PM100"


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestPM100Initialization:
    def test_default_state(self, pm100):
        pm, _ = pm100
        assert pm.state is DeviceState.DISCONNECTED
        assert not pm.is_connected()
        assert pm.background == 0.0
        assert pm.last_reading is None
        assert pm.trace.shape == (100,)
        assert np.all(pm.trace == 0.0)
        assert pm.idn == ""

    def test_registered_parameters(self, pm100):
        pm, _ = pm100
        assert set(pm.list_parameters()) >= {"wavelength", "unit", "background"}
        wl = pm.get_parameter("wavelength")
        assert wl.unit == "nm"
        assert wl.min_value == 200.0
        assert wl.max_value == 11000.0
        assert pm.get_parameter_value("background") == 0.0
        assert not pm.get_parameter("unit").writable

    def test_registered_capabilities(self, pm100):
        pm, _ = pm100
        assert "measure_power" in pm.list_capabilities()
        assert "set_wavelength" in pm.list_capabilities()
        assert "switch_unit" in pm.list_capabilities()

    def test_repr(self, pm100):
        pm, _ = pm100
        assert "ThorlabsPM100" in repr(pm)
        assert "PM100" in repr(pm)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestPM100FindDevices:
    def test_finds_pm100_among_mixed_resources(self):
        rm = FakeResourceManager(
            {
                "USB0::0x1313::0x8072::P1000001::INSTR": "Thorlabs,PM100D,P1000001,1.6.0",
                "USB0::0x1313::0x8072::P1000002::INSTR": "Thorlabs,PM100A,P1000002,1.0.0",
                "USB0::0x2A8D::0x0101::SN123::INSTR": "Keysight,34461A,SN123,2.4.0",
            }
        )
        devices = ThorlabsPM100.find_devices(rm)
        assert len(devices) == 2
        assert devices["Thorlabs,PM100D,P1000001,1.6.0"] == "USB0::0x1313::0x8072::P1000001::INSTR"
        assert devices["Thorlabs,PM100A,P1000002,1.0.0"] == "USB0::0x1313::0x8072::P1000002::INSTR"
        # non-PM100 resource was still probed and closed
        assert len(rm.opened) == 3

    def test_returns_empty_when_no_pm100(self):
        rm = FakeResourceManager({"USB0::0x2A8D::0x0101::SN123::INSTR": "Keysight,34461A,SN123,2.4.0"})
        assert ThorlabsPM100.find_devices(rm) == {}

    def test_missing_visa_raises_pm100_error(self):
        if is_pyvisa_available():
            pytest.skip("PyVISA is installed; cannot exercise missing-VISA path")
        with pytest.raises(PM100Error):
            ThorlabsPM100.find_devices()


# ---------------------------------------------------------------------------
# Open / close
# ---------------------------------------------------------------------------


class TestPM100OpenClose:
    def test_open_success(self, open_pm100):
        pm, fake = open_pm100
        assert pm.state is DeviceState.READY
        assert pm.is_connected()
        assert pm.idn == "Thorlabs,PM100D,P1001234,1.6.0"
        assert pm.serial_number == "P1001234"
        assert pm.firmware_version == "1.6.0"
        assert pm.wavelength == 1064.0
        assert pm.unit == "W"
        # configuration + identification sequence
        assert fake.writes == ["CONF:POW"]
        assert fake.queries[0] == "*IDN?"
        assert "SENS:CORR:WAV?" in fake.queries
        assert "SENS:POW:DC:UNIT?" in fake.queries

    def test_open_rejects_non_pm100(self):
        fake = FakeVisaInstrument(idn="Keysight,34461A,SN123,2.4.0")
        pm = ThorlabsPM100(instrument=fake)
        with pytest.raises(PM100Error, match="not a Thorlabs PM100"):
            pm.open()
        assert not pm.is_connected()

    def test_double_open_is_noop(self, open_pm100):
        pm, fake = open_pm100
        writes_before = list(fake.writes)
        pm.open()
        assert pm.is_connected()
        assert fake.writes == writes_before

    def test_close(self, open_pm100):
        pm, fake = open_pm100
        pm.close()
        assert pm.state is DeviceState.DISCONNECTED
        assert not pm.is_connected()
        # injected instrument is not owned by the driver, so not closed
        assert not fake.closed

    def test_context_manager(self):
        fake = FakeVisaInstrument()
        with ThorlabsPM100(instrument=fake) as pm:
            assert pm.is_connected()
        assert not pm.is_connected()
        assert pm.state is DeviceState.DISCONNECTED

    def test_open_auto_discovers_resource(self):
        rm = FakeResourceManager(
            {"USB0::0x1313::0x8072::P1000001::INSTR": "Thorlabs,PM100D,P1000001,1.6.0"}
        )
        pm = ThorlabsPM100(resource_manager=rm)
        pm.open()
        assert pm.is_connected()
        assert pm.resource_name == "USB0::0x1313::0x8072::P1000001::INSTR"
        assert pm.serial_number == "P1000001"
        pm.close()

    def test_open_no_device_found(self):
        rm = FakeResourceManager({})
        pm = ThorlabsPM100(resource_manager=rm)
        with pytest.raises(PM100NotFoundError):
            pm.open()

    def test_open_without_visa_raises_pm100_error(self):
        if is_pyvisa_available():
            pytest.skip("PyVISA is installed; cannot exercise missing-VISA path")
        pm = ThorlabsPM100()
        with pytest.raises(PM100Error):
            pm.open()


# ---------------------------------------------------------------------------
# Power reading
# ---------------------------------------------------------------------------


class TestPM100Read:
    def test_read_returns_scripted_value(self, open_pm100):
        pm, fake = open_pm100
        value = pm.read()
        assert value == pytest.approx(0.001234)
        assert pm.last_reading == pytest.approx(0.001234)
        assert pm.trace[-1] == pytest.approx(0.001234)
        assert pm.trace.shape == (100,)
        # read sequence matches ThorlabsPM100: INIT -> *OPC? -> FETCH?
        assert fake.writes[-1] == "INIT"
        assert fake.queries[-2:] == ["*OPC?", "FETCH?"]

    def test_read_pure_returns_raw(self, open_pm100):
        pm, _ = open_pm100
        pm.background = 0.001
        assert pm.read(pure=True) == pytest.approx(0.001234)

    def test_read_subtracts_background(self, open_pm100):
        pm, _ = open_pm100
        pm.background = 0.0002
        assert pm.read() == pytest.approx(0.001234 - 0.0002)

    def test_read_when_not_connected(self, pm100):
        pm, _ = pm100
        with pytest.raises(PM100NotConnectedError):
            pm.read()

    def test_trace_rolls_over(self, open_pm100):
        pm, fake = open_pm100
        fake.responses["FETCH?"] = "1.0"
        for _ in range(105):
            pm.read()
        assert pm.trace.shape == (100,)
        assert np.all(pm.trace == 1.0)


# ---------------------------------------------------------------------------
# Wavelength
# ---------------------------------------------------------------------------


class TestPM100Wavelength:
    def test_get_wavelength(self, open_pm100):
        pm, _ = open_pm100
        assert pm.get_wavelength() == pytest.approx(1064.0)
        assert pm.wavelength == pytest.approx(1064.0)

    def test_set_wavelength_writes_scpi(self, open_pm100):
        pm, fake = open_pm100
        result = pm.set_wavelength(532)
        assert "SENS:CORR:WAV 532" in fake.writes
        assert result == pytest.approx(532.0)
        assert pm.wavelength == pytest.approx(532.0)

    def test_set_wavelength_out_of_range(self, open_pm100):
        pm, fake = open_pm100
        with pytest.raises(ValueError, match="200"):
            pm.set_wavelength(50)
        assert "SENS:CORR:WAV 50" not in fake.writes

    def test_get_wavelength_when_not_connected(self, pm100):
        pm, _ = pm100
        with pytest.raises(PM100NotConnectedError):
            pm.get_wavelength()


# ---------------------------------------------------------------------------
# Unit
# ---------------------------------------------------------------------------


class TestPM100Unit:
    def test_get_unit(self, open_pm100):
        pm, _ = open_pm100
        assert pm.get_unit() == "W"

    def test_set_unit_lowercase_input(self, open_pm100):
        pm, fake = open_pm100
        result = pm.set_unit("dBm")
        assert "SENS:POW:DC:UNIT DBM" in fake.writes
        assert result == "DBM"
        assert pm.unit == "DBM"

    def test_set_unit_invalid(self, open_pm100):
        pm, fake = open_pm100
        with pytest.raises(ValueError, match="W"):
            pm.set_unit("mV")
        assert "SENS:POW:DC:UNIT mV" not in fake.writes

    def test_switch_unit(self, open_pm100):
        pm, fake = open_pm100
        assert pm.switch_unit() == "DBM"
        assert "SENS:POW:DC:UNIT DBM" in fake.writes
        assert pm.switch_unit() == "W"
        assert "SENS:POW:DC:UNIT W" in fake.writes


# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------


class TestPM100Background:
    def test_get_background(self, open_pm100):
        pm, _ = open_pm100
        background = pm.get_background()
        assert background == pytest.approx(0.001234)
        assert pm.background == pytest.approx(0.001234)
        # subsequent read() is zeroed by the background
        assert pm.read() == pytest.approx(0.0)

    def test_set_background(self, pm100):
        pm, _ = pm100
        pm.set_background(0.5)
        assert pm.background == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Parameter registry integration
# ---------------------------------------------------------------------------


class TestPM100Parameters:
    def test_set_wavelength_parameter_writes_device(self, open_pm100):
        pm, fake = open_pm100
        assert pm.set_parameter_value("wavelength", 532) is True
        assert pm.get_parameter_value("wavelength") == 532
        assert "SENS:CORR:WAV 532" in fake.writes

    def test_set_wavelength_parameter_invalid(self, open_pm100):
        pm, fake = open_pm100
        assert pm.set_parameter_value("wavelength", 1) is False
        assert pm.get_parameter_value("wavelength") == 1064
        assert "SENS:CORR:WAV 1" not in fake.writes

    def test_unit_parameter_readonly(self, open_pm100):
        pm, _ = open_pm100
        with pytest.raises(PermissionError):
            pm.set_parameter_value("unit", "DBM")


# ---------------------------------------------------------------------------
# Device interface conformance
# ---------------------------------------------------------------------------


class TestPM100DeviceInterface:
    def test_get_hardware_info(self, open_pm100):
        pm, _ = open_pm100
        info = pm.get_hardware_info()
        assert info["manufacturer"] == "Thorlabs"
        assert info["model"] == "PM100"
        assert info["serial_number"] == "P1001234"
        assert info["firmware_version"] == "1.6.0"
        assert "PM100" in info["idn"]

    def test_health_check_disconnected(self, pm100):
        pm, _ = pm100
        healthy, message = pm.health_check()
        assert healthy is False
        assert "not connected" in message

    def test_health_check_connected(self, open_pm100):
        pm, _ = open_pm100
        healthy, message = pm.health_check()
        assert healthy is True
        assert message == "OK"

    def test_get_status(self, open_pm100):
        pm, _ = open_pm100
        status = pm.get_status()
        assert status["connected"] is True
        assert status["model"] == "Thorlabs_PM100"

    def test_get_twin_state(self, open_pm100):
        pm, _ = open_pm100
        state = pm.get_twin_state()
        assert state["device_type"] == "POWER_METER"
        assert "wavelength" in state["parameters"]


# ---------------------------------------------------------------------------
# Hardware-dependent tests (skipped without a physical PM100)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pm100_hardware():
    """Real PM100 session; skipped when no device is reachable."""
    pm = ThorlabsPM100()
    try:
        pm.open()
    except Exception as e:
        pytest.skip(f"PM100 hardware not available: {e}")
    yield pm
    pm.close()


def test_hardware_idn(pm100_hardware):
    assert "PM100" in pm100_hardware.idn


def test_hardware_read(pm100_hardware):
    value = pm100_hardware.read()
    assert isinstance(value, float)
    assert value == value  # not NaN


def test_hardware_wavelength_roundtrip(pm100_hardware):
    original = pm100_hardware.get_wavelength()
    pm100_hardware.set_wavelength(original)
    assert pm100_hardware.get_wavelength() == pytest.approx(original)
