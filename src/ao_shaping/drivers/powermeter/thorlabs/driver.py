"""Thorlabs PM100 series optical power meter driver.

Implements the SCPI command set used by Thorlabs PM100/PM100A/PM100D power
meters over USB-VISA (see the Thorlabs PM100 SCPI command reference and
https://asrc-photonics.github.io/2025-python-for-photonics/python/powermeter/).

The driver follows the repository device contract (``Device`` ABC from
``ao_shaping.drivers.device_base``): parameter/capability registration, state
tracking, digital-twin support and context-manager semantics.

Design notes:
- PyVISA is optional. Pass an ``instrument`` (anything exposing ``write()`` /
  ``query()``) or a ``resource_manager`` (anything exposing ``list_resources()`` /
  ``open_resource()``) to run fully hardware-free, e.g. with test doubles.
- When neither is provided, the driver discovers a physical PM100 through
  ``VisaResourceManager`` and requires PyVISA to be installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from ao_shaping.drivers.device_base import (
    Device,
    DeviceError,
    DeviceState,
    DeviceType,
)
from ao_shaping.drivers.visa_base import (
    VisaInstrument,
    VisaResourceManager,
    is_pyvisa_available,
)


class PM100Error(DeviceError):
    """Base exception for Thorlabs PM100 driver errors."""


class PM100NotFoundError(PM100Error):
    """Raised when no Thorlabs PM100 is found on the VISA bus."""


class PM100NotConnectedError(PM100Error):
    """Raised when an operation requires an open connection."""


class ThorlabsPM100(Device):
    """Thorlabs PM100 series optical power meter.

    Example:
        >>> from ao_shaping.drivers.powermeter import ThorlabsPM100
        >>> with ThorlabsPM100() as pm:  # requires PyVISA + hardware
        ...     pm.set_wavelength(1064)
        ...     power = pm.read()
    """

    device_type = DeviceType.POWER_METER
    manufacturer = "Thorlabs"
    model = "PM100"
    version = "1.0.0"

    # Measurement configuration
    VALID_UNITS: tuple[str, ...] = ("W", "DBM")
    TRACE_SIZE: int = 100
    MIN_WAVELENGTH: float = 200.0
    MAX_WAVELENGTH: float = 11000.0

    # SCPI commands
    SCPI_CONFIGURE_POWER = "CONF:POW"
    SCPI_READ_POWER = "FETCH?"
    SCPI_WAVELENGTH = "SENS:CORR:WAV"
    SCPI_UNIT = "SENS:POW:DC:UNIT"
    SCPI_IDN = "*IDN?"
    SCPI_INIT = "INIT"
    SCPI_OPC = "*OPC?"

    def __init__(
        self,
        device_id: str = "",
        resource_name: str = "",
        instrument: Any = None,
        resource_manager: Any = None,
        default_wavelength: float = 1064.0,
    ):
        """Initialize the PM100 driver.

        Args:
            device_id: Unique device identifier (auto-generated if empty).
            resource_name: Preferred VISA resource address to connect to.
            instrument: Injected instrument-like object (``write``/``query``).
                Used for tests or pre-wrapped VISA instruments.
            resource_manager: Injected resource manager (``list_resources``/
                ``open_resource``) enabling auto-discovery without PyVISA.
            default_wavelength: Wavelength (nm) assumed before the device is
                queried.
        """
        super().__init__(device_id)

        # Connection objects (injected test doubles or real VISA objects)
        self._instrument = instrument
        self._resource_manager = resource_manager
        self._resource = None
        self._owns_resource = False
        self._owns_resource_manager = False
        self._resource_name = resource_name

        # Device identity (populated on open)
        self.idn = ""
        self.serial_number = ""
        self.firmware_version = ""

        # Measurement state
        self.wavelength = float(default_wavelength)
        self.unit = "W"
        self.background = 0.0
        self.last_reading: float | None = None
        self.trace = np.zeros(self.TRACE_SIZE, dtype=float)

        self._register_parameters()
        self._register_capabilities()

    # ==================== Device interface ====================

    def open(self) -> None:
        """Open the connection and configure the meter for power readings.

        Raises:
            PM100Error: If the resource is not a Thorlabs PM100 or PyVISA
                is unavailable.
            PM100NotFoundError: If no PM100 is found on the VISA bus.
        """
        if self.is_connected():
            return

        if self._instrument is not None:
            self._open_with_instrument(self._instrument)
        elif self._resource_manager is not None:
            self._open_with_discovery()
        else:
            self._open_real_device()

    def close(self) -> None:
        """Close the connection and release resources.

        Injected instruments/resource managers are not owned by the driver
        and are therefore left open.
        """
        if not self.is_connected():
            return

        if self._owns_resource and self._resource is not None:
            try:
                self._resource.close()
            except Exception as e:
                logger.warning(f"Error closing PM100 resource: {e}")
        self._resource = None

        if self._owns_resource_manager and self._resource_manager is not None:
            try:
                self._resource_manager.close()
            except Exception as e:
                logger.warning(f"Error closing PM100 resource manager: {e}")
            self._resource_manager = None

        self._owns_resource = False
        self._owns_resource_manager = False
        self._set_state(DeviceState.DISCONNECTED)
        logger.info(f"PM100 {self.serial_number or self._device_id} closed")

    def is_connected(self) -> bool:
        """Return True when the meter is open and ready."""
        return self._state is DeviceState.READY

    @property
    def resource_name(self) -> str:
        """VISA resource address the meter is connected to (empty if unknown)."""
        return self._resource_name

    def get_hardware_info(self) -> dict[str, Any]:
        """Return hardware identification information."""
        return {
            "manufacturer": self.manufacturer,
            "model": self.model,
            "serial_number": self.serial_number,
            "firmware_version": self.firmware_version,
            "idn": self.idn,
        }

    # ==================== Discovery ====================

    @classmethod
    def find_devices(cls, rm: Any = None) -> dict[str, str]:
        """Probe all VISA resources and return matching PM100 devices.

        Args:
            rm: Resource manager (``list_resources``/``open_resource``). When
                None, a real ``VisaResourceManager`` is created, which requires
                PyVISA.

        Returns:
            Mapping of ``*IDN?`` string to VISA resource address for every
            resource identifying itself as a Thorlabs PM100.

        Raises:
            PM100Error: If PyVISA is unavailable and no manager was injected.
        """
        owns_rm = False
        if rm is None:
            if not is_pyvisa_available():
                raise PM100Error(
                    "PyVISA is not available; install pyvisa or pass a resource manager"
                )
            rm = VisaResourceManager()
            owns_rm = True

        devices: dict[str, str] = {}
        try:
            for addr in rm.list_resources():
                try:
                    resource = rm.open_resource(
                        addr, read_termination="\n", write_termination="\r\n"
                    )
                    try:
                        idn = resource.query(cls.SCPI_IDN)
                        if "PM100" in idn:
                            devices[idn.strip()] = addr
                    finally:
                        resource.close()
                except Exception as e:
                    logger.warning(f"Failed to probe resource {addr}: {e}")
        finally:
            if owns_rm:
                try:
                    rm.close()
                except Exception as e:
                    logger.warning(f"Failed to close resource manager: {e}")
        return devices

    # ==================== Power measurement ====================

    def read(self, pure: bool = False) -> float:
        """Perform a single power measurement.

        Args:
            pure: If True, return the raw reading without subtracting the
                background.

        Returns:
            Measured power in the current display unit (W or dBm).

        Raises:
            PM100NotConnectedError: If the meter is not open.
        """
        if not self.is_connected():
            raise PM100NotConnectedError(
                "PM100 is not connected; call open() first"
            )
        self._instrument.write(self.SCPI_INIT)
        self._instrument.query(self.SCPI_OPC)
        raw_value = float(self._instrument.query(self.SCPI_READ_POWER))
        value = raw_value if pure else raw_value - self.background
        self.last_reading = value
        self.trace = np.roll(self.trace, -1)
        self.trace[-1] = value
        self._emit_data("power", value)
        logger.debug(f"PM100 reading: {value:.6g}")
        return value

    def get_background(self) -> float:
        """Measure and store the current reading as the background level."""
        self.background = self.read(pure=True)
        return self.background

    def set_background(self, value: float) -> None:
        """Set the background level subtracted from subsequent readings."""
        self.background = float(value)
        self._set_parameter_value_internal("background", self.background)

    # ==================== Wavelength ====================

    def get_wavelength(self) -> float:
        """Query and return the measurement wavelength in nanometers."""
        if not self.is_connected():
            raise PM100NotConnectedError(
                "PM100 is not connected; call open() first"
            )
        value = float(self._instrument.query(f"{self.SCPI_WAVELENGTH}?"))
        self._cache_wavelength(value)
        return value

    def set_wavelength(self, nm: float) -> float:
        """Set the measurement wavelength.

        Args:
            nm: Wavelength in nanometers (200-11000 nm).

        Returns:
            The applied wavelength.

        Raises:
            ValueError: If the wavelength is out of range.
            PM100NotConnectedError: If the meter is not open.
        """
        if not self.is_connected():
            raise PM100NotConnectedError(
                "PM100 is not connected; call open() first"
            )
        value = float(nm)
        if not self.MIN_WAVELENGTH <= value <= self.MAX_WAVELENGTH:
            raise ValueError(
                f"Wavelength {value} out of range "
                f"[{self.MIN_WAVELENGTH}, {self.MAX_WAVELENGTH}] nm"
            )
        self._instrument.write(f"{self.SCPI_WAVELENGTH} {value:g}")
        self._cache_wavelength(value)
        logger.info(f"PM100 wavelength set to {value:g} nm")
        return value

    # ==================== Unit ====================

    def get_unit(self) -> str:
        """Query and return the display unit (W or DBM)."""
        if not self.is_connected():
            raise PM100NotConnectedError(
                "PM100 is not connected; call open() first"
            )
        unit = self._instrument.query(f"{self.SCPI_UNIT}?").strip().upper()
        self._cache_unit(unit)
        return unit

    def set_unit(self, unit: str) -> str:
        """Set the display unit.

        Args:
            unit: "W" or "DBM" (case-insensitive).

        Returns:
            The applied unit, uppercased.

        Raises:
            ValueError: If the unit is not supported.
            PM100NotConnectedError: If the meter is not open.
        """
        if not self.is_connected():
            raise PM100NotConnectedError(
                "PM100 is not connected; call open() first"
            )
        normalized = unit.strip().upper()
        if normalized not in self.VALID_UNITS:
            raise ValueError(
                f"Invalid unit {unit!r}; must be one of {self.VALID_UNITS}"
            )
        self._instrument.write(f"{self.SCPI_UNIT} {normalized}")
        self._cache_unit(normalized)
        logger.info(f"PM100 unit set to {normalized}")
        return normalized

    def switch_unit(self) -> str:
        """Toggle the display unit between W and DBM.

        Returns:
            The new unit after toggling.
        """
        current = self.get_unit()
        return self.set_unit("DBM" if current == "W" else "W")

    # ==================== Registration ====================

    def _register_parameters(self) -> None:
        """Register the driver parameters on the device registry."""
        self.register_parameter(
            "wavelength",
            self.wavelength,
            min_value=self.MIN_WAVELENGTH,
            max_value=self.MAX_WAVELENGTH,
            unit="nm",
            description="Measurement wavelength in nanometers",
        )
        self.register_parameter(
            "unit",
            self.unit,
            writable=False,
            description="Display unit (W or DBM)",
        )
        self.register_parameter(
            "background",
            0.0,
            unit="W",
            description="Background level subtracted from readings",
        )

    def _register_capabilities(self) -> None:
        """Register the driver capabilities on the device registry."""
        self.register_capability(
            "measure_power",
            description="Measure optical power",
            return_type=float,
        )
        self.register_capability(
            "set_wavelength",
            description="Set the measurement wavelength",
            parameters=["wavelength"],
        )
        self.register_capability(
            "switch_unit",
            description="Toggle the display unit between W and DBM",
        )

    # ==================== Internals ====================

    def _open_with_instrument(self, instrument: Any) -> None:
        """Open using an instrument-like object exposing write()/query()."""
        idn = instrument.query(self.SCPI_IDN)
        if "PM100" not in idn:
            raise PM100Error(f"Device {idn!r} is not a Thorlabs PM100")
        self._parse_idn(idn)
        instrument.write(self.SCPI_CONFIGURE_POWER)
        wavelength = float(instrument.query(f"{self.SCPI_WAVELENGTH}?"))
        unit = instrument.query(f"{self.SCPI_UNIT}?").strip().upper()
        self._cache_wavelength(wavelength)
        self._cache_unit(unit)
        self._set_state(DeviceState.READY)
        logger.info(f"PM100 connected: {idn}")

    def _open_with_discovery(self) -> None:
        """Open the first matching PM100 found through the resource manager."""
        devices = self.find_devices(self._resource_manager)
        if not devices:
            raise PM100NotFoundError(
                "No Thorlabs PM100 found on the VISA bus"
            )
        addr = self._pick_address(devices)
        self._resource = self._resource_manager.open_resource(
            addr, read_termination="\n", write_termination="\r\n"
        )
        idn = self._resource.query(self.SCPI_IDN)
        if "PM100" not in idn:
            raise PM100Error(f"Device {idn!r} is not a Thorlabs PM100")
        self._parse_idn(idn)
        self._resource_name = addr
        self._owns_resource = True
        self._set_state(DeviceState.READY)
        logger.info(f"PM100 connected: {idn}")

    def _open_real_device(self) -> None:
        """Open a physical PM100 through PyVISA (requires hardware)."""
        if not is_pyvisa_available():
            raise PM100Error(
                "PyVISA is not available; install pyvisa or pass "
                "instrument/resource_manager"
            )
        rm = VisaResourceManager()
        resource = None
        try:
            devices = self.find_devices(rm)
            if not devices:
                raise PM100NotFoundError(
                    "No Thorlabs PM100 found on the VISA bus"
                )
            addr = self._pick_address(devices)
            resource = rm.open_resource(
                addr, read_termination="\n", write_termination="\r\n"
            )
            instrument = VisaInstrument(resource, auto_init=False)
            self._resource_manager = rm
            self._owns_resource_manager = True
            self._resource = resource
            self._owns_resource = True
            self._instrument = instrument
            self._resource_name = addr
            self._open_with_instrument(instrument)
        except Exception:
            if resource is not None:
                try:
                    resource.close()
                except Exception:
                    pass
            rm.close()
            self._resource_manager = None
            self._resource = None
            self._owns_resource = False
            self._owns_resource_manager = False
            raise

    def _pick_address(self, devices: dict[str, str]) -> str:
        """Choose the resource address to connect to.

        Prefers a match against the configured ``resource_name`` (either the
        raw address or the full IDN string), otherwise returns the first
        discovered PM100.
        """
        if self._resource_name:
            for idn, addr in devices.items():
                if addr == self._resource_name or idn == self._resource_name:
                    return addr
        return next(iter(devices.values()))

    def _parse_idn(self, idn: str) -> None:
        """Parse a ``*IDN?`` response of the form
        ``Thorlabs,PM100D,<serial>,<firmware>``."""
        self.idn = idn.strip()
        parts = [part.strip() for part in self.idn.split(",")]
        if len(parts) >= 4:
            self.serial_number = parts[2]
            self.firmware_version = parts[3]
        elif len(parts) >= 3:
            self.serial_number = parts[2]
        self._metadata.serial_number = self.serial_number
        self._metadata.firmware_version = self.firmware_version

    def _cache_wavelength(self, value: float) -> None:
        """Update the wavelength cache and the registered parameter."""
        self.wavelength = float(value)
        self._set_parameter_value_internal("wavelength", self.wavelength)

    def _cache_unit(self, unit: str) -> None:
        """Update the unit cache and the registered parameter."""
        self.unit = unit.upper()
        self._set_parameter_value_internal("unit", self.unit)

    def _set_parameter_value_internal(self, name: str, value: Any) -> None:
        """Update a registered parameter without triggering the device hook."""
        param = self._parameters.get(name)
        if param is not None:
            param.value = value

    def _on_parameter_changed(self, name: str, old_value: Any, new_value: Any) -> None:
        """Push parameter changes to the physical device."""
        if name == "wavelength":
            self.set_wavelength(float(new_value))
