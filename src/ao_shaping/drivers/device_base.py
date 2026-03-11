"""Device base class for digital twin management.

This module provides a unified interface for all hardware devices,
enabling device registration, metadata management, and digital twin support.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, ClassVar

from loguru import logger


class DeviceType(Enum):
    """Device type classification."""

    CAMERA = auto()
    SLM = auto()
    DM = auto()  # Deformable Mirror
    WFS = auto()  # Wavefront Sensor
    STAGE = auto()  # Motion stage
    LASER = auto()
    FILTER = auto()
    OTHER = auto()


class DeviceState(Enum):
    """Device operational state."""

    UNKNOWN = auto()
    DISCONNECTED = auto()
    CONNECTING = auto()
    READY = auto()
    BUSY = auto()
    ERROR = auto()
    CALIBRATING = auto()


@dataclass
class DeviceParameter:
    """Device parameter metadata.

    Attributes:
        name: Parameter name.
        value: Current value.
        value_type: Data type of the parameter.
        min_value: Minimum allowed value (optional).
        max_value: Maximum allowed value (optional).
        unit: Physical unit (e.g., "ms", "nm", "V").
        description: Human-readable description.
        writable: Whether the parameter can be set.
    """

    name: str
    value: Any
    value_type: type = float
    min_value: float | None = None
    max_value: float | None = None
    unit: str = ""
    description: str = ""
    writable: bool = True

    def validate(self, value: Any) -> bool:
        """Validate if value is within allowed range."""
        if not isinstance(value, self.value_type):
            try:
                value = self.value_type(value)
            except (TypeError, ValueError):
                return False

        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return True


@dataclass
class DeviceCapability:
    """Device capability metadata.

    Attributes:
        name: Capability name.
        description: Human-readable description.
        parameters: Required parameters for this capability.
        return_type: Expected return type.
    """

    name: str
    description: str = ""
    parameters: list[str] = field(default_factory=list)
    return_type: type | None = None


@dataclass
class DeviceMetadata:
    """Device metadata for digital twin registration.

    Attributes:
        device_id: Unique device identifier (UUID).
        device_type: Device classification.
        manufacturer: Device manufacturer.
        model: Device model name.
        serial_number: Hardware serial number.
        firmware_version: Firmware version.
        hardware_version: Hardware version.
        connection_info: Connection parameters (address, port, etc.).
        registration_time: When the device was registered.
        last_seen: Last communication timestamp.
    """

    device_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    device_type: DeviceType = DeviceType.OTHER
    manufacturer: str = "Unknown"
    model: str = "Unknown"
    serial_number: str = ""
    firmware_version: str = ""
    hardware_version: str = ""
    connection_info: dict[str, Any] = field(default_factory=dict)
    registration_time: datetime = field(default_factory=datetime.now)
    last_seen: datetime | None = None


class Device(ABC):
    """Abstract base class for all hardware devices.

    This class provides a unified interface for device management,
    enabling digital twin functionality and standardized device control.

    All device drivers must inherit from this class and implement
    the abstract methods.

    Example:
        >>> class MyCamera(Device):
        ...     device_type = DeviceType.CAMERA
        ...     manufacturer = "MyCam"
        ...     model = "MC-100"
        ...
        ...     def __init__(self, device_id: str = ""):
        ...         super().__init__(device_id)
        ...         self._register_parameters()
        ...
        ...     def _register_parameters(self) -> None:
        ...         self.register_parameter(
        ...             "exposure_time",
        ...             20.0,
        ...             min_value=1.0,
        ...             max_value=1000.0,
        ...             unit="ms",
        ...             description="Exposure time in milliseconds"
        ...         )
    """

    # Class-level device identification
    device_type: ClassVar[DeviceType] = DeviceType.OTHER
    manufacturer: ClassVar[str] = "Unknown"
    model: ClassVar[str] = "Unknown"
    version: ClassVar[str] = "1.0.0"

    def __init__(self, device_id: str = ""):
        """Initialize device base.

        Args:
            device_id: Unique device identifier. If empty, auto-generated.
        """
        # Device identification
        self._device_id = device_id or str(uuid.uuid4())
        self._state = DeviceState.DISCONNECTED
        self._error_message: str | None = None

        # Parameter registry: name -> DeviceParameter
        self._parameters: dict[str, DeviceParameter] = {}

        # Capability registry: name -> DeviceCapability
        self._capabilities: dict[str, DeviceCapability] = {}

        # Data callbacks for streaming acquisition
        self._data_callbacks: list[Callable[[str, Any], None]] = []

        # Initialize metadata
        self._metadata = DeviceMetadata(
            device_id=self._device_id,
            device_type=self.device_type,
            manufacturer=self.manufacturer,
            model=self.model,
        )

        # Register standard capabilities
        self._register_standard_capabilities()

        logger.debug(f"Device {self._device_id} ({self.model}) initialized")

    # ==================== Abstract Methods ====================

    @abstractmethod
    def open(self) -> None:
        """Open connection to the device.

        Raises:
            ConnectionError: If connection fails.
            DeviceError: If device initialization fails.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close connection and release resources."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if device is connected and ready."""
        pass

    @abstractmethod
    def get_hardware_info(self) -> dict[str, Any]:
        """Get hardware-specific information.

        Returns:
            Dictionary containing hardware info (serial, firmware, etc.).
        """
        pass

    # ==================== Context Manager ====================

    def __enter__(self) -> "Device":
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self._device_id[:8]}, "
            f"model={self.manufacturer}_{self.model}, "
            f"state={self._state.name}"
            f")"
        )

    # ==================== Device Registration ====================

    @property
    def device_id(self) -> str:
        """Unique device identifier."""
        return self._device_id

    @property
    def metadata(self) -> DeviceMetadata:
        """Device metadata for registration."""
        # Update last seen timestamp
        if self.is_connected():
            self._metadata.last_seen = datetime.now()
        return self._metadata

    @property
    def state(self) -> DeviceState:
        """Current device state."""
        return self._state

    def _set_state(self, state: DeviceState, error_msg: str | None = None) -> None:
        """Update device state."""
        old_state = self._state
        self._state = state
        self._error_message = error_msg

        if state == DeviceState.ERROR and error_msg:
            logger.error(f"Device {self._device_id} error: {error_msg}")
        elif old_state != state:
            logger.debug(f"Device {self._device_id} state: {old_state.name} -> {state.name}")

    # ==================== Parameter Management ====================

    def register_parameter(
        self,
        name: str,
        default_value: Any,
        min_value: float | None = None,
        max_value: float | None = None,
        unit: str = "",
        description: str = "",
        writable: bool = True,
    ) -> None:
        """Register a device parameter.

        Args:
            name: Parameter name.
            default_value: Default value.
            min_value: Minimum allowed value.
            max_value: Maximum allowed value.
            unit: Physical unit.
            description: Human-readable description.
            writable: Whether the parameter can be modified.
        """
        self._parameters[name] = DeviceParameter(
            name=name,
            value=default_value,
            value_type=type(default_value),
            min_value=min_value,
            max_value=max_value,
            unit=unit,
            description=description,
            writable=writable,
        )

    def get_parameter(self, name: str) -> DeviceParameter | None:
        """Get parameter metadata."""
        return self._parameters.get(name)

    def get_parameter_value(self, name: str) -> Any:
        """Get current parameter value."""
        if name not in self._parameters:
            raise KeyError(f"Parameter '{name}' not found")
        return self._parameters[name].value

    def set_parameter_value(self, name: str, value: Any) -> bool:
        """Set parameter value with validation.

        Args:
            name: Parameter name.
            value: New value.

        Returns:
            True if successful, False if validation fails.

        Raises:
            KeyError: If parameter doesn't exist.
            PermissionError: If parameter is read-only.
        """
        if name not in self._parameters:
            raise KeyError(f"Parameter '{name}' not found")

        param = self._parameters[name]
        if not param.writable:
            raise PermissionError(f"Parameter '{name}' is read-only")

        if not param.validate(value):
            logger.warning(f"Invalid value {value} for parameter '{name}'")
            return False

        # Update value
        old_value = param.value
        param.value = value

        # Call parameter change hook
        self._on_parameter_changed(name, old_value, value)

        logger.debug(f"Parameter '{name}': {old_value} -> {value}")
        return True

    def list_parameters(self) -> list[str]:
        """List all registered parameter names."""
        return list(self._parameters.keys())

    def get_all_parameters(self) -> dict[str, DeviceParameter]:
        """Get all registered parameters."""
        return self._parameters.copy()

    def _on_parameter_changed(self, name: str, old_value: Any, new_value: Any) -> None:
        """Hook called when a parameter changes.

        Override in subclass to implement device-specific logic.
        """
        pass

    # ==================== Capability Management ====================

    def register_capability(
        self,
        name: str,
        description: str = "",
        parameters: list[str] | None = None,
        return_type: type | None = None,
    ) -> None:
        """Register a device capability.

        Args:
            name: Capability name.
            description: Human-readable description.
            parameters: Required parameter names.
            return_type: Expected return type.
        """
        self._capabilities[name] = DeviceCapability(
            name=name,
            description=description,
            parameters=parameters or [],
            return_type=return_type,
        )

    def has_capability(self, name: str) -> bool:
        """Check if device has a capability."""
        return name in self._capabilities

    def get_capability(self, name: str) -> DeviceCapability | None:
        """Get capability metadata."""
        return self._capabilities.get(name)

    def list_capabilities(self) -> list[str]:
        """List all registered capability names."""
        return list(self._capabilities.keys())

    def _register_standard_capabilities(self) -> None:
        """Register standard capabilities common to all devices."""
        self.register_capability(
            "connect",
            description="Connect to the device",
        )
        self.register_capability(
            "disconnect",
            description="Disconnect from the device",
        )
        self.register_capability(
            "get_status",
            description="Get device status information",
            return_type=dict,
        )

    # ==================== Data Acquisition ====================

    def register_data_callback(self, callback: Callable[[str, Any], None]) -> None:
        """Register a callback for data streaming.

        Args:
            callback: Function called with (data_type, data) on new data.
        """
        self._data_callbacks.append(callback)

    def unregister_data_callback(self, callback: Callable[[str, Any], None]) -> None:
        """Unregister a data callback."""
        if callback in self._data_callbacks:
            self._data_callbacks.remove(callback)

    def _emit_data(self, data_type: str, data: Any) -> None:
        """Emit data to all registered callbacks."""
        for callback in self._data_callbacks:
            try:
                callback(data_type, data)
            except Exception as e:
                logger.warning(f"Data callback error: {e}")

    # ==================== Digital Twin Support ====================

    def get_twin_state(self) -> dict[str, Any]:
        """Get current state for digital twin synchronization.

        Returns:
            Dictionary containing device state, parameters, and metadata.
        """
        return {
            "device_id": self._device_id,
            "device_type": self.device_type.name,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "state": self._state.name,
            "parameters": {
                name: {
                    "value": param.value,
                    "unit": param.unit,
                }
                for name, param in self._parameters.items()
            },
            "capabilities": list(self._capabilities.keys()),
            "metadata": {
                "serial_number": self._metadata.serial_number,
                "firmware_version": self._metadata.firmware_version,
                "hardware_version": self._metadata.hardware_version,
            },
        }

    def sync_from_twin(self, twin_state: dict[str, Any]) -> None:
        """Synchronize device state from digital twin.

        Args:
            twin_state: State dictionary from digital twin.
        """
        # Update parameters
        if "parameters" in twin_state:
            for name, param_data in twin_state["parameters"].items():
                if name in self._parameters:
                    try:
                        self.set_parameter_value(name, param_data["value"])
                    except Exception as e:
                        logger.warning(f"Failed to sync parameter '{name}': {e}")

    # ==================== Utility Methods ====================

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive device status."""
        return {
            "device_id": self._device_id,
            "model": f"{self.manufacturer}_{self.model}",
            "state": self._state.name,
            "error": self._error_message,
            "connected": self.is_connected(),
            "parameter_count": len(self._parameters),
            "capability_count": len(self._capabilities),
        }

    def health_check(self) -> tuple[bool, str]:
        """Perform health check on the device.

        Returns:
            Tuple of (is_healthy, message).
        """
        if not self.is_connected():
            return False, "Device not connected"

        if self._state == DeviceState.ERROR:
            return False, self._error_message or "Device in error state"

        return True, "OK"


class DeviceError(Exception):
    """Base exception for device errors."""

    pass


class DeviceNotFoundError(DeviceError):
    """Raised when device is not found."""

    pass


class DeviceBusyError(DeviceError):
    """Raised when device is busy."""

    pass
