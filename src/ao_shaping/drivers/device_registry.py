"""Device registry for digital twin management.

Provides centralized device registration, discovery, and management
for building digital twin systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger

from ao_shaping.drivers.device_base import (
    Device,
    DeviceError,
    DeviceMetadata,
    DeviceState,
    DeviceType,
)


@dataclass
class RegisteredDevice:
    """Wrapper for registered device with additional management info."""

    device: Device
    alias: str = ""  # User-friendly name
    tags: list[str] = field(default_factory=list)  # For grouping/filtering
    priority: int = 0  # Device priority (higher = more important)
    auto_connect: bool = False  # Auto-connect on registry load
    twin_sync_enabled: bool = True  # Enable digital twin sync


class DeviceRegistry:
    """Central registry for managing all hardware devices.

    This class provides:
    - Device registration and discovery
    - Unified access to all devices
    - Digital twin state management
    - Batch operations on device groups

    Example:
        >>> registry = DeviceRegistry()
        >>>
        >>> # Register devices
        >>> registry.register(camera, alias="main_camera", tags=["imaging"])
        >>> registry.register(slm, alias="phase_modulator", tags=["modulation"])
        >>>
        >>> # Find devices
        >>> cameras = registry.find_by_type(DeviceType.CAMERA)
        >>> imaging_devices = registry.find_by_tag("imaging")
        >>>
        >>> # Batch operations
        >>> registry.connect_all()
        >>> states = registry.get_all_twin_states()
    """

    def __init__(self):
        """Initialize empty device registry."""
        # device_id -> RegisteredDevice
        self._devices: dict[str, RegisteredDevice] = {}

        # alias -> device_id (for quick lookup)
        self._aliases: dict[str, str] = {}

        # Callbacks for registry events
        self._on_device_registered: list[Callable[[str, Device], None]] = []
        self._on_device_unregistered: list[Callable[[str, Device], None]] = []
        self._on_state_change: list[Callable[[str, DeviceState, DeviceState], None]] = []

        logger.info("Device registry initialized")

    # ==================== Registration ====================

    def register(
        self,
        device: Device,
        alias: str = "",
        tags: list[str] | None = None,
        priority: int = 0,
        auto_connect: bool = False,
        twin_sync_enabled: bool = True,
    ) -> str:
        """Register a device in the registry.

        Args:
            device: Device instance to register.
            alias: User-friendly name for the device.
            tags: Tags for grouping/filtering devices.
            priority: Device priority (higher = more important).
            auto_connect: Whether to auto-connect when registry loads.
            twin_sync_enabled: Enable digital twin synchronization.

        Returns:
            Device ID.

        Raises:
            ValueError: If alias is already in use.
        """
        device_id = device.device_id

        # Check alias uniqueness
        if alias and alias in self._aliases:
            raise ValueError(f"Alias '{alias}' is already registered")

        # Create registration record
        reg = RegisteredDevice(
            device=device,
            alias=alias or device_id[:8],
            tags=tags or [],
            priority=priority,
            auto_connect=auto_connect,
            twin_sync_enabled=twin_sync_enabled,
        )

        self._devices[device_id] = reg
        if alias:
            self._aliases[alias] = device_id

        logger.info(f"Registered device: {alias or device_id[:8]} ({device.model})")

        # Notify callbacks
        for callback in self._on_device_registered:
            try:
                callback(device_id, device)
            except Exception as e:
                logger.warning(f"Registration callback error: {e}")

        return device_id

    def unregister(self, device_id: str) -> bool:
        """Unregister a device.

        Args:
            device_id: Device ID or alias.

        Returns:
            True if device was found and removed.
        """
        # Resolve alias if needed
        device_id = self._resolve_id(device_id)

        if device_id not in self._devices:
            return False

        reg = self._devices.pop(device_id)

        # Remove alias mapping
        if reg.alias in self._aliases:
            del self._aliases[reg.alias]

        # Close device if connected
        try:
            reg.device.close()
        except (DeviceError, ConnectionError, RuntimeError) as e:
            logger.warning(f"Error closing device during unregister: {e}")

        logger.info(f"Unregistered device: {reg.alias}")

        # Notify callbacks
        for callback in self._on_device_unregistered:
            try:
                callback(device_id, reg.device)
            except Exception as e:
                logger.warning(f"Unregistration callback error: {e}")

        return True

    def _resolve_id(self, device_id_or_alias: str) -> str:
        """Resolve alias to device ID if needed."""
        if device_id_or_alias in self._aliases:
            return self._aliases[device_id_or_alias]
        return device_id_or_alias

    # ==================== Device Access ====================

    def get(self, device_id: str) -> Device | None:
        """Get device by ID or alias."""
        device_id = self._resolve_id(device_id)
        reg = self._devices.get(device_id)
        return reg.device if reg else None

    def __getitem__(self, device_id: str) -> Device:
        """Get device by ID or alias (dict-like access)."""
        device = self.get(device_id)
        if device is None:
            raise KeyError(f"Device '{device_id}' not found")
        return device

    def __contains__(self, device_id: str) -> bool:
        """Check if device is registered."""
        device_id = self._resolve_id(device_id)
        return device_id in self._devices

    def list_devices(self) -> list[str]:
        """List all registered device IDs."""
        return list(self._devices.keys())

    def list_aliases(self) -> list[str]:
        """List all registered aliases."""
        return list(self._aliases.keys())

    def get_registration_info(self, device_id: str) -> RegisteredDevice | None:
        """Get full registration information for a device."""
        device_id = self._resolve_id(device_id)
        return self._devices.get(device_id)

    # ==================== Discovery ====================

    def find_by_type(self, device_type: DeviceType) -> list[Device]:
        """Find all devices of a specific type."""
        return [
            reg.device
            for reg in self._devices.values()
            if reg.device.device_type == device_type
        ]

    def find_by_tag(self, tag: str) -> list[Device]:
        """Find all devices with a specific tag."""
        return [
            reg.device
            for reg in self._devices.values()
            if tag in reg.tags
        ]

    def find_by_manufacturer(self, manufacturer: str) -> list[Device]:
        """Find all devices from a specific manufacturer."""
        return [
            reg.device
            for reg in self._devices.values()
            if reg.device.manufacturer == manufacturer
        ]

    def find_by_model(self, model: str) -> list[Device]:
        """Find all devices with a specific model."""
        return [
            reg.device
            for reg in self._devices.values()
            if reg.device.model == model
        ]

    def find_by_state(self, state: DeviceState) -> list[Device]:
        """Find all devices in a specific state."""
        return [
            reg.device
            for reg in self._devices.values()
            if reg.device.state == state
        ]

    # ==================== Batch Operations ====================

    def connect_all(self, device_type: DeviceType | None = None) -> dict[str, bool]:
        """Connect all devices (optionally filtered by type).

        Returns:
            Dictionary of device_id -> success status.
        """
        results = {}
        for device_id, reg in self._devices.items():
            if device_type and reg.device.device_type != device_type:
                continue
            try:
                reg.device.open()
                results[device_id] = True
            except Exception as e:
                logger.error(f"Failed to connect {reg.alias}: {e}")
                results[device_id] = False
        return results

    def disconnect_all(self, device_type: DeviceType | None = None) -> dict[str, bool]:
        """Disconnect all devices."""
        results = {}
        for device_id, reg in self._devices.items():
            if device_type and reg.device.device_type != device_type:
                continue
            try:
                reg.device.close()
                results[device_id] = True
            except Exception as e:
                logger.error(f"Failed to disconnect {reg.alias}: {e}")
                results[device_id] = False
        return results

    def health_check_all(self) -> dict[str, tuple[bool, str]]:
        """Perform health check on all devices.

        Returns:
            Dictionary of device_id -> (is_healthy, message).
        """
        return {
            device_id: reg.device.health_check()
            for device_id, reg in self._devices.items()
        }

    def get_all_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all devices."""
        return {
            device_id: reg.device.get_status()
            for device_id, reg in self._devices.items()
        }

    # ==================== Digital Twin ====================

    def get_all_twin_states(self) -> dict[str, dict[str, Any]]:
        """Get twin states for all registered devices.

        Returns:
            Dictionary of device_id -> twin state.
        """
        states = {}
        for device_id, reg in self._devices.items():
            if reg.twin_sync_enabled:
                try:
                    states[device_id] = reg.device.get_twin_state()
                except Exception as e:
                    logger.warning(f"Failed to get twin state for {reg.alias}: {e}")
        return states

    def sync_from_twin_states(self, twin_states: dict[str, dict[str, Any]]) -> dict[str, bool]:
        """Synchronize devices from twin states.

        Args:
            twin_states: Dictionary of device_id -> twin state.

        Returns:
            Dictionary of device_id -> success status.
        """
        results = {}
        for device_id, state in twin_states.items():
            reg = self._devices.get(device_id)
            if not reg or not reg.twin_sync_enabled:
                results[device_id] = False
                continue

            try:
                reg.device.sync_from_twin(state)
                results[device_id] = True
            except Exception as e:
                logger.error(f"Failed to sync {reg.alias}: {e}")
                results[device_id] = False

        return results

    def get_twin_snapshot(self) -> dict[str, Any]:
        """Get complete snapshot for digital twin initialization.

        Returns:
            Dictionary containing all device states and metadata.
        """
        return {
            "registry_info": {
                "device_count": len(self._devices),
                "device_types": list(set(
                    reg.device.device_type.name
                    for reg in self._devices.values()
                )),
            },
            "devices": self.get_all_twin_states(),
            "aliases": {
                alias: device_id
                for alias, device_id in self._aliases.items()
            },
        }

    # ==================== Event Callbacks ====================

    def on_device_registered(self, callback: Callable[[str, Device], None]) -> None:
        """Register callback for device registration events."""
        self._on_device_registered.append(callback)

    def on_device_unregistered(self, callback: Callable[[str, Device], None]) -> None:
        """Register callback for device unregistration events."""
        self._on_device_unregistered.append(callback)

    # Note: on_state_change callback is reserved for future use
    # when automatic state change notifications are implemented.

    # ==================== Import/Export ====================

    def export_config(self) -> dict[str, Any]:
        """Export registry configuration.

        Returns:
            Configuration dictionary for persistence.
        """
        return {
            "devices": [
                {
                    "device_id": device_id,
                    "alias": reg.alias,
                    "tags": reg.tags,
                    "priority": reg.priority,
                    "auto_connect": reg.auto_connect,
                    "twin_sync_enabled": reg.twin_sync_enabled,
                    "device_type": reg.device.device_type.name,
                    "manufacturer": reg.device.manufacturer,
                    "model": reg.device.model,
                    "parameters": {
                        name: param.value
                        for name, param in reg.device.get_all_parameters().items()
                    },
                }
                for device_id, reg in self._devices.items()
            ]
        }

    def __len__(self) -> int:
        """Number of registered devices."""
        return len(self._devices)

    def __iter__(self):
        """Iterate over registered devices."""
        return iter(self._devices.values())

    def __repr__(self) -> str:
        return f"DeviceRegistry(devices={len(self._devices)}, aliases={len(self._aliases)})"


# Global registry instance for convenience
_global_registry: DeviceRegistry | None = None


def get_global_registry() -> DeviceRegistry:
    """Get or create the global device registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = DeviceRegistry()
    return _global_registry
