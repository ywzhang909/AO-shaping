"""Device registration for AO RL training (virtual + physical devices)."""

from __future__ import annotations

from dataclasses import dataclass
@dataclass(frozen=True)
class DeviceSpec:
    """Registered device metadata."""

    name: str
    device_type: str
    implementation: str
    is_virtual: bool
    description: str


class DeviceRegistry:
    """In-memory registry with grouped device selection support."""

    def __init__(self) -> None:
        self._specs: dict[str, dict[str, DeviceSpec]] = {
            "dm": {},
            "ccd": {},
            "wfs": {},
        }

    def register(self, spec: DeviceSpec) -> None:
        if spec.device_type not in self._specs:
            msg = f"Unsupported device_type={spec.device_type}"
            raise ValueError(msg)
        self._specs[spec.device_type][spec.name] = spec

    def names(self, device_type: str) -> list[str]:
        return sorted(self._specs[device_type].keys())

    def get(self, device_type: str, name: str) -> DeviceSpec:
        try:
            return self._specs[device_type][name]
        except KeyError as exc:
            msg = f"Device '{name}' is not registered for type '{device_type}'."
            raise ValueError(msg) from exc


def build_default_registry() -> DeviceRegistry:
    registry = DeviceRegistry()

    # virtual devices
    registry.register(
        DeviceSpec(
            name="sim_dm",
            device_type="dm",
            implementation="TraditionalAOSystem.dm",
            is_virtual=True,
            description="Virtual deformable mirror based on simulation influence function.",
        )
    )
    registry.register(
        DeviceSpec(
            name="sim_ccd",
            device_type="ccd",
            implementation="TraditionalAOSystem.get_image",
            is_virtual=True,
            description="Virtual CCD image from simulated focal-plane intensity.",
        )
    )
    registry.register(
        DeviceSpec(
            name="sim_wfs",
            device_type="wfs",
            implementation="TraditionalAOSystem.measure_wavefront",
            is_virtual=True,
            description="Virtual Shack-Hartmann slopes from simulation.",
        )
    )

    # physical devices (registered and selectable from click)
    registry.register(
        DeviceSpec(
            name="nlight_dm",
            device_type="dm",
            implementation="ao_shaping.drivers.NlightDM",
            is_virtual=False,
            description="Physical NLight deformable mirror.",
        )
    )
    registry.register(
        DeviceSpec(
            name="miicam_ccd",
            device_type="ccd",
            implementation="ao_shaping.drivers.CameraStreamManager",
            is_virtual=False,
            description="Physical MiiCam CCD camera stream.",
        )
    )
    registry.register(
        DeviceSpec(
            name="thorlabs_wfs",
            device_type="wfs",
            implementation="ao_shaping.drivers.wfs.thorlab_wfs.WFS20",
            is_virtual=False,
            description="Physical Thorlabs wavefront sensor.",
        )
    )

    return registry
