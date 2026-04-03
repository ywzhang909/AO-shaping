# drivers/ - Hardware Drivers

Hardware SDK wrappers for SLM, DM, WFS, CCD, TM devices.

## STRUCTURE

```
drivers/
├── device_base.py       # Device base class (Device, DeviceState, DeviceType)
├── device_registry.py    # Device registration and management
├── visa_base.py         # VISA communication layer
├── mock_devices.py      # Mock devices for testing
├── ccd/                 # Cameras
│   ├── base.py          # BaseCamera abstract class
│   ├── daheng.py        # Daheng GigE camera
│   ├── miicam.py        # MiiCam SDK
│   └── miicam_device.py # MIICAMDevice
├── dm/                  # Deformable Mirrors
│   ├── base.py          # DM abstract class
│   ├── NLight.py        # NLight DM driver
│   └── simulateDM.py    # Simulation DM
├── slm/                 # Spatial Light Modulators
│   ├── santec_slm200.py # Santec SLM-200 SDK
│   ├── santec_slm200_visa.py
│   ├── slm_calibration.py
│   └── slm_pattern_helper.py
├── wfs/                 # Wavefront Sensors
│   └── thorlab_wfs.py  # Thorlabs WFS
├── tm/                  # Timing Modules
│   └── serial_port_fsm.py
└── sim/                 # Simulation (digital twin)
    ├── base.py          # SimulatedDevice, OpticalDevice
    ├── ccd/             # SimulatedCCD
    ├── laser/           # SimulatedLaser
    ├── optics/          # SimulatedSLM, SimulatedLens
    └── atmos/           # Turbulence screens
```

## KEY CLASSES

| Class | File | Purpose |
|-------|------|---------|
| `Device` | device_base.py | Base class for all devices |
| `DeviceState` | device_base.py | State enum (DISCONNECTED, READY, BUSY, etc.) |
| `DeviceType` | device_base.py | Type enum (CAMERA, SLM, DM, WFS, etc.) |
| `NLightDM` | dm/NLight.py | DM control |
| `BaseDM` | dm/base.py | DM abstract base |
| `ThorlabWFS` | wfs/thorlab_wfs.py | WFS control |

## REQUIRED INTERFACE

All drivers must implement:
```python
class Device(ABC):
    @abstractmethod
    def open(self) -> None: ...
    @abstractmethod
    def close(self) -> None: ...
    @abstractmethod
    def is_connected(self) -> bool: ...
    @abstractmethod
    def get_hardware_info(self) -> dict: ...
    
    # Context manager support
    def __enter__(self) -> "Device": ...
    def __exit__(self, ...): ...
```

## CONVENTIONS

- Custom exceptions: `*Error` suffix (e.g., `DeviceError`, `CameraError`)
- State tracking via `_set_state(state, error_msg)` 
- SDK imports in `__init__`, handle failure gracefully with try/except
- Use `loguru.logger` for logging
- Parameter registration via `register_parameter()` method
