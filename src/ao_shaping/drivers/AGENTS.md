# drivers/ - Hardware Drivers

Hardware SDK wrappers for SLM, DM, WFS, CCD, TM devices.

## STRUCTURE

```
drivers/
├── slm/           # Spatial Light Modulator
│   ├── _slm_win.py        # Santec SDK bindings (ctypes)
│   ├── santec_slm200.py    # Main driver class
│   ├── santec_slm200_visa.py
│   ├── slm_calibration.py  # Phase-gamma calibration
│   └── slm_pattern_helper.py
├── dm/             # Deformable Mirror
│   ├── base.py             # Abstract base class
│   ├── NLight.py           # NLight DM driver
│   └── simulateDM.py       # Simulation driver
├── wfs/            # Wavefront Sensor
│   └── thorlab_wfs.py      # Thorlabs WFS
├── ccd/            # Camera
│   └── daheng.py           # Daheng camera
├── tm/             # Timing Module
│   └── serial_port_fsm.py   # Serial FSM
└── visa_base.py    # VISA base class
```

## KEY CLASSES

| Class | File | Purpose |
|-------|------|---------|
| `SantecSLM200` | santec_slm200.py | SLM control |
| `NLightDM` | dm/NLight.py | DM control |
| `BaseDM` | dm/base.py | DM abstract base |

## CONVENTIONS

- Driver must implement: `open()`, `close()`, `__enter__`, `__exit__`
- Custom exceptions: `*Error` suffix
- State tracking: `self.is_open`
- SDK import in `__init__`, handle failure gracefully
