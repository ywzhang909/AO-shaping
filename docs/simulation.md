# AO Simulation Guide

## Overview

This simulation system provides wave optics simulation for Adaptive Optics (AO) wavefront correction, using `sim.digitaltwin` as the physics engine wrapped by the `ao_shaping.drivers.sim` device layer.

## Architecture

```
ao_shaping/
├── drivers/sim/                    # Device driver interface
│   ├── base.py                    # SimulatedDevice, OpticalDevice, WavefrontProcessor
│   ├── wave.py                    # Physics utilities + WaveGenerator, WavePropagator
│   ├── atmos/screens.py            # SimulatedTurbulentScreen, SimulatedThermalScreen
│   ├── optics/simulated_slm.py   # SimulatedSLM, SimulatedLens, SimulatedAperture
│   ├── ccd/simulated_ccd.py       # SimulatedCCD
│   ├── laser/simulated_laser.py   # SimulatedLaser
│   └── compat.py                  # Legacy AOConfig, TraditionalAOSystem (for old optimizers)
│
├── optimize_pib_reference.py       # Main optimizer script (uses drivers/sim)
│
└── sim/digitaltwin/              # Core physics (DO NOT MODIFY)
    ├── base.py                    # Wave, Environment classes
    ├── screens.py                 # TurbulentScreen, ThermalScreen
    ├── utilities.py               # wave_angle_spectrum propagation
    └── params.py                 # WaveIndex (PIB, radius, centroid)
```

## Quick Start

### Basic Wave Simulation

```python
from ao_shaping.drivers.sim import (
    SimulatedTurbulentScreen,
    create_wave,
    apply_aperture,
    apply_focus,
    propagate,
    power_bucket,
    radius_metric,
    Environment,
)

# Create wave
wave = create_wave(npix=256, dpix=0.1e-3, wavelength=1550e-9)

# Apply optical elements
apply_aperture(wave, radius=0.05)        # 50mm aperture
apply_focus(wave, focal_length=0.5)       # 0.5m focal length

# Add turbulence
turb = SimulatedTurbulentScreen(Cn2=1e-9, L0=10.0, l0=0.01)
turb.process(wave)

# Propagate to focal plane
propagate(wave, distance=0.5)

# Compute metrics
pib = power_bucket(wave.intensity, wave.x, wave.y, 'origin', r_bucket=5e-3)
r80 = radius_metric(wave.intensity, wave.x, wave.y, 'origin', energy=0.8)
```

### AO System with DM Control

See `optimize_pib_reference.py` for a complete example with Zernike-based DM control and SPGD/PSO/GA/SA optimization.

## Physics Utilities

All utilities wrap `sim.digitaltwin` functions:

| Function | Description |
|----------|-------------|
| `create_wave(npix, dpix, wavelength)` | Create a Wave object with plane wave |
| `apply_aperture(wave, radius)` | Apply circular pupil mask |
| `apply_focus(wave, focal_length)` | Apply thin lens phase (`-πr²/(λf)`) |
| `propagate(wave, distance)` | Angular spectrum propagation |
| `power_bucket(intensity, x, y, center, r_bucket)` | Power-in-bucket metric |
| `radius_metric(intensity, x, y, center, energy)` | Radius containing energy fraction |
| `SimulatedTurbulentScreen.process(wave)` | Apply Kolmogorov turbulence |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `npix` | 256 | Grid size |
| `aperture` | 0.005 m | Physical aperture diameter |
| `wavelength` | 1550e-9 m | Wavelength |
| `focal_length` | 0.5 m | Lens focal length |
| `Cn2` | 1e-9 | Refractive index structure constant |
| `L0` | 10.0 m | Outer scale |
| `l0` | 0.01 m | Inner scale |
| `harmonic` | 0 | Subharmonic order (0 = no subharmonics, fixes ~100° phase artifact) |

## Simulation Results

With turbulence (Cn2=1e-9), 8x8 DM, and SPGD optimization:

![PIB Convergence](pib_convergence.png)

![Spot Comparison](pib_spots.png)

| Algorithm | Final PIB Ratio | Converged to 80% |
|-----------|----------------|-------------------|
| SPGD (Zernike, 11 modes) | 85.95% | YES |
| SPGD-V (voltage-based) | 81.24% | YES |
| GA | 82.26% | YES |
| PSO | 68.46% | NO |
| SA | 77.77% | NO |

## Device Interface vs Raw Physics

The `drivers/sim/` classes provide a Device-compatible interface (with `open()`, `close()`, `is_connected()`). However, for high-performance optimization, use the **physics utilities directly** — they have no state management overhead:

```python
# Device interface (has state management)
turb = SimulatedTurbulentScreen(Cn2=1e-9)
turb.open()
turb.process(wave)     # requires open()
turb.close()

# Physics utilities (direct, no overhead)
from ao_shaping.drivers.sim import SimulatedTurbulentScreen
turb = SimulatedTurbulentScreen(Cn2=1e-9)
turb.process(wave)     # works without open()
```

## Modifying the Simulation

### Adding a New Optical Element

Add to `drivers/sim/wave.py`:

```python
def apply_mirror(wave: Any, reflectivity: float = 1.0) -> None:
    """Apply reflective mirror.
    
    Args:
        wave: Wave object.
        reflectivity: Mirror reflectivity.
    """
    wave.wavefront = wave.wavefront * reflectivity
```

### Adding a New Device Class

Add to `drivers/sim/wave.py` (or a new file):

```python
from ao_shaping.drivers.sim.base import WavefrontProcessor

class MyDevice(WavefrontProcessor):
    device_type = DeviceType.OTHER
    manufacturer = "Simulation"
    model = "My Device"
    
    def process(self, wave: Any) -> Any:
        # Process wavefront
        return wave
```

Then export in `drivers/sim/__init__.py`:

```python
from ao_shaping.drivers.sim.wave import MyDevice
__all__ = [..., "MyDevice"]
```

### Adding a New Turbulence Model

Add to `drivers/sim/atmos/screens.py`:

```python
class SimulatedCustomScreen(WavefrontProcessor):
    def __init__(self, ...):
        ...
    
    def process(self, wave: Any) -> Any:
        # Custom turbulence logic
        return wave
```

### Running the Optimizer

```bash
cd /data/llm_models/AO-shaping
PYTHONPATH=src .venv/bin/python optimize_pib_reference.py
```

Plots are saved to the current directory. Modify `plot_convergence()` and `plot_spots()` in `optimize_pib_reference.py` to change save paths.

## Turbulence: Subharmonic Fix

The `TurbulentScreen` in `sim.digitaltwin` has a bug with subharmonics (`harmonic=1`) that produces ~100° phase artifacts even at Cn2=1e-14. **Always use `harmonic=0`**:

```python
# WRONG - produces phase artifacts
turb = SimulatedTurbulentScreen(Cn2=1e-9, harmonic=1)

# CORRECT
turb = SimulatedTurbulentScreen(Cn2=1e-9, harmonic=0)
```

## Legacy Compatibility

The `drivers/sim/compat.py` module provides `AOConfig` and `TraditionalAOSystem` interfaces for legacy optimizer scripts:

```python
from ao_shaping.drivers.sim.compat import AOConfig, TraditionalAOSystem

config = AOConfig(N=64, L=0.1, Cn2=1e-9, dm_actuators=8)
ao = TraditionalAOSystem(config)
img = ao.get_image()
ao.set_dm_voltages(voltages)
```

This is used by `sim_spgd.py` and `envs.py` (RL environments).
