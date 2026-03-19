# AGENTS.md - AO-Shaping Development Guide

## Project Overview

AO-Shaping is an Adaptive Optics (AO) system using reinforcement learning for wavefront correction and beam shaping. It integrates multiple optimization algorithms including WFS-based and wavefront-sensorless methods.

## Project Structure

```
src/ao_shaping/
├── main.py              # CLI entry point
├── algorithm/           # Optimization algorithms (Adam, SGD, etc.)
├── drivers/              # Hardware drivers (SLM, DM, WFS, CCD)
│   ├── slm/             # Spatial Light Modulator
│   ├── dm/              # Deformable Mirror
│   ├── wfs/             # Wavefront Sensor
│   └── ccd/             # Camera
├── optimizer/           # High-level optimizers
│   ├── wf/              # Wavefront-based
│   ├── wfless/          # Wavefront-sensorless
│   └── rl/              # Reinforcement learning
├── utils/               # Utilities (spots_calc, wavefront_calc, etc.)
├── display/             # Visualization
└── sim/                 # Simulation components
```

---

## Build/Lint/Test Commands

### Running Tests

```bash
# Run all tests
pytest

# Run all tests with verbose output
pytest -v

# Run tests with coverage
pytest --cov=ao_shaping

# Run a specific test file
pytest tests/ao_shaping/utils/test_spots_calc.py

# Run a specific test class
pytest tests/ao_shaping/utils/test_spots_calc.py::TestCentroid

# Run a specific test function
pytest tests/ao_shaping/utils/test_spots_calc.py::TestCentroid::test_centroid_uniform

# Run tests matching a pattern
pytest -k "spots_calc"
pytest -k "test_centroid"

# Run with live output (no capture)
pytest -s

# Run with specific markers
pytest -m "not slow"
```

### Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -e .
```

---

## Code Style Guidelines

### Python Version
- **Python 3.12+ required** (see `pyproject.toml`)

### Imports

**Ordering (PEP 8 standard library ordering):**
```python
from __future__ import annotations  # Future imports first

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, ClassVar

import numpy as np

from loguru import logger
```

**Avoid relative imports in package code:**
```python
# Good
from ao_shaping.drivers import CameraStreamManager
from ao_shaping.utils.spots_calc import centroid

# Avoid (unless necessary)
from .drivers import ...
```

### Type Hints

**Use modern type hints with `|` syntax (Python 3.12+):**
```python
def set_parameter_value(self, name: str, value: Any) -> bool:
    min_value: float | None = None
    error_message: str | None = None
```

**Return type hints on all public methods:**
```python
def get_parameter_value(self, name: str) -> Any:
    pass

def is_connected(self) -> bool:
    pass
```

**Generic types:**
```python
from typing import TypeVar

T = TypeVar('T')

def get_item(self, key: str) -> DeviceParameter | None:
    return self._parameters.get(key)
```

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Classes | PascalCase | `DeviceBase`, `NLightDM` |
| Functions | snake_case | `calculate_sharpness`, `get_centroid` |
| Variables | snake_case | `exposure_time_ms`, `dm_unit_mask` |
| Constants | SCREAMING_SNAKE | `MAX_VOLTAGE`, `DEFAULT_THRESHOLD` |
| Private attrs | _leading_underscore | `_device_id`, `_parameters` |
| Type vars | PascalCase | `T`, `T_co` |

### Data Classes

Use `@dataclass` for structured data containers:
```python
@dataclass
class DeviceParameter:
    name: str
    value: Any
    value_type: type = float
    min_value: float | None = None
    max_value: float | None = None
    unit: str = ""
    description: str = ""
    writable: bool = True
```

### Enums

Use `Enum` with `auto()` for state/type definitions:
```python
class DeviceState(Enum):
    UNKNOWN = auto()
    DISCONNECTED = auto()
    CONNECTING = auto()
    READY = auto()
    BUSY = auto()
    ERROR = auto()
```

### Error Handling

**Custom exceptions with `*Error` suffix:**
```python
class DeviceError(Exception):
    pass

class DeviceNotFoundError(DeviceError):
    pass

class DeviceBusyError(DeviceError):
    pass
```

**Context manager support for resources:**
```python
class BaseDM(ABC):
    @abstractmethod
    def open(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    def __enter__(self) -> "BaseDM":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
```

**Usage:**
```python
with NlightDM() as dm:
    dm.send_voltages(vs, 0.1)
# Automatically closed
```

**Graceful exception handling:**
```python
try:
    import cupy as cp
    CUPY_AVAILABLE = cp.cuda.is_available()
except (ImportError, AttributeError):
    CUPY_AVAILABLE = False
```

### Docstrings

Use docstrings for public APIs (Google style):
```python
def validate(self, value: Any) -> bool:
    """Validate if value is within allowed range.
    
    Args:
        value: The value to validate.
        
    Returns:
        True if valid, False otherwise.
    """
    pass
```

### Logging

Use `loguru.logger` (configured in pyproject.toml):
```python
from loguru import logger

logger.debug(f"Device {self._device_id} initialized")
logger.info("Starting optimization")
logger.warning(f"Invalid value {value} for parameter '{name}'")
logger.error(f"Device {self._device_id} error: {error_msg}")
```

### Hardware Drivers

**Required interface for all drivers:**
```python
class Device(ABC):
    @abstractmethod
    def open(self) -> None:
        """Open connection to the device."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close connection and release resources."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if device is connected and ready."""
        pass
```

**State tracking:**
- Use `DeviceState` enum for state management
- Use `_set_state(state, error_msg)` helper method
- Track `self.is_open` or similar for connection state

### Performance-Critical Code

**Numba JIT compilation:**
```python
@numba.njit(cache=True)
def calculate_sharpness_numba(img: np.ndarray):
    # JIT-compiled code here
    pass
```

**NumPy as default, provide alternatives:**
```python
def calculate_sharpness(img: np.ndarray):
    # NumPy version (default)
    pass

def calculate_sharpness_numba(img: np.ndarray):
    # Numba-accelerated version
    pass

def calculate_sharpness_cupy(img: cp.ndarray):
    # CuPy GPU version
    pass
```

---

## Environment Variables

Configuration via `.env` file:
```
Far_Cam_ID=0
Near_Cam_ID=1
IDEAL_SPOT_RADIUS=7
CENTER=577,655
```

Access in code:
```python
import os
cam_id = int(os.environ.get('Far_Cam_ID', 0))
```

---

## Configuration

Pytest configuration in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
```

VS Code settings in `.vscode/settings.json` set PYTHONPATH to `src` and `libs` directories.
