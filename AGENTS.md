# AGENTS.md - AO-Shaping Development Guide

**Generated:** 2026-04-25

## Project Overview

AO-Shaping is an Adaptive Optics (AO) system using reinforcement learning for wavefront correction and beam shaping. It integrates multiple optimization algorithms including WFS-based and wavefront-sensorless methods.

## Project Structure

```
AO-shaping/
├── src/
│   ├── ao_shaping/          # Main package
│   │   ├── main.py              # CLI entry point (Click-based)
│   │   ├── wf_runner.py         # Wavefront RMS optimizer
│   │   ├── axis_beam_runner.py  # PIB optimizer
│   │   ├── pipeline_runner.py    # Serial WF→PIB pipeline
│   │   ├── algorithm/            # Optimization algorithms (Adam, SGD, etc.)
│   │   ├── drivers/              # Hardware drivers (see drivers/AGENTS.md)
│   │   │   ├── ccd/              # Cameras (Daheng, MiiCam)
│   │   │   ├── dm/               # Deformable Mirrors (NLight)
│   │   │   ├── slm/              # Spatial Light Modulators (Santec)
│   │   │   ├── wfs/              # Wavefront Sensors (Thorlabs)
│   │   │   ├── tm/               # Timing modules (Serial/FSM)
│   │   │   ├── sim/              # Digital twin simulation
│   │   │   └── mock_devices.py   # Mock devices for testing
│   │   ├── optimizer/            # High-level optimizers
│   │   │   ├── wf/               # Wavefront-based (RMS)
│   │   │   ├── wfless/           # Wavefront-sensorless (PIB)
│   │   │   └── rl/               # Reinforcement learning (SAC)
│   │   ├── utils/                # Utilities (spots_calc, wavefront_calc, zernike_calc)
│   │   ├── ml/                  # Machine learning (U-Net+GAN, training, models)
│   │   ├── tools/                # Standalone tools (SLM phase capture, data collection)
│   │   ├── display/              # Visualization (Windows, frames for GUI)
│   │   └── gui/                  # GUI components (Streamlit)
│   ├── calculators/               # Cython extensions (standalone)
│   └── optical_ui/                # [DEPRECATED] Empty package
├── tests/ao_shaping/              # Tests (mirrors src structure)
├── libs/                          # Third-party SDK binaries (gxipy, Drv_UDPST)
└── scripts/                       # Utility scripts
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Hardware drivers | `src/ao_shaping/drivers/` | See drivers/AGENTS.md |
| Optimization algorithms | `src/ao_shaping/algorithm/` | Adam, SGD, Muon, Tabu search, etc. |
| Wavefront optimizers | `src/ao_shaping/optimizer/wf/` | RMS optimization |
| Zernike response matrix | `src/ao_shaping/optimizer/wf/zernike_response_matrix.py` | SLM→WFS Zernike校准 |
| PIB optimizers | `src/ao_shaping/optimizer/wfless/` | Power-in-bucket |
| RL training | `src/ao_shaping/optimizer/rl/` | SAC, LR-WFS |
| Simulation | `src/ao_shaping/drivers/sim/` | Digital twin devices |
| Utilities | `src/ao_shaping/utils/` | spots_calc, wavefront_calc, zernike_calc, display |
| ML training | `src/ao_shaping/ml/` | U-Net+GAN, trainer, wandb_logger |
| Standalone tools | `src/ao_shaping/tools/` | SLM phase capture, train data collection |
| Visualization | `src/ao_shaping/display/` | Windows, frames for GUI |
| GUI | `src/ao_shaping/gui/` | Streamlit components |
| Tests | `tests/ao_shaping/` | Mirror of src structure |

---

## optimizer/ Module

High-level optimizers for wavefront correction and beam shaping:

| Submodule | File | Purpose |
|----------|------|---------|
| wf/ | `rms.py` | Wavefront sensor-based RMS optimization |
| wf/ | `interaction_matrix.py` | DM-WFS interaction matrix |
| wf/ | `zernike_response_matrix.py` | Zernike calibration |
| wfless/ | `pib.py` | Power-in-bucket optimization |
| wfless/ | `sim_spgd.py` | Simulated SPGD |
| rl/ | `sac_train.py` | SAC reinforcement learning |
| rl/ | `lr_wfs.py` | Learning-based wavefront sensing |

---

## utils/ Module

Utility functions for image processing and calculations:

| File | Purpose |
|------|---------|
| `spots_calc.py` | Centroid calculation, sharpness metrics |
| `wavefront_calc.py` | Wavefront reconstruction from spots |
| `zernike_calc.py` | Zernike polynomial generation |
| `matrix_utils.py` | Matrix operations |
| `display.py` | Visualization utilities |
| `pattern_helper.py` | SLM pattern generation |
| `file.py` | File I/O utilities |

---

## Entry Points

**CLI Commands (Click-based):**
```bash
# Via main.py hub
python src/ao_shaping/main.py [COMMAND]

# Direct runners (standalone)
python src/ao_shaping/wf_runner.py
python src/ao_shaping/axis_beam_runner.py
python src/ao_shaping/pipeline_runner.py
```

**CLI Structure:**
```
main (click.group)
├── wf             → wf_runner.run()              [Wavefront RMS optimization]
├── pib            → axis_beam_run()              [Power-in-Bucket optimization]
├── pipeline       → pipeline_run()               [Serial WF→PIB pipeline]
└── zernike-matrix → zernike_matrix_run()         [Zernike响应矩阵校准]
```

**Note:** `combined_runner.py` exists but is NOT wired to main CLI (documented in README only).

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
# Create virtual environment with uv
uv venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
uv add -e .
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

---

## ANTI-PATTERNS (THIS PROJECT)

| Pattern | Forbidden Because |
|---------|------------------|
| Relative imports in package | Use `from ao_shaping.xxx import yyy` instead of `from .xxx import yyy` |
| `as any`, `@ts-ignore` | Never suppress type errors |
| Empty catch blocks | Always handle exceptions or log |
| Deleting failing tests | Fix the code, not the test |
| `combined_runner.py` with main CLI | Use `pipeline_runner.py` instead |

---

## UNIQUE STYLES

- **Mock-first testing**: Tests use simulation classes (`SimTurbulenceAOEnv`, `sim_spgd`) to avoid hardware
- **Hardware skip pattern**: Tests requiring physical hardware use `pytest.skip("Requires DM hardware")`
- **Recorder pattern**: Optimization tests validate history dictionaries with expected fields
- **Optional backend testing**: CuPy/Numba tested conditionally with try/except guards
- **No fixtures**: No `conftest.py`, fixtures defined inline in test methods

---

## MISSING INFRASTRUCTURE

- **No CI/CD**: No GitHub Actions, no automated testing on push
- **No linting**: No ruff/mypy/flake8 configured
- **No pre-commit**: No hooks for lint/format before commits
- **No requirements.txt**: Only `pyproject.toml` and `uv.lock`

Consider adding: `.github/workflows/ci.yml`, `ruff.toml`, `.pre-commit-config.yaml`
