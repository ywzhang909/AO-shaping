# AGENTS.md - Agent Coding Guidelines for AO-shaping

## Project Overview

AO-shaping is a Python project for Adaptive Optics shaping, controlling hardware like SLMs (Spatial Light Modulators), DMs (Deformable Mirrors), WFSs (Wavefront Sensors), and CCDs. It uses PyTorch for deep learning, numpy for computation, and various hardware SDKs.

## Build, Test, and Development Commands

### Running Tests

Run all tests:
```bash
pytest
```

Run a single test file:
```bash
pytest tests/ao_shaping/drivers/test_slm.py
```

Run a single test function:
```bash
pytest tests/ao_shaping/drivers/test_slm.py::TestSLMInitialization::test_valid_slm_number
```

Run tests with verbose output:
```bash
pytest -v
```

Run tests with coverage:
```bash
pytest --cov=ao_shaping --cov-report=html
```

Run tests matching a pattern:
```bash
pytest -k "test_slm"
```

### Development Tools

The project uses:
- **pytest** for testing
- **uv** for dependency management (pyproject.toml based)
- **loguru** for logging

### Installing Dependencies

```bash
uv sync
```

### Type Checking (if needed)

```bash
mypy src/ao_shaping
```

### Linting (if ruff is added)

```bash
ruff check src/ao_shaping
```

## Code Style Guidelines

### Imports

- Standard library imports first
- Third-party imports second
- Local imports third
- Use explicit relative imports within the package:
  ```python
  from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200
  ```

### Naming Conventions

- **Classes**: PascalCase (e.g., `SantecSLM200`, `SerialPortFSM`)
- **Functions/variables**: snake_case (e.g., `send()`, `is_open`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_GRAYSCALE_VALUE`)
- **Private methods**: prefix with underscore (e.g., `_ensure_open()`)

### Type Hints

- Use type hints for function parameters and return types:
  ```python
  def send(self, x: float, y: float) -> bytes:
  ```
- Use `Optional[T]` for nullable types:
  ```python
  def get_rx(self) -> Optional[bytes]:
  ```
- Use `Union` or `|` syntax for multiple types:
  ```python
  video_mode: int | VideoMode = 0
  ```

### Error Handling

- Create custom exception classes for driver-specific errors:
  ```python
  class SantecSLM200Error(Exception):
      """Santec SLM-200 驱动错误"""
      pass
  ```
- Use descriptive error messages in Chinese or English:
  ```python
  raise SantecSLM200Error(f"SLM编号必须在1-8之间，当前: {slm_number}")
  ```
- Validate inputs early with clear assertions:
  ```python
  assert 450 <= wavelength <= 1600, f"{wavelength=} not in range(450, 1600)"
  ```

### Logging

- Use `loguru.logger` for logging:
  ```python
  from loguru import logger
  logger.info(f"成功打开SLM #{self.slm_number}")
  logger.debug(f"SLM #{self.slm_number} 状态正常")
  logger.warning(f"SLM #{self.slm_number} 已经处于打开状态")
  ```

### Context Managers

- Implement `__enter__` and `__exit__` for resource management:
  ```python
  def __enter__(self):
      self.open()
      return self
  
  def __exit__(self, exc_type, exc_val, exc_tb):
      self.close()
  ```

### Docstrings

- Use English docstrings for public APIs (see `santec_slm200.py` for reference)
- Include Examples in docstrings:
  ```python
  """
  Santec SLM-200 空间光调制器驱动类
  
  Example:
      >>> with SantecSLM200(slm_number=1) as slm:
      ...     slm.set_wavelength(1064, 200)
      ...     phase_data = np.zeros((1080, 1920), dtype=np.uint16)
      ...     slm.write_phase(phase_data, memory_number=1)
  """
  ```

### File Organization

- Source code in `src/ao_shaping/`
- Tests in `tests/ao_shaping/`
- Follow the module structure:
  - `drivers/` - Hardware drivers (slm/, dm/, wfs/, ccd/, tm/)
  - `algorithm/` - Algorithms (adam.py)
  - `wf/` - Wavefront control
  - `wfless/` - Wavefrontless methods
  - `utils/` - Utility functions
  - `display/` - GUI/display utilities

### Hardware-Specific Guidelines

- Driver classes should have:
  - `open()` method to connect to device
  - `close()` method to disconnect
  - Context manager support
  - Proper state tracking (`self.is_open`)
- Use abstract base classes for common interfaces (see `drivers/dm/base.py`)
- Validate hardware parameters before SDK calls
- Handle SDK import failures gracefully

### Testing Guidelines

- Test files in `tests/ao_shaping/` mirror source structure
- Use pytest fixtures for setup/teardown
- Test both success and failure cases
- Use descriptive test names: `test_<method>_<scenario>`
- Group tests in classes by functionality

### NumPy Conventions

- Use `np.ndarray` for array types
- Specify dtype explicitly:
  ```python
  phase = np.zeros((1080, 1920), dtype=np.uint16)
  ```
- Validate array shape and dtype before hardware operations

### Ctypes for Hardware SDKs

- Use ctypes for C/C++ DLL bindings
- Create proper type conversions:
  ```python
  dat = (ctypes.c_ushort * (width * height))()
  ctypes.memmove(dat, phase.ctypes.data, phase.nbytes)
  ```

### General Python Practices

- Use f-strings for formatting
- Avoid bare `except:` - catch specific exceptions
- Use early returns to reduce nesting
- Keep functions focused and reasonably sized
