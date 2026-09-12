# AGENTS.md - AO-Shaping Development Guide

**Generated:** 2026-06-20

## Project Overview

AO-Shaping is an Adaptive Optics (AO) system using reinforcement learning for wavefront correction and beam shaping. It integrates multiple optimization algorithms including WFS-based and wavefront-sensorless methods.

## Project Structure

```
AO-shaping/
├── src/
│   ├── ao_shaping/          # Main package
│   │   ├── main.py              # CLI entry point (Click-based)
│   │   ├── runners/             # Runner scripts package
│   │   │   ├── __init__.py     # Re-exports for backward compatibility
│   │   │   ├── wf_runner.py         # Wavefront RMS optimizer
│   │   │   ├── axis_beam_runner.py  # PIB optimizer
│   │   │   ├── pipeline_runner.py   # Serial WF→PIB pipeline
│   │   │   ├── zernike_matrix_runner.py  # Zernike response matrix
│   │   │   └── slm_square_runner.py  # SLM方形光斑 SPGD 整形 (spgd-square)
│   │   ├── algorithm/            # Optimization algorithms (Adam, SGD, etc.)
│   │   ├── drivers/              # Hardware drivers (see drivers/AGENTS.md)
│   │   │   ├── ccd/              # Cameras (Daheng, MiiCam)
│   │   │   ├── dm/               # Deformable Mirrors (NLight)
│   │   │   ├── slm/              # Spatial Light Modulators (Santec, WavefrontCorrection)
│   │   │   ├── wfs/              # Wavefront Sensors (Thorlabs)
│   │   │   ├── tm/               # Timing modules (Serial/FSM)
│   │   │   ├── sim/              # Digital twin simulation
│   │   │   └── mock_devices.py   # Mock devices for testing
│   │   ├── optimizer/            # High-level optimizers
│   │   │   ├── wf/               # Wavefront-based (RMS)
│   │   │   ├── wfless/           # Wavefront-sensorless (PIB)
│   │   │   │   └── slm_square_shaping.py  # SLM方形光斑 SPGD 整形优化器
│   │   │   └── rl/               # Reinforcement learning (SAC)
│   │   ├── utils/                # Utilities (spots_calc, wavefront_calc, zernike_calc, zernike_utils, wfs_utils)
│   │   ├── ml/                  # Machine learning (U-Net+GAN, training, models) — NOTE: lives at src/ml/ as a separate standalone package
│   │   │   ├── trainer/         # Training utilities
│   │   │   ├── models/          # Neural network models
│   │   │   └── wandb_logger.py  # WandB integration
│   │   ├── tools/                # Standalone tools (SLM phase capture, Micro-DM image collection, data collection)
│   │   ├── display/              # Visualization (Windows, frames for GUI)
│   │   └── gui/                  # GUI components (Streamlit), 按设备域分包:
│   │       ├── r50/              #   R50Power 控制器 UI (r50_controller_ui.py 入口)
│   │       ├── dm/               #   变形镜 UI (micro_dm_ui.py, ceramic_viewer.py)
│   │       ├── slm/              #   SLM UI (multi_slm_controller.py, slm_calibration_ui.py)
│   │       ├── zernike/          #   Zernike UI (response matrix 校准/调试查看)
│   │       └── ccd/              #   CCD 图像分析 UI (ccd_analyzer.py)
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
| SLM方形光斑整形 (SPGD) | `src/ao_shaping/optimizer/wfless/slm_square_shaping.py` + `runners/slm_square_runner.py` | SPGD 优化 Zernike 系数 → 均匀方形远场 (CLI: `spgd-square`) |
| Zernike 工具 | `src/ao_shaping/utils/zernike_utils.py` | 系数解析 (Noll/(n,m)/数组) + 相位生成，Noll 1976 约定 |
| RL training | `src/ao_shaping/optimizer/rl/` | SAC, LR-WFS |
| Simulation | `src/ao_shaping/drivers/sim/` | Digital twin devices |
| Utilities | `src/ao_shaping/utils/` | spots_calc, wavefront_calc, zernike_calc, display |
| ML training | `src/ml/` (standalone, not inside `ao_shaping/`) | U-Net+GAN, trainer, wandb_logger |
| Standalone tools | `src/ao_shaping/tools/` | SLM phase capture, Micro-DM per-channel image collection, train data collection |
| Visualization | `src/ao_shaping/display/` | Windows, frames for GUI |
| GUI | `src/ao_shaping/gui/{r50,dm,slm,zernike,ccd}/` | Streamlit components, 按设备域分包 (见上方目录树) |
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
| wfless/ | `slm_square_shaping.py` | SLM 方形光斑 SPGD 整形 (均匀性 CV + 环围能量) |
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
| `zernike_utils.py` | Zernike 系数解析/校验/相位生成 (独立于 PatternHelper，含 Noll 约定文档) |
| `matrix_utils.py` | Matrix operations |
| `display.py` | Visualization utilities |
| `pattern_helper.py` | SLM pattern generation |
| `file.py` | File I/O utilities |
| `cli_helpers.py` | CLI common utilities (parse_tuple, coredumpy setup) |
| `wfs_utils.py` | WFS utilities (flatten_slopes, compute_snr, DitheredReference) |

---

## Configuration

### Environment Variables (.env)

Project uses `.env` file for environment configuration:

```bash
# Hardware device IDs
Far_Cam_ID=0
Near_Cam_ID=1

# Optical parameters
IDEAL_SPOT_RADIUS=7
CENTER=577,655

# Library paths
PYTHONPATH=src;libs
PATH=libs\Drv_UDPST\x64\Release;libs\gxipy;${PATH}
```

### VSCode Settings

VSCode settings are configured in `.vscode/settings.json`:
- Python path includes `src/` and `libs/`
- Pytest integration enabled
- Terminal environment variables from `.env`

### Config Module

Centralized configuration in `src/ao_shaping/config.py`:

```python
from ao_shaping.config import DM_N_ACTUATORS, DEFAULTS, PATHS

# Hardware constants
DM_N_ACTUATORS = 64

# Default optimization parameters
defaults = DEFAULTS
print(defaults.WF_EPOCHS)  # 20000

# Path configuration
paths = PATHS
print(paths.root_dir)  # data/
```

---

## Entry Points

**CLI Commands (Click-based):**
```bash
# Via main.py hub
python src/ao_shaping/main.py [COMMAND]

# Direct runners (standalone)
python -m ao_shaping.runners.wf_runner
python -m ao_shaping.runners.axis_beam_runner
python -m ao_shaping.runners.pipeline_runner
python -m ao_shaping.runners.zernike_matrix_runner
```

**CLI Structure:**
```
main (click.group)
├── wf             → wf_runner.run()              [Wavefront RMS optimization]
├── pib            → axis_beam_run()              [Power-in-Bucket optimization]
├── pipeline       → pipeline_run()               [Serial WF→PIB pipeline]
├── zernike-matrix → zernike_matrix_run()         [Zernike响应矩阵校准]
├── rms-zernike    → rms_zernike_run()            [Zernike RMS optimization]
├── ga-zernike     → ga_zernike_run()             [GA Zernike optimization]
└── spgd-square    → slm_square_run()             [SLM方形光斑 SPGD 整形]
```

**Note:** `combined_runner.py` is DEPRECATED — use `pipeline_runner.py` instead.

**Refactoring Notes:**
- All runner scripts now use centralized config from `config.py` (DM_N_ACTUATORS, PATHS, DEFAULTS)
- Common CLI helpers moved to `utils/cli_helpers.py` (parse_tuple, setup_coredumpy)
- Duplicate code eliminated across runner files

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
| Passing uint16 grayscale through `create_phase_from_array()` | `create_phase_from_array()` treats input as **radians** (mod 2π → grayscale = rad/2π × 1023). uint16 grayscale values get silently corrupted. Use `np.full((h,w), gray, dtype=np.uint16)` for flat phase or direct grayscale patterns. |
| Consecutive `write_phase` + `display_memory` to the **same** memory slot | Santec SLM firmware treats `display_memory(slot)` as a no-op when that slot is already being displayed — the LCOS panel does **not** refresh. Consecutive writes must ALWAYS target different slots. Preferred pattern (used by diff-shaping runner): pick a **random slot in 2..125** each write, excluding the currently displayed slot — this also survives process restarts (`get_displayed_memory_number()` before the first write). Older tools rotate a small pool like `itertools.cycle([3,4,5])` — works within one process only. The built-in `display_data()` cycles through all 127 slots. |
| Opening the SLM in DVI mode (`video_mode=1`) for diagnostics | `SantecSLM200(..., video_mode=1).open()` can **hang** (observed 120s/300s timeouts), and a hung controller then also hangs memory-mode `open()` until a **physical power cycle**. Never auto-try DVI mode; use memory mode (`video_mode=0`) only. See `tools/slm/slm_diagnose.py` |
| Treating `get_displayed_memory_number` error-code 1 as a fault | In `set_grayscale` mode there is no memory slot being displayed, so `SLM_Ctrl_ReadDS` returns error-code 1 — this is **normal**, not a failure. Slots only exist in memory mode. |
| Assuming the 0-order spot sits at the camera frame center | In the 2f Fourier bench the optical axis (0-order = frame **global maximum**) lands at the camera center only by luck. Observed: frame center (1344,760) vs 0-order spot (1441-1443, 705-706). Always locate 0-order by `argmax`, never by geometry. |
| diff-shaping 硬件闭环挂起（日志止于 `成功打开SLM #1`） | 2026-09-08 实测 (与相机无关): SLM open() 成功后, 首次 `camera.get_numpy_image()` 前无任何日志输出即无限阻塞 (900s 超时被强杀; 分步探针脚本同挂)。`WaitImageV3` 的原生等待由 SDK 内部驱动, 不受 Python 侧超时保护。处置: 强杀后先确认无残留 python 进程 (Get-Process python*)；重跑前 SLM memory 模式 open() 若超过数秒无日志, 对 SLM 控制器物理断电重置 (与 DVI 挂起同一处置)。见 `diff_shaping_runner.py` SLM 连接段注释与 `docs/slm_shaping_diff/readme.md` 故障排查。 |
| Generating SLM phase via `PatternHelper._zernike_to_uint16` | It **min-max normalises** the phase (`(p-pmin)/(pmax-pmin)*1023`) instead of `mod 2π` radians→grayscale, making the pattern **scale-invariant** (coefficients ×1 and ×4 produce byte-identical patterns; verified `np.array_equal == True`). Always convert radian phase through the SLM driver `slm.create_phase_from_array()` (2π=993 + wavefront correction + LUT). See `docs/slm_square_spgd/README.md`. |
| Square shaping with low-order Zernike (n≤4) | Zernike modes are a **circularly symmetric smooth** basis; they physically cannot synthesise a square far-field (needs 2D-sinc-like near field / high spatial frequencies). Use full-pixel phase freedom (GS / differentiable / free-form), not Zernike. |
| Optimising `-CV` alone as the SPGD objective for square shaping | With no energy term the optimizer **empties the target box** to minimise CV (hardware observed EE→0.002). The objective must include encircled energy (use the combined quality score). |
| Trusting `reset_window()`'s returned centre | When the spot is near the frame edge the ROI offset is clamped but the returned `(w//2, h//2)` is not the true spot position → the target box lands off the beam (hardware observed epoch-0 `mean_b=0.01`). Re-locate the spot by `argmax`/centroid on the **windowed** image. |

> 方形光斑 SPGD 整形的完整分析、硬件实测与修复记录见 [`docs/slm_square_spgd/README.md`](docs/slm_square_spgd/README.md)。

---

## UNIQUE STYLES

- **Mock-first testing**: Tests use simulation classes (`SimTurbulenceAOEnv`, `sim_spgd`) to avoid hardware
- **Zernike Noll 约定统一** (aotools Noll 1976): Noll 4 = (2,0) defocus, Noll 5 = (2,-2) astig, Noll 11 = (4,0) spherical, Noll 13 = (4,-2)。**注意** `optimizer/wf/ga_zernike.py` / `rms_by_zernike.py` 里硬编码查表是另一套 (Noll 5 = (2,0)); 新代码一律用 `zernike_calc.noll_to_nm()` / `utils/zernike_utils.py`, 勿混用。zernike_utils 模块文档含完整前 15 阶映射表。
- **Hardware skip pattern**: Tests requiring physical hardware use `pytest.skip("Requires DM hardware")`
- **Recorder pattern**: Optimization tests validate history dictionaries with expected fields
- **Optional backend testing**: CuPy/Numba tested conditionally with try/except guards
- **No fixtures**: No `conftest.py`, fixtures defined inline in test methods
- **SLM flat-phase gray RAW path**: Always send raw uint16 grayscale values to SLM via `np.full((h,w), gray, dtype=np.uint16)`. Never route flat phase through `create_phase_from_array()` (radian conversion). The SLM has amplitude coupling: different flat-phase gray levels produce different camera intensities at 1064nm (periodic with 2π ≈ 993 gray). Use `scripts/validate_flat_phase_gray.py` to verify. Parameters for stable observation: `--exposure-ms 0.8 --wait-time-s 0.3 --discard-count 3`.
- **SLM memory-slot rotation**: When writing consecutive phases to memory mode, always target **different** slots — calling `display_memory(slot)` for the slot already displayed is a no-op and the LCOS panel will not refresh. Preferred pattern (diff-shaping runner): random slot in **2..125**, excluding the currently displayed one (`get_displayed_memory_number()` on start, survives process restarts). Older tools rotate a small pool (`itertools.cycle([3,4,5])`) — in-process only. `display_data()` cycles all 127 slots internally.
- **Streamlit background threads**: Any background loop that drives hardware must never touch `st.session_state` directly — it causes `missing ScriptRunContext` warnings and race conditions. Use the R50 `run_loop` pattern: pass a snapshot of parameters as a plain `dict`, communicate state changes via `threading.Event` for stop signals and mutable containers (e.g. `list`) for live-updated values like frequency, and drain feedback through a `queue.Queue`. The main thread reads/writes `st.session_state` only on rerun. See `src/ao_shaping/gui/r50/r50_voltage_send.py:run_loop` and `src/ao_shaping/gui/slm/multi_slm_controller.py:_toggle_phases_task` for implementations.
- **Timing in background loops**: Prefer `time.time()` wall-clock deltas over counters for state machines. Example: `int(elapsed * 2.0 * freq) % 2 == 0` toggles at exactly the requested frequency without drift, instead of sleeping fixed half-periods and accumulating error.
- **2f Fourier bench geometry** (SLM front focus → f=125mm lens → CCD back focus): the lens Fourier-transforms the SLM field, so CCD coordinates represent **spatial frequency** — the +1 orders of an upper-half-grating and a lower-half-grating land on the **same CCD row (optical-axis row)**, differing only in x-offset. Expected diffraction offset `Δx_px = λ·f/(d_SLM·p_cam) ≈ 5021/Λ` (P64→78px, P32→157px, P96→52, P40→126). Do NOT assume half-screen gratings separate in y. 0-order = frame global max (`argmax`), re-check its location after any bench change.
- **SLM panel "not modulating" diagnostic chain** (2026-09, encoded in `tools/slm/slm_diagnose.py`): ① freeze check — write flat/full-grating/top-half/bottom-half to **rotated memory slots**, frames must differ (all-identical ⇒ LCOS frozen); ② modulation check — `set_grayscale` sweep 0..1023, 0-order bucket must vary with ~993-gray period (flat ⇒ no amplitude coupling ⇒ panel not modulating); ③ linearity check — exposure ×4, ×20 must grow peak brightness (constant peak incl. at 0.1ms/2ms ⇒ light is >100× weaker than the known ~0.02ms near-saturation baseline). Verdict thresholds: freeze ≥2/3 frames differ, modulation bucket rel-spread >15%, linearity growth >2×.

---

## MISSING INFRASTRUCTURE

- **No CI/CD**: No GitHub Actions, no automated testing on push
- **No linting**: No ruff/mypy/flake8 configured
- **No pre-commit**: No hooks for lint/format before commits
- **No requirements.txt**: Only `pyproject.toml` and `uv.lock`

Consider adding: `.github/workflows/ci.yml`, `ruff.toml`, `.pre-commit-config.yaml`

---

## CodeGraph (Pre-indexed Knowledge Graph)

This project has CodeGraph initialized (`.codegraph/` exists, 6,111 nodes, 12,339 edges).

### For Explore agents
Use `codegraph_explore` as your PRIMARY tool — it returns full source code sections from all relevant files in one call.

**Rules:**
1. Follow the explore call budget in the `codegraph_explore` tool description.
2. Do NOT re-read files that `codegraph_explore` already returned source code for.
3. Only fall back to grep/glob/read for files listed under "Additional relevant files" if you need more detail.

### For the main session
Only use these lightweight tools directly:

| Tool | Use For |
|------|---------|
| `codegraph_search` | Find symbols by name |
| `codegraph_callers` / `codegraph_callees` | Trace call flow |
| `codegraph_impact` | Check what's affected before editing |
| `codegraph_node` | Get a single symbol's details |
| `codegraph_context` | Build relevant context for a task |
| `codegraph_files` | Get indexed file structure |
| `codegraph_status` | Check index health

<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them, including dynamic-dispatch hops grep can't follow. Name a file or symbol in the query to read its current line-numbered source. If it's listed but deferred, load it by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` prints the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.
<!-- CODEGRAPH_END -->
