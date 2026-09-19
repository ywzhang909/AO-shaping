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
| CLI runner scripts | `src/ao_shaping/runners/` | 硬件编排层: Click CLI 命令注册, 设备生命周期 (open/close), 结果保存 |
| Optimization algorithms | `src/ao_shaping/algorithm/` | 5 子包: gradient/, heuristic/, signal_processing/, tabu/, goal_functions/. Class-based optimizer convention: see src/ao_shaping/algorithm/README.md |
| Wavefront optimizers | `src/ao_shaping/optimizer/wf/` | RMS optimization |
| Zernike response matrix | `src/ao_shaping/optimizer/wf/zernike_response_matrix.py` | SLM→WFS Zernike校准 |
| PIB optimizers | `src/ao_shaping/optimizer/wfless/` | Power-in-bucket |
| SLM方形光斑整形 (SPGD) | `src/ao_shaping/optimizer/wfless/slm_square_shaping.py` + `runners/slm_square_runner.py` | SPGD 优化 Zernike 系数 → 均匀方形远场 (CLI: `spgd-square`) |
| Zernike 工具 | `src/ao_shaping/utils/wavefront/zernike_utils.py` | 系数解析 (Noll/(n,m)/数组) + 相位生成，Noll 1976 约定 |
| RL training | `src/ao_shaping/optimizer/rl/` | SAC, LR-WFS |
| Simulation | `src/ao_shaping/drivers/sim/` | Digital twin devices |
| Utilities | `src/ao_shaping/utils/{io,image,wavefront,slm}/` | 4 子包: io/, image/, wavefront/, slm/ (spots_calc, wavefront_calc, zernike_calc, display 等) |
| Standalone runners (未注册) | `src/ao_shaping/runners/` | `shaping_runner` 计划迁移至 `scripts/`, `slm_offset_runner` 计划迁移至 `tools/slm/` |
| ML training | `src/ml/` (standalone, not inside `ao_shaping/`) | U-Net+GAN, trainer, wandb_logger |
| Standalone tools | `src/ao_shaping/tools/` | SLM phase capture, Micro-DM per-channel image collection, train data collection |
| Visualization | `src/ao_shaping/display/` | Windows, frames for GUI |
| GUI | `src/ao_shaping/gui/{r50,dm,slm,zernike,ccd}/` | Streamlit components, 按设备域分包 (见上方目录树) |
| Tests | `tests/ao_shaping/` | Mirror of src structure |

---

## optimizer/ Module

高层次优化策略层, 负责实现特定优化目标 (RMS、PIB、Zernike 标定等) 并编排硬件与算法的协作。

### 职责

- 实现优化策略 (RMS via DM电压 / RMS via SLM Zernike / PIB / GA Zernike / 响应矩阵标定等)
- 管理优化过程中的硬件状态 (电压/相位/图像记录)
- 调用 algorithm 包中的基础算法进行参数更新
- 子包:
  - `wf/`: 基于波前传感器的优化 (DM 电压 RMS, SLM Zernike RMS, Zernike 响应矩阵, GA Zernike)
  - `wfless/`: 无波前传感器优化 (PIB via DM电压, SPGD, 模拟 SPGD, 微分波束整形, SLM 方形光斑 SPGD)
  - `rl/`: 强化学习优化 (SAC, LR-WFS)

### 优化策略与 Runner 对应关系

| Runner (main.py 命令) | Optimizer 函数 | 子包 | 优化方式 | 硬件 |
|---|---|---|---|---|
| `wf` | `optimizer.wf.rms:optimizer_rms_dm()` | wf | DM 电压 RMS (SPGD) | DM + WFS |
| `pib` | `optimizer.wfless.pib:optimize_pib()` | wfless | DM 电压 PIB (SPGD) | DM + CCD |
| `pipeline` | `wf.rms:optimizer_rms_dm()` + `wfless.pib:optimize_pib()` | wf + wfless | WF RMS → PIB 串行 | DM + WFS + CCD |
| \zernike-matrix\ | \optimizer.wf.zernike_response_matrix:calibrate_zernike_response_matrix\ | wf | Zernike 响应矩阵标定 + 闭环优化 | SLM + WFS |
| `rms-zernike` | `optimizer.wf.rms_by_zernike:optimizer_rms_slm()` | wf | SLM Zernike RMS | SLM + WFS |
| `ga-zernike` | `optimizer.wf.ga_zernike:optimizer_ga()` | wf | GA Zernike | SLM + WFS |
| `combined` | `optimizer.combined_optimizer:optimize_pib()` | wfless | AdaMOD + SPGD 混合 PIB | DM + CCD |

> **注意**: `optimizer/wf/rms.py` 和 `optimizer/wf/rms_by_zernike.py` 的函数名冲突已通过重命名解决:
> - `rms.py:optimizer_rms_dm()`: DM 电压控制 + WFS 测量 (用于 `wf` 和 `pipeline` 命令)
> - `rms_by_zernike.py:optimizer_rms_slm()`: SLM Zernike 相位控制 + WFS 测量 (用于 `rms-zernike` 命令)

---

### algorithm/ Module — 基础算法层

`src/ao_shaping/algorithm/` 提供纯数学优化算法, 不包含任何硬件知识。

#### Class-based Optimizer Convention

New optimizers added to `src/ao_shaping/algorithm/` MUST follow the class-based API: `__init__` does validation + state setup, `update()` performs one step and returns the next state/solution, and an optional `run()` returns a result dataclass. A one-shot function is kept only as a thin wrapper for backward compatibility. Torch/numpy **simulation-first tests are required before any hardware use**. Canonical example: `DifferentiableBeamOptimizer` (`src/ao_shaping/algorithm/differentiable_beam.py`). Full principle: `src/ao_shaping/algorithm/README.md`.

#### 算法分类

| 类别 | 算法 | 说明 |
|------|------|------|
| 梯度优化 | `Base`, `SGD`, `Adam`, `AdamW`, `AdaMOD`, `Muno`, `MuonW`, `AdamNS` | `update(grad) → next_step` 模式 |
| 启发式搜索 | `GeneticAlgorithm`, `PSO`, `SA`, `CEM`, `DE`, `HC`, `RandomSearch` | 无梯度全局搜索 |
| Tabu 搜索 | `TabuMemory`, `AdaptiveSearchState`, `TabuSearchRunner` | 禁忌搜索 |
| 信号处理 | `PhaseWrapOptimizer`, `GerchbergSaxton`, `SLMPhaseController`, `ControlLaw` | 相位包裹/光强重建/控制律 |
| 可微分 shaping | `DifferentiableBeamOptimizer`, `DifferentiableShapingResult` | PyTorch 可微分波前优化 (需 GPU) |
| 目标函数 | `ImageTargetFunc` | 图像质量指标 |

#### 子包结构

| 子包 | 内容 | 说明 |
|------|------|------|
| `gradient/` | `adam`, `acceleration` | 梯度优化器 |
| `heuristic/` | `ga`, `pso`, `sa`, `hc`, `rs`, `cem`, `de` | 无梯度启发式搜索 |
| `signal_processing/` | `gerchberg_saxton`, `phase_wrap`, `controller`, `iterative_base`, `differentiable_beam`, `differentiable_shaping`, `beam_shaping_utils`, `wavefront`, `beam_shaping_benchmark` | 相位恢复/控制律/可微分整形 |
| `tabu/` | Tabu 搜索 | 禁忌搜索 |
| `goal_functions/` | `target_func`, `image_metrics` | 目标函数与图像质量指标 |

> **注意**: top-level `algorithm/*.py` 文件是 *-re-export shims (向后兼容)。

---

### 三层架构关系

```
CLI (main.py Click 命令)
  │
  ├─ wf ──────────────→ runners/wf_runner.py ──→ optimizer/wf/rms.py:optimizer_rms_dm() ──→ algorithm: Adam/AdaMOD
  ├─ pib ─────────────→ runners/axis_beam_runner.py ──→ optimizer/wfless/pib.py:optimize_pib() ──→ algorithm: AdaMOD/Adam/SGD/Muno
  ├─ pipeline ────────→ runners/pipeline_runner.py ──→ optimizer/wf/rms.py:optimizer_rms_dm() + wfless/pib.py ──→ algorithm: Adam/AdaMOD
  ├─ zernike-matrix ──→ runners/zernike_matrix_runner.py ──→ optimizer/wf/zernike_response_matrix.py ──→ (标定)
  ├─ rms-zernike ────→ runners/rms_zernike_runner.py ──→ optimizer/wf/rms_by_zernike.py ──→ algorithm: Adam/AdaMOD
  ├─ ga-zernike ─────→ runners/ga_zernike_runner.py ──→ optimizer/wf/ga_zernike.py ──→ algorithm: GA
  └─ combined ────────→ runners/combined_runner.py ──→ optimizer/combined_optimizer.py ──→ algorithm: AdaMOD/SPGD

runners/       硬件编排层  — Click CLI, 设备生命周期 (open/close), 结果保存
optimizer/     策略实现层  — 优化流程编排, 硬件状态管理, 调用 algorithm 更新参数
algorithm/     算法基础层  — 纯数学优化器 (update/grad), 无硬件知识
```

**调用链**: `Runner` 打开硬件 → 调用 `Optimizer` 函数 → `Optimizer` 创建 `Algorithm` 实例 → `Algorithm.update(grad)` 返回参数更新 → `Optimizer` 应用更新并记录历史 → `Runner` 关闭硬件并保存结果。

---

## runners/ Module

硬件编排层 — Click CLI 命令注册, 设备生命周期 (open/close), 结果保存。

### 职责

- 注册 CLI 命令 (main.py Click group) 并解析参数
- 打开/关闭硬件设备, 管理设备生命周期
- 调用 optimizer 层函数执行优化流程
- 保存优化结果与调试产物

### 已注册 CLI 命令 vs 独立 Runner

| 类型 | 说明 |
|------|------|
| 已注册 CLI 命令 | main.py 注册 19 个命令 (含 `spgd-square`, `combined` 等), 见 Entry Points 节 |
| 独立 Runner (未注册) | 需直接运行 `python -m ao_shaping.runners.xxx` 或 standalone 脚本; `shaping_runner` 计划迁移至 `scripts/`, `slm_offset_runner` 计划迁移至 `tools/slm/` |

### 共享辅助

`runner_common.py` 是共享辅助函数的主目录: `resolve_dm` (DM 解析), debug-artifact 写入器等。

> **注意**: `closed_loop.py` 不是 runner — 它是 `AOClosedLoop` 控制类, 归属 `optimizer/wf/`。

---

## utils/ Module

Utility functions for image processing and calculations, organized into 4 subpackages:

| Subpackage | Contents | Notes |
|------|---------|-------|
| `utils/io/` | `file`, `timestamp`, `cli_helpers`, `device_config`, `network`, `handler` | `handler` becomes a backward-compat shim to `display/` |
| `utils/image/` | `spots_calc`, `beam_metrics`, `targets`, `resample`, `display`, `gs_visualization`, `hardware_utils` | pygame display relocating to `display/`; `gs_visualization` relocating to `display/` |
| `utils/wavefront/` | `zernike_calc`, `zernike_utils`, `wavefront_calc`, `wfs_utils`, `phase_unwrap`, `hadamard_calc`, `matrix_utils` | |
| `utils/slm/` | `pattern_helper`, `slm_lut`, `slm_utils` | |

> **注意**: legacy top-level `ao_shaping.utils.X` paths remain importable via shims.

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
python src/ao_shaping/main.py wf
python src/ao_shaping/main.py pib
python src/ao_shaping/main.py pipeline
python src/ao_shaping/main.py zernike-matrix
python src/ao_shaping/main.py rms-zernike
python src/ao_shaping/main.py ga-zernike
python src/ao_shaping/main.py combined
```

**CLI Structure (main.py 注册关系):**
```
main (click.group)
├── wf             ← wf_runner.run        [Wavefront RMS via DM电压 + WFS]
├── pib            ← axis_beam_runner.run  [Power-in-Bucket via DM电压 + CCD]
├── pipeline       ← pipeline_runner.run   [Serial WF RMS → PIB]
├── zernike-matrix ← zernike_matrix_runner.run [Zernike响应矩阵标定 + 闭环优化 (closed_loop_run)]
├── rms-zernike    ← rms_zernike_runner.run    [SLM Zernike RMS]
├── ga-zernike     ← ga_zernike_runner.run     [GA Zernike]
└── combined       ← combined_runner.run       [AdaMOD+SPGD 混合 PIB]
```

> **注意**: `spgd-square` 命令 (`runners/slm_square_runner.py:run`) 已注册到 main.py (main.py:112)。

**Note:** `combined` 命令仍注册于 main.py (main.py:106) 且功能可用, 作为 legacy 保留。`pipeline_runner.py` 是推荐的 WF→PIB 串行方案。

> **未注册到 main.py 的独立 Runner** (需直接运行 `python -m ao_shaping.runners.xxx` 或 standalone 脚本):
> `shaping_runner` (计划迁移至 `scripts/`), `slm_offset_runner` (计划迁移至 `tools/slm/`), `hadamard_matrix_runner` (正在注册为 `hadamard-matrix` 命令)

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
from ao_shaping.drivers import MIICamera, DahengCamera
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
| `combined_runner.py` with main CLI | Prefer `pipeline_runner.py` for new serial flows; `combined` remains registered as legacy (main.py:106) |
| Passing uint16 grayscale through `create_phase_from_array()` | `create_phase_from_array()` treats input as **radians** (mod 2π → grayscale = rad/2π × 1023). uint16 grayscale values get silently corrupted. Use `np.full((h,w), gray, dtype=np.uint16)` for flat phase or direct grayscale patterns. |
| SLM 相位生成函数自行 `np.mod(phase, 2π)` (2026-09 raw-only 契约) | All phase generators must return **raw unwrapped radians** — the only mod-2π wrap lives in the driver `Santec.create_phase_from_array()` on radian→grayscale conversion (`santec/driver.py` L1382). Generators that self-wrap duplicate the driver contract and hide the true phase. Convert via `utils/slm_utils.phase_to_slm_grayscale(phase, slm=slm)` (hardware) or the pure fallback (offline/tests). Exception: `_zernike_phase_radians` keeps a wrapped output **only** as a test-only reference. |
| Consecutive `write_phase` + `display_memory` to the **same** memory slot | Santec SLM firmware treats `display_memory(slot)` as a no-op when that slot is already being displayed — the LCOS panel does **not** refresh. Consecutive writes must ALWAYS target different slots. Preferred pattern (used by diff-shaping runner): pick a **random slot in 2..125** each write, excluding the currently displayed slot — this also survives process restarts (`get_displayed_memory_number()` before the first write). Older tools rotate a small pool like `itertools.cycle([3,4,5])` — works within one process only. The built-in `display_data()` cycles through all 127 slots. |
| Opening the SLM in DVI mode (`video_mode=1`) for diagnostics | `Santec(..., video_mode=1).open()` can **hang** (observed 120s/300s timeouts), and a hung controller then also hangs memory-mode `open()` until a **physical power cycle**. Never auto-try DVI mode; use memory mode (`video_mode=0`) only. See `tools/slm/slm_diagnose.py` |
| Treating `get_displayed_memory_number` error-code 1 as a fault | In `set_grayscale` mode there is no memory slot being displayed, so `SLM_Ctrl_ReadDS` returns error-code 1 — this is **normal**, not a failure. Slots only exist in memory mode. |
| Re-writing a **cached displayed phase** back through `write_phase` | Cached phases (`get_displayed_phase()`) already contain base overlay + wavefront correction; `write_phase` re-applies both → corrected **twice**. Rewriting shifted phases must go through the driver-level `Santec.apply_shift()` (raw write via `_write_to_memory` + slot rotation + config save), which is the single public entry point for shift-and-redisplay (identical semantics to the multi-SLM controller "应用平移" button). Shift math's only implementation is the static `Santec.shift_phase()`. See `src/ao_shaping/drivers/slm/AGENTS.md`. |
| Assuming the 0-order spot sits at the camera frame center | In the 2f Fourier bench the optical axis (0-order = frame **global maximum**) lands at the camera center only by luck. Observed: frame center (1344,760) vs 0-order spot (1441-1443, 705-706). Always locate 0-order by `argmax`, never by geometry. |
| diff-shaping 硬件闭环挂起（日志止于 `成功打开SLM #1`） | 2026-09-08 实测 (与相机无关): SLM open() 成功后, 首次 `camera.get_numpy_image()` 前无任何日志输出即无限阻塞 (900s 超时被强杀; 分步探针脚本同挂)。`WaitImageV3` 的原生等待由 SDK 内部驱动, 不受 Python 侧超时保护。处置: 强杀后先确认无残留 python 进程 (Get-Process python*)；重跑前 SLM memory 模式 open() 若超过数秒无日志, 对 SLM 控制器物理断电重置 (与 DVI 挂起同一处置)。见 `diff_shaping_runner.py` SLM 连接段注释与 `docs/slm_shaping_diff/readme.md` 故障排查。 |
| Generating SLM phase via `PatternHelper._zernike_to_uint16` | It **min-max normalises** the phase (`(p-pmin)/(pmax-pmin)*1023`) instead of `mod 2π` radians→grayscale, making the pattern **scale-invariant** (coefficients ×1 and ×4 produce byte-identical patterns; verified `np.array_equal == True`). Always convert radian phase through the SLM driver `slm.create_phase_from_array()` (2π=993 + wavefront correction + LUT). See `docs/slm_square_spgd/README.md`. |
| **`ZernikeDM.generate_phase` min-max normalises the phase** (same anti-pattern, still live) | `drivers/dm/zernike_dm.py:106-129` does `(raw−min)/(max−min)` then `×2π` (rad) / `×max_val` (gray) → output is **scale-invariant**: coefficients ×1 and ×4 give **byte-identical** phases (verified `np.array_equal == True`, PV always 2π). Since `ZernikeSLM.send_zernike` routes through it, **the Zernike coefficient amplitude is uncontrollable** for every consumer (`zernike-matrix`, `rms-zernike`, `ga-zernike`, `greedy-zernike`, `runners/zernike_matrix_runner.py`, `runners/rms_zernike_runner.py`, `gui/zernike/`). Fix: drop the normalisation (treat coefficients as radians) — note this intentionally changes those callers' behaviour. See `docs/slm/report2.md` §2.3. |
| **Mixing units across the WFS→SLM correction loop** | The WFS `get_zernike()` returns **µm**; the response matrix must be built in the **same unit** as the correction's `w` (λ). Two real bugs (2026-09-16): matrix built in µm vs `w` in λ → coefficients inflated **1/0.532 = 1.88×**; and the solved coefficients are in **λ (waves)** but `make_phase`/`generate_zernike_polynomial` take **radians** → applied phase shrunk **2π = 6.28×**. Both fixed → closed-loop RMS improvement 13.8% → **42.1%**. Always route through `tools/slm/slm_zernike_common.um_to_waves()` and multiply λ→rad by `2π` before `make_phase`. See `docs/slm/report2.md` §0. |
| Square shaping with low-order Zernike (n≤4) | Zernike modes are a **circularly symmetric smooth** basis; they physically cannot synthesise a square far-field (needs 2D-sinc-like near field / high spatial frequencies). Use full-pixel phase freedom (GS / differentiable / free-form), not Zernike. |
| Optimising `-CV` alone as the SPGD objective for square shaping | With no energy term the optimizer **empties the target box** to minimise CV (hardware observed EE→0.002). The objective must include encircled energy (use the combined quality score). |
| Trusting `reset_window()`'s returned centre | When the spot is near the frame edge the ROI offset is clamped but the returned `(w//2, h//2)` is not the true spot position → the target box lands off the beam (hardware observed epoch-0 `mean_b=0.01`). Re-locate the spot by `argmax`/centroid on the **windowed** image. |
| Function-only optimizers in ao_shaping/algorithm (no class API) | New optimizers must expose __init__ (validation + state) + update() (one step) + optional run() (result dataclass); one-shot functions are legacy/thin wrappers only. See src/ao_shaping/algorithm/README.md. |
| Placing markdown/report **generation** under `src/ao_shaping/tools/` | **All markdown/illustrated-report generation MUST live in `scripts/`** (naming: `scripts/generate_*_report.py`, e.g. `generate_zernike_wfs_report.py`, `generate_diff_shaping_report.py`). `src/ao_shaping/tools/` is reserved for hardware-interaction tools (CLI + driver orchestration), not report writers. See scripts/README.md. |
| Report generation inside `algorithm/` | `signal_processing/beam_shaping_benchmark` writes CSV/MD/GIF reports — report generation MUST live in `scripts/` (see scripts/README.md), not in the algorithm layer. |
| Pygame/viz code inside `utils/` | `utils/image/display.py` and `utils/image/gs_visualization.py` are visualization code — they belong in `display/`, not the leaf utils layer. |
| Re-implementing the GS loop | `gs_visualization.gerchberg_saxton_with_visualization` must drive the canonical `gerchberg_saxton` via a callback, not duplicate the loop. |
| Dead vendored ctypes VISA code | `utils/wavefront/vi.py` is dead vendored ctypes VISA code — remove it or move it out of utils. |
| Min-max scale-invariant `PatternHelper._zernike_to_uint16` | `_zernike_to_uint16` min-max normalises the phase, making patterns scale-invariant (coefficients ×1 and ×4 → byte-identical). Use `utils/slm/slm_utils.phase_to_slm_grayscale(phase, slm=slm)` instead. |
| Vector-beam demo `sim.py` living in `algorithm/` | The vector-beam demo `sim.py` lives in `algorithm/` — it belongs in `scripts/`. |
| `utils/slm/pattern_helper.py` importing `from ao_shaping.algorithm.phase_wrap` at module top level | utils is the leaf layer and must not depend on `algorithm/` at import time — use deferred function-local imports. |

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
- **SLM phase raw-only contract + unified grayscale entry** (2026-09): Every SLM phase generator (optimizer/runner layer) returns **raw unwrapped radians**; do NOT `mod 2π` in generators — the driver `create_phase_from_array()` is the single wrap point (radian→grayscale, `santec/driver.py` L1382). Convert radian phase to grayscale through `utils/slm_utils.phase_to_slm_grayscale(phase, slm=slm)` — passes a live SLM it delegates to the driver pipeline (grayscale + wavefront correction + LUT + shift, 2π = device `_max_gray`); `slm=None` (simulation/offline save/unit tests) falls back to the built-in pure math conversion (wrap→scale→clip→uint16, default 1023). Counter-examples still live: `PatternHelper._zernike_to_uint16` and `ZernikeDM.generate_phase` (scale-invariant min-max normalisers) must NOT be used.
- **SLM memory-slot rotation**: When writing consecutive phases to memory mode, always target **different** slots — calling `display_memory(slot)` for the slot already displayed is a no-op and the LCOS panel will not refresh. Preferred pattern (diff-shaping runner): random slot in **2..125**, excluding the currently displayed one (`get_displayed_memory_number()` on start, survives process restarts). Older tools rotate a small pool (`itertools.cycle([3,4,5])`) — in-process only. `display_data()` cycles all 127 slots internally.
- **Streamlit background threads**: Any background loop that drives hardware must never touch `st.session_state` directly — it causes `missing ScriptRunContext` warnings and race conditions. Use the R50 `run_loop` pattern: pass a snapshot of parameters as a plain `dict`, communicate state changes via `threading.Event` for stop signals and mutable containers (e.g. `list`) for live-updated values like frequency, and drain feedback through a `queue.Queue`. The main thread reads/writes `st.session_state` only on rerun. See `src/ao_shaping/gui/r50/r50_voltage_send.py:run_loop` and `src/ao_shaping/gui/slm/multi_slm_controller.py:_toggle_phases_task` for implementations.
- **Timing in background loops**: Prefer `time.time()` wall-clock deltas over counters for state machines. Example: `int(elapsed * 2.0 * freq) % 2 == 0` toggles at exactly the requested frequency without drift, instead of sleeping fixed half-periods and accumulating error.
- **2f Fourier bench geometry** (SLM front focus → f=125mm lens → CCD back focus): the lens Fourier-transforms the SLM field, so CCD coordinates represent **spatial frequency** — the +1 orders of an upper-half-grating and a lower-half-grating land on the **same CCD row (optical-axis row)**, differing only in x-offset. Expected diffraction offset `Δx_px = λ·f/(d_SLM·p_cam) ≈ 5021/Λ` (P64→78px, P32→157px, P96→52, P40→126). Do NOT assume half-screen gratings separate in y. 0-order = frame global max (`argmax`), re-check its location after any bench change.
- **SLM panel "not modulating" diagnostic chain** (2026-09, encoded in `tools/slm/slm_diagnose.py`): ① freeze check — write flat/full-grating/top-half/bottom-half to **rotated memory slots**, frames must differ (all-identical ⇒ LCOS frozen); ② modulation check — `set_grayscale` sweep 0..1023, 0-order bucket must vary with ~993-gray period (flat ⇒ no amplitude coupling ⇒ panel not modulating); ③ linearity check — exposure ×4, ×20 must grow peak brightness (constant peak incl. at 0.1ms/2ms ⇒ light is >100× weaker than the known ~0.02ms near-saturation baseline). Verdict thresholds: freeze ≥2/3 frames differ, modulation bucket rel-spread >15%, linearity growth >2×.
- **Class-based optimizer convention**: optimizers in ao_shaping/algorithm expose __init__ (validate + set state), update() (one step → next state), optional run() → result dataclass; the one-shot function stays a thin wrapper. Simulation-first (torch/numpy) tests before hardware. Canonical example: DifferentiableBeamOptimizer. See src/ao_shaping/algorithm/README.md.
- **Report generation lives in `scripts/`**: any functionality that writes markdown/illustrated reports MUST live in `scripts/` (naming `generate_*_report.py`, e.g. `generate_zernike_wfs_report.py`, `generate_zernike_response_matrix_report.py`, `generate_diff_shaping_report.py`) — NEVER under `src/ao_shaping/tools/`, which is reserved for hardware-interaction tools (CLI + driver orchestration). Scripts follow the repo script conventions: `matplotlib.use("Agg")` BEFORE importing pyplot, `sys.path` bootstrap (`ROOT = Path(__file__).resolve().parents[1]`), CJK font rcParams (`Microsoft YaHei`/`SimHei` + `axes.unicode_minus = False`), output under `docs/<topic>/` or `logs/<timestamp>/`, and MUST be documented in `scripts/README.md`. Prefer **offline** report generators (read saved artefacts) so a report can be regenerated without hardware.

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
