"""ao_epics_common - IOC 公共框架包别名。

让各 IOC 能以 `import ao_epics_common` 引入公共组件。
运行方式(从 IOC 目录):
    python -m ao_epics_common.serve <ioc.yaml> <module> <PVGroupClass>
或更简单:各 IOC 自带 main 脚本,通过 PYTHONPATH 指向本目录。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 使本包可作为顶层包导入(IOC 运行前将本目录加入 PYTHONPATH)
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from .device_manager import DeviceManager, OpenCloseDevice  # noqa: E402
from .ioc_config import (  # noqa: E402
    DeviceSpec,
    IocConfigError,
    IocSpec,
    load_ioc_config,
    validate,
)
from .ioc_runner import PVGroup, build_pv_group, pvproperty, run_ioc  # noqa: E402
from .slm_rules import (  # noqa: E402
    DEFAULT_SLOTS,
    GRAYSCALE_MAX,
    GRAYSCALE_MIN,
    SLM_PANEL_HEIGHT,
    SLM_PANEL_WIDTH,
    MemorySlotRotator,
    SlmRuleError,
    flat_phase_grayscale,
    validate_dm_voltages,
    validate_grayscale,
    validate_phase_array,
)

__all__ = [
    "DeviceManager",
    "OpenCloseDevice",
    "DeviceSpec",
    "IocConfigError",
    "IocSpec",
    "load_ioc_config",
    "validate",
    "PVGroup",
    "build_pv_group",
    "pvproperty",
    "run_ioc",
    "DEFAULT_SLOTS",
    "GRAYSCALE_MAX",
    "GRAYSCALE_MIN",
    "SLM_PANEL_HEIGHT",
    "SLM_PANEL_WIDTH",
    "MemorySlotRotator",
    "SlmRuleError",
    "flat_phase_grayscale",
    "validate_dm_voltages",
    "validate_grayscale",
    "validate_phase_array",
]
