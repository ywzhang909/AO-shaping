"""AO-Shaping unified configuration module.

This module provides centralized configuration management for the AO-Shaping system,
including hardware constants, paths, and default parameters.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal


def _resolve_class_attribute(cls: type, name: str) -> Any:
    """Read a class attribute, resolving instance-level ``@property`` descriptors.

    Some DM types (ZernikeDM, HadamardDM) expose ``DM_NUM`` as an instance
    ``@property`` derived from the generator, so ``cls.DM_NUM`` returns the
    descriptor object instead of the value.  Instantiate the class to read the
    real value; returns ``None`` if the attribute is absent or cannot be read.
    """
    value = getattr(cls, name, None)
    if isinstance(value, property):
        try:
            value = getattr(cls(), name)
        except Exception:
            return None
    return value


def _resolve_dm_n_actuators() -> int:
    """Resolve DM actuator count from device driver.

    Uses the DM registry to find the first reachable DM type.
    Falls back to 64 (NLight default) if no DM is reachable.
    """
    try:
        from ao_shaping.drivers.dm._registry import get_dm_registry
        registry = get_dm_registry()
        reachable = registry.list_reachable_types()
        if len(reachable) >= 1:
            cls = registry.get_class(reachable[0])
            dm_num = _resolve_class_attribute(cls, "DM_NUM")
            if isinstance(dm_num, int):
                return dm_num
        return 64
    except Exception:
        return 64


def _resolve_disabled_actuators() -> list[int]:
    """Resolve disabled actuators from device driver.

    Uses the DM registry to find the first reachable DM type.
    Falls back to [0] if no DM is reachable.
    """
    try:
        from ao_shaping.drivers.dm._registry import get_dm_registry
        registry = get_dm_registry()
        reachable = registry.list_reachable_types()
        if len(reachable) >= 1:
            cls = registry.get_class(reachable[0])
            disabled = _resolve_class_attribute(cls, "disabled_actuators")
            if isinstance(disabled, list):
                return disabled
        return [0]
    except Exception:
        return [0]


DEFAULT_OPTIMIZATION_DEFAULTS = dict(
    WF_EPOCHS=20_000,
    WF_EARLY_STOP_THRESHOLD=0.12,
    WF_WFS_RES="768",
    WF_PUPIL_DIAMETER=2.7,
    PIB_EPOCHS=4_000,
    PIB_DELTA=2.0,
    PIB_LR=0.0,
    PIB_R_BUCKET=0,
    PIB_SHRINK_ITER=200,
    PIB_SHRINK_RATIO=0.8,
    PIPELINE_WF_EPOCHS=8_000,
    PIPELINE_PIB_EPOCHS=8_000,
    PIPELINE_RMS_THRESHOLD=0.12,
    GA_POPULATION_SIZE=50,
    GA_N_GENERATIONS=2000,
    GA_CROSSOVER_PROB=0.7,
    GA_MUTATION_PROB=0.15,
    GA_TOURNAMENT_SIZE=3,
    GA_ELITE_COUNT=2,
    GA_N_MAX=4,
)

DEFAULT_PATHS = dict(
    root_dir="data",
    voltages_dir="flatten_voltages",
    zernike_dir="flatten_zernike",
    log_dir="logs/debug/error",
    wf_subdir="wf",
    pib_subdir="wf-less",
    pipeline_subdir="pipeline",
    rms_zernike_subdir="rms_zernike",
)

DEFAULT_DEVICE_CONFIG = dict(
    far_cam_id=0,
    near_cam_id=1,
    slm_number=1,
    slm_wavelength=532,
    exposure_time_ms=60,
    cam_size=200,
    target_max_brightness=90,
)


class OptimizationDefaults:
    def __init__(self):
        for k, v in DEFAULT_OPTIMIZATION_DEFAULTS.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)

    def __getattr__(self, name: str):
        if name in DEFAULT_OPTIMIZATION_DEFAULTS:
            return DEFAULT_OPTIMIZATION_DEFAULTS[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class PathConfig:
    def __init__(self):
        for k, v in DEFAULT_PATHS.items():
            setattr(self, k, Path(v) if k == "log_dir" else v)
        self.root_dir = Path("data")

    def get_voltages_path(self, date_str: str | None = None) -> Path:
        from datetime import datetime
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")
        return self.root_dir / self.voltages_dir / date_str

    def get_debug_path(self, subdir: str) -> Path:
        from datetime import datetime
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.root_dir / subdir / date_str
        path.mkdir(parents=True, exist_ok=True)
        return path

    def __getattr__(self, name: str):
        if name in DEFAULT_PATHS:
            return DEFAULT_PATHS[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class DeviceConfig:
    def __init__(self):
        self.far_cam_id = int(os.environ.get("Far_Cam_ID", 0))
        self.near_cam_id = int(os.environ.get("Near_Cam_ID", 1))

    def __getattr__(self, name: str):
        if name in DEFAULT_DEVICE_CONFIG:
            return DEFAULT_DEVICE_CONFIG[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


DEFAULTS = OptimizationDefaults()
PATHS = PathConfig()
DEVICES = DeviceConfig()


def get_dm_unit_mask(available: Literal["all", "inner", "outer"] = "all") -> list[bool]:
    n_actuators = _resolve_dm_n_actuators()
    disabled = _resolve_disabled_actuators()
    mask = [True] * n_actuators

    for idx in disabled:
        mask[idx] = False

    if available == "inner":
        for i in range(21, n_actuators):
            mask[i] = False
    elif available == "outer":
        for i in range(39):
            mask[i] = False

    return mask


def get_init_voltages() -> list[float]:
    return [0.0] * _resolve_dm_n_actuators()


def get_coredumpy_directory() -> str:
    return str(PATHS.log_dir)


def __getattr__(name: str):
    if name == "DM_N_ACTUATORS":
        return _resolve_dm_n_actuators()
    if name == "DM_DISABLED_ACTUATORS":
        return _resolve_disabled_actuators()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
