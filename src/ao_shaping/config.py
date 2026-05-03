"""AO-Shaping unified configuration module.

This module provides centralized configuration management for the AO-Shaping system,
including hardware constants, paths, and default parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# ============================================
# Hardware Constants
# ============================================
DM_N_ACTUATORS: int = 64
"""Number of deformable mirror actuators."""

DM_DISABLED_ACTUATORS: list[int] = [0]
"""List of actuator indices to disable (0-indexed)."""


# ============================================
# Default Optimization Parameters
# ============================================
@dataclass(frozen=True)
class OptimizationDefaults:
    """Default parameters for optimization algorithms."""
    
    # WF (Wavefront) optimization
    WF_EPOCHS: int = 20_000
    WF_EARLY_STOP_THRESHOLD: float = 0.12
    WF_WFS_RES: str = "768"
    WF_PUPIL_DIAMETER: float = 2.7
    
    # PIB (Power-in-Bucket) optimization  
    PIB_EPOCHS: int = 4_000
    PIB_DELTA: float = 2.0
    PIB_LR: float = 0.0  # 0 means adaptive
    PIB_R_BUCKET: int = 0  # 0 means auto-adjust
    PIB_SHRINK_ITER: int = 200
    PIB_SHRINK_RATIO: float = 0.8
    
    # Pipeline optimization
    PIPELINE_WF_EPOCHS: int = 8_000
    PIPELINE_PIB_EPOCHS: int = 8_000
    PIPELINE_RMS_THRESHOLD: float = 0.12
    
    # GA Zernike optimization
    GA_POPULATION_SIZE: int = 50
    GA_N_GENERATIONS: int = 2000
    GA_CROSSOVER_PROB: float = 0.7
    GA_MUTATION_PROB: float = 0.15
    GA_TOURNAMENT_SIZE: int = 3
    GA_ELITE_COUNT: int = 2
    GA_N_MAX: int = 4


# ============================================
# Path Configuration
# ============================================
@dataclass
class PathConfig:
    """Path configuration for data storage and logs."""
    
    root_dir: Path = field(default_factory=lambda: Path("data"))
    """Root directory for all data."""
    
    voltages_dir: str = "flatten_voltages"
    """Subdirectory for saved voltages."""
    
    zernike_dir: str = "flatten_zernike"
    """Subdirectory for saved Zernike coefficients."""
    
    log_dir: Path = field(default_factory=lambda: Path("logs/debug/error"))
    """Directory for debug/error logs."""
    
    wf_subdir: str = "wf"
    """Subdirectory for wavefront optimization results."""
    
    pib_subdir: str = "wf-less"
    """Subdirectory for PIB optimization results."""
    
    pipeline_subdir: str = "pipeline"
    """Subdirectory for pipeline optimization results."""
    
    rms_zernike_subdir: str = "rms_zernike"
    """Subdirectory for RMS Zernike optimization results."""
    
    def get_voltages_path(self, date_str: str | None = None) -> Path:
        """Get path for voltage files."""
        if date_str is None:
            from datetime import datetime
            date_str = datetime.now().strftime("%Y%m%d")
        return self.root_dir / self.voltages_dir / date_str
    
    def get_debug_path(self, subdir: str) -> Path:
        """Get debug save path with date subdirectory."""
        from datetime import datetime
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.root_dir / subdir / date_str
        path.mkdir(parents=True, exist_ok=True)
        return path


# ============================================
# Device Configuration
# ============================================
@dataclass
class DeviceConfig:
    """Hardware device configuration."""
    
    far_cam_id: int = field(default_factory=lambda: int(os.environ.get("Far_Cam_ID", 0)))
    """Far-field camera ID."""
    
    near_cam_id: int = field(default_factory=lambda: int(os.environ.get("Near_Cam_ID", 1)))
    """Near-field camera ID."""
    
    slm_number: int = 1
    """SLM device number."""
    
    slm_wavelength: int = 532
    """SLM wavelength in nm."""
    
    exposure_time_ms: int = 60
    """Default camera exposure time in milliseconds."""
    
    cam_size: int = 200
    """Default camera window size."""
    
    target_max_brightness: int = 90
    """Target maximum brightness for auto-exposure."""


# ============================================
# Global Instances
# ============================================
DEFAULTS = OptimizationDefaults()
PATHS = PathConfig()
DEVICES = DeviceConfig()


# ============================================
# Utility Functions
# ============================================
def get_dm_unit_mask(available: Literal["all", "inner", "outer"] = "all") -> list[bool]:
    """Generate DM unit mask based on availability pattern.
    
    Args:
        available: Which actuators to enable ("all", "inner", or "outer")
        
    Returns:
        List of boolean values indicating which actuators are enabled.
    """
    mask = [True] * DM_N_ACTUATORS
    
    # Disable problematic actuators
    for idx in DM_DISABLED_ACTUATORS:
        mask[idx] = False
    
    if available == "inner":
        # Only inner 21 actuators (indices 0-20)
        for i in range(21, DM_N_ACTUATORS):
            mask[i] = False
    elif available == "outer":
        # Only outer actuators (indices 39-63)
        for i in range(39):
            mask[i] = False
    
    return mask


def get_init_voltages() -> list[float]:
    """Get initial voltage values (all zeros)."""
    return [0.0] * DM_N_ACTUATORS


def get_coredumpy_directory() -> str:
    """Get coredumpy error directory path."""
    return str(PATHS.log_dir)