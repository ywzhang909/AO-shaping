"""SLM phase -> grayscale conversion and memory-slot rotation helpers.

Pure helpers that operate on an SLM device object through duck-typing (any
object exposing ``get_displayed_memory_number``, ``create_phase_from_array``,
``write_phase`` and ``display_memory``). No hardware import is performed here;
callers pass an open SLM device.

Migrated from :mod:`ao_shaping.algorithm.beam_shaping_utils`; the old module
re-exports these names for backward compatibility.

Memory-slot rules (AGENTS.md iron rules):
- Consecutive writes must ALWAYS target different slots; ``display_memory``
  on the slot already displayed is a no-op and the LCOS panel will not refresh.
- Preferred pattern: random slot in 2..125, excluding the currently displayed
  one (``get_displayed_memory_number()`` on first call, survives restarts).
- Memory mode only (``video_mode=0``); never auto-try DVI mode.
"""

from __future__ import annotations

import random
import time
from typing import Any

import numpy as np

from loguru import logger

__all__ = [
    "DEFAULT_WAVELENGTH",
    "DEFAULT_SLM_PIXEL_SIZE",
    "DEFAULT_DISTANCE",
    "DEFAULT_MAX_GRAYSCALE",
    "phase_to_slm_grayscale",
    "pick_slm_slot",
    "display_phase",
]

# Shared physical / SLM constants
DEFAULT_WAVELENGTH: float = 1064e-9  # 1064 nm YAG laser (meters)
DEFAULT_SLM_PIXEL_SIZE: float = 8e-6  # SLM pixel pitch (meters)
DEFAULT_DISTANCE: float = 0.1  # default propagation distance (meters)
DEFAULT_MAX_GRAYSCALE: int = 1023  # grayscale value corresponding to 2*pi


def phase_to_slm_grayscale(
    phase: np.ndarray,
    max_grayscale: int = DEFAULT_MAX_GRAYSCALE,
) -> np.ndarray:
    """Convert a phase map (radians) to an SLM uint16 grayscale map.

    Wraps phase into ``[0, 2*pi)`` and scales to the ``[0, max_grayscale]``
    range. Values are clamped to the grayscale range and cast to ``uint16``.

    Args:
        phase: 2D phase array in radians.
        max_grayscale: The grayscale value that corresponds to ``2*pi``
            (default 1023 for a 10-bit LCOS device).

    Returns:
        ``uint16`` 2D grayscale array.
    """
    phase = np.asarray(phase, dtype=np.float32)
    phase = np.mod(phase, 2 * np.pi)
    gray = (phase / (2 * np.pi)) * max_grayscale
    return np.clip(gray, 0, max_grayscale).astype(np.uint16)


# ---------------------------------------------------------------------------
# SLM memory-slot rotation (shared by diff_beam_runner and diff_shaping_runner)
# ---------------------------------------------------------------------------
_SLOT_MIN: int = 2
_SLOT_MAX: int = 125
_last_slm_slot: int | None = None


def pick_slm_slot(slm: Any) -> int:
    """Pick a random SLM memory slot in ``[_SLOT_MIN, _SLOT_MAX]`` that differs
    from the last used slot.

    On first call the currently displayed slot is read from the device so the
    rotation survives process restarts (memory mode only).

    Args:
        slm: Open SLM device object exposing ``get_displayed_memory_number``.

    Returns:
        Slot number in ``[2, 125]``.
    """
    global _last_slm_slot
    if _last_slm_slot is None:
        try:
            _last_slm_slot = slm.get_displayed_memory_number()
            logger.info("SLM当前显示槽: {}", _last_slm_slot)
        except Exception as exc:
            logger.debug("读取 SLM 当前显示槽失败: {}", exc)
            _last_slm_slot = None
    candidates = [s for s in range(_SLOT_MIN, _SLOT_MAX + 1) if s != _last_slm_slot]
    slot = random.choice(candidates)
    _last_slm_slot = slot
    return slot


def display_phase(slm: Any, phase_rad: np.ndarray, settle_time_s: float) -> None:
    """Display a radian phase pattern on the SLM using memory-slot rotation.

    Uses ``create_phase_from_array`` (device-authentic 2π conversion +
    correction/LUT) and memory mode only (never DVI).

    Args:
        slm: Open SLM device.
        phase_rad: 2D phase array in radians.
        settle_time_s: Wait time after ``display_memory``.
    """
    gray = slm.create_phase_from_array(phase_rad)
    slot = pick_slm_slot(slm)
    slm.write_phase(gray, memory_number=slot)
    time.sleep(0.05)
    slm.display_memory(slot)
    time.sleep(settle_time_s)