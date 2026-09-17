"""Shared SLM phase and camera-frame helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_PANEL_RES = (1200, 1920)


def flat_gray(
    panel_res: tuple[int, int] | None = None,
    gray_value: int = 0,
) -> np.ndarray:
    """Create a raw uint16 flat grayscale pattern."""
    height, width = panel_res or DEFAULT_PANEL_RES
    return np.full((height, width), gray_value, dtype=np.uint16)


def capture_frame(
    camera: Any,
    n_sample: int = 1,
    skip_first: bool = True,
    discard_count: int = 0,
) -> np.ndarray:
    """Capture a fresh averaged frame with optional discarded leading frames."""
    if n_sample < 1:
        raise ValueError("n_sample must be at least 1")
    discard = 1 if skip_first and n_sample == 1 else 0
    for _ in range(discard + discard_count):
        camera.get_numpy_image(n_sample=1)
    return camera.get_numpy_image(n_sample=n_sample, skip_first=skip_first)


__all__ = ["DEFAULT_PANEL_RES", "capture_frame", "flat_gray"]
