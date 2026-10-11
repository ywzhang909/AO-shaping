"""Hardware measurement helpers for SLM+CCD closed loop (private)."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from ao_shaping.utils.image.beam_metrics import locate_peak, prepare_frame


def locate_zero_order(frame: np.ndarray) -> tuple[int, int]:
    return locate_peak(frame, finite=True)


def display_and_average(
    cam: Any,
    slm: Any,
    phase_rad: np.ndarray,
    *,
    n_frames: int,
    settle_wait_s: float,
    settle_tol: float,
    settle_max_wait_s: float,
    settle_max_discard: int,
    memory_number: int | None = None,
) -> np.ndarray:
    gray = slm.create_phase_from_array(phase_rad)
    if memory_number is None:
        slm.display_data(gray)
    else:
        slm.display_data(gray, memory_number=memory_number)
    if settle_wait_s > 0:
        time.sleep(settle_wait_s)

    def probe() -> tuple[float, float, float]:
        img = np.asarray(cam.get_numpy_image(n_sample=1), dtype=np.float64)
        cy, cx = locate_zero_order(img)
        return float(img[cy, cx]), float(cy), float(cx)

    prev = probe()
    deadline = time.time() + max(0.0, settle_max_wait_s - settle_wait_s)
    for _ in range(int(settle_max_discard)):
        if time.time() >= deadline:
            break
        cur = probe()
        if (
            abs(cur[0] - prev[0]) <= settle_tol * max(prev[0], 1.0)
            and abs(cur[1] - prev[1]) <= settle_tol * 40.0
            and abs(cur[2] - prev[2]) <= settle_tol * 40.0
        ):
            break
        prev = cur
    acc = [
        np.asarray(cam.get_numpy_image(n_sample=1), dtype=np.float64)
        for _ in range(max(1, int(n_frames)))
    ]
    return prepare_frame(np.mean(acc, axis=0))
