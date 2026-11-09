"""locate() zero-order masking: a bright zero order must not capture the ramp spot."""

from __future__ import annotations

import numpy as np

from ao_shaping.tools.slm.transfer.capture import ZERO_ORDER_EXCLUDE_RADIUS, locate


def _gauss(shape, cx, cy, amp, sigma=6.0):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))


def test_default_locate_is_unchanged_for_single_peak():
    frame = _gauss((200, 200), 120.0, 80.0, 100.0)
    x, y = locate(frame)
    assert abs(x - 120) < 0.5 and abs(y - 80) < 0.5


def test_masked_locate_skips_brighter_zero_order():
    shape = (300, 300)
    zero = (60.0, 150.0)
    first = (170.0, 150.0)  # 110 px away, outside the mask
    frame = _gauss(shape, *zero, 1.2) + _gauss(shape, *first, 1.0)
    assert np.hypot(first[0] - zero[0], first[1] - zero[1]) > ZERO_ORDER_EXCLUDE_RADIUS
    # Unmasked argmax locks the brighter zero order (the 2026-11-09 failure mode).
    x0, _ = locate(frame)
    assert abs(x0 - zero[0]) < 2
    x, y = locate(frame, exclude_xy=zero)
    assert abs(x - first[0]) < 2 and abs(y - first[1]) < 2

