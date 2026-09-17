"""Test the gs_square_runner center-detection helper.

Offline-only (no hardware): the square-shaping runner locates the 0-order spot
center via ``_detect_center``, which shares its three methods with the CCD GUI
target-shape helper (``gui/ccd/target_shape_helper.py``):

- ``argmax``: peak position (default) — AGENTS.md: 0-order located by argmax,
  never by geometry (stray-light halo drags the plain centroid off-spot).
- ``centroid_thresh``: brightness centroid with threshold=0.1×max.
- ``centroid``: full-image intensity centroid (stray-light sensitive).
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.runners.gs_square_runner import _detect_center


def _gaussian_spot(
    peak: float, cx: float, cy: float, sigma: float = 8.0, size: int = 300
) -> np.ndarray:
    """轴对称高斯光斑 (uint16 相机量级)."""
    yy, xx = np.ogrid[:size, :size]
    return peak * np.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / sigma**2)


class TestDetectCenter:
    def test_argmax_returns_peak_pixel(self):
        frame = _gaussian_spot(200.0, 150.0, 120.0)
        cx, cy = _detect_center(frame, "argmax")
        assert (cx, cy) == (150.0, 120.0)

    def test_centroid_thresh_centers_symmetric_spot(self):
        # 对称光斑阈值质心应落在光斑中心 (~1px 内, 离散化误差)
        frame = _gaussian_spot(200.0, 150.0, 120.0)
        cx, cy = _detect_center(frame, "centroid_thresh")
        assert abs(cx - 150.0) < 1.0
        assert abs(cy - 120.0) < 1.0

    def test_centroid_matches_centroid_thresh_without_halo(self):
        # 无杂散光时两种质心应基本一致 (阈值只截掉低于 0.1×max 的尾巴)
        frame = _gaussian_spot(200.0, 150.0, 120.0)
        cx_c, cy_c = _detect_center(frame, "centroid")
        cx_t, cy_t = _detect_center(frame, "centroid_thresh")
        assert np.hypot(cx_c - cx_t, cy_c - cy_t) < 2.0

    def test_centroid_thresh_resists_stray_light_halo(self):
        # 弱而宽的杂散光晕 (实验实测会把全图质心拉偏数十~数百px):
        # 晕峰值 (15) 刻意低于截断阈值 (0.1×200=20) → 阈值质心应将晕完全
        # 排除并精确定位主光斑 (~0.1px), 而无阈值质心被晕拖离 ~66px。
        spot = _gaussian_spot(200.0, 150.0, 120.0)
        halo = _gaussian_spot(15.0, 60.0, 40.0, sigma=60.0)
        frame = spot + halo

        true_cx, true_cy = 150.0, 120.0
        cx_c, cy_c = _detect_center(frame, "centroid")
        cx_t, cy_t = _detect_center(frame, "centroid_thresh")

        dist_c = np.hypot(cx_c - true_cx, cy_c - true_cy)
        dist_t = np.hypot(cx_t - true_cx, cy_t - true_cy)
        # 阈值质心精确定位 (晕被截断排除)
        assert dist_t < 2.0
        # 无阈值质心确实被拖离光斑 (验证测试场景有效性 + AGENTS.md 警告物证)
        assert dist_c > 20.0

    def test_unknown_mode_raises(self):
        frame = np.zeros((50, 50))
        with pytest.raises(ValueError, match="Unknown center mode"):
            _detect_center(frame, "bogus")