"""``utils.image.zero_order``: 零级中心四种口径的选择层 (默认椭圆中心)。

基础原语 (``despike_frame`` / ``locate_peak``) 由 ``beam_metrics`` 提供, 这里
只验证选择与组合: ``argmax`` 与 ``zero_order_center(refine=False)`` 等价,
四种口径都落在合成高斯的解析中心, 框内质心不被框外亮斑拉偏。
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.utils.image.beam_metrics import zero_order_center
from ao_shaping.utils.image.zero_order import zero_order_center_by_method


class TestZeroOrderCenterByMethod:
    """``zero_order_center_by_method``: 四种口径 + 默认椭圆中心。"""

    @staticmethod
    def _gaussian(
        height: int = 120,
        width: int = 140,
        cx: float = 71.3,
        cy: float = 54.6,
        sigma_x: float = 9.0,
        sigma_y: float = 5.0,
        background: float = 4.0,
    ) -> np.ndarray:
        yy, xx = np.mgrid[:height, :width].astype(np.float64)
        beam = 200.0 * np.exp(
            -((xx - cx) ** 2) / (2.0 * sigma_x**2)
            - ((yy - cy) ** 2) / (2.0 * sigma_y**2)
        )
        return background + beam

    def test_argmax_equals_zero_order_center_refine_false(self):
        frame = np.zeros((40, 40))
        frame[20, 25] = 100.0
        frame[20, 28] = 80.0
        assert zero_order_center_by_method(frame, "argmax") == (25, 20)
        assert zero_order_center_by_method(frame, "argmax") == zero_order_center(
            frame, refine=False
        )

    def test_four_methods_near_analytic_centre(self):
        frame = self._gaussian()
        for method in ("argmax", "centroid", "box_centroid", "ellipse"):
            x, y = zero_order_center_by_method(frame, method)
            assert abs(x - 71.3) <= 3, method
            assert abs(y - 54.6) <= 3, method

    def test_box_centroid_ignores_blob_outside_half_max(self):
        # 亮斑远在光斑外且低于半高: 它拉伸全画幅质心, 但不进半高连通域,
        # 因此框内质心仍钉在光束上 (这正是该口径存在的理由)。
        frame = self._gaussian()
        frame[5:15, 5:15] += 60.0
        x_full, _ = zero_order_center_by_method(frame, "centroid")
        x_box, _ = zero_order_center_by_method(frame, "box_centroid")
        assert abs(x_box - 71.3) <= 3
        assert abs(x_box - x_full) >= 2

    def test_all_dark_returns_frame_centre(self):
        # (w // 2, h // 2)
        assert zero_order_center_by_method(np.zeros((40, 80))) == (40, 20)

    def test_unknown_method_raises(self):
        frame = self._gaussian()
        with pytest.raises(ValueError, match="未知"):
            zero_order_center_by_method(frame, "nope")

    def test_default_is_ellipse(self):
        frame = self._gaussian()
        assert zero_order_center_by_method(frame) == zero_order_center_by_method(
            frame, "ellipse"
        )
