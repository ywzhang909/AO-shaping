"""Guard tests for the R-48 (#64) frame-analysis sink into ``utils/image``.

The 13 pure-array symbols moved from ``tools/kernels/bench_kernels`` to
``utils/image/frame_analysis``; ``bench_kernels`` re-exports them and
``model_in_loop_shaping`` aliases ``_spot_fwhm``. These tests pin the wiring so a
future "cleanup" cannot silently break the ~15 importers that still pull these
names from ``bench_kernels`` or import ``_spot_fwhm`` directly.

No hardware, no device objects: numpy on synthetic frames, so the file runs in CI.
"""

from __future__ import annotations

import numpy as np

from ao_shaping.tools.kernels import bench_kernels
from ao_shaping.utils.image import beam_metrics, frame_analysis

MOVED = (
    "SpotMeasurement",
    "core_fraction",
    "crop_around_zero_order",
    "crop_roi",
    "estimate_shift",
    "finite_clip",
    "finite_median_subtract",
    "flat_to_flat_floor",
    "measure_spot",
    "roi_l2",
    "smooth_frame",
)


def _blob() -> np.ndarray:
    yy, xx = np.mgrid[0:64, 0:64]
    return np.exp(-((xx - 32.0) ** 2 + (yy - 32.0) ** 2) / (2 * 4.0**2))


class TestReexportIdentity:
    """bench_kernels must hand out the *same* objects, not copies."""

    def test_moved_names_are_identical(self):
        for name in MOVED:
            assert getattr(bench_kernels, name) is getattr(frame_analysis, name), name

    def test_despike_is_single_source(self):
        assert bench_kernels.despike_frame is beam_metrics.despike_frame
        assert frame_analysis.despike_frame is beam_metrics.despike_frame

    def test_all_star_surface_intact(self):
        missing = [n for n in bench_kernels.__all__ if not hasattr(bench_kernels, n)]
        assert not missing, f"bench_kernels.__all__ names unresolved: {missing}"


class TestSpotFwhmAlias:
    """model_in_loop_shaping keeps its private name pointing at the sunk fn."""

    def test_alias_is_canonical(self):
        from ao_shaping.optimizer.wfless import model_in_loop_shaping as mil

        assert mil._spot_fwhm is frame_analysis.spot_fwhm


class TestSmokeBehaviour:
    """The moved functions still behave on synthetic frames."""

    def test_measure_spot_centres_blob(self):
        m = frame_analysis.measure_spot(_blob())
        assert abs(m.centroid_x - 32.0) < 1.0
        assert abs(m.centroid_y - 32.0) < 1.0
        assert m.fwhm_px > 0.0

    def test_estimate_shift_recovers_translation(self):
        img = _blob()
        dy, dx, _ = frame_analysis.estimate_shift(img, np.roll(img, 5, axis=1))
        assert abs(dx - 5.0) < 0.5
        assert abs(dy) < 0.5

    def test_crop_roi_pads_off_frame(self):
        out = frame_analysis.crop_roi(_blob(), (2, 2), 10)
        assert out.shape == (20, 20)
        assert out.min() >= 0.0


class TestMaxTargetSideAtZeroOrder:
    """``target_side == 0`` = 最大边长：0 级到最近帧边 − 留白。

    这条语义把「目标方斑必然落在帧内」变成**结构性质**，所以
    「目标边长 > 读出窗口边长」那类冲突在 0 下不可能出现。
    """

    def test_margin_constant_is_ten_px(self):
        assert frame_analysis.TARGET_EDGE_MARGIN_PX == 10

    def test_side_is_twice_the_radius_after_the_margin(self):
        # 100x100、中心 (50, 50)：到最近边 49 px，扣 10 留白 → 半径 39 → 边长 78
        frame = np.zeros((100, 100))
        assert frame_analysis.max_target_side_at_zero_order(frame, (50, 50)) == 78
        assert (
            frame_analysis.max_target_side_at_zero_order(frame, (50, 50), margin_px=0)
            == 98
        )

    def test_offcentre_spot_is_limited_by_its_nearest_edge(self):
        # (x=30, y=20) → 最近边是上边 (20 px)；扣 10 → 半径 10 → 边长 20
        frame = np.zeros((200, 200))
        assert frame_analysis.max_target_side_at_zero_order(frame, (30, 20)) == 20

    def test_center_defaults_to_the_zero_order_detector(self):
        blob = _blob()  # 64x64 高斯，峰在 (32, 32)
        auto = frame_analysis.max_target_side_at_zero_order(blob)
        explicit = frame_analysis.max_target_side_at_zero_order(blob, (32, 32))
        assert abs(auto - explicit) <= 2, (auto, explicit)

    def test_tiny_frame_falls_back_to_min_side(self):
        frame = np.zeros((4, 4))
        assert frame_analysis.max_target_side_at_zero_order(frame, (2, 2)) == 1
        assert (
            frame_analysis.max_target_side_at_zero_order(frame, (2, 2), min_side=8) == 8
        )

    def test_side_always_fits_the_frame(self):
        # 0 级落在任何位置（含四个角）边长都必须放得下
        for center in ((0, 0), (0, 63), (63, 0), (63, 63), (32, 32), (5, 40)):
            side = frame_analysis.max_target_side_at_zero_order(_blob(), center)
            assert 1 <= side <= 64, (center, side)
