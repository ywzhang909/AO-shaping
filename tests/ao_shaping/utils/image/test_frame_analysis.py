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
