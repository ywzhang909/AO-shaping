"""Regression-anchor tests for the auto-find camera utilities.

Pins the exact current behavior of :func:`find_exposure_ms`,
:func:`find_zero_order_center` and :func:`auto_find_exposure_and_center`
from :mod:`ao_shaping.utils.hardware_utils` (the deterministic-probe
auto-exposure / 0-order-centre helpers shared by the SLM+CCD runners).

All tests are mock-based: a duck-typed ``_StubCam`` stands in for the real
camera (``reset_exposure_time`` + ``get_numpy_image`` only) so no hardware
or native SDK is touched. These are read-only anchors describing what the
source does today.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.utils.image.hardware_utils import (
    _DEFAULT_EXPOSURE_PROBES_MS,
    auto_find_exposure_and_center,
    find_exposure_ms,
    find_zero_order_center,
    resolve_spot_center,
)


class _StubCam:
    """Duck-typed camera stub: exposure-scaled gaussian spot at a fixed center.

    ``get_numpy_image`` returns a frame whose global max sits at
    ``spot_center`` and whose peak scales linearly with the current exposure
    (``peak = peak_per_ms * exposure_ms``). ``dark_below_exposure`` makes the
    spot vanish (all-dark frame) until the exposure reaches the threshold,
    which mirrors the real camera being dark at the smallest probes.
    """

    def __init__(
        self,
        spot_center: tuple[int, int] = (60, 40),
        peak_per_ms: float = 100.0,
        dark_below_exposure: float | None = None,
        shape: tuple[int, int] = (80, 120),
    ) -> None:
        self.spot_center = spot_center
        self.peak_per_ms = peak_per_ms
        self.dark_below_exposure = dark_below_exposure
        self.shape = shape
        self.exposure_ms = 0.0
        self.reset_calls: list[float] = []
        self.grab_calls = 0

    def reset_exposure_time(self, ms: float) -> float:
        self.exposure_ms = float(ms)
        self.reset_calls.append(float(ms))
        return self.exposure_ms

    def get_numpy_image(self, n_sample: int = 1, skip_first: bool = True) -> np.ndarray:
        self.grab_calls += 1
        h, w = self.shape
        cx, cy = self.spot_center
        y, x = np.mgrid[0:h, 0:w]
        spot = np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * 3.0**2)))
        amp = self.peak_per_ms * self.exposure_ms
        if (
            self.dark_below_exposure is not None
            and self.exposure_ms < self.dark_below_exposure
        ):
            amp = 0.0
        return amp * spot


class _HotPixelCam(_StubCam):
    """Adds a corner hot pixel on selected 1-based grab calls.

    Used to exercise the hot-pixel guard in ``find_exposure_ms`` and the
    median robustness of ``find_zero_order_center``.
    """

    def __init__(
        self,
        hot_grabs: set[int] | None = None,
        hot_peak: float = 300.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hot_grabs = set(hot_grabs or ())
        self.hot_peak = hot_peak

    def get_numpy_image(self, n_sample: int = 1, skip_first: bool = True) -> np.ndarray:
        frame = super().get_numpy_image(n_sample=n_sample, skip_first=skip_first)
        if self.grab_calls in self.hot_grabs:
            frame = frame.copy()
            frame[0, 0] = self.hot_peak
        return frame


class TestFindExposureMs:
    def test_requires_reset_exposure_time(self):
        class NoExposureCam:
            def get_numpy_image(
                self, n_sample: int = 1, skip_first: bool = True
            ) -> np.ndarray:
                return np.zeros((10, 10))

        with pytest.raises(ValueError, match="reset_exposure_time"):
            find_exposure_ms(NoExposureCam())

    def test_returns_first_probe_in_target_band(self):
        # peaks 50, 100, 200, 300 -> 1.0ms (peak 100) is the first in [100, 245].
        cam = _StubCam(peak_per_ms=100.0)
        ms = find_exposure_ms(cam, probe_exposures_ms=(0.5, 1.0, 2.0, 3.0))
        assert ms == 1.0

    def test_probes_sorted_ascending(self):
        cam = _StubCam(peak_per_ms=100.0)
        ms = find_exposure_ms(cam, probe_exposures_ms=(3.0, 1.0, 0.5, 2.0))
        assert ms == 1.0
        assert cam.reset_calls == [0.5, 1.0, 2.0, 3.0]

    def test_default_probes_used_when_none(self):
        cam = _StubCam(peak_per_ms=100.0)
        ms = find_exposure_ms(cam)
        assert ms == 1.0  # default probes: 1.0ms -> peak 100
        assert cam.reset_calls == list(_DEFAULT_EXPOSURE_PROBES_MS)

    def test_default_probes_constant(self):
        assert _DEFAULT_EXPOSURE_PROBES_MS == (
            0.05,
            0.1,
            0.2,
            0.5,
            1.0,
            2.0,
            3.0,
            5.0,
            10.0,
        )

    def test_hot_pixel_probe_discarded(self):
        # Grab 1 (0.5ms) is a hot pixel (300 > 245) immediately followed by a
        # probe with peak 50 (< 100) -> physically impossible, discarded.
        cam = _HotPixelCam(hot_grabs={1}, hot_peak=300.0, peak_per_ms=50.0)
        ms = find_exposure_ms(cam, probe_exposures_ms=(0.5, 1.0, 2.0))
        assert ms == 2.0  # 2.0ms -> peak 100 in band

    def test_empty_probe_sequence_falls_back_to_defaults(self):
        # An empty/None probe list is falsy -> the module defaults are used.
        cam = _StubCam(peak_per_ms=100.0)
        ms = find_exposure_ms(cam, probe_exposures_ms=[])
        assert ms == 1.0
        assert cam.reset_calls == list(_DEFAULT_EXPOSURE_PROBES_MS)

    def test_branch2_linear_extrapolation(self):
        # peaks 20, 40, 80 all below the floor -> est = 2.0 * (160/80) = 4.0.
        cam = _StubCam(peak_per_ms=40.0)
        ms = find_exposure_ms(cam, probe_exposures_ms=(0.5, 1.0, 2.0))
        assert ms == pytest.approx(4.0)
        assert cam.exposure_ms == pytest.approx(4.0)

    def test_branch2_extrapolation_clamped_to_1000ms(self):
        # last peak 0.2 -> est = 2.0 * (160/0.2) = 1600 -> clamped to 1000.
        cam = _StubCam(peak_per_ms=0.1)
        ms = find_exposure_ms(cam, probe_exposures_ms=(0.5, 1.0, 2.0))
        assert ms == 1000.0

    def test_branch2_all_dark_raises(self):
        cam = _StubCam(peak_per_ms=100.0, dark_below_exposure=10.0)
        with pytest.raises(ValueError, match="全暗"):
            find_exposure_ms(cam, probe_exposures_ms=(0.5, 1.0, 2.0))

    def test_branch3_lowest_probe_saturated(self):
        # peaks 300, 600, 1200 all > 245 -> down-probe from the minimum:
        # est = 0.5 * (160/300) = 0.2667.
        cam = _StubCam(peak_per_ms=600.0)
        ms = find_exposure_ms(cam, probe_exposures_ms=(0.5, 1.0, 2.0))
        expected = 0.5 * 160.0 / 300.0
        assert ms == pytest.approx(expected)
        assert cam.exposure_ms == pytest.approx(expected)

    def test_branch3_down_probe_clamped_to_0_02ms(self):
        # est = 0.5 * (160/50000) = 0.0016 -> max(min(0.0016, 0.5), 0.02) = 0.02.
        cam = _StubCam(peak_per_ms=100000.0)
        ms = find_exposure_ms(cam, probe_exposures_ms=(0.5, 1.0))
        assert ms == 0.02


class TestFindZeroOrderCenter:
    def test_returns_median_argmax_center(self):
        cam = _StubCam(spot_center=(60, 40), peak_per_ms=100.0)
        cam.exposure_ms = 5.0
        center = find_zero_order_center(cam, n_frames=5)
        assert center == (60, 40)
        assert cam.grab_calls == 5

    def test_n_frames_less_than_one_raises(self):
        cam = _StubCam()
        with pytest.raises(ValueError, match="n_frames"):
            find_zero_order_center(cam, n_frames=0)

    def test_does_not_change_exposure(self):
        cam = _StubCam(spot_center=(60, 40))
        cam.exposure_ms = 5.0
        find_zero_order_center(cam, n_frames=3)
        assert cam.exposure_ms == 5.0
        assert cam.reset_calls == []

    def test_robust_to_hot_pixel_frames(self):
        # Frames 1 and 3 carry a corner hot pixel (1000 > gaussian peak 500);
        # the per-axis median still lands on the true spot.
        cam = _HotPixelCam(
            hot_grabs={1, 3},
            hot_peak=1000.0,
            spot_center=(60, 40),
            peak_per_ms=100.0,
        )
        cam.exposure_ms = 5.0
        center = find_zero_order_center(cam, n_frames=5)
        assert center == (60, 40)


class TestAutoFindExposureAndCenter:
    def test_finds_exposure_then_center(self):
        cam = _StubCam(spot_center=(60, 40), peak_per_ms=100.0)
        exposure, center = auto_find_exposure_and_center(
            cam, probe_exposures_ms=(0.5, 1.0, 2.0)
        )
        assert exposure == 1.0
        assert center == (60, 40)

    def test_find_exposure_disabled(self):
        cam = _StubCam(spot_center=(60, 40), peak_per_ms=100.0)
        cam.exposure_ms = 5.0
        exposure, center = auto_find_exposure_and_center(
            cam, find_exposure=False, probe_exposures_ms=(0.5, 1.0, 2.0)
        )
        assert exposure is None
        assert center == (60, 40)
        assert cam.reset_calls == []

    def test_find_center_disabled(self):
        cam = _StubCam(spot_center=(60, 40), peak_per_ms=100.0)
        exposure, center = auto_find_exposure_and_center(
            cam, find_center=False, probe_exposures_ms=(0.5, 1.0, 2.0)
        )
        assert exposure == 1.0
        assert center is None

    def test_both_disabled(self):
        cam = _StubCam(spot_center=(60, 40), peak_per_ms=100.0)
        exposure, center = auto_find_exposure_and_center(
            cam,
            find_exposure=False,
            find_center=False,
            probe_exposures_ms=(0.5, 1.0, 2.0),
        )
        assert exposure is None
        assert center is None
        assert cam.grab_calls == 0


class TestResolveSpotCenter:
    def test_none_uses_detect_fn(self):
        cam = _StubCam(spot_center=(60, 40), peak_per_ms=100.0)
        cam.exposure_ms = 5.0
        img = cam.get_numpy_image(n_sample=1)
        center = resolve_spot_center(cam, img, None, recapture=False)
        assert center == (60, 40)

    def test_explicit_tuple_returned_as_is(self):
        cam = _StubCam(spot_center=(60, 40))
        img = np.zeros((80, 120))
        center = resolve_spot_center(cam, img, (30, 25), recapture=False)
        assert center == (30, 25)

    def test_string_mode_max(self):
        cam = _StubCam(spot_center=(60, 40), peak_per_ms=100.0)
        cam.exposure_ms = 5.0
        img = cam.get_numpy_image(n_sample=1)
        center = resolve_spot_center(cam, img, "max")
        assert center == (60, 40)

    def test_string_mode_shape(self):
        cam = _StubCam(spot_center=(60, 40), peak_per_ms=100.0)
        cam.exposure_ms = 5.0
        center = resolve_spot_center(cam, np.zeros((80, 120)), "shape")
        assert center == (60, 40)

    def test_unknown_string_raises(self):
        cam = _StubCam()
        with pytest.raises(ValueError, match="known center"):
            resolve_spot_center(cam, np.zeros((10, 10)), "bogus")

    def test_returns_int_tuple(self):
        cam = _StubCam(spot_center=(60, 40), peak_per_ms=100.0)
        cam.exposure_ms = 5.0
        img = cam.get_numpy_image(n_sample=1)
        center = resolve_spot_center(cam, img, None, recapture=False)
        assert isinstance(center, tuple)
        assert all(isinstance(v, int) for v in center)
