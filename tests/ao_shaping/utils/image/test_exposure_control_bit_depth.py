"""Full scale in the exposure probe follows the camera's effective bit depth.

Issue #246: a MONO12 sensor delivers 12-bit samples inside uint16 containers,
so ``np.iinfo(uint16).max`` reads the saturation ceiling as 65535 instead of
4095.  A ladder judged against 65535 tops out at ~6 % of "full scale", the
probe never registers saturation, and the extrapolation it falls back to is
far above the bench's true ceiling.  ``probe_exposure_ceiling_ms`` and
``estimate_exposure_ceiling_ms`` therefore accept an optional duck-typed
``camera`` (``BaseCamera.effective_full_scale`` contract) and resolve every
rung through ``common.full_scale(frame, camera)``.

Offline only: the camera below is synthetic, deterministic, and never opened.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.utils.image.exposure_control import (
    estimate_exposure_ceiling_ms,
    probe_exposure_ceiling_ms,
)


class _Mono12Camera:
    """Duck-typed MONO12 camera: 12 effective bits in a uint16 container.

    Peak response is linear at ``counts_per_ms`` and clips at 4095, so with the
    default 2048 counts/ms the sensor saturates at ~2 ms of exposure.
    """

    bit_depth = 12

    def __init__(self, counts_per_ms: float = 2048.0) -> None:
        self.counts_per_ms = counts_per_ms
        self.current_ms = 0.0
        self.applied: list[float] = []

    def effective_full_scale(self, img: np.ndarray) -> float:
        return 4095.0

    def reset_exposure_time(self, ms: float) -> float:
        self.current_ms = float(ms)
        self.applied.append(self.current_ms)
        return self.current_ms

    def get_numpy_image(self, n_sample: int = 1) -> np.ndarray:
        peak = int(min(self.current_ms * self.counts_per_ms, 4095.0))
        frame = np.zeros((8, 8), dtype=np.uint16)
        frame[4, 4] = peak
        return frame


def _probe(cam: _Mono12Camera, **kwargs):
    return probe_exposure_ceiling_ms(
        cam.reset_exposure_time,
        lambda: np.asarray(cam.get_numpy_image(n_sample=1)),
        **kwargs,
    )


class TestProbeWithMono12Camera:
    def test_the_ladder_stops_at_the_effective_full_scale(self) -> None:
        """With the camera, 4095 counts *is* saturation and the walk stops there.

        The doubling ladder 0.02 ... 2.56 ms clips at the 2.56 ms rung
        (2.56 ms x 2048 counts/ms = 5242 -> 4095), whose fraction of the
        12-bit full scale is 1.0 >= 0.9.
        """
        cam = _Mono12Camera()
        estimate = _probe(cam, camera=cam)

        assert estimate.saturated_ms == pytest.approx(2.56)
        assert estimate.peak_fractions[-1] == pytest.approx(1.0)
        assert max(cam.applied) == pytest.approx(2.56)
        assert 0.0 < estimate.ceiling_ms <= 2.56

    def test_without_the_camera_the_uint16_container_misleads(self) -> None:
        """Pin the #246 failure mode: dtype alone reads full scale as 65535.

        Every rung stays at ~6 % of that, no rung reaches the 90 % stop, and
        the estimate carries no measured saturation point at all.
        """
        cam = _Mono12Camera()
        estimate = _probe(cam)  # camera omitted -> dtype fallback

        assert estimate.saturated_ms is None
        assert max(estimate.peak_fractions) == pytest.approx(4095 / 65535)
        # All eight rungs were walked because none ever looked saturated.
        assert len(cam.applied) == 8

    def test_the_dtype_only_ceiling_overshoots_the_measured_one(self) -> None:
        """The misread is not cosmetic: it hands back a far too high ceiling."""
        cam = _Mono12Camera()
        by_dtype = _probe(cam)
        cam.applied.clear()
        with_camera = _probe(cam, camera=cam)

        assert with_camera.ceiling_ms < by_dtype.ceiling_ms


class TestEstimateWithMono12Camera:
    def _ladder(self) -> list[tuple[float, np.ndarray]]:
        """Hand-built uint16 ladder: 1024, 2048, then clipped at 4095."""
        rungs = []
        for ms, peak in ((0.5, 1024), (1.0, 2048), (2.0, 4095)):
            frame = np.zeros((8, 8), dtype=np.uint16)
            frame[4, 4] = peak
            rungs.append((ms, frame))
        return rungs

    def test_peak_fractions_use_the_effective_bit_depth(self) -> None:
        cam = _Mono12Camera()
        estimate = estimate_exposure_ceiling_ms(self._ladder(), camera=cam)

        assert estimate.peak_fractions == pytest.approx(
            [1024 / 4095, 2048 / 4095, 1.0]
        )
        assert estimate.saturated_ms == pytest.approx(2.0)

    def test_camera_none_keeps_the_dtype_fallback(self) -> None:
        """Unchanged #246-free behavior: no camera means the dtype decides."""
        estimate = estimate_exposure_ceiling_ms(self._ladder())

        assert estimate.peak_fractions == pytest.approx(
            [1024 / 65535, 2048 / 65535, 4095 / 65535]
        )
        assert estimate.saturated_ms is None

    def test_uint8_ladder_is_untouched_by_the_threading(self) -> None:
        """An 8-bit bench reads the same numbers with and without a camera."""
        rungs = []
        for ms, peak in ((0.5, 128), (1.0, 255)):
            frame = np.zeros((8, 8), dtype=np.uint8)
            frame[4, 4] = peak
            rungs.append((ms, frame))

        class _Mono8Camera:
            def effective_full_scale(self, img: np.ndarray) -> float:
                return 255.0

        plain = estimate_exposure_ceiling_ms(rungs)
        with_camera = estimate_exposure_ceiling_ms(rungs, camera=_Mono8Camera())

        assert plain.peak_fractions == pytest.approx([128 / 255, 1.0])
        assert with_camera.peak_fractions == pytest.approx(plain.peak_fractions)
        assert with_camera.ceiling_ms == pytest.approx(plain.ceiling_ms)
