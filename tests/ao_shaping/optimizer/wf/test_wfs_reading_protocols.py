"""Pins the two WFS reading protocols in ``optimizer/wf``, which are not interchangeable.

``dm_response_matrix`` issues one acquisition per reading (``n_sample=1``) so a per-channel
truncated mean can discard a bad frame; ``interaction_matrix`` issues a single
``take_image(n_sample=n_averages)`` and reads one deviation pair, leaving ``cancel_tile`` at
its default. Both mean "average N readings" in prose. The acquisition traces below are what
make them look different to a reader, and what fails if someone collapses them into one.
"""

from __future__ import annotations

import numpy as np

from ao_shaping.optimizer.wf.dm_response_matrix import _measure_averaged_slopes
from ao_shaping.optimizer.wf.interaction_matrix import _single_shot_slopes
from ao_shaping.utils.wavefront.wfs_utils import flatten_slopes, truncated_mean


class _ProtocolWFS:
    """Records the acquisition protocol and hands back deterministic readings."""

    def __init__(self, n_readings: int) -> None:
        rng = np.random.default_rng(0)
        self.readings_x = [rng.normal(size=(3, 4)) for _ in range(n_readings)]
        self.readings_y = [rng.normal(size=(3, 4)) for _ in range(n_readings)]
        self.take_calls: list[int] = []
        self.cancel_flags: list[bool] = []
        self._next = 0

    def take_image(self, n_sample: int = 10, dynamicNoiseCut: bool = True) -> None:
        self.take_calls.append(n_sample)

    def get_spot_deviation(self, cancel_tile: bool = False):
        self.cancel_flags.append(cancel_tile)
        x = self.readings_x[self._next]
        y = self.readings_y[self._next]
        self._next += 1
        return x, y


class TestPerReadingProtocol:
    def test_one_acquisition_per_reading_with_cancel_tile_forwarded(self) -> None:
        wfs = _ProtocolWFS(7)
        _measure_averaged_slopes(wfs, n_averages=7, cancel_tile=True)
        assert wfs.take_calls == [1] * 7
        assert wfs.cancel_flags == [True] * 7

    def test_values_match_stacking_the_readings_then_truncating(self) -> None:
        wfs = _ProtocolWFS(8)
        slopes, mean_x, mean_y = _measure_averaged_slopes(
            wfs, n_averages=8, cancel_tile=False
        )
        np.testing.assert_array_equal(mean_x, truncated_mean(np.array(wfs.readings_x)))
        np.testing.assert_array_equal(mean_y, truncated_mean(np.array(wfs.readings_y)))
        np.testing.assert_array_equal(slopes, flatten_slopes(mean_x, mean_y))

    def test_gate_boundary_is_visible_from_the_caller(self) -> None:
        for n in (5, 6):
            wfs = _ProtocolWFS(n)
            _slopes, mean_x, _mean_y = _measure_averaged_slopes(
                wfs, n_averages=n, cancel_tile=False
            )
            stacked = np.array(wfs.readings_x)
            want = (
                np.mean(stacked, axis=0)
                if n <= 5
                else np.mean(np.sort(stacked, axis=0)[1:-1], axis=0)
            )
            np.testing.assert_array_equal(mean_x, want)


class TestSingleShotProtocol:
    def test_exactly_one_acquisition_sized_to_the_average_count(self) -> None:
        wfs = _ProtocolWFS(1)
        _single_shot_slopes(wfs, n_averages=9)
        assert wfs.take_calls == [9]
        assert wfs.cancel_flags == [False]

    def test_the_one_reading_is_returned_without_truncation(self) -> None:
        wfs = _ProtocolWFS(1)
        slopes = _single_shot_slopes(wfs, n_averages=4)
        np.testing.assert_array_equal(
            slopes, flatten_slopes(wfs.readings_x[0], wfs.readings_y[0])
        )


class TestTheProtocolsDiffer:
    def test_same_request_produces_different_acquisition_traces(self) -> None:
        per_reading = _ProtocolWFS(6)
        single_shot = _ProtocolWFS(6)
        _measure_averaged_slopes(per_reading, n_averages=6, cancel_tile=False)
        _single_shot_slopes(single_shot, n_averages=6)
        assert per_reading.take_calls != single_shot.take_calls

    def test_only_the_per_reading_path_can_discard_a_bad_frame(self) -> None:
        # A corrupt frame moves one channel by a lot. The per-reading protocol sees it as
        # a separable row and truncates it away; the single-shot protocol cannot, because
        # the driver has already mixed the frames into one reading.
        wfs = _ProtocolWFS(6)
        wfs.readings_x[3] = wfs.readings_x[3] + 50.0
        _slopes, mean_x, _mean_y = _measure_averaged_slopes(
            wfs, n_averages=6, cancel_tile=False
        )
        assert np.max(np.abs(mean_x)) < 20.0
