"""Regression-anchor tests for :mod:`ao_shaping.utils.slm.phase_display`.

Pins the EXACT current behavior of the SLM phase->grayscale conversion and
memory-slot rotation helpers so future refactors of ``diff_beam_runner`` (which
imports these helpers) can be verified against a stable baseline.

All device interaction is exercised through a plain mock SLM object; no real
hardware is ever touched.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from ao_shaping.utils.slm import phase_display
from ao_shaping.utils.slm.phase_display import (
    DEFAULT_DISTANCE,
    DEFAULT_MAX_GRAYSCALE,
    DEFAULT_SLM_PIXEL_SIZE,
    DEFAULT_WAVELENGTH,
    display_phase,
    phase_to_slm_grayscale,
    pick_slm_slot,
)


class MockSLM:
    """Duck-typed stand-in for an open Santec SLM device (memory mode)."""

    def __init__(self, displayed_slot: int = 7, fail_query: bool = False) -> None:
        self.displayed_slot = displayed_slot
        self.fail_query = fail_query
        self.query_count = 0
        self.calls: list[tuple] = []

    def get_displayed_memory_number(self) -> int:
        self.query_count += 1
        if self.fail_query:
            raise RuntimeError("no device connected")
        return self.displayed_slot

    def create_phase_from_array(self, phase_rad: np.ndarray) -> str:
        self.calls.append(("create_phase_from_array", phase_rad))
        return "GRAY"

    def _write_phase(self, gray: str, memory_number: int | None = None) -> None:
        self.calls.append(("write_phase", gray, memory_number))

    def _display_memory(self, slot: int) -> None:
        self.calls.append(("display_memory", slot))

    def display_phase(
        self,
        phase_rad: np.ndarray,
        wait_time_s: float | None = None,
        memory_number: int | None = None,
        memory_mode: int = 0,
    ) -> None:
        gray = self.create_phase_from_array(phase_rad)
        slot = memory_number if memory_number is not None else 2
        self._write_phase(gray, memory_number=slot)
        self._display_memory(slot)


@pytest.fixture(autouse=True)
def _reset_slot_state() -> None:
    """Reset the module-global last-slot tracker before and after each test."""
    phase_display._last_slm_slot = None
    yield
    phase_display._last_slm_slot = None


@pytest.fixture
def first_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force ``random.choice`` to always return the first candidate."""

    def _first(seq: list[int]) -> int:
        return seq[0]

    monkeypatch.setattr(phase_display.random, "choice", _first)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_default_wavelength(self) -> None:
        assert DEFAULT_WAVELENGTH == 1064e-9

    def test_default_slm_pixel_size(self) -> None:
        assert DEFAULT_SLM_PIXEL_SIZE == 8e-6

    def test_default_distance(self) -> None:
        assert DEFAULT_DISTANCE == 0.1

    def test_default_max_grayscale(self) -> None:
        assert DEFAULT_MAX_GRAYSCALE == 1023


# ---------------------------------------------------------------------------
# phase_to_slm_grayscale
# ---------------------------------------------------------------------------


class TestPhaseToSlmGrayscale:
    def test_returns_uint16_with_preserved_shape(self) -> None:
        out = phase_to_slm_grayscale(np.zeros((4, 5)))
        assert out.dtype == np.uint16
        assert out.shape == (4, 5)

    def test_zero_phase_maps_to_zero(self) -> None:
        assert phase_to_slm_grayscale(np.array([[0.0]]))[0, 0] == 0

    def test_pi_maps_to_half_max_grayscale(self) -> None:
        # 0.5 * 1023 = 511.5, truncated to uint16 -> 511
        assert phase_to_slm_grayscale(np.array([[np.pi]]))[0, 0] == 511

    def test_pi_over_two_maps_to_quarter(self) -> None:
        # 0.25 * 1023 = 255.75, truncated -> 255
        assert phase_to_slm_grayscale(np.array([[np.pi / 2]]))[0, 0] == 255

    def test_three_pi_over_two_maps_to_three_quarters(self) -> None:
        # 0.75 * 1023 = 767.25, truncated -> 767
        assert phase_to_slm_grayscale(np.array([[3 * np.pi / 2]]))[0, 0] == 767

    def test_two_pi_wraps_to_zero(self) -> None:
        assert phase_to_slm_grayscale(np.array([[2 * np.pi]]))[0, 0] == 0

    def test_phase_beyond_two_pi_wraps(self) -> None:
        # 3*pi wraps to pi -> 511
        assert phase_to_slm_grayscale(np.array([[3 * np.pi]]))[0, 0] == 511

    def test_negative_phase_wraps_into_range(self) -> None:
        # -pi/2 wraps to 3*pi/2 -> 767
        assert phase_to_slm_grayscale(np.array([[-np.pi / 2]]))[0, 0] == 767

    def test_accepts_plain_list_input(self) -> None:
        out = phase_to_slm_grayscale([0.0, np.pi])
        assert out.dtype == np.uint16
        assert out.tolist() == [0, 511]

    def test_accepts_float64_input(self) -> None:
        assert (
            phase_to_slm_grayscale(np.array([[np.pi]], dtype=np.float64))[0, 0] == 511
        )

    def test_custom_max_grayscale(self) -> None:
        # 0.5 * 255 = 127.5, truncated -> 127
        out = phase_to_slm_grayscale(np.array([[np.pi]]), max_grayscale=255)
        assert out[0, 0] == 127
        assert out.dtype == np.uint16

    def test_output_values_never_exceed_max_grayscale(self) -> None:
        phases = np.linspace(0.0, 4 * np.pi, 1000).reshape(50, 20)
        out = phase_to_slm_grayscale(phases)
        assert out.min() >= 0
        assert out.max() <= DEFAULT_MAX_GRAYSCALE


# ---------------------------------------------------------------------------
# pick_slm_slot
# ---------------------------------------------------------------------------


class TestPickSlmSlot:
    def test_returns_slot_within_range(self, first_candidate: None) -> None:
        slot = pick_slm_slot(MockSLM(displayed_slot=7))
        assert phase_display._SLOT_MIN <= slot <= phase_display._SLOT_MAX

    def test_first_call_excludes_currently_displayed_slot(
        self, first_candidate: None
    ) -> None:
        slm = MockSLM(displayed_slot=7)
        slot = pick_slm_slot(slm)
        assert slot != 7
        # With choice pinned to the first candidate, slot 2 is returned.
        assert slot == 2

    def test_device_queried_only_on_first_call(self, first_candidate: None) -> None:
        slm = MockSLM(displayed_slot=7)
        pick_slm_slot(slm)
        pick_slm_slot(slm)
        pick_slm_slot(slm)
        assert slm.query_count == 1

    def test_consecutive_calls_never_return_same_slot(
        self, first_candidate: None
    ) -> None:
        slm = MockSLM(displayed_slot=7)
        slots = [pick_slm_slot(slm) for _ in range(10)]
        assert all(a != b for a, b in zip(slots, slots[1:]))

    def test_works_without_device_query(self, first_candidate: None) -> None:
        # get_displayed_memory_number raises -> falls back to full range.
        slm = MockSLM(fail_query=True)
        slot = pick_slm_slot(slm)
        assert phase_display._SLOT_MIN <= slot <= phase_display._SLOT_MAX
        assert slot == 2

    def test_out_of_range_displayed_slot_still_yields_valid_slot(
        self, first_candidate: None
    ) -> None:
        # Device reports slot 200 (outside 2..125): nothing is excluded.
        slm = MockSLM(displayed_slot=200)
        slot = pick_slm_slot(slm)
        assert phase_display._SLOT_MIN <= slot <= phase_display._SLOT_MAX

    def test_updates_module_last_slot_state(self, first_candidate: None) -> None:
        slm = MockSLM(displayed_slot=7)
        slot = pick_slm_slot(slm)
        assert phase_display._last_slm_slot == slot

    def test_many_calls_never_repeat_and_stay_in_range(self, monkeypatch) -> None:
        rng = random.Random(42)
        monkeypatch.setattr(phase_display.random, "choice", rng.choice)
        slm = MockSLM(displayed_slot=5)
        slots = [pick_slm_slot(slm) for _ in range(200)]
        assert all(
            phase_display._SLOT_MIN <= s <= phase_display._SLOT_MAX for s in slots
        )
        assert all(a != b for a, b in zip(slots, slots[1:]))
        assert slots[0] != 5


# ---------------------------------------------------------------------------
# display_phase
# ---------------------------------------------------------------------------


class TestDisplayPhase:
    def test_full_call_sequence(self, first_candidate: None, monkeypatch) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr(phase_display.time, "sleep", sleeps.append)
        slm = MockSLM(displayed_slot=3)
        phase = np.zeros((2, 2))

        display_phase(slm, phase, settle_time_s=0.7)

        # create_phase_from_array receives the radian phase unchanged.
        assert slm.calls[0][0] == "create_phase_from_array"
        np.testing.assert_array_equal(slm.calls[0][1], phase)
        # write_phase receives the converted gray plus a memory slot.
        assert slm.calls[1] == ("write_phase", "GRAY", 2)
        # display_memory receives the same slot.
        assert slm.calls[2] == ("display_memory", 2)
        # Fixed settle time only (0.05s settle is now internal to display_phase)
        assert sleeps == [0.7]

    def test_slot_used_is_within_range(
        self, first_candidate: None, monkeypatch
    ) -> None:
        monkeypatch.setattr(phase_display.time, "sleep", lambda _t: None)
        slm = MockSLM(displayed_slot=3)
        display_phase(slm, np.zeros((2, 2)), settle_time_s=0.1)
        slot = slm.calls[1][2]
        assert phase_display._SLOT_MIN <= slot <= phase_display._SLOT_MAX
        assert slm.calls[2][1] == slot

    def test_uses_different_slot_on_consecutive_displays(
        self, first_candidate: None, monkeypatch
    ) -> None:
        monkeypatch.setattr(phase_display.time, "sleep", lambda _t: None)
        slm = MockSLM(displayed_slot=3)
        display_phase(slm, np.zeros((2, 2)), settle_time_s=0.1)
        display_phase(slm, np.zeros((2, 2)), settle_time_s=0.1)
        slot1 = slm.calls[1][2]
        slot2 = slm.calls[4][2]
        assert slot1 != slot2
        assert phase_display._SLOT_MIN <= slot1 <= phase_display._SLOT_MAX
        assert phase_display._SLOT_MIN <= slot2 <= phase_display._SLOT_MAX
