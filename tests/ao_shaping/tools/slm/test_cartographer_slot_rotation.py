"""F-1: the compensation loop must never pin an SLM memory slot.

Santec firmware treats ``display_memory(slot)`` as a **no-op** when ``slot`` is
already the displayed slot: the data is written, but the LCOS panel does not
refresh. ``compensate_once`` used to call
``apply_compensation(comp_gs, memory_slot=2)`` inside its loop, so iterations
2..N re-wrote slot 2 while slot 2 was still the one on screen. Every iteration
after the first therefore re-measured the same panel state -- a closed loop that
silently cannot make progress, reporting whatever the hardware noise produced.

The fix is to pass **no** ``memory_number`` and let ``display_data`` rotate
(``santec/driver.py:1230-1249``, which also skips the currently displayed slot so
the first write after ``open()`` cannot collide with another process's pattern).

These tests are pure/offline: ``DynamicCompensator`` receives its SLM and WFS by
injection, so fakes are enough -- no SDK, no device.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ao_shaping.tools.slm.cartographer.dynamic_compensation import (
    CompensationConfig,
    DynamicCompensator,
)


class _RecordingSLM:
    """Duck-typed Santec stand-in that records every ``display_data`` call."""

    is_open = True

    #: ``compensate_once`` builds the cosine probe pattern from the panel size.
    Panel_Res = (8, 8)

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.displayed = 2
        #: slot left on screen after each write, in order
        self.shown: list[int] = []

    def display_data(self, gray, wait_time_s=None, memory_number=None, memory_mode=None):
        # Emulate the firmware no-op the real driver warns about: a write to the
        # already-displayed slot does not change what the panel shows. When no
        # slot is pinned, rotate the way the driver does (and skip the displayed
        # one), so the test can assert the panel really did move.
        self.calls.append(
            {
                "gray": np.asarray(gray).copy(),
                "wait_time_s": wait_time_s,
                "memory_number": memory_number,
                "memory_mode": memory_mode,
            }
        )
        if memory_number is not None:
            self.displayed = int(memory_number)
        else:
            self.displayed = 3 if self.displayed == 2 else 2
        self.shown.append(self.displayed)
        return self.displayed


def _compensator(tmp_path: Path, slm: _RecordingSLM) -> DynamicCompensator:
    return DynamicCompensator(
        slm=slm,  # type: ignore[arg-type]
        wfs=object(),  # type: ignore[arg-type]
        config=CompensationConfig(n_correction_iterations=3),
        lut={0: 0.0, 255: 1.0},
        storage_dir=tmp_path / "storage",
    )


class TestApplyCompensationSlot:
    def test_never_pins_a_memory_slot(self, tmp_path: Path) -> None:
        """The regression itself: no ``memory_number`` reaches the driver."""
        slm = _RecordingSLM()
        comp = _compensator(tmp_path, slm)

        comp.apply_compensation(np.zeros((8, 8), dtype=np.uint16))

        assert len(slm.calls) == 1
        assert slm.calls[0]["memory_number"] is None, (
            "apply_compensation must not pass memory_number: the firmware treats a "
            "write to the displayed slot as a no-op and the panel never refreshes"
        )

    def test_does_not_pass_memory_mode_either(self, tmp_path: Path) -> None:
        """``memory_mode`` already defaults to MEMORY_MODE_INTERNAL at the driver.

        TODO F-1 claimed line 276 was "missing" it; passing it would only add a
        second source of truth for a value the driver already owns.
        """
        slm = _RecordingSLM()
        comp = _compensator(tmp_path, slm)

        comp.apply_compensation(np.zeros((8, 8), dtype=np.uint16))

        assert slm.calls[0]["memory_mode"] is None

    def test_returns_the_slot_the_driver_chose(self, tmp_path: Path) -> None:
        slm = _RecordingSLM()
        comp = _compensator(tmp_path, slm)

        slot = comp.apply_compensation(np.zeros((8, 8), dtype=np.uint16))

        assert isinstance(slot, int)
        assert slot == slm.displayed

    def test_refuses_to_write_to_a_closed_slm(self, tmp_path: Path) -> None:
        slm = _RecordingSLM()
        slm.is_open = False
        comp = _compensator(tmp_path, slm)

        with pytest.raises(AssertionError):
            comp.apply_compensation(np.zeros((8, 8), dtype=np.uint16))
        assert slm.calls == []


class TestCompensateOnceSlot:
    """The loop is where the bug actually lived: a pinned slot, N times."""

    def test_no_iteration_pins_a_slot(self, tmp_path: Path) -> None:
        slm = _RecordingSLM()
        comp = _compensator(tmp_path, slm)

        wf = np.linspace(-1.0, 1.0, 64).reshape(8, 8)
        # Never converging keeps all 3 iterations running, which is what makes
        # "iteration 2..N are no-ops" observable.
        comp.measure_initial_aberration = lambda: wf  # type: ignore[method-assign]
        comp.compute_compensation = lambda w: np.zeros((8, 8), dtype=np.uint16)  # type: ignore[method-assign]
        comp.verify_correction = lambda: (  # type: ignore[method-assign]
            wf,
            {"rms": 0.5},
        )
        comp.config.convergence_threshold_rms_waves = -1.0

        result = comp.compensate_once(max_iterations=3)

        assert len(slm.calls) == 3, "expected all 3 iterations to reach the SLM"
        for i, call in enumerate(slm.calls):
            assert call["memory_number"] is None, f"iteration {i + 1} pinned a slot"

    def test_reports_every_iteration(self, tmp_path: Path) -> None:
        """Guard the guard: if iterations were dropped, the slot check could pass vacuously."""
        slm = _RecordingSLM()
        comp = _compensator(tmp_path, slm)

        wf = np.linspace(-1.0, 1.0, 64).reshape(8, 8)
        comp.measure_initial_aberration = lambda: wf  # type: ignore[method-assign]
        comp.compute_compensation = lambda w: np.zeros((8, 8), dtype=np.uint16)  # type: ignore[method-assign]
        comp.verify_correction = lambda: (wf, {"rms": 0.5})  # type: ignore[method-assign]
        comp.config.convergence_threshold_rms_waves = -1.0

        result = comp.compensate_once(max_iterations=3)

        assert result.iterations_used == 3
        assert len(result.per_iteration_data) == 3

    def test_consecutive_writes_land_on_different_slots(self, tmp_path: Path) -> None:
        """The user-visible consequence: the panel actually changes pattern.

        With a pinned slot every write after the first is a firmware no-op, so the
        loop re-measures one unchanged panel. Asserting the slot *moves* is what
        ties the ``memory_number is None`` check back to that physical fact.
        """
        slm = _RecordingSLM()
        comp = _compensator(tmp_path, slm)

        wf = np.linspace(-1.0, 1.0, 64).reshape(8, 8)
        comp.measure_initial_aberration = lambda: wf  # type: ignore[method-assign]
        comp.compute_compensation = lambda w: np.zeros((8, 8), dtype=np.uint16)  # type: ignore[method-assign]
        comp.verify_correction = lambda: (wf, {"rms": 0.5})  # type: ignore[method-assign]
        comp.config.convergence_threshold_rms_waves = -1.0

        comp.compensate_once(max_iterations=3)

        assert [c["memory_number"] for c in slm.calls] == [None, None, None]
        assert len(slm.shown) == 3
        # consecutive writes must never land on the slot already on screen
        for previous, current in zip(slm.shown, slm.shown[1:], strict=False):
            assert current != previous, f"slot repeated: {slm.shown}"
