"""Two ``SlotRotator`` classes exist on purpose -- do not "converge" them.

Issue #52 asked for the three SLM memory-slot rotation spellings to be collapsed
into one. Measured before acting, that instruction is not safe to execute:

* ``drivers/slm/santec/driver.SlotRotator`` picks **at random** from
  ``[SLOT_MIN, SLOT_MAX]`` while *excluding the slot the device is currently
  showing*, which it reads from the panel via ``get_displayed_memory_number()``.
  That exclusion is what makes it survive a process restart.
* ``optimizer/wfless/slm_zernike_primitives.SlotRotator`` **increments
  sequentially** and holds no device reference at all, so it stays deterministic.

Two probes writing the same panel in one process must not interleave, so each
consumer owns an instance -- that part is intentional and already documented in
the wfless module docstring. The policies themselves are different, and unifying
them would silently change which physical slot a bench writes to. The issue's own
section 2.3 warns against exactly this: "只动 CLI 层, 不合并命令体" and
"会产生看似可信的错数据".

The third spelling -- ``tools/slm/calibration/phase_capture.py`` computing
``(memory_slot + i - 1) % 128 + 1`` -- deliberately covers slots 1..128, whereas
every other path uses 2..125. That range is a bench fact, not a typo, so it is
left alone here for the same reason: changing which physical slot gets written
cannot be validated without the hardware.

So this file pins the difference instead. Without it, "there are two classes with
the same name" reads as an obvious duplication, and the next sweep would merge
them.
"""

from __future__ import annotations

import inspect

from ao_shaping.drivers.slm.santec import SLOT_MAX, SLOT_MIN
from ao_shaping.drivers.slm.santec.driver import SlotRotator as DriverSlotRotator
from ao_shaping.optimizer.wfless.slm_zernike_primitives import (
    SlotRotator as PrimitiveSlotRotator,
)

#: The canonical rotation range, from the driver's rationale comment: consecutive
#: writes must not target the slot already on screen, because ``display_memory``
#: on the displayed slot is a firmware no-op.
CANONICAL_RANGE = (SLOT_MIN, SLOT_MAX)


def test_the_two_rotators_are_distinct_implementations() -> None:
    """Same name, same purpose, different policy -- so not a duplication."""
    assert DriverSlotRotator is not PrimitiveSlotRotator
    assert DriverSlotRotator.__module__ != PrimitiveSlotRotator.__module__


def test_primitive_rotator_is_deterministic_and_sequential() -> None:
    """It holds no device and no randomness, so a run is reproducible."""
    rotator = PrimitiveSlotRotator()
    slots = [rotator.next() for _ in range(5)]

    assert slots == [2, 3, 4, 5, 6], f"expected a sequential walk from slot_min, got {slots}"
    assert all(rotator.slot_min <= s <= rotator.slot_max for s in slots)


def test_primitive_rotator_wraps_within_range_and_never_repeats() -> None:
    """The invariant that actually matters: no two consecutive writes share a slot."""
    rotator = PrimitiveSlotRotator(slot_min=2, slot_max=4)
    slots = [rotator.next() for _ in range(6)]

    assert slots == [2, 3, 4, 2, 3, 4]
    assert all(a != b for a, b in zip(slots, slots[1:])), f"consecutive repeat: {slots}"
    assert min(slots) == CANONICAL_RANGE[0] and max(slots) <= CANONICAL_RANGE[1]


def test_primitive_rotator_rejects_an_inverted_range() -> None:
    """A min above max would otherwise produce a silently wrong slot."""
    import pytest

    with pytest.raises(ValueError, match="slot_min"):
        PrimitiveSlotRotator(slot_min=10, slot_max=2)


def test_driver_rotator_reads_the_slot_the_panel_is_showing() -> None:
    """Cross-process safety comes from the device read, not from local state.

    A rotator that only remembered its own last write would happily re-pick that
    slot after a restart, because the panel's displayed slot is not in memory.
    """
    src = inspect.getsource(DriverSlotRotator.__init__)
    assert "read_current_slot" in src, (
        "DriverSlotRotator must seed itself from the panel's displayed slot; "
        "without that read, a restart can repeat the on-screen slot, which the "
        "firmware treats as a no-op"
    )


def test_driver_rotator_is_random_not_sequential() -> None:
    """Pin the other half of the difference: the driver side picks at random."""
    sig = inspect.signature(DriverSlotRotator.__init__)
    assert "choice" in sig.parameters, "the driver rotator's randomness hook disappeared"

    # Deterministic stand-in for random.choice so the assertion is stable.
    rotator = DriverSlotRotator(
        slm=object(),
        slot_min=2,
        slot_max=3,
        choice=lambda candidates: candidates[-1],
    )
    first = rotator.next_slot()
    second = rotator.next_slot()

    assert {first, second} == {2, 3}
    assert first != second, "a repeat would be the firmware no-op this class exists to avoid"
