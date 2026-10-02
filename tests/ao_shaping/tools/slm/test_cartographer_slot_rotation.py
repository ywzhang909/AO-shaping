"""SLM memory-slot rotation guard for the cartographer's dynamic compensation.

Bench rule (README "四条台架铁律" #3): pinning ``memory_number=`` is a firmware
NO-OP — ``display_memory(slot)`` does not refresh the LCOS panel when that slot
is already displayed, so every frame afterwards is a stale picture.

``DynamicCompensator.apply_compensation`` used to hardcode slot 2 inside the
``for i in range(max_iter)`` loop, so from iteration 2 onward the loop re-measured
the identical panel state while recording a fabricated ``improvement_rms``.

These tests are fully hardware-free: ``DynamicCompensator`` receives the SLM as
a constructor parameter, and the driver-side rotation is exercised on a bare
``Santec`` instance built with ``object.__new__`` (no ``open()``, no SDK load).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ao_shaping.drivers.slm.santec import MAX_MEM_SLOTS, Santec, VideoMode
from ao_shaping.tools.slm.cartographer import dynamic_compensation
from ao_shaping.tools.slm.cartographer.dynamic_compensation import (
    CompensationConfig,
    DynamicCompensator,
)


class _RecordingSlm:
    """Fake SLM recording the ``memory_number`` of every ``display_data`` call.

    Mirrors the real signature ``Santec.display_data(phase_gray, wait_time_s=None,
    memory_number=None, memory_mode=...)`` closely enough for the compensator.
    """

    Panel_Res = (1920, 1200)

    def __init__(self) -> None:
        self.is_open = True
        self.calls: list[int | None] = []

    def display_data(
        self,
        phase_gray: np.ndarray,
        wait_time_s: float | None = None,
        memory_number: int | None = None,
        memory_mode: int = 0,
    ) -> int | None:
        self.calls.append(memory_number)
        return memory_number


class _StubWfs:
    """Minimal WFS stub — ``apply_compensation`` never touches it."""

    def is_connected(self) -> bool:
        return True


def _make_compensator(slm: Any, storage_dir: Path) -> DynamicCompensator:
    return DynamicCompensator(
        slm=slm,
        wfs=_StubWfs(),
        config=CompensationConfig(),
        lut={0: 0.0, 512: 3.14159, 1023: 6.28318},
        storage_dir=storage_dir,
    )


def test_apply_compensation_does_not_pin_a_memory_slot(tmp_path: Path) -> None:
    """Repeated applications must never pin a slot (driver rotates instead)."""
    slm = _RecordingSlm()
    compensator = _make_compensator(slm, tmp_path)
    gray = np.zeros((1200, 1920), dtype=np.uint16)

    for _ in range(3):
        compensator.apply_compensation(gray)

    assert slm.calls == [None, None, None]


def test_no_consecutive_writes_reuse_the_same_slot(monkeypatch: Any) -> None:
    """Premise check: the driver's own rotation really does change the slot.

    Drives the REAL ``Santec.display_data`` without ``memory_number`` 10 times on
    a bare instance; every ``_write_phase`` / ``_display_memory`` slot must be
    distinct, otherwise omitting ``memory_number`` would not be a fix at all.
    """
    slm = object.__new__(Santec)
    slm.slm_number = 1
    slm.video_mode = VideoMode.Memory
    slm.is_open = True
    slm._current_memory_slot = 1
    slm._displayed_phase_cache = None

    write_slots: list[int] = []
    display_slots: list[int] = []

    def fake_write_phase(
        phase: np.ndarray, memory_number: int, memory_mode: int = 0
    ) -> None:
        write_slots.append(memory_number)

    def fake_display_memory(memory_number: int) -> None:
        display_slots.append(memory_number)

    monkeypatch.setattr(slm, "_write_phase", fake_write_phase)
    monkeypatch.setattr(slm, "_display_memory", fake_display_memory)

    gray = np.zeros((1200, 1920), dtype=np.uint16)
    for _ in range(10):
        slm.display_data(gray, wait_time_s=0)

    assert len(write_slots) == 10
    assert write_slots == display_slots
    assert len(set(write_slots)) == 10
    assert all(1 <= slot <= MAX_MEM_SLOTS for slot in write_slots)


def test_no_memory_number_kwarg_in_source() -> None:
    """Guard against re-introducing a pinned slot in the cartographer module."""
    # Anchored to the package location, not CWD: a CWD-relative path makes this
    # guard silently vacuous whenever pytest is invoked from another directory.
    source = Path(dynamic_compensation.__file__)
    text = source.read_text(encoding="utf-8")
    assert "memory_number" not in text, (
        "dynamic_compensation.py must not pass a pinned memory slot: "
        "display_memory() is a firmware no-op for the already-displayed slot, "
        "so the panel would never refresh. Let the driver rotate instead."
    )
