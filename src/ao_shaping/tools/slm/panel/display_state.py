"""Preserve the SLM display while a bench probe uses temporary memory slots."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


@contextmanager
def preserve_slm_display(slm) -> Iterator[None]:
    """Restore either the displayed memory slot or grayscale mode on exit."""
    from ao_shaping.drivers.slm.santec import SantecError

    try:
        original_slot = slm.get_displayed_memory_number()
    except SantecError as exc:
        if exc.code != 1:
            raise
        original_slot = None
    original_gray = slm.get_current_grayscale() if original_slot is None else None
    try:
        yield
    finally:
        if original_slot is None:
            slm.set_grayscale(original_gray)
        else:
            slm._display_memory(original_slot)
