"""N-SLM session registry: one derived view of "which SLMs are on the bench".

Extracted from ``bench_settings`` / ``multi_slm_controller`` when the #150
layout change pushed both modules over their size ratchet. Keeping the view in
one leaf module also keeps the *single* source of truth: the layout, the
sidebar and the shaping row-width helper all ask this module, so a third (or
fourth) connected SLM can never be half-recognised.

Storage stays flat (``slm{N}_connected`` / ``slm{N}`` keys) because Streamlit
widget-key ownership forbids nesting it. Nothing is cached: a cached list would
outlive a disconnect and hand callers a closed handle.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

__all__ = [
    "MAX_SLMS",
    "chain_role_caption",
    "connected_slm_handles",
    "connected_slm_numbers",
]

#: Highest SLM ordinal the layout and the discovery scan consider (1..MAX_SLMS).
MAX_SLMS = 8


def connected_slm_numbers() -> list[int]:
    """Ordinals of the SLMs that are connected *and* hold a live handle."""
    return [
        n
        for n in range(1, MAX_SLMS + 1)
        if st.session_state.get(f"slm{n}_connected")
        and st.session_state.get(f"slm{n}") is not None
    ]


def connected_slm_handles() -> list[tuple[int, Any]]:
    """``(ordinal, handle)`` pairs for every connected SLM, ascending."""
    return [(n, st.session_state[f"slm{n}"]) for n in connected_slm_numbers()]


def chain_role_caption(slm_num: int) -> str | None:
    """Chaining semantics: stage k is fed by stage k-1's exit pupil.

    Beyond two devices an operator easily reads a later stage as its own beam
    path; the caption states the physical chain. Ordinal 0 is not a device
    (only test doubles pass it), so it gets no caption.
    """
    if slm_num == 1:
        return "串联首台：入射面 = 激光入射"
    if slm_num > 1:
        return f"串联第 {slm_num} 台：入射面 = SLM {slm_num - 1} 出射面"
    return None
