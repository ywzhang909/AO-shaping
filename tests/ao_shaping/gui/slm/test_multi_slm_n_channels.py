"""Issue #150: the SLM layout is N-capable, not hardcoded to ordinals 1 and 2.

The device layer was already N-safe (``connect_slm`` keys every parameter by
``slm{N}``); only the *layout* loops hardcoded the pair, so a connected SLM #3
was written into ``st.session_state`` and then silently ignored.

These tests pin the three properties that make the fix stick:

* the connected-ordinal view is derived per call (never cached) and follows the
  real ordinals, not ``1..2``;
* state initialisation covers every ordinal the scan can probe;
* no layout module loops over the literal pair again (AST, so comments and
  docstrings cannot trip it).
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest

from ao_shaping.gui.slm import bench_settings as bs
from ao_shaping.gui.slm import multi_slm_controller as gui
from ao_shaping.gui.slm import shaping_controls as sc
from ao_shaping.gui.slm import slm_registry as reg


class _St:
    """Minimal ``streamlit`` stand-in: only ``session_state`` is read."""

    def __init__(self, session_state: dict[str, Any]) -> None:
        self.session_state = session_state


@pytest.fixture()
def state(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    session: dict[str, Any] = {}
    fake = _St(session)
    # Both modules read ``st``: the view moved to ``slm_registry`` (#150 size ratchet).
    monkeypatch.setattr(bs, "st", fake)
    monkeypatch.setattr(reg, "st", fake)
    return session


def _connect(session: dict[str, Any], *nums: int, handle: Any = None) -> None:
    for n in nums:
        session[f"slm{n}_connected"] = True
        session[f"slm{n}"] = object() if handle is None else handle


def test_connected_ordinals_are_not_capped_at_two(state: dict[str, Any]) -> None:
    _connect(state, 1, 2, 3)
    assert bs.connected_slm_numbers() == [1, 2, 3]


def test_disconnected_or_handle_less_slots_are_excluded(state: dict[str, Any]) -> None:
    _connect(state, 1)
    state["slm2_connected"] = True  # flagged connected but the handle is gone
    state["slm3_connected"] = False
    state["slm3"] = object()
    assert bs.connected_slm_numbers() == [1]


def test_the_view_is_recomputed_every_call(state: dict[str, Any]) -> None:
    _connect(state, 1, 2)
    assert bs.connected_slm_numbers() == [1, 2]
    state["slm2_connected"] = False
    assert bs.connected_slm_numbers() == [1]


def test_handles_pair_each_ordinal(state: dict[str, Any]) -> None:
    first, fifth = object(), object()
    _connect(state, 1, handle=first)
    _connect(state, 5, handle=fifth)
    assert bs.connected_slm_handles() == [(1, first), (5, fifth)]


def test_state_initialises_every_ordinal_the_scan_can_reach(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _St({})
    monkeypatch.setattr(gui, "st", stub)
    gui._initialize_slm_state()
    for n in range(1, bs.MAX_SLMS + 1):
        assert stub.session_state[f"slm{n}"] is None
        assert stub.session_state[f"slm{n}_connected"] is False


def test_scan_default_is_the_shared_maximum() -> None:
    default = inspect.signature(gui._scan_available_slms).parameters["max_slms"].default
    assert default == bs.MAX_SLMS
    assert bs.MAX_SLMS >= 3, "a two-SLM ceiling would re-hide the third device"


def test_row_width_helper_sees_every_connected_slm(state: dict[str, Any]) -> None:
    _connect(state, 1, 2, 3)
    assert sc._slms_on_screen() == 3


def test_chain_role_caption_names_the_previous_stage() -> None:
    assert gui._chain_role_caption(1) == "串联首台：入射面 = 激光入射"
    assert gui._chain_role_caption(3) == "串联第 3 台：入射面 = SLM 2 出射面"
    assert gui._chain_role_caption(0) is None, "ordinal 0 is not a device"


@pytest.mark.parametrize(
    "path",
    [Path(gui.__file__), Path(bs.__file__), Path(sc.__file__), Path(reg.__file__)],
    ids=lambda p: p.name,
)
def test_no_loop_over_the_literal_pair(path: Path) -> None:
    """A return of ``for slm_num in (1, 2)`` is the defect this issue fixed."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.comprehension)):
            continue
        iterator = node.iter
        if (
            isinstance(iterator, ast.Tuple)
            and len(iterator.elts) == 2
            and [getattr(elt, "value", None) for elt in iterator.elts] == [1, 2]
        ):
            offenders.append(getattr(node, "lineno", 0))
    assert not offenders, f"{path.name} still loops over the literal (1, 2) at {offenders}"
