"""Behaviour pins for the parse-failure handler in ``build_position_ip_table`` (issue #174).

The 1300-unit wiring CSV is read into a position/IP lookup table. When a row is malformed the
table degrades to an empty frame instead of taking the whole control page down. That catch used to
be blind (``except Exception``); it is now enumerated to the three types the guarded statements can
actually raise (``int()`` on a bad cell, column access on a missing header). These tests pin **both
halves** of the resulting contract, because a narrowing that is not tested is just an assumption:

- the realistic parse failures still degrade to an empty table (no regression in resilience);
- an error outside that enumeration is **no longer swallowed** (the behaviour change itself).
"""

from __future__ import annotations

import pandas as pd
import pytest

from ao_shaping.gui.r50.r50_channel_select import build_position_ip_table


def _row(**overrides) -> dict[str, object]:
    """One complete wiring row (column names as they appear in 1300-5-enriched.csv)."""
    row: dict[str, object] = {
        "36×36行": 1,
        "36×36列": 2,
        "IP组": 3,
        "序号": 7,
        "组": "A",
        "引脚编号": 11,
        "连接器": "J1",
    }
    row.update(overrides)
    return row


def test_valid_rows_build_position_ordered_table() -> None:
    # pos = row * GRID_SIZE(36) + col + 1, so (1,2) -> 39 and (0,0) -> 1; CSV order is by pin.
    df = pd.DataFrame([_row(), _row(**{"36×36行": 0, "36×36列": 0})])

    out = build_position_ip_table(df)

    assert list(out["位置序号"]) == [1, 39]
    assert out["IP"].iloc[-1] == "192.168.0.3"


def test_empty_frame_degrades_to_empty_table() -> None:
    assert build_position_ip_table(pd.DataFrame()).empty


def test_non_numeric_cell_still_degrades_to_empty_table() -> None:
    """ValueError path — the failure a hand-edited wiring CSV actually produces."""
    assert build_position_ip_table(pd.DataFrame([_row(**{"36×36行": "not-a-number"})])).empty


def test_missing_header_still_degrades_to_empty_table() -> None:
    """KeyError path — a re-exported CSV with renamed/dropped columns."""
    assert build_position_ip_table(pd.DataFrame([{"IP组": 1, "序号": 0}])).empty


def test_null_cell_still_degrades_to_empty_table() -> None:
    """TypeError path — int(None) from a blank cell."""
    assert build_position_ip_table(pd.DataFrame([_row(**{"序号": None})])).empty


def test_error_outside_the_enumeration_is_no_longer_swallowed() -> None:
    """The narrowing itself: a defect in the builder must surface, not hide as an empty table.

    Before the narrowing this returned an empty frame and logged a warning, so a real bug in
    this function was indistinguishable from a malformed CSV.
    """

    class _ExplodingFrame:
        empty = False

        def iterrows(self):
            raise RuntimeError("unexpected internal fault")

    with pytest.raises(RuntimeError, match="unexpected internal fault"):
        build_position_ip_table(_ExplodingFrame())
