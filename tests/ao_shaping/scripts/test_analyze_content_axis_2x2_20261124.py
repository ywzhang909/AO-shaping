"""Offline tests for the read-only S1b content-axis judge script."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

JUDGE = importlib.import_module("scripts.slm.analyze_content_axis_2x2_20261124")


def _gaussian(frame_shape: tuple[int, int], center: tuple[float, float],
              amp: float, sigma: float = 3.0) -> np.ndarray:
    img = np.zeros(frame_shape, dtype=np.float64)
    rr, cc = np.indices(frame_shape)
    img = amp * np.exp(-((rr - center[0]) ** 2 + (cc - center[1]) ** 2)
                       / (2.0 * sigma ** 2))
    return img


def test_axis_of_row_col_ambiguous() -> None:
    assert JUDGE.axis_of(56.0, -1.2) == "row"
    assert JUDGE.axis_of(-0.7, 95.0) == "column"
    assert JUDGE.axis_of(0.0, 0.0) == "ambiguous"


def test_principal_excludes_zero_order_core() -> None:
    frame = _gaussian((2200, 2600), (1105.25, 1227.56), 1200.0)
    # displaced brightest order on the row axis
    frame += _gaussian((2200, 2600), (1105.25 + 56.3, 1227.56 - 1.2), 610.0)
    p = JUDGE.principal(frame, (1105.25, 1227.56))
    assert p is not None
    assert abs(p["dr"] - 56.3) < 1.5
    assert abs(p["dc"] + 1.2) < 1.5
    assert p["peak"] > 500.0


def test_judge_group_content_correlated_4of4() -> None:
    """Synthetic in-spec A->column(compressed), B->row(K_row*N) = 4/4."""
    def cell(axis: str, n: float, amp: float) -> dict:
        return {
            "principal": {"dr": amp if axis == "row" else -1.0,
                          "dc": amp if axis == "column" else -1.0,
                          "peak": 600.0},
            "axis": axis,
            "amplitude_px": amp,
            "weak_signal": False,
            "residual_vs_KcolN_compressed_px": abs(
                amp - JUDGE.K_COL * n / JUDGE.X_FAMILY_COMPRESSION),
            "residual_vs_KrowN_px": abs(amp - JUDGE.K_ROW * n),
        }
    cells = {
        "A128": cell("column", 15.0, JUDGE.K_COL * 15.0
                     / JUDGE.X_FAMILY_COMPRESSION),  # 60px
        "A256": cell("column", 7.5, JUDGE.K_COL * 7.5
                     / JUDGE.X_FAMILY_COMPRESSION),  # 30px
        "B128": cell("row", 15.0, JUDGE.K_ROW * 15.0),  # 93px
        "B256": cell("row", 7.5, JUDGE.K_ROW * 7.5),    # 46.5px
    }
    assert JUDGE.judge_group(cells)["verdict"] == "content_correlated_4of4"


def test_judge_group_insufficient_signal() -> None:
    """All four cells weak -> insufficient_signal (axis not judgeable)."""
    cells = {
        c: {"principal": None, "axis": "row", "amplitude_px": None,
            "weak_signal": True}
        for c in JUDGE.CELLS
    }
    verdict = JUDGE.judge_group(cells)
    assert verdict["verdict"] == "insufficient_signal"
    assert len(verdict["weak_cells"]) == 4


def test_real_capture_verdicts() -> None:
    """Real 2026-11-24 paired capture: D=0 insufficient_consistency,
    D=-0.5 insufficient_signal (needs the data directory present)."""
    base = JUDGE.DEFAULT_BASE
    if not (base / "d0_20261124" / "raw" / "A128.npz").exists():
        return  # data not checked out; skip silently
    for group, expected in (("d0_20261124", "insufficient_consistency"),
                            ("dm0p5_20261124", "insufficient_signal")):
        cells = {c: JUDGE.analyze_cell(group, c, base) for c in JUDGE.CELLS}
        assert JUDGE.judge_group(cells)["verdict"] == expected
