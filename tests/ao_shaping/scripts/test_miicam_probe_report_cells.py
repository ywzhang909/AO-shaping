"""The miicam probe escapes table cells at render time -- offline.

``scripts/ccd/miicam_auto_exposure_probe.py`` drives a real camera, but its
``_write_report`` is a pure ``dict -> markdown`` function, so the table contract is
testable with no device at all. This exists because seven committed rows in
``report/miicam_auto_exposure/miicam_auto_exposure_probe/report.md`` were split by the
``mean|diff|=`` text their ``detail`` strings carry: a bare pipe inside a GFM table
cell ends the cell, which re-wrote the row's column count.
"""

from __future__ import annotations

import re
from pathlib import Path

from scripts.ccd.miicam_auto_exposure_probe import _write_report

#: Unescaped pipes only -- an escaped ``\|`` is cell content, not a delimiter.
_BARE_PIPE = re.compile(r"(?<!\\)\|")

RAW_DETAIL = "17.0 us: mean|diff|=1.91 counts, exact duplicates=0"


def _fixture() -> tuple[dict, dict]:
    """The smallest ``results``/``config`` pair ``_write_report`` accepts.

    ``response_map``, ``knee_steps`` and ``dirty_data`` are guarded by ``or {}`` /
    ``or []`` and are omitted; ``runs`` is read unguarded so it is present but empty.
    That leaves the verdict table -- the section that produced the seven split rows --
    as the only thing rendered.
    """
    results = {
        "started_at": "2026-10-25T00:00:00",
        "camera": {"sn": "SN", "width": 1200, "height": 1000,
                   "bit_depth": 8, "capture_mode": "continuous"},
        "premise": {},
        "summary": {"finished_at": "2026-10-25T00:01:00", "n_passed": 1,
                    "n_checks": 2, "all_passed": False},
        # ``results["runs"]`` is read unguarded by the per-run trajectory section, so it
        # has to exist; an empty list keeps that section out of the fixture while the
        # verdict table -- the part under test -- still renders.
        "runs": [],
        "checks": [
            {"check": "frames_not_duplicated", "passed": True, "detail": RAW_DETAIL},
            {"check": "frames_fresh[target|220]", "passed": None,
             "detail": "at 0.0200 ms: peaks ['221'], mean|diff|=2.08 counts"},
        ],
    }
    return results, {"initial_exposure_ms": 0.017}


def _verdict_block(text: str) -> list[str]:
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("| 检查项 |"))
    return lines[start:start + 4]          # header, delimiter, the two fixture rows


def test_verdict_table_cells_escape_bare_pipes(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    _write_report(out, *_fixture())
    rows = _verdict_block(out.read_text(encoding="utf-8"))[2:]
    for row in rows:
        assert "mean\\|diff\\|=" in row, f"detail pipe not escaped: {row}"
        # the bare form must be GONE -- its presence means the cell still splits
        assert RAW_DETAIL not in row, f"row still carries the raw text: {row}"


def test_verdict_table_rows_agree_with_their_delimiter(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    _write_report(out, *_fixture())
    block = _verdict_block(out.read_text(encoding="utf-8"))
    widths = {len(_BARE_PIPE.findall(l)) for l in block}
    assert len(widths) == 1, f"arity drifted across the block: {block}"


def test_escaping_is_render_only_and_leaves_the_results_dict_alone(tmp_path: Path) -> None:
    """``results.json`` stores the raw detail; only the rendered row is escaped.

    Escaping at the ``record()`` site would have put a backslash into the data
    artifact, so the same string would read differently in JSON than in markdown.
    """
    results, config = _fixture()
    _write_report(tmp_path / "report.md", results, config)
    assert results["checks"][0]["detail"] == RAW_DETAIL
