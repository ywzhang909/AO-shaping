"""Pin AGENTS.md's ``utils/wavefront/`` inventory row to the directory.

The row documents every wavefront module with a dated provenance
annotation and is the discovery surface later censuses rely on
(TODO 8.9 item 82): after the issue #204 batch added ``link_design``
(``d10bb483``), the row drifted -- 13 modules listed, 14 on disk.  The
guard pins the module-name cell to the actual directory listing in both
directions: a module without a row entry, or a row entry without a
module, both fail loudly naming the diff.  Text-only: no source imports.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENTS = ROOT / "AGENTS.md"
WAVEFRONT = ROOT / "src/ao_shaping/utils/wavefront"


def _row_cells() -> list[str]:
    """The three non-empty pipe cells of the utils/wavefront row."""
    text = AGENTS.read_text(encoding="utf-8")
    rows = [l for l in text.split("\n") if l.startswith("| `utils/wavefront/` |")]
    assert len(rows) == 1, f"expected exactly one utils/wavefront row, found {len(rows)}"
    cells = [c for c in rows[0].split("|") if c.strip()]
    assert len(cells) == 3, f"row structure changed, got {len(cells)} cells"
    assert cells[0].strip().strip("`") == "utils/wavefront/", repr(cells[0])
    return cells


def test_source_layout_is_where_this_file_assumes() -> None:
    """A silent path slip would make the other check vacuously pass."""
    assert AGENTS.is_file() and WAVEFRONT.is_dir(), f"layout moved: {AGENTS} / {WAVEFRONT}"


def test_wavefront_row_names_match_directory() -> None:
    stems = sorted(p.stem for p in WAVEFRONT.glob("*.py") if p.stem != "__init__")
    names_cell = _row_cells()[1]
    listed = sorted(t.strip().strip("`") for t in names_cell.split(",") if t.strip())
    assert listed == stems, (
        "AGENTS.md utils/wavefront row drifted from the directory:\n"
        f"missing from row: {sorted(set(stems) - set(listed))}\n"
        f"in row but not on disk: {sorted(set(listed) - set(stems))}"
    )
