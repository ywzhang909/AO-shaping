"""Pin AGENTS.md's inventory table rows to their on-disk subpackages.

Two table families in AGENTS.md document every module with a dated
provenance annotation and are the discovery surface later censuses rely
on (TODO 8.9 items 82/83/84): the utils rows and the algorithm
subpackage rows.  Both drifted: the wavefront row missed
``link_design`` (items 82), the io/image rows missed
``recorder_keys``/``npz_summary``/``centroid`` (item 83), and the
algorithm table drifted harder still (item 84) -- the heuristic row
used alias tokens for module names that never existed as files, the
signal_processing row missed 10 modules, the tabu row was bare prose,
and the goal_functions row listed a phantom ``image_metrics`` that
never entered git history.  The guard pins each names cell to the
actual directory listing in both directions: a module without a row
entry, or a row entry without a module, both fail loudly naming the
diff.  A `` `name` (+ note)`` decoration maps to ``name`` (the image
row uses that grammar to track the ``target/`` subpackage directory).
Text-only: no source imports.

``utils/cli/`` has no table row by design -- it is documented in its
own red-line section, and the guard deliberately does not force one.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENTS = ROOT / "AGENTS.md"
SUBPACKAGES = {
    "utils/io/": ROOT / "src/ao_shaping/utils/io",
    "utils/image/": ROOT / "src/ao_shaping/utils/image",
    "utils/wavefront/": ROOT / "src/ao_shaping/utils/wavefront",
    "utils/slm/": ROOT / "src/ao_shaping/utils/slm",
    "gradient/": ROOT / "src/ao_shaping/algorithm/gradient",
    "heuristic/": ROOT / "src/ao_shaping/algorithm/heuristic",
    "signal_processing/": ROOT / "src/ao_shaping/algorithm/signal_processing",
    "tabu/": ROOT / "src/ao_shaping/algorithm/tabu",
    "goal_functions/": ROOT / "src/ao_shaping/algorithm/goal_functions",
}


def _row_cells(label: str) -> list[str]:
    """The three non-empty pipe cells of one inventory row."""
    anchor = f"| `{label}` |"
    text = AGENTS.read_text(encoding="utf-8")
    rows = [l for l in text.split("\n") if l.startswith(anchor)]
    assert len(rows) == 1, f"expected exactly one {label} row, found {len(rows)}"
    cells = [c for c in rows[0].split("|") if c.strip()]
    assert len(cells) == 3, f"{label}: expected 3 non-empty cells, got {len(cells)}"
    assert cells[0].strip().strip("`") == label, repr(cells[0])
    return cells


def _listed_tokens(names_cell: str) -> list[str]:
    """Names-cell tokens; a ' `name` (+ note)` decoration maps to `name`."""
    return sorted(
        t.split(" (")[0].strip().strip("`") for t in names_cell.split(",") if t.strip()
    )


def test_source_layout_is_where_this_file_assumes() -> None:
    """A silent path slip would make the other check vacuously pass."""
    assert AGENTS.is_file(), AGENTS
    for label, pkg in SUBPACKAGES.items():
        assert pkg.is_dir(), (label, pkg)


def test_inventory_rows_match_directories() -> None:
    """One unified diff message across all pinned rows keeps born-red legible."""
    diffs = []
    for label, pkg in SUBPACKAGES.items():
        stems = sorted(p.stem for p in pkg.glob("*.py") if p.stem != "__init__")
        listed = _listed_tokens(_row_cells(label)[1])
        missing = sorted(set(stems) - set(listed))
        extra = sorted(set(listed) - set(stems))
        if missing or extra:
            diffs.append(
                f"{label}: missing from row {missing}, in row but not on disk {extra}"
            )
    assert not diffs, (
        "AGENTS.md inventory rows drifted from the directories:\n"
        + "\n".join(diffs)
    )
