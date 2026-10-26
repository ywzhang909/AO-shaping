"""Root AGENTS.md utils/ + display/ module coverage guard (item 89, 2026-10-26).

Contract: every module under src/ao_shaping/utils/ and
src/ao_shaping/display/ must be referenced in the root AGENTS.md either
by full file name (``name.py``, lead-boundary lookbehind) or by a bare
stem token NOT continued by a word character or a dot. The dot
exclusion keeps dotted non-module references from counting (e.g.
``exposure.npy`` must not cover the ``exposure.py`` module; dotted
module forms like ``cli_params.py`` are covered by the full-name
branch instead).

This is deliberately a token-level FLOOR contract: leaves of
utils/image/target/ and utils/cli/ are additionally enumerated in the
AGENTS.md census table (the item-89 fill-ins) even where a boundary
hit somewhere else in the file would have satisfied the token floor.

ROOT depth is parents[2]: this file sits directly in
tests/ao_shaping/ (package-nested guards use parents[3] - the
item-87 depth lesson).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENTS = ROOT / "AGENTS.md"
STEM_TAIL = r"(?![A-Za-z0-9_.])"
LEAD = r"(?<![A-Za-z0-9_])"


def _drifts(rel_dir: str) -> list[str]:
    """Modules of rel_dir with neither a full-name nor a bare-stem mention."""
    agents = AGENTS.read_text(encoding="utf-8")
    missing: list[str] = []
    for path in sorted((ROOT / "src" / "ao_shaping" / rel_dir).rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        name_hit = re.search(LEAD + re.escape(path.name), agents)
        stem_hit = re.search(LEAD + re.escape(path.stem) + STEM_TAIL, agents)
        if not (name_hit or stem_hit):
            missing.append(str(path.relative_to(ROOT / "src" / "ao_shaping")))
    return missing


def test_every_utils_module_is_documented() -> None:
    """Every utils module name/stem must occur in the root AGENTS.md."""
    missing = _drifts("utils")
    assert not missing, f"utils modules missing from AGENTS.md: {missing}"


def test_every_display_module_is_documented() -> None:
    """Every display module name/stem must occur in the root AGENTS.md."""
    missing = _drifts("display")
    assert not missing, f"display modules missing from AGENTS.md: {missing}"
