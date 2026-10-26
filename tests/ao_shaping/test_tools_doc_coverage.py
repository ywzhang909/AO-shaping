"""Root AGENTS.md tools/ module coverage guard (item 90, 2026-10-26).

Contract identical to the utils/display guard (item 89): every module
under src/ao_shaping/tools/ must be referenced in the root AGENTS.md by
full file name (``name.py``, lead-boundary lookbehind) or by a bare stem
token NOT continued by a word character or a dot. The dot exclusion
keeps dotted non-module references from covering a module; dotted
module forms are covered by the full-name branch. Generic ``__main__``
mentions satisfy ``__main__.py`` files via the same floor-contract
precedent as ``__init__.py``.

ROOT depth is parents[2]: this file sits directly in
tests/ao_shaping/ (package-nested guards use parents[3]).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENTS = ROOT / "AGENTS.md"
LEAD = r"(?<![A-Za-z0-9_])"
STEM_TAIL = r"(?![A-Za-z0-9_.])"


def test_every_tools_module_is_documented() -> None:
    """Every tools module name/stem must occur in the root AGENTS.md."""
    agents = AGENTS.read_text(encoding="utf-8")
    missing: list[str] = []
    for path in sorted((ROOT / "src" / "ao_shaping" / "tools").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        name_hit = re.search(LEAD + re.escape(path.name), agents)
        stem_hit = re.search(LEAD + re.escape(path.stem) + STEM_TAIL, agents)
        if not (name_hit or stem_hit):
            missing.append(str(path.relative_to(ROOT / "src" / "ao_shaping")))
    assert not missing, f"tools modules missing from AGENTS.md: {missing}"
