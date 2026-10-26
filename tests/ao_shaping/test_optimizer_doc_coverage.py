"""Root AGENTS.md optimizer/ module coverage guard (item 88, 2026-10-26).

Contract: every module under src/ao_shaping/optimizer/ (all .py names)
must be mentioned by file name in the root AGENTS.md narrative. Name
matching is lookbehind-bounded so a longer name can never satisfy a
shorter one via substring (the zernike_matrix_runner.py / matrix_
runner.py precedent from the root runners guard).

ROOT depth is parents[2]: this file sits directly in tests/ao_shaping/
(one level less than package-nested guards, which are parents[3]).

Pre-census drifts (32 modules, split 4 top / 8 rl / 3 wf / 17 wfless)
are listed in TODO.md ledger item 88; the doc table lives in the root
AGENTS.md "optimizer/ Module" section.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENTS = ROOT / "AGENTS.md"
PKG = ROOT / "src" / "ao_shaping" / "optimizer"


def test_every_optimizer_module_is_documented() -> None:
    """Every optimizer module file name must occur in the root AGENTS.md."""
    agents = AGENTS.read_text(encoding="utf-8")
    drifts: list[str] = []
    for path in sorted(PKG.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if not re.search(rf"(?<![A-Za-z0-9_]){re.escape(path.name)}", agents):
            drifts.append(str(path.relative_to(ROOT / "src")))
    assert not drifts, f"optimizer modules missing from AGENTS.md: {drifts}"
