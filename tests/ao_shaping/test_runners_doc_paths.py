"""Lock the root AGENTS.md runners/optimizer narrative to the on-disk tree.

Contract (verified 2026-10-26):
- Every path-style ``runners/*.py`` / ``optimizer/*.py`` mention in the root
  AGENTS.md resolves to a real file under ``src/ao_shaping/``; test-path
  mentions fall back to ``tests/ao_shaping/``.
- Every module of the ``runners`` package (it spans ``nlight_dm/``,
  ``micro_drive/`` and ``slm/`` since the topology split) is mentioned
  somewhere in AGENTS.md at basename granularity - the structural overview
  tree is deliberately abbreviated, so per-directory mirroring is not the
  expectation. Both the mention test and the tree sentinels use
  lead-boundary matching, so a longer name can never satisfy a shorter one
  (``zernike_matrix_runner.py`` must not false-resolve ``matrix_runner.py``).
- The Project Structure fenced tree carries the runner subpackages and the
  late-added top modules (``matrix_runner``, ``runner_common``, ...).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENTS = ROOT / "AGENTS.md"
PKG = ROOT / "src/ao_shaping"
RUNNERS = PKG / "runners"
TESTS = ROOT / "tests/ao_shaping"

RUNNER_TREE_DIRS = ("nlight_dm/", "micro_drive/", "slm/")

# File entries are matched with a lead-boundary lookbehind so a longer name can
# never satisfy a shorter one via substring (the first control run went
# false-green exactly for that reason).
RUNNER_TREE_FILES = (
    "runner_common.py",
    "matrix_runner.py",
    "zernike_search_runner.py",
    "voltage_runner.py",
    "model_in_loop_session.py",
    "offset_runner.py",
)


def _agents() -> str:
    return AGENTS.read_text(encoding="utf-8")


def _runners_tree_slice() -> str:
    text = _agents()
    fences = re.findall(r"```[^\n]*\n(.*?)```", text, flags=re.DOTALL)
    fence = next(f for f in fences if "├── runners/" in f)
    ls = fence.splitlines()
    i = next(k for k, l in enumerate(ls) if l.startswith("│   │   ├── runners/"))
    j = next(
        k
        for k in range(i + 1, len(ls))
        if ls[k].startswith("│   │   ├── ") or ls[k].startswith("│   │   └── ")
    )
    return "\n".join(ls[i:j]) + "\n"


def test_runners_path_mentions_resolve() -> None:
    agents = _agents()
    drifts = []
    for mention in sorted(set(re.findall(r"runners/[A-Za-z0-9_/.]+\.py", agents))):
        if (PKG / mention).is_file():
            continue
        if (TESTS / mention).is_file():
            continue
        drifts.append(mention)
    assert not drifts, f"AGENTS.md runners/ path mentions not on disk: {drifts}"


def test_optimizer_path_mentions_resolve() -> None:
    agents = _agents()
    drifts = [
        m
        for m in sorted(set(re.findall(r"optimizer/[A-Za-z0-9_/.]+\.py", agents)))
        if not (PKG / m).is_file()
    ]
    assert not drifts, f"AGENTS.md optimizer/ path mentions not on disk: {drifts}"


def test_every_runner_module_is_documented() -> None:
    agents = _agents()
    drifts = [
        str(p.relative_to(ROOT))
        for p in sorted(RUNNERS.rglob("*.py"))
        if "__pycache__" not in p.parts
        and not re.search(rf"(?<![A-Za-z0-9_]){re.escape(p.name)}", agents)
    ]
    assert not drifts, f"runner modules missing from AGENTS.md: {drifts}"


def test_project_tree_covers_runner_subpackages() -> None:
    tree = _runners_tree_slice()
    missing = [d for d in RUNNER_TREE_DIRS if d not in tree]
    missing += [
        n for n in RUNNER_TREE_FILES if not re.search(rf"(?<![A-Za-z0-9_]){re.escape(n)}", tree)
    ]
    assert not missing, f"Project Structure tree missing runner entries: {missing}"
