"""Guard: the CI shard list must cover the whole test tree.

Added because the shard list in .github/workflows/pytest.yml was written by hand and
was already wrong on arrival: it covered 11 of the 12 top-level test areas and listed
two root-level test modules but omitted ``tests/ao_shaping/test_config.py`` (81
collected items). A shard list that quietly stops covering a directory is worse than
no CI at all, because a green job then *asserts* those tests ran. This is the same
failure mode as a stale allowlist, so it is guarded the same way: two-way.

The check is regex-based rather than YAML-based on purpose: PyYAML is not a declared
dependency of this repo, and a guard that cannot run is not a guard. The workflow is
therefore required to keep one ``run_shard`` invocation per line, which is also what
makes the shard list reviewable in a diff.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
WORKFLOW = REPO / ".github" / "workflows" / "pytest.yml"
TESTS_ROOT = REPO / "tests" / "ao_shaping"

#: Areas the container harness deliberately does not collect (see EXCLUDES in
#: scripts/build/run_pytest_ci.sh and pyproject addopts). Anything else must be in a
#: shard.
NOT_SHARDED = {"ml", "__pycache__"}

SHARD_RE = re.compile(
    r'^\s*run_shard\s+"(?P<name>[^"]+)"\s+(?P<per_file>[01])\s+(?P<blocking>[01])\s+(?P<targets>\S.*)$'
)


def _shards() -> list[dict[str, object]]:
    text = WORKFLOW.read_text(encoding="utf-8")
    shards: list[dict[str, object]] = []
    for line in text.splitlines():
        match = SHARD_RE.match(line)
        if match is None:
            continue
        shards.append(
            {
                "name": match.group("name"),
                "per_file": match.group("per_file") == "1",
                "blocking": match.group("blocking") == "1",
                "targets": match.group("targets").split(),
            }
        )
    return shards


def _expected_areas() -> set[str]:
    """Top-level test areas and root-level test modules that must be sharded."""
    areas: set[str] = set()
    for child in TESTS_ROOT.iterdir():
        if child.name in NOT_SHARDED:
            continue
        if child.is_dir():
            areas.add(child.name)
        elif child.is_file() and child.name.startswith("test_") and child.suffix == ".py":
            areas.add(child.name)
    return areas


def _covered(shards: list[dict[str, object]]) -> set[str]:
    covered: set[str] = set()
    for shard in shards:
        for target in shard["targets"]:  # type: ignore[attr-defined]
            rel = target.removeprefix("tests/ao_shaping/")
            covered.add(rel)
    return covered


def test_workflow_exists_and_has_shards() -> None:
    assert WORKFLOW.is_file(), f"missing {WORKFLOW}"
    shards = _shards()
    assert shards, (
        "no run_shard invocations parsed -- either the shards were removed or the "
        "one-invocation-per-line rule was broken (a continued line is invisible to "
        "this guard, which would silently disable coverage checking)."
    )


def test_every_test_area_is_sharded() -> None:
    shards = _shards()
    missing = sorted(_expected_areas() - _covered(shards))
    assert not missing, (
        f"{missing} live under tests/ao_shaping but appear in no shard of "
        f"{WORKFLOW.relative_to(REPO)}; the job would go green without running them. "
        f"Add a run_shard line (one per line, no continuations)."
    )


def test_every_shard_target_exists() -> None:
    """Catches a shard pointing at a renamed or deleted directory (silently 0 items)."""
    shards = _shards()
    dangling: list[str] = []
    for shard in shards:
        for target in shard["targets"]:  # type: ignore[attr-defined]
            if not (REPO / target).exists():
                dangling.append(f"{shard['name']}: {target}")
    assert not dangling, (
        f"{dangling} do not exist; pytest exits 0 on an empty selection unless "
        "--strict-markers/last-fail turns it red, so a stale shard would report PASS "
        "while testing nothing."
    )


def test_shard_targets_are_not_redundantly_nested() -> None:
    """A shard that repeats a directory already covered by another shard double-runs it.

    Double-running is not merely slow: the two runs land in one summary, so a
    flaky area can appear to pass in one shard and mask the other.
    """
    shards = _shards()
    targets: list[tuple[str, str]] = [
        (str(shard["name"]), t) for shard in shards for t in shard["targets"]
    ]  # type: ignore[attr-defined]
    overlap: list[str] = []
    for name, target in targets:
        for other_name, other in targets:
            if name == other_name or target == other:
                continue
            if target.startswith(other + "/"):
                overlap.append(f"{name}:{target} inside {other_name}:{other}")
    assert not overlap, f"nested shard targets duplicate work: {overlap}"


def test_at_least_one_hard_gate_shard_exists() -> None:
    """If every shard were advisory the job could never fail, i.e. it is not a gate."""
    shards = _shards()
    assert any(shard["blocking"] for shard in shards), (
        "every shard is advisory -- the workflow can never go red, which recreates "
        "the exact condition that let an authored-red test survive a day (TODO.md 8.13)."
    )
