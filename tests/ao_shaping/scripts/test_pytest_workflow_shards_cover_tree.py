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

#: Nothing under tests/ao_shaping may be exempt from a shard. It used to hold ``ml``,
#: which is how three authored-red ml tests stayed invisible: the job went green while
#: never collecting them. Areas the container harness genuinely cannot collect are
#: exempted one sub-path at a time in scripts/build/run_pytest_ci.sh, where each entry
#: carries its reason, and the checks below keep that list honest.
NOT_SHARDED: set[str] = {"__pycache__"}

#: The harness's own exclusion list -- the single place an ml sub-path may be hidden.
HARNESS = REPO / "scripts" / "build" / "run_pytest_ci.sh"

IGNORE_RE = re.compile(
    r'"--ignore=\$\{?(?P<base>WORKDIR_IN_CONTAINER|REPO)\}?/(?P<target>tests/\S+)"'
)

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


def _ignores_from(text: str) -> list[tuple[int, str]]:
    """``(line_number, target)`` for every ``--ignore`` in a harness script body."""
    entries: list[tuple[int, str]] = []
    for number, line in enumerate(text.splitlines(), 1):
        match = IGNORE_RE.search(line)
        if match is not None:
            entries.append((number, match.group("target")))
    return entries


def _harness_ignores() -> list[tuple[int, str]]:
    """``(line_number, target)`` for every ``--ignore`` the real harness passes."""
    return _ignores_from(HARNESS.read_text(encoding="utf-8"))


def test_harness_ignore_parser_is_not_vacuous() -> None:
    """A parser that matches nothing turns all three exemption checks green by default.

    Fed the two spellings the harness actually uses, it must name the exact target --
    the same discipline as ``test_guard_is_not_vacuous`` above for the depth scanner.
    """
    planted = (
        "EXCLUDES=(\n"
        '    "--ignore=$WORKDIR_IN_CONTAINER/tests/ao_shaping/ml/zernike"\n'
        '    "--ignore=${WORKDIR_IN_CONTAINER}/tests/ao_shaping/ml/x"\n'
        ")\n"
    )
    assert _ignores_from(planted) == [
        (2, "tests/ao_shaping/ml/zernike"),
        (3, "tests/ao_shaping/ml/x"),
    ], _ignores_from(planted)
    assert _harness_ignores(), "the real harness carries no --ignore entries at all"


def test_harness_ignore_targets_exist() -> None:
    """A dangling exemption is a silent hole: it excludes nothing and explains less."""
    dangling = [
        f"line {number}: {target}"
        for number, target in _harness_ignores()
        if not (REPO / target).exists()
    ]
    assert not dangling, f"harness --ignore paths that do not exist: {dangling}"


def test_no_harness_ignore_blanks_a_whole_test_area() -> None:
    """The blanket ``tests/ao_shaping/<area>`` pattern is what hid the red ml tests.

    A sub-path exemption is reviewable and keeps the rest of its area running; an
    area-wide one means the job reports PASS having collected none of it.
    """
    blanked = [
        f"line {number}: {target}"
        for number, target in _harness_ignores()
        if target.removeprefix("tests/ao_shaping/").count("/") == 0
        and target.startswith("tests/ao_shaping/")
    ]
    assert not blanked, (
        f"these harness ignores exclude an entire top-level area: {blanked}; exempt "
        "the specific sub-path that cannot run, with its reason, instead."
    )


def test_every_harness_ignore_carries_a_reason() -> None:
    """An exemption without a reason becomes permanent by accident.

    The reason must sit **between the array opener (or the previous entry) and this
    entry**: a policy comment above ``EXCLUDES=(`` explains the list, not the entry, so
    letting the first entry borrow it would leave an unexplained exemption green.
    """
    unexplained = _reason_gaps(HARNESS.read_text(encoding="utf-8"))
    assert not unexplained, (
        f"these --ignore entries carry no reason of their own: {unexplained}"
    )


def test_reason_check_is_not_vacuous() -> None:
    """Stripping an entry's own comment must be caught, not excused by the header.

    Without this control the check could pass on a harness where no entry explains
    itself, because the policy block above ``EXCLUDES=(`` is always nearby.
    """
    lines = HARNESS.read_text(encoding="utf-8").splitlines()
    opener = next(i for i, line in enumerate(lines) if line.strip().endswith("EXCLUDES=("))
    closer = next(i for i in range(opener + 1, len(lines)) if lines[i].strip() == ")")
    planted = [
        line
        for i, line in enumerate(lines)
        if not (opener < i < closer and line.lstrip().startswith("#"))
    ]
    stripped = "\n".join(planted) + "\n"
    assert "tests/ao_shaping/ml/zernike" in stripped, (
        "the planted file lost an entry; the control no longer tests anything"
    )
    assert _reason_gaps(stripped), (
        "the reason check passed with every comment inside the EXCLUDES block removed "
        "-- it is vacuous"
    )


def _reason_gaps(text: str) -> list[str]:
    """Entries in ``text`` that carry no reason of their own."""
    lines = text.splitlines()
    gaps: list[str] = []
    for number, target in _ignores_from(text):
        boundary = 0
        for previous in range(number - 2, -1, -1):
            stripped = lines[previous].strip()
            if IGNORE_RE.search(lines[previous]) or stripped.endswith("("):
                boundary = previous + 1
                break
        if not any(
            line.lstrip().startswith("#") for line in lines[boundary : number - 1]
        ):
            gaps.append(f"line {number}: {target}")
    return gaps

