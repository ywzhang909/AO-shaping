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


def _test_files(root: Path) -> set[str]:
    """Every collected test module under ``root``, relative to ``tests/ao_shaping``."""
    units: set[str] = set()
    for path in root.rglob("test_*.py"):
        rel = path.relative_to(TESTS_ROOT)
        if any(part in NOT_SHARDED for part in rel.parts):
            continue
        units.add(str(rel).replace("\\", "/"))
    return units


def test_workflow_exists_and_has_shards() -> None:
    assert WORKFLOW.is_file(), f"missing {WORKFLOW}"
    shards = _shards()
    assert shards, (
        "no run_shard invocations parsed -- either the shards were removed or the "
        "one-invocation-per-line rule was broken (a continued line is invisible to "
        "this guard, which would silently disable coverage checking)."
    )


def test_every_test_area_is_sharded() -> None:
    """Every test module under ``tests/ao_shaping`` must fall inside some shard.

    Coverage is proved per file rather than per area: naming an area directory covers
    its whole subtree, and an area split across shards -- which is what an area that
    does not fit one container needs -- must name the rest file by file. Either way a
    module added later turns the job red instead of going uncollected, which is the
    failure mode a hand-maintained list is prone to.
    """
    targets = [
        t.removeprefix("tests/ao_shaping/") for shard in _shards() for t in shard["targets"]
    ]
    uncovered = sorted(
        unit
        for unit in _test_files(TESTS_ROOT)
        if not any(unit == t or unit.startswith(t.rstrip("/") + "/") for t in targets)
    )
    assert not uncovered, (
        f"{uncovered[:12]} live under tests/ao_shaping but fall in no shard of "
        f"{WORKFLOW.relative_to(REPO)}; the job would go green without running them. "
        f"Name the area, or the files, in a run_shard line (one per line, no "
        f"continuations)."
    )


def test_advisory_shards_state_their_flip_condition() -> None:
    """An advisory shard that never says when it becomes blocking stays forever.

    ``blocking=0`` exists for one measured reason at a time; without a written flip
    condition it degrades into the permanent hiding the workflow was rebuilt to stop.
    """
    lines = WORKFLOW.read_text(encoding="utf-8").splitlines()
    unexplained: list[str] = []
    for shard in _shards():
        if shard["blocking"]:  # type: ignore[truthy-bool]
            continue
        line = next(i for i, text in enumerate(lines) if f'run_shard "{shard["name"]}"' in text)
        window = " ".join(lines[max(0, line - 6) : line]).lower()
        if "advisory" not in window or "flip" not in window:
            unexplained.append(str(shard["name"]))
    assert not unexplained, (
        f"advisory shards without an 'advisory:' note and a 'flip' condition "
        f"above the invocation: {unexplained}"
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
    """Proves the reason scan on planted text, in both directions.

    Reading the live harness would tie this control to the current exemption list: when
    the last exemption is legitimately removed the control fails, detecting nothing. So
    the probe supplies its own list -- one entry with a reason, one without -- and the
    scan must pass the first and flag the second.
    """
    explained = (
        "EXCLUDES=(\n"
        "    # because: measured failure on a corpus-free runner\n"
        '    "--ignore=$WORKDIR_IN_CONTAINER/tests/ao_shaping/a/b"\n'
        ")\n"
    )
    unexplained = (
        "EXCLUDES=(\n"
        '    "--ignore=$WORKDIR_IN_CONTAINER/tests/ao_shaping/a/b"\n'
        ")\n"
    )
    assert _reason_gaps(explained) == [], _reason_gaps(explained)
    assert _reason_gaps(unexplained) == ["line 2: tests/ao_shaping/a/b"], (
        "the reason check passed an exemption with no comment -- it is vacuous"
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

