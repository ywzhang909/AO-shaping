"""R-28 D1/D2: one implementation behind the several date/目录 spellings.

Before this change the package carried two pairs of byte-identical copies plus
two structurally-identical directory builders:

    cli_helpers.get_timestamp_str()  ==  file.gen_date_str()      # %Y%m%d_%H%M%S
    cli_helpers.get_date_dir_name()  ==  file.py:140, 167 inline # %Y%m%d
    cli_helpers.create_save_dir()    ==  file.gen_date_dir()      # up to fmt

All of them are exported through ``ao_shaping.utils.__all__`` and all have live
callers, so the public names stay. What these tests pin is that the names now
share ONE implementation and that the two directory conventions keep their
different granularity -- ``create_save_dir`` stamps a bare date while
``gen_date_dir`` stamps a full timestamp, and collapsing that difference would
move every runner's output directory.

The collision guard matters because two builders that both call
``datetime.now()`` can disagree across a second boundary; the frozen-clock
tests below pin the strings without freezing the clock.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pytest

from ao_shaping.utils.io import file as file_mod
from ao_shaping.utils.io import cli_helpers, timestamp

FULL = re.compile(r"^\d{8}_\d{6}$")
DATE = re.compile(r"^\d{8}$")


# ---------------------------------------------------------------------------
# the canonical primitive
# ---------------------------------------------------------------------------


def test_format_ts_defaults_to_a_full_timestamp() -> None:
    assert FULL.match(timestamp.format_ts())


def test_format_ts_accepts_an_explicit_datetime_and_format() -> None:
    ts = datetime(2024, 1, 2, 3, 4, 5)
    assert timestamp.format_ts(ts) == "20240102_030405"
    assert timestamp.format_ts(ts, "%Y%m%d") == "20240102"
    assert timestamp.format_ts(ts, "%H%M%S") == "030405"


def test_format_ts_and_now_agree_on_the_same_instant() -> None:
    """The `ts=None` path must not use a different clock than the `ts` path."""
    ts = datetime.now()
    # tolerance, not equality: the two calls are microseconds apart
    assert abs(timestamp.parse_stamp(timestamp.format_ts(ts)) - ts).total_seconds() < 1.0


# ---------------------------------------------------------------------------
# D1 -- the public names are the same function
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("module", "name", "pattern"),
    [
        (file_mod, "gen_date_str", FULL),
        (cli_helpers, "get_timestamp_str", FULL),
    ],
)
def test_timestamp_names_agree_and_match_the_pattern(
    module, name: str, pattern: re.Pattern[str]
) -> None:
    value = getattr(module, name)()
    assert pattern.match(value), f"{name}() produced {value!r}"


def test_the_two_timestamp_spellings_are_byte_identical() -> None:
    """Not merely similar -- the same string from the same instant.

    They call `datetime.now()` independently, so equality can only hold when both
    land in the same second. Retried briefly rather than asserted once.
    """
    for _ in range(50):
        a = file_mod.gen_date_str()
        b = cli_helpers.get_timestamp_str()
        if a == b:
            return
    pytest.fail("gen_date_str() and get_timestamp_str() never agreed")


@pytest.mark.parametrize(
    ("module", "name"),
    [
        (file_mod, "gen_date_str"),
        (cli_helpers, "get_timestamp_str"),
        (cli_helpers, "get_date_dir_name"),
    ],
)
def test_date_name_does_not_reimplement_strftime(module, name: str) -> None:
    """A copy here is the regression this commit exists to prevent.

    Asserted as "no raw strftime in the body" rather than "calls a particular
    name", so renaming the shared helper cannot silently pass this while a
    second implementation creeps back in.
    """
    body = _function_body(Path(module.__file__).read_text(encoding="utf-8"), name)
    assert "strftime" not in body, f"{module.__name__}.{name} reimplements the format"


@pytest.mark.parametrize("module", [file_mod, cli_helpers])
def test_neither_module_calls_strftime_outside_the_shared_module(module) -> None:
    source = Path(module.__file__).read_text(encoding="utf-8")
    offenders = [
        i
        for i, line in enumerate(source.splitlines(), 1)
        if "strftime" in line and not line.lstrip().startswith("#")
    ]
    assert not offenders, f"{module.__name__} still formats inline at lines {offenders}"


def _function_body(source: str, name: str) -> str:
    """Crude but sufficient: the `def name(...)` block up to the next top-level def."""
    lines = source.splitlines()
    start = next(
        (i for i, l in enumerate(lines) if l.startswith(f"def {name}(")), None
    )
    assert start is not None, f"def {name} not found"
    rest = lines[start + 1 :]
    end = next(
        (i for i, l in enumerate(rest) if l.startswith(("def ", "@", "class "))), len(rest)
    )
    return "\n".join(lines[start : start + 1 + end])


# ---------------------------------------------------------------------------
# D2 -- two directory conventions that must NOT be collapsed
# ---------------------------------------------------------------------------


def test_gen_date_dir_stamps_a_full_timestamp(tmp_path: Path) -> None:
    d = file_mod.gen_date_dir(tmp_path)
    assert d.parent == tmp_path
    assert FULL.match(d.name), d.name
    assert d.is_dir()


def test_create_save_dir_stamps_a_bare_date(tmp_path: Path) -> None:
    d = cli_helpers.create_save_dir(tmp_path, "wf")
    assert d.parent == tmp_path / "wf"
    assert DATE.match(d.name), d.name
    assert d.is_dir()


def test_the_two_directory_conventions_differ_in_granularity(tmp_path: Path) -> None:
    """Documented, intentional divergence.

    `gen_date_dir` gives every run its own second-resolution directory;
    `create_save_dir` buckets a whole day. Merging them would either merge
    same-day runs into one directory or rename every existing output tree, so
    both spellings stay and only their implementation is shared.
    """
    assert not FULL.match(cli_helpers.create_save_dir(tmp_path, "a").name)
    assert FULL.match(file_mod.gen_date_dir(tmp_path).name)


@pytest.mark.parametrize("as_str", [True, False])
def test_gen_date_dir_accepts_str_and_path(tmp_path: Path, as_str: bool) -> None:
    base = str(tmp_path) if as_str else tmp_path
    d = file_mod.gen_date_dir(base)
    assert isinstance(d, Path)
    assert d.is_dir()


def test_both_builders_are_idempotent(tmp_path: Path) -> None:
    """Two calls in the same second must not raise on an existing directory."""
    for _ in range(3):
        assert file_mod.gen_date_dir(tmp_path).is_dir()
        assert cli_helpers.create_save_dir(tmp_path, "wf").is_dir()


def test_builders_create_missing_parents(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c"
    assert file_mod.gen_date_dir(nested).is_dir()
    assert cli_helpers.create_save_dir(nested, "wf").is_dir()