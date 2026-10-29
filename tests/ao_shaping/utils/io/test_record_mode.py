"""``RecordMode`` is defined once and behaves like the historical strings.

``ao_shaping.utils.io.record_mode`` is the only definition of the ``"max"`` /
``"min"`` vocabulary. Issue #130 was filed because a second copy had grown in
``utils/image/target/objective.py`` and because 24 call sites still spelled the
literals out; these tests pin the three contracts that make the single copy
possible:

1. identity -- every import path reaches the *same* class object;
2. the ``str`` mixin -- old call sites that compare ``== "max"`` keep working;
3. :meth:`RecordMode.coerce` -- the legacy ``self.mode == "max"`` semantics,
   where anything that is not exactly ``"max"`` falls through to ``"min"``.
"""

from __future__ import annotations

import json

import pytest

from ao_shaping.utils.io import record_mode
from ao_shaping.utils.io.file import Recorder
from ao_shaping.utils.io.record_mode import RecordMode


def test_file_module_reuses_the_single_definition() -> None:
    # ``utils/io/file.py`` used to declare its own ``class RecordMode(str, Enum)``.
    # The rename to a shared class is only safe if the identity is the same object:
    # two structurally equal enums are still different types (``isinstance`` and
    # ``is`` comparisons across them fail).
    import ao_shaping.utils.io.file as file_module

    assert file_module.RecordMode is RecordMode
    assert file_module.RecordMode is record_mode.RecordMode


def test_utils_package_lazy_attribute_reaches_the_same_class() -> None:
    # ``ao_shaping.utils.__init__`` resolves ``RecordMode`` lazily (R-27/PEP-562);
    # a stale target there would hand callers a second class object.
    import ao_shaping.utils as utils

    assert utils.RecordMode is RecordMode


def test_members_compare_equal_to_the_historical_strings() -> None:
    # 24 call sites passed the literal and every sidecar/CSV spells it out.
    assert RecordMode.MAX == "max"
    assert RecordMode.MIN == "min"
    assert isinstance(RecordMode.MAX, str)
    assert "x" + RecordMode.MAX == "xmax"


def test_str_renders_the_wire_value() -> None:
    # ``ShapingObjectiveParams.Mode.from_string`` and the CSV/JSON sidecars read
    # ``str(mode)``; ``RecordMode.MAX`` would break both.
    assert str(RecordMode.MAX) == "max"
    assert str(RecordMode.MIN) == "min"
    assert f"{RecordMode.MAX}" == "max"
    assert json.dumps({"mode": RecordMode.MAX}) == '{"mode": "max"}'


@pytest.mark.parametrize(
    "value,expected",
    [
        (RecordMode.MAX, RecordMode.MAX),
        (RecordMode.MIN, RecordMode.MIN),
        ("max", RecordMode.MAX),
        ("min", RecordMode.MIN),
        (" MAX ", RecordMode.MAX),
        ("Min", RecordMode.MIN),
        # Legacy semantics: ``self.mode == "max"`` is the only ascent trigger, so
        # everything unrecognised must land on MIN -- never on MAX.
        (None, RecordMode.MIN),
        ("", RecordMode.MIN),
        ("oops", RecordMode.MIN),
        ("maximum", RecordMode.MIN),
    ],
)
def test_coerce_preserves_the_legacy_fallthrough(
    value: object, expected: RecordMode
) -> None:
    assert RecordMode.coerce(value) is expected


def test_maximizing_and_sign_agree_with_the_member() -> None:
    assert RecordMode.MAX.maximizing is True
    assert RecordMode.MIN.maximizing is False
    assert RecordMode.MAX.sign == -1.0
    assert RecordMode.MIN.sign == 1.0
    # ``sign`` feeds ``objective + sign * delta``; the two must stay complementary.
    assert RecordMode.MAX.sign == -RecordMode.MIN.sign


def test_recorder_forwards_the_mode_through_the_enum() -> None:
    # The constructor accepts the literal for backward compatibility but stores a
    # member, so ``get_best_iter`` no longer compares against a bare string.
    for raw in ("max", "min", RecordMode.MAX, RecordMode.MIN):
        assert isinstance(Recorder(mode=raw).mode, RecordMode)
    assert Recorder(mode="max").mode is RecordMode.MAX
    assert Recorder(mode="min").mode is RecordMode.MIN


@pytest.mark.parametrize("mode", [RecordMode.MAX, "max"])
def test_get_best_iter_ascends_for_max(mode: object) -> None:
    recorder = Recorder(mark="score", mode=mode)  # type: ignore[arg-type]
    for score in (1.0, 3.0, 2.0):
        recorder.append({"score": score})
    _, (index, value) = recorder.get_best_iter("score")
    assert (int(index), value) == (1, 3.0)


@pytest.mark.parametrize("mode", [RecordMode.MIN, "min", None])
def test_get_best_iter_descends_for_min(mode: object) -> None:
    recorder = Recorder(mark="score", mode=mode)  # type: ignore[arg-type]
    for score in (1.0, 3.0, 2.0):
        recorder.append({"score": score})
    _, (index, value) = recorder.get_best_iter("score")
    assert (int(index), value) == (0, 1.0)
