"""Characterization tests pinning the ``with_params`` click mechanism contract.

``with_params`` (``ao_shaping.runners.runner_common``) turns dataclass fields
declared as ``name: Annotated[T, option(...)] = default`` into click options and
delivers a fully-populated parameter instance to the wrapped command under a
keyword named by ``kw_name``. It is about to be moved out of
``runner_common.py``; these tests lock the behaviour that silently depends on
**four different escape hatches** for union-typed fields, plus the three
``TypeError`` raise sites that keep the convention honest.

Union-typed fields in the repo, and the single reason each one imports at all:

| Field                                | Annotation                       | Escape hatch              |
|--------------------------------------|----------------------------------|---------------------------|
| ``WfsParams.pupil_center``           | ``str \\| tuple[float, float]``      | ``callback=parse_tuple``  |
| ``PibRunnerParams.center``           | ``str \\| tuple[float, float] \\| None`` | ``callback=parse_tuple``  |
| ``CameraParams.center``              | ``str \\| tuple[int, int] \\| None``    | ``type=click.STRING``     |
| ``CameraParamsPib.center``           | ``str \\| tuple[int, int] \\| None``    | ``type=click.STRING``     |

Remove any one escape hatch and the *module import itself* raises, because the
decoration happens at import time. That is the point: the coupling is
load-bearing, and the falsification half of this file's history (``git log``)
records each of them going red.

No hardware is touched: ``runner_common`` only *imports* driver modules at
module scope (registry + panel constants), and constructing a click command
never opens a device.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any

import click
import pytest
from click.testing import CliRunner

from ao_shaping.runners.runner_common import (
    DM_TYPES,
    DM_TYPES_PRE_ASYN_MICRO,
    CameraParams,
    CameraParamsPib,
    ClickGroup,
    PibRunnerParams,
    WfsParams,
    option,
    with_params,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def build_command(arg_class: type, name: str) -> tuple[click.Command, list[Any]]:
    """Wrap ``arg_class`` in a throwaway click command and capture the instance.

    Mirrors the production idiom exactly (``@click.command(...)`` stacked
    *above* ``@with_params(Config, kw_name=...)``, as in
    ``slm_gsnet_runner.spgd`` / ``slm_pib_runner``). The returned list receives
    the delivered parameter instance so the test can assert on the *parsed*
    value without touching any device.
    """
    captured: list[Any] = []

    @click.command(name=name)
    @with_params(arg_class, kw_name="params")
    def command(params: Any) -> None:
        captured.append(params)

    return command, captured


def parse_with(arg_class: type, argv: list[str], name: str) -> Any:
    """Invoke a command built from ``arg_class`` and return the delivered instance."""
    command, captured = build_command(arg_class, name)
    result = CliRunner().invoke(command, argv)
    assert result.exit_code == 0, result.output
    assert len(captured) == 1, f"command body ran {len(captured)} times"
    return captured[0]


# ---------------------------------------------------------------------------
# A. behavioural pins — the four union couplings
# ---------------------------------------------------------------------------


class TestUnionCouplings:
    """Each union field survives import only via its own escape hatch."""

    def test_wfs_pupil_center_callback_yields_float_tuple(self) -> None:
        """``WfsParams.pupil_center`` is saved by ``callback=parse_tuple``.

        A 2-member union with *no* ``None``, so ``_strip_optional`` cannot
        reduce it — only the callback can turn ``"960,600"`` into a tuple.
        """
        params = parse_with(WfsParams, ["-c", "960,600"], "wfs-params")

        assert isinstance(params.pupil_center, tuple)
        assert params.pupil_center == (960.0, 600.0)
        assert all(isinstance(v, float) for v in params.pupil_center)

    def test_pib_runner_center_callback_yields_float_tuple(self) -> None:
        """``PibRunnerParams.center`` is saved by ``callback=parse_tuple``."""
        params = parse_with(PibRunnerParams, ["--center", "3,4"], "pib-params")

        assert isinstance(params.center, tuple)
        assert params.center == (3.0, 4.0)

    def test_camera_params_center_explicit_string_type_stays_string(self) -> None:
        """``CameraParams.center`` keeps the raw string via ``type=click.STRING``.

        The explicit ``type=`` short-circuits ``_patch_click_types`` before
        ``_strip_optional`` is reached, so the union annotation is never
        inspected and the value stays an unparsed ``str``. Consumers call
        ``parse_center()`` later, deliberately.
        """
        params = parse_with(CameraParams, ["--center", "3,4"], "camera-params")

        assert isinstance(params.center, str)
        assert not isinstance(params.center, tuple)
        assert params.center == "3,4"

    def test_camera_params_pib_center_explicit_string_type_stays_string(self) -> None:
        """``CameraParamsPib.center`` overrides the parent with the same hatch."""
        params = parse_with(
            CameraParamsPib, ["--center", "3,4"], "camera-params-pib"
        )

        assert isinstance(params.center, str)
        assert not isinstance(params.center, tuple)
        assert params.center == "3,4"


class TestKwNameContract:
    """``kw_name`` is keyword-only and required — the delivery key."""

    def test_kw_name_must_be_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            with_params(CameraParams, "params")  # type: ignore[misc]

    def test_kw_name_is_required(self) -> None:
        with pytest.raises(TypeError):
            with_params(CameraParams)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# B. negative pins — the raise sites are real, not decorative
# ---------------------------------------------------------------------------


@dataclass
class TwoMemberUnionParams:
    """A union with two concrete members and no escape hatch: rejected."""

    x: Annotated[int | str, option("--x", help="Two concrete members.")] = 0


@dataclass
class ThreeMemberUnionParams:
    """A union with three concrete members and no escape hatch: rejected."""

    x: Annotated[int | str | float, option("--x", help="Three concrete members.")] = 0


@dataclass
class DefaultInOptionParams:
    """``default=`` inside ``option(...)`` is rejected: the field owns the default."""

    x: Annotated[int, option("--x", default=5, help="default= belongs on the field.")] = 7


@dataclass
class DuplicateNameChild:
    x: Annotated[int, option("--x", help="Declared by the nested ClickGroup child.")] = 1


@dataclass
class DuplicateNameOuter:
    x: Annotated[int, option("--x", help="Declared by the outer class.")] = 0
    child: Annotated[DuplicateNameChild, ClickGroup()] = field(
        default_factory=DuplicateNameChild
    )


class TestNegativePins:
    """Each negative case must raise ``TypeError`` at decoration time."""

    def test_two_member_union_without_hatch_raises(self) -> None:
        """``_strip_optional`` rejects a union without exactly one non-None member."""
        with pytest.raises(TypeError, match="union"):
            build_command(TwoMemberUnionParams, "neg-two-member-union")

    def test_three_member_union_without_hatch_raises(self) -> None:
        with pytest.raises(TypeError, match="union"):
            build_command(ThreeMemberUnionParams, "neg-three-member-union")

    def test_default_inside_option_raises(self) -> None:
        """``_patch_defaults`` refuses a ``default=`` passed inside ``option(...)``."""
        with pytest.raises(TypeError, match="default must live on the dataclass field"):
            build_command(DefaultInOptionParams, "neg-default-in-option")

    def test_duplicate_click_name_in_group_tree_raises(self) -> None:
        """``_add`` refuses a child option that shadows an existing parent name."""
        with pytest.raises(TypeError, match="declared twice"):
            build_command(DuplicateNameOuter, "neg-duplicate-name")


# ---------------------------------------------------------------------------
# C. DM type registry pins
# ---------------------------------------------------------------------------


class TestDmTypes:
    """``DM_TYPES`` depends on import order plus one side-effect import.

    ``runner_common`` snapshots ``DM_TYPES_PRE_ASYN_MICRO`` *before*
    ``import ao_shaping.drivers.dm.asyn_micro_dm`` registers ``asyn_micro``,
    then re-reads the registry for ``DM_TYPES``. Dropping that import shrinks
    the ``--dm_type`` choice list by one entry with no error.
    """

    def test_pre_asyn_micro_contents(self) -> None:
        assert DM_TYPES_PRE_ASYN_MICRO == [
            "hadamard",
            "micro",
            "nlight",
            "sim",
            "sim_micro",
            "zernike",
        ]

    def test_dm_types_is_sorted_pre_plus_asyn_micro(self) -> None:
        assert DM_TYPES == sorted(DM_TYPES_PRE_ASYN_MICRO + ["asyn_micro"])

    def test_dm_types_adds_exactly_one_entry(self) -> None:
        assert len(DM_TYPES) == len(DM_TYPES_PRE_ASYN_MICRO) + 1
        assert set(DM_TYPES) - set(DM_TYPES_PRE_ASYN_MICRO) == {"asyn_micro"}
