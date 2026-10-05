"""Pin the ``micro-dm-collect`` CLI contract introduced by the R-33 migration.

The migration moved this tool from loose ``@click.option`` kwargs onto the
canonical ``with_params`` dataclass binding in
:mod:`ao_shaping.utils.cli.params`. Two things about that binding are easy to
get wrong and invisible to the on-disk-layout tests in the sibling module:

1. **A dataclass field without a default is NOT automatically required.**
   ``with_params`` only forwards a ``default=`` when the field actually has one
   (``cli_params._patch_defaults``). An option whose field has no default is
   therefore registered as *optional*, and click hands the callback ``None``.
   ``--voltage`` is the field this hit: the collector would then push
   ``None`` volts to every channel instead of refusing to start.

2. The fix must follow the convention the rest of the repo already uses rather
   than teaching the shared mechanism a new rule: declare ``required=True``
   inside ``option(...)`` and give the field a ``default_factory`` sentinel,
   exactly as ``runner_common.FullVoltageRunnerParams.alt_voltage`` and
   ``AltVoltageRunnerParams.ip`` do. Changing ``cli_params`` instead would
   silently alter every ``with_params`` command in the repo, so these tests
   exist to hold the local declaration to the established pattern.

No hardware is touched: the command is only introspected and invoked far enough
for click to reject the missing option, which happens before any device opens.
"""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner
from dataclasses import MISSING

from ao_shaping.tools.micro_dm.micro_dm_image_collect import (
    MicroDMImageCollectParams,
    run,
)


def test_run_is_a_click_command_named_micro_dm_collect():
    assert isinstance(run, click.Command)
    assert run.name == "micro-dm-collect"


def test_voltage_is_required_on_the_command():
    """The regression this module exists for: --voltage must be required.

    Before the fix ``with_params`` left ``required`` unset, so ``required``
    was ``[]`` and a bare ``micro-dm-collect`` proceeded with ``voltage=None``.
    """
    required = [p.name for p in run.params if getattr(p, "required", False)]
    assert required == ["voltage"]


def test_bare_invocation_is_rejected_before_any_device_opens():
    result = CliRunner().invoke(run, [])
    assert result.exit_code == 2
    assert "Missing option" in result.output
    assert "--voltage" in result.output


def test_voltage_uses_the_repo_wide_required_sentinel_convention():
    """Required-ness is declared locally, not inferred by the shared mechanism.

    A ``required=True`` in ``option()`` plus a ``default_factory`` sentinel is
    the pattern ``runner_common`` already uses; this asserts we did not drift
    from it (e.g. by leaving the field default-less and hoping click infers it).
    """
    field = MicroDMImageCollectParams.__dataclass_fields__["voltage"]
    # A default_factory (not a plain default) is what keeps the dataclass
    # constructible while click still enforces the option.
    assert field.default is MISSING
    assert callable(field.default_factory)

    option = next(p for p in run.params if p.name == "voltage")
    assert option.required is True
    assert MicroDMImageCollectParams(voltage=5.0).voltage == 5.0


def test_all_seventeen_options_are_declared():
    assert len(run.params) == len(MicroDMImageCollectParams.__dataclass_fields__) == 17


def test_help_does_not_crash_and_lists_voltage():
    result = CliRunner().invoke(run, ["--help"])
    assert result.exit_code == 0
    assert "--voltage" in result.output


@pytest.mark.parametrize("bad_channel", ["abc", "-1", "999"])
def test_channel_choice_is_still_validated(bad_channel: str):
    """`--channel` keeps its own bounds check; the migration must not bypass it."""
    result = CliRunner().invoke(
        run, ["--voltage", "5", "--channel", bad_channel]
    )
    assert result.exit_code != 0