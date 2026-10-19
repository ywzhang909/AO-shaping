"""CLI surface tests for the B2-converted slm-gsnet runner (2026-09).

``run`` is a click group with ``spgd`` / ``heuristic`` subcommands; options
live on the subcommands. Locks: group help lists both subcommands, spgd
options never leak into heuristic (and vice-versa), and the main ``cli``
group registers ``slm-gsnet``.
"""

from __future__ import annotations

from click.testing import CliRunner
import click
import pytest

from ao_shaping.runners.runner_common import CameraParams
from ao_shaping.runners.slm.gsnet_runner import _validate_daheng_exposure, run

SHARED_OPTIONS = (
    "-e",
    "--epochs",
    "--cam-type",
    "--exposure-time-ms",
    "--cam-size",
    "--slm-number",
    "--slm-wavelength",
    "--zernike-radius",
    "--target-side",
    "--side-factor",
    "--w-uniformity",
    "--w-efficiency",
    "--w-aspect",
)


def test_group_help_lists_both_subcommands():
    result = CliRunner().invoke(run, ["--help"])

    assert result.exit_code == 0, result.output
    assert "spgd" in result.output
    assert "heuristic" in result.output


def test_spgd_subcommand_option_surface():
    result = CliRunner().invoke(run, ["spgd", "--help"])

    assert result.exit_code == 0, result.output
    for opt in SHARED_OPTIONS + ("--delta", "--lr", "--optimizer-type"):
        assert opt in result.output, f"missing spgd option: {opt}"
    # heuristic-only knobs must not leak into spgd
    for opt in ("--algorithm", "--pop-size"):
        assert opt not in result.output, f"heuristic-only option leaked: {opt}"


def test_heuristic_subcommand_option_surface():
    result = CliRunner().invoke(run, ["heuristic", "--help"])

    assert result.exit_code == 0, result.output
    for opt in SHARED_OPTIONS + ("--algorithm", "--pop-size"):
        assert opt in result.output, f"missing heuristic option: {opt}"
    # spgd-only knobs must not leak into heuristic
    for opt in ("--delta", "--optimizer-type"):
        assert opt not in result.output, f"spgd-only option leaked: {opt}"


def test_main_cli_registers_slm_gsnet():
    from ao_shaping.main import cli

    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0, result.output
    assert "slm-gsnet" in result.output


@pytest.mark.parametrize("exposure", ["auto", "AUTO", "default"])
def test_daheng_auto_exposure_rejected_before_hardware_open(exposure):
    """Both non-numeric tokens are refused by name.

    Before the three-state vocabulary this gate refused a non-positive number;
    ``auto``/``default`` together with a literal ``0`` all meant "no fixed
    value", so refusing only one of the tokens would have opened the camera
    where the old code refused.
    """
    with pytest.raises(click.BadParameter, match="explicit positive"):
        _validate_daheng_exposure(CameraParams(cam_type="daheng", exposure_time_ms=exposure))


@pytest.mark.parametrize("exposure", [0, 0.0, -0.01, "0", "-1", "", "abc"])
def test_unusable_exposure_is_rejected_by_the_shared_resolver(exposure):
    """``0`` is no longer a spelling of either token -- it is an error naming them.

    This is strictly stronger than the pre-change gate, which silently accepted
    ``0`` for a non-Daheng backend and rejected it here with a message that
    never mentioned an alternative.
    """
    with pytest.raises(click.BadParameter, match="auto"):
        _validate_daheng_exposure(CameraParams(cam_type="daheng", exposure_time_ms=exposure))


@pytest.mark.parametrize("exposure", [0.401, 1.5])
def test_daheng_exposure_above_env_ceiling_rejected(exposure, monkeypatch):
    monkeypatch.setenv("AO_DAHENG_MAX_EXPOSURE_MS", "0.4")
    with pytest.raises(click.BadParameter, match="0.4"):
        _validate_daheng_exposure(CameraParams(cam_type="daheng", exposure_time_ms=exposure))


def test_daheng_exposure_accepts_safe_fixed_value_and_sim_auto():
    _validate_daheng_exposure(CameraParams(cam_type="daheng", exposure_time_ms=0.02))
    # A non-Daheng backend still lets both tokens through: the gate exists
    # because Daheng clamps rather than auto-exposes.
    _validate_daheng_exposure(CameraParams(cam_type="sim", exposure_time_ms="auto"))
    _validate_daheng_exposure(CameraParams(cam_type="sim", exposure_time_ms="default"))


def test_live_cli_rejects_over_limit_exposure_without_opening_devices(monkeypatch):
    monkeypatch.setenv("AO_DAHENG_MAX_EXPOSURE_MS", "0.4")
    result = CliRunner().invoke(
        run, ["spgd", "--cam-type", "daheng", "--exposure-time-ms", "0.5"]
    )
    assert result.exit_code != 0
    assert "require exposure in (0, 0.4] ms" in result.output
