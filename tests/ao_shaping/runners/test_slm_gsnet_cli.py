"""CLI surface tests for the B2-converted slm-gsnet runner (2026-09).

``run`` is a click group with ``spgd`` / ``heuristic`` subcommands; options
live on the subcommands. Locks: group help lists both subcommands, spgd
options never leak into heuristic (and vice-versa), and the main ``cli``
group registers ``slm-gsnet``.
"""

from __future__ import annotations

from click.testing import CliRunner

from ao_shaping.runners.slm_gsnet_runner import run

SHARED_OPTIONS = (
    "-e",
    "--epochs",
    "--cam_type",
    "--exposure_time_ms",
    "--cam_size",
    "--slm_number",
    "--slm_wavelength",
    "--zernike_radius",
    "--target-side",
    "--side-factor",
    "--w_uniformity",
    "--w_efficiency",
    "--w_aspect",
)


def test_group_help_lists_both_subcommands():
    result = CliRunner().invoke(run, ["--help"])

    assert result.exit_code == 0, result.output
    assert "spgd" in result.output
    assert "heuristic" in result.output


def test_spgd_subcommand_option_surface():
    result = CliRunner().invoke(run, ["spgd", "--help"])

    assert result.exit_code == 0, result.output
    for opt in SHARED_OPTIONS + ("--delta", "--lr", "--optimizer_type"):
        assert opt in result.output, f"missing spgd option: {opt}"
    # heuristic-only knobs must not leak into spgd
    for opt in ("--algorithm", "--pop_size"):
        assert opt not in result.output, f"heuristic-only option leaked: {opt}"


def test_heuristic_subcommand_option_surface():
    result = CliRunner().invoke(run, ["heuristic", "--help"])

    assert result.exit_code == 0, result.output
    for opt in SHARED_OPTIONS + ("--algorithm", "--pop_size"):
        assert opt in result.output, f"missing heuristic option: {opt}"
    # spgd-only knobs must not leak into heuristic
    for opt in ("--delta", "--optimizer_type"):
        assert opt not in result.output, f"spgd-only option leaked: {opt}"


def test_main_cli_registers_slm_gsnet():
    from ao_shaping.main import cli

    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0, result.output
    assert "slm-gsnet" in result.output