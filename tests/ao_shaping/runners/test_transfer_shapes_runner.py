"""The transfer-shapes runner: registration, plan-only behaviour, stage recipe."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from ao_shaping.runners.slm.transfer_shapes_runner import STAGE_RECIPES, run

REPO = Path(__file__).resolve().parents[3]


def test_the_command_is_registered_on_the_top_level_group():
    from ao_shaping.main import cli

    assert "slm-transfer-shapes" in cli.commands
    assert cli.commands["slm-transfer-shapes"] is run


def test_importing_the_runner_does_not_import_a_device_driver():
    """The runner is imported by ``main.py`` at start-up; it must stay device-free."""
    code = (
        "import sys\n"
        "import ao_shaping.runners.slm.transfer_shapes_runner as module\n"
        "leaked = sorted(n for n in sys.modules if n.startswith('ao_shaping.drivers'))\n"
        "assert not leaked, leaked\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_plan_only_prints_the_plan_and_opens_no_device(tmp_path):
    result = CliRunner().invoke(run, ["--out", str(tmp_path / "session"),
                                      "--shapes", "square", "--shapes", "circle"])
    assert result.exit_code == 0, result.output
    assert "未指定 --execute" in result.output
    plan = json.loads(result.output[result.output.index("{"):result.output.rindex("}") + 1])
    assert plan["shapes"] == ["square", "circle"]
    assert plan["execute"] is False
    # Nothing was written: the plan path must not create the session directory.
    assert not (tmp_path / "session").exists()


def test_the_stage_recipe_is_the_square1200_chain():
    names = [recipe.name for recipe in STAGE_RECIPES]
    assert names == ["transport_cv", "mono12_cv", "mono12_continue", "mono12_dense"]
    first, second, third, fourth = STAGE_RECIPES
    # Stage 1 accepts on the model's own loss (no CCD verdict measured).
    assert (first.ccd_check, first.bit_depth, first.accept_metric) == (False, 8, "loss")
    # Stages 2-4 accept only when the measured core CV drops.
    for recipe in (second, third, fourth):
        assert recipe.ccd_check and recipe.bit_depth == 12 and recipe.accept_metric == "cv"
    # Only the last stage uses the wide-ramp model and the finer sampling.
    assert [recipe.wide_model for recipe in STAGE_RECIPES] == [False, False, False, True]
    assert [recipe.window_step for recipe in STAGE_RECIPES] == [4.0, 4.0, 4.0, 2.0]
    assert fourth.core_margin == 8.0
    assert fourth.seed_policy == "transfer"


def test_the_shape_vocabulary_of_the_command_is_the_transfer_family():
    from ao_shaping.utils.image.target import TRANSFER_SHAPE_CHOICES

    choices = {param.type.choices for param in run.params if param.name == "shapes"}
    assert choices == {tuple(TRANSFER_SHAPE_CHOICES)}
    assert "circle" in TRANSFER_SHAPE_CHOICES


@pytest.mark.parametrize("path", ["src/ao_shaping/runners/slm/transfer_shapes_runner.py",
                                  "src/ao_shaping/tools/slm/transfer/recording.py"])
def test_the_new_modules_are_tracked_by_the_repository(path):
    """The convention guard reads the git index; a new file must be added to it.

    ``cwd=REPO`` is load-bearing: the container harness starts pytest in its scratch
    directory (``run_pytest_ci.sh`` passes ``-w "$SCRATCH"``), so a git call without a
    working directory answers "fatal: not a git repository" with rc=128 -- which this test
    used to report as "not tracked by git", sending the runners shard red over files that
    are plainly in the index. A checkout where git cannot answer at all is now skipped with
    that reason instead of being mistaken for a missing file (the tri-state pattern already
    used by test_report_provenance._is_gitignored).
    """
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", path],
        capture_output=True, text=True, cwd=str(REPO), check=False,
    )
    if result.returncode not in (0, 1):
        pytest.skip(
            f"git cannot answer for this checkout (rc={result.returncode}): "
            f"{result.stderr.strip()}"
        )
    assert result.returncode == 0, (
        f"{path} is not tracked by git -- add it with `git add {path}`"
    )
