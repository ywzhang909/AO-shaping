"""The transfer-shapes runner: registration, plan-only behaviour, stage recipe."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from ao_shaping.runners.slm.transfer_shapes_runner import STAGE_RECIPES, _Journal, run

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


# ---------------------------------------------------------------------------
# resume / stage order: a half-finished session must be continuable
# ---------------------------------------------------------------------------


def _stage_args(tmp_path, *, stage_resume=True, order="stage", shapes=("square", "circle")):
    """Everything ``run_stages`` needs, with paths that never touch a device."""
    return dict(
        out=tmp_path, shapes=list(shapes), order=order, resume=stage_resume,
        journal=_Journal(path=tmp_path / "journal.json"),
        chosen={shape: "backprojection" for shape in shapes},
        model_640=tmp_path / "model640", wide_dir=tmp_path / "model_wide640",
        zero_xy=[10., 10.], initial_phase="auto", device="cpu", camera="custom",
        cam_id=0, slm_number=1, frames=2, side=20., period=64., window_span=60.,
        zero_radius=0., window_step=4., recorder=None)


def _record_entered_stages(monkeypatch):
    """Replace ``validate.run`` with a stub that only leaves a manifest behind."""
    from ao_shaping.runners.slm import transfer_shapes_runner as module

    entered = []

    def fake_validate(args):
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "manifest.json").write_text("[]")
        entered.append((args.out.parent.name, args.out.name, args.seed_run, args.resume))

    monkeypatch.setattr(module.validate_stage, "run", fake_validate)
    return entered


def _fake_stage(tmp_path, shape, recipe_name):
    stage_dir = tmp_path / "shape" / shape / recipe_name
    stage_dir.mkdir(parents=True)
    (stage_dir / "manifest.json").write_text("[]")
    return stage_dir


def test_resume_reuses_a_stage_whose_manifest_exists(tmp_path, monkeypatch):
    """A finished stage is journaled ``reused`` and never re-entered."""
    from ao_shaping.runners.slm.transfer_shapes_runner import STAGE_RECIPES, run_stages

    entered = _record_entered_stages(monkeypatch)
    frozen = _fake_stage(tmp_path, "square", STAGE_RECIPES[0].name)

    run_stages(**_stage_args(tmp_path, shapes=("square",)))

    states = [entry["status"] for entry in json.loads((tmp_path / "journal.json").read_text())]
    reused = [entry for entry in json.loads((tmp_path / "journal.json").read_text())
              if entry["status"] == "reused"]
    assert [entry["stage"] for entry in reused] == [f"square/{STAGE_RECIPES[0].name}"]
    assert reused[0]["out"] == str(frozen)
    assert states.count("running") == len(STAGE_RECIPES) - 1
    # Stage 2 continued from the reused directory, not from nothing.
    assert entered[0][:2] == ("square", STAGE_RECIPES[1].name)
    assert entered[0][2] == frozen


def test_without_resume_a_populated_stage_directory_still_fails(tmp_path):
    """The historical contract is unchanged: no --resume means no reuse.

    This drives the production ``validate.run`` itself, with a stub forward
    model, and nothing else stubbed out: the ``FileExistsError`` is the real
    ``mkdir(parents=True, exist_ok=resume)`` guard, and it now lives inside
    ``validate.run`` instead of the runner.
    """
    from ao_shaping.runners.slm.transfer_shapes_runner import run_stages
    from ao_shaping.tools.slm.transfer import validate

    model = tmp_path / "model640"
    model.mkdir()
    (model / "frozen_parameters.json").write_text(json.dumps(dict(
        residual_noll_radians={}, defocus_diopter=0., origin=[0., 0.],
        amplitude_center=[0., 0.], amplitude_width=[1., 1.], gain=1., background=0.,
        sigma_ccd_px=1., ramp_matrix=[[1., 0.], [0., 1.]],
        grid_size=8, fft_size=16, pixel_samples=3)))
    frozen = _fake_stage(tmp_path, "square", STAGE_RECIPES[0].name)
    args = _stage_args(tmp_path, stage_resume=False, shapes=("square",))
    args["initial_phase"] = "random"
    with pytest.raises(FileExistsError):
        run_stages(**args)
    assert validate.run is not None
    assert frozen.is_dir()


def test_stage_major_order_visits_every_shape_at_one_stage_before_the_next(tmp_path, monkeypatch):
    from ao_shaping.runners.slm.transfer_shapes_runner import (
        STAGE_RECIPES, _stage_sequence, run_stages,
    )

    expected = [(shape, index) for index in range(len(STAGE_RECIPES))
                for shape in ("square", "circle")]
    assert _stage_sequence(["square", "circle"], "stage") == expected
    assert _stage_sequence(["square", "circle"], "shape") == [
        (shape, index) for shape in ("square", "circle")
        for index in range(len(STAGE_RECIPES))]

    entered = _record_entered_stages(monkeypatch)
    run_stages(**_stage_args(tmp_path, order="stage"))
    assert [(shape, recipe) for shape, recipe, *_ in entered] == [
        (shape, STAGE_RECIPES[index].name) for shape, index in expected]


def test_the_new_flags_reach_the_plan_without_opening_a_device(tmp_path):
    result = CliRunner().invoke(run, ["--out", str(tmp_path / "session"), "--shapes", "square",
                                      "--resume", "--order", "stage"])
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output[result.output.index("{"):result.output.rindex("}") + 1])
    assert (plan["resume"], plan["order"]) == (True, "stage")
    # --resume makes the plan say what is actually left, in stage-major order.
    assert plan["session_plan"]["frozen"] == []
    assert [(entry[0], entry[3]) for entry in plan["session_plan"]["todo"]] == [
        ("square", recipe.blocks) for recipe in STAGE_RECIPES]
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
