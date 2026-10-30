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
                                  "src/ao_shaping/tools/slm/transfer/recording.py",
                                  "src/ao_shaping/algorithm/signal_processing/weight_schedule.py"])
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

def test_objective_and_weight_schedule_reach_the_plan(tmp_path):
    """Both default to the historical behaviour (cv / constant)."""
    result = CliRunner().invoke(run, ["--out", str(tmp_path / "session"), "--shapes", "square"])
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output[result.output.index("{"):result.output.rindex("}") + 1])
    assert (plan["objective"], plan["weight_schedule"]) == ("cv", "constant")


def test_staged_weights_with_the_cv_objective_is_refused(tmp_path):
    """staged would be a silent no-op on a CV-only loss, so the runner fails."""
    result = CliRunner().invoke(run, ["--out", str(tmp_path / "session"), "--shapes", "square",
                                      "--weight-schedule", "staged"])
    assert result.exit_code != 0
    assert "--weight-schedule staged needs --objective" in result.output


def test_staged_weights_with_a_shaped_objective_reaches_the_plan(tmp_path):
    result = CliRunner().invoke(run, ["--out", str(tmp_path / "session"), "--shapes", "square",
                                      "--objective", "shape", "--weight-schedule", "staged"])
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output[result.output.index("{"):result.output.rindex("}") + 1])
    assert (plan["objective"], plan["weight_schedule"]) == ("shape", "staged")


# -- 光路无光的退出契约 ----------------------------------------------------
#
# 2026-10-28 的失败是一条 lane 花了 48 s 才报一句「曝光不收敛」, 而那句话既没说
# 数字也没说该去查光路还是查参数。这里的契约是: 光路无光 ⇒ 退出码 3 + 实测数字,
# 且**不打印 traceback** (那是预期状态, 不是缺陷)。普通异常仍是 1。

def _dark_path(*, what: str = "capture") -> Exception:
    from ao_shaping.tools.slm.transfer.light_guard import LadderReading, OpticalPathInvalid
    from ao_shaping.utils.image.exposure_control import FrameNoiseFloor

    noise = FrameNoiseFloor(median_counts=0.0, sigma_counts=1.0, threshold_counts=4.0)
    return OpticalPathInvalid(
        reason="flat", cause="blocked", budget=12, what=what,
        readings=(LadderReading(1, 0.02, 0.0, noise), LadderReading(2, 0.08, 0.0, noise)))


def test_a_dark_path_exits_with_three_and_no_traceback(tmp_path, monkeypatch):
    from ao_shaping.tools.slm.transfer import capture as capture_module

    def dark(_args):
        raise _dark_path()

    monkeypatch.setattr(capture_module, "capture", dark)
    result = CliRunner().invoke(run, ["--out", str(tmp_path / "session"),
                                      "--shapes", "square", "--execute"])
    assert result.exit_code == 3, result.output
    assert "光路无光" in result.output
    assert "0.02→0.08 ms" in result.output          # 实测数字, 不是一句结论
    assert "Traceback" not in result.output


def test_reusing_a_model_checks_the_light_before_it_is_loaded(tmp_path, monkeypatch):
    """--model-dir 跳过采集, 于是没有任何人看一眼光路: 启动时必须先查。"""
    from ao_shaping.tools.slm.transfer import capture as capture_module

    entered: list[str] = []

    def dark(**_kwargs):
        entered.append("precheck")
        raise _dark_path(what="model reuse")

    monkeypatch.setattr(capture_module, "precheck_light", dark)
    model = tmp_path / "model"
    model.mkdir()
    result = CliRunner().invoke(run, ["--out", str(tmp_path / "session"), "--shapes", "square",
                                      "--model-dir", str(model), "--execute"])
    assert entered == ["precheck"]
    assert result.exit_code == 3, result.output
    assert "model reuse" in result.output and "Traceback" not in result.output


def test_an_ordinary_failure_keeps_exit_code_one(tmp_path, monkeypatch):
    """别把两个码混成一个: 3 是光路, 1 才是代码/流程出错。"""
    from ao_shaping.tools.slm.transfer import capture as capture_module

    def broken(_args):
        raise RuntimeError("exposure did not converge")

    monkeypatch.setattr(capture_module, "capture", broken)
    result = CliRunner().invoke(run, ["--out", str(tmp_path / "session"),
                                      "--shapes", "square", "--execute"])
    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)
    assert "exposure did not converge" in str(result.exception)


# --------------------------------------------------------------------------- #
# execute_campaign: the executing half the GUI drives from a background thread
# --------------------------------------------------------------------------- #
def test_execute_runs_the_callable_body_not_a_copy(tmp_path, monkeypatch):
    """``run --execute`` must delegate, and hand over *every* click parameter.

    The GUI page calls ``execute_campaign(**CampaignSettings.kwargs(), out=...)``;
    a parameter that ``run`` fails to forward silently falls back to the click
    default, which is exactly the "one lane lost a flag" failure the campaign layer
    exists to prevent.
    """
    from ao_shaping.runners.slm import transfer_shapes_runner as module

    captured: dict = {}

    def fake_execute(**kwargs):
        captured.update(kwargs)
        return {"out": str(kwargs["out"]), "stages": 0, "blocks": 0, "stopped": False}

    monkeypatch.setattr(module, "execute_campaign", fake_execute)
    result = CliRunner().invoke(run, ["--out", str(tmp_path / "s"), "--shapes", "square",
                                      "--side", "200", "--window-span", "420", "--execute"])
    assert result.exit_code == 0, result.output
    assert captured["out"] == (tmp_path / "s").resolve()
    assert captured["shapes"] == ["square"]
    assert captured["side"] == 200.0
    assert captured["window_span"] == 420.0
    forwarded = {param.name for param in run.params} - {"out_dir", "execute", "records_only"}
    assert forwarded <= set(captured), sorted(forwarded - set(captured))


def test_execute_campaign_rejects_a_misspelled_keyword(tmp_path):
    """A typo must fail loudly: a silently ignored key would run with another default."""
    from ao_shaping.runners.slm.transfer_shapes_runner import execute_campaign

    with pytest.raises(TypeError, match="不认识"):
        execute_campaign(out=tmp_path, windo_span=1.0)


def _drive_run_stages(tmp_path, monkeypatch, *, stops_at: int | None = None):
    """Drive ``run_stages`` with ``validate.run`` faked: no device is opened."""
    from ao_shaping.runners.slm import transfer_shapes_runner as module

    entered: list[Path] = []

    def fake_validate(args):
        entered.append(Path(args.out))
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "manifest.json").write_text("[]", encoding="utf-8")

    monkeypatch.setattr(module.validate_stage, "run", fake_validate)
    journal = module._Journal(path=tmp_path / "journal.json")
    calls = {"n": 0}

    def should_stop() -> bool:
        calls["n"] += 1
        return stops_at is not None and calls["n"] >= stops_at

    stopped = module.run_stages(
        out=tmp_path, shapes=["square"], order="shape", resume=False, journal=journal,
        chosen={"square": "transport"}, model_640=tmp_path, wide_dir=tmp_path,
        zero_xy=(0.0, 0.0), initial_phase="auto", device="cpu", camera="sim", cam_id=0,
        slm_number=1, frames=1, side=200.0, period=32.0, window_span=420.0,
        zero_radius=60.0, window_step=4.0, recorder=None, should_stop=should_stop)
    return stopped, entered, journal


def test_no_stop_runs_every_segment(tmp_path, monkeypatch):
    stopped, entered, journal = _drive_run_stages(tmp_path, monkeypatch)
    assert stopped is False
    assert len(entered) == len(STAGE_RECIPES)
    assert [entry["status"] for entry in journal.entries] == ["running", "complete"] * len(STAGE_RECIPES)


def test_should_stop_ends_at_a_segment_boundary_and_journals_stopped(tmp_path, monkeypatch):
    """Stop is segment-granularity: the next segment is journalled ``stopped``, no raise."""
    stopped, entered, journal = _drive_run_stages(tmp_path, monkeypatch, stops_at=2)
    assert stopped is True
    # One full segment ran; the second one was never entered (a block is atomic).
    assert len(entered) == 1
    statuses = [entry["status"] for entry in journal.entries]
    assert statuses == ["running", "complete", "stopped"]
    assert journal.entries[-1]["stage"] == f"square/{STAGE_RECIPES[1].name}"
    assert "operator stop" in journal.entries[-1]["reason"]


# --------------------------------------------------------------------------- #
# 段成本与段间暂停 (issue #214 的数据层接进 runner)
# --------------------------------------------------------------------------- #
def _write_blocks(stage_dir: Path, blocks: int) -> None:
    """写一份有 ``blocks`` 行的段 manifest (第一行是 seed 重测)。"""
    stage_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"block": index} for index in range(blocks)]
    (stage_dir / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")


def test_a_completed_stage_records_its_measured_cost(tmp_path, monkeypatch):
    """Every finished segment leaves a ``cost.json`` of journal time × manifest blocks."""
    from ao_shaping.runners.slm import transfer_shapes_runner as module
    from ao_shaping.tools.slm.transfer.cost_table import COST_FILE, read_campaign_costs

    def fake_validate(args):
        _write_blocks(Path(args.out), blocks=4)

    monkeypatch.setattr(module.validate_stage, "run", fake_validate)
    assert module.run_stages(**_stage_args(tmp_path, shapes=("square",))) is False

    stage_dir = tmp_path / "shape" / "square" / STAGE_RECIPES[0].name
    payload = json.loads((stage_dir / COST_FILE).read_text(encoding="utf-8"))
    assert payload["stage"] == STAGE_RECIPES[0].name
    assert payload["shape"] == "square"
    assert payload["source"] == "measured"
    # manifest 4 行里第一行是 seed 重测, 真正测过的块数是 3.
    assert payload["blocks"] == 3
    assert payload["seconds"] >= 0.0
    assert payload["seconds_per_block"] == pytest.approx(payload["seconds"] / 3)
    # 成本落在段目录里, 扫回来按 <形状>/<阶段> 归并.
    assert set(read_campaign_costs(tmp_path)) == {
        f"square/{recipe.name}" for recipe in STAGE_RECIPES}


def test_a_pause_request_stops_after_the_completed_stage(tmp_path, monkeypatch):
    """The on-disk pause is honoured after a *finished* segment, then cleared."""
    from ao_shaping.runners.slm import transfer_shapes_runner as module
    from ao_shaping.tools.slm.transfer.campaign_control import (
        request_stage_stop,
        stage_stop_requested,
    )
    from ao_shaping.tools.slm.transfer.cost_table import COST_FILE

    entered = _record_entered_stages(monkeypatch)
    request_stage_stop(tmp_path, reason="先看这一段")
    stopped = module.run_stages(**_stage_args(tmp_path, shapes=("square",),
                                              stage_resume=False))

    assert stopped is True
    assert len(entered) == 1
    assert stage_stop_requested(tmp_path) is False
    statuses = [entry["status"] for entry in
                json.loads((tmp_path / "journal.json").read_text(encoding="utf-8"))]
    assert statuses == ["running", "complete", "paused"]
    # 暂停之前这一段已经完整落盘: 产物与成本都在.
    stage_dir = tmp_path / "shape" / "square" / STAGE_RECIPES[0].name
    assert (stage_dir / "manifest.json").is_file()
    assert (stage_dir / COST_FILE).is_file()


def test_resuming_a_paused_run_reuses_the_finished_stage(tmp_path, monkeypatch):
    """``--resume`` picks a paused campaign up at the *next* segment, cost untouched."""
    from ao_shaping.runners.slm import transfer_shapes_runner as module
    from ao_shaping.tools.slm.transfer.campaign_control import request_stage_stop
    from ao_shaping.tools.slm.transfer.cost_table import COST_FILE

    first = _record_entered_stages(monkeypatch)
    request_stage_stop(tmp_path, reason="暂停")
    assert module.run_stages(**_stage_args(tmp_path, shapes=("square",))) is True
    assert len(first) == 1
    cost_path = tmp_path / "shape" / "square" / STAGE_RECIPES[0].name / COST_FILE
    cost_before = json.loads(cost_path.read_text(encoding="utf-8"))

    second = _record_entered_stages(monkeypatch)
    assert module.run_stages(**_stage_args(tmp_path, shapes=("square",))) is False

    assert second[0][:2] == ("square", STAGE_RECIPES[1].name)
    journal = json.loads((tmp_path / "journal.json").read_text(encoding="utf-8"))
    reused = [entry["stage"] for entry in journal if entry["status"] == "reused"]
    assert reused == [f"square/{STAGE_RECIPES[0].name}"]
    # 已测成本是覆盖写的: 续跑不重算, 也不抹掉.
    assert json.loads(cost_path.read_text(encoding="utf-8")) == cost_before


def test_the_released_summary_carries_the_fingerprint_only_when_one_exists(tmp_path):
    """No fingerprint in the session directory ⇒ no field invented at the end."""
    from ao_shaping.tools.slm.transfer.optical_session import save_fingerprint
    from ao_shaping.tools.slm.transfer.session_record import optical_session_field

    assert optical_session_field(tmp_path) == {}

    save_fingerprint({"created_at": "2026-01-01T00:00:00+00:00", "window_span": 820.0},
                     tmp_path)
    field = optical_session_field(tmp_path)
    assert set(field) == {"optical_session"}
    assert field["optical_session"]["window_span"] == 820.0


# ---------------------------------------------------------------------------
# 收尾写侧: 指纹落盘 + 检查点登记 (免设备)
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _writing_session(tmp_path: Path) -> Path:
    """一条只含已有产物的假会话目录 (含一个检查点), 不需要设备。"""
    root = tmp_path / "session"
    _write_json(root / "journal.json", [
        {"stage": "plan", "status": "complete",
         "timestamp_utc": "2026-01-01T00:00:00+00:00",
         "plan": {"shapes": ["square"], "window_span": 820.0, "window_step": 4.0,
                  "zero_radius": 0.0, "period": 128.0, "control_grid": [640, 640],
                  "fft_size": 1024}},
        {"stage": "geometry", "status": "complete", "zero_xy": [101.5, 208.25]},
    ])
    _write_json(root / "shape" / "square" / "capture" / "config.json",
                {"pixel_size_um": 3.45})
    _write_json(root / "shape" / "square" / "capture" / "manifest.json",
                [{"peak": 900.0, "full_scale": 4095.0, "exposure_ms": 0.6},
                 {"peak": 1500.0, "full_scale": 4095.0, "exposure_ms": 0.6}])
    _write_json(root / "fit" / "provenance.json",
                {"grid_size": [320, 320], "fft_size": 1024, "origin": [50.0, 60.0],
                 "ramp_matrix": [[1, 2], [3, 4]]})
    _write_json(root / "camera_probe" / "formats.json",
                {"observations": [{"format": "Mono12", "exposure_ms": 0.6}]})
    _write_json(root / "optical_path_probe.json", {"state": "ok"})
    _write_json(root / "nominal" / "frozen_parameters.json",
                {"ramp_matrix": [[1.0, 0.0], [0.0, 1.0]], "grid_size": [320, 320],
                 "fft_size": 1024})
    return root


def test_session_completion_writes_the_fingerprint_and_registers_checkpoints(tmp_path):
    """收尾写侧: 用已有产物算指纹落盘, 并把检查点登记进索引 (免设备)。"""
    from ao_shaping.tools.slm.transfer.checkpoint_registry import INDEX_NAME
    from ao_shaping.tools.slm.transfer.optical_session import (
        FINGERPRINT_NAME, compute_optical_fingerprint)
    from ao_shaping.tools.slm.transfer.session_record import record_optical_session

    out = _writing_session(tmp_path)
    record_optical_session(out)

    assert (out / FINGERPRINT_NAME).is_file()
    written = json.loads((out / FINGERPRINT_NAME).read_text(encoding="utf-8"))
    expected = compute_optical_fingerprint(out)
    assert written.keys() == expected.keys()
    for key, value in expected.items():
        if key == "created_at":
            continue
        assert written[key] == value, key
    # 检查点是登记了的 (指纹先落盘, 所以条目里带着它)。
    index = json.loads((out / INDEX_NAME).read_text(encoding="utf-8"))
    assert [entry["kind"] for entry in index] == ["nominal"]
    assert index[0]["fingerprint"]["window_span"] == 820.0


def test_session_completion_survives_an_incomplete_directory(tmp_path):
    """残缺目录也要写一条缺字段的指纹, 不能让跑完的会话失败。"""
    from ao_shaping.tools.slm.transfer.optical_session import FINGERPRINT_NAME
    from ao_shaping.tools.slm.transfer.session_record import record_optical_session

    out = tmp_path / "empty_session"
    record_optical_session(out)  # 不得抛异常

    written = json.loads((out / FINGERPRINT_NAME).read_text(encoding="utf-8"))
    assert written["window_span"] is None
    assert written["created_at"]
