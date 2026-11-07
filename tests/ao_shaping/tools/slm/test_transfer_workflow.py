"""Workflow boundaries, archive integrity and lazy source-package imports."""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ao_shaping.tools.slm.transfer.artifacts import audit_run
from ao_shaping.tools.slm.transfer.capture import to_panel
from ao_shaping.tools.slm.transfer.validate import ShapeObjective, image_metrics
from ao_shaping.tools.slm.transfer.workflow import TransferSession, run_session


def test_source_workflow_imports_without_scripts_or_drivers():
    code = '''
import importlib, importlib.abc, sys
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "scripts" or fullname.startswith(("scripts.", "ao_shaping.drivers")):
            raise AssertionError("Forbidden import: " + fullname)
sys.meta_path.insert(0, Reject())
for name in ("capture", "fit", "validate", "bootstrap", "capture_response", "response", "blur", "direction", "summary", "workflow", "hardware", "adaptation", "simulation", "seeding", "paired"):
    importlib.import_module("ao_shaping.tools.slm.transfer." + name)
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_default_cli_only_prints_plan_and_creates_no_session(tmp_path):
    out = tmp_path / "not-created"
    result = subprocess.run([sys.executable, "-m", "ao_shaping.tools.slm.transfer", "run", "--out", str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "CCD_anchored_blocks" in result.stdout
    assert not out.exists()


def test_failed_stage_is_journaled_and_previous_output_retained(tmp_path, monkeypatch):
    from ao_shaping.tools.slm.transfer import fit
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("[]")
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{}")
    config = TransferSession(tmp_path / "run", dataset=dataset, checkpoint=checkpoint)
    def fail(_):
        raise ValueError("bad recorded geometry")
    monkeypatch.setattr(fit, "fit", fail)
    with pytest.raises(ValueError, match="bad recorded geometry"):
        run_session(config, offline=True)
    journal = json.loads((config.out / "journal.json").read_text())
    assert journal[0]["status"] == "failed"
    assert "bad recorded geometry" in journal[0]["error"]
    assert (config.out / "session.json").exists()
    with pytest.raises(FileExistsError):
        run_session(config, offline=True)


def make_run(tmp_path):
    yy, xx = np.mgrid[:9, :9]
    xy = np.stack((xx, yy), axis=-1)
    target = ((abs(xx-4) <= 2) & (abs(yy-4) <= 2)).astype(np.float32)
    objective = ShapeObjective(target)
    phase = np.zeros((8, 8), dtype=np.float32)
    panel = to_panel(phase)
    rows = []
    for label, level in (("seed", 40), ("0_A1", 40), ("0_B1", 45), ("0_B2", 45), ("0_A2", 40), ("final", 45)):
        raw = np.ones((2, 20, 20), dtype=np.uint8)
        raw[:, :9, :9] += (target*level).astype(np.uint8)
        file = label + ".npz"
        np.savez_compressed(tmp_path / file, raw_frames=raw, model_phase=phase,
                            panel_phase=panel, panel_gray=panel.astype(np.uint16),
                            target=target, core=objective.core, camera_xy=xy)
        image = raw.mean(0)[yy, xx]
        metrics = image_metrics(image, objective)
        loss = metrics["cv"]**2+2*(1-metrics["ee_roi"])+target.sum()*np.square(image/image.sum()-target/target.sum()).sum()
        rows.append(dict(label="square_"+label, file=file, exposure_ms=1.,
                         metrics=metrics, loss=float(loss),
                         frames=[dict(timestamp_utc="2026-10-23T00:00:00+00:00", monotonic_ns=1, end_monotonic_ns=2, exposure_ms=1.)]*2))
    (tmp_path / "config.json").write_text(json.dumps(dict(shapes=["square"], blocks=1)))
    (tmp_path / "manifest.json").write_text(json.dumps(rows))
    (tmp_path / "summary.json").write_text(json.dumps([dict(shape="square", block=0, accepted=True)]))
    return rows


def test_audit_recomputes_raw_metrics_and_acceptance(tmp_path):
    rows = make_run(tmp_path)
    result = audit_run(tmp_path)
    assert result["raw_frames"] == 12
    assert result["accepted_blocks"] == 1
    rows[-1]["metrics"]["ee_roi"] = .01
    (tmp_path / "manifest.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="raw/manifest ee_roi mismatch"):
        audit_run(tmp_path)


def test_audit_cv_acceptance_uses_raw_cv_instead_of_composite_loss(tmp_path):
    make_run(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    config["accept_metric"] = "cv"
    (tmp_path / "config.json").write_text(json.dumps(config))
    # Synthetic target core is constant: both candidates have CV=0. No strict
    # improvement exists, although the composite energy objective improved.
    summary = json.loads((tmp_path / "summary.json").read_text())
    summary[0]["accepted"] = False
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert audit_run(tmp_path)["accepted_blocks"] == 0


def test_exploration_audit_replays_rng_and_requires_champion_export(tmp_path):
    from ao_shaping.tools.slm.transfer.exploration import NonmonotoneAcceptance
    rows = make_run(tmp_path)
    rows[2]["label"] = "square_0_B"
    (tmp_path/"manifest.json").write_text(json.dumps(rows))
    config = dict(shapes=["square"], blocks=1, accept_metric="cv",
                  exploration_temperature=.03, exploration_budget=.25, exploration_seed=42)
    (tmp_path/"config.json").write_text(json.dumps(config))
    policy = NonmonotoneAcceptance(0., temperature=.03, relative_budget=.25, seed=42)
    decision = policy.update(0., 0.)
    summary = [dict(shape="square", block=0, acceptance="nonmonotone_ccd", **decision)]
    (tmp_path/"summary.json").write_text(json.dumps(summary))
    np.save(tmp_path/"square_best.npy", np.zeros((8, 8), dtype=np.float32))
    assert audit_run(tmp_path)["status"] == "passed"
    summary[0]["random_draw"] = .99
    (tmp_path/"summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="exploration decision differs"):
        audit_run(tmp_path)
    summary[0]["random_draw"] = decision["random_draw"]
    (tmp_path/"summary.json").write_text(json.dumps(summary))
    np.save(tmp_path/"square_best.npy", np.ones((8, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="export differs from champion"):
        audit_run(tmp_path)


def test_session_rejects_incomplete_calibration_shapes(tmp_path):
    with pytest.raises(ValueError, match="skip-calibration"):
        TransferSession(tmp_path, shapes=("square",), calibration=True).validate()


def test_relocated_archive_resolves_against_run_ancestors(tmp_path):
    from ao_shaping.tools.slm.transfer.artifacts import resolve_recorded_path
    checkpoint = tmp_path / "data/debug/model/frozen_parameters.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("{}")
    run = tmp_path / "data/debug/run"
    run.mkdir()
    result = resolve_recorded_path("Z:/old-workspace/data/debug/model/frozen_parameters.json", relative_to=run)
    assert result == checkpoint.resolve()


def test_capture_uses_common_exposure_readback_and_releases_devices(tmp_path, monkeypatch):
    from ao_shaping.tools.slm.transfer import capture
    closed, _ = _patch_capture_devices(monkeypatch)
    _one_probe(monkeypatch)
    capture.capture(_capture_args(tmp_path))
    row = json.loads((tmp_path / "capture/manifest.json").read_text())[0]
    assert row["exposure_ms"] == .123
    assert all(frame["exposure_ms"] == .123 for frame in row["frames"])
    assert closed == ["slm", "camera"]


# ---------------------------------------------------------------------------
# capture stage: in-memory measurements by default, npz only on request
# ---------------------------------------------------------------------------


class _FakeCamera:
    """Deterministic CCD: a dim Gaussian spot, settable exposure, no noise."""

    readback_ms = .123

    def __init__(self, closed: list[str]) -> None:
        self._closed = closed

    def open(self) -> None:
        pass

    def close(self) -> None:
        self._closed.append("camera")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._closed.append("camera")

    def reset_exposure_time(self, value):
        self.readback_ms = value

    def get_numpy_image(self, **kwargs):
        yy, xx = np.mgrid[:50, :60]
        return (1+80*np.exp(-((xx-30)**2+(yy-25)**2)/20)).astype(np.uint8)


class _FakeSlm:
    def __init__(self, closed: list[str]) -> None:
        self._closed = closed

    def open(self) -> None:
        pass

    def close(self) -> None:
        self._closed.append("slm")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._closed.append("slm")

    def set_phase_overlay(self, value):
        pass

    def create_phase_from_array(self, phase):
        return np.zeros(phase.shape, dtype=np.uint16)

    def display_data(self, data, *args, **kwargs):
        pass


@dataclass
class _Setting:
    status: str = "ok"
    exposure_ms: float = .123
    peak: float = 0.
    fraction: float = .48
    reads: int = 1


def _patch_capture_devices(monkeypatch, *, statuses=None) -> tuple[list[str], list[dict]]:
    """Replace ``capture.capture``'s device layer.

    Returns the close log and every ``adapt_exposure`` keyword set, so a test can
    pin how the exposure is armed as well as what was written. ``statuses`` feeds
    the ladder's verdicts one per call (``"ok"`` once the list runs out): that is
    how the re-arm path is driven without a camera.
    """
    from types import SimpleNamespace
    from ao_shaping.drivers.ccd import common
    from ao_shaping.tools.slm.panel import focus_probe
    from ao_shaping.tools.slm.transfer import capture as capture_module, hardware

    closed: list[str] = []
    calls: list[dict] = []
    camera = _FakeCamera(closed)
    monkeypatch.setattr(hardware, "prepare_camera_sdk", lambda _: None)
    monkeypatch.setattr(common, "create_camera", lambda *a, **kw: camera)
    monkeypatch.setattr(common, "get_camera_exposure_range", lambda _: (.02, 10.))
    monkeypatch.setattr(common, "get_camera_exposure_ms", lambda cam: cam.readback_ms)
    monkeypatch.setitem(sys.modules, "ao_shaping.drivers.slm.santec",
                        SimpleNamespace(Santec=lambda **kwargs: _FakeSlm(closed)))

    outcomes = list(statuses or [])

    def adapt(setter, read, **kwargs):
        calls.append(dict(kwargs))
        setter(.123)
        read()
        outcome = outcomes.pop(0) if outcomes else "ok"
        return outcome if isinstance(outcome, _Setting) else _Setting(status=outcome)

    monkeypatch.setattr(focus_probe, "adapt_exposure", adapt)
    monkeypatch.setattr(capture_module.time, "sleep", lambda _: None)
    return closed, calls


def _capture_args(tmp_path, **over):
    from types import SimpleNamespace
    base = dict(out=tmp_path / "capture", camera="custom", cam_id=7, slm_number=2,
                grid=8, count=6, frames=3, seed=1, execute=True)
    base.update(over)
    return SimpleNamespace(**base)


def _one_probe(monkeypatch):
    from ao_shaping.tools.slm.transfer import capture as capture_module
    monkeypatch.setattr(capture_module, "probes", lambda *a: [
        ("reference_start", "calibration", np.zeros((8, 8), np.float32))])


def test_capture_writes_no_sample_npz_by_default(tmp_path, monkeypatch):
    """The probe stage hands the fit its payload in memory; the disk stays small."""
    from ao_shaping.tools.slm.transfer import capture

    _patch_capture_devices(monkeypatch)
    _one_probe(monkeypatch)

    samples = capture.capture(_capture_args(tmp_path))

    out = tmp_path / "capture"
    assert list(out.glob("*.npz")) == []
    assert (out / "manifest.json").is_file()
    assert (out / "config.json").is_file()
    row = json.loads((out / "manifest.json").read_text())[0]
    payload = samples[row["file"]]
    assert set(payload) == {"model_phase", "raw_frames"}
    assert payload["model_phase"].shape == (8, 8)
    assert payload["raw_frames"].dtype == np.uint8
    assert payload["raw_frames"].shape == (3, 50, 60)
    config = json.loads((out / "config.json").read_text())
    assert config["save_dataset"] is False
    assert config["exposure_adaptation"] == {"max_steps": 2, "ratio_bounds": [0.1, 50.0]}


def test_save_dataset_writes_the_historical_npz_keys(tmp_path, monkeypatch):
    from ao_shaping.tools.slm.transfer import capture

    _patch_capture_devices(monkeypatch)
    _one_probe(monkeypatch)

    samples = capture.capture(_capture_args(tmp_path, save_dataset=True))

    files = sorted((tmp_path / "capture").glob("sample_*.npz"))
    assert [f.name for f in files] == ["sample_0000.npz"]
    with np.load(files[0]) as saved:
        assert set(saved.files) == {"raw_frames", "frame", "model_phase", "panel_phase",
                                    "panel_gray", "exposure_probe_frames"}
        # The in-memory payload is the same measurement, not a second one.
        np.testing.assert_array_equal(saved["raw_frames"], samples["sample_0000.npz"]["raw_frames"])
        np.testing.assert_array_equal(saved["model_phase"], samples["sample_0000.npz"]["model_phase"])
    assert json.loads((tmp_path / "capture/config.json").read_text())["save_dataset"] is True


def test_capture_writes_done_marker_with_early_stopped_false(tmp_path, monkeypatch):
    """A completed capture records its own end so a parallel worker can tell."""
    from ao_shaping.tools.slm.transfer import capture

    _patch_capture_devices(monkeypatch)
    _one_probe(monkeypatch)

    capture.capture(_capture_args(tmp_path))

    done = tmp_path / "capture" / ".capture_done"
    assert done.is_file()
    marker = json.loads(done.read_text(encoding="utf-8"))
    assert marker["early_stopped"] is False
    assert marker["probes"] == 1
    assert "timestamp_utc" in marker and "monotonic_ns" in marker


def test_capture_stops_when_acquisition_requested(tmp_path, monkeypatch):
    """A parallel-fit worker can ask capture to stop early via a marker file."""
    from ao_shaping.tools.slm.transfer import capture as capture_module

    _patch_capture_devices(monkeypatch)
    out = tmp_path / "capture"

    def stopping_probes(*a):
        states = [
            ("reference_start", "calibration", np.zeros((8, 8), np.float32)),
            ("train_0", "train", np.zeros((8, 8), np.float32)),
            ("train_1", "train", np.zeros((8, 8), np.float32)),
        ]
        for index, state in enumerate(states):
            if index == 1:
                (out / "stop_acquisition.request").write_text("{}")
            yield state

    monkeypatch.setattr(capture_module, "probes", stopping_probes)

    capture_module.capture(_capture_args(tmp_path, save_dataset=True))

    done = out / ".capture_done"
    assert done.is_file()
    marker = json.loads(done.read_text(encoding="utf-8"))
    assert marker["early_stopped"] is True
    assert marker["probes"] == 1  # only the first probe was recorded
    assert len(json.loads((out / "manifest.json").read_text())) == 1
    assert (out / "stop_acquisition.request").is_file()


def test_capture_arms_exposure_with_one_correction_and_a_wide_bound(tmp_path, monkeypatch):
    """One exact correction + one verification read, with a bound that reaches ×9."""
    from ao_shaping.tools.slm.transfer import capture

    _, calls = _patch_capture_devices(monkeypatch)
    _one_probe(monkeypatch)

    capture.capture(_capture_args(tmp_path))

    assert calls[0]["max_steps"] == 2
    assert tuple(calls[0]["ratio_bounds"]) == (0.1, 50.0)
    # The operator's cold start read 13/255 at the 0.02 ms floor: a factor of 8.8,
    # which the historical (0.15, 6.0) clamp cannot apply in one step.
    assert capture.EXPOSURE_RATIO_BOUNDS[1] >= 0.45 / (13/255)


def test_capture_rearms_the_exposure_ladder_instead_of_failing_the_lane(tmp_path, monkeypatch):
    """A probe that overshoots into saturation gets another ladder, not a dead lane.

    2026-11-03 17:40 的会话死在第 13 条探针上: probe_012_zernike 从上一个探针停住的
    1.428 ms 起阶梯, 峰值 255 (饱和), 乘法修正到 0.642 ms 仍是 255 —— 两档用完, 阶梯
    已经把下一档算成 0.289 ms 却没有档位去测它, 于是整条 24 条探针的会话 abort, 前面
    12 条好数据一起作废。走不到接受带只是跨度问题, 换一档就回来。
    """
    from ao_shaping.tools.slm.transfer import capture as capture_module

    _, calls = _patch_capture_devices(monkeypatch, statuses=["clipped", "ok"])
    _one_probe(monkeypatch)

    capture_module.capture(_capture_args(tmp_path))

    row = json.loads((tmp_path / "capture/manifest.json").read_text())[0]
    assert [call["max_steps"] for call in calls] == [2, capture_module.EXPOSURE_RETRY_STEPS]
    assert row["exposure_setting"]["status"] == "ok"
    assert row["exposure_attempts"] == 2


def test_capture_resumes_from_the_exposure_the_ladder_computed(tmp_path, monkeypatch):
    """The retry starts at the rung the ladder never got to measure, not at the clipped one.

    阶梯跑到最后一档就停时会先把下一档算进 ``ExposureSetting.exposure_ms`` (那次是
    0.642 → 0.289) 却没档位去测它。重试就该从 0.289 起, 而不是回到 0.642 再重测一次
    饱和帧 —— 那一档换来的读数必然是 255, 白花一次阶梯。
    """
    from ao_shaping.tools.slm.transfer import capture as capture_module

    _, calls = _patch_capture_devices(
        monkeypatch,
        statuses=[_Setting(status="clipped", exposure_ms=.289, peak=255., fraction=1.), "ok"],
    )
    _one_probe(monkeypatch)

    capture_module.capture(_capture_args(tmp_path))

    assert calls[1]["start_ms"] == pytest.approx(.289)


def test_capture_still_fails_when_every_ladder_attempt_misses_the_band(tmp_path, monkeypatch):
    """A genuinely impossible exposure (too bright at the floor) is still fatal — loudly."""
    from ao_shaping.tools.slm.transfer import capture as capture_module

    _, calls = _patch_capture_devices(monkeypatch, statuses=["clipped"] * 9)
    _one_probe(monkeypatch)

    with pytest.raises(RuntimeError, match="Unusable exposure"):
        capture_module.capture(_capture_args(tmp_path))
    assert len(calls) == 1 + capture_module.EXPOSURE_RETRIES


# ---------------------------------------------------------------------------
# fit stage: the in-memory payload produces the recorded result
# ---------------------------------------------------------------------------


def _tiny_dataset(tmp_path, *, write_npz=True, size=8):
    """Ramps + reference + one train/validation pair, with both payload forms."""
    yy, xx = np.mgrid[:40, :40]
    base = (5 + 60*np.exp(-((xx-20)**2+(yy-20)**2)/30)).astype(np.uint8)
    rows, samples = [], {}

    def add(label, split, spot, phase_value):
        name = f"sample_{len(rows):04d}.npz"
        raw = np.stack([base, base])
        phase = np.full((size, size), phase_value, np.float32)
        if write_npz:
            np.savez_compressed(tmp_path/name, raw_frames=raw, frame=raw.mean(0),
                                model_phase=phase, panel_phase=to_panel(phase),
                                panel_gray=np.zeros((1200, 1920), np.uint16))
        samples[name] = dict(model_phase=phase, raw_frames=raw)
        rows.append(dict(file=name, label=label, split=split, exposure_ms=1., peak=float(base.max()),
                         spot_xy=list(spot), exposure_setting=dict(status="ok")))

    add("reference_start", "calibration", (20., 20.), 0.)
    for axis, shift in (("x", (6., 0.)), ("y", (0., 6.))):
        add(f"ramp_{axis}_64_1", "calibration", (20.+shift[0], 20.+shift[1]), 0.)
        add(f"ramp_{axis}_64_-1", "calibration", (20.-shift[0], 20.-shift[1]), 0.)
    add("probe_000_a", "train", (20., 20.), .1)
    add("probe_003_b", "validation", (20., 20.), .2)
    (tmp_path/"manifest.json").write_text(json.dumps(rows))
    (tmp_path/"config.json").write_text(json.dumps(dict(grid=size)))
    return samples


def test_fit_reads_in_memory_samples_exactly_like_the_recorded_npz(tmp_path):
    """No npz files are read: the arrays must be bit-identical to the disk path."""
    from ao_shaping.tools.slm.transfer import fit

    samples = _tiny_dataset(tmp_path)
    from_disk = fit.load_dataset(tmp_path, 8, 10)
    from_memory = fit.load_dataset(tmp_path, 8, 10, samples=samples)

    assert [r["label"] for r in from_disk[0]] == [r["label"] for r in from_memory[0]]
    for index, name in ((1, "phases"), (2, "images"), (3, "camera_xy"),
                        (4, "ramp_matrix"), (6, "train mask")):
        np.testing.assert_array_equal(from_disk[index], from_memory[index], err_msg=name)


def test_fit_still_reads_the_recorded_dataset_without_samples(tmp_path):
    """The standalone ``--dataset`` path is unchanged: npz only, no mapping."""
    from ao_shaping.tools.slm.transfer import fit

    _tiny_dataset(tmp_path)
    selected, phases, images, *_ = fit.load_dataset(tmp_path, 8, 10)

    assert [r["label"] for r in selected] == ["probe_000_a", "probe_003_b"]
    assert phases.shape == (2, 8, 8)
    assert images.shape == (2, 11, 11)


def test_build_fit_model_threads_the_in_memory_samples_through(tmp_path, monkeypatch):
    """``save_dataset=False`` writes no npz, so the fit must use the in-memory payload.

    2026-11-03 19:26 的会话在 **fit** 段炸掉: ``FileNotFoundError:
    .../capture/sample_0009.npz``。采集段成功, 但 ``save_dataset=False`` 意味着
    那些 npz 从来没被写出来 —— 拟合本该走内存里的 payload。``build_fit_model``
    被抽出来时 (``a35d5af0``) 把 ``samples=`` 丢了, 于是 ``load_dataset`` 掉回
    磁盘分支, 去读一个本来就不该存在的文件。
    """
    from ao_shaping.tools.slm.transfer import fit

    samples = _tiny_dataset(tmp_path)
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps(dict(
        defocus_diopter=0., origin=[20., 20.], amplitude_center=[960., 600.],
        amplitude_width=[400., 300.], residual_noll_radians={str(j): 0. for j in range(5, 12)},
        gain=1., background=0., sigma_ccd_px=0., ramp_matrix=[[0., 1.], [1., 0.]],
        grid_size=8, fft_size=64, pixel_samples=3)), encoding="utf-8")
    captured: dict = {}

    class _Reached(Exception):
        """Stops the test right after the wiring under test."""

    def fake_load(directory, size, span, samples=None):
        captured["samples"] = samples
        raise _Reached

    monkeypatch.setattr(fit, "load_dataset", fake_load)

    with pytest.raises(_Reached):
        fit.build_fit_model(tmp_path, checkpoint, span=10, samples=samples)

    assert captured["samples"] is samples


def test_fit_forwards_the_in_memory_samples_to_build_fit_model(tmp_path, monkeypatch):
    """``fit.fit`` must hand its ``samples`` down to ``build_fit_model``.

    Same defect as above, one hop further up: ``fit`` records
    ``sample_source="memory"`` in its provenance, so a lost mapping shows up as a
    provenance that claims memory while the arrays came from disk.
    """
    from ao_shaping.tools.slm.transfer import fit

    samples = _tiny_dataset(tmp_path)
    captured: dict = {}

    class _Reached(Exception):
        """Stops the test right after the wiring under test."""

    def fake_build(dataset, checkpoint, *, span, **kwargs):
        captured.update(kwargs)
        raise _Reached

    monkeypatch.setattr(fit, "build_fit_model", fake_build)

    with pytest.raises(_Reached):
        fit.fit(SimpleNamespace(dataset=tmp_path, checkpoint=tmp_path / "checkpoint.json",
                                 out=tmp_path / "fit", iterations=1, span=10, samples=samples,
                                 adaptation="full"))

    assert captured["samples"] is samples


# ---------------------------------------------------------------------------
# acceptance: A-B-B-A is optional; the model-only path must say so
# ---------------------------------------------------------------------------


def test_model_only_decision_needs_a_previous_loss_and_an_improvement():
    from ao_shaping.tools.slm.transfer.validate import model_only_decision

    assert model_only_decision([])["accepted"] is False
    single = model_only_decision([dict(loss=1.)])
    assert single["accepted"] is False
    assert single["model_loss_previous"] is None
    improved = model_only_decision([dict(loss=1.), dict(loss=.5)])
    assert improved["accepted"] is True
    assert improved["acceptance"] == "model_only"
    assert (improved["model_loss_previous"], improved["model_loss_last"]) == (1., .5)
    assert model_only_decision([dict(loss=.5), dict(loss=.5)])["accepted"] is False
    assert model_only_decision([dict(loss=.5), dict(loss=.9)])["accepted"] is False


class _PhaseCamera:
    """CCD whose brightness follows the last displayed grayscale level."""

    readback_ms = .5
    level = 0

    def __init__(self, closed):
        self._closed = closed
        self.exposures = []

    def open(self) -> None:
        pass

    def close(self) -> None:
        self._closed.append("camera")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._closed.append("camera")

    def reset_exposure_time(self, value):
        self.readback_ms = value
        self.exposures.append(float(value))

    def get_numpy_image(self, **kwargs):
        yy, xx = np.mgrid[:50, :60]
        base = 5 + 40*np.exp(-((xx-4)**2+(yy-4)**2)/20)
        return np.clip(base*(1+.5*self.level)+3*self.level, 0, 240).astype(np.uint8)


class _LevelSlm:
    """SLM stub: encodes the phase's mean amplitude as a grayscale level."""

    def __init__(self, camera, closed):
        self._camera = camera
        self._closed = closed
        self.displays = []

    def open(self) -> None:
        pass

    def close(self) -> None:
        self._closed.append("slm")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._closed.append("slm")

    def set_phase_overlay(self, value):
        pass

    def create_phase_from_array(self, phase):
        level = 3 + int(round(float(np.asarray(phase).mean())*10))
        return np.full(np.asarray(phase).shape, max(level, 0), dtype=np.uint16)

    def display_data(self, gray, *args, **kwargs):
        self._camera.level = int(np.asarray(gray).ravel()[0])
        self.displays.append(self._camera.level)


def _run_validate(tmp_path, monkeypatch, *, abba, mode="model", updates=2, device=None, seen=None,
                  adaptive_exposure=False, blocks=1, seed_value=0., target_at_impl=None, **extra):
    """Drive ``validate.run`` end to end with fakes: devices, forward and optimizer.

    Everything above the device layer is the production code, so the set of
    captures actually taken is what is asserted -- not a re-implementation of it.
    ``seen`` (a dict) collects the ``device`` each stubbed stage construction
    received, which is how the resolver plumbing is asserted without a GPU.
    ``target_at_impl`` overrides the ``validate.target_at`` stub (same signature),
    so a test can return a box that would touch the window edge and pin the
    ``tilt`` boundary-skip behaviour.
    """
    from types import SimpleNamespace
    import torch
    from ao_shaping.algorithm.signal_processing import differentiable_defocus_uniformity as ddu
    from ao_shaping.drivers.ccd import common
    from ao_shaping.tools.slm.panel import focus_probe
    from ao_shaping.tools.slm.transfer import hardware, validate

    seen = {} if seen is None else seen
    seen.setdefault("forward", [])
    seen.setdefault("optimizer", [])

    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps(dict(ramp_matrix=[[1., 0.], [0., 1.]], grid_size=8,
                                          fft_size=16, pixel_samples=3)))
    yy, xx = np.mgrid[:9, :9]
    xy = np.stack((xx, yy), axis=-1).astype(float)

    class _Params:
        @classmethod
        def from_dicts(cls, parameters, ramp_matrix, *, grid_size, fft_size, pixel_samples):
            params = SimpleNamespace(ramp_matrix=np.asarray(ramp_matrix, dtype=float),
                                     grid_size=grid_size)
            params.effective_parameters = lambda: params
            return params

    class _Forward:
        def __init__(self, params, xy, device="cpu"):
            seen["forward"].append(device)
            self.xy = np.asarray(xy)

        def predict(self, phase):
            level = 1.0 + float(np.asarray(phase).mean())
            ramp = np.arange(self.xy.shape[0]*self.xy.shape[1], dtype=np.float32).reshape(self.xy.shape[:2])
            return torch.tensor(level*(1.0+ramp), dtype=torch.float32)[None]

    class _Optimizer:
        # ``**kwargs``: the production optimizer keeps growing optional hooks
        # (e.g. ``spectral_mask`` from issue #213 item 1b). A fake that names only
        # the arguments it happens to use breaks every caller the day a hook is
        # added, and the failure reads as "validate is broken" rather than "the
        # fake is stale".
        def __init__(self, forward, objective, seed, *, lr, step_rms_cap, device, **kwargs):
            seen.setdefault("optimizer_kwargs", []).append(sorted(kwargs))
            seen["optimizer"].append(device)
            self._phase = np.asarray(seed, np.float32).copy()
            self.loss_history = []
            self.accepted = 0

        @property
        def current_phase(self):
            return self._phase

        def update(self, measured=None):
            self.loss_history.append(dict(step=len(self.loss_history)+1,
                                          loss=1.0/(1+len(self.loss_history))))
            self._phase = self._phase + np.float32(.05)
            return self._phase

        def accept(self):
            self.accepted += 1

    monkeypatch.setattr(ddu, "FrozenDefocusParameters", _Params)
    monkeypatch.setattr(ddu, "FrozenDefocusForward", _Forward)
    monkeypatch.setattr(ddu, "DifferentiableUniformityOptimizer", _Optimizer)
    monkeypatch.setattr(ddu, "recentre_window", lambda centre, *, half_span_px, step_px: xy)
    if target_at_impl is None:
        target_at_impl = lambda xy, shape, centre, side, aspect_ratio=2.0: \
            np.pad(np.ones((5, 5), np.float32), 2)
    monkeypatch.setattr(validate, "target_at", target_at_impl)
    monkeypatch.setattr(validate, "gs_seed",
                        lambda forward, centre, side, shape: np.full((8, 8), seed_value, np.float32))
    monkeypatch.setattr(hardware, "prepare_camera_sdk", lambda _: None)
    monkeypatch.setattr(validate.time, "sleep", lambda _: None)

    closed = []
    camera = _PhaseCamera(closed)
    seen["camera"] = camera
    monkeypatch.setattr(common, "create_camera", lambda *a, **kw: camera)
    monkeypatch.setattr(common, "get_camera_exposure_range", lambda _: (.02, 10.))
    monkeypatch.setattr(common, "get_camera_exposure_ms", lambda cam: cam.readback_ms)
    monkeypatch.setitem(sys.modules, "ao_shaping.drivers.slm.santec",
                        SimpleNamespace(Santec=lambda **kwargs: _LevelSlm(camera, closed)))

    def adapt(setter, read, **kwargs):
        seen.setdefault("exposure_state", []).append((camera.level, camera.readback_ms))
        level = .5 + .2*len(seen.setdefault("exposure", [])) if adaptive_exposure else .5
        seen.setdefault("exposure", []).append(level)
        setter(level)
        read()
        return _Setting(status="ok", exposure_ms=.5)

    monkeypatch.setattr(focus_probe, "adapt_exposure", adapt)

    args = SimpleNamespace(checkpoint=checkpoint, out=tmp_path/"run", zero_xy=[10., 10.],
                           period=64., side=20., shapes=["square"], blocks=blocks, updates=updates,
                           frames=2, mode=mode, seed_run=None, seed_policy="strict",
                           accept_metric="loss", objective="shape", camera="custom",
                           cam_id=7, slm_number=2, execute=True, abba=abba, device=device,
                           adaptive_exposure=adaptive_exposure, **extra)
    validate.run(args)
    return args


def _labels(args):
    return [row["label"] for row in json.loads((args.out/"manifest.json").read_text())]


def test_flat_light_precheck_restores_shaped_state_before_exposure_ladder(tmp_path, monkeypatch):
    seen = {}
    _run_validate(
        tmp_path,
        monkeypatch,
        abba=False,
        updates=0,
        blocks=0,
        seed_value=.2,
        seen=seen,
    )

    # The flat probe used 0.05 ms; both restoration and the shaped ladder start
    # from the original 0.5 ms, and the ladder sees the restored shaped level 5.
    assert seen["camera"].exposures[:3] == [.05, .5, .5]
    assert seen["exposure_state"][0] == (5, .5)


def test_runtime_exposure_changes_between_blocks_but_not_within_pair(tmp_path, monkeypatch):
    args = _run_validate(tmp_path, monkeypatch, abba=True, mode="anchored", adaptive_exposure=True, blocks=2)
    rows = {r["label"]: r for r in json.loads((args.out/"manifest.json").read_text())}
    for block, expected in ((0, .5), (1, .7)):
        assert {rows[f"square_{block}_{suffix}"]["exposure_ms"] for suffix in ("A1", "B1", "B2", "A2")} == {expected}
    assert rows["square_seed_replay"]["exposure_ms"] == rows["square_final"]["exposure_ms"] == .5
    assert (args.out/"square_1_exposure.npz").exists()


def test_tilt_off_skips_boundary_check_without_target_xy(tmp_path, monkeypatch):
    """``tilt=False`` 且无 ``target_xy`` 时, 即使目标其实会越界也不抛 boundary 错。

    构造一个 ``target_at`` 返回满窗矩形 (box 顶到窗口边缘) 的 stub: 若倾斜开着
    该 branch 会 ``raise``, 而 ``tilt=False`` 时要跳过检查并正常跑完 blocks=0。
    同时确认 ``centre == zero_xy`` (目标停在零级, ``shift`` 被置零)。
    """
    full_rect = np.ones((9, 9), np.float32)  # 填满 9x9 窗口, 必然触碰边缘
    args = _run_validate(
        tmp_path,
        monkeypatch,
        abba=False,
        mode="anchored",
        updates=0,
        blocks=0,
        target_at_impl=lambda xy, shape, centre, side, aspect_ratio=2.0: full_rect,
        tilt=False,
        target_xy=None,
    )
    # centre 必须等于 zero_xy (shift=0)。
    assert args.target_xy is None
    assert (args.out / "config.json").exists()  # 跑完了, 没有抛 boundary


def test_tilt_on_still_enforces_boundary_check(tmp_path, monkeypatch):
    """``tilt=True`` (默认 aspect=2.0) 保持既有 boundary 判定: 越界仍抛错。"""
    full_rect = np.ones((9, 9), np.float32)
    with pytest.raises(ValueError, match="window boundary"):
        _run_validate(
            tmp_path,
            monkeypatch,
            abba=False,
            mode="anchored",
            updates=0,
            blocks=0,
            target_at_impl=lambda xy, shape, centre, side, aspect_ratio=2.0: full_rect,
            tilt=True,
        )
    rows = make_run(tmp_path)
    config = json.loads((tmp_path/"config.json").read_text())
    config["exposure_policy"] = "per_block"
    (tmp_path/"config.json").write_text(json.dumps(config))
    for row in (rows[0], rows[-1]):
        row["exposure_ms"] = 2.
        row["frames"] = [dict(frame, exposure_ms=2.) for frame in row["frames"]]
    (tmp_path/"manifest.json").write_text(json.dumps(rows))
    assert audit_run(tmp_path)["status"] == "passed"
    rows[2]["exposure_ms"] = 3.
    rows[2]["frames"] = [dict(frame, exposure_ms=3.) for frame in rows[2]["frames"]]
    (tmp_path/"manifest.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="exposure differs within paired block"):
        audit_run(tmp_path)


def test_abba_off_in_model_mode_skips_a1_b1_b2_a2(tmp_path, monkeypatch):
    args = _run_validate(tmp_path, monkeypatch, abba=False)

    # In "model" mode A1 only feeds the paired decision, so it goes too.
    assert _labels(args) == ["square_seed", "square_final"]
    assert not (args.out/"square_direction.json").exists()
    summary = json.loads((args.out/"summary.json").read_text())
    assert summary[0]["acceptance"] == "model_only"
    assert summary[0]["accepted"] is True          # losses 1.0 -> 0.5
    assert summary[0]["model_loss_last"] < summary[0]["model_loss_previous"]
    assert "baseline" not in summary[0]
    config = json.loads((args.out/"config.json").read_text())
    assert config["abba"] is False
    assert config["acceptance"] == "model_only"


def test_abba_off_in_anchored_mode_measures_the_candidate_on_ccd(tmp_path, monkeypatch):
    args = _run_validate(tmp_path, monkeypatch, abba=False, mode="anchored")

    # 非配对默认路径: 用户要求 CCD 实测主导 —— 锚点 A1 之外再捕 B1,
    # 由实测指标决定接受/拒绝 (sequential_ccd), 模型损失仍作参考。
    assert _labels(args) == ["square_seed", "square_0_A1", "square_0_B", "square_final"]
    entry = json.loads((args.out/"summary.json").read_text())[0]
    assert entry["acceptance"] == "sequential_ccd"
    assert entry["accept_metric"] == "loss"
    # 接受与否由 CCD 实测判定 (candidate < baseline - tolerance), 不依赖模型损失。
    assert entry["accepted"] == (entry["candidate"] < entry["baseline"] - entry["tolerance"])
    assert "model_loss_last" in entry and "model_loss_previous" in entry
    assert entry["model_loss_last"] < entry["model_loss_previous"]  # 模型损失其实在改善


def test_abba_off_looped_round_keeps_model_only_acceptance(tmp_path, monkeypatch):
    # 验收阈值循环路径不动: acceptance_enabled=True 时即使有锚点也不加 B1,
    # 维持原 model_only 判定, 避免改变循环时序。
    args = _run_validate(tmp_path, monkeypatch, abba=False, mode="anchored",
                         acceptance_enabled=True)
    assert _labels(args) == ["square_seed", "square_0_A1", "square_final"]
    assert json.loads((args.out/"summary.json").read_text())[0]["acceptance"] == "model_only"


def test_abba_on_still_lets_the_paired_decision_decide(tmp_path, monkeypatch):
    from ao_shaping.tools.slm.transfer.paired import paired_decision

    args = _run_validate(tmp_path, monkeypatch, abba=True)

    assert _labels(args) == ["square_seed", "square_0_A1", "square_0_B1", "square_0_B2",
                             "square_0_A2", "square_direction_minus", "square_direction_reference",
                             "square_final"]
    records = json.loads((args.out/"manifest.json").read_text())
    by_label = {row["label"]: row for row in records}
    expected = paired_decision(by_label["square_0_A1"], by_label["square_0_B1"],
                               by_label["square_0_B2"], by_label["square_0_A2"])
    summary = json.loads((args.out/"summary.json").read_text())
    # Today's entry shape: the paired numbers, and no model-only marker.
    assert "acceptance" not in summary[0]
    assert "model_loss_last" not in summary[0]
    assert summary[0]["baseline"] == expected["baseline"]
    assert summary[0]["candidate"] == expected["candidate"]
    assert summary[0]["drift"] == expected["drift"]
    assert summary[0]["accepted"] == expected["accepted"]
    assert (args.out/"square_direction.json").exists()
    assert json.loads((args.out/"config.json").read_text())["acceptance"] == "paired_abba"


# ---------------------------------------------------------------------------
# workflow plumbing: capture samples -> fit, abba -> validate
# ---------------------------------------------------------------------------


def _fake_fit(seen):
    """Stand-in for ``fit.fit``: record the args and write the checkpoint.

    The session's geometry pre-flight reads ``fit/frozen_parameters.json`` after
    the fit stage, so the stub has to leave one behind (identity ramp matrix:
    no target displacement, nothing for the pre-flight to reject).
    """
    def _fit(args):
        seen["fit"] = args
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out/"frozen_parameters.json").write_text(json.dumps(dict(
            ramp_matrix=[[1., 0.], [0., 1.]], grid_size=8, fft_size=16, pixel_samples=3)))
    return _fit


def test_run_session_hands_capture_samples_to_the_fit(tmp_path, monkeypatch):
    from ao_shaping.tools.slm.transfer import (
        artifacts, bootstrap, capture, direction, fit, summary, validate, workflow,
    )

    payload = {"sample_0000.npz": dict(model_phase=np.zeros((8, 8), np.float32),
                                       raw_frames=np.zeros((2, 8, 8), np.uint8))}
    seen = {}

    def fake_capture(args, *, on_frame=None):
        seen["capture"] = args
        # 逐帧观察者以关键字传入 (config.json 是 {**vars(args)}, 观察者不进那里)。
        seen["capture_on_frame"] = on_frame
        rows = [dict(label="reference_start", file="sample_0000.npz", split="calibration",
                     exposure_ms=1., peak=10., spot_xy=[4., 4.],
                     exposure_setting=dict(status="ok"))]
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out/"manifest.json").write_text(json.dumps(rows))
        (args.out/"config.json").write_text(json.dumps(dict(grid=8)))
        return payload

    monkeypatch.setattr(capture, "capture", fake_capture)
    monkeypatch.setattr(bootstrap, "bootstrap",
                        lambda dataset: (dict(grid_size=8), dict(status="nominal")))
    monkeypatch.setattr(fit, "fit", _fake_fit(seen))
    monkeypatch.setattr(validate, "run", lambda args: seen.setdefault("validate", []).append(args))
    monkeypatch.setattr(artifacts, "audit_run", lambda run: dict(status="passed"))
    monkeypatch.setattr(summary, "summarize", lambda run, out: None)
    monkeypatch.setattr(direction, "analyze", lambda run, out: seen.setdefault("direction", []).append(run))

    def session(name, **over):
        return TransferSession(tmp_path/name, shapes=("square",), count=6, frames=3,
                               iterations=1, blocks=1, updates=1, **over)

    workflow.run_session(session("with-abba", abba=True))
    assert seen["capture"].save_dataset is False
    assert seen["fit"].samples is payload
    assert [args.abba for args in seen["validate"]] == [True, True]
    assert len(seen["direction"]) == 1

    seen.clear()
    workflow.run_session(session("model-only", abba=False))
    assert [args.abba for args in seen["validate"]] == [False, False]
    # The direction check reads B1/B2, which do not exist without the paired blocks.
    assert "direction" not in seen
    plan = json.loads((tmp_path/"model-only/session.json").read_text())
    assert plan["resolved"]["acceptance"] == "model_only"
    assert "A-B-B-A acceptance skipped" in plan["note"].replace("\u2013", "-")
    assert "direction_check" not in plan["stages"]

    # 逐帧观察者要走到每一个开相机的阶段。两种调用形状都在用: capture 收关键字,
    # validate 收 namespace 上的字段 (validate._resolve_observer 归一两者)。
    frames: list[dict] = []

    def marker(payload):
        frames.append(payload)

    seen.clear()
    workflow.run_session(session("observed", abba=False), on_frame=marker)
    assert seen["capture_on_frame"] is marker
    assert seen["validate"] and all(
        getattr(args, "on_frame", None) is marker for args in seen["validate"])
    # 假的 capture/validate 不发帧, 所以上面断言的是「转发到」而不是「被调用」。
    assert frames == []


# ---------------------------------------------------------------------------
# device: the stages take the resolved device, and the artefacts record it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("available,requested,expected", [
    (False, None, "cpu"),
    (True, None, "cuda"),
    (True, "auto", "cuda"),
    (False, "auto", "cpu"),
    (True, "cpu", "cpu"),      # an explicit pin wins over a visible GPU
    (False, "cuda", "cuda"),   # an explicit cuda is not silently downgraded
])
def test_validate_stages_use_the_resolved_device(tmp_path, monkeypatch, available,
                                                 requested, expected):
    """Forward *and* optimizer get the resolver's answer, and config.json records it.

    The fakes ignore the device, so this needs no GPU: what is asserted is the
    value the production code passed down, which is exactly what a hardcoded
    ``device="cpu"`` would break.
    """
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: bool(available))
    seen: dict = {}

    args = _run_validate(tmp_path, monkeypatch, abba=False, device=requested, seen=seen)

    assert set(seen["forward"]) == {expected}
    assert set(seen["optimizer"]) == {expected}
    config = json.loads((args.out/"config.json").read_text())
    assert config["device"] == expected
    assert config["device_requested"] == requested


def test_the_session_records_the_resolved_device(tmp_path, monkeypatch):
    """session.json must say what the run actually used, not what was asked for."""
    from ao_shaping.tools.slm.transfer import (
        artifacts, bootstrap, capture, direction, fit, summary, validate, workflow,
    )

    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    payload = {"sample_0000.npz": dict(model_phase=np.zeros((8, 8), np.float32),
                                       raw_frames=np.zeros((2, 8, 8), np.uint8))}
    seen = {}

    def fake_capture(args, *, on_frame=None):
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out/"manifest.json").write_text(json.dumps([dict(
            label="reference_start", file="sample_0000.npz", split="calibration",
            exposure_ms=1., peak=10., spot_xy=[4., 4.], exposure_setting=dict(status="ok"))]))
        (args.out/"config.json").write_text(json.dumps(dict(grid=8)))
        return payload

    monkeypatch.setattr(capture, "capture", fake_capture)
    monkeypatch.setattr(bootstrap, "bootstrap",
                        lambda dataset: (dict(grid_size=8), dict(status="nominal")))
    monkeypatch.setattr(fit, "fit", _fake_fit(seen))
    monkeypatch.setattr(validate, "run", lambda args: seen.setdefault("validate", []).append(args))
    monkeypatch.setattr(artifacts, "audit_run", lambda run: dict(status="passed"))
    monkeypatch.setattr(summary, "summarize", lambda run, out: None)
    monkeypatch.setattr(direction, "analyze", lambda run, out: None)

    config = TransferSession(tmp_path/"session", shapes=("square",), count=6, frames=3,
                             iterations=1, blocks=1, updates=1, device="auto")
    workflow.run_session(config)

    session = json.loads((config.out/"session.json").read_text())
    # "auto" is the request; the resolved value is what the run used.
    assert session["settings"]["device"] == "auto"
    assert session["resolved"]["device"] == "cpu"
    assert seen["fit"].device == "cpu"
    assert [args.device for args in seen["validate"]] == ["cpu", "cpu"]


def test_an_unknown_device_is_refused_before_any_device_opens(tmp_path):
    with pytest.raises(ValueError, match="unknown device"):
        TransferSession(tmp_path / "session", device="gpu").validate()


# ---------------------------------------------------------------------------
# plan mode: what a session still has to measure, decided before any device
# ---------------------------------------------------------------------------


def _plan_args(out, shapes, **over):
    from types import SimpleNamespace

    base = dict(out=out, shapes=list(shapes), bit_depth=8, window_step=4., blocks=2, frames=3)
    base.update(over)
    return SimpleNamespace(**base)


def _block_summary(path, shape, blocks):
    """A ``summary.json`` as ``validate.run`` writes it: one row per block."""
    path.mkdir(parents=True)
    (path / "summary.json").write_text(json.dumps(
        [dict(shape=shape, block=block, accepted=False) for block in blocks]))


def test_session_plan_without_session_groups_is_one_tuple_per_shape(tmp_path):
    from ao_shaping.tools.slm.transfer.validate import session_plan

    plan = session_plan(_plan_args(tmp_path / "run", ["square", "circle"]))

    assert plan["frozen"] == []
    assert [tuple(entry[:2]) for entry in plan["todo"]] == [("square", 8), ("circle", 8)]
    assert [entry[2:] for entry in plan["todo"]] == [(4., 2), (4., 2)]
    # Order of magnitude only, not wall clock: 2 shapes x 2 groups x
    # (0.6 s settle + 3 frames x 0.9 s) = 13.2 s.
    assert plan["estimated_seconds"] == 13.2


def test_a_finished_shape_lands_in_frozen_not_todo(tmp_path):
    """A shape whose directory already holds two measured blocks needs no block.

    The directory is written in the recorded form ``validate.run`` produces (one
    ``summary.json`` row per block), so this asserts the on-disk contract plan
    mode reads rather than a hand-built shortcut.
    """
    from ao_shaping.tools.slm.transfer.validate import session_plan

    _block_summary(tmp_path / "square", "square", range(2))

    plan = session_plan(_plan_args(tmp_path, ["square", "circle"]))

    assert plan["frozen"] == [("square", 8, 4.0, 2)]
    assert [entry[0] for entry in plan["todo"]] == ["circle"]


def test_a_malformed_summary_is_a_value_error_not_an_index_error(tmp_path):
    from ao_shaping.tools.slm.transfer.validate import session_plan

    broken = tmp_path / "triangle"
    broken.mkdir()
    (broken / "summary.json").write_text(json.dumps([dict(shape="triangle")]))  # no "block"

    with pytest.raises(ValueError, match="lacks shape/block keys"):
        session_plan(_plan_args(tmp_path, ["triangle"]))


def test_profile_timing_writes_a_five_part_split_per_capture(tmp_path, monkeypatch):
    """The split is the evidence the speed-up work is argued from.

    ``scripts/slm/analyze_session_time_budget.py`` can only attribute what a
    session records; before this flag the ~5 s between two captures had no
    per-segment breakdown at all, so every proposal rested on a guess.
    """
    args = _run_validate(tmp_path, monkeypatch, abba=False, mode="anchored", blocks=1,
                         profile_timing=True)
    rows = json.loads((args.out/"manifest.json").read_text())
    assert rows
    for row in rows:
        assert set(row["timing"]) == {"slm_write", "settle", "read", "reduce", "save"}
        assert all(value >= 0 for value in row["timing"].values())


def test_timing_is_absent_unless_it_is_asked_for(tmp_path, monkeypatch):
    args = _run_validate(tmp_path, monkeypatch, abba=False, mode="anchored", blocks=1)
    rows = json.loads((args.out/"manifest.json").read_text())
    assert rows and all("timing" not in row for row in rows)


def test_async_capture_write_stores_the_same_measurement(tmp_path, monkeypatch):
    """The background writer may only change *when* the bytes land.

    A lane's manifest is what the campaign compares, so the row and the npz it
    names must match the synchronous path exactly -- otherwise the flag would
    silently fork the protocol.
    """
    (tmp_path/"sync").mkdir(parents=True, exist_ok=True)
    (tmp_path/"async").mkdir(parents=True, exist_ok=True)
    sync = _run_validate(tmp_path/"sync", monkeypatch, abba=False, mode="anchored", blocks=1)
    background = _run_validate(tmp_path/"async", monkeypatch, abba=False, mode="anchored",
                               blocks=1, async_capture_write=True)

    def comparable(run):
        rows = json.loads((run.out/"manifest.json").read_text())
        keys = ("label", "file", "peak", "core_peak", "saturation_pixels", "loss")
        return [{key: row[key] for key in keys} for row in rows]

    assert comparable(sync) == comparable(background)
    assert (sorted(path.name for path in sync.out.glob("capture_*.npz"))
            == sorted(path.name for path in background.out.glob("capture_*.npz")))
    with np.load(sync.out/"capture_0000.npz") as left, np.load(background.out/"capture_0000.npz") as right:
        assert sorted(left.files) == sorted(right.files)
        for key in left.files:
            assert np.array_equal(left[key], right[key])

# ---------------------------------------------------------------------------
# 光路复用跳过记录: 复用被跳过也要落一份可读说明, 供「运行处」显示原因。
# ---------------------------------------------------------------------------


def _probe(serial="SLM-A", zero=(10.0, 20.0), exposure=(0.02, 25.0),
           camera_format="MONO8", bit_depth=8):
    """一份可比的合成探针 (序列号默认有, 传 None 模拟读不到)。"""
    from ao_shaping.tools.slm.transfer.optical_probe import OpticalProbe
    return OpticalProbe(
        zero_xy=zero,
        exposure_range_ms=exposure,
        camera_format=camera_format,
        bit_depth=bit_depth,
        slm_serial=serial,
    )


def _skip_record(out):
    data = json.loads((out / "optical_probe.json").read_text(encoding="utf-8"))
    assert data.get("skipped") is True
    return data


def test_optical_reuse_probe_failure_writes_skip(tmp_path, monkeypatch):
    """探针失败 → 返回 None 且写一份「跳过: 探针失败」记录 (不带异常上抛)。"""
    from ao_shaping.tools.slm.transfer import optical_probe as probe_mod
    from ao_shaping.tools.slm.transfer import workflow as workflow_mod

    config = TransferSession(tmp_path / "run", reuse_latest=True)

    def boom(**kwargs):
        raise RuntimeError("camera open failed")
    monkeypatch.setattr(probe_mod, "measure_probe", boom)

    assert workflow_mod._attempt_optical_reuse(config, config.out) is None
    data = _skip_record(config.out)
    assert "探针失败" in data["reason"]
    assert "RuntimeError" in data["reason"]
    assert "camera open failed" in data["reason"]


def test_optical_reuse_no_serial_writes_skip(tmp_path, monkeypatch):
    """读不到 SLM 序列号 → 跳过记录, 并记下探针本身 (便于人工排查)。"""
    from ao_shaping.tools.slm.transfer import optical_probe as probe_mod
    from ao_shaping.tools.slm.transfer import workflow as workflow_mod

    config = TransferSession(tmp_path / "run", reuse_latest=True)
    monkeypatch.setattr(probe_mod, "measure_probe", lambda **kwargs: _probe(serial=None))

    assert workflow_mod._attempt_optical_reuse(config, config.out) is None
    data = _skip_record(config.out)
    assert data["reason"] == "读不到 SLM 序列号"
    assert data["current"]["slm_serial"] is None


def test_optical_reuse_no_latest_writes_skip(tmp_path, monkeypatch):
    """该序列号没有 latest 模型 → 跳过记录, 并记下序列号。"""
    from ao_shaping.tools.slm.transfer import optical_probe as probe_mod
    from ao_shaping.tools.slm.transfer import workflow as workflow_mod

    monkeypatch.setenv("SLM_MODELS_DIR", str(tmp_path / "models"))
    config = TransferSession(tmp_path / "run", reuse_latest=True)
    monkeypatch.setattr(probe_mod, "measure_probe", lambda **kwargs: _probe(serial="SLM-A"))

    assert workflow_mod._attempt_optical_reuse(config, config.out) is None
    data = _skip_record(config.out)
    assert data["reason"] == "该 SLM 没有 latest 模型"
    assert data["serial"] == "SLM-A"


def test_optical_reuse_same_path_returns_plan(tmp_path, monkeypatch):
    """光路一致 → 照旧返回复用计划, 判定写 same_path=True (回归保护)。"""
    from ao_shaping.tools.slm.transfer import optical_probe as probe_mod
    from ao_shaping.tools.slm.transfer import workflow as workflow_mod

    monkeypatch.setenv("SLM_MODELS_DIR", str(tmp_path / "models"))
    latest = tmp_path / "models" / "SLM-A" / "latest"
    latest.mkdir(parents=True)
    (latest / "frozen_parameters.json").write_text(
        json.dumps({"origin": [10.0, 20.0]}), encoding="utf-8"
    )
    (latest / probe_mod.FLAT_FROZEN_SIDECAR).write_text(
        json.dumps({
            "exposure_range_ms": [0.02, 25.0],
            "camera_format": "MONO8",
            "bit_depth": 8,
            "slm_serial": "SLM-A",
        }),
        encoding="utf-8",
    )
    config = TransferSession(tmp_path / "run", reuse_latest=True)
    config.out.mkdir(parents=True, exist_ok=True)  # run_session 里 out 已建好
    monkeypatch.setattr(probe_mod, "measure_probe", lambda **kwargs: _probe(serial="SLM-A"))

    plan = workflow_mod._attempt_optical_reuse(config, config.out)
    assert plan is not None
    data = json.loads((config.out / "optical_probe.json").read_text(encoding="utf-8"))
    assert data["same_path"] is True


def test_run_session_writes_reuse_skip_when_disabled(tmp_path, monkeypatch):
    """开关未开 → 正常全流程照旧, 但落一份「光路复用未开启」说明。"""
    from ao_shaping.tools.slm.transfer import fit

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("[]")
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{}")
    config = TransferSession(tmp_path / "run", dataset=dataset, checkpoint=checkpoint)

    def fail(_):
        raise ValueError("boom")
    monkeypatch.setattr(fit, "fit", fail)
    with pytest.raises(ValueError):
        run_session(config, offline=True)
    assert _skip_record(config.out)["reason"] == "光路复用未开启"


def test_run_session_writes_reuse_skip_when_dataset_session(tmp_path, monkeypatch):
    """数据集会话 + 开关开 → 仍走全流程, 但落「数据集会话不参与」说明。"""
    from ao_shaping.tools.slm.transfer import fit

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("[]")
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{}")
    config = TransferSession(
        tmp_path / "run", dataset=dataset, checkpoint=checkpoint, reuse_latest=True
    )

    def fail(_):
        raise ValueError("boom")
    monkeypatch.setattr(fit, "fit", fail)
    with pytest.raises(ValueError):
        run_session(config, offline=True)
    assert _skip_record(config.out)["reason"] == "数据集会话不参与光路复用"
