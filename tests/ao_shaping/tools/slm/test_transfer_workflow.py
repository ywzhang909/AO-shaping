"""Workflow boundaries, archive integrity and lazy source-package imports."""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

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


def _patch_capture_devices(monkeypatch) -> tuple[list[str], list[dict]]:
    """Replace ``capture.capture``'s device layer.

    Returns the close log and every ``adapt_exposure`` keyword set, so a test can
    pin how the exposure is armed as well as what was written.
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

    def adapt(setter, read, **kwargs):
        calls.append(dict(kwargs))
        setter(.123)
        read()
        return _Setting()

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

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._closed.append("camera")

    def reset_exposure_time(self, value):
        self.readback_ms = value

    def get_numpy_image(self, **kwargs):
        yy, xx = np.mgrid[:50, :60]
        base = 5 + 40*np.exp(-((xx-4)**2+(yy-4)**2)/20)
        return np.clip(base*(1+.5*self.level)+3*self.level, 0, 240).astype(np.uint8)


class _LevelSlm:
    """SLM stub: encodes the phase's mean amplitude as a grayscale level."""

    def __init__(self, camera, closed):
        self._camera = camera
        self._closed = closed

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


def _run_validate(tmp_path, monkeypatch, *, abba, mode="model", updates=2, device=None, seen=None,
                  adaptive_exposure=False, blocks=1, **extra):
    """Drive ``validate.run`` end to end with fakes: devices, forward and optimizer.

    Everything above the device layer is the production code, so the set of
    captures actually taken is what is asserted -- not a re-implementation of it.
    ``seen`` (a dict) collects the ``device`` each stubbed stage construction
    received, which is how the resolver plumbing is asserted without a GPU.
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
    monkeypatch.setattr(validate, "target_at",
                        lambda xy, shape, centre, side: np.pad(np.ones((5, 5), np.float32), 2))
    monkeypatch.setattr(validate, "gs_seed",
                        lambda forward, centre, side, shape: np.zeros((8, 8), np.float32))
    monkeypatch.setattr(hardware, "prepare_camera_sdk", lambda _: None)
    monkeypatch.setattr(validate.time, "sleep", lambda _: None)

    closed = []
    camera = _PhaseCamera(closed)
    monkeypatch.setattr(common, "create_camera", lambda *a, **kw: camera)
    monkeypatch.setattr(common, "get_camera_exposure_range", lambda _: (.02, 10.))
    monkeypatch.setattr(common, "get_camera_exposure_ms", lambda cam: cam.readback_ms)
    monkeypatch.setitem(sys.modules, "ao_shaping.drivers.slm.santec",
                        SimpleNamespace(Santec=lambda **kwargs: _LevelSlm(camera, closed)))

    def adapt(setter, read, **kwargs):
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


def test_runtime_exposure_changes_between_blocks_but_not_within_pair(tmp_path, monkeypatch):
    args = _run_validate(tmp_path, monkeypatch, abba=True, mode="anchored", adaptive_exposure=True, blocks=2)
    rows = {r["label"]: r for r in json.loads((args.out/"manifest.json").read_text())}
    for block, expected in ((0, .5), (1, .7)):
        assert {rows[f"square_{block}_{suffix}"]["exposure_ms"] for suffix in ("A1", "B1", "B2", "A2")} == {expected}
    assert rows["square_seed_replay"]["exposure_ms"] == rows["square_final"]["exposure_ms"] == .5
    assert (args.out/"square_1_exposure.npz").exists()


def test_audit_allows_block_exposure_but_rejects_mixed_pair(tmp_path):
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


def test_abba_off_in_anchored_mode_keeps_only_the_anchor(tmp_path, monkeypatch):
    args = _run_validate(tmp_path, monkeypatch, abba=False, mode="anchored")

    # The optimizer is anchored on A1, so that one capture stays.
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

    def fake_capture(args):
        seen["capture"] = args
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

    def fake_capture(args):
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
