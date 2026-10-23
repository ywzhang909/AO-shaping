"""Workflow boundaries, archive integrity and lazy source-package imports."""
from __future__ import annotations

import json
import subprocess
import sys
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
for name in ("capture", "fit", "validate", "bootstrap", "capture_response", "response", "blur", "direction", "summary", "workflow", "hardware"):
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


def test_session_rejects_incomplete_calibration_shapes(tmp_path):
    with pytest.raises(ValueError, match="skip-calibration"):
        TransferSession(tmp_path, shapes=("square",)).validate()


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
    from dataclasses import dataclass
    from types import SimpleNamespace
    from ao_shaping.drivers.ccd import common
    from ao_shaping.tools.slm.panel import focus_probe
    from ao_shaping.tools.slm.transfer import capture, hardware

    closed = []
    class Camera:
        readback_ms = .123
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            closed.append("camera")
        def reset_exposure_time(self, value):
            self.readback_ms = value
        def get_numpy_image(self, **kwargs):
            yy, xx = np.mgrid[:50, :60]
            return (1+80*np.exp(-((xx-30)**2+(yy-25)**2)/20)).astype(np.uint8)
    class Slm:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            closed.append("slm")
        def set_phase_overlay(self, value):
            pass
        def create_phase_from_array(self, phase):
            return np.zeros(phase.shape, dtype=np.uint16)
        def display_data(self, data):
            pass
    @dataclass
    class Setting:
        status: str = "ok"
    camera = Camera()
    monkeypatch.setattr(hardware, "prepare_camera_sdk", lambda _: None)
    monkeypatch.setattr(common, "create_camera", lambda *a, **kw: camera)
    monkeypatch.setattr(common, "get_camera_exposure_range", lambda _: (.02, 10.))
    monkeypatch.setattr(common, "get_camera_exposure_ms", lambda cam: cam.readback_ms)
    monkeypatch.setitem(sys.modules, "ao_shaping.drivers.slm.santec", SimpleNamespace(Santec=Slm))
    def adapt(setter, read, **kwargs):
        setter(.123)
        read()
        return Setting()
    monkeypatch.setattr(focus_probe, "adapt_exposure", adapt)
    monkeypatch.setattr(capture.time, "sleep", lambda _: None)
    monkeypatch.setattr(capture, "probes", lambda *a: [("reference_start", "calibration", np.zeros((8, 8), np.float32))])
    out = tmp_path / "capture"
    capture.capture(SimpleNamespace(out=out, camera="custom", cam_id=7, slm_number=2, grid=8, count=6, frames=3, seed=1))
    row = json.loads((out / "manifest.json").read_text())[0]
    assert row["exposure_ms"] == .123
    assert all(frame["exposure_ms"] == .123 for frame in row["frames"])
    assert closed == ["slm", "camera"]
