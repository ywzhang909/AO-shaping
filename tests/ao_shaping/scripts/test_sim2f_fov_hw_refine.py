"""Dry-run contract tests for measured-acceptance sim2f hardware refinement.

All acquisitions use ``SimInstrument`` through a real completed dry-run source
session.  The hardware constructor is explicitly poisoned so an accidental
non-dry acquisition fails the test instead of opening a device.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def _load_script(name: str, filename: str):
    path = ROOT / "scripts" / "ml" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


session_mod = _load_script("sim2f_fov_hw_session_for_refine_test", "sim2f_fov_hw_session.py")
refine_mod = _load_script("sim2f_fov_hw_refine_under_test", "sim2f_fov_hw_refine.py")


@pytest.fixture(autouse=True)
def _no_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("a refinement test attempted to open real hardware")

    monkeypatch.setattr(refine_mod, "HwInstrument", forbidden)


@pytest.fixture(scope="module")
def source_session(tmp_path_factory: pytest.TempPathFactory) -> Path:
    session = tmp_path_factory.mktemp("refine_source") / "session"
    session_mod.main(
        [
            "plan",
            "--out",
            str(session),
            "--window",
            "128",
            "--screens",
            "1",
            "--repeats",
            "2",
            "--targets",
            "ring",
        ]
    )
    session_mod.main(["capture", "--session", str(session), "--dry-run"])
    return session


@pytest.fixture()
def calibration(tmp_path: Path) -> Path:
    directory = tmp_path / "fitted_calibration"
    directory.mkdir()
    return directory


def _arguments(
    source: Path, calibration: Path, out: Path, *, rounds: int = 1
) -> list[str]:
    return [
        "--source",
        str(source),
        "--calibration",
        str(calibration),
        "--out",
        str(out),
        "--target",
        "ring",
        "--start-phase",
        "phases/shape_ring_nominal_r0.npy",
        "--rounds",
        str(rounds),
        "--iters",
        "2",
        "--lr",
        "0.02",
        "--dry-run",
    ]


def _summary(out: Path) -> dict:
    return json.loads((out / refine_mod.SUMMARY_NAME).read_text(encoding="utf-8"))


def test_physical_source_dry_run_requires_recaptured_reference(
    source_session: Path, calibration: Path, tmp_path: Path
) -> None:
    physical_source = tmp_path / "physical_source"
    shutil.copytree(source_session, physical_source)
    manifest_path = physical_source / "capture_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["backend"] = "hw"
    manifest["native_shape"] = [500, 600]
    manifest["zero_order_xy"] = [300, 200]
    for item in manifest["items"].values():
        frame_path = physical_source / item["frame"]
        frame = np.load(frame_path, allow_pickle=False)
        canvas = np.zeros((500, 600), dtype=frame.dtype)
        canvas[172 : 172 + frame.shape[0], 72 : 72 + frame.shape[1]] = frame
        np.save(frame_path, canvas)
        item["frame_sha256"] = refine_mod._array_sha256(canvas)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    refused = tmp_path / "refused"
    with pytest.raises(SystemExit):
        refine_mod.main(_arguments(physical_source, calibration, refused))
    assert not (refused / refine_mod.SUMMARY_NAME).exists()

    out = tmp_path / "physical_dry_run"
    refine_mod.main(
        [*_arguments(physical_source, calibration, out), "--recapture-reference"]
    )
    summary = _summary(out)
    assert summary["status"] == "complete"
    assert summary["request"]["dry_run"] is True
    assert summary["native_shape"] == [500, 600]
    assert summary["source_zero_order_xy"] == [300, 200]
    assert summary["zero_order_xy"] == [300, 250]
    assert summary["reference"]["capture_method"] == "sequential_native_full_frame"
    assert summary["reference"]["capture"]["dry_run_canvas"]["zero_order_xy"] == [
        300,
        250,
    ]
    for key in ("reference", "rounds"):
        records = [summary[key]] if key == "reference" else summary[key]
        for record in records:
            if key == "reference":
                measured = record["measured"]
                predicted = record["predicted"]
            else:
                measured = record["measured"]["candidate"]
                predicted = record["predicted"]["candidate"]
            assert measured == pytest.approx(predicted, abs=1e-6)
    assert "not real CCD success" in summary["dry_run_warning"]


def test_native_capture_acceptance_and_frozen_calibration(
    source_session: Path, calibration: Path, tmp_path: Path
) -> None:
    out = tmp_path / "refine"
    source_frame = source_session / "frames" / "shape_ring_nominal_r0.npy"
    source_phase = source_session / "phases" / "shape_ring_nominal_r0.npy"
    before = (source_frame.read_bytes(), source_phase.read_bytes())

    returned = refine_mod.main(
        _arguments(source_session, calibration, out, rounds=2)
    )
    summary = _summary(out)

    assert returned["status"] == summary["status"] == "complete"
    assert summary["acceptance"] == "sequential_ccd"
    assert summary["measured_source"] == "native_full_frames_cropped_offline_no_device_roi"
    assert summary["calibration_mode"] == "nominal_twin_dry_run_calibration_not_loaded"
    assert "frozen" in summary["limitation"]
    assert "not real CCD success" in summary["dry_run_warning"]
    assert summary["reference"]["capture_method"] == "reused_source_native_frame"
    assert np.load(out / "target.npy").sum() == pytest.approx(1.0)
    assert (out / "accepted_phase.npy").is_file()
    assert [record["id"] for record in summary["rounds"]] == [
        "round_0001",
        "round_0002",
    ]

    for record in [summary["reference"], *summary["rounds"]]:
        native = np.load(out / record["native_frame"], allow_pickle=False)
        assert native.shape == tuple(summary["native_shape"])
        assert refine_mod._sha256(out / record["native_frame"]) == record[
            "native_frame_sha256"
        ]
        assert refine_mod._sha256(out / record["phase"]) == record["phase_sha256"]
        assert record["exposure_ms"] == summary["request"]["exposure_ms"]
        if record["panel_gray"] is not None:
            assert (out / record["panel_gray"]).is_file()
        measured = record["measured"]
        objective = measured["objective"] if "objective" in measured else measured["candidate"]["objective"]
        assert np.isfinite(objective)

    for record in summary["rounds"]:
        assert set(record["predicted"]) == {"reference", "candidate", "gain"}
        assert set(record["measured"]) == {"reference", "candidate", "gain"}
        assert record["status"] == ("accepted" if record["accepted"] else "rejected")
        if record["accepted"]:
            assert record["measured"]["gain"] > summary["request"]["measured_min_gain"]
    assert before == (source_frame.read_bytes(), source_phase.read_bytes())


def test_resume_skips_valid_capture_and_rejects_modified_phase(
    source_session: Path,
    calibration: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "resume"
    arguments = _arguments(source_session, calibration, out)
    first = refine_mod.main(arguments)
    round_path = out / first["rounds"][0]["native_frame"]
    phase_path = out / first["rounds"][0]["phase"]
    frame_bytes = round_path.read_bytes()
    phase_bytes = phase_path.read_bytes()

    def no_capture(*args, **kwargs):
        raise AssertionError("a valid record was captured again")

    monkeypatch.setattr(refine_mod, "_capture_artifact", no_capture)
    resumed = refine_mod.main(arguments)
    assert resumed["status"] == "complete"
    assert [record["id"] for record in resumed["rounds"]] == ["round_0001"]
    assert round_path.read_bytes() == frame_bytes
    assert phase_path.read_bytes() == phase_bytes

    modified = np.load(phase_path, allow_pickle=False)
    modified[0, 0] += 0.25
    np.save(phase_path, modified, allow_pickle=False)
    with pytest.raises(SystemExit):
        refine_mod.main(arguments)
    assert not (out / "frames" / "round_0002.npy").exists()
    assert round_path.read_bytes() == frame_bytes


def test_capture_failure_closes_sim_instrument_and_saves_failure(
    source_session: Path,
    calibration: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances = []
    base = refine_mod.SimInstrument

    class FailingInstrument(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.test_closed = False
            instances.append(self)

        def capture(self, phase):
            raise RuntimeError("injected capture failure")

        def close(self):
            self.test_closed = True
            super().close()

    monkeypatch.setattr(refine_mod, "SimInstrument", FailingInstrument)
    out = tmp_path / "failure"
    with pytest.raises(SystemExit):
        refine_mod.main(_arguments(source_session, calibration, out))

    assert len(instances) == 1
    assert instances[0].test_closed
    summary = _summary(out)
    assert summary["status"] == "failed"
    assert "injected capture failure" in summary["error"]
    assert (out / "accepted_phase.npy").is_file()


def _fake_hw_class(captured: dict, *, applied_exposure: float):
    real = refine_mod.HwInstrument

    class FakeHw:
        def __init__(self, **kwargs):
            # Bind against the real signature: an unknown or renamed keyword
            # argument must fail here, not on the rig.
            inspect.signature(real.__init__).bind(self, **kwargs)
            captured.update(kwargs)
            self.exposure_ms = applied_exposure
            self.native_shape = (16, 24)
            self.zero_order_xy = (12, 8)
            self.closed = False

        def enable_full_frame(self):
            return self.native_shape

        def close(self):
            self.closed = True

    return FakeHw


def test_hardware_open_passes_only_supported_constructor_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        refine_mod, "HwInstrument", _fake_hw_class(captured, applied_exposure=1.753)
    )
    lazy = refine_mod._LazyInstrument(
        dry_run=False,
        full_cfg=refine_mod.Sim2fConfig.calibrated(),
        native_shape=(16, 24),
        zero_order_xy=(12, 8),
        exposure_ms=1.753,
        settle_s=0.05,
        cam_type="daheng",
        cam_id=0,
        window_px=128,
        panel_shape=None,
        pupil_center_xy=None,
    )
    instrument = lazy.open()
    try:
        assert captured["ccd_px"] == 128
        assert captured["exposure_ms"] == 1.753
        assert instrument is lazy.instrument
    finally:
        lazy.close()
    assert instrument.closed


def test_hardware_open_refuses_a_different_applied_exposure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        refine_mod, "HwInstrument", _fake_hw_class(captured, applied_exposure=2.0)
    )
    lazy = refine_mod._LazyInstrument(
        dry_run=False,
        full_cfg=refine_mod.Sim2fConfig.calibrated(),
        native_shape=(16, 24),
        zero_order_xy=(12, 8),
        exposure_ms=1.753,
        settle_s=0.05,
        cam_type="daheng",
        cam_id=0,
        window_px=128,
        panel_shape=None,
        pupil_center_xy=None,
    )
    with pytest.raises(RuntimeError, match="refusing to score a different exposure"):
        lazy.open()
    assert lazy.instrument is None


def test_physical_output_outside_data_debug_is_rejected_before_opening(
    source_session: Path, calibration: Path, tmp_path: Path
) -> None:
    arguments = _arguments(source_session, calibration, tmp_path / "physical")
    arguments.remove("--dry-run")
    with pytest.raises(SystemExit):
        refine_mod.main(arguments)
    assert not (tmp_path / "physical").exists()
