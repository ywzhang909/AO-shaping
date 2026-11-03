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
from ml.sim2f.calibrate import CalibrationResult, save_calibration
from ml.sim2f.config import Sim2fConfig


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


def _fake_hw_class(
    captured: dict,
    *,
    applied_exposure: float,
    native_shape: tuple[int, int] = (16, 24),
    zero_order_xy: tuple[int, int] = (12, 8),
):
    real = refine_mod.HwInstrument

    class FakeHw:
        def __init__(self, **kwargs):
            # Bind against the real signature: an unknown or renamed keyword
            # argument must fail here, not on the rig.
            inspect.signature(real.__init__).bind(self, **kwargs)
            captured.update(kwargs)
            self.exposure_ms = applied_exposure
            self.native_shape = native_shape
            self.zero_order_xy = zero_order_xy
            self.closed = False

        def enable_full_frame(self):
            return self.native_shape

        def sent_grayscale(self, phase):
            return None

        def capture_full_frame(self, phase):
            return np.ones(self.native_shape, dtype=np.float64)

        def describe(self):
            return {
                "backend": "fake_hw",
                "native_shape": list(self.native_shape),
                "zero_order_xy": list(self.zero_order_xy),
                "exposure_ms": float(self.exposure_ms),
            }

        def close(self):
            self.closed = True

    return FakeHw


_HW_EXPOSURE_MS = 0.5


def _hardware_source(source_session: Path, tmp_path: Path) -> Path:
    """Restamp the completed dry-run session as a hardware session.

    The frames become (500, 600) canvases and the manifest declares
    ``backend="hw"`` with the recorded zero order ``(300, 200)`` -- the shape a
    physical source session has when the rig has moved since.

    Args:
        source_session: The completed dry-run session.
        tmp_path: Where to copy it.

    Returns:
        The hardware source session directory.
    """
    physical = tmp_path / "hardware_source"
    shutil.copytree(source_session, physical)
    manifest_path = physical / "capture_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["backend"] = "hw"
    manifest["native_shape"] = [500, 600]
    manifest["zero_order_xy"] = [300, 200]
    manifest["exposure_ms"] = _HW_EXPOSURE_MS
    for item in manifest["items"].values():
        frame_path = physical / item["frame"]
        frame = np.load(frame_path, allow_pickle=False)
        canvas = np.zeros((500, 600), dtype=frame.dtype)
        canvas[172 : 172 + frame.shape[0], 72 : 72 + frame.shape[1]] = frame
        np.save(frame_path, canvas)
        item["frame_sha256"] = refine_mod._array_sha256(canvas)
        item["exposure_ms"] = _HW_EXPOSURE_MS
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return physical


def _write_calibration(directory: Path, grid: int) -> None:
    """Write the nominal calibration a non-dry run loads.

    Args:
        directory: The calibration directory (``calibration.json`` + ``.npz``).
        grid: The plan grid the bench is built at.
    """
    save_calibration(
        directory,
        CalibrationResult(
            ok=True,
            reason=None,
            sim_config=Sim2fConfig.calibrated(),
            aberration=np.zeros((grid, grid), dtype=np.float32),
            illumination=np.ones((grid, grid), dtype=np.float32),
            camera={"zoom": 1.0, "shift_x_px": 0.0, "shift_y_px": 0.0},
        ),
    )


def _hardware_arguments(
    source: Path, calibration: Path, out: Path, *extra: str
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
        "1",
        "--iters",
        "2",
        "--lr",
        "0.02",
        *extra,
    ]


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


def _prepare_hardware_run(
    source_session: Path,
    calibration: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    live_zero_order_xy: tuple[int, int],
    out_name: str,
) -> tuple[Path, Path, list[str]]:
    """Stage a non-dry run whose live zero order differs from the recorded one.

    ``refine_mod.ROOT`` is redirected so the mandatory ``data/debug`` output lands
    in ``tmp_path`` instead of the repository.

    Args:
        source_session: The completed dry-run session.
        calibration: The calibration directory to fill.
        tmp_path: Scratch root.
        monkeypatch: Fixture used to swap the hardware constructor and root.
        live_zero_order_xy: What the rig reports when it opens.
        out_name: Output directory name under ``tmp_path/data/debug``.

    Returns:
        ``(source, out, arguments)``.
    """
    source = _hardware_source(source_session, tmp_path)
    plan = json.loads((source / "plan.json").read_text(encoding="utf-8"))
    _write_calibration(calibration, int(plan["grid"]))
    out = tmp_path / "data" / "debug" / out_name
    monkeypatch.setattr(refine_mod, "ROOT", tmp_path)
    monkeypatch.setattr(
        refine_mod,
        "HwInstrument",
        _fake_hw_class(
            {},
            applied_exposure=_HW_EXPOSURE_MS,
            native_shape=(500, 600),
            zero_order_xy=live_zero_order_xy,
        ),
    )
    arguments = _hardware_arguments(
        source, calibration, out, "--exposure-ms", repr(_HW_EXPOSURE_MS)
    )
    return source, out, arguments


def test_default_tolerance_refuses_a_two_pixel_zero_order_drift(
    source_session: Path,
    calibration: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, out, arguments = _prepare_hardware_run(
        source_session,
        calibration,
        tmp_path,
        monkeypatch,
        live_zero_order_xy=(300, 202),
        out_name="drift_refused",
    )
    with pytest.raises(SystemExit):
        refine_mod.main(arguments)
    assert "differs from the source session" in capsys.readouterr().err
    assert not (out / refine_mod.SUMMARY_NAME).exists()


def test_zero_order_tolerance_adopts_the_live_zero_order_and_recaptures_the_reference(
    source_session: Path,
    calibration: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, out, arguments = _prepare_hardware_run(
        source_session,
        calibration,
        tmp_path,
        monkeypatch,
        live_zero_order_xy=(300, 202),
        out_name="drift_adopted",
    )
    refine_mod.main([*arguments, "--zero-order-tolerance-px", "5"])
    summary = _summary(out)
    assert summary["status"] == "complete"
    assert summary["zero_order_recorded_xy"] == [300, 200]
    assert summary["zero_order_effective_xy"] == [300, 202]
    assert summary["zero_order_drift_px"] == 2
    assert summary["zero_order_tolerance_px"] == 5
    assert summary["zero_order_reference_recaptured"] is True
    assert summary["request"]["effective_zero_order_xy"] == [300, 202]
    # Reuse would have been allowed at this exposure, so the recapture above is
    # the drift forcing it, not --recapture-reference.
    assert summary["request"]["recapture_reference"] is False
    assert summary["request"]["exposure_ms"] == _HW_EXPOSURE_MS
    reference = summary["reference"]
    assert reference["capture_method"] == "sequential_native_full_frame"
    assert reference["zero_order_xy"] == [300, 202]
    assert reference["frame_max"] == 1.0
    assert reference["window_peak"] == 1.0
    assert set(summary["rounds"][0]) >= {"frame_max", "window_peak"}


def test_zero_order_drift_beyond_the_tolerance_still_refuses(
    source_session: Path,
    calibration: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, out, arguments = _prepare_hardware_run(
        source_session,
        calibration,
        tmp_path,
        monkeypatch,
        live_zero_order_xy=(300, 204),
        out_name="drift_too_large",
    )
    with pytest.raises(SystemExit):
        refine_mod.main([*arguments, "--zero-order-tolerance-px", "1"])
    assert "differs from the source session" in capsys.readouterr().err
    assert not (out / refine_mod.SUMMARY_NAME).exists()


def test_min_window_peak_count_fails_the_run_and_marks_the_summary_failed(
    source_session: Path, calibration: Path, tmp_path: Path
) -> None:
    out = tmp_path / "low_snr"
    with pytest.raises(SystemExit):
        refine_mod.main(
            [
                *_arguments(source_session, calibration, out),
                "--min-window-peak-count",
                "1e9",
            ]
        )
    summary = _summary(out)
    assert summary["status"] == "failed"
    assert "round_0001" in summary["error"]
    assert "1e+09" in summary["error"]
    assert "peaks at" in summary["error"]
    assert "--min-window-peak-count" in summary["error"]
    assert (out / "accepted_phase.npy").is_file()


def test_rejected_round_advances_the_search_start_to_its_candidate(
    source_session: Path, calibration: Path, tmp_path: Path
) -> None:
    out = tmp_path / "rejected_advance"
    refine_mod.main([*_arguments(source_session, calibration, out, rounds=2), "--min-gain", "1e9"])
    summary = _summary(out)
    rounds = summary["rounds"]
    assert len(rounds) == 2
    assert rounds[0]["status"] == "rejected"
    # A rejected round must not replay the identical candidate next round.
    assert rounds[1]["phase_sha256"] != rounds[0]["phase_sha256"]
