"""Offline tests for the A1 zero-order alignment probe (single flat pattern)."""

from __future__ import annotations

import argparse
import builtins
import importlib
import json
import sys
import types

import numpy as np
import pytest

SCRIPT = importlib.import_module("scripts.slm.probe_a1_zero_order_alignment_20261127")
S1 = importlib.import_module("scripts.slm.capture_s1_mapping_disambiguation_20261124")


def _plan_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "exposure_ms": SCRIPT.DEFAULT_EXPOSURE_MS,
        "wait_until_idle": SCRIPT.DEFAULT_WAIT_IDLE_S,
        "slm_number": SCRIPT.DEFAULT_SLM_NUMBER,
        "cam_type": SCRIPT.DEFAULT_CAM_TYPE,
        "cam_id": SCRIPT.DEFAULT_CAM_ID,
        "frames": SCRIPT.DEFAULT_FRAMES,
        "settle_s": SCRIPT.DEFAULT_SETTLE_S,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _plan() -> dict[str, object]:
    return SCRIPT.build_plan(_plan_args())


# ---------------------------------------------------------------------------
# Frozen plan / constants
# ---------------------------------------------------------------------------
def test_plan_reference_and_tolerance_match_s1_criterion_one() -> None:
    plan = _plan()
    criteria = plan["counter_criteria"]
    assert criteria["zero_order_reference_px"] == {"row": 1105.249, "col": 1227.556}
    assert criteria["tolerance_px"] == 5.0
    assert criteria["observed_20261126_within_tolerance"] is False
    assert criteria["observed_20261126_centroid_px"] == {"row": 1018.9, "col": 1273.0}
    # Same values as the S1 module's frozen criterion ①, without touching it.
    assert criteria["zero_order_reference_px"] == S1.ZERO_ORDER_REFERENCE_PX
    assert criteria["tolerance_px"] == S1.ZERO_ORDER_REFERENCE_TOLERANCE_PX
    assert SCRIPT.ZERO_ORDER_REFERENCE_PX == {"row": 1105.249, "col": 1227.556}
    assert SCRIPT.TOLERANCE_PX == 5.0


def test_plan_is_one_flat_pattern_with_device_scale_993_and_modulus_994() -> None:
    plan = _plan()
    pattern = plan["pattern"]
    assert pattern["id"] == "flat_scale993"
    assert pattern["kind"] == "flat"
    assert pattern["two_pi_gray_scale"] == 993
    assert pattern["gray_modulus"] == 994
    assert pattern["gray_modulus"] == pattern["two_pi_gray_scale"] + 1
    assert pattern["panel_shape_rows_cols"] == [1200, 1920]
    assert pattern["focus_d_m1"] == 0.0 and pattern["compensation_applied"] is False
    assert len(pattern["phase_sha256"]) == 64
    assert len(pattern["gray_sha256"]) == 64
    # A flat phase encodes to gray 0 everywhere at scale 993.
    assert pattern["gray_max"] == 0
    assert pattern["gray_unique_count"] == 1


def test_plan_frames_count_is_a_cli_knob_but_not_six_patterns() -> None:
    plan = _plan()
    assert plan["n_frames"] == SCRIPT.DEFAULT_FRAMES == 3
    assert "patterns" not in plan  # single pattern, not the six-frame protocol
    assert SCRIPT.build_plan(_plan_args(frames=5))["n_frames"] == 5
    with pytest.raises(ValueError, match="frames"):
        SCRIPT.build_plan(_plan_args(frames=0))


def test_plan_device_parameters_and_scope_note() -> None:
    plan = _plan()
    acquisition = plan["acquisition"]
    assert acquisition["slm_number"] == 1
    assert acquisition["cam_type"] == "daheng"
    assert acquisition["cam_id"] == 0
    assert acquisition["exposure_ms"] == pytest.approx(0.087)
    assert acquisition["wait_until_idle_s"] == pytest.approx(600.0)
    assert acquisition["settle_s"] == pytest.approx(0.8)
    assert plan["bit_depth_required"] == 12
    assert plan["full_scale_required"] == 4095
    assert plan["native_ccd_shape_rows_cols"] == [1944, 2592]
    assert plan["panel_shape_expected_rows_cols"] == [1200, 1920]
    assert plan["preflight"]["discard_frames"] == 3
    assert plan["preflight"]["abort_on_saturation"] is True
    assert plan["preflight"]["saturation_fraction"] == pytest.approx(0.9)
    assert "不替代六帧判据②-⑥" in plan["counter_criteria"]["scope_note"]
    assert plan["exit_codes"]["3"] == "measured, outside 5 px tolerance"


# ---------------------------------------------------------------------------
# Centroid convention (must be S1's own function)
# ---------------------------------------------------------------------------
def test_centroid_function_is_the_s1_function_object() -> None:
    assert SCRIPT._centroid_px is S1._centroid_px


def test_centroid_on_synthetic_single_peak() -> None:
    frame = np.zeros((61, 61), dtype=np.uint16)
    frame[29:32, 40:43] = 500
    assert SCRIPT._centroid_px(frame) == [30.0, 41.0]
    assert SCRIPT._centroid_px(frame) == S1._centroid_px(frame)


def test_centroid_blank_frame_falls_back_to_geometric_centre() -> None:
    blank = np.zeros((5, 7), dtype=np.uint16)
    assert SCRIPT._centroid_px(blank) == [2.0, 3.0]


# ---------------------------------------------------------------------------
# No-signal gate (must reuse S1 v2's flat_signal_measurement)
# ---------------------------------------------------------------------------
def test_signal_gate_is_the_s1_function_object() -> None:
    assert SCRIPT.flat_signal_measurement is S1.flat_signal_measurement


def test_signal_gate_fails_closed_on_blank_and_noise_frames() -> None:
    blank = np.zeros((61, 61), dtype=np.uint16)
    blank_measure = SCRIPT.flat_signal_measurement(blank)
    assert blank_measure["has_signal"] is False
    assert blank_measure["centroid_is_fallback"] is True
    assert blank_measure["centroid_px"] == [30.0, 30.0]
    assert blank_measure["peak_snr"] == 0.0
    assert blank_measure["signal_pixels"] == 0
    assert blank_measure["signal_weight_counts"] == 0.0

    noise = np.random.default_rng(20261127).integers(0, 6, size=(61, 61))
    noise_measure = SCRIPT.flat_signal_measurement(noise.astype(np.uint16))
    assert noise_measure["has_signal"] is False
    assert noise_measure["centroid_is_fallback"] is True


def test_signal_gate_passes_a_synthetic_single_peak() -> None:
    frame = np.zeros((61, 61), dtype=np.uint16)
    frame[29:32, 40:43] = 500
    measure = SCRIPT.flat_signal_measurement(frame)
    assert measure["has_signal"] is True
    assert measure["centroid_is_fallback"] is False
    assert measure["centroid_px"] == [30.0, 41.0]


def test_plan_registers_exit_code_four_and_s1_gate_values() -> None:
    plan = _plan()
    assert plan["exit_codes"]["4"] == (
        "no_signal (fail-closed; centroid is fallback, not a measurement)"
    )
    assert SCRIPT.EXIT_NO_SIGNAL == 4
    gate = plan["signal_gate"]
    assert gate["source"].endswith("flat_signal_measurement")
    assert gate["peak_snr_min"] == S1.SIGNAL_PEAK_SNR_MIN == 50.0
    assert gate["signal_pixels_min"] == S1.SIGNAL_MIN_PIXELS == 4
    assert gate["signal_weight_counts_min"] == S1.SIGNAL_MIN_WEIGHT == 2000.0
    assert gate["fail_closed_exit_code"] == SCRIPT.EXIT_NO_SIGNAL


# ---------------------------------------------------------------------------
# Pure offset / verdict function
# ---------------------------------------------------------------------------
def test_offset_at_reference_is_zero_and_within() -> None:
    verdict = SCRIPT.offset_from_reference([1105.249, 1227.556])
    assert verdict["within_tolerance"] is True
    assert verdict["dist_px"] == 0.0
    assert verdict["drow_px"] == 0.0
    assert verdict["dcol_px"] == 0.0


def test_offset_of_the_20261126_observation_fails_criterion_one() -> None:
    verdict = SCRIPT.offset_from_reference([1018.9, 1273.0])
    assert verdict["drow_px"] == pytest.approx(-86.349)
    assert verdict["dcol_px"] == pytest.approx(45.444)
    assert verdict["dist_px"] == pytest.approx(97.577, abs=0.01)
    assert verdict["within_tolerance"] is False


def test_offset_boundary_at_five_px_is_inclusive() -> None:
    inside = SCRIPT.offset_from_reference([1105.249 + 5.0, 1227.556])
    assert inside["dist_px"] == 5.0
    assert inside["within_tolerance"] is True
    # 3-4-5 diagonal: hypotenuse exactly 5.0.
    diagonal = SCRIPT.offset_from_reference([1105.249 + 3.0, 1227.556 + 4.0])
    assert diagonal["dist_px"] == pytest.approx(5.0)
    assert diagonal["within_tolerance"] is True
    outside = SCRIPT.offset_from_reference([1105.249 + 5.0 + 0.01, 1227.556])
    assert outside["within_tolerance"] is False


def test_mean_centroid_averages_frames() -> None:
    assert SCRIPT.mean_centroid([[10.0, 20.0], [12.0, 24.0]]) == [11.0, 22.0]
    with pytest.raises(ValueError):
        SCRIPT.mean_centroid([])


# ---------------------------------------------------------------------------
# Plan mode and self-test: no drivers, no directories, no hardware
# ---------------------------------------------------------------------------
def test_plan_mode_does_not_import_drivers_or_create_directory(
    tmp_path, capsys, monkeypatch
) -> None:
    real_import = builtins.__import__
    seen: list[str] = []

    def spy(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.startswith("ao_shaping.drivers"):
            seen.append(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", spy)
    out = tmp_path / "run"
    assert SCRIPT.main(["--out", str(out)]) == 0
    assert not out.exists()
    assert seen == []
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == SCRIPT.PLAN_SCHEMA
    assert payload["pattern"]["id"] == "flat_scale993"
    assert payload["n_frames"] == 3


def test_invalid_cli_values_are_rejected(tmp_path) -> None:
    with pytest.raises(SystemExit):
        SCRIPT.main(["--out", str(tmp_path / "run"), "--frames", "0"])
    with pytest.raises(SystemExit):
        SCRIPT.main(["--out", str(tmp_path / "run"), "--settle-s", "-1"])


def test_self_test_returns_zero(capsys) -> None:
    assert SCRIPT.main(["--self-test"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["self_test"] == "passed"


# ---------------------------------------------------------------------------
# Execute path with fake devices (no driver import, no hardware)
# ---------------------------------------------------------------------------
class _FakeLock:
    instances: list["_FakeLock"] = []

    def __init__(self) -> None:
        self.released = False
        _FakeLock.instances.append(self)

    def release(self) -> None:
        self.released = True


class _FakeSLM:
    def __init__(self, **_: object) -> None:
        self.Panel_Res = (1920, 1200)  # columns, rows (driver convention)
        self.opens = 0
        self.grays: list[np.ndarray] = []
        self.closed = False
        self.gray0 = None

    def open(self) -> None:
        self.opens += 1

    def set_phase_overlay(self, value: object) -> None:
        pass

    def display_data(self, gray: np.ndarray) -> None:
        self.grays.append(np.array(gray, copy=True))

    def set_grayscale(self, value: int) -> None:
        self.gray0 = int(value)

    def close(self) -> None:
        self.closed = True


class _FakeCam:
    def __init__(self, fill: int, *, peak_row: int, peak_col: int) -> None:
        self.bit_depth = 12
        self.fill = int(fill)
        self.peak_row = int(peak_row)
        self.peak_col = int(peak_col)
        self.opens = 0
        self.reads = 0
        self.closed = False
        self.exposure_ms = 0.0

    def open(self) -> None:
        self.opens += 1

    def close(self) -> None:
        self.closed = True

    def get_numpy_image(self, n: int, full: bool) -> np.ndarray:
        self.reads += 1
        frame = np.zeros(SCRIPT.NATIVE_CCD_SHAPE, dtype=np.uint16)
        frame[self.peak_row - 1 : self.peak_row + 2,
              self.peak_col - 1 : self.peak_col + 2] = self.fill
        return frame

    def effective_full_scale(self, _probe: np.ndarray) -> float:
        return float(SCRIPT.FULL_SCALE_REQUIRED)


def _install_fake_devices(
    monkeypatch, *, fill: int, peak_row: int = 1105, peak_col: int = 1227
) -> tuple[_FakeSLM, _FakeCam]:
    slm = _FakeSLM()
    cam = _FakeCam(fill, peak_row=peak_row, peak_col=peak_col)
    lock = _FakeLock()

    monkeypatch.setattr(SCRIPT, "acquire_device_lock", lambda _wait: lock)
    monkeypatch.setattr(SCRIPT, "open_slm_guarded", lambda device: device.open())

    ccd = types.ModuleType("ao_shaping.drivers.ccd.common")
    ccd.create_camera = lambda *a, **k: cam
    ccd.get_camera_exposure_ms = lambda device: device.exposure_ms
    ccd.set_camera_exposure_ms = lambda device, value: setattr(
        device, "exposure_ms", float(value)
    )
    santec = types.ModuleType("ao_shaping.drivers.slm.santec")
    santec.Santec = lambda **kwargs: slm
    monkeypatch.setitem(sys.modules, "ao_shaping.drivers.ccd.common", ccd)
    monkeypatch.setitem(sys.modules, "ao_shaping.drivers.slm.santec", santec)
    return slm, cam


def _run_execute(monkeypatch, tmp_path, *, fill: int, frames: int = 3, **peaks):
    slm, cam = _install_fake_devices(monkeypatch, fill=fill, **peaks)
    out = tmp_path / "run"
    rc = SCRIPT.main(
        [
            "--execute",
            "--out",
            str(out),
            "--wait-until-idle",
            "0",
            "--frames",
            str(frames),
        ]
    )
    return rc, out, slm, cam


def test_execute_writes_frames_measurement_and_release_artifacts(
    monkeypatch, tmp_path
) -> None:
    rc, out, slm, cam = _run_execute(monkeypatch, tmp_path, fill=1000)
    assert rc == 0
    assert slm.opens == 1 and slm.closed and slm.gray0 == 0
    assert cam.opens == 1 and cam.closed
    assert _FakeLock.instances[-1].released is True
    assert sorted(p.name for p in (out / "raw").iterdir()) == [
        "00_flat.npy",
        "01_flat.npy",
        "02_flat.npy",
    ]
    for name in (
        "plan.json",
        "capture_journal.jsonl",
        "alignment_measurement.json",
        "released.json",
    ):
        assert (out / name).is_file(), name

    measurement = json.loads(
        (out / "alignment_measurement.json").read_text(encoding="utf-8")
    )
    assert measurement["status"] == "ok"
    assert measurement["n_frames_requested"] == measurement["n_frames_used"] == 3
    assert measurement["pattern"]["two_pi_gray_scale"] == 993
    assert measurement["pattern"]["gray_modulus"] == 994
    assert measurement["camera"]["bit_depth"] == 12
    assert measurement["camera"]["full_scale"] == 4095
    assert measurement["camera"]["exposure_ms_actual"] == pytest.approx(0.087)
    assert measurement["panel_shape_rows_cols"] == [1200, 1920]
    assert measurement["any_frame_saturated"] is False
    assert len(measurement["frames"]) == 3
    for row in measurement["frames"]:
        assert row["pattern_id"] == "flat_scale993"
        assert len(row["frame_sha256"]) == 64
        assert row["frame_shape"] == [1944, 2592]
        assert row["saturated"] is False
        assert row["raw_file"].startswith("raw/")
        assert row["has_signal"] is True
        assert row["centroid_is_fallback"] is False
        assert row["offset_from_reference"] is not None
    assert measurement["has_signal"] is True
    assert measurement["centroid_is_fallback"] is False
    assert measurement["n_frames_with_signal"] == 3
    assert measurement["mean_centroid_px"] == pytest.approx([1105.0, 1227.0])
    assert measurement["mean_offset_from_reference"]["drow_px"] == pytest.approx(-0.249)
    assert measurement["mean_offset_from_reference"]["dcol_px"] == pytest.approx(-0.556)
    assert measurement["within_tolerance"] is True

    released = json.loads((out / "released.json").read_text(encoding="utf-8"))
    assert released["slm_closed"] is True
    assert released["camera_closed"] is True
    assert released["lock_released"] is True
    assert released["closed_normally"] is True
    assert released["status"] == "ok"
    assert released["error"] is None
    for key in ("acquired_at_utc", "finished_at_utc", "released_at_utc"):
        assert released[key]

    events = [
        json.loads(line)["event"]
        for line in (out / "capture_journal.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[0] == "lock_acquired"
    assert events[1] == "devices_ready"
    assert events.count("frame") == 3
    assert events.count("centroid") == 3
    assert "preflight" in events
    assert "verdict" in events
    assert events[-1] == "release"


def test_execute_reports_outside_tolerance_without_failing_the_measurement(
    monkeypatch, tmp_path
) -> None:
    rc, out, _slm, _cam = _run_execute(
        monkeypatch, tmp_path, fill=1000, peak_row=1019, peak_col=1273
    )
    assert rc == 3
    measurement = json.loads(
        (out / "alignment_measurement.json").read_text(encoding="utf-8")
    )
    assert measurement["status"] == "ok"
    assert measurement["within_tolerance"] is False
    assert measurement["mean_offset_from_reference"]["dist_px"] > 5.0
    released = json.loads((out / "released.json").read_text(encoding="utf-8"))
    assert released["status"] == "ok"
    assert released["closed_normally"] is True


def test_execute_without_signal_fails_closed_with_exit_code_four(
    monkeypatch, tmp_path
) -> None:
    rc, out, slm, cam = _run_execute(monkeypatch, tmp_path, fill=0)
    assert rc == SCRIPT.EXIT_NO_SIGNAL == 4
    assert slm.closed and cam.closed
    assert _FakeLock.instances[-1].released is True
    measurement = json.loads(
        (out / "alignment_measurement.json").read_text(encoding="utf-8")
    )
    assert measurement["status"] == "no_signal"
    assert measurement["has_signal"] is False
    assert measurement["centroid_is_fallback"] is True
    assert measurement["n_frames_with_signal"] == 0
    assert measurement["within_tolerance"] is None
    assert measurement["mean_centroid_px"] is None
    assert measurement["mean_offset_from_reference"] is None
    assert measurement["exit_code"] == SCRIPT.EXIT_NO_SIGNAL
    geometric_row = (SCRIPT.NATIVE_CCD_SHAPE[0] - 1) / 2.0
    geometric_col = (SCRIPT.NATIVE_CCD_SHAPE[1] - 1) / 2.0
    for row in measurement["frames"]:
        assert row["has_signal"] is False
        assert row["centroid_is_fallback"] is True
        assert row["centroid_px"] == pytest.approx([geometric_row, geometric_col])
        assert row["offset_from_reference"] is None
        assert row["peak_snr"] == 0.0
        assert row["signal_pixels"] == 0
        assert row["signal_weight_counts"] == 0.0
        assert row["raw_file"].startswith("raw/")
    events = [
        json.loads(line)["event"]
        for line in (out / "capture_journal.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert "no_signal" in events
    assert "verdict" not in events
    released = json.loads((out / "released.json").read_text(encoding="utf-8"))
    assert released["status"] == "no_signal"
    assert released["closed_normally"] is True
    assert released["error"] is None


def test_execute_aborts_on_saturated_preflight_without_science_frames(
    monkeypatch, tmp_path
) -> None:
    rc, out, slm, cam = _run_execute(monkeypatch, tmp_path, fill=4000)
    assert rc == 1
    assert cam.reads == SCRIPT.PREFLIGHT_DISCARD_FRAMES + 1
    assert slm.closed and cam.closed
    assert _FakeLock.instances[-1].released is True
    assert not list((out / "raw").iterdir())
    measurement = json.loads(
        (out / "alignment_measurement.json").read_text(encoding="utf-8")
    )
    assert measurement["status"] == "error"
    assert measurement["n_frames_used"] == 0
    assert measurement["mean_centroid_px"] is None
    assert measurement["within_tolerance"] is None
    assert measurement["preflight"]["saturated"] is True
    assert "preflight" in measurement["error"]
    released = json.loads((out / "released.json").read_text(encoding="utf-8"))
    assert released["status"] == "error"
    assert released["closed_normally"] is False
    assert released["error"] == measurement["error"]


def test_execute_refuses_an_existing_out_without_opening_devices(tmp_path) -> None:
    out = tmp_path / "exists"
    out.mkdir()
    marker = out / "keep.txt"
    marker.write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit):
        SCRIPT.main(["--execute", "--out", str(out)])
    assert marker.read_text(encoding="utf-8") == "x"
    assert sorted(p.name for p in out.iterdir()) == ["keep.txt"]
