"""Offline tests for the prereg §12.11.5 six-frame disambiguation capture."""

from __future__ import annotations

import argparse
import importlib
import json

import numpy as np
import pytest

SCRIPT = importlib.import_module(
    "scripts.slm.capture_s1_mapping_disambiguation_20261124"
)

#: Frozen by the user protocol; none of these may be a CLI knob.
EXPECTED_ORDER = (
    "flat_scale993",
    "grating_x_p128_scale993",
    "grating_x_p128_scale1024",
    "grating_y_p128_scale993",
    "blaze2d_px128_py192_sx+1_sy+1_scale993",
    "ladder_x_scale993",
)
EXPECTED_SCALES = (993, 993, 1024, 993, 993, 993)


def _plan_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "exposure_ms": SCRIPT.DEFAULT_EXPOSURE_MS,
        "wait_until_idle": SCRIPT.DEFAULT_WAIT_IDLE_S,
        "slm_number": SCRIPT.DEFAULT_SLM_NUMBER,
        "cam_type": SCRIPT.DEFAULT_CAM_TYPE,
        "cam_id": SCRIPT.DEFAULT_CAM_ID,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _plan() -> dict[str, object]:
    return SCRIPT.build_plan(_plan_args())


def test_plan_is_exactly_six_frozen_frames_in_order() -> None:
    plan = _plan()
    assert plan["capture_order"] == list(EXPECTED_ORDER)
    assert plan["n_frames_expected"] == 6
    assert plan["frames_per_pattern"] == 1
    assert [item["id"] for item in plan["patterns"]] == list(EXPECTED_ORDER)
    assert len(plan["patterns"]) == 6
    assert [item["frames"] for item in plan["patterns"]] == [1] * 6
    assert [item["index"] for item in plan["patterns"]] == list(range(6))


def test_plan_scales_match_the_user_protocol() -> None:
    plan = _plan()
    assert [item["two_pi_gray_scale"] for item in plan["patterns"]] == list(
        EXPECTED_SCALES
    )
    by_id = {item["id"]: item for item in plan["patterns"]}
    assert by_id["grating_x_p128_scale993"]["a1_pattern_id"] == "grating_x_p128"
    assert by_id["grating_x_p128_scale1024"]["a1_pattern_id"] == "grating_x_p128"
    assert by_id["grating_y_p128_scale993"]["a1_pattern_id"] == "grating_y_p128"
    assert (
        by_id["blaze2d_px128_py192_sx+1_sy+1_scale993"]["a1_pattern_id"]
        == "blaze2d_px128_py192_sx+1_sy+1"
    )
    assert by_id["ladder_x_scale993"]["a1_pattern_id"] == "ladder_x"
    for item in plan["patterns"]:
        assert item["gray_modulus"] == item["two_pi_gray_scale"] + 1


def test_scale_pair_shares_phase_and_differs_only_in_gray() -> None:
    plan = _plan()
    by_id = {item["id"]: item for item in plan["patterns"]}
    x993 = by_id["grating_x_p128_scale993"]
    x1024 = by_id["grating_x_p128_scale1024"]
    assert x993["phase_sha256"] == x1024["phase_sha256"]
    assert x993["gray_sha256"] != x1024["gray_sha256"]
    assert x993["gray_max"] != x1024["gray_max"]
    assert len(x993["phase_sha256"]) == 64
    assert all(len(item["phase_sha256"]) == 64 for item in plan["patterns"])
    assert all(len(item["gray_sha256"]) == 64 for item in plan["patterns"])

    patterns = {item["id"]: item for item in SCRIPT.protocol_patterns()}
    assert patterns["grating_x_p128_scale993"]["gray_unique_count"] == 2
    assert patterns["grating_x_p128_scale1024"]["gray_unique_count"] == 2


def test_no_focus_compensation_and_frozen_acquisition_settings() -> None:
    plan = _plan()
    focus = plan["focus_compensation"]
    assert focus["focus_d_m1"] == 0.0
    assert focus["compensation_applied"] is False
    assert focus["focal_mm"] is None
    assert focus["fresnel_phase_sha256"] == SCRIPT.digest(
        np.zeros(SCRIPT.DEFAULT_PANEL_SHAPE, dtype=np.float64)
    )
    assert plan["exposure_ms_requested"] == 0.087
    assert SCRIPT.DEFAULT_EXPOSURE_MS == 0.087
    assert plan["bit_depth_required"] == 12
    assert plan["full_scale_required"] == 4095
    assert plan["panel_shape_expected_rows_cols"] == [1200, 1920]
    assert plan["settle_s"] == 0.8
    assert plan["wait_until_idle_s"] == 600.0
    assert plan["preflight"]["discard_frames"] == 3
    assert plan["preflight"]["abort_on_saturation"] is True


def test_a1_zero_order_reference_and_prereg_section_are_frozen() -> None:
    plan = _plan()
    ref = plan["a1_zero_order_reference"]
    assert ref["row"] == 1105.249
    assert ref["col"] == 1227.556
    assert plan["prereg_section"] == "§12.11.5"
    assert "92–97 px" in plan["prereg_criteria_verbatim"]
    assert "56–59 px" not in plan["prereg_criteria_verbatim"]
    assert "95 px" in plan["prereg_criteria_verbatim"]
    assert "blaze2d" in plan["prereg_criteria_verbatim"]
    assert SCRIPT.PREREG_SECTION == "§12.11.5"
    assert plan["design_source"].endswith("§12.11.5")


def test_plan_only_creates_no_directory(tmp_path, capsys) -> None:
    out = tmp_path / "run"
    assert SCRIPT.main(["--out", str(out)]) == 0
    assert not out.exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_frames_expected"] == 6
    assert [item["id"] for item in payload["patterns"]] == list(EXPECTED_ORDER)


def test_self_test_passes(capsys) -> None:
    assert SCRIPT.main(["--self-test"]) == 0
    capsys.readouterr()


def test_fixed_protocol_has_no_scale_or_focus_knobs(tmp_path) -> None:
    with pytest.raises(SystemExit):
        SCRIPT.main(["--out", str(tmp_path / "run"), "--focus-d-m1", "0.5"])
    with pytest.raises(SystemExit):
        SCRIPT.main(["--out", str(tmp_path / "run"), "--frames-per-pattern", "2"])
    with pytest.raises(SystemExit):
        SCRIPT.main(["--out", str(tmp_path / "run"), "--two-pi-gray-scale", "1023"])


def test_execute_refuses_an_existing_out_without_opening_devices(tmp_path) -> None:
    out = tmp_path / "exists"
    out.mkdir()
    marker = out / "keep.txt"
    marker.write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit):
        SCRIPT.main(["--out", str(out), "--execute"])
    assert marker.read_text(encoding="utf-8") == "x"
    assert sorted(p.name for p in out.iterdir()) == ["keep.txt"]


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
    def __init__(self, fill: int) -> None:
        self.bit_depth = 12
        self.fill = int(fill)
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
        return np.full(SCRIPT.NATIVE_CCD_SHAPE, self.fill, dtype=np.uint16)

    def effective_full_scale(self, _probe: np.ndarray) -> float:
        return float(SCRIPT.FULL_SCALE_REQUIRED)


def _install_fake_devices(monkeypatch, *, fill: int) -> tuple[_FakeSLM, _FakeCam]:
    import sys
    import types

    slm = _FakeSLM()
    cam = _FakeCam(fill)
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


def _run_execute(monkeypatch, tmp_path, *, fill: int):
    slm, cam = _install_fake_devices(monkeypatch, fill=fill)
    out = tmp_path / "run"
    rc = SCRIPT.main(["--execute", "--out", str(out), "--wait-until-idle", "0"])
    return rc, out, slm, cam


def test_execute_writes_six_frames_and_release_artifacts(monkeypatch, tmp_path) -> None:
    rc, out, slm, cam = _run_execute(monkeypatch, tmp_path, fill=100)
    assert rc == 0
    assert slm.opens == 1 and slm.closed and slm.gray0 == 0
    assert cam.opens == 1 and cam.closed
    assert _FakeLock.instances[-1].released is True
    for name in ("plan.json", "summary.json", "released.json",
                 "capture_journal.jsonl"):
        assert (out / name).is_file(), name
    for item in SCRIPT.protocol_patterns():
        assert (out / "raw" / f"{item['index']:02d}_{item['id']}.npy").is_file()
    assert sorted(p.name for p in (out / "raw").iterdir()) == sorted(
        f"{i['index']:02d}_{i['id']}.npy" for i in SCRIPT.protocol_patterns()
    )

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert summary["n_frames"] == summary["expected_frames"] == 6
    assert summary["saturation_count"] == 0
    assert summary["same_session_flat_centroid_px"] is not None
    rows = summary["frames"]
    assert [row["pattern_id"] for row in rows] == list(EXPECTED_ORDER)
    assert [row["two_pi_gray_scale"] for row in rows] == list(EXPECTED_SCALES)
    for row in rows:
        assert row["focus_d_m1"] == 0.0
        assert row["compensation_applied"] is False
        assert row["bit_depth"] == 12
        assert row["full_scale"] == 4095
        assert row["panel_shape_rows_cols"] == [1200, 1920]
        assert row["exposure_ms_requested"] == 0.087
        assert row["exposure_ms_actual"] == 0.087
        assert len(row["frame_sha256"]) == 64
        assert row["role"]
    x993, x1024 = rows[1], rows[2]
    assert x993["phase_sha256"] == x1024["phase_sha256"]
    assert x993["gray_sha256"] != x1024["gray_sha256"]

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
        for line in (out / "capture_journal.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert events.count("frame_saved") == 6
    assert events[0] == "devices_ready"
    assert "preflight" in events
    assert events[-1] == "capture_finished"


def test_execute_aborts_on_saturated_preflight_without_retry(
    monkeypatch, tmp_path
) -> None:
    rc, out, slm, cam = _run_execute(monkeypatch, tmp_path, fill=4000)
    assert rc == 1
    # Preflight only: 3 discards + 1 read; no science frame is attempted.
    assert cam.reads == SCRIPT.PREFLIGHT_DISCARD_FRAMES + 1
    assert slm.closed and cam.closed
    assert _FakeLock.instances[-1].released is True
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "error"
    assert summary["n_frames"] == 0
    assert summary["preflight"]["saturated"] is True
    assert "preflight" in summary["error"]
    released = json.loads((out / "released.json").read_text(encoding="utf-8"))
    assert released["closed_normally"] is False
    assert released["status"] == "error"
    assert released["error"] == summary["error"]

