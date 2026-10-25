"""Offline tests for the sidebar 「CCD参数设置」 block (bench_settings).

Everything here runs without Streamlit's server and without hardware:
``st.session_state`` is patched to a plain dict and ``st.rerun`` is a no-op in
bare mode, so the real render path can be driven (same convention as
``test_shaping_controls.py::TestDeviceLifecycle``).  Camera probing is stubbed
so no gxipy import ever happens.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import streamlit as st

from ao_shaping.drivers.ccd.common import WindowSize
from ao_shaping.gui.slm import bench_settings as bs


class _FakeSLM:
    is_open = True

    def close(self) -> None:
        self.is_open = False

    def open(self) -> None:
        self.is_open = True

    def load_config(self) -> dict[str, Any]:
        return {}

    # attributes ``_sync_slm_session_state`` reads after open()
    video_mode = 0
    shift_x = 0
    shift_y = 0
    wavelength = 1064
    Panel_Res = (1920, 1200)
    Pixel_Size_um = 7.8
    Gray_Scale_bits = 10
    _use_120hz = False
    _serial_number = "FAKE"
    slm_number = 1


@pytest.fixture(autouse=True)
def state(monkeypatch):
    """Patch ``st.session_state`` with a plain dict and stub the camera probe."""
    state: dict[str, Any] = {}
    monkeypatch.setattr(st, "session_state", state)
    monkeypatch.setattr(bs, "probe_camera_window", lambda *a, **k: None)
    return state


@pytest.fixture
def fake_santec(monkeypatch):
    """Replace the Santec class so the handle re-acquire needs no hardware."""
    from ao_shaping.gui.slm import shaping_controls

    monkeypatch.setattr(shaping_controls, "Santec", lambda slm_number: _FakeSLM())
    return _FakeSLM


def _connect_slm(state: dict[str, Any], slm_num: int = 1) -> None:
    prefix = f"slm{slm_num}"
    state[prefix] = _FakeSLM()
    state[f"{prefix}_connected"] = True


# ---------------------------------------------------------------------------
# session key contract
# ---------------------------------------------------------------------------


class TestSessionKeyContract:
    def test_documented_keys(self) -> None:
        """The four tables named in the design contract, plus the busy flag."""
        assert bs.detected_key("slm1") == "slm1_bench_detected"
        assert bs.manual_key("slm1") == "slm1_bench_manual"
        assert bs.pending_key("slm1") == "slm1_bench_pending"
        assert bs.effective_key("slm1") == "slm1_bench_effective"
        assert bs.BUSY_KEY == "bench_busy"

    def test_experiment_level_keys_are_unprefixed(self, state) -> None:
        assert bs.detected_key("") == "bench_detected"
        assert bs.manual_key("") == "bench_manual"

    def test_per_slm_namespaces_are_separate(self, state) -> None:
        """Dual-SLM: SLM 2's entrance plane is SLM 1's exit plane, so the
        pupil centre and geometry MUST NOT share one value."""
        bs.record_detection("slm1", {"pupil_center": (960, 690)})
        bs.record_detection("slm2", {"pupil_center": (900, 610)})
        assert bs.bench_pupil_center("slm1") == "960,690"
        assert bs.bench_pupil_center("slm2") == "900,610"


# ---------------------------------------------------------------------------
# pending-flush: the Streamlit widget-lock workaround
# ---------------------------------------------------------------------------


class TestPendingFlush:
    def test_queue_then_flush_writes_widget_key(self, state) -> None:
        bs.queue_pending("slm1", {"slm1_bench_pupil_center": "948,690"})
        # Nothing is written yet: this is what the calibration button's handler
        # does, while the sidebar widget already holds this key.
        assert "slm1_bench_pupil_center" not in state
        assert bs.pending_state("slm1") == {"slm1_bench_pupil_center": "948,690"}

        written = bs.flush_pending("slm1")

        assert written == {"slm1_bench_pupil_center": "948,690"}
        assert state["slm1_bench_pupil_center"] == "948,690"
        assert bs.pending_state("slm1") == {}

    def test_flush_is_idempotent(self, state) -> None:
        bs.queue_pending("", {"camera_pixel_um": 2.4})
        assert bs.flush_pending("") == {"camera_pixel_um": 2.4}
        assert bs.flush_pending("") == {}

    def test_flush_does_not_swallow_errors(self, state) -> None:
        """A locked widget key must raise, not fail silently.

        ``_safe_set_state`` is deliberately NOT used for calibration back-fill:
        swallowing the ``StreamlitAPIException`` would make the back-fill look
        like it worked while the widget kept the old value.
        """

        class _Locked(dict):
            def __setitem__(self, key, value):
                if key == "camera_pixel_um":
                    raise RuntimeError("simulated StreamlitAPIException")
                return super().__setitem__(key, value)

        st.session_state = _Locked(  # type: ignore[assignment]
            {"camera_pixel_um": 2.2}
        )
        bs.queue_pending("", {"camera_pixel_um": 2.4})
        with pytest.raises(RuntimeError, match="StreamlitAPIException"):
            bs.flush_pending("")


# ---------------------------------------------------------------------------
# three states
# ---------------------------------------------------------------------------


class TestThreeStates:
    def test_unset_before_any_calibration(self, state) -> None:
        assert bs.field_state("slm1", "pupil_center") == "unset"
        assert bs.field_state("", "camera_pixel_um") == "unset"
        assert bs.manual_fields("slm1") == ()
        assert "○" in bs.field_hint("slm1", "pupil_center")

    def test_detected_after_calibration(self, state) -> None:
        bs.record_detection("slm1", {"pupil_center": (960, 690)})
        state["slm1_bench_pupil_center"] = "960,690"
        bs.sync_manual_states("slm1")

        assert bs.field_state("slm1", "pupil_center") == "detected"
        assert bs.manual_fields("slm1") == ()
        hint = bs.field_hint("slm1", "pupil_center")
        assert hint.startswith(bs.MARK_DETECTED)
        assert "960" in hint and "690" in hint

    def test_manual_when_value_diverges(self, state) -> None:
        bs.record_detection("slm1", {"pupil_center": (960, 690)})
        state["slm1_bench_pupil_center"] = "948,690"  # operator typed it
        bs.sync_manual_states("slm1")

        assert bs.field_state("slm1", "pupil_center") == "manual"
        assert bs.manual_fields("slm1") == ("pupil_center",)
        hint = bs.field_hint("slm1", "pupil_center")
        assert hint.startswith(bs.MARK_MANUAL)
        # current value, detected value AND the delta are all visible
        assert "948, 690" in hint
        assert "960, 690" in hint
        assert "12" in hint

    def test_restore_detected_value_clears_manual(self, state) -> None:
        bs.record_detection("slm1", {"pupil_center": (960, 690)})
        state["slm1_bench_pupil_center"] = "948,690"
        bs.sync_manual_states("slm1")
        assert bs.manual_fields("slm1") == ("pupil_center",)

        bs.queue_pending("slm1", {"slm1_bench_pupil_center": "960,690"})
        bs.reset_manual("slm1")
        bs.flush_pending("slm1")
        bs.sync_manual_states("slm1")

        assert bs.manual_fields("slm1") == ()
        assert bs.field_state("slm1", "pupil_center") == "detected"

    def test_recalibration_never_overwrites_manual(
        self, state, monkeypatch, fake_santec
    ) -> None:
        """重新标定不覆盖手动值 —— structural, not a caller responsibility."""
        _connect_slm(state)
        bs.record_detection("slm1", {"pupil_center": (960, 690)})
        state["slm1_bench_pupil_center"] = "948,690"
        bs.sync_manual_states("slm1")

        seen = _patch_calibrate_session(
            monkeypatch, bs, _fake_calibration(center=(1000, 700), geometry=None)
        )

        assert bs.run_session_calibration("slm1", 1) is None

        # The GUI handed its handle over and flagged itself busy while measuring.
        assert seen["handle_released"] is True
        assert seen["busy_inside"] is True
        # The probe moved; the manual value and the flag both survive. The detected
        # centre is stored as a list so it is JSON-serialisable for params.json.
        assert bs.detected_state("slm1")["pupil_center"] == [1000, 700]
        assert bs.pending_state("slm1") == {}  # nothing queued for a manual field
        assert state["slm1_bench_pupil_center"] == "948,690"

    def test_calibration_backfills_non_manual_fields(
        self, state, monkeypatch, fake_santec
    ) -> None:
        _connect_slm(state)
        bs.queue_pending("slm1", {"slm1_bench_pupil_center": "940,690"})
        bs.flush_pending("slm1")
        bs.sync_manual_states("slm1")
        assert bs.manual_fields("slm1") == ()

        _patch_calibrate_session(
            monkeypatch, bs, _fake_calibration(center=(960, 690), geometry=None)
        )

        assert bs.run_session_calibration("slm1", 1) is None

        # queued for the NEXT render, then flushed into the widget
        assert bs.pending_state("slm1") == {"slm1_bench_pupil_center": "960,690"}
        bs.flush_pending("slm1")
        assert state["slm1_bench_pupil_center"] == "960,690"

    def test_calibration_failure_is_surfaced_not_swallowed(
        self, state, monkeypatch, fake_santec
    ) -> None:
        """``calibrate_session`` raises plain exceptions; the GUI shows them."""
        _connect_slm(state)
        from ao_shaping.runners.slm import model_in_loop_session as session

        def _boom(*a, **k):
            raise RuntimeError("panel scan response +0.004 is too weak")

        monkeypatch.setattr(session, "calibrate_session", _boom)

        message = bs.run_session_calibration("slm1", 1)

        assert message is not None and "too weak" in message
        # Nothing recorded, and the handle is back.
        assert bs.detected_state("slm1") == {}
        assert state["slm1"] is not None
        assert state["slm1_connected"] is True
        assert bs.bench_busy() is False

    def test_reuse_source_skips_the_sweep(
        self, state, monkeypatch, fake_santec, tmp_path
    ) -> None:
        _connect_slm(state)
        geometry = tmp_path / "bench_geometry.json"
        geometry.write_text(
            json.dumps({"method": "sweep", "region": 64}), encoding="utf-8"
        )
        state["slm1_bench_geometry_source"] = "reuse"
        state["slm1_bench_geometry_file"] = str(geometry)
        # The real ``calibrate_session`` returns ``None`` for a skipped step; the
        # stub must be faithful or the "skipped != detected" rule is untested.
        seen = _patch_calibrate_session(
            monkeypatch, bs, _fake_calibration(center=None, geometry=None)
        )

        assert bs.run_session_calibration("slm1", 1) is None
        assert seen["locate"] is False
        assert seen["geometry"] is False
        # A skipped step must NOT be recorded as a detection: labelling the
        # caller's own guess "● detected" would be a lie the three-state exists
        # to prevent.
        assert "pupil_center" not in bs.detected_state("slm1")
        assert "geometry_file" not in bs.detected_state("slm1")


def _fake_calibration(center, geometry):
    from ao_shaping.runners.slm.model_in_loop_session import SessionCalibration

    return SessionCalibration(
        pupil_center=center,
        geometry_file=geometry,
        geometry_correlation=0.87 if geometry else None,
        evidence_dir="data/debug/slm_bench_calibration/x",
        notes=("fake",),
    )


def _patch_calibrate_session(monkeypatch, module, calibration) -> dict[str, Any]:
    """Swap the shared ``calibrate_session`` entry point for a stub.

    It is patched **on the module that defines it**, not on the GUI: the point
    of the test is that ``bench_settings`` calls the canonical entry point, so
    the stub has to sit where the real one lives.
    """
    from ao_shaping.runners.slm import model_in_loop_session as session

    seen: dict[str, Any] = {"busy_inside": False, "handle_released": False}

    def _stub(params, config, save_dir, *, locate=True, geometry=True):
        seen["locate"] = locate
        seen["geometry"] = geometry
        seen["busy_inside"] = module.bench_busy()
        seen["handle_released"] = module.st.session_state.get("slm1") is None
        return calibration

    monkeypatch.setattr(session, "calibrate_session", _stub)
    return seen


# ---------------------------------------------------------------------------
# diff thresholds
# ---------------------------------------------------------------------------


class TestDiffThresholds:
    def test_center_tolerance_is_one_pixel(self) -> None:
        field = bs.bench_fields("slm1")["pupil_center"]
        assert bs.field_diff(field, "960,690", (960, 690)) == 0.0
        assert bs.field_diff(field, "961,690", (960, 690)) == pytest.approx(1.0)
        # The threshold is "≥ 1 px", so exactly 1 px is NOT reported as manual.
        assert not _is_manual(field, "961,690", (960, 690))
        assert _is_manual(field, "962,690", (960, 690))

    def test_camera_pixel_relative_tolerance_is_half_a_percent(self) -> None:
        field = bs.bench_fields("")["camera_pixel_um"]
        assert not _is_manual(field, 2.21, 2.2)  # 0.45 %
        assert _is_manual(field, 2.22, 2.2)  # 0.91 %

    def test_exposure_relative_tolerance_is_one_percent(self) -> None:
        field = bs.bench_fields("")["exposure_ms"]
        assert not _is_manual(field, 0.0504, 0.05)  # 0.8 %
        assert _is_manual(field, 0.051, 0.05)  # 2 %

    def test_correlation_threshold_constant(self) -> None:
        assert bs.CORRELATION_ABS_TOLERANCE == pytest.approx(0.02)

    def test_text_field_differs_on_any_change(self) -> None:
        field = bs.bench_fields("slm1")["geometry_file"]
        assert bs.field_diff(field, "a.json", "a.json") is None
        assert bs.field_diff(field, "b.json", "a.json") == pytest.approx(1.0)

    def test_panel_span_tolerance_is_one_step(self) -> None:
        field = bs.bench_fields("slm1")["panel_span_px"]
        assert not _is_manual(field, 910.0, 900.0)  # exactly one step
        assert _is_manual(field, 920.0, 900.0)


def _is_manual(field, current, detected) -> bool:
    return bs._is_manual(field, current, detected)


# ---------------------------------------------------------------------------
# camera window bounds
# ---------------------------------------------------------------------------


class TestCameraWindowBounds:
    def test_fallback_when_camera_unreachable(self, state) -> None:
        bounds = bs.window_bounds()
        assert bounds.source == "fallback"
        assert bounds.max_side == bs.FALLBACK_WINDOW_MAX_SIDE
        assert bounds.step == bs.FALLBACK_WINDOW_STEP
        assert "兜底" in bounds.detail

    def test_device_bounds_use_windowsize_inc(self, state, monkeypatch) -> None:
        """``inc`` must come from the device feature grid.

        Daheng gxipy keeps the OLD window when a value is off-grid and only
        prints noise, so a step that is not ``inc`` silently reads the wrong ROI.
        """
        monkeypatch.setattr(
            bs,
            "probe_camera_window",
            lambda *a, **k: WindowSize(
                width=2592, height=1944, max_width=2592, max_height=1944, inc=4
            ),
        )
        bounds = bs.window_bounds(probe=True)

        assert bounds.is_device
        assert bounds.max_side == 1944  # min(max_width, max_height)
        assert bounds.step == 4
        assert bounds.min_side == 4
        assert "inc" in bounds.detail

    def test_bounds_are_cached_per_camera(self, state, monkeypatch) -> None:
        calls: list[tuple] = []

        def _probe(cam_type, cam_id):
            calls.append((cam_type, cam_id))
            return WindowSize(2592, 1944, 2592, 1944, 4)

        monkeypatch.setattr(bs, "probe_camera_window", _probe)
        state["bench_cam_type"] = "daheng"
        state["bench_cam_id"] = 0
        bs.window_bounds()
        bs.window_bounds()
        assert len(calls) == 1

        state["bench_cam_id"] = 1
        bs.window_bounds()
        assert len(calls) == 2

    def test_target_side_upper_bound_follows_the_window(self, state) -> None:
        state["bench_cam_crop"] = True
        state["bench_cam_size"] = 320
        assert bs.bench_window_side() == 320

        # Cropping off → the full sensor short side is the ceiling.
        state["bench_cam_crop"] = False
        assert bs.bench_window_side() == bs.FALLBACK_WINDOW_MAX_SIDE

    def test_clamp_snaps_to_the_grid(self, state, monkeypatch) -> None:
        monkeypatch.setattr(
            bs,
            "probe_camera_window",
            lambda *a, **k: WindowSize(2592, 1944, 2592, 1944, 4),
        )
        bounds = bs.window_bounds(probe=True)
        assert bounds.clamp(319) % 4 == 0
        assert bounds.clamp(319) == 320
        assert bounds.clamp(100000) == bounds.max_side


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


class TestBenchValidation:
    def test_window_side_must_be_on_the_device_grid(self, state, monkeypatch) -> None:
        monkeypatch.setattr(
            bs,
            "probe_camera_window",
            lambda *a, **k: WindowSize(2592, 1944, 2592, 1944, 4),
        )
        bs.window_bounds(probe=True)
        state["bench_cam_crop"] = True
        state["bench_cam_size"] = 251
        message = bs.validate_bench_inputs()
        assert message is not None
        assert "整数倍" in message

        state["bench_cam_size"] = 252
        assert bs.validate_bench_inputs() is None

    def test_exposure_over_the_session_ceiling_is_rejected(self, state, monkeypatch) -> None:
        monkeypatch.setenv("AO_DAHENG_MAX_EXPOSURE_MS", "0.4")
        state["bench_cam_type"] = "daheng"
        state["bench_exposure_ms"] = 0.5
        message = bs.validate_bench_inputs()
        assert message is not None and "0.4" in message

    def test_negative_exposure_is_rejected(self, state) -> None:
        state["bench_exposure_ms"] = -1.0
        assert "≥ 0" in (bs.validate_bench_inputs() or "")


# ---------------------------------------------------------------------------
# busy flag
# ---------------------------------------------------------------------------


class TestBusyFlag:
    def test_busy_disables_sidebar_controls(self, state, monkeypatch) -> None:
        """While busy, every bench control must come out disabled.

        ``disabled=True`` is what stops the operator from editing the optical
        path while the optimizer / calibrator owns a non-re-entrant handle.
        """
        _connect_slm(state)
        calls: list[dict[str, Any]] = []
        _record_calls(monkeypatch, calls)

        bs.render_bench_settings()
        assert any(spec.get("disabled") is False for spec in calls)

        calls.clear()
        bs.set_bench_busy(True)
        bs.render_bench_settings()

        widgets = [spec for spec in calls if "key" in spec]
        bench_widgets = [w for w in widgets if "bench" in w["key"] or w["key"] == "camera_pixel_um"]
        assert bench_widgets, "no bench widgets captured"
        assert all(w["disabled"] for w in bench_widgets)
        # The calibrate button too.
        assert any(
            w["key"] == "slm1_bench_calibrate_btn" and w["disabled"] for w in widgets
        )

    def test_reuse_mode_does_not_offer_a_measurement(
        self, state, monkeypatch
    ) -> None:
        """Under 复用, ``calibrate_session`` is called with both steps off, so
        the button would only restamp ``calibrated_at`` while implying the bench
        had just been re-measured."""
        _connect_slm(state)
        calls: list[dict[str, Any]] = []
        _record_calls(monkeypatch, calls)
        # A bare-mode selectbox ignores session_state and answers with the first
        # option, so the branch has to be driven through the widget itself.
        selectbox = st.selectbox
        monkeypatch.setattr(
            st,
            "selectbox",
            lambda label, *a, **k: (
                "reuse" if label == "几何来源" else selectbox(label, *a, **k)
            ),
        )

        bs.render_bench_settings()

        button = next(
            w for w in calls if w.get("key") == "slm1_bench_calibrate_btn"
        )
        assert button["disabled"] is True

    def test_busy_is_cleared_after_the_calibration(
        self, state, monkeypatch, fake_santec
    ) -> None:
        _connect_slm(state)
        _patch_calibrate_session(
            monkeypatch, bs, _fake_calibration(center=(960, 690), geometry=None)
        )

        assert bs.run_session_calibration("slm1", 1) is None
        assert bs.bench_busy() is False


def _record_calls(monkeypatch, sink: list[dict[str, Any]]) -> None:
    """Capture the kwargs of every widget call so disabled flags can be asserted."""
    originals = {
        name: getattr(st, name)
        for name in (
            "number_input",
            "text_input",
            "selectbox",
            "checkbox",
            "button",
        )
    }

    def _wrap(name, fn):
        def _inner(*args, **kwargs):
            sink.append({"widget": name, **kwargs})
            return fn(*args, **kwargs)

        return _inner

    for name, fn in originals.items():
        monkeypatch.setattr(st, name, _wrap(name, fn))


# ---------------------------------------------------------------------------
# summary card + geometry region check
# ---------------------------------------------------------------------------


class TestSummaryAndGeometry:
    def test_uncalibrated_summary_points_at_the_calibration_button(self, state) -> None:
        """未标定 ≠ 「运行时会自动标定」。

        运行路径不再测量（2026-10-24 删掉 ``auto_calibrate``），所以这一行必须指向
        侧栏的「标定 … 光路」按钮，而不是承诺一次运行期的标定。
        """
        lines = bs.bench_summary_lines("slm1")
        line = lines[0]
        assert line.startswith(bs.MARK_UNSET)
        assert "标定" in line
        assert "光路" in line
        assert "运行时会先" not in line
        assert "运行时自动" not in line

    def test_calibrated_summary_lists_the_measured_facts(self, state) -> None:
        bs.record_detection(
            "slm1",
            {
                "pupil_center": (960, 690),
                "geometry_file": "data/g.json",
                "geometry_correlation": 0.87,
                "exposure_ms": 1.5,
                "calibrated_at": "20261016_120000",
                "evidence_dir": "data/ev",
            },
        )
        line = bs.bench_summary_lines("slm1")[0]
        assert line.startswith(bs.MARK_DETECTED)
        assert "960, 690" in line
        assert "0.8700" in line
        assert "data/g.json" in line

    def test_geometry_region_mismatch(self, state, tmp_path) -> None:
        good = tmp_path / "g.json"
        good.write_text(json.dumps({"method": "sweep", "region": 64}), encoding="utf-8")
        assert bs.geometry_region_mismatch(str(good), 64) is None
        assert bs.geometry_region_mismatch(str(good), 32) == 64
        assert bs.geometry_region_mismatch(str(tmp_path / "missing"), 64) is None


# ---------------------------------------------------------------------------
# merge into the method values table
# ---------------------------------------------------------------------------


class TestMergeBenchValues:
    def test_merge_supplies_the_camera_and_geometry_fields(self, state) -> None:
        _connect_slm(state)
        state.update(
            {
                "bench_cam_type": "sim",
                "bench_cam_id": 2,
                "bench_cam_crop": True,
                "bench_cam_size": 256,
                "bench_exposure_ms": 1.25,
                "camera_pixel_um": 3.45,
                "bench_blaze_period_px": 96,
                "bench_blaze_direction": "horizontal",
            }
        )
        values: dict[str, Any] = {}
        bs.merge_bench_values(values, "slm1", 1, "model-in-loop")

        assert values["cam_type"] == "sim"
        assert values["cam_id"] == 2
        assert values["cam_size"] == 256
        assert values["exposure_time_ms"] == pytest.approx(1.25)
        assert values["camera_pixel_um"] == pytest.approx(3.45)
        assert values["blaze_period_px"] == 96
        assert values["blaze_direction"] == "horizontal"
        assert values["bench_prefix"] == "slm1"

    def test_crop_off_means_full_frame(self, state) -> None:
        state["bench_cam_crop"] = False
        state["bench_cam_size"] = 256
        values: dict[str, Any] = {}
        bs.merge_bench_values(values, "slm1", 1, "model-in-loop")
        assert values["cam_size"] == 0

    def test_gs_beam_radius_is_half_the_model_span(self, state) -> None:
        state["slm1_bench_panel_span_px"] = 1000.0
        values: dict[str, Any] = {}
        bs.merge_bench_values(values, "slm1", 1, "gs-refine")
        assert values["panel_span_px"] == pytest.approx(1000.0)
        assert values["beam_radius_px"] == pytest.approx(500.0)

    def test_blank_pupil_center_falls_back_to_detected_then_default(self, state) -> None:
        state["slm1_bench_pupil_center"] = ""
        assert bs.bench_pupil_center("slm1") == bs.PUPIL_CENTER_FALLBACK

        bs.record_detection("slm1", {"pupil_center": (1010, 700)})
        assert bs.bench_pupil_center("slm1") == "1010,700"

        state["slm1_bench_pupil_center"] = " 123,456 "
        assert bs.bench_pupil_center("slm1") == "123,456"


# ---------------------------------------------------------------------------
# reusable geometry (the run path no longer measures)
# ---------------------------------------------------------------------------


class TestReusableGeometryFile:
    """``reusable_geometry_file`` —— 运行路径唯一的几何来源。

    三条规则，按优先级：几何来源「复用」时文本框里的路径 → 本会话已标定的探测产物
    → 空串（= 没有可复用的几何，由 ``ModelInLoopMethod.validate`` 拒绝运行）。
    """

    def test_reuse_mode_uses_the_text_box(self, state, tmp_path) -> None:
        box = tmp_path / "box.json"
        box.write_text("{}", encoding="utf-8")
        state["slm1_bench_geometry_source"] = "reuse"
        state["slm1_bench_geometry_file"] = f"  {box}  "
        bs.record_detection("slm1", {"geometry_file": "data/detected.json"})

        assert bs.reusable_geometry_file("slm1") == str(box)

    def test_reuse_mode_with_an_empty_box_uses_the_detected_artefact(self, state) -> None:
        state["slm1_bench_geometry_source"] = "reuse"
        bs.record_detection("slm1", {"geometry_file": "data/detected.json"})

        assert bs.reusable_geometry_file("slm1") == "data/detected.json"

    @pytest.mark.parametrize("source", ["auto", "keep"])
    def test_the_dropdown_does_not_gate_the_run(self, state, source) -> None:
        """``auto``/``keep`` 只影响按钮下一次测什么；运行一律复用已有产物。

        文本框属于「复用」模式（下次点按钮才测量时不该被一个手填路径挡住），所以
        这两种来源下解析到的是本会话的探测产物。
        """
        state["slm1_bench_geometry_source"] = source
        state["slm1_bench_geometry_file"] = "data/box.json"
        bs.record_detection("slm1", {"geometry_file": "data/detected.json"})

        assert bs.reusable_geometry_file("slm1") == "data/detected.json"

    def test_nothing_anywhere_is_the_empty_string(self, state) -> None:
        assert bs.reusable_geometry_file("slm1") == ""

        # A detection that recorded no geometry is still "nothing".
        bs.record_detection("slm1", {"geometry_file": ""})
        assert bs.reusable_geometry_file("slm1") == ""

    def test_namespaces_do_not_leak_between_slms(self, state) -> None:
        """Dual SLM: SLM 2's entrance plane is SLM 1's exit plane."""
        bs.record_detection("slm1", {"geometry_file": "data/slm1.json"})
        bs.record_detection("slm2", {"geometry_file": "data/slm2.json"})

        assert bs.reusable_geometry_file("slm1") == "data/slm1.json"
        assert bs.reusable_geometry_file("slm2") == "data/slm2.json"
class TestCalibrationMarker:
    """折叠标题自报标定状态。

    操作者看的是 D/E 组里的字段；字段为空并不等于「没标定」，所以在**标题**上
    给出状态，折叠着也能判断这台 SLM 现在能不能直接整形。
    """

    def test_uncalibrated_is_the_marker_when_nothing_was_measured(self, state) -> None:
        assert bs.is_calibrated("slm1") is False
        assert bs.calibration_marker("slm1") == f"{bs.MARK_UNSET} 未标定"

    def test_marker_flips_once_a_centre_is_detected(self, state) -> None:
        bs.record_detection("slm1", {"pupil_center": (960, 690)})
        assert bs.calibration_marker("slm1") == f"{bs.MARK_DETECTED} 已标定"

    def test_sweep_geometry_alone_also_counts_as_calibrated(self, state) -> None:
        """sweep 几何是标定的另一半证据：只有它、没有中心也必须算已标定。"""
        bs.record_detection("slm1", {"geometry_file": "data/geo.json"})
        assert bs.is_calibrated("slm1") is True
        assert bs.calibration_marker("slm1") == f"{bs.MARK_DETECTED} 已标定"

    def test_namespaces_do_not_share_the_state(self, state) -> None:
        bs.record_detection("slm2", {"pupil_center": (900, 610)})
        assert bs.calibration_marker("slm1") == f"{bs.MARK_UNSET} 未标定"
        assert bs.calibration_marker("slm2") == f"{bs.MARK_DETECTED} 已标定"

    def test_expander_title_carries_the_marker(self, state, monkeypatch) -> None:
        labels: list[str] = []
        expander = st.expander

        def _record(label, *args, **kwargs):
            labels.append(label)
            return expander(label, *args, **kwargs)

        monkeypatch.setattr(st, "expander", _record)
        _connect_slm(state)

        bs.render_bench_settings()

        titles = [label for label in labels if "光路几何与标定" in label]
        assert titles, f"no per-SLM calibration expander rendered; got {labels}"
        assert all(bs.MARK_UNSET in label for label in titles)

    def test_expander_title_reports_detected_after_a_calibration(
        self, state, monkeypatch
    ) -> None:
        labels: list[str] = []
        expander = st.expander

        def _record(label, *args, **kwargs):
            labels.append(label)
            return expander(label, *args, **kwargs)

        monkeypatch.setattr(st, "expander", _record)
        _connect_slm(state)
        bs.record_detection("slm1", {"pupil_center": (960, 690)})

        bs.render_bench_settings()

        titles = [label for label in labels if "光路几何与标定" in label]
        assert titles
        assert all(bs.MARK_DETECTED in label for label in titles)


class TestSingleCalibrationEntry:
    """标定只有**一个**入口：摘要卡 + D + E 同在一个折叠块里。

    2026-10-24 合并前，侧栏有两块：「SLM N · 光路几何与标定」与
    「SLM N · 一键光路标定」。两个主按钮调的是同一个
    ``run_session_calibration``，后者只是把摘要卡放在自己的折叠块里，并且在
    「复用」来源下**不禁用** —— 点下去等于只刷新 ``calibrated_at``。
    """

    def test_summary_renders_in_the_same_block_as_the_button(
        self, state, monkeypatch
    ) -> None:
        """摘要卡必须落在存活的那一块里（否则合并就把摘要弄丢了）。"""
        prefixes: list[str] = []
        original = bs.bench_summary_lines

        def _record(prefix, *args, **kwargs):
            prefixes.append(prefix)
            return original(prefix, *args, **kwargs)

        monkeypatch.setattr(bs, "bench_summary_lines", _record)
        _connect_slm(state)

        bs.render_bench_settings()

        assert "slm1" in prefixes, f"summary card never rendered; got {prefixes}"

    def test_no_second_calibration_button_remains(self, state, monkeypatch) -> None:
        _connect_slm(state)
        calls: list[dict[str, Any]] = []
        _record_calls(monkeypatch, calls)

        bs.render_bench_settings()

        keys = [spec["key"] for spec in calls if "key" in spec]
        assert "slm1_bench_calibrate_btn" in keys
        assert not [k for k in keys if "auto_calibrate" in k], keys

    def test_one_click_entry_is_gone(self, state) -> None:
        """``render_auto_calibrate_entry`` 是那个重复入口，已整体删除。"""
        from ao_shaping.gui.slm import multi_slm_controller

        assert not hasattr(multi_slm_controller, "render_auto_calibrate_entry")
