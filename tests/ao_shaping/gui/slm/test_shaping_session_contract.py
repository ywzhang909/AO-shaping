"""Offline tests for the shaping page's contract layer.

Three things that must not regress silently:

- :func:`with_slm_ownership` — the three-step SLM handover shared by the run
  button and the sidebar calibration button (stop toggle → release → reacquire
  in ``finally``, with the ``bench_busy`` flag around the whole body);
- the page **order** — 状态条 / 校验区 before the run button so the button can be
  ``disabled``;
- the effective optical-path snapshot — written to the session at run start and
  to ``{save_dir}/params.json`` so a result can be traced back to the optical
  path that produced it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import streamlit as st

from ao_shaping.gui.slm import bench_settings as bs
from ao_shaping.gui.slm import shaping_controls as sc
from ao_shaping.gui.slm.shaping_controls import GsRefineMethod, SlmOwnership
from ao_shaping.gui.utils.phase_overlay import OverlaidSantec


class _FakeSLM:
    def __init__(self) -> None:
        self.is_open = True
        self.close_calls = 0
        self.displayed: list[Any] = []
        self.memory_mode: int | None = None
        self.video_mode = 0
        self.shift_x = 0
        self.shift_y = 0
        self.wavelength = 1064
        self.Panel_Res = (1920, 1200)
        self.Pixel_Size_um = 7.8
        self.Gray_Scale_bits = 10
        self._use_120hz = False
        self._serial_number = "FAKE"
        self.slm_number = 1

    def close(self) -> None:
        self.close_calls += 1
        self.is_open = False

    def open(self) -> None:
        self.is_open = True

    def load_config(self) -> dict[str, Any]:
        return {}

    def create_phase_from_array(self, phase: Any) -> Any:
        import numpy as np

        return np.asarray(phase * 10, dtype="uint16")

    def display_data(self, phase_gray: Any, *, memory_mode: int) -> None:
        self.displayed.append(phase_gray)
        self.memory_mode = memory_mode

    def get_displayed_phase(self) -> tuple[None, str]:
        return None, "fake"


@pytest.fixture(autouse=True)
def state(monkeypatch):
    state: dict[str, Any] = {}
    monkeypatch.setattr(st, "session_state", state)
    monkeypatch.setattr(bs, "probe_camera_window", lambda *a, **k: None)
    monkeypatch.setattr(sc, "Santec", lambda slm_number: _FakeSLM())
    monkeypatch.setattr(
        "ao_shaping.gui.slm.multi_slm_controller._stop_toggle", lambda n: None
    )
    return state


def _recipe_values(slm_number: int = 1) -> dict[str, Any]:
    values = dict(sc.GS_REFINE_RECIPE)
    values["slm_number"] = slm_number
    return values


def _connect(state: dict[str, Any], slm_num: int = 1) -> _FakeSLM:
    prefix = f"slm{slm_num}"
    slm = _FakeSLM()
    slm.slm_number = slm_num
    state[prefix] = slm
    state[f"{prefix}_connected"] = True
    return slm


# ---------------------------------------------------------------------------
# with_slm_ownership
# ---------------------------------------------------------------------------


class TestWithSlmOwnership:
    def test_handover_releases_then_reacquires(self, state) -> None:
        fake = _connect(state)
        seen: dict[str, Any] = {}

        with sc.with_slm_ownership(1) as owned:
            seen["during"] = state["slm1"]
            seen["busy"] = bs.bench_busy()

        assert seen["during"] is None  # the GUI owns nothing inside the body
        assert seen["busy"] is True
        assert fake.close_calls == 1
        assert isinstance(owned, SlmOwnership)
        assert isinstance(owned.slm, OverlaidSantec)
        assert state["slm1"] is owned.slm
        assert state["slm1_connected"] is True
        assert bs.bench_busy() is False

    def test_reacquires_even_when_the_body_raises(self, state) -> None:
        fake = _connect(state)

        with pytest.raises(RuntimeError, match="boom"):
            with sc.with_slm_ownership(1):
                raise RuntimeError("boom")

        assert fake.close_calls == 1
        assert isinstance(state["slm1"], OverlaidSantec)
        assert bs.bench_busy() is False

    def test_busy_flag_is_cleared_even_when_reacquire_raises(self, state, monkeypatch) -> None:
        """A stuck ``bench_busy`` would disable the sidebar optical-path controls
        for the rest of the session, with no control that could clear it."""
        _connect(state)

        def boom(_slm_num):
            raise RuntimeError("SLM controller is wedged")

        monkeypatch.setattr(sc, "_reacquire_slm_handle", boom)

        with pytest.raises(RuntimeError, match="wedged"):
            with sc.with_slm_ownership(1):
                pass

        assert bs.bench_busy() is False

    def test_caller_supplied_slot_is_filled_in(self, state) -> None:
        """``shaping_session`` must be able to read the handle after the ``with``
        even though the ``with`` statement is inside a ``try``."""
        _connect(state)
        slot = SlmOwnership(1)
        with sc.with_slm_ownership(1, slot):
            pass
        assert isinstance(slot.slm, OverlaidSantec)

    def test_only_the_target_slm_is_released(self, state) -> None:
        first = _connect(state, 1)
        second = _connect(state, 2)

        with sc.with_slm_ownership(1):
            assert state["slm1"] is None
            assert state["slm2"] is second  # the other SLM stays connected

        assert first.close_calls == 1
        assert second.close_calls == 0

    def test_run_button_reuses_the_same_handover(self, state, monkeypatch) -> None:
        """``@shaping_session`` is now a thin wrapper over the CM — same behaviour."""
        from ao_shaping.optimizer.wfless import slm_gs_refine

        fake = _connect(state)
        seen: dict[str, Any] = {}

        def _boom(config: Any) -> None:
            seen["released"] = state["slm1"] is None
            seen["busy"] = bs.bench_busy()
            raise RuntimeError("simulated hardware failure")

        monkeypatch.setattr(slm_gs_refine, "optimize_slm_gs_refine", _boom)

        outcome = GsRefineMethod("slm1", 1).run(_recipe_values(), debug=False)

        assert outcome is None
        assert seen == {"released": True, "busy": True}
        assert fake.close_calls == 1
        assert isinstance(state["slm1"], OverlaidSantec)
        assert bs.bench_busy() is False


# ---------------------------------------------------------------------------
# validation that must land *before* the run button
# ---------------------------------------------------------------------------


class TestValidationGate:
    def test_target_side_larger_than_window_is_blocked(self) -> None:
        base = _recipe_values()
        base.update({"target_side": 400, "cam_size": 250})
        assert sc.validate_target_fits_window(base) is not None

        base.update({"target_side": 100})
        assert sc.validate_target_fits_window(base) is None

    def test_crop_off_means_the_target_always_fits(self) -> None:
        base = _recipe_values()
        base.update({"target_side": 400, "cam_size": 0})
        assert sc.validate_target_fits_window(base) is None

    def test_target_side_zero_is_auto_derived(self) -> None:
        base = _recipe_values()
        base.update({"target_side": 0, "cam_size": 8})
        assert sc.validate_target_fits_window(base) is None

    def test_gs_validate_includes_the_window_check(self) -> None:
        values = _recipe_values()
        values.update({"cam_type": "sim", "exposure_time_ms": 0.0})
        values.update({"target_side": 400, "cam_size": 250})
        assert GsRefineMethod("slm1", 1).validate(values) is not None

    def test_mil_reuse_geometry_region_mismatch_is_blocked(
        self, state, tmp_path, monkeypatch
    ) -> None:
        import json

        geometry = tmp_path / "bench_geometry.json"
        geometry.write_text(json.dumps({"method": "sweep", "region": 32}), encoding="utf-8")
        values = dict(sc.MIL_WIDGET_DEFAULTS)
        values.update(
            {
                "cam_type": "sim",
                "cam_id": 0,
                "exposure_time_ms": 0.0,
                "max_exposure_ms": 0.0,
                "cam_size": 250,
                "w_uniformity": 0.4,
                "w_efficiency": 0.6,
                "min_holdout_correlation": 0.5,
                "region": 64,
                "geometry_file": str(geometry),
            }
        )
        method = sc.ModelInLoopMethod("slm1", 1)
        message = method.validate(values)
        assert message is not None and "region" in message

        values["region"] = 32
        assert method._validate_geometry_region(values) is None

    def test_mil_zero_holdout_is_rejected_with_a_pointer(
        self, state, tmp_path, monkeypatch
    ) -> None:
        """``holdout_probe_count`` bound is 0, but Step A's fit needs ≥ 1."""
        monkeypatch.setitem(__import__("sys").modules, "torch", object())
        method = sc.ModelInLoopMethod("slm1", 1)
        message = method._validate_holdout({"holdout_probe_count": 0})
        assert message is not None and "自由相位留出探针" in message
        assert method._validate_holdout({"holdout_probe_count": 2}) is None


# ---------------------------------------------------------------------------
# effective optical-path snapshot
# ---------------------------------------------------------------------------


class TestEffectiveSnapshot:
    def test_snapshot_marks_each_field_state(self, state) -> None:
        bs.record_detection("slm1", {"pupil_center": (960, 690), "geometry_correlation": 0.87})
        bs.record_detection("", {"exposure_ms": 1.5})
        bs.record_detection("slm1", {"panel_span_px": 900.0})
        state["slm1_bench_pupil_center"] = "948,690"  # hand-edited
        state["slm1_bench_panel_span_px"] = 900.0
        bs.sync_manual_states("slm1")
        bs.sync_manual_states("")

        values: dict[str, Any] = {
            "bench_prefix": "slm1",
            "method": "model-in-loop",
            "cam_type": "daheng",
            "cam_id": 0,
            "cam_crop_enabled": True,
            "cam_size": 250,
            "camera_pixel_um": 2.2,
            "exposure_time_ms": 1.5,
            "pupil_center": "948,690",
            "panel_span_px": 900.0,
            "slm_number": 1,
        }
        snap = sc.effective_bench_snapshot(values)

        assert snap["pupil_center__state"] == "manual"
        assert snap["panel_span_px__state"] == "detected"
        assert snap["exposure_ms__state"] == "detected"
        assert snap["cam_type__state"] == "unset"
        assert snap["geometry_correlation"] == pytest.approx(0.87)
        assert snap["method"] == "model-in-loop"

    def test_summary_line_counts_detected_and_manual(self, state) -> None:
        assert "无快照" in sc.snapshot_summary_line("slm1")

        bs.record_detection("slm1", {"pupil_center": (960, 690)})
        bs.record_detection("slm1", {"panel_span_px": 900.0})
        bs.record_detection("", {"exposure_ms": 1.5})
        state["slm1_bench_pupil_center"] = "948,690"
        state["slm1_bench_panel_span_px"] = 900.0
        bs.sync_manual_states("slm1")
        bs.sync_manual_states("")
        bs.set_effective_state(
            "slm1",
            sc.effective_bench_snapshot(
                {
                    "bench_prefix": "slm1",
                    "exposure_time_ms": 1.5,
                    "pupil_center": "948,690",
                    "panel_span_px": 900.0,
                }
            ),
        )

        line = sc.snapshot_summary_line("slm1")
        assert "项探测" in line
        assert "项手动" in line
        assert "pupil_center=948,690" in line

    def test_snapshot_is_written_next_to_the_results(self, state, tmp_path) -> None:
        bs.set_effective_state("slm1", {"pupil_center": "960,690", "method": "gs-refine"})

        written = sc.write_bench_snapshot(tmp_path, "slm1")

        assert written == tmp_path / "params.json"
        assert '"pupil_center": "960,690"' in written.read_text(encoding="utf-8")

    def test_missing_run_dir_is_not_created(self, state, tmp_path) -> None:
        """``save()`` returning a path that does not exist means the saver was a
        stub; creating the directory there would litter the repo."""
        bs.set_effective_state("slm1", {"pupil_center": "960,690"})
        absent = tmp_path / "nope"

        assert sc.write_bench_snapshot(absent, "slm1") is None
        assert not absent.exists()

    def test_existing_params_json_is_left_alone(self, state, tmp_path) -> None:
        target = tmp_path / "params.json"
        target.write_text("{}", encoding="utf-8")
        bs.set_effective_state("slm1", {"pupil_center": "960,690"})

        assert sc.write_bench_snapshot(tmp_path, "slm1") == target
        assert target.read_text(encoding="utf-8") == "{}"

    def test_run_records_the_snapshot_in_the_session(
        self, state, tmp_path, monkeypatch
    ) -> None:
        from ao_shaping.optimizer.wfless import slm_gs_refine

        _connect(state)
        monkeypatch.setattr(
            slm_gs_refine,
            "optimize_slm_gs_refine",
            lambda cfg: type(
                "_Rec",
                (),
                {
                    "history": [{"stage": "flat", "score": 0.5, "pib": 0.8, "cv": 4.0}],
                    "best_phase": None,
                },
            )(),
        )
        monkeypatch.setattr(
            sc,
            "save_gs_refine_results",
            lambda *a, **k: type("R", (), {"save_dir": tmp_path}),
        )

        assert GsRefineMethod("slm1", 1).run(_recipe_values(), debug=False) is not None
        assert bs.effective_state("slm1")["method"] == "gs-refine"
        assert (tmp_path / "params.json").is_file()


# ---------------------------------------------------------------------------
# relaxed upper bounds (the "换光路就不能用" complaint)
# ---------------------------------------------------------------------------


class TestRelaxedBounds:
    @staticmethod
    def _capture(monkeypatch, render) -> dict[str, dict[str, Any]]:
        """Capture every ``st.number_input`` call, keyed by its label.

        The label arrives **positionally** (``st.number_input(label, ...)`` in the
        shared ``num_input`` helper), so the first positional arg has to be
        folded into the kwargs mapping.
        """
        sink: list[dict[str, Any]] = []
        original = st.number_input

        def _wrapped(*args, **kwargs):
            spec = dict(kwargs)
            if args and "label" not in spec:
                spec["label"] = args[0]
            sink.append(spec)
            return original(*args, **kwargs)

        monkeypatch.setattr(st, "number_input", _wrapped)
        render()
        by_label = {}
        for spec in sink:
            by_label[str(spec.get("label", spec.get("key")))] = spec
        return by_label

    def test_mil_advanced_bounds(self, monkeypatch) -> None:
        widgets = self._capture(
            monkeypatch,
            lambda: sc.ModelInLoopMethod("slm1", 1).render_advanced({}),
        )

        assert widgets["探针数"]["max_value"] == 64
        assert widgets["模型网格"]["max_value"] == 512
        assert widgets["模型网格"]["step"] == 8
        assert widgets["远场补零"]["max_value"] == 64
        assert widgets["探针 σ (rad)"]["max_value"] == pytest.approx(12.0)
        assert widgets["探针 σ (rad)"]["min_value"] == pytest.approx(3.0)  # hard floor
        assert widgets["留出探针"]["min_value"] == 0
        # The padding default stays MIL's 8; GS's 3 lives in its own hook.
        assert widgets["远场补零"]["value"] == 8

    def test_gs_advanced_bounds(self, monkeypatch) -> None:
        widgets = self._capture(
            monkeypatch, lambda: sc.GsRefineMethod("slm1", 1).render_advanced({})
        )

        assert widgets["相位网格"]["max_value"] == 128
        assert widgets["远场补零"]["max_value"] == 64
        # The GS padding default stays 3 — deliberately NOT merged with MIL's 8.
        assert widgets["远场补零"]["value"] == 3

    def test_panel_span_upper_bound_is_the_panel(self, state) -> None:
        """模型覆盖宽度 used to be capped at a hard 1920; now it is the panel."""
        from ao_shaping.drivers.slm.santec.slm200_constants import PANEL_RES

        assert bs._SLM_FIELD_NAMES == ("pupil_center", "panel_span_px", "geometry_file")
        # The clamp helper is what the widget value goes through.
        assert bs.bounds_clamp_panel_span("slm1", 99999) == PANEL_RES[0]
        assert bs.bounds_clamp_panel_span("slm1", 0) == 32  # lower bound is 32
        # The grid is 32-based (the widget's min), so 905 snaps to 902.
        assert bs.bounds_clamp_panel_span("slm1", 905) == 902
        assert bs.bounds_clamp_panel_span("slm1", 100000) == PANEL_RES[0]

    def test_target_side_upper_bound_is_the_window(self, state, monkeypatch) -> None:
        """Was 3–1000 hard-coded; now it follows the read-out window."""
        from ao_shaping.drivers.ccd.common import WindowSize

        monkeypatch.setattr(
            bs,
            "probe_camera_window",
            lambda *a, **k: WindowSize(2592, 1944, 2592, 1944, 4),
        )
        state["bench_cam_crop"] = True
        state["bench_cam_size"] = 700

        widgets = self._capture(
            monkeypatch,
            lambda: sc.ModelInLoopMethod("slm1", 1).render_objective({}),
        )
        assert widgets["目标方边长"]["max_value"] == 700
        assert widgets["目标方边长"]["min_value"] == 8  # hard floor

        # Cropping off → the full sensor short side becomes the ceiling.
        state["bench_cam_crop"] = False
        widgets = self._capture(
            monkeypatch,
            lambda: sc.ModelInLoopMethod("slm1", 1).render_objective({}),
        )
        assert widgets["目标方边长"]["max_value"] == 1944

    def test_target_side_value_is_clamped_when_the_window_is_smaller(
        self, state, monkeypatch
    ) -> None:
        """窗口比目标默认边长**小**时，``value`` 必须夹到 ``max_value`` 以内。

        Streamlit 对 ``value > max_value`` 抛 ``StreamlitValueAboveMaxError`` 而不是
        夹取，而「窗口边长 < 目标默认边长」是合法状态（操作员刚把窗口调小；或窗口
        默认值还没建立）。2026-10-16 启动崩溃就是窗口默认值塌成设备网格步长 4 px，
        而 ``target_side`` 的默认值仍是 100（GS）/ 60（MIL）。
        """
        from ao_shaping.drivers.ccd.common import WindowSize

        monkeypatch.setattr(
            bs,
            "probe_camera_window",
            lambda *a, **k: WindowSize(2592, 1944, 2592, 1944, 4),
        )
        state["bench_cam_crop"] = True
        state["bench_cam_size"] = 4  # 网格步长，不是窗口边长

        for method in (sc.ModelInLoopMethod("slm1", 1), sc.GsRefineMethod("slm1", 1)):
            spec = self._capture(monkeypatch, lambda: method.render_objective({}))[
                "目标方边长"
            ]
            assert spec["min_value"] <= spec["value"] <= spec["max_value"]

    def test_w_efficiency_floor_stays_positive(self, monkeypatch) -> None:
        widgets = self._capture(
            monkeypatch,
            lambda: sc.ModelInLoopMethod("slm1", 1).render_objective({}),
        )
        assert widgets["权重 w_eff"]["min_value"] == pytest.approx(0.01)
        assert "掏空" in widgets["权重 w_eff"]["help"]

    def test_budget_preset_matches_the_doc_defaults(self) -> None:
        assert sc.MIL_BUDGET_PRESETS["标准 6/80/600"] == (6, 80, 600)
        assert sc.MIL_BUDGET_PRESETS["快速 2/40/300"] == (2, 40, 300)
        assert sc.MIL_BUDGET_PRESETS["长跑 12/160/1200"] == (12, 160, 1200)
        # GS is validated evidence, never a preset.
        assert sc.GS_VALIDATED_BUDGET == (12, 12)


class TestResultPlaceholder:
    def test_placeholder_before_the_first_run(self, state, monkeypatch) -> None:
        _connect(state)
        monkeypatch.setattr(
            GsRefineMethod,
            "render_inputs",
            lambda self: {"bench_prefix": "slm1", "cam_type": "sim", "cam_size": 0},
        )
        captions: list[str] = []
        original = st.caption

        def _captured(text=None, *a, **k):
            captions.append(str(text))
            return original(text, *a, **k)

        monkeypatch.setattr(st, "caption", _captured)
        sc.render_closed_loop_shaping(1)

        assert "尚无运行结果。" in captions


class TestPageOrder:
    def test_status_bar_says_what_will_happen_when_uncalibrated(self, state) -> None:
        line = sc.render_bench_status("slm1", 1)
        assert line is not None
        assert "未标定" in line
        assert "自动标定" in line

    def test_status_bar_reports_manual_overrides(self, state) -> None:
        bs.record_detection(
            "slm1",
            {
                "pupil_center": (960, 690),
                "geometry_correlation": 0.87,
                "calibrated_at": "20261016_120000",
            },
        )
        state["slm1_bench_pupil_center"] = "948,690"
        bs.sync_manual_states("slm1")

        line = sc.render_bench_status("slm1", 1)
        assert line is not None
        assert "已标定" in line
        assert "0.8700" in line
        assert "1 项手动覆盖" in line
        assert "pupil_center" in line

    def test_button_disabled_when_validation_fails(self, state, monkeypatch) -> None:
        _connect(state)
        calls: list[dict[str, Any]] = []
        originals = {name: getattr(st, name) for name in ("button", "checkbox")}

        def _wrap(fn):
            def _inner(*args, **kwargs):
                calls.append(kwargs)
                return fn(*args, **kwargs)

            return _inner

        for name, fn in originals.items():
            monkeypatch.setattr(st, name, _wrap(fn))

        state["slm1_bench_manual"] = {}
        bs.record_detection("slm1", {})
        # A target far bigger than the window: the cross-check must fire.
        monkeypatch.setattr(
            GsRefineMethod,
            "render_inputs",
            lambda self: {
                "bench_prefix": "slm1",
                "cam_type": "sim",
                "cam_size": 64,
                "target_side": 400,
                "exposure_time_ms": 0.0,
            },
        )
        sc.render_closed_loop_shaping(1)

        buttons = [c for c in calls if "key" in c and "run" in str(c.get("key"))]
        assert buttons, "run button was never rendered"
        assert buttons[0]["disabled"] is True