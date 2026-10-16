"""Offline tests for the closed-loop shaping section (shaping_controls).

Covers the parts that must not regress without hardware:
- the widget-values → canonical ``SlmGsRefineParams`` / ``SlmModelInLoopParams``
  mapping, and that the GUI defaults ARE the interleaved-replay validated
  recipe from ``report/shape_trials/gs_refine_live_20261012.md``;
- the resolved ``SlmGsRefineConfig`` (derived focal length, guard validation);
- the Daheng exposure guards mirroring the CLI runner;
- the device lifecycle: the GUI handle is released before the optimizer runs
  and re-acquired afterwards even when the optimizer raises;
- the pattern registry no longer carries the removed open-loop shaping types.

No hardware, no Streamlit server: ``st.session_state`` and the no-op bare-mode
element calls work outside a script run (same convention as
``test_pattern_controls.py`` calling ``render()``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import streamlit as st

from ao_shaping.gui.slm.pattern_controls import PATTERN_REGISTRY
from ao_shaping.gui.slm.shaping_controls import (
    GS_REFINE_RECIPE,
    build_gs_refine_params,
    build_model_in_loop_params,
    validate_daheng_exposure,
    validate_daheng_fixed_exposure,
)

REMOVED_PATTERN_TYPES = ("GS方形整形", "GS方形整形(高斯版)", "稳像法整形")


class _FakeSLM:
    """Minimal stand-in for an open Santec handle."""

    def __init__(self) -> None:
        self.is_open = True
        self.close_calls = 0
        self.displayed: list[np.ndarray] = []
        # attributes _sync_slm_session_state reads after open()
        self.video_mode = 0
        self.shift_x = 0
        self.shift_y = 0
        self.wavelength = 1064
        self.Panel_Res = (128, 96)
        self.Pixel_Size_um = 8.0
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

    def display_phase(self, phase: np.ndarray, wait_time_s: float | None = 0.0) -> None:
        self.displayed.append(np.asarray(phase))

    def create_phase_from_array(self, phase: np.ndarray) -> np.ndarray:
        return np.asarray(phase * 10, dtype=np.uint16)

    def display_data(self, phase_gray: np.ndarray, *, memory_mode: int) -> None:
        self.displayed.append(np.asarray(phase_gray))
        self.memory_mode = memory_mode

    def get_displayed_phase(self) -> tuple[None, str]:
        # refresh_phase_preview falls back to "no preview" without crashing.
        return None, "fake"


class _FakeSantecFactory(_FakeSLM):
    """Callable stand-in for the ``Santec`` class in shaping_controls."""

    instances: list[_FakeSantecFactory] = []

    def __new__(cls, *args: Any, **kwargs: Any) -> _FakeSantecFactory:
        del args, kwargs
        obj = super().__new__(cls)
        return obj

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        del args, kwargs
        super().__init__()
        _FakeSantecFactory.instances.append(self)


def _recipe_values(slm_number: int = 1) -> dict[str, Any]:
    values = dict(GS_REFINE_RECIPE)
    values["slm_number"] = slm_number
    return values


class TestRegistryCleanup:
    @pytest.mark.parametrize("pattern_type", REMOVED_PATTERN_TYPES)
    def test_removed_open_loop_types_gone(self, pattern_type: str) -> None:
        assert pattern_type not in PATTERN_REGISTRY

    def test_surviving_types_intact(self) -> None:
        for pattern_type in ("平场", "Zernike", "半半相位", "涡旋相位"):
            assert pattern_type in PATTERN_REGISTRY


class TestParamsMapping:
    def test_gs_refine_defaults_match_validated_recipe(self) -> None:
        params = build_gs_refine_params(_recipe_values(slm_number=2))
        assert params.slm_number == 2
        assert params.pupil_center == "960,690"
        assert params.beam_radius_px == 450.0
        assert params.target_side == 100
        assert params.cam_size == 250
        assert params.gs_iters == 12
        assert params.gs_method == "plain"
        assert params.far_field_padding == 3
        assert params.phase_grid == 12
        assert params.epochs == 12
        assert params.seed == 0
        assert params.confirm_best is True
        assert params.exposure_time_ms == pytest.approx(0.05)
        assert params.camera_pixel_um == pytest.approx(2.2)

    def test_gs_refine_config_resolves_bench_geometry(self) -> None:
        from ao_shaping.runners.runner_common import build_gs_refine_config

        params = build_gs_refine_params(_recipe_values())
        config = build_gs_refine_config(params, debug=False)
        assert config.pupil_center == (960, 690)
        assert config.target_side == 100
        assert config.phase_grid == 12
        # The derived 2f focal length for this bench is ~0.122 m (125 mm lens
        # within 2%); a wrong anchor would move it far outside this band.
        assert 0.10 < config.focal_length_m < 0.15

    def test_model_in_loop_params_mapping(self) -> None:
        values = {
            "n_rounds": 3,
            "seed": 7,
            "target_shape": "square",
            "target_side": 80,
            "w_uniformity": 0.4,
            "w_efficiency": 0.6,
            "probe_count": 8,
            "probe_spread": 4.0,
            "step_a_iterations": 80,
            "step_b_iterations": 300,
            "holdout_probe_count": 2,
            "min_holdout_correlation": 0.5,
            "region": 64,
            "panel_span_px": 900.0,
            "pupil_center": "960,690",
            "far_field_padding": 8,
            "cam_type": "sim",
            "cam_id": 0,
            "exposure_time_ms": 0.0,
            "max_exposure_ms": 0.4,
            "cam_size": 250,
            "slm_number": 1,
        }
        params = build_model_in_loop_params(values)
        assert params.n_rounds == 3
        assert params.target_side == 80
        assert params.step_b_iterations == 300
        assert params.pupil_center == "960,690"
        from ao_shaping.runners.runner_common import build_model_in_loop_config

        config = build_model_in_loop_config(params, debug=False)
        assert config.freeform_holdout_count == 0
        assert config.holdout_probe_count == 2

        from dataclasses import replace

        config = build_model_in_loop_config(
            replace(params, freeform_holdout_count=3), debug=False
        )
        assert config.freeform_holdout_count == 3

    def test_model_in_loop_geometry_file_reaches_the_config(self, tmp_path) -> None:
        """The GUI must be able to hand the optimiser a measured geometry.

        Without one a hardware run uses the abandoned speckle route (measured
        correlation ~0.07 against the 0.5 bake-off floor) and aborts — exactly
        the failure this entry point exists to prevent.
        """
        from ao_shaping.gui.slm.shaping_controls import MIL_WIDGET_DEFAULTS
        from ao_shaping.runners.runner_common import build_model_in_loop_config

        values = dict(MIL_WIDGET_DEFAULTS)
        values.update(
            w_uniformity=0.4,
            w_efficiency=0.6,
            min_holdout_correlation=0.5,
            cam_type="sim",
            cam_id=0,
            exposure_time_ms=0.0,
            max_exposure_ms=0.4,
            cam_size=250,
        )

        geometry = tmp_path / "bench_geometry.json"
        geometry.write_text("{}", encoding="utf-8")
        values["geometry_file"] = str(geometry)

        config = build_model_in_loop_config(
            build_model_in_loop_params(values), debug=False
        )
        assert config.geometry_file == str(geometry)

        # Blank must mean "not specified" (``None``), not "": the config
        # validator rejects a path that does not exist, and an empty box is the
        # documented "no geometry file" state.
        for blank in ("", "   "):
            values["geometry_file"] = blank
            assert build_model_in_loop_params(values).geometry_file is None

    @pytest.mark.parametrize(
        "shape", ["square", "rectangle", "triangle", "star", "circle"]
    )
    def test_model_in_loop_shape_selection_reaches_config(self, shape: str) -> None:
        from ao_shaping.gui.slm.shaping_controls import MIL_TARGET_SHAPE_LABELS
        from ao_shaping.runners.runner_common import build_model_in_loop_config

        assert shape in MIL_TARGET_SHAPE_LABELS
        values = {
            "n_rounds": 1,
            "seed": 0,
            "target_shape": shape,
            "target_side": 60,
            "w_uniformity": 0.4,
            "w_efficiency": 0.6,
            "probe_count": 2,
            "probe_spread": 4.0,
            "step_a_iterations": 0,
            "step_b_iterations": 4,
            "holdout_probe_count": 2,
            "freeform_holdout_count": 1,
            "min_holdout_correlation": 0.5,
            "region": 32,
            "panel_span_px": 900.0,
            "pupil_center": "960,670",
            "far_field_padding": 4,
            "cam_type": "sim",
            "cam_id": 0,
            "exposure_time_ms": 0.0,
            "max_exposure_ms": 0.4,
            "cam_size": 128,
            "slm_number": 1,
        }
        config = build_model_in_loop_config(
            build_model_in_loop_params(values), debug=False
        )
        assert config.target_shape == shape
        assert config.freeform_holdout_count == 1


class TestExposureGuards:
    def test_daheng_over_env_ceiling_rejected(self, monkeypatch) -> None:
        monkeypatch.setenv("AO_DAHENG_MAX_EXPOSURE_MS", "0.4")
        assert validate_daheng_exposure("daheng", 0.5) is not None
        assert validate_daheng_fixed_exposure("daheng", 0.5) is not None

    def test_daheng_exposure_without_env_ceiling_is_not_static_rejected(
        self, monkeypatch
    ) -> None:
        # The ceiling is optical-path dependent; with no env var the UI cannot
        # bound it, and the optimizer probes the image at run time.
        monkeypatch.delenv("AO_DAHENG_MAX_EXPOSURE_MS", raising=False)
        assert validate_daheng_exposure("daheng", 0.5) is None

    def test_daheng_zero_exposure_rejected_for_gs_refine(self) -> None:
        # gs-refine mirrors the CLI: an explicit exposure is required.
        assert validate_daheng_fixed_exposure("daheng", 0.0) is not None
        # model-in-loop may start at 0 (auto from 0.02 ms).
        assert validate_daheng_exposure("daheng", 0.0) is None

    def test_valid_exposure_passes(self) -> None:
        assert validate_daheng_fixed_exposure("daheng", 0.05) is None
        assert validate_daheng_fixed_exposure("sim", 0.0) is None

    def test_config_post_init_also_guards(self, monkeypatch) -> None:
        from ao_shaping.optimizer.wfless.slm_gs_refine import SlmGsRefineConfig

        monkeypatch.setenv("AO_DAHENG_MAX_EXPOSURE_MS", "0.4")
        with pytest.raises(ValueError):
            SlmGsRefineConfig(cam_type="daheng", exposure_time_ms=0.5)


class TestDeviceLifecycle:
    """``st.session_state`` is a no-op proxy in bare mode, so the tests patch a
    plain dict onto the ``streamlit`` module — every consumer (this module and
    ``multi_slm_controller``) resolves ``st.session_state`` at call time."""

    @staticmethod
    def _patch_state(monkeypatch) -> dict[str, Any]:
        state: dict[str, Any] = {}
        monkeypatch.setattr(st, "session_state", state)
        return state

    def test_release_closes_handle_and_clears_state(self, monkeypatch) -> None:
        from ao_shaping.gui.slm import shaping_controls

        state = self._patch_state(monkeypatch)
        fake = _FakeSLM()
        state["slm1"] = fake
        state["slm1_connected"] = True

        shaping_controls._release_slm_handle(1)

        assert fake.close_calls == 1
        assert state["slm1"] is None
        assert state["slm1_connected"] is False

    def test_reacquire_opens_and_syncs_state(self, monkeypatch) -> None:
        from ao_shaping.gui.slm import shaping_controls

        state = self._patch_state(monkeypatch)
        _FakeSantecFactory.instances.clear()
        monkeypatch.setattr(shaping_controls, "Santec", _FakeSantecFactory)

        slm = shaping_controls._reacquire_slm_handle(1)

        assert slm is not None
        assert slm.is_open
        assert state["slm1"] is slm
        assert state["slm1_connected"] is True
        assert state["slm1_phase_preview"] is None
        assert state["slm1_loaded_config"] == {}

    def test_reacquire_syncs_widget_state(self, monkeypatch) -> None:
        """When the sidebar widgets have already been instantiated (re-acquire
        path), ``_safe_set_state`` must not raise even though the widget keys
        are locked — non-widget keys like ``_phase_preview`` still propagate."""
        from ao_shaping.gui.slm import shaping_controls

        state = self._patch_state(monkeypatch)
        _FakeSantecFactory.instances.clear()
        monkeypatch.setattr(shaping_controls, "Santec", _FakeSantecFactory)
        # Simulate already-instantiated widgets by pre-populating the keys.
        state["slm1_wavelength"] = 1064
        state["slm1_shift_x"] = 0
        state["slm1_shift_y"] = 0

        slm = shaping_controls._reacquire_slm_handle(1)

        assert slm is not None
        assert state["slm1"] is slm
        assert state["slm1_connected"] is True
        assert state["slm1_phase_preview"] is None

    def test_sync_reuses_the_config_open_already_read(self, monkeypatch) -> None:
        """``Santec.open()`` reads the per-serial config; the session sync must
        reuse that snapshot instead of reading the same JSON a second time."""
        from ao_shaping.gui.slm import multi_slm_controller

        state = self._patch_state(monkeypatch)
        fake = _FakeSLM()
        reads: list[str] = []

        def load_config() -> dict[str, Any]:
            reads.append("disk")
            return {"wavelength": 9999}

        fake.load_config = load_config  # type: ignore[method-assign]
        fake.loaded_config = {"wavelength": 1064}  # type: ignore[attr-defined]

        multi_slm_controller._sync_slm_session_state(1, fake)

        assert state["slm1_loaded_config"] == {"wavelength": 1064}
        assert reads == []

    def test_sync_falls_back_when_open_did_not_cache(self, monkeypatch) -> None:
        """A handle without ``loaded_config`` (older driver or test double)
        still populates the sidebar from ``load_config()``."""
        from ao_shaping.gui.slm import multi_slm_controller

        state = self._patch_state(monkeypatch)
        fake = _FakeSLM()

        multi_slm_controller._sync_slm_session_state(1, fake)

        assert state["slm1_loaded_config"] == {}

    def test_run_reacquires_even_when_optimizer_raises(self, monkeypatch) -> None:
        """The ``@shaping_session`` contract: the GUI must own the handle again
        even when the optimizer dies mid-run (SDK handles are not re-entrant)."""
        from ao_shaping.gui.slm import shaping_controls
        from ao_shaping.optimizer.wfless import slm_gs_refine

        state = self._patch_state(monkeypatch)
        fake = _FakeSLM()
        state["slm1"] = fake
        state["slm1_connected"] = True

        released_at_call: list[bool] = []

        def _boom(config: Any) -> None:
            # The GUI handle must already be released when the optimizer starts.
            released_at_call.append(state.get("slm1") is None)
            raise RuntimeError("simulated hardware failure")

        _FakeSantecFactory.instances.clear()
        monkeypatch.setattr(slm_gs_refine, "optimize_slm_gs_refine", _boom)
        monkeypatch.setattr(shaping_controls, "Santec", _FakeSantecFactory)

        method = shaping_controls.GsRefineMethod("slm1", 1)
        outcome = method.run(_recipe_values(), debug=False)

        assert outcome is None
        assert released_at_call == [True]
        assert fake.close_calls == 1
        reacquired = state.get("slm1")
        assert reacquired is not None and reacquired.is_open

    def test_run_displays_best_phase_after_success(self, monkeypatch) -> None:
        from ao_shaping.gui.slm import shaping_controls
        from ao_shaping.optimizer.wfless import slm_gs_refine

        state = self._patch_state(monkeypatch)
        fake = _FakeSLM()
        state["slm1"] = fake
        state["slm1_connected"] = True

        class _Recorder:
            """Duck-typed optimizer result: the adapter reads ``history`` plus
            the ``best_phase`` attribute the optimizer attaches on return."""

            def __init__(self) -> None:
                self.history = [
                    {"stage": "flat", "score": 0.5, "pib": 0.8, "cv": 4.0},
                    {"stage": "gs", "score": 0.7, "pib": 0.9, "cv": 0.8, "epoch": 0},
                ]
                self.best_phase = np.full((96, 128), 1.25, dtype=np.float64)
                self.best_frame = np.zeros((250, 250))
                self.best_stage = "gs"

        recorder = _Recorder()

        monkeypatch.setattr(
            slm_gs_refine, "optimize_slm_gs_refine", lambda cfg: recorder
        )
        monkeypatch.setattr(
            shaping_controls,
            "save_gs_refine_results",
            lambda *a, **k: type("R", (), {"save_dir": Path("data/debug/x")}),
        )
        _FakeSantecFactory.instances.clear()
        monkeypatch.setattr(shaping_controls, "Santec", _FakeSantecFactory)

        method = shaping_controls.GsRefineMethod("slm1", 1)
        outcome = method.run(_recipe_values(), debug=False)

        assert outcome is not None
        assert outcome.best_score == 0.7
        assert outcome.flat_score == 0.5
        assert outcome.saved_dir == Path("data/debug/x")
        assert "已采用" in outcome.note
        reacquired = state.get("slm1")
        assert reacquired is not None
        assert len(reacquired.displayed) == 1
        assert np.array_equal(
            reacquired.displayed[0],
            reacquired.create_phase_from_array(recorder.best_phase),
        )
        assert reacquired.memory_mode == shaping_controls.MEMORY_MODE_INTERNAL


class TestMethodRegistry:
    """The registration decorator is the extension point: a third algorithm
    must be addable without touching the section renderer."""

    def test_both_algorithms_registered(self) -> None:
        from ao_shaping.gui.slm import shaping_controls

        assert set(shaping_controls.SHAPING_METHODS) == {"gs-refine", "model-in-loop"}

    def test_lookup_and_unknown_key(self) -> None:
        from ao_shaping.gui.slm import shaping_controls

        assert (
            shaping_controls.get_shaping_method("gs-refine")
            is shaping_controls.GsRefineMethod
        )
        with pytest.raises(KeyError, match="未知闭环整形方法"):
            shaping_controls.get_shaping_method("nope")

    def test_registration_rejects_duplicate_and_keyless(self) -> None:
        from ao_shaping.gui.slm.shaping_controls import (
            ShapingMethod,
            register_shaping_method,
        )

        class _Keyless(ShapingMethod):
            pass

        with pytest.raises(ValueError, match="必须声明非空 key"):
            register_shaping_method(_Keyless)

        class _Duplicate(ShapingMethod):
            key = "gs-refine"

        with pytest.raises(ValueError, match="key 重复"):
            register_shaping_method(_Duplicate)

    def test_widget_keys_are_method_scoped(self) -> None:
        """Switching methods must not leak one algorithm's inputs into the
        other's defaults, so every widget key carries the method key."""
        from ao_shaping.gui.slm.shaping_controls import (
            GsRefineMethod,
            ModelInLoopMethod,
        )

        assert (
            GsRefineMethod("slm1", 1).wkey("exposure") == "slm1_shp_gs-refine_exposure"
        )
        assert (
            ModelInLoopMethod("slm1", 1).wkey("exposure")
            == "slm1_shp_model-in-loop_exposure"
        )


class TestOutcomeNormalisation:
    """Adapters translate optimizer results into ShapingOutcome; the page only
    ever sees the normalised view, which is what keeps the UI stable while the
    optimizers change."""

    def test_gs_recorder_becomes_outcome(self) -> None:
        from ao_shaping.gui.slm.shaping_controls import outcome_from_gs_recorder

        class _Rec:
            history = [
                {"stage": "flat", "score": 0.5, "pib": 0.8, "cv": 4.0},
                {"stage": "gs", "score": 0.7, "pib": 0.9, "cv": 0.8},
                {"stage": "refine", "score": 0.72, "pib": 0.91, "cv": 0.7, "epoch": 3},
            ]
            best_phase = "PHASE"

        out = outcome_from_gs_recorder(_Rec())
        assert out.panel_phase == "PHASE"
        assert out.best_score == 0.72
        assert out.flat_score == 0.5
        assert [row["stage"] for row in out.rows] == ["flat", "gs", "best (refine)"]
        assert out.rows[-1]["epoch"] == "3"
        assert "已采用" in out.note

    def test_gs_bakeoff_rejection_note(self) -> None:
        from ao_shaping.gui.slm.shaping_controls import outcome_from_gs_recorder

        class _Rec:
            history = [
                {"stage": "flat", "score": 0.6, "pib": 0.8, "cv": 4.0},
                {"stage": "gs", "score": 0.5, "pib": 0.8, "cv": 5.0},
            ]
            best_phase = None

        out = outcome_from_gs_recorder(_Rec())
        assert out.best_score == 0.6
        assert "未胜平场" in out.note

    def test_model_in_loop_embeds_model_grid_onto_panel(self) -> None:
        from ao_shaping.gui.slm.shaping_controls import outcome_from_model_in_loop

        class _Cfg:
            panel_span_px = 90.0
            pupil_center = (960, 690)

        class _Res:
            best_phase = np.zeros((64, 64))
            best_score = 0.71
            flat_score = 0.55
            status = "completed"
            target_side_camera = 60
            target_side_model = 22

        out = outcome_from_model_in_loop(_Res(), _Cfg())
        # 900x1200 Santec panel: phase lives inside a disc of radius 45 px.
        assert out.panel_phase is not None
        assert out.panel_phase.shape == (1200, 1920)
        assert out.best_score == 0.71
        assert "completed" in out.note
        assert out.rows[0]["stage"] == "flat"
