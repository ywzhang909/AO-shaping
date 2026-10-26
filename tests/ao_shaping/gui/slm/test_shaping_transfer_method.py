"""Offline contract tests for :class:`TransferMethod` (迁移模型闭环整形).

The adapter is deliberately thin — ``TransferSession`` is the canonical parameter
object and ``run_session`` the canonical orchestrator, exactly as the CLI uses
them.  What must not regress silently:

- the widget values → ``TransferSession`` mapping covers **every** rendered key
  (the shipped bug was a rendered-free ``seed`` read by ``build_params``, so the
  two halves are pinned together here);
- ``validate`` turns every authoritative ``TransferSession.validate`` rejection
  into a message instead of a traceback, and rejects a non-Daheng camera;
- the carrier contract: this phase already carries ``ramp_matrix``, so it takes
  no sidebar send overlay, and showing it must swap the carrier layer out and
  **restore** it even when sending fails;
- ``execute`` calls the pipeline once, and the outcome is read back from what the
  session wrote (nothing is re-derived in the GUI).

No hardware, no Streamlit server: ``st.session_state`` is a plain dict and only
the ``st`` widget calls that need a return value are stubbed.
"""

from __future__ import annotations

import builtins
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import streamlit as st

from ao_shaping.gui.slm import bench_settings as bs
from ao_shaping.gui.slm import shaping_controls as sc
from ao_shaping.tools.slm.transfer.workflow import TransferSession


@pytest.fixture(autouse=True)
def state(monkeypatch):
    state: dict[str, Any] = {}
    monkeypatch.setattr(st, "session_state", state)
    return state


def _values(**over: Any) -> dict[str, Any]:
    """Everything ``render_*`` writes, with the recipe defaults."""
    values = dict(sc.TRANSFER_RECIPE)
    values.update(
        checkpoint="",
        seed_run="",
        calibration=True,
        cam_type="daheng",
        cam_id=3,
        slm_number=1,
    )
    values.update(over)
    return values


def _method() -> sc.TransferMethod:
    return sc.TransferMethod("slm1", 1)


def _record_number_inputs(monkeypatch) -> list[dict[str, Any]]:
    """Capture every ``st.number_input`` spec and answer with its own default.

    Answering with the spec's ``value`` keeps the render offline and keeps the
    widget's own bound validation out of the way — ``st.number_input`` *raises*
    on an out-of-range default, which is pinned separately by
    ``TestSeedIsConsumed::test_the_rendered_seed_fits_the_widget_bounds``.
    """
    specs: list[dict[str, Any]] = []

    def _wrapped(*args, **kwargs):
        spec = dict(kwargs)
        if args and "label" not in spec:
            spec["label"] = args[0]
        specs.append(spec)
        return spec.get("value")

    monkeypatch.setattr(st, "number_input", _wrapped)
    return specs


def _spec_for(specs: list[dict[str, Any]], key_suffix: str) -> dict[str, Any] | None:
    for spec in specs:
        if str(spec.get("key", "")).endswith(key_suffix):
            return spec
    return None


# ---------------------------------------------------------------------------
# widget values → canonical parameter object
# ---------------------------------------------------------------------------


class TestParamMapping:
    def test_every_recipe_field_reaches_the_session(self) -> None:
        session = _method().build_params(_values())

        assert isinstance(session, TransferSession)
        assert session.seed == 102319
        assert session.count == 24
        assert session.shapes == ("square",)
        assert session.iterations == 60
        assert session.blocks == 4
        assert session.updates == 10
        assert session.frames == 3
        assert session.side == pytest.approx(80.0)
        assert session.period == pytest.approx(64.0)
        assert session.adaptation == "auto"
        assert session.objective == "shape"
        assert session.accept_metric == "loss"
        assert session.seed_policy == "transfer"
        # True means "let the shape selection decide" (single shape → skipped),
        # never "force calibration on".
        assert session.calibration is None
        assert session.camera == "daheng"
        assert session.cam_id == 3
        assert session.slm_number == 1
        # Both switches default to "measured behaviour": A–B–B–A on, npz off.
        assert session.abba is False
        assert session.save_dataset is False
        # Empty boxes are a cold start, not a path named "".
        assert session.checkpoint is None
        assert session.seed_run is None

    def test_an_empty_checkpoint_box_is_a_cold_start(self) -> None:
        session = _method().build_params(_values(checkpoint="   ", seed_run="  "))
        assert session.checkpoint is None
        assert session.seed_run is None

    def test_a_typed_checkpoint_and_seed_run_are_paths(self, tmp_path) -> None:
        checkpoint = tmp_path / "frozen_parameters.json"
        checkpoint.write_text("{}", encoding="utf-8")
        seed_run = tmp_path / "anchored"
        seed_run.mkdir()

        session = _method().build_params(
            _values(checkpoint=str(checkpoint), seed_run=str(seed_run))
        )
        assert session.checkpoint == checkpoint
        assert session.seed_run == seed_run

    def test_the_session_dir_is_a_fresh_timestamped_path(self) -> None:
        session = _method().build_params(_values())
        assert re.match(r"^\d{8}_\d{6}(-\d+)?$", session.out.name)
        assert session.out.parent == Path("data") / "debug" / "slm_transfer"


class TestAcceptanceAndDatasetSwitches:
    """The two cost knobs: paired acceptance, and writing the probe npz files."""

    def test_both_switches_reach_the_session(self) -> None:
        session = _method().build_params(_values(abba=False, save_dataset=True))

        assert session.abba is False
        assert session.save_dataset is True

    def test_render_advanced_renders_both_checkboxes_with_recipe_defaults(
        self, monkeypatch
    ) -> None:
        specs: list[dict[str, Any]] = []

        def _checkbox(label, value=None, key=None, help=None):
            specs.append(dict(label=label, value=value, key=key))
            return value

        monkeypatch.setattr(st, "checkbox", _checkbox)

        _method().render_advanced({})

        by_key = {str(spec["key"]): spec for spec in specs}
        abba = by_key["slm1_shp_transfer_abba"]
        dataset = by_key["slm1_shp_transfer_save_dataset"]
        assert abba["value"] is sc.TRANSFER_RECIPE["abba"] is False
        assert dataset["value"] is sc.TRANSFER_RECIPE["save_dataset"] is False

    def test_switching_acceptance_off_is_visible_in_the_plan(self, tmp_path) -> None:
        """A session.json reader must see that no CCD verdict was measured."""
        session = TransferSession(out=tmp_path / "session", shapes=("square",), abba=False)

        plan = session.plan()

        assert plan["resolved"]["acceptance"] == "model_only"
        assert "direction_check" not in plan["stages"]
        assert "A–B–B–A acceptance skipped" in plan["note"]
        assert TransferSession(out=tmp_path / "on", shapes=("square",), abba=True).plan()["resolved"]["acceptance"] == "paired_abba"


class TestDeviceControl:
    """The device selectbox: ``auto`` by default, and it reaches the session."""

    def test_the_device_choice_reaches_the_session(self) -> None:
        assert _method().build_params(_values()).device == "auto"
        assert _method().build_params(_values(device="cpu")).device == "cpu"

    def test_render_advanced_renders_the_three_device_choices(self, monkeypatch) -> None:
        specs: list[dict[str, Any]] = []

        def _selectbox(label, options=None, index=0, key=None, help=None, **kwargs):
            specs.append(dict(label=label, options=list(options or ()), index=index, key=key))
            return list(options)[index]

        monkeypatch.setattr(st, "selectbox", _selectbox)

        _method().render_advanced({})

        spec = next(s for s in specs if str(s["key"]).endswith("_shp_transfer_device"))
        assert spec["options"] == ["auto", "cpu", "cuda"]
        # The rendered default is the recipe's, so the operator sees "auto".
        assert spec["options"][spec["index"]] == sc.TRANSFER_RECIPE["device"] == "auto"

    def test_an_unknown_device_is_refused_before_running(self) -> None:
        message = _method().validate(_values(device="gpu"))

        assert isinstance(message, str)
        assert "unknown device" in message


class TestPlanTransferSessionDir:
    def test_planning_does_not_create_the_directory(self, tmp_path) -> None:
        """``run_session`` makes it with ``exist_ok=False`` — a pre-created path
        would be refused, so planning must stay a pure name computation."""
        planned = sc.plan_transfer_session_dir(tmp_path)

        assert not planned.exists()
        assert not planned.parent.exists()
        assert planned.parent == tmp_path / "debug" / "slm_transfer"
        assert re.match(r"^\d{8}_\d{6}(-\d+)?$", planned.name)


class TestSeedIsConsumed:
    """The shipped bug: ``seed`` was read by ``build_params`` but never rendered."""

    def test_a_missing_seed_raises_instead_of_defaulting(self) -> None:
        values = _values()
        values.pop("seed")
        with pytest.raises(KeyError, match="seed"):
            _method().build_params(values)

    def test_render_budget_renders_the_seed_widget(self, monkeypatch) -> None:
        specs = _record_number_inputs(monkeypatch)

        _method().render_budget({})

        seed = _spec_for(specs, "_shp_transfer_seed")
        assert seed is not None, [spec.get("key") for spec in specs]
        assert seed["value"] == sc.TRANSFER_RECIPE["seed"]

    def test_the_rendered_seed_fits_the_widget_bounds(self, monkeypatch) -> None:
        """Every rendered row must satisfy ``min_value <= value <= max_value``.

        Streamlit *raises* (``StreamlitValueAboveMaxError``) instead of clamping,
        so an out-of-range default is not a cosmetic issue: the whole section
        fails to render and the method cannot be selected at all.
        """
        specs = _record_number_inputs(monkeypatch)

        _method().render_budget({})

        offenders = [
            spec
            for spec in specs
            if spec.get("value") is not None
            and not (spec["min_value"] <= spec["value"] <= spec["max_value"])
        ]
        assert offenders == []


class TestMakeConfig:
    def test_debug_never_switches_the_evidence_off(self, tmp_path) -> None:
        session = TransferSession(out=tmp_path / "session")
        assert _method().make_config(session, debug=False) is session
        assert _method().make_config(session, debug=True) is session


# ---------------------------------------------------------------------------
# validate — the session's own rules, surfaced instead of raised
# ---------------------------------------------------------------------------


class TestValidate:
    def test_a_valid_recipe_passes(self) -> None:
        assert _method().validate(_values()) is None

    def test_a_non_daheng_camera_is_refused(self) -> None:
        message = _method().validate(_values(cam_type="sim"))

        assert isinstance(message, str)
        assert "大恒" in message

    def test_an_empty_shape_selection_is_refused(self) -> None:
        message = _method().validate(_values(shapes=[]))

        assert isinstance(message, str)
        assert "形状" in message

    def test_a_count_below_the_widget_floor_surfaces_the_session_error(self) -> None:
        message = _method().validate(_values(count=2))

        assert isinstance(message, str)
        assert "count >= 6" in message

    def test_an_unsupported_shape_surfaces_the_session_error(self) -> None:
        message = _method().validate(_values(shapes=("hexagon",)))

        assert isinstance(message, str)
        assert "shape" in message

    def test_a_missing_seed_run_manifest_surfaces_as_a_message(self, tmp_path) -> None:
        empty = tmp_path / "no_manifest"
        empty.mkdir()

        message = _method().validate(_values(seed_run=str(empty)))

        assert isinstance(message, str)
        assert "manifest.json" in message

    def test_a_missing_torch_is_reported_not_raised(self, monkeypatch) -> None:
        real_import = builtins.__import__

        def _no_torch(name: str, *args: Any, **kwargs: Any):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("No module named 'torch'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_torch)

        message = _method().validate(_values())

        assert isinstance(message, str)
        assert "torch" in message


# ---------------------------------------------------------------------------
# execute — one call to the canonical pipeline
# ---------------------------------------------------------------------------


class TestExecuteDelegatesOnce:
    def test_run_session_is_called_once_with_the_stage_hook(
        self, monkeypatch, tmp_path
    ) -> None:
        from ao_shaping.tools.slm.transfer import workflow

        calls: list[dict[str, Any]] = []

        def _run_session(config, *, offline=False, on_stage=None):
            calls.append({"config": config, "offline": offline, "on_stage": on_stage})

        monkeypatch.setattr(workflow, "run_session", _run_session)

        sentinel = object()
        read_back: list[tuple[Any, Any]] = []

        def _outcome(session_dir, shapes):
            read_back.append((session_dir, shapes))
            return sentinel

        monkeypatch.setattr(sc, "outcome_from_transfer", _outcome)

        session = TransferSession(out=tmp_path / "session", shapes=("square",))
        result = _method().execute(session)

        assert result is sentinel
        assert len(calls) == 1
        assert calls[0]["config"] is session
        assert calls[0]["offline"] is False
        assert callable(calls[0]["on_stage"])
        assert read_back == [(Path(session.out), ("square",))]


class _ProgressRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[float, str | None]] = []

    def progress(self, value: float, text: str | None = None) -> None:
        self.calls.append((value, text))


class _PlannedConfig:
    """Just the surface ``prepare_config`` reads (4 stages, one of them unlabelled)."""

    stages = ["capture", "fit", "summary", "unlabelled_stage"]

    def plan(self) -> dict[str, Any]:
        return {"stages": list(self.stages)}


class TestStageProgress:
    def test_completed_stages_advance_the_bar(self, monkeypatch) -> None:
        bar = _ProgressRecorder()
        monkeypatch.setattr(st, "progress", lambda value, text=None: bar)

        method = _method()
        method.prepare_config(_PlannedConfig(), {})

        for name in _PlannedConfig.stages:
            method._on_stage(name, "complete")

        assert len(bar.calls) == 4
        assert bar.calls[-1][0] == pytest.approx(1.0)
        # An unmapped stage name falls back to itself rather than KeyError.
        assert bar.calls[-1][1].endswith("unlabelled_stage")
        assert sc.TRANSFER_STAGE_LABELS["summary"] in bar.calls[2][1]

    @pytest.mark.parametrize("status", ["running", "failed"])
    def test_only_completion_advances_the_bar(self, monkeypatch, status: str) -> None:
        bar = _ProgressRecorder()
        monkeypatch.setattr(st, "progress", lambda value, text=None: bar)

        method = _method()
        method.prepare_config(_PlannedConfig(), {})
        method._on_stage("capture", status)

        assert bar.calls == []
        assert method._stage_index == 0

    def test_a_direct_execute_call_does_not_raise(self) -> None:
        """``__init__`` initialises the stage counters, so ``_on_stage`` is safe
        even when ``prepare_config`` never ran (no bar → no-op)."""
        method = _method()
        assert (method._stage_total, method._stage_index) == (0, 0)

        method._on_stage("capture", "complete")

        assert method._stage_index == 0


# ---------------------------------------------------------------------------
# carrier contract — the phase carries ``ramp_matrix`` itself
# ---------------------------------------------------------------------------


class TestCarrierContract:
    def test_transfer_is_the_only_carrier_bearing_method(self) -> None:
        assert sc.TransferMethod.phase_carries_carrier is True
        assert sc.TransferMethod.applies_send_overlay is False
        # No override: it uses the shared swap-and-restore implementation.
        assert "show_phase" not in sc.TransferMethod.__dict__
        assert "show_phase" in sc.ShapingMethod.__dict__

    def test_show_phase_swaps_the_carrier_out_and_restores_it(self, monkeypatch) -> None:
        built: list[dict[str, Any]] = []
        applied: list[Any] = []

        def _build(prefix, slm=None, *, include_carrier=True):
            built.append({"prefix": prefix, "include_carrier": include_carrier})
            return f"overlay(carrier={include_carrier})"

        monkeypatch.setattr(bs, "build_send_overlay", _build)
        monkeypatch.setattr(
            sc,
            "apply_overlays_to_handle",
            lambda handle, overlay: applied.append(overlay),
        )

        class _Slm:
            _max_gray = 993

            def create_phase_from_array(self, phase):
                return np.asarray(phase, dtype=np.uint16)

            def display_data(self, gray, *, memory_mode):
                raise RuntimeError("设备忙")

        outcome = sc.ShapingOutcome(panel_phase=np.zeros((4, 4)), best_score=0.1)
        _method().show_phase(_Slm(), outcome)

        assert [entry["include_carrier"] for entry in built] == [False, True]
        assert applied == ["overlay(carrier=False)", "overlay(carrier=True)"]

    def test_no_phase_means_no_overlay_swap(self, monkeypatch) -> None:
        applied: list[Any] = []
        monkeypatch.setattr(
            sc, "apply_overlays_to_handle", lambda handle, overlay: applied.append(overlay)
        )

        _method().show_phase(object(), sc.ShapingOutcome(panel_phase=None, best_score=0.0))

        assert applied == []


# ---------------------------------------------------------------------------
# outcome — read back only what the session wrote
# ---------------------------------------------------------------------------


def _session_dir(
    tmp_path: Path,
    *,
    with_best: bool = True,
    journal: list[dict[str, Any]] | None = None,
) -> Path:
    root = tmp_path / "session"
    anchored = root / "anchored"
    anchored.mkdir(parents=True)
    (anchored / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "label": "square_seed",
                    "metrics": {"cv": 0.80, "ee_roi": 0.41},
                    "exposure_ms": 0.31,
                },
                {
                    "label": "square_final",
                    "metrics": {"cv": 0.61, "ee_roi": 0.47},
                    "exposure_ms": 0.29,
                },
            ]
        ),
        encoding="utf-8",
    )
    if with_best:
        # 8×8 exactly tiles the 1920×1200 panel (capture.to_panel's contract).
        np.save(anchored / "square_best.npy", np.zeros((8, 8), dtype=np.float64))
    if journal is not None:
        (root / "journal.json").write_text(json.dumps(journal), encoding="utf-8")
    return root


class TestOutcomeFromTransfer:
    def test_reads_the_audited_metrics_and_tiles_the_phase(self, tmp_path) -> None:
        root = _session_dir(
            tmp_path,
            journal=[
                {"stage": "capture", "status": "complete"},
                {"stage": "summary", "status": "failed"},
            ],
        )

        outcome = sc.outcome_from_transfer(root, ("square",))

        assert len(outcome.rows) == 2
        assert outcome.rows[0]["阶段"] == "接受初值 (seed)"
        assert outcome.rows[0]["CV"] == pytest.approx(0.80)
        assert outcome.rows[1]["阶段"] == "验收后 (final)"
        assert outcome.rows[1]["CV"] == pytest.approx(0.61)
        assert outcome.rows[1]["EE_ROI"] == pytest.approx(0.47)
        assert outcome.panel_phase is not None
        # The model control grid is tiled onto the physical panel, no resampling.
        assert outcome.panel_phase.shape == (1200, 1920)
        assert outcome.best_score == pytest.approx(0.61)
        assert outcome.flat_score == pytest.approx(0.80)
        assert outcome.saved_dir == root
        assert "正方形" in outcome.note
        assert "失败阶段" in outcome.note

    def test_without_an_accepted_phase_it_reports_instead_of_crashing(
        self, tmp_path
    ) -> None:
        root = _session_dir(tmp_path, with_best=False)

        outcome = sc.outcome_from_transfer(root, ("square",))

        assert outcome.panel_phase is None
        assert math.isnan(outcome.best_score)
        assert "没有可下发的接受相位" in outcome.note
        assert outcome.saved_dir == root
        # The metrics that *were* audited are still shown.
        assert len(outcome.rows) == 2

    def test_a_multi_shape_run_says_which_phase_it_put_on_the_panel(
        self, tmp_path
    ) -> None:
        root = _session_dir(tmp_path)

        outcome = sc.outcome_from_transfer(root, ("square", "triangle"))

        assert outcome.panel_phase is not None
        assert "多个形状" in outcome.note
