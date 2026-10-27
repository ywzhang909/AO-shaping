"""Offline tests for the 迁移学习闭环 progress detail panel.

``ao_shaping.gui.slm.transfer_progress`` has two halves: a pure reader
(:func:`collect_progress`, no streamlit, no hardware) and a renderer
(:func:`render_progress`). The renderer is only worth testing through the text it
emits, so ``streamlit`` is stubbed down to ``markdown`` / ``caption`` — no server,
no script run.

What these tests hold down:

1. **The stage list and its statuses come from ``session.json`` + ``journal.json``**
   — the panel must never re-implement the stage order (``run_session`` is the only
   orchestrator), and a stage that the plan lists but the run skipped reads
   ``skipped`` instead of staying 「待运行」 forever.
2. **Every metric is read from what the pipeline wrote.** The capture / fit /
   acceptance numbers below are the shapes ``capture.py``, ``fit.py`` and
   ``paired.py`` actually produce; the fit quality is pinned to the *validation*
   probes because that is the same call ``fit/adaptation.json`` escalates on.
3. **「未产出」is not ``0``.** A stage that just started, a failed stage and a
   model-only acceptance block all report ``value is None`` + 「—」. The pipeline's
   real zeros (drift 0, CV error 0) must stay distinguishable from "not measured".
4. **Nothing raises on a partial session.** An empty / missing / half-written
   session directory is the normal state during a run, and ``collect_progress`` is
   called from the stage hook with whatever exists at that instant.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest
import streamlit as st

from ao_shaping.gui.slm import transfer_progress as tp

STAGES = [
    "capture",
    "nominal_initialization",
    "fit",
    "CCD_anchored_blocks",
    "raw_data_audit",
    "summary",
    "direction_check",
]


# ---------------------------------------------------------------------------
# synthetic session directory
# ---------------------------------------------------------------------------


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _entry(
    name: str, status: str, *, start: float = 0.0, seconds: float | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """One ``journal.json`` row, shaped exactly like ``workflow.stage()``'s."""
    entry: dict[str, Any] = {
        "stage": name,
        "status": status,
        "timestamp_utc": "2026-10-26T10:00:00.000000+00:00",
        "monotonic_ns": int(start * 1e9),
    }
    if seconds is not None:
        entry["end"] = {
            "timestamp_utc": "2026-10-26T10:00:12.000000+00:00",
            "monotonic_ns": int((start + seconds) * 1e9),
        }
    if error is not None:
        entry["error"] = error
    return entry


def _probe(
    label: str, split: str, exposure_ms: float, peak: float, *, status: str = "ok"
) -> dict[str, Any]:
    """One ``capture/manifest.json`` row (the keys ``fit.valid`` reads included)."""
    return {
        "file": f"sample_{label}.npz",
        "label": label,
        "split": split,
        "exposure_ms": exposure_ms,
        "peak": peak,
        "spot_xy": [812.5, 640.25],
        "exposure_setting": {"status": status},
        "frames": [
            {
                "timestamp_utc": "2026-10-26T10:00:01.000000+00:00",
                "monotonic_ns": 1,
                "end_monotonic_ns": 2,
                "exposure_ms": exposure_ms,
            }
        ],
    }


CAPTURE = [
    _probe("reference_start", "calibration", 0.02, 180.0),
    _probe("probe_000_zernike", "train", 0.85, 200.0),
    _probe("probe_003_zernike", "validation", 0.40, 120.0),
]

#: The fit selects on train loss; the reported quality is the *validation* one.
FIT_RESULT = {
    "iteration": 12,
    "loss": 0.0412,
    "metrics": [
        {"correlation": 0.10, "cv_prediction": 0.50, "cv_measured": 0.60,
         "cv_abs_error": 0.50, "relative_l2": 0.90},
        {"correlation": 0.95, "cv_prediction": 0.60, "cv_measured": 0.61,
         "cv_abs_error": 0.01, "relative_l2": 0.11},
    ],
}
FIT_PROVENANCE = {"rows": [{"split": "train"}, {"split": "validation"}]}

ANCHORED = [
    {"shape": "square", "block": 0, "accept_metric": "loss", "baseline": 0.09,
     "candidate": 0.07, "drift": 0.002, "accepted": True},
    {"shape": "square", "block": 1, "accept_metric": "loss", "baseline": 0.08,
     "candidate": 0.09, "drift": 0.004, "accepted": False},
]


def _session(
    tmp_path: Path,
    *,
    journal: list[dict[str, Any]],
    stages: list[str] | None = None,
    capture: list[dict[str, Any]] | None = None,
    fit: tuple[dict[str, Any], dict[str, Any]] | None = None,
    anchored: list[dict[str, Any]] | None = None,
) -> Path:
    out = tmp_path / "session"
    out.mkdir(parents=True, exist_ok=True)
    if stages is not None:
        _write(out / "session.json", {"stages": list(stages)})
    _write(out / "journal.json", journal)
    if capture is not None:
        _write(out / "capture" / "manifest.json", capture)
    if fit is not None:
        _write(out / "fit" / "result.json", fit[0])
        _write(out / "fit" / "provenance.json", fit[1])
    if anchored is not None:
        _write(out / "anchored" / "summary.json", anchored)
    return out


def _stage(view: dict[str, Any], name: str) -> dict[str, Any]:
    return next(stage for stage in view["stages"] if stage["name"] == name)


def _metric(stage: dict[str, Any], key: str) -> dict[str, Any]:
    return next(metric for metric in stage["metrics"] if metric["key"] == key)


def _keys(stage: dict[str, Any]) -> list[str]:
    return [metric["key"] for metric in stage["metrics"]]


# ---------------------------------------------------------------------------
# stage list / statuses / current stage
# ---------------------------------------------------------------------------


class TestStageList:
    def test_the_plan_supplies_the_order_and_the_journal_the_statuses(self, tmp_path):
        out = _session(
            tmp_path,
            stages=STAGES,
            journal=[
                _entry("capture", "complete", seconds=12.0),
                _entry("nominal_initialization", "complete", start=12.0, seconds=1.5),
                _entry("fit", "running", start=13.5),
            ],
        )

        view = tp.collect_progress(out)

        assert [stage["name"] for stage in view["stages"]] == STAGES
        assert [stage["status"] for stage in view["stages"]] == [
            "complete", "complete", "running", "pending", "pending", "pending", "pending",
        ]
        assert view["current"]["name"] == "fit"
        assert view["next"]["name"] == "CCD_anchored_blocks"
        assert view["counts"] == {
            "pending": 4, "running": 1, "complete": 2, "failed": 0, "skipped": 0, "total": 7,
        }
        assert view["finished"] is False
        assert view["failed"] is False
        assert view["exists"] is True
        assert view["out_dir"] == str(out)

    def test_a_stage_carries_its_label_position_and_duration(self, tmp_path):
        out = _session(
            tmp_path, stages=STAGES,
            journal=[_entry("capture", "complete", seconds=12.0)],
        )

        stage = _stage(tp.collect_progress(out), "capture")

        assert stage["label"] == tp.STAGE_LABELS["capture"]
        assert stage["index"] == 1
        assert stage["duration_s"] == pytest.approx(12.0)
        assert stage["started"] == "2026-10-26T10:00:00.000000+00:00"
        assert stage["ended"] == "2026-10-26T10:00:12.000000+00:00"
        assert _metric(stage, "duration")["text"] == "12.0 s"

    def test_the_last_planned_stage_completing_finishes_the_run(self, tmp_path):
        out = _session(
            tmp_path, stages=STAGES,
            journal=[_entry(name, "complete", seconds=1.0) for name in STAGES],
        )

        view = tp.collect_progress(out)

        assert view["finished"] is True
        assert view["next"] is None
        assert view["current"]["name"] == "direction_check"
        assert view["counts"]["complete"] == len(STAGES)

    def test_a_failed_stage_keeps_its_error_and_the_run_is_not_finished(self, tmp_path):
        out = _session(
            tmp_path, stages=STAGES,
            journal=[
                _entry("capture", "complete", seconds=2.0),
                _entry("nominal_initialization", "failed", start=2.0, seconds=0.5,
                       error="ValueError: degenerate camera geometry"),
            ],
        )

        view = tp.collect_progress(out)

        failed = _stage(view, "nominal_initialization")
        assert failed["status"] == "failed"
        assert failed["error"] == "ValueError: degenerate camera geometry"
        assert failed["details"][0].startswith("错误：ValueError")
        assert view["failed"] is True
        assert view["finished"] is False
        assert view["current"]["name"] == "nominal_initialization"

    def test_a_planned_stage_that_never_ran_reads_skipped(self, tmp_path):
        """``reuse_capture`` is planned for a recorded dataset but never journaled."""
        out = _session(
            tmp_path,
            stages=["reuse_capture", "fit", "summary"],
            journal=[_entry("fit", "running", seconds=1.0)],
        )

        view = tp.collect_progress(out)

        assert _stage(view, "reuse_capture")["status"] == "skipped"
        assert view["counts"]["skipped"] == 1
        assert view["counts"]["pending"] == 1

    def test_a_stage_the_plan_forgot_still_appears(self, tmp_path):
        out = _session(
            tmp_path, stages=["capture"],
            journal=[_entry("capture", "complete", seconds=1.0),
                     _entry("summary", "running", start=1.0)],
        )

        view = tp.collect_progress(out)

        assert [stage["name"] for stage in view["stages"]] == ["capture", "summary"]
        assert view["counts"]["total"] == 2


# ---------------------------------------------------------------------------
# per-stage metrics, read from the artefacts the pipeline writes
# ---------------------------------------------------------------------------


class TestCaptureMetrics:
    def test_probes_splits_usable_exposure_and_peak(self, tmp_path):
        out = _session(
            tmp_path, stages=STAGES,
            journal=[_entry("capture", "running")], capture=CAPTURE,
        )

        stage = _stage(tp.collect_progress(out), "capture")

        assert _metric(stage, "probes")["text"] == "3 条"
        assert _metric(stage, "splits")["text"] == "校准 1 / 训练 1 / 验证 1"
        assert _metric(stage, "usable")["text"] == "3 条"
        assert _metric(stage, "exposure")["text"] == "0.02–0.85 ms"
        assert _metric(stage, "peak")["text"] == "200/255"

    def test_an_unusable_probe_is_counted_but_not_usable(self, tmp_path):
        """``fit.valid`` rejects a saturated probe — 探针 still counts it."""
        capture = [*CAPTURE, _probe("probe_005_zernike", "train", 0.9, 255.0)]
        out = _session(
            tmp_path, stages=STAGES,
            journal=[_entry("capture", "complete", seconds=1.0)], capture=capture,
        )

        stage = _stage(tp.collect_progress(out), "capture")

        assert _metric(stage, "probes")["text"] == "4 条"
        assert _metric(stage, "usable")["text"] == "3 条"


class TestFitMetrics:
    def test_the_quality_is_reported_on_the_validation_probes(self, tmp_path):
        out = _session(
            tmp_path, stages=STAGES,
            journal=[_entry("fit", "complete", seconds=30.0)],
            fit=(FIT_RESULT, FIT_PROVENANCE),
        )

        stage = _stage(tp.collect_progress(out), "fit")

        assert _metric(stage, "iteration")["text"] == "12"
        assert _metric(stage, "loss")["text"] == "0.0412"
        # The train probe says 0.10 / 0.90; the validation probe says 0.95 / 0.11.
        assert _metric(stage, "correlation")["label"] == "相关系数（验证）"
        assert _metric(stage, "correlation")["value"] == pytest.approx(0.95)
        assert _metric(stage, "correlation")["text"] == "0.950"
        assert _metric(stage, "cv_abs_error")["text"] == "0.0100"
        assert _metric(stage, "relative_l2")["text"] == "0.110"

    def test_without_provenance_the_quality_falls_back_to_every_probe(self, tmp_path):
        out = _session(
            tmp_path, stages=STAGES,
            journal=[_entry("fit", "complete", seconds=30.0)],
            fit=(FIT_RESULT, {"rows": []}),
        )

        stage = _stage(tp.collect_progress(out), "fit")

        assert _metric(stage, "correlation")["label"] == "相关系数（全部）"
        assert _metric(stage, "correlation")["value"] == pytest.approx(0.525)

    def test_the_automatic_adaptation_decision_is_a_detail_line(self, tmp_path):
        out = _session(
            tmp_path, stages=STAGES,
            journal=[_entry("fit", "complete", seconds=30.0)],
            fit=(FIT_RESULT, FIT_PROVENANCE),
        )
        _write(
            out / "fit" / "adaptation.json",
            {
                "selected_stage": "alignment",
                "escalated": False,
                "escalation_heuristic": {"correlation_below": 0.85, "relative_l2_above": 0.25},
            },
        )

        stage = _stage(tp.collect_progress(out), "fit")

        assert stage["details"] == [
            "自动适配：选中 alignment（未升级）；升级判定阈 相关系数 < 0.85 或 相对 L2 > 0.25"
        ]


class TestAcceptanceMetrics:
    def test_accepted_baseline_candidate_and_drift(self, tmp_path):
        out = _session(
            tmp_path, stages=STAGES,
            journal=[_entry("CCD_anchored_blocks", "complete", seconds=90.0)],
            anchored=ANCHORED,
        )

        stage = _stage(tp.collect_progress(out), "CCD_anchored_blocks")

        assert _metric(stage, "accepted")["value"] == 1
        assert _metric(stage, "accepted")["text"] == "1/2 块"
        assert _metric(stage, "baseline_candidate")["text"] == "基线 0.0850 → 候选 0.0800"
        assert _metric(stage, "drift")["text"] == "0.0040"
        assert _metric(stage, "scope")["text"] == "配对 A–B–B–A（损失）"

    def test_each_block_is_one_nested_detail_line(self, tmp_path):
        out = _session(
            tmp_path, stages=STAGES,
            journal=[_entry("CCD_anchored_blocks", "complete", seconds=90.0)],
            anchored=ANCHORED,
        )

        stage = _stage(tp.collect_progress(out), "CCD_anchored_blocks")

        assert stage["details"] == [
            "第 1 块（正方形）：接受 · 基线 0.0900 → 候选 0.0700 · 漂移 0.0020",
            "第 2 块（正方形）：拒绝 · 基线 0.0800 → 候选 0.0900 · 漂移 0.0040",
        ]

    def test_a_model_only_block_reports_the_paired_metrics_as_absent(self, tmp_path):
        """No A–B–B–A apparatus ⇒ no baseline/candidate/drift, and no fake zero."""
        summary = [
            {"shape": "square", "block": 0, "acceptance": "model_only",
             "accept_metric": "model", "accepted": True,
             "model_loss_previous": 0.09, "model_loss_last": 0.07},
            {"shape": "square", "block": 1, "acceptance": "model_only",
             "accept_metric": "model", "accepted": False,
             "model_loss_previous": 0.07, "model_loss_last": 0.08},
        ]
        out = _session(
            tmp_path, stages=STAGES,
            journal=[_entry("CCD_anchored_blocks", "complete", seconds=90.0)],
            anchored=summary,
        )

        stage = _stage(tp.collect_progress(out), "CCD_anchored_blocks")

        assert _metric(stage, "accepted")["text"] == "1/2 块"
        for key in ("baseline_candidate", "drift"):
            assert _metric(stage, key)["value"] is None
            assert _metric(stage, key)["text"] == tp.MISSING
        assert _metric(stage, "scope")["text"] == "仅模型损失"
        assert stage["details"][0] == "第 1 块（正方形）：接受 · 模型损失 0.0900 → 0.0700"

    def test_more_blocks_than_the_cap_point_at_the_session_directory(self, tmp_path):
        summary = [
            {"shape": "square", "block": index, "accept_metric": "loss",
             "baseline": 0.09, "candidate": 0.08, "drift": 0.001, "accepted": True}
            for index in range(tp.MAX_DETAIL_LINES + 3)
        ]
        out = _session(
            tmp_path, stages=STAGES,
            journal=[_entry("CCD_anchored_blocks", "complete", seconds=90.0)],
            anchored=summary,
        )

        details = _stage(tp.collect_progress(out), "CCD_anchored_blocks")["details"]

        assert len(details) == tp.MAX_DETAIL_LINES + 1
        assert details[-1].startswith("…其余 3 块见 ")
        assert details[-1].endswith("anchored/summary.json") or details[-1].endswith(
            "anchored\\summary.json"
        )

    def test_the_summary_stage_reports_seed_and_final_cv_per_shape(self, tmp_path):
        out = _session(
            tmp_path, stages=STAGES,
            journal=[_entry("summary", "complete", seconds=20.0)],
        )
        _write(
            out / "summary" / "results.json",
            {
                "acceptance": {"scopes": ["paired"], "accepted_blocks": 3,
                               "model_only_blocks": 0},
                "shapes": [
                    {
                        "shape": "square",
                        "seed": {"measured": {"cv": 0.612}},
                        "final": {"measured": {"cv": 0.524}, "correlation": 0.941},
                    }
                ],
            },
        )

        stage = _stage(tp.collect_progress(out), "summary")

        assert _metric(stage, "acceptance")["text"] == "配对 A–B–B–A"
        assert _metric(stage, "accepted")["text"] == "3 块"
        assert stage["details"] == ["正方形：CV 0.612 → 0.524 · 模型相关系数 0.941"]


# ---------------------------------------------------------------------------
# robustness: missing / partial artefacts never raise and never read as 0
# ---------------------------------------------------------------------------


class TestMissingArtefacts:
    def test_a_session_that_has_not_written_anything_yet(self, tmp_path):
        out = _session(tmp_path, stages=STAGES, journal=[])

        view = tp.collect_progress(out)

        assert [stage["status"] for stage in view["stages"]] == ["pending"] * len(STAGES)
        assert view["current"] is None
        assert view["finished"] is False
        assert all(stage["metrics"] == [] for stage in view["stages"])

    def test_a_missing_directory_returns_an_empty_view(self, tmp_path):
        view = tp.collect_progress(tmp_path / "does-not-exist")

        assert view["stages"] == []
        assert view["exists"] is False
        assert view["counts"]["total"] == 0
        assert view["current"] is None

    def test_none_is_accepted_before_the_session_directory_is_planned(self):
        """``_on_stage`` may fire before ``build_params`` set ``_session_dir``."""
        view = tp.collect_progress(None)

        assert view["stages"] == []
        assert view["out_dir"] is None

    def test_a_stage_with_no_artefact_has_no_metrics_not_zero_metrics(self, tmp_path):
        out = _session(
            tmp_path, stages=STAGES,
            journal=[_entry("fit", "complete", seconds=30.0)],
        )

        stage = _stage(tp.collect_progress(out), "fit")

        # 只有 journal 里读得到的耗时；没有任何被 0 填满的指标。
        assert _keys(stage) == ["duration"]
        assert stage["details"] == []

    def test_a_half_written_artefact_is_treated_as_missing(self, tmp_path):
        out = _session(
            tmp_path, stages=STAGES,
            journal=[_entry("fit", "complete", seconds=30.0)],
        )
        (out / "fit").mkdir(parents=True, exist_ok=True)
        (out / "fit" / "result.json").write_text('{"iteration": 3, "loss":', encoding="utf-8")

        stage = _stage(tp.collect_progress(out), "fit")

        assert _keys(stage) == ["duration"]

    def test_an_artefact_with_renamed_fields_reports_the_survivors_only(self, tmp_path):
        out = _session(
            tmp_path, stages=STAGES,
            journal=[_entry("fit", "complete", seconds=30.0)],
            fit=({"iteration": 7, "metrics": [{"correlation": 0.8}]}, {"rows": []}),
        )

        stage = _stage(tp.collect_progress(out), "fit")

        assert _metric(stage, "iteration")["text"] == "7"
        assert _metric(stage, "loss")["value"] is None
        assert _metric(stage, "loss")["text"] == tp.MISSING
        assert _metric(stage, "correlation")["value"] == pytest.approx(0.8)
        assert _metric(stage, "cv_abs_error")["value"] is None

    def test_a_real_zero_is_not_confused_with_a_missing_value(self, tmp_path):
        """``drift = 0`` is a measurement; it must render as 0, not 「—」."""
        out = _session(
            tmp_path, stages=STAGES,
            journal=[_entry("CCD_anchored_blocks", "complete", seconds=90.0)],
            anchored=[{"shape": "square", "block": 0, "accept_metric": "loss",
                       "baseline": 0.09, "candidate": 0.08, "drift": 0.0,
                       "accepted": True}],
        )

        stage = _stage(tp.collect_progress(out), "CCD_anchored_blocks")

        assert _metric(stage, "drift")["value"] == 0.0
        assert _metric(stage, "drift")["text"] == "0.0000"


# ---------------------------------------------------------------------------
# import boundary: the reader must not drag streamlit in
# ---------------------------------------------------------------------------


class TestImportBoundary:
    def test_streamlit_is_only_imported_inside_the_renderer(self):
        """``collect_progress`` must stay testable (and importable) without a server.

        This is the AST form of the module docstring's claim: a module-level
        ``import streamlit`` would make the pure half depend on a Streamlit
        session, which is exactly the coupling ``link_design_assist`` avoids.
        """
        tree = ast.parse(Path(tp.__file__).read_text(encoding="utf-8"))
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "streamlit":
                found.append(node)
            elif isinstance(node, ast.Import) and any(
                alias.name == "streamlit" for alias in node.names
            ):
                found.append(node)

        assert found, "render_progress is expected to import streamlit"
        assert all(node not in tree.body for node in found), (
            "streamlit must be imported inside render_progress, not at module scope"
        )


class TestJournalArgument:
    def test_an_explicit_journal_overrides_the_file(self, tmp_path):
        out = _session(
            tmp_path, stages=STAGES,
            journal=[_entry("capture", "complete", seconds=1.0)],
        )

        view = tp.collect_progress(
            out,
            journal=[_entry("capture", "complete", seconds=1.0),
                     _entry("fit", "running", start=1.0)],
        )

        assert view["current"]["name"] == "fit"
        assert view["counts"]["complete"] == 1

    def test_the_in_memory_journal_also_works_before_it_is_written(self):
        view = tp.collect_progress(
            None, journal=[_entry("capture", "running")]
        )

        assert view["current"]["name"] == "capture"
        assert view["current"]["label"] == tp.STAGE_LABELS["capture"]
        assert view["stages"][0]["status"] == "running"


# ---------------------------------------------------------------------------
# renderer (streamlit stubbed to markdown / caption)
# ---------------------------------------------------------------------------


class _Stub:
    def __init__(self) -> None:
        self.markdown: list[str] = []
        self.caption: list[str] = []


@pytest.fixture
def stub(monkeypatch) -> _Stub:
    stub = _Stub()
    monkeypatch.setattr(st, "markdown", lambda text, *a, **k: stub.markdown.append(text))
    monkeypatch.setattr(st, "caption", lambda text, *a, **k: stub.caption.append(text))
    return stub


class TestRenderProgress:
    def _running_view(self, tmp_path: Path) -> dict[str, Any]:
        out = _session(
            tmp_path, stages=STAGES,
            journal=[
                _entry("capture", "complete", seconds=12.0),
                _entry("nominal_initialization", "complete", start=12.0, seconds=1.5),
                _entry("fit", "running", start=13.5),
            ],
            capture=CAPTURE,
            fit=(FIT_RESULT, FIT_PROVENANCE),
        )
        return tp.collect_progress(out)

    def test_the_first_line_names_the_running_stage_and_its_count(self, tmp_path, stub):
        tp.render_progress(self._running_view(tmp_path), running=True)

        assert stub.markdown[0] == (
            "**运行中** · 当前阶段：物理预测模型迁移 / 拟合（3/7）"
        )

    def test_the_counts_line_follows(self, tmp_path, stub):
        tp.render_progress(self._running_view(tmp_path), running=True)

        assert stub.caption[0] == "已完成 2 · 进行中 1 · 待运行 4"

    def test_one_row_per_started_stage_and_none_for_pending_ones(self, tmp_path, stub):
        tp.render_progress(self._running_view(tmp_path), running=True)
        body = stub.markdown[1]

        assert body.count("\n- ") == 2  # three rows, two of them after the first
        assert f"- **{tp.STAGE_LABELS['capture']}**：已完成" in body
        assert f"- **{tp.STAGE_LABELS['fit']}**：进行中" in body
        assert tp.STAGE_LABELS["summary"] not in body

    def test_the_row_carries_that_stage_key_metrics(self, tmp_path, stub):
        tp.render_progress(self._running_view(tmp_path), running=True)
        body = stub.markdown[1]

        assert "探针 3 条 · 划分 校准 1 / 训练 1 / 验证 1 · 可用 3 条" in body
        assert "曝光 0.02–0.85 ms · 峰值 200/255" in body
        assert "迭代 12 · 损失 0.0412 · 相关系数（验证） 0.950" in body

    def test_a_stage_without_artefacts_says_so_instead_of_zero(self, tmp_path, stub):
        out = _session(tmp_path, stages=STAGES, journal=[_entry("fit", "running")])

        tp.render_progress(tp.collect_progress(out), running=True)

        assert "**物理预测模型迁移 / 拟合**：进行中 —— 尚未产出指标" in stub.markdown[1]

    def test_block_details_are_nested_under_their_stage_row(self, tmp_path, stub):
        out = _session(
            tmp_path, stages=STAGES,
            journal=[_entry("CCD_anchored_blocks", "running")], anchored=ANCHORED,
        )

        tp.render_progress(tp.collect_progress(out), running=True)

        assert "\n  - 第 1 块（正方形）：接受 · 基线 0.0900" in stub.markdown[1]

    def test_the_session_directory_is_the_last_caption(self, tmp_path, stub):
        view = self._running_view(tmp_path)

        tp.render_progress(view, running=True)

        assert stub.caption[-1] == f"会话目录：{view['out_dir']}"

    def test_a_finished_run_says_all_stages_completed(self, tmp_path, stub):
        out = _session(
            tmp_path, stages=STAGES,
            journal=[_entry(name, "complete", seconds=1.0) for name in STAGES],
        )

        tp.render_progress(tp.collect_progress(out), running=True)

        assert stub.markdown[0] == "**已完成** · 全部 7 个阶段完成"

    def test_a_failed_run_names_the_failing_stage(self, tmp_path, stub):
        out = _session(
            tmp_path, stages=STAGES,
            journal=[_entry("capture", "complete", seconds=2.0),
                     _entry("nominal_initialization", "failed", start=2.0,
                            error="RuntimeError: Unusable exposure")],
        )

        tp.render_progress(tp.collect_progress(out), running=False)

        assert stub.markdown[0] == (
            "**已结束** · 第 2 个阶段失败：名义模型初始化（冷启动，未标定）"
        )
        assert "错误：RuntimeError: Unusable exposure" in stub.markdown[1]

    def test_between_stages_a_live_run_points_at_the_next_stage(self, tmp_path, stub):
        out = _session(
            tmp_path, stages=STAGES,
            journal=[_entry("capture", "complete", seconds=2.0)],
        )

        tp.render_progress(tp.collect_progress(out), running=True)

        assert stub.markdown[0] == (
            "**运行中** · 下一步：名义模型初始化（冷启动，未标定）（2/7）"
        )

    def test_a_run_that_ended_between_stages_says_where_it_stopped(self, tmp_path, stub):
        out = _session(
            tmp_path, stages=STAGES,
            journal=[_entry("capture", "complete", seconds=2.0)],
        )

        tp.render_progress(tp.collect_progress(out), running=False)

        assert stub.markdown[0] == (
            "**已停止** · 停在 当前光路采集（探针 + 逐帧自动曝光）（1/7）"
        )

    def test_an_empty_view_renders_one_caption(self, tmp_path, stub):
        tp.render_progress(tp.collect_progress(tmp_path / "nothing"), running=True)

        assert stub.markdown == []
        assert len(stub.caption) == 1
        assert "还没有阶段记录" in stub.caption[0]

    def test_a_skipped_stage_gets_a_row_too(self, tmp_path, stub):
        out = _session(
            tmp_path,
            stages=["reuse_capture", "fit"],
            journal=[_entry("fit", "running")],
        )

        tp.render_progress(tp.collect_progress(out), running=True)

        assert "**复用已有采集**：跳过 —— 尚未产出指标" in stub.markdown[1]
