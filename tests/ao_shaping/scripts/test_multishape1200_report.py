"""The multishape1200 report generator reads a session directory into a report.

The generator is the only place that decides what the report claims, so the
numbers it prints have to come from the session files -- in particular the
*adopted* initial phase, which is recorded in the journal (the bake-off
directory names are alphabetical and would otherwise pick the wrong one).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.slm.generate_multishape1200_report import (
    _session_facts,
    _transport_flow_facts,
    build_report,
)


def _capture(directory: Path, label: str, cv: float, *, exposure=50.0, bit_depth=12) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    name = f"capture_{len(list(directory.glob('capture_*.npz'))):04d}.npz"
    import numpy as np

    np.savez_compressed(directory / name, raw_frames=np.zeros((2, 40, 40), dtype=np.uint8),
                        camera_xy=np.stack(np.meshgrid(np.arange(5, 15), np.arange(5, 15)),
                                           axis=-1).astype(float),
                        target=np.ones((10, 10)), core=np.ones((10, 10), dtype=bool))
    return dict(label=label, file=name, metrics=dict(cv=cv, mean=cv, ee_roi=0.9, target_peak=1.0),
                exposure_ms=exposure, bit_depth=bit_depth, core_peak=1.0, saturation_pixels=0,
                timestamp_utc="2026-10-27T00:00:00+00:00")


def _session(tmp_path: Path) -> Path:
    session = tmp_path / "session"
    (session / "seed_bakeoff" / "square_gs").mkdir(parents=True)
    (session / "seed_bakeoff" / "square_backprojection").mkdir(parents=True)
    for kind, cv in (("gs", 0.65), ("backprojection", 1.29)):
        rows = [_capture(session / "seed_bakeoff" / f"square_{kind}", "square_seed", cv)]
        (session / "seed_bakeoff" / f"square_{kind}" / "manifest.json").write_text(
            json.dumps(rows), encoding="utf-8")
    (session / "journal.json").write_text(json.dumps([
        dict(stage="seed_bakeoff", status="complete", shape="square",
             scores={"gs": 0.65, "backprojection": 1.29}, chosen="gs"),
    ]), encoding="utf-8")
    stage = session / "shape" / "square" / "mono12_cv"
    rows = [_capture(stage, "square_seed", 0.9), _capture(stage, "square_0_A1", 0.88),
            _capture(stage, "square_0_B", 0.71), _capture(stage, "square_final", 0.43)]
    (stage / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")
    (stage / "summary.json").write_text(json.dumps([
        dict(shape="square", block=0, accepted=True, acceptance="sequential_ccd",
             accept_metric="cv", baseline=0.88, candidate=0.71),
    ]), encoding="utf-8")
    replay = session / "common_replay"
    replay.mkdir(parents=True)
    (replay / "manifest.json").write_text(json.dumps([
        dict(label="square_seed", exposure_ms=50.0, core_peak=12.0,
             metrics=dict(cv=0.9, mean=0.9, ee_roi=0.9, target_peak=1.0)),
    ]), encoding="utf-8")
    return session


def test_the_report_states_the_measured_seed_cv_and_the_adopted_seed(tmp_path):
    facts = _session_facts(_session(tmp_path))
    report = build_report(facts)
    assert "0.6500" in report          # the measured GS seed CV
    assert "1.2900" in report          # the measured back-projection seed CV
    # The adopted seed comes from the journal, not from the directory order.
    assert "GS (模型 80 次迭代)" in report
    assert "反投影 (非 GS, 形状通用) |" in report


def test_the_report_carries_the_measured_trajectory_and_acceptance(tmp_path):
    facts = _session_facts(_session(tmp_path))
    report = build_report(facts)
    assert "0.430" in report           # the final measured core CV
    assert "1/1" in report              # the accepted-block count
    assert "共同重放" in report


def test_a_missing_transport_flow_is_simply_absent(tmp_path):
    facts = _session_facts(_session(tmp_path))
    assert facts["transport_flow"] is None
    report = build_report(facts)
    assert "传输」初相位版本" not in report


def test_the_transport_flow_is_read_when_it_exists(tmp_path):
    session = _session(tmp_path)
    stage = session / "shape" / "square" / "transport_cv"
    rows = [_capture(stage, "square_seed", 1.3), _capture(stage, "square_final", 0.82)]
    (stage / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")
    flow = _transport_flow_facts(session)
    assert flow is not None
    assert flow["stages"]["transport_cv"]["final_cv"] == pytest.approx(0.82)
