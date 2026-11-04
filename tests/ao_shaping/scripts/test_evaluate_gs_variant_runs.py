import json
import tempfile
import csv
import pickle
import numpy as np
from pathlib import Path
import pytest

from scripts.slm.evaluate_gs_variant_runs import summarize_run, _aggregate_grouped

# Helper to create dummy run data
def _create_dummy_run(run_dir: Path, gs_method: str, stages: list[str], rows: list[dict]) -> Path:
    # Create CSV
    csv_path = run_dir / "slm_gs_refine_dummy.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    # Create PKL with dummy frames
    pkl_path = run_dir / "slm_gs_refine_dummy.pkl"
    frames = []
    for i, row in enumerate(rows):
        frames.append({
            "_img": np.random.rand(1944, 2592)  # random data
        })
    with pkl_path.open("wb") as f:
        pickle.dump(frames, f)
    # Create JSON config
    config_path = run_dir / "slm_gs_refine_dummy.json"
    config = {
        "gs_method": gs_method,
        "exposure_time_ms": 50.0,
        "target_side": 100,
        "seed": 123,
        "cy": 690.0,
        "cx": 960.0,
    }
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f)
    return run_dir


def test_summarize_run_basic():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run1"
        run_dir.mkdir()
        stages = ["stage1", "stage2"]
        rows = [
            {
                "stage": "stage1",
                "timestamp_utc": "2026-01-01T00:00:00",
                "score": "0.5",
                "pib": "0.4",
                "cv": "0.6",
                "target_side": "100",
                "cy": "690",
                "cx": "960",
            },
            {
                "stage": "stage2",
                "timestamp_utc": "2026-01-01T00:00:01",
                "score": "0.55",
                "pib": "0.45",
                "cv": "0.5",
                "target_side": "100",
                "cy": "690",
                "cx": "960",
            },
        ]
        _create_dummy_run(run_dir, "plain", stages, rows)
        result = summarize_run(run_dir)
        assert result["run_dir"] == str(run_dir)
        assert result["gs_method"] == "plain"
        assert len(result["measurements"]) == 2
        m1 = result["measurements"][0]
        assert m1["stage"] == "stage1"
        assert m1["score"] == 0.5
        assert m1["pib"] == 0.4
        assert m1["cv"] == 0.6


def test_aggregate_grouped_basic():
    summary = [
        {
            "run_dir": "/tmp/run1",
            "gs_method": "plain",
            "measurements": [
                {"stage": "stage1", "score": 0.5, "pib": 0.4, "cv": 0.6, "shape_iou": 0.08, "frame_peak": 100.0},
                {"stage": "stage2", "score": 0.55, "pib": 0.45, "cv": 0.5, "shape_iou": 0.07, "frame_peak": 102.0},
            ],
        },
        {
            "run_dir": "/tmp/run2",
            "gs_method": "plain",
            "measurements": [
                {"stage": "stage1", "score": 0.55, "pib": 0.45, "cv": 0.55, "shape_iou": 0.09, "frame_peak": 101.0},
                {"stage": "stage2", "score": 0.6, "pib": 0.5, "cv": 0.48, "shape_iou": 0.08, "frame_peak": 103.0},
            ],
        },
    ]
    grouped = _aggregate_grouped(summary)
    # Expect 2 groups (plain_stage1, plain_stage2)
    assert len(grouped) == 2
    # Sort by stage then method
    grouped = sorted(grouped, key=lambda g: (g["stage"], g["gs_method"]))
    assert grouped[0]["gs_method"] == "plain"
    assert grouped[0]["stage"] == "stage1"
    assert grouped[0]["n"] == 2
    # Check metrics have mean and std (n>1)
    assert grouped[0]["metrics"]["score"]["mean"] == pytest.approx((0.5 + 0.55) / 2)
    assert grouped[0]["metrics"]["score"]["std"] is not None
    # Check std is sample std (ddof=1)
    expected_std = np.std([0.5, 0.55], ddof=1)
    assert grouped[0]["metrics"]["score"]["std"] == pytest.approx(expected_std)


def test_aggregate_grouped_n_one():
    summary = [
        {
            "run_dir": "/tmp/run1",
            "gs_method": "weighted",
            "measurements": [
                {"stage": "stage1", "score": 0.5, "pib": 0.4, "cv": 0.6, "shape_iou": 0.08, "frame_peak": 100.0},
            ],
        },
    ]
    grouped = _aggregate_grouped(summary)
    assert len(grouped) == 1
    assert grouped[0]["n"] == 1
    # std should be None for n == 1
    assert grouped[0]["metrics"]["score"]["std"] is None


def test_aggregate_grouped_mixed_methods():
    summary = [
        {
            "run_dir": "/tmp/run1",
            "gs_method": "plain",
            "measurements": [
                {"stage": "stage1", "score": 0.5, "pib": 0.4, "cv": 0.6, "shape_iou": 0.08, "frame_peak": 100.0},
            ],
        },
        {
            "run_dir": "/tmp/run2",
            "gs_method": "weighted",
            "measurements": [
                {"stage": "stage1", "score": 0.45, "pib": 0.38, "cv": 0.7, "shape_iou": 0.07, "frame_peak": 99.0},
            ],
        },
    ]
    grouped = _aggregate_grouped(summary)
    # Expect 2 groups: plain_stage1, weighted_stage1
    assert len(grouped) == 2
    # Check both are present
    methods = {g["gs_method"] for g in grouped}
    stages = {g["stage"] for g in grouped}
    assert "plain" in methods
    assert "weighted" in methods
    assert "stage1" in stages
    # Check n = 1 for both since different methods
    for g in grouped:
        assert g["n"] == 1
        assert g["metrics"]["score"]["std"] is None
