"""Offline contracts for the real-machine optical-path reuse gate trial harness.

The trial runs the reuse gate on real hardware and judges whether its result is
*trustworthy* per docs/dev/next_hw_plan_optical_path_reuse_gate_trial_20261124.md §4,
which is independent of whether the gate itself said passed or failed. These tests
pin that judgement offline, with no device access.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
TRIAL = ROOT / "scripts" / "slm" / "optical_path_reuse_gate_trial.py"


def _load_trial():
    spec = importlib.util.spec_from_file_location("optical_path_reuse_gate_trial", TRIAL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


trial = _load_trial()
gate = trial.gate


def _passed_result(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema": gate.GATE_SCHEMA,
        "reuse_gate": "passed",
        "reasons": [],
        "reference": {"probe_json": "p.json", "date": "2026-11-24"},
        "criteria": {"same": True},
        "exposure_check": {
            "passed": True,
            "peak": 1.0,
            "peak_limit": 2.0,
            "saturated_pixels": 0,
        },
        "devices": {"drive_board_id": "b1"},
        "closed_normally": True,
        "remedy": "n/a",
    }
    base.update(overrides)
    return base


def _failed_result(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema": gate.GATE_SCHEMA,
        "reuse_gate": "failed",
        "reasons": ["驱动板 ID 不一致"],
        "reference": {"probe_json": "p.json", "date": "2026-11-24"},
        "criteria": {},
        "exposure_check": {"passed": False},
        "devices": {"drive_board_id": "b2"},
        "closed_normally": True,
        "remedy": "改跑完整 Step 3–4",
    }
    base.update(overrides)
    return base


def test_valid_passed_result() -> None:
    verdict = trial.validate_trial(_passed_result(), gate.EXIT_PASSED)
    assert verdict["trial"] == "valid"
    assert all(c["passed"] for c in verdict["checks"])


def test_valid_failed_result() -> None:
    verdict = trial.validate_trial(_failed_result(), gate.EXIT_FAILED)
    assert verdict["trial"] == "valid"
    assert all(c["passed"] for c in verdict["checks"])


def test_invalid_when_not_a_dict() -> None:
    verdict = trial.validate_trial(None, None)
    assert verdict["trial"] == "invalid"
    assert any(c["name"] == "result_shape" and not c["passed"] for c in verdict["checks"])


def test_invalid_when_exit_inconsistent_with_reuse() -> None:
    # reuse_gate=failed 却拿到退出码 0 → 不一致，不可信
    verdict = trial.validate_trial(_failed_result(), gate.EXIT_PASSED)
    assert verdict["trial"] == "invalid"
    assert any(
        c["name"] == "gate_exit_consistent" and not c["passed"] for c in verdict["checks"]
    )


def test_invalid_when_schema_wrong() -> None:
    verdict = trial.validate_trial(
        _passed_result(schema="optical_path_reuse_gate/9"), gate.EXIT_PASSED
    )
    assert verdict["trial"] == "invalid"
    assert any(c["name"] == "schema_ok" and not c["passed"] for c in verdict["checks"])


def test_invalid_passed_missing_exposure() -> None:
    verdict = trial.validate_trial(
        _passed_result(exposure_check={"passed": False}), gate.EXIT_PASSED
    )
    assert verdict["trial"] == "invalid"
    assert any(c["name"] == "passed_exposure" and not c["passed"] for c in verdict["checks"])


def test_invalid_passed_missing_driveboard() -> None:
    verdict = trial.validate_trial(_passed_result(devices={}), gate.EXIT_PASSED)
    assert verdict["trial"] == "invalid"
    assert any(
        c["name"] == "passed_driveboard" and not c["passed"] for c in verdict["checks"]
    )


def test_invalid_failed_empty_reasons() -> None:
    verdict = trial.validate_trial(_failed_result(reasons=[]), gate.EXIT_FAILED)
    assert verdict["trial"] == "invalid"
    assert any(c["name"] == "failed_reasons" and not c["passed"] for c in verdict["checks"])


def test_invalid_failed_empty_remedy() -> None:
    verdict = trial.validate_trial(_failed_result(remedy=""), gate.EXIT_FAILED)
    assert verdict["trial"] == "invalid"
    assert any(c["name"] == "failed_remedy" and not c["passed"] for c in verdict["checks"])


def test_gate_args_passthrough_all_fields() -> None:
    args = trial.build_parser().parse_args(
        [
            "--out-root", "o",
            "--run-dir", "r",
            "--slm-number", "2",
            "--wavelength", "532",
            "--cam-type", "daheng",
            "--cam-id", "3",
            "--wait-until-idle", "12.5",
            "--peak-fraction", "0.6",
            "--saturation-fraction", "0.9",
        ]
    )
    g = trial._build_gate_args(args)
    assert g.out_root == "o"
    assert g.run_dir == "r"
    assert g.slm_number == 2
    assert g.wavelength == 532
    assert g.cam_type == "daheng"
    assert g.cam_id == 3
    assert g.wait_until_idle == 12.5
    assert g.peak_fraction == 0.6
    assert g.saturation_fraction == 0.9


def test_parser_defaults_match_gate() -> None:
    t = trial.build_parser().parse_args([])
    g = gate.build_parser().parse_args([])
    for attr in (
        "out_root",
        "slm_number",
        "wavelength",
        "cam_type",
        "cam_id",
        "wait_until_idle",
        "peak_fraction",
        "saturation_fraction",
    ):
        assert getattr(t, attr) == getattr(g, attr), attr
    assert t.run_dir is None
    assert t.peak_fraction == 0.7
    assert t.saturation_fraction == 0.95


def _trial_args(tmp_path: Path) -> Any:
    return trial.build_parser().parse_args(
        [
            "--out-root", str(tmp_path),
            "--run-dir", str(tmp_path / "run"),
            "--slm-number", "1",
            "--wavelength", "1064",
            "--cam-type", "daheng",
            "--cam-id", "0",
        ]
    )


def test_run_trial_passed_writes_valid_json(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(gate, "run_reuse_gate", lambda args: _passed_result())
    code = trial.run_trial(_trial_args(tmp_path))
    assert code == trial.EXIT_TRIAL_VALID
    doc = json.loads((tmp_path / "run" / trial.TRIAL_FILE).read_text(encoding="utf-8"))
    assert doc["schema"] == trial.TRIAL_SCHEMA
    assert doc["trial"] == "valid"
    assert doc["gate"]["reuse_gate"] == "passed"
    assert doc["gate"]["exit_code"] == gate.EXIT_PASSED
    assert doc["gate"]["result_file"].endswith(gate.GATE_FILE)


def test_run_trial_failed_still_valid_json(tmp_path: Path, monkeypatch) -> None:
    # 门判 failed 也是可信结果：试跑仍有效，退出码 0（区别于门退出码 3）。
    monkeypatch.setattr(gate, "run_reuse_gate", lambda args: _failed_result())
    code = trial.run_trial(_trial_args(tmp_path))
    assert code == trial.EXIT_TRIAL_VALID
    doc = json.loads((tmp_path / "run" / trial.TRIAL_FILE).read_text(encoding="utf-8"))
    assert doc["trial"] == "valid"
    assert doc["gate"]["reuse_gate"] == "failed"
    assert doc["gate"]["exit_code"] == gate.EXIT_FAILED


def test_run_trial_invalid_when_gate_crashes(tmp_path: Path, monkeypatch) -> None:
    def boom(_args: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(gate, "run_reuse_gate", boom)
    code = trial.run_trial(_trial_args(tmp_path))
    assert code == trial.EXIT_TRIAL_INVALID
    doc = json.loads((tmp_path / "run" / trial.TRIAL_FILE).read_text(encoding="utf-8"))
    assert doc["trial"] == "invalid"
    assert doc["gate"]["exit_code"] is None
    assert any("boom" in r for r in doc["reasons"])
