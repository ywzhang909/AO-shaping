"""Issue #247 同窗口重复性检查的纯离线验证；只使用合成帧，不连接硬件。"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SCRIPT = importlib.import_module(
    "scripts.slm.issue247_reproducibility_check_20261124"
)
ANALYZER = SCRIPT._analyzer()
HELD_OUT_BASE_IDS = frozenset(
    base for base in SCRIPT.EVALUABLE_BASE_IDS if base.endswith("_p192")
)


def _sha(seed: int) -> str:
    return f"{seed:064x}"


def _pattern(
    base_id: str,
    round_index: int,
    angle: float | None,
    *,
    flat_shape: bool = False,
    role: str | None = "fit",
    edge_offset: tuple[float, float] | None = None,
) -> dict[str, Any]:
    round_id = f"r{round_index + 1}"
    entry: dict[str, Any] = {
        "id": f"{base_id}__{round_id}" if flat_shape else f"{round_id}__{base_id}",
        "phase_sha256": _sha(SCRIPT.EXPECTED_BASE_IDS.index(base_id) + 1),
        "frames": 4,
        "settle_s": 0.8,
        "kind": "flat" if base_id.startswith("flat_") else None,
        "angle_for_test": angle,
    }
    if edge_offset is not None:
        entry["edge_for_test"] = [float(edge_offset[0]), float(edge_offset[1])]
    if role is not None:
        entry["role"] = role
    if flat_shape:
        entry["base_pattern_id"] = base_id
        entry["repeat_index"] = round_index + 1
    else:
        entry["base_id"] = base_id
        entry["round"] = round_id
    return entry


def _make_capture(
    root: Path,
    *,
    angles_by_round: list[float | None] | None = None,
    duplicate_id: bool = False,
    flat_drift: bool = False,
    flat_shape: bool = False,
    missing_peak_id: str | None = "patch_x_minus_p192",
    plan_role: bool = True,
    edge_peaks: dict[str, list[tuple[float, float] | None]] | None = None,
) -> Path:
    capture_dir = root / "capture"
    frames_dir = capture_dir / "frames"
    frames_dir.mkdir(parents=True)
    angles = angles_by_round or [22.0, 22.5, 22.8]
    rounds: dict[str, list[dict[str, Any]]] = {}
    digests_by_pattern: dict[str, list[str]] = {}
    y, x = np.mgrid[:64, :64]
    image = (100.0 + 800.0 * np.exp(-((y - 32) ** 2 + (x - 32) ** 2) / 20)).astype(
        np.float32
    )
    for round_index, round_id in enumerate(SCRIPT.ROUND_IDS):
        base_order = list(SCRIPT.EXPECTED_BASE_IDS)
        if round_id == "r2":
            base_order.reverse()
        entries: list[dict[str, Any]] = []
        for base_id in base_order:
            angle = None if base_id == missing_peak_id else angles[round_index]
            entry = _pattern(
                base_id,
                round_index,
                angle,
                flat_shape=flat_shape,
                role=(
                    None
                    if not plan_role
                    else ("held_out" if base_id in HELD_OUT_BASE_IDS else "fit")
                ),
                edge_offset=(
                    None
                    if not edge_peaks or not edge_peaks.get(base_id)
                    else edge_peaks[base_id][round_index]
                ),
            )
            if duplicate_id and round_id == "r3" and base_id == "flat_post":
                entry["id"] = "flat_post__r2" if flat_shape else "r2__flat_post"
            entries.append(entry)
            entry_digests: list[str] = []
            for frame_index in range(4):
                frame = image.copy()
                if flat_drift and base_id == "flat_post":
                    frame = np.roll(frame, 4, axis=1)
                frame[0, 0] = 20 + round_index
                np.save(frames_dir / f"{entry['id']}_{frame_index:02d}.npy", frame)
                entry_digests.append(
                    hashlib.sha256(
                        np.ascontiguousarray(frame).tobytes()
                    ).hexdigest()
                )
            digests_by_pattern[entry["id"]] = entry_digests
        rounds[round_id] = entries

    if flat_shape:
        flat_entries = [entry for entries in rounds.values() for entry in entries]
        plan = {
            "patterns": flat_entries,
            "frames_per_pattern": {entry["id"]: 4 for entry in flat_entries},
            "requested_exposure_ms": 0.04,
            "settle_s": 0.8,
            "full_scale": 4095,
            "reproduction": {
                "prereg": (
                    "docs/dev/hw_window_20261124_issue247_reproducibility_prereg.md"
                ),
                "rounds": list(SCRIPT.ROUND_IDS),
                "round_order": ["frozen", "reversed", "frozen"],
                "base_pattern_count": 23,
                "n_unique_patterns": 69,
                "frames_per_pattern": 4,
                "n_frames_expected": 276,
                "id_pattern": "{base_pattern_id}__{round_id}",
                "phase_hash_identical_across_rounds": True,
                "single_slm_session": True,
            },
        }
    else:
        plan = {
            "rounds": rounds,
            "frames_per_pattern": 4,
            "exposure_ms": 0.04,
            "settle_s": 0.8,
            "full_scale": 4095,
        }
    summary = {
        "status": "ok",
        "n_frames": SCRIPT.EXPECTED_TOTAL_FRAMES,
        "saturation_count": 0,
        "exposure_ms": 0.04,
        "per_pattern": {
            entry["id"]: {
                "frames": 4,
                "digests": digests_by_pattern[entry["id"]],
            }
            for entries in rounds.values()
            for entry in entries
        },
    }
    released = {
        "status": "ok",
        "closed_normally": True,
        "slm_closed": True,
        "camera_closed": True,
        "lock_released": True,
        "error": None,
    }
    # Mirrors the real capture-side record: device identity and exposure self-check
    # only, no six-step precheck verdict and no camera-index field.
    optical_record = {
        "devices": {
            "slm": {
                "model": "Santec SLM-200",
                "slm_number": 1,
                "wavelength_nm": 1064,
            },
            "camera": {
                "model": "Daheng MER2-507-23GM",
                "serial": "TEST-CAM-0001",
                "bit_depth": 12,
                "full_scale": 4095,
                "exposure_used_ms": 0.04,
            },
        },
        "exposure_check": {"saturated": False, "settle_timeout": False},
        "occupancy": {"closed_normally": True},
        "missing": {"optical_path.config_id": "调用方未提供该字段"},
    }
    (capture_dir / "plan.json").write_text(
        json.dumps(plan), encoding="utf-8"
    )
    (capture_dir / "summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    (capture_dir / "released.json").write_text(
        json.dumps(released), encoding="utf-8"
    )
    (capture_dir / "optical_path_record.json").write_text(
        json.dumps(optical_record), encoding="utf-8"
    )
    return capture_dir


def _write_probe(
    probe_dir: Path,
    *,
    state: str = "ok",
    exit_code: int = 0,
    reasons: list[str] | None = None,
    write_file: bool = True,
) -> Path:
    probe_dir.mkdir(parents=True, exist_ok=True)
    if not write_file:
        return probe_dir
    if reasons is None:
        reasons = [
            "可用曝光窗口 0.04, 0.05, 0.06 ms (峰值高于噪声底 20.9 counts 且未饱和)"
        ]
    payload = {
        "schema": "optical-path-probe/v1",
        "verdict": {
            "state": state,
            "reasons": reasons,
            "advice": [],
            "exit_code": exit_code,
        },
    }
    (probe_dir / SCRIPT.PROBE_FILE_NAME).write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return probe_dir


def _check_capture(capture_dir: Path, baseline_dir: Path | None = None) -> dict[str, Any]:
    probe_dir = capture_dir.parent / "probe"
    if not (probe_dir / SCRIPT.PROBE_FILE_NAME).is_file():
        _write_probe(probe_dir)
    return SCRIPT.check_capture(capture_dir, baseline_dir, probe_dir)


@pytest.fixture(autouse=True)
def _stub_analyzer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub only expensive signal interpretation; retain real validation and records."""

    def discover_frames(frames_dir: Path, pattern_id: str) -> list[Path]:
        return sorted(frames_dir.glob(f"{pattern_id}_*.npy"))

    def analyze(capture: Any, _thresholds: dict[str, float]) -> dict[str, Any]:
        patterns: dict[str, dict[str, Any]] = {}
        flat_centroids: dict[str, dict[str, float]] = {}
        for spec, entry in zip(capture.catalog, capture.plan["patterns"], strict=True):
            mean = capture.means[spec.pattern_id]
            center = ANALYZER._argmax_center(mean)
            angle = entry.get("angle_for_test")
            is_flat = spec.pattern_id.startswith("flat_")
            if is_flat:
                flat_centroids[spec.pattern_id] = {
                    "row": float(center[0]),
                    "col": float(center[1]),
                }
            rad = math.radians(float(angle or 0.0))
            offset = entry.get("edge_for_test")
            if offset:
                dr, dc = float(offset[0]), float(offset[1])
                principal = {
                    "dr": dr,
                    "dc": dc,
                    "angle_deg": math.degrees(math.atan2(dr, dc)),
                    "magnitude": math.hypot(dr, dc),
                    "peak": 24.0,
                }
            else:
                principal = {
                    "dr": math.sin(rad) * 10.0,
                    "dc": math.cos(rad) * 10.0,
                    "angle_deg": float(angle or 0.0) + 70.0,
                    "magnitude": 10.0,
                    "peak": 50.0,
                }
            patterns[spec.pattern_id] = {
                "family": spec.family,
                "axis": spec.axis,
                "role": spec.role,
                "axis_angle_deg": None if angle is None else float(angle),
                "principal": principal,
                "zero_order": {"row": center[0], "col": center[1]},
                "peak_count": 0 if angle is None else 1,
                "peaks": [],
                "multi_peak": False,
                "counter_lobe_missing": False,
                "counter_lobe_weak": False,
                "background_counts": 1.0,
                "max_counts": float(mean.max()),
                "saturated": False,
            }
        gates = {
            "drift_gate": {"pass": True},
            "saturation_gate": {"pass": True},
        }
        attribution = {
            key: {"pass": False, "measured": 0.0}
            for key in SCRIPT.GATE_KEYS
        }
        return {
            "verdict": "not_assessable",
            "failed_gates": [],
            "gates": gates,
            "attribution_gates": attribution,
            "held_out_validation": {"axis_accuracy": None},
            "flat_reference": {
                "centroids": flat_centroids,
                "sigma_est_px": 2.0,
            },
            "patterns": patterns,
        }

    def measure_pattern(
        spec: Any,
        frame: np.ndarray,
        reference: tuple[float, float],
        _sigma: float,
        _stats: dict[str, Any],
    ) -> dict[str, Any]:
        round_marker = int(round(float(frame[0, 0]))) - 20
        entry_angle = [22.0, 22.5, 22.8][round_marker]
        if spec.pattern_id == "patch_x_minus_p192":
            angle: float | None = None
        else:
            angle = entry_angle
        rad = math.radians(float(angle or 0.0))
        principal = {
            "dr": math.sin(rad) * 10.0,
            "dc": math.cos(rad) * 10.0,
            # Deliberately divergent: repeatability must use axis_angle_deg.
            "angle_deg": float(angle or 0.0) + round_marker * 70.0,
            "magnitude": 10.0,
            "peak": 50.0,
        }
        return {
            "axis_angle_deg": angle,
            "principal": principal,
            "peak_count": 0 if angle is None else 1,
            "peaks": [],
            "lobes": [],
            "multi_peak": False,
            "counter_lobe_missing": False,
            "counter_lobe_weak": False,
            "background_counts": 1.0,
            "saturated": False,
        }

    monkeypatch.setattr(ANALYZER, "discover_frames", discover_frames)
    monkeypatch.setattr(ANALYZER, "analyze", analyze)
    monkeypatch.setattr(ANALYZER, "_measure_pattern", measure_pattern)


def test_check_capture_uses_axis_angle_and_preserves_missing_peak(tmp_path: Path) -> None:
    capture = _make_capture(tmp_path)

    result = _check_capture(capture)

    assert result["invariants"]["total_frames"] == 276
    assert result["invariants"]["flat_global_drift"]["max_drift_px"] <= 2.0
    repeatability = result["repeatability"]
    assert repeatability["axis_metric"] == "axis_angle_deg"
    assert repeatability["overall_pass"] is False
    assert repeatability["counts"]["not_assessable"] == 1
    patterns = {item["base_id"]: item for item in repeatability["patterns"]}
    assert patterns["grating_x_p128"]["round_max_diff_deg"] == pytest.approx(0.8)
    missing = patterns["patch_x_minus_p192"]
    assert missing["status"] == "not_assessable"
    assert missing["missing_rounds"] == {
        "r1": "missing_peak",
        "r2": "missing_peak",
        "r3": "missing_peak",
    }
    first_frame = patterns["grating_x_p128"]["per_round"][0]
    assert first_frame["principal_angle_deg"] != pytest.approx(
        first_frame["axis_angle_deg"]
    )
    assert first_frame["frame_angles_deg"] == [22.0, 22.0, 22.0, 22.0]


def test_principal_peak_on_frame_edge_is_excluded_as_not_assessable(
    tmp_path: Path,
) -> None:
    capture = _make_capture(
        tmp_path,
        missing_peak_id=None,
        edge_peaks={"grating_x_p128": [(-31.5, -31.0)] * 3},
    )

    result = _check_capture(capture)

    assert result["invariants"]["frame_shape"] == {"rows": 64, "cols": 64}
    rows = {item["base_id"]: item for item in result["repeatability"]["patterns"]}
    item = rows["grating_x_p128"]
    assert item["status"] == "not_assessable"
    assert item["assessable_rounds"] == []
    assert item["edge_clipped_rounds"] == ["r1", "r2", "r3"]
    assert item["round_max_diff_deg"] is None
    assert item["reason"] == "edge_clipped_rounds_excluded"
    for row in item["per_round"]:
        assert row["status"] == "not_assessable"
        # 原始值保留（审计用），但不作为门禁输入。
        assert row["axis_angle_deg"] is not None
        assert row["gate_axis_angle_deg"] is None
        assert "principal peak on frame edge" in item["missing_rounds"][row["round"]]
        assert row["edge_clipped"]["distance_to_edge_px"] < SCRIPT.EDGE_BAND_PX
    counts = result["repeatability"]["counts"]
    assert counts["edge_clipped_rounds"] == 3
    assert counts["assessable_rounds_total"] == (len(SCRIPT.EVALUABLE_BASE_IDS) - 1) * 3
    assert counts["reproduced"] == len(SCRIPT.EVALUABLE_BASE_IDS) - 1
    assert counts["not_assessable"] == 1
    assert counts["insufficient_evidence"] == 0
    assert result["repeatability"]["overall_pass"] is True
    # 非贴边图样不受影响。
    assert rows["tilt_x_plus"]["status"] == "reproduced"


def test_single_edge_round_is_excluded_but_pattern_still_reproduced(
    tmp_path: Path,
) -> None:
    capture = _make_capture(
        tmp_path,
        missing_peak_id=None,
        edge_peaks={"grating_x_p128": [None, (-31.5, -31.0), None]},
    )

    result = _check_capture(capture)

    item = next(
        row
        for row in result["repeatability"]["patterns"]
        if row["base_id"] == "grating_x_p128"
    )
    assert item["status"] == "reproduced"
    assert item["assessable_rounds"] == ["r1", "r3"]
    assert item["edge_clipped_rounds"] == ["r2"]
    assert set(item["missing_rounds"]) == {"r2"}
    assert "principal peak on frame edge" in item["missing_rounds"]["r2"]
    assert item["round_max_diff_deg"] == pytest.approx(0.8)
    assert result["repeatability"]["counts"]["edge_clipped_rounds"] == 1


def test_markdown_provenance_block_comes_from_registry(tmp_path: Path) -> None:
    capture = _make_capture(tmp_path, missing_peak_id=None)

    result = _check_capture(capture)
    markdown = SCRIPT.render_markdown(result)
    block = SCRIPT.provenance_block_for(SCRIPT.REPORT_KEY)
    if block:
        assert block in markdown
        assert markdown.count("<!-- provenance:start -->") == 1
    else:
        assert "<!-- provenance:start -->" not in markdown
    # 数据/分析器/预注册/生成时间行留在注册表块外，照常显示。
    assert "- **数据**:" in markdown
    assert "- **分析器**:" in markdown
    assert "- **预注册**:" in markdown
    assert "- **生成时间**:" in markdown


def test_report_key_matches_default_output_path() -> None:
    series = SCRIPT.DEFAULT_MD_NAME[: -len(".md")]
    assert SCRIPT.REPORT_KEY == (
        f"report/shape_trials/{series}/{SCRIPT.DEFAULT_MD_NAME}"
    )


def test_accepts_flat_suffix_encoding_from_capture(tmp_path: Path) -> None:
    capture = _make_capture(tmp_path, flat_shape=True)

    result = _check_capture(capture)

    assert result["invariants"]["rounds"] == list(SCRIPT.ROUND_IDS)
    assert result["invariants"]["order"]["r1"] == list(SCRIPT.EXPECTED_BASE_IDS)
    assert result["invariants"]["order"]["r2"] == list(
        reversed(SCRIPT.EXPECTED_BASE_IDS)
    )
    assert [len(row["analysis"]["patterns"]) for row in result["rounds"]] == [
        23,
        23,
        23,
    ]
    assert result["repeatability"]["n_evaluable_patterns"] == 20


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"repeat_index": 2}, "repeat_index=2"),
        ({"base_pattern_id": "grating_y_p128"}, "base_pattern_id"),
    ],
)
def test_flat_encoding_cross_checks_fail_loudly(
    tmp_path: Path, mutation: dict[str, Any], message: str
) -> None:
    capture = _make_capture(tmp_path, flat_shape=True)
    plan_path = capture / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["patterns"][1].update(mutation)
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(SCRIPT.CheckError, match=message):
        _check_capture(capture)


def test_rejects_duplicate_round_pattern_id(tmp_path: Path) -> None:
    capture = _make_capture(tmp_path, duplicate_id=True)

    with pytest.raises(SCRIPT.CheckError, match="pattern id 重复"):
        _check_capture(capture)


def test_rejects_phase_digest_mismatch(tmp_path: Path) -> None:
    capture = _make_capture(tmp_path)
    plan_path = capture / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["rounds"]["r2"][0]["phase_sha256"] = _sha(999)
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(SCRIPT.CheckError, match="相位摘要三轮不一致"):
        _check_capture(capture)


def test_rejects_wrong_exposure_and_saturation(tmp_path: Path) -> None:
    capture = _make_capture(tmp_path)
    summary_path = capture / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["exposure_ms"] = 0.05
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(SCRIPT.CheckError, match="固定曝光"):
        _check_capture(capture)

    # A 3x3 plateau at 0.9 full scale must be rejected.
    summary["exposure_ms"] = 0.04
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    frame_path = next((capture / "frames").glob("*.npy"))
    frame = np.load(frame_path)
    frame[10:13, 10:13] = 4095
    np.save(frame_path, frame)
    with pytest.raises(SCRIPT.CheckError, match="判定饱和"):
        _check_capture(capture)


def test_rejects_bad_release_and_flat_drift(tmp_path: Path) -> None:
    drift_capture = _make_capture(tmp_path / "drift", flat_drift=True)
    with pytest.raises(SCRIPT.CheckError, match="跨轮 flat 质心最大漂移"):
        _check_capture(drift_capture)

    capture = _make_capture(tmp_path / "release")
    release_path = capture / "released.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    release["lock_released"] = False
    release_path.write_text(json.dumps(release), encoding="utf-8")
    with pytest.raises(SCRIPT.CheckError, match="lock_released"):
        _check_capture(capture)


def test_baseline_requires_analysis_artifact(tmp_path: Path) -> None:
    capture = _make_capture(tmp_path)
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()

    with pytest.raises(SCRIPT.CheckError, match="缺少 plan.json"):
        _check_capture(capture, baseline_dir)


def _write_baseline(baseline_dir: Path, *, bad_phase: bool = False) -> None:
    baseline_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "plan.json": {
            "patterns": [
                {
                    "id": base_id,
                    "phase_sha256": _sha(999 if bad_phase else index + 1),
                }
                for index, base_id in enumerate(SCRIPT.EXPECTED_BASE_IDS)
            ]
        },
        "summary.json": {"status": "ok", "n_frames": 92, "saturation_count": 0},
        "released.json": {
            "status": "ok",
            "closed_normally": True,
            "slm_closed": True,
            "camera_closed": True,
            "lock_released": True,
        },
        "waveform_probe_analysis.json": {
            "patterns": {
                base_id: {"axis_angle_deg": 22.0}
                for base_id in SCRIPT.EXPECTED_BASE_IDS
            },
            "gates": {
                "drift_gate": {"pass": True},
                "saturation_gate": {"pass": True},
            },
            "attribution_gates": {
                key: {"pass": False} for key in SCRIPT.GATE_KEYS
            },
            "held_out_validation": {"axis_accuracy": 0.0},
            "verdict": "map_not_reproduced",
        },
    }
    for name, value in artifacts.items():
        (baseline_dir / name).write_text(json.dumps(value), encoding="utf-8")


def test_valid_baseline_analysis_is_validated(tmp_path: Path) -> None:
    capture = _make_capture(tmp_path / "capture")
    baseline_dir = tmp_path / "baseline"
    _write_baseline(baseline_dir)

    result = _check_capture(capture, baseline_dir)

    assert result["baseline"]["supplied"] is True
    assert result["baseline"]["phase_checked"] == 23
    assert result["repeatability"]["patterns"][0]["baseline"][
        "round_vs_baseline_diff_deg"
    ]
    missing_gate = result["repeatability"]["patterns"][0]["baseline"]
    assert all(
        isinstance(value, float)
        for value in missing_gate["round_vs_baseline_diff_deg"].values()
    )


def test_baseline_phase_mismatch_is_rejected(tmp_path: Path) -> None:
    capture = _make_capture(tmp_path / "capture")
    baseline_dir = tmp_path / "baseline"
    _write_baseline(baseline_dir, bad_phase=True)

    with pytest.raises(SCRIPT.CheckError, match="与基线 attribution 不一致"):
        _check_capture(capture, baseline_dir)


def test_full_gate_run_requires_probe_dir(tmp_path: Path) -> None:
    capture = _make_capture(tmp_path)

    with pytest.raises(SCRIPT.CheckError, match="--probe-dir"):
        SCRIPT.check_capture(capture)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"state": "fail"}, "verdict.state='fail'"),
        ({"exit_code": 1}, "exit_code=1"),
        ({"reasons": ["可用曝光窗口 0.05, 0.06 ms"]}, "不在预检可用曝光窗口"),
        ({"reasons": ["预检完成，但未列出曝光窗口"]}, "找不到可解析"),
        ({"reasons": ["可用曝光窗口 0.04, 0.4.5 ms"]}, "无法保守解析"),
    ],
)
def test_probe_verdict_gates(
    tmp_path: Path, mutation: dict[str, Any], message: str
) -> None:
    capture = _make_capture(tmp_path)
    probe_dir = tmp_path / "probe"
    _write_probe(probe_dir, **mutation)

    with pytest.raises(SCRIPT.CheckError, match=message):
        SCRIPT.check_capture(capture, None, probe_dir)


def test_missing_probe_file_is_rejected(tmp_path: Path) -> None:
    capture = _make_capture(tmp_path)
    probe_dir = _write_probe(tmp_path / "probe", write_file=False)

    with pytest.raises(SCRIPT.CheckError, match="缺少文件"):
        SCRIPT.check_capture(capture, None, probe_dir)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("saturated", True, "exposure_check.saturated"),
        ("settle_timeout", True, "exposure_check.settle_timeout"),
    ],
)
def test_capture_optical_record_exposure_self_check(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    capture = _make_capture(tmp_path)
    path = capture / "optical_path_record.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["exposure_check"][field] = value
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(SCRIPT.CheckError, match=message):
        _check_capture(capture)


@pytest.mark.parametrize(
    ("target", "key", "value", "message"),
    [
        ("slm", "model", "SLM-100", "slm.model"),
        ("camera", "bit_depth", 8, "camera.bit_depth"),
        ("camera", "serial", "", "camera.serial"),
        ("camera", "exposure_used_ms", 0.05, "exposure_used_ms"),
    ],
)
def test_capture_optical_record_device_identity(
    tmp_path: Path,
    target: str,
    key: str,
    value: object,
    message: str,
) -> None:
    capture = _make_capture(tmp_path)
    path = capture / "optical_path_record.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["devices"][target][key] = value
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(SCRIPT.CheckError, match=message):
        _check_capture(capture)


def _strip_settle(capture: Path) -> dict[str, Any]:
    plan_path = capture / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    for item in plan.get("patterns", []):
        item.pop("settle_s", None)
    plan.pop("settle_s", None)
    plan.pop("capture_settings", None)
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return plan


def test_settle_less_capture_is_rejected(tmp_path: Path) -> None:
    """Real 2026-11-24 capture shape (no settle record) must keep failing."""
    capture = _make_capture(tmp_path, flat_shape=True)
    _strip_settle(capture)

    with pytest.raises(SCRIPT.CheckError, match="缺少 settle 记录"):
        _check_capture(capture)


def test_capture_settings_settle_s_is_read(tmp_path: Path) -> None:
    capture = _make_capture(tmp_path, flat_shape=True)
    plan = _strip_settle(capture)
    plan["capture_settings"] = {
        "settle_s": 0.8,
        "exposure_ms": 0.04,
        "saturation_fraction": 0.9,
        "wait_until_idle_s": 1800,
        "plan_name": "issue247_reproducibility",
        "slm_number": 1,
        "wavelength_nm": 1064,
        "cam_id": 0,
    }
    (capture / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    result = _check_capture(capture)

    sources = result["invariants"]["fixed_parameters"]["settle_sources"]
    assert "plan.capture_settings.settle_s" in sources
    assert result["invariants"]["fixed_parameters"]["settle_s"] == 0.8


def test_frame_digest_uses_capture_array_canonicalization(tmp_path: Path) -> None:
    """Digests must hash array bytes (capture lane `digest(array)`), not file bytes."""
    capture = _make_capture(tmp_path)
    frame = next((capture / "frames").glob("*.npy"))
    array_digest = hashlib.sha256(
        np.ascontiguousarray(np.load(frame)).tobytes()
    ).hexdigest()
    file_digest = hashlib.sha256(frame.read_bytes()).hexdigest()
    assert array_digest != file_digest  # .npy container bytes differ from payload
    assert _check_capture(capture)["invariants"]["total_frames"] == 276


def test_frame_digest_mismatch_is_rejected(tmp_path: Path) -> None:
    capture = _make_capture(tmp_path)
    path = capture / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["per_pattern"]["r1__flat_pre"]["digests"][0] = _sha(0)
    path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(SCRIPT.CheckError, match="与实际帧不一致"):
        _check_capture(capture)


def test_capture_settings_settle_s_must_be_0_8(tmp_path: Path) -> None:
    capture = _make_capture(tmp_path, flat_shape=True)
    plan = _strip_settle(capture)
    plan["capture_settings"] = {"settle_s": 0.5, "exposure_ms": 0.04}
    (capture / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(SCRIPT.CheckError, match="capture_settings.settle_s=0.5"):
        _check_capture(capture)


def test_report_states_evidence_sources(tmp_path: Path) -> None:
    capture = _make_capture(tmp_path)
    probe_dir = _write_probe(tmp_path / "probe")

    result = SCRIPT.check_capture(capture, None, probe_dir)

    assert result["probe_dir"] == SCRIPT._rel(probe_dir)
    assert result["invariants"]["precheck_probe"]["verdict_state"] == "ok"
    assert result["invariants"]["precheck_probe"]["usable_exposure_window_ms"] == [
        0.04,
        0.05,
        0.06,
    ]
    sources = result["evidence_sources"]
    assert "optical_path_probe.json" in sources["probe_run"]
    assert "六步预检 verdict 不在其中" in sources["capture_optical_record"]
    markdown = SCRIPT.render_markdown(result)
    assert "`probe_run`" in markdown
    assert "`capture_optical_record`" in markdown


def test_default_output_names_follow_shape_trials_convention() -> None:
    assert SCRIPT.DEFAULT_JSON_NAME == "issue247_reproducibility_20261124.json"
    assert SCRIPT.DEFAULT_MD_NAME == "issue247_reproducibility_20261124.md"


def test_plan_role_populates_repeatability(tmp_path: Path) -> None:
    capture = _make_capture(tmp_path)

    result = _check_capture(capture)

    rows = result["repeatability"]["patterns"]
    assert len(rows) == len(SCRIPT.EVALUABLE_BASE_IDS)
    roles = {row["base_id"]: row["role"] for row in rows}
    assert all(role is not None for role in roles.values())
    for base_id, role in roles.items():
        assert role == ("held_out" if base_id in HELD_OUT_BASE_IDS else "fit")
    markdown = SCRIPT.render_markdown(result)
    assert "| base pattern | 族 | 角色 |" in markdown
    assert "| grating_x_p192 |" in markdown and "held_out" in markdown


def test_missing_plan_role_stays_null_and_is_hidden_in_markdown(
    tmp_path: Path,
) -> None:
    capture = _make_capture(tmp_path, plan_role=False)

    result = _check_capture(capture)

    assert all(row["role"] is None for row in result["repeatability"]["patterns"])
    markdown = SCRIPT.render_markdown(result)
    assert "| 角色 |" not in markdown


def test_conflicting_plan_roles_do_not_pick_one(tmp_path: Path) -> None:
    capture = _make_capture(tmp_path, missing_peak_id=None)
    plan_path = capture / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    for entry in plan["rounds"]["r2"]:
        base = entry.get("base_pattern_id") or entry.get("base_id")
        if base == "grating_x_p128":
            entry["role"] = "held_out"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    result = _check_capture(capture)

    role = next(
        row["role"]
        for row in result["repeatability"]["patterns"]
        if row["base_id"] == "grating_x_p128"
    )
    assert role is None
    assert result["repeatability"]["overall_pass"] is True


def test_repeatability_needs_at_least_two_rounds_and_not_principal() -> None:
    round_analyses: dict[str, dict[str, Any]] = {}
    frame_metrics: dict[str, dict[str, Any]] = {}
    for round_id, axis_angle in zip(SCRIPT.ROUND_IDS, (10.0, 11.0, None), strict=True):
        patterns = {
            base_id: {
                "axis_angle_deg": axis_angle,
                "principal": {"angle_deg": 120.0 + len(round_analyses)},
                "peak_count": 0 if axis_angle is None else 1,
                "family": "grating",
                "axis": "x",
                "role": "fit",
                "background_counts": 1.0,
                "max_counts": 10.0,
            }
            for base_id in SCRIPT.EVALUABLE_BASE_IDS
        }
        round_analyses[round_id] = {"patterns": patterns}
        frame_metrics[round_id] = {}

    result = SCRIPT._repeatability(round_analyses, frame_metrics, None)

    item = result["patterns"][0]
    assert item["status"] == "reproduced"
    assert item["assessable_rounds"] == ["r1", "r2"]
    assert item["round_max_diff_deg"] == pytest.approx(1.0)


def test_rejects_missing_or_extra_frames(tmp_path: Path) -> None:
    capture = _make_capture(tmp_path / "missing")
    first_frame = next((capture / "frames").glob("*.npy"))
    first_frame.unlink()
    with pytest.raises(SCRIPT.CheckError, match="275 个 .npy，预注册要求 276"):
        _check_capture(capture)

    capture = _make_capture(tmp_path / "extra")
    for path in list((capture / "frames").glob("*.npy"))[:2]:
        np.save(capture / "frames" / f"000_extra_{path.stem}.npy", np.load(path))
    with pytest.raises(SCRIPT.CheckError, match="276"):
        _check_capture(capture)


def test_repeatability_failure_when_axis_diff_exceeds_two_degrees() -> None:
    round_analyses: dict[str, dict[str, Any]] = {}
    frame_metrics: dict[str, dict[str, Any]] = {}
    for round_id, axis_angle in zip(SCRIPT.ROUND_IDS, (10.0, 15.0, 10.0), strict=True):
        patterns = {
            base_id: {
                "axis_angle_deg": axis_angle,
                "principal": {"angle_deg": 0.0},
                "peak_count": 1,
                "family": "grating",
                "axis": "x",
                "role": "fit",
                "background_counts": 1.0,
                "max_counts": 10.0,
            }
            for base_id in SCRIPT.EVALUABLE_BASE_IDS
        }
        round_analyses[round_id] = {"patterns": patterns}
        frame_metrics[round_id] = {}

    result = SCRIPT._repeatability(round_analyses, frame_metrics, None)

    assert result["overall_pass"] is False
    assert result["patterns"][0]["status"] == "not_reproduced"
    assert result["patterns"][0]["round_max_diff_deg"] == pytest.approx(5.0)


def test_cli_refuses_existing_output_without_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / SCRIPT.DEFAULT_JSON_NAME).write_text("{}", encoding="utf-8")

    code = SCRIPT.main(
        [
            "--dir",
            str(tmp_path / "missing"),
            "--probe-dir",
            str(tmp_path / "probe"),
            "--out-dir",
            str(output_dir),
        ]
    )

    assert code == 4


def test_cli_writes_json_and_markdown_with_probe(tmp_path: Path) -> None:
    capture = _make_capture(tmp_path, missing_peak_id=None)
    probe_dir = _write_probe(tmp_path / "probe")
    out_dir = tmp_path / "out"

    code = SCRIPT.main(
        [
            "--dir",
            str(capture),
            "--probe-dir",
            str(probe_dir),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert code == 0
    payload = json.loads(
        (out_dir / SCRIPT.DEFAULT_JSON_NAME).read_text(encoding="utf-8")
    )
    assert payload["probe_dir"] == SCRIPT._rel(probe_dir)
    assert payload["invariants"]["precheck_probe"]["verdict_state"] == "ok"
    markdown = (out_dir / SCRIPT.DEFAULT_MD_NAME).read_text(encoding="utf-8")
    assert "六步预检" in markdown
