"""Offline contract tests for ``scripts/slm/summarize_optics_change_blocks.py``.

The summarizer turns one closed-loop D-block collection into a referenceable
table/curve/JSON. Everything here is synthetic: two fake campaign roots mirror
the two real layouts,

* ``stage_d``-style — old optics path, no ``rounds.json``, rounds live in
  ``run_block{N}/`` next to a ``chain.json``;
* ``stage_d1000``-style — new optics path, ``rounds.json`` with model
  side data and ``measured.run_dir`` pointing at a *different* stage dir
  (``stage_r/run_round{N}/``), some data still on the way (missing files).

All per-round numbers are hand-computed below and asserted verbatim, so a
drift in the baseline/candidate/delta/drift/replay math fails the suite.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "slm" / "summarize_optics_change_blocks.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("summarize_optics_change_blocks", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_module()

NA = "—"


def _run_json(cv_by_label: dict[str, list[float]], *, accepted: bool | None = None) -> dict:
    """Build a synthetic ``stage_d_block1.json`` with 2 frames per present label."""
    order = ["A1", "B1", "B2", "A2"]
    records = []
    position = 0
    for label in order:
        values = cv_by_label.get(label)
        if values is None:
            continue
        for frame, cv in enumerate(values):
            records.append({"position": position, "label": label, "frame": frame, "cv_core": cv})
        position += 1
    data: dict = {
        "order": order,
        "exposure_ms": 4.0,
        "frozen_mask": {"centre_yx": [960, 600], "side_px": 60, "box_px": 3600, "core_px": 3025},
        "records": records,
    }
    if accepted is not None:
        data["accepted"] = accepted
    return data


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_np_placeholder(path: Path) -> None:
    """The summarizer never reads the .npy frames; an empty file proves it doesn't either."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _build_stage_d_fixture(base: Path) -> Path:
    """Old-style layout: run_block{N} + chain.json, no rounds.json.

    Hand math (noise floor 3σ = 0.1):
      round 1: A1 mean 1.1, A2 mean 1.5, B1 mean 2.1, B2 mean 2.5
        baseline 1.3, candidate 2.3, Δ +1.0, Δ÷3σ +10.0, drift +0.4, replay +0.4
      round 2: no JSON at all → skipped, every cell —
      round 3: only A1 (mean 0.55) + A2 (mean 0.75) → baseline 0.65, drift +0.2,
        candidate/Δ/replay are — (not 0)
    """
    run_root = base / "run_stage_d"
    _write(
        run_root / "stage_d" / "run_block1" / "stage_d_block1.json",
        _run_json({"A1": [1.0, 1.2], "A2": [1.4, 1.6], "B1": [2.0, 2.2], "B2": [2.4, 2.6]}, accepted=True),
    )
    (run_root / "stage_d" / "run_block2").mkdir(parents=True, exist_ok=True)  # empty: JSON never landed
    _write(
        run_root / "stage_d" / "run_block3" / "stage_d_block1.json",
        _run_json({"A1": [0.5, 0.6], "A2": [0.7, 0.8]}, accepted=False),
    )
    _write(
        run_root / "stage_d" / "chain.json",
        {
            "blocks": [
                {"block": 1, "cap": 0.1, "delta": 1.0, "accepted": True, "run_dir": "stage_d/run_block1"},
                {"block": 2, "cap": 0.1, "delta": None, "accepted": False, "run_dir": "stage_d/run_block2"},
                {"block": 3, "cap": 0.1, "delta": None, "accepted": False, "run_dir": "stage_d/run_block3"},
            ],
            "cap": 0.1,
        },
    )
    _write(base / "noise_60.json", {"three_sigma_resolution": 0.1, "side_px": 60, "exposure_ms": 4.0})
    return run_root


def _build_stage_d1000_fixture(base: Path) -> Path:
    """New-style layout: rounds.json (+model side), measured.run_dir → stage_r.

    Hand math (noise floor 3σ = 0.0568):
      round 1 (per-frame JSON in stage_r/run_round1):
        A1 mean 5.11, A2 mean 5.14, B1 mean 5.01, B2 mean 5.04
        → baseline 5.125, candidate 5.025, Δ −0.1, Δ÷3σ ≈ −1.76056, drift +0.03, replay +0.03
      round 2 (per-frame JSON missing → rounds.json ``measured`` aggregate only):
        baseline 5.2, candidate 5.18, Δ −0.02, drift +0.001, replay — (not 0)
    Model side: round 1 cv 13.045186 → 12.305614; round 2 cv 12.5 → 11.9 (upper panel drawn).
    """
    run_root = base / "run_stage_d1000"
    _write(
        run_root / "stage_d1000" / "rounds.json",
        [
            {
                "round": 1,
                "cap": 0.06,
                "model_before": {"cv_sq": 170.18, "cv": 13.045186, "mean_core": 0.0225, "peak": 20.6},
                "model_after": {"cv_sq": 151.43, "cv": 12.305614, "mean_core": 0.0225, "peak": 18.7},
                "measured": {
                    "delta": -0.1,
                    "accepted": True,
                    "baseline": 5.125,
                    "candidate": 5.025,
                    "drift": 0.03,
                    "run_dir": "stage_r/run_round1",
                },
            },
            {
                "round": 2,
                "cap": 0.06,
                "model_before": {"cv_sq": 156.25, "cv": 12.5, "mean_core": 0.0225, "peak": 20.0},
                "model_after": {"cv_sq": 141.61, "cv": 11.9, "mean_core": 0.0225, "peak": 18.5},
                "measured": {
                    "delta": -0.02,
                    "accepted": True,
                    "baseline": 5.2,
                    "candidate": 5.18,
                    "drift": 0.001,
                    "run_dir": "stage_r/run_round2",  # directory intentionally absent
                },
            },
        ],
    )
    _write(
        run_root / "stage_r" / "run_round1" / "stage_d_block1.json",
        _run_json({"A1": [5.10, 5.12], "A2": [5.13, 5.15], "B1": [5.00, 5.02], "B2": [5.03, 5.05]}, accepted=True),
    )
    # stage_d1000's own run_round1 is still being written: empty dir, no JSON, no panels
    _write_np_placeholder(run_root / "stage_d1000" / "run_round1" / "p0_A1_f0.npy")
    (run_root / "stage_d1000" / "round1").mkdir(parents=True, exist_ok=True)
    _write(base / "noise_1000.json", {"three_sigma_resolution": 0.0568, "side_px": 1000, "exposure_ms": 4.0})
    return run_root


def _run(main_args: list[str]) -> int:
    return mod.main(main_args)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# stage_d-style (old layout, no rounds.json)
# ---------------------------------------------------------------------------


def test_stage_d_table_matches_hand_math(tmp_path):
    run_root = _build_stage_d_fixture(tmp_path)
    out = tmp_path / "out_d"
    rc = _run(["--run-root", str(run_root), "--stage", "stage_d", "--noise-floor", str(tmp_path / "noise_60.json"), "--out", str(out)])
    assert rc == 0
    payload = _read_json(out / "blocks.json")
    rounds = {r["round"]: r for r in payload["rounds"]}

    r1 = rounds[1]
    assert r1["baseline_cv"] == pytest.approx(1.3)
    assert r1["candidate_cv"] == pytest.approx(2.3)
    assert r1["delta"] == pytest.approx(1.0)
    assert r1["delta_over_3sigma"] == pytest.approx(10.0)
    assert r1["drift"] == pytest.approx(0.4)
    assert r1["replay_b2_minus_b1"] == pytest.approx(0.4)
    assert r1["accepted"] is True
    assert r1["missing_labels"] == []

    r3 = rounds[3]
    assert r3["baseline_cv"] == pytest.approx(0.65)
    assert r3["drift"] == pytest.approx(0.2)
    assert r3["candidate_cv"] is None
    assert r3["delta"] is None
    assert r3["replay_b2_minus_b1"] is None
    assert set(r3["missing_labels"]) == {"B1", "B2"}

    md = (out / "blocks.md").read_text(encoding="utf-8")
    for expected in ("| 1 | 1.30000 | 2.30000 | 1.00000 | 10.00000 | 0.40000 | 0.40000 |", "| 3 | 0.65000 | —"):
        assert expected in md
    assert " 0.00000 " not in md  # missing cells are —, never 0 (avoid matching inside "10.00000")


def test_stage_d_missing_round_and_sources(tmp_path):
    run_root = _build_stage_d_fixture(tmp_path)
    out = tmp_path / "out_d2"
    _run(["--run-root", str(run_root), "--stage", "stage_d", "--noise-floor", str(tmp_path / "noise_60.json"), "--out", str(out)])
    payload = _read_json(out / "blocks.json")
    rounds = {r["round"]: r for r in payload["rounds"]}

    # round 2 skipped with all nulls (not zeros)
    r2 = rounds[2]
    assert r2["source"] == "missing"
    for key in ("baseline_cv", "candidate_cv", "delta", "drift", "replay_b2_minus_b1"):
        assert r2[key] is None
    assert payload["missing_rounds"] == [2]

    # md row for round 2 is all dashes
    md = (out / "blocks.md").read_text(encoding="utf-8")
    assert f"| 2 | {NA} | {NA} | {NA} | {NA} | {NA} | {NA} |" in md

    # sources: no rounds.json → reconstructed; per-round origins recorded
    assert payload["sources"]["manifest"] == "reconstructed"
    per_round = payload["sources"]["rounds"]
    assert per_round["1"].startswith("重建自 stage_d/run_block1/stage_d_block1.json")
    assert per_round["3"].startswith("重建自 stage_d/run_block3/stage_d_block1.json")
    assert "缺失" in per_round["2"]
    assert per_round["2"].startswith("缺失")
    assert 2 in payload["sources"]["per_run_json_missing"]
    assert payload["sources"]["chain_json"] == "chain.json"


def test_stage_d_above_3sigma_count(tmp_path):
    run_root = _build_stage_d_fixture(tmp_path)
    out = tmp_path / "out_d3"
    _run(["--run-root", str(run_root), "--stage", "stage_d", "--noise-floor", str(tmp_path / "noise_60.json"), "--out", str(out)])
    payload = _read_json(out / "blocks.json")
    # only round 1 exceeds: |Δ÷3σ| = 10 > 1; round 3 has no Δ
    assert payload["above_3sigma"] == 1
    assert payload["above_3sigma_rounds"] == [1]
    # conclusion sentence: max |drift| = 0.4 > 3σ = 0.1 in this fixture
    md = (out / "blocks.md").read_text(encoding="utf-8")
    assert "1 轮 |Δ| 超过 3σ" in md
    assert "0.40000" in md
    assert "漂移不可忽略" in md


def test_stage_d_model_panel_omitted_with_note(tmp_path):
    run_root = _build_stage_d_fixture(tmp_path)
    out = tmp_path / "out_d4"
    _run(["--run-root", str(run_root), "--stage", "stage_d", "--noise-floor", str(tmp_path / "noise_60.json"), "--out", str(out)])
    payload = _read_json(out / "blocks.json")
    assert payload["model"]["available"] is False
    assert payload["model"]["before"] is None
    assert payload["model"]["after"] is None
    assert "省略" in payload["model"]["note"]
    md = (out / "blocks.md").read_text(encoding="utf-8")
    assert "上栏" in md and "省略" in md


def test_stage_d_png_generated_nonempty(tmp_path):
    run_root = _build_stage_d_fixture(tmp_path)
    out = tmp_path / "out_d5"
    _run(["--run-root", str(run_root), "--stage", "stage_d", "--noise-floor", str(tmp_path / "noise_60.json"), "--out", str(out)])
    png = out / "blocks.png"
    assert png.is_file()
    assert png.stat().st_size > 0
    assert (out / "blocks.md").is_file()
    assert (out / "blocks.json").is_file()


# ---------------------------------------------------------------------------
# stage_d1000-style (rounds.json + model side + measured.run_dir cross-stage)
# ---------------------------------------------------------------------------


def test_stage_d1000_table_matches_hand_math(tmp_path):
    run_root = _build_stage_d1000_fixture(tmp_path)
    out = tmp_path / "out_1000"
    rc = _run(
        ["--run-root", str(run_root), "--stage", "stage_d1000", "--noise-floor", str(tmp_path / "noise_1000.json"), "--out", str(out)]
    )
    assert rc == 0
    payload = _read_json(out / "blocks.json")
    rounds = {r["round"]: r for r in payload["rounds"]}

    r1 = rounds[1]
    # computed from the per-frame JSON in stage_r/run_round1, not from the measured aggregate
    assert r1["baseline_cv"] == pytest.approx(5.125)
    assert r1["candidate_cv"] == pytest.approx(5.025)
    assert r1["delta"] == pytest.approx(-0.1)
    assert r1["delta_over_3sigma"] == pytest.approx(-0.1 / 0.0568)
    assert r1["drift"] == pytest.approx(0.03)
    assert r1["replay_b2_minus_b1"] == pytest.approx(0.03)

    r2 = rounds[2]
    # per-frame JSON missing → measured aggregate, replay is — (not 0)
    assert r2["baseline_cv"] == pytest.approx(5.2)
    assert r2["candidate_cv"] == pytest.approx(5.18)
    assert r2["delta"] == pytest.approx(-0.02)
    assert r2["drift"] == pytest.approx(0.001)
    assert r2["replay_b2_minus_b1"] is None
    assert r2["source"].startswith("rounds.json:measured")

    md = (out / "blocks.md").read_text(encoding="utf-8")
    assert "| 1 | 5.12500 | 5.02500 | -0.10000 |" in md
    assert f"| 2 | 5.20000 | 5.18000 | -0.02000 | -0.35211 | 0.00100 | {NA} |" in md


def test_stage_d1000_sources_and_model_available(tmp_path):
    run_root = _build_stage_d1000_fixture(tmp_path)
    out = tmp_path / "out_1000b"
    _run(
        ["--run-root", str(run_root), "--stage", "stage_d1000", "--noise-floor", str(tmp_path / "noise_1000.json"), "--out", str(out)]
    )
    payload = _read_json(out / "blocks.json")

    assert payload["sources"]["manifest"] == "rounds.json"
    per_round = payload["sources"]["rounds"]
    assert per_round["1"] == "rounds.json 清单 + 逐帧 stage_r/run_round1/stage_d_block1.json"
    assert "measured 汇总" in per_round["2"]
    assert 2 in payload["sources"]["per_run_json_missing"]
    assert payload["missing_rounds"] == []
    assert payload["noise_floor"]["three_sigma_resolution"] == pytest.approx(0.0568)

    # model side present → upper panel drawn (not omitted)
    assert payload["model"]["available"] is True
    assert payload["model"]["before"] == [[1, 13.045186], [2, 12.5]]
    assert payload["model"]["after"] == [[1, 12.305614], [2, 11.9]]
    md = (out / "blocks.md").read_text(encoding="utf-8")
    assert "上栏" not in md  # no omission note when model data exists

    png = out / "blocks.png"
    assert png.is_file() and png.stat().st_size > 0


def test_stage_d1000_above_3sigma_count(tmp_path):
    run_root = _build_stage_d1000_fixture(tmp_path)
    out = tmp_path / "out_1000c"
    _run(
        ["--run-root", str(run_root), "--stage", "stage_d1000", "--noise-floor", str(tmp_path / "noise_1000.json"), "--out", str(out)]
    )
    payload = _read_json(out / "blocks.json")
    # |−0.1/0.0568| = 1.76 > 1; |−0.02/0.0568| = 0.35 < 1
    assert payload["above_3sigma"] == 1
    assert payload["above_3sigma_rounds"] == [1]


def test_stale_per_run_json_discarded_cross_stage(tmp_path):
    """rounds.json 的 measured.run_dir 指向别的 stage 的旧逐帧 JSON (不同窗宽/光路)
    时, 交叉核对必须判其为陈旧并丢弃, 退回 measured 汇总; stage 自身目录里的
    新 JSON 与 measured 一致时优先采用。"""
    base = tmp_path / "run_stale"
    run_root = base / "campaign"
    _write(base / "noise.json", {"three_sigma_resolution": 0.1, "side_px": 1000})

    stale_dir = run_root / "stage_old" / "run_round1"
    stale_dir.mkdir(parents=True)
    (stale_dir / "stage_d_block1.json").write_text(
        json.dumps(_run_json({"A1": [1.0, 1.0], "A2": [1.0, 1.0], "B1": [1.0, 1.0], "B2": [1.0, 1.0]}, accepted=False)),
        encoding="utf-8",
    )
    fresh_dir = run_root / "stage_new" / "run_round1"
    fresh_dir.mkdir(parents=True)
    (fresh_dir / "stage_d_block1.json").write_text(
        json.dumps(_run_json({"A1": [5.1, 5.2], "A2": [5.3, 5.4], "B1": [4.9, 5.0], "B2": [5.1, 5.2]}, accepted=True)),
        encoding="utf-8",
    )
    (run_root / "stage_new").joinpath("rounds.json").write_text(
        json.dumps(
            [
                {
                    "round": 1,
                    "model_before": {"cv": 13.0},
                    "model_after": {"cv": 12.5},
                    "measured": {"delta": -0.25, "baseline": 5.25, "candidate": 5.05, "drift": 0.2, "run_dir": "stage_old/run_round1"},
                }
            ]
        ),
        encoding="utf-8",
    )

    out = tmp_path / "out_stale"
    _run(["--run-root", str(run_root), "--stage", "stage_new", "--noise-floor", str(base / "noise.json"), "--out", str(out)])
    payload = _read_json(out / "blocks.json")
    r1 = payload["rounds"][0]
    # fresh per-frame JSON (stage 自身目录) 与 measured 一致 → 采用, 重放差可算
    assert r1["baseline_cv"] == pytest.approx((5.15 + 5.35) / 2)  # 5.25
    assert r1["candidate_cv"] == pytest.approx((4.95 + 5.15) / 2)  # 5.05
    assert r1["replay_b2_minus_b1"] == pytest.approx(0.2)
    assert r1["drift"] == pytest.approx(0.2)
    # fresh JSON 先命中 (stage 自身目录优先) 且与 measured 一致, 陈旧文件未被读取
    assert payload["sources"]["stale_per_run_json"] == []
    assert payload["sources"]["per_run_json"]["1"] == "stage_new/run_round1/stage_d_block1.json"

    # 若 stage 自身目录也没有新 JSON, 则陈旧文件被丢弃、退回 measured 汇总
    (run_root / "stage_new" / "run_round1" / "stage_d_block1.json").unlink()
    out2 = tmp_path / "out_stale2"
    _run(["--run-root", str(run_root), "--stage", "stage_new", "--noise-floor", str(base / "noise.json"), "--out", str(out2)])
    payload2 = _read_json(out2 / "blocks.json")
    r1b = payload2["rounds"][0]
    assert r1b["baseline_cv"] == pytest.approx(5.25)  # measured 汇总
    assert r1b["replay_b2_minus_b1"] is None
    assert payload2["sources"]["per_run_json"].get("1") is None
    assert payload2["sources"]["stale_per_run_json"], "stale file must be recorded even when it is the only candidate"


def test_end_to_end_stage_d_and_stage_d1000(tmp_path):
    """One end-to-end pass per stage value, exercising both layouts and the
    default out dir (<run-root>/<stage>/summary/)."""
    noise_by_stage = {"stage_d": "noise_60.json", "stage_d1000": "noise_1000.json"}
    for stage, builder in (("stage_d", _build_stage_d_fixture), ("stage_d1000", _build_stage_d1000_fixture)):
        run_root = builder(tmp_path)
        noise = tmp_path / noise_by_stage[stage]
        rc = _run(["--run-root", str(run_root), "--stage", stage, "--noise-floor", str(noise)])
        assert rc == 0
        summary = run_root / stage / "summary"
        for name in ("blocks.md", "blocks.png", "blocks.json"):
            path = summary / name
            assert path.is_file(), f"{stage}: missing {name}"
            if name != "blocks.png":
                continue
            assert path.stat().st_size > 0
        payload = _read_json(summary / "blocks.json")
        assert payload["stage"] == stage
        assert len(payload["rounds"]) >= 2
        # the em-dash, never 0.00000, marks missing values
        md = (summary / "blocks.md").read_text(encoding="utf-8")
        assert NA in md


def test_stage_missing_errors_cleanly(tmp_path):
    run_root = _build_stage_d_fixture(tmp_path)
    rc = _run(["--run-root", str(run_root), "--stage", "stage_nope", "--out", str(tmp_path / "out")])
    assert rc == 1
