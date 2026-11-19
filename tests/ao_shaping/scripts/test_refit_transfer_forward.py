# -*- coding: utf-8 -*-
"""Tests for scripts/slm/refit_transfer_forward.py (no capture data, no device).

The heavy seams are replaced: ``fit.fit``, ``fit.build_fit_model``,
``refit.optical_prediction`` and ``refit.score_splits``. The split logic, the
mask assertions, the provenance assembly and the dry-run contract run for real.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.slm import refit_transfer_forward as refit

SELECTED_KEYS = {
    "kind", "generated", "script", "script_sha256", "dataset", "manifest_sha256",
    "base_checkpoint", "base_checkpoint_sha256", "refit_checkpoint",
    "refit_checkpoint_sha256", "span", "adaptation", "radiometry", "split_counts",
    "train_rows", "calibration_excluded", "test_scored_once", "forward_api",
    "metrics", "gates",
}
GATE_KEYS = {"train_only_fit", "calibration_excluded", "split_counts_match",
             "refit_beats_base_on_score_split"}


def write_session(tmp_path, counts=(4, 2, 2)):
    """A synthetic session directory: manifest.json, config.json, checkpoint."""
    rows = []
    for split, count in zip(("train", "test", "calibration"), counts):
        for index in range(count):
            rows.append({"split": split, "label": f"{split}_{index}",
                         "file": f"{split}_{index}.npz", "exposure_ms": 1.0})
    (tmp_path / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")
    (tmp_path / "config.json").write_text(json.dumps({"grid": 8}), encoding="utf-8")
    checkpoint = tmp_path / "frozen_parameters.json"
    checkpoint.write_text(json.dumps({"grid_size": 8}), encoding="utf-8")
    return rows, checkpoint


def selected_rows(rows):
    return [row for row in rows if row["split"] in ("train", "test")]


def fake_components(rows):
    chosen = selected_rows(rows)
    return SimpleNamespace(
        rows=chosen,
        train=np.array([row["split"] == "train" for row in chosen]),
        images=np.zeros((len(chosen), 2, 2)),
        exposure_values=np.ones(len(chosen)),
        kernel=None,
        model=object(),
    )


# ---------------------------------------------------------------------------
# split counting / calibration exclusion
# ---------------------------------------------------------------------------
def test_split_plan_counts_and_excludes_calibration(tmp_path):
    rows, _ = write_session(tmp_path)
    plan = refit.split_plan(rows)
    assert plan["counts"] == {"train": 4, "test": 2, "calibration": 2}
    assert plan["train"] == 4
    assert plan["selected"] == 6
    assert plan["calibration"] == 2


def test_split_plan_refuses_unknown_split(tmp_path):
    rows, _ = write_session(tmp_path)
    rows.append({"split": "holdout", "label": "rogue", "file": "rogue.npz"})
    with pytest.raises(SystemExit, match="split"):
        refit.split_plan(rows)


def test_split_plan_refuses_duplicate_payload(tmp_path):
    rows, _ = write_session(tmp_path)
    rows.append({"split": "train", "label": "clone", "file": "test_0.npz"})
    with pytest.raises(SystemExit, match="more than once"):
        refit.split_plan(rows)


def test_split_plan_requires_train_rows(tmp_path):
    rows, _ = write_session(tmp_path, counts=(0, 2, 2))
    with pytest.raises(SystemExit, match="no train rows"):
        refit.split_plan(rows)


def test_verify_mask_refuses_calibration_rows_in_selection(tmp_path):
    rows, _ = write_session(tmp_path)
    plan = refit.split_plan(rows)
    leaking = SimpleNamespace(
        rows=rows,
        train=np.array([True, True, True, True, False, False, True, True]),
    )
    with pytest.raises(SystemExit, match="reached the fit selection"):
        refit.verify_split_mask(leaking, plan)


def test_verify_mask_refuses_test_rows_in_train_mask(tmp_path):
    rows, _ = write_session(tmp_path)
    plan = refit.split_plan(rows)
    leaking = fake_components(rows)
    leaking.train = np.array([True, True, True, True, True, False])
    with pytest.raises(SystemExit, match="mask would include non-train rows"):
        refit.verify_split_mask(leaking, plan)


def test_verify_mask_accepts_train_only_mask(tmp_path):
    rows, _ = write_session(tmp_path)
    plan = refit.split_plan(rows)
    refit.verify_split_mask(fake_components(rows), plan)


# ---------------------------------------------------------------------------
# --expect-splits
# ---------------------------------------------------------------------------
def test_parse_expected_splits():
    assert refit.parse_expected_splits("train=4,test=2,calibration=2") == {
        "train": 4, "test": 2, "calibration": 2}
    assert refit.parse_expected_splits(None) is None
    with pytest.raises(SystemExit, match="NAME=COUNT"):
        refit.parse_expected_splits("train")


def test_expect_splits_mismatch_raises(tmp_path):
    rows, _ = write_session(tmp_path)
    with pytest.raises(SystemExit, match="expect-splits mismatch"):
        refit.split_plan(rows, {"train": 80, "test": 2})


def test_main_refuses_expect_splits_mismatch(tmp_path):
    _, checkpoint = write_session(tmp_path)
    with pytest.raises(SystemExit, match="expect-splits mismatch"):
        refit.main([
            "--dataset", str(tmp_path),
            "--checkpoint", str(checkpoint),
            "--out", str(tmp_path / "refit"),
            "--expect-splits", "train=80,test=24,calibration=10",
        ])


# ---------------------------------------------------------------------------
# provenance assembly (heavy seams monkeypatched)
# ---------------------------------------------------------------------------
def test_provenance_has_required_keys_and_gates(tmp_path, monkeypatch):
    from ao_shaping.tools.slm.transfer import fit as fitmod

    rows, checkpoint = write_session(tmp_path)
    out = tmp_path / "refit"
    comps = fake_components(rows)

    def fake_fit(namespace):
        namespace.out.mkdir(parents=True)
        (namespace.out / "frozen_parameters.json").write_text(
            json.dumps({"grid_size": 8, "refit": True}), encoding="utf-8")

    monkeypatch.setattr(fitmod, "fit", fake_fit)
    monkeypatch.setattr(refit, "build_components", lambda *a, **k: comps)
    monkeypatch.setattr(refit, "optical_prediction", lambda comps: np.zeros((6, 2, 2)))

    # Base and refit are scored consistently; the refit is the better one here.
    scores = iter([
        {"test": {"correlation": 0.5, "relative_l2": 0.3, "cv_abs_error": 0.02}},
        {"test": {"correlation": 0.9, "relative_l2": 0.05, "cv_abs_error": 0.01}},
    ])
    monkeypatch.setattr(
        refit, "score_splits",
        lambda comps, optical, fitmod, names, radiometry="exposure": next(scores))

    assert refit.main([
        "--dataset", str(tmp_path),
        "--checkpoint", str(checkpoint),
        "--out", str(out),
        "--expect-splits", "train=4,test=2,calibration=2",
    ]) == 0

    provenance = json.loads((out / "refit_provenance.json").read_text(encoding="utf-8"))
    assert SELECTED_KEYS <= set(provenance)
    assert provenance["kind"] == "transfer_forward_refit"
    assert provenance["calibration_excluded"] is True
    assert provenance["test_scored_once"] is True
    assert provenance["train_rows"] == 4
    assert provenance["split_counts"] == {"train": 4, "test": 2, "calibration": 2}
    assert provenance["base_checkpoint"].endswith("frozen_parameters.json")
    refit_path = Path(provenance["refit_checkpoint"])
    assert refit_path.name == "frozen_parameters.json"
    assert refit_path.parent.name == "refit"
    assert set(provenance["metrics"]) == {"base", "refit"}
    assert set(provenance["metrics"]["refit"]["test"]) >= {
        "correlation", "relative_l2", "cv_abs_error"}
    assert GATE_KEYS <= set(provenance["gates"])
    assert provenance["gates"]["train_only_fit"] is True
    assert provenance["gates"]["calibration_excluded"] is True
    assert provenance["gates"]["split_counts_match"] is True
    assert provenance["gates"]["refit_beats_base_on_score_split"] is True
    modules = {item["module"] for item in provenance["forward_api"]["sources"]}
    assert "ao_shaping.tools.slm.transfer.fit" in modules
    assert "ao_shaping.algorithm.signal_processing.physical_forward_fit" in modules


def test_provenance_gate_flags_a_worse_refit(tmp_path, monkeypatch):
    from ao_shaping.tools.slm.transfer import fit as fitmod

    rows, checkpoint = write_session(tmp_path)
    out = tmp_path / "refit"
    comps = fake_components(rows)

    def fake_fit(namespace):
        namespace.out.mkdir(parents=True)
        (namespace.out / "frozen_parameters.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(fitmod, "fit", fake_fit)
    monkeypatch.setattr(refit, "build_components", lambda *a, **k: comps)
    monkeypatch.setattr(refit, "optical_prediction", lambda comps: np.zeros((6, 2, 2)))
    scores = iter([
        {"test": {"correlation": 0.9, "relative_l2": 0.05, "cv_abs_error": 0.01}},
        {"test": {"correlation": 0.5, "relative_l2": 0.3, "cv_abs_error": 0.02}},
    ])
    monkeypatch.setattr(
        refit, "score_splits",
        lambda comps, optical, fitmod, names, radiometry="exposure": next(scores))

    assert refit.main([
        "--dataset", str(tmp_path), "--checkpoint", str(checkpoint), "--out", str(out),
    ]) == 0
    provenance = json.loads((out / "refit_provenance.json").read_text(encoding="utf-8"))
    assert provenance["gates"]["refit_beats_base_on_score_split"] is False
    # Without --expect-splits the count gate is not a comparison.
    assert provenance["gates"]["split_counts_match"] is True
    assert provenance["expect_splits"] is None


# ---------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------
def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    from ao_shaping.tools.slm.transfer import fit as fitmod

    rows, checkpoint = write_session(tmp_path)
    out = tmp_path / "refit"
    comps = fake_components(rows)

    def explode(*_args, **_kwargs):  # pragma: no cover - would fail the test
        raise AssertionError("fit must not run under --dry-run")

    monkeypatch.setattr(fitmod, "fit", explode)
    monkeypatch.setattr(fitmod, "fit_loss", lambda *a, **k: (0.25, {"loss": 0.25}, None))
    monkeypatch.setattr(refit, "build_components", lambda *a, **k: comps)

    assert refit.main([
        "--dataset", str(tmp_path), "--checkpoint", str(checkpoint),
        "--out", str(out), "--dry-run",
    ]) == 0
    assert not out.exists()
