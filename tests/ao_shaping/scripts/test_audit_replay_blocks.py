"""Tests for the offline replay audit (device-free; reads only synthetic archives).

The script exists to make one class of defect impossible: a replay block whose
frames were captured at different exposures, bit depths or with a clipped peak
being compared as if it were one condition. These tests build a tiny campaign on
disk and assert the audit refuses to do that.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "slm" / "audit_replay_blocks.py"


def _load_audit_module():
    """Import the script by path (``scripts/`` is not an installable package)."""
    spec = importlib.util.spec_from_file_location("audit_replay_blocks_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_audit_module()


FULL_SCALE = 4095.0


def _stage_config(tmp: Path) -> Path:
    """A source stage config: 42 px target, 2 px core erosion, small zero disk.

    The target is deliberately *not* exactly filled by the synthetic frames: a
    half-filled target makes CV sensitive to the level, which is what the
    pair-comparison tests need to move.
    """
    stage = tmp / "shape" / "square" / "transport_cv"
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "config.json").write_text(json.dumps(dict(
        target_xy=[100.0, 100.0], side=40.0, zero_xy=[100.0, 100.0],
        zero_radius=4.0, core_margin=2.0)), encoding="utf-8")
    return stage


def _write_block(tmp: Path, rows: list[dict], *, initial_run: Path) -> Path:
    """Materialise one replay block: config, manifest and one npz per row."""
    block = tmp / "common_replay"
    block.mkdir(parents=True, exist_ok=True)
    (block / "config.json").write_text(
        json.dumps(dict(initial_run=str(initial_run))), encoding="utf-8")
    manifest = []
    for index, row in enumerate(rows):
        frame = np.zeros((200, 200), dtype=np.float64)
        frame[80:120, 80:120] = row.pop("level", 100.0)
        if row.pop("clip", False):
            frame[90:110, 90:110] = FULL_SCALE
        name = f"replay_{index:04d}.npz"
        np.savez_compressed(block / name, raw_frames=frame[None, ...])
        manifest.append(dict(label=row["label"], file=name,
                             exposure_ms=row.get("exposure_ms", 20.0),
                             bit_depth=row.get("bit_depth", 12), full_scale=FULL_SCALE,
                             metrics=row.get("metrics", {})))
    (block / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (block / "released.json").write_text("{}", encoding="utf-8")
    return block


def test_audit_reads_the_frozen_aperture_from_the_source_stage(tmp_path):
    stage = _stage_config(tmp_path)
    block = _write_block(tmp_path, [dict(label="transport_cv_square_seed")], initial_run=stage)
    result = audit.audit_block(block)
    assert result["present"] and result["released"]
    row = result["rows"][0]
    assert "error" not in row

    assert row["shape"] == "square" and row["phase"] == "seed"
    assert row["stage"] == "transport_cv"
    # The aperture is the source stage's own geometry, in sensor pixels.
    assert result["aperture"]["half_side"] == pytest.approx(20.0)
    assert result["aperture"]["core_margin_px"] == pytest.approx(2.0)


def test_clipped_frame_is_flagged_from_the_frame_peak_not_the_manifest(tmp_path):
    stage = _stage_config(tmp_path)
    block = _write_block(tmp_path, [
        dict(label="mono12_cv_square_final", level=200.0),
        dict(label="mono12_cv_square_seed", clip=True),
    ], initial_run=stage)
    summary = audit.summarise_block(audit.audit_block(block))
    assert summary["clipped"] == ["mono12_cv_square_seed"]


def test_exposures_and_bit_depths_are_never_pooled(tmp_path):
    stage = _stage_config(tmp_path)
    block = _write_block(tmp_path, [
        dict(label="mono12_cv_square_seed", exposure_ms=20.0),
        dict(label="mono12_cv_square_final", exposure_ms=90.0),
    ], initial_run=stage)
    summary = audit.summarise_block(audit.audit_block(block))
    assert sorted(summary["groups"]) == ["20ms/12bit/square", "90ms/12bit/square"]


def test_one_measurement_per_phase_cannot_claim_a_supported_gain(tmp_path):
    stage = _stage_config(tmp_path)
    block = _write_block(tmp_path, [
        dict(label="mono12_dense_square_seed", level=100.0),
        dict(label="mono12_dense_square_final", level=400.0),
    ], initial_run=stage)
    group = audit.summarise_block(audit.audit_block(block))["groups"]["20ms/12bit/square"]
    assert group["replicates"] == {}
    # Same CV at two different brightnesses: the pair cannot be called either way
    # without a replicate spread, so the audit must not claim one.
    assert group["pairs"][0]["verdict"]["supported"] is False
    assert group["pairs"][0]["verdict"]["reason"] == "no_difference"


def test_repeated_phase_supplies_the_noise_the_verdict_uses(tmp_path):
    stage = _stage_config(tmp_path)
    block = _write_block(tmp_path, [
        dict(label="mono12_cv_square_seed", level=400.0),
        dict(label="mono12_cv_square_final", level=400.0),
        dict(label="mono12_cv_square_final", level=402.0),
        dict(label="mono12_cv_square_final", level=398.0),
    ], initial_run=stage)
    group = audit.summarise_block(audit.audit_block(block))["groups"]["20ms/12bit/square"]
    assert group["phase_counts"]["final"] == 3
    assert "final" in group["replicates"]
    # A flat reference and a 1% -brighter candidate are the same shape: with no
    # replicate spread the pair cannot be called either way.
    assert group["pairs"][0]["verdict"]["supported"] in {None, False}
    assert group["pairs"][0]["verdict"]["reason"] in {"insufficient_repeats", "no_difference"}



def test_missing_block_is_reported_not_crashed(tmp_path):
    summary = audit.summarise_block(audit.audit_block(tmp_path / "absent"))
    assert summary["present"] is False and summary["groups"] == {}


def test_label_parse_handles_underscored_stage_names():
    assert audit._label_parts("mono12_continue_triangle_final") == (
        "mono12_continue", "triangle", "final")
    assert audit._label_parts("nonsense") == ("<unparsed>", "nonsense", "")


def test_unreleased_block_is_marked(tmp_path):
    stage = _stage_config(tmp_path)
    block = _write_block(tmp_path, [dict(label="transport_cv_square_seed")], initial_run=stage)
    (block / "released.json").unlink()
    assert audit.summarise_block(audit.audit_block(block))["released"] is False


def test_cli_writes_json_and_reports_missing_session(tmp_path):
    stage = _stage_config(tmp_path)
    _write_block(tmp_path, [dict(label="transport_cv_square_seed")], initial_run=stage)
    out = tmp_path / "audit.json"
    assert audit.main(["--campaign", str(tmp_path), "--session", ".", "--json", str(out),
                       "--no-table"]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["blocks"][0]["n_usable"] == 1
    with pytest.raises(SystemExit):
        audit.main(["--campaign", str(tmp_path), "--session", "absent", "--no-table"])
