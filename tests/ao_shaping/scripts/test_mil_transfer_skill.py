"""Behavioral checks for the offline helper bundled with the MIL skill."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


TOOL = Path(__file__).resolve().parents[3]/"libs/skills/slm-mil-transfer/scripts/session_review.py"
SPEC = importlib.util.spec_from_file_location("mil_session_review", TOOL)
REVIEW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REVIEW)


@pytest.fixture
def recorded(tmp_path):
    from ao_shaping.algorithm.signal_processing.multishape_objective import ShapeObjective, image_metrics
    y,x = np.mgrid[:9,:9]
    xy = np.stack((x,y),-1)
    target = ((abs(x-4)<4)&(abs(y-4)<4)).astype(np.float32)
    objective = ShapeObjective(target)
    rows = []
    for exposure in (1.,2.):
        for label,weight,split in (("A",20,"train"),("B",5,"validation")):
            image = ((200+weight*x)*exposure).astype(np.uint16)
            phase = np.full((8,8),weight,np.float32)
            raw = np.stack((image,image))
            filename = f"{label}_{exposure}.npz"
            np.savez_compressed(tmp_path/filename, raw_frames=raw,model_phase=phase,
                                camera_xy=xy,target=target,core=objective.core)
            rows.append(dict(label=label,file=filename,split=split,exposure_ms=exposure,full_scale=4095,
                metrics=image_metrics(image,objective),frames=[dict(timestamp_utc="2026-10-27T00:00:00+00:00",
                monotonic_ns=1,end_monotonic_ns=2,exposure_ms=exposure)]*2))
    (tmp_path/"manifest.json").write_text(json.dumps(rows))
    return tmp_path,rows


def test_metadata_cli_works_without_site_packages_and_refuses_overwrite(recorded):
    run,_ = recorded
    target = run/"result.json"
    command = [sys.executable,"-S",str(TOOL),"review","--run",str(run),"--out",str(target)]
    first = subprocess.run(command,capture_output=True,text=True)
    assert first.returncode == 0, first.stderr
    assert json.loads(target.read_text())["captures"] == 4
    original = target.read_bytes()
    second = subprocess.run(command,capture_output=True,text=True)
    assert second.returncode == 2 and "already exists" in second.stderr
    assert target.read_bytes() == original


def test_raw_comparison_recomputes_only_same_exposure_and_mask(recorded):
    run,_ = recorded
    result = REVIEW.compare(run,"A","B",check_raw=True)
    assert [r["exposure_ms"] for r in result["comparison"]] == [1.,2.]
    assert all(r["delta_cv"]<0 and r["relative_cv_reduction"]>0 for r in result["comparison"])
    assert result["comparison"][0]["baseline_cv"] == pytest.approx(result["comparison"][1]["baseline_cv"])
    assert result["comparison"][0]["raw_checks"][0]["saturated_core_samples"] == 0


def test_comparison_refuses_mismatched_masks(recorded):
    run,rows = recorded
    path = run/rows[1]["file"]
    with np.load(path) as saved:
        payload = {k:saved[k].copy() for k in saved.files}
    payload["core"][4,4] = False
    np.savez_compressed(path,**payload)
    # B has a linear ramp; removing its central value leaves the mean unchanged
    # but changes its variance, so remove manifest CV to reach mask comparison.
    rows[1]["metrics"].pop("cv")
    (run/"manifest.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError,match="core differs"):
        REVIEW.compare(run,"A","B",check_raw=True)


def test_compare_refuses_exposure_mismatch_and_ambiguous_duplicates(recorded):
    run,rows = recorded
    (run/"manifest.json").write_text(json.dumps(rows[:-1]))
    with pytest.raises(ValueError,match="exposure sets differ"):
        REVIEW.compare(run,"A","B")
    (run/"manifest.json").write_text(json.dumps(rows+[rows[0]]))
    with pytest.raises(ValueError,match="ambiguous"):
        REVIEW.compare(run,"A","B")


def test_raw_audit_detects_stale_metrics_and_same_phase_split_overlap(recorded):
    run,rows = recorded
    rows[0]["metrics"]["cv"] = .999
    (run/"manifest.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError,match="CV mismatch"):
        REVIEW.review(run,check_raw=True)
    rows[0]["metrics"].pop("cv")
    rows[2]["split"] = "validation"
    (run/"manifest.json").write_text(json.dumps(rows))
    result = REVIEW.review(run,check_raw=True)
    assert list(result["phase_split_overlap"].values()) == [["train","validation"]]


def test_no_record_path_escape_or_nonpositive_exposure(recorded):
    run,rows = recorded
    rows[0]["file"] = "../outside.npz"
    (run/"manifest.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError,match="escapes"):
        REVIEW.review(run)
    rows[0]["file"] = "A_1.0.npz"
    rows[0]["exposure_ms"] = 0
    (run/"manifest.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError,match="positive exposure"):
        REVIEW.review(run)


def test_comparison_rejects_format_or_phase_changes_across_exposures(recorded):
    run,rows = recorded
    rows[1]["full_scale"] = 255
    (run/"manifest.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError,match="full_scale differs"):
        REVIEW.compare(run,"A","B")
    rows[1]["full_scale"] = 4095
    (run/"manifest.json").write_text(json.dumps(rows))
    path = run/rows[-1]["file"]
    with np.load(path) as saved:
        payload = {k:saved[k].copy() for k in saved.files}
    payload["model_phase"] += 1.
    np.savez_compressed(path,**payload)
    with pytest.raises(ValueError,match="phase changes across exposures"):
        REVIEW.compare(run,"A","B",check_raw=True)
