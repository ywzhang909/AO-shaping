"""Validation CV must affect checkpoint selection when explicitly requested."""
import json
import pickle

from ao_shaping.tools.slm.transfer.progress import fit_trace, select_checkpoint


def test_validation_cv_rejects_correlation_only_overfit(tmp_path):
    run = tmp_path / "fit"
    run.mkdir()
    (run / "provenance.json").write_text(json.dumps(dict(rows=[dict(split="train"), dict(split="validation")])))
    (run / "frozen_parameters.json").write_text(json.dumps(dict(grid_size=16, gain=999)))
    rows = []
    for i, corr, cv in ((0, .9, .1), (1, .95, 1.)):
        metric = dict(correlation=corr, relative_l2=.2, cv_abs_error=cv)
        rows.append(dict(iteration=i, loss=1-i*.1, timestamp_utc="2026-10-26T00:00:00+00:00",
                         parameters=dict(gain=i+1), metrics=[metric, metric]))
    (run / "iterations.pkl").write_bytes(pickle.dumps(rows))
    trace = fit_trace(run)
    assert trace[-1]["validation_cv_abs_error"] == 1.
    result = select_checkpoint([run], tmp_path / "selected", cv_weight=.25)
    assert result["iteration"] == 0
    assert json.loads((tmp_path / "selected/frozen_parameters.json").read_text())["gain"] == 1
    assert len(result["candidates"]) == 2
