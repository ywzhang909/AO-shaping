"""Resume must reuse completed model stages without reopening their directories."""
from __future__ import annotations

import json

from ao_shaping.runners.slm import transfer_shapes_runner as module
from ao_shaping.tools.slm.transfer import bootstrap, geometric_seed


def test_resume_reuses_completed_capture_fit_and_geometry(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(module.capture_stage, "capture",
                        lambda _args: calls.append("capture"))
    monkeypatch.setattr(module.fit_stage, "fit",
                        lambda _args: calls.append("fit"))
    monkeypatch.setattr(module.fit_stage, "spot", lambda _row: (1300.0, 942.0))
    monkeypatch.setattr(
        bootstrap, "bootstrap",
        lambda _capture: calls.append("bootstrap") or ({}, {}))
    monkeypatch.setattr(geometric_seed, "export_frozen_parameters",
                        lambda *args, **kwargs: calls.append("export"))
    monkeypatch.setattr(module, "run_stages", lambda **kwargs: False)
    monkeypatch.setattr(module.replay_phases, "replay", lambda _args: None)

    (tmp_path / "capture").mkdir()
    (tmp_path / "capture" / "manifest.json").write_text(
        json.dumps([{"label": "reference_start"}]), encoding="utf-8")
    for relative in (
        "initial/frozen_parameters.json",
        "fit/frozen_parameters.json",
        "model640/frozen_parameters.json",
        "model_wide640/frozen_parameters.json",
    ):
        artifact = tmp_path / relative
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("{}", encoding="utf-8")

    (tmp_path / "journal.json").write_text(
        json.dumps([{"stage": "previous", "status": "complete"}]), encoding="utf-8")
    module.execute_campaign(out=tmp_path, shapes=("square",), resume=True,
                            seed_bakeoff=False)

    assert calls == []
    journal = json.loads((tmp_path / "journal.json").read_text(encoding="utf-8"))
    assert journal[0] == {"stage": "previous", "status": "complete"}
    reused = {(entry["stage"], entry["status"]) for entry in journal}
    assert ("capture", "reused") in reused
    assert ("fit", "reused") in reused
