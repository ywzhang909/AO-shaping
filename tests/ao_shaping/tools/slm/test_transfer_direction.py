"""Archived float64 phases must support float32 propagation and autograd."""
import json

import numpy as np
import pytest


def test_direction_analysis_accepts_archived_double_precision_phase(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from ao_shaping.algorithm.signal_processing import differentiable_defocus_uniformity as physical
    from ao_shaping.tools.slm.transfer.direction import analyze

    class Forward:
        def __init__(self, params, xy, **kwargs):
            self.pattern = torch.tensor(xy[..., 0], dtype=torch.float32)*.0001

        def predict(self, phase):
            value = phase if torch.is_tensor(phase) else torch.tensor(phase, dtype=torch.float32)
            assert value.dtype == torch.float32
            return (10 + self.pattern*value.sum())[None]

    monkeypatch.setattr(physical, "FrozenDefocusForward", Forward)
    monkeypatch.setattr(physical, "load_frozen_parameters", lambda _: None)
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{}")
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps(dict(checkpoint=str(checkpoint), shapes=["square"])))
    yy, xx = np.mgrid[:15, :15]
    xy = np.stack((xx, yy), axis=-1)
    target = np.zeros((15, 15))
    target[3:12, 3:12] = 1
    rows = []
    for label, delta in (("square_0_A1", 0), ("square_0_B1", .1), ("square_direction_minus", -.1)):
        file = label + ".npz"
        np.savez(run / file, model_phase=np.full((8, 8), delta, dtype=np.float64),
                 camera_xy=xy, target=target, raw_frames=np.ones((3, 15, 15))*10)
        rows.append(dict(label=label, file=file, exposure_ms=1., loss=1.))
    (run / "manifest.json").write_text(json.dumps(rows))
    out = tmp_path / "analysis"
    analyze(run, out)
    assert len(json.loads((out / "result.json").read_text())["results"][0]["derivative_checks"]) == 3
