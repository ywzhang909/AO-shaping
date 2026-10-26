"""Axial scan keeps repeated phase/exposure pairs in the same split."""
import json
from types import SimpleNamespace

import numpy as np


def test_scan_groups_same_phase_and_retains_unproven_boundary_candidate(tmp_path, monkeypatch):
    import torch
    from ao_shaping.algorithm.signal_processing import differentiable_defocus_uniformity as models
    from ao_shaping.tools.slm.transfer.axial import scan
    params = models.FrozenDefocusParameters(.02, (1., 1.), (960., 600.), (300., 300.),
             {}, 1., 0., 0., ((1000., 0.), (0., 1000.)), grid_size=8, fft_size=16)
    monkeypatch.setattr(models, "load_frozen_parameters", lambda _:params)

    class Forward:
        def __init__(self, params, xy, device):
            self.xy = xy

        def predict(self, phase):
            return torch.as_tensor((1+float(phase[0,0])+.1*self.xy[...,0])[None], dtype=torch.float32)

    monkeypatch.setattr(models, "FrozenDefocusForward", Forward)
    source = tmp_path/"capture"
    source.mkdir()
    y, x = np.mgrid[:3, :3]
    xy = np.stack((x,y),-1)
    rows = []
    for index in range(3):
        for exposure in (1.,2.):
            phase = np.full((8,8),index,np.float32)
            raw = np.round((10+index+x)*exposure).astype(np.uint16)[None]
            filename = f"{len(rows)}.npz"
            np.savez_compressed(source/filename, raw_frames=raw, camera_xy=xy, core=np.ones((3,3),bool), model_phase=phase)
            rows.append(dict(file=filename, exposure_ms=exposure, full_scale=4095, counts_scale=1.))
    (source/"manifest.json").write_text(json.dumps(rows))
    out = tmp_path/"scan"
    scan(SimpleNamespace(checkpoint=tmp_path/"unused", capture=source, out=out, sample_stride=1,
                         offsets=[-.1,0.,.1], device="cpu"))
    config = json.loads((out/"config.json").read_text())
    for start in (0,2,4):
        assert config["phase_hashes"][start] == config["phase_hashes"][start+1]
        assert config["train"][start] == config["train"][start+1]
    decision = json.loads((out/"selection.json").read_text())
    assert not decision["promoted"]
    assert json.loads((out/"frozen_parameters.json").read_text())["slm_axial_offset_m"] == 0.
    assert (out/"candidate_parameters.json").exists()
