"""Recover a known positive differential gain and camera blur, without devices."""
import json
from dataclasses import replace

import numpy as np


def test_signed_pair_recovers_known_gain_and_blur(tmp_path):
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusForward, FrozenDefocusParameters,
    )
    from ao_shaping.tools.slm.transfer.local_response import fit_local_response
    p = FrozenDefocusParameters(0., (20., 20.), (960., 600.), (300., 260.), {},
        3., 0., 0., ((7500., 0.), (0., 7500.)), grid_size=16, fft_size=64, pixel_samples=1)
    yy, xx = np.mgrid[:41, :41]
    xy = np.stack((xx, yy), axis=-1)
    phase = np.random.default_rng(4).normal(size=(16, 16)).astype(np.float32)
    model = FrozenDefocusForward(replace(p, gain=1.2, sigma_ccd_px=2.), xy, device="cpu")
    checkpoint = tmp_path / "model.json"
    checkpoint.write_text(json.dumps(p.to_dict()))
    (tmp_path / "config.json").write_text(json.dumps(dict(checkpoint=str(checkpoint))))
    rows = []
    for label, control in (("square_0_B1", phase), ("square_0_B2", phase), ("square_direction_minus", -phase)):
        raw = model.predict(control).detach().numpy()+7.
        file = label+".npz"
        np.savez_compressed(tmp_path / file, raw_frames=raw, camera_xy=xy, model_phase=control)
        rows.append(dict(label=label, file=file, exposure_ms=1.))
    (tmp_path / "manifest.json").write_text(json.dumps(rows))
    result = fit_local_response(tmp_path, tmp_path / "local", sigmas=(0., 2., 4.))
    assert result["selected"]["sigma_ccd_px"] == 2.
    assert abs(result["selected"]["gain_ratio"]-.4) < .001
