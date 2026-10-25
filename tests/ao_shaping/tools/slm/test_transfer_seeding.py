"""Known-model phase transfer preserves the accepted solution without GS."""
import json
from dataclasses import replace

import numpy as np
import pytest

from ao_shaping.tools.slm.transfer.seeding import accepted_seed


def test_session_preserves_existing_positional_shape_argument(tmp_path):
    from ao_shaping.tools.slm.transfer.workflow import TransferSession
    session = TransferSession(tmp_path, None, None, ("triangle",))
    assert session.shapes == ("triangle",)
    assert session.seed_run is None


def test_cli_continuation_defaults_to_ccd_anchored_mode(tmp_path, monkeypatch):
    import sys
    from ao_shaping.tools.slm.transfer import validate
    seen = []
    monkeypatch.setattr(validate, "run", lambda args: seen.append(args))
    monkeypatch.setattr(sys, "argv", ["validate", "--checkpoint", str(tmp_path / "model.json"),
        "--out", str(tmp_path / "out"), "--zero-xy", "1", "2", "--seed-run", str(tmp_path / "source")])
    validate.main()
    assert seen[0].mode == "anchored"


def test_transfer_static_phase_and_carrier_have_known_sign(tmp_path, monkeypatch):
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusForward, FrozenDefocusParameters,
    )
    from ao_shaping.tools.slm.transfer import validate
    monkeypatch.setattr(validate, "gs_seed", lambda *a: pytest.fail("GS must not run"))
    p = FrozenDefocusParameters(.01, (40., 40.), (960., 600.), (300., 260.),
                                {5: .1}, 1., 0., 0., ((1000., 0.), (0., 1000.)),
                                grid_size=16, fft_size=64, pixel_samples=1)
    checkpoint = tmp_path / "old.json"
    checkpoint.write_text(json.dumps(p.to_dict()))
    yy, xx = np.mgrid[:9, :9]
    xy = np.stack((xx, yy), axis=-1)
    phase = np.arange(256).reshape(16, 16)*.01
    target = np.ones((9, 9))
    np.savez(tmp_path / "seed.npz", model_phase=phase, camera_xy=xy, target=target)
    (tmp_path / "manifest.json").write_text(json.dumps([dict(label="square_final", file="seed.npz")]))
    (tmp_path / "config.json").write_text(json.dumps(dict(checkpoint=str(checkpoint), target_xy=[50., 50.])))
    old = FrozenDefocusForward(p, xy, device="cpu")
    same, evidence = accepted_seed(tmp_path, "square", old, xy, target, policy="transfer", centre=[50., 50.])
    np.testing.assert_allclose(same, phase, atol=1e-7)
    new = FrozenDefocusForward(replace(p, defocus_diopter=.012, origin_xy=(42., 40.)), xy, device="cpu")
    moved, evidence = accepted_seed(tmp_path, "square", new, xy, target, policy="transfer", centre=[50., 50.])
    correction = (.01-.012)*p.diopter_scale*old.z4.numpy()
    expected = phase + correction + 2*np.pi*(-.002)*(np.mgrid[:16, :16][1]*120.)
    np.testing.assert_allclose(moved, expected, atol=2e-6)
    assert evidence["gs_calls"] == 0
    with pytest.raises(ValueError, match="grid/target"):
        accepted_seed(tmp_path, "square", old, xy+1, target, policy="strict")
