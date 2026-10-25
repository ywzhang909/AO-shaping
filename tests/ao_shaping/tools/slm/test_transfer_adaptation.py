"""Parameter identifiability and validation fallback for mild optical drift."""
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from ao_shaping.tools.slm.transfer.adaptation import (
    configure_adaptation, needs_full_adaptation, validation_quality,
)
from ao_shaping.tools.slm.transfer import fit


def test_single_shape_automatically_uses_existing_radiometric_calibration(tmp_path):
    from ao_shaping.tools.slm.transfer.workflow import TransferSession
    config = TransferSession(tmp_path, shapes=("star",))
    assert not config.calibration_enabled
    assert "exposure_response_capture" not in config.plan()["stages"]
    assert TransferSession(tmp_path).calibration_enabled


def entry(error, center=960.):
    return dict(parameters=dict(amplitude_center=[center, 600.]), metrics=[
        dict(correlation=.99, relative_l2=.01, cv_abs_error=.01),
        dict(correlation=1-error, relative_l2=error, cv_abs_error=error)])


def test_validation_ignores_better_training_metrics():
    rows = [dict(split="train"), dict(split="validation")]
    quality = validation_quality(entry(.3), rows)
    assert quality["correlation"] == .7
    assert needs_full_adaptation(quality)
    with pytest.raises(ValueError, match="validation"):
        validation_quality(entry(.3), [dict(split="train")]*2)


def test_alignment_does_not_attribute_power_to_aberrations():
    torch = pytest.importorskip("torch")
    model = SimpleNamespace(**{name: torch.nn.Parameter(torch.zeros(2))
                              for name in ("d", "log_width", "residual", "center", "origin")})
    configure_adaptation(model, "alignment")
    assert model.center.requires_grad and model.origin.requires_grad
    assert not any(getattr(model, name).requires_grad for name in ("d", "log_width", "residual"))
    configure_adaptation(model, "full")
    assert all(p.requires_grad for p in vars(model).values())


@pytest.mark.parametrize("initial_error,fit_error,expected,full", [
    (.1, .05, "alignment", False),
    (.05, .1, "alignment_initial", False),
    (.4, .3, "full", True),
])
def test_auto_stage_selection_preserves_baseline(tmp_path, monkeypatch, initial_error, fit_error, expected, full):
    def fake_fit(args):
        args.out.mkdir()
        initial = entry(initial_error)
        result = entry(.02 if args.adaptation == "full" else fit_error, center=970.)
        for filename, payload in (("result.json", result),
                                  ("provenance.json", dict(rows=[dict(split="train"), dict(split="validation")])),
                                  ("frozen_parameters.json", result["parameters"] | dict(grid_size=32))):
            (args.out / filename).write_text(json.dumps(payload))
        with (args.out / "iterations.pkl").open("wb") as stream:
            pickle.dump([initial, result], stream)
    monkeypatch.setattr(fit, "fit", fake_fit)
    args = SimpleNamespace(out=tmp_path / "auto", checkpoint=tmp_path / "old.json", iterations=60)
    fit.fit_adaptive(args)
    choice = json.loads((args.out / "adaptation.json").read_text())
    saved = json.loads((args.out / "frozen_parameters.json").read_text())
    assert choice["selected_stage"] == expected
    assert choice["escalated"] is full
    assert saved["amplitude_center"][0] == (960. if expected.endswith("initial") else 970.)


def test_synthetic_data_preserves_exposure_and_power_scaling(tmp_path):
    pytest.importorskip("torch")
    from ao_shaping.tools.slm.transfer.simulation import generate_dataset
    a, b = tmp_path / "a", tmp_path / "b"
    generate_dataset(a, power=1., count=6)
    generate_dataset(b, power=1.3, count=6)
    rows = json.loads((a / "manifest.json").read_text())
    for row in rows:
        with np.load(a / row["file"]) as left, np.load(b / row["file"]) as right:
            np.testing.assert_allclose(right["raw_frames"], 1.3*left["raw_frames"], rtol=2e-6)
    truth = json.loads((b / "truth.json").read_text())
    assert truth["synthetic"] is True
    assert truth["parameters"]["amplitude_center"] == [972., 592.]
