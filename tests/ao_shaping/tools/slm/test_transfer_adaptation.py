"""Parameter identifiability and validation fallback for mild optical drift."""
import json
import pickle
from pathlib import Path
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


def test_stale_origin_is_reanchored_to_the_measured_zero(tmp_path):
    """A warm checkpoint's origin must not deadlock the fit after a realignment.

    ``load_dataset`` centres the camera window on the **measured** zero order of
    the current capture, so a checkpoint that still carries the previous bench's
    origin displaces the predicted field: the NNLS radiometry clips the gain to
    zero, the prediction becomes a constant and every gradient vanishes. The
    origin is therefore re-anchored from the measurement before the first
    forward pass (2026-10-27: a 76 px bench shift produced exactly that).
    """
    pytest.importorskip("torch")
    from ao_shaping.tools.slm.transfer.simulation import generate_dataset

    dataset = tmp_path / "synthetic"
    checkpoint = generate_dataset(dataset, power=1., count=6)
    stale = json.loads(Path(checkpoint).read_text())
    stale["origin"] = [stale["origin"][0] + 40., stale["origin"][1] - 25.]
    stale_path = tmp_path / "stale.json"
    stale_path.write_text(json.dumps(stale), encoding="utf-8")

    fit.fit(SimpleNamespace(dataset=dataset, checkpoint=stale_path, out=tmp_path / "fit",
                            iterations=2, span=120, device="cpu", adaptation="alignment",
                            radiometry="exposure", samples=None))
    result = json.loads((tmp_path / "fit" / "result.json").read_text())
    assert result["parameters"]["gain"] > 0
    correlations = [m["correlation"] for m in result["metrics"]]
    assert np.isfinite(correlations).all()


def test_zero_gain_is_reported_instead_of_deadlocking(tmp_path, monkeypatch):
    """A constant prediction makes the loss independent of the field.

    Then no parameter can move, and the run used to end with an opaque
    "nonfinite validation quality" from a later stage. It has to fail at the
    place that can name the cause.
    """
    pytest.importorskip("torch")
    from ao_shaping.tools.slm.transfer import radiometry
    from ao_shaping.tools.slm.transfer.simulation import generate_dataset

    dataset = tmp_path / "synthetic"
    checkpoint = generate_dataset(dataset, power=1., count=6)

    def zero_gain(optical, measured, exposure, train, *, mode="exposure"):
        return np.asarray(optical) * 0., dict(gain=0., background=1., bias_counts=0.)

    monkeypatch.setattr(radiometry, "fit_radiometry", zero_gain)
    with pytest.raises(ValueError, match="gain collapsed"):
        fit.fit(SimpleNamespace(dataset=dataset, checkpoint=checkpoint, out=tmp_path / "fit",
                                iterations=2, span=120, device="cpu", adaptation="alignment",
                                radiometry="exposure", samples=None))


def test_plane_start_skips_the_model_stage(tmp_path):
    """A cold plane start has no GS phase to verify on the bench.

    The model-only stage exists to confirm a GS seed before the anchored blocks;
    with a plane start it is skipped and the anchored stage begins from the flat
    phase plus the measured steering carrier.
    """
    from ao_shaping.tools.slm.transfer.workflow import TransferSession

    config = TransferSession(tmp_path, shapes=("square",), initial_phase="plane")
    stages = config.plan()["stages"]
    assert "CCD_model_blocks" not in stages
    assert "CCD_anchored_blocks" in stages
    assert TransferSession(tmp_path, shapes=("square",)).plan()["stages"][2:4] == [
        "fit", "CCD_model_blocks"]


def test_cold_plane_start_is_carrier_only():
    from ao_shaping.tools.slm.transfer.cold_start import cold_phase

    class Params:
        grid_size = 8
        ramp_matrix = np.eye(2) * 64.
        origin_xy = (0., 0.)

    class Forward:
        params = Params()
        pitch = 3.

    phase, provenance = cold_phase(Forward(), (0., 0.), kind="plane")
    assert phase.shape == (8, 8)
    assert provenance["shaping_rms_rad"] == 0.0
    assert provenance["gs_calls"] == 0
    assert provenance["static_correction"] is False


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
