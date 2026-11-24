"""Optional bounded geometry residual for the transfer forward fit."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ao_shaping.algorithm.signal_processing.physical_forward_fit import PhysicalForwardFit
from ao_shaping.tools.slm.transfer import fit as fit_module


def _model(*, learn_geometry: bool = False) -> PhysicalForwardFit:
    yy, xx = np.mgrid[-2:3, -2:3]
    camera_xy = np.stack((xx + 1360, yy + 1020), axis=-1)
    return PhysicalForwardFit(
        np.zeros((2, 8, 8), dtype=np.float32), np.ones((2, 5, 5), dtype=np.float32),
        camera_xy, np.diag([10000.0, 12000.0]), 0.0, [True, False], [960.0, 600.0],
        fft_size=32, learn_geometry=learn_geometry,
    )


def test_geometry_is_disabled_without_changing_inverse() -> None:
    model = _model()
    assert model.geometry_raw is None
    assert model.effective_inverse() is model.inverse
    np.testing.assert_array_equal(model.geometry_value(), np.eye(2))


def test_learned_geometry_is_bounded_differentiable_and_uses_camera_frame_formula() -> None:
    import torch

    model = _model(learn_geometry=True)
    assert model.geometry_raw is not None and model.geometry_raw.requires_grad
    model.geometry_raw.data.copy_(torch.tensor([[0.25, -0.5], [0.75, -1.0]]))
    correction = model.geometry_correction()
    assert torch.all((correction - torch.eye(2)).abs() <= 0.05)
    torch.testing.assert_close(model.effective_inverse(), model.inverse @ correction)
    model.geometry_raw.data.fill_(100.0)
    correction = model.geometry_correction()
    assert torch.all((correction - torch.eye(2)).abs() <= 0.05)
    from ao_shaping.tools.slm.transfer.adaptation import configure_adaptation

    configure_adaptation(model, "alignment", learn_geometry=True)
    assert model.geometry_raw.requires_grad
    configure_adaptation(model, "alignment", learn_geometry=False)
    assert not model.geometry_raw.requires_grad


def test_geometry_uses_calibration_ramps_only() -> None:
    expected = np.array([[10.0, 7400.0], [-7600.0, -5.0]])
    rows = []
    for column, axis in enumerate(("x", "y")):
        for sign in (-1, 1):
            spot = np.array([1000.0, 900.0]) + sign * expected[:, column] / 64
            rows.append({"label": f"ramp_{axis}_64_{sign}", "split": "calibration",
                         "branch": "cal", "spot_xy": spot.tolist()})
            # A held-out calibration-looking pair with very different geometry
            # must not affect the estimate.
            rows.append({"label": f"ramp_{axis}_64_{sign}", "split": "validation",
                         "branch": "heldout", "spot_xy": (spot * 2).tolist()})
    matrix, _ = fit_module.geometry(rows)
    np.testing.assert_allclose(matrix, expected)
    with pytest.raises(ValueError, match="both ramp axes required"):
        fit_module.geometry([row for row in rows if row["split"] != "calibration"])
    # A capture manifest before the fitter assigns splits declares no ``split``
    # key at all; those rows stay usable, exactly as before this change.
    undeclared = [{key: value for key, value in row.items() if key != "split"}
                  for row in rows if row["split"] == "calibration"]
    matrix, _ = fit_module.geometry(undeclared)
    np.testing.assert_allclose(matrix, expected)


def test_cli_and_fit_outputs_geometry_provenance(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("[]", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{}", encoding="utf-8")
    out = tmp_path / "fit-out"

    parsed = {}
    real_fit = fit_module.fit
    monkeypatch.setattr("sys.argv", ["fit.py", "--dataset", str(dataset), "--checkpoint",
                                      str(checkpoint), "--out", str(out), "--learn-geometry"])
    monkeypatch.setattr(fit_module, "fit", lambda args: parsed.update(vars(args)))
    fit_module.main()
    assert parsed["learn_geometry"] is True

    # Run the output-writing path with the fit loop replaced: no model training,
    # GPU, or capture data is needed to pin the output schema and matrix export.
    matrix = np.array([[10.0, 2.0], [1.0, 8.0]])
    correction = np.array([[1.02, 0.01], [-0.02, 0.98]])

    class FakeModel:
        _device = SimpleNamespace(type="cpu")
        optimizer = SimpleNamespace(zero_grad=lambda: None)
        registration_value = None

        @staticmethod
        def geometry_value():
            return correction

    comps = SimpleNamespace(
        model=FakeModel(), rows=[], phases=np.zeros((1, 8, 8)), images=np.zeros((1, 5, 5)),
        xy=np.zeros((5, 5, 2)), train=np.array([True]), matrix=matrix,
        ramp_samples={}, old=SimpleNamespace(grid_size=8, fft_size=32, pixel_samples=1,
                                             origin_xy=(0.0, 0.0)),
        measured_origin=np.zeros(2), exposure_values=np.ones(1), kernel=None,
    )
    monkeypatch.setattr(fit_module, "build_fit_model", lambda *args, **kwargs: comps)

    def fake_loss(_model, _comps, _radiometry, iteration):
        entry = {
            "iteration": iteration, "loss": 1.0,
            "parameters": {"gain": 1.0, "defocus_diopter": 0.0},
            "learned_geometry_correction": correction.tolist(),
            "base_ramp_matrix": matrix.tolist(),
            "effective_ramp_matrix": fit_module.effective_ramp_matrix(matrix, correction).tolist(),
        }
        return __import__("torch").tensor(1.0), entry, np.zeros((1, 5, 5))

    monkeypatch.setattr(fit_module, "fit_loss", fake_loss)
    fit_module.fit = real_fit
    args = SimpleNamespace(
        adaptation="full", dataset=dataset, checkpoint=checkpoint, out=out,
        span=20, iterations=0, patience=8, min_delta=1e-4, device=None,
        amplitude_center_lr=None, radiometry="exposure", samples=None, train_all=False,
        record_responses=False, debug_weights=False, learn_geometry=True,
    )
    fit_module.fit(args)
    provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    frozen = json.loads((out / "frozen_parameters.json").read_text(encoding="utf-8"))
    expected = np.linalg.inv(correction) @ matrix
    np.testing.assert_allclose(provenance["base_ramp_matrix"], matrix)
    np.testing.assert_allclose(provenance["learned_geometry_correction"], correction)
    np.testing.assert_allclose(provenance["effective_ramp_matrix"], expected)
    np.testing.assert_allclose(result["effective_ramp_matrix"], expected)
    np.testing.assert_allclose(frozen["ramp_matrix"], expected)
