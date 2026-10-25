"""Measured uniformity and composite acceptance must remain distinguishable."""
import pytest
import numpy as np

from ao_shaping.tools.slm.transfer.paired import paired_decision


def row(cv, loss):
    return dict(metrics=dict(cv=cv), loss=loss)


def test_cv_acceptance_can_prioritize_uniformity_over_composite_objective():
    a, b = row(.35, 1.5), row(.32, 1.6)
    assert paired_decision(a, b, b, a, metric="cv")["accepted"]
    assert not paired_decision(a, b, b, a, metric="loss")["accepted"]


def test_cv_improvement_smaller_than_reference_drift_is_rejected():
    assert not paired_decision(row(.35, 1), row(.34, 1), row(.34, 1), row(.39, 1), metric="cv")["accepted"]
    with pytest.raises(ValueError, match="nonfinite"):
        paired_decision(row(float("nan"), 1), row(.3, 1), row(.3, 1), row(.3, 1), metric="cv")


def test_cv_only_objective_is_invariant_to_energy_outside_fixed_core():
    torch = pytest.importorskip("torch")
    from ao_shaping.algorithm.signal_processing.multishape_objective import ShapeObjective
    target = np.zeros((15, 15))
    target[3:12, 3:12] = 1
    cv = ShapeObjective(target, cv_only=True)
    composite = ShapeObjective(target)
    a = torch.ones((15, 15), requires_grad=True)
    b = a.detach().clone()
    b[~torch.tensor(target.astype(bool))] = 20
    phase = torch.zeros((8, 8))
    assert cv.evaluate(a, phase)[0].item() == cv.evaluate(b, phase)[0].item()
    assert composite.evaluate(a, phase)[0].item() != composite.evaluate(b, phase)[0].item()
