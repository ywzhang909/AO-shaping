import numpy as np
import pytest

from ao_shaping.utils.image.model_response import summarize_response


def test_real_response_reports_range_and_nonfinite():
    result = summarize_response(np.array([0., 1., 2., np.nan]))
    assert result["finite_fraction"] == .75
    assert result["mean"] == 1
    assert result["zero_fraction"] == pytest.approx(1/3)
    assert result["percentiles"][1] == 1


def test_complex_field_is_magnitude_not_intensity():
    result = summarize_response(np.array([3+4j, 0j]))
    assert result["complex"]
    assert result["mean"] == 2.5
    assert result["real_mean"] == 1.5
    assert result["rms"] == pytest.approx(np.sqrt(12.5))


def test_tensor_does_not_retain_autograd():
    torch = pytest.importorskip("torch")
    value = torch.tensor([-2., 3.], requires_grad=True)
    result = summarize_response(value)
    assert result["min"] == -2
    assert value.requires_grad
    with pytest.raises(ValueError):
        summarize_response(np.array([]))


def test_physical_telemetry_preserves_prediction_and_gradient():
    torch = pytest.importorskip("torch")
    from ao_shaping.algorithm.signal_processing.physical_forward_fit import PhysicalForwardFit

    yy, xx = np.mgrid[-2:3, -2:3]
    xy = np.stack((xx+1360, yy+1020), axis=-1)
    model = PhysicalForwardFit(np.random.default_rng(7).normal(size=(2, 8, 8)),
        np.ones((2, 5, 5)), xy, np.diag([10000., 10000.]), 0., [True, False],
        [960., 600.], fft_size=32)
    before = model.prediction()
    gradient_before = torch.autograd.grad(before.sum(), model.d)[0]
    events = []
    model.diagnostic_callback = lambda name, value, start: events.append(
        (name, start, summarize_response(value)))
    after = model.prediction()
    gradient_after = torch.autograd.grad(after.sum(), model.d)[0]
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    torch.testing.assert_close(gradient_before, gradient_after, rtol=0, atol=0)
    assert {e[0] for e in events} == {"input_amplitude", "panel_phase", "panel_field",
                                     "propagated_intensity", "ccd_integrated"}
