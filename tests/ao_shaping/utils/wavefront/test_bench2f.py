"""Physics-contract tests for the bench-2f forward (issue #165, Module 1).

Every MUST of the spec's physics contract is pinned here: the pupil
amplitude formula and hard mask, the centred-FFT forward chain (asserted
against a hand-written torch expression), differentiability, the Zernike
reduction actually reducing, coefficient recovery, the ground-truth twin's
extra physics and its optionality, determinism, JSON loading, and the GS
back step.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from ao_shaping.utils.wavefront.bench2f import (
    Bench2FGeometry,
    Bench2FPropagator,
    Bench2FGroundTruth,
    ZernikeAberrationField,
)

REGION = 64
FAR = 256


def _geometry(**overrides) -> Bench2FGeometry:
    kwargs = {"region": REGION, "far_field_size": FAR}
    kwargs.update(overrides)
    return Bench2FGeometry(**kwargs)


def _span_phase(propagator: Bench2FPropagator, batch: int = 1) -> torch.Tensor:
    """A phase that IS a Zernike field (inside the reduction span)."""
    rng = np.random.default_rng(3)
    coeffs = torch.tensor(rng.normal(0.0, 0.3, (batch, propagator.field.n_terms)))
    return propagator.field.render(coeffs)


# ----------------------------------------------------------------------
# Bench2FGeometry
# ----------------------------------------------------------------------


def test_geometry_defaults_and_validation() -> None:
    geo = Bench2FGeometry()
    assert geo.region == 64 and geo.far_field_size == 512
    assert geo.gain == 1.0 and geo.background == 0.0
    for bad in ({"region": 63}, {"region": 8}, {"far_field_size": 8},
                {"panel_disc_radius_px": 0.0}):
        try:
            Bench2FGeometry(**bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


def test_from_bench_geometry_json(tmp_path) -> None:
    payload = {
        # the keys BenchGeometry.to_json emits, plus one unknown key
        "panel_disc_radius": 470,
        "region": 128,
        "beam_waist_panel_px": 460.0,
        "far_field_size": 1024,
        "camera_px_per_model_px": 7.5,
        "spot_fwhm_camera_px": 12.0,
        "spot_fwhm_model_px": 1.6,
        "correlation": 0.07,
        "method": "speckle",
        "calibration_notes": "ignored",
        "totally_unknown_future_key": 42,
    }
    path = tmp_path / "bench_geometry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    geo = Bench2FGeometry.from_bench_geometry_json(path)
    assert geo.panel_disc_radius_px == 470
    assert geo.region == 128
    assert geo.beam_waist_panel_px == 460.0
    assert geo.far_field_size == 1024
    assert geo.camera_px_per_model_px == 7.5
    # gain / background are not in the schema -> defaults
    assert geo.gain == 1.0 and geo.background == 0.0
    # overrides win
    geo2 = Bench2FGeometry.from_bench_geometry_json(path, gain=4.0, background=0.1)
    assert geo2.gain == 4.0 and geo2.background == 0.1
    # missing keys keep dataclass defaults
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"region": 64}), encoding="utf-8")
    geo3 = Bench2FGeometry.from_bench_geometry_json(partial)
    assert geo3.far_field_size == 512 and geo3.panel_disc_radius_px == 450.0


def test_from_bench_geometry_json_errors(tmp_path) -> None:
    missing = tmp_path / "nope.json"
    try:
        Bench2FGeometry.from_bench_geometry_json(missing)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing file must raise FileNotFoundError")
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    try:
        Bench2FGeometry.from_bench_geometry_json(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("malformed JSON must raise ValueError")


# ----------------------------------------------------------------------
# pupil amplitude (physics contract #1)
# ----------------------------------------------------------------------


def test_pupil_amplitude_waist_mask_and_peak() -> None:
    # beam_waist_panel_px * region / (2 * panel_disc_radius_px) = 225*64/600 = 24
    geo = _geometry(panel_disc_radius_px=300.0, beam_waist_panel_px=225.0)
    prop = Bench2FPropagator(geo)
    amp = prop.pupil_amplitude()
    assert amp.shape == (1, 1, REGION, REGION)
    assert amp.dtype is torch.float32
    axis = np.arange(REGION) - REGION / 2.0
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    rr = np.hypot(xx, yy)
    expected = np.exp(-2.0 * rr**2 / 24.0**2) * (rr <= REGION / 2.0)
    assert torch.allclose(amp[0, 0], torch.tensor(expected, dtype=torch.float32),
                          atol=1e-6)
    # hard mask: exactly zero outside radius region/2
    outside = rr > REGION / 2.0
    assert (amp[0, 0][outside] == 0).all()
    # peak == 1
    assert amp.max().item() == 1.0


# ----------------------------------------------------------------------
# forward chain (physics contract #2) + differentiability (contract #3)
# ----------------------------------------------------------------------


def _handwritten_chain(propagator: Bench2FPropagator, phase: torch.Tensor,
                       gain: float, background: float,
                       reduce: bool = True) -> torch.Tensor:
    """Independent inline copy of the padded centred-FFT chain."""
    region = propagator.geometry.region
    out = propagator.geometry.far_field_size
    phi = propagator.reduce_phase(phase) if reduce else phase
    u = propagator.pupil_amplitude() * torch.exp(1j * phi)
    n = u.shape[-1]
    if out > n:
        half = (out - n) // 2  # odd remainder -> extra row/column right/bottom
        u = torch.nn.functional.pad(
            u, (half, out - n - half, half, out - n - half)
        )
    spectrum = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(u, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )
    intensity = gain * (spectrum.real**2 + spectrum.imag**2) + background
    return intensity / intensity.amax(dim=(-2, -1), keepdim=True)


def test_forward_matches_handwritten_chain_odd_pad() -> None:
    # far_field_size with an ODD pad remainder pins the right/bottom rule
    geo = _geometry(far_field_size=REGION + 69)  # (69)//2=34 left, 35 right
    prop = Bench2FPropagator(geo)
    phase = _span_phase(prop, batch=2)
    out = prop.forward(phase)
    expected = _handwritten_chain(prop, phase, gain=1.0, background=0.0)
    assert out.shape == (2, 1, REGION + 69, REGION + 69)
    assert torch.allclose(out, expected, atol=1e-6)
    # peak-normalised: max == 1 within 1e-6
    assert torch.allclose(
        out.amax(dim=(-2, -1)), torch.ones(2, dtype=out.dtype), atol=1e-6
    )


def test_forward_gain_background_affine() -> None:
    geo = _geometry(gain=3.0, background=0.25)
    prop = Bench2FPropagator(geo)
    phase = _span_phase(prop)
    out = prop.forward(phase)
    expected = _handwritten_chain(prop, phase, gain=3.0, background=0.25)
    assert torch.allclose(out, expected, atol=1e-6)
    # background raises the floor above the background-free floor
    plain = Bench2FPropagator(_geometry()).forward(phase)
    assert out.amin().item() > plain.amin().item()


def test_forward_is_differentiable() -> None:
    prop = Bench2FPropagator(_geometry())
    rng = torch.Generator().manual_seed(7)
    phase = torch.randn(1, 1, REGION, REGION, generator=rng)
    # break any residual symmetry
    phase[..., : REGION // 2, :] += 0.7
    phase.requires_grad_(True)
    out = prop.forward(phase)
    out.sum().backward()
    assert phase.grad is not None
    assert torch.isfinite(phase.grad).all()
    assert phase.grad.abs().max().item() > 0.0


# ----------------------------------------------------------------------
# ZernikeAberrationField + reduce_phase (physics contract #4)
# ----------------------------------------------------------------------


def test_field_recovers_known_coefficients() -> None:
    field = ZernikeAberrationField(REGION, n_max=10)
    assert field.n_terms == 66  # calc_n_zernike_terms(10), piston included
    basis = field.basis()
    assert basis.shape == (66, REGION, REGION)
    assert basis.dtype is torch.float32
    rng = np.random.default_rng(11)
    coeffs = torch.tensor(rng.normal(0.0, 0.3, (3, 66)))
    phase = field.render(coeffs)
    recovered = field.fit(phase)
    assert recovered.shape == (3, 66)
    assert (recovered - coeffs).abs().max().item() < 1e-4


def test_reduce_phase_really_reduces() -> None:
    prop = Bench2FPropagator(_geometry(), n_max=10)
    # a Zernike field: residual must vanish
    phase = _span_phase(prop)
    residual = phase - prop.reduce_phase(phase)
    assert residual.pow(2).mean().sqrt().item() < 1e-4
    # a random per-pixel phase: residual stays large
    rng = torch.Generator().manual_seed(5)
    random_phase = 0.5 * torch.randn(1, 1, REGION, REGION, generator=rng)
    residual = random_phase - prop.reduce_phase(random_phase)
    assert residual.pow(2).mean().sqrt().item() > 0.05


# ----------------------------------------------------------------------
# back_propagate
# ----------------------------------------------------------------------


def test_back_propagate_shape_and_finiteness() -> None:
    prop = Bench2FPropagator(_geometry())
    rng = torch.Generator().manual_seed(9)
    far = (torch.randn(2, 1, FAR, FAR, generator=rng)
           + 1j * torch.randn(2, 1, FAR, FAR, generator=rng))
    pupil = prop.back_propagate(far)
    assert pupil.shape == (2, 1, REGION, REGION)
    assert pupil.is_complex()
    assert torch.isfinite(pupil).all()


def test_back_propagate_is_forward_adjoint() -> None:
    """Feeding the centred forward's complex field back recovers the pupil.

    The two shifts cancel: ``ifft2(ifftshift(focal_field(U, S)))`` returns
    ``U`` (on the padded grid) scaled by the ortho/un-normalised-FFT factor,
    so the cropped window equals ``U * (region / S)**2 / S`` — the adjoint up
    to the pinned padding normalisation.
    """
    prop = Bench2FPropagator(_geometry())
    u = prop.pupil_amplitude() * torch.exp(1j * _span_phase(prop))
    far = torch.fft.fftshift(
        torch.fft.fft2(
            torch.fft.ifftshift(
                torch.nn.functional.pad(
                    u, ((FAR - REGION) // 2,) * 4,
                ),
                dim=(-2, -1),
            ),
            norm="ortho",
        ),
        dim=(-2, -1),
    )
    back = prop.back_propagate(far)
    scale = (REGION / FAR) ** 2 / FAR
    assert torch.allclose(back, u * scale, atol=1e-8)
    assert back.abs().max().item() > 1e-5


# ----------------------------------------------------------------------
# Bench2FGroundTruth (the digital twin)
# ----------------------------------------------------------------------


def _freeform(seed: int = 13, scale: float = 1.0) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    return scale * torch.randn(REGION, REGION, generator=rng)


def test_ground_truth_adds_physics_outside_the_reduced_model() -> None:
    geo = _geometry()
    prop = Bench2FPropagator(geo)
    gt = Bench2FGroundTruth(geo, freeform=_freeform())  # default super-Gaussian
    rng = torch.Generator().manual_seed(17)
    phase = 0.5 * torch.randn(1, 1, REGION, REGION, generator=rng)
    diff = (gt.forward(phase) - prop.forward(phase)).abs().max().item()
    # far above the 1e-6 numerical noise floor
    assert diff > 1e-3


def test_ground_truth_extra_physics_is_optional() -> None:
    geo = _geometry()
    prop = Bench2FPropagator(geo)
    gt = Bench2FGroundTruth(geo, illumination_exponent=2.0)  # freeform None
    phase = _span_phase(prop, batch=2)
    out_gt = gt.forward(phase)
    out_prop = prop.forward(phase)
    assert out_gt.shape == out_prop.shape
    assert (out_gt - out_prop).abs().max().item() < 1e-6


def test_freeform_is_not_in_the_zernike_span() -> None:
    field = ZernikeAberrationField(REGION, n_max=10)
    freeform = _freeform()
    residual = freeform.unsqueeze(0).unsqueeze(0) - field.render(
        field.fit(freeform.unsqueeze(0).unsqueeze(0))
    )
    assert residual.pow(2).mean().sqrt().item() > 0.05


def test_ground_truth_determinism_same_seed() -> None:
    geo = _geometry()
    phase = _span_phase(Bench2FPropagator(geo))
    kwargs = dict(freeform=_freeform(), shot_noise=0.05, read_noise=0.01, seed=42)
    a = Bench2FGroundTruth(geo, **kwargs).forward(phase)
    b = Bench2FGroundTruth(geo, **kwargs).forward(phase)
    assert torch.equal(a, b)
    noiseless = Bench2FGroundTruth(
        geo, freeform=_freeform(), illumination_exponent=2.6, seed=42
    ).forward(phase)
    assert (a - noiseless).abs().max().item() > 0.0  # noise really applied


def test_forward_from_coefficients_splits_span_and_residual() -> None:
    prop = Bench2FPropagator(_geometry())
    coeffs = torch.tensor(
        np.random.default_rng(21).normal(0.0, 0.3, (1, prop.field.n_terms))
    )
    span = prop.field.render(coeffs)
    rng = torch.Generator().manual_seed(23)
    residual = 0.3 * torch.randn(1, 1, REGION, REGION, generator=rng)
    out = prop.forward_from_coefficients(coeffs, residual)
    assert out.shape == (1, 1, FAR, FAR)
    # equals the plain (hand-written) chain on the summed phase: the Zernike
    # reduction is NOT re-applied, the out-of-span residual is kept
    plain = _handwritten_chain(prop, span + residual, gain=1.0,
                               background=0.0, reduce=False)
    assert (out - plain).abs().max().item() < 1e-6
    # and differs from the reducing forward, which would discard the residual
    assert (out - prop.forward(span + residual)).abs().max().item() > 1e-4
