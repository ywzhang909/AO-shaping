"""Tests for :mod:`ml.sim2f.model` -- the phasor U-Net.

No optics and no cache: the network is built from a dataclass configuration and
an explicit context tensor, so these tests run while ``ml.sim2f.optics`` is
still being written by a parallel change.  ``torch`` lives in the optional
``ml`` dependency group, hence the ``importorskip``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch lives in the optional ml group")

from dataclasses import fields, replace

from ml.sim2f.config import Sim2fConfig, TrainConfig
from ml.sim2f.model import Sim2fUNet, build_unet, network_input

_GRID = 64
_CCD = 64


def _sim_config(**overrides: object) -> Sim2fConfig:
    """Build a :class:`Sim2fConfig` from field overrides only.

    The twin-error block and the detector fields are the calibrated twin's, so
    the tests name only the geometry they need and filter the rest through the
    dataclass.  A field that is renamed or removed turns into "not applied"
    instead of a collection-time ``TypeError``.

    Args:
        **overrides: Field values to apply.

    Returns:
        The configuration.
    """
    known = {item.name for item in fields(Sim2fConfig)}
    values: dict[str, object] = {"grid": _GRID, "ccd_px": _CCD}
    values.update(overrides)
    return replace(
        Sim2fConfig(), **{key: value for key, value in values.items() if key in known}
    )


def _context(size: int = _CCD) -> torch.Tensor:
    """A stand-in flat-phase frame: one bright central pixel.

    Args:
        size: Camera grid side.

    Returns:
        ``(1, 1, size, size)`` with a single hot pixel, i.e. roughly what an
        aberration-free bench produces.
    """
    frame = torch.zeros(1, 1, size, size)
    frame[0, 0, size // 2, size // 2] = 10.0
    return frame


def _model(**overrides: object) -> Sim2fUNet:
    """Build a small network on the toy grid.

    Args:
        **overrides: Keyword arguments forwarded to :class:`Sim2fUNet`.

    Returns:
        The network.
    """
    kwargs = {
        "width": 8,
        "depth": 2,
        "grid": _GRID,
        "ccd_px": _CCD,
        "context": _context(),
    }
    kwargs.update(overrides)
    return Sim2fUNet(**kwargs)  # type: ignore[arg-type]


def test_forward_shape_modulus_and_aperture() -> None:
    """The output is a unit phasor, free-space outside the pupil."""
    model = _model()
    batch = 3
    output = model(torch.rand(batch, 2, _CCD, _CCD))

    assert output.shape == (batch, 2, _GRID, _GRID)
    modulus = output.pow(2).sum(dim=1).sqrt()
    assert torch.allclose(modulus, torch.ones_like(modulus), rtol=1e-5, atol=1e-6)

    outside = (model.pupil_mask == 0).expand_as(output[:, 0:1])
    assert torch.allclose(output[:, 0:1][outside], torch.ones_like(output[:, 0:1][outside]))
    assert torch.equal(output[:, 1:2][outside], torch.zeros_like(output[:, 1:2][outside]))
    assert bool((model.pupil_mask > 0).any())


def test_grid_differs_from_camera_grid() -> None:
    """A 2x pupil/camera ratio resamples the head output instead of failing."""
    model = _model(ccd_px=32)
    output = model(torch.rand(2, 2, 32, 32))

    assert output.shape == (2, 2, _GRID, _GRID)
    modulus = output.pow(2).sum(dim=1).sqrt()
    assert torch.allclose(modulus, torch.ones_like(modulus), rtol=1e-5, atol=1e-6)


def test_gradient_reaches_every_parameter() -> None:
    """Every parameter receives a finite, non-zero gradient."""
    model = _model()
    output = model(torch.rand(2, 2, _CCD, _CCD))
    output.pow(2).sum().backward()

    parameters = list(model.parameters())
    assert parameters
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().max()) > 0.0
    assert model.n_parameters() == sum(p.numel() for p in parameters)
    assert model.n_parameters() > 0


def test_context_is_peak_normalised_and_broadcast() -> None:
    """The stored context is peak-normalised; ``network_input`` broadcasts it."""
    model = _model()
    assert float(model.flat_psf.amax()) == pytest.approx(1.0)
    assert model.flat_psf.shape == (1, 1, _CCD, _CCD)

    target = torch.zeros(4, 1, _CCD, _CCD)
    frames = network_input(target, model.flat_psf)
    assert frames.shape == (4, 2, _CCD, _CCD)
    assert torch.equal(frames[:, 1:2], model.flat_psf.expand(4, -1, -1, -1))
    with pytest.raises(ValueError, match="grid"):
        network_input(target, torch.zeros(1, 1, _CCD // 2, _CCD // 2))


def test_missing_context_is_an_empty_buffer() -> None:
    """Without a context the buffer is empty rather than zeros of a wrong size."""
    model = _model(context=None)
    assert model.flat_psf.numel() == 0
    assert model(torch.rand(1, 2, _CCD, _CCD)).shape == (1, 2, _GRID, _GRID)


def test_build_unet_rejects_bad_depth_budget() -> None:
    """``ccd_px % 2 ** depth != 0`` fails at build time with a useful message."""
    cfg = TrainConfig(width=8, depth=5, steps=1, val_every=1)
    with pytest.raises(ValueError, match="divisible"):
        build_unet(cfg, _sim_config(ccd_px=48), device="cpu", context=_context(48))


def test_build_unet_uses_the_supplied_context() -> None:
    """``build_unet`` wires the bench geometry and the context through."""
    cfg = TrainConfig(width=8, depth=2, steps=1, val_every=1)
    context = _context()
    model = build_unet(cfg, _sim_config(), device="cpu", context=context)

    assert model.grid == _GRID
    assert model.ccd_px == _CCD
    assert model.depth == 2
    assert model.width == 8
    assert torch.allclose(model.flat_psf, context / context.amax())
    assert next(model.parameters()).device.type == "cpu"


def test_forward_rejects_wrong_input_arity() -> None:
    """A wrongly shaped input is a clear error, not a silent broadcast."""
    model = _model()
    with pytest.raises(ValueError, match="expects"):
        model(torch.rand(2, 1, _CCD, _CCD))
