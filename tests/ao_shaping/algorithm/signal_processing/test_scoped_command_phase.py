"""Unit tests for the phase_sign scope helpers (issue #235)."""
from __future__ import annotations

import pytest
import torch

from ao_shaping.algorithm.signal_processing.defocus_forward_core import (
    PHASE_SIGN_SCOPES,
    scoped_command_phase,
    tilt_plane_split,
)


def _grid(n: int = 16):
    y, x = torch.meshgrid(torch.arange(n, dtype=torch.float64),
                          torch.arange(n, dtype=torch.float64), indexing="ij")
    return x, y


def test_command_scope_is_exact_negation():
    x, y = _grid()
    total = torch.sin(x / 3.0) + 0.1 * y
    out = scoped_command_phase(total, x, y, phase_sign=-1.0, scope="command")
    assert torch.equal(out, -total)


def test_positive_phase_sign_is_identity_for_every_scope():
    x, y = _grid()
    total = torch.cos(y / 2.0)
    for scope in PHASE_SIGN_SCOPES:
        assert torch.equal(scoped_command_phase(total, x, y, phase_sign=1.0, scope=scope), total)


def test_tilt_plane_split_is_reversible():
    x, y = _grid()
    total = 0.3 * x - 0.2 * y + torch.sin(x) * torch.cos(y)
    plane, residual = tilt_plane_split(total, x, y)
    assert torch.allclose(plane + residual, total, atol=1e-12)


def test_aberration_scope_flips_plane_and_residual_differently():
    x, y = _grid()
    total = 0.5 * x + 0.1 * torch.sin(y)
    plane, residual = tilt_plane_split(total, x, y)
    out = scoped_command_phase(total, x, y, phase_sign=-1.0, scope="aberration")
    assert torch.allclose(out, plane - residual, atol=1e-12)


def test_focus_scope_subtracts_twice_focus_and_never_adds_to_total():
    x, y = _grid()
    total = 0.2 * x + torch.cos(y)
    focus = 0.05 * torch.sin(x * y / 10.0)
    base = scoped_command_phase(total, x, y, phase_sign=-1.0, scope="aberration")
    out = scoped_command_phase(total, x, y, phase_sign=-1.0, scope="aberration_focus",
                               focus_component=focus)
    assert torch.allclose(out, base - 2 * focus, atol=1e-12)


def test_carrier_scope_without_component_equals_aberration():
    x, y = _grid()
    total = 0.4 * y + torch.sin(x)
    a = scoped_command_phase(total, x, y, phase_sign=-1.0, scope="aberration")
    c = scoped_command_phase(total, x, y, phase_sign=-1.0, scope="aberration_carrier")
    assert torch.allclose(a, c, atol=1e-12)


def test_invalid_scope_raises():
    x, y = _grid()
    with pytest.raises(ValueError, match="phase_sign_scope"):
        scoped_command_phase(x, x, y, phase_sign=-1.0, scope="bogus")
