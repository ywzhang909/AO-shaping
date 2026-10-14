"""R-47: ``phase_to_panel`` is SLM panel geometry, not a bench measurement kernel.

Characterization test written **before** the move (issue note: "大原子单元，需先补
特征测试"). It pins the exact numeric behaviour of :func:`phase_to_panel` so the
verbatim relocation from ``tools/kernels/bench_kernels`` to
``utils/slm/panel_geometry`` is provably behaviour-preserving.

The naming trap this refactor is named for: ``phase_to_panel`` is a **spatial
placement** op (model-grid phase -> panel-sized radians array, zero outside the
disc). It is NOT the radians->uint16 grayscale conversion in
``utils/slm/phase_display.phase_to_slm_grayscale``. The two must not be merged.
"""

from __future__ import annotations

import numpy as np
import pytest

# Canonical home after R-47 (the utils/slm leaf). The old bench_kernels path is
# asserted to re-export the *same* objects below, so existing callers stay green.
from ao_shaping.utils.slm.bench_constants import SLM_PANEL_H, SLM_PANEL_W
from ao_shaping.utils.slm.panel_geometry import phase_to_panel


def test_panel_shape_and_dtype() -> None:
    """Output is always the full Santec panel, float64 radians."""
    out = phase_to_panel(np.zeros((64, 64)), 32, (960, 600))
    assert out.shape == (SLM_PANEL_H, SLM_PANEL_W) == (1200, 1920)
    assert out.dtype == np.float64


def test_non_square_input_raises() -> None:
    with pytest.raises(ValueError, match="square"):
        phase_to_panel(np.zeros((64, 32)), 32, (960, 600))


def test_zoom_factor_one_places_input_verbatim() -> None:
    """region=64, r=32 -> zoom factor 2r/region = 1.0 -> identity placement.

    This is the strongest exact characterization: with a unit zoom the disc must
    equal the input array bit-for-bit at the centred window.
    """
    rng = np.random.default_rng(0)
    phase = rng.uniform(-np.pi, np.pi, (64, 64))
    r = 32
    cx, cy = 960, 600
    out = phase_to_panel(phase, r, (cx, cy))
    assert np.array_equal(out[cy - r : cy + r, cx - r : cx + r], phase)


def test_zero_outside_disc() -> None:
    """Everything outside the placed window stays exactly zero."""
    rng = np.random.default_rng(1)
    phase = rng.uniform(-np.pi, np.pi, (64, 64))
    r = 100
    cx, cy = 960, 600
    out = phase_to_panel(phase, r, (cx, cy))
    mask = np.zeros_like(out, dtype=bool)
    mask[cy - r : cy + r, cx - r : cx + r] = True
    assert np.count_nonzero(out[~mask]) == 0


def test_constant_phase_preserved_in_disc() -> None:
    """A flat phase stays flat inside the disc (order-1 zoom of a constant)."""
    out = phase_to_panel(np.full((64, 64), 1.5), 100, (960, 600))
    assert np.allclose(out[500:700, 860:1060], 1.5)


def test_off_centre_oversized_disc_clips_in_bounds() -> None:
    """An off-centre disc near the corner is clipped, never raises / wraps."""
    out = phase_to_panel(np.ones((64, 64)), 100, (1900, 1150))
    assert out.shape == (SLM_PANEL_H, SLM_PANEL_W)
    # Placed window is clipped to the panel: x in [1800,1920), y in [1050,1200).
    assert np.count_nonzero(out[:, :1800]) == 0
    assert np.count_nonzero(out[:1050, :]) == 0
    assert np.allclose(out[1050:1200, 1800:1920], 1.0)


def test_bench_kernels_reexports_same_objects() -> None:
    """The old ``bench_kernels`` path must alias the leaf, not copy it.

    13+ modules still import these names from ``bench_kernels``; identity (not
    equality) proves there is a single definition, so a bench change cannot
    drift between the two paths.
    """
    from ao_shaping.tools.kernels import bench_kernels

    assert bench_kernels.phase_to_panel is phase_to_panel
    assert bench_kernels.SLM_PANEL_H is SLM_PANEL_H
    assert bench_kernels.SLM_PANEL_W is SLM_PANEL_W


def test_panel_geometry_is_a_leaf() -> None:
    """``panel_geometry`` may import numpy/scipy/utils only -- never tools/drivers.

    This is the R-47 point: ``optimizer/`` reaches panel geometry without
    pulling in ``tools/``. AST (not grep) so a deferred in-function import of a
    driver cannot slip past.
    """
    import ast
    import pathlib

    import ao_shaping.utils.slm.panel_geometry as panel_geometry

    tree = ast.parse(pathlib.Path(panel_geometry.__file__).read_text())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    for mod in imported:
        assert not mod.startswith("ao_shaping.tools"), mod
        assert not mod.startswith("ao_shaping.drivers"), mod
