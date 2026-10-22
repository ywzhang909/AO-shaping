"""Tests for the pred-vs-true montage in :mod:`ml.shaping.viz`.

No hardware, no ``.pkl``: the network and the forward bank are tiny stand-ins and
the batch is built by hand.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch lives in the optional ml group")

from ml.shaping.viz import (
    PANEL_KEYS_FULL,
    PANEL_KEYS_PHASE_ONLY,
    build_comparison,
    render_comparison,
)

GRID = 8


def _batch(fov: list[int], seed: int = 0) -> dict[str, Any]:
    """A minimal shaping batch with a known target phasor.

    Args:
        fov: Per-sample FOV written into ``fov_px``.
        seed: RNG seed for the image.

    Returns:
        A batch dict shaped like :func:`ml.shaping.dataset.shaping_collate`.
    """
    generator = torch.Generator().manual_seed(seed)
    n = len(fov)
    angle = torch.rand((n, GRID, GRID), generator=generator) * (2 * math.pi)
    return {
        "image": torch.rand((n, 1, GRID, GRID), generator=generator),
        "phase_cos": torch.cos(angle).unsqueeze(1),
        "phase_sin": torch.sin(angle).unsqueeze(1),
        "fov_px": torch.tensor(fov, dtype=torch.int64),
        "exposure_log10": torch.zeros((n, 1)),
    }


class _Identity(torch.nn.Module):
    """A network that returns the true phasor exactly, for exactness tests."""

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        del image
        return torch.cat(
            (
                self.target_cos,
                self.target_sin,
            ),
            dim=1,
        )


class _Zero(torch.nn.Module):
    """A network returning a fixed phasor, so the error is known analytically."""

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((image.shape[0], 2, GRID, GRID))
        out[:, 0] = 1.0
        return out


class _StubBank:
    """A forward bank that marks a fixed set of FOVs as uncalibrated."""

    def __init__(self, supported: set[int]) -> None:
        self.supported = supported

    def forward(self, phase: torch.Tensor, fov: torch.Tensor):
        excluded = torch.tensor(
            [int(v) not in self.supported for v in fov.tolist()],
            dtype=torch.bool,
        )
        pred = torch.ones((phase.shape[0], 1, GRID, GRID))
        return pred, excluded


def _figure(network: torch.nn.Module, fov: list[int], supported: set[int], **kw: Any):
    """Build a comparison figure from the stand-ins above."""
    batch = _batch(fov)
    target_cos = batch["phase_cos"]
    target_sin = batch["phase_sin"]
    network.target_cos = target_cos
    network.target_sin = target_sin
    return build_comparison(
        network,
        batch,
        _StubBank(supported),
        device=torch.device("cpu"),
        grid=GRID,
        in_channels=2,
        **kw,
    )


# ----------------------------------------------------------------------
# Panel content
# ----------------------------------------------------------------------
def test_full_panels_appear_when_fov_is_calibrated() -> None:
    """A calibrated batch gets the far-field columns."""
    figure = _figure(_Identity(), [248, 248], {248})
    assert figure.panel_keys == PANEL_KEYS_FULL
    assert len(figure.rows) == 2
    for row in figure.rows:
        assert set(row) == set(PANEL_KEYS_FULL)


def test_far_field_panels_drop_when_nothing_is_calibrated() -> None:
    """No calibrated FOV => phase-only columns, never zero-filled far fields.

    Zero-filling would render as a black prediction, which is indistinguishable
    from a genuinely failed sample.
    """
    figure = _figure(_Identity(), [1944, 1944], set())
    assert figure.panel_keys == PANEL_KEYS_PHASE_ONLY
    for row in figure.rows:
        assert "far_pred" not in row
        assert "far_diff" not in row


def test_uncalibrated_rows_are_dropped_when_some_are_calibrated() -> None:
    """Mixed batch => only the propagatable rows are rendered, and it is counted."""
    figure = _figure(_Identity(), [248, 1944], {248})
    assert figure.panel_keys == PANEL_KEYS_FULL
    assert len(figure.rows) == 1
    assert figure.n_candidates == 2
    assert figure.n_physics_skipped == 1


def test_perfect_prediction_has_zero_phasor_error() -> None:
    """The montage must be a real diagnostic, so an exact model reads as exact."""
    figure = _figure(_Identity(), [248], {248})
    error = figure.rows[0]["phasor_err"]
    assert float(np.max(error)) == pytest.approx(0.0, abs=1e-6)


def test_phasor_error_is_bounded_by_two() -> None:
    """``|unit - unit| <= 2``, so the colour scale has a fixed upper limit."""
    figure = _figure(_Zero(), [248], {248})
    error = figure.rows[0]["phasor_err"]
    # Target phasors are unit vectors; (1, 0) vs a unit vector is never > 2.
    assert float(np.max(error)) <= 2.0 + 1e-6
    assert float(np.max(error)) > 0.0


def test_phasor_error_has_no_branch_cut() -> None:
    """Opposite phases are a *large* phasor error, not a small wrapped one.

    An angle-difference panel would report pi - pi = 0 here, hiding a total
    failure. This is the reason the panel uses the phasor distance.
    """
    network = _Zero()  # outputs phasor (1, 0)
    figure = _figure(network, [248], {248})
    error = figure.rows[0]["phasor_err"]
    # Half the target phasors point away from (1, 0); those must read ~2, not ~0.
    # The target angles are random, so the max lands just under 2 by sampling.
    assert float(np.max(error)) == pytest.approx(2.0, abs=1e-3)


def test_panel_shapes_match_the_grid() -> None:
    """Every panel must be a 2-D ``(grid, grid)`` array for imshow."""
    figure = _figure(_Identity(), [248], {248}, max_samples=1)
    for row in figure.rows:
        for value in row.values():
            assert value.shape == (GRID, GRID)


def test_max_samples_caps_rows() -> None:
    """The row cap is honoured and the remainder is reported, not hidden."""
    figure = _figure(_Identity(), [248] * 5, {248}, max_samples=2)
    assert len(figure.rows) == 2
    assert figure.n_candidates == 5
    # Nothing was dropped for calibration reasons; the shortfall is purely the cap.
    assert figure.n_physics_skipped == 0


def test_drop_exposure_channel_still_renders() -> None:
    """The 1-channel ablation must produce a montage too."""
    batch = _batch([248])
    network = _Identity()
    network.target_cos = batch["phase_cos"]
    network.target_sin = batch["phase_sin"]
    figure = build_comparison(
        network,
        batch,
        _StubBank({248}),
        device=torch.device("cpu"),
        grid=GRID,
        in_channels=1,
        max_samples=1,
    )
    assert len(figure.rows) == 1


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------
def test_render_writes_a_png(tmp_path: Path) -> None:
    """A run without W&B must still leave a visual record on disk."""
    figure = _figure(_Identity(), [248], {248}, max_samples=2)
    out = render_comparison(figure, tmp_path / "sub" / "cmp.png", "epoch 0")
    assert out.is_file()
    assert out.stat().st_size > 0


def test_render_survives_an_empty_batch(tmp_path: Path) -> None:
    """Zero rows must still produce a file rather than raising."""
    figure = _figure(_Identity(), [248], {248}, max_samples=0)
    out = render_comparison(figure, tmp_path / "empty.png", "no rows")
    assert out.is_file()


def test_render_does_not_leak_figures() -> None:
    """pyplot figures must be closed, or a long run leaks memory."""
    import matplotlib.pyplot as plt

    before = len(plt.get_fignums())
    figure = _figure(_Identity(), [248], {248}, max_samples=2)
    render_comparison(figure, Path("cmp.png"), "epoch 0")
    assert len(plt.get_fignums()) == before