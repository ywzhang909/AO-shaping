"""Smoke tests for the differentiable beam shaping runner (offline path).

These tests verify the runner module imports cleanly and that the offline
differentiable optimization path produces a result. They are gated on torch.
No hardware is touched.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import click  # noqa: E402

from ao_shaping.runners.diff_shaping_runner import run  # noqa: E402


class TestRunnerCLI:
    """Verify the Click command is defined with expected options."""

    def test_is_click_command(self):
        assert isinstance(run, click.Command)

    def test_expected_options_present(self):
        names = {p.name for p in run.params}
        for expected in (
            "camera_type",
            "target_shape",
            "target_size",
            "propagation",
            "optimizer",
            "dl_iterations",
            "lr",
            "outer_iterations",
            "output",
        ):
            assert expected in names, f"missing option {expected}"


class TestOfflinePath:
    """Verify the offline (simulation) optimization path works via the algorithm module."""

    def test_train_beam_shaping_offline(self):
        from ao_shaping.algorithm.differentiable_shaping import (
            create_target_mask,
            train_beam_shaping,
        )

        grid = (64, 64)
        target = create_target_mask("square", grid, 20)
        result = train_beam_shaping(target, grid, iterations=20, seed=0)
        assert result.phase.shape == grid
        assert np.isfinite(result.phase).all()
        assert len(result.loss_history) > 0
