"""Tests for the surrogate-iteration experiment script.

The script exists to answer one question -- does a *better* surrogate produce a
*better* phase? -- so the things worth pinning are the ones that could make that
question unanswerable: the zoo must actually span an under- and an over-fitted
model, the held-out score must come from pickles the model never saw, and the
cross-model matrix must reuse the producing instances so its diagonal really is
self-assessment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from iterate_surrogate_synthesis import (  # noqa: E402
    SYNTH_N_MAX,
    ZOO,
    _peak_norm,
    held_out_r2,
)


class TestZoo:
    def test_zoo_spans_under_and_over_fitting(self):
        assert len(ZOO) >= 3
        assert min(z[1] for z in ZOO) < SYNTH_N_MAX, "zoo must include a low-order model"
        assert max(z[2] for z in ZOO) > min(z[2] for z in ZOO), "zoo must vary epochs"
        assert len({z[1] for z in ZOO}) > 1, "zoo must vary n_max"

    def test_labels_are_unique(self):
        labels = [z[0] for z in ZOO]
        assert len(labels) == len(set(labels))

    def test_synthesis_band_matches_the_validated_surrogate(self):
        # Synthesising outside n_max=15 is the out-of-distribution mistake this
        # whole experiment exists to avoid.
        assert SYNTH_N_MAX == 15


class TestHeldOutR2:
    @staticmethod
    def _split(n: int = 8):
        phase_cos = torch.full((n, 4, 4), 0.5)
        phase_sin = torch.zeros((n, 4, 4))
        target = torch.rand(n, 1, 4, 4)
        return {"phase_cos": phase_cos, "phase_sin": phase_sin, "target": target}

    def test_perfect_prediction_scores_one(self):
        tensors = self._split()
        # held_out_r2 peak-normalises the target, so the stubs must too.
        normalised = _peak_norm(tensors["target"].clone())

        class _Echo(torch.nn.Module):
            def forward(self, cos, sin):  # noqa: D102, ANN001
                return normalised

        assert held_out_r2(_Echo(), tensors) == pytest.approx(1.0, abs=1e-5)

    def test_mean_prediction_scores_about_zero(self):
        tensors = self._split()
        normalised = _peak_norm(tensors["target"].clone())

        class _Mean(torch.nn.Module):
            def forward(self, cos, sin):  # noqa: D102, ANN001
                return normalised.mean()

        assert held_out_r2(_Mean(), tensors) == pytest.approx(0.0, abs=1e-5)

    def test_accepts_a_measured_phasor_pair(self):
        # Regression guard: this used to forward a (cos, sin) tuple into a helper
        # that expected a raw-radian phase tensor.
        tensors = self._split()
        normalised = _peak_norm(tensors["target"].clone())

        class _Model(torch.nn.Module):
            def forward(self, cos, sin):  # noqa: D102, ANN001
                assert isinstance(cos, torch.Tensor) and isinstance(sin, torch.Tensor)
                return normalised

        assert held_out_r2(_Model(), tensors) == pytest.approx(1.0, abs=1e-5)
