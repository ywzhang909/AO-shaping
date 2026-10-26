"""Recorded nonmonotone CCD exploration with a separately retained champion."""
from __future__ import annotations

import numpy as np


class NonmonotoneAcceptance:
    """Metropolis acceptance in absolute metric units, with a best-value guard."""

    def __init__(self, initial, *, temperature, relative_budget=.25, seed=0):
        if not np.isfinite([initial, temperature, relative_budget]).all() or min(initial, temperature, relative_budget) < 0:
            raise ValueError("finite nonnegative initial value, temperature and budget required")
        self.best = float(initial)
        self.temperature = float(temperature)
        self.relative_budget = float(relative_budget)
        self.rng = np.random.default_rng(seed)

    def update(self, baseline, candidate):
        if not np.isfinite([baseline, candidate]).all() or min(baseline, candidate) < 0:
            raise ValueError("finite nonnegative measurements required")
        delta = float(candidate-baseline)
        guard = float(candidate) <= self.best*(1+self.relative_budget)
        probability = (1. if delta < 0 else float(np.exp(-delta/self.temperature))) if self.temperature > 0 else float(delta < 0)
        draw = float(self.rng.random())
        accepted = bool(guard and draw < probability)
        improved_best = bool(accepted and candidate < self.best)
        previous_best = self.best
        if improved_best:
            self.best = float(candidate)
        return dict(accepted=accepted, improved_best=improved_best, uphill=delta > 0,
                    delta=delta, temperature=self.temperature, probability=probability,
                    random_draw=draw, within_budget=bool(guard), previous_best=previous_best,
                    best_value=self.best, relative_budget=self.relative_budget)
