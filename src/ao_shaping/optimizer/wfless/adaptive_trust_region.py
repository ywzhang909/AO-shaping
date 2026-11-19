"""Adaptive trust-region radius policy for the model-in-the-loop round loop.

**Why the radius should adapt.** Step A produces ``c_t`` and Step B produces
``phi_t`` conditioned on ``c_t``; left alone the two estimators co-adapt and the
error shows up on hardware as a slow divergence. :func:`trust_region_clamp` bounds
that by capping the per-round L2 step. But a *fixed* cap is right for neither
regime: too small and a genuinely large correction is clipped every round and the
loop crawls; too large and an accepted-then-wrong round lands far from the fit.

This module supplies only the **radius schedule**. The clamp itself stays in
:func:`ao_shaping.optimizer.wfless.slm_model_in_loop.trust_region_clamp` — there is
exactly one implementation of "bound this step", and this policy never re-derives
it, so there is no second opinion about where the boundary is.

**What counts as evidence.** Growth is earned by consecutive accepts; a single
rejection shrinks hard. That asymmetry is deliberate: the cost of a too-large step
is a committed bad round, while the cost of a too-small step is only slowness.

**Device-free.** This module imports only NumPy and the standard library, so it
can be unit-tested and reasoned about without a bench. It is wired into
:func:`~ao_shaping.optimizer.wfless.slm_model_in_loop.optimize_slm_model_in_loop`
behind ``SlmModelInLoopConfig.adaptive_trust_region``, which **defaults to False**
— with the flag off the loop calls the clamp with its fixed configured radius,
exactly as before.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from loguru import logger

__all__ = ["AdaptiveTrustRegion", "AdaptiveTrustRegionConfig"]


@dataclass(frozen=True)
class AdaptiveTrustRegionConfig:
    """Bounds and step sizes for :class:`AdaptiveTrustRegion`.

    Attributes:
        initial_c_l2: Starting per-round L2 cap, radians. Must be > 0.
        min_c_l2: Floor the radius may fall to. Must be > 0 and <= ``initial_c_l2``.
        max_c_l2: Ceiling the radius may rise to. Must be >= ``initial_c_l2``.
        grow_factor: Multiplicative growth per earned accept. Must be > 1.
        shrink_factor: Multiplicative shrink on a rejection. Must be in ``(0, 1)``.
        accepts_before_growth: Consecutive accepts required before growing once.
            Must be >= 1, so a single good round cannot inflate the cap.
    """

    initial_c_l2: float
    min_c_l2: float
    max_c_l2: float
    grow_factor: float = 1.25
    shrink_factor: float = 0.5
    accepts_before_growth: int = 1

    def __post_init__(self) -> None:
        if not math.isfinite(self.initial_c_l2) or self.initial_c_l2 <= 0.0:
            raise ValueError(
                f"initial_c_l2 must be > 0 and finite, got {self.initial_c_l2!r}"
            )
        if not math.isfinite(self.min_c_l2) or self.min_c_l2 <= 0.0:
            raise ValueError(f"min_c_l2 must be > 0 and finite, got {self.min_c_l2!r}")
        if not math.isfinite(self.max_c_l2) or self.max_c_l2 <= 0.0:
            raise ValueError(f"max_c_l2 must be > 0 and finite, got {self.max_c_l2!r}")
        if self.min_c_l2 > self.initial_c_l2:
            raise ValueError(
                f"min_c_l2 ({self.min_c_l2:g}) must not exceed initial_c_l2 "
                f"({self.initial_c_l2:g})"
            )
        if self.max_c_l2 < self.initial_c_l2:
            raise ValueError(
                f"max_c_l2 ({self.max_c_l2:g}) must not be below initial_c_l2 "
                f"({self.initial_c_l2:g})"
            )
        if not math.isfinite(self.grow_factor) or self.grow_factor <= 1.0:
            raise ValueError(
                f"grow_factor must be > 1 and finite, got {self.grow_factor!r}"
            )
        if not (0.0 < self.shrink_factor < 1.0):
            raise ValueError(
                f"shrink_factor must lie in (0, 1), got {self.shrink_factor!r}"
            )
        if self.accepts_before_growth < 1:
            raise ValueError(
                f"accepts_before_growth must be >= 1, got {self.accepts_before_growth}"
            )


class AdaptiveTrustRegion:
    """Radius schedule for the per-round coefficient step.

    State is the current radius plus the accept streak. One :meth:`update` call per
    round advances the schedule by one step.

    Attributes are exposed read-only; the radius is read via :attr:`radius`.
    """

    def __init__(self, config: AdaptiveTrustRegionConfig) -> None:
        """Validate the configuration and initialise the schedule.

        Args:
            config: Bounds and step sizes. Validated in full by its
                ``__post_init__``, so an unusable schedule fails here rather than
                mid-run.
        """
        self._config = config
        self._radius = float(config.initial_c_l2)
        self._accept_streak = 0
        self._rounds = 0
        self._n_growth = 0
        self._n_shrink = 0

    @property
    def config(self) -> AdaptiveTrustRegionConfig:
        """The immutable configuration this schedule was built from."""
        return self._config

    @property
    def radius(self) -> float:
        """Current per-round L2 cap in radians."""
        return self._radius

    @property
    def accept_streak(self) -> int:
        """Consecutive accepted rounds so far (reset by any rejection)."""
        return self._accept_streak

    @property
    def rounds(self) -> int:
        """Number of :meth:`update` calls so far."""
        return self._rounds

    def update(self, *, accepted: bool) -> float:
        """Advance the schedule by one round and return the next radius.

        A rejection shrinks immediately and resets the streak; a run of
        ``accepts_before_growth`` accepts grows the radius once. Both ends clamp to
        the configured bounds, so the radius can never reach zero or diverge.

        Args:
            accepted: Whether the round that just finished was accepted.

        Returns:
            The radius to use for the **next** round.
        """
        self._rounds += 1
        if not accepted:
            self._accept_streak = 0
            before = self._radius
            new_radius = before * self._config.shrink_factor
            self._radius = max(self._config.min_c_l2, min(new_radius, self._config.max_c_l2))
            # Count a shrink only when the radius actually came down; a
            # rejection at the floor is not a shrink, it is the floor holding.
            if self._radius < before:
                self._n_shrink += 1
            logger.debug(
                "trust region rejected a round: {:.4f} -> {:.4f} rad (floor {:.4f})",
                before,
                self._radius,
                self._config.min_c_l2,
            )
            return self._radius

        self._accept_streak += 1
        if self._accept_streak >= self._config.accepts_before_growth:
            self._accept_streak = 0
            before = self._radius
            new_radius = before * self._config.grow_factor
            self._radius = min(max(new_radius, self._config.min_c_l2), self._config.max_c_l2)
            # Count a growth only when the radius actually rose; an accept at the
            # ceiling is the ceiling holding, not a growth.
            if self._radius > before:
                self._n_growth += 1
        return self._radius

    def state(self) -> dict[str, float | int]:
        """Snapshot of the schedule, for the per-round history row."""
        return {
            "trust_region_c_l2": float(self._radius),
            "trust_region_accept_streak": int(self._accept_streak),
            "trust_region_rounds": int(self._rounds),
            "trust_region_growths": int(self._n_growth),
            "trust_region_shrinks": int(self._n_shrink),
        }

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"AdaptiveTrustRegion(radius={self._radius:.4g}, "
            f"streak={self._accept_streak}, rounds={self._rounds})"
        )