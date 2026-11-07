"""Online training-set refeeding between differentiable-optimization blocks.

Issue #229 step 5 (plan §5.2). The differentiable loop already captures four
ABBA frames per block (A1/B1/B2/A2) and currently *discards* them after the
verdict — the most valuable data for the model, because the loop visits
exactly the phase neighbourhood the model is worst at. This module provides
the device-free bookkeeping and gate logic:

- :class:`SessionDataset` — the accumulated ``(phase, image, meta)`` store
  with the ``split="augment"`` marker;
- :func:`should_refit` — the ``k % K == 0`` inter-block trigger;
- :class:`ModelSwitchGate` — the model-switch gate: the refit model may only
  replace the incumbent when the *independent* validation set does not
  degrade (NRMSE within 10%) and the direction guard does not regress;
  two consecutive failures force a baseline rebuild (re-run of the
  structured probes).

Isolation invariants (plan §5.2 steps 4-5, §7.5):

- refeeding happens **between** blocks only; the model is frozen during a
  block's iterations (the caller enforces, this module only records);
- refed frames never enter the *current* block's verdict (the verdict uses
  only that block's four ABBA frames);
- refed frames never enter the validation set — the validation set is the
  held-out amplitude level / mode of the structured probe plan and is
  managed by the caller, never appended here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Default inter-block refeed cadence (plan: K = 3, configurable 1-5).
DEFAULT_REFIT_EVERY = 3

#: Max relative NRMSE degradation on the independent validation set before
#: the new model is rejected (plan §5.2 step 3: <= 10%).
MAX_VALIDATION_DEGRADATION = 0.10

#: Consecutive model-switch gate failures that force a baseline rebuild.
MAX_CONSECUTIVE_GATE_FAILURES = 2


@dataclass
class BlockRecord:
    """One ABBA block's verdict and the four frames it produced."""

    block: int
    phase_anchor: np.ndarray
    phase_candidate: np.ndarray
    frames: dict  # {"A1": image, "B1": image, "B2": image, "A2": image}
    accepted: bool
    exposure_ms: float


class SessionDataset:
    """Accumulated session training store (plan §5.2 step 1).

    Frames are appended **after** each block's verdict. Rejected phases are
    included as well: a rejected block is a negative sample (``direction not
    improving``) and is equally valuable for calibration.
    """

    def __init__(self) -> None:
        self.entries: list[dict] = []
        self._counters: dict[str, int] = {"train": 0, "validation": 0, "holdout": 0, "augment": 0}

    def seed(self, rows: list[dict]) -> int:
        """Load the structured-probe anchor set (train/validation/holdout)."""
        for row in rows:
            if row["split"] not in self._counters:
                raise ValueError(f"bad split {row['split']!r}")
            self.entries.append(dict(**row, source="probe"))
            self._counters[row["split"]] += 1
        return len(self.entries)

    def add_block(self, record: BlockRecord) -> int:
        """Refed the block's four ABBA frames (marker ``split="augment"``)."""
        required = ("A1", "B1", "B2", "A2")
        missing = [k for k in required if k not in record.frames]
        if missing:
            raise ValueError(f"block {record.block} missing ABBA frames {missing}")
        added = 0
        for tag, frame in record.frames.items():
            phase = record.phase_anchor if tag in ("A1", "A2") else record.phase_candidate
            self.entries.append(dict(
                phase=np.asarray(phase), image=np.asarray(frame), split="augment",
                block=record.block, role=tag,
                exposure_ms=float(record.exposure_ms),
                accepted=bool(record.accepted),
                source="block",
            ))
            self._counters["augment"] += 1
            added += 1
        return added

    def frames(self, splits: tuple[str, ...] = ("train", "validation", "augment")) -> list[dict]:
        """Frames eligible for a block-boundary refit (never holdout)."""
        return [e for e in self.entries if e["split"] in splits]

    def counts(self) -> dict[str, int]:
        return dict(self._counters)


def should_refit(block: int, every: int = DEFAULT_REFIT_EVERY) -> bool:
    """Inter-block refeed trigger: every ``every`` blocks (plan: K = 3)."""
    if every < 1:
        raise ValueError("every must be >= 1")
    if block < 1:
        raise ValueError("blocks are 1-based; nothing to refeed at block < 1")
    return block % every == 0


@dataclass
class ModelSwitchGate:
    """Version-gated model switch after each refit (plan §5.2 step 3).

    ``check()`` compares the refit model's NRMSE on the independent
    validation set against the incumbent's, and the next-block direction
    guard correlation against the incumbent's. Only when *both* pass does
    the new model become incumbent. Two consecutive failures force a
    baseline rebuild (re-run of the structured probes).
    """

    max_degradation: float = MAX_VALIDATION_DEGRADATION
    max_consecutive_failures: int = MAX_CONSECUTIVE_GATE_FAILURES
    incumbent_validation_nrmse: float | None = None
    incumbent_guard_correlation: float | None = None
    consecutive_failures: int = 0
    rebuild_required: bool = False
    history: list[dict] = field(default_factory=list)

    def check(self, refit_validation_nrmse: float,
              next_guard_correlation: float | None = None) -> dict:
        """Evaluate one refit candidate. Returns the gate verdict dict."""
        if self.incumbent_validation_nrmse is None or not np.isfinite(self.incumbent_validation_nrmse):
            raise ValueError("no incumbent validation NRMSE recorded")
        degradation = ((refit_validation_nrmse - self.incumbent_validation_nrmse)
                       / max(self.incumbent_validation_nrmse, 1e-9))
        nrmse_ok = degradation <= self.max_degradation
        guard_ok = True
        guard_detail = "no incumbent correlation recorded; accepted"
        if (self.incumbent_guard_correlation is not None
                and next_guard_correlation is not None
                and np.isfinite(self.incumbent_guard_correlation)
                and np.isfinite(next_guard_correlation)):
            guard_ok = next_guard_correlation >= self.incumbent_guard_correlation
            guard_detail = (f"{next_guard_correlation:.3f} vs incumbent "
                             f"{self.incumbent_guard_correlation:.3f}")
        passed = bool(nrmse_ok and guard_ok)
        if passed:
            self.consecutive_failures = 0
            self.incumbent_validation_nrmse = float(refit_validation_nrmse)
            if next_guard_correlation is not None and np.isfinite(next_guard_correlation):
                self.incumbent_guard_correlation = float(next_guard_correlation)
        else:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_consecutive_failures:
                self.rebuild_required = True
        verdict = dict(passed=passed, degradation=float(degradation), nrmse_ok=bool(nrmse_ok),
                       guard_ok=bool(guard_ok), guard_detail=guard_detail,
                       consecutive_failures=self.consecutive_failures,
                       rebuild_required=self.rebuild_required)
        self.history.append(verdict)
        return verdict

    def record_block_guard(self, correlation: float) -> None:
        """The incumbent's own per-block guard correlation (for trend log)."""
        if self.incumbent_guard_correlation is None:
            self.incumbent_guard_correlation = float(correlation)
