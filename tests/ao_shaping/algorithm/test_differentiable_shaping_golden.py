"""Golden: the numeric trace of ``train_beam_shaping``.

Issue #167 splits this long training function; the split must not move a single
number. So the numbers are pinned: the per-iteration loss history plus digests
of the final phase and the simulated intensity, for a fixed seed.

``device="cpu"`` is passed explicitly. Without it the function picks CUDA when a
GPU is visible, and GPU float32 does not match CPU float32 bit for bit -- the
golden would then only reproduce on machines with the same GPU. Pinning the
device keeps the trace reproducible everywhere.

The golden is regenerated with::

    AO_DIFF_SHAPING_GOLDEN_UPDATE=1 pytest tests/ao_shaping/algorithm/test_differentiable_shaping_golden.py
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ao_shaping.algorithm.signal_processing.differentiable_shaping import (
    train_beam_shaping,
)

GOLDEN = Path(__file__).with_name("differentiable_shaping_golden.json")
UPDATE = os.environ.get("AO_DIFF_SHAPING_GOLDEN_UPDATE") == "1"


def _square_target(size: int = 16) -> np.ndarray:
    target = np.zeros((size, size), dtype=np.float64)
    target[size // 4 : size - size // 4, size // 4 : size - size // 4] = 1.0
    return target


def _digest(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()[:16]


def _trace() -> dict[str, Any]:
    result = train_beam_shaping(
        _square_target(),
        (16, 16),
        iterations=5,
        seed=7,
        device="cpu",
        propagation="fft",
        optimizer="adam",
    )
    return {
        "loss_history": result.loss_history,
        "iterations": result.iterations,
        "converged": result.converged,
        "phase_digest": _digest(result.phase),
        "intensity_digest": _digest(result.simulated_intensity),
    }


def test_numeric_trace_matches_the_golden() -> None:
    trace = _trace()

    if UPDATE:
        GOLDEN.write_text(
            json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        expected = trace
    else:
        assert GOLDEN.exists(), (
            f"missing golden {GOLDEN.name}; regenerate with "
            "AO_DIFF_SHAPING_GOLDEN_UPDATE=1"
        )
        expected = json.loads(GOLDEN.read_text(encoding="utf-8"))

    assert trace == expected


def test_the_trace_is_reproducible() -> None:
    """A golden that only reproduces once pins nothing."""
    assert _trace() == _trace()
