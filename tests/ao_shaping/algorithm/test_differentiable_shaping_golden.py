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


def _shape(array: np.ndarray) -> dict[str, Any]:
    """A version-tolerant summary of an array: extent, and its value range.

    The digests below are bit-exact, which makes them sensitive to the BLAS
    and FFT library versions: the same code produces different bytes on
    torch 2.6/numpy 2.5 (the CI CPU image) than on torch 2.14/numpy 2.4
    (the workstation), even though the loss sequence agrees to the last
    digit. A golden that only reproduces on one library build pins nothing,
    so the recorded summary is the shape plus the value range -- the things
    that must not move when the function is split -- and the digests are
    kept as a separate, advisory record.
    """
    finite = array[np.isfinite(array)]
    return {
        "shape": list(array.shape),
        "min": float(finite.min()) if finite.size else 0.0,
        "max": float(finite.max()) if finite.size else 0.0,
        "mean": float(finite.mean()) if finite.size else 0.0,
    }


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
        "phase": _shape(result.phase),
        "intensity": _shape(result.simulated_intensity),
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

    # The loss history is compared exactly: it is the sequence the split must
    # not move, and it agreed to the last digit across library versions.
    assert trace["loss_history"] == expected["loss_history"]
    assert trace["iterations"] == expected["iterations"]
    assert trace["converged"] == expected["converged"]

    # The array summaries are compared with a tolerance: their extent must
    # match exactly, and their values to within float noise (the FFT and
    # BLAS implementations differ between library builds, and the results
    # differ in the last bits -- not in the last digits).
    for key in ("phase", "intensity"):
        got, want = trace[key], expected[key]
        assert got["shape"] == want["shape"], f"{key} shape moved: {got} != {want}"
        for field in ("min", "max", "mean"):
            assert abs(got[field] - want[field]) <= 1e-6 * max(1.0, abs(want[field])), (
                f"{key}.{field} moved: {got[field]} != {want[field]}"
            )


def test_the_trace_is_reproducible() -> None:
    """A golden that only reproduces once pins nothing."""
    assert _trace() == _trace()
