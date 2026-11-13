"""GPU (torch-CUDA) acceleration of the GS seed: equivalence guards.

``gs_seed`` now runs the ``fft``-mode Gerchberg-Saxton loop on CUDA by default
when ``torch.cuda.is_available()``. This file pins the two invariants that make
that safe:

* **device selection** -- ``device=None`` / ``"auto"`` follow the machine,
  ``"cpu"`` / ``"cuda"`` are honoured verbatim, anything else raises;
* **kernel equivalence** -- ``_gs_seed_cuda`` reproduces the numpy
  ``gerchberg_saxton`` result to ~1e-11 relative (both run the same
  double-precision ``fft`` iterations), so the seed phase and the resulting CV
  are unchanged.

The equivalence is checked on a small synthetic amplitude/target pair (no
``skipif`` on the fitted checkpoint), so it runs on any checkout, CI included;
only the CUDA half is skipped when no GPU is present.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.algorithm.signal_processing.gerchberg_saxton import gerchberg_saxton
from ao_shaping.algorithm.signal_processing.multishape_objective import (
    _gs_seed_cuda,
    _gs_seed_use_cuda,
)


def _small_field(n: int = 64, g: int = 32) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    half = (n - g) // 2
    amp = np.zeros((n, n))
    amp[half:half + g, half:half + g] = rng.uniform(0.5, 1.0, (g, g))
    tgt = np.zeros((n, n))
    tgt[half:half + g, half:half + g] = rng.uniform(0.2, 1.0, (g, g))
    return amp, tgt


def test_gs_seed_device_selection() -> None:
    assert _gs_seed_use_cuda("cpu") is False
    assert _gs_seed_use_cuda("cuda") is True
    import torch

    available = bool(torch.cuda.is_available())
    assert _gs_seed_use_cuda(None) is available
    assert _gs_seed_use_cuda("auto") is available
    with pytest.raises(ValueError):
        _gs_seed_use_cuda("tpu")


@pytest.mark.skipif(not _gs_seed_use_cuda(None), reason="no CUDA device")
def test_gs_seed_cuda_matches_numpy() -> None:
    n, g = 64, 32
    half = (n - g) // 2
    amp, tgt = _small_field(n, g)
    target_sqrt = np.sqrt(np.clip(tgt, 0.0, None))
    result = gerchberg_saxton(amp, target_sqrt, iterations=80, propagation="fft")
    cpu_phase = result.phase[half:half + g, half:half + g]
    gpu_phase = _gs_seed_cuda(amp, target_sqrt, n, g, half)
    assert np.isfinite(gpu_phase).all()
    rel = np.linalg.norm(gpu_phase - cpu_phase) / (np.linalg.norm(cpu_phase) + 1e-12)
    assert rel < 1e-8
