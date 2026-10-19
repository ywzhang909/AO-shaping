"""Parity check: the Cython Gerchberg-Saxton must match the Python one.

The Cython extension (``gerchberg_saxton_cython``) is built in place from
``src/calculators/gerchberg_saxton.pyx``; it is not part of the packaged build,
so this test **skips** unless someone has compiled it locally
(``python src/calculators/setup.py build_ext --inplace`` -- note the extension is
not yet listed in ``setup.py``, only ``adam_cython`` and ``target_func_cython``
are).

This file previously lived at ``src/calculators/test_gerchberg_saxton.py``,
where ``testpaths = ["tests"]`` meant pytest never collected it: it looked like
coverage while running for nobody. See
``tests/ao_shaping/test_conventions.py::test_src_contains_no_test_files_pytest_would_skip``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# The extension is compiled next to its .pyx source, so that directory has to be
# importable before ``importorskip`` probes for the module.
_CALCULATORS = Path(__file__).resolve().parents[2] / "src" / "calculators"
if str(_CALCULATORS) not in sys.path:
    sys.path.insert(0, str(_CALCULATORS))

cython_gs = pytest.importorskip(
    "gerchberg_saxton_cython",
    reason="the optional Cython GS extension is not built in this checkout",
)

from ao_shaping.algorithm.signal_processing.gerchberg_saxton import (  # noqa: E402
    gerchberg_saxton as python_gerchberg_saxton,
)


def test_cython_matches_python():
    """Identical inputs -> identical phase / amplitude / error history."""
    size = 64
    source_amplitude = np.ones((size, size), dtype=np.float64)

    # A simple target: a filled circle in the centre.
    target_amplitude = np.zeros((size, size), dtype=np.float64)
    center = size // 2
    radius = size // 8
    yy, xx = np.ogrid[:size, :size]
    target_amplitude[(yy - center) ** 2 + (xx - center) ** 2 <= radius**2] = 1.0
    target_amplitude = target_amplitude / np.sqrt(np.sum(target_amplitude**2))

    kwargs = dict(
        source_amplitude=source_amplitude,
        target_amplitude=target_amplitude,
        iterations=20,
        cell_spacing=8e-6,
        distance=0.1,
        wavelength=1064e-9,
        error_threshold=1e-6,
    )

    python_result = python_gerchberg_saxton(**kwargs, propagation="asm")
    cython_result = cython_gs.gerchberg_saxton(**kwargs, propagation=0)

    # Unpack: the Python side may return a GSResult or a plain tuple.
    if hasattr(python_result, "phase"):
        python_phase = python_result.phase
        python_amplitude = python_result.amplitude
        python_error_history = python_result.error_history
    else:
        python_phase, python_amplitude, python_error_history, *_ = python_result

    cython_phase, cython_amplitude, cython_error_history, *_ = cython_result

    assert np.allclose(python_phase, cython_phase, rtol=1e-10, atol=1e-10)
    assert np.allclose(python_amplitude, cython_amplitude, rtol=1e-10, atol=1e-10)
    assert np.allclose(python_error_history, cython_error_history, rtol=1e-10, atol=1e-10)
