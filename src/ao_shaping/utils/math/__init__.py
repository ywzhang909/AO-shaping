"""Numeric helpers that are neither image, wavefront nor device aware.

Created for R-49 (#65): the fitting and stability-judgement kernels that used
to live in ``tools/kernels/bench_kernels.py`` now have a single leaf home here,
so any probe, runner or script can import them without dragging in the bench
kernel surface (camera / SLM duck-types, panel geometry, frame analysis).

Leaf contract: every module here imports only the standard library and
``numpy`` -- no ``ao_shaping.*`` import, at module level or deferred. Guarded by
``tests/ao_shaping/utils/math/test_math_leaf.py``.
"""

from __future__ import annotations
