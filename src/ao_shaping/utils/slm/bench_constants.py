"""Zero-import leaf for bench-tuning constants that must be defined exactly once.

The 2f focal-scale constant ``TILT_SHIFT_SCALE`` was spelled out as four
separate literal assignments (``bench_kernels``, ``runner_common``,
``slm_gs_refine``, ``scripts/slm/generate_bench_probe_report``). A bench
change would update one copy and silently leave the other three stale -- the
same failure mode the stale-geometry guard exists to catch, but in the code.

Every consumer now imports this one value from here instead of re-declaring
it. This module must stay a **leaf**: no ``import``/``from`` of anything
non-stdlib, and no module-level side effects, so that importing it never
drags in a hardware SDK / ``ctypes`` / ``torch``. The guard is
``tests/ao_shaping/utils/slm/test_bench_constants_leaf.py``.

``TILT_SHIFT_SCALE`` -- the measured 2f focal scale for this bench: a 2*pi
phase ramp over ``P`` SLM panel px displaces the spot by
``TILT_SHIFT_SCALE / P`` camera px. Measured 2026-09-30 with
``slm_tilt_probe --periods 120,240,480,960,1920`` (3 repeats each, agreeing to
0.2 px). Displacements from flat: 62.1, 30.3, 14.7, 6.9, 3.4 px -- cleanly
1/P, halving each time the period doubles. The naive ``f*lambda/d_slm`` is
~17.5x larger on this bench and the panel/camera axes are swapped 90 deg, so
neither may be derived from first principles.
"""

from __future__ import annotations

#: Measured 2f focal scale (camera px per full 2*pi ramp per panel pixel): see
#: module docstring. The single source of truth -- do not re-declare it.
TILT_SHIFT_SCALE = 7400.0
