"""Zero-import leaf for bench constants that must be defined exactly once.

The 2f focal-scale constant ``TILT_SHIFT_SCALE`` was spelled out as four
separate literal assignments (``bench_kernels``, ``runner_common``,
``slm_gs_refine``, ``scripts/slm/generate_bench_probe_report``). A bench
change would update one copy and silently leave the other three stale -- the
same failure mode the stale-geometry guard exists to catch, but in the code.

The SLM panel-geometry constants (``SLM_PANEL_W`` / ``SLM_PANEL_H`` /
``SLM_PITCH_M`` / ``BEAM_CENTER_PANEL`` / ``BEAM_RADIUS_PANEL``) joined here
under R-47: they were declared in ``tools/kernels/bench_kernels`` yet are pure
bench geometry, imported by 13+ modules across ``tools/``, ``scripts/``,
``runners/`` and ``optimizer/``. Keeping them in the leaf lets ``optimizer/``
reach panel geometry without importing ``tools/`` (the R-46 reverse edge).

Every consumer now imports these values from here instead of re-declaring
them. This module must stay a **leaf**: no ``import``/``from`` of anything
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

# ---------------------------------------------------------------------------
# SLM panel geometry (measured 2026-09-30; single source of truth, R-47).
# Santec SLM-200 #1 22030108, 1920x1200, 10-bit, 2pi = 993 gray at 1064 nm;
# Daheng MER2-507, 2592x1944, 2.2 um pixel. Provenance:
# report/slm/model_in_loop_bench_calibration.md
# ---------------------------------------------------------------------------

#: Panel width in px (the shape the Santec driver expects, last axis).
SLM_PANEL_W = 1920
#: Panel height in px (first axis of every panel-sized array).
SLM_PANEL_H = 1200
#: SLM pixel pitch in metres.
SLM_PITCH_M = 8e-6
#: Illuminated beam centre in **panel** px (x, y). NOT a camera coordinate: the
#: two frames are swapped 90 deg and differ >10x in scale.
BEAM_CENTER_PANEL = (960, 600)
#: Illuminated beam radius in panel px.
BEAM_RADIUS_PANEL = 450

# ---------------------------------------------------------------------------
# Device / grating limits shared by the off-axis shaping configs (batch of
# #173). These were spelled out as bare literals in the per-mode config
# validators (``slm_gs_refine`` / ``slm_model_in_loop`` / ``slm_square_shaping``
# / ``_blaze_calibration``). A bench change (different SLM bank, larger CCD
# crop) would have to update several copies by hand -- the same drift failure
# the bench-geometry constants above already prevent.
# ---------------------------------------------------------------------------

#: Minimum blaze (diffraction-grating) period in panel px. A period below this
#: is not a resolvable grating for the off-axis zero-order overlay, so the
#: configs reject ``0 < period < BLAZE_PERIOD_MIN_PX``.
BLAZE_PERIOD_MIN_PX = 4

#: Largest SLM bank number the bench can address (slm_number is 1-based).
MAX_SLM_NUMBER = 8

#: Smallest full-frame CCD crop (px) the auto-calibration can target; smaller
#: crops lack the zero-order separation the blaze probe needs.
MIN_CAM_SIZE_PX = 16
