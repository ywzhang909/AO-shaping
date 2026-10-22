"""SLM panel probes: characterise the SLM panel itself with a CCD (no WFS).

Modules:
- ``diagnose`` — three-step hardware self-test (freeze / modulate / linearity),
  registered as the ``slm-diagnose`` CLI command. Default camera is ``miicam``;
  the Daheng bench must pass ``--camera-type daheng``.
- ``tilt_probe`` — is the panel actually modulating? A tilt ramp is more
  reliable than a grating: the spot shift depends only on the ramp period, not
  on diffraction efficiency.
- ``panel_locate`` — locate the spot centre on *panel* coordinates (the CCD
  0-order coordinate is not a panel coordinate: axes swapped 90°, scale >10x).
- ``beam_extent`` — half-plane random-phase boundary scan → spot centre/radius.
- ``display_state`` — context manager that preserves the SLM display (memory
  slot or grayscale mode) while a probe uses temporary memory slots.
- ``focus_probe`` — sweep the Fresnel focal length on a plain phase → the 1/f
  that brings the 0-order spot back onto the detector (auto-exposure per point,
  coarse scan + local refinement).
- ``focus_preflight`` — first-class pre-acquisition stage that layers the
  measured focus (defocus) compensation + de-steer ramp onto every displayed
  pupil phase; pure compute, no devices (the live-probe path is gated by the
  caller). Order/symbol are frozen: focus → de-steer → pupil.
- ``alignment_probe`` — measure focus and two coma corrections *before*
  locating the CCD spot (reuses the canonical Fresnel sweep + Zernike
  generator; keeps every tested phase, raw frame, exposure readback and UTC
  timestamp for replay; corrections are never left on the panel implicitly).
- ``bench_fingerprint`` — did the optical path change since calibration?
  Two known phases (flat, flat + known Zernike tilt) → FWHM / tilt shift /
  centroid fingerprint: catches camera repositioning / SLM swap that would
  silently invalidate a stale calibration.
- ``phase_resolution`` — per-pixel random phase vs smooth Zernike: the panel's
  effective phase resolution (decides whether a speckle route is usable).
- ``phase_response`` — phase → CCD response curve (defocus / lens cases) with a
  shared acquisition/render path.

Run with ``python -m ao_shaping.tools.slm.panel.<module>``.

This package deliberately does **no** eager re-export: the submodules import
click / numpy / the SLM and camera drivers at module scope, and a package-level
re-export would make every ``import ao_shaping.tools.slm.panel.<x>`` pay for all
of them.
"""
from __future__ import annotations
