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
- ``focus_probe`` — sweep the Fresnel focal length on a plain phase → the 1/f
  that brings the 0-order spot back onto the detector (auto-exposure per point,
  coarse scan + local refinement).
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
