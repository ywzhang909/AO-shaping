"""SLM + WFS probes and Zernike response / correction tools.

Modules:
- ``wfs_probe`` — SLM + WFS intensity and pupil check; the pre-calibration
  hardware gate (writes a JSON report and sets ``wfs.pupil`` explicitly).
- ``wfs_reference`` — WFS reference-wavefront calibration and tilt linearity
  (three steps: pure-flat reference → restore/load cross-check → tilt scan).
- ``zernike_response`` — SLM Zernike mode → WFS reading response-matrix
  calibration (push-pull per mode, then invert to a modal control law).
- ``zernike_correction`` — three-stage Zernike modal wavefront correction
  (auto-calibration + reference / robust response matrix / closed-loop solve).

Run with ``python -m ao_shaping.tools.wfs.<module>``.

This package deliberately does **no** eager re-export.
"""
