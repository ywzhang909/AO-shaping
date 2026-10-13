"""Camera-only (CCD) bench probes.

No SLM is driven here: these characterise the *camera* side of the 2f-Fourier
bench, whose numbers every other probe depends on.

Modules:
- ``exposure_check`` — is the camera really in manual exposure, and how much
  does it drift at a fixed setting? Separates "camera gain drift" from "the SLM
  retained the last displayed pattern" (two causes of the same symptom).
- ``drift_probe`` — flat-field drift (region L2 norm, **not** peak) plus an
  exposure ladder judged by ``exposure_monotonicity``.
- ``floor_probe`` — read-floor / settle-time / SNR-vs-averages characterisation;
  answers whether the noise is read-limited or drift-limited.

Run with ``python -m ao_shaping.tools.ccd.<module>``; each takes ``--no-hw`` to
print the acquisition plan without touching a device.

This package deliberately does **no** eager re-export.
"""
