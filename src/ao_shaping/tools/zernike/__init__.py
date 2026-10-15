"""SLM Zernike tools, the smooth-Zernike sweep probe and the ABBA probe.
from __future__ import annotations

Split out of ``tools/slm/`` (2026-10-12) so the WFS-Zernike correction pipeline,
the CCD probes and the model-in-loop sweep share one Zernike module set without
depending on the SLM panel package.

Modules:
- ``common`` — shared constants and measurement primitives for the SLM Zernike
  tools. Fixes the ``ThorlabWFS.get_zernike()`` index convention (67-long array,
  ``coeff[1..66]`` valid, **sequential-m** enumeration, not standard Noll 1976)
  and the µm↔waves unit discipline.
- ``sweep_probe`` — smooth-Zernike bench sweep (ramp + tilt + defocus + astig +
  coma + spherical, 42 points) with a settled read and Recorder output; owns
  ``capture_settled`` and ``DEFAULT_PANEL_SHAPE``.
- ``abba_probe`` — ABBA dense-random-phase detectability probe (``+ - - +``
  palindrome cancels linear drift) that answers "can a random-phase
  perturbation be measured above the bench floor?".

Run with ``python -m ao_shaping.tools.zernike.<module>``; each probe takes
``--no-hw`` to print the acquisition plan without touching a device.
"""
