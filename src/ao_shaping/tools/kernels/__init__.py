"""Shared SLM bench measurement kernels and scan/sweep analysis.

Split out of ``tools/slm/`` (2026-10-12) so the CCD, panel, WFS and Zernike
probes can share one source for frame maths and scan analysis without importing
a device-facing package. Device instances are injected by callers — there is no
CLI here and nothing in this package constructs hardware.

Modules:
- ``bench_kernels`` — frame maths (``smooth_frame`` / ``despike_frame`` /
  ``finite_clip`` / ``crop_roi`` / ``roi_l2`` / ``finite_median_subtract``),
  spot measurement, panel phase placement (``ramp_panel`` / ``zernike_panel``),
  tilt ramp, exposure monotonicity, ``display_and_average`` / ``capture_settled``
  and the bench constants (``SLM_PANEL_W`` / ``BEAM_CENTER_PANEL`` / ...).
- ``sweep_analysis`` — scan linearity (``outlier_mask`` / ``group_raw_scan`` /
  ``analyze_linearity``), SPGD ``delta`` exploration and SNR measurement
  (``abba_signal`` / ``snr_sweep`` / ``snr_verdict``).

This package deliberately does **no** eager re-export.
"""
