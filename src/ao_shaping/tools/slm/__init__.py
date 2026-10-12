"""SLM-specific bench tools (panel probes + calibration + cartographer).

After the 2026-10-12 ``tools/`` reorganisation this package holds only the
SLM-specific pieces; the shared measurement kernels, the shared Zernike tools
and the camera-only / WFS probes now live beside it::

    tools/
      kernels/   bench_kernels, sweep_analysis      (shared, device-injected)
      zernike/   common, sweep_probe, abba_probe    (shared Zernike + sweep)
      ccd/       exposure_check, drift_probe, floor_probe
      wfs/       wfs_probe, wfs_reference, zernike_response, zernike_correction
      slm/
        panel/         diagnose, tilt_probe, panel_locate, beam_extent,
                       phase_resolution, phase_response
        calibration/   calibration, gray_response, lut_runner, phase_capture,
                       train_data_collect
        cartographer/  cosine_pattern, hartmann_capture,
                       wavefront_reconstruction, phase_grayscale_lut,
                       dynamic_compensation
        params.py      shared click parameter groups
                       (``SlmBenchParams`` / ``SlmAcquireParams``)

Registered CLI commands that live here: ``slm-diagnose``
(``panel/diagnose.py``) and ``slm-lut`` (``calibration/lut_runner.py``). Every
other module runs with ``python -m ao_shaping.tools.slm.<subpackage>.<module>``.

This package deliberately does **no** eager re-export: the submodules import
click / loguru / numpy / the SLM and camera drivers at module scope, and a
package-level re-export would make every ``import ao_shaping.tools.slm.<x>`` pay
for all of them.
"""
