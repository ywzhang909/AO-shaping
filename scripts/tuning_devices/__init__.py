"""Device tuning / calibration helpers.

This directory holds **scripts and MATLAB sources**, not a Python API. The
Python functions that used to be re-exported here
(``calculate_derotation``, ``centroid_calculation``, ``normalize_01``) never
existed as modules in this package -- the names pointed at MATLAB files
(``calculateDerotation.m``, ``centroidcaculation.m``, ``normalize01.m``), so
importing this package died with ``ModuleNotFoundError``.

The canonical home of all of them is
:mod:`ao_shaping.utils.wavefront.wavefront_calc`:

* ``calculate_derotation``
* ``centroid_calculation``
* ``normalize_01``
* ``to_color``
* ``get_zernike_base_matrixs``
* ``ZernikeCentroidCalculator``

Import them from there. ``dm_unit_compute.py`` in this directory is a thin
pygame/WFS front end that imports the same canonical names (R-34, 2026-10-04).
"""