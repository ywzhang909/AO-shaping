"""Single shared camera-injection point for the sensorless PIB optimizers.

Two optimizer modules used to each carry their own ``_CAMERA_CLASS_OVERRIDE``
global plus a copy of ``_camera_cls`` / ``_camera_override`` (``wfless/pib.py``
and ``combined_optimizer.py``). Because the globals were independent, a test or
the sim-eval harness that injected the simulator through ``pib`` only could
silently leave ``combined_optimizer`` on the real driver (or vice versa) -- i.e.
a hardware run could quietly use the simulator.

This module collapses that into ONE global and ONE pair of helpers, so any
injection point pokes the same state. Both optimizers import
``_camera_cls`` / ``_camera_override`` from here; tests and
``pib_sim_eval`` poke :data:`_CAMERA_CLASS_OVERRIDE` here.

Leaf of the optimizer package: camera drivers are resolved lazily inside
:func:`_camera_cls` (the ``MIICamera``/``SimPibCCD`` imports live in the
function body) so importing this module has no driver side effect.
"""

from __future__ import annotations

from contextlib import contextmanager

#: Test / simulation injection point. Set to ``SimCamera`` (or any camera
#: stand-in) to replace the real driver; ``None`` means use the real driver.
_CAMERA_CLASS_OVERRIDE: type | None = None


def _camera_cls(cam_type: str = "") -> type:
    """Resolve the camera driver class on demand.

    ``ao_shaping.drivers`` exposes ``MIICamera`` through a PEP 562
    ``__getattr__``, so a module-level ``from ao_shaping.drivers import
    MIICamera`` would load the native MiiCam SDK (``_setup_miicam_sdk()`` ->
    ``ctypes.CDLL``) on every ``import ao_shaping`` -- defeating the lazy
    contract and forcing the SDK onto offline consumers such as
    ``scripts/slm/generate_shape_objective_comparison.py``. Resolving here keeps
    the optimizer importable without the side effect.

    Precedence: an explicit :data:`_CAMERA_CLASS_OVERRIDE` wins, so tests can
    force any class; then ``cam_type="sim"`` for the 2f-Fourier simulator;
    otherwise the real MiiCam driver.
    """
    if _CAMERA_CLASS_OVERRIDE is not None:
        return _CAMERA_CLASS_OVERRIDE

    if cam_type == "sim":
        from ao_shaping.drivers.sim.slm_pib_sim import SimPibCCD

        return SimPibCCD

    from ao_shaping.drivers import MIICamera

    if MIICamera is None:
        raise RuntimeError("MiiCam driver not available. Install the MiiCam SDK.")
    return MIICamera


@contextmanager
def _camera_override(camera_cls: type | None):
    """Temporarily swap the camera class used by the optimizers.

    ``_CAMERA_CLASS_OVERRIDE`` is module state, so assigning it without
    restoring would silently make a later hardware run use the simulator. This
    restores the previous value on exit, including on exception, and nests
    correctly.
    """
    global _CAMERA_CLASS_OVERRIDE
    previous = _CAMERA_CLASS_OVERRIDE
    _CAMERA_CLASS_OVERRIDE = camera_cls
    try:
        yield camera_cls
    finally:
        _CAMERA_CLASS_OVERRIDE = previous
