"""W1 — the sensorless runners must be runnable under simulation.

``slm-pib`` and ``slm-gsnet`` already expose ``--cam_type`` because their
parameter dataclasses inherit ``CameraParams``. Two of the three DM/SLM
sensorless runners do not, so they cannot be exercised without hardware:

===================  ==========================================  ==============
runner               param dataclass                             ``cam_type``
===================  ==========================================  ==============
``pib``              ``PibRunnerParams``                          added
``spgd-square``      ``SlmSquareParams``                          added
``combined``         ``CombinedRunnerParams``                     already present
===================  ==========================================  ==============

``combined`` is the exception: its CLI is wired with a second
``@with_params(CameraParams, kw_name="camera")`` decorator, so ``--cam_type``
already reaches it. It must **not** be declared again on
``CombinedRunnerParams`` — that would register a duplicate Click option name.

``optimize_slm_square`` already accepts ``cam_type`` and resolves the camera
through the ``create_camera`` registry, so ``spgd-square`` only needs the CLI
field. ``optimize_pib`` has no such parameter; it resolves the driver through
``pib._CAMERA_CLASS_OVERRIDE``, an explicit injection point added for exactly
this purpose.

There is also no simulated NLight-equivalent DM. Only ``sim_micro`` (a 50-channel
Micro-DM) is registered, while ``wf`` / ``pipeline`` / ``dm-matrix`` / ``pib`` /
``combined`` all expect a 64-actuator NLight, so ``--dm_type`` has no
simulation-friendly value to select.
"""

from __future__ import annotations

import pytest

from ao_shaping.drivers.dm import list_dm_types


class TestSimDmRegistration:
    def test_simulate_dm_is_registered(self):
        assert "sim" in list_dm_types(), (
            f"a simulated 64-channel DM must be selectable via --dm_type; "
            f"registered: {sorted(list_dm_types())}"
        )

    def test_registration_is_visible_from_the_registry(self):
        from ao_shaping.drivers.dm._registry import create_dm

        dm = create_dm("sim")
        try:
            assert dm.DM_NUM == 64, f"the NLight stand-in must expose 64 actuators, got {dm.DM_NUM}"
        finally:
            dm.close()

    def test_sim_micro_still_registered(self):
        """The existing simulation type must not be displaced."""
        assert "sim_micro" in list_dm_types()


class TestRunnerParamsExposeCamType:
    @pytest.mark.parametrize(
        ("module", "cls"),
        [
            ("ao_shaping.runners.runner_common", "PibRunnerParams"),
            ("ao_shaping.runners.runner_common", "SlmSquareParams"),
        ],
    )
    def test_param_class_has_cam_type(self, module, cls):
        import importlib

        params = getattr(importlib.import_module(module), cls)
        assert "cam_type" in params.__dataclass_fields__, (
            f"{cls} cannot select a simulated camera without a cam_type field"
        )

    def test_combined_gets_cam_type_from_camera_params(self):
        """``combined`` must NOT duplicate the field.

        Its CLI is wired as ``@with_params(CombinedRunnerParams, ...)`` plus
        ``@with_params(CameraParams, kw_name="camera")``, so ``--cam_type``
        already arrives via ``CameraParams``. Declaring it again on
        ``CombinedRunnerParams`` would register a second Click option with the
        same name.
        """
        from ao_shaping.runners.runner_common import CameraParams, CombinedRunnerParams

        assert "cam_type" in CameraParams.__dataclass_fields__
        assert "cam_type" not in CombinedRunnerParams.__dataclass_fields__, (
            "combined already receives --cam_type through CameraParams; a second "
            "declaration collides on the Click option name"
        )

    def test_slm_square_params_offers_sim(self):
        from ao_shaping.runners.runner_common import SlmSquareParams

        assert SlmSquareParams().cam_type in {"daheng", "miicam", "sim"}


class TestPibCameraInjectionSeam:
    def test_override_seam_exists_and_defaults_to_none(self):
        from ao_shaping.optimizer.wfless import pib

        assert hasattr(pib, "_CAMERA_CLASS_OVERRIDE")
        assert pib._CAMERA_CLASS_OVERRIDE is None, (
            "the seam must default to the real driver so hardware runs are unaffected"
        )

    def test_override_changes_the_resolved_camera_class(self):
        from ao_shaping.drivers.sim.slm_pib_sim import SimPibCCD
        from ao_shaping.optimizer.wfless import pib

        original = pib._CAMERA_CLASS_OVERRIDE
        try:
            pib._CAMERA_CLASS_OVERRIDE = SimPibCCD
            assert pib._camera_cls() is SimPibCCD
        finally:
            pib._CAMERA_CLASS_OVERRIDE = original

    def test_seam_is_restored_after_use(self):
        """The seam is module state; leaking it would poison later hardware runs."""
        from ao_shaping.drivers.sim.slm_pib_sim import SimPibCCD
        from ao_shaping.optimizer.wfless import pib

        original = pib._CAMERA_CLASS_OVERRIDE
        try:
            with pib._camera_override(SimPibCCD):
                assert pib._CAMERA_CLASS_OVERRIDE is SimPibCCD
            assert pib._CAMERA_CLASS_OVERRIDE is original
        finally:
            pib._CAMERA_CLASS_OVERRIDE = original

    def test_nested_overrides_unwind_correctly(self):
        from ao_shaping.drivers.sim.slm_pib_sim import SimPibCCD
        from ao_shaping.optimizer.wfless import pib

        original = pib._CAMERA_CLASS_OVERRIDE
        try:
            with pib._camera_override(SimPibCCD):
                with pib._camera_override(None):
                    assert pib._CAMERA_CLASS_OVERRIDE is None
                assert pib._CAMERA_CLASS_OVERRIDE is SimPibCCD
        finally:
            pib._CAMERA_CLASS_OVERRIDE = original
