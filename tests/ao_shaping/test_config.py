import os
from pathlib import Path

import pytest

from ao_shaping.config import (
    DEFAULTS,
    PATHS,
    DEVICES,
    OptimizationDefaults,
    PathConfig,
    DeviceConfig,
    DEFAULT_OPTIMIZATION_DEFAULTS,
    DEFAULT_PATHS,
    DEFAULT_DEVICE_CONFIG,
    _resolve_class_attribute,
    _resolve_dm_n_actuators,
    get_dm_unit_mask,
    get_init_voltages,
    get_coredumpy_directory,
)


class TestOptimizationDefaults:
    def test_all_defaults_accessible_as_attributes(self):
        for key, expected in DEFAULT_OPTIMIZATION_DEFAULTS.items():
            assert getattr(DEFAULTS, key) == expected

    def test_getitem_access(self):
        assert DEFAULTS["WF_EPOCHS"] == 20_000

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError):
            _ = DEFAULTS.NONEXISTENT_KEY

    def test_fresh_instance_matches_singleton(self):
        fresh = OptimizationDefaults()
        assert fresh.WF_EPOCHS == DEFAULTS.WF_EPOCHS
        assert fresh.PIB_EPOCHS == DEFAULTS.PIB_EPOCHS

    def test_wf_defaults(self):
        assert DEFAULTS.WF_EPOCHS == 20_000
        assert DEFAULTS.WF_EARLY_STOP_THRESHOLD == 0.12
        assert DEFAULTS.WF_WFS_RES == "768"
        assert DEFAULTS.WF_PUPIL_DIAMETER == 2.7

    def test_pib_defaults(self):
        assert DEFAULTS.PIB_EPOCHS == 4_000
        assert DEFAULTS.PIB_DELTA == 2.0
        assert DEFAULTS.PIB_LR == 0.0
        assert DEFAULTS.PIB_SHRINK_RATIO == 0.8

    def test_pipeline_defaults(self):
        assert DEFAULTS.PIPELINE_WF_EPOCHS == 8_000
        assert DEFAULTS.PIPELINE_PIB_EPOCHS == 8_000
        assert DEFAULTS.PIPELINE_RMS_THRESHOLD == 0.12

    def test_ga_defaults(self):
        assert DEFAULTS.GA_POPULATION_SIZE == 50
        assert DEFAULTS.GA_N_GENERATIONS == 2000
        assert DEFAULTS.GA_CROSSOVER_PROB == 0.7
        assert DEFAULTS.GA_MUTATION_PROB == 0.15


class TestPathConfig:
    def test_root_dir_is_path(self):
        assert isinstance(PATHS.root_dir, Path)

    def test_root_dir_default(self):
        assert PATHS.root_dir == Path("data")

    def test_subdir_defaults(self):
        assert PATHS.voltages_dir == "flatten_voltages"
        assert PATHS.wf_subdir == "wf"
        assert PATHS.pib_subdir == "wf-less"

    def test_log_dir_is_path(self):
        assert isinstance(PATHS.log_dir, Path)

    def test_get_voltages_path_returns_path(self):
        result = PATHS.get_voltages_path("20260101")
        assert isinstance(result, Path)
        assert "20260101" in str(result)

    def test_get_voltages_path_uses_today_when_none(self):
        result = PATHS.get_voltages_path()
        assert isinstance(result, Path)

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError):
            _ = PATHS.NONEXISTENT_KEY


class TestDeviceConfig:
    def test_far_cam_id_default(self):
        assert DEVICES.far_cam_id == int(os.environ.get("Far_Cam_ID", 0))

    def test_near_cam_id_default(self):
        assert DEVICES.near_cam_id == int(os.environ.get("Near_Cam_ID", 1))

    def test_fallback_defaults(self):
        assert DEVICES.slm_number == 1
        assert DEVICES.slm_wavelength == 532
        assert DEVICES.exposure_time_ms == 0.0

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError):
            _ = DEVICES.NONEXISTENT_KEY


class TestGetDmUnitMask:
    def test_all_mask_length(self):
        mask = get_dm_unit_mask("all")
        assert len(mask) >= 64

    def test_all_mask_mostly_true(self):
        mask = get_dm_unit_mask("all")
        true_count = sum(1 for m in mask if m)
        assert true_count > len(mask) * 0.8

    def test_inner_mask_shorter(self):
        mask = get_dm_unit_mask("inner")
        true_count = sum(1 for m in mask if m)
        assert true_count <= 21

    def test_outer_mask(self):
        mask = get_dm_unit_mask("outer")
        true_count = sum(1 for m in mask if m)
        assert true_count < len(mask) * 0.5


class TestGetInitVoltages:
    def test_returns_list(self):
        voltages = get_init_voltages()
        assert isinstance(voltages, list)

    def test_all_zeros(self):
        voltages = get_init_voltages()
        assert all(v == 0.0 for v in voltages)

    def test_length_matches_actuator_count(self):
        voltages = get_init_voltages()
        mask = get_dm_unit_mask("all")
        assert len(voltages) == len(mask)


class TestResolveClassAttribute:
    """Anchor the property-descriptor resolution semantics of the helper."""

    def test_plain_class_attribute_passthrough(self):
        class Dummy:
            DM_NUM = 64

        assert _resolve_class_attribute(Dummy, "DM_NUM") == 64

    def test_instance_property_resolves_to_value(self):
        class Dummy:
            @property
            def DM_NUM(self):
                return 64

        assert _resolve_class_attribute(Dummy, "DM_NUM") == 64

    def test_missing_attribute_returns_none(self):
        class Dummy:
            pass

        assert _resolve_class_attribute(Dummy, "DM_NUM") is None

    def test_property_instantiation_failure_returns_none(self):
        class Dummy:
            @property
            def DM_NUM(self):
                raise RuntimeError("cannot instantiate")

        assert _resolve_class_attribute(Dummy, "DM_NUM") is None


class TestResolveDmWithPropertyDm:
    """Regression: ZernikeDM/HadamardDM expose ``DM_NUM`` as an instance
    ``@property``; ``config._resolve_dm_n_actuators()`` previously returned the
    property descriptor object (not an int) when such a DM was the first
    reachable type, breaking ``get_dm_unit_mask`` with a TypeError in
    ``[True] * n_actuators`` (self-registering modules populate the registry).

    Uses a fake registry that resolves to the REAL HadamardDM class, so the
    exact production code path (lazy import -> registry.get_class ->
    _resolve_class_attribute -> instantiate) is exercised without hitting
    network probes for unreachable hardware DMs."""

    def test_registry_property_dm_resolves_to_consistent_mask(self, monkeypatch):
        import ao_shaping.drivers.dm.hadamard_dm  # noqa: F401  (注册 HadamardDM)
        import ao_shaping.drivers.dm.zernike_dm  # noqa: F401  (注册 ZernikeDM)
        from ao_shaping.drivers.dm.hadamard_dm import HadamardDM

        assert isinstance(getattr(HadamardDM, "DM_NUM"), property)

        class FakeRegistry:
            def list_reachable_types(self):
                return ["hadamard"]

            def get_class(self, name):
                return HadamardDM

        monkeypatch.setattr(
            "ao_shaping.drivers.dm._registry.get_dm_registry",
            lambda: FakeRegistry(),
        )

        n = _resolve_dm_n_actuators()
        assert isinstance(n, int)
        assert n == HadamardDM().DM_NUM

        mask = get_dm_unit_mask("all")
        assert isinstance(mask, list)
        assert len(mask) == n
        assert all(isinstance(m, bool) for m in mask)

        voltages = get_init_voltages()
        assert len(voltages) == n
        assert all(v == 0.0 for v in voltages)


class TestNonActuatorDMsAreExcluded:
    """``DM_N_ACTUATORS`` must not be sourced from a virtual or modal count.

    ``registry.is_reachable()`` hardcodes ``True`` for exactly the classes that
    have no device behind them, and they sort ahead of a real NLight entry, so
    "first reachable" resolved to ``SimMicroDM``'s 50 -- and, once that was
    excluded, to ``ZernikeDM``'s ``n_modes`` (15), which is a Zernike modal
    order rather than an actuator count at all.
    """

    def test_simulated_and_zernike_dms_are_rejected(self):
        from ao_shaping.config import _counts_physical_actuators

        class FakeSimDM:
            __module__ = "ao_shaping.drivers.sim.dm.simulated_micro_dm"

        class FakeZernikeDM:
            __module__ = "ao_shaping.drivers.dm.zernike_dm"

        class FakeNLightDM:
            __module__ = "ao_shaping.drivers.dm.nlight.udp"

        assert _counts_physical_actuators(FakeSimDM) is False
        assert _counts_physical_actuators(FakeZernikeDM) is False
        assert _counts_physical_actuators(FakeNLightDM) is True

    def test_real_registry_excludes_the_virtual_dms(self):
        from ao_shaping.config import _counts_physical_actuators
        from ao_shaping.drivers.dm._registry import get_dm_registry

        registry = get_dm_registry()
        reachable = registry.list_reachable_types()
        assert "sim_micro" in reachable, "fixture assumes the sim DM is reachable"

        for name in reachable:
            cls = registry.get_class(name)
            if cls.__module__.startswith("ao_shaping.drivers.sim"):
                assert _counts_physical_actuators(cls) is False, name

    def test_resolver_falls_back_to_64_when_only_virtual_dms_are_reachable(self):
        import ao_shaping.config as config

        class VirtualOnlyRegistry:
            def list_reachable_types(self):
                return ["sim_micro", "zernike"]

            def get_class(self, name):
                from ao_shaping.drivers.dm._registry import get_dm_registry

                return get_dm_registry().get_class(name)

        original = config._resolve_dm_n_actuators
        import ao_shaping.drivers.dm._registry as registry_mod

        real_get = registry_mod.get_dm_registry
        registry_mod.get_dm_registry = lambda: VirtualOnlyRegistry()
        try:
            config.reset_dm_resolution_cache()
            assert config._resolve_dm_n_actuators() == 64
        finally:
            registry_mod.get_dm_registry = real_get
            config.reset_dm_resolution_cache()
            assert config._resolve_dm_n_actuators() == original()


class TestGetCoredumpyDirectory:
    def test_returns_string(self):
        result = get_coredumpy_directory()
        assert isinstance(result, str)

    def test_contains_log_path(self):
        result = get_coredumpy_directory()
        assert "log" in result.lower() or "debug" in result.lower() or "error" in result.lower()


class TestModuleGetattr:
    def test_dm_n_actuators_accessible(self):
        import ao_shaping.config as config
        assert isinstance(config.DM_N_ACTUATORS, int)
        assert config.DM_N_ACTUATORS >= 1

    def test_dm_disabled_actuators_accessible(self):
        import ao_shaping.config as config
        assert isinstance(config.DM_DISABLED_ACTUATORS, list)

    def test_unknown_module_attribute_raises(self):
        import ao_shaping.config as config
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = config.NONEXISTENT_MODULE_ATTR
