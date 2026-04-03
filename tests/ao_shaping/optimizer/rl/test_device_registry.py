from ao_shaping.optimizer.rl.device_registry import build_default_registry


def test_default_registry_contains_virtual_and_physical_devices() -> None:
    registry = build_default_registry()

    dm_names = registry.names("dm")
    ccd_names = registry.names("ccd")
    wfs_names = registry.names("wfs")

    assert "sim_dm" in dm_names
    assert "nlight_dm" in dm_names
    assert "sim_ccd" in ccd_names
    assert "miicam_ccd" in ccd_names
    assert "sim_wfs" in wfs_names
    assert "thorlabs_wfs" in wfs_names


def test_registry_get_device_spec() -> None:
    registry = build_default_registry()
    sim_dm = registry.get("dm", "sim_dm")
    real_dm = registry.get("dm", "nlight_dm")

    assert sim_dm.is_virtual is True
    assert real_dm.is_virtual is False
