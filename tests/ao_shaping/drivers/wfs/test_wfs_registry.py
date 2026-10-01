"""Contract tests for the wavefront-sensor registry.

Cameras got ``register_camera``/``create_camera`` and DMs got
``register_dm``/``resolve_dm``, but WFS had no seam at all: nine runners
constructed ``ThorlabWFS(...)`` directly at thirteen call sites, so a simulated
sensor could not be substituted even after one existed.

The registry also has to preserve two contracts the drivers already rely on:

* constructing a driver must not load its native SDK (that is why ``ThorlabWFS``
  defers ``load_dll()`` to ``open()``), so ``create_wfs("thorlab")`` has to work
  on a machine with no WFS hardware;
* ``resolve_wfs(None)`` must return Thorlab, because that is what the hardcoded
  call sites already do today.
"""

from __future__ import annotations

import pytest

from ao_shaping.drivers.wfs._registry import (
    WFSRegistry,
    create_wfs,
    get_wfs_registry,
    list_wfs_types,
    register_wfs,
    resolve_wfs,
)
from ao_shaping.drivers.wfs.base import BaseWFS


@pytest.fixture(scope="module")
def registry():
    return get_wfs_registry()


class TestDiscovery:
    def test_sim_and_thorlab_are_both_listed(self, registry):
        types = list_wfs_types()
        assert "sim" in types, "the simulated sensor must be selectable by name"
        assert "thorlab" in types

    def test_listing_works_without_importing_sim_first(self, registry):
        """A fresh process that touches only the hardware path must still see ``sim``.

        Binding happens on first registry use, so a process that never reaches the
        simulation package by name still resolves both types. (It cannot be
        asserted that ``drivers.sim`` stays out of ``sys.modules``: importing
        anything under ``ao_shaping.drivers`` runs ``ao_shaping/__init__.py``,
        which already pulls in ``sim``.)
        """
        import subprocess
        import sys

        code = (
            "import sys; sys.path.insert(0, 'src');"
            "import ao_shaping.drivers.wfs.thorlab_wfs;"
            "from ao_shaping.drivers.wfs._registry import list_wfs_types;"
            "t = list_wfs_types();"
            "assert 'sim' in t and 'thorlab' in t, t;"
            "print('ok')"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
        )
        assert out.returncode == 0, out.stderr[-2000:]
        assert "ok" in out.stdout

    def test_get_wfs_registry_is_a_singleton(self):
        assert get_wfs_registry() is get_wfs_registry()


class TestCreate:
    def test_create_sim_returns_a_base_wfs(self, registry):
        wfs = create_wfs("sim")
        assert isinstance(wfs, BaseWFS)
        assert wfs.get_hardware_info()["simulated"] is True

    def test_create_is_case_insensitive(self, registry):
        assert isinstance(create_wfs("SIM"), BaseWFS)

    def test_create_thorlab_does_not_require_hardware(self, registry):
        """Construction must stay SDK-free; only ``open()`` may touch the DLL.

        ``ThorlabWFS`` used to call ``load_dll()`` in ``__init__``, so merely
        constructing it raised ``OSError`` without the vendor library.
        """
        wfs = create_wfs("thorlab", mla_index=512)
        assert isinstance(wfs, BaseWFS)
        assert not wfs.is_connected()

    def test_unknown_type_lists_the_alternatives(self, registry):
        with pytest.raises(ValueError, match="sim"):
            create_wfs("no_such_wfs")

    def test_has_type(self, registry):
        assert registry.has_type("sim")
        assert not registry.has_type("nope")


class TestKwargFiltering:
    def test_kwargs_for_another_type_are_dropped(self, registry):
        """``exposure_time`` belongs to Thorlab, not to the simulated sensor."""
        wfs = create_wfs("sim", exposure_time=33.0, resolution=96)
        assert wfs.grid == 96

    def test_sim_specific_kwargs_are_forwarded(self, registry):
        wfs = create_wfs("sim", n_subap=8, resolution=64, wavelength_nm=532)
        assert wfs.n_subap == 8

    def test_unknown_kwargs_are_dropped_not_rejected(self, registry):
        """Mirrors ``create_dm``: runners pass one uniform bag of options.

        Silently dropping is the established behaviour for the DM registry, and a
        runner migrating from a direct constructor would otherwise start raising.
        """
        wfs = create_wfs("sim", resolution=96, not_a_real_kwarg=1)
        assert wfs.grid == 96


class TestResolve:
    def test_explicit_type_is_used(self):
        assert resolve_wfs("sim").get_hardware_info()["simulated"] is True

    def test_none_defaults_to_thorlab(self):
        """Preserves what the thirteen hardcoded ``ThorlabWFS()`` call sites do."""
        assert not resolve_wfs(None).get_hardware_info().get("simulated", False)

    def test_resolve_forwards_kwargs(self):
        assert resolve_wfs("sim", resolution=96).grid == 96


class TestRegistration:
    def test_non_base_wfs_is_rejected(self):
        with pytest.raises(TypeError):

            @register_wfs("bogus")
            class NotAWFS:  # noqa: D401 - deliberately not a BaseWFS
                pass

    def test_registering_returns_the_class_unchanged(self):
        class Dummy(BaseWFS):
            pass

        assert register_wfs("dummy_for_test")(Dummy) is Dummy

    def test_get_class_returns_the_registered_class(self, registry):
        from ao_shaping.drivers.sim.wfs.simulated_wfs import SimulatedWFS

        assert registry.get_class("sim") is SimulatedWFS
