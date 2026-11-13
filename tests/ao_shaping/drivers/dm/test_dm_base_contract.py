"""``BaseDM`` is the single DM base class and its public export contract.

The Micro-DM refactor introduced a real ``BaseDM`` parent so that a single
controller (:class:`R50Controller`) can reuse the DM channel model without ever
becoming a registered DM type. The former ``DM`` compatibility subclass has
since been removed; ``BaseDM`` is now the *only* name. These tests pin the
invariants that removal depends on:

* every driver inherits from ``BaseDM`` directly;
* ``DM`` is gone from the package surface (not importable, not in ``__all__``);
* the DM registry / lazy attribute map still resolves exactly the same types
  (``BaseDM`` itself is *not* a backend and ``R50Controller`` is *not*
  registered);
* importing the package still does not import any driver module.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import ao_shaping.drivers.dm as dm_package
import ao_shaping.drivers.dm.micro as micro_package
from ao_shaping.drivers.dm import (
    BaseDM,
    MicroDM,
    list_dm_types,
)
from ao_shaping.drivers.dm.micro import MAX_CHANNELS, R50Controller
from tests.ao_shaping.drivers._subprocess_env import subprocess_env_with_pythonpath

_ROOT = Path(__file__).resolve().parents[4]


class TestBaseDmIsTheOnlyBase:
    """``BaseDM`` is the implementation and the only exported base name."""

    def test_drivers_inherit_base_dm(self) -> None:
        assert issubclass(MicroDM, BaseDM)

    def test_dm_alias_is_gone(self) -> None:
        # The compatibility subclass was deleted; the name must not linger on
        # the package or in its export list, and it must not be a lazy backend.
        assert not hasattr(dm_package, "DM")
        assert "DM" not in dm_package.__all__
        assert "DM" not in dm_package._LAZY_BACKENDS


class TestExports:
    """``BaseDM`` is importable from the historical package paths."""

    def test_dm_package_exports_base_dm(self) -> None:
        assert "BaseDM" in dm_package.__all__
        assert dm_package.BaseDM is BaseDM

    def test_micro_package_exports_the_same_base_dm(self) -> None:
        assert "BaseDM" in micro_package.__all__
        assert micro_package.BaseDM is BaseDM

    def test_base_dm_is_not_a_lazy_backend(self) -> None:
        # BaseDM lives in ``base`` which the package already imports eagerly;
        # putting it in the lazy map would shadow that binding and change the
        # import graph.
        assert "BaseDM" not in dm_package._LAZY_BACKENDS
        assert "BaseDM" not in micro_package._LAZY_BACKENDS


class TestRegistrySnapshot:
    """Registry and CLI-visible type list are untouched by the removal."""

    def test_micro_types_still_registered(self) -> None:
        types = list_dm_types()
        assert "micro" in types
        assert "asyn_micro" in types

    def test_base_dm_is_not_a_registered_dm_type(self) -> None:
        assert "BaseDM" not in list_dm_types()

    def test_r50_controller_is_not_a_registered_dm_type(self) -> None:
        # The controller is an internal building block; only ``MicroDM`` is a
        # selectable DM type for runners/CLI.
        assert "r50" not in list_dm_types()
        assert "R50Controller" not in list_dm_types()


class TestR50ControllerBaseDmContract:
    """Single-controller client is a BaseDM, but not a registered DM type."""

    def test_is_base_dm(self) -> None:
        assert issubclass(R50Controller, BaseDM)

    def test_channel_model_matches_hardware(self) -> None:
        assert R50Controller.DM_NUM == MAX_CHANNELS
        assert MAX_CHANNELS == 50

    @pytest.mark.parametrize("member", ["open", "close", "is_connected", "readback"])
    def test_has_dm_surface(self, member: str) -> None:
        assert callable(getattr(R50Controller, member))


class TestLazyImportPreserved:
    """The eager ``BaseDM`` export did not replace the lazy backend map."""

    def test_fresh_interpreter_resolves_base_dm_and_lazy_backends(self) -> None:
        # ``BaseDM`` is bound eagerly (``base`` only needs numpy), while every
        # backend name must still resolve through ``install_lazy_attrs``. This
        # runs in a pristine interpreter so an already-imported module in the
        # pytest process cannot mask a regression.
        snippet = """
            import ao_shaping.drivers.dm as dm_pkg
            import ao_shaping.drivers.dm.micro as micro_pkg

            assert dm_pkg.BaseDM is micro_pkg.BaseDM
            assert not hasattr(dm_pkg, "DM")

            # Attribute access still resolves the historical path.
            assert dm_pkg.MicroDM.__name__ == "MicroDM"
            assert micro_pkg.MicroDM is dm_pkg.MicroDM
            print("lazy-ok")
        """
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(snippet)],
            capture_output=True,
            text=True,
            cwd=_ROOT,
            env=subprocess_env_with_pythonpath(root=str(_ROOT), absolute=True),
            timeout=180,
        )
        assert result.returncode == 0, result.stderr
        assert "lazy-ok" in result.stdout
