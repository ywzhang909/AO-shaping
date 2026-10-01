"""Contract tests for the shared WFS base class.

Motivation (2026-10-01): the WFS family had **no** shared base class —
``ThorlabWFS`` extended ``Device`` directly — while the camera family has
``BaseCamera`` and the DM family has ``DM``, and the corresponding *simulated*
devices conform to those bases (``SimPibCCD(BaseCamera)``, ``SimulateDM(DM)``,
``SimMicroDM(DM)``). That missing shared contract is precisely why no simulated
WFS could be written against the real driver's surface.

These tests pin the new contract so a simulated WFS can be a genuine drop-in:

* ``BaseWFS`` exists, is abstract, and derives from ``Device``.
* ``ThorlabWFS`` is a subclass of it (so real and simulated share one contract).
* Every method the optimizers/runners actually call is part of the contract, so
  a partial implementation fails loudly at instantiation.
* The shared WFS parameters are registered by the base, not re-declared per driver.
* ``calc_n_zernike_terms`` keeps its documented piston offset instead of
  silently diverging from the canonical ``zernike_calc`` helper.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.drivers.device_base import Device
from ao_shaping.drivers.wfs.base import BaseWFS
from ao_shaping.drivers.wfs.thorlab_wfs import ThorlabWFS
from ao_shaping.utils.wavefront.zernike_calc import (
    calc_n_zernike_terms as canonical_calc_n_zernike_terms,
)

#: Every WFS entry point the optimizers / runners actually call, with the
#: call counts observed by grepping ``optimizer/`` and ``runners/`` (2026-10-01).
REQUIRED_METHODS = (
    "take_image",
    "get_wavefront",
    "get_spot_deviation",
    "get_zernike",
    "get_spots_statics",
    "build_subaperture_mask",
)

#: ``Device``'s own abstract surface, stubbed so a test subclass fails only
#: because of a missing *WFS* method.
DEVICE_METHODS = ("open", "close", "is_connected", "get_hardware_info")


def _wfs_subclass(name: str, omit: str | None = None) -> type:
    """Build a WFS subclass missing only ``omit`` (a WFS-contract method)."""
    namespace: dict[str, object] = {
        m: (lambda self, *a, **k: None) for m in DEVICE_METHODS
    }
    namespace.update(
        {m: (lambda self, *a, **k: None) for m in REQUIRED_METHODS if m != omit}
    )
    return type(name, (BaseWFS,), namespace)


class TestBaseWFSShape:
    def test_is_a_device_subclass(self):
        assert issubclass(BaseWFS, Device)

    def test_is_abstract(self):
        with pytest.raises(TypeError):
            BaseWFS()  # type: ignore[abstract]

    def test_thorlab_wfs_shares_the_contract(self):
        assert issubclass(ThorlabWFS, BaseWFS), (
            "the real driver must implement the shared WFS contract, otherwise a "
            "simulated WFS is not a drop-in replacement"
        )


class TestContractCompleteness:
    @pytest.mark.parametrize("name", REQUIRED_METHODS)
    def test_real_driver_satisfies_contract(self, name):
        assert callable(getattr(ThorlabWFS, name, None)), f"{name} missing"

    @pytest.mark.parametrize("name", REQUIRED_METHODS)
    def test_contract_is_enforced(self, name):
        """A partial implementation must fail at instantiation, not at run time."""
        with pytest.raises(TypeError):
            _wfs_subclass("PartialWFS", omit=name)()  # type: ignore[abstract]

    def test_enforcement_is_not_vacuous(self):
        """Guard the guard: the complete subclass really is instantiable."""
        assert isinstance(_wfs_subclass("ConcreteWFS")(), BaseWFS)

    def test_minimal_subclass_can_be_instantiated(self):
        """Once the contract is met, subclassing must be straightforward.

        This is the shape a SimulatedWFS will take.
        """
        obj = _wfs_subclass("ConcreteWFS")()  # type: ignore[abstract]
        assert isinstance(obj, BaseWFS)


class TestSharedParameters:
    def test_base_registers_the_shared_wfs_parameters(self):
        """The base owns the common parameters; drivers must not re-declare them."""
        obj = _wfs_subclass("ParamWFS")()  # type: ignore[abstract]
        names = set(obj.list_parameters())
        for expected in ("exposure_time_ms", "remove_tilt"):
            assert expected in names, (
                f"{expected} must be registered by BaseWFS so every WFS driver "
                f"exposes it; got {sorted(names)}"
            )

    def test_driver_still_exposes_its_own_parameters(self):
        """Refactoring must not strip the real driver's parameter set."""
        names = set(ThorlabWFS.__dict__)
        assert "_register_parameters" in names or "register_parameter" in {
            m for m in dir(ThorlabWFS)
        }, "ThorlabWFS must keep declaring its own parameters"

    def test_driver_may_override_a_shared_parameter(self):
        """A driver re-declaring a shared parameter must win, not crash.

        ``ThorlabWFS`` declares ``exposure_time_ms`` itself; the base declares
        it too. Last registration wins, so the driver's hardware-specific bounds
        survive the split.
        """
        namespace: dict[str, object] = {
            m: (lambda self, *a, **k: None) for m in DEVICE_METHODS
        }
        namespace.update(
            {m: (lambda self, *a, **k: None) for m in REQUIRED_METHODS}
        )

        def _own(self):
            self.register_parameter(
                "exposure_time_ms",
                default_value=0.0,
                min_value=0.0,
                max_value=7.0,
                unit="ms",
                description="driver override",
            )

        def __init__(self, device_id: str = "") -> None:
            BaseWFS.__init__(self, device_id)
            self._register_parameters()

        namespace["_register_parameters"] = _own
        namespace["__init__"] = __init__
        concrete = type("OverrideWFS", (BaseWFS,), namespace)
        obj = concrete()  # type: ignore[abstract]
        param = obj.get_parameter("exposure_time_ms")
        assert param is not None
        assert param.max_value == 7.0
        assert param.description == "driver override"


class TestZernikeTermCountDivergence:
    """Pin the +1 piston offset so it cannot drift silently.

    ``ThorlabWFS.calc_n_zernike_terms`` counts **with** piston; the canonical
    ``zernike_calc.calc_n_zernike_terms`` counts **without**. Both are correct
    for their own consumers, but the two must not be confused — so the
    relationship is asserted rather than left implicit.
    """

    @pytest.mark.parametrize("n", [0, 1, 2, 4, 6, 10])
    def test_wfs_counts_include_piston(self, n):
        assert ThorlabWFS.calc_n_zernike_terms(n) == (n + 1) * (n + 2) // 2 + 1

    @pytest.mark.parametrize("n", [0, 1, 2, 4, 6, 10])
    def test_canonical_excludes_piston(self, n):
        assert canonical_calc_n_zernike_terms(n) == (n + 1) * (n + 2) // 2

    @pytest.mark.parametrize("n", [0, 1, 2, 4, 6, 10])
    def test_offset_is_exactly_piston(self, n):
        assert ThorlabWFS.calc_n_zernike_terms(n) == canonical_calc_n_zernike_terms(
            n
        ) + 1
