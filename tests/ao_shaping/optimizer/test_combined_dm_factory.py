"""#91 residual: the ``combined_optimizer`` DM factory must be the FILTERING one.

``combined_optimizer.optimize_pib(dm=None)`` used to build its default DM
through the unfiltered registry factory ``create()`` while forwarding
``max_voltage`` / ``min_voltage``. No registered hardware DM accepts either
at construction time -- ``NLight.__init__`` takes only ``max_iter_diff`` /
``max_neibor_diff`` / ``keep_when_exit`` / ``safety_mode``, and on
``base.DM`` those two limits are read-only ``V_Min`` / ``V_Max`` properties
enforced when updates are clipped -- so a direct call with ``dm=None``
raised ``TypeError`` on hardware. The simulation harness hid this because
its ``SimDM`` shim deliberately models the limits and accepts the kwargs.

Fix: route the call site through ``create_dm()``, which filters kwargs per
registered type (``_KWARG_FILTERS``). These tests pin:

1. behaviour: running the combined path with a strict NLight-shaped DM
   through a registry that mirrors the real per-type filtering completes
   without ``TypeError``, and the factory actually received the dead kwargs
   (i.e. the call site still forwards them, the factory drops them);
2. the call site uses ``create_dm`` and never the unfiltered ``create`` --
   with a bite self-check so the scan cannot be vacuous;
3. the inertness contract itself: the two limits are not in the accepted
   set for ``nlight`` (if someone adds them, this test forces a re-read of
   #91 before behaviour on hardware changes silently).
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np

from ao_shaping.drivers.dm._registry import _KWARG_FILTERS
from ao_shaping.optimizer import combined_optimizer as combined_module
from ao_shaping.optimizer.wfless.pib_sim_eval import (
    ACTIVE_MASK,
    LANDSCAPE,
    SimDM,
    patched_sensorless_simulation,
)

SOURCE = Path(__file__).resolve().parents[3] / (
    "src/ao_shaping/optimizer/combined_optimizer.py"
)

_ARGS = dict(
    center=LANDSCAPE.center,
    epochs=2,
    r_bucket=10,
    delta=1.2,
    lr=0.5,
    exposure_time_ms=20,
    shrink_iter=0,
    cam_id=0,
    show=False,
    init_v=np.zeros(8, dtype=np.float64),
    cam_size=LANDSCAPE.shape[0],
    target_max_brightness=0,
    dm_unit_mask=ACTIVE_MASK.copy(),
    dm_neibor_diff=100,
    dm_max_voltage=12,
    dm_min_voltage=-12,
)


class StrictNLightDM(SimDM):
    """``SimDM`` behaviour with ``NLight``'s exact constructor signature."""

    def __init__(
        self,
        max_iter_diff: int = 20,
        max_neibor_diff: float = 200,
        keep_when_exit: bool = True,
        safety_mode: bool = True,
    ) -> None:
        super().__init__()
        self.max_iter_diff = max_iter_diff
        self.max_neibor_diff = max_neibor_diff
        self.keep_when_exit = keep_when_exit
        self.safety_mode = safety_mode


class _FilteringRegistry:
    """Registry stub mirroring the real ``create_dm`` kwarg filtering."""

    def __init__(self) -> None:
        self.create_dm_kwargs: dict[str, Any] | None = None
        self.built: StrictNLightDM | None = None

    def create(self, dm_type: str, **kwargs: Any) -> SimDM:
        raise AssertionError(
            "combined_optimizer must build its default DM through the"
            f" filtering factory create_dm(), got create({dm_type!r})"
        )

    def create_dm(self, dm_type: str, **kwargs: Any) -> SimDM:
        self.create_dm_kwargs = dict(kwargs)
        accepted = _KWARG_FILTERS.get(dm_type.lower(), set())
        filtered = (
            {k: v for k, v in kwargs.items() if k in accepted} if accepted else kwargs
        )
        self.built = StrictNLightDM(**filtered)
        return self.built


def test_combined_none_dm_survives_strict_nlight_signature() -> None:
    """The ``dm=None`` path must reach a strict NLight-shaped constructor."""
    fake = _FilteringRegistry()
    original = combined_module.get_dm_registry
    state = np.random.get_state()
    try:
        with patched_sensorless_simulation():
            # Re-patch *after* the harness installed its own stub, so this
            # seam is the one the optimizer sees.
            combined_module.get_dm_registry = lambda: fake
            np.random.seed(1)
            combined_module.optimize_pib(**_ARGS)
    finally:
        combined_module.get_dm_registry = original
        np.random.set_state(state)

    assert fake.create_dm_kwargs is not None, "factory was never reached"
    assert {"max_voltage", "min_voltage"} <= set(fake.create_dm_kwargs), (
        "the call site must still forward the limits (sim-parity contract)"
    )
    assert isinstance(fake.built, StrictNLightDM), (
        "the DM was not built through the strict-signature path"
    )


def test_create_site_uses_the_filtering_factory() -> None:
    """AST: exactly one ``get_dm_registry().create_dm`` and no ``.create``."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    unfiltered, filtered = _registry_factory_calls(tree)
    assert not unfiltered, (
        "combined_optimizer builds a DM through the unfiltered create();"
        " it must go through create_dm() (#91: NLight rejects voltage kwargs)"
    )
    assert filtered == 1, f"expected exactly one create_dm call site, got {filtered}"


def test_the_factory_scan_actually_bites() -> None:
    """A scan that cannot fail is not a guard: seed a violation tree."""
    bad = ast.parse(
        "from x import get_dm_registry\n"
        "f = lambda: get_dm_registry().create('nlight', max_voltage=12)\n"
    )
    unfiltered, filtered = _registry_factory_calls(bad)
    assert unfiltered == 1 and filtered == 0


def test_limits_are_inert_for_registered_nlight() -> None:
    """Why filtering is the fix: the limits are not constructor kwargs."""
    accepted = _KWARG_FILTERS.get("nlight", set())
    assert "max_voltage" not in accepted
    assert "min_voltage" not in accepted


def _registry_factory_calls(tree: ast.AST) -> tuple[int, int]:
    """Count ``get_dm_registry().create(...)`` / ``.create_dm(...)`` calls."""
    unfiltered = filtered = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        value = node.func.value
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "get_dm_registry"
        ):
            continue
        if node.func.attr == "create":
            unfiltered += 1
        elif node.func.attr == "create_dm":
            filtered += 1
    return unfiltered, filtered
