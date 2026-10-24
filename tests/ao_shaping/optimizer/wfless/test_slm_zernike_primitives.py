"""Direct tests for the shared SLM+Zernike closed-loop primitives.

These helpers used to be duplicated verbatim in ``slm_zernike_pib`` and
``slm_zernike_shaping``. Before the extraction the only end-to-end offline
coverage targeted ``slm_zernike_shaping`` (the sandbox copy), so the engine
production actually runs -- ``slm_zernike_pib`` -- had no direct coverage of
the hardware primitives at all. These tests pin the shared behaviour and, just
as importantly, pin that the shared module reads *no* module-level aperture
constant: the two consumers disagree on that value (300 vs 600), so it has to
travel as an argument.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.optimizer.wfless import slm_zernike_primitives as common


class FakeSlm:
    """Minimal stand-in recording every ``display_data`` call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def display_data(self, gray, memory_number=None, memory_mode=None) -> None:
        self.calls.append(
            {"memory_number": memory_number, "mode": memory_mode, "shape": gray.shape}
        )


def _mem(rotator: common.SlotRotator) -> int:
    return rotator.next()


class TestSlotRotation:
    def test_slot_stays_within_bounds(self):
        rotator = common.SlotRotator(slot_min=2, slot_max=6)
        seen = [_mem(rotator) for _ in range(20)]
        assert set(seen) == {2, 3, 4, 5, 6}
        assert min(seen) == 2
        assert max(seen) == 6

    def test_never_repeats_consecutively(self):
        rotator = common.SlotRotator(slot_min=2, slot_max=125)
        seen = [_mem(rotator) for _ in range(500)]
        assert all(a != b for a, b in zip(seen, seen[1:]))

    def test_wraps_from_max_back_to_min(self):
        rotator = common.SlotRotator(slot_min=2, slot_max=4)
        assert [_mem(rotator) for _ in range(5)] == [2, 3, 4, 2, 3]

    def test_first_slot_is_min_not_min_minus_one(self):
        # The pre-extraction code seeded the counter at _SLOT_MIN - 1 and
        # incremented before use, so the first written slot was _SLOT_MIN.
        rotator = common.SlotRotator(slot_min=2, slot_max=125)
        assert rotator.next() == 2

    def test_independent_rotators_do_not_share_state(self):
        a = common.SlotRotator(slot_min=2, slot_max=10)
        b = common.SlotRotator(slot_min=2, slot_max=10)
        assert a.next() == b.next() == 2
        assert a.next() == b.next() == 3
        # Advancing one must not move the other.
        _ = a.next()
        assert b.next() == 4


class TestDisplayRotating:
    def test_passes_rotated_slot_and_memory_mode(self):
        slm = FakeSlm()
        rotator = common.SlotRotator(slot_min=2, slot_max=125)
        common.display_rotating(slm, np.zeros((4, 4)), rotator)
        common.display_rotating(slm, np.zeros((4, 4)), rotator)
        assert [c["memory_number"] for c in slm.calls] == [2, 3]

    def test_returns_the_slot_written(self):
        slm = FakeSlm()
        rotator = common.SlotRotator(slot_min=2, slot_max=125)
        assert common.display_rotating(slm, np.zeros((2, 2)), rotator) == 2
        assert common.display_rotating(slm, np.zeros((2, 2)), rotator) == 3

    def test_forwards_internal_memory_mode(self):
        from ao_shaping.drivers.slm.santec import MEMORY_MODE_INTERNAL

        slm = FakeSlm()
        common.display_rotating(slm, np.zeros((2, 2)), common.SlotRotator())
        assert slm.calls[0]["mode"] is MEMORY_MODE_INTERNAL


class _SpyPatternHelper:
    """Captures the radius the phase generator is asked for."""

    def __init__(self) -> None:
        self.radius: float | None = None

    def generate_zernike_polynomial(
        self, coefficients=None, radius=None, n_max=None
    ):
        self.radius = radius
        # Deterministic stand-in; the real one is irrelevant to this contract.
        return np.full((4, 4), float(radius))


class TestZernikeToPhase:
    def test_radius_is_honoured_as_passed(self):
        spy = _SpyPatternHelper()
        common.zernike_to_phase(
            {(2, 0): 1.0}, 4, spy, default_radius=300.0
        )
        assert spy.radius == 300.0
        spy2 = _SpyPatternHelper()
        common.zernike_to_phase({(2, 0): 1.0}, 4, spy2, default_radius=600.0)
        assert spy2.radius == 600.0

    def test_explicit_radius_overrides_default(self):
        spy = _SpyPatternHelper()
        common.zernike_to_phase({(2, 0): 1.0}, 4, spy, 123.0, default_radius=300.0)
        assert spy.radius == 123.0

    def test_matches_parse_then_generate_pipeline(self):
        from ao_shaping.utils.wavefront.pattern_helper import PatternHelper
        from ao_shaping.utils.wavefront.zernike_utils import parse_zernike_coefficients

        coeffs = {(2, 0): 0.4, (4, 0): 0.1}
        helper = PatternHelper(resolution=(1920, 1200))
        ours = common.zernike_to_phase(
            coeffs, 4, helper, default_radius=300.0
        )
        reference = helper.generate_zernike_polynomial(
            parse_zernike_coefficients(coeffs), 300.0, n_max=4
        )
        assert np.array_equal(ours, reference)

    def test_shared_module_declares_no_aperture_constant(self):
        # The two consumers disagree (300 vs 600). A module-level constant here
        # would silently re-forge that 600 -> 300 divergence.
        src = Path(common.__file__).read_text(encoding="utf-8-sig")
        assert "ZERNIKE_APERTURE_RADIUS" not in src


class TestLearningSchedule:
    def _schedule(self, **kw):
        base = dict(
            power_radius=10.0,
            ideal_r=7.0,
            gradient_history=None,
            pib_history=None,
            epoch=0,
            max_lr=1.0,
            max_delta=0.5,
        )
        base.update(kw)
        return common.learning_schedule(**base)

    def test_returns_finite_positive_pair(self):
        lr, delta = self._schedule()
        assert math.isfinite(lr) and math.isfinite(delta)
        assert lr > 0.0
        assert delta > 0.0

    def test_respects_supplied_caps(self):
        lr, delta = self._schedule(power_radius=1e6, max_lr=0.01, max_delta=0.002)
        assert lr <= 0.01
        assert delta <= 0.002

    def test_caps_are_load_bearing(self):
        loose_lr, loose_delta = self._schedule(power_radius=1e6)
        tight_lr, tight_delta = self._schedule(power_radius=1e6, max_lr=0.01, max_delta=0.002)
        assert tight_lr < loose_lr
        assert tight_delta < loose_delta

    def test_ideal_r_drives_the_segment_boundaries(self):
        # power_radius=5 sits exactly on a segment boundary, so ideal_r alone
        # decides the bucket. Caps are raised well above both outcomes; any
        # other power_radius saturates them and makes this test vacuous.
        tight = self._schedule(
            power_radius=5.0, ideal_r=1.0, max_lr=100.0, max_delta=100.0
        )
        loose = self._schedule(
            power_radius=5.0, ideal_r=1000.0, max_lr=100.0, max_delta=100.0
        )
        assert tight != loose
        assert tight == (4.5, 5)
        assert loose == (1.5, 1)

    def test_identical_inputs_are_deterministic(self):
        assert self._schedule() == self._schedule()

    def test_accepts_history_arguments(self):
        lr, delta = self._schedule(
            gradient_history=[0.1, 0.2, 0.3], pib_history=[0.4, 0.5, 0.6]
        )
        assert math.isfinite(lr) and lr > 0.0
        assert math.isfinite(delta) and delta > 0.0


class TestSharedModuleContract:
    def test_module_defines_no_module_level_mutable_state(self):
        # Slot rotation state must be owned per-consumer, never shared through a
        # module global -- otherwise two importers in one process would interleave.
        # Widen beyond dict: a module-level list or set interleaves exactly the same
        # way, so a dict-only check would not deliver what this test's name claims.
        tree = ast.parse(Path(common.__file__).read_text(encoding="utf-8-sig"))
        offenders: list[str] = []
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            value = node.value
            kind: str | None = None
            if isinstance(value, (ast.Dict, ast.List, ast.Set)):
                kind = type(value).__name__
            elif isinstance(value, (ast.ListComp, ast.SetComp, ast.DictComp)):
                kind = f"{type(value).__name__} comprehension"
            elif isinstance(value, ast.Call):
                name = getattr(value.func, "id", None) or getattr(value.func, "attr", None)
                if name in {
                    "dict", "list", "set", "deque", "defaultdict", "Counter", "OrderedDict",
                }:
                    kind = f"{name}()"
            if kind:
                targets = ", ".join(ast.unparse(t) for t in node.targets)
                offenders.append(f"{targets} = <{kind}> at line {node.lineno}")
        assert not offenders, "module-level mutable state: " + "; ".join(offenders)

    def test_public_surface(self):
        for name in (
            "SlotRotator",
            "display_rotating",
            "zernike_to_phase",
            "learning_schedule",
            "default_camera",
            "default_slm",
        ):
            assert hasattr(common, name), name

    def test_module_defines_no_device_classes(self):
        # The module owns the *how to drive* layer, not the devices: it must not
        # bind a driver class, only the MEMORY_MODE_INTERNAL constant.
        assert "Santec" not in common.__dict__
        assert isinstance(common.MEMORY_MODE_INTERNAL, int)

    def test_import_does_not_load_a_native_sdk(self):
        """The module docstring claims it stays importable without native SDKs.

        Assert that for real, in a clean interpreter: ``gxipy`` is the Daheng
        SDK wrapper, so its absence from ``sys.modules`` is the observable
        proof that no ``ctypes.CDLL`` was taken.
        """
        import subprocess
        import sys

        code = (
            "import sys;"
            "import ao_shaping.optimizer.wfless.slm_zernike_primitives;"
            "assert 'gxipy' not in sys.modules, sorted(m for m in sys.modules if 'gxipy' in m)"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
