"""Device policy for the transfer torch stages — no GPU required.

Two things must not regress:

1. the policy itself (:func:`resolve_device`): ``auto`` follows the machine and
   falls back to CPU, an explicit choice is honoured, a typo is rejected;
2. the *call sites* actually use it. A literal ``device="cpu"`` (or ``"cuda"``)
   inside a stage is the failure mode this guard exists for — it silently
   overrides the operator's ``--device`` and is invisible to any functional test
   that fakes the devices anyway.

The functional half of (2) — that ``validate.run`` hands the resolved value to
the forward model *and* the optimizer — lives in
``test_transfer_workflow.py::test_validate_stages_use_the_resolved_device``,
where the fake harness already exists. Here the guard is at source level, so it
holds for every construction site in the three modules, GPU or not.
"""

from __future__ import annotations

import ast
import builtins
from pathlib import Path

import pytest

from ao_shaping.tools.slm.transfer import device as device_module
from ao_shaping.tools.slm.transfer.device import DEVICE_CHOICES, resolve_device

TRANSFER = Path(device_module.__file__).parent
#: The stages that must take their device from the resolver.
STAGE_MODULES = ("fit.py", "validate.py", "workflow.py")
#: torch constructs whose device decides where the work happens.
STAGE_CONSTRUCTS = frozenset({"FrozenDefocusForward", "DifferentiableUniformityOptimizer",
                              "PhysicalForwardFit"})


@pytest.fixture
def cuda(monkeypatch):
    """Pretend the machine has (or has not) a usable CUDA device."""

    def _set(available: bool) -> None:
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: bool(available))

    return _set


class TestResolveDevice:
    def test_auto_follows_the_machine(self, cuda) -> None:
        cuda(True)
        assert resolve_device("auto") == "cuda"
        cuda(False)
        assert resolve_device("auto") == "cpu"

    def test_none_is_auto(self, cuda) -> None:
        cuda(True)
        assert resolve_device(None) == "cuda"
        cuda(False)
        assert resolve_device(None) == "cpu"

    def test_an_explicit_choice_is_returned_unchanged(self, cuda) -> None:
        # Pinning CPU must win over a visible GPU: that is how a run is compared
        # against a historical CPU artefact.
        cuda(True)
        assert resolve_device("cpu") == "cpu"
        cuda(False)
        assert resolve_device("cpu") == "cpu"
        # And an explicit cuda is not silently downgraded — torch must fail loudly.
        assert resolve_device("cuda") == "cuda"

    def test_case_and_whitespace_are_normalised(self, cuda) -> None:
        cuda(False)
        assert resolve_device(" CPU ") == "cpu"
        assert resolve_device("Auto") == "cpu"

    @pytest.mark.parametrize("requested", ["gpu", "cuda:0", "", "tpu"])
    def test_an_unknown_device_is_rejected(self, requested: str) -> None:
        """A typo must not silently become CPU: that would look like a fallback."""
        with pytest.raises(ValueError, match="unknown device"):
            resolve_device(requested)

    def test_without_torch_auto_is_cpu(self, monkeypatch) -> None:
        """torch is an optional extra for this package; the probe must not raise."""
        real_import = builtins.__import__

        def _no_torch(name: str, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("No module named 'torch'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_torch)
        assert device_module.cuda_available() is False
        assert resolve_device("auto") == "cpu"
        assert resolve_device("cpu") == "cpu"

    def test_the_choices_are_the_documented_three(self) -> None:
        assert DEVICE_CHOICES == ("auto", "cpu", "cuda")


def _called_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


class TestCallSitesGoThroughTheResolver:
    """Source-level guard: no stage may pin a device literal."""

    def test_every_stage_construction_takes_a_resolved_device(self) -> None:
        offenders: list[str] = []
        seen_constructs = 0
        for name in STAGE_MODULES:
            tree = ast.parse((TRANSFER / name).read_text(encoding="utf-8-sig"), filename=name)
            resolver_calls = [n for n in ast.walk(tree)
                              if isinstance(n, ast.Call) and _called_name(n) == "resolve_device"]
            if not resolver_calls:
                offenders.append(f"{name}: never calls resolve_device()")
            for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
                if _called_name(call) not in STAGE_CONSTRUCTS:
                    continue
                seen_constructs += 1
                value = next((kw.value for kw in call.keywords if kw.arg == "device"), None)
                if value is None:
                    offenders.append(f"{name}:{call.lineno} {_called_name(call)} passes no device=")
                elif isinstance(value, ast.Constant):
                    offenders.append(
                        f"{name}:{call.lineno} {_called_name(call)} pins device={value.value!r}")
                elif not (isinstance(value, ast.Name) and value.id == "device"):
                    offenders.append(
                        f"{name}:{call.lineno} {_called_name(call)} passes a non-resolved device")
        assert seen_constructs >= 6, "the guard found fewer stage constructions than expected"
        assert not offenders, (
            "transfer stage(s) do not take their device from resolve_device(); a literal "
            "overrides the operator's --device and cannot be tested on a GPU-less machine:\n  "
            + "\n  ".join(offenders))

    def test_the_guard_catches_a_pinned_literal(self) -> None:
        """Guards the guard, on the exact pattern it is meant to stop."""
        sample = ast.parse('FrozenDefocusForward(params, xy, device="cpu")\n'
                           'FrozenDefocusForward(params, xy, device=device)\n')
        pinned = [call for call in ast.walk(sample) if isinstance(call, ast.Call)
                  and isinstance(next((kw.value for kw in call.keywords if kw.arg == "device"), None),
                                 ast.Constant)]
        assert len(pinned) == 1
