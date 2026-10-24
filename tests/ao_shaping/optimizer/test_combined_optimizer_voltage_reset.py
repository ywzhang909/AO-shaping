"""A failed DM voltage reset must be reported, never swallowed.

``optimize_pib`` zeroes the DM in a ``finally``. Raising there would mask whatever
exception ended the run, so the handler must not re-raise -- but swallowing it leaves
the mirror holding its last (possibly large) voltages, which is a physical state the
next person at the bench cannot observe.

There is no offline harness that runs ``optimize_pib`` (it opens a camera and a DM
through ``open_pib_bench``), so this is pinned as an AST guard on the module rather
than as a behavioural test. The guard asserts the guarded reset *exists* first, so it
cannot pass vacuously if the teardown is moved or deleted.
"""

from __future__ import annotations

import ast
from pathlib import Path

MODULE = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "ao_shaping"
    / "optimizer"
    / "combined_optimizer.py"
)


def _guarded_voltage_resets() -> list[tuple[ast.Try, ast.ExceptHandler]]:
    tree = ast.parse(MODULE.read_text(encoding="utf-8-sig"))
    pairs: list[tuple[ast.Try, ast.ExceptHandler]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        guarded = any(
            isinstance(call, ast.Call) and getattr(call.func, "attr", "") == "send_voltages"
            for call in ast.walk(node)
        )
        if not guarded:
            continue
        for handler in node.handlers:
            pairs.append((node, handler))
    return pairs


def test_the_module_still_guards_its_voltage_resets() -> None:
    """Non-vacuity: if this fails, the teardown reset no longer has a handler."""
    assert _guarded_voltage_resets(), (
        f"{MODULE.name}: no try/except around a send_voltages call -- "
        "the reset invariant below would be checking nothing"
    )


def test_every_guarded_voltage_reset_reports_its_failure() -> None:
    for node, handler in _guarded_voltage_resets():
        reported = [
            call
            for call in ast.walk(handler)
            if isinstance(call, ast.Call)
            and getattr(call.func, "attr", "") in {"error", "warning", "exception", "critical"}
            and "logger"
            in str(
                getattr(call.func.value, "id", "")
                or getattr(call.func.value, "attr", "")
            ).lower()
        ]
        assert reported, (
            f"{MODULE.name}:{handler.lineno}: a send_voltages failure (guarded at "
            f":{node.lineno}) is swallowed without a log -- the DM may keep its last "
            "voltages and nothing records it"
        )
