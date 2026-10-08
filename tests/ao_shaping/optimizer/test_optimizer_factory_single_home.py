"""R-13 guard: the optimizer factory and the optimizer map must have exactly one home.

Issue #46 filed five copies of ``_create_optimizer``. Measuring them (rather than
trusting the count) showed **four** distinct bodies, of which only one pair was
byte-identical, and that two of them already carried the loud-failure implementation
while three silently swallowed unsupported kwargs. It also turned up three files
carrying their own ``OPTIMIZER_MAP`` even though ``optimizer/constants.py`` exists
specifically to centralise it.

Both duplications are now gone. This test fails if either comes back:

1. any ``def _create_optimizer`` anywhere in the package;
2. any module-level ``OPTIMIZER_MAP`` assignment outside ``optimizer/constants.py``;
3. ``create_optimizer`` not defined in ``optimizer/constants.py``.

It also pins the behaviour that motivated the whole issue, because a future
"simplification" of the filter would silently reintroduce the two documented bugs:
a dropped kwarg must be reported, and a callee declaring ``**kwargs`` must receive
what it was given.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
OPTIMIZER_ROOT = REPO_ROOT / "src" / "ao_shaping" / "optimizer"
CANONICAL = OPTIMIZER_ROOT / "constants.py"


def _module_assign_names(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
        return set()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_no_private_create_optimizer_anywhere() -> None:
    offenders: list[str] = []
    for path in sorted(OPTIMIZER_ROOT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        if any(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == "_create_optimizer"
            for n in ast.walk(tree)
        ):
            offenders.append(path.relative_to(REPO_ROOT).as_posix())
    assert not offenders, (
        f"_create_optimizer is defined again in {offenders}. The canonical factory is "
        f"ao_shaping.optimizer.constants.create_optimizer -- import that instead."
    )


def test_optimizer_map_has_exactly_one_home() -> None:
    offenders: list[str] = []
    for path in sorted(OPTIMIZER_ROOT.rglob("*.py")):
        if path == CANONICAL:
            continue
        if "OPTIMIZER_MAP" in _module_assign_names(path):
            offenders.append(path.relative_to(REPO_ROOT).as_posix())
    assert not offenders, (
        f"OPTIMIZER_MAP is redefined in {offenders}. It belongs to "
        f"ao_shaping/optimizer/constants.py -- import it from there."
    )


def test_canonical_factory_lives_in_constants() -> None:
    tree = ast.parse(CANONICAL.read_text(encoding="utf-8"))
    assert any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "create_optimizer"
        for n in tree.body
    ), "create_optimizer is missing from optimizer/constants.py"


# --- the behaviour that motivated the issue -----------------------------------------
def test_dropped_kwarg_is_reported_not_swallowed(monkeypatch) -> None:
    """`--optimizer sgd` with `momentum` looked accepted and silently did nothing."""
    from ao_shaping.algorithm.gradient.adam import SGD
    from ao_shaping.optimizer import constants

    # loguru does not feed pytest's caplog, so record the call directly
    recorded: list[tuple] = []
    monkeypatch.setattr(
        constants, "logger", type("L", (), {"warning": staticmethod(lambda *a, **k: recorded.append(a))})()
    )

    opt = constants.create_optimizer("sgd", dim=4, lr=0.1, momentum=0.9)
    assert isinstance(opt, SGD)
    assert any("momentum" in " ".join(str(x) for x in call) for call in recorded), (
        f"a kwarg the target cannot accept must be reported, not silently dropped; got {recorded}"
    )


def test_var_keyword_callee_receives_the_kwargs() -> None:
    """A callee declaring **kwargs must get it -- the `**config.kwargs` escape hatch."""
    from ao_shaping.optimizer.constants import OPTIMIZER_MAP, create_optimizer

    seen: dict[str, object] = {}

    class _Sink:
        def __init__(self, dim: int, lr: float = 0.0, **kwargs: object) -> None:
            seen.update(kwargs)
            self.dim, self.lr = dim, lr

    original = dict(OPTIMIZER_MAP)
    try:
        OPTIMIZER_MAP["sink"] = _Sink
        create_optimizer("sink", dim=4, lr=0.1, freeform=7)
    finally:
        OPTIMIZER_MAP.clear()
        OPTIMIZER_MAP.update(original)

    assert seen == {"freeform": 7}, (
        "a callee declaring **kwargs must receive what it was given; "
        f"got {seen}"
    )


def test_unknown_optimizer_name_falls_back_to_adamod() -> None:
    from ao_shaping.algorithm.gradient.adam import AdaMOD
    from ao_shaping.optimizer.constants import create_optimizer

    assert isinstance(create_optimizer("no-such-optimizer", dim=3, lr=0.2), AdaMOD)


@pytest.mark.parametrize("name", ["adam", "adamw", "adamod", "sgd", "muno", "munow"])
def test_every_registered_optimizer_is_constructible(name: str) -> None:
    from ao_shaping.optimizer.constants import create_optimizer

    assert create_optimizer(name, dim=4, lr=0.1) is not None
