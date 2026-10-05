"""R-36: the dataclass-click mechanism must stay a zero-``ao_shaping`` leaf.

The mechanism was extracted out of ``runners/runner_common.py`` into
:mod:`ao_shaping.utils.cli.params` so that ``tools/slm/params.py`` (R-37) can
declare its 22 probes' options with the same ``Annotated[..., option(...)]``
convention. That only works while the mechanism sits **below** both of them.

The tempting regression is a single convenience import -- ``from
ao_shaping.runners.runner_common import DM_TYPES`` to build a ``click.Choice``,
or a lazy helper import inside a function. Either one silently welds ``tools/``
to ``runners/``, so the checks below walk the AST instead of grepping: a
deferred import inside a function body passes ``grep " ao_shaping"`` and fails
here.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import pytest

from ao_shaping.utils.cli import params as cli_params

_MODULE = Path(cli_params.__file__).resolve()
_REPO_ROOT = _MODULE.parents[3]
_LEAF = _MODULE.relative_to(_REPO_ROOT).as_posix()

#: Names that must be defined exactly once in the whole repo. A second
#: definition is a copy that will drift from the first -- the failure mode that
#: produced the two diverging ``_fmt`` copies R-25 had to reconcile.
@dataclass
class _Child:
    """Nested parameter group used by the shadowing guard below."""

    epochs: Annotated[int, cli_params.option("--epochs")] = 1


@dataclass
class _ParentWithShadow:
    """Declares ``--epochs`` twice -- once flat, once via the nested group."""

    # The group field goes FIRST and stays undefaulted: dataclasses forbid a
    # non-default field after a defaulted one, and `X | None` would make
    # `Annotated`'s first argument a UnionType instead of the dataclass itself.
    child: Annotated[_Child, cli_params.ClickGroup()]
    epochs: Annotated[int, cli_params.option("--epochs")] = 1


_MECHANISM_NAMES = (
    "_DelayedCall",
    "_DelayedFunction",
    "_strip_optional",
    "_patch_names",
    "_patch_click_types",
    "_patch_defaults",
    "_copy_delayed_call",
    "_is_click_group",
    "_collect_click_annotations",
    "with_params",
    "ClickGroup",
)

#: Third-party roots the leaf is allowed to depend on.
_ALLOWED_ROOTS = frozenset({"click"})


def _stdlib_roots() -> frozenset[str]:
    """Top-level module names that ship with CPython."""
    roots = set(sys.stdlib_module_names)
    # `__future__` is a compiler directive, and `sysconfig`/`site`-installed
    # helpers behave like stdlib for our purposes.
    roots |= {"__future__"}
    return frozenset(roots)


_STDLIB = _stdlib_roots()


def _imported_roots(path: Path) -> set[str]:
    """Every ``import`` / ``from ... import`` root in ``path``, at any depth."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{path.name}: relative import {node.module!r}"
            roots.add((node.module or "").split(".")[0])
    return roots


# ---------------------------------------------------------------------------
# 1. the leaf invariant: nothing from ao_shaping, anywhere
# ---------------------------------------------------------------------------


def test_leaf_imports_nothing_from_ao_shaping() -> None:
    offenders = sorted(_imported_roots(_MODULE) & {"ao_shaping"})
    assert not offenders, f"{_LEAF} must not import {offenders}"


def test_leaf_imports_only_stdlib_and_click() -> None:
    """A non-stdlib, non-click dependency needs the same scrutiny as ao_shaping."""
    roots = _imported_roots(_MODULE)
    unknown = sorted(r for r in roots if r not in _STDLIB and r not in _ALLOWED_ROOTS)
    assert not unknown, (
        f"{_LEAF} may only import stdlib + {sorted(_ALLOWED_ROOTS)}; got {unknown}. "
        "A new dependency needs a reason recorded in the module docstring."
    )


def test_no_deferred_ao_shaping_import_can_hide_in_a_function() -> None:
    """AST, not grep: an import inside a function body must still be caught."""
    tree = ast.parse(_MODULE.read_text(encoding="utf-8"), filename=str(_MODULE))
    nested = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and any("ao_shaping" in a.name for a in getattr(node, "names", []) or ())
        or (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0] == "ao_shaping"
        )
    ]
    assert not nested, f"deferred ao_shaping import at lines {[n.lineno for n in nested]}"


# ---------------------------------------------------------------------------
# 2. the mechanism lives HERE, exactly once
# ---------------------------------------------------------------------------


def test_mechanism_is_defined_in_the_leaf() -> None:
    defined = {n.name for n in ast.parse(_MODULE.read_text(encoding="utf-8")).body
               if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    defined |= {t.id for n in ast.parse(_MODULE.read_text(encoding="utf-8")).body
                if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    missing = [n for n in _MECHANISM_NAMES if n not in defined]
    assert not missing, f"missing from {_LEAF}: {missing}"


def _repo_sources() -> list[Path]:
    return [
        p
        for base in ("src/ao_shaping", "tests/ao_shaping")
        for p in ( _REPO_ROOT / base).rglob("*.py")
    ]


def test_mechanism_is_not_duplicated_anywhere_else() -> None:
    """The R-25 lesson applied to the mechanism itself.

    ``option`` and ``with_params`` are *imported* by many modules -- that is not
    duplication. Duplication is *defining* them again, which is how two ``_fmt``
    copies drifted into different behaviour in the first place.
    """
    offenders: dict[str, list[str]] = {}
    for path in _repo_sources():
        if path.resolve() == _MODULE:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in tree.body:
            name = None
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                name = node.name
            elif isinstance(node, ast.Assign):
                name = next(
                    (t.id for t in node.targets if isinstance(t, ast.Name)), None
                )
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                name = node.target.id
            if name in _MECHANISM_NAMES:
                offenders.setdefault(name, []).append(
                    path.relative_to(_REPO_ROOT).as_posix()
                )
    assert not offenders, (
        f"mechanism redefined outside {_LEAF}: {offenders}. "
        "Import from ao_shaping.utils.cli.params instead of re-implementing."
    )


def test_runner_common_shares_the_leaf_objects_not_copies() -> None:
    """``runner_common`` must import the mechanism, not re-own it.

    Identity (``is``) rather than equality: a copy that happens to compare equal
    today is still free to drift tomorrow.
    """
    from ao_shaping.runners import runner_common

    assert runner_common.with_params is cli_params.with_params
    assert runner_common.ClickGroup is cli_params.ClickGroup
    assert runner_common.option is cli_params.option


def test_explicit_default_in_option_still_raises() -> None:
    """The documented invariant moved with the code -- re-pin it at the new home.

    Two sources of truth for a default is how a flag ends up documented at one
    value while applying another, so ``default=`` inside ``option(...)`` must
    keep raising ``TypeError``.

    NOTE: ``Annotated`` / ``dataclass`` are module-level imports on purpose --
    ``get_type_hints`` evaluates annotations in the *defining module's*
    globals, so a function-local import makes every field raise ``NameError``.
    """
    with pytest.raises(TypeError, match="default must live on the dataclass field"):

        @dataclass
        class Bad:
            x: Annotated[int, cli_params.option("--x", default=3)] = 1

        cli_params.with_params(Bad, kw_name="p")(lambda p: None)


def test_duplicate_click_parameter_name_is_rejected() -> None:
    """A nested ClickGroup must not shadow an existing option name."""
    with pytest.raises(TypeError, match="declared twice"):
        cli_params._collect_click_annotations(_ParentWithShadow)