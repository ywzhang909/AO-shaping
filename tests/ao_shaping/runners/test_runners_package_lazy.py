"""R-27: ``ao_shaping.runners`` must actually be lazy.

The package docstring states the reason for the PEP 562 ``__getattr__``:

    "This lets every runner be executed directly with ``python -m
    ao_shaping.runners.<name>`` without the ``RuntimeWarning: '<name>' found in
    sys.modules after import of package`` that eager imports trigger."

It did not do that. Fifteen eager ``from ... import run as ...`` statements sat at
the top of ``__init__.py``, so every name was already in ``globals()`` by the time
``__getattr__`` could ever be consulted -- the lazy path was unreachable dead
code, and the warning it promised to avoid was live:

    RuntimeWarning: 'ao_shaping.runners.slm.shaping_runner' found in sys.modules
    after import of package 'ao_shaping.runners', but prior to execution of
    'ao_shaping.runners.slm.shaping_runner'; this may result in unpredictable
    behaviour

The subprocess assertions are the real contract. In-process ``sys.modules``
checks cannot see this, because pytest has already imported the runners by the
time any test body runs.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

from ao_shaping.runners import __all__ as RUNNER_EXPORTS
from ao_shaping.runners import _IMPORT_ATTRS, _LAZY_RUNNERS

#: Every ``python -m`` entry point the package documents. Two of them live in
#: sub-packages, which is where the warning was easiest to trigger from.
_M_ENTRY_POINTS = [
    "ao_shaping.runners.slm.shaping_runner",
    "ao_shaping.runners.nlight_dm.wf_runner",
    "ao_shaping.runners.slm.rms_zernike_runner",
]


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


# ---------------------------------------------------------------------------
# the warning the docstring promises to avoid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module", _M_ENTRY_POINTS)
def test_python_dash_m_emits_no_runtime_warning(module: str) -> None:
    """The docstring's own claim, checked against a real interpreter.

    ``--help`` is enough: it exercises the module body and exits before any
    hardware is touched, and these are click commands.
    """
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    noise = [
        line
        for line in result.stderr.splitlines()
        if "RuntimeWarning" in line or "found in sys.modules" in line
    ]
    assert not noise, (
        f"{module} is already in sys.modules before runpy executes it, which is "
        f"the exact warning the package docstring claims to prevent:\n"
        + "\n".join(noise)
    )


def test_importing_the_package_loads_no_runner_module() -> None:
    code = (
        "import sys, ao_shaping.runners\n"
        "leaked = sorted(m for m in sys.modules "
        "if m.startswith('ao_shaping.runners.') and 'runner' in m.rsplit('.', 1)[-1])\n"
        "print('\\n'.join(leaked))\n"
    )
    result = _run(code)
    leaked = [line for line in result.stdout.split() if line.strip()]
    assert not leaked, f"importing the package eagerly pulled in {leaked}"


# ---------------------------------------------------------------------------
# the lazy path has to be reachable, and correct
# ---------------------------------------------------------------------------


def test_every_exported_name_resolves_to_a_callable() -> None:
    for name in RUNNER_EXPORTS:
        value = getattr(sys.modules["ao_shaping.runners"], name)
        assert callable(value), f"{name} resolved to {type(value).__name__}"


def test_getattr_caches_so_the_second_access_is_the_same_object() -> None:
    """PEP 562 only stays lazy if the resolved value is written back.

    Without the ``globals()[name] = value`` line every access would re-enter
    ``__getattr__`` and re-import, so this pins the caching, not just the lookup.
    """
    import importlib

    module = importlib.import_module("ao_shaping.runners")
    importlib.reload(module)
    first = module.wf_run
    assert module.wf_run is first


def test_unknown_attribute_raises_attribute_error() -> None:
    import importlib

    module = importlib.import_module("ao_shaping.runners")
    with pytest.raises(AttributeError, match="has no attribute"):
        module.definitely_not_a_runner


def test_dir_lists_the_exports() -> None:
    import importlib

    module = importlib.import_module("ao_shaping.runners")
    listed = dir(module)
    assert set(RUNNER_EXPORTS).issubset(listed)


# ---------------------------------------------------------------------------
# the guard that would have caught the original bug
# ---------------------------------------------------------------------------


def test_all_and_the_lazy_map_cannot_drift() -> None:
    """``__all__`` and ``_LAZY_RUNNERS`` must name the same set.

    This is the root cause, stated as an invariant. While the eager imports
    covered every name, a name added to ``__all__`` without a mapping still
    resolved -- so the map was never exercised and never had to be correct. Once
    the eager imports are gone, a missing mapping becomes an AttributeError at
    call time instead, in whatever runner the user happened to invoke.
    """
    assert set(RUNNER_EXPORTS) == set(_LAZY_RUNNERS), (
        f"only in __all__: {sorted(set(RUNNER_EXPORTS) - set(_LAZY_RUNNERS))}; "
        f"only in the map: {sorted(set(_LAZY_RUNNERS) - set(RUNNER_EXPORTS))}"
    )


def test_every_name_main_imports_is_resolvable() -> None:
    """Self-consistency was not enough; the real consumer is ``main.py``.

    ``slm_gsnet_run`` was in neither ``__all__`` nor ``_LAZY_RUNNERS`` and still
    worked, because an eager import happened to cover it. So the invariant above
    passed while ``main.py`` was one line away from breaking -- and removing the
    eager imports is what finally surfaced it.

    This test parses main.py's actual import list, so the two lists cannot drift
    apart again without a failure.
    """
    import ast
    import re
    from pathlib import Path

    import ao_shaping

    main_py = Path(ao_shaping.__file__).resolve().parent / "main.py"
    required: set[str] = set()
    for node in ast.parse(main_py.read_text(encoding="utf-8")).body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "ao_shaping.runners"
        ):
            required |= {a.name for a in node.names}

    assert required, "main.py no longer imports from ao_shaping.runners"
    missing = sorted(required - set(RUNNER_EXPORTS))
    assert not missing, (
        f"main.py imports {missing} from ao_shaping.runners but the package does "
        "not export them; they used to resolve only via the eager imports"
    )

    unresolved = sorted(required - set(_LAZY_RUNNERS))
    assert not unresolved, f"main.py needs {unresolved} but _LAZY_RUNNERS lacks them"

    # And they must actually resolve, not merely be declared.
    package = importlib.import_module("ao_shaping.runners")
    for name in sorted(required):
        assert callable(getattr(package, name)), f"{name} does not resolve"


def test_closures_that_name_a_second_attribute_have_an_entry() -> None:
    """``zernike_closed_loop_run`` and ``zernike_matrix_run`` share a module.

    They must therefore each carry an explicit attribute; a missing one would
    silently fall back to ``run`` and hand back the wrong command.
    """
    for name in RUNNER_EXPORTS:
        if list(_LAZY_RUNNERS.values()).count(_LAZY_RUNNERS[name]) > 1:
            assert name in _IMPORT_ATTRS, (
                f"{name} shares a module with a sibling export but has no "
                "_IMPORT_ATTRS entry, so it would silently resolve to 'run'"
            )


def test_no_module_is_imported_at_scope_time() -> None:
    """Belt and braces: nothing at module scope may import a runner.

    Asserted over the AST rather than ``sys.modules`` so it holds no matter what
    else has already been imported during the test session.
    """
    import ast
    from pathlib import Path

    import ao_shaping.runners as pkg

    source = Path(pkg.__file__).read_text(encoding="utf-8")
    offenders = []
    for node in ast.parse(source).body:  # module scope only
        targets: list[str] = []
        if isinstance(node, ast.ImportFrom):
            targets = [node.module or ""]
        elif isinstance(node, ast.Import):
            targets = [a.name for a in node.names]
        for target in targets:
            if target.startswith("ao_shaping.runners") and target != "ao_shaping.runners":
                offenders.append((node.lineno, target))
    assert not offenders, f"module-scope runner imports: {offenders}"