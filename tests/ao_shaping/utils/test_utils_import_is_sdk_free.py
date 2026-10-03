"""R-20 — ``import ao_shaping.utils`` must not require optional third-party SDKs.

The failure this pins is an **import-time** one, which is the worst kind: a machine
without ``aotools`` installed cannot even ``import ao_shaping.utils``, so every
consumer (tests, scripts, an IDE's indexer, ``python -c``) dies before reaching
any code it cares about.

Two independent causes, both fixed:

1. ``utils/wavefront/pattern_helper.py`` did a bare module-scope
   ``from aotools.turbulence.infinitephasescreen import PhaseScreenKolmogorov``,
   and ``utils/__init__.py`` eagerly imported ``pattern_helper``.
2. ``utils/__init__.py`` re-exported ~87 names eagerly, so the whole subtree was
   loaded whether or not the caller needed any of it.

After the fix the failure mode must **move from import time to call time**: only
actually constructing a Kolmogorov phase screen needs ``aotools``.

The heavy lifting is done in a subprocess with ``aotools`` blocked from
``sys.modules``, because by the time this test runs ``aotools`` is already
imported in-process.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

BLOCK_AOTOOLS = """
import sys

class _Block:
    def find_module(self, name, path=None):
        return self if name == "aotools" or name.startswith("aotools.") else None
    def load_module(self, name):
        raise ImportError("No module named 'aotools' (blocked by R-20 test)")
    def find_spec(self, name, path=None, target=None):
        if name == "aotools" or name.startswith("aotools."):
            raise ImportError("No module named 'aotools' (blocked by R-20 test)")
        return None

sys.meta_path.insert(0, _Block())
for mod in [m for m in sys.modules if m == "aotools" or m.startswith("aotools.")]:
    del sys.modules[mod]
"""

PROBE_IMPORT = """
import ao_shaping.utils
print("IMPORT_OK")
"""

PROBE_NAMES = """
import ao_shaping.utils as u
names = {n for n in dir(u) if not n.startswith("_")}
print("NAMES", len(names))
import ao_shaping.utils.wavefront.pattern_helper as ph
print("PH", ph.PatternHelper.__name__)
"""

PROBE_USE = """
import numpy as np
from ao_shaping.utils.wavefront.pattern_helper import PatternHelper
h = PatternHelper((64, 64))
print("CONSTRUCTED")
try:
    h.init_turbulence_screen(r0=0.1, L0=10.0, pixel_scale=1e-3)
except ImportError as exc:
    print("CALL_IMPORTERROR", "aotools" in str(exc))
else:
    print("CALL_OK")
"""


def _run(body: str) -> subprocess.CompletedProcess[str]:
    script = BLOCK_AOTOOLS + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )


def test_import_ao_shaping_utils_survives_without_aotools() -> None:
    """The headline guarantee: no optional SDK is needed to import the package."""
    result = _run(PROBE_IMPORT)
    assert "IMPORT_OK" in result.stdout, (
        "import ao_shaping.utils must not need aotools\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_the_public_surface_is_intact_without_aotools() -> None:
    result = _run(PROBE_NAMES)
    assert "PH PatternHelper" in result.stdout, result.stdout + result.stderr
    count = next(
        (int(l.split()[1]) for l in result.stdout.splitlines() if l.startswith("NAMES ")),
        0,
    )
    assert count >= 87, f"public surface shrank to {count} names"


def test_the_failure_moved_to_call_time_not_import_time() -> None:
    """``aotools`` must only be needed when a phase screen is actually built."""
    result = _run(PROBE_USE)
    assert "CONSTRUCTED" in result.stdout, (
        "constructing PatternHelper must not need aotools\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert (
        "CALL_IMPORTERROR True" in result.stdout
        or "CALL_OK" in result.stdout
    ), result.stdout + result.stderr


def test_pattern_helper_has_no_module_scope_aotools_import() -> None:
    """Static guard: the bare import must not creep back to module scope."""
    import ast
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "ao_shaping"
        / "utils"
        / "wavefront"
        / "pattern_helper.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:  # module scope only
        if isinstance(node, ast.Import):
            assert all(not a.name.startswith("aotools") for a in node.names), (
                f"module-scope `import aotools` at line {node.lineno}"
            )
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("aotools"), (
                f"module-scope `from aotools... import` at line {node.lineno}"
            )


def test_utils_package_uses_pep562_lazy_reexports() -> None:
    """Static guard: ``utils/__init__`` must not eagerly import its own submodules.

    ``loguru`` / stdlib imports are fine (they are cheap and dependency-free); what
    must not appear is ``from ao_shaping.utils...`` at module scope, because that
    is what drags the whole subtree -- and its optional SDKs -- into every
    ``import ao_shaping.utils``.
    """
    import ast
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[3] / "src" / "ao_shaping" / "utils" / "__init__.py"
    )
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    eager: list[str] = []
    for node in tree.body:  # module scope only; a TYPE_CHECKING block is fine
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "ao_shaping.utils"
        ):
            eager.append(f"line {node.lineno}: from {node.module}")
        elif isinstance(node, ast.Import) and any(
            a.name.startswith("ao_shaping") for a in node.names
        ):
            eager.append(f"line {node.lineno}: import ao_shaping...")
    assert not eager, (
        "utils/__init__.py must not import submodules at module scope; "
        f"found {eager}"
    )
    assert "__getattr__" in source, "utils/__init__.py needs a PEP 562 __getattr__"
    assert "_LAZY_EXPORTS" in source, "utils/__init__.py needs the lazy name -> module map"


def test_utils_all_is_never_shrunk() -> None:
    """``__all__`` must keep every name the eager version exported (R-20: delay, never delete)."""
    import ao_shaping.utils as u

    assert len(u.__all__) == 85
    assert "PatternHelper" in u.__all__
    assert "calculate_sharpness" in u.__all__
    assert "parse_tuple" in u.__all__
    # The historical alias must survive too.
    assert "calc_n_zernike_terms" in u.__all__
    assert "calc_n_zernike_terms_zern" in u.__all__
    # Every declared name must actually resolve.
    unresolvable = [n for n in u.__all__ if not hasattr(u, n)]
    assert not unresolvable, unresolvable


def test_unknown_name_still_raises_attribute_error() -> None:
    import ao_shaping.utils as u

    with pytest.raises(AttributeError):
        u.definitely_not_a_real_name  # noqa: B018