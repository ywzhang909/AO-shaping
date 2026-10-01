"""``import ao_shaping`` must not depend on the current working directory.

Found 2026-10-01 while making the drivers lazy. ``NLight.py`` loads the actuator
adjacency matrix in its **class body**::

    Units_Adj_Mat = _load_adj_txt()      # NLight.py:63 — runs at import time

and ``_load_adj_txt`` reads a **hardcoded relative** path::

    np.loadtxt("data/dm_adj.txt")       # NLight.py:48

``drivers/__init__.py`` imported ``NLight`` eagerly, so merely importing the
package performed disk I/O against a CWD-relative path. Any consumer running
from another directory — a cron job, a service, an installed console script, a
test with ``cwd=tmp_path`` — died with::

    FileNotFoundError: data/dm_adj.txt not found.

This also duplicated the loader: ``SimulateDM._load_adj_txt`` reads the same path
but *does* fall back to a synthetic grid, so the two copies disagreed on
behaviour. These tests pin the path-independence and the single implementation.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[4]


def _run(snippet: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(snippet)],
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env={"PYTHONPATH": f"{_ROOT / 'src'}:{_ROOT / 'libs'}", "HOME": "/tmp"},
        timeout=180,
    )


class TestImportIsCwdIndependent:
    @pytest.mark.parametrize("name", ["tmp", "root"])
    def test_import_works_from_any_directory(self, name):
        cwd = Path("/tmp") if name == "tmp" else _ROOT
        result = _run(
            """
            import ao_shaping
            from ao_shaping.drivers.dm.NLight import NLight
            print("OK", NLight.__name__)
            """,
            cwd,
        )
        assert result.returncode == 0, (
            f"import failed with cwd={cwd}:\n{result.stderr[-2000:]}"
        )
        assert "OK NLight" in result.stdout

    def test_importing_drivers_does_not_read_data_files(self):
        """The package import must not touch ``data/`` at all."""
        result = _run(
            """
            import builtins
            _real = builtins.open
            opened = []

            def _spy(file, *a, **k):
                opened.append(str(file))
                return _real(file, *a, **k)

            builtins.open = _spy
            import ao_shaping  # noqa: F401
            builtins.open = _real
            bad = [p for p in opened if "dm_adj" in p]
            assert not bad, f"import read adjacency data: {bad}"
            print("OK")
            """,
            _ROOT,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "OK" in result.stdout

    def test_nlight_adjacency_is_still_available(self):
        """Laziness must not remove the public matrix the safety checks use."""
        result = _run(
            """
            from ao_shaping.drivers.dm.NLight import NLight
            m = NLight.Units_Adj_Mat
            print("OK", m.shape, m.sum())
            """,
            Path("/tmp"),
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "OK (64, 64)" in result.stdout

    def test_neighbour_lookup_still_works(self):
        """``default_dm_unit_mask``-style neighbour queries must still function."""
        result = _run(
            """
            import numpy as np
            from ao_shaping.drivers.dm.NLight import NLight
            mat = NLight.Units_Adj_Mat
            nbrs = np.where(mat[1, :] == 1)[0]
            print("OK", len(nbrs) > 0)
            """,
            Path("/tmp"),
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "OK True" in result.stdout


class TestSingleAdjacencyImplementation:
    def test_shared_loader_exists(self):
        from ao_shaping.drivers.dm import _adjacency

        assert hasattr(_adjacency, "load_adjacency")

    def test_no_duplicate_loadtxt_of_the_adjacency_file(self):
        """Only the shared module may actually *read* ``dm_adj.txt``.

        A prose mention in a docstring is fine; a second ``loadtxt`` call is the
        duplication this test exists to prevent.
        """
        offenders = []
        for path in sorted((_ROOT / "src/ao_shaping/drivers").rglob("*.py")):
            if path.name == "_adjacency.py":
                continue
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                stripped = line.strip()
                if stripped.startswith(("#", '"', "'")) or "*" in stripped:
                    continue
                if "dm_adj" in line and ("loadtxt" in line or "open(" in line):
                    offenders.append(f"{path.relative_to(_ROOT)}:{lineno}")
        assert not offenders, f"dm_adj.txt read outside the shared loader: {offenders}"

    def test_loader_is_cached_and_shaped(self):
        from ao_shaping.drivers.dm._adjacency import load_adjacency

        first = load_adjacency()
        assert first.shape == (64, 64)
        assert load_adjacency() is first, "the matrix must be cached, not re-read"

    def test_loader_survives_a_missing_file(self):
        """A missing asset must degrade, not explode at import."""
        from ao_shaping.drivers.dm import _adjacency

        original = _adjacency.ADJACENCY_PATH
        try:
            _adjacency.ADJACENCY_PATH = Path("/nonexistent/dm_adj.txt")
            _adjacency._cache_clear()
            mat = _adjacency.load_adjacency()
            assert mat.shape == (64, 64)
        finally:
            _adjacency.ADJACENCY_PATH = original
            _adjacency._cache_clear()
