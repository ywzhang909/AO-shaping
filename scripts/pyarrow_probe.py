"""Pure pyarrow-probe helpers — used by tests and the ZernikeControl debug panel.

These helpers are deliberately kept out of ``pattern_controls`` module level:
the control calls ``probe_pyarrow()`` inline inside
``ZernikeControl.render()``, so a broken pyarrow never blocks the module
from being imported (and the rest of the page from rendering).

Tests monkeypatch the probe to simulate a broken pyarrow environment without
actually breaking the real one.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path


def pyarrow_version() -> str | None:
    """Version of the importable ``pyarrow``, or ``None`` when it is broken."""
    try:
        import pyarrow
    except ImportError:
        return None
    return str(getattr(pyarrow, "__version__", "未知"))


def pyarrow_pandas_compat_ok() -> bool:
    """Probe whether ``pyarrow.pandas_compat`` can be imported.

    ``st.data_editor`` calls ``pa.Table.from_pandas(data_df)`` at
    ``data_editor.py:1092``, which internally does
    ``from pyarrow.pandas_compat import dataframe_to_arrays``.  When that
    submodule is missing (e.g. a mismatched wheel), the call raises
    ``ModuleNotFoundError: No module named 'pyarrow.pandas_compat'`` — the
    same fatal abort as a plain ``pyarrow.lib`` failure.
    """
    try:
        import pyarrow.pandas_compat  # noqa: F401
    except ImportError:
        return False
    return True


def probe_pyarrow() -> tuple[bool, str]:
    """Probe whether ``st.data_editor`` can run — returns ``(available, reason)``.

    ``st.data_editor`` imports pyarrow lazily *inside* the widget call.  When
    pyarrow is missing or built for another Python minor version (e.g. a
    ``cp313`` wheel left in a ``cp314`` site-packages), that import raises
    ``ModuleNotFoundError`` from within the widget, which aborts the whole
    Streamlit script run — every element after the crashing widget disappears
    and the page looks like it "exited" by itself.  Probes the capability
    before any widget is called, so the control can fall back to plain,
    pyarrow-free widgets instead of taking the page down.

    The returned ``reason`` is the exact import failure (empty when available).
    """
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if not pyarrow_pandas_compat_ok():
        return False, "ModuleNotFoundError: No module named 'pyarrow.pandas_compat'"
    return True, ""


def pyarrow_diagnostics() -> list[tuple[str, str]]:
    """``(label, value)`` rows pinpointing *why* pyarrow is unusable.

    Separates "not installed" from "installed for another Python minor
    version": the latter leaves e.g. ``lib.cp313-win_amd64.pyd`` in a ``cp314``
    interpreter, whose import machinery only accepts ``.cp314-win_amd64.pyd``
    and therefore reports ``ModuleNotFoundError: No module named
    'pyarrow.lib'`` even though the package directory is present.
    """
    import importlib.machinery
    import importlib.util

    available, reason = probe_pyarrow()
    rows: list[tuple[str, str]] = [
        (
            "系数编辑器",
            "st.data_editor (系数表)" if available else "st.number_input (逐项回退)",
        ),
        ("pyarrow", pyarrow_version() or f"导入失败 — {reason}"),
        ("pyarrow.pandas_compat", "可用" if available else "不可用"),
        ("Python", f"{platform.python_version()} @ {sys.executable}"),
        ("Streamlit", _streamlit_version()),
    ]
    if available:
        return rows

    try:
        spec = importlib.util.find_spec("pyarrow")
    except (ImportError, ValueError):
        spec = None
    package_dir = Path(spec.origin).parent if spec is not None and spec.origin else None
    rows.append(("pyarrow 包目录", str(package_dir) if package_dir else "未找到"))

    try:
        suffixes = importlib.machinery.EXTENSION_SUFFIXES
        installed = [p.name for p in (package_dir or Path(".")).glob("*.pyd")]
        rows.append(("已安装 lib*.pyd", ", ".join(installed) or "无"))
        rows.append(("解释器接受的扩展后缀", ", ".join(suffixes)))
    except Exception:
        pass

    rows.append(
        ("修复命令", "pip install --force-reinstall --only-binary :all: pyarrow")
    )
    return rows


def _streamlit_version() -> str:
    """Streamlit version — imported lazily so this module is importable
    without streamlit (useful for unit tests that only probe pyarrow)."""
    try:
        import streamlit as st
    except ImportError:
        return "streamlit 未安装"
    return str(getattr(st, "__version__", "未知"))


__all__ = [
    "pyarrow_diagnostics",
    "pyarrow_pandas_compat_ok",
    "pyarrow_version",
    "probe_pyarrow",
]
