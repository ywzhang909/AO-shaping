"""Regression test for issue #143 problem 2 (sweep empty-result clobber guard).

``scripts/_common.refuse_empty_overwrite`` is the single source of truth for
"must a crashed/aborted sweep be allowed to overwrite a good prior output with
``{}``?". It is pure (one ``Path.stat``), so it is unit-tested directly rather
than by running the torch-heavy sweep. ``sweep_far_field_padding.py`` and any
future sweep write point must route through it, so an empty result can never
silently replace a good ``padding_sweep.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from scripts._common import refuse_empty_overwrite  # noqa: E402


def test_empty_result_over_existing_nonempty_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "padding_sweep.json"
    out.write_text('{"family": {"best_padding": 10}}', encoding="utf-8")
    # An empty sweep result must NOT clobber the good prior file.
    assert refuse_empty_overwrite(out, {}) is True
    assert refuse_empty_overwrite(out, None) is True
    assert refuse_empty_overwrite(out, []) is True


def test_empty_result_with_no_existing_file_is_allowed(tmp_path: Path) -> None:
    # Nothing to protect: a fresh (even empty) write is fine.
    out = tmp_path / "padding_sweep.json"
    assert refuse_empty_overwrite(out, {}) is False
    assert refuse_empty_overwrite(out, None) is False


def test_nonempty_result_is_always_allowed(tmp_path: Path) -> None:
    out = tmp_path / "padding_sweep.json"
    out.write_text('{"family": {"best_padding": 10}}', encoding="utf-8")
    # A real result overwrites freely (normal re-run of the sweep).
    assert refuse_empty_overwrite(out, {"family": {"best_padding": 10}}) is False
    assert refuse_empty_overwrite(out, [("a", 1)]) is False


def test_empty_result_over_an_already_empty_file_is_allowed(tmp_path: Path) -> None:
    out = tmp_path / "padding_sweep.json"
    out.write_text("", encoding="utf-8")  # 0-byte prior: nothing worth protecting
    assert refuse_empty_overwrite(out, {}) is False


def test_helper_is_torch_free_leaf() -> None:
    """The guard is a leaf in ``scripts._common``: it must not import torch.

    Sweeps are torch-heavy; the guard is extracted precisely so it can be
    imported (and tested) without loading torch.
    """
    import inspect
    import importlib

    mod = importlib.import_module("scripts._common")
    src = inspect.getsource(mod.refuse_empty_overwrite)
    assert "torch" not in src
