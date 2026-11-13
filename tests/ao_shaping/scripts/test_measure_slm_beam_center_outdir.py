"""Offline guard for #235's per-frame output layout.

2026-11-11: a full knife-edge scan died on the very first ``np.save`` because the
``frames/`` subdirectory was never created — the camera work was already spent.
These tests pin the layout without opening any device.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "slm" / "measure_slm_beam_center.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "measure_slm_beam_center_under_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_module()


def test_prepare_out_dir_creates_frames_subdir(mod, tmp_path: Path) -> None:
    out = mod._prepare_out_dir(tmp_path / "run")
    assert out == tmp_path / "run"
    assert out.is_dir()
    # The exact path shape used by `_run` when saving each pattern.
    target = out / "frames" / "x_00.npy"
    assert target.parent.is_dir()


def test_prepare_out_dir_creates_missing_parents(mod, tmp_path: Path) -> None:
    out = mod._prepare_out_dir(tmp_path / "a" / "b" / "run")
    assert (out / "frames").is_dir()


def test_prepare_out_dir_refuses_existing_run(mod, tmp_path: Path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    with pytest.raises(FileExistsError):
        mod._prepare_out_dir(out)
