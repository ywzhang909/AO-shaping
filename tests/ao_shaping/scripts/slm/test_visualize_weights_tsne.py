"""Offline tests for the t-SNE weights-trajectory visualization script.

The script turns ``--debug-weights``' ``weights_trajectory.npz`` into a 2D t-SNE
scatter/trajectory PNG. These tests build small synthetic trajectories (no
hardware, no real run), exercise the ``zernike`` and ``all`` modes, and cover the
``D < 2`` failure path.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts" / "slm" / "visualize_weights_tsne.py"


def _load_script():
    """Load the visualization script by path (it lives under ``scripts/slm``)."""
    spec = importlib.util.spec_from_file_location("visualize_weights_tsne", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


visualize = _load_script()

#: Seven residual_noll modes, matching the real ``_flatten_parameters`` key set.
RESIDUAL_MODES = ("5", "6", "7", "8", "9", "10", "11")


def _write_trajectory(path: Path, n_iter: int, include_extra: bool = True) -> Path:
    """Write a synthetic ``weights_trajectory.npz`` with ``n_iter`` rounds."""
    rng = np.random.default_rng(1234)
    data: dict[str, np.ndarray] = {"iterations": np.arange(n_iter, dtype=float)}
    for mode in RESIDUAL_MODES:
        data[f"residual_noll_{mode}"] = rng.normal(loc=0.0, scale=0.01, size=n_iter)
    if include_extra:
        data["amplitude_center_0"] = np.linspace(960.0, 962.0, n_iter)
        data["amplitude_center_1"] = np.linspace(600.0, 601.0, n_iter)
        data["gain"] = np.linspace(1.0, 1.05, n_iter)
    np.savez(path, **data)
    return path


def test_zernike_mode_writes_png(tmp_path: Path) -> None:
    """``--mode zernike`` (default) produces a non-empty PNG over 6 rounds."""
    weights_dir = tmp_path / "debug_weights"
    weights_dir.mkdir()
    _write_trajectory(weights_dir / "weights_trajectory.npz", n_iter=6)

    out = tmp_path / "weights_tsne.png"
    visualize.main(["--input", str(weights_dir), "--out", str(out)])

    assert out.is_file()
    assert out.stat().st_size > 0


def test_all_mode_writes_png(tmp_path: Path) -> None:
    """``--mode all`` includes the non-zernike parameter columns and writes a PNG."""
    weights_dir = tmp_path / "debug_weights"
    weights_dir.mkdir()
    _write_trajectory(weights_dir / "weights_trajectory.npz", n_iter=6, include_extra=True)

    out = tmp_path / "weights_tsne_all.png"
    visualize.main(["--input", str(weights_dir), "--out", str(out), "--mode", "all"])

    assert out.is_file()
    assert out.stat().st_size > 0


def test_single_column_mode_errors_out(tmp_path: Path) -> None:
    """``D < 2`` (a single selected parameter column) makes the script exit."""
    weights_dir = tmp_path / "debug_weights"
    weights_dir.mkdir()
    np.savez(
        weights_dir / "weights_trajectory.npz",
        iterations=np.arange(5, dtype=float),
        residual_noll_5=np.zeros(5),
    )

    out = tmp_path / "weights_tsne.png"
    with pytest.raises(SystemExit):
        visualize.main(["--input", str(weights_dir), "--out", str(out)])

    assert not out.exists()


def test_missing_trajectory_file_errors_out(tmp_path: Path) -> None:
    """A missing ``weights_trajectory.npz`` makes the script exit cleanly."""
    weights_dir = tmp_path / "debug_weights"
    weights_dir.mkdir()

    out = tmp_path / "weights_tsne.png"
    with pytest.raises(SystemExit):
        visualize.main(["--input", str(weights_dir), "--out", str(out)])

    assert not out.exists()
