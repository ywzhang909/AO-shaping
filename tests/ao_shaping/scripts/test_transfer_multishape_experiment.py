"""Checks for exact phase delivery and independent phase-family splits."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/slm/transfer_multishape_experiment.py"
spec = importlib.util.spec_from_file_location("transfer_multishape_experiment", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_panel_preserves_raw_phase_at_every_control_pixel():
    grid = np.random.default_rng(7).normal(0, 12, (128, 128))
    panel = module.to_panel(grid)
    assert panel.shape == (1200, 1920)
    np.testing.assert_array_equal(panel[::15, ::15], grid[:80])
    np.testing.assert_array_equal(panel[14::15, 14::15], grid[:80])


def test_non_tiling_grid_is_rejected():
    with pytest.raises(ValueError):
        module.to_panel(np.zeros((127, 127)))


def test_train_and_validation_cover_all_families_with_distinct_phases():
    probes = list(module.probes(128, 12, 102319))
    train = [(label, grid) for label, split, grid in probes if split == "train"]
    validation = [(label, grid) for label, split, grid in probes if split == "validation"]
    for group in (train, validation):
        assert {label.split("_", 2)[2] for label, _ in group} == {"zernike", "smooth", "bandlimited"}
    assert all(not np.array_equal(a, b) for _, a in train for _, b in validation)


def test_dominant_component_locator_uses_both_camera_coordinates():
    yy, xx = np.mgrid[:200, :300]
    frame = 0.5 + 100 * np.exp(-((xx - 184.3) ** 2 + (yy - 82.7) ** 2) / 40)
    x, y = module.locate(frame)
    assert x == pytest.approx(184.3, abs=.02)
    assert y == pytest.approx(82.7, abs=.02)
