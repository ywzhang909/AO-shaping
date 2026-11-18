from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


_ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "issue237_registration_audit", _ROOT / "scripts/slm/issue237_registration_audit_20261118.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_fixed_support_ncc_excludes_masked_signal() -> None:
    rng = np.random.default_rng(237)
    image = rng.normal(0.0, 0.01, (80, 90))
    template = rng.normal(0.0, 1.0, (13, 17))
    mask = np.ones(image.shape, dtype=bool)
    image[31 : 44, 42 : 59] += template
    mask[31:44, 42:59] = False
    score, _ = _MODULE.fixed_support_ncc_map(image, template, mask)
    assert np.max(score) < 0.2


def test_fixed_support_ncc_recovers_known_peak() -> None:
    rng = np.random.default_rng(238)
    template = rng.normal(0.0, 1.0, (11, 15))
    image = rng.normal(0.0, 0.01, (70, 80))
    image[24 : 35, 37 : 52] += template
    mask = np.ones(image.shape, dtype=bool)
    score, _ = _MODULE.fixed_support_ncc_map(image, template, mask)
    peak = np.unravel_index(np.argmax(score), score.shape)
    assert peak == (24, 37)
    assert float(score[peak]) > 0.99


def test_concentration_drops_one_percent_of_template_not_detector() -> None:
    rng = np.random.default_rng(239)
    image = rng.normal(size=(100, 100))
    placed = np.zeros_like(image)
    placed[20:30, 30:40] = image[20:30, 30:40]
    result = _MODULE._concentration(image, placed, np.ones_like(image, dtype=bool))
    assert result["top_1pct_template_pixels"] == 1
    assert np.isfinite(result["ncc_without_top_1pct_contributors"])


def test_run_refuses_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        _MODULE.run(tmp_path / "missing-source", output)
