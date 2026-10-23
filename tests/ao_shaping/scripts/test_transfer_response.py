"""A repeated exposure is independent validation, never an extra fitting sample."""

import importlib.util
import json
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[3] / "scripts/slm/analyze_transfer_response.py"
)
spec = importlib.util.spec_from_file_location("analyze_transfer_response", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_corrupted_repeat_does_not_change_fitted_intercept(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    yy, xx = np.mgrid[:8, :8]
    xy = np.stack((xx, yy), axis=-1)
    np.savez(source / "square_probe.npz", camera_xy=xy, target=np.ones((8, 8)))
    slope, intercept = 3 + xx / 8, 1 + yy / 8
    exposures = (1.0, 0.25, 1.0, 0.5, 1.0, 0.125, 1.0)
    rows = []
    for index, exposure in enumerate(exposures):
        image = slope * exposure + intercept
        if index == 2:
            image = image + 2  # independent repeated reference is deliberately wrong
        file = f"capture_{index}.npz"
        np.savez(source / file, raw_frames=np.stack([image] * 3))
        rows.append(dict(phase="square", exposure_ms=exposure, index=index, file=file))
    (source / "manifest.json").write_text(json.dumps(rows))
    output = tmp_path / "analysis"
    module.analyze(source, output)
    with np.load(output / "square_arrays.npz") as data:
        np.testing.assert_allclose(data["intercept"], intercept, atol=1e-10)
        np.testing.assert_allclose(data["slope"], slope, atol=1e-10)
    result = json.loads((output / "result.json").read_text())["square"]
    assert result["validation_rmse_counts"] > 1
    assert 2 in result["validation_indices"]
