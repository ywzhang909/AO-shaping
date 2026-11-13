"""Offline tests for the ``--debug-dir`` frame dump of the geometry calibration.

The calibration itself needs SLM + camera, so these tests pin only the
device-free contract: the three raw ``capture()`` frames must land exactly as
captured, the JSON metadata must be complete after every single frame (an
early ``return`` or a failing ``measure_orders`` must not lose data), and no
run must ever reuse or overwrite an existing directory.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "slm" / "calib_geometry_from_orders.py"

_RUN_DIR_NAME = re.compile(r"^\d{8}_\d{6}(?:_\d{3})?$")


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "calib_geometry_from_orders_under_test", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


calib = _load_script()


def _frames() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20261111)
    return {
        "flat": rng.normal(100.0, 3.0, size=(6, 8)).astype(np.float64),
        "x": rng.normal(50.0, 2.0, size=(6, 8)).astype(np.float64),
        "y": rng.normal(40.0, 2.0, size=(6, 8)).astype(np.float64),
    }


def _meta(run_dir: Path) -> dict:
    return json.loads((run_dir / calib.DEBUG_META_NAME).read_text(encoding="utf-8"))


def test_sink_writes_each_frame_byte_identical_with_metadata(tmp_path):
    run_dir = calib._reserve_debug_run_dir(tmp_path / "debug")
    sink = calib._DebugFrameSink(
        run_dir,
        {
            "period_px": 32,
            "predicted_k": 7546.4,
            "predicted_sep": 7546.4 / 32,
            "exposure_ms": 0.4,
        },
    )

    frames = _frames()
    for axis, frame in frames.items():
        sink.save(axis, frame)

    for axis, frame in frames.items():
        saved = np.load(run_dir / f"{axis}.npy")
        assert saved.dtype == frame.dtype
        assert np.array_equal(saved, frame)

    meta = _meta(run_dir)
    assert meta["period_px"] == 32
    assert meta["predicted_k"] == pytest.approx(7546.4)
    assert meta["predicted_sep"] == pytest.approx(7546.4 / 32)
    assert meta["exposure_ms"] == pytest.approx(0.4)
    assert [item["axis"] for item in meta["frames"]] == ["flat", "x", "y"]
    for item, frame in zip(meta["frames"], frames.values(), strict=True):
        assert item["file"] == f"{item['axis']}.npy"
        assert item["shape"] == list(frame.shape)
        assert item["dtype"] == str(frame.dtype)


def test_metadata_is_complete_after_the_flat_frame_alone(tmp_path):
    """A later early ``return`` (no signal / saturation) keeps flat + meta."""
    sink = calib._open_debug_sink(
        tmp_path, period_px=16, predicted_k=640.0, exposure_ms=1.5
    )
    assert _meta(sink.run_dir)["frames"] == []

    flat = _frames()["flat"]
    sink.save("flat", flat)

    meta = _meta(sink.run_dir)
    assert meta["period_px"] == 16
    assert meta["predicted_sep"] == pytest.approx(640.0 / 16)
    assert [item["axis"] for item in meta["frames"]] == ["flat"]
    assert (sink.run_dir / "flat.npy").exists()
    assert not (sink.run_dir / "x.npy").exists()


def test_metadata_dtype_follows_the_raw_frame(tmp_path):
    sink = calib._open_debug_sink(
        tmp_path, period_px=32, predicted_k=7546.4, exposure_ms=0.4
    )
    sink.save("x", np.zeros((3, 5), dtype=np.float32))

    saved = np.load(sink.run_dir / "x.npy")
    assert saved.dtype == np.float32
    assert _meta(sink.run_dir)["frames"][0]["dtype"] == "float32"


def test_each_calibration_gets_a_fresh_run_directory(tmp_path):
    root = tmp_path / "debug"
    first = calib._reserve_debug_run_dir(root).name
    second = calib._reserve_debug_run_dir(root).name
    third = calib._reserve_debug_run_dir(root).name

    assert len({first, second, third}) == 3
    assert {first, second, third} <= {p.name for p in root.iterdir() if p.is_dir()}
    # Pre-seeded directories from an earlier run are untouched.
    for name in (first, second, third):
        assert _RUN_DIR_NAME.match(name), name


def test_existing_run_directory_is_never_reused_or_overwritten(tmp_path):
    root = tmp_path / "debug"
    existing = root / "20260101_000000"
    existing.mkdir(parents=True)
    sentinel = existing / "flat.npy"
    sentinel.write_bytes(b"earlier-run")

    fresh = calib._reserve_debug_run_dir(root)

    assert fresh != existing
    assert fresh.is_dir()
    assert list(fresh.iterdir()) == []
    assert sentinel.read_bytes() == b"earlier-run"


def test_saving_the_same_axis_twice_is_refused(tmp_path):
    sink = calib._open_debug_sink(
        tmp_path, period_px=32, predicted_k=7546.4, exposure_ms=0.4
    )
    flat = _frames()["flat"]
    path = sink.save("flat", flat)

    with pytest.raises(FileExistsError):
        sink.save("flat", flat + 1.0)

    assert np.array_equal(np.load(path), flat)
    assert [item["axis"] for item in _meta(sink.run_dir)["frames"]] == ["flat"]
