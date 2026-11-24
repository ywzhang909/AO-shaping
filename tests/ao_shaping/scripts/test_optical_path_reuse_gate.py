"""Offline contracts for the optional optical-path reuse gate.

The gate must never turn "an old diagnostic was valid" into "this path is
valid now". These tests pin the fail-closed behaviour, the current-frame
requirement (never compare a stored flat with itself), and the single-exposure
saturation guard that replaces -- but never skips -- Step 4.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "slm" / "optical_path_reuse_gate.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("optical_path_reuse_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


def _write_probe(
    directory: Path,
    *,
    state: str = "ok",
    created_utc: str = "2026-11-24T01:00:00+00:00",
    exposure_ms: float = 0.05,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    frames = np.zeros((2, 64, 64), dtype=np.uint16)
    frames[:, 20:24, 30:34] = 1800
    np.savez_compressed(directory / "flat.npz", frames=frames)
    doc = {
        "schema": "optical_path_probe/1",
        "created_utc": created_utc,
        "verdict": {"state": state},
        "config": {
            "bit_depth": 12,
            "defocus_exposure_ms": exposure_ms,
            "slm_number": 1,
            "cam_type": "daheng",
            "cam_id": 0,
        },
        "device": {
            "slm_serial": "SIM-SLM",
            "cam_exposure_range_ms": [0.02, 1000.0],
            "cam_bit_depth": 12,
        },
        "defocus_sweep": [
            {
                "tag": "flat_repeat",
                "npz": "flat.npz",
                "centroid_x": 32.0,
                "centroid_y": 22.0,
            }
        ],
    }
    (directory / gate.PROBE_FILE).write_text(
        json.dumps(doc, ensure_ascii=False), encoding="utf-8"
    )
    return directory / gate.PROBE_FILE


def _write_record(directory: Path) -> Path:
    doc = {
        "date": "2026-11-24",
        "devices": {
            "slm": {"drive_board_id": "BOARD-1", "two_pi_grayscale": 993}
        },
        "optical_path": {
            "config_id": "synthetic",
            "fresnel_compensation_m_inv": None,
            "probe_status": "ok",
        },
    }
    path = directory / gate.RECORD_FILE
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return path


def test_normalize_camera_format_reconciles_device_and_config() -> None:
    """gxipy returns 'Mono12'; the probe JSON only stores bit_depth=12."""
    assert gate.normalize_camera_format("Mono12", 12) == "MONO12"
    assert gate.normalize_camera_format(None, 12) == "MONO12"
    assert gate.normalize_camera_format("MONO8", 8) == "MONO8"
    assert gate.normalize_camera_format(None, None) is None
    assert gate.normalize_camera_format("weird-format", None) == "WEIRD-FORMAT"


def test_exposure_guard_uses_device_full_scale_not_frame_dtype() -> None:
    """A MONO12 frame in a uint16 container must be checked against 4095."""
    ok = gate.check_exposure_frame(np.full((8, 8), 2800.0), 4095.0)
    assert ok["passed"] is True
    assert ok["peak_limit_counts"] == pytest.approx(0.7 * 4095)

    too_bright = gate.check_exposure_frame(np.full((8, 8), 3000.0), 4095.0)
    assert too_bright["passed"] is False
    assert "峰值" in too_bright["reason"]

    saturated = gate.check_exposure_frame(np.full((8, 8), 3900.0), 4095.0)
    assert saturated["passed"] is False
    assert saturated["saturated_pixels"] == 64

    assert gate.check_exposure_frame(np.zeros((8,)), 4095.0)["passed"] is False
    assert gate.check_exposure_frame(np.zeros((8, 8)), 0.0)["passed"] is False


def test_load_flat_frame_accepts_single_and_stacked_frames(tmp_path: Path) -> None:
    single = tmp_path / "single.npz"
    np.savez_compressed(single, frame=np.arange(16, dtype=np.float64).reshape(4, 4))
    stacked = tmp_path / "stacked.npz"
    np.savez_compressed(
        stacked, frames=np.stack([np.zeros((4, 4)), np.full((4, 4), 10.0)])
    )
    assert np.array_equal(gate.load_flat_frame_any(single), np.arange(16).reshape(4, 4))
    assert np.all(gate.load_flat_frame_any(stacked) == 5.0)
    assert gate.load_flat_frame_any(tmp_path / "missing.npz") is None


def test_find_latest_ok_probe_filters_failed_and_orders_by_time(tmp_path: Path) -> None:
    root = tmp_path / "probe_root"
    older = _write_probe(root / "older", created_utc="2026-11-24T00:00:00+00:00")
    latest = _write_probe(root / "latest", created_utc="2026-11-24T03:00:00+00:00")
    _write_probe(root / "failed", state="low_snr", created_utc="2026-11-24T09:00:00+00:00")
    assert gate.find_latest_ok_probe(root) == latest
    assert gate.find_latest_ok_probe(root / "failed") is None

    # A probe without its standard record cannot be a reuse reference.
    with pytest.raises(ValueError, match="标准记录"):
        gate.load_reference(latest)
    _write_record(older.parent)
    reference = gate.load_reference(older)
    assert reference.probe.bit_depth == 12
    assert reference.probe.camera_format == "MONO12"
    assert reference.probe.flat_frame == "flat.npz"
    assert reference.exposure_ms == pytest.approx(0.05)
    assert reference.drive_board_id == "BOARD-1"
    assert reference.frame.shape == (64, 64)


def test_custom_saturation_fraction_can_be_binding(tmp_path: Path) -> None:
    """0.7x peak headroom alone cannot catch a 0.65x hot pixel unless the
    saturation fraction is lowered below the peak fraction."""
    frame = np.full((4, 4), 0.65 * 4095.0)
    assert gate.check_exposure_frame(frame, 4095.0)["passed"] is True
    strict = gate.check_exposure_frame(frame, 4095.0, saturation_fraction=0.6)
    assert strict["passed"] is False
    assert strict["saturated_pixels"] == 16
    assert gate.build_parser().parse_args([]).saturation_fraction == 0.95


def test_latest_probe_mixes_iso_and_mtime_without_string_bias(tmp_path: Path) -> None:
    """An old ISO timestamp must not outrank a newer file that lacks one."""
    root = tmp_path / "mixed"
    old_iso = _write_probe(root / "old_iso", created_utc="2020-01-01T00:00:00+00:00")
    mtime_only = root / "mtime_only"
    mtime_only.mkdir(parents=True)
    frames = np.zeros((2, 8, 8), dtype=np.uint16)
    np.savez_compressed(mtime_only / "flat.npz", frames=frames)
    doc = {
        "schema": "optical_path_probe/1",
        "verdict": {"state": "ok"},
        "config": {"bit_depth": 12, "defocus_exposure_ms": 0.05},
        "device": {"cam_exposure_range_ms": [0.02, 1000.0], "cam_bit_depth": 12},
        "defocus_sweep": [
            {"tag": "flat_repeat", "npz": "flat.npz", "centroid_x": 4.0, "centroid_y": 4.0}
        ],
    }
    (mtime_only / gate.PROBE_FILE).write_text(
        json.dumps(doc, ensure_ascii=False), encoding="utf-8"
    )
    assert gate.find_latest_ok_probe(root) == mtime_only / gate.PROBE_FILE
    assert old_iso.exists()


def test_no_reference_fails_closed_and_writes_result(tmp_path: Path) -> None:
    """Missing evidence never reaches the device; it writes reuse_gate=failed."""
    args = argparse.Namespace(
        out_root=str(tmp_path / "empty"),
        run_dir=str(tmp_path / "run"),
        slm_number=1,
        wavelength=1064,
        cam_type="daheng",
        cam_id=0,
        wait_until_idle=0.0,
        peak_fraction=0.7,
        saturation_fraction=0.95,
    )
    result = gate.run_reuse_gate(args)
    assert result["reuse_gate"] == "failed"
    assert any("verdict=ok" in reason for reason in result["reasons"])
    assert result["remedy"].endswith("Step 3–4")
    target = Path(args.run_dir) / gate.GATE_FILE
    payload: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
    assert payload["reuse_gate"] == "failed"
    assert payload["criteria"] == {}
    assert gate.build_parser().parse_args([]).peak_fraction == 0.7
