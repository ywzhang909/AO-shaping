"""Tests for ``slm_pib_runner``: debug artifacts + CLI options (no hardware).

The debug figure is exercised through ``_save_debug_artifacts`` with a synthetic
:class:`Recorder`, so the whole image-output path is covered offline.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner

from ao_shaping.runners.slm_pib_runner import (
    CameraParams,
    HeuristicParams,
    ObjectiveParams,
    SlmParams,
    _resolve_auto_camera,
    _save_debug_artifacts,
    run,
)
from ao_shaping.utils.io.file import Recorder


def _make_recorder(objective: str, mode: str, n: int = 3) -> Recorder:
    rec = Recorder(mark=objective, mode=mode)
    for i in range(n):
        img = np.full((16, 16), 10 * (i + 1), dtype=np.uint8)
        rec.append(
            {
                "J": 0.1 * i,
                objective: 0.1 * (i + 1),
                "_p%": 0.5,
                "_max_r": 5.0,
                "_c": np.linspace(-1.0, 1.0, 6),
                "_img": img,
                "_diff": 0.0,
                "lr": 0.1,
                "r": 5.0,
                "delta": 0.1,
                "_epoch": i,
                "exp_t": 10.0,
                "max_brt": float(img.max()),
                "_grad": np.zeros(6),
            }
        )
    return rec


@pytest.mark.parametrize(
    "objective,mode", [("pib", "max"), ("radiu", "min"), ("avg_radiu", "max")]
)
def test_debug_artifacts_written_for_every_objective(tmp_path, objective, mode):
    rec = _make_recorder(objective, mode)

    png = _save_debug_artifacts(
        rec,
        ObjectiveParams(name=objective),
        SlmParams(),
        ObjectiveParams(name=objective),
        str(tmp_path),
    )

    assert png.suffix == ".png"
    assert png.exists() and png.stat().st_size > 0
    assert png.with_suffix(".pkl").exists()
    assert png.with_suffix(".json").exists()


def test_debug_artifact_json_round_trips_config(tmp_path):
    rec = _make_recorder("pib", "max")
    png = _save_debug_artifacts(
        rec,
        ObjectiveParams(name="pib"),
        SlmParams(),
        HeuristicParams(algorithm="ga"),
        str(tmp_path),
    )

    payload = json.loads(png.with_suffix(".json").read_text(encoding="utf8"))
    assert payload == {"algorithm": "ga"}


def test_cli_exposes_debug_and_heuristic_options():
    result = CliRunner().invoke(run, ["heuristic", "--help"])

    assert result.exit_code == 0, result.output
    for opt in ("--debug", "--algorithm", "--pop_size", "--cam_type"):
        assert opt in result.output
    for algo in ("spgd", "ga", "pso", "sa", "hc", "rs", "cem", "de"):
        assert algo in result.output


def test_algorithm_choices_match_optimizer():
    from ao_shaping.optimizer.wfless.slm_zernike_pib import ALGORITHM_CHOICES

    assert set(ALGORITHM_CHOICES) == {"spgd", "ga", "pso", "sa", "hc", "rs", "cem", "de"}


def test_main_group_exposes_debug_flag():
    from ao_shaping.main import cli

    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0, result.output
    assert "--debug" in result.output


# --- HDF5 debug export + auto-camera probe (2026-09) ------------------------


def _make_recorder_with_phase(objective: str, mode: str, n: int = 3) -> Recorder:
    """Like ``_make_recorder`` but each row also carries a uint16 2D ``_phase``."""
    rec = _make_recorder(objective, mode, n)
    for i, row in enumerate(rec.history):
        row["_phase"] = np.full((16, 16), 100 * (i + 1), dtype=np.uint16)
    return rec


class _StubProbeCamera:
    """Minimal probe-camera stub; records whether ``close`` was called."""

    def __init__(self) -> None:
        self.closed = False
        self.exposure_time_ms = 80.0

    def get_numpy_image(self) -> np.ndarray:
        return np.full((16, 16), 100, dtype=np.uint8)

    def reset_exposure_time(self, ms: float) -> None:
        self.exposure_time_ms = float(ms)

    def close(self) -> None:
        self.closed = True


def _patch_probe(
    monkeypatch: pytest.MonkeyPatch,
    result: tuple[float | None, tuple[int, int] | None] = (12.5, (34, 56)),
    exc: Exception | None = None,
) -> tuple[_StubProbeCamera, dict[str, Any]]:
    """Monkeypatch the lazy-imported probe helpers; return (probe, calls).

    ``_resolve_auto_camera`` re-imports ``open_camera`` /
    ``auto_find_exposure_and_center`` from ``hardware_utils`` at call time, so
    patching the module attributes intercepts the probe.
    """
    probe = _StubProbeCamera()
    calls: dict[str, Any] = {}

    def fake_open_camera(camera_type: str, cam_id: int, exposure_ms: float):
        calls["open"] = (camera_type, cam_id, exposure_ms)
        return probe

    def fake_auto_find(
        camera,
        *,
        find_exposure: bool,
        find_center: bool,
        target_peak: float,
        n_frames: int,
    ):
        calls["auto"] = (find_exposure, find_center, target_peak, n_frames)
        if exc is not None:
            raise exc
        return result

    monkeypatch.setattr(
        "ao_shaping.utils.image.hardware_utils.open_camera", fake_open_camera
    )
    monkeypatch.setattr(
        "ao_shaping.utils.image.hardware_utils.auto_find_exposure_and_center",
        fake_auto_find,
    )
    return probe, calls


def test_debug_artifacts_write_hdf5(tmp_path):
    import h5py

    rec = _make_recorder("pib", "max")
    png = _save_debug_artifacts(
        rec,
        ObjectiveParams(name="pib"),
        SlmParams(),
        HeuristicParams(algorithm="ga"),
        str(tmp_path),
    )

    h5 = png.with_suffix(".h5")
    assert h5.exists() and h5.stat().st_size > 0
    assert png.with_suffix(".pkl").exists()
    assert png.with_suffix(".json").exists()

    with h5py.File(h5, "r") as f:
        assert f["metadata"].attrs["objective"] == "pib"
        assert f["metadata"].attrs["algorithm"] == "ga"
        assert len(f["scalars"]["J"][:]) == 3


def test_debug_artifacts_hdf5_exports_phase(tmp_path):
    import h5py

    rec = _make_recorder_with_phase("pib", "max")
    png = _save_debug_artifacts(
        rec,
        ObjectiveParams(name="pib"),
        SlmParams(),
        HeuristicParams(algorithm="ga"),
        str(tmp_path),
    )

    h5 = png.with_suffix(".h5")
    assert h5.exists() and h5.stat().st_size > 0

    with h5py.File(h5, "r") as f:
        assert f["metadata"].attrs["objective"] == "pib"
        epoch_id = f"{rec.history[0]['_id']:04d}"
        assert "_phase" in f["epochs"][epoch_id]
        np.testing.assert_array_equal(
            f["epochs"][epoch_id]["_phase"][:],
            np.full((16, 16), 100, dtype=np.uint16),
        )


def test_resolve_auto_camera_sets_exposure_and_center(monkeypatch):
    probe, calls = _patch_probe(monkeypatch, result=(12.5, (34, 56)))
    camera = CameraParams(center="auto", exposure_time_ms=80.0)

    _resolve_auto_camera(camera, find_exposure=True, auto_target_peak=160.0, auto_n_frames=5)

    assert camera.exposure_time_ms == 12.5
    assert camera.center == (34, 56)
    assert probe.closed is True
    assert calls["auto"] == (True, True, 160.0, 5)


def test_resolve_auto_camera_keeps_exposure_when_find_exposure_false(monkeypatch):
    probe, calls = _patch_probe(monkeypatch, result=(None, (34, 56)))
    camera = CameraParams(center="auto", exposure_time_ms=80.0)

    _resolve_auto_camera(camera, find_exposure=False, auto_target_peak=160.0, auto_n_frames=5)

    assert camera.exposure_time_ms == 80.0
    assert camera.center == (34, 56)
    assert probe.closed is True
    assert calls["auto"] == (False, True, 160.0, 5)


def test_resolve_auto_camera_falls_back_on_probe_error(monkeypatch):
    probe, calls = _patch_probe(monkeypatch, exc=ValueError("all-dark frame"))
    camera = CameraParams(center="auto", exposure_time_ms=80.0)

    _resolve_auto_camera(camera, find_exposure=True, auto_target_peak=160.0, auto_n_frames=5)

    assert camera.center == "shape"
    assert camera.exposure_time_ms == 80.0
    assert probe.closed is True


def test_cli_auto_exposure_help_has_full_sentence():
    result = CliRunner().invoke(run, ["heuristic", "--help"])

    assert result.exit_code == 0, result.output
    assert "probe pass" in result.output
