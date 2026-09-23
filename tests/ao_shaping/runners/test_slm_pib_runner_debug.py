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


def _make_rms_pib_recorder_with_target_box(n: int = 5) -> Recorder:
    """rms_pib records carrying the target-box geometry fields + rms_pib terms.

    ``_ref_center`` / ``_target_shape`` / ``_target_size`` / ``_target_aspect``
    are the fields the optimizer adds via ``_target_box_meta``; the runner
    reads them from ``history[0]`` to overlay the ROI on the spot panels.
    """
    rec = Recorder(mark="rms_pib", mode="max")
    for i in range(n):
        img = np.full((24, 24), 10 * (i + 1), dtype=np.uint8)
        rec.append(
            {
                "J": 0.5 + 0.03 * i,
                "rms_pib": 0.4 + 0.05 * i,
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
                "w_pib": 0.4,
                "w_rms": 0.3,
                "w_ee": 0.3,
                "pib_term": 0.3,
                "rms_term": 0.5,
                "ee_term": 0.7,
                # target-box geometry (window-local centre, same frame as _img)
                "_ref_center": (12.0, 12.0),
                "_target_shape": "rectangle",
                "_target_size": 10.0,
                "_target_aspect": 1.0,
                f"best_rms_pib": 0.4 + 0.05 * i,
            }
        )
    return rec


def test_debug_artifacts_rms_pib_with_target_box_overlay(tmp_path):
    """rms_pib runs draw the objective curve + overlay the target ROI.

    The data-mode key list must recognise ``rms_pib`` (it did not before the
    target-box work, so the curve panel was empty) and the target-box fields
    must flow ``history[0] -> _save_debug_artifacts -> PNG`` without error.
    """
    rec = _make_rms_pib_recorder_with_target_box()

    png = _save_debug_artifacts(
        rec,
        ObjectiveParams(name="rms_pib"),
        SlmParams(),
        HeuristicParams(algorithm="spgd"),
        str(tmp_path),
    )

    assert png.suffix == ".png"
    assert png.exists() and png.stat().st_size > 0
    assert png.with_suffix(".pkl").exists()
    assert png.with_suffix(".json").exists()
    # the h5 sidecar must carry the new scalar/array/str columns
    import h5py

    with h5py.File(png.with_suffix(".h5"), "r") as f:
        assert "rms_pib" in f["scalars"]
        assert "w_ee" in f["scalars"]
        assert "ee_term" in f["scalars"]
        assert "_target_shape" in f["scalars"]  # str col -> scalars
        assert "_target_size" in f["scalars"]  # float col -> scalars
        # window-local centre is a tuple -> per-epoch array col
        epoch_id = f"{rec.history[0]['_id']:04d}"
        assert "_ref_center" in f["epochs"][epoch_id]


def test_save_data_mode_debug_artifacts_best_epoch_and_overlay(tmp_path):
    """Data-mode PNG picks the best-epoch frame/coeffs and overlays the ROI.

    Regression: the old panel showed the *last* ``_img``/``_c`` and only
    recognised ``pib``/``radiu``/``avg_radiu`` objective columns; the new code
    must use the row with the optimal ``best_rms_pib`` and draw a contour for
    the supplied ``target_box``.
    """
    from ao_shaping.runners.runner_common import _save_data_mode_debug_artifacts

    rec = _make_rms_pib_recorder_with_target_box()
    data = {int(r["_epoch"]): r for r in rec.history}
    target_box = {
        "center": (12.0, 12.0),
        "shape": "rectangle",
        "size": 10.0,
        "aspect_ratio": 1.0,
    }

    png = _save_data_mode_debug_artifacts(
        data=data,
        png_path=tmp_path / "fig.png",
        pkl_path=tmp_path / "fig.pkl",
        json_path=tmp_path / "fig.json",
        title="slm-pib rms_pib search",
        json_payload={"algorithm": "spgd"},
        target_box=target_box,
    )

    assert png == tmp_path / "fig.png"
    assert png.exists() and png.stat().st_size > 0
    assert (tmp_path / "fig.pkl").exists()
    assert (tmp_path / "fig.json").exists()


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


def test_debug_artifacts_with_guard_penalised_rows(tmp_path):
    """Guard-penalised rows (J < -100) must not crash the debug figure.

    The energy-loss guard records ``J-1e3`` for abandoned evaluations.  The
    data-mode plot filters those out of *both* the x and y arrays — a
    dimension mismatch (tk_xs longer than the cleaned y) is a regression.
    """
    rec = Recorder(mark="pib", mode="max")
    for i in range(10):
        img = np.full((16, 16), 10 * (i + 1), dtype=np.uint8)
        j_val = -999.0 if i in (3, 7) else 0.1 * i  # penalised at epochs 3, 7
        rec.append(
            {
                "J": j_val,
                "pib": 0.1 * (i + 1) if i not in (3, 7) else -0.5,
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
                "best_pib": 0.1 * i,
            }
        )

    png = _save_debug_artifacts(
        rec,
        ObjectiveParams(name="pib"),
        SlmParams(),
        HeuristicParams(algorithm="spgd"),
        str(tmp_path),
    )

    assert png.suffix == ".png"
    assert png.exists() and png.stat().st_size > 0


def test_cli_exposes_debug_and_heuristic_options():
    result = CliRunner().invoke(run, ["heuristic", "--help"])

    assert result.exit_code == 0, result.output
    for opt in ("--debug", "--algorithm", "--pop_size", "--cam_type"):
        assert opt in result.output
    for algo in ("spgd", "ga", "pso", "sa", "hc", "rs", "cem", "de"):
        assert algo in result.output


def test_algorithm_choices_match_optimizer():
    from ao_shaping.optimizer.wfless.slm_zernike_pib import ALGORITHM_CHOICES

    assert set(ALGORITHM_CHOICES) == {
        "spgd",
        "ga",
        "pso",
        "sa",
        "hc",
        "rs",
        "cem",
        "de",
    }


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

    _resolve_auto_camera(
        camera, find_exposure=True, auto_target_peak=160.0, auto_n_frames=5
    )

    assert camera.exposure_time_ms == 12.5
    assert camera.center == (34, 56)
    assert probe.closed is True
    assert calls["auto"] == (True, True, 160.0, 5)


def test_resolve_auto_camera_keeps_exposure_when_find_exposure_false(monkeypatch):
    probe, calls = _patch_probe(monkeypatch, result=(None, (34, 56)))
    camera = CameraParams(center="auto", exposure_time_ms=80.0)

    _resolve_auto_camera(
        camera, find_exposure=False, auto_target_peak=160.0, auto_n_frames=5
    )

    assert camera.exposure_time_ms == 80.0
    assert camera.center == (34, 56)
    assert probe.closed is True
    assert calls["auto"] == (False, True, 160.0, 5)


def test_resolve_auto_camera_falls_back_on_probe_error(monkeypatch):
    probe, calls = _patch_probe(monkeypatch, exc=ValueError("all-dark frame"))
    camera = CameraParams(center="auto", exposure_time_ms=80.0)

    _resolve_auto_camera(
        camera, find_exposure=True, auto_target_peak=160.0, auto_n_frames=5
    )

    assert camera.center == "shape"
    assert camera.exposure_time_ms == 80.0
    assert probe.closed is True


def test_cli_auto_exposure_help_has_full_sentence():
    result = CliRunner().invoke(run, ["heuristic", "--help"])

    assert result.exit_code == 0, result.output
    assert "probe pass" in result.output
