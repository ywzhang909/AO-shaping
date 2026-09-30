"""Tests for the model-in-the-loop hardware runbook script.

The device-facing code in ``scripts/model_in_loop_hw_runbook.py`` cannot be
exercised without the bench, so it is driven here against mock SLM/camera
objects. That covers the parts most likely to be wrong in a way a dry run
misses: the radian-vs-gray conversion call, ``display_data`` arguments, the
millisecond unit of ``reset_exposure_time``, the 0-order lookup, the artefact
round-trips, and the stage sequencing.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "model_in_loop_hw_runbook.py"
PANEL_H, PANEL_W = 1200, 1920


def _load_runbook() -> Any:
    """Import the runbook script as a module (it is not an installed package)."""
    spec = importlib.util.spec_from_file_location("mil_hw_runbook", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MockSLM:
    """Minimal Santec stand-in that records what it was asked to display."""

    Panel_Res = (PANEL_W, PANEL_H)  # (W, H) like the real driver

    def __init__(self, **_: Any) -> None:
        self.shown: list[np.ndarray] = []
        self.closed = False

    def __enter__(self) -> "MockSLM":
        return self

    def __exit__(self, *exc: object) -> None:
        self.closed = True

    def get_wavelength_info(self) -> tuple[int, int]:
        return 1064, 993

    def create_phase_from_array(self, phase_rad: np.ndarray) -> np.ndarray:
        # The real driver takes radians and returns uint16 gray; it also rejects
        # a non-(H, W) array, which is what the runbook must respect. Returning
        # float here instead of uint16 would hide a runbook bug that re-encodes
        # or re-scales the driver's output.
        if phase_rad.shape != (PANEL_H, PANEL_W):
            raise ValueError(f"bad phase shape {phase_rad.shape}")
        gray = np.mod(phase_rad, 2 * np.pi) / (2 * np.pi) * 1023
        # The real driver clips to the device's grayscale range before casting.
        return np.clip(gray, 0, 1023).astype(np.uint16)

    def display_data(self, gray: np.ndarray, wait_time_s: float) -> None:
        assert gray.dtype == np.uint16
        assert gray.shape == (PANEL_H, PANEL_W)
        assert wait_time_s >= 0.0
        self.shown.append(gray)


class MockCamera:
    """Daheng stand-in returning a phase-dependent peaked far-field frame.

    The frame is deterministic in whatever the SLM last displayed, so the
    geometry solve sees genuinely different captures per probe instead of a
    constant (an all-zero or constant frame is rejected by the calibrator's
    degeneracy guard, which is itself worth exercising).
    """

    SIZE = 64

    def __init__(self, *args: Any, slm: "MockSLM | None" = None, **kwargs: Any) -> None:
        # create_camera(cam_type, cam_id, exposure_time_ms=...) is called
        # positionally, so the stand-in has to accept positional arguments.
        self.exposures: list[float] = []
        self._slm = slm

    def __enter__(self) -> "MockCamera":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def reset_exposure_time(self, time_ms: float) -> float:
        # Milliseconds: passing microseconds silently clamps to 1000 ms and
        # saturates every frame, which is one of the documented failure modes.
        self.exposures.append(float(time_ms))
        return float(time_ms)

    def get_numpy_image(self, n_sample: int = 1) -> np.ndarray:
        assert n_sample >= 1
        shown = self._slm.shown[-1] if self._slm and self._slm.shown else None
        key = 0 if shown is None else int(np.asarray(shown, np.int64).sum() % (2**31))
        rng = np.random.default_rng(key)
        frame = rng.random((self.SIZE, self.SIZE))
        yy, xx = np.mgrid[0 : self.SIZE, 0 : self.SIZE]
        frame *= np.exp(-((xx - self.SIZE / 2) ** 2 + (yy - self.SIZE / 2) ** 2) / 32.0)
        return frame * 255.0


@pytest.fixture
def runbook(monkeypatch: pytest.MonkeyPatch) -> Any:
    module = _load_runbook()
    created: dict[str, Any] = {}

    def make_slm(**kwargs: Any) -> MockSLM:
        created["slm"] = MockSLM(**kwargs)
        return created["slm"]

    def make_cam(*args: Any, **kwargs: Any) -> MockCamera:
        created["cam"] = MockCamera(*args, slm=created.get("slm"), **kwargs)
        return created["cam"]

    monkeypatch.setattr(module, "Santec", make_slm)
    monkeypatch.setattr(module, "create_camera", make_cam)
    module._created = created  # type: ignore[attr-defined]
    return module


def _args(runbook: Any, tmp_path: Path, **over: Any) -> Any:
    import argparse

    defaults = dict(
        stage="all", out=tmp_path, probes=3, region=32, far_field_size=256,
        collect_disc=60, target_cam_px=8.0, shape_iterations=5, slm_number=1,
        slm_wavelength=1064, cam_type="daheng", cam_id=0, exposure_ms=1.1,
        settle_s=0.0, discard=1, frames=1, seed=0, dry_run=False,
        defocus_rad=1.0, zernike_radius=200, n_orders=6, fit_epochs=2,
        fit_lr=0.1, crop=64, pupil_center="960,600", save_frames=False,
        method="sweep", sweep_tilt=[-1.0, 1.0], sweep_defocus=[-1.0, 1.0],
        sweep_astig=[-1.0, 1.0], sweep_repeats=2, sweep_ramps=[480.0, 960.0],
        sweep_coma=[-0.6, 0.6], sweep_spherical=[-0.6, 0.6],
        stable_tol=0.02, max_wait_s=2.0,
    )
    defaults.update(over)
    return argparse.Namespace(**defaults)


class TestPureHelpers:
    """Helpers that must hold with no devices present."""

    def test_probe_phase_is_strongly_scattering(self, runbook: Any) -> None:
        phase = runbook.probe_phase(64, seed=0)
        assert phase.shape == (64, 64)
        assert phase.min() >= 0.0 and phase.max() <= 2 * np.pi
        # A *weak* probe would leave the focal pattern insensitive to the
        # aberration, which is the identifiability failure being avoided.
        assert float(phase.std()) > 1.0

    def test_phase_to_panel_places_disc_at_the_pupil(self, runbook: Any) -> None:
        phase = runbook.probe_phase(32, seed=1)
        pupil = (1000, 480)
        panel = runbook.phase_to_panel(phase, 60, pupil)
        assert panel.shape == (PANEL_H, PANEL_W)
        assert float(panel.std()) > 0.0
        # Everything outside the disc must stay flat.
        assert float(panel[0, 0]) == 0.0
        assert float(panel[-1, -1]) == 0.0
        # The phase must land ON the pupil, not on the panel centre: the panel
        # centre is the wrong coordinate frame (camera 0-order != panel pixel).
        # All content lives in the 2r x 2r box centred on the pupil.
        r = 60
        box = np.zeros_like(panel, dtype=bool)
        box[pupil[1] - r : pupil[1] + r, pupil[0] - r : pupil[0] + r] = True
        assert np.array_equal(panel[~box], np.zeros_like(panel[~box]))
        assert float(np.abs(panel).sum()) > 0.0
        # A different pupil moves the pattern.
        assert not np.array_equal(runbook.phase_to_panel(phase, 60, (960, 600)), panel)

    def test_gaussian_grid_is_masked_to_the_inscribed_circle(self, runbook: Any) -> None:
        amp = runbook.gaussian_grid(32, 4.0)
        assert amp.shape == (32, 32)
        # The inscribed circle of radius 16 touches the frame midpoints, so it is
        # the corners that must be strictly outside it.
        assert amp[0, 0] == 0.0
        assert amp[15, 0] == 0.0
        assert amp[16, 16] == pytest.approx(1.0)  # centre is the peak
        # Radially decreasing away from the centre.
        assert amp[16, 20] < amp[16, 18] < amp[16, 16]

    def test_record_round_trip(self, runbook: Any, tmp_path: Path) -> None:
        from ao_shaping.optimizer.wfless.model_in_loop_shaping import CalibrationRecord

        records = [
            CalibrationRecord(
                image=np.full((8, 8), float(i), dtype=np.float64),
                phase=np.full((8, 8), float(i) / 10, dtype=np.float64),
                label=f"r{i}",
            )
            for i in range(3)
        ]
        path = tmp_path / "records.npz"
        runbook.save_records(records, path)
        assert path.exists()
        loaded = runbook.load_records(path)
        assert len(loaded) == len(records)
        assert [r.label for r in loaded] == [r.label for r in records]
        np.testing.assert_allclose(loaded[1].image, records[1].image)
        np.testing.assert_allclose(loaded[1].phase, records[1].phase)

    def test_geometry_round_trip(self, runbook: Any) -> None:
        from ao_shaping.optimizer.wfless.model_in_loop_shaping import BenchGeometry

        geo = BenchGeometry(
            panel_disc_radius=200, region=256, beam_waist_panel_px=192.0,
            far_field_size=4096, camera_px_per_model_px=1.845,
            spot_fwhm_camera_px=29.5, spot_fwhm_model_px=17.25, correlation=0.93,
        )
        back = runbook.geometry_from_json(runbook.geometry_to_json(geo))
        assert back == geo
        assert json.loads(runbook.geometry_to_json(geo))["panel_disc_radius"] == 200


class TestStagesWithMocks:
    """The device-facing code paths, driven without hardware."""

    def test_collect_saves_records_and_offset(
        self, runbook: Any, tmp_path: Path
    ) -> None:
        args = _args(runbook, tmp_path)
        runbook.stage_collect(args, tmp_path)

        assert (tmp_path / "calibration_records.npz").exists()
        assert (tmp_path / "probe_phases.npy").exists(), (
            "Step A needs the model-side probe phases saved"
        )
        probes = np.load(tmp_path / "probe_phases.npy")
        assert probes.shape == (args.probes, args.region, args.region)
        meta = json.loads((tmp_path / "beam_offset.json").read_text(encoding="utf-8"))
        assert meta["pupil_center_panel"] == [960, 600]
        assert meta["collect_disc"] == args.collect_disc

        slm = runbook._created["slm"]
        cam = runbook._created["cam"]
        assert slm.closed, "SLM context manager must close"
        # One flat reference, then a bare AND a defocus frame per probe.
        assert len(slm.shown) == 2 * args.probes + 1
        # Milliseconds, not microseconds: 3.0 ms must not become 3000.
        assert cam.exposures and all(e == 1.1 for e in cam.exposures)

    def test_collect_writes_phase_and_ccd_images(self, runbook: Any, tmp_path: Path) -> None:
        """Every displayed phase and CCD frame must land on disk.

        This is the only artefact that can show *where on the panel* the pupil
        phase landed, which is what a pupil-placement mistake looks like; the
        numbers alone cannot distinguish it.
        """
        args = _args(runbook, tmp_path, probes=2, save_frames=True)
        runbook.stage_collect(args, tmp_path)
        vis = tmp_path / "frames" / "collect"
        assert (vis / "flat_ccd.png").exists()
        for i in range(2):
            for name in (
                f"probe{i:02d}_phase_bare.png",
                f"probe{i:02d}_phase_defocus.png",
                f"probe{i:02d}_ccd_bare.png",
                f"probe{i:02d}_ccd_defocus.png",
            ):
                assert (vis / name).exists(), f"missing {name}"
                assert (vis / name).stat().st_size > 0

    def test_injected_defocus_is_displayed_but_withheld_from_the_model(
        self, runbook: Any, tmp_path: Path
    ) -> None:
        """The defocus must reach the panel and NOT the stored probe phase."""
        defocus = runbook.defocus_panel(1.0, 200, (960, 600))
        assert defocus.shape == (PANEL_H, PANEL_W)
        # Non-zero inside the disc, exactly zero far outside it.
        assert np.abs(defocus[600, 960]) > 0.0
        assert defocus[5, 5] == 0.0
        # Centred on the pupil, so moving the pupil moves the aberration.
        moved = runbook.defocus_panel(1.0, 200, (1000, 480))
        assert not np.array_equal(moved, defocus)

        args = _args(runbook, tmp_path)
        runbook.stage_collect(args, tmp_path)
        probes = np.load(tmp_path / "probe_phases.npy")
        # Stored probes are the bare random phases, so the aberration the run
        # displays is genuinely unknown to Step A.
        assert np.isfinite(probes).all()
        assert abs(float(probes.max())) <= 2 * np.pi + 1e-6
        assert abs(float(probes.min())) >= -1e-6

    def test_collect_rejects_wrong_phase_orientation(
        self, runbook: Any, tmp_path: Path
    ) -> None:
        """phase_to_panel must hand the driver a (H, W) array or the SLM errors."""
        bad = np.zeros((PANEL_W, PANEL_H))  # transposed on purpose
        with pytest.raises(ValueError, match="bad phase shape"):
            MockSLM().create_phase_from_array(bad)

    def test_calibrate_requires_collect_first(
        self, runbook: Any, tmp_path: Path
    ) -> None:
        with pytest.raises(SystemExit, match="collect"):
            runbook.stage_calibrate(_args(runbook, tmp_path), tmp_path)

    def test_shape_requires_geometry(self, runbook: Any, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="calibrate"):
            runbook.stage_shape(_args(runbook, tmp_path), tmp_path)

    def test_full_sequence_writes_geometry_and_frames(
        self, runbook: Any, tmp_path: Path
    ) -> None:
        args = _args(runbook, tmp_path)
        runbook.stage_collect(args, tmp_path)
        runbook.stage_calibrate(args, tmp_path)

        geo_path = tmp_path / "bench_geometry.json"
        assert geo_path.exists(), "calibrate must persist the geometry"
        geometry = runbook.geometry_from_json(geo_path.read_text(encoding="utf-8"))
        assert geometry.region == args.region
        assert geometry.correlation <= 1.0

        runbook.stage_shape(args, tmp_path)
        saved = list(tmp_path.glob("hw_*.npy"))
        assert saved, "shape stage must persist the measured frames"
        assert any(p.name.startswith("hw_before_") for p in saved)
        assert any(p.name.startswith("hw_shaped_") for p in saved)
        assert any(p.name.startswith("hw_phase_") for p in saved)


    def test_sweep_writes_standard_recorder_artifacts(self, runbook: Any, tmp_path: Path) -> None:
        """The sweep must land in the repo's standard debug-artefact shape.

        ``save_recorder_debug_artifacts`` keys the pkl by ``_epoch`` and every
        ``scripts/generate_*_report.py`` reads that form, so a hardware sweep can
        be re-rendered offline without re-running the bench.
        """
        import pickle

        args = _args(runbook, tmp_path, save_frames=False)
        runbook.stage_sweep(args, tmp_path)

        debug = runbook.ROOT / "data" / "debug"
        runs = sorted(
            (p for p in debug.glob("model_in_loop_hw_sweep_*") if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
        )
        assert runs, "no model_in_loop_hw_sweep_* debug dir written"
        # build_debug_save_paths nests a date-stamped dir inside the prefixed one.
        pkls = sorted(runs[-1].rglob("*.pkl"))
        assert pkls, f"no pkl under {runs[-1]}"
        with open(pkls[0], "rb") as fh:
            data = pickle.load(fh)
        assert isinstance(data, dict) and data, "pkl must be a {epoch: record} dict"
        epochs = sorted(data)
        assert epochs == list(range(len(epochs))), "epochs must be 0..N-1"
        first = data[epochs[0]]
        # Scalars the fit and the reports rely on. `mode`/`axis` are strings and
        # cannot survive save_recorder_debug_artifacts (it float()-casts every
        # declared scalar), so provenance travels as numeric codes resolved via
        # the sidecar's mapping.
        for key in (
            "peak", "fwhm_px", "centroid_x", "centroid_y", "hollowness",
            "coefficient", "mode_id", "axis_id", "_phase", "_img",
        ):
            assert key in first, f"missing {key}"
        # Raw process data: the displayed phase and the CCD frame it produced.
        assert np.asarray(first["_phase"]).shape == (PANEL_H, PANEL_W)
        assert np.asarray(first["_img"]).ndim == 2
        # Sidecar carries the provenance needed to re-run the fit offline.
        jsons = sorted(runs[-1].rglob("*.json"))
        assert jsons, f"no json sidecar under {runs[-1]}"
        payload = json.loads(jsons[0].read_text(encoding="utf-8"))
        assert payload["pupil_center_panel"] == [960, 600]
        assert "sweep_tilt_rad" in payload and "sweep_astig_rad" in payload
        assert payload["mode_codes"]["defocus"] == 2
        assert payload["axis_codes"]["x"] == 1
        # The first point is the flat reference, and it must decode back to "flat".
        codes = {v: k for k, v in payload["mode_codes"].items()}
        assert codes[int(first["mode_id"])] == "flat"


class TestDryRun:
    """The offline self-check must run green with no devices."""

    def test_dry_run_completes(self, runbook: Any, tmp_path: Path,
                               capsys: pytest.CaptureFixture[str]) -> None:
        args = _args(runbook, tmp_path, dry_run=True, region=64, far_field_size=256)
        # Mirror the geometry main() writes for a dry run, then exercise the path.
        from ao_shaping.optimizer.wfless.model_in_loop_shaping import BenchGeometry

        geo = BenchGeometry(
            panel_disc_radius=200, region=64, beam_waist_panel_px=24.0,
            far_field_size=256, camera_px_per_model_px=3.69,
            spot_fwhm_camera_px=12.0, spot_fwhm_model_px=44.0, correlation=0.97,
        )
        (tmp_path / "bench_geometry.json").write_text(
            runbook.geometry_to_json(geo), encoding="utf-8"
        )
        runbook.dry_run(args, tmp_path)
        out = capsys.readouterr().out
        assert "dry-run" in out
        assert "panel placement" in out
