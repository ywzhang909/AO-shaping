"""Offline tests for scripts/slm/fit_bench_twin.py's focal-plane scale convention.

The script's whole purpose is to fit the **real bench's** twin; upstream
(``drivers/sim/bench_twin.py``, "两种焦面标度口径") pins that the bench must be
fitted at its **measured** far-field scale (2.614022 cam-px/model-px ⇒ equivalent
focal 311.32 mm), because the nominal focal length gives a scale off by 2.54×.
These tests pin the three scale routes (CLI option / preflight geometry file /
nominal fallback with loud warning) and the padding-consistency rescale, by
checking the written config's ``focal_length_m`` against the pure function
``equivalent_focal_length_m`` and the round-trip through
``BenchTwin(...).camera_px_per_model_px()``.
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
SCRIPT = ROOT / "scripts" / "slm" / "fit_bench_twin.py"

REGION = 16
MODEL_SIDE = 32
MEASURED_SCALE = 2.614022
WAVELENGTH_M = 1064e-9
SLM_PIXEL_UM = 8.0
CCD_PIXEL_UM = 2.2
REGION_SPAN_PX = 900.0


def _load_script() -> Any:
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("_fit_bench_twin", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_fit_bench_twin"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> Any:
    return _load_script()


def _make_run_dir(
    tmp_path: Path,
    *,
    geometry: dict[str, Any] | None = None,
    region: int = REGION,
    model_side: int = MODEL_SIDE,
) -> Path:
    """Fake ``slm-model-in-loop --debug`` run dir: probes/*.npz + amplitude."""
    run_dir = tmp_path / "run"
    probes = run_dir / "probes"
    probes.mkdir(parents=True)
    rng = np.random.default_rng(7)
    for i in range(3):
        phase = rng.normal(0.0, 0.5, size=(region, region))
        frame = rng.uniform(0.0, 100.0, size=(model_side, model_side))
        np.savez_compressed(probes / f"{i:04d}.npz", phase=phase, model_grid=frame)
    np.save(
        run_dir / "source_amplitude.npy",
        rng.uniform(0.1, 1.0, size=(model_side, model_side)),
    )
    if geometry is not None:
        preflight = run_dir / "preflight"
        preflight.mkdir()
        (preflight / "geometry.json").write_text(
            json.dumps(geometry, indent=2), encoding="utf-8"
        )
    return run_dir


def _expected_focal(scale: float, *, far_field_padding: int) -> float:
    from ao_shaping.drivers.sim.bench_twin import equivalent_focal_length_m

    return equivalent_focal_length_m(
        scale,
        wavelength_m=WAVELENGTH_M,
        slm_pixel_um=SLM_PIXEL_UM,
        ccd_pixel_um=CCD_PIXEL_UM,
        region=REGION,
        region_span_px=REGION_SPAN_PX,
        far_field_padding=far_field_padding,
    )


def _run(script: Any, run_dir: Path, out: Path, *extra: str) -> None:
    rc = script.main(
        [
            "--from-dir", str(run_dir),
            "--out", str(out),
            "--maxiter", "1",
            "--popsize", "2",
            "--maxiter-fine", "3",
            *extra,
        ]
    )
    assert rc == 0


class TestMeasuredScaleRoutes:
    def test_geometry_file_scale_becomes_equivalent_focal(
        self, script: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        run_dir = _make_run_dir(
            tmp_path,
            geometry={
                "panel_disc_radius": 450,
                "region": REGION,
                "beam_waist_panel_px": 250.0,
                "far_field_size": MODEL_SIDE,
                "camera_px_per_model_px": MEASURED_SCALE,
                "spot_fwhm_camera_px": 10.0,
                "spot_fwhm_model_px": 4.0,
                "correlation": 0.9,
            },
        )
        out = tmp_path / "fit.json"
        _run(script, run_dir, out)

        from ao_shaping.tools.slm.bench_twin_calibrate import load_fit
        from ao_shaping.drivers.sim.bench_twin import BenchTwin

        config = load_fit(out).config
        assert config.focal_length_m == pytest.approx(
            _expected_focal(MEASURED_SCALE, far_field_padding=MODEL_SIDE // REGION),
            rel=1e-12,
        )
        assert BenchTwin(config).camera_px_per_model_px() == pytest.approx(
            MEASURED_SCALE, rel=1e-9
        )
        captured = capsys.readouterr().out
        assert "实测标度" in captured
        assert "geometry.json" in captured
        assert "标称焦距口径** (无实测标度" not in captured

    def test_cli_option_overrides_geometry_file(
        self, script: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        run_dir = _make_run_dir(
            tmp_path,
            geometry={
                "region": REGION,
                "far_field_size": MODEL_SIDE,
                "camera_px_per_model_px": MEASURED_SCALE,
            },
        )
        out = tmp_path / "fit.json"
        _run(script, run_dir, out, "--camera-px-per-model-px", "1.5")

        from ao_shaping.tools.slm.bench_twin_calibrate import load_fit

        config = load_fit(out).config
        assert config.focal_length_m == pytest.approx(
            _expected_focal(1.5, far_field_padding=MODEL_SIDE // REGION), rel=1e-12
        )
        assert "操作员显式给定" in capsys.readouterr().out

    def test_geometry_scale_rescaled_between_padding_sizes(
        self, script: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        run_dir = _make_run_dir(
            tmp_path,
            geometry={
                "region": REGION,
                "far_field_size": 64,
                "camera_px_per_model_px": MEASURED_SCALE,
            },
        )
        out = tmp_path / "fit.json"
        _run(script, run_dir, out)

        from ao_shaping.tools.slm.bench_twin_calibrate import load_fit

        rescaled = MEASURED_SCALE * 64 / MODEL_SIDE
        config = load_fit(out).config
        assert config.focal_length_m == pytest.approx(
            _expected_focal(rescaled, far_field_padding=MODEL_SIDE // REGION),
            rel=1e-12,
        )
        assert "标度 ∝ 1/补零尺寸" in capsys.readouterr().out

    def test_geometry_region_mismatch_exits(
        self, script: Any, tmp_path: Path
    ) -> None:
        run_dir = _make_run_dir(
            tmp_path,
            geometry={
                "region": 64,
                "far_field_size": MODEL_SIDE,
                "camera_px_per_model_px": MEASURED_SCALE,
            },
        )
        with pytest.raises(SystemExit) as excinfo:
            _run(script, run_dir, tmp_path / "fit.json")
        assert "region 64" in str(excinfo.value)


class TestNominalFallback:
    def test_missing_geometry_file_uses_nominal_and_warns(
        self, script: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        run_dir = _make_run_dir(tmp_path)  # no preflight/ at all
        out = tmp_path / "fit.json"
        _run(script, run_dir, out)

        from ao_shaping.tools.slm.bench_twin_calibrate import load_fit

        config = load_fit(out).config
        assert config.focal_length_m == pytest.approx(0.125), (
            "fallback must keep the nominal BenchTwinConfig default"
        )
        captured = capsys.readouterr().out
        assert "警告 (焦面标度口径)" in captured
        assert "2.614022" in captured
        assert "不可直接比较" in captured
        assert "标称焦距口径" in captured


class TestDefocusDegeneracyNote:
    def test_panel_noll4_note_printed(self, script: Any, tmp_path: Path,
                                      capsys: pytest.CaptureFixture[str]) -> None:
        run_dir = _make_run_dir(tmp_path)
        _run(script, run_dir, tmp_path / "fit.json", "--fit-modes", "4,5")
        assert "近似简并" in capsys.readouterr().out

    def test_no_note_without_panel_defocus(self, script: Any, tmp_path: Path,
                                           capsys: pytest.CaptureFixture[str]) -> None:
        run_dir = _make_run_dir(tmp_path)
        _run(script, run_dir, tmp_path / "fit.json", "--fit-modes", "5,6")
        assert "近似简并" not in capsys.readouterr().out
