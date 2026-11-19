"""The focus (defocus) compensation preflight inside the MIL hardware path.

The 2f bench sits permanently **out of focus**.  A forward model fitted on an
uncorrected bench predicts a **broader** spot than the hardware produces
(measured width ratio 1.328, CV bias -0.3009 over 51/51 records --
``report/shape_trials/experiment_summary_20261022.md`` §2).  So the Fresnel
focus phase has to be on the panel for every displayed pupil phase, including
the Step-A probes.

What is under test:

1. with the compensation **off** the panel phase that reaches the SLM is
   bit-identical to the one produced before the stage existed;
2. with it **on** the panel phase is exactly ``pupil + focus``;
3. the evidence file is written either way, and a requested-but-missing
   compensation aborts instead of quietly collecting on an uncorrected bench.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ao_shaping.optimizer.wfless import slm_model_in_loop as mil
from ao_shaping.optimizer.wfless.slm_model_in_loop import (
    SlmModelInLoopConfig,
    _HardwareBench,
)
from ao_shaping.utils.slm.panel_geometry import phase_to_panel

PANEL_H, PANEL_W = 1200, 1920
FOCAL_MM = 1783.5418257111967


def _focus_json(directory: Path, *, focal_mm: float = FOCAL_MM) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "focus.json"
    path.write_text(
        json.dumps(
            {
                "focus": {
                    "best_focal_mm": focal_mm,
                    "best_diopter_m1": 1.0 / focal_mm,
                    "gain_vs_flat": 31.36,
                },
                "settings": {
                    "center_x": 960,
                    "center_y": 600,
                    "radius": 450,
                    "wavelength": 1064,
                    "slm_number": 1,
                    "cam_type": "daheng",
                    "cam_id": 0,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


class _MockCamera:
    """Just enough camera for the bench to be constructed."""

    is_mock = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.exposure_time = 0.02
        self.opened = False

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.opened = False

    def reset_exposure_time(self, ms: float) -> float:
        self.exposure_time = float(ms)
        return float(ms)

    def get_numpy_image(self, n_sample: int = 1) -> np.ndarray:
        return np.zeros((8, 8), dtype=np.uint8)


class _MockSLM:
    """Just enough panel for the bench to be constructed."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.shown: list[np.ndarray] = []

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def get_displayed_memory_number(self) -> int:
        return 5

    def get_current_grayscale(self) -> np.ndarray:
        return np.zeros((PANEL_H, PANEL_W), dtype=np.uint16)

    def create_phase_from_array(self, phase_rad: np.ndarray) -> np.ndarray:
        return np.clip(
            np.mod(phase_rad, 2 * np.pi) / (2 * np.pi) * 1023, 0, 1023
        ).astype(np.uint16)

    def display_data(self, gray: np.ndarray, wait_time_s: float = 0.0) -> None:
        self.shown.append(np.asarray(gray))


def _patch_devices(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    import ao_shaping.drivers.ccd.common as ccd_common
    import ao_shaping.drivers.slm.santec as santec_module

    created: dict[str, Any] = {}

    def make_cam(*args: Any, **kwargs: Any) -> _MockCamera:
        created["cam"] = _MockCamera(*args, **kwargs)
        return created["cam"]

    def make_slm(*args: Any, **kwargs: Any) -> _MockSLM:
        created["slm"] = _MockSLM(*args, **kwargs)
        return created["slm"]

    monkeypatch.setattr(ccd_common, "create_camera", make_cam)
    monkeypatch.setattr(santec_module, "Santec", make_slm)
    return created


def _config(tmp_path: Path, **over: Any) -> SlmModelInLoopConfig:
    values: dict[str, Any] = dict(
        cam_type="mock",
        cam_id=0,
        cam_size=0,  # skip the full-frame pre-capture
        slm_number=1,
        slm_wavelength=1064,
        region=32,
        panel_span_px=900.0,
        pupil_center=(960, 600),
        target_shape="square",
        target_side=16,
        n_orders=2,
        far_field_padding=2,
        max_exposure_ms=0.4,
        debug_dir=str(tmp_path),
    )
    values.update(over)
    return SlmModelInLoopConfig(**values)


def _displayed_panel(bench: _HardwareBench, phase_model: np.ndarray) -> np.ndarray:
    bench.display(phase_model)
    assert bench._pending is not None
    return np.asarray(bench._pending)


class TestPreflightWiring:
    def test_off_is_bit_identical_to_the_pre_change_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_devices(monkeypatch)
        bench = _HardwareBench(_config(tmp_path))
        phase = np.random.default_rng(0).uniform(0, 2 * np.pi, (32, 32))
        displayed = _displayed_panel(bench, phase)
        expected = phase_to_panel(phase, 450, (960, 600))
        assert np.array_equal(displayed, expected)

    def test_on_adds_the_focus_phase_to_the_pupil_phase(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ao_shaping.tools.kernels.bench_kernels import fresnel_panel

        focus_dir = tmp_path / "focus"
        _focus_json(focus_dir)
        _patch_devices(monkeypatch)
        bench = _HardwareBench(
            _config(tmp_path, focus_dir=str(focus_dir), focus_compensation=True)
        )
        phase = np.random.default_rng(1).uniform(0, 2 * np.pi, (32, 32))
        displayed = _displayed_panel(bench, phase)
        pupil = phase_to_panel(phase, 450, (960, 600))
        focus = fresnel_panel(
            FOCAL_MM,
            radius=450,
            pupil_center=(960, 600),
            panel_shape=(PANEL_H, PANEL_W),
            wavelength_nm=1064.0,
        )
        assert np.array_equal(displayed, focus + pupil)

    def test_the_evidence_file_is_written_when_the_compensation_is_off(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_devices(monkeypatch)
        _HardwareBench(_config(tmp_path))
        path = tmp_path / "focus_preflight.json"
        assert path.exists(), "the off state must be recorded too"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["requested"] is False
        assert payload["enabled"] is False
        assert payload["disable_reason"]

    def test_the_evidence_file_records_the_calibration_when_enabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        focus_dir = tmp_path / "focus"
        _focus_json(focus_dir)
        _patch_devices(monkeypatch)
        _HardwareBench(
            _config(tmp_path, focus_dir=str(focus_dir), focus_compensation=True)
        )
        payload = json.loads(
            (tmp_path / "focus_preflight.json").read_text(encoding="utf-8")
        )
        assert payload["requested"] is True
        assert payload["enabled"] is True
        assert payload["calibration"]["focal_mm"] == pytest.approx(FOCAL_MM)
        assert payload["overlay_order"] == ["focus", "tilt", "panel"]

    def test_a_broken_evidence_file_aborts_before_any_device_opens(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        focus_dir = tmp_path / "focus"
        focus_dir.mkdir()
        (focus_dir / "focus.json").write_text("{not json", encoding="utf-8")
        created = _patch_devices(monkeypatch)
        with pytest.raises(RuntimeError, match="焦面补偿前置阶段失败"):
            _HardwareBench(
                _config(tmp_path, focus_dir=str(focus_dir), focus_compensation=True)
            )
        assert not created, "no device may be opened when the evidence is broken"

    def test_requesting_the_compensation_without_evidence_aborts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_devices(monkeypatch)
        with pytest.raises(RuntimeError, match="焦面补偿被要求但没有生效"):
            _HardwareBench(_config(tmp_path, focus_compensation=True))
        payload = json.loads(
            (tmp_path / "focus_preflight.json").read_text(encoding="utf-8")
        )
        assert payload["requested"] is True
        assert payload["enabled"] is False

    def test_the_config_defaults_keep_the_compensation_off(self) -> None:
        config = SlmModelInLoopConfig()
        assert config.focus_compensation is False
        assert config.focus_dir is None
        assert config.focus_live_probe is False

    def test_the_simulated_bench_is_not_given_a_focus_phase(self) -> None:
        """The digital twin is in focus by construction; a focus phase there
        would defocus it, so the combination is rejected outright."""
        with pytest.raises(ValueError, match="already in focus"):
            _config(tmp_path=Path("."), cam_type="sim", focus_compensation=True)
        config = _config(tmp_path=Path("."), cam_type="sim")
        bench = mil._SimBench(config)
        phase = np.zeros((config.region, config.region), dtype=np.float64)
        bench.display(phase)
        assert np.array_equal(bench._phase, phase)
