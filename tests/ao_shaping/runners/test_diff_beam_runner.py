"""Tests for the dual-algorithm diff_beam_runner module.

The runner is loaded by *file path* (``importlib``) because importing it
through the ``ao_shaping.runners`` package triggers ``runners/__init__.py``,
whose import chain performs a network ping in ``config.py`` resolving DM
actuators (pre-existing infra issue that hangs in offline environments).
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from ao_shaping.utils.beam_metrics import compute_metrics
from ao_shaping.utils.targets import square_target_from_measurement

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNNER_PATH = _REPO_ROOT / "src" / "ao_shaping" / "runners" / "diff_beam_runner.py"


@pytest.fixture(scope="module")
def runner_module():
    """Load diff_beam_runner.py directly by file path (bypasses package __init__)."""
    spec = importlib.util.spec_from_file_location("diff_beam_runner", _RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestCliOptions:
    """Test the Click command surface."""

    def test_command_exists(self, runner_module):
        assert hasattr(runner_module.run, "callback")
        assert callable(runner_module.run)

    def test_algorithm_choice(self, runner_module):
        """--algorithm must accept only backprop/gs."""
        choice = [p for p in runner_module.run.params if p.name == "algorithm"][0]
        assert sorted(choice.type.choices) == ["backprop", "gs"]

    def test_target_shape_choice(self, runner_module):
        """--target-shape must accept gaussian/circle/square."""
        choice = [p for p in runner_module.run.params if p.name == "target_shape"][0]
        assert sorted(choice.type.choices) == ["circle", "gaussian", "square"]

    def test_expected_options_exist(self, runner_module):
        param_names = [p.name for p in runner_module.run.params]
        expected = [
            "algorithm",
            "target_image",
            "target_shape",
            "target_size",
            "target_px",
            "epochs",
            "lr",
            "iterations",
            "distance",
            "wavelength",
            "slm_wavelength",
            "slm_number",
            "cam_id",
            "cam_center",
            "cam_size",
            "cam_exposure",
            "cam_exposure_us",
            "auto_exposure",
            "auto_exposure_target",
            "auto_exposure_tol",
            "p_cam",
            "settle_time",
            "capture_timeout",
            "adaptive",
            "adaptive_iterations",
            "device",
            "seed",
            "save_dir",
            "use_hardware",
            "show",
        ]
        for name in expected:
            assert name in param_names, f"Missing CLI option: {name}"


class _FakeAutoExposureCamera:
    """Minimal camera stub exposing only ``reset_exposure_time``."""

    def __init__(self) -> None:
        self.calls: list[float] = []
        self.exposure_ms = 50.0

    def reset_exposure_time(self, time_ms: float) -> float:
        self.calls.append(float(time_ms))
        self.exposure_ms = float(time_ms)
        return self.exposure_ms


class TestAutoExposure:
    """Tests for the pure exposure-tuning logic (匀化过程自动曝光)."""

    def test_in_band_unchanged(self, runner_module):
        """Peak inside target±tol must not change exposure."""
        ms = runner_module.auto_exposure_target_ms(
            current_ms=1.2, peak=170.0, target_brightness=180.0, tol=0.2
        )
        assert ms == 1.2

    def test_low_peak_boosts_exposure(self, runner_module):
        """Weak signal boosts exposure by target/peak (no step cap hit)."""
        ms = runner_module.auto_exposure_target_ms(
            current_ms=1.2, peak=60.0, target_brightness=180.0, tol=0.2
        )
        assert ms == pytest.approx(1.2 * 180.0 / 60.0)  # 3.6 ms

    def test_low_peak_boost_capped(self, runner_module):
        """A single step never boosts more than ``max_boost=4x``."""
        ms = runner_module.auto_exposure_target_ms(
            current_ms=1.2, peak=10.0, target_brightness=180.0, tol=0.2
        )
        assert ms == pytest.approx(1.2 * 4.0, rel=1e-9)  # capped at 4x

    def test_saturated_peak_cuts_hard(self, runner_module):
        """peak >= 245 (8-bit saturation) forces a big cut regardless of caps."""
        ms = runner_module.auto_exposure_target_ms(
            current_ms=1.2, peak=255.0, target_brightness=180.0, tol=0.2
        )
        assert ms == pytest.approx(1.2 * 180.0 / 255.0)  # ~0.847 ms

    def test_high_peak_cuts_floor(self, runner_module):
        """Too-bright but unsaturated peak cuts at target/peak, floored 0.25x."""
        ms = runner_module.auto_exposure_target_ms(
            current_ms=1.2, peak=220.0, target_brightness=180.0, tol=0.2
        )
        expected = 1.2 * 180.0 / 220.0  # 0.9818... within [0.25, 4] band
        assert ms == pytest.approx(expected)

    def test_extreme_saturation_floor(self, runner_module):
        """Hugely over-exposed peak cuts by target/peak, floored at 0.1x."""
        ms = runner_module.auto_exposure_target_ms(
            current_ms=1.2, peak=10000.0, target_brightness=180.0, tol=0.2
        )
        assert ms == pytest.approx(1.2 * 0.1, rel=1e-9)

    def test_clamped_to_bounds(self, runner_module):
        """Result is clamped to [min_ms, max_ms]."""
        tiny = runner_module.auto_exposure_target_ms(
            current_ms=0.02, peak=255.0, target_brightness=180.0, tol=0.2
        )
        assert tiny == pytest.approx(0.02)  # saturated cut would go below min
        huge = runner_module.auto_exposure_target_ms(
            current_ms=1000.0, peak=100.0, target_brightness=180.0, tol=0.2
        )
        assert huge == pytest.approx(1000.0)  # above max clamp

    def test_apply_auto_exposure_changes_camera(self, runner_module):
        """Out-of-band peak drives reset_exposure_time on a real-ish camera."""
        cam = _FakeAutoExposureCamera()
        actual, changed = runner_module._apply_auto_exposure(
            cam, peak=60.0, current_ms=1.2, target_brightness=180.0, tol=0.2
        )
        assert changed is True
        assert actual == pytest.approx(3.6)
        assert len(cam.calls) == 1
        assert cam.calls[0] == pytest.approx(3.6)

    def test_apply_auto_exposure_in_band_skips_camera(self, runner_module):
        """In-band peak neither touches the camera nor reports a change."""
        cam = _FakeAutoExposureCamera()
        actual, changed = runner_module._apply_auto_exposure(
            cam, peak=180.0, current_ms=1.2, target_brightness=180.0, tol=0.2
        )
        assert changed is False
        assert actual == 1.2
        assert cam.calls == []

    def test_auto_exposure_possible_detects_api(self, runner_module):
        assert runner_module._auto_exposure_possible(_FakeAutoExposureCamera()) is True
        assert runner_module._auto_exposure_possible(object()) is False


class TestRunBackprop:
    """Test `run --algorithm backprop` end-to-end in mock mode."""

    def test_backprop_e2e(self, runner_module, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            runner_module.run,
            [
                "--algorithm",
                "backprop",
                "--target-shape",
                "gaussian",
                "--epochs",
                "2",
                "--lr",
                "0.01",
                "--device",
                "cpu",
                "--seed",
                "0",
                "--save-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Results must be persisted.
        dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert len(dirs) == 1
        out_dir = dirs[0]
        for fname in [
            "config.json",
            "phase_pattern.npy",
            "target_intensity.npy",
            "measured_intensity.npy",
            "loss_history.npy",
        ]:
            assert (out_dir / fname).exists(), f"Missing output: {fname}"

        # config.json must record the algorithm and metrics.
        import json

        cfg = json.loads((out_dir / "config.json").read_text())
        assert cfg["algorithm"] == "backprop"
        assert set(cfg["metrics"]) == {"mse", "correlation", "efficiency"}

        # Shared metric definition must be usable on the saved arrays.
        import numpy as np

        measured = np.load(out_dir / "measured_intensity.npy")
        target = np.load(out_dir / "target_intensity.npy")
        m = compute_metrics(measured, target)
        assert set(m) == {"mse", "correlation", "efficiency"}


class TestRunGs:
    """Test `run --algorithm gs` end-to-end in mock mode (plain and adaptive)."""

    def test_gs_e2e(self, runner_module, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            runner_module.run,
            [
                "--algorithm",
                "gs",
                "--target-shape",
                "gaussian",
                "--iterations",
                "5",
                "--save-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert len(dirs) == 1
        out_dir = dirs[0]
        for fname in [
            "config.json",
            "phase_pattern.npy",
            "target_intensity.npy",
            "measured_intensity.npy",
            "loss_history.npy",
        ]:
            assert (out_dir / fname).exists(), f"Missing output: {fname}"

        import json

        cfg = json.loads((out_dir / "config.json").read_text())
        assert cfg["algorithm"] == "gs"

    def test_gs_adaptive_without_hardware_falls_back(self, runner_module, tmp_path):
        """--adaptive without --use-hardware must warn and fall back to plain GS."""
        runner = CliRunner()
        result = runner.invoke(
            runner_module.run,
            [
                "--algorithm",
                "gs",
                "--target-shape",
                "gaussian",
                "--iterations",
                "5",
                "--adaptive",
                "--save-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Fallback warning is emitted by loguru (stderr); verify we still ran.
        assert "Differentiable beam shaping complete!" in result.output


class TestSquareShaping:
    """Test the square-shaping surface (the primary hardware use case)."""

    def test_backprop_square_e2e(self, runner_module, tmp_path):
        """--target-shape square --target-size 40 end-to-end (mock mode)."""
        runner = CliRunner()
        result = runner.invoke(
            runner_module.run,
            [
                "--algorithm",
                "backprop",
                "--target-shape",
                "square",
                "--target-size",
                "40",
                "--epochs",
                "2",
                "--lr",
                "0.01",
                "--device",
                "cpu",
                "--seed",
                "0",
                "--save-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert len(dirs) == 1
        out_dir = dirs[0]

        import json

        import numpy as np

        cfg = json.loads((out_dir / "config.json").read_text())
        assert cfg["target_size"] == 40
        assert cfg["target_px"] is None

        target = np.load(out_dir / "target_intensity.npy")
        measured = np.load(out_dir / "measured_intensity.npy")
        # Full-panel SLM200 grid (1200, 1920) must be used.
        assert target.shape == (1200, 1920)
        assert measured.shape == (1200, 1920)
        # The square bright region is exactly target_size x target_size grid bins.
        bright = target > 0.5 * target.max()
        rows = np.where(bright.any(axis=1))[0]
        cols = np.where(bright.any(axis=0))[0]
        assert rows[-1] - rows[0] + 1 == 40
        assert cols[-1] - cols[0] + 1 == 40

    def test_gs_square_e2e(self, runner_module, tmp_path):
        """--target-shape square works with the gs algorithm too."""
        runner = CliRunner()
        result = runner.invoke(
            runner_module.run,
            [
                "--algorithm",
                "gs",
                "--target-shape",
                "square",
                "--target-size",
                "40",
                "--iterations",
                "5",
                "--save-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert len(dirs) == 1
        out_dir = dirs[0]

        import json

        import numpy as np

        cfg = json.loads((out_dir / "config.json").read_text())
        assert cfg["target_size"] == 40
        target = np.load(out_dir / "target_intensity.npy")
        assert target.shape == (1200, 1920)

    def test_square_target_px_without_hardware_falls_back(self, runner_module, tmp_path):
        """--target-px without --use-hardware warns and falls back to --target-size."""
        runner = CliRunner()
        result = runner.invoke(
            runner_module.run,
            [
                "--algorithm",
                "gs",
                "--target-shape",
                "square",
                "--target-px",
                "20",
                "--target-size",
                "40",
                "--iterations",
                "3",
                "--save-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        import json

        dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert len(dirs) == 1
        cfg = json.loads((dirs[0] / "config.json").read_text())
        # Fallback: --target-size kept its value; --target-px recorded as passed.
        assert cfg["target_size"] == 40
        assert cfg["target_px"] == 20

    def test_cam_exposure_us_priority(self, runner_module, tmp_path):
        """--cam-exposure-us 900 must win over --cam-exposure and reach config."""
        runner = CliRunner()
        result = runner.invoke(
            runner_module.run,
            [
                "--algorithm",
                "gs",
                "--target-shape",
                "gaussian",
                "--iterations",
                "3",
                "--cam-exposure",
                "50",
                "--cam-exposure-us",
                "900",
                "--save-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        import json

        dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert len(dirs) == 1
        cfg = json.loads((dirs[0] / "config.json").read_text())
        assert cfg["cam_exposure_us"] == 900
        assert cfg["effective_exposure_ms"] == 0.9  # 900 us = 0.9 ms


class TestFixedSideSquare:
    """CCD 图片空间固定边长方形 target: 固定边长 side_px (CCD 像素)、
    质心定位、亮度 = 1/边长² (总和 = 1)。loss 用实测帧/总亮度 归一化
    对比, 因此曝光/总亮度变化不影响 target square。"""

    def _make_spot_frame(self) -> tuple:
        """1000x1000 frame: constant background 100 + Gaussian spot (peak 1000)
        planted off-center at (600, 300), sigma=20 px."""
        h, w = 1000, 1000
        y, x = np.mgrid[0:h, 0:w]
        bg, amp = 100.0, 900.0
        cy, cx = 600.0, 300.0
        sigma = 20.0
        frame = bg + amp * np.exp(
            -((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2)
        )
        return frame.astype(np.float32), (cy, cx)

    def test_fixed_side_value_and_sum(self):
        """固定边长 (CCD 像素): 值 = 1/side², CCD 空间方形总和 ≈ 1,
        中心 = 实测质心。"""
        frame, (cy, cx) = self._make_spot_frame()
        target, info = square_target_from_measurement(frame, side_px=20.0)

        # Background subtraction and peak extraction unchanged.
        assert info["background"] == pytest.approx(100.0, abs=0.5)
        assert info["max_brightness"] == pytest.approx(900.0, rel=1e-3)
        # Centroid must follow the *measured* beam (off-center plant).
        assert info["centroid"][0] == pytest.approx(cy, abs=2.0)
        assert info["centroid"][1] == pytest.approx(cx, abs=2.0)

        # Fixed side in CCD image space; value = 1/n_pixels so the square
        # sums to exactly 1 — even with a sub-pixel centroid (floor/ceil can
        # fill side or side+1 pixels per dimension).
        assert info["side_cam_px"] == pytest.approx(20.0)
        assert info["target_ccd"].shape == frame.shape
        n_px = info["n_pixels"]
        assert info["target_ccd"].max() == pytest.approx(1.0 / n_px, rel=1e-6)
        assert info["target_ccd"].sum() == pytest.approx(1.0, rel=1e-6)
        # Square spans ~side x side pixels around the centroid.
        bright = info["target_ccd"] > 0
        rows = np.where(bright.any(axis=1))[0]
        cols = np.where(bright.any(axis=0))[0]
        assert rows[-1] - rows[0] + 1 in (20, 21)
        assert cols[-1] - cols[0] + 1 in (20, 21)
        # Peak is uniform across the whole square (exactly 1/n_px everywhere).
        assert np.allclose(
            info["target_ccd"][bright], 1.0 / n_px, rtol=1e-6
        )

        # Grid-space target: full SLM200 panel, normalized to [0, 1].
        assert target.shape == (1200, 1920)
        assert target.min() >= 0.0
        assert target.max() <= 1.0 + 1e-6
        assert target.max() > 0.0
        # Grid side follows the camera->grid mapping (crop_h = 625 here).
        assert info["side_grid_bins"][0] == pytest.approx(
            info["side_cam_px"] * 1200 / 625, rel=1e-3
        )

    def test_exposure_invariance(self):
        """曝光/总亮度变化不改变 target square: 帧缩放后 (乘任意正因子)
        得到的 CCD 空间 target 必须逐像素相同 (loss 用帧/总亮度对比)。"""
        frame, _ = self._make_spot_frame()
        target_1, info_1 = square_target_from_measurement(frame, side_px=20.0)
        target_2, info_2 = square_target_from_measurement(frame * 3.7, side_px=20.0)

        assert np.array_equal(target_1, target_2)
        assert info_1["side_cam_px"] == info_2["side_cam_px"]
        assert np.array_equal(info_1["target_ccd"], info_2["target_ccd"])
        # Normalized intensity (frame / total) is the comparison operand.
        norm = frame / frame.sum()
        assert norm.shape == info_2["target_ccd"].shape
        assert norm.sum() == pytest.approx(1.0)

    def test_no_signal_raises(self):
        """A zero/dead frame must raise instead of producing a garbage target."""
        frame = np.zeros((100, 100), dtype=np.float32)
        with pytest.raises(ValueError, match="无有效信号"):
            square_target_from_measurement(frame, side_px=20.0)

    def test_invalid_side_raises(self):
        """side_px <= 0 must raise."""
        frame, _ = self._make_spot_frame()
        with pytest.raises(ValueError, match="side_px"):
            square_target_from_measurement(frame, side_px=0.0)

    def _make_stray_light_frame(self) -> tuple:
        """Regress 2026-09-11 hardware discovery (1200us flat frame): the Daheng
        full-frame carries a pervasive *gray==1* stray-light layer (3.5% of
        pixels, every row/col, ~62% of total energy) that drags any intensity
        centroid hundreds of px off the real 0-order spot. Median == 0 so
        percentile background subtraction leaves the layer untouched. Only
        argmax survives; assert square_target_from_measurement keeps pinning
        the planted spot.

        Builds a 1944x2592 frame: 1-gray layer (3.51% random pixels) + Gaussian
        spot (peak 255, sigma=9) planted at the measured-real-spot (689, 1776).
        """
        h, w = 1944, 2592
        rng = np.random.default_rng(0)
        frame = np.zeros((h, w), dtype=np.float32)
        frame[rng.random((h, w)) < 0.0351] = 1.0  # 1-gray layer everywhere
        cy, cx = 689.0, 1776.0
        sigma = 9.0
        y, x = np.mgrid[0:h, 0:w]
        spot = 255.0 * np.exp(
            -((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2)
        )
        frame = np.maximum(frame.astype(np.float32), spot.astype(np.float32))
        return frame, (cy, cx)

    def test_stray_light_layer_does_not_drag_argmax(self):
        """Regression for the 1200us flat-frame finding: a frame whose dominant
        energy is a gray==1 stray layer (median 0 → background subtraction
        ineffective) must still pin the target square on the real spot via
        argmax, not on the drag-biased intensity centroid."""
        frame, (cy, cx) = self._make_stray_light_frame()

        # Sanity: the layer really is there and median background is 0, i.e.
        # percentile-based subtraction cannot remove it (measured 2026-09-11).
        assert np.median(frame) == 0.0
        assert (frame == 1.0).mean() == pytest.approx(0.0351, abs=1e-3)
        bg = float(np.percentile(frame, 10))
        assert bg == 0.0

        # The raw intensity centroid IS dragged far off the spot (the whole
        # reason argmax is used) — confirm the test frame reproduces the trap.
        tot = float(frame.sum())
        yy = (frame * np.arange(frame.shape[0])[:, None]).sum() / tot
        xx = (frame * np.arange(frame.shape[1])[None, :]).sum() / tot
        assert (yy - cy) ** 2 + (xx - cx) ** 2 > 50.0**2

        # square_target_from_measurement must anchor on argmax == real spot.
        target, info = square_target_from_measurement(frame, side_px=20.0)
        assert info["centroid"][0] == pytest.approx(cy, abs=2.0)
        assert info["centroid"][1] == pytest.approx(cx, abs=2.0)
        # The square actually covers the spot → its max overlaps the planted
        # location (argmax of target_ccd sits on the spot row/col span).
        bright = info["target_ccd"] > 0
        rows = np.where(bright.any(axis=1))[0]
        cols = np.where(bright.any(axis=0))[0]
        assert rows[0] <= cy <= rows[-1]
        assert cols[0] <= cx <= cols[-1]
        assert target.shape == (1200, 1920)


class TestRunValidation:
    """Test CLI input validation."""

    def test_invalid_algorithm_rejected(self, runner_module, tmp_path):
        result = CliRunner().invoke(
            runner_module.run,
            ["--algorithm", "bogus", "--save-dir", str(tmp_path)],
        )
        assert result.exit_code == 2  # Click Choice validation failure

    def test_missing_target_image_rejected(self, runner_module, tmp_path):
        result = CliRunner().invoke(
            runner_module.run,
            [
                "--algorithm",
                "backprop",
                "--target-image",
                str(tmp_path / "does_not_exist.png"),
                "--epochs",
                "1",
                "--save-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 2  # click.Path(exists=True) validation failure


if __name__ == "__main__":
    pytest.main([__file__, "-v"])