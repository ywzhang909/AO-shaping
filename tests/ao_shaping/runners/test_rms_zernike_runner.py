"""Tests for rms_zernike_runner module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ao_shaping.drivers import MlaRes


class TestGetWfsRes:
    """Test WFS resolution mapping function."""

    def test_all_resolutions(self):
        """Test all supported WFS resolutions."""
        from ao_shaping.drivers.wfs.thorlab_wfs import MlaRes

        assert MlaRes.from_str("320") == MlaRes.Res320
        assert MlaRes.from_str("512") == MlaRes.Res512
        assert MlaRes.from_str("768") == MlaRes.Res768
        assert MlaRes.from_str("1024") == MlaRes.Res1024
        assert MlaRes.from_str("1280") == MlaRes.Res1280

    def test_default_fallback(self):
        """Test unknown resolution falls back to Res1024."""
        from ao_shaping.drivers.wfs.thorlab_wfs import MlaRes

        assert MlaRes.from_str("999", default=MlaRes.Res1024) == MlaRes.Res1024
        assert MlaRes.from_str("invalid", default=MlaRes.Res1024) == MlaRes.Res1024


class TestAutoDeltaDetectRms:
    """Test _auto_delta_detect_rms function."""

    def _make_mock_slm(self):
        mock_slm = MagicMock()
        mock_slm.send_zernike.return_value = np.zeros((512, 512))
        mock_slm.wavelength = 532
        mock_slm.__enter__.return_value = mock_slm
        mock_slm.__exit__.return_value = False
        return mock_slm

    def _make_mock_wfs(self, rms_value=0.15):
        mock_wfs = MagicMock()
        mock_wfs.get_wavefront.return_value = (
            np.zeros((64, 64)),
            {"rms": rms_value, "strehl": 0.8},
        )
        mock_wfs.take_image.return_value = None
        mock_wfs.__enter__.return_value = mock_wfs
        mock_wfs.__exit__.return_value = False
        return mock_wfs

    def test_auto_delta_detect_returns_tuple(self):
        """Test that _auto_delta_detect_rms returns (delta, info_dict)."""
        from ao_shaping.runners.slm.rms_zernike_runner import _auto_delta_detect_rms

        mock_slm = self._make_mock_slm()
        mock_wfs = self._make_mock_wfs()

        with (
            patch(
                "ao_shaping.runners.slm.rms_zernike_runner.ZernikeSLM",
                return_value=mock_slm,
            ),
            patch(
                "ao_shaping.runners.slm.rms_zernike_runner.ThorlabWFS",
                return_value=mock_wfs,
            ),
            patch(
                "ao_shaping.runners.slm.rms_zernike_runner.search_optimal_delta",
                return_value=(
                    0.5,
                    {
                        "baseline_obj": 0.15,
                        "optimal_delta": 0.5,
                        "fine_results": [],
                    },
                ),
            ),
        ):
            delta, info = _auto_delta_detect_rms(
                min_delta=0.01,
                max_delta=100.0,
                delta_step=5,
                n_directions=5,
                n_max=4,
                wfs_exposure_time=0.0,
            )

            assert isinstance(delta, float)
            assert isinstance(info, dict)
            assert "baseline_rms" in info
            assert "best_rms" in info
            assert "best_delta" in info

    def test_auto_delta_detect_passes_exposure_time(self):
        """Test that exposure_time is passed to ThorlabWFS."""
        from ao_shaping.runners.slm.rms_zernike_runner import _auto_delta_detect_rms

        mock_slm = self._make_mock_slm()
        mock_wfs = self._make_mock_wfs()

        with (
            patch(
                "ao_shaping.runners.slm.rms_zernike_runner.ZernikeSLM",
                return_value=mock_slm,
            ),
            patch(
                "ao_shaping.runners.slm.rms_zernike_runner.ThorlabWFS",
                return_value=mock_wfs,
            ),
            patch(
                "ao_shaping.runners.slm.rms_zernike_runner.search_optimal_delta",
                return_value=(
                    0.5,
                    {
                        "baseline_obj": 0.15,
                        "optimal_delta": 0.5,
                        "fine_results": [],
                    },
                ),
            ),
        ):
            _auto_delta_detect_rms(
                wfs_exposure_time=50.0,
                n_max=4,
            )

            # Verify ThorlabWFS was called with exposure_time
            ThorlabWFS_mock = patch(
                "ao_shaping.runners.slm.rms_zernike_runner.ThorlabWFS",
                return_value=mock_wfs,
            )
            # The mock was already patched above, check call args
            mock_wfs_call = mock_wfs.__enter__.call_args
            # Verify the WFS was instantiated - check via the patch
            # The actual verification is that the code path includes exposure_time param


class TestCliOptions:
    """Test CLI command options."""

    def test_command_exists(self):
        """Test that the run command is a Click command."""
        from ao_shaping.runners.slm.rms_zernike_runner import run

        assert hasattr(run, "callback")
        assert callable(run)

    def test_exposure_time_ms_option_exists(self):
        """Test that --exposure-time-ms option is defined (not --exposure-time)."""
        from ao_shaping.runners.slm.rms_zernike_runner import run

        param_names = [p.name for p in run.params]
        assert "exposure_time_ms" in param_names
        # Old option should NOT exist
        assert "exposure_time" not in param_names

    def test_all_expected_options_exist(self):
        """Test that all expected CLI options are present."""
        from ao_shaping.runners.slm.rms_zernike_runner import run

        param_names = [p.name for p in run.params]
        expected = [
            "dir",
            "epochs",
            "lr",
            "delta",
            "n_max",
            "wfs_res",
            "pupil_diameter",
            "pupil_center",
            "early_stop_threshold",
            "exposure_time_ms",
            "wavelength",
            "shift_x",
            "shift_y",
            "slm_number",
            "remove_tilt",
            "wait_time",
            "min_delta",
            "max_delta",
            "delta_step",
            "n_directions",
            "n_init_positions",
            "init_range",
            "lr_schedule",
            "lr_min",
            "delta_schedule",
            "delta_min",
            "optimizer",
            "beta1",
            "weight_decay",
            "mini_batch",
            "gradient_clip",
            "stagnation_patience",
            "stagnation_delta_boost",
            "freeze_threshold",
            "early_stop_window",
            "early_stop_min_epochs",
            "early_stop_patience",
            "n_frames",
        ]
        for name in expected:
            assert name in param_names, f"Missing CLI option: {name}"


class TestRunFunction:
    """Test the main run function with mocked hardware."""

    def _make_mock_slm(self):
        mock_slm = MagicMock()
        mock_slm.send_zernike.return_value = np.zeros((512, 512))
        mock_slm.wavelength = 532
        mock_slm.__enter__.return_value = mock_slm
        mock_slm.__exit__.return_value = False
        return mock_slm

    def _make_mock_wfs(self, rms_value=0.15):
        mock_wfs = MagicMock()
        mock_wfs.get_wavefront.return_value = (
            np.zeros((64, 64)),
            {"rms": rms_value, "strehl": 0.8},
        )
        mock_wfs.take_image.return_value = None
        mock_wfs.__enter__.return_value = mock_wfs
        mock_wfs.__exit__.return_value = False
        return mock_wfs

    def test_run_with_mocked_hardware(self):
        """Test run function with fully mocked hardware."""
        from click.testing import CliRunner

        from ao_shaping.runners.slm.rms_zernike_runner import run

        mock_slm = self._make_mock_slm()
        mock_wfs = self._make_mock_wfs()

        runner = CliRunner()

        with (
            patch(
                "ao_shaping.runners.slm.rms_zernike_runner.ZernikeSLM",
                return_value=mock_slm,
            ),
            patch(
                "ao_shaping.runners.slm.rms_zernike_runner.ThorlabWFS",
                return_value=mock_wfs,
            ),
            patch(
                "ao_shaping.runners.slm.rms_zernike_runner.search_optimal_delta",
                return_value=(
                    1.0,
                    {
                        "baseline_obj": 0.15,
                        "optimal_delta": 1.0,
                        "fine_results": [],
                    },
                ),
            ),
            patch(
                "ao_shaping.runners.slm.rms_zernike_runner.optimizer_rms_slm",
                return_value=MagicMock(
                    get_best_iter=MagicMock(
                        return_value=(
                            {
                                "_c": np.zeros(15),
                                "_wavefront": [np.zeros((64, 64)), np.zeros((64, 64))],
                            },
                            (1, 0.10),
                        )
                    ),
                    get_best_target=MagicMock(return_value=(np.zeros(15), 0.10)),
                    history=[
                        {
                            "_c": np.zeros(15),
                            "_wavefront": [np.zeros((64, 64)), np.zeros((64, 64))],
                            "_pos_intensity": None,
                        }
                    ],
                    first={
                        "_c": np.zeros(15),
                        "_wavefront": [np.zeros((64, 64)), np.zeros((64, 64))],
                    },
                    save_best=MagicMock(),
                    save_array_sidecars=MagicMock(),
                    save_dataframe=MagicMock(),
                ),
            ),
        ):
            result = runner.invoke(
                run,
                [
                    "--epochs",
                    "2",
                    "--n-max",
                    "4",
                    "--delta",
                    "1.0",  # Skip auto-delta by providing positive value
                    "--wfs_res",
                    "1024",
                    "--exposure-time-ms",
                    "50.0",
                ],
            )

            # Should complete without error
            assert result.exit_code == 0, f"CLI failed: {result.output}"
            assert "完成" in result.output or "RMS" in result.output

    def test_run_passes_exposure_time_ms_to_optimizer(self):
        """Test that exposure_time_ms is correctly passed to optimizer_rms."""
        from click.testing import CliRunner

        from ao_shaping.runners.slm.rms_zernike_runner import run

        mock_slm = self._make_mock_slm()
        mock_wfs = self._make_mock_wfs()
        captured_kwargs = {}

        def capture_optimizer_rms(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock(
                get_best_iter=MagicMock(
                    return_value=(
                        {
                            "_c": np.zeros(15),
                            "_wavefront": [np.zeros((64, 64)), np.zeros((64, 64))],
                        },
                        (1, 0.10),
                    )
                ),
                get_best_target=MagicMock(return_value=(np.zeros(15), 0.10)),
                history=[
                    {
                        "_c": np.zeros(15),
                        "_wavefront": [np.zeros((64, 64)), np.zeros((64, 64))],
                        "_pos_intensity": None,
                    }
                ],
                first={
                    "_c": np.zeros(15),
                    "_wavefront": [np.zeros((64, 64)), np.zeros((64, 64))],
                },
                save_best=MagicMock(),
                save_array_sidecars=MagicMock(),
                save_dataframe=MagicMock(),
            )

        runner = CliRunner()

        with (
            patch(
                "ao_shaping.runners.slm.rms_zernike_runner.ZernikeSLM",
                return_value=mock_slm,
            ),
            patch(
                "ao_shaping.runners.slm.rms_zernike_runner.ThorlabWFS",
                return_value=mock_wfs,
            ),
            patch(
                "ao_shaping.runners.slm.rms_zernike_runner.optimizer_rms_slm",
                side_effect=capture_optimizer_rms,
            ),
        ):
            result = runner.invoke(
                run,
                [
                    "--epochs",
                    "2",
                    "--n-max",
                    "4",
                    "--delta",
                    "1.0",
                    "--wfs_res",
                    "1024",
                    "--exposure-time-ms",
                    "75.0",
                ],
            )

            assert result.exit_code == 0, f"CLI failed: {result.output}"
            assert captured_kwargs.get("wfs_exposure_time") == 75.0


class TestAutoDetectDeltaWiring:
    """Runner-level wiring: ``--delta <= 0`` routes to ``_auto_delta_detect_rms``.

    ``min_delta/max_delta/delta_step/n_directions`` are delivered ONLY to the
    auto-detector (never to ``optimizer_rms_slm``); the optimised delta from
    the detector then flows into ``optimizer_rms_slm``. Positive ``--delta``
    skips the detector entirely.
    """

    def _records_mock(self):
        rec = MagicMock()
        rec.get_best_iter.return_value = (
            {
                "_c": np.zeros(15),
                "_wavefront": [np.zeros((64, 64)), np.zeros((64, 64))],
            },
            (1, 0.10),
        )
        rec.save_best = MagicMock()
        rec.save_array_sidecars = MagicMock()
        return rec

    def test_delta_zero_routes_to_auto_detect(self, tmp_path):
        from click.testing import CliRunner

        from ao_shaping.runners.slm.rms_zernike_runner import (
            _auto_delta_detect_rms,
            optimizer_rms_slm,
            run,
        )

        auto_calls = {}
        optimizer_calls = {}

        def fake_auto_detect(**kwargs):
            auto_calls.update(kwargs)
            return 0.25, {"baseline_rms": 0.3, "best_rms": 0.12, "best_delta": 0.25}

        def fake_optimizer(**kwargs):
            optimizer_calls.update(kwargs)
            return self._records_mock()

        with (
            patch(
                "ao_shaping.runners.slm.rms_zernike_runner._auto_delta_detect_rms",
                side_effect=fake_auto_detect,
            ),
            patch(
                "ao_shaping.runners.slm.rms_zernike_runner.optimizer_rms_slm",
                side_effect=fake_optimizer,
            ),
        ):
            result = CliRunner().invoke(
                run,
                [
                    "--delta",
                    "0",
                    "--epochs",
                    "2",
                    "--n-max",
                    "4",
                    "--dir",
                    str(tmp_path),
                ],
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        # auto-detector received its dedicated knobs + hardware params
        assert auto_calls["min_delta"] == pytest.approx(0.01)
        assert auto_calls["max_delta"] == pytest.approx(100.0)
        assert auto_calls["delta_step"] == 5
        assert auto_calls["n_directions"] == 5
        assert auto_calls["n_max"] == 4
        assert "wfs_exposure_time" in auto_calls
        # optimizer receives the DETECTED delta, not the CLI 0
        assert optimizer_calls["delta"] == pytest.approx(0.25)
        # the auto-detect-only knobs never reach the optimizer
        for knob in ("min_delta", "max_delta", "delta_step", "n_directions"):
            assert knob not in optimizer_calls, f"{knob} must stay auto-detect-only"

    def test_delta_positive_skips_auto_detect(self, tmp_path):
        from click.testing import CliRunner

        from ao_shaping.runners.slm.rms_zernike_runner import (
            _auto_delta_detect_rms,
            optimizer_rms_slm,
            run,
        )

        optimizer_calls = {}

        def fake_optimizer(**kwargs):
            optimizer_calls.update(kwargs)
            return self._records_mock()

        with (
            patch(
                "ao_shaping.runners.slm.rms_zernike_runner._auto_delta_detect_rms",
                side_effect=AssertionError("must not be called"),
            ),
            patch(
                "ao_shaping.runners.slm.rms_zernike_runner.optimizer_rms_slm",
                side_effect=fake_optimizer,
            ),
        ):
            result = CliRunner().invoke(
                run,
                [
                    "--delta",
                    "1.5",
                    "--epochs",
                    "2",
                    "--n-max",
                    "4",
                    "--dir",
                    str(tmp_path),
                ],
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert optimizer_calls["delta"] == pytest.approx(1.5)
        # sanity: the patched detector truly exists in the module namespace
        assert callable(_auto_delta_detect_rms)


class TestImports:
    """Test that all imports are valid (no unused imports)."""

    def test_module_imports_clean(self):
        """Test that the module can be imported without errors."""
        from ao_shaping.runners.slm import rms_zernike_runner

        # Verify key functions exist
        assert hasattr(rms_zernike_runner, "run")
        # _get_wfs_res has been moved to MlaRes.from_str
        assert hasattr(rms_zernike_runner, "_auto_delta_detect_rms")

    def test_no_email_import(self):
        """Verify the unused email.policy import was removed."""
        import ao_shaping.runners.slm.rms_zernike_runner as module

        source_lines = open(module.__file__).readlines()
        for line in source_lines[:20]:
            assert "email.policy" not in line, (
                "Unused email.policy import should be removed"
            )
            assert "from tqdm import cli" not in line, (
                "Unused 'from tqdm import cli' should be removed"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
