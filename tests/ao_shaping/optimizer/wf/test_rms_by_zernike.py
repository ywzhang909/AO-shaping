"""Tests for rms_by_zernike optimizer module."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestImport:
    """Test that module can be imported."""

    def test_import(self):
        """Test importing the optimizer function."""
        from ao_shaping.optimizer.wf.rms_by_zernike import optimizer_rms

        assert callable(optimizer_rms)


class TestHelperFunctions:
    """Test helper functions in the module."""

    def test_noll_to_nm(self):
        """Test Noll index to (n, m) conversion with known values."""
        from ao_shaping.optimizer.wf.rms_by_zernike import noll_to_nm

        # Test known Noll indices
        assert noll_to_nm(1) == (0, 0)   # piston
        assert noll_to_nm(2) == (1, -1)  # tilt x
        assert noll_to_nm(3) == (1, 1)  # tilt y
        assert noll_to_nm(4) == (2, -2)  # oblique astigmatism
        assert noll_to_nm(5) == (2, 0)  # defocus
        assert noll_to_nm(6) == (2, 2)  # oblique astigmatism

    def test_noll_to_nm_invalid(self):
        """Test that invalid Noll indices raise ValueError."""
        from ao_shaping.optimizer.wf.rms_by_zernike import noll_to_nm

        with pytest.raises(ValueError):
            noll_to_nm(0)  # too small

        with pytest.raises(ValueError):
            noll_to_nm(100)  # too large

    def test_zernike_indices(self):
        """Test Zernike indices generation with n_max=4."""
        from ao_shaping.optimizer.wf.rms_by_zernike import _zernike_indices

        modes = _zernike_indices(n_max=4)

        # n_max=4 should give 15 modes (including piston)
        assert len(modes) == 15

        # First few should be piston and tilts
        assert modes[0] == (0, 0)
        assert modes[1] == (1, -1)
        assert modes[2] == (1, 1)

    def test_zernike_indices_n_max_2(self):
        """Test Zernike indices with n_max=2."""
        from ao_shaping.optimizer.wf.rms_by_zernike import _zernike_indices

        modes = _zernike_indices(n_max=2)

        # n_max=2 gives 6 modes
        assert len(modes) == 6


class TestOptimizerReturnsRecorder:
    """Test that optimizer returns a Recorder object with expected fields."""

    def test_optimizer_rms_returns_recorder(self):
        """Test that optimizer_rms returns a Recorder with expected fields."""
        from ao_shaping.optimizer.wf.rms_by_zernike import optimizer_rms

        # Create mock objects for SLM and WFS
        mock_slm = MagicMock()
        mock_slm.send_zernike.return_value = np.zeros((512, 512))
        mock_slm.wavelength = 1064

        mock_wfs = MagicMock()
        mock_wfs.get_wavefront.return_value = (
            np.zeros((64, 64)),
            {"rms": 0.15, "strehl": 0.8}
        )
        mock_wfs.take_image.return_value = None

        with patch(
            "ao_shaping.optimizer.wf.rms_by_zernike.ZernikeSLM",
            return_value=mock_slm,
        ), patch(
            "ao_shaping.optimizer.wf.rms_by_zernike.Thorlab_WFS",
            return_value=mock_wfs,
        ), patch(
            "ao_shaping.optimizer.wf.rms_by_zernike.tqdm"
        ):
            # Run optimizer with minimal epochs
            recorder = optimizer_rms(
                epochs=2,
                n_max=4,
                slm_number=1,
            )

            # Verify return type and basic properties
            assert recorder is not None
            assert hasattr(recorder, "history")
            assert len(recorder.history) > 0

            # Check expected fields in first record
            first_record = recorder.history[0]
            assert "rms" in first_record
            assert "_c" in first_record
            assert "_epoch" in first_record

    def test_recorder_initial_state(self):
        """Test that initial state is recorded correctly."""
        from ao_shaping.optimizer.wf.rms_by_zernike import optimizer_rms

        mock_slm = MagicMock()
        mock_slm.send_zernike.return_value = np.zeros((512, 512))
        mock_slm.wavelength = 1064

        mock_wfs = MagicMock()
        mock_wfs.get_wavefront.return_value = (
            np.zeros((64, 64)),
            {"rms": 0.2, "strehl": 0.7}
        )
        mock_wfs.take_image.return_value = None

        with patch(
            "ao_shaping.optimizer.wf.rms_by_zernike.ZernikeSLM",
            return_value=mock_slm,
        ), patch(
            "ao_shaping.optimizer.wf.rms_by_zernike.Thorlab_WFS",
            return_value=mock_wfs,
        ), patch(
            "ao_shaping.optimizer.wf.rms_by_zernike.tqdm"
        ):
            recorder = optimizer_rms(
                epochs=1,
                n_max=2,
            )

            first_record = recorder.history[0]
            assert first_record["_epoch"] == 0
            assert isinstance(first_record["rms"], float)
            assert first_record["rms"] > 0


class TestScheduleLearningRate:
    """Test learning rate scheduling function."""

    def test_schedule_lr_delta_high_rms(self):
        """Test schedule with high RMS values."""
        from ao_shaping.optimizer.wf.rms_by_zernike import schedule_lr_delta

        lr, delta = schedule_lr_delta(rms=0.20)
        assert lr == 1
        assert delta == 1

        lr, delta = schedule_lr_delta(rms=0.16)
        assert lr == 1
        assert delta == 1

    def test_schedule_lr_delta_medium_rms(self):
        """Test schedule with medium RMS values."""
        from ao_shaping.optimizer.wf.rms_by_zernike import schedule_lr_delta

        lr, delta = schedule_lr_delta(rms=0.12)
        assert lr == 1
        assert delta == 0.5

        lr, delta = schedule_lr_delta(rms=0.11)
        assert lr == 0.8
        assert delta == 0.3

    def test_schedule_lr_delta_low_rms(self):
        """Test schedule with low RMS values."""
        from ao_shaping.optimizer.wf.rms_by_zernike import schedule_lr_delta

        lr, delta = schedule_lr_delta(rms=0.07)
        assert lr == 0.8
        assert delta == 0.3

        lr, delta = schedule_lr_delta(rms=0.05)
        assert lr == 0.5
        assert delta == 0.1


class TestConstants:
    """Test module constants."""

    def test_zernike_bounds(self):
        """Test Zernike coefficient bounds."""
        from ao_shaping.optimizer.wf.rms_by_zernike import (
            ZERNIKE_MIN,
            ZERNIKE_MAX,
        )

        assert ZERNIKE_MIN == -50.0
        assert ZERNIKE_MAX == 50.0

    def test_slm_wavelength_default(self):
        """Test default SLM wavelength."""
        from ao_shaping.optimizer.wf.rms_by_zernike import (
            SLM_WAVELENGTH_DEFAULT,
        )

        assert SLM_WAVELENGTH_DEFAULT == 532


class TestZernikeCoefficientHandling:
    """Test Zernike coefficient handling."""

    def test_init_z_as_none(self):
        """Test initialization with None."""
        from ao_shaping.optimizer.wf.rms_by_zernike import optimizer_rms
        from ao_shaping.utils.matrix_utils import calc_n_zernike_terms

        n_max = 4
        n_zernike = calc_n_zernike_terms(n_max)

        mock_slm = MagicMock()
        mock_slm.send_zernike.return_value = np.zeros((512, 512))
        mock_slm.wavelength = 1064

        mock_wfs = MagicMock()
        mock_wfs.get_wavefront.return_value = (
            np.zeros((64, 64)),
            {"rms": 0.15, "strehl": 0.8}
        )
        mock_wfs.take_image.return_value = None

        with patch(
            "ao_shaping.optimizer.wf.rms_by_zernike.ZernikeSLM",
            return_value=mock_slm,
        ), patch(
            "ao_shaping.optimizer.wf.rms_by_zernike.Thorlab_WFS",
            return_value=mock_wfs,
        ), patch(
            "ao_shaping.optimizer.wf.rms_by_zernike.tqdm"
        ):
            recorder = optimizer_rms(
                epochs=1,
                n_max=n_max,
                init_z=None,
            )

            # Should start from zeros
            first_coeffs = recorder.history[0]["_c"]
            assert len(first_coeffs) == n_zernike

    def test_init_z_as_dict(self):
        """Test initialization with dict {(n,m): value}."""
        from ao_shaping.optimizer.wf.rms_by_zernike import optimizer_rms

        mock_slm = MagicMock()
        mock_slm.send_zernike.return_value = np.zeros((512, 512))
        mock_slm.wavelength = 1064

        mock_wfs = MagicMock()
        mock_wfs.get_wavefront.return_value = (
            np.zeros((64, 64)),
            {"rms": 0.15, "strehl": 0.8}
        )
        mock_wfs.take_image.return_value = None

        init_dict = {(2, 0): 1.0, (2, -2): 0.5}

        with patch(
            "ao_shaping.optimizer.wf.rms_by_zernike.ZernikeSLM",
            return_value=mock_slm,
        ), patch(
            "ao_shaping.optimizer.wf.rms_by_zernike.Thorlab_WFS",
            return_value=mock_wfs,
        ), patch(
            "ao_shaping.optimizer.wf.rms_by_zernike.tqdm"
        ):
            recorder = optimizer_rms(
                epochs=1,
                n_max=4,
                init_z=init_dict,
            )

            # Coefficients should include the dict values
            coeffs = recorder.history[0]["_c"]
            # (2, 0) is at index 4, (2, -2) is at index 3
            assert coeffs[4] == 1.0
            assert coeffs[3] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])