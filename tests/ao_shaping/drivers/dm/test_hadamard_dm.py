import numpy as np
import pytest

from ao_shaping.drivers.dm.hadamard_dm import HadamardDM


class TestHadamardDM:
    """Tests for HadamardDM class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        hdm = HadamardDM()
        assert hdm.mode_order == 8
        assert hdm.resolution == (1920, 1080)
        assert hdm.bits == 10
        assert hdm.mask_type == "circular"
        assert not hdm.is_open
        assert hdm._safety_mode is True

    def test_safety_mode_off(self):
        """Test initialization with safety_mode=False."""
        hdm = HadamardDM(safety_mode=False)
        assert hdm._safety_mode is False

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        hdm = HadamardDM(
            mode_order=16,
            resolution=(100, 100),
            radius=0.5,
            bits=12,
            mask_type="rectangular",
        )
        assert hdm.mode_order == 16
        assert hdm.resolution == (100, 100)
        assert hdm.bits == 12
        assert hdm.mask_type == "rectangular"
        assert hdm.DM_NUM == 256  # 16x16

    def test_dm_num_property(self):
        """Test DM_NUM property."""
        hdm = HadamardDM(mode_order=4)
        assert hdm.DM_NUM == 16  # 4x4

    def test_generate_phase_shape(self):
        """Test generate_phase returns correct shape."""
        hdm = HadamardDM(resolution=(100, 100))
        coeffs = np.random.randn(64) * 0.1
        phase = hdm.generate_phase(coeffs)
        assert phase.shape == (100, 100)
        assert phase.dtype == np.float64

    def test_generate_phase_2pi_shape(self):
        """Test generate_phase_2pi returns correct shape."""
        hdm = HadamardDM(resolution=(100, 100))
        coeffs = np.random.randn(64) * 0.1
        phase = hdm.generate_phase_2pi(coeffs)
        assert phase.shape == (100, 100)
        assert phase.dtype == np.uint16

    def test_generate_phase_range(self):
        """Test generate_phase returns values in radians [0, 2π]."""
        hdm = HadamardDM(resolution=(100, 100))
        coeffs = np.random.randn(64) * 0.1
        phase = hdm.generate_phase(coeffs)
        assert np.all(phase >= 0)
        assert np.all(phase <= 2 * np.pi)

    def test_generate_phase_2pi_range(self):
        """Test generate_phase_2pi returns values in [0, 1023]."""
        hdm = HadamardDM(resolution=(100, 100), bits=10)
        coeffs = np.random.randn(64) * 0.1
        phase = hdm.generate_phase_2pi(coeffs)
        assert np.all(phase >= 0)
        assert np.all(phase <= 1023)

    def test_transform_array(self):
        """Test transform with numpy array."""
        hdm = HadamardDM(resolution=(100, 100))
        coeffs = np.zeros(64)
        coeffs[0] = 0.5
        result = hdm.transform(coeffs)
        assert result.shape == (100, 100)
        assert result.dtype == np.uint16

    def test_transform_invalid_type(self):
        """Test transform with invalid type raises ValueError."""
        hdm = HadamardDM()
        with pytest.raises(ValueError):
            hdm.transform("invalid")
        with pytest.raises(ValueError):
            hdm.transform(123)
        with pytest.raises(ValueError):
            hdm.transform([1, 2, 3])  # List should fail

    def test_phase_shortcuts_removed(self):
        """HadamardDM 不再自带 send / send_hadamard 薄包装。

        这两个方法过去只是 ``transform`` / ``generate_phase_2pi`` 的转发,
        而继承来的 :meth:`BaseDM.send` 是**电压**语义 —— 同名不同义。
        取相位图一律走 ``transform`` / ``generate_phase_2pi``。
        """
        assert "send" not in vars(HadamardDM)
        assert "send_hadamard" not in vars(HadamardDM)
        hdm = HadamardDM(resolution=(100, 100))
        coeffs = np.random.randn(64) * 0.1
        np.testing.assert_array_equal(
            hdm.transform(coeffs), hdm.generate_phase_2pi(coeffs)
        )

    def test_open_close(self):
        """Test open and close methods."""
        hdm = HadamardDM()
        assert not hdm.is_open
        hdm.open()
        assert hdm.is_open
        hdm.close()
        assert not hdm.is_open

    def test_context_manager(self):
        """Test context manager usage."""
        with HadamardDM() as hdm:
            assert hdm.is_open
        assert not hdm.is_open

    def test_get_actuator_positions_none(self):
        """Test get_actuator_positions before any command."""
        hdm = HadamardDM()
        positions = hdm.get_actuator_positions()
        assert len(positions) == 0

    def test_get_actuator_positions_after_transform(self):
        """Test get_actuator_positions after transform."""
        hdm = HadamardDM(resolution=(100, 100))
        coeffs = np.random.randn(64) * 0.1
        hdm.transform(coeffs)
        positions = hdm.get_actuator_positions()
        np.testing.assert_array_almost_equal(positions, coeffs)

    def test_get_phase_none(self):
        """Test get_phase before any command."""
        hdm = HadamardDM()
        assert hdm.get_phase() is None

    def test_get_phase_after_transform(self):
        """Test get_phase after transform."""
        hdm = HadamardDM(resolution=(100, 100))
        coeffs = np.random.randn(64) * 0.1
        hdm.transform(coeffs)
        phase = hdm.get_phase()
        assert phase is not None
        assert phase.shape == (100, 100)

    def test_is_connected(self):
        """Test is_connected method."""
        hdm = HadamardDM()
        assert not hdm.is_connected()
        hdm.open()
        assert hdm.is_connected()
        hdm.close()
        assert not hdm.is_connected()

    def test_get_hardware_info(self):
        """Test get_hardware_info method."""
        hdm = HadamardDM(mode_order=8, resolution=(1920, 1080), mask_type="circular")
        info = hdm.get_hardware_info()
        assert info["type"] == "HadamardDM"
        assert info["mode_order"] == 8
        assert info["n_modes"] == 64
        assert info["resolution"] == (1920, 1080)
        assert info["mask_type"] == "circular"
        assert info["bits"] == 10

    def test_repr(self):
        """Test __repr__ method."""
        hdm = HadamardDM(mode_order=8, resolution=(1920, 1080))
        repr_str = repr(hdm)
        assert "HadamardDM" in repr_str
        assert "mode_order=8" in repr_str
        assert "resolution=(1920, 1080)" in repr_str
