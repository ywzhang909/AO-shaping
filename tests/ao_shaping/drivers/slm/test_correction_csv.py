"""测试Santec SLM-200驱动的误差矫正CSV载入功能"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# 在 import SantecSLM200 之前 mock SLM SDK 模块
# （_slm_win 仅在 Windows + SDK installed 时可用）
sys.modules["ao_shaping.drivers.slm._slm_win"] = MagicMock()

from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200
from ao_shaping.drivers.slm.wavefront_correction import WavefrontCorrection


# 确保所有测试都使用 mock 的 SLM SDK
@pytest.fixture(autouse=True)
def _ensure_mock_slm_sdk() -> None:
    with patch.dict(
        "sys.modules",
        {"ao_shaping.drivers.slm._slm_win": MagicMock()},
        clear=False,
    ):
        yield


class TestCorrectionCSVLoading:
    """测试误差矫正CSV载入功能"""

    # ── WavefrontCorrection + 初始化 ───────────────────

    def test_init_without_correction(self):
        """不指定 correction_csv_path → is_valid=False"""
        slm = SantecSLM200(slm_number=1)
        assert isinstance(slm._correction, WavefrontCorrection)
        assert not slm._correction.is_valid

    def test_init_with_nonexistent_path(self):
        """指定不存在的路径 → is_valid=False"""
        slm = SantecSLM200(slm_number=1, correction_csv_path="/nonexistent/path.csv")
        assert isinstance(slm._correction, WavefrontCorrection)
        assert not slm._correction.is_valid

    def test_init_with_empty_path(self):
        """correction_csv_path="" → is_valid=False"""
        slm = SantecSLM200(slm_number=1, correction_csv_path="")
        assert isinstance(slm._correction, WavefrontCorrection)
        assert not slm._correction.is_valid

    def test_init_with_existing_csv(self, tmp_path: Path):
        """指定存在的 CSV 路径 → WavefrontCorrection 实例（lazy load, raw_data=None）"""
        csv = tmp_path / "test.csv"
        csv.write_text("Y/X,0,1,2\n0,100,200,300\n1,400,500,600\n")

        slm = SantecSLM200(slm_number=1, correction_csv_path=str(csv))
        assert slm._correction.is_valid
        assert slm._correction.csv_path == Path(csv)
        assert slm._correction.raw_data is None  # 懒加载

    def test_wavefront_correction_standalone(self, tmp_path: Path):
        """单独测试 WavefrontCorrection 类"""
        csv = tmp_path / "test.csv"
        csv.write_text("Y/X,0,1,2\n0,100,200,300\n1,400,500,600\n")

        wc = WavefrontCorrection(str(csv))
        wc.load_csv()
        assert wc.raw_data is not None
        assert wc.raw_data.shape == (2, 3)
        assert wc.raw_data.dtype == np.uint16

    # ── Backward compatibility: 旧 wrapper ─────────────

    def test_load_phase_from_csv_backward_compat(self, tmp_path: Path):
        """load_phase_from_csv 仍可正常工作"""
        csv = tmp_path / "test.csv"
        csv.write_text("Y/X,0,1,2\n0,100,200,300\n1,400,500,600\n")

        slm = SantecSLM200(slm_number=1)
        data = slm.load_phase_from_csv(csv)
        assert data.shape == (2, 3)
        assert data.dtype == np.uint16

    def test_resize_to_panel_backward_compat(self):
        """_resize_to_panel 委托至 WavefrontCorrection.resize_to_panel"""
        data = np.array([[100, 200], [300, 400]], dtype=np.uint16)
        slm = SantecSLM200(slm_number=1)
        resized = slm._resize_to_panel(data)
        assert resized.shape == (1200, 1920)
        assert resized.dtype == np.float64
        # 验证右上角 2×2 区域保留原值
        assert resized[599, 959] == 100  # 居中后的位置

    # ── Shift 参数 ────────────────────────────────────

    def test_shift_parameters_set_in_init(self):
        """shift_x/y 在 __init__ 时立即生效"""
        slm_no = SantecSLM200(slm_number=1, shift_x=0, shift_y=0)
        assert slm_no.shift_x == 0
        assert slm_no.shift_y == 0

        slm_shift = SantecSLM200(slm_number=1, shift_x=10, shift_y=-5)
        assert slm_shift.shift_x == 10
        assert slm_shift.shift_y == -5

    def test_create_phase_with_shift_no_correction(self):
        """create_phase_from_array 在有 shift 无 correction 时正常工作"""
        phase_rad = np.linspace(0, 2 * np.pi, 100).reshape(10, 10)
        slm = SantecSLM200(slm_number=1, shift_x=5, shift_y=-3)
        grayscale = slm.create_phase_from_array(phase_rad)
        assert grayscale.dtype == np.uint16
        assert np.all(grayscale >= 0)
        assert np.all(grayscale <= 1023)
