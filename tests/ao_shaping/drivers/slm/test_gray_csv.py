"""测试Santec SLM-200驱动的灰度CSV加载功能"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.modules["ao_shaping.drivers.slm.santec._slm_win"] = MagicMock()

from ao_shaping.drivers.slm.santec import Santec


@pytest.fixture(autouse=True)
def _ensure_mock_slm_sdk() -> None:
    with patch.dict(
        "sys.modules",
        {"ao_shaping.drivers.slm.santec._slm_win": MagicMock()},
        clear=False,
    ):
        yield


CSV_PATH = Path(r"C:\santec\SLM-200\Files\All 1023.csv")


class TestLoadGrayFromCSV:
    """测试 load_gray_from_csv 方法"""

    @pytest.mark.skipif(
        not CSV_PATH.exists(),
        reason=f"测试CSV文件不存在: {CSV_PATH}",
    )
    def test_load_gray_shape_and_dtype(self):
        """加载CSV灰度矩阵，验证形状和数据类型"""
        slm = Santec(slm_number=1)
        gray = slm.load_gray_from_csv(CSV_PATH)

        assert gray.shape == (1200, 1920)
        assert gray.dtype == np.uint16

    @pytest.mark.skipif(
        not CSV_PATH.exists(),
        reason=f"测试CSV文件不存在: {CSV_PATH}",
    )
    def test_load_gray_value_range(self):
        """灰度值应在 0~1023 范围内"""
        slm = Santec(slm_number=1)
        gray = slm.load_gray_from_csv(CSV_PATH)

        assert gray.min() >= 0
        assert gray.max() <= 1023

    @pytest.mark.skipif(
        not CSV_PATH.exists(),
        reason=f"测试CSV文件不存在: {CSV_PATH}",
    )
    def test_load_gray_dimensions(self):
        """验证维度: 行0~1199, 列0~1919"""
        slm = Santec(slm_number=1)
        gray = slm.load_gray_from_csv(CSV_PATH)

        assert gray.shape[0] == 1200  # 行: 0~1199
        assert gray.shape[1] == 1920  # 列: 0~1919

    def test_load_gray_file_not_found(self):
        """文件不存在时应抛出 FileNotFoundError"""
        slm = Santec(slm_number=1)
        with pytest.raises(FileNotFoundError):
            slm.load_gray_from_csv("nonexistent.csv")


class TestCsvToPhase:
    """测试 csv_to_phase 静态方法"""

    @pytest.mark.skipif(
        not CSV_PATH.exists(),
        reason=f"测试CSV文件不存在: {CSV_PATH}",
    )
    def test_csv_to_phase_shape(self):
        """灰度转弧度，相位数组形状应与原始数据一致"""
        slm = Santec(slm_number=1)
        gray = slm.load_gray_from_csv(CSV_PATH)
        phase_rad = Santec.csv_to_phase(CSV_PATH)

        assert phase_rad.shape == gray.shape

    def test_csv_to_phase_dtype(self):
        """csv_to_phase 返回 float64 弧度数组"""
        if not CSV_PATH.exists():
            pytest.skip(f"CSV文件不存在: {CSV_PATH}")
        phase_rad = Santec.csv_to_phase(CSV_PATH)
        assert phase_rad.dtype == np.float64

    @pytest.mark.skipif(
        not CSV_PATH.exists(),
        reason=f"测试CSV文件不存在: {CSV_PATH}",
    )
    def test_csv_to_phase_value_range(self):
        """弧度值应在 0~2π 范围内"""
        phase_rad = Santec.csv_to_phase(CSV_PATH)
        assert phase_rad.min() >= 0.0
        assert phase_rad.max() <= 2 * np.pi

    @pytest.mark.skipif(
        not CSV_PATH.exists(),
        reason=f"测试CSV文件不存在: {CSV_PATH}",
    )
    def test_csv_to_phase_roundtrip(self):
        """灰度→弧度→灰度 往返一致（无矫正/LUT/平移）"""
        slm = Santec(slm_number=1)
        gray_orig = slm.load_gray_from_csv(CSV_PATH)
        phase_rad = Santec.csv_to_phase(CSV_PATH)

        gray_roundtrip = slm.create_phase_from_array(phase_rad)

        np.testing.assert_array_equal(gray_orig, gray_roundtrip)

    def test_csv_to_phase_with_custom_max_gray(self):
        """使用自定义 max_grayscale 参数验证比例关系"""
        phase_default = Santec.csv_to_phase(CSV_PATH, max_grayscale=1023)
        phase_custom = Santec.csv_to_phase(CSV_PATH, max_grayscale=511)
        ratio = phase_custom.max() / phase_default.max()
        assert abs(ratio - 2.0) < 1e-2

    def test_csv_to_phase_file_not_found(self):
        """文件不存在时应抛出 FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            Santec.csv_to_phase("nonexistent.csv")


class TestGrayToPhasePipeline:
    """测试灰度CSV → 弧度 → create_phase_from_array 完整流程"""

    @pytest.mark.skipif(
        not CSV_PATH.exists(),
        reason=f"测试CSV文件不存在: {CSV_PATH}",
    )
    def test_full_pipeline(self):
        """完整流程: CSV → csv_to_phase → create_phase_from_array"""
        slm = Santec(slm_number=1)
        gray = slm.load_gray_from_csv(CSV_PATH)
        phase_rad = Santec.csv_to_phase(CSV_PATH)
        gray_from_phase = slm.create_phase_from_array(phase_rad)

        assert gray_from_phase.shape == gray.shape
        assert gray_from_phase.dtype == np.uint16
        np.testing.assert_array_equal(gray, gray_from_phase)
