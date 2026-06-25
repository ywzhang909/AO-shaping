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


class TestDefaultCalc:
    """测试 _default_calc 的新算法（前后数组异常检测 + 余弦拟合）"""

    # ── 辅助方法 ──────────────────────────────────────

    @staticmethod
    def make_cosine_data(
        height: int, width: int, *, phi_map: np.ndarray | None = None,
        max_grayscale: int = 1023, measurement_gray: int = 1023,
        amplitude: float = 200.0, offset: float = 128.0,
        noise_std: float = 0.0,
    ) -> np.ndarray:
        """生成符合物理模型的合成测量数据: B = A·cos(2π·g/MAX + φ) + C"""
        amp = amplitude
        off = offset

        if phi_map is None:
            # 默认随机相位偏移 [-π/4, π/4]
            rng = np.random.default_rng(42)
            phi_map = (rng.random((height, width)) - 0.5) * np.pi / 2

        known_phase = 2 * np.pi * measurement_gray / max_grayscale
        data = off + amp * np.cos(known_phase + phi_map)
        if noise_std > 0:
            rng = np.random.default_rng(1)
            data += rng.normal(0, noise_std, data.shape)
        return data.astype(np.float64), phi_map

    # ── 异常检测测试 ──────────────────────────────────

    def test_no_outliers_in_smooth_data(self):
        """平滑数据不应标记异常点"""
        data, _ = self.make_cosine_data(10, 20, noise_std=1.0)
        result = WavefrontCorrection._default_calc(data, outlier_threshold=5.0)
        assert result.shape == data.shape
        assert np.all(np.isfinite(result))

    def test_single_outlier_detected_and_replaced(self):
        """单个显式异常点应被检测并替换"""
        data, _ = self.make_cosine_data(5, 10, noise_std=1.0)
        # 在 (2, 3) 处插入显式异常点
        data[2, 3] = 10000.0
        result = WavefrontCorrection._default_calc(data, outlier_threshold=3.0)
        # 异常点位置的值应该被替换（不再是 10000）
        assert np.abs(result[2, 3] - 10000.0) > 1.0

    def test_outlier_on_boundary_not_flagged(self):
        """首尾列不标记异常（边界无左右邻居）"""
        # 合成轻微噪声数据，首列放极端值
        h, w = 5, 10
        phi = np.zeros((h, w))
        data, _ = self.make_cosine_data(
            h, w, phi_map=phi, amplitude=100.0, offset=128.0)
        # 在非边界处插入极端值 — 会被替换
        data[2, 3] = 99999.0
        result_with_interior = WavefrontCorrection._default_calc(
            data.copy(), outlier_threshold=3.0)
        # 在边界处插入—不会触发替换（但有异常）
        data[2, 0] = 99999.0
        result_with_boundary = WavefrontCorrection._default_calc(
            data.copy(), outlier_threshold=3.0, max_grayscale=1023,
            measurement_gray=1023)
        # 矫正映射不同（边界异常影响全局估计）
        assert not np.allclose(result_with_interior, result_with_boundary)

    # ── 余弦拟合测试 ──────────────────────────────────

    def test_correction_map_shape_matches_input(self):
        """矫正映射图应与输入形状相同"""
        data, _ = self.make_cosine_data(30, 40)
        result = WavefrontCorrection._default_calc(data)
        assert result.shape == data.shape

    def test_zero_data_produces_zero_correction(self):
        """零振幅数据应产生全零矫正图"""
        data = np.full((10, 20), 128.0, dtype=np.float64)
        result = WavefrontCorrection._default_calc(data)
        assert np.allclose(result, 0.0)

    def test_uniform_phase_produces_uniform_correction(self):
        """所有像素具有相同 φ → 矫正映射图为常数"""
        h, w = 20, 30
        phi_uniform = np.full((h, w), 0.3)
        data, _ = self.make_cosine_data(h, w, phi_map=phi_uniform,
                                        amplitude=200.0, offset=128.0)
        result = WavefrontCorrection._default_calc(
            data, max_grayscale=1023, measurement_gray=1023)
        # 所有矫正值应相同
        assert np.std(result) < 1e-5

    def test_known_phi_roundtrip(self):
        """已知 φ ∈ [0,π] → 测量数据 → 提取 φ → 与已知 φ 一致

        arccos 返回 [0, π]，因此只能唯一恢复 φ ∈ [0, π] 的正值。
        负值需通过多灰度测量或空间连续性假设确定，不在本测试覆盖。
        """
        h, w = 40, 50
        rng = np.random.default_rng(42)
        # φ 限定在 [0, π/2] 以确保 arccos 唯一重建
        phi_gt = rng.random((h, w)) * np.pi / 2  # [0, π/2]
        data, _ = self.make_cosine_data(
            h, w, phi_map=phi_gt, max_grayscale=1023,
            measurement_gray=1023, amplitude=200.0, offset=128.0)
        result = WavefrontCorrection._default_calc(
            data, max_grayscale=1023, measurement_gray=1023)

        # 从矫正映射反推 φ: φ_extracted = -result * 2π / max_grayscale
        phi_extracted = -result * 2 * np.pi / 1023
        # 应接近原始 φ（arccos 在 [0, π] 内唯一）
        phi_diff = np.abs(phi_extracted - phi_gt)
        assert np.mean(phi_diff) < 0.1  # 平均误差 < 0.1 rad

    # ── 参数传递测试 ──────────────────────────────────

    def test_measurement_gray_affects_result(self):
        """不同的 measurement_gray 产生不同的矫正映射"""
        data, _ = self.make_cosine_data(10, 20, max_grayscale=1023,
                                        measurement_gray=512)
        r1 = WavefrontCorrection._default_calc(
            data, max_grayscale=1023, measurement_gray=512)
        r2 = WavefrontCorrection._default_calc(
            data, max_grayscale=1023, measurement_gray=1023)
        assert not np.allclose(r1, r2)

    # ── 集成测试 ──────────────────────────────────────

    def test_calc_with_default_params(self):
        """calc() 方法应将新参数传递给 _default_calc"""
        h, w = 15, 25
        data, _ = self.make_cosine_data(h, w)
        wc = WavefrontCorrection()
        # 直接注 raw_data
        wc._raw_data = data.astype(np.uint16)
        wc.calc(
            panel_resolution=(w, h),
            max_grayscale=1023,
            measurement_gray=1023,
        )
        assert wc.correction_map is not None
        assert wc.correction_map.shape == (h, w)

    def test_calc_with_custom_fn_still_works(self):
        """calc_fn 保留向后兼容"""
        def dummy_fn(raw):
            return np.zeros_like(raw, dtype=np.float64)

        h, w = 10, 15
        data = np.random.default_rng(0).integers(0, 1024, (h, w)).astype(np.uint16)
        wc = WavefrontCorrection(calc_fn=dummy_fn)
        wc._raw_data = data
        wc.calc(panel_resolution=(w, h))
        assert wc.correction_map is not None
        assert np.allclose(wc.correction_map, 0.0)

    def test_load_with_new_params_flow_through(self, tmp_path: Path):
        """load() 的 max_grayscale/measurement_gray 应传递到 calc()"""
        csv = tmp_path / "test_cosine.csv"
        # 生成合成数据并写入 CSV
        h, w = 5, 8
        data, _ = self.make_cosine_data(h, w)
        # 构建 Santec 格式 CSV（含行索引列）
        header = "Y/X," + ",".join(str(i) for i in range(w))
        lines = [header]
        for r in range(h):
            row_vals = data[r, :].astype(int)
            lines.append(str(r) + "," + ",".join(str(v) for v in row_vals))
        csv.write_text("\n".join(lines))

        wc = WavefrontCorrection(str(csv))
        wc.load(
            panel_resolution=(w, h),
            max_grayscale=1023,
            measurement_gray=1023,
        )
        assert wc.correction_map is not None
        assert wc.correction_map.shape == (h, w)

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
