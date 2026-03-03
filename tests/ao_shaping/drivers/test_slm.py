"""Santec SLM-200 驱动测试模块

测试 SLM 驱动的基本功能，包括设备连接、波长设置、相位图显示等。
由于硬件依赖，使用 Mock 来模拟 SLM SDK。
"""

import pytest
import numpy as np

from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200, SantecSLM200Error


@pytest.fixture
def slm():
    """创建并返回一个已初始化的 SLM 实例"""
    return SantecSLM200(slm_number=1, wavelength=1064, phase_range=200)


@pytest.fixture
def open_slm(slm):
    """创建并返回一个已打开的 SLM 实例"""
    slm.open()
    return slm


class TestSLMInitialization:
    """测试 SLM 初始化功能"""

    def test_valid_slm_number(self):
        """测试有效的 SLM 编号"""
        for num in range(1, 9):
            slm = SantecSLM200(slm_number=num)
            assert slm.slm_number == num
            assert not slm.is_open

    def test_invalid_slm_number(self):
        """测试无效的 SLM 编号应抛出异常"""
        with pytest.raises(SantecSLM200Error, match="SLM编号必须在1-8之间"):
            SantecSLM200(slm_number=0)
        with pytest.raises(SantecSLM200Error, match="SLM编号必须在1-8之间"):
            SantecSLM200(slm_number=9)
        with pytest.raises(SantecSLM200Error, match="SLM编号必须在1-8之间"):
            SantecSLM200(slm_number=-1)

    def test_default_parameters(self):
        """测试默认参数"""
        slm = SantecSLM200()
        assert slm.slm_number == 1
        assert slm.wavelength == 1064
        assert slm.phase_range == 200
        assert slm.flags == 0  # 不使用 120Hz

    def test_120hz_flag(self):
        """测试 120Hz 刷新率标志"""
        slm = SantecSLM200(use_120hz=True)
        assert slm.flags == 1

    def test_custom_parameters(self):
        """测试自定义参数"""
        slm = SantecSLM200(
            slm_number=2,
            wavelength=532,
            phase_range=150,
            use_120hz=True
        )
        assert slm.slm_number == 2
        assert slm.wavelength == 532
        assert slm.phase_range == 150
        assert slm.flags == 1

    def test_str_representation_closed(self, slm):
        """测试字符串表示（未连接状态）"""
        repr_str = repr(slm)
        assert "编号=1" in repr_str
        assert "状态=未连接" in repr_str
        assert "波长=1064nm" in repr_str


class TestSLMOpenClose:
    """测试 SLM 打开和关闭功能"""

    def test_open_success(self, slm):
        """测试成功打开设备"""
        assert not slm.is_open
        slm.open()
        assert slm.is_open

    def test_open_already_open(self, open_slm):
        """测试重复打开已连接的设备"""
        open_slm.open()  # 再次打开不应报错
        assert open_slm.is_open

    def test_close_success(self, open_slm):
        """测试成功关闭设备"""
        assert open_slm.is_open
        open_slm.close()
        assert not open_slm.is_open

    def test_close_not_open(self, slm):
        """测试关闭未打开的设备"""
        slm.close()  # 不应抛出异常
        assert not slm.is_open

    def test_context_manager(self, slm):
        """测试上下文管理器"""
        with slm as s:
            assert s.is_open
            assert s == slm
        assert not slm.is_open

    def test_str_representation_open(self, open_slm):
        """测试字符串表示（已连接状态）"""
        repr_str = repr(open_slm)
        assert "编号=1" in repr_str
        assert "状态=已连接" in repr_str


class TestWavelengthSetting:
    """测试波长设置功能"""

    def test_set_wavelength_success(self, open_slm):
        """测试成功设置波长"""
        open_slm.set_wavelength(532, 150)
        assert open_slm.wavelength == 532
        assert open_slm.phase_range == 150

    def test_set_wavelength_without_save(self, open_slm):
        """测试设置波长但不保存"""
        open_slm.set_wavelength(800, 200, save_to_device=False)

    def test_set_wavelength_not_open(self, slm):
        """测试在未打开设备时设置波长"""
        with pytest.raises(RuntimeError, match="SLM设备未打开"):
            slm.set_wavelength(532, 200)


class TestPhaseWriting:
    """测试相位写入功能"""

    def test_write_phase_success(self, open_slm):
        """测试成功写入相位数据"""
        phase = np.zeros((1080, 1920), dtype=np.uint16)
        open_slm.write_phase(phase, memory_number=1)

    def test_write_phase_invalid_memory_number(self, open_slm):
        """测试无效的内存编号"""
        phase = np.zeros((1080, 1920), dtype=np.uint16)
        with pytest.raises(ValueError, match="内存编号必须在1-128之间"):
            open_slm.write_phase(phase, memory_number=0)
        with pytest.raises(ValueError, match="内存编号必须在1-128之间"):
            open_slm.write_phase(phase, memory_number=129)

    def test_write_phase_wrong_dtype(self, open_slm):
        """测试错误的数据类型"""
        phase = np.zeros((1080, 1920), dtype=np.float32)
        with pytest.raises(ValueError, match="相位数据类型必须是uint16"):
            open_slm.write_phase(phase)

    def test_write_phase_wrong_dimensions(self, open_slm):
        """测试错误的数据维度"""
        phase = np.zeros((1080,), dtype=np.uint16)
        with pytest.raises(ValueError, match="相位数据必须是2D数组"):
            open_slm.write_phase(phase)


class TestPhasePatternGeneration:
    """测试相位图案生成功能（参考 notebooks/slm_test.py）"""

    def test_create_phase_from_array_default(self, open_slm):
        """测试从弧度数组创建相位图（使用默认参数）"""
        # 创建 0-2π 的相位
        phase_rad = np.linspace(0, 2*np.pi, 100).reshape(10, 10)
        grayscale = open_slm.create_phase_from_array(phase_rad)

        assert grayscale.dtype == np.uint16
        assert grayscale.shape == (10, 10)
        assert np.all(grayscale >= 0)
        assert np.all(grayscale <= 1023)

    def test_create_phase_from_array_custom_max(self, open_slm):
        """测试从弧度数组创建相位图（自定义最大值）"""
        phase_rad = np.linspace(0, 2*np.pi, 100).reshape(10, 10)
        grayscale = open_slm.create_phase_from_array(phase_rad, max_grayscale=512)

        assert np.all(grayscale >= 0)
        assert np.all(grayscale <= 512)

    def test_create_phase_clip_values(self, open_slm):
        """测试灰度值裁剪到有效范围"""
        # 超出 2π 的相位值
        phase_rad = np.array([[0, 4*np.pi]])  # 2π 的 2 倍
        grayscale = open_slm.create_phase_from_array(phase_rad)
        assert np.all(grayscale <= 1023)

    def test_load_phase_from_csv(self, open_slm, tmp_path):
        """测试从 CSV 文件加载相位数据"""
        # 创建测试 CSV 文件
        csv_content = "Y/X,0,1,2\n"
        csv_content += "0,100,200,300\n"
        csv_content += "1,400,500,600\n"

        csv_file = tmp_path / "test_phase.csv"
        csv_file.write_text(csv_content)

        phase = open_slm.load_phase_from_csv(str(csv_file))

        assert phase.dtype == np.uint16
        assert phase.shape == (2, 3)
        assert phase[0, 0] == 100
        assert phase[1, 2] == 600

    def test_load_phase_from_csv_not_found(self, open_slm):
        """测试加载不存在的 CSV 文件"""
        with pytest.raises(FileNotFoundError):
            open_slm.load_phase_from_csv("/nonexistent/path.csv")


class TestPatternTypes:
    """测试不同类型的相位图案生成（参考 notebooks/slm_test.py）"""

    RESOLUTION = (1920, 1200)  # SLM 分辨率

    def generate_checkerboard(self, period: int = 100) -> np.ndarray:
        """生成棋盘格相位图案"""
        height, width = self.RESOLUTION[1], self.RESOLUTION[0]
        max_val = 1023

        y = np.arange(height) // period
        x = np.arange(width) // period
        X, Y = np.meshgrid(x, y)

        checker = (X + Y) % 2
        img = (checker * max_val).astype(np.uint16)
        return img

    def generate_blazed_grating(self, period: int = 100, direction: str = 'horizontal') -> np.ndarray:
        """生成闪耀光栅相位图案"""
        height, width = self.RESOLUTION[1], self.RESOLUTION[0]
        max_val = 1023

        if direction == 'horizontal':
            y = np.arange(height)
            grating = ((y % period) / period * max_val)
            img = np.tile(grating[:, np.newaxis], (1, width))
        else:
            x = np.arange(width)
            grating = ((x % period) / period * max_val)
            img = np.tile(grating[np.newaxis, :], (height, 1))

        return img.astype(np.uint16)

    def generate_binary_grating(self, b: int = 2, a: int = 3, direction: str = 'horizontal') -> np.ndarray:
        """生成二元光栅相位图案"""
        height, width = self.RESOLUTION[1], self.RESOLUTION[0]
        max_val = 511  # 半值对应 π

        if direction == 'horizontal':
            y = np.arange(height)
            grating = np.where(y % (a + b) < b, 0, max_val)
            img = np.tile(grating[:, np.newaxis], (1, width))
        else:
            x = np.arange(width)
            grating = np.where(x % (a + b) < b, 0, max_val)
            img = np.tile(grating[np.newaxis, :], (height, 1))

        return img.astype(np.uint16)

    def generate_focus(self, focal_length: float, wavelength: float = 532e-9, pixel_size: float = 8e-6) -> np.ndarray:
        """生成聚焦相位图案（抛物面）"""
        height, width = self.RESOLUTION[1], self.RESOLUTION[0]
        max_val = 1023

        x = np.arange(width) - width // 2
        y = np.arange(height) - height // 2
        X, Y = np.meshgrid(x, y)

        R2 = X**2 + Y**2
        phase = (np.pi / wavelength / focal_length) * (R2 * pixel_size**2)
        phase_wrapped = np.mod(phase, 2 * np.pi)
        img = (phase_wrapped / (2 * np.pi) * max_val).astype(np.uint16)

        return img

    def test_write_checkerboard_pattern(self, open_slm):
        """测试写入棋盘格相位图"""
        phase = self.generate_checkerboard(period=50)
        open_slm.write_phase(phase, memory_number=1)

    def test_write_blazed_grating_horizontal(self, open_slm):
        """测试写入水平闪耀光栅"""
        phase = self.generate_blazed_grating(period=20, direction='horizontal')
        open_slm.write_phase(phase, memory_number=2)
        
    def test_write_blazed_grating_vertical(self, open_slm):
        """测试写入垂直闪耀光栅"""
        phase = self.generate_blazed_grating(period=20, direction='vertical')
        open_slm.write_phase(phase, memory_number=3)

    def test_write_binary_grating(self, open_slm):
        """测试写入二元光栅"""
        phase = self.generate_binary_grating(b=5, a=10, direction='horizontal')
        open_slm.write_phase(phase, memory_number=4)

    def test_write_focus_pattern(self, open_slm):
        """测试写入聚焦相位图"""
        phase = self.generate_focus(focal_length=0.5)
        open_slm.write_phase(phase, memory_number=5)

    def test_full_pattern_workflow(self, open_slm):
        """测试完整的相位图案工作流程"""
        patterns = [
            ("checkerboard", self.generate_checkerboard(period=100), 1),
            ("blazed_h", self.generate_blazed_grating(period=20, direction='horizontal'), 2),
            ("blazed_v", self.generate_blazed_grating(period=20, direction='vertical'), 3),
            ("binary", self.generate_binary_grating(b=3, a=7, direction='horizontal'), 4),
            ("focus", self.generate_focus(focal_length=1.0), 5),
        ]

        for name, phase, mem_num in patterns:
            open_slm.write_phase(phase, memory_number=mem_num)
            open_slm.display_memory(mem_num)


class TestIntegration:
    """集成测试"""

    def test_full_workflow(self):
        """测试完整的工作流程"""
        # 模拟所有 SDK 调用成功

        with SantecSLM200(slm_number=1) as slm:
            # 设置波长
            slm.set_wavelength(1064, 200)

            # 生成并写入相位图案
            phase = np.zeros((1080, 1920), dtype=np.uint16)
            phase[500:580, 900:1020] = 511  # 添加一个中心图案
            slm.write_phase(phase, memory_number=1)

            # 显示相位图
            slm.display_memory(1)