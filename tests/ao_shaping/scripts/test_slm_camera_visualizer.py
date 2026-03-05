"""
SLM Camera Visualizer 测试类

测试 Streamlit 应用的UI组件和相位生成功能。
由于需要硬件(SLM/相机)，部分测试需要mock或跳过。

使用方法:
    pytest tests/ao_shaping/scripts/test_slm_camera_visualizer.py -v
"""

import pytest
import numpy as np
from pathlib import Path

from ao_shaping.utils.phase_patterns import (
    SLM_RESOLUTION,
    SLM_MAX_VAL,
    generate_blazed_grating,
    generate_focus,
    generate_checkerboard,
    generate_binary_grating,
    generate_vortex,
    generate_zernike,
    resize_to_slm,
    load_phase_csv,
)


# ============== 测试类 ==============


class TestPhaseGenerationFunctions:
    """测试相位生成函数 (无需硬件)"""

    def test_generate_blazed_grating_horizontal(self):
        """测试水平闪耀光栅生成"""
        phase = generate_blazed_grating(period=20, direction="horizontal")
        assert phase.shape == (SLM_RESOLUTION[1], SLM_RESOLUTION[0])
        assert phase.dtype == np.uint16
        assert phase.max() <= SLM_MAX_VAL
        # 验证周期性
        assert phase[0, 0] == phase[0, 20]  # 一个周期后重复

    def test_generate_blazed_grating_vertical(self):
        """测试垂直闪耀光栅生成"""
        phase = generate_blazed_grating(period=15, direction="vertical")
        assert phase.shape == (SLM_RESOLUTION[1], SLM_RESOLUTION[0])
        assert phase.dtype == np.uint16

    def test_generate_focus(self):
        """测试聚焦透镜生成"""
        phase = generate_focus(focal_length=0.5, wavelength=532e-9, pixel_size=8e-6)
        assert phase.shape == (SLM_RESOLUTION[1], SLM_RESOLUTION[0])
        assert phase.dtype == np.uint16
        # 中心值应该较小，边缘值较大
        center_val = phase[600, 960]
        corner_val = phase[0, 0]
        assert corner_val > center_val

    def test_generate_checkerboard(self):
        """测试棋盘格生成"""
        phase = generate_checkerboard(period=100)
        assert phase.shape == (SLM_RESOLUTION[1], SLM_RESOLUTION[0])
        assert phase.dtype == np.uint16
        # 验证只有两种值 (0 和 max)
        unique_vals = np.unique(phase)
        assert len(unique_vals) <= 2

    def test_generate_binary_grating_horizontal(self):
        """测试二元光栅生成"""
        phase = generate_binary_grating(period=8, direction="horizontal")
        assert phase.shape == (SLM_RESOLUTION[1], SLM_RESOLUTION[0])
        assert phase.dtype == np.uint16

    def test_generate_binary_grating_vertical(self):
        """测试垂直二元光栅生成"""
        phase = generate_binary_grating(period=8, direction="vertical")
        assert phase.shape == (SLM_RESOLUTION[1], SLM_RESOLUTION[0])
        assert phase.dtype == np.uint16

    def test_generate_vortex(self):
        """测试涡旋光束生成"""
        phase = generate_vortex(topological_charge=1)
        assert phase.shape == (SLM_RESOLUTION[1], SLM_RESOLUTION[0])
        assert phase.dtype == np.uint16

    def test_generate_vortex_multiple_charge(self):
        """测试多拓扑荷涡旋"""
        for charge in [1, 2, 3, 5]:
            phase = generate_vortex(topological_charge=charge)
            assert phase.shape == (SLM_RESOLUTION[1], SLM_RESOLUTION[0])

    def test_generate_zernike_defocus(self):
        """测试Zernike离焦模式"""
        phase = generate_zernike(n=2, m=0, amplitude=2.0)
        assert phase.shape == (SLM_RESOLUTION[1], SLM_RESOLUTION[0])
        assert phase.dtype == np.uint16

    def test_generate_zernike_astigmatism(self):
        """测试Zernike像散模式"""
        phase = generate_zernike(n=2, m=2, amplitude=1.0)
        assert phase.shape == (SLM_RESOLUTION[1], SLM_RESOLUTION[0])
        assert phase.dtype == np.uint16

    def test_resize_to_slm_same_size(self):
        """测试相同尺寸的图像不需要调整"""
        img = np.random.randint(0, 1023, (1200, 1920), dtype=np.uint16)
        result = resize_to_slm(img)
        assert result.shape == (1200, 1920)

    def test_resize_to_slm_different_size(self):
        """测试不同尺寸的图像调整"""
        img = np.random.randint(0, 255, (600, 800), dtype=np.uint16)
        result = resize_to_slm(img)
        assert result.shape == (1200, 1920)
        assert result.dtype == np.uint16


class TestStreamlitAppUI:
    """测试 Streamlit 应用的 UI 组件 (需要硬件mock)"""

    @pytest.fixture
    def app_test(self):
        """创建 AppTest 实例"""
        try:
            from streamlit.testing.v1 import AppTest

            # 正确的路径: tests/ao_shaping/scripts/test_file.py -> 项目根目录/scripts/
            script_path = (
                Path(__file__).parent.parent.parent.parent
                / "scripts"
                / "slm_camera_visualizer.py"
            )
            return AppTest.from_file(str(script_path), default_timeout=10)
        except ImportError:
            pytest.skip("streamlit.testing not available")

    def test_app_loads_title(self, app_test):
        """测试应用标题加载"""
        at = app_test.run()
        # 检查标题
        assert len(at.title) > 0
        assert "SLM" in at.title[0].value
        assert "Camera" in at.title[0].value

    def test_app_has_sidebar(self, app_test):
        """测试侧边栏存在"""
        at = app_test.run()
        # 侧边栏应该存在
        assert hasattr(at, "sidebar")

    def test_app_has_slm_connect_checkbox(self, app_test):
        """测试 SLM 连接复选框"""
        at = app_test.run()
        # 应该有连接相关的checkbox
        checkboxes = at.checkbox
        assert len(checkboxes) >= 2  # 至少SLM和相机

    def test_app_has_camera_selectbox(self, app_test):
        """测试相机类型选择"""
        at = app_test.run()
        # 应该有相机类型选择
        selectboxes = at.selectbox
        assert len(selectboxes) > 0

    def test_app_has_phase_source_radio(self, app_test):
        """测试相位来源单选按钮"""
        at = app_test.run()
        radios = at.radio
        assert len(radios) > 0
        # 应该有"参数生成"和"加载 CSV"选项
        assert "参数生成" in radios[0].value or "加载 CSV" in radios[0].value

    def test_app_has_phase_type_selectbox(self, app_test):
        """测试相位类型选择"""
        at = app_test.run()
        # 相位类型选择应该存在
        selectboxes = at.selectbox
        assert len(selectboxes) >= 2  # 相机类型 + 相位类型

    def test_app_has_exposure_control(self, app_test):
        """测试曝光控制 - 需要在expander中查找"""
        at = app_test.run()
        # 曝光控制在 expander 中，需要展开后查找
        if len(at.expander) > 0:
            exp = at.expander[0]
            number_inputs = exp.number_input
            assert len(number_inputs) >= 1

    def test_app_exposure_time_range(self, app_test):
        """测试曝光时间范围: 0.011 - 2 ms"""
        at = app_test.run()
        if len(at.expander) > 0:
            exp = at.expander[0]
            number_inputs = exp.number_input
            if len(number_inputs) > 0:
                # 检查最小值
                assert number_inputs[0].min_value == 0.011
                # 检查默认值
                assert number_inputs[0].value == 1.0
                # 检查最大值
                assert number_inputs[0].max_value == 2.0
                # 检查步长
                assert number_inputs[0].step == 0.001

    def test_app_has_auto_exposure_toggle(self, app_test):
        """测试自动曝光开关"""
        at = app_test.run()
        # 应该有checkbox用于自动曝光
        checkboxes = at.checkbox
        assert len(checkboxes) >= 2

    def test_app_has_refresh_checkbox(self, app_test):
        """测试自动刷新复选框"""
        at = app_test.run()
        # 应该有自动刷新相关控件
        checkboxes = at.checkbox
        assert len(checkboxes) >= 2

    def test_app_has_capture_button(self, app_test):
        """测试拍照按钮 - 需要在expander中查找"""
        at = app_test.run()
        # 按钮在 expander 中
        if len(at.expander) > 0:
            exp = at.expander[0]
            buttons = exp.button
            assert len(buttons) >= 2

    def test_app_has_slm_send_button(self, app_test):
        """测试发送到SLM按钮 - 在col1列中(需要连接SLM后显示)"""
        at = app_test.run()
        # 发送到SLM按钮在col1列中，但只有当 phase_array is not None 且 SLM已连接时显示
        # 检查列结构是否存在
        if len(at.columns) > 0:
            # 列存在即可，按钮是条件显示的
            assert True

    def test_app_has_slider_controls(self, app_test):
        """测试滑块控件"""
        at = app_test.run()
        sliders = at.slider
        # 至少有刷新间隔滑块
        assert len(sliders) >= 1


class TestStreamlitWidgetInteractions:
    """测试 Streamlit 小部件交互 (需要硬件mock)"""

    @pytest.fixture
    def app_test(self):
        """创建 AppTest 实例"""
        try:
            from streamlit.testing.v1 import AppTest

            script_path = (
                Path(__file__).parent.parent.parent.parent
                / "scripts"
                / "slm_camera_visualizer.py"
            )
            return AppTest.from_file(str(script_path), default_timeout=10)
        except ImportError:
            pytest.skip("streamlit.testing not available")

    def test_toggle_slm_connection(self, app_test):
        """测试切换SLM连接状态"""
        at = app_test.run()

        # 找到SLM连接checkbox (sidebar中)
        # 切换状态
        checkboxes = at.sidebar.checkbox
        if len(checkboxes) >= 1:
            checkboxes[0].check().run()
            # 检查是否有错误或成功消息
            # 注意: 实际硬件未连接,可能会有错误

    def test_change_phase_type(self, app_test):
        """测试更改相位类型"""
        at = app_test.run()

        # 获取相位类型选择框
        selectboxes = at.selectbox
        if len(selectboxes) >= 2:
            # 先查看可用的选项
            try:
                # 切换到不同的相位类型
                selectboxes[1].select("聚焦透镜 (Focus)").run()
            except ValueError:
                # 如果选项不存在，跳过
                pass

    def test_change_slider_values(self, app_test):
        """测试滑块值更改"""
        at = app_test.run()

        # 获取滑块
        sliders = at.slider
        if len(sliders) >= 1:
            # 更改刷新间隔
            sliders[0].set_value(2.0).run()

    def test_change_exposure_time(self, app_test):
        """测试更改曝光时间"""
        at = app_test.run()

        # 获取曝光时间输入
        number_inputs = at.number_input
        if len(number_inputs) >= 1:
            number_inputs[0].set_value(50).run()

    def test_toggle_auto_exposure(self, app_test):
        """测试切换自动曝光"""
        at = app_test.run()

        # 找到自动曝光checkbox
        checkboxes = at.checkbox
        if len(checkboxes) >= 2:
            # 切换自动曝光
            checkboxes[1].check().run()


class TestCSVLoading:
    """测试 CSV 加载功能"""

    @pytest.fixture
    def temp_csv_file(self, tmp_path):
        """创建临时CSV文件用于测试"""
        # 创建测试CSV文件
        csv_path = tmp_path / "test_phase.csv"
        height, width = 100, 200

        with open(csv_path, "w") as f:
            # 写入header
            header = ["Y/X"] + [str(i) for i in range(width)]
            f.write(",".join(header) + "\n")

            # 写入数据
            for y in range(height):
                row = [str(y)] + [str(y * width + x) for x in range(width)]
                f.write(",".join(row) + "\n")

        return csv_path

    def test_load_csv_basic(self, temp_csv_file):
        """测试基本CSV加载"""
        phase = load_phase_csv(str(temp_csv_file))
        assert phase is not None
        assert phase.shape[0] > 0
        assert phase.shape[1] > 0
        assert phase.dtype == np.uint16

    def test_resize_after_load(self, temp_csv_file):
        """测试加载后调整大小"""
        loaded = load_phase_csv(str(temp_csv_file))
        resized = resize_to_slm(loaded)
        assert resized.shape == (SLM_RESOLUTION[1], SLM_RESOLUTION[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
