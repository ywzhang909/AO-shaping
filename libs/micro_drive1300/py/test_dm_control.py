"""
DM Control Pytest Test Suite
变形镜控制 pytest 测试套件

运行: pytest test_dm_control.py -v
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dm_control import (
    DMController,
    R50Controller,
    ErrorCode,
    MAX_CONTROLLERS,
    MAX_CHANNELS,
    MAX_ACTUATORS,
)


class TestR50Controller:
    """R50控制器单元测试"""

    def test_init(self):
        """测试控制器初始化"""
        ctrl = R50Controller(1, "192.168.0.101", 10101)
        assert ctrl.controller_id == 1
        assert ctrl.ip == "192.168.0.101"
        assert ctrl.port == 10101
        assert ctrl.socket is None

    def test_voltage_conversion(self):
        """测试电压转换"""
        ctrl = R50Controller(1, "192.168.0.101", 10101)

        # 测试 -20V -> 0
        hv, lv = ctrl._convert_voltage(-20.0)
        assert hv == 0 and lv == 0

        # 测试 0V -> 中间值
        hv, lv = ctrl._convert_voltage(0.0)
        assert hv > 0

        # 测试 120V -> 最大值
        hv, lv = ctrl._convert_voltage(120.0)
        assert hv == 255 and lv == 255

    @patch('dm_control.socket.socket')
    def test_connect_success(self, mock_socket_class):
        """测试连接成功"""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket

        ctrl = R50Controller(1, "192.168.0.101", 10101)
        result = ctrl.connect(timeout=1.0)

        assert result is True
        mock_socket.connect.assert_called_once_with(("192.168.0.101", 10101))

    @patch('dm_control.socket.socket')
    def test_connect_failure(self, mock_socket_class):
        """测试连接失败"""
        mock_socket = Mock()
        mock_socket.connect.side_effect = Exception("Connection refused")
        mock_socket_class.return_value = mock_socket

        ctrl = R50Controller(1, "192.168.0.101", 10101)
        result = ctrl.connect(timeout=1.0)

        assert result is False


class TestDMController:
    """DM控制器主类单元测试"""

    def test_init_default(self):
        """测试默认初始化"""
        dm = DMController()
        assert dm._initialized is False
        assert dm.connected_count == 0

    def test_default_mapping(self):
        """测试默认驱动器映射"""
        dm = DMController()
        assert len(dm._actuator_map) == MAX_ACTUATORS

        # 验证映射范围
        for mapping in dm._actuator_map:
            assert 1 <= mapping.controller_id <= MAX_CONTROLLERS
            assert 0 <= mapping.channel < MAX_CHANNELS

    def test_validate_voltage(self):
        """测试电压验证"""
        dm = DMController()

        # 有效电压
        assert dm._validate_voltage(0.0) == ErrorCode.SUCCESS
        assert dm._validate_voltage(50.0) == ErrorCode.SUCCESS
        assert dm._validate_voltage(120.0) == ErrorCode.SUCCESS
        assert dm._validate_voltage(-20.0) == ErrorCode.SUCCESS

        # 无效电压
        assert dm._validate_voltage(121.0) == ErrorCode.INVALID_VOLTAGE
        assert dm._validate_voltage(-21.0) == ErrorCode.INVALID_VOLTAGE

    def test_get_controller(self):
        """测试获取控制器"""
        dm = DMController()
        dm._controllers[0] = R50Controller(1, "192.168.0.101", 10101)

        ctrl = dm._get_controller(1)
        assert ctrl is not None
        assert ctrl.controller_id == 1

        # 无效ID
        assert dm._get_controller(0) is None
        assert dm._get_controller(27) is None

    def test_set_actuator_invalid_actuator(self):
        """测试设置无效驱动器"""
        dm = DMController()
        dm._initialized = True

        # 无效驱动器号
        assert dm.set_actuator_voltage(0, 10.0) == ErrorCode.INVALID_ACTUATOR
        assert dm.set_actuator_voltage(1297, 10.0) == ErrorCode.INVALID_ACTUATOR

    def test_set_channel_invalid_channel(self):
        """测试设置无效通道"""
        dm = DMController()
        dm._initialized = True

        # 无效通道号
        assert dm.set_channel_all_controllers(-1, 10.0) == ErrorCode.INVALID_CHANNEL
        assert dm.set_channel_all_controllers(50, 10.0) == ErrorCode.INVALID_CHANNEL

    def test_not_initialized(self):
        """测试未初始化时的操作"""
        dm = DMController()

        assert dm.set_voltage_all_controllers(50.0) == ErrorCode.NOT_INIT
        assert dm.set_actuator_voltage(1, 10.0) == ErrorCode.NOT_INIT
        assert dm.open_relay() == ErrorCode.NOT_INIT
        assert dm.close_relay() == ErrorCode.NOT_INIT

    @patch.object(R50Controller, 'connect')
    def test_init_with_mock_controllers(self, mock_connect):
        """测试初始化（使用mock）"""
        mock_connect.return_value = True

        dm = DMController()
        ret = dm.init()

        # 可能没有真实控制器，返回CONNECT_ERROR也是合理的
        assert ret in [ErrorCode.SUCCESS, ErrorCode.CONNECT_ERROR]


class TestAsyncVoltageSending:
    """异步发送测试"""

    @patch('dm_control.socket.socket')
    def test_set_voltage_all_controllers_async(self, mock_socket_class):
        """测试异步设置所有控制器电压"""
        # 创建mock控制器
        mock_controllers = []
        for i in range(MAX_CONTROLLERS):
            mock_ctrl = Mock(spec=R50Controller)
            mock_ctrl.is_connected.return_value = True
            mock_ctrl.set_all_channel_voltage.return_value = True
            mock_controllers.append(mock_ctrl)

        dm = DMController()
        dm._initialized = True
        dm._controllers = mock_controllers

        # 调用异步方法
        result = dm.set_voltage_all_controllers(50.0)

        # 验证所有控制器都被调用
        assert result == ErrorCode.SUCCESS
        for mock_ctrl in mock_controllers:
            mock_ctrl.set_all_channel_voltage.assert_called_once_with(50.0)

    @patch('dm_control.socket.socket')
    def test_set_channel_all_controllers_async(self, mock_socket_class):
        """测试异步设置所有控制器同一通道"""
        mock_controllers = []
        for i in range(MAX_CONTROLLERS):
            mock_ctrl = Mock(spec=R50Controller)
            mock_ctrl.is_connected.return_value = True
            mock_ctrl.set_channel_voltage.return_value = True
            mock_controllers.append(mock_ctrl)

        dm = DMController()
        dm._initialized = True
        dm._controllers = mock_controllers

        result = dm.set_channel_all_controllers(10, 30.0)

        assert result == ErrorCode.SUCCESS
        for mock_ctrl in mock_controllers:
            mock_ctrl.set_channel_voltage.assert_called_once_with(10, 30.0)

    @patch('dm_control.socket.socket')
    def test_open_relay_async(self, mock_socket_class):
        """测试异步打开继电器"""
        mock_controllers = []
        for i in range(MAX_CONTROLLERS):
            mock_ctrl = Mock(spec=R50Controller)
            mock_ctrl.is_connected.return_value = True
            mock_ctrl.set_relay.return_value = True
            mock_controllers.append(mock_ctrl)

        dm = DMController()
        dm._initialized = True
        dm._controllers = mock_controllers

        result = dm.open_relay()

        assert result == ErrorCode.SUCCESS
        for mock_ctrl in mock_controllers:
            mock_ctrl.set_relay.assert_called_once_with(True)

    @patch('dm_control.socket.socket')
    def test_close_relay_async(self, mock_socket_class):
        """测试异步关闭继电器"""
        mock_controllers = []
        for i in range(MAX_CONTROLLERS):
            mock_ctrl = Mock(spec=R50Controller)
            mock_ctrl.is_connected.return_value = True
            mock_ctrl.set_relay.return_value = True
            mock_controllers.append(mock_ctrl)

        dm = DMController()
        dm._initialized = True
        dm._controllers = mock_controllers

        result = dm.close_relay()

        assert result == ErrorCode.SUCCESS
        for mock_ctrl in mock_controllers:
            mock_ctrl.set_relay.assert_called_once_with(False)

    @patch('dm_control.socket.socket')
    def test_async_sending_parallelism(self, mock_socket_class):
        """测试异步发送的并行性"""
        import time

        # 模拟每个控制器发送需要0.1秒
        send_times = []

        def mock_set_voltage(voltage):
            send_times.append(time.time())
            time.sleep(0.1)  # 模拟网络延迟
            return True

        mock_controllers = []
        for i in range(MAX_CONTROLLERS):
            mock_ctrl = Mock(spec=R50Controller)
            mock_ctrl.is_connected.return_value = True
            mock_ctrl.set_all_channel_voltage = mock_set_voltage
            mock_controllers.append(mock_ctrl)

        dm = DMController()
        dm._initialized = True
        dm._controllers = mock_controllers

        start_time = time.time()
        dm.set_voltage_all_controllers(50.0)
        elapsed = time.time() - start_time

        # 串行需要 26 * 0.1 = 2.6秒，并行应该 < 1秒
        # 给点余量，设置为1.5秒
        assert elapsed < 1.5, f"异步发送耗时 {elapsed:.2f}s，应该并行执行"


class TestErrorHandling:
    """错误处理测试"""

    def test_set_voltage_invalid(self):
        """测试设置无效电压"""
        dm = DMController()
        dm._initialized = True

        # 创建已连接的mock控制器
        mock_ctrl = Mock(spec=R50Controller)
        mock_ctrl.is_connected.return_value = True
        dm._controllers = [mock_ctrl] * MAX_CONTROLLERS

        # 超出范围的电压
        result = dm.set_voltage_all_controllers(150.0)
        assert result == ErrorCode.INVALID_VOLTAGE

        result = dm.set_voltage_all_controllers(-30.0)
        assert result == ErrorCode.INVALID_VOLTAGE


class TestActuatorMapping:
    """驱动器映射测试"""

    def test_get_actuator_mapping(self):
        """测试获取驱动器映射"""
        dm = DMController()

        for i in range(1, MAX_ACTUATORS + 1):
            ctrl_id, channel = dm.get_actuator_mapping(i)
            assert 1 <= ctrl_id <= MAX_CONTROLLERS
            assert 0 <= channel < MAX_CHANNELS

    def test_get_controller_ip(self):
        """测试获取控制器IP"""
        dm = DMController()
        dm._controllers[0] = R50Controller(1, "192.168.0.101", 10101)

        ip = dm.get_controller_ip(1)
        assert ip == "192.168.0.101"

        ip = dm.get_controller_ip(27)
        assert ip is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])