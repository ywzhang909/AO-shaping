"""
DM Control - Python Native Implementation
变形镜控制 - 原生Python实现

控制26个R50电源控制器，每个控制器50通道，共1296个驱动器
IP范围: 192.168.0.101 - 192.168.0.126
端口: IP + 10100 (如 192.168.0.101:10101)

Protocol:
- 电压转换: value = (voltage + 20) / 20 / 3.4 / 3.3 * 65535
- 设置全部通道电压: 0xAA 0xBB 0x08 hv lv 0xCC 0xDD
- 设置单通道电压: 0xAA 0xBB 0x04 ch hv lv 0xCC 0xDD
- 批量设置50通道: 0xAA 0xBB 0x09 hv1 lv1 ... hv50 lv50 0xCC 0xDD
- 打开继电器: 0xAA 0xBB 0x06 0xCC 0xDD
- 关闭继电器: 0xAA 0xBB 0x07 0xCC 0xDD

Usage:
    from dm_control import DMController
    
    dm = DMController()
    dm.init()  # 连接所有控制器
    dm.set_voltage_all(50.0)  # 所有通道设为50V
    dm.set_actuator(1, 10.0)  # 1号驱动器设为10V
    dm.open_relay()  # 打开继电器
    dm.disconnect()  # 断开连接
"""

import socket
import struct
import threading
import time
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import IntEnum
from concurrent.futures import ThreadPoolExecutor, as_completed

__version__ = "1.0.0"


class ErrorCode(IntEnum):
    """错误码"""
    SUCCESS = 0
    NOT_INIT = -1
    CONNECT_ERROR = -2
    SEND_ERROR = -3
    INVALID_VOLTAGE = -4
    INVALID_CHANNEL = -5
    INVALID_ACTUATOR = -6
    NOT_CONNECTED = -7
    TIMEOUT = -8


# 常量
MAX_CONTROLLERS = 26
MAX_CHANNELS = 50
MAX_ACTUATORS = 1296
VOLTAGE_MIN = -20.0
VOLTAGE_MAX = 120.0
TIMEOUT_MS = 5000

# 默认IP地址
DEFAULT_IPS = [f"192.168.0.{100 + i}" for i in range(1, 27)]

# 命令协议
CMD_HEADER = bytes([0xAA, 0xBB])
CMD_FOOTER = bytes([0xCC, 0xDD])
CMD_SET_ALL_VOLTAGE = bytes([0x08])  # 设置全部通道电压
CMD_SET_CHANNEL_VOLTAGE = bytes([0x04])  # 设置单通道电压
CMD_SET_ALL_ARRAY = bytes([0x09])  # 批量设置50通道
CMD_RELAY_OPEN = bytes([0x06])  # 打开继电器
CMD_RELAY_CLOSE = bytes([0x07])  # 关闭继电器


@dataclass
class ActuatorMapping:
    """驱动器映射"""
    controller_id: int  # 1-26
    channel: int  # 0-49


class DMError(Exception):
    """DM控制异常"""
    def __init__(self, code: ErrorCode, message: str = ""):
        self.code = code
        self.message = message or self._get_default_message(code)
        super().__init__(f"[{code.value}] {self.message}")
    
    @staticmethod
    def _get_default_message(code: ErrorCode) -> str:
        messages = {
            ErrorCode.SUCCESS: "成功",
            ErrorCode.NOT_INIT: "系统未初始化",
            ErrorCode.CONNECT_ERROR: "连接失败",
            ErrorCode.SEND_ERROR: "发送命令失败",
            ErrorCode.INVALID_VOLTAGE: "电压值无效",
            ErrorCode.INVALID_CHANNEL: "通道号无效 (0-49)",
            ErrorCode.INVALID_ACTUATOR: "驱动器号无效 (1-1296)",
            ErrorCode.NOT_CONNECTED: "控制器未连接",
            ErrorCode.TIMEOUT: "连接超时",
        }
        return messages.get(code, "未知错误")


class R50Controller:
    """R50控制器封装"""
    
    def __init__(self, controller_id: int, ip: str, port: int):
        self.controller_id = controller_id
        self.ip = ip
        self.port = port
        self.socket: Optional[socket.socket] = None
        self._lock = threading.Lock()
    
    def connect(self, timeout: float = 5.0) -> bool:
        """连接到控制器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(timeout)
            self.socket.connect((self.ip, self.port))
            return True
        except (socket.timeout, socket.error) as e:
            self.disconnect()
            return False
    
    def disconnect(self):
        """断开连接"""
        with self._lock:
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass
                self.socket = None
    
    def is_connected(self) -> bool:
        """检查是否连接"""
        return self.socket is not None
    
    def send_command(self, data: bytes) -> bool:
        """发送命令"""
        with self._lock:
            if not self.socket:
                return False
            try:
                self.socket.sendall(data)
                return True
            except socket.error:
                return False
    
    def _convert_voltage(self, voltage: float) -> Tuple[int, int]:
        """转换电压到高低字节
        
        NOTE: This implementation uses consistent base-256 byte extraction:
            high = raw // 256, low = raw % 256
        This ensures high * 256 + low == raw (mathematically correct).
        
        The MATLAB reference (R50PowerV1.m) uses an inconsistent scheme:
            high = floor(value / 255), low = floor(mod(value, 256))
        This creates a non-injective mapping due to different base values.
        
        The main AO-shaping codebase preserves MATLAB's behavior for hardware
        compatibility, but this reference implementation follows the cleaner
        approach. Be aware of the discrepancy when comparing outputs.
        """
        value = (voltage + 20.0) / 20.0 / 3.4 / 3.3 * 65535.0
        raw = int(value + 0.5)
        high = raw // 256
        low = raw % 256
        return high, low
    
    def set_all_channel_voltage(self, voltage: float) -> bool:
        """设置所有通道电压"""
        hv, lv = self._convert_voltage(voltage)
        cmd = CMD_HEADER + CMD_SET_ALL_VOLTAGE + bytes([hv, lv]) + CMD_FOOTER
        return self.send_command(cmd)
    
    def set_channel_voltage(self, channel: int, voltage: float) -> bool:
        """设置单通道电压"""
        if channel < 0 or channel > 49:
            return False
        hv, lv = self._convert_voltage(voltage)
        cmd = CMD_HEADER + CMD_SET_CHANNEL_VOLTAGE + bytes([channel, hv, lv]) + CMD_FOOTER
        return self.send_command(cmd)
    
    def set_all_voltage_array(self, voltages: List[float]) -> bool:
        """批量设置50通道电压"""
        if len(voltages) != 50:
            return False
        
        cmd = CMD_HEADER + CMD_SET_ALL_ARRAY
        for v in voltages:
            v = max(-20.0, min(120.0, v))  # 限制范围
            hv, lv = self._convert_voltage(v)
            cmd += bytes([hv, lv])
        cmd += CMD_FOOTER
        return self.send_command(cmd)
    
    def set_relay(self, open: bool) -> bool:
        """设置继电器状态"""
        cmd = CMD_HEADER + (CMD_RELAY_OPEN if open else CMD_RELAY_CLOSE) + CMD_FOOTER
        return self.send_command(cmd)


class DMController:
    """
    变形镜控制器主类
    
    控制26个R50电源控制器，每个50通道，共1296个驱动器
    
    Attributes:
        initialized: 是否已初始化
        connected_count: 已连接控制器数量
    """
    
    def __init__(self):
        self._controllers: List[Optional[R50Controller]] = [None] * MAX_CONTROLLERS
        self._actuator_map: List[ActuatorMapping] = []
        self._initialized = False
        self._lock = threading.Lock()
        
        # 构建默认驱动器映射
        self._build_default_mapping()
    
    def _build_default_mapping(self):
        """构建默认驱动器映射 (36x36网格)"""
        self._actuator_map = []
        for i in range(MAX_ACTUATORS):
            row = i // 36
            col = i % 36
            
            # 将36x36分成16个区域
            region_x = col // 9
            region_y = row // 9
            region = region_y * 4 + region_x
            
            controller_id = (region % 16) + 1
            channel = (i % 50)
            
            self._actuator_map.append(ActuatorMapping(controller_id, channel))
    
    def _load_actuator_mapping_from_excel(self, excel_path: str):
        """
        从Excel文件加载驱动器映射
        Excel格式: page | volume | issue
        """
        try:
            # 尝试使用openpyxl (更推荐)
            try:
                import openpyxl
                wb = openpyxl.load_workbook(excel_path, data_only=True)
                ws = wb.active
                
                self._actuator_map = []
                for row in ws.iter_rows(min_row=2, values_only=True):
                    if row[0] is None:
                        break
                    page = int(row[0])
                    volume = int(row[1]) - 100  # MATLAB索引转换
                    issue = int(row[2])
                    
                    if 1 <= page <= MAX_ACTUATORS:
                        self._actuator_map.append(ActuatorMapping(volume, issue))
                
                # 填充缺失的映射
                while len(self._actuator_map) < MAX_ACTUATORS:
                    self._actuator_map.append(ActuatorMapping(1, 0))
                    
            except ImportError:
                # 回退到xlrd (旧版Excel)
                import xlrd
                wb = xlrd.open_workbook(excel_path)
                ws = wb.sheet_by_index(0)
                
                self._actuator_map = []
                for row_idx in range(1, ws.nrows):
                    page = int(ws.cell_value(row_idx, 0))
                    volume = int(ws.cell_value(row_idx, 1)) - 100
                    issue = int(ws.cell_value(row_idx, 2))
                    
                    if 1 <= page <= MAX_ACTUATORS:
                        self._actuator_map.append(ActuatorMapping(volume, issue))
                
                while len(self._actuator_map) < MAX_ACTUATORS:
                    self._actuator_map.append(ActuatorMapping(1, 0))
                    
        except Exception as e:
            print(f"加载Excel映射失败，使用默认映射: {e}")
            self._build_default_mapping()
    
    def _load_ip_from_file(self, file_path: str) -> List[str]:
        """从文件加载IP地址"""
        try:
            with open(file_path, 'r') as f:
                ips = []
                for line in f:
                    line = line.strip()
                    if line and '.' in line:  # IP格式
                        ips.append(line)
                return ips if ips else DEFAULT_IPS
        except:
            return DEFAULT_IPS
    
    def init(self, ip_file: Optional[str] = None, mapping_file: Optional[str] = None) -> ErrorCode:
        """
        初始化系统，连接所有控制器
        
        Args:
            ip_file: IP地址文件路径 (None使用默认)
            mapping_file: 驱动器映射Excel文件 (None使用默认)
        
        Returns:
            ErrorCode: 错误码
        """
        with self._lock:
            if self._initialized:
                return ErrorCode.SUCCESS
            
            # 加载映射
            if mapping_file:
                self._load_actuator_mapping_from_excel(mapping_file)
            else:
                self._build_default_mapping()
            
            # 加载IP
            ips = self._load_ip_from_file(ip_file) if ip_file else DEFAULT_IPS
            
            # 初始化控制器
            connected = 0
            for i in range(MAX_CONTROLLERS):
                ip = ips[i] if i < len(ips) else DEFAULT_IPS[i]
                port = 10000 + (i + 1)
                
                ctrl = R50Controller(i + 1, ip, port)
                if ctrl.connect():
                    connected += 1
                self._controllers[i] = ctrl
            
            if connected == 0:
                self._initialized = True  # 标记以便后续清理
                return ErrorCode.CONNECT_ERROR
            
            self._initialized = True
            return ErrorCode.SUCCESS
    
    def disconnect(self) -> ErrorCode:
        """断开所有连接并清理资源"""
        with self._lock:
            for ctrl in self._controllers:
                if ctrl:
                    ctrl.disconnect()
            self._initialized = False
            return ErrorCode.SUCCESS
    
    def is_connected(self) -> bool:
        """检查是否初始化"""
        return self._initialized
    
    @property
    def connected_count(self) -> int:
        """获取已连接控制器数量"""
        return sum(1 for c in self._controllers if c and c.is_connected())
    
    def _get_controller(self, controller_id: int) -> Optional[R50Controller]:
        """获取控制器实例"""
        if 1 <= controller_id <= MAX_CONTROLLERS:
            return self._controllers[controller_id - 1]
        return None
    
    def _validate_voltage(self, voltage: float, higher: float = 120.0, lower: float = -20.0) -> ErrorCode:
        """验证电压值"""
        if voltage > higher or voltage < lower:
            return ErrorCode.INVALID_VOLTAGE
        return ErrorCode.SUCCESS
    
    def set_voltage_all_controllers(self, voltage: float,
                                   higher: float = 120.0,
                                   lower: float = -20.0) -> ErrorCode:
        """
        设置所有控制器所有通道的电压 (异步并行)

        Args:
            voltage: 电压值
            higher: 上限
            lower: 下限

        Returns:
            ErrorCode: 错误码
        """
        if not self._initialized:
            return ErrorCode.NOT_INIT

        ret = self._validate_voltage(voltage, higher, lower)
        if ret != ErrorCode.SUCCESS:
            return ret

        # 异步并行发送电压到所有控制器
        def send_voltage(ctrl):
            if ctrl and ctrl.is_connected():
                ctrl.set_all_channel_voltage(voltage)

        with ThreadPoolExecutor(max_workers=MAX_CONTROLLERS) as executor:
            futures = [executor.submit(send_voltage, ctrl) for ctrl in self._controllers]
            # 等待所有发送完成
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    pass

        return ErrorCode.SUCCESS
    
    def set_actuator_voltage(self, actuator_number: int, voltage: float) -> ErrorCode:
        """
        设置单个驱动器的电压
        
        Args:
            actuator_number: 驱动器号 (1-1296)
            voltage: 电压值
        
        Returns:
            ErrorCode: 错误码
        """
        if not self._initialized:
            return ErrorCode.NOT_INIT
        
        if not (1 <= actuator_number <= MAX_ACTUATORS):
            return ErrorCode.INVALID_ACTUATOR
        
        ret = self._validate_voltage(voltage)
        if ret != ErrorCode.SUCCESS:
            return ret
        
        mapping = self._actuator_map[actuator_number - 1]
        ctrl = self._get_controller(mapping.controller_id)
        
        if not ctrl or not ctrl.is_connected():
            return ErrorCode.NOT_CONNECTED
        
        if ctrl.set_channel_voltage(mapping.channel, voltage):
            return ErrorCode.SUCCESS
        return ErrorCode.SEND_ERROR
    
    def set_controller_voltage(self, controller_id: int, voltage: float) -> ErrorCode:
        """
        设置某个控制器所有通道的电压
        
        Args:
            controller_id: 控制器ID (1-26)
            voltage: 电压值
        
        Returns:
            ErrorCode: 错误码
        """
        if not self._initialized:
            return ErrorCode.NOT_INIT
        
        ctrl = self._get_controller(controller_id)
        if not ctrl:
            return ErrorCode.CONNECT_ERROR
        
        ret = self._validate_voltage(voltage)
        if ret != ErrorCode.SUCCESS:
            return ret
        
        if not ctrl.is_connected():
            return ErrorCode.NOT_CONNECTED
        
        if ctrl.set_all_channel_voltage(voltage):
            return ErrorCode.SUCCESS
        return ErrorCode.SEND_ERROR
    
    def set_channel_all_controllers(self, channel: int, voltage: float) -> ErrorCode:
        """
        设置所有控制器的同一个通道电压 (异步并行)

        Args:
            channel: 通道号 (0-49)
            voltage: 电压值

        Returns:
            ErrorCode: 错误码
        """
        if not self._initialized:
            return ErrorCode.NOT_INIT

        if not (0 <= channel <= 49):
            return ErrorCode.INVALID_CHANNEL

        ret = self._validate_voltage(voltage)
        if ret != ErrorCode.SUCCESS:
            return ret

        # 异步并行发送
        def send_channel(ctrl):
            if ctrl and ctrl.is_connected():
                ctrl.set_channel_voltage(channel, voltage)

        with ThreadPoolExecutor(max_workers=MAX_CONTROLLERS) as executor:
            futures = [executor.submit(send_channel, ctrl) for ctrl in self._controllers]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    pass

        return ErrorCode.SUCCESS
    
    def control_relay(self, state: int) -> ErrorCode:
        """
        控制继电器
        
        Args:
            state: 1打开, 0关闭
        
        Returns:
            ErrorCode: 错误码
        """
        return self.open_relay() if state else self.close_relay()
    
    def open_relay(self) -> ErrorCode:
        """打开继电器 (异步并行)"""
        if not self._initialized:
            return ErrorCode.NOT_INIT

        def open_relay(ctrl):
            if ctrl and ctrl.is_connected():
                ctrl.set_relay(True)

        with ThreadPoolExecutor(max_workers=MAX_CONTROLLERS) as executor:
            futures = [executor.submit(open_relay, ctrl) for ctrl in self._controllers]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    pass

        return ErrorCode.SUCCESS
    
    def close_relay(self) -> ErrorCode:
        """关闭继电器 (异步并行)"""
        if not self._initialized:
            return ErrorCode.NOT_INIT

        def close_relay(ctrl):
            if ctrl and ctrl.is_connected():
                ctrl.set_relay(False)

        with ThreadPoolExecutor(max_workers=MAX_CONTROLLERS) as executor:
            futures = [executor.submit(close_relay, ctrl) for ctrl in self._controllers]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    pass

        return ErrorCode.SUCCESS
    
    def set_controller_voltage_array(self, controller_id: int, voltages: List[float]) -> ErrorCode:
        """
        批量设置控制器所有通道电压
        
        Args:
            controller_id: 控制器ID (1-26)
            voltages: 50个电压值数组
        
        Returns:
            ErrorCode: 错误码
        """
        if not self._initialized:
            return ErrorCode.NOT_INIT
        
        ctrl = self._get_controller(controller_id)
        if not ctrl:
            return ErrorCode.CONNECT_ERROR
        
        if not ctrl.is_connected():
            return ErrorCode.NOT_CONNECTED
        
        if ctrl.set_all_voltage_array(voltages):
            return ErrorCode.SUCCESS
        return ErrorCode.SEND_ERROR
    
    def init_all_actuators(self) -> ErrorCode:
        """初始化所有驱动器为0V"""
        return self.set_voltage_all_controllers(0.0)
    
    def get_actuator_mapping(self, actuator_number: int) -> Tuple[int, int]:
        """
        获取驱动器映射
        
        Args:
            actuator_number: 驱动器号 (1-1296)
        
        Returns:
            Tuple[controller_id, channel]
        """
        if 1 <= actuator_number <= MAX_ACTUATORS:
            mapping = self._actuator_map[actuator_number - 1]
            return mapping.controller_id, mapping.channel
        return -1, -1
    
    def get_controller_ip(self, controller_id: int) -> Optional[str]:
        """获取控制器IP地址"""
        ctrl = self._get_controller(controller_id)
        return ctrl.ip if ctrl else None
    
    def set_multiple_actuators(self, counts: List[int], voltages: List[float]) -> ErrorCode:
        """
        批量设置多个驱动器电压
        
        Args:
            counts: 驱动器号列表
            voltages: 电压值列表
        
        Returns:
            ErrorCode: 最后一个错误码
        """
        if not self._initialized:
            return ErrorCode.NOT_INIT
        
        if len(counts) != len(voltages):
            return ErrorCode.INVALID_ACTUATOR
        
        last_error = ErrorCode.SUCCESS
        for i, (act, volt) in enumerate(zip(counts, voltages)):
            ret = self.set_actuator_voltage(act, volt)
            if ret != ErrorCode.SUCCESS:
                last_error = ret
        
        return last_error
    
    def __enter__(self):
        """上下文管理器入口"""
        self.init()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.disconnect()
        return False
    
    def __del__(self):
        """析构函数"""
        try:
            self.disconnect()
        except:
            pass


# 便捷函数
_default_controller: Optional[DMController] = None


def init(ip_file: Optional[str] = None, mapping_file: Optional[str] = None) -> ErrorCode:
    """初始化默认控制器"""
    global _default_controller
    _default_controller = DMController()
    return _default_controller.init(ip_file, mapping_file)


def disconnect() -> ErrorCode:
    """断开默认控制器"""
    global _default_controller
    if _default_controller:
        return _default_controller.disconnect()
    return ErrorCode.SUCCESS


def set_voltage_all(voltage: float, higher: float = 120.0, lower: float = -20.0) -> ErrorCode:
    """设置所有控制器电压"""
    if _default_controller:
        return _default_controller.set_voltage_all_controllers(voltage, higher, lower)
    return ErrorCode.NOT_INIT


def set_actuator(actuator: int, voltage: float) -> ErrorCode:
    """设置单个驱动器电压"""
    if _default_controller:
        return _default_controller.set_actuator_voltage(actuator, voltage)
    return ErrorCode.NOT_INIT


def open_relay() -> ErrorCode:
    """打开继电器"""
    if _default_controller:
        return _default_controller.open_relay()
    return ErrorCode.NOT_INIT


def close_relay() -> ErrorCode:
    """关闭继电器"""
    if _default_controller:
        return _default_controller.close_relay()
    return ErrorCode.NOT_INIT


if __name__ == "__main__":
    # 测试代码
    print(f"DM Control Python v{__version__}")
    print("初始化控制器...")
    
    dm = DMController()
    ret = dm.init()
    
    if ret == ErrorCode.SUCCESS:
        print(f"已连接 {dm.connected_count} 个控制器")
        print(f"设置所有通道为0V...")
        dm.init_all_actuators()
        print("完成!")
        dm.disconnect()
    else:
        print(f"初始化失败: {ret}")