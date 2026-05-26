"""PyVISA 兼容的 SLM-200 驱动

结合 Santec SLM-200 SDK 和 PyVISA 接口，
允许通过 VISA 协议控制 SLM 设备。

注意：Santec SLM-200 使用专用 SDK (_slm_win)，
此模块提供 PyVISA 兼容的包装层，但底层仍依赖 SDK。
"""

import numpy as np
from loguru import logger

from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200, SantecSLM200Error
from ao_shaping.drivers.visa_base import VisaError


class SantecSLM200Visa:
    """PyVISA 兼容的 Santec SLM-200 包装器

    提供 PyVISA 风格的接口来控制 SLM-200，包括：
    - 资源名称支持 (如 'SLM::1::INSTR')
    - SCPI 风格命令接口
    - 标准 VISA 仪器方法

    注意：这是包装器而非原生 VISA 实现，底层仍使用 Santec SDK。

    Attributes:
        slm: 底层 SantecSLM200 实例
        resource_name: VISA 风格的资源名称
        slm_number: SLM 设备编号

    Example:
        >>> from ao_shaping.drivers.slm import SantecSLM200Visa
        >>>
        >>> # 使用 VISA 资源名称打开
        >>> with SantecSLM200Visa('SLM::1::INSTR') as slm:
        ...     # 使用 SCPI 风格命令
        ...     slm.write('WAVELENGTH 1064')
        ...     slm.write('PHASE:DISPLAY 1')
        ...
        ...     # 或使用高级 API
        ...     slm.write_phase(phase_data, memory_number=1)
        ...     slm.display_memory(1)
    """

    def __init__(
        self,
        resource_name: str,
        wavelength: int = 1064,
        use_120hz: bool = False,
        auto_open: bool = True,
    ):
        """初始化 SLM VISA 包装器

        Args:
            resource_name: VISA 资源名称 (格式: 'SLM::<number>::INSTR')
            wavelength: 工作波长（nm）
            use_120hz: 是否使用 120Hz 模式
            auto_open: 是否自动打开设备

        Raises:
            VisaError: 资源名称格式错误或 SDK 不可用
            SantecSLM200Error: 设备初始化失败
        """
        self.resource_name = resource_name
        self._slm_number = self._parse_resource_name(resource_name)
        self._wavelength = wavelength
        self._use_120hz = use_120hz
        self._is_open = False

        # 创建底层 SLM 实例
        self._slm = SantecSLM200(
            slm_number=self._slm_number,
            use_120hz=use_120hz,
            wavelength=wavelength,
        )

        if auto_open:
            self.open()

    def _parse_resource_name(self, resource_name: str) -> int:
        """解析 VISA 资源名称获取 SLM 编号

        支持的格式：
        - 'SLM::<number>::INSTR'
        - 'SLM::<number>'
        - '<number>' (纯数字)

        Args:
            resource_name: 资源名称

        Returns:
            SLM 编号 (1-8)

        Raises:
            VisaError: 格式无效
        """
        try:
            # 尝试纯数字
            if resource_name.isdigit():
                return int(resource_name)

            # 解析 VISA 格式
            if resource_name.startswith("SLM::"):
                parts = resource_name.split("::")
                if len(parts) >= 2:
                    return int(parts[1])

            raise ValueError(f"无法解析资源名称: {resource_name}")
        except (ValueError, IndexError) as e:
            raise VisaError(
                f"无效的 SLM 资源名称: {resource_name}. "
                f"支持的格式: 'SLM::<number>::INSTR' 或 '<number>'"
            ) from e

    @property
    def slm(self) -> SantecSLM200:
        """获取底层 SLM 实例"""
        return self._slm

    @property
    def slm_number(self) -> int:
        """获取 SLM 设备编号"""
        return self._slm_number

    @property
    def timeout(self) -> int:
        """获取/设置超时（模拟属性，实际由 SDK 控制）"""
        return 5000  # 默认 5 秒

    @timeout.setter
    def timeout(self, value: int) -> None:
        # SDK 不支持动态超时设置
        logger.debug(f"SLM SDK 不支持动态超时设置（请求: {value}ms）")

    def open(self) -> None:
        """打开 SLM 设备"""
        if self._is_open:
            return

        self._slm.open()
        self._is_open = True
        logger.info(f"VISA SLM {self.resource_name} 已打开")

    def close(self) -> None:
        """关闭 SLM 设备"""
        if not self._is_open:
            return

        self._slm.close()
        self._is_open = False
        logger.info(f"VISA SLM {self.resource_name} 已关闭")

    def write(self, command: str) -> None:
        """发送 SCPI 风格命令

        支持的命令：
        - '*IDN?' - 获取设备标识
        - '*RST' - 重置设备
        - 'WAVELENGTH <value>' - 设置波长
        - 'PHASE:RANGE <value>' - 设置相位范围
        - 'PHASE:DISPLAY <memory>' - 显示指定内存的相位图
        - 'GRAYSCALE <value>' - 设置灰度值

        Args:
            command: SCPI 命令字符串

        Raises:
            VisaError: 命令执行失败
        """
        command = command.strip().upper()

        try:
            if command == "*IDN?":
                # 查询命令，不执行操作
                return

            elif command == "*RST":
                # 重置 - 设置为均匀灰度
                self._slm.set_grayscale(0)

            elif command.startswith("WAVELENGTH"):
                parts = command.split()
                if len(parts) >= 2:
                    wavelength = int(parts[1])
                    self._slm.set_wavelength(wavelength)
                else:
                    raise VisaError("WAVELENGTH 命令需要参数")

            elif command.startswith("PHASE:RANGE"):
                # 相位范围已固定为 200 (2π)，此命令不再支持
                logger.warning("PHASE:RANGE 命令已不再支持，SLM 固定使用 2π 相位范围")

            elif command.startswith("PHASE:DISPLAY"):
                parts = command.split()
                if len(parts) >= 2:
                    memory = int(parts[1])
                    self._slm.display_memory(memory)
                else:
                    raise VisaError("PHASE:DISPLAY 命令需要内存编号")

            elif command.startswith("GRAYSCALE"):
                parts = command.split()
                if len(parts) >= 2:
                    gs = int(parts[1])
                    self._slm.set_grayscale(gs)
                else:
                    raise VisaError("GRAYSCALE 命令需要参数")

            else:
                raise VisaError(f"未知命令: {command}")

            logger.debug(f"VISA -> {command}")

        except SantecSLM200Error as e:
            raise VisaError(f"命令执行失败: {e}") from e

    def query(self, command: str) -> str:
        """发送查询命令

        支持的查询：
        - '*IDN?' - 设备标识
        - 'WAVELENGTH?' - 当前波长
        - 'PHASE:RANGE?' - 当前相位范围

        Args:
            command: SCPI 查询命令

        Returns:
            响应字符串
        """
        command = command.strip().upper()

        try:
            if command == "*IDN?":
                return f"Santec,SLM-200,{self._slm_number},SDK"

            elif command == "WAVELENGTH?":
                wl, _ = self._slm.get_wavelength_info()
                return str(wl)

            elif command == "PHASE:RANGE?":
                # 相位范围固定为 200 (2π)
                return "200"

            else:
                raise VisaError(f"未知查询: {command}")

        except SantecSLM200Error as e:
            raise VisaError(f"查询失败: {e}") from e

    def write_phase(self, phase: np.ndarray, memory_number: int = 1) -> None:
        """写入相位数据

        直接调用底层 SLM API，跳过 SCPI 解析。

        Args:
            phase: 相位数据数组 (uint16)
            memory_number: 内存位置 (1-128)
        """
        self._slm.write_phase(phase, memory_number)

    def display_memory(self, memory_number: int) -> None:
        """显示指定内存的相位图

        Args:
            memory_number: 内存位置 (1-128)
        """
        self._slm.display_memory(memory_number)

    def set_wavelength(self, wavelength: int) -> None:
        """设置波长

        Args:
            wavelength: 波长（nm）
        """
        self._slm.set_wavelength(wavelength)
        self._wavelength = wavelength

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self) -> str:
        status = "open" if self._is_open else "closed"
        return f"SantecSLM200Visa({self.resource_name}, {status})"


def create_slm_visa_instrument(resource_name: str, **kwargs) -> SantecSLM200Visa:
    """创建 SLM VISA 仪器的工厂函数

    Args:
        resource_name: VISA 资源名称
        **kwargs: 其他参数

    Returns:
        SantecSLM200Visa 实例
    """
    return SantecSLM200Visa(resource_name, **kwargs)
