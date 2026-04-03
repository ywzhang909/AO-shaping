"""PyVISA 兼容层 - 统一仪器控制接口

提供基于 PyVISA 的仪器控制基础架构，使各类硬件设备可以通过
标准的 VISA 接口进行通信（GPIB、USB、Serial、Ethernet）。

这个模块提供：
- VisaInstrument: PyVISA 仪器基类
- VisaResourceManager: 资源管理器封装
- 与现有驱动系统的集成支持

Example:
    >>> from ao_shaping.drivers.visa_base import VisaInstrument, VisaResourceManager
    >>> 
    >>> # 列出所有可用资源
    >>> rm = VisaResourceManager()
    >>> print(rm.list_resources())
    ('USB0::0x1234::0x5678::SN001::INSTR', 'GPIB0::12::INSTR')
    >>> 
    >>> # 连接到仪器
    >>> with VisaInstrument('USB0::0x1234::0x5678::SN001::INSTR') as inst:
    ...     print(inst.query('*IDN?'))
    ...     inst.write('VOLT 12.0')
"""

from typing import Optional, Union, List, Dict, Any, Callable
from pathlib import Path
import contextlib
from loguru import logger

# PyVISA 可选导入
try:
    import pyvisa
    from pyvisa import ResourceManager, constants
    from pyvisa.resources import Resource, MessageBasedResource
    PYVISA_AVAILABLE = True
except ImportError:
    PYVISA_AVAILABLE = False
    pyvisa = None
    ResourceManager = None
    constants = None
    Resource = None
    MessageBasedResource = None


class VisaError(Exception):
    """VISA 通信错误"""
    pass


class VisaResourceManager:
    """PyVISA 资源管理器封装
    
    提供统一的资源管理接口，支持：
    - 资源发现和列举
    - 仪器连接管理
    - 多后端支持 (NI-VISA, pyvisa-py)
    
    Attributes:
        backend: 使用的 VISA 后端 ('@ni' 表示 NI-VISA, '@py' 表示 pyvisa-py)
        resource_manager: PyVISA ResourceManager 实例
    
    Example:
        >>> rm = VisaResourceManager()  # 自动检测后端
        >>> resources = rm.list_resources()
        >>> print(resources)
        ('USB0::0x1234::0x5678::SN001::INSTR',)
        >>> 
        >>> # 使用特定后端
        >>> rm_py = VisaResourceManager(backend='@py')  # 使用纯 Python 后端
    """
    
    def __init__(self, backend: Optional[str] = None, library_path: Optional[str] = None):
        """初始化资源管理器
        
        Args:
            backend: VISA 后端 ('@ni', '@py', '@sim')，None表示自动检测
            library_path: VISA 库文件路径，用于手动指定
        
        Raises:
            VisaError: PyVISA 未安装或无法初始化
        """
        if not PYVISA_AVAILABLE:
            raise VisaError(
                "PyVISA 未安装。请运行: uv add pyvisa "
                "或 pip install pyvisa"
            )
        
        self.backend = backend
        self._library_path = library_path
        self._rm: Optional[ResourceManager] = None
        
        try:
            if library_path:
                self._rm = ResourceManager(library_path)
            elif backend:
                self._rm = ResourceManager(backend)
            else:
                # 尝试自动检测
                try:
                    self._rm = ResourceManager()
                    self.backend = '@ni'  # 默认使用 NI-VISA
                except Exception:
                    # 回退到 pyvisa-py
                    self._rm = ResourceManager('@py')
                    self.backend = '@py'
            
            logger.info(f"VISA 资源管理器已初始化，后端: {self.backend}")
        except Exception as e:
            raise VisaError(f"无法初始化 VISA 资源管理器: {e}")
    
    @property
    def resource_manager(self) -> ResourceManager:
        """获取底层 ResourceManager 实例"""
        if self._rm is None:
            raise VisaError("资源管理器未初始化")
        return self._rm
    
    def list_resources(self, query: str = '?*::INSTR') -> tuple:
        """列出可用的 VISA 资源
        
        Args:
            query: 资源查询表达式，默认 '?*::INSTR' 表示所有仪器
        
        Returns:
            资源名称元组
        """
        try:
            return self._rm.list_resources(query)
        except Exception as e:
            logger.error(f"列举资源失败: {e}")
            return ()
    
    def list_resources_info(self, query: str = '?*::INSTR') -> Dict[str, Any]:
        """获取详细的资源信息
        
        Args:
            query: 资源查询表达式
        
        Returns:
            资源名称到 ResourceInfo 的映射
        """
        try:
            return self._rm.list_resources_info(query)
        except Exception as e:
            logger.error(f"获取资源信息失败: {e}")
            return {}
    
    def open_resource(
        self,
        resource_name: str,
        timeout: int = 5000,
        read_termination: Optional[str] = '\n',
        write_termination: Optional[str] = '\n',
        **kwargs
    ) -> Resource:
        """打开 VISA 资源
        
        Args:
            resource_name: VISA 资源名称
            timeout: 超时时间（毫秒）
            read_termination: 读取终止符
            write_termination: 写入终止符
            **kwargs: 其他资源特定参数
        
        Returns:
            Resource 实例
        
        Raises:
            VisaError: 打开资源失败
        """
        try:
            resource = self._rm.open_resource(
                resource_name,
                timeout=timeout,
                **kwargs
            )
            
            # 配置消息基资源
            if isinstance(resource, MessageBasedResource):
                if read_termination is not None:
                    resource.read_termination = read_termination
                if write_termination is not None:
                    resource.write_termination = write_termination
            
            logger.info(f"已打开资源: {resource_name}")
            return resource
        except Exception as e:
            raise VisaError(f"无法打开资源 {resource_name}: {e}")
    
    def open_instrument(
        self,
        resource_name: str,
        auto_init: bool = True,
        **kwargs
    ) -> 'VisaInstrument':
        """打开仪器并返回 VisaInstrument 包装
        
        Args:
            resource_name: VISA 资源名称
            auto_init: 是否自动初始化仪器
            **kwargs: 传递给 open_resource 的参数
        
        Returns:
            VisaInstrument 实例
        """
        resource = self.open_resource(resource_name, **kwargs)
        return VisaInstrument(resource, auto_init=auto_init)
    
    def find_instruments(
        self,
        manufacturer_id: Optional[str] = None,
        model_code: Optional[str] = None,
        serial_number: Optional[str] = None
    ) -> List[str]:
        """根据条件查找仪器
        
        Args:
            manufacturer_id: 制造商ID (如 '0x1234')
            model_code: 型号代码 (如 '0x5678')
            serial_number: 序列号
        
        Returns:
            匹配的资源名称列表
        """
        resources = self.list_resources()
        matches = []
        
        for resource in resources:
            # USB 资源格式: USB[board]::manufacturer_id::model_code::serial_number[::interface]::INSTR
            if 'USB' in resource:
                parts = resource.split('::')
                match = True
                
                if manufacturer_id and len(parts) > 1:
                    match = match and manufacturer_id.lower() in parts[1].lower()
                if model_code and len(parts) > 2:
                    match = match and model_code.lower() in parts[2].lower()
                if serial_number and len(parts) > 3:
                    match = match and serial_number.lower() in parts[3].lower()
                
                if match:
                    matches.append(resource)
            else:
                matches.append(resource)
        
        return matches
    
    def close(self) -> None:
        """关闭资源管理器"""
        if self._rm:
            try:
                self._rm.close()
                logger.info("VISA 资源管理器已关闭")
            except Exception as e:
                logger.error(f"关闭资源管理器时出错: {e}")
            finally:
                self._rm = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __repr__(self) -> str:
        if self._rm:
            resources = self.list_resources()
            return f"VisaResourceManager(backend={self.backend}, resources={len(resources)})"
        return "VisaResourceManager(closed)"


class VisaInstrument:
    """PyVISA 仪器封装类
    
    为 VISA 仪器提供高级接口，支持：
    - SCPI 命令发送和查询
    - 二进制数据传输
    - 自动重连和错误恢复
    - 上下文管理器支持
    
    Attributes:
        resource: PyVISA Resource 实例
        resource_name: 资源名称
        timeout: 通信超时（毫秒）
    
    Example:
        >>> with VisaInstrument('GPIB0::12::INSTR') as inst:
        ...     # 标准 SCPI 查询
        ...     idn = inst.query('*IDN?')
        ...     
        ...     # 设置参数
        ...     inst.write('VOLT 12.0')
        ...     
        ...     # 读取测量值
        ...     voltage = inst.query_float('MEAS:VOLT?')
    """
    
    def __init__(
        self,
        resource: Union[str, Resource],
        auto_init: bool = True,
        resource_manager: Optional[VisaResourceManager] = None,
        **resource_kwargs
    ):
        """初始化仪器
        
        Args:
            resource: 资源名称字符串或 Resource 实例
            auto_init: 是否自动初始化（发送 *RST 和 *CLS）
            resource_manager: 可选的资源管理器实例
            **resource_kwargs: 传递给 open_resource 的参数
        """
        if not PYVISA_AVAILABLE:
            raise VisaError("PyVISA 未安装")
        
        self._resource_manager = resource_manager
        self._own_resource_manager = False
        self._resource: Optional[Resource] = None
        self._resource_name: Optional[str] = None
        
        if isinstance(resource, str):
            # 从资源名称打开
            self._resource_name = resource
            if resource_manager is None:
                self._resource_manager = VisaResourceManager()
                self._own_resource_manager = True
            self._resource = self._resource_manager.open_resource(resource, **resource_kwargs)
        elif isinstance(resource, Resource):
            # 使用现有的 Resource 实例
            self._resource = resource
            self._resource_name = resource.resource_name
        else:
            raise ValueError(f"不支持的资源类型: {type(resource)}")
        
        self._auto_init = auto_init
        self._is_open = True
        
        if auto_init:
            self.initialize()
    
    @property
    def resource(self) -> Resource:
        """获取底层 Resource 实例"""
        if self._resource is None:
            raise VisaError("仪器未连接")
        return self._resource
    
    @property
    def resource_name(self) -> str:
        """获取资源名称"""
        return self._resource_name or "unknown"
    
    @property
    def timeout(self) -> int:
        """获取/设置超时时间（毫秒）"""
        return self.resource.timeout
    
    @timeout.setter
    def timeout(self, value: int) -> None:
        self.resource.timeout = value
    
    def initialize(self) -> None:
        """初始化仪器（清除状态和错误）"""
        try:
            # 清除状态
            self.write('*CLS')
            # 重置仪器（可选）
            # self.write('*RST')
            logger.debug(f"仪器 {self.resource_name} 已初始化")
        except Exception as e:
            logger.warning(f"初始化仪器时出错: {e}")
    
    def write(self, command: str) -> None:
        """发送 SCPI 命令
        
        Args:
            command: SCPI 命令字符串
        
        Raises:
            VisaError: 通信错误
        """
        try:
            self.resource.write(command)
            logger.debug(f"-> {command}")
        except Exception as e:
            raise VisaError(f"写入命令失败 '{command}': {e}")
    
    def read(self) -> str:
        """读取仪器响应
        
        Returns:
            响应字符串
        
        Raises:
            VisaError: 通信错误
        """
        try:
            response = self.resource.read()
            logger.debug(f"<- {response.strip()}")
            return response
        except Exception as e:
            raise VisaError(f"读取响应失败: {e}")
    
    def query(self, command: str) -> str:
        """发送查询命令并读取响应
        
        Args:
            command: SCPI 查询命令
        
        Returns:
            响应字符串
        """
        try:
            response = self.resource.query(command)
            logger.debug(f"-> {command}")
            logger.debug(f"<- {response.strip()}")
            return response.strip()
        except Exception as e:
            raise VisaError(f"查询失败 '{command}': {e}")
    
    def query_float(self, command: str) -> float:
        """查询并解析为浮点数
        
        Args:
            command: SCPI 查询命令
        
        Returns:
            浮点数值
        """
        response = self.query(command)
        try:
            return float(response)
        except ValueError:
            raise VisaError(f"无法将响应解析为浮点数: {response}")
    
    def query_int(self, command: str) -> int:
        """查询并解析为整数
        
        Args:
            command: SCPI 查询命令
        
        Returns:
            整数值
        """
        response = self.query(command)
        try:
            return int(response)
        except ValueError:
            raise VisaError(f"无法将响应解析为整数: {response}")
    
    def query_binary(
        self,
        command: str,
        datatype: str = 'f',
        is_big_endian: bool = False
    ) -> bytes:
        """查询二进制数据
        
        Args:
            command: SCPI 查询命令
            datatype: 数据类型 ('b', 'h', 'i', 'f', 'd')
            is_big_endian: 是否大端字节序
        
        Returns:
            二进制数据
        """
        try:
            return self.resource.query_binary_values(
                command,
                datatype=datatype,
                is_big_endian=is_big_endian
            )
        except Exception as e:
            raise VisaError(f"二进制查询失败 '{command}': {e}")
    
    def read_raw(self) -> bytes:
        """读取原始字节数据
        
        Returns:
            原始字节数据
        """
        try:
            return self.resource.read_raw()
        except Exception as e:
            raise VisaError(f"原始读取失败: {e}")
    
    def write_raw(self, data: bytes) -> None:
        """写入原始字节数据
        
        Args:
            data: 原始字节数据
        """
        try:
            self.resource.write_raw(data)
        except Exception as e:
            raise VisaError(f"原始写入失败: {e}")
    
    def clear(self) -> None:
        """清除仪器状态"""
        try:
            self.resource.clear()
            logger.debug(f"仪器 {self.resource_name} 已清除")
        except Exception as e:
            logger.warning(f"清除仪器时出错: {e}")
    
    def get_idn(self) -> str:
        """获取仪器标识 (*IDN?)
        
        Returns:
            仪器标识字符串
        """
        return self.query('*IDN?')
    
    def reset(self) -> None:
        """重置仪器 (*RST)"""
        self.write('*RST')
        logger.info(f"仪器 {self.resource_name} 已重置")
    
    def wait_for_operation_complete(self, timeout: Optional[int] = None) -> bool:
        """等待操作完成 (*OPC?)
        
        Args:
            timeout: 超时时间（毫秒），None表示使用默认超时
        
        Returns:
            是否成功完成
        """
        old_timeout = self.timeout
        if timeout:
            self.timeout = timeout
        
        try:
            response = self.query('*OPC?')
            return response.strip() == '1'
        except Exception as e:
            logger.warning(f"等待操作完成时出错: {e}")
            return False
        finally:
            if timeout:
                self.timeout = old_timeout
    
    def close(self) -> None:
        """关闭仪器连接"""
        if not self._is_open:
            return
        
        if self._resource:
            try:
                self._resource.close()
                logger.info(f"仪器 {self.resource_name} 已关闭")
            except Exception as e:
                logger.error(f"关闭仪器时出错: {e}")
            finally:
                self._resource = None
        
        # 如果资源管理器是我们创建的，也关闭它
        if self._own_resource_manager and self._resource_manager:
            self._resource_manager.close()
            self._resource_manager = None
        
        self._is_open = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __repr__(self) -> str:
        if self._is_open:
            return f"VisaInstrument({self.resource_name}, open=True)"
        return f"VisaInstrument(closed)"


class VisaInstrumentFactory:
    """VISA 仪器工厂
    
    用于批量创建和管理 VISA 仪器连接。
    
    Example:
        >>> factory = VisaInstrumentFactory()
        >>> 
        >>> # 添加仪器配置
        >>> factory.register('power_supply', 'USB0::0x1234::0x5678::SN001::INSTR')
        >>> factory.register('multimeter', 'GPIB0::12::INSTR')
        >>> 
        >>> # 批量打开
        >>> with factory.open_all() as instruments:
        ...     ps = instruments['power_supply']
        ...     dmm = instruments['multimeter']
        ...     print(ps.get_idn())
    """
    
    def __init__(self, resource_manager: Optional[VisaResourceManager] = None):
        """初始化工厂
        
        Args:
            resource_manager: 可选的资源管理器实例
        """
        self._rm = resource_manager or VisaResourceManager()
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._instruments: Dict[str, VisaInstrument] = {}
    
    def register(
        self,
        name: str,
        resource_name: str,
        **kwargs
    ) -> None:
        """注册仪器配置
        
        Args:
            name: 仪器别名
            resource_name: VISA 资源名称
            **kwargs: 其他连接参数
        """
        self._configs[name] = {
            'resource_name': resource_name,
            'kwargs': kwargs
        }
        logger.debug(f"已注册仪器 '{name}': {resource_name}")
    
    def open(self, name: str) -> VisaInstrument:
        """打开指定仪器
        
        Args:
            name: 仪器别名
        
        Returns:
            VisaInstrument 实例
        """
        if name not in self._configs:
            raise VisaError(f"未找到仪器配置: {name}")
        
        config = self._configs[name]
        instrument = VisaInstrument(
            config['resource_name'],
            resource_manager=self._rm,
            **config['kwargs']
        )
        self._instruments[name] = instrument
        return instrument
    
    def open_all(self) -> Dict[str, VisaInstrument]:
        """打开所有注册的仪器
        
        Returns:
            仪器名称到实例的映射
        """
        for name in self._configs:
            if name not in self._instruments:
                self.open(name)
        return self._instruments.copy()
    
    def close_all(self) -> None:
        """关闭所有仪器"""
        for name, instrument in list(self._instruments.items()):
            instrument.close()
        self._instruments.clear()
        logger.info("所有仪器已关闭")
    
    def close(self) -> None:
        """关闭工厂和所有资源"""
        self.close_all()
        if self._rm:
            self._rm.close()
            self._rm = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 便捷函数
def list_visa_resources(query: str = '?*::INSTR') -> tuple:
    """列出所有可用的 VISA 资源
    
    Args:
        query: 资源查询表达式
    
    Returns:
        资源名称元组
    """
    with VisaResourceManager() as rm:
        return rm.list_resources(query)


def open_visa_instrument(resource_name: str, **kwargs) -> VisaInstrument:
    """快速打开 VISA 仪器
    
    Args:
        resource_name: VISA 资源名称
        **kwargs: 其他连接参数
    
    Returns:
        VisaInstrument 实例
    """
    return VisaInstrument(resource_name, **kwargs)


def is_pyvisa_available() -> bool:
    """检查 PyVISA 是否可用"""
    return PYVISA_AVAILABLE
