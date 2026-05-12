"""
DM Control Python Package
变形镜控制Python包

Usage:
    from dm_control import DMController
    
    # 使用上下文管理器
    with DMController() as dm:
        dm.init()
        dm.set_voltage_all(50.0)
        dm.set_actuator(1, 10.0)
    
    # 或使用便捷函数
    from dm_control import init, set_actuator, open_relay
    
    init()
    set_actuator(1, 10.0)
    open_relay()
"""

from .dm_control import (
    DMController,
    DMError,
    ErrorCode,
    R50Controller,
    ActuatorMapping,
    MAX_CONTROLLERS,
    MAX_CHANNELS,
    MAX_ACTUATORS,
    VOLTAGE_MIN,
    VOLTAGE_MAX,
    # 便捷函数
    init,
    disconnect,
    set_voltage_all,
    set_actuator,
    open_relay,
    close_relay,
)

__all__ = [
    "DMController",
    "DMError", 
    "ErrorCode",
    "R50Controller",
    "ActuatorMapping",
    "MAX_CONTROLLERS",
    "MAX_CHANNELS",
    "MAX_ACTUATORS",
    "VOLTAGE_MIN",
    "VOLTAGE_MAX",
    "init",
    "disconnect",
    "set_voltage_all",
    "set_actuator",
    "open_relay",
    "close_relay",
]

__version__ = "1.0.0"