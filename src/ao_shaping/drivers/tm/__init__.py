"""TM (Timing Module / 定时模块) 驱动器包.

当前仅暴露 :class:`~ao_shaping.drivers.tm.serial_port_fsm.SerialPortFSM`
(串口定时模块, 有限状态机控制) 的 ``TM`` 别名, 用于定时模块设备的
串口命令收发与状态机管理。
"""

from __future__ import annotations
from ao_shaping.drivers.tm.serial_port_fsm import SerialPortFSM as TM
