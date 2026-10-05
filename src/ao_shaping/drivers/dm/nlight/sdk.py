"""NLight C SDK (Drv_UDPST.dll) ctypes 绑定.

与 :mod:`ao_shaping.drivers.dm.nlight.udp` 的 UDP 批量下发互补: 这条通道
负责高压开关 (``SetHV``) 与电压读回 (``GetVoltages``)。

DLL 位置由 ``findlibs`` 在 PATH 中解析 (``Drv_UDPST``)，不在此硬编码路径。
"""

from __future__ import annotations

from ctypes import byref, c_bool, c_int32, cdll

import findlibs
import numpy as np
import numpy.typing as npt

from ao_shaping.drivers.dm.nlight.nlight_constants import DM_NUM


class DMSdk:
    """``Drv_UDPST.dll`` 的 ctypes 包装。

    构造过程会加载 DLL 并断言驱动板已连接, 因此调用方只应在预期硬件已上电时实例化它。
    """

    def __init__(self) -> None:
        self.dm_num = DM_NUM
        path = findlibs.find("Drv_UDPST")
        if path is None:
            raise Exception("Drv_UDPST.dll not found.")

        dll = cdll.LoadLibrary(path)

        dll.GetConnection2.restype = c_bool
        dll.GetConnection2.argtypes = []

        dll.SetVoltages.restype = c_bool
        dll.SetVoltages.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double, ndim=1, shape=(self.dm_num)),
            c_int32,
            c_int32,
        ]

        dll.SetVoltagesNoEcho.restype = c_bool
        dll.SetVoltagesNoEcho.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double, ndim=1, shape=(self.dm_num)),
            c_int32,
            c_int32,
        ]

        dll.SetHV.restype = c_bool
        dll.SetHV.argtypes = [c_bool, c_bool]

        dll.GetVoltages.restype = c_bool
        dll.GetVoltages.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double, ndim=1, shape=(self.dm_num)),
            c_int32,
            c_int32,
        ]

        self._dll = dll
        assert self._dll.GetConnection2(), "device connection error."

    def set_voltages(
        self, vs: npt.NDArray[np.floating], with_echo: bool = False
    ) -> bool:
        func = self._dll.SetVoltages if with_echo else self._dll.SetVoltagesNoEcho
        return func(vs, c_int32(0), c_int32(self.dm_num))

    def reset_all(self) -> bool:
        return self._dll.ResetAll()

    def set_hv(self, hv: bool) -> bool:
        return self._dll.SetHV(c_bool(hv), c_bool(True))

    def get_hv(self) -> c_bool:
        hv_status = c_bool(False)
        if self._dll.GetHV(byref(hv_status)):
            return hv_status
        raise Exception("device connection error.")

    def get_voltages(self) -> npt.NDArray[np.float64]:
        voltages = np.zeros(self.dm_num, dtype=np.double)
        if self._dll.GetVoltages(voltages, c_int32(0), c_int32(self.dm_num)):
            return voltages
        raise Exception("device connection error.")


__all__ = ["DMSdk"]
