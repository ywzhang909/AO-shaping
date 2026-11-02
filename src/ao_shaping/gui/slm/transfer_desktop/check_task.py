"""迁移学习闭环页的「设备自检」后台任务。

自检要枚举两台相机的 SDK、并真正打开一次 SLM（秒级、且 SDK 调用可能挂住），
所以不能落在 Qt 主线程。本模块把它包成 :class:`QRunnable`, 结论经信号回 UI
线程 —— 与 :class:`~ao_shaping.gui.slm.transfer_desktop.page._SessionTask`
同一套做法: 工作线程只发信号, 绝不碰控件。

**只有按钮点击会创建它。** 表单校验 (``_revalidate``) 每改一次控件就跑一遍,
把自检挂上去等于每敲一个数字都枚举一次硬件 —— 既慢, 又会让「通过」失去意义。

``hardware_check`` 在 :meth:`HardwareCheckTask.run` 内才 import: 那个模块的顶层
import 会拉起 Daheng (``gxipy``) 与 MiiCam 的 SDK, 页面冷启动不该为一次可能的
自检先付这笔代价 (见 ``drivers/AGENTS.md`` 的惰性加载契约)。
"""

from __future__ import annotations

from loguru import logger
from PySide6.QtCore import QObject, QRunnable, Signal

__all__ = ["HardwareCheckTask"]


class _CheckSignals(QObject):
    """``QRunnable`` 不是 ``QObject``, 用伴随对象把结论送回主线程。"""

    #: ``HardwareReport`` 或异常实例 —— 自检本身失败也是一条结论, 不能吞掉。
    done = Signal(object)


class HardwareCheckTask(QRunnable):
    """在后台线程跑一次 ``check_hardware``。"""

    def __init__(self, camera: str, cam_id: int, slm_number: int) -> None:
        super().__init__()
        self._camera = camera
        self._cam_id = cam_id
        self._slm_number = slm_number
        self.signals = _CheckSignals()

    def run(self) -> None:
        """线程池入口 (在后台线程执行)。"""
        try:
            from ao_shaping.gui.slm.transfer_desktop.hardware_check import (
                check_hardware,
            )

            report = check_hardware(self._camera, self._cam_id, self._slm_number)
        # policy: 工作线程不碰界面; 任何失败都变成一条回到主线程的结论
        except Exception as exc:
            logger.opt(exception=exc).warning("硬件自检任务失败")
            self.signals.done.emit(exc)
        else:
            self.signals.done.emit(report)
