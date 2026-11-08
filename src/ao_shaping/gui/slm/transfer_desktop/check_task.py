"""迁移学习闭环页的两个后台硬件任务。

两个任务都只读枚举, 但代价与触发时机不同:

* :class:`HardwareCheckTask` —— 完整自检 (枚举两台相机 + 真正打开一次 SLM,
  秒级, 且 SDK 调用可能挂住)。**只有点「设备自检」按钮会创建它。**
* :class:`CcdDetectTask` —— 页面加载时跑一次的相机自动探测: 只枚举 Daheng 与
  MiiCam, **不打开相机、不探针 SLM**, 结论用来把 ① 设备区的相机类型 / cam_id
  摆到真正在线的那台上。

两者都不能落在 Qt 主线程 (SDK 枚举会卡住界面), 所以都包成 :class:`QRunnable`,
结论经信号回 UI 线程 —— 与 :class:`~ao_shaping.gui.slm.transfer_desktop.page._SessionTask`
同一套做法: 工作线程只发信号, 绝不碰控件。

**自检仍然不跟表单走。** ``_revalidate`` 每改一次控件就跑一遍, 把自检挂上去等于每敲
一个数字都枚举一次硬件 —— 既慢, 又会让「通过」失去意义。加载时的相机探测则只在
页面构造末尾跑一次 (见 ``page.TransferPage(auto_detect_ccd=...)``)。

``hardware_check`` 在各自的 ``run`` 内才 import: 那个模块的顶层 import 会拉起
Daheng (``gxipy``) 与 MiiCam 的 SDK, 页面冷启动不该为一次可能的自检先付这笔代价
(见 ``drivers/AGENTS.md`` 的惰性加载契约)。
"""

from __future__ import annotations

from collections.abc import Callable

from loguru import logger
from PySide6.QtCore import QObject, QRunnable, Signal

__all__ = ["CcdDetectTask", "HardwareCheckTask"]


class _TaskSignals(QObject):
    """``QRunnable`` 不是 ``QObject``, 用伴随对象把结论送回主线程。"""

    #: 任务结论 (``HardwareReport`` / ``CcdChoice``), 或异常实例 ——
    #: 任务本身失败也是一条结论, 不能吞掉。
    done = Signal(object)


def _emit_result(
    label: str, work: Callable[[], object], signals: _TaskSignals
) -> None:
    """跑一次只读工作, 把结论 (失败时是异常实例) 送回主线程。

    两个任务共用这一处 catch: 后台只读任务失败也是一条结论 (页面据此降级提示),
    不能吞掉; 而宽泛捕获与它的判据都只留一份, 不随任务数量扩散。
    """
    try:
        payload = work()
    # policy: 工作线程不碰界面; SDK 的任何失败都变成一条回到主线程的结论
    except Exception as exc:
        logger.opt(exception=exc).warning(f"{label}失败")
        signals.done.emit(exc)
    else:
        signals.done.emit(payload)


class HardwareCheckTask(QRunnable):
    """在后台线程跑一次 ``check_hardware``。"""

    def __init__(self, camera: str, cam_id: int, slm_number: int) -> None:
        super().__init__()
        self._camera = camera
        self._cam_id = cam_id
        self._slm_number = slm_number
        self.signals = _TaskSignals()

    def run(self) -> None:
        """线程池入口 (在后台线程执行)。"""

        def work() -> object:
            from ao_shaping.gui.slm.transfer_desktop.hardware_check import (
                check_hardware,
            )

            return check_hardware(self._camera, self._cam_id, self._slm_number)

        _emit_result("硬件自检任务", work, self.signals)


class CcdDetectTask(QRunnable):
    """在后台线程跑一次 ``select_ccd`` (只枚举相机, 不打开相机、不探针 SLM)。"""

    def __init__(self, preferred: str, cam_id: int) -> None:
        super().__init__()
        self._preferred = preferred
        self._cam_id = cam_id
        self.signals = _TaskSignals()

    def run(self) -> None:
        """线程池入口 (在后台线程执行)。"""

        def work() -> object:
            from ao_shaping.gui.slm.transfer_desktop.hardware_check import select_ccd

            return select_ccd(self._preferred, self._cam_id)

        _emit_result("相机自动探测任务", work, self.signals)