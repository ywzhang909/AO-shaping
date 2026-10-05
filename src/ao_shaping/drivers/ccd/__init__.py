"""CCD 相机驱动包。

本包提供统一接口的相机驱动。

基于 SDK 的后端以**惰性**方式暴露 (PEP 562): ``MIICamera`` 与 ``DahengCamera``
在首次属性访问时才解析, 而不是在包导入时。急切导入后端并非无副作用 ——
``miicam.driver`` 会调用 ``_setup_miicam_sdk()``, 它修改 ``sys.path`` 并经
``ctypes.CDLL`` 加载原生 ``MIIUSB.dll``, 而 ``daheng`` 会导入 ``gxipy`` 绑定。
由于 Python 在任何 ``ao_shaping.drivers.ccd.common`` 导入之前就先初始化本包,
此前那些急切导入让相机注册表 (:func:`create_camera`) 的**每一个**使用者都要为
两个 SDK 买单。把它们推迟, 与 ``ccd/AGENTS.md`` 里已为注册表记录的惰性约定一致:
缺失或损坏的后端 SDK 只在该后端真正被请求时才失败。

公开名称未变, 因此 ``from ao_shaping.drivers.ccd import MIICamera`` 继续可用;
无法导入的后端仍与以前一样退化为 ``None``。
"""

from typing import TYPE_CHECKING

from ao_shaping.drivers._lazy import install_lazy_attrs
from ao_shaping.drivers.ccd.base import BaseCamera, CameraError
from ao_shaping.drivers.ccd.ffmpeg import (
    FFmpegCamera,
    FFmpegCameraError,
    ImageFolderCamera,
)

# 公开属性名 -> (要导入的模块, 该模块内的属性)。
_LAZY_BACKENDS: dict[str, tuple[str, str]] = {
    "MIICamera": ("ao_shaping.drivers.ccd.miicam.driver", "MIICamera"),
    "MIICAMError": ("ao_shaping.drivers.ccd.miicam.driver", "MIICAMError"),
    "DahengCamera": ("ao_shaping.drivers.ccd.daheng", "DahengCamera"),
}


if TYPE_CHECKING:
    # 静态分析器需要看见这些名字; 运行时它们由首次访问时的 ``__getattr__`` 绑定。
    from ao_shaping.drivers.ccd.daheng import DahengCamera
    from ao_shaping.drivers.ccd.miicam.driver import MIICAMError, MIICamera


__getattr__ = install_lazy_attrs(globals(), _LAZY_BACKENDS, __name__)



def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "BaseCamera",
    "CameraError",
    "MIICamera",
    "MIICAMError",
    "DahengCamera",
    "FFmpegCamera",
    "FFmpegCameraError",
    "ImageFolderCamera",
]
