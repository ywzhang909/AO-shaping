"""波前传感器驱动的共享抽象契约。

WFS 家族曾是唯一没有共享基类的设备家族: 真实驱动直接继承
:class:`~ao_shaping.drivers.device_base.Device`, 因此不存在任何可供仿真 WFS 实现的
契约 —— 也就无法作为即插即用的替代品。相机家族用 ``BaseCamera``、DM 家族用 ``DM``
解决了同一问题, 本模块为 WFS 补齐同样的缺口。

抽象接口面恰好就是 optimizer 与 runner 实际调用的那一组方法
(通过 grep ``optimizer/`` 与 ``runners/`` 验证), 因此实现它就足以让其中任意一个
跑在仿真传感器上。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ao_shaping.drivers.device_base import Device, DeviceType
from ao_shaping.utils.wavefront.zernike_calc import (
    calc_n_zernike_terms as _canonical_calc_n_zernike_terms,
)


class BaseWFS(Device, ABC):
    """波前传感器 (真实或仿真) 的抽象基类。"""

    device_type: DeviceType = DeviceType.WFS
    manufacturer: str = "Unknown"
    model: str = "BaseWFS"

    def __init__(self, device_id: str = "") -> None:
        super().__init__(device_id)
        self._register_wfs_parameters()

    def _register_wfs_parameters(self) -> None:
        """注册每个 WFS 驱动都必须暴露的参数。"""
        self.register_parameter(
            "exposure_time_ms",
            default_value=0.0,
            min_value=0.0,
            max_value=70.0,
            unit="ms",
            description="WFS camera exposure time; 0 selects auto exposure.",
        )
        self.register_parameter(
            "remove_tilt",
            default_value=False,
            unit="",
            description="Remove tip/tilt from the measured wavefront.",
        )
        self.register_parameter(
            "pupil_diameter",
            default_value=2.7,
            min_value=0.0,
            unit="mm",
            description="Assumed pupil diameter used for phase reconstruction.",
        )
        self.register_parameter(
            "pupil_center",
            default_value=(0.0, 0.0),
            unit="mm",
            description="Pupil centroid used for phase reconstruction.",
        )

    @staticmethod
    def calc_n_zernike_terms(n: int) -> int:
        """到径向阶次 ``n`` (含 piston) 为止的 Zernike 项数。

        与 :func:`~ao_shaping.utils.wavefront.zernike_calc.calc_n_zernike_terms`
        恰好相差 piston 项: WFS 的 slope 拟合带着这一项, 而 canonical 辅助函数不计。
        这里只保留唯一一份实现, 避免两者在无人察觉的情况下各自漂移。
        """
        return _canonical_calc_n_zernike_terms(n) + 1

    @abstractmethod
    def take_image(self, n_sample: int = 10, dynamicNoiseCut: bool = True) -> None:
        """采集点场图像, 并由其导出斑点质心与直径。"""

    @abstractmethod
    def get_spots_statics(
        self,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """返回最近一次采集的 ``(intensities, (centroid_x, centroid_y))``。"""

    @abstractmethod
    def build_subaperture_mask(
        self,
        n_avg: int = 30,
        threshold_ratio: float = 0.3,
        edge_clip: int = 1,
        plot: bool = False,
    ) -> np.ndarray:
        """通过通量阈值的子孔径布尔掩码。"""

    @abstractmethod
    def get_spot_deviation(
        self, cancel_tile: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """斑点相对参考位置的位移, 单位 m。"""

    @abstractmethod
    def get_wavefront(self, cancel_tile: bool = False) -> tuple[np.ndarray, dict]:
        """重建出的波前图 (单位 m) 及其元数据。"""

    @abstractmethod
    def get_zernike(self, zernike_order: int = 10) -> np.ndarray:
        """对最近一次波前拟合出的 Zernike 系数, 单位 **微米 (µm)**。

        微米这一单位是 WFS 家族的历史契约, 并由
        :mod:`tests.ao_shaping.drivers.wfs.test_base_wfs` 断言。调用方若要以 waves
        为单位做任何运算, 必须先经
        :func:`~ao_shaping.utils.wavefront.zernike_utils.um_to_waves` 换算。
        """


__all__ = ["BaseWFS"]
