from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from ao_shaping.drivers.dm.zernike_dm import ZernikeDM
from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200


class ZernikeSLMError(Exception):
    pass


class ZernikeSLM:
    """Zernike系数驱动的SLM接口

    封装SantecSLM200，将Zernike系数转换为相位图并发送到SLM显示。
    内部使用ZernikeDM进行相位计算。

    Attributes:
        slm: 底层的SantecSLM200实例
        zernike_dm: Zernike相位计算器

    Example:
        >>> with ZernikeSLM(slm_number=1, wavelength=1064, n_max=4) as zslm:
        ...     coeffs = {
        ...         (1, -1): 0.5,  # tilt X
        ...         (1, 1): 0.3,   # tilt Y
        ...         (2, 0): 0.2,   # defocus
        ...     }
        ...     zslm.send_zernike(coeffs)
    """

    def __init__(
        self,
        slm_number: int = 1,
        wavelength: int|None = None,
        n_max: int = 4,
        slm_resolution: tuple[int, int] | None = None,
        use_120hz: bool = False,
        shift_x: int = 0,
        shift_y: int = 0,
        correction_csv_path: str | Path | None = None,
    ):
        self.slm_number = slm_number
        self.wavelength = wavelength
        self.n_max = n_max

        if slm_resolution is None:
            slm_resolution = SantecSLM200.Panel_Res  # (1920, 1200)

        self._slm = SantecSLM200(
            slm_number=slm_number,
            use_120hz=use_120hz,
            wavelength=wavelength,
            shift_x=shift_x,
            shift_y=shift_y,
            correction_csv_path=correction_csv_path,
        )
        self._zernike_dm = ZernikeDM(
            n_max=n_max,
            resolution=slm_resolution,
        )

        self.is_open = False
        self._current_phase: np.ndarray | None = None
        self._current_coeffs: dict[tuple[int, int], float] = {}

    def open(self) -> None:
        """打开SLM设备并初始化"""
        self._slm.open()
        self._zernike_dm.open()

        try:
            current_wl, max_gray = self._slm.get_wavelength_info()
            if not self.wavelength:
                self.wavelength = current_wl
            if current_wl != self.wavelength:
                self._slm.set_wavelength(self.wavelength)
        except Exception as e:
            logger.warning(f"读取波长信息失败: {e}，跳过波长设置")

        self.is_open = True
        logger.info(
            f"ZernikeSLM opened: slm_number={self.slm_number}, "
            f"wavelength={self.wavelength}nm, n_max={self.n_max}"
        )

    def close(self) -> None:
        """关闭SLM设备"""
        self._zernike_dm.close()
        self._slm.close()
        self.is_open = False
        logger.info("ZernikeSLM closed")

    def _ensure_open(self) -> None:
        if not self.is_open:
            raise RuntimeError("ZernikeSLM not open, call open() first")

    def set_shift(self, shift_x: int, shift_y: int) -> None:
        """设置平移参数

        Args:
            shift_x: X方向平移像素数（正=右，负=左）
            shift_y: Y方向平移像素数（正=下，负=上）
        """
        self._slm.set_shift(shift_x, shift_y)

    @property
    def shift_x(self) -> int:
        """X方向平移像素数"""
        return self._slm.shift_x

    @property
    def shift_y(self) -> int:
        """Y方向平移像素数"""
        return self._slm.shift_y

    # TODO wait_time移动到init中去
    def send_zernike(
        self,
        coefficients: dict[tuple[int, int], float] | np.ndarray,
        wait_time_s:float = 0.2
    ) -> np.ndarray:
        """发送Zernike系数到SLM

        Args:
            coefficients: Zernike系数，可以是:
                - dict: {(n, m): value} 形式的系数
                - np.ndarray: 按Noll顺序排列的系数向量
            display: 是否立即显示到SLM，默认为True

        Returns:
            发送的灰度相位图
        """
        phase_rad = self._zernike_dm.generate_phase_2pi(coefficients)
        self._current_phase = self._slm.create_phase_from_array(phase_rad)

        self._current_coeffs = (
            coefficients if isinstance(coefficients, dict)
            else self._zernike_dm._noll_to_dict(coefficients)
        )
        self._slm.display_data(self._current_phase, wait_time_s)

        return self._current_phase

    def send_zernike_to_memory(
        self,
        coefficients: dict[tuple[int, int], float] | np.ndarray,
        memory_number: int = 1,
    ) -> np.ndarray:
        """发送Zernike系数到SLM内存

        Args:
            coefficients: Zernike系数
            memory_number: 内存位置编号（1-128）

        Returns:
            写入的灰度相位图
        """
        self._ensure_open()

        phase_gray = self._zernike_dm.generate_phase_2pi(coefficients)
        self._slm.display_data(phase_gray)

        return phase_gray

    def display_memory(self, memory_number: int) -> None:
        """显示指定内存的相位图"""
        self._ensure_open()
        self._slm.display_memory(memory_number)

    def set_flat(self) -> None:
        """设置SLM为平相位（清零）"""
        self._ensure_open()
        self._current_phase = np.zeros(self._zernike_dm.resolution, dtype=np.uint16).T
        self._slm.display_data(self._current_phase)

    def set_grayscale(self, gs: int) -> None:
        """设置SLM为均匀灰度值"""
        self._ensure_open()
        self._slm.set_grayscale(gs)

    def get_current_phase(self) -> np.ndarray | None:
        """获取当前显示的相位缓存"""
        return self._zernike_dm.get_phase()

    def get_current_zernike_coeffs(self) -> dict[tuple[int, int], float]:
        """获取当前Zernike系数"""
        return self._zernike_dm._current_coeffs.copy()

    def is_connected(self) -> bool:
        return self.is_open

    def get_hardware_info(self) -> dict:
        return {
            "type": "ZernikeSLM",
            "slm_number": self.slm_number,
            "wavelength": self.wavelength,
            "n_max": self.n_max,
            "resolution": self._zernike_dm.resolution,
            "slm_info": self._slm.get_wavelength_info() if self.is_open else None,
        }

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self) -> str:
        status = "connected" if self.is_open else "disconnected"
        return f"ZernikeSLM(slm_number={self.slm_number}, wavelength={self.wavelength}nm, n_max={self.n_max}, status={status})"

    @property
    def n_modes(self):
        return self._zernike_dm.DM_NUM