from __future__ import annotations

import numpy as np
from loguru import logger

from ao_shaping.drivers.dm.base import DM
from ao_shaping.utils.zernike_calc import ZernikeGenerator


class ZernikeDM(DM):
    """Zernike系数驱动的变形镜接口

    接受Zernike系数作为输入，将其转换为Zernike拟合相位面型。
    内部使用 ZernikeGenerator 进行相位计算。

    Attributes:
        n_max: Zernike多项式的最大阶数
        resolution: 输出相位图的分辨率 (width, height)
        radius: 归一化半径（像素）

    Example:
        >>> zdm = ZernikeDM(n_max=4, resolution=(1920, 1080))
        >>> zdm.open()
        >>> coeffs = {  # (n, m): coefficient
        ...     (1, -1): 0.5,  # tilt X
        ...     (1, 1): 0.3,   # tilt Y
        ...     (2, 0): 0.2,   # defocus
        ... }
        >>> phase = zdm.send_zernike(coeffs)
        >>> zdm.close()
    """

    def __init__(
        self,
        n_max: int = 4,
        resolution: tuple[int, int] = (1920, 1080),
        radius: float | None = None,
        bits: int = 10,
    ):
        self.n_max = n_max
        self.resolution = resolution
        self.bits = bits

        self._generator = ZernikeGenerator(resolution=resolution, radius=radius, n_orders=n_max)
        self._generator.set_bits(bits)

        self._current_coeffs: dict[tuple[int, int], float] = {}
        self._current_phase: np.ndarray | None = None
        self.is_open = False

    def generate_phase(
        self,
        coefficients: dict[tuple[int, int], float] | np.ndarray,
    ) -> np.ndarray:
        """根据Zernike系数生成相位面型（弧度）

        Args:
            coefficients: Zernike系数，可以是:
                - dict: {(n, m): value} 形式的系数
                - np.ndarray: 按Noll顺序排列的系数向量

        Returns:
            相位面型（弧度），shape为 (height, width)
        """
        if isinstance(coefficients, np.ndarray):
            coefficients = self._noll_to_dict(coefficients)

        phase_rad = self._generate_phase_rad(coefficients)

        self._current_coeffs = coefficients
        self._current_phase = phase_rad.copy()
        return phase_rad

    def _generate_phase_rad(
        self,
        coefficients: dict[tuple[int, int], float],
    ) -> np.ndarray:
        """使用ZernikeGenerator生成弧度相位"""
        height, width = self.resolution[1], self.resolution[0]
        phase_total = np.zeros((height, width), dtype=np.float64)

        for (n, m), amp in coefficients.items():
            if abs(amp) < 1e-10:
                continue
            single_phase = self._generator.generate(n, m, amplitude=amp)
            phase_rad = single_phase.astype(np.float64) / (2**self.bits - 1) * 2 * np.pi
            phase_total += phase_rad

        return phase_total

    def generate_phase_2pi(
        self,
        coefficients: dict[tuple[int, int], float] | np.ndarray,
    ) -> np.ndarray:
        """生成0~2π范围的相位图（用于SLM显示）

        Args:
            coefficients: Zernike系数，支持:
                - dict: {(n, m): value} 形式
                - np.ndarray: Noll顺序的系数向量

        Returns:
            灰度相位图，dtype=uint16
        """
        if isinstance(coefficients, np.ndarray):
            phase_gray = self._generator.generate_noll(coefficients)
            self._current_coeffs = self._noll_to_dict(coefficients)
        else:
            phase_gray = self._generator.generate_polynomial(coefficients)
            self._current_coeffs = coefficients

        self._current_phase = phase_gray.copy()
        return phase_gray

    def _noll_to_dict(self, coeffs: np.ndarray) -> dict[tuple[int, int], float]:
        """将Noll顺序的系数向量转换为字典形式"""
        result = {}
        for j, amp in enumerate(coeffs):
            if abs(amp) < 1e-10:
                continue
            n, m = self._generator.noll_to_nm(j + 1)
            result[(n, m)] = float(amp)
        return result

    def transform(self, cmd) -> np.ndarray:
        if isinstance(cmd, np.ndarray):
            return self.generate_phase_2pi(cmd)
        if isinstance(cmd, dict):
            return self.generate_phase_2pi(cmd)
        raise ValueError(f"Unsupported command type: {type(cmd)}")

    def send(self, cmd) -> np.ndarray:
        return self.transform(cmd)

    def send_zernike(
        self,
        coefficients: dict[tuple[int, int], float] | np.ndarray,
    ) -> np.ndarray:
        """发送Zernike系数并返回相位图（快捷方法）"""
        return self.generate_phase_2pi(coefficients)

    def open(self) -> None:
        self.is_open = True
        logger.info(f"ZernikeDM opened: n_max={self.n_max}, resolution={self.resolution}")

    def close(self) -> None:
        self.is_open = False
        logger.info("ZernikeDM closed")

    def get_actuator_positions(self) -> np.ndarray:
        return np.array(list(self._current_coeffs.values()))

    def get_phase(self) -> np.ndarray | None:
        return self._current_phase.copy() if self._current_phase is not None else None

    def is_connected(self) -> bool:
        return self.is_open

    def get_hardware_info(self) -> dict:
        return {
            "type": "ZernikeDM",
            "n_max": self.n_max,
            "resolution": self.resolution,
            "radius": self._generator.radius,
        }

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self) -> str:
        return f"ZernikeDM(n_max={self.n_max}, resolution={self.resolution})"
