from __future__ import annotations

from warnings import deprecated

import numpy as np
from loguru import logger

from ao_shaping.drivers.dm._registry import register_dm
from ao_shaping.drivers.dm.base import DM
from ao_shaping.utils.zernike_calc import ZernikeGenerator


@register_dm("zernike")
class ZernikeDM(DM):
    """Zernike系数驱动的变形镜接口

    接受Zernike系数作为输入，将其转换为Zernike拟合相位面型。
    内部使用 ZernikeGenerator 进行相位计算。

    Attributes:
        n_max: Zernike多项式的最大阶数
        resolution: 输出相位图的分辨率 (width, height)
        radius: 归一化半径（像素）
    """

    def __init__(
        self,
        n_max: int = 4,
        resolution: tuple[int, int] = (1920, 1080),
        radius: float | None = None,
        bits: int = 10,
        safety_mode: bool = True,
    ):
        self.n_max = n_max
        self.resolution = resolution
        self.bits = bits

        # Initialize generator BEFORE super().__init__
        # because DM_NUM property depends on _generator
        self._generator = ZernikeGenerator(
            resolution=resolution, radius=radius, n_orders=n_max
        )
        self._generator.set_bits(bits)

        super().__init__(safety_mode=safety_mode)

        self._current_coeffs: dict[tuple[int, int], float] = {}
        self._current_phase: np.ndarray | None = None
        self.is_open = False

    @classmethod
    def is_reachable(cls) -> bool:
        return True

    @property
    def DM_NUM(self) -> int:
        return self._generator.n_modes

    @property
    def V_Min(self) -> float:
        return 0.0

    @property
    def V_Max(self) -> float:
        return float(2**self.bits - 1)

    @property
    def max_neibor_diff(self) -> float:
        return float("inf")

    @property
    def default_dm_unit_mask(self) -> np.ndarray:
        return np.ones(self.DM_NUM, dtype=bool)

    def generate_phase(
        self,
        coefficients: dict[tuple[int, int], float] | np.ndarray,
        output_mode: str = "rad",
    ) -> np.ndarray:
        """根据Zernike系数生成相位面型

        Args:
            coefficients: Zernike系数，可以是:
                - dict: {(n, m): value} 形式的系数
                - np.ndarray: 按Noll顺序排列的系数向量
            output_mode: 输出模式:
                - "rad": 返回弧度相位 (0-2π)，用于波形计算
                - "gray": 返回灰度相位 (0-1023)，用于SLM显示

        Returns:
            相位面型，shape为 (height, width)
            output_mode="rad" 时 dtype=float64，output_mode="gray" 时 dtype=uint16
        """
        if isinstance(coefficients, np.ndarray):
            coeffs_dict = self._noll_to_dict(coefficients)
        else:
            coeffs_dict = coefficients

        height, width = self.resolution[1], self.resolution[0]
        max_val = float(2**self.bits - 1)

        # Generate phase using ZernikeGenerator's generate_polynomial
        phase_raw = self._generator.generate_polynomial(coeffs_dict)

        # phase_raw is in arbitrary units (typically -2.5 to +2.5)
        # Normalize to [0, 1] range then scale appropriately
        phase_min = np.nanmin(phase_raw)
        phase_max = np.nanmax(phase_raw)
        phase_range = phase_max - phase_min

        if phase_range > 1e-10:
            phase_normalized = (phase_raw - phase_min) / phase_range
        else:
            phase_normalized = np.zeros_like(phase_raw)

        # Set outputs
        self._current_coeffs = coeffs_dict
        if output_mode == "rad":
            phase_out = phase_normalized * 2 * np.pi
            self._current_phase = phase_out.copy()
            return phase_out
        else:
            phase_out = (phase_normalized * max_val).astype(np.uint16)
            self._current_phase = phase_out.copy()
            return phase_out

    @deprecated("Use generate_phase with output_mode='gray'")
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
        return self.generate_phase(coefficients, output_mode="gray")

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
        logger.info(
            f"ZernikeDM opened: n_max={self.n_max}, resolution={self.resolution}"
        )

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
        info = super().get_hardware_info()
        info.update(
            {
                "n_max": self.n_max,
                "resolution": self.resolution,
                "radius": self._generator.radius,
            }
        )
        return info

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self) -> str:
        return f"ZernikeDM(n_max={self.n_max}, resolution={self.resolution})"
