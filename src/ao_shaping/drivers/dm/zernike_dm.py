from __future__ import annotations

from warnings import deprecated

import numpy as np
from loguru import logger

from ao_shaping.drivers.dm._registry import register_dm
from ao_shaping.drivers.dm.base import DM
from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator
from ao_shaping.utils.wavefront.zernike_utils import parse_zernike_coefficients


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

        ⚠️ **单位约定 (2026-09-16 修复后)**: ``coefficients`` 的数值**即弧度**,
        输出保留**绝对幅度** —— 系数 ×4 得到 ×4 的相位 PV。
        旧实现曾做 min-max 归一化, 使输出对系数缩放不变 (幅度不可控), 已移除。

        Args:
            coefficients: Zernike系数 (单位: 弧度)，可以是:
                - dict: {(n, m): value} 形式的系数
                - np.ndarray: 按Noll顺序排列的系数向量
            output_mode: 输出模式:
                - "rad": 返回**弧度**相位 (系数原值, 不归一化, 不 mod 2π)
                - "gray": 返回灰度相位 (mod 2π → 0..2^bits−1)

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

        # Generate phase using ZernikeGenerator's generate_polynomial.
        # Outside the aperture the zernike package yields NaN; zero them.
        phase_raw = self._generator.generate_polynomial(coeffs_dict)
        phase_raw = np.nan_to_num(phase_raw, nan=0.0, posinf=0.0, neginf=0.0)

        # ⚠️ 不做 min-max 归一化 (2026-09-16 修复)。
        # 旧实现 `(raw−min)/(max−min) × 2π` 使输出**对系数缩放不变**
        # (系数 ×1 与 ×4 产生逐字节相同相位, PV 恒为 2π, 实测
        #  np.array_equal == True) → Zernike 系数的"幅度"维度被完全抹掉,
        # 所有 ZernikeSLM 消费方 (zernike-matrix / rms-zernike / ga-zernike /
        # greedy-zernike / GUI) 都无法控制相位幅度。
        # 现改为: **系数即弧度**, 直接输出, 保留绝对幅度 (与
        # PatternHelper.generate_zernike_polynomial 语义一致)。
        self._current_coeffs = coeffs_dict
        if output_mode == "rad":
            phase_out = phase_raw
            self._current_phase = phase_out.copy()
            return phase_out

        # "gray": 弧度 → 灰度 (mod 2π), 语义与 SLM 驱动
        # `create_phase_from_array` 一致 (2π 对应 max_val)
        phase_out = np.mod(phase_raw / (2.0 * np.pi) * max_val, max_val).astype(
            np.uint16
        )
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
        # parse_zernike_coefficients 保持 Noll j = idx+1 映射 (与旧实现逐条一致),
        # 仅跳过 |amp| < 1e-15; 此处补 1e-10 后过滤恢复原阈值语义。
        parsed = parse_zernike_coefficients(coeffs)
        return {nm: float(amp) for nm, amp in parsed.items() if abs(amp) >= 1e-10}

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
