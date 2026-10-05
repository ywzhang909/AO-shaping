from __future__ import annotations

import numpy as np
from loguru import logger

from ao_shaping.drivers.dm.base import DM
from ao_shaping.drivers.dm._registry import register_dm
from ao_shaping.utils.wavefront.hadamard_calc import HadamardGenerator


# HadamardDM 刻意没有 from_params 工厂: 驱动层的任何参数类
# 都没有定义它的 mode_order、resolution、radius、bits、mask_type
# 和 safety_mode 这些构造函数字段。


@register_dm("hadamard")
class HadamardDM(DM):
    """Hadamard系数驱动的变形镜/SLM接口.

    接受Hadamard系数向量作为输入，将其转换为Walsh-Hadamard拟合相位面型。
    内部使用 HadamardGenerator 进行相位计算。

    Attributes:
        mode_order: Hadamard模式阶数
        resolution: 输出相位图的分辨率 (width, height)
        mask_type: 掩码类型
        radius: 归一化半径（像素）
    """

    @classmethod
    def is_reachable(cls) -> bool:
        return True

    def __init__(
        self,
        mode_order: int = 8,
        resolution: tuple[int, int] = (1920, 1080),
        radius: float | None = None,
        bits: int = 10,
        mask_type: str = "circular",
        safety_mode: bool = True,
    ):
        """初始化 Hadamard DM。

        Args:
            mode_order: Hadamard 矩阵的阶 N。必须是 2 的幂。
            resolution: 输出相位分辨率, 形式为 (宽, 高)。
            radius: 归一化坐标下的光瞳半径。
            bits: SLM 位深 (例如 10 表示 0-1023 范围)。
            mask_type: 光瞳掩码类型 ("circular" 或 "rectangular")。
            safety_mode: 仅为接口一致性而接受 (对相位型 DM 无作用)。
        """
        self.mode_order = mode_order
        self.resolution = resolution
        self.bits = bits
        self.mask_type = mask_type
        self._radius = radius

        # 先于 super().__init__ 初始化 Hadamard 生成器,
        # 因为 DM_NUM 属性依赖 _generator
        self._generator = HadamardGenerator(
            resolution=resolution,
            mode_order=mode_order,
            mask_type=mask_type,
            radius=radius,
        )
        self._generator.set_bits(bits)

        super().__init__(safety_mode=safety_mode)

        # 跟踪当前状态
        self._current_coeffs: np.ndarray | None = None
        self._current_phase: np.ndarray | None = None
        self.is_open = False

    @property
    def DM_NUM(self) -> int:
        """该 DM 的致动器 (模式) 数量。"""
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

    def generate_phase(self, coefficients: np.ndarray) -> np.ndarray:
        """根据Hadamard系数生成相位面型（弧度）

        Args:
            coefficients: Hadamard 模式系数的一维数组,
                         长度应 ≤ n_modes (mode_order²)。

        Returns:
            相位面型（弧度），shape为 (height, width)
        """
        # 用生成器产生灰度相位
        phase_gray = self._generator.generate_modes(coefficients)

        # 从灰度值转换为弧度
        max_val = 2**self.bits - 1
        phase_rad = phase_gray.astype(np.float64) / max_val * 2 * np.pi

        # 保存当前状态
        self._current_coeffs = coefficients.copy()
        self._current_phase = phase_rad.copy()

        return phase_rad

    def generate_phase_2pi(self, coefficients: np.ndarray) -> np.ndarray:
        """生成0~2π范围的相位图（用于SLM显示）

        Args:
            coefficients: Hadamard 模式系数的一维数组。

        Returns:
            灰度相位图，dtype=uint16
        """
        phase_gray = self._generator.generate_modes(coefficients)
        self._current_coeffs = coefficients.copy()
        self._current_phase = phase_gray.copy()
        return phase_gray

    def transform(self, cmd) -> np.ndarray:
        """把命令转换为相位图案。

        Args:
            cmd: 待转换的命令。可以是:
                - np.ndarray: 系数的一维数组

        Returns:
            灰度标度的二维相位数组 (uint16)。

        Raises:
            ValueError: 命令类型不受支持时。
        """
        if isinstance(cmd, np.ndarray):
            return self.generate_phase_2pi(cmd)
        raise ValueError(
            f"Unsupported command type: {type(cmd)}. Expected numpy array."
        )

    def send(self, cmd) -> np.ndarray:
        """向 DM 发送命令并返回相位图案。

        Args:
            cmd: 待发送的命令 (系数的一维 numpy 数组)。

        Returns:
            灰度标度的二维相位数组 (uint16)。
        """
        return self.transform(cmd)

    def send_hadamard(self, coefficients: np.ndarray) -> np.ndarray:
        """发送Hadamard系数并返回相位图（快捷方法）

        Args:
            coefficients: Hadamard 模式系数的一维数组。

        Returns:
            灰度相位图 (uint16)
        """
        return self.generate_phase_2pi(coefficients)

    def open(self) -> None:
        """打开 Hadamard DM 连接。"""
        self.is_open = True
        logger.info(
            f"HadamardDM opened: mode_order={self.mode_order}, "
            f"n_modes={self.DM_NUM}, resolution={self.resolution}, "
            f"mask_type={self.mask_type}"
        )

    def close(self) -> None:
        """关闭 Hadamard DM 连接。"""
        self.is_open = False
        logger.info("HadamardDM closed")

    def get_actuator_positions(self) -> np.ndarray:
        """获取当前致动器位置 (系数)。

        Returns:
            当前系数的一维数组, 未设置时返回空数组。
        """
        if self._current_coeffs is None:
            return np.array([])
        return self._current_coeffs.copy()

    def get_phase(self) -> np.ndarray | None:
        """获取当前相位图案。

        Returns:
            当前的相位数组, 若尚未生成任何相位则为 None。
        """
        if self._current_phase is None:
            return None
        return self._current_phase.copy()

    def is_connected(self) -> bool:
        """检查 DM 是否已连接/打开。

        Returns:
            已打开返回 True, 否则返回 False。
        """
        return self.is_open

    def get_hardware_info(self) -> dict:
        info = super().get_hardware_info()
        info.update(
            {
                "mode_order": self.mode_order,
                "n_modes": self.DM_NUM,
                "resolution": self.resolution,
                "radius": self._generator.radius,
                "mask_type": self.mask_type,
                "bits": self.bits,
            }
        )
        return info

    def __enter__(self):
        """上下文管理器入口。"""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口。"""
        self.close()

    def __repr__(self) -> str:
        """字符串表示。"""
        return (
            f"HadamardDM(mode_order={self.mode_order}, "
            f"resolution={self.resolution}, mask_type='{self.mask_type}')"
        )
