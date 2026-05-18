from __future__ import annotations

import numpy as np
from loguru import logger

from ao_shaping.drivers.dm.base import DM
from ao_shaping.utils.hadamard_calc import HadamardGenerator


class HadamardDM(DM):
    """Hadamard系数驱动的变形镜/SLM接口.

    接受Hadamard系数向量作为输入，将其转换为Walsh-Hadamard拟合相位面型。
    内部使用 HadamardGenerator 进行相位计算。

    Attributes:
        mode_order: Hadamard矩阵阶数 (2的幂次)
        resolution: 输出相位图分辨率 (width, height)
        radius: 归一化半径（像素）
        mask_type: 光瞳掩码类型 ("circular" 或 "rectangular")
        bits: SLM位深度

    Example:
        >>> hdm = HadamardDM(mode_order=8, resolution=(1920, 1080))
        >>> hdm.open()
        >>> coeffs = np.zeros(64)  # 8x8 = 64 modes
        >>> coeffs[0] = 0.5  # First Hadamard mode
        >>> coeffs[5] = 0.3  # Another mode
        >>> phase = hdm.send(coeffs)
        >>> hdm.close()
    """

    def __init__(
        self,
        mode_order: int = 8,
        resolution: tuple[int, int] = (1920, 1080),
        radius: float | None = None,
        bits: int = 10,
        mask_type: str = "circular",
    ):
        """Initialize the Hadamard DM.

        Args:
            mode_order: The order N of the Hadamard matrix. Must be a power of 2.
                       Default is 8, giving 64 total 2D modes.
            resolution: Output phase resolution as (width, height).
                       Default is (1920, 1080).
            radius: Aperture radius in normalized coordinates. Default is 1.0.
            bits: SLM bit depth (e.g., 10 for 0-1023 range). Default is 10.
            mask_type: Pupil mask type ("circular" or "rectangular").
                      Default is "circular".
        """
        self.mode_order = mode_order
        self.resolution = resolution
        self.bits = bits
        self.mask_type = mask_type
        self._radius = radius

        # Initialize the Hadamard generator
        self._generator = HadamardGenerator(
            resolution=resolution,
            mode_order=mode_order,
            mask_type=mask_type,
            radius=radius,
        )
        self._generator.set_bits(bits)

        # Track current state
        self._current_coeffs: np.ndarray | None = None
        self._current_phase: np.ndarray | None = None
        self.is_open = False

    @property
    def DM_NUM(self) -> int:
        """Number of actuators (modes) for this DM."""
        return self._generator.n_modes

    def generate_phase(self, coefficients: np.ndarray) -> np.ndarray:
        """根据Hadamard系数生成相位面型（弧度）

        Args:
            coefficients: 1D array of Hadamord mode coefficients.
                         Length should be ≤ n_modes (mode_order²).

        Returns:
            相位面型（弧度），shape为 (height, width)
        """
        # Generate gray phase using the generator
        phase_gray = self._generator.generate_modes(coefficients)

        # Convert from gray values to radians
        max_val = 2**self.bits - 1
        phase_rad = phase_gray.astype(np.float64) / max_val * 2 * np.pi

        # Store current state
        self._current_coeffs = coefficients.copy()
        self._current_phase = phase_rad.copy()

        return phase_rad

    def generate_phase_2pi(self, coefficients: np.ndarray) -> np.ndarray:
        """生成0~2π范围的相位图（用于SLM显示）

        Args:
            coefficients: 1D array of Hadamord mode coefficients.

        Returns:
            灰度相位图，dtype=uint16
        """
        phase_gray = self._generator.generate_modes(coefficients)
        self._current_coeffs = coefficients.copy()
        self._current_phase = phase_gray.copy()
        return phase_gray

    def transform(self, cmd) -> np.ndarray:
        """Transform command to phase pattern.

        Args:
            cmd: Command to transform. Can be:
                - np.ndarray: 1D array of coefficients

        Returns:
            2D phase array in gray scale (uint16).

        Raises:
            ValueError: If command type is not supported.
        """
        if isinstance(cmd, np.ndarray):
            return self.generate_phase_2pi(cmd)
        raise ValueError(f"Unsupported command type: {type(cmd)}. Expected numpy array.")

    def send(self, cmd) -> np.ndarray:
        """Send command to DM and return phase pattern.

        Args:
            cmd: Command to send (1D numpy array of coefficients).

        Returns:
            2D phase array in gray scale (uint16).
        """
        return self.transform(cmd)

    def send_hadamard(self, coefficients: np.ndarray) -> np.ndarray:
        """发送Hadamard系数并返回相位图（快捷方法）

        Args:
            coefficients: 1D array of Hadamord mode coefficients.

        Returns:
            灰度相位图 (uint16)
        """
        return self.generate_phase_2pi(coefficients)

    def open(self) -> None:
        """Open the Hadamard DM connection."""
        self.is_open = True
        logger.info(
            f"HadamardDM opened: mode_order={self.mode_order}, "
            f"n_modes={self.DM_NUM}, resolution={self.resolution}, "
            f"mask_type={self.mask_type}"
        )

    def close(self) -> None:
        """Close the Hadamard DM connection."""
        self.is_open = False
        logger.info("HadamardDM closed")

    def get_actuator_positions(self) -> np.ndarray:
        """Get current actuator positions (coefficients).

        Returns:
            1D array of current coefficients, or empty array if none set.
        """
        if self._current_coeffs is None:
            return np.array([])
        return self._current_coeffs.copy()

    def get_phase(self) -> np.ndarray | None:
        """Get the current phase pattern.

        Returns:
            Current phase array, or None if no phase has been generated.
        """
        if self._current_phase is None:
            return None
        return self._current_phase.copy()

    def is_connected(self) -> bool:
        """Check if the DM is connected/open.

        Returns:
            True if open, False otherwise.
        """
        return self.is_open

    def get_hardware_info(self) -> dict:
        """Get hardware information about this DM.

        Returns:
            Dictionary with hardware specifications.
        """
        return {
            "type": "HadamardDM",
            "mode_order": self.mode_order,
            "n_modes": self.DM_NUM,
            "resolution": self.resolution,
            "radius": self._generator.radius,
            "mask_type": self.mask_type,
            "bits": self.bits,
        }

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"HadamardDM(mode_order={self.mode_order}, "
            f"resolution={self.resolution}, mask_type='{self.mask_type}')"
        )
