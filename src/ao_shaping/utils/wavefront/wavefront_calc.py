from __future__ import annotations

from pathlib import Path

import numpy as np

from ao_shaping.utils.image.spots_calc import centroid as _centroid

# Measured Zernike base matrices (66 modes, 360x360). Anchored to the
# repository, NOT to the process CWD: the previous default was the relative
# string "scripts/tuning_devices/stdWavefront", which made
# ``ZernikeCentroidCalculator()`` work only from the repo root.
_STD_WAVEFRONT_PATH = (
    Path(__file__).resolve().parents[4]
    / "scripts"
    / "tuning_devices"
    / "stdWavefront"
    / "std_wavefront.npz"
)


def normalize_01(matrix):
    """将矩阵归一化到 [0, 1]（**min-max** 归一化）。

    ⚠️ **不委托**给 :func:`~ao_shaping.utils.image.beam_metrics.normalize_pattern`
    （R-28 实测更正）。两者是不同的函数，且
    ``normalize_pattern`` 的默认 mode 是 ``"peak"``：

    | 输入 | ``normalize_01``（本函数，min-max） | ``normalize_pattern``（默认 peak） |
    |---|---|---|
    | ``[[0, 1], [2, 4]]`` | 拉伸到 ``[0, 1]`` | ``/ 4``，``min`` 仍是 0 |
    | ``[[10, 11], [12, 14]]`` | 拉伸到 ``[0, 1]`` | ``/ 14``，``min`` 仍是 0.71 |
    | 常量矩阵 | **全零** | 全一（``pmax > 0``） |
    | 含 ``NaN`` | **``NaN`` 传播** | ``nan_to_num`` 后有限 |
    | 输出 dtype | 跟输入 | 强制 ``float32`` |

    保留原实现行为：常量（``max == min``）返回全零。
    需要「按最大值归一 + 暗帧安全 + NaN 清除」时用 ``normalize_pattern``。
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    if max_val == min_val:
        return np.zeros_like(matrix)
    return (matrix - min_val) / (max_val - min_val)


def centroid_calculation(matrix):
    """计算矩阵的质心坐标 (委托至 spots_calc.centroid, return_float=True)。"""
    return _centroid(matrix, return_float=True)


def calculate_derotation(x_actual, y_actual, theta):
    """
    计算消旋坐标变换（反向旋转theta角）

    参数:
    x_actual: 实际x坐标
    y_actual: 实际y坐标
    theta: 旋转角度

    返回:
    x_derotated: 消旋后的x坐标
    y_derotated: 消旋后的y坐标
    """
    # 步骤2：计算旋转角的余弦值和正弦值
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # 步骤3：执行消旋坐标变换（反向旋转theta角），公式依据专利消旋原理推导
    x_derotated = x_actual * cos_theta + y_actual * sin_theta
    y_derotated = -x_actual * sin_theta + y_actual * cos_theta

    # 步骤4：输出消旋后的坐标（保留6位小数，与专利实施例数据精度一致，如0.025mm、-0.144mm）
    x_derotated = np.round(x_derotated, 6)
    y_derotated = np.round(y_derotated, 6)

    return x_derotated, y_derotated


def get_zernike_base_matrixs(path: str | Path | None = None) -> np.ndarray:
    """Load the 66 measured Zernike base matrices, ordered by **mode number**.

    The data set is 66 square-wavefront maps of 360x360 samples covering
    Zernike modes 1..66 in canonical order (1 = piston, 2 = tilt x, 3 = tilt y,
    4 = defocus, 5 = astigmatism, ...). It ships as one compressed ``.npz``
    holding a float64 ``matrices`` stack of shape ``(66, 360, 360)`` plus the
    matching ``modes`` vector.

    Three defects this replaced (R-34, 2026-10-04):

    * **Mode order was silently wrong.** The previous loader iterated
      ``Path(folder_path).glob("*.txt")``, which is *lexicographic*:
      ``1, 10, 11, ..., 66, 7, 8, 9``. Since :meth:`ZernikeCentroidCalculator.get_centroid`
      is a positional weighted sum (``sum_i matrices[i] * coef[i]``), 65 of
      the 66 slots held the wrong map -- a pure tilt-x command (``coef[1] = 1``)
      returned a centroid of (179.500, 179.496), i.e. dead centre, instead of
      the correct (144.291, 179.500). The ``modes`` vector is now checked for
      contiguity so the mapping cannot rot again unnoticed.
    * **CWD-relative default.** The path was resolved against the process CWD,
      so ``ZernikeCentroidCalculator()`` only worked from the repo root. It is
      now anchored to the repository, like
      :func:`ao_shaping.drivers.dm._adjacency.load_adjacency`.
    * **56.4 MB of ASCII.** The 66 ``.txt`` files held 129 600 whitespace
      separated decimals each and took 0.71 s to parse. The ``.npz`` is
      bit-exact float64 (verified with ``np.array_equal``), 23.2 MB, and loads
      in 0.13 s.

    Args:
        path: Override for the ``.npz`` location. Defaults to the repository's
            own copy; a missing file raises ``FileNotFoundError`` rather than
            falling back to another source.

    Returns:
        ``(66, 360, 360)`` float64 array indexed by ``mode - 1``.

    Raises:
        FileNotFoundError: The data file is absent.
        ValueError: The stored mode numbers are not ``1..N`` in order, or the
            stack is not square.
    """
    import loguru

    data_path = Path(path) if path is not None else _STD_WAVEFRONT_PATH
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Zernike base matrices not found at {data_path}. The data file is "
            "tracked in the repository (scripts/tuning_devices/stdWavefront/"
            "std_wavefront.npz); it is not regenerated at runtime."
        )

    with np.load(data_path) as payload:
        modes = payload["modes"]
        matrices = payload["matrices"]

    expected = np.arange(1, matrices.shape[0] + 1)
    if not np.array_equal(modes, expected):
        raise ValueError(
            f"{data_path.name}: mode numbers must be 1..{matrices.shape[0]} in "
            f"ascending order, got {modes.tolist()[:5]}...{modes.tolist()[-3:]}. "
            "get_centroid() multiplies matrices[i] by coef[i] positionally, so a "
            "reordered stack silently returns wrong centroids."
        )
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError(
            f"{data_path.name}: expected a (N, S, S) stack, got {matrices.shape}"
        )

    loguru.logger.debug(
        "loaded {} Zernike base matrices {} from {}",
        matrices.shape[0],
        matrices.shape[1:],
        data_path,
    )
    return np.ascontiguousarray(matrices, dtype=np.float64)


def to_color(matrix, max_val=1):
    # 将矩阵转换为RGB图像（归一化到0-255范围）
    normalized_matrix = (matrix) / (max_val + 1e-8)
    rgb_matrix = np.stack([normalized_matrix * 255] * 3, axis=-1).astype(np.uint8)
    return rgb_matrix


class ZernikeCentroidCalculator:
    """Map a Zernike coefficient vector to a beam centroid.

    ⚠️ **Coefficients are positional**: ``coef[i]`` multiplies Zernike mode
    ``i + 1`` in the measured order (1 = piston, 2 = tilt x, 3 = tilt y, ...).
    """

    # 计算100mm×100mm对应的像素尺寸  —（233，220）  （237，223）
    mm_size = 100  # 实际尺寸(mm)
    resolution_for_70mm = 360  # 70mm对应的像素数
    shape = (resolution_for_70mm, resolution_for_70mm)
    pixel_per_mm = resolution_for_70mm / 70  # 每毫米的像素数
    pixel_size = int(round(mm_size * pixel_per_mm))  # 100mm对应的像素数
    mm_per_pixel = 1 / pixel_per_mm  # 每像素对应的毫米数 (约0.1944mm/像素)

    # print(f"像素尺寸: {self.pixel_size}x{self.pixel_size}")
    # print(f"每毫米像素数: {self.pixel_per_mm:.4f} mm/pixel")
    def __init__(self, path: str | Path | None = None, black_level: float = 0.0):
        self.wavefront_matrices = get_zernike_base_matrixs(path)
        self.num_files = self.wavefront_matrices.shape[0]
        if self.wavefront_matrices.shape[1:] != self.shape:
            raise ValueError(
                f"base matrices are {self.wavefront_matrices.shape[1:]}, but this "
                f"class is calibrated for {self.shape}"
            )
        self.black_level = black_level

    def get_centroid(self, zernike_coef: np.ndarray):
        """
        计算给定Zernike系数组合的波前矩阵的质心坐标
        """
        zer_class = min(len(zernike_coef), self.num_files)
        _zernike_coef = zernike_coef[:zer_class]
        _wavefront_matrices = self.wavefront_matrices[:zer_class]
        _zernike_base_matrix = np.sum(
            _wavefront_matrices * _zernike_coef[:, np.newaxis, np.newaxis], axis=0
        )
        _zernike_base_matrix = normalize_01(_zernike_base_matrix)
        _zernike_base_matrix = _zernike_base_matrix - self.black_level
        _zernike_base_matrix = np.where(
            _zernike_base_matrix < 0, 0, _zernike_base_matrix
        )
        cx, cy = centroid_calculation(_zernike_base_matrix)
        return (cx, cy), _zernike_base_matrix

    def pix_to_mm(self, pix):
        """
        将像素坐标转换为毫米坐标
        """
        return (pix - self.resolution_for_70mm / 2) * self.mm_per_pixel

    def center_coordinate(self, cx, cy):
        """
        将像素坐标转换为毫米坐标
        """
        return (cx - self.resolution_for_70mm / 2), (cy - self.resolution_for_70mm / 2)
