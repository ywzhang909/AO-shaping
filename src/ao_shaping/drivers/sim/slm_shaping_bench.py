"""闭环 SLM 远场整形仿真实验系统 —— 重新导出 + sim 物理预测模型。

本模块作为 **数字孪生 (数字双胞胎)** 层的门面:

* **算法实现** (GS、WeightedGS/HIO、可微分、SPGD、analytic-amplitude) 已迁移到
  :mod:`ao_shaping.algorithm.signal_processing.square_shaping` —— 算法层不
  导入任何驱动 / sim 代码, 保持纯数学。
* **sim-specific 物理预测模型** (``forward_intensity``) 与其物理参数
  ``ShapingBenchConfig`` 保留在这里, 封装 2f 傅里 Fourier 实验系统的物理。
* **ShapingBenchConfig** 继承自 ``SquareShapingConfig``, 增加 ``cn2`` /
  ``l_max`` / ``l_min`` 与 ``make_beam_config()`` —— 这些是 sim-only 的大
  时湍流参数, 算法层不需要 (算法层的物理预测模型是纯 FFT, 不路由 ``cn2``)。

关于物理预测模型的说明
------------------
2f 傅里 Fourier 实验系统 (SLM 前焦面 → f 透镜 → 相机后焦面) 把 SLM 瞳面场映射到
它的**傅里叶变换** (透镜相位 + 传播到 ``z=f``, 除一个全局相位外就等于夫琅
禾费变换)。因此在 FFT 之前瞳面要补零到 ``far_field_size``, 使焦平面被正确
采样: 同尺寸的 FFT 是以 ``w_far/dx_far = aperture/(pi*w0) = 3.5/pi = 1.11`` px
每腰半径对焦平面采样的 —— 这一个与 ``n_grid`` 无关的模型常数 —— 它会把经透镜
相位调制的瞳面混叠成一个点阵 (见 ``far_field_padding``)。0 级光斑位于全局强
度最大值处 (用 ``argmax`` 定位, 从不按几何定位)。

⚠️ ``cn2`` 是**死配置** (2026-10-01 实测):
:func:`forward_intensity` 使用 :meth:`SquareShapingConfig.make_gaussian_pupil`
(内联 FFT, 不经过 ``beam_backend`` 后端路由), 因此在 ``cn2=0`` 与
``cn2=5e-14`` 下输出字节一致 (max|ΔI| = 0)。不要在本 bench 上调 ``cn2`` 后
期待看到湍流效果; 需真实湍流路由请用 :class:`SimulatedTurbulentScreen` 或
:class:`SimTurbulenceAOEnv`。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ao_shaping.drivers.sim.beam_backend import (
    BeamSimConfig,
    make_beam_config,
)

# ---------------------------------------------------------------------------
# 算法层 re-exports (single source of truth)
# ---------------------------------------------------------------------------
from ao_shaping.algorithm.signal_processing.square_shaping import (
    GS_METHODS,
    GS_SELECT,
    ShapingResult,
    SquareShapingConfig,
    UniformSquareShaper,
    _fraunhofer_intensity_from_field,
    _initial_field as _alg_initial_field,
    _pad_centred as _alg_pad_centred,
    analytic_amplitude_target,
    compute_bench_metrics,
    composite_from_pib_cv,
    composite_score,
    differentiable_shape,
    efficiency,
    gs_shape,
    make_square_target,
    power_in_bucket,
    spgd_shape,
    strehl,
    uniformity_cv,
    zero_order_fraction,
)

# ``make_target`` is the legacy name; alias the canonical ``make_square_target``.
make_target = make_square_target

__all__ = [
    "ShapingBenchConfig",
    "SquareShapingConfig",
    "ShapingResult",
    "forward_intensity",
    "make_target",
    "make_square_target",
    "compute_bench_metrics",
    "composite_score",
    "composite_from_pib_cv",
    "power_in_bucket",
    "uniformity_cv",
    "efficiency",
    "strehl",
    "zero_order_fraction",
    "GS_METHODS",
    "GS_SELECT",
    "UniformSquareShaper",
    "gs_shape",
    "differentiable_shape",
    "spgd_shape",
    "analytic_amplitude_target",
]


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShapingBenchConfig(SquareShapingConfig):
    """2f 傅里 Fourier SLM 整形实验系统的参数 (sim-only 扩展)。

    继承自 :class:`~ao_shaping.algorithm.signal_processing.square_shaping.SquareShapingConfig`,
    增加 sim-only 的大气湍流参数 (``cn2`` / ``l_max`` / ``l_min``) 与
    :meth:`make_beam_config` (桥接到 ``beam_backend``)。

    这些额外字段在算法层的物理预测模型中是**死配置** (永不路由), 但保留它们
    是为了 ``ShapingBenchConfig`` 仍可用于需要湍流程径 (如 ``turbulence_phase``)
    的 sim 场景。
    """

    cn2: float = 0.0
    l_max: float = 0.1
    l_min: float = 1e-3

    def make_beam_config(self) -> BeamSimConfig:
        """构造 ``beam_backend.BeamSimConfig`` (桥接湍流后端路由)。"""
        return make_beam_config(
            n_grid=self.n_grid,
            aperture_size=self.aperture_size,
            wavelength=self.wavelength,
            cn2=self.cn2,
            l_max=self.l_max,
            l_min=self.l_min,
            propagation_distance=self.focal_length,
        )


# ---------------------------------------------------------------------------
# sim 物理预测模型 (2f 傅里 Fourier) —— 留在 sim 层
# ---------------------------------------------------------------------------


def _pad_centred(arr: np.ndarray, cfg: ShapingBenchConfig) -> np.ndarray:
    """把方形瞳面网格数组居中补零到远场网格。

    委托 :func:`~ao_shaping.algorithm.signal_processing.square_shaping._pad_centred`
    (仅 ``far_field_size`` 取自 ``cfg``)。
    """
    return _alg_pad_centred(arr, cfg.far_field_size)


def _fraunhofer_intensity(field: np.ndarray, cfg: ShapingBenchConfig) -> np.ndarray:
    """瞳面场的补零夫琅禾费 (焦平面) 强度。

    ``field`` 在 ``n_grid`` 网格上 (会被补零), 或者已补零到 ``far_field_size``
    (补零是 no-op)。委托算法层的
    :func:`~ao_shaping.algorithm.signal_processing.square_shaping._fraunhofer_intensity_from_field`。
    """
    return _fraunhofer_intensity_from_field(field, cfg.far_field_size)


def _gs_initial_field(
    cfg: ShapingBenchConfig,
    beam_cfg=None,
    *,
    seed: int | None = None,
    base_phase: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """GS 的初始场 (Gaussian 入射 + 随机扰动), 已补零到远场网格。

    .. deprecated:: 委托 :func:`~ao_shaping.algorithm.signal_processing.square_shaping._initial_field`。
       ``beam_cfg`` 参数已弃用 (算法层内联 Gaussian 瞳面), 保留仅为向后兼容。
    """
    return _alg_initial_field(cfg, seed=seed, base_phase=base_phase)


def forward_intensity(
    phase: np.ndarray, cfg: SquareShapingConfig | ShapingBenchConfig
) -> np.ndarray:
    """把纯相位 SLM 图样传播到远场; 返回强度 (未归一化)。

    Gaussian 入射场乘以 ``exp(1j*phase)`` (纯相位 SLM), 补零 FFT 到焦平面。
    使用 :meth:`SquareShapingConfig.make_gaussian_pupil` (内联), 与
    ``cn2`` 无关。
    """
    pupil = cfg.make_gaussian_pupil().astype(np.complex128)
    field = pupil * np.exp(1j * np.asarray(phase, dtype=np.complex128))
    return _fraunhofer_intensity_from_field(field, cfg.far_field_size)


#: Default camera pixel pitch (um) for the Daheng MER2-507 used on this bench.
DEFAULT_CAMERA_PIXEL_UM = 2.2


def camera_window_from_far_field(
    intensity: np.ndarray,
    cfg: SquareShapingConfig | ShapingBenchConfig,
    window_shape: tuple[int, int],
    *,
    camera_pixel_um: float = DEFAULT_CAMERA_PIXEL_UM,
) -> np.ndarray:
    """把远场重采样到**相机像元网格**并居中裁剪成 ``window_shape``。

    为什么需要这一步: :func:`forward_intensity` 的远场像元间距是
    ``lambda*f/(aperture_size*padding)`` (见
    :attr:`SquareShapingConfig.far_field_pixel_size`), 与相机像元间距**无关**。
    在本实验系统上 ``n_grid = pupil_side(450) = 900``、``padding = 3`` 时远场视场约
    16.6 mm, 而相机 250 px 窗只覆盖 0.55 mm —— 相差约 30 倍。直接并排显示会让
    第三栏与第一栏无法逐像素比较。

    两个方向都要处理: 远场比相机**细**时抽稀, 比相机**粗**时插值放大 (否则光斑
    会被画得比真实更小)。本实验系统 ``padding=3`` 属后者。

    Args:
        intensity: :func:`forward_intensity` 的输出 (焦平面强度)。
        cfg: 产生该远场的配置, 用于取 ``far_field_pixel_size``。
        window_shape: 目标 ``(rows, cols)``, 通常是实测帧的形状。
        camera_pixel_um: 相机像元间距 (um)。

    Returns:
        形状为 ``window_shape``、单位为相机像元的强度。

    Raises:
        ValueError: 若重采样后仍小于请求的窗口。
    """
    from scipy.ndimage import zoom

    px_far_m = cfg.far_field_pixel_size
    px_cam_m = camera_pixel_um * 1e-6
    if px_far_m <= 0 or px_cam_m <= 0:
        raise ValueError("pixel sizes must be positive")

    image: np.ndarray = np.asarray(intensity, dtype=np.float64)
    ratio = px_far_m / px_cam_m  # >1: far field is finer than the camera
    if abs(ratio - 1.0) > 1e-9:
        image = np.asarray(zoom(image, ratio, order=1, mode="nearest"))

    rows, cols = window_shape
    if image.shape[0] < rows or image.shape[1] < cols:
        raise ValueError(
            f"far field {image.shape} too small for camera window {window_shape}"
        )
    r0 = (image.shape[0] - rows) // 2
    c0 = (image.shape[1] - cols) // 2
    return image[r0 : r0 + rows, c0 : c0 + cols]
