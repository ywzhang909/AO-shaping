"""闭环 SLM 远场整形仿真台架 —— 重新导出 + sim 前向模型。

本模块作为 **数字孪生 (数字双胞胎)** 层的门面:

* **算法实现** (GS、WeightedGS/HIO、可微分、SPGD、analytic-amplitude) 已迁移到
  :mod:`ao_shaping.algorithm.signal_processing.square_shaping` —— 算法层不
  导入任何驱动 / sim 代码, 保持纯数学。
* **sim-specific 前向模型** (``forward_intensity``) 与其物理参数
  ``ShapingBenchConfig`` 保留在这里, 封装 2f 傅里 Fourier 台架的物理。
* **ShapingBenchConfig** 继承自 ``SquareShapingConfig``, 增加 ``cn2`` /
  ``l_max`` / ``l_min`` 与 ``make_beam_config()`` —— 这些是 sim-only 的大
  时湍流参数, 算法层不需要 (算法层的正向模型是纯 FFT, 不路由 ``cn2``)。

关于物理模型的说明
------------------
2f 傅里 Fourier 台架 (SLM 前焦面 → f 透镜 → 相机后焦面) 把 SLM 瞳孔场映射到
它的**傅里叶变换** (透镜相位 + 传播到 ``z=f``, 除一个全局相位外就等于夫琅
禾费变换)。因此在 FFT 之前瞳孔要补零到 ``far_field_size``, 使焦平面被正确
采样: 同尺寸的 FFT 是以 ``w_far/dx_far = aperture/(pi*w0) = 3.5/pi = 1.11`` px
每腰半径对焦平面采样的 —— 这一个与 ``n_grid`` 无关的模型常数 —— 它会把经透镜
相位调制的瞳孔混叠成一个点阵 (见 ``far_field_padding``)。0 级光斑位于全局强
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
    """2f 傅里 Fourier SLM 整形台架的参数 (sim-only 扩展)。

    继承自 :class:`~ao_shaping.algorithm.signal_processing.square_shaping.SquareShapingConfig`,
    增加 sim-only 的大气湍流参数 (``cn2`` / ``l_max`` / ``l_min``) 与
    :meth:`make_beam_config` (桥接到 ``beam_backend``)。

    这些额外字段在算法层的前向模型中是**死配置** (永不路由), 但保留它们
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
# sim 前向模型 (2f 傅里 Fourier) —— 留在 sim 层
# ---------------------------------------------------------------------------


def _pad_centred(arr: np.ndarray, cfg: ShapingBenchConfig) -> np.ndarray:
    """把方形瞳孔网格数组居中补零到远场网格。

    委托 :func:`~ao_shaping.algorithm.signal_processing.square_shaping._pad_centred`
    (仅 ``far_field_size`` 取自 ``cfg``)。
    """
    return _alg_pad_centred(arr, cfg.far_field_size)


def _fraunhofer_intensity(field: np.ndarray, cfg: ShapingBenchConfig) -> np.ndarray:
    """瞳孔场的补零夫琅禾费 (焦平面) 强度。

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
