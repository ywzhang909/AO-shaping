"""`from_response_matrix` 的模态空间回退必须校验 pinv 与 matrix 是同一份标定。

矩阵与伪逆都来自同一个 h5, 但可以是**不同次**标定留下的 (旧文件、中途重写)。构造时不校验的话,
不一致会立刻被固化进控制器 —— `mask_indices` 按 matrix 的行数生成, `s_ref` 按 pinv 的列数生成,
两者当场就不等; 而失败要等到闭环迭代里的 matmul 才暴露, 报的是 numpy 的 core dimension 错,
不说是哪两份数据不匹配。
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.optimizer.wf.zernike_response_matrix import ZernikeResponseMatrixResult
from ao_shaping.runners.slm.zernike_matrix_runner import LoopConfig, from_response_matrix


def _modal_only_result(matrix: np.ndarray, pinv: np.ndarray) -> ZernikeResponseMatrixResult:
    """只有 matrix + pinv 的结果对象: 走模态空间回退分支。"""
    return ZernikeResponseMatrixResult(
        matrix=matrix,
        variance_matrix=np.zeros_like(matrix),
        deviation_response_matrix=None,
        subaperture_mask=None,
        n_max=10,
        magnitude=0.1,
        wavelength_nm=532,
        n_averages=1,
        n_cycles=1,
        timestamp="test",
        excluded_piston=True,
        excluded_tip_tilt=False,
        pinv_matrix=pinv,
        lstsq_matrix=None,
        amplitude_optimization=None,
    )


def test_consistent_pair_builds_a_self_consistent_controller() -> None:
    matrix = np.random.default_rng(0).normal(size=(66, 65))
    loop = from_response_matrix(
        _modal_only_result(matrix, np.linalg.pinv(matrix)), LoopConfig(n_modes=65)
    )
    assert loop.D_pinv.shape == (matrix.shape[1], matrix.shape[0])
    assert loop.mask_indices.size == matrix.shape[0]
    assert loop.s_ref.size == matrix.shape[0]


def test_pinv_from_another_calibration_is_rejected_at_construction() -> None:
    matrix = np.random.default_rng(0).normal(size=(66, 65))
    foreign_pinv = np.linalg.pinv(np.random.default_rng(1).normal(size=(64, 65)))

    with pytest.raises(ValueError, match=r"pinv|伪逆"):
        from_response_matrix(_modal_only_result(matrix, foreign_pinv), LoopConfig(n_modes=65))
