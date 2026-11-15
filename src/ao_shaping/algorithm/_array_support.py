"""算法层共享的数组工具 (numpy, 无 torch 依赖)。

把 ``algorithm/`` 下逐字重复、且语义确实一致的数组操作收敛到一处。只收
**可互换**的实现 —— 语义不同的相似代码 (FFT 变体、相位 wrap 区间等) 不在此
模块的范围内。
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def symmetric_clip(
    init_params: npt.NDArray[np.float64],
    disturb: npt.NDArray[np.float64],
    clip_min: npt.NDArray[np.float64],
    clip_max: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """对初始参数施加 ±扰动, 并各自裁剪到 ``[clip_min, clip_max]``。

    用于 SOD 幅值扫描里对称的正/负扰动评估。正负两支用**同一个**扰动方向
    (``init + disturb`` 与 ``init - disturb``) 再各自裁剪, 保证两支的扰动
    幅度对称。

    Args:
        init_params: 基线参数向量。
        disturb: 扰动向量 (已含方向与幅度)。
        clip_min: 每个参数的下界。
        clip_max: 每个参数的上界。

    Returns:
        ``(pos_params, neg_params)``, 均已裁剪。
    """
    return (
        np.clip(init_params + disturb, clip_min, clip_max),
        np.clip(init_params - disturb, clip_min, clip_max),
    )
