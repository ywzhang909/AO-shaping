"""SLM+Zernike PIB 闭环的环境鲁棒性门: 差分噪声统计与噪声门。

自 :mod:`ao_shaping.optimizer.wfless.slm_zernike_pib` 拆出的纯函数叶子:

- :func:`_rolling_sigma` —— 从最近的 ``diff`` 历史估计 ``sigma_hat``;
- :func:`_noise_gate` —— ``|diff| <= k * sigma_hat`` 时把 SPGD 更新置零, 使系数
  诚实停滞而不是随机游走。

两者都不碰硬件; ``optimize_slm_zernike_pib`` 的 SPGD 分支原样调用, 行为不变。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _rolling_sigma(diffs: Sequence[float]) -> float:
    """Std of the recent ``diff`` history (0.0 when degenerate/empty)."""
    if len(diffs) < 2:
        return 0.0
    arr = np.asarray(diffs, dtype=np.float64)
    s = float(np.std(arr))
    return s if np.isfinite(s) else 0.0


def _noise_gate(diff: float, sigma_hat: float, k: float) -> bool:
    """True when ``diff`` is indistinguishable from the recent diff noise
    (``|diff| <= k * sigma_hat``) - the SPGD update would be pure measurement
    noise and is zeroed so the coefficients stall honestly instead of
    random-walking. ``k <= 0`` (or an unusable ``sigma_hat``) disables the gate.
    """
    if k <= 0.0 or not np.isfinite(sigma_hat) or sigma_hat <= 0.0:
        return False
    return abs(diff) <= k * sigma_hat
