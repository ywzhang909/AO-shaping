"""光路标定 ``.npz`` 的读取与几何系数校验 (衍射效率测量的叶子依赖)。

自 :mod:`ao_shaping.tools.slm.calibration.diffraction_efficiency.measure` 拆出:
把几何标定产物读进内存、校验必需键, 并取出零级中心对应的 K 放大系数。这样
``measure`` 的扫描分支与产物分支都复用同一份校验, 而无须把其全部 import 面
拖进任何一个消费者。

术语
----
中文叙述遵循仓库术语表: 光路标定, 面板像元, 实验系统。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

#: 光路标定里必须存在的键 (缺一即拒绝继续)。
REQUIRED_CALIBRATION_KEYS: tuple[str, ...] = ("center", "Kx", "Ky")


def _load_calibration(path: Path) -> dict[str, Any]:
    """读取光路标定 ``.npz``, 校验必需键。

    Args:
        path: 标定文件路径。

    Returns:
        ``{键: 数组}``。

    Raises:
        RuntimeError: 文件不存在或缺少 ``center``/``Kx``/``Ky``。
    """
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(
            f"光路标定文件不存在: {path}。"
            "请先运行 `python -m ao_shaping.tools.slm.calibration` 生成光路标定 (.npz)。"
        )
    with np.load(path, allow_pickle=True) as data:
        calib: dict[str, Any] = {key: np.asarray(data[key]) for key in data.files}
    missing = [key for key in REQUIRED_CALIBRATION_KEYS if key not in calib]
    if missing:
        raise RuntimeError(
            f"光路标定 {path} 缺少必需键 {missing} (需要 center/Kx/Ky)。"
            "请先运行 `python -m ao_shaping.tools.slm.calibration` 重新生成光路标定。"
        )
    return calib


def _k_scale(calib: dict[str, Any], axis: str) -> float:
    """取该光栅方向的放大系数 (CCD 像素 / 面板像元)。"""
    key = "Ky" if str(axis).strip().lower() == "y" else "Kx"
    value = float(np.asarray(calib[key], dtype=np.float64).ravel()[0])
    if value <= 0.0:
        raise RuntimeError(f"光路标定的 {key} 非正 ({value}), 几何标定无效")
    return value
