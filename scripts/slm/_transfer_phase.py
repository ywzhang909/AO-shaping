"""共享的“最新 transfer 整形相位 → SLM 面板相位”读入口。

上机脚本（泽尼克匀化、离焦扫描、均匀性重放）的被测相位统一从这里取：
``ab_session_20261124`` 的 accepted 终稿 ``*_best.npy`` (640x640 弧度) 或等价的
``capture_*.npz`` / ``replay_*.npz`` 中的 ``model_phase``。面板展开沿用
:func:`ao_shaping.tools.slm.transfer.capture.to_panel`，与
``replay_phases.replay`` 的上屏路径逐位一致；不需要 ramp 矩阵、几何校正或
frozen_parameters（那些只用于前向模型，不参与相位重放）。

dry-run 不打开设备：本模块只 import 无驱动的 ``capture.to_panel``。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ao_shaping.tools.slm.transfer.capture import to_panel

#: 当前最新已验收的 transfer 整形终稿（2026-11-24 A/B 会话，star dense）。
#: 方形对照用同目录的 ``square/mono12_dense/square_best.npy``。
DEFAULT_TRANSFER_PHASE = Path(
    "data/debug/slm_transfer/ab_session_20261124/shape/"
    "star/mono12_dense/star_best.npy"
)


def load_transfer_phase(path: str | Path | None = None) -> tuple[Path, np.ndarray, np.ndarray]:
    """读 transfer 整形相位并返回 ``(路径, 模型相位, 面板弧度)``。

    ``.npy`` 必须是 640x640 弧度模型网格；``.npz`` 必须有 ``model_phase``
    （``panel_gray`` 是单向灰度结果，不能反推弧度，故不作为输入）。面板相位
    与 ``replay_phases`` 的 ``to_panel(model_phase)`` 完全一致，供已有脚本的
    rad→gray 转换链直接消费。

    Args:
        path: 相位文件路径；``None`` 用 :data:`DEFAULT_TRANSFER_PHASE`。

    Returns:
        ``(path, model, panel)``，``panel`` 为 ``(1200, 1920)`` float64 弧度。

    Raises:
        FileNotFoundError: 路径不存在。
        ValueError: 数组形状不能平铺物理面板，或缺 ``model_phase``。
    """
    resolved = Path(path) if path is not None else DEFAULT_TRANSFER_PHASE
    if not resolved.is_file():
        raise FileNotFoundError(f"transfer 整形相位不存在: {resolved}")
    if resolved.suffix == ".npy":
        model = np.load(resolved, allow_pickle=False)
    elif resolved.suffix == ".npz":
        with np.load(resolved, allow_pickle=False) as payload:
            if "model_phase" not in payload:
                raise ValueError(f"{resolved} 没有 model_phase（panel_gray 不可反推弧度）")
            model = payload["model_phase"]
    else:
        raise ValueError(f"不支持的 transfer 相位格式: {resolved.suffix}")
    model = np.asarray(model, dtype=np.float64)
    if model.shape != (1200, 1920) and (
        model.ndim != 2
        or model.shape[0] != model.shape[1]
        or 1920 % model.shape[0]
        or 1200 % (1920 // model.shape[0])
    ):
        raise ValueError(f"{resolved} 相位形状 {model.shape} 不能平铺 1200x1920 面板")
    panel = model if model.shape == (1200, 1920) else to_panel(model)
    return resolved, model, np.asarray(panel, dtype=np.float64)
