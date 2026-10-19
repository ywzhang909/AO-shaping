"""T 系列仿真脚本共享的评分与量定义。

T1 (noise relaxation) 与 T2 (mixed fitting) 报告里发布的 2 分量复合分数
``0.5·EE + 0.5/(1+CV)`` 必须由本模块统一给出，而不是在每个脚本里内联一遍 ——
重复实现迟早漂移（这个仓库已经被"重复实现漂移"咬伤过多次）。

与硬件 bake-off 分数 ``compute_quality_score``
(``src/ao_shaping/utils/image/beam_metrics.py``, ``0.3·AR + 0.4/(1+CV) + 0.3·EE``)
的区别：T 系列只有 2 分量、没有 AR (aperture ratio) 维度；真源是 3 分量、含 AR，
用于硬件 bake-off / 采纳判据。两者**不是同一个量**，行文不要混用。

另注意：论文 §7.7 里 T2 的**优化损失** ``0.4·CV + 0.6·(1−EE)`` 与这里的
composite 分数 ``0.5·EE + 0.5/(1+CV)`` 也不是同一个量 —— 前者用于优化（越小越
好），后者用于报告（越大越好）。

用法 (脚本在本目录下运行, 本目录自动在 ``sys.path`` 上)::

    from _common import shaping_score
"""
from __future__ import annotations

__all__ = ["shaping_score"]


def shaping_score(ee: float, cv: float) -> float:
    """T 系列 composite 分数: ``0.5·EE + 0.5/(1+CV)`` (越大越好)。

    Args:
        ee: Encircled energy (0~1)。
        cv: In-box coefficient of variation (>=0)。

    Returns:
        与 T1/T2 历史发布数字逐位一致的分数值 (同一输入下不改变结果口径)。
    """
    return 0.5 * ee + 0.5 / (1.0 + cv)
