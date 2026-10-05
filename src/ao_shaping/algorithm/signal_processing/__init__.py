"""信号处理 / 光束整形算法。

包含 GS / 可微分整形 / 相位包裹 / 波前工具函数, 控制律闭环, 以及共享的迭代
优化器基类。

子模块可按全路径直接导入, 例如::

    from ao_shaping.algorithm.signal_processing.gerchberg_saxton import gerchberg_saxton
    from ao_shaping.algorithm.signal_processing.differentiable_beam import DifferentiableBeamOptimizer
    from ao_shaping.algorithm.signal_processing.iterative_base import IterativeOptimizer

本 ``__init__`` 刻意*不*重新导入子模块: 只有 docstring 的包 init 可以避免
循环导入级联 (子模块会导入顶层 ``ao_shaping.algorithm`` 门面并互相导入,
在这里急加载它们会让顶层 ``__init__`` 在初始化中途被执行)。所有公开名字都
经由顶层 ``ao_shaping.algorithm`` 门面暴露。
"""
