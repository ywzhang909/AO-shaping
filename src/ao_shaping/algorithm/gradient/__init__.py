"""梯度优化器 (Adam 家族、加速器)。

子模块可按全路径直接导入, 例如::

    from ao_shaping.algorithm.gradient.adam import Adam, AdamW, AdaMOD
    from ao_shaping.algorithm.gradient.acceleration import _njit

本 ``__init__`` 刻意*不*重新导入子模块: 只有 docstring 的包 init 可以避免
循环导入级联 (子模块之间互相导入, 也会导入顶层 ``ao_shaping.algorithm`` 门面,
在这里急加载它们会让顶层 ``__init__`` 在初始化中途被执行)。所有公开名字改为
经由顶层 ``ao_shaping.algorithm`` 门面暴露。
"""
