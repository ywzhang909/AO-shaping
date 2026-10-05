"""启发式 / 黑盒优化器 (GA、PSO、SA、HC、Random、CEM、DE)。

子模块可按全路径直接导入, 例如::

    from ao_shaping.algorithm.heuristic.ga import GeneticAlgorithm, GAParams
    from ao_shaping.algorithm.heuristic.heuristic_base import HeuristicOptimizer

本 ``__init__`` 刻意*不*重新导入子模块: 只有 docstring 的包 init 可以避免
循环导入级联 (子模块会导入共享的 ``heuristic_base`` 以及顶层
``ao_shaping.algorithm`` 门面, 在这里急加载它们会让顶层 ``__init__`` 在初始化
中途被执行)。所有公开名字都经由顶层 ``ao_shaping.algorithm`` 门面暴露。

具体优化器通过声明 ``_optimizer_type`` 把自己注册到 ``HeuristicOptimizer``
(见 ``heuristic_base``), 因此导入本包下的任何东西都会填好注册表, 而
``__init__`` 无需知道那些具体模块。
"""
# 引导式变异 (GM) 实现在 ``gm.py`` 中; 两种算子定义各自用在哪里, 见该模块的
# docstring。
# ref https://www.researching.cn/ArticlePdf/m00009/2023/52/6/0629002.pdf