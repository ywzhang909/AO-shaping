---
title: run_heuristic_search（启发式搜索统一驱动）
created: 2026-10-13
updated: 2026-10-13
type: function
tags: [algorithm,heuristic,ga,pso]
source: src/ao_shaping/algorithm/heuristic/search.py:113
---

## 签名

```python
run_heuristic_search(algorithm, evaluate, dim, iterations, bounds, x0, maximize, seed, pop_size, on_evaluate, should_stop) → `HeuristicSearchResult`
```

## docstring（原文）

```
用指定名字的启发式算法驱动 ``evaluate``。

    ``evaluate`` 必须返回**原始**目标值 (按调用方自己的单位/方向); 当
    ``maximize=True`` 时, 本驱动把 ``-value`` 交给 (做最小化的) 启发式, 从而
    搜索它的最大值。

    Args:
        algorithm: :data:`HEURISTIC_ALGORITHM_MAP` 中的一个 (大小写不敏感)。
        evaluate: 黑盒目标函数; 收到的是已裁剪到边界内的向量。
        dim: 问题维度 (例如 Zernike 项数)。
        iterations: 启发式迭代数/代数 (调用方把它们的 ``epochs`` 映射到这里;
            注意种群类方法每一代会做 ``pop_size`` 次求值)。
        bounds: 每个参数的 ``(低, 高)`` 裁剪/搜索边界。
        x0: 可选的初始向量 (GA 会把它纳入种群, 局部方法则用它作起点)。
        maximize: 为 True 时按 ``evaluate`` 上升搜索 (PIB 类), 为 False 时下降
            搜索 (RMS 类)。
        seed: 可选的随机种子。
        pop_size: 可选的种群规模 (GA/PSO/CEM/DE)。
        on_evaluate: 可选的 ``(x, value, evaluation_index)`` 回调, 在每次求值
            之后触发 -- 用它来追加 Recorder 行 / 刷新实时显示。
            ``evaluation_index`` 从 1 开始。
        should_stop: 可选的判定式; 它返回 True 时搜索中止并返回当前最好的结果
            (不抛异常)。

    Returns:
        HeuristicSearchResult, 含最佳向量、它的原始值以及历史。

    Raises:
        ValueError: 未知的 ``algorithm``。
```

## 直接调用（出边）

- `create_heuristic`

## 谁调用它（入边）

- [[functions/optimizer-greedy|`optimizer_greedy`]]
- [[functions/optimizer-rms-slm|`optimizer_rms_slm`]]

## 相关页面

- [[entry/cli]]
