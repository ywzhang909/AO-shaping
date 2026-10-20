---
title: greedy-zernike（贪婪局部搜索 Zernike 波前优化）
created: 2026-10-13
updated: 2026-10-13
type: command
tags: [command]
source: src/ao_shaping/runners/zernike_search_runner.py:202
---

## 一句话

贪婪局部搜索 Zernike 波前优化

## 入口函数

`runners.zernike_search_runner.greedy_zernike_run`（`src/ao_shaping/runners/zernike_search_runner.py:202`）

```python
greedy_zernike_run(ctx, params, wfs, slm)
```

## docstring（原文节选）

```
Zernike波前优化器 - 贪婪局部搜索 / 启发式搜索

    通过SLM进行波前校正，最小化WFS测量的波前RMS值。
    ``--algorithm spgd`` 使用贪婪局部搜索:
    1. 随机初始化N个位置，选取最优作为起始点
    2. 每次迭代采样n个随机扰动方向
    3. 评估所有候选(当前位置+n个扰动)，选择最优
    其他取值 (ga/pso/sa/hc/rs/cem/de) 使用 algorithm/heuristic 的启发式搜索。

    调试模式: ``main.py --debug greedy-zernike`` / 本命令 ``--debug`` / ``DEBUG=1``。
```

## 调用链分析（自动）

**硬件**：WFS 波前传感器

**调用的优化器/算法**：

- `_generate_random_direction`
- `ao_shaping.algorithm.heuristic.search.heuristic_algorithm_choices`
- [[functions/run-heuristic-search|`run_heuristic_search`]]
- [[functions/optimizer-greedy|`optimizer_greedy`]]
- `create_heuristic`

## 直接调用（1 级，按调用次数）

- [[functions/with-params|`with_params`]]
- `_persist_results`
- [[functions/optimizer-greedy|`optimizer_greedy`]]
- `ao_shaping.utils.io.cli_helpers.resolve_debug`

## 谁调用它

- [[entry/cli|`cli`]]（`cli.add_command` 注册为 `greedy-zernike`）

## 相关页面

- [[entry/cli]]
- [[functions/run-heuristic-search|`run_heuristic_search`]]
- [[functions/optimizer-greedy|`optimizer_greedy`]]
