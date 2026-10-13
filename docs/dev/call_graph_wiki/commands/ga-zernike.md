---
title: ga-zernike（遗传算法 Zernike 波前优化）
created: 2026-10-13
updated: 2026-10-13
type: command
tags: [command]
source: src/ao_shaping/runners/zernike_search_runner.py:129
---

## 一句话

遗传算法 Zernike 波前优化

## 入口函数

`runners.zernike_search_runner.run`（`src/ao_shaping/runners/zernike_search_runner.py:129`）

```python
run(ctx, params, wfs, slm)
```

## docstring（原文节选）

```
使用遗传算法优化Zernike系数进行波前校正.
```

## 调用链分析（自动）

**硬件**：WFS 波前传感器

**调用的优化器/算法**：

- `_blend_crossover`
- `_gaussian_mutation`
- `_tournament_selection`
- [[functions/optimizer-ga|`optimizer_ga`]]

## 直接调用（1 级，按调用次数）

- [[functions/with-params|`with_params`]]
- `_persist_results`
- [[functions/optimizer-ga|`optimizer_ga`]]
- [[functions/debug-mode|`get_debug_mode`]]

## 谁调用它

- [[entry/cli|`cli`]]（`cli.add_command` 注册为 `ga-zernike`）

## 相关页面

- [[entry/cli]]
- [[functions/optimizer-ga|`optimizer_ga`]]
