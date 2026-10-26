---
title: slm-pib（SLM Zernike 整形（spgd/heuristic 子命令））
created: 2026-10-13
updated: 2026-10-13
type: command
tags: [command]
source: src/ao_shaping/runners/slm/slm_shaping_runner.py:464
---

## 一句话

SLM Zernike 整形（spgd/heuristic 子命令）

## 入口函数

`runners.slm.slm_shaping_runner.run`（`src/ao_shaping/runners/slm/slm_shaping_runner.py:464`）

```python
run(ctx, run)
```

## docstring（原文节选）

```
SLM shaping with a Santec SLM driven by Zernike coefficients.

    Without a subcommand, the SPGD (gradient) search runs.

    Subcommands: ``spgd`` / ``heuristic`` shape an arbitrary target shape
    (Power-in-Bucket family); ``square`` optimises a uniform square far field.
```

## 调用链分析（自动）

**硬件**：（无直接硬件，纯计算/仿真）

## 直接调用（1 级，按调用次数）

- [[functions/with-params|`with_params`]]

## 谁调用它

- [[entry/cli|`cli`]]（`cli.add_command` 注册为 `slm-pib`）

## 相关页面

- [[entry/cli]]
