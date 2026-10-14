---
title: slm-model-in-loop（正向模型闭环方形整形（反复迭代））
created: 2026-10-13
updated: 2026-10-13
type: command
tags: [command]
source: src/ao_shaping/runners/slm/model_in_loop_runner.py:39
---

## 一句话

正向模型闭环方形整形（反复迭代）

## 入口函数

`runners.slm.model_in_loop_runner.run`（`src/ao_shaping/runners/slm/model_in_loop_runner.py:39`）

```python
run(ctx, params)
```

## docstring（原文节选）

```
SLM model-in-the-loop square shaping (SLM + CCD closed loop).

    Every round: fit the forward model's aberration from measured probe frames,
    then synthesise a square far-field spot with that aberration frozen. The
    fitted phase is only adopted if it measurably beats what is already held, and
    the final phase is kept only if it beats the flat baseline.

    
    Exit codes:
      0  a shaped phase was adopted (or flat was correctly kept)
      2  the geometry bake-off failed, or the bench stayed outside the model --
         the run refused to commit a phase. Use slm-gs-refine for that bench.
```

## 调用链分析（自动）

**硬件**：（无直接硬件，纯计算/仿真）

**结果/调试落盘**：`save_model_in_loop_results`

## 直接调用（1 级，按调用次数）

- [[functions/runner-common|`build_model_in_loop_config`]]
- [[functions/runner-common|`parse_point`]]
- [[functions/runner-common|`save_model_in_loop_results`]]
- [[functions/with-params|`with_params`]]
- [[functions/debug-mode|`get_debug_mode`]]

## 谁调用它

- [[entry/cli|`cli`]]（`cli.add_command` 注册为 `slm-model-in-loop`）

## 相关页面

- [[entry/cli]]
