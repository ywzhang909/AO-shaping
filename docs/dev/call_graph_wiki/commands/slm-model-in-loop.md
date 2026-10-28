---
title: slm-model-in-loop（物理预测模型闭环方形整形（反复迭代））
created: 2026-10-13
updated: 2026-10-13
type: command
tags: [command]
source: src/ao_shaping/runners/slm/model_in_loop_runner.py:60
---

## 一句话

物理预测模型闭环方形整形（反复迭代）

## 入口函数

`runners.slm.model_in_loop_runner.run`（`src/ao_shaping/runners/slm/model_in_loop_runner.py:60`）

```python
run(ctx, params)
```

## docstring（原文节选）

```
SLM model-in-the-loop shaping (SLM + CCD closed loop).

    Every round: fit the forward model's aberration from measured probe frames,
    then synthesise a target far-field spot with that aberration frozen. The
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

**结果/调试落盘**：`save_model_in_loop_config`、`save_model_in_loop_results`、`save_model_in_loop_config`

## 直接调用（1 级，按调用次数）

- `ao_shaping.runners.runner_common.save_model_in_loop_config`
- [[functions/runner-common|`build_model_in_loop_config`]]
- [[functions/runner-common|`parse_point`]]
- [[functions/runner-common|`save_model_in_loop_results`]]
- `ao_shaping.runners.slm.model_in_loop_session.apply_session_calibration`
- `ao_shaping.runners.slm.model_in_loop_session.calibrate_session`
- `ao_shaping.utils.cli.exposure.exposure_ms_legacy`
- `ao_shaping.utils.cli.exposure.resolve_exposure`
- [[functions/with-params|`with_params`]]
- `ao_shaping.utils.io.cli_helpers.resolve_debug`

## 谁调用它

- [[entry/cli|`cli`]]（`cli.add_command` 注册为 `slm-model-in-loop`）

## 相关页面

- [[entry/cli]]
