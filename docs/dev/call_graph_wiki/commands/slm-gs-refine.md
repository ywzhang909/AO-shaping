---
title: slm-gs-refine（GS 预整形 + 自由相位 SPGD 细化）
created: 2026-10-13
updated: 2026-10-13
type: command
tags: [command]
source: src/ao_shaping/runners/slm/gs_refine_runner.py:85
---

## 一句话

GS 预整形 + 自由相位 SPGD 细化

## 入口函数

`runners.slm.gs_refine_runner.run`（`src/ao_shaping/runners/slm/gs_refine_runner.py:85`）

```python
run(ctx, params)
```

## docstring（原文节选）

```
SLM GS 预整形 + 自由相位 SPGD 细化 (SLM + CCD 闭环)。

    把远场光斑整形为均匀方形。先用 GS 算一个开环预整形相位 (只在实测优于平场时
    采用), 再用无感知 SPGD 细化自由相位。

    目标函数与仿真一致: ``0.5*PIB + 0.5/(1+CV)``, 读取真实 CCD 帧计算。
```

## 调用链分析（自动）

**硬件**：（无直接硬件，纯计算/仿真）

**结果/调试落盘**：`save_gs_refine_results`

## 直接调用（1 级，按调用次数）

- [[functions/runner-common|`build_gs_refine_config`]]
- [[functions/runner-common|`save_gs_refine_results`]]
- `ao_shaping.utils.cli.exposure.resolve_exposure`
- [[functions/with-params|`with_params`]]
- `ao_shaping.utils.io.cli_helpers.resolve_debug`

## 谁调用它

- [[entry/cli|`cli`]]（`cli.add_command` 注册为 `slm-gs-refine`）

## 相关页面

- [[entry/cli]]
