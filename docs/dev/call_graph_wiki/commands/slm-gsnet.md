---
title: slm-gsnet（SLM freeform 方形整形（spgd/heuristic/train））
created: 2026-10-13
updated: 2026-10-13
type: command
tags: [command]
source: src/ao_shaping/runners/slm/gsnet_runner.py:379
---

## 一句话

SLM freeform 方形整形（spgd/heuristic/train）

## 入口函数

`runners.slm.gsnet_runner.run`（`src/ao_shaping/runners/slm/gsnet_runner.py:379`）

```python
run(ctx, run)
```

## docstring（原文节选）

```
Square far-field shaping via FREEFORM per-pixel SLM phase.

    The phase basis is always freeform (per-pixel) — the only DOF that can
    synthesise a true square far-field (low-order Zernike cannot). Without a
    subcommand, the SPGD (gradient) search runs.

    Subcommands: ``spgd`` (gradient search), ``heuristic`` (black-box search)
    and ``train`` (offline FourierGSNet training on the recorded debug corpus —
    the only one that touches no hardware).
```

## 调用链分析（自动）

**硬件**：（无直接硬件，纯计算/仿真）

## 直接调用（1 级，按调用次数）

- [[functions/with-params|`with_params`]]

## 谁调用它

- [[entry/cli|`cli`]]（`cli.add_command` 注册为 `slm-gsnet`）

## 相关页面

- [[entry/cli]]
