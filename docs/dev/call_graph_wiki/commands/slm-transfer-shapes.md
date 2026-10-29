---
title: slm-transfer-shapes（多形状物理预测模型闭环整形（从零训练 → 逐形状整形 → 共同重放 → 记录））
created: 2026-10-13
updated: 2026-10-13
type: command
tags: [command]
source: src/ao_shaping/runners/slm/transfer_shapes_runner.py:721
---

## 一句话

多形状物理预测模型闭环整形（从零训练 → 逐形状整形 → 共同重放 → 记录）

## 入口函数

`runners.slm.transfer_shapes_runner.run`（`src/ao_shaping/runners/slm/transfer_shapes_runner.py:721`）

```python
run(out_dir, shapes, side, period, window_span, window_step, zero_radius, capture_grid, capture_count, fit_iterations, fit_span, amplitude_center_lr, … (+29 个))
```

## docstring（原文节选）

```
物理预测模型闭环整形：从零训练模型 + 目标整形，逐形状执行。

    典型用法::

        python src/ao_shaping/main.py slm-transfer-shapes \
            --out data/debug/slm_transfer/session \
            --shapes square --shapes triangle --execute
```

## 调用链分析（自动）

**硬件**：（无直接硬件，纯计算/仿真）

## 直接调用（1 级，按调用次数）

- `_json`
- `_measured_peak_rates`
- `_rebuild_records`
- `_seed_kind`
- `_validate_args`
- `ao_shaping.tools.slm.transfer.protocol.derive_replay_exposures`
- `run_stages`

## 谁调用它

- [[entry/cli|`cli`]]（`cli.add_command` 注册为 `slm-transfer-shapes`）

## 相关页面

- [[entry/cli]]
