---
title: closed-loop（基于响应矩阵的闭环波前优化）
created: 2026-10-13
updated: 2026-10-13
type: command
tags: [command]
source: src/ao_shaping/runners/slm/zernike_matrix_runner.py:1580
---

## 一句话

基于响应矩阵的闭环波前优化

## 入口函数

`runners.slm.zernike_matrix_runner.closed_loop_run`（`src/ao_shaping/runners/slm/zernike_matrix_runner.py:1580`）

```python
closed_loop_run(ctx, params)
```

## docstring（原文节选）

```
基于响应矩阵的闭环波前优化

    载入已保存的Zernike响应矩阵 (.h5), 恢复硬件参数,
    使用指定控制律进行闭环优化, 保存优化结果。
```

## 调用链分析（自动）

**硬件**：（无直接硬件，纯计算/仿真）

## 直接调用（1 级，按调用次数）

- `_expand_to_noll`
- `_measure_wfs_step`
- `ao_shaping.algorithm.signal_processing.controller.HardwareConfig` *(未解析/外部)*
- `ao_shaping.algorithm.signal_processing.controller.LoopConfig` *(未解析/外部)*
- `ao_shaping.drivers.wfs.MlaRes` *(未解析/外部)*
- `ao_shaping.drivers.wfs.ThorlabWFS` *(未解析/外部)*
- [[functions/with-params|`with_params`]]
- [[functions/path-dirs|`get_timestamp_str`]]
- `from_response_matrix`

## 谁调用它

- [[entry/cli|`cli`]]（`cli.add_command` 注册为 `closed-loop`）

## 相关页面

- [[entry/cli]]
