---
title: hadamard-matrix（Hadamard 响应矩阵标定（并行扰动））
created: 2026-10-13
updated: 2026-10-13
type: command
tags: [command]
source: src/ao_shaping/runners/matrix_runner.py:274
---

## 一句话

Hadamard 响应矩阵标定（并行扰动）

## 入口函数

`runners.matrix_runner.hadamard_matrix_run`（`src/ao_shaping/runners/matrix_runner.py:274`）

```python
hadamard_matrix_run(ctx, params, wfs, wfs_drv)
```

## docstring（原文节选）

```
获取Hadamard响应矩阵

    支持 N 次正负交替循环测量 + M 次 WFS 读取取平均 + 方差跟踪 + 逆矩阵计算。

    Hadamard模式数量 = mode_order²，例如:
    - mode_order=8: 64个模式
    - mode_order=16: 256个模式
    - mode_order=32: 1024个模式
```

## 调用链分析（自动）

**硬件**：（无直接硬件，纯计算/仿真）

## 直接调用（1 级，按调用次数）

- [[functions/with-params|`with_params`]]
- `ao_shaping.utils.wavefront.wfs_utils.flatten_slopes`
- `_open_wfs`
- `_refresh_wfs_reference`
- `ao_shaping.drivers.dm.hadamard_dm.HadamardDM` *(未解析/外部)*
- `ao_shaping.utils.wavefront.hadamard_calc.calc_n_hadamard_modes`

## 谁调用它

- [[entry/cli|`cli`]]（`cli.add_command` 注册为 `hadamard-matrix`）

## 相关页面

- [[entry/cli]]
