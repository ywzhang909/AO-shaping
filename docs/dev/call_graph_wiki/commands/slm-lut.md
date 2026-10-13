---
title: slm-lut（SLM 灰度-相位 LUT 标定（衍射效率反演））
created: 2026-10-13
updated: 2026-10-13
type: command
tags: [command]
source: src/ao_shaping/tools/slm/calibration/lut_runner.py:488
---

## 一句话

SLM 灰度-相位 LUT 标定（衍射效率反演）

## 入口函数

`tools.slm.calibration.lut_runner.run`（`src/ao_shaping/tools/slm/calibration/lut_runner.py:488`）

```python
run(params)
```

## docstring（原文节选）

```
SLM gray-to-phase LUT calibration.

    Drives the Santec SLM-200 in half-screen blazed-grating mode, scans gray
    depth or offset, measures +1 diffraction efficiency of both halves (drift-
    canceled ratio), inverts to a phase LUT, and saves artifacts.
```

## 调用链分析（自动）

**硬件**：（无直接硬件，纯计算/仿真）

**结果/调试落盘**：`save_lut`

## 直接调用（1 级，按调用次数）

- [[functions/slm-lut|`depth_pattern`]]
- `_check_spot_drift`
- `_measure_power`
- `ao_shaping.utils.image.hardware_utils.open_camera`
- [[functions/slm-lut|`stack_halves`]]
- `_joint_exposure_settle`
- `_locate_spots`
- `ao_shaping.drivers.slm.santec.Santec` *(未解析/外部)*
- [[functions/with-params|`with_params`]]
- [[functions/slm-lut|`build_inverse_lut`]]
- [[functions/slm-lut|`invert_depth_scan`]]
- [[functions/slm-lut|`invert_offset_scan`]]
- [[functions/slm-lut|`offset_pattern`]]
- [[functions/slm-lut|`save_lut`]]

## 谁调用它

- [[entry/cli|`cli`]]（`cli.add_command` 注册为 `slm-lut`）

## 相关页面

- [[entry/cli]]
