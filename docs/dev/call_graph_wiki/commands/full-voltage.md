---
title: full-voltage（AsyncMicroDM 全量交替电压下发）
created: 2026-10-13
updated: 2026-10-13
type: command
tags: [command]
source: src/ao_shaping/runners/micro_drive/voltage_runner.py:651
---

## 一句话

AsyncMicroDM 全量交替电压下发

## 入口函数

`runners.micro_drive.voltage_runner.full_voltage_run`（`src/ao_shaping/runners/micro_drive/voltage_runner.py:651`）

```python
full_voltage_run(ctx, params)
```

## docstring（原文节选）

```
全量交替电压下发工具 (AsyncMicroDM)

    所有单元同时、均匀地在 0V 和指定电压之间交替。基于 asyncio 异步驱动。

    Examples:

        # 全部 26 台控制器 (.101~.126) 全部单元交替 20V, 1Hz, 持续运行直到 Ctrl+C
        python -m ao_shaping.runners.micro_drive.voltage_runner full-voltage --voltage 20

        # 两个控制器, 30V, 2Hz, 持续 10 秒
        python -m ao_shaping.runners.micro_drive.voltage_runner full-voltage             --ips 192.168.0.101,192.168.0.102 --voltage 30 --freq 2.0 --duration 10
```

## 调用链分析（自动）

**硬件**：（无直接硬件，纯计算/仿真）

## 直接调用（1 级，按调用次数）

- `_amain`
- `_enable_debug`
- [[functions/with-params|`with_params`]]

## 谁调用它

- [[entry/cli|`cli`]]（`cli.add_command` 注册为 `full-voltage`）

## 相关页面

- [[entry/cli]]
