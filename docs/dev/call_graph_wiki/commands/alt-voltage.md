---
title: alt-voltage（R50 交替电压下发 + ADC 同步采集）
created: 2026-10-13
updated: 2026-10-13
type: command
tags: [command]
source: src/ao_shaping/runners/micro_drive/voltage_runner.py:211
---

## 一句话

R50 交替电压下发 + ADC 同步采集

## 入口函数

`runners.micro_drive.voltage_runner.run`（`src/ao_shaping/runners/micro_drive/voltage_runner.py:211`）

```python
run(ctx, params)
```

## docstring（原文节选）

```
交替电压下发工具

    在 0V 和指定电压之间循环交替发送到 R50Power 控制器的指定单元。
    可选同步采集 NI DAQ ADC 电压信号，结果自动保存为 CSV。

    Examples:

        # 全部 50 个通道交替 20V, 1Hz, 持续运行直到 Ctrl+C
        python -m ao_shaping.runners.micro_drive.voltage_runner --ip 192.168.0.101 --voltage 20

        # 通道 0-5 交替 30V, 2Hz, 持续 10 秒
        python -m ao_shaping.runners.micro_drive.voltage_runner --ip 192.168.0.101 --voltage 30 --freq 2.0 --duration 10 --channels 0,1,2,3,4,5

        # 全部通道, 0.5Hz, 跳过 ping 和自动上电
        python -m ao_shaping.runners.micro_drive.voltage_runner --ip 192.168.0.101 --voltage 15 --freq 0.5 --no-ping-first --no-relay-on

        # 同步采集 ADC (Dev1/ai0, 5kHz, 10 samples/read)
        python -m ao_shaping.runners.micro_drive.voltage_runner --ip 192.168.0.101 --voltage 20 --adc-enabled --adc-device Dev1 --adc-channel ai0
```

## 调用链分析（自动）

**硬件**：（无直接硬件，纯计算/仿真）

## 直接调用（1 级，按调用次数）

- `_banner`
- `_duration_desc`
- `_enable_debug`
- `_parse_channels`
- `_resolve_port`
- `_safe_shutdown`
- `ao_shaping.drivers.dm.micro.R50Controller` *(未解析/外部)*
- [[functions/with-params|`with_params`]]
- `ao_shaping.utils.io.network.ping_reachable`

## 谁调用它

- [[entry/cli|`cli`]]（`cli.add_command` 注册为 `alt-voltage`）

## 相关页面

- [[entry/cli]]
