# dm/ - 变形镜驱动

变形镜 (Deformable Mirror) 驱动模块。

## 结构

```
dm/
├── base.py              # DM 抽象基类 (DM/ABC)
├── NLight.py            # NLight DM 驱动 (SDK + UDP)
├── asyn_micro_dm.py     # 异步 Micro-DM
├── hadamard_dm.py       # Hadamard DM
├── MicroDM.py           # Micro-DM 驱动
├── zernike_dm.py        # Zernike DM
├── _registry.py         # DM 注册表
├── __init__.py
├── Drv_UDPST/           # NLight SDK (x64/x86, Debug/Release)
├── micro/               # Micro-DM 子包
└── nlight/              # NLight 子包
```

> 注: `simulateDM.py` (模拟 DM) 已移除, 模拟功能见 [sim/](../sim/AGENTS.md) 及 `mock_devices.py`。

## 关键类

| 类 | 文件 | 说明 |
|----|------|------|
| `DM` | `base.py` | DM 抽象基类 |
| `NLight` | `NLight.py` | NLight 变形镜 (主用) |
| `MicroDM` | `MicroDM.py` | Micro-DM |
| `HadamardDM` | `hadamard_dm.py` | Hadamard DM |
| `ZernikeDM` | `zernike_dm.py` | Zernike DM |

## DM 接口

`DM` 抽象基类定义 (`dm/base.py`):

| 方法 | 说明 |
|------|------|
| `transform(cmd)` | 将命令转换为 DM 值 |
| `send(cmd)` | 发送命令到 DM (委托给 send_voltages) |
| `send_voltages(vs, wait_time_s)` | 发送电压数组 (含安全斜坡) |
| `open()` | 打开 DM 连接 |
| `close()` | 关闭 DM 连接 |
| `is_connected()` | 检查连接状态 |
| `get_hardware_info()` | 获取硬件信息 |
| `get_actuator_positions()` | 获取致动器位置 |
| `is_reachable()` | 检查硬件是否可达 (classmethod, 子类实现) |
| `transform_voltage(cmd)` | [-1,1] → [V_Min, V_Max] 电压转换 |
| `set_channel_voltage(ch, v)` | 设置单通道电压 |
| `set_all_voltage_by_arr(voltages)` | 按数组设置所有电压 |
| `check_dm_unit_grad_safe(vs)` | 检查相邻电压差是否安全 |

### NLight DM

`NLight` (`dm/NLight.py`) 是主要的 NLight 变形镜驱动, 继承 `DM`, 需实现 `_apply_voltages` 等抽象方法。使用 UDP 通信 (SOCK_STREAM 到 192.168.6.10:1001), SDK 位于 `Drv_UDPST/` 目录。
