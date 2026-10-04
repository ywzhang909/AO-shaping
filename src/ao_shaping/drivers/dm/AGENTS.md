# dm/ - 变形镜驱动

变形镜 (Deformable Mirror) 驱动模块。

## 结构

```
dm/
├── base.py              # DM 抽象基类 (DM/ABC)
├── hadamard_dm.py       # Hadamard DM
├── zernike_dm.py        # Zernike DM
├── _adjacency.py        # 致动器邻接矩阵单一加载器
├── _registry.py         # DM 注册表
├── __init__.py
├── Drv_UDPST/           # NLight SDK (x64/x86, Debug/Release)
├── micro/               # Micro-DM (R50Power) 子包
│   ├── driver.py            # MicroDM / R50Controller / voltages_to_payload
│   ├── asyn_driver.py       # AsyncMicroDM (asyncio 版本)
│   ├── constants.py         # 帧协议 / 电压→载荷编码
│   ├── micro_constants.py   # 型号专属硬件规格
│   └── wiring_map.py        # 接线表类型化 JSON schema
└── nlight/              # NLight 子包
    ├── driver.py            # NLight / NLightParams / NLIGHT_CONFIG
    ├── constants.py         # 驱动层常量 (协议 / 编码 / 端点)
    ├── nlight_constants.py  # 型号专属硬件规格
    ├── udp.py               # DMUdp —— UDP 批量下发
    └── sdk.py               # DMSdk —— Drv_UDPST.dll ctypes 绑定
```

实体设备按 [santec 的子包标准](../slm/santec/) 组织: 每个设备一个子包, 内含
`driver.py` + `constants.py` + `<型号>_constants.py`, `__init__.py` 做 re-export。
旧路径 `dm/MicroDM.py` / `dm/NLight.py` / `dm/asyn_micro_dm.py` 已降级为
**纯 re-export shim** (向后兼容, 不再新增符号) —— 新代码一律 import 子包。

> ⚠️ **`micro/__init__.py` 的异步驱动是惰性的。** `@register_dm("asyn_micro")`
> 是 import 副作用, 而 `runners/runner_common.py` 先快照 `list_dm_types()`
> (`DM_TYPES_PRE_ASYN_MICRO`, 6 项) 再 import 异步模块 (`DM_TYPES`, 7 项) ——
> 这个两步顺序决定 `--dm_type` 的 help 文本。改成 eager 会塌成一步,
> `--dm_type` 列表从 7 项静默缩成 6 项。故 `micro/__init__.py` 用
> `install_lazy_attrs` 解析异步符号, 而 `runner_common` 直接 import
> `ao_shaping.drivers.dm.micro.asyn_driver` 子模块。契约由
> `tests/ao_shaping/runners/test_cli_contract_freeze.py` 逐字节冻结。

> 注: `simulateDM.py` (模拟 DM) 已移除, 模拟功能见 [sim/](../sim/AGENTS.md) 及 `mock_devices.py`。

## 关键类

| 类 | 文件 | 说明 |
|----|------|------|
| `DM` | `base.py` | DM 抽象基类 |
| `NLight` | `nlight/driver.py` | NLight 变形镜 (主用) |
| `DMUdp` / `DMSdk` | `nlight/udp.py` / `nlight/sdk.py` | NLight 两条传输通道 |
| `MicroDM` | `micro/driver.py` | Micro-DM |
| `R50Controller` | `micro/driver.py` | 单台 R50Power 同步 TCP 客户端 |
| `AsyncMicroDM` | `micro/asyn_driver.py` | Micro-DM 异步版 |
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

`NLight` (`dm/nlight/driver.py`) 是主要的 NLight 变形镜驱动, 继承 `DM`, 需实现 `_apply_voltages` 等抽象方法。两条互补通道: `udp.py::DMUdp` 走 UDP datagram 批量下发 64 通道电压, `sdk.py::DMSdk` 走 `Drv_UDPST/` 下的 C SDK 做高压开关与读回。
