# Chapter 5: Micro-DM 1300（R50Power 多控制器）

## Core Idea
Micro-DM 1300 是"36×36 单元、26 个 R50Power 控制器、每控 50 通道"的分布式执行器；AO 闭环下驱动的第一约束不是"能通"而是**确定性延迟 + 零 GC 抖动**——热路径里禁止重连、禁止每帧分配、drain 必须硬超时。

## Frameworks Introduced
- **AO 闭环驱动五原则（区别于通用网络编程）**
  - When to use: 修改 Micro-DM 驱动或任何 AO 实时链路。
  - How: (1) 确定性延迟——单次发送+固定超时，失败立即返回交上层决策，不做重连/退避；(2) 零 GC 抖动——预分配缓冲原地更新电压，不每帧建 `bytes`；(3) 背压安全——`drain` 硬超时（8ms），超时标记丢帧不阻塞节拍；(4) 多控制器同步——`gather`+独立超时，部分失败不拖慢整体帧率；(5) 线程模型——直接嵌 AO 主事件循环，不建后台线程桥接。
- **零分配稳态（LUT + 预分配）**
  - When to use: 1kHz 量级热路径。
  - How: 电压量化为 12-bit LUT 索引，LUT 预计算 `index → big-endian 2-byte payload`；`fill_buffer(voltages, buf)` 把 LUT 项原地拷进预分配的 50 通道输出缓冲（无分配）。对比：每帧 50 次浮点乘除+numpy clip 会触发 GC 毫秒级暂停；整数查表约 10× 更快且无浮点单元竞争。
- **TCP 帧协议**
  - When to use: 与 R50 控制器通信。
  - How: 帧 = `HEADER(0xAA 0xBB)` + `CMD(1)` + `PAYLOAD(100)` + `FOOTER(0xCC 0xDD)`；关键命令 `0x09`=按数组设全部电压、`0x06`=继电器开、`0x07`=继电器关；电压域 -20.0 ~ 120.0 V，每控 50 通道。连接时 `TCP_NODELAY` 禁 Nagle，消除延迟 ACK 对小火花的缓冲延迟。
- **wiring map 坐标映射**
  - When to use: 在 (x,y) 阵列坐标、控制器 IP、通道号之间换算。
  - How: 自动从 `libs/micro_drive1300/wiring_map.json` 加载控制器 IP 与通道映射；`get_channel_by_xy(x, y)` 按 36×36 阵列查；`get_channel_by_ip_position(ip_suffix, payload_position)` 反向查物理位置；类型安全 `WiringMap`/`ChannelEntry`/`ChannelInfo` dataclass；`use_wiring_map=False` 回退默认 IP。

## Key Concepts
- **R50Power 控制器**: 每控 50 通道、-20V~120V；全系统 26 个控（IP 101-126）、共 1300 单元（36×36=1296 有效）。
- **wiring map**: `位置序号(硬件ID) → 36×36 行列 → IP组+序号(0-based) → 物理执行器` 的映射链；序号 0-49，位置序号≠Excel 行号。
- **AsyncMicroDM**: asyncio 驱动，LUT 预查表 + `StreamReader/StreamWriter` 非阻塞 TCP + 内联命令字节快路径。
- **prebuild_command / build_frame_commands**: 把电压帧一次性编码为命令字节；热循环用 `send_bytes`/`send_frame_commands` 直接重放——零编码、零分配、仅 `write`+`drain`。
- **SEND_DRAIN_TIMEOUT=0.008**: 每控 8ms 硬上限（1kHz 预算内）。
- **同步/异步双模式**: `open()/close()` 同步桥接；原生 `connect_all()/send_frame()/shutdown()`。

## Mental Models
- Use "热路径里一切都要预分配" 的心智：每帧新建对象=GC 暂停=闭环抖动，提前把缓冲/帧/查表都备好。
- 把 控制器 理解为 26 条独立车道：一条超时只丢它那一条，`gather` 独立超时不让慢车道拖停整帧。
- Think of 0x09 命令 as "整帧电压快照"：预构建后重放，等价于录像回放。

## Anti-patterns
- **热路径里自动重连/重试/指数退避**：破坏确定性延迟；失败应立即返回交上层决策。
- **每帧分配 bytes/命令**：触发 Python GC 毫秒级暂停；必须预分配+原地更新。
- **`drain()` 无限等待**：背压时阻塞整个闭环节拍；必须 8ms 硬超时标记丢帧。
- **`gather` 等全部控制器完成**：一个慢/挂死的控拖慢整体帧率；要独立超时，部分失败继续。
- **后台线程 + Loop 桥接**：跨线程通信开销；应直接嵌 AO 主事件循环。
- **Nagle 未关**：延迟 ACK 缓冲小火花，破坏低延迟；连接时设 `TCP_NODELAY`。

## Code Examples
```python
# 内联快路径（零编码零分配重放）
dm = AsyncMicroDM(ips=["192.168.0.101", "192.168.0.102"])
dm.open()
dm.send_voltages(np.zeros(dm.DM_Num))   # 同步桥接
dm.close()

# 原生 asyncio + 预构建命令重放
await dm.connect_all()
cmd_off = dm.build_frame_commands(np.zeros(dm.DM_Num))
cmd_on  = dm.build_frame_commands(np.full(dm.DM_Num, 20.0))
await dm.send_frame_commands(cmd_off)    # 仅 write + drain
await dm.send_frame_commands(cmd_on)
await dm.shutdown()
```
- 演示：同步/异步双模式 + 内联命令字节快路径（热循环零编码）。

```python
# 工厂创建 + wiring map 坐标查询
from ao_shaping.drivers.dm._registry import create_dm
dm = create_dm("asyn_micro", ips=["192.168.0.101"])

info = dm.get_channel_by_xy(x=1, y=3)
print(info.ip_address, info.payload_position, info.physical_label)
```
- 演示：注册表工厂创建 + 36×36 阵列坐标到通道的双向查询。

## Reference Tables
| 协议常量 | 值 | 含义 |
|---|---|---|
| `HEADER` / `FOOTER` | `0xAA 0xBB` / `0xCC 0xDD` | 帧头/帧尾 |
| `CMD_SET_ALL_VOLTAGE_BY_ARR` | `0x09` | 按数组设全部电压 |
| `CMD_RELAY_ON` / `OFF` | `0x06` / `0x07` | 继电器开/关 |
| `VOLTAGE_MIN` / `MAX` | -20.0 / 120.0 | 电压域 (V) |
| `MAX_CHANNELS` | 50 | 每控通道数 |
| `SEND_DRAIN_TIMEOUT` | 0.008 s | 每控 drain 硬超时 |

| wiring map 列 | 含义 |
|---|---|
| A 位置序号 | 硬件物理 ID（1-1296） |
| B/C 36×36 行/列 | 网格坐标（0-35） |
| D IP 组 | 控制器 IP 后缀（101-126） |
| E 序号 | 通道号（0-based, 0-49） |
| G/H 引脚/连接器 | 物理引脚号 / 连接器编号 |

| 图像文件命名 | 格式 |
|---|---|
| 单元图 | `192.168.0.{IP组}-{序号:03d}.png` |

## Key Takeaways
1. 热路径五原则：确定性延迟、零 GC、drain 硬超时、gather 独立超时、无后台线程桥接。
2. 预分配 + LUT 查表 + 原地拷贝，杜绝每帧分配。
3. 帧协议 `AA BB + CMD + PAYLOAD + CC DD`，0x09 设全电压，Nagle 必关。
4. 26 控 × 50 通道 = 1300 单元；wiring map 做 (x,y)↔IP↔通道换算。
5. 同步 `send_voltages` / 异步 `send_frame_commands` 双入口，热循环用预构建重放。

## Connects To
- **ch06 (驱动层)**: Micro-DM 是 `DM` 统一接口的一个实现（`--dm-type micro`/`asyn_micro`）。
- **ch02 (SLM-200)**: 同为空间执行器，但 SLM 走相位灰度，Micro-DM 走电压通道。
