# Chapter 2: Santec SLM-200（LCOS-SLM）

## Core Idea
SLM-200 是"慢启动、双模式、128 内存槽"的设备：上电/切模式/设波长/写 EEPROM 每一步都有约 40 秒时序；所有相位图案先写入 1-128 号内存槽再指定显示，**正在显示的槽不可覆盖**——这是本设备一切驱动代码的第一约束。

## Frameworks Introduced
- **推荐调用流程（Open 后八步）**
  - When to use: 任何 USB 控制路径（内存模式）的程序骨架。
  - How: `SLM_Ctrl_Open` → 轮询 `ReadSU` 至非 `SLM_BS`（启动约 40 秒）→ `ReadVI` 读当前视频模式 → 必要时 `WriteVI` 切模式（再等 40 秒）→ `WriteWL(波长, phase)`（phase 单位 0.01π，200=2π；后再等 40 秒）→ `WriteAW` 存 EEPROM（约 40 秒，仅首次）→ `WriteGS` 设灰度 → `WriteMI` 写相位入槽 → `WriteDS(mem)` 显示 / `WriteDR` 连续播放。
- **内存模式 vs DVI 模式**
  - When to use: 需要随机访问 128 个已载图案、触发器控制、快速换图用内存模式（Video Mode=0）；直接灌 DVI 信号、与主机显示管线联动用 DVI 模式（Mode=1，走 `SLM_Disp_*` 函数族）。
  - How: 内存模式数据路径 `WriteMI/WriteMI_BMP/WriteMI_CSV`（槽号 1-128）+ `WriteDS`；DVI 模式数据路径 `SLM_Disp_Data/SLM_Disp_BMP/SLM_Disp_ReadBMP/SLM_Disp_ReadCSV`（按 DisplayNumber 1-8）。
- **显示表（Display Table）机制**
  - When to use: 多图案序列、A-B-B-A 验收、连续播放。
  - How: Table n 默认映射 Memory n（`WriteMZ` 恢复默认）；`WriteMT(table, mem)` 改映射、`WriteMR(start, end)` 设有效范围、`WriteMP(table)` 设起始表、`WriteMW(frames)` 设帧间隔（0-120 帧）；`WriteDR(order)` 连续播放（0 降序/1 升序）、`WriteDB` 停止；触发用 `WriteTI/WriteTM/WriteTC/WriteTS`（软件触发等同触发输入）。
- **设备标识读取降级链**
  - When to use: 多 SLM 共存、配置按 SN 归档（`save_config` 依赖 `_serial_number`）。
  - How: 按序尝试 `ReadSDO`(Drive+Option 各 16B) → `ReadSD`(Drive 16B) → `ReadSO`(Option 16B) → `ReadPS`(产品 SN 12 位, board=0/1) → `ReadLS`(LCOS SN ≤20 位) → `ReadPN`(Display Name ≤13 位) → `ReadVR`(64B 版本串 `DLL:x.x.x,Drive:xxxx,Option:xxxx,FPGA:xxxx`)。`ReadSD` 快速 open/close 后可能挂起，失败必须回退下一方法，挂死则 `Reboot` 后重连。

## Key Concepts
- **SLM Number**: Windows 自动分配的 USB 设备编号（1-8），`SLM_Ctrl_*` 全部以此寻址。
- **Display Number**: DVI 模式下的显示编号，与 SLM Number 不同源；多显示器时用 `SLM_Disp_Info2` 确认。
- **Memory Number**: 内存槽 1-128；**显示中的槽写入即失败**（官方脚注：先切走显示目标或先显示灰度）。
- **SLM_STATUS**: `SLM_OK=0, SLM_NG=1, SLM_BS=2(忙), SLM_ER=3(参数错)`；负值为 DVI/文件/USB 错误（`-101` 文件读、`-200` USB 未开、`-10002` 设备未找到查电源）。
- **灰度域**: 0-1023（10 bit），线性对应 0-2π 相位。
- **BMP/CSV 格式限制**: BMP 仅 8/24 bit，10 bit 数据必须走 `SLM_Disp_Data` 或 CSV；CSV 每行 `Y, Gray0, Gray1, ...`。
- **FT601**: SLM 的 USB 3.0 桥芯片，设备管理器须识别 `FTDI FT601 USB 3.0 Bridge Device`。
- **振幅耦合**: 1064nm 附近不同灰度值平场时光斑亮度随灰度周期性变化（周期 ≈2π ≈993 灰度值），SLM 固有特性。

## Mental Models
- Use "一切皆 40 秒" 的心智：启动、切模式、设波长、写 EEPROM 都按 40 秒量级预算，代码里必须轮询 `ReadSU` 而不是 `sleep(5)`。
- 把 内存槽 理解为 128 个只读显示屏的后台：先备货（WriteMI）再上架（WriteDS）；上架的货架不能直接换货，必须先撤下。
- Think of 显示表 as 播放列表：`WriteDS` 是切歌，`WriteDR` 是连续播放，触发是外部切歌按钮。

## Anti-patterns
- **覆盖正在显示的内存槽**：`WriteMI` 到显示中槽号会失败或静默无效；先 `WriteDS(其他槽)` 或 `set_grayscale(0)`。
- **连续两次写同一内存槽（内存模式）**：`display_memory(slot)` 对已在显示的槽是 no-op，LCOS 不刷新；连续写入必须换槽（本仓 `slm-gsnet` 在 2-125 槽随机选并排除当前显示槽，启动用 `get_displayed_memory_number()` 续接）。
- **跳过 ReadSU 轮询直接设参数**：40 秒内所有 `Write*` 都会失败。
- **把 uint16 灰度数组塞进按弧度解释的封装**：`create_phase_from_array()` 把输入当弧度（mod 2π 后 ×1023），灰度值会静默损坏；平场灰度必须 `np.full((h,w), gray, dtype=np.uint16)` 直走灰度路径（详见 ch06）。
- **频繁 `WriteAW` 写 EEPROM**：约 40 秒/次且磨损寿命；波长/相位固定后只写一次。
- **依赖 `ReadSD` 单点取 SN**：快速 open/close 后可能永久挂起，必须有降级链 + `Reboot` 恢复。

## Code Examples
```cpp
// 内存模式最小闭环（C/DLL 视角）
SLM_Ctrl_Open(1);
while (SLM_Ctrl_ReadSU(1) == SLM_BS) Sleep(1000);   // 启动约 40s
SLM_Ctrl_WriteVI(1, 0);                               // 内存模式；再等 40s
SLM_Ctrl_WriteWL(1, 1064, 200);                       // 1064nm, phase=2π(0.01π 单位)
SLM_Ctrl_WriteGS(1, 512);                             // 全屏灰度
SLM_Ctrl_WriteMI(1, 1, 1200, 1920, 0, data);          // 写入槽 1
SLM_Ctrl_WriteDS(1, 1);                               // 显示槽 1
SLM_Ctrl_Close(1);
```
- 演示：官方推荐时序；每个 40 秒点不可省略。

```python
# Python 驱动映射（src/ao_shaping/drivers/slm/santec/driver.py）
slm.open()                      # SLM_Ctrl_Open
slm._check_status()             # SLM_Ctrl_ReadSU 轮询
slm._set_memory_mode()          # SLM_Ctrl_WriteVI
slm.set_wavelength(1064)        # SLM_Ctrl_WriteWL
slm.write_phase(phase, mem=1)   # SLM_Ctrl_WriteMI
slm.display_memory(1)           # SLM_Ctrl_WriteDS（同槽 no-op，见 Anti-patterns）
n = slm.get_displayed_memory_number()  # SLM_Ctrl_ReadDS
```
- 演示：官方 API 与 Python 方法一一对应，改驱动时按此表定位。

## Reference Tables
| SLM_STATUS | 值 | 含义 |
|---|---|---|
| `SLM_OK` / `SLM_NG` / `SLM_BS` / `SLM_ER` | 0/1/2/3 | 成功 / 一般错误 / 忙（~40s）/ 参数错 |
| `SLM_INVAID_MONITOR`…`SLM_DATA_FORMAT_ERR` | -1…-4 | DVI 显示设备错误族 |
| `SLM_FILE_READ_ERR` | -101 | 文件读取错误 |
| `SLM_NOT_OPEN_USB` | -200 | 未打开 USB |
| `FT_DEVICE_NOT_FOUND` | -10002 | 设备未找到，检查电源 |

| 40 秒时序点 | 函数 | 等待方式 |
|---|---|---|
| 上电启动 | `ReadSU` | 轮询至非 `SLM_BS` |
| 模式切换 | `WriteVI` | 固定等待或轮询 |
| 波长/相位设置 | `WriteWL` | 之后才可写相位数据 |
| EEPROM 保存 | `WriteAW` | 低频操作 |

| 显示数据函数（DVI 族） | 签名要点 |
|---|---|
| `SLM_Disp_GrayScale` | `(DisplayNumber, type, GrayScale)`，`type` 可带 `FLAG_RATE120` |
| `SLM_Disp_Data` | `(DisplayNumber, width, height, type, USHORT* data)`，10bit 数组入口 |
| `SLM_Disp_BMP` / `ReadBMP` / `ReadCSV` | `(DisplayNumber, type, HBITMAP/LPWSTR 文件)`，BMP 仅 8/24bit |

## Key Takeaways
1. 时序第一：启动/切模式/设波长/写 EEPROM 各约 40 秒，全部用 `ReadSU` 轮询。
2. 槽 1-128 备货-上架模型：显示中的槽不可覆盖；连续换图必须轮换槽位。
3. 灰度 0-1023 对应 0-2π；10bit 数据走 `SLM_Disp_Data`/CSV，BMP 只 8/24bit。
4. SN 读取走 7 级降级链；`ReadSD` 挂起时 `Reboot` 重连。
5. 1064nm 附近存在振幅耦合（亮度随灰度周期变化），实验判读时注意。

## Connects To
- **ch06**: 本仓 Santec 驱动的平场灰度两条关键规则（`np.full` 直写、槽轮换）与 CSV 加载/导出管线。
- **ch01**: 相机侧曝光融合/回调采集与 SLM 换图时序配合。
