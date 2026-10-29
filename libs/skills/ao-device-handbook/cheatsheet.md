# Cheatsheet — 设备操作决策速查

**按"设备 → 你遇到的问题"查表。每条都是 if/then 决策规则。**

## 设备选型与入口

| 你要做的事 | 用哪个 | 入口 |
|---|---|---|
| 打开相机（任何品牌） | 统一工厂 | `open_camera(camera_type, cam_id, exposure_ms, bit_depth)` |
| 创建 DM（任何类型） | 注册表 | `create_dm("nlight"\|"micro"\|"asyn_micro"\|"zernike"\|"hadamard")` |
| 创建 WFS | 注册表 | `create_wfs("thorlab"\|"sim")`；无硬件用 sim |
| 无硬件跑 WFS runner | 仿真 | `--wfs-type sim`（8 条命令透传；`closed-loop` 无此 flag） |
| 无硬件跑 DM 逻辑 | 仿真 DM | 注册表延迟绑定，硬件包不 import 仿真包 |

## 相机（MiiCam / 大恒）

| 症状 / 决策点 | 规则 |
|---|---|
| 判断 SDK 调用成功 | **`hr >= 0` 即成功**；`S_FALSE=0x1` 也是成功。`==0`/`==S_OK` 判法必错 |
| 高帧率实时采集 | 用 callback 模式（独立回调线程）；回调内**禁**调 `Stop/Close/put_Option/put_Roi`（死锁/E_WRONG_THREAD），重活投工作队列 |
| 运行中改曝光 | 直接 `put_ExpoTime` **不生效**（残帧症状：max 恒定）→ 必须 `Stop→设置→Start` + 丢 1-2 帧 |
| 非运行时选项（RAW/RGB/BITDEPTH/ISP/FRAMERATE…） | 一律拉流**前**设；运行中改返回 `E_UNEXPECTED` |
| 重配置相机 | 走 `Stop→Start`，**不** `Close→Open`（重量级） |
| NumPy 互操作 | `rowPitch=-1`（紧密打包）；宽度是 4 的倍数时 `rowPitch=0` 等价 |
| 大恒高速采集 | 回调推送；拉取模式仅低频单帧 |
| 大恒每帧必查 | `raw.get_status() == INCOMPLETE` 则丢弃 |

## SLM-200

| 症状 / 决策点 | 规则 |
|---|---|
| 任何写操作前 | 先 `ReadSU` 轮询至非 `SLM_BS`；启动/切模式/设波长/写 EEPROM 各约 **40 秒** |
| `WriteMI` 失败或无变化 | 目标槽**正在显示** → 先 `WriteDS(其他槽)` 或 `set_grayscale(0)` 撤下 |
| 连续换图 | 必须**轮换槽位**（本仓 2-125 随机 + 排除当前显示槽）；同槽 `display_memory` 是 no-op |
| 10bit 图案数据 | 走 `SLM_Disp_Data`（数组）或 CSV；BMP 只支持 8/24bit |
| 平场灰度 | `np.full((h,w), gray, dtype=np.uint16)` 直走灰度路径；**不要**经 `create_phase_from_array`（被当弧度 mod 2π 损坏） |
| 读 SN 挂起/超时 | `ReadSD` 在快速 open/close 后可能挂起 → 走 7 级降级链，挂死则 `Reboot` 重连 |
| 频繁改波长/相位 | 只 `WriteAW` 一次（~40s + EEPROM 寿命）；后续 `WriteWL` 不落 EEPROM |
| 1064nm 实验判读 | 注意**振幅耦合**：光斑亮度随灰度周期变化（≈993 灰度值/周期） |

## Micro-DM

| 症状 / 决策点 | 规则 |
|---|---|
| 闭环抖动/帧时间不稳 | 检查热路径是否有**每帧分配**或**自动重连**；改预分配 + 固定超时 |
| 一控挂死拖慢整帧 | `gather` 独立超时，部分失败继续；不要等全部完成 |
| 延迟 ACK 小火花 | 连接时 `TCP_NODELAY` 禁 Nagle |
| (x,y)↔通道换算 | `get_channel_by_xy(x,y)` / `get_channel_by_ip_position(ip_suffix, pos)`；序号 0-based |
| 老化/寿命验证 | `full-voltage`（全量 0V↔V 交替）；小范围用 `alt-voltage`（+ NI DAQ ADC 同步采集） |

## Thorlabs WFS

| 症状 / 决策点 | 规则 |
|---|---|
| 返回值判定 | `0` 放行 / **正数记 warning 不抛** / 负数抛 `WfsError` |
| 2D 数组绑定 | 一律 `MAX_SPOTS[::-1]` 保持 **Y 主序**；传 (80,80) 直序=整幅转置 |
| `Get*` 返回空/垃圾 | 先查前置链：`Calc*` 是否先跑（`GetSpotDiameters` 要 `calcDias=1`；`CalcReconstrDeviations` 先于 `CalcWavefront`） |
| Zernike 出现巨大假 tip/tilt（\|z\| 数 λ~10λ） | pupil（瞳面） 被硬编码污染 → `wfs.pupil = wfs.optimize_pupil()`（只计算不写回，必须显式赋值） |
| `GetInstrumentListLen` 报错 | 传 `VI_NULL()`（`c_ulong()`），不是 `None` |
| Highspeed 模式 | 只剩质心流；图像/波前/Zernike/直径全禁用 |
| 光瞳过暗/过亮 | status bit PTL/PTH → 自动曝光循环调曝光/增益 |
| 仿真 WFS 数值 | 由 `get_zernike()` 推的 RMS/Strehl **必须标"未验证"**；piston 测不到（SH 物理限制） |

## 通用红线（跨设备）

1. 热路径零分配、无重连、硬超时。
2. 回调线程是禁区：不调 SDK、不取主线程锁。
3. 灰度与弧度是两个口径，不互通（SLM 灰度直写 / CSV 灰度入弧度出）。
4. 历史数值只作经验来源，新光路必须实测重标定。
5. 换设备=换注册表条目，上层代码不动。
