# Patterns — AO-shaping 设备手册技术模式

每个模式给出"何时用 / 怎么做 / 权衡"。

## AO 闭环零分配热路径
**When to use**: 1kHz 量级 DM/闭环热路径。
**How**: 预分配缓冲与命令帧，电压量化为 LUT 索引原地拷贝，`build_frame_commands` 预构建后 `send_frame_commands` 重放（仅 write+drain）。
**Trade-offs**: 启动多一次性预计算；换来热循环零编码零分配零 GC。

## 确定性延迟（无热路径重连）
**When to use**: 实时闭环 I/O。
**How**: 单次发送 + 固定超时，失败立即返回交上层决策；`drain` 硬超时（8ms）标记丢帧。
**Trade-offs**: 放弃自动恢复，换延迟可预测、不阻塞节拍。

## 独立超时 gather（多控制器）
**When to use**: 多控制器并行下发。
**How**: `asyncio.gather` + 每控独立超时，部分失败不拖慢整体帧率。
**Trade-offs**: 需上层判断"部分成功"语义。

## 注册表 + 工厂延迟绑定
**When to use**: 多品牌设备抽象。
**How**: `register_dm/create_dm/resolve_dm`，硬件包不 import 仿真包；上层只认 `DM`/`WFS` 接口。
**Trade-offs**: 多一层间接；换来设备可替换、仿真/硬件同接口。

## 回调推送采集
**When to use**: 高速相机连续采集 + 实时推理。
**How**: `register_capture_callback`/`StartPullModeWithCallback` + 独立回调线程，回调内只做轻活。
**Trade-offs**: 回调线程禁调 SDK（死锁/E_WRONG_THREAD）；换来高实时低延迟。

## 曝光 Stop-设置-Start 复位
**When to use**: 运行中改曝光。
**How**: `Stop → put_ExpoTime → Start`，丢弃前 1-2 帧残帧。
**Trade-offs**: 多一次 Stop/Start（重量级操作）；换来曝光真实生效。

## SLM 内存槽备货-上架
**When to use**: SLM 连续换图。
**How**: `WriteMI` 备货入槽 → `WriteDS` 上架；连续写必须轮换槽，不覆盖显示中槽。
**Trade-offs**: 多管理 128 槽状态；换来快速无阻塞换图。

## 40 秒时序轮询
**When to use**: SLM 上电/切模式/设波长/写 EEPROM。
**How**: `ReadSU` 轮询至非 `SLM_BS`，不 `sleep(5)`。
**Trade-offs**: 启动慢（~40s）；换来时序正确不失败。

## 设备标识降级链
**When to use**: 多设备唯一标识读取。
**How**: 按序尝试 `ReadSDO→ReadSD→ReadSO→ReadPS→ReadLS→ReadPN→ReadVR`，单点挂起 `Reboot` 恢复。
**Trade-offs**: 多回退逻辑；换来标识读取鲁棒。

## WFS 调用前置链
**When to use**: 波前/Zernike 测量。
**How**: `optimize_pupil 写回 → Calc* → Get*`，先算后取，pupil（瞳面） 必须自动获取。
**Trade-offs**: 步骤多；换来系数/波前正确（避免假 tip/tilt）。

## HRESULT 三态判定
**When to use**: MiiCam 一切 SDK 调用。
**How**: `hr >= 0` 即成功（含 `S_FALSE`），负值分类处置。
**Trade-offs**: 无；`==0` 判断会误判。

## 灰度直写 vs 弧度封装
**When to use**: SLM 平场/灰度图案。
**How**: `np.full((h,w), gray, dtype=np.uint16)` 直走灰度路径，不经 `create_phase_from_array`。
**Trade-offs**: 弧度图案才用封装；混用灰度会静默损坏。

## 仿真 WFS 替身
**When to use**: 无硬件开发/测试。
**How**: `SimulatedWFS` 接口与 `ThorlabWFS` 1:1 对齐（12 成员 + stats 键），`--wfs-type sim`。
**Trade-offs**: 保真度有限（SH 测不到 piston、串扰）；换无硬件可运行，数值须标"未验证"。
