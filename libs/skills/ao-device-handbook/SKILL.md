---
name: ao-device-handbook
version: 1.0.0
description: "Knowledge base from the AO-shaping repo device manuals (docs/): Daheng/Galaxy camera SDK, Santec SLM-200 LCOS, MiiCam high-speed camera, Thorlabs WFS, Micro-DM 1300, and the unified drivers/ abstraction layer. Use when writing, modifying, or debugging device drivers and hardware-facing code in this repo — open/close sequences, timing (40s SLM warmup, 8ms drain timeout), memory-slot and callback-thread traps, HRESULT/SLM_STATUS/WFS error handling, wiring maps, or deciding which factory entry point to call."
---

<!-- argument-hint: [topic, device name, or chapter number ch01..ch06] -->

# AO-shaping 设备手册
**来源**: `docs/` 设备手册 19 份源文件（大恒 SDK 说明书、SLM-200 开发者/DLL 指南+Agent Wiki、MiiCam Agent Wiki+评审、Thorlabs WFS 编写指南、Micro-DM 重构/README、hardware_drivers.md），由 book-to-skill 流程提取生成 | **章节**: 6 | **Generated**: 2026-10-29

## How to Use This Skill

- **无参数** — 加载下面的核心框架与红线
- **带主题** — 问 `slm 槽轮换`、`miicam 回调`、`wfs zernike 假倾斜` 等，我读对应章节后回答
- **带章节** — `ch02` 加载 Santec SLM-200 章
- **浏览** — 问"有哪些章节？"看完整索引

问到的主题若不在下方核心框架里，我会先读相应 `chapters/chNN-*.md` 再回答。

---

## 核心框架与红线

**五类设备，一条抽象**：WFS / DM / 相机 / SLM / ADC 全部走"注册表 + 统一接口 + 工厂"（ch06）。上层只认 `DM`/`WFS` 抽象接口与 `open_camera()` 工厂；换品牌=换注册表条目，业务代码不动。

**各设备第一约束**：

| 设备 | 一句话红线 | 章节 |
|---|---|---|
| 大恒相机 | GenICam Feature 模型：`is_implemented` 探活后按 8 类访问；高速采集必须回调推送 | ch01 |
| Santec SLM-200 | 启动/切模式/设波长/写 EEPROM 各约 40 秒（`ReadSU` 轮询）；128 内存槽"备货-上架"，显示中的槽不可覆盖 | ch02 |
| MiiCam | `hr >= 0` 即成功（`S_FALSE` 也是）；回调线程内禁调 SDK（死锁）；改曝光必须 Stop-设置-Start | ch03 |
| Thorlabs WFS | 0=成功/正=警告/负=抛错；先 `Calc*` 后 `Get*`；2D 数组一律 Y 主序 `MAX_SPOTS[::-1]`；pupil（瞳面） 必须 `optimize_pupil()` 写回 | ch04 |
| Micro-DM 1300 | 热路径零分配、无重连、drain 8ms 硬超时、gather 独立超时；26 控×50 通道，wiring map 做坐标换算 | ch05 |
| 驱动层 | 灰度与弧度是两个口径不互通；仿真 WFS 数值必须标"未验证" | ch06 |

**跨设备通用红线（最常踩的 8 个坑）**：

1. **SLM 同槽 no-op**：内存模式连续写同一内存槽，`display_memory` 不刷新 LCOS → 必须轮换槽位（2-125 随机 + 排除当前显示槽）。
2. **灰度经弧度封装被静默损坏**：`create_phase_from_array()` 把输入当弧度（mod 2π ×1023）；平场灰度必须 `np.full((h,w), gray, dtype=np.uint16)` 直写。
3. **CSV 单向**：SLM 灰度入（`load_gray_from_csv→csv_to_phase`）与弧度出（`save_phase_to_csv`）不互通，导出文件不能交回加载管线。
4. **MiiCam `==0` 判成功**：`S_FALSE=0x1` 是"成功但无操作"，`hr >= 0` 才是唯一判据。
5. **MiiCam 运行中改曝光不生效**：抓到旧曝光残帧（max 恒定、图像雷同）→ `Stop→put_ExpoTime→Start` + 丢 1-2 帧。
6. **WFS 硬编码 pupil（瞳面）**：边界无效子孔径污染 `WFS_ZernikeLsf`，产生巨大假 tip/tilt（实测 |z|=4.6~12.8λ，正确 pupil（瞳面） 后 ≤0.22λ）；`optimize_pupil()` 只计算不写回，必须 `wfs.pupil = wfs.optimize_pupil()`。
7. **热路径每帧分配/重连**：触发 GC 毫秒级暂停或延迟不可预测 → 预分配 + LUT 查表 + 固定超时。
8. **回调线程调 SDK**：MiiCam `Stop/Close/put_Option`、大恒重计算，都是死锁或 `E_WRONG_THREAD` 来源。

---

## Chapter Index

| # | Title | 关键内容 |
|---|-------|---------|
| [ch01](chapters/ch01-daheng-camera-sdk.md) | 大恒（Daheng/Galaxy）相机 SDK | GX 主流程、拉取/回调双模式、Feature 8 类、曝光融合 |
| [ch02](chapters/ch02-santec-slm200.md) | Santec SLM-200（LCOS-SLM） | 40s 时序、内存槽/显示表、SLM_STATUS、SN 降级链、振幅耦合 |
| [ch03](chapters/ch03-miicam.md) | MiiCam 相机（C SDK + ctypes） | HRESULT 黄金规则、双采集模式、回调线程禁区、曝光复位 |
| [ch04](chapters/ch04-thorlabs-wfs.md) | Thorlabs WFS 波前传感器 | ctypes 绑定五约定、71 API×10 族前置链、WFS_STATUS、pupil（瞳面） 写回 |
| [ch05](chapters/ch05-micro-dm.md) | Micro-DM 1300（R50Power 多控制器） | AO 热路径五原则、TCP 帧协议、wiring map、同步/异步双模式 |
| [ch06](chapters/ch06-driver-layer.md) | 驱动层总览（drivers/ 统一抽象） | 注册表+工厂、统一接口、仿真 WFS 保真度红线、CSV 单向管道 |

## Topic Index

- **40 秒时序 / 上电等待** → ch02
- **内存槽 / 槽轮换 / no-op** → ch02, ch06
- **HRESULT / S_FALSE / E_WRONG_THREAD** → ch03
- **回调 / 回调线程 / 死锁** → ch01, ch03
- **曝光 / 残帧 / Stop-Start 复位** → ch01, ch03
- **pupil（瞳面） / 假 tip/tilt / ZernikeLsf** → ch04
- **Y 主序 / MAX_SPOTS / 数组布局** → ch04
- **热路径 / 零 GC / drain 超时 / TCP_NODELAY** → ch05
- **wiring map / 通道映射 / (x,y) 换算** → ch05
- **灰度直写 / 弧度封装 / CSV 单向** → ch02, ch06
- **注册表 / 工厂 / 延迟绑定 / 仿真 WFS** → ch04, ch06
- **SLM_STATUS / FT601 / 振幅耦合** → ch02
- **Feature 模型 / INCOMPLETE 帧** → ch01
- **曝光融合 / 拉普拉斯金字塔** → ch01

## Supporting Files

- [glossary.md](glossary.md) — 全部关键术语与定义
- [patterns.md](patterns.md) — 全部技术与设计模式（何时用/怎么做/权衡）
- [cheatsheet.md](cheatsheet.md) — 决策速查表（按设备×症状）

---

## Scope & Limits

本 skill 只覆盖 `docs/` 设备手册内容（19 份源文件）与 `hardware_drivers.md` 的实测红线。具体代码位置以 `src/ao_shaping/drivers/` 为准，本 skill 不替代代码；术语遵循 `docs/terminology_glossary.md`。历史数值（曝光响应、k≈0.414 缩放、40 秒时序等）是特定光路/批次实测，新光路必须重测，不得直接继承为默认标定。
