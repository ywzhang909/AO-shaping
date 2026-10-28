# Glossary — AO-shaping 设备手册关键术语

按字母/拼音排序。`(chN)` 指主要出处章节。

**ADC（NidaqADC）** — NI DAQ 模拟电压采集驱动，`read(samples)`/`read_mean()`，无硬件用 `MockADC`（ch06）。
**振幅耦合** — SLM 在 1064nm 附近加载不同灰度平场时光斑亮度随灰度周期变化（周期≈2π≈993 灰度值），SLM 固有特性（ch02/ch06）。
**AsyncMicroDM** — Micro-DM 的 asyncio 驱动，LUT 预查表 + 非阻塞 TCP + 内联命令字节快路径（ch05）。
**bits（MiiCam）** — `PullImageV4` 输出位深，0=默认，8/16/24/32/48/64（ch03）。
**BMP 格式限制** — SLM-200 的 BMP 仅 8/24 bit，10bit 数据必须走 `SLM_Disp_Data` 或 CSV（ch02）。
**callback 模式（MiiCam）** — 独立回调线程 + 软件触发 `Trigger(1)` + `PullImageV4` 取帧（ch03）。
**CALC→GET 前置链** — WFS 中 `Calc*`（计算）必须先于 `Get*`（取值），跳链取到垃圾（ch04）。
**ch01 大恒主流程** — 枚举→打开→`stream_on`→取图→`stream_off`→`close_device`（ch01）。
**CSV 单向管道** — SLM 灰度入（`load_gray_from_csv→csv_to_phase`）与弧度出（`save_phase_to_csv`）不互通（ch06）。
**Display Number** — SLM-200 DVI 模式显示编号，与 SLM Number 不同源（ch02）。
**DM 注册表** — 延迟绑定，`resolve_dm(None)` 默认硬件实现，硬件包不 import 仿真包（ch06）。
**drain 硬超时** — Micro-DM 每控 8ms（`SEND_DRAIN_TIMEOUT`），超时标记丢帧不阻塞节拍（ch05）。
**Feature 属性模型** — 大恒 GenICam 风格，8 类 Feature（Int/Float/Enum/Bool/String/Buffer/Command）按类型访问（ch01）。
**full-voltage** — Micro-DM 全量单元 0V↔V 同时交替，用于老化/寿命验证（ch05）。
**GS 预整形** — 高斯预矫正 + 全频随机相位 SPGD 细化的 SLM 整形入口（ch02 关联）。
**HMiicam** — MiiCam 相机句柄，`Miicam_Open` 返回，`Close` 后失效（ch03）。
**HadamardDM** — Walsh-Hadamard 系数驱动的 DM/SLM 接口（ch06）。
**Highspeed 模式** — WFS10/20 专用，仅质心流，图像/波前/Zernike 全禁用（ch04）。
**HRESULT 黄金规则** — MiiCam `hr >= 0` 即成功；`S_FALSE=0x1` 是成功但无操作（ch03）。
**LUT 预查表** — Micro-DM 电压量化为 12-bit 索引，`index→big-endian 2-byte payload` 预计算（ch05）。
**Memory Number（SLM）** — 1-128 号内存槽；显示中的槽不可覆盖（ch02）。
**MiiCam** — 高速相机 C SDK + ctypes 封装，Pull/Snap/Trigger 三模式（ch03）。
**Micro-DM 1300** — 36×36 单元、26 个 R50Power 控制器、1300 单元分布式执行器（ch05）。
**Nagle 禁用（TCP_NODELAY）** — Micro-DM 连接时禁 Nagle，消除延迟 ACK 缓冲（ch05）。
**NidaqADC / MockADC** — NI DAQ 采集驱动与无硬件替身（ch06）。
**optimize_pupil 写回** — Thorlabs WFS `optimize_pupil()` 只计算不写回，返回元组必须显式赋 `wfs.pupil`（ch04）。
**Piston 不可测** — 仿真 WFS（SH）子孔径绝对相位偏移不移动光斑，piston 必须排除出拟合基（ch06）。
**pull 模式（MiiCam）** — 连续拉流，`WaitImageV3` 阻塞拉取（ch03）。
**RoC = Z[5]** — Noll 第 5 阶（离焦后），与 `zernike_utils` 约定一致（ch04）。
**rowPitch（MiiCam）** — 行跨度；0=4 字节对齐，-1=紧密打包（NumPy 互操作推荐）（ch03）。
**Santec SLM-200** — LCOS 空间光调制器，双模式（内存/DVI）+ 128 槽 + 40 秒时序（ch02）。
**SLM Number** — Windows 自动分配的 USB 设备编号 1-8，`SLM_Ctrl_*` 寻址（ch02）。
**SLM_STATUS** — SLM-200 状态码，0=OK/1=NG/2=BUSY(~40s)/3=ER，负值为 DVI/文件/USB 错（ch02）。
**SLM 槽轮换** — 内存模式连续换图必须换槽，同槽 `display_memory` 是 no-op（ch02/ch06）。
**SimulatedWFS** — OOPAO Shack-Hartmann 仿真 WFS，接口与 `ThorlabWFS` 1:1 对齐，可无硬件运行（ch06）。
**Snap 模式（MiiCam）** — 高分辨率静止图，`MIICAM_EVENT_STILLIMAGE` 后 `PullStillImage`（ch03）。
**TDIBWIDTHBYTES** — 4 字节对齐行宽宏 `((bits+31)&~31)/8`（ch03）。
**TCP_NODELAY** — 见 Nagle 禁用（ch05）。
**Trigger 模式（MiiCam）** — 软件/外部触发精确控制时机，`put_Option(TRIGGER, 0/1/2/3)`（ch03）。
**trigger_sync** — MiiCam 单触发+同步等待 `TriggerSyncV4`，默认超时 `ExpoTime×102%+4000ms`（ch03）。
**VI_NULL()** — WFS 无 session 查询的句柄占位符 `c_ulong()`（ch04）。
**ViStatus 三态** — WFS 返回值 0=成功/正=警告记录不抛/负=抛 `WfsError`（ch04）。
**wiring map** — Micro-DM `(x,y)↔IP组↔通道` 映射链，`wiring_map.json` 加载（ch05）。
**WFS_STATUS 17 位** — Thorlabs WFS 状态位图，关键位 PTH/PTL/SCL/PUD/LOS（ch04）。
**Y 主序数组** — WFS 一切 2D 数组 `[Y][X]` row-major，绑定 `MAX_SPOTS[::-1]`（ch04）。
**零 GC 抖动** — Micro-DM 热路径预分配+原地更新，杜绝每帧分配触发 GC（ch05）。
**0x09 命令** — Micro-DM 按数组设全部电压的 R50 协议命令（ch05）。
