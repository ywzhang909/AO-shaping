# 硬件驱动详情

> 本文件从 README.md 拆出（2026-10-13）。驱动分层与惰性加载契约见 [drivers/AGENTS.md](../src/ao_shaping/drivers/AGENTS.md) 与 [docs/drivers_architecture.md](drivers_architecture.md)；本文按设备域给出各驱动的用法与实测红线。

## 波前传感器

- **Thorlabs WFS系列**: 支持自动图像采集和倾斜去除
- **WFS 注册表** (`drivers/wfs/_registry.py`): `register_wfs` / `create_wfs` / `list_wfs_types` / `resolve_wfs`，镜像 DM 的注册表；采用延迟绑定，硬件包**不会** import 仿真包。`resolve_wfs(None)` 仍返回 `ThorlabWFS`，故硬件运行行为不变。
- **`--wfs_type [thorlab|sim]`** 定义在共享的 `WfsParams` 上，8 个已注册命令**既暴露也真正透传**该 flag: `wf`、`pipeline`、`rms-zernike`、`ga-zernike`、`greedy-zernike`、`dm-matrix`、`hadamard-matrix`、`zernike-matrix`。`closed-loop` **刻意没有** —— 它回放已保存的响应矩阵。

### 仿真 WFS (`SimulatedWFS`)

`drivers/sim/wfs/simulated_wfs.py`，OOPAO Shack-Hartmann 实现 `BaseWFS`，可直接顶替 `ThorlabWFS`，使所有 WFS 类 runner 无需硬件即可执行。

- **测瞳面梯度而非焦平面传播**: SH 通过微透镜阵列成像瞳孔，探测器上光斑位移编码各子孔径的局部 tip/tilt，因此测量量是**瞳面相位梯度**。早前计划复用本仓 `wave.py` 的焦平面传播 —— 那产出的是**相机图像**而非 slope，二者不可互换。
- **slope 数组是行块状布局**: 第 `0:n_subap` 行为 x-slope，第 `n_subap:` 行为 y-slope (纯倾斜探针实测: 纯 x 倾斜 → 上半 rms 1.29e-4, 下半 0)。
- 🔴 **SH 测不到 piston**: 子孔径的绝对相位偏移不移动光斑。`list_zernike_modes` 从 Noll 1 = piston 起，把它放进拟合基会留下一个近零空间列，`pinv` 把它放大成巨大的伪系数 (实测纯离焦瞳孔下 **+0.50 rad**)。故 piston 必须排除出拟合基 (`_FIT_FIRST_MODE`) —— **这不是 off-by-one**。
- **单位**: `get_wavefront()` → **waves** = `phase_rad / 2π`; `get_zernike()` → **µm** = `phase_rad * λ_nm*1e-3 / 2π`。`um_to_waves()` 写死 532 nm，故默认波长必须 532 nm。
- ⚠️ **保真度不足以当波前基准 (实测，勿高估)**: 注入已知模式再读回系数 —— noll 2 tilt 单独 24% / 与其他模式同时 **83%**; noll 4 defocus 1.4%; noll 11 spherical 32%。tilt 单独注入只差 24%，与其他模式同注入却差到 83% ⇒ 主误差是**模式间串扰 (cross-talk)**，不是逐模式噪声。已排除质心量化与标定 pass 污染 (平场读数严格为 0)。**任何由 `get_zernike()` 推出的 RMS 改善率或 Strehl 都必须标注为「未验证」**; 要可信数值请用硬件标定。
- **接口契约必须与 `ThorlabWFS` 完全对齐**，缺一个就直接崩: `get_wavefront()` 的 stats 键必须是 `min/max/diff/mean/rms/wighted_rms`; `build_subaperture_mask()` 返回 **2 元组** `(mask_2d, valid_indices_flat)` 且按 **subaperture 网格**定尺寸 (**不能**按 flux 定 —— OOPAO 的 flux 报在 8×8 lenslet 网格而 slope 跨 6×6 网格); runner/优化器直接读取 12 个成员 (`num_spots_x/num_spots_y`、`mla_index`、`serial_num`、`device_name`、`exposure_time`、`high_speed`、`use_custom_ref`、`pupil`、`d_x`、`get_mla_name()`、`set_ref_plane()`)。锁定测试: `tests/ao_shaping/drivers/sim/test_simulated_wfs.py::TestRunnerFacingSurface`。

## 变形镜

- **统一 DM 接口**: 所有变形镜继承自 `ao_shaping.drivers.dm.base.DM`，提供 `transform`/`send`/`open`/`close`/`is_connected`/`get_actuator_positions` 等标准方法
- **NLight系列**: 支持电压控制和电压差安全检查
- **R50Power MicroDM (同步)**: 通过 TCP 控制多路 R50Power 控制器（每路 50 通道，-20V~120V）
  - 自动从 `libs/micro_drive1300/wiring_map.json` 加载控制器 IP 和通道映射
  - 支持 39×39 阵列坐标 `(x, y)` 到控制器通道的双向查询
  - 支持 `(ip_suffix, payload_position)` 到物理位置的映射查询
  - 类型安全的 WiringMap dataclass 解析（`WiringMap`, `ChannelEntry`, `ChannelInfo`）
  - 可通过 `use_wiring_map=False` 回退到默认 IP 配置
  - **容错连接**: `open()` 允许个别控制器连接失败，记录 warning 后继续，仅当全部失败时抛出异常
  - **连接状态检查**: `get_connection_status()` 返回所有控制器的 ping 可达性和 TCP 连接状态
  - **单控制器管理**: 支持 `connect_controller(id)`、`disconnect_controller(id)`、`reconnect_controller(id)` 独立控制
  - **排除控制器**: 初始化时可通过 `exclude_ips` 或 `exclude_ids` 跳过指定控制器

  ```python
  from ao_shaping.drivers.dm.micro import MicroDM

  # 默认加载 wiring map，自动识别控制器 IP
  dm = MicroDM()

  # 排除特定控制器（按 IP 或 ID）
  dm = MicroDM(exclude_ips=["192.168.0.103"], exclude_ids=[4, 5])

  # 通过 39×39 阵列坐标查询通道信息
  info = dm.get_channel_by_xy(x=1, y=3)
  print(info.ip_address, info.payload_position, info.physical_label)

  # 通过控制器 IP 和通道标号查询
  info = dm.get_channel_by_ip_position(ip_suffix=101, payload_position=13)
  print(info.physical_position, info.physical_label)

  # 发送电压指令
  dm.open()
  dm.send_voltages(np.zeros(dm.DM_Num))

  # 检查所有控制器连接状态
  for status in dm.get_connection_status():
      print(f"ID={status.controller_id} IP={status.ip} "
            f"ping={status.ping_reachable} tcp={status.tcp_connected}")

  # 单独管理控制器
  dm.connect_controller(3)       # 连接控制器 3
  dm.disconnect_controller(2)    # 断开控制器 2
  dm.reconnect_controller(1)     # 重连控制器 1

  dm.close()
  ```

- **R50Power AsyncMicroDM (异步)**: 基于 asyncio 的高性能异步 TCP 驱动，专为 AO 快速闭环优化
  - LUT-based 预查表电压转换，零 GC 稳态运行（`VoltageConverter`）
  - `asyncio.StreamReader/StreamWriter` 非阻塞 TCP 通信
  - **TCP_NODELAY**：连接时禁用 Nagle，消除延迟 ACK 对小火花的缓冲延迟
  - 预分配命令缓冲区，避免帧间内存分配
  - **内联命令字节快路径**：`prebuild_command()` / `build_frame_commands()` 将电压帧一次性编码为命令字节，热循环用 `send_bytes()` / `send_frame_commands()` 直接重放——零编码、零分配、仅 `write`+`drain`（见 `full-voltage` 工具）
  - 支持同步/异步双模式使用（`open()`/`close()` 同步桥接）
  - 并行控制器通信，独立超时控制

  ```python
  from ao_shaping.drivers.dm.micro import AsyncMicroDM

  # 同步用法（内部桥接到异步）
  dm = AsyncMicroDM(ips=["192.168.0.101", "192.168.0.102"])
  dm.open()
  dm.send_voltages(np.zeros(dm.DM_Num))
  dm.close()

  # 异步用法（原生 asyncio）
  dm = AsyncMicroDM(ips=["192.168.0.101"])
  await dm.connect_all()
  await dm.send_frame(np.zeros(dm.DM_Num))
  await dm.shutdown()

  # 内联快路径：预构建状态字节，热循环零开销重放
  cmd_off = dm.build_frame_commands(np.zeros(dm.DM_Num))
  cmd_on = dm.build_frame_commands(np.full(dm.DM_Num, 20.0))
  await dm.send_frame_commands(cmd_off)  # 仅 write + drain
  await dm.send_frame_commands(cmd_on)

  # 工厂创建
  from ao_shaping.drivers.dm._registry import create_dm
  dm = create_dm("asyn_micro", ips=["192.168.0.101"])
  ```

- **ZernikeDM**: Zernike 系数驱动的 DM/SLM 接口，支持 Zernike 多项式相位生成
- **HadamardDM**: Hadamard 系数驱动的 DM/SLM 接口，支持 Walsh-Hadamard 模式相位生成
- **PIB 优化器多 DM 支持**: `optimize_pib` 现接受任意 `DM` 子类实例，命令行支持 `--dm_type` 参数
  - 支持类型: `nlight`, `micro`, `asyn_micro`, `zernike`, `hadamard`
  - 自动检测: 未指定 `--dm_type` 时自动探测在线 DM，仅一个时自动选取，多个时报错提示

## 相机

- **大恒相机系列**: DahengCamera，支持14位和16位模式
- **MIICAM系列**: MIICamera，支持高速采集
- 统一相机工厂: `utils/image/hardware_utils.open_camera(camera_type, cam_id, exposure_ms, bit_depth)`（v0.10.0 起所有打开调用点走此工厂）

## 空间光调制器

- **Santec SLM200**: 支持相位图案生成、缓存和CSV加载/导出
  - `open()` 方法已重构为子方法 (`_apply_config_params`, `_load_correction`, `_setup_wavelength`)，逻辑更清晰
  - 波前误差矫正通过独立 `WavefrontCorrection` 类管理（CSV加载→异常点检测→矫正映射图）
  - 矫正数据自动按优先级加载: `__init__` 显式指定 > 配置文件 > 默认路径
  - **CSV 相位加载与导出**（`multi_slm_controller.py` GUI）：加载格式为 1200×1920、值 0~1023、首行/首列为 `Y/X` 索引；管线为 `load_gray_from_csv`（驱动层格式校验）→ `csv_to_phase`（灰度→弧度）→ `display_phase`（`create_phase_from_array()` 弧度→灰度+矫正+LUT+平移），与 GUI 预览共用同一路径保证字节级一致。导出使用 `Santec.save_phase_to_csv(phase_rad, destination)`，数据区为弧度值并保留 `Y/X` 行列索引，目标可为路径、`BytesIO` 或文本流；导出文件不能交给灰度加载管线。详见 `docs/slm/slm_gui_manual.md`。

> **⚠️ SLM 平场灰度生成注意事项**
>
> Santec SLM200在1064nm附近存在**振幅耦合**效应——SLM加载不同灰度值的平场相位时，相机采集到的光斑亮度会随灰度值变化（周期 ≈ 2π，即约993灰度值）。这是SLM的固有特性，已在实验中验证（参见 `scripts/validate_flat_phase_gray.py`）。
>
> **关键规则1（灰度值路径）**: 平场相位（以及其他直接灰度图案）**必须**使用 `np.full((height, width), gray, dtype=np.uint16)` 生成，**不能**通过 `create_phase_from_array()` 传递。因为 `create_phase_from_array()` 将输入作为**弧度**处理（mod 2π → 弧度/2π × 1023），uint16灰度值会经过不必要的弧度转换而被静默损坏。
>
> **关键规则2（内存模式槽轮换）**: Santec SLM 在内存模式下，**前后两次写入不能使用同一个内存槽**（memory slot）。当 `display_memory(slot)` 被调用时，如果该槽已经在显示，设备会将此调用视为空操作（no-op），LCOS 面板不会刷新，屏幕上仍显示上一次的相位图案。连续写入时必须使用不同的槽位——`slm-gsnet` runner 在 **2~125 槽范围内随机选取**并排除当前显示槽（启动时 `get_displayed_memory_number()` 续接，跨进程也不冲突）。`display_data()` 内置的 127 槽循环机制同样满足这一约束。
>
> 验证命令:
> ```bash
> python scripts/validate_flat_phase_gray.py --exposure-ms 0.8 --wait-time-s 0.3
> ```

## 数据采集卡

- **NI DAQ (nidaqmx)**: 用于多设备同步控制和模拟电压采集
- **ADC Driver** (`ao_shaping.drivers.adc.NidaqADC`): NI DAQ模拟输入电压采集驱动
  - 基于 nidaqmx 的 `HW_TIMED_SINGLE_POINT` 采样模式
  - 可配置设备名 (Dev1)、通道 (ai0)、采样率和每批采样数
  - 提供 `read(samples)` 返回原始电压数组和 `read_mean()` 返回均值
  - 无硬件时可使用 `MockADC` 进行开发和测试

  ```python
  from ao_shaping.drivers.adc import NidaqADC

  with NidaqADC(device_name="Dev1", channel="ai0", sample_rate=5000, samples_per_channel=10) as adc:
      voltages = adc.read()        # shape (10,) array of voltages
      mean_v = adc.read_mean()     # float
  ```
