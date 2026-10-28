# Chapter 6: 驱动层总览（drivers/ 统一抽象）

## Core Idea
`drivers/` 用一个"注册表 + 统一接口 + 工厂"的模型把 WFS/DM/相机/SLM/ADC 五类设备抽象成同一套 `open/close/send/is_connected` 生命周期，上层 runner/优化器不碰具体品牌——换设备只换注册表条目，不动业务代码。

## Frameworks Introduced
- **统一 DM 接口 + 注册表**
  - When to use: 写/改任何变形镜相关代码。
  - How: 所有 DM 继承 `ao_shaping.drivers.dm.base.DM`，提供 `transform`/`send`/`open`/`close`/`is_connected`/`get_actuator_positions` 标准方法。注册表 `register_dm`/`create_dm`/`list_dm_types`/`resolve_dm` 延迟绑定（硬件包不 import 仿真包）。`optimize_pib` 接受任意 `DM` 子类，CLI `--dm-type` 支持 `nlight`/`micro`/`asyn_micro`/`zernike`/`hadamard`；未指定时自动探测在线 DM（仅一个自动选，多个报错）。
- **WFS 注册表 + 仿真 WFS**
  - When to use: 无硬件开发或测试。
  - How: `register_wfs`/`create_wfs`/`list_wfs_types`/`resolve_wfs` 镜像 DM 注册表；`resolve_wfs(None)` 仍返回 `ThorlabWFS`（硬件行为不变）。`SimulatedWFS`（OOPAO Shack-Hartmann）可直接顶替 `ThorlabWFS` 让所有 WFS runner 无硬件运行。`--wfs-type [thorlab|sim]` 在 `WfsParams` 上，8 条命令透传（`wf`/`pipeline`/`rms-zernike`/`ga-zernike`/`greedy-zernike`/`dm-matrix`/`hadamard-matrix`/`zernike-matrix`），`closed-loop` 刻意无此 flag（它回放已存矩阵）。
- **相机统一工厂**
  - When to use: 一切打开相机的调用点。
  - How: `utils/image/hardware_utils.open_camera(camera_type, cam_id, exposure_ms, bit_depth)` 是 v0.10.0 起唯一入口；大恒走 `DahengCamera`（14/16 位），MiiCam 走 `MIICamera`（高速）。
- **SLM 统一接口 + CSV 管线**
  - When to use: SLM 相位加载/导出。
  - How: Santec SLM200 `open()` 拆为 `_apply_config_params`/`_load_correction`/`_setup_wavelength`；波前误差矫正由 `WavefrontCorrection` 类管理（CSV 加载→异常点检测→矫正映射），按"显式>配置>默认"优先级加载。CSV 加载 `load_gray_from_csv`（驱动层校验）→ `csv_to_phase`（灰度→弧度）→ `display_phase`（弧度→灰度+矫正+LUT+平移）；导出 `Santec.save_phase_to_csv(phase_rad, dest)` 保留 Y/X 索引。**导出文件不能交回灰度加载管线**（弧度 vs 灰度口径不同）。

## Key Concepts
- **DM 注册表**: 延迟绑定，`resolve_dm(None)` 默认硬件实现，硬件包不依赖仿真包。
- **`--dm-type` / `--wfs-type`**: 共享参数上定义、多命令透传的设备选择开关。
- **振幅耦合**: SLM 在 1064nm 附近不同灰度平场时光斑亮度随灰度周期变化（周期≈2π≈993 灰度值），SLM 固有特性。
- **NidaqADC**: NI DAQ 模拟电压采集（`HW_TIMED_SINGLE_POINT`），`read(samples)`/`read_mean()`；无硬件用 `MockADC`。
- **ZernikeDM / HadamardDM**: 按 Zernike/Walsh-Hadamard 系数驱动的 DM/SLM 接口。

## Mental Models
- Use "注册表是设备插口" 的心智：上层只认 `DM`/`WFS` 抽象接口，换品牌=换注册表条目，业务零改动。
- 把 仿真 WFS 理解为 数字孪生替身：接口契约与硬件 1:1 对齐（12 个成员 + stats 键），缺一个 runner 就崩。
- Think of CSV 加载/导出 as 两条单向管道：灰度入（`load_gray→csv_to_phase`）与弧度出（`save_phase_to_csv`）不互通，混用即口径错。

## Anti-patterns
- **上层直接 import 具体品牌驱动**：绕过注册表会破坏可替换性；一律走 `create_dm`/`create_wfs`/`open_camera` 工厂。
- **把 uint16 灰度数组经 `create_phase_from_array()` 传**：该函数把输入当弧度（mod 2π ×1023），灰度值被静默损坏；平场灰度必须 `np.full((h,w), gray, dtype=np.uint16)` 走灰度路径。
- **内存模式连续写同一 SLM 槽**：`display_memory(slot)` 对已显示槽是 no-op（LCOS 不刷新）；必须换槽（本仓 `slm-gsnet` 在 2-125 随机选并排除当前显示槽）。
- **把 CSV 导出文件再喂给灰度加载管线**：弧度文件≠灰度文件，口径不一致。
- **仿真 WFS 当波前基准**：`SimulatedWFS` 保真度不足以当基准（SH 测不到 piston，tilt 与其他模式同注入串扰达 83%）；由 `get_zernike()` 推的 RMS/Strehl 必须标注"未验证"，可信数值用硬件标定。
- **硬件包 import 仿真包**：破坏延迟绑定契约；注册表用 `resolve_*` 惰性解析。

## Code Examples
```python
# 统一相机工厂（唯一打开入口）
from ao_shaping.utils.image.hardware_utils import open_camera
cam = open_camera(camera_type="miicam", cam_id=0, exposure_ms=0.8, bit_depth=16)
```
- 演示：一切相机调用走工厂，`camera_type` 决定品牌实现。

```python
# 统一 DM 注册表 + 工厂
from ao_shaping.drivers.dm._registry import create_dm
dm = create_dm("asyn_micro", ips=["192.168.0.101"])
dm.open()
dm.send_voltages(np.zeros(dm.DM_Num))
dm.close()
```
- 演示：`--dm-type` 对应注册表 key，上层只认 `DM` 接口。

```python
# 平场灰度正确写法（第 1 条关键规则）
gray = 512
flat = np.full((height, width), gray, dtype=np.uint16)   # ✅ 直接灰度
# ❌ 不要 slm.display_phase(np.full((h,w), 512)) —— 会被当弧度 mod 2π 损坏
```
- 演示：灰度直写 vs 弧度封装的口径陷阱（1064nm 振幅耦合场景尤其注意）。

## Reference Tables
| 设备域 | 统一接口/工厂 | 支持类型 |
|---|---|---|
| DM 变形镜 | `create_dm` + `DM` 基类 | nlight / micro / asyn_micro / zernike / hadamard |
| WFS 波前传感器 | `create_wfs` + `WFS` 接口 | thorlab / sim |
| 相机 | `open_camera` | 大恒 (DahengCamera) / MiiCam (MIICamera) |
| SLM | Santec 驱动 + `WavefrontCorrection` | SLM200 |
| ADC | `NidaqADC` | NI DAQ / MockADC |

| 关键红线 | 规则 | 后果 |
|---|---|---|
| 灰度直写 | `np.full(..., dtype=np.uint16)` | 经弧度封装会静默损坏 |
| SLM 槽轮换 | 不写已显示槽 | 同槽 no-op，LCOS 不刷新 |
| CSV 单向 | 灰度入 / 弧度出 | 混用口径错 |
| 仿真 WFS | 不当波前基准 | 串扰致假系数，须标"未验证" |
| 延迟绑定 | 硬件不 import 仿真 | 破坏可替换性 |

## Key Takeaways
1. 一切设备走注册表+工厂，上层只认抽象接口。
2. 灰度用 `np.full(uint16)` 直写，不经过弧度封装。
3. SLM 连续换图必须换槽，同槽 no-op。
4. CSV 灰度入/弧度出是两条单向管道，不互通。
5. 仿真 WFS 接口对齐但保真度有限，波前基准必须用硬件。
6. 相机唯一入口 `open_camera`；ADC 无硬件用 `MockADC`。

## Connects To
- **ch01-ch05**: 本章是各品牌驱动章（大恒/SLM-200/MiiCam/WFS/Micro-DM）在统一抽象层的汇总视图。
- **ch04 (WFS)**: 仿真 WFS 与硬件 WFS 的接口契约对齐细节。
