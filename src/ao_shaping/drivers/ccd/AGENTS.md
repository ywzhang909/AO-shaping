# ccd/ - 相机驱动

相机驱动模块，提供统一的相机接口。

## 结构

```
ccd/
├── base.py              # BaseCamera 抽象基类
├── common.py            # 通用工具
├── daheng/              # 大恒相机 (GigE)
│   ├── driver.py
│   ├── constants.py     # 大恒驱动常量
│   └── __init__.py
├── ffmpeg/              # FFmpeg 相机
│   ├── driver.py
│   └── __init__.py
├── miicam/              # MiiCam SDK
│   ├── driver.py
│   ├── _sdk_setup.py
│   └── __init__.py
└── __init__.py          # 包入口 (MIICamera, DahengCamera, FFmpegCamera 等)
```

## 关键类

| 类 | 文件 | 说明 |
|----|------|------|
| `BaseCamera` | `base.py` | 相机抽象基类 (ABC) |
| `CameraError` | `base.py` | 相机异常基类 |
| `MIICamera` | `miicam/driver.py` | MiiCam 相机 |
| `MIICAMError` | `miicam/driver.py` | MiiCam 特定异常 |
| `DahengCamera` | `daheng/driver.py` | 大恒相机管理器 |
| `FFmpegCamera` | `ffmpeg/driver.py` | FFmpeg 相机 |
| `ImageFolderCamera` | `ffmpeg/driver.py` | 图片文件夹相机 |

## 相机类型注册表 (common.py)

`common.py` 提供按 **相机类型 + id 获取“未 open”实例** 的注册表 (对应 DM 的 `drivers/dm/_registry.create_dm`):

| 符号 | 说明 |
|------|------|
| `CAMERA_TYPES` | `类型名 → CameraSpec` (`module:attr` 惰性目标 + 构造 kwargs 白名单) |
| `register_camera(name, target, accepted_kwargs=None)` | 注册/覆盖后端; `target` 可为类或 `"module:attr"`; 白名单 `None` = 透传全部 kwargs |
| `create_camera(camera_type, cam_id=0, exposure_time_ms=20.0, **kwargs)` | 构造但**不 open** 的相机实例; 未知类型 `ValueError`, 后端 SDK 缺失 `ImportError` |
| `list_camera_types()` | 已注册类型名排序列表 |

已注册: `daheng` / `miicam` / `ffmpeg` / `image_folder`。惰性解析保证某个后端 SDK 缺失时仅在该类型被请求时失败, 也避免与从本模块导入 `ExposureTime` 的后端发生循环导入。

> Daheng 后端在模块导入时吞掉 SDK `ImportError` (留下未定义的 `gx`), 因此 `create_camera("daheng")` 会把构造期的 `NameError` 归一化为 `ImportError`。

### 自动曝光 `auto_exposure` (统一契约, 两个硬件后端都实现)

```python
cam.auto_exposure(
    target_max=40.0,     # 目标峰值亮度 (0-255 灰度)
    tolerance=5.0,       # 峰值容差 (0-255, 绝对)
    twice_valid=True,    # 需连续两次落入容差才收敛
    max_iterations=20,
    n_sample=1,          # 每次估计峰值的采样帧数
) -> np.ndarray          # 返回调整后的图像
```

- `target_max` / `tolerance` 一律 **0-255 灰度** (与 SDK `ExpectedGrayValue` 及历史 `autoset_exposure_time_ms` 一致)。**不要再传 0-1**。
- `DahengCamera.auto_exposure` 额外有 `use_sdk_auto` / `sdk_settle_frames` (先走 SDK `ExposureAuto="Once"` 快路径, 再比例迭代)。
- `MIICamera.auto_exposure` 无 SDK 快路径, 只做比例迭代 (单步放大上限 3×); 每次改曝光走 `reset_exposure_time` (Stop→put_ExpoTime→重启拉流), 较慢; 峰值可能锁在 ~85, 高目标 (如 220) 常不可达 → 以边界曝光退出。
- 历史函数 `autoset_exposure_time_ms` **已移除**, 所有调用点改用 `auto_exposure`。

### 曝光兼容助手 (common.py)

后端曝光 API 不一致 (Daheng 用 `exposure_time` 属性; MiiCam 用 `exposure_time_ms` + `reset_exposure_time()` 的 Stop→set→Start), 统一入口:

| 函数 | 说明 |
|------|------|
| `get_camera_exposure_ms(cam)` | 读取当前**生效**曝光 (ms), **优先设备实时回读** `cam.get_exposure_time_ms()`; 后端无回读 (ffmpeg/image_folder/sim) 才回落 `exposure_time_ms` / `exposure_time` 缓存 |
| `set_camera_exposure_ms(cam, ms)` | 写曝光, 优先 `reset_exposure_time()` (MiiCam 必需) |
| `get_camera_exposure_range(cam)` | `(min_ms, max_ms)`, 后端未暴露时回退 `0.011–10000` |
| `auto_exposure(cam, target_max, tolerance=5.0, max_iterations=20, n_sample=1)` | 有原生 `auto_exposure` (Daheng/MiiCam) 则委托, 否则通用比例环 (ffmpeg) |

> **曝光「回读」= 设备实际生效值, 不是你请求的值** (见下方已知坑第 2 条)。
> 后端实现该契约的单一入口是 `get_exposure_time_ms()`:
> `DahengCamera` 读 GenICam `ExposureTime` 特征 (设备真值);
> `MIICamera` 返回驱动侧已钳制+量化的缓存 (SDK 无可靠 `ExpoTime` getter)。
> 新增后端请实现它, 否则会静默退回缓存 —— 那正是 issue #32 的缺陷形态。

> **曝光「改完」也不等于「下一帧就是新曝光」** (见下方已知坑第 4 条)。
> 大恒在取流进行中直接写 `ExposureTime`, 改动之前就开始积分的帧仍留在 SDK 流水线里;
> `DahengCamera.reset_exposure_time` 因此在曝光真的改变后消费掉
> `EXPOSURE_CHANGE_DISCARD_FRAMES`(=3) 帧, 让「改曝光后的第一帧就是新曝光」成为
> **驱动层的不变量**。其他后端若也有此行为, 必须在各自的曝光写入口做同样的事 ——
> `get_numpy_image` 自带的 `skip_first` 只丢 1 帧, **不够**。

## BaseCamera 接口

`BaseCamera` 定义相机驱动的统一接口 (继承 `Device` 的相机需实现):

| 方法 | 说明 |
|------|------|
| `initialize()` | 初始化相机 |
| `open()` | 打开相机 |
| `close()` | 关闭相机 |
| `reset_exposure_time(time_ms)` | 设置曝光时间 (MIICAM 须 Stop→设置→重启拉流) |
| `reset_window(center, size)` | 设置 ROI 窗口 |
| `get_numpy_image(n_sample, skip_first)` | 获取图像 |
| `enable_auto_exposure(enable, mode)` | 启用/禁用自动曝光 |
| `get_auto_exposure_state()` | 获取自动曝光状态 |
| `set_auto_exposure_range(...)` | 设置自动曝光范围 |
| `get_cam_list()` | 获取可用相机列表 (static) |
| `pixel_size_um` | 传感器像元边长 (µm), **类属性**, 默认 `0` = 未知 |
| `require_pixel_size_um()` | 取像元边长; 未知时抛 `CameraError` 并说明补救办法 |

具体驱动 (MIICAM/FFmpeg/Daheng) 还提供 `set_auto_exposure_target(target)` 方法 (非抽象, 由各驱动自行实现)。

### 像元尺寸 `pixel_size_um` (2026-10-29)

像元尺寸决定全部远场物理标度 (像素 ↔ 空间频率 → 焦距 / 束腰 / 目标边长), 因此它是
**驱动的类属性**, 不是各脚本的 CLI 默认值 —— 调用方一律读 `cam.pixel_size_um`
(或 `cam.require_pixel_size_um()`), 不得再硬编码数值。

| 后端 | 来源 | 行为 |
|------|------|------|
| `BaseCamera` | — | `pixel_size_um = 0.0` (未知)。`require_pixel_size_um()` 抛 `CameraError`, 错误信息带后端名与补救命令 |
| `DahengCamera` | 型号级配置 `<CCD_CONFIG_DIR>/ccd/models.json` → 已知型号表 → 报错 | 属性按型号解析 (见下); **未登记就报错**, 不猜数 |
| `MIICamera` | 写死 | `pixel_size_um = 2.9` —— 该后端目前只支持一款相机, 换型号改这一处 |
| `SimPibCCD` (`--cam-type sim`) | 孪生口径 | `pixel_size_um = CCD_PIXEL_UM` (与 `SimPibSystem` 的远场标度同源) |
| `SimulatedCCD` | 构造参数 | `SimulatedCCD(pixel_size_um=...)`, 默认 `0` (通用仿真相机没有固定物理口径) |
| `FFmpegCamera` / `ImageFolderCamera` | — | 沿用基类 `0` (视频/图集回放没有可靠的物理标度) |

**大恒为什么按型号而不是按序列号**: 像元尺寸是**型号属性**, 同一型号的多台相机共用同一个值;
按序列号存 (`CCDParams`) 会让每接一台新相机都要重新登记同一个数。配置文件
`data/ccd_configs/ccd/models.json`:

```json
{
  "MER2-507-23GM": {"pixel_size_um": 2.2}
}
```

解析顺序: 型号配置 → `KNOWN_MODEL_PIXEL_SIZE_UM` (历史口径, 前缀匹配, 只作回落) →
抛 `CameraError` (信息里给出登记命令)。实现集中在零导入叶子
`drivers/ccd/pixel_size.py` (不 import 任何 SDK, 模块作用域不读盘)。

**缺登记时怎么补** (驱动只报错、不交互, 交互在脚本/命令行侧):

```bash
python -m ao_shaping.drivers.ccd.pixel_size                 # 列出已登记型号
python -m ao_shaping.drivers.ccd.pixel_size --prompt        # 询问当前连接的大恒相机型号并录入
python -m ao_shaping.drivers.ccd.pixel_size --store MER2-507-23GM 2.2
```

脚本侧对应 `cam.ensure_pixel_size_um(prompt=input)` (未登记则提问 + 写入配置 + 本实例生效)。

**runner / 脚本的统一入口**是 `camera_pixel_size_um_for(cam_type, cam_id, *, camera=None, prompt=None)`:
命令行把「未指定」写成 `0`, 由它按后端换成真实值 —— 已打开的 `camera` 实例优先
(`cam.require_pixel_size_um()`), 否则按类型查型号配置 (大恒**只枚举设备不打开**),
`miicam` 取写死的 2.9, `sim` 取孪生口径。型号未登记时, 交互终端才提问 (默认写盘),
非交互环境直接抛驱动侧的 `CameraError`。

⚠️ **GenICam 的 `PixelSize` 特征不是像元边长** —— 它是**位深** (`GxPixelSizeEntry`: BPP8=8 …),
读它当像元尺寸会得到 `8 µm`。物理像元间距只能来自型号配置或器件手册。

## 已知坑

- **Daheng gxipy `IntFeature.set` 越界会「静默保留旧窗口」** (SDK gxiapi.py `IntFeature.set`/`range_check`): 请求值不在特征 `[min, max, inc]` 网格 (如 `Height.range=[2, 1138, 2]`) 时, gxipy **不抛异常**, 只 `print()` 噪声 "IntFeature.set: int_value out of bounds, ..." 然后**保留上一次的窗口** — 调用方拿到的仍是旧窗口而无从知晓 (2026-09-23 实测: 开窗设置/关闭时缓冲报错, 窗口被钳到 930×632)。修复: `DahengCamera.reset_window` 在 set 前用 `_clamp_int(value, feature.get_range())` 把宽高/偏移钳到合法网格 (warning 记录钳制), 并 readback (`Width.get()`/`Height.get()`/`OffsetX.get()`) 校验 — 若 SDK 确保留旧值则 warning + 返回实际生效窗口 `((w,h), center)`。调用方必须用返回的生效尺寸/中心, 不要拿请求值当生效值。新代码一律经 `reset_window` 返回的 readback 值; 已在 `tests/ao_shaping/drivers/ccd/daheng/test_reset_window.py` 离线锁定 (钳制/读回/全帧分支)。
- **MiiCam 曝光修改**: 须通过 `reset_exposure_time()` (Stop→put_ExpoTime→重启拉流); 曝光 <0.1ms 信号淹没在噪声 (max≤10), 建议 ≥0.2ms; 峰值可能锁在 ~85, 质量指标优先用 mean/亮区而非 max。
- **MiiCam 响应呈台阶状, 比例迭代存在死区** (2026-10-22 实测, 探针 `scripts/ccd/miicam_auto_exposure_probe.py`, 产物 `data/debug/miicam_auto_exposure_2026102x_*`): 当前光照下峰值/均值随曝光不是线性而是阶梯 —— 档位 (设备曝光, 均值 counts): ≤17us≈31, 18-22us≈42, 25-30us≈51, 40us≈65, 50us≈71; 峰值对应 ≈185/220/242/255/255 (40us 起饱和)。台阶边界在 17→18us (+6% 曝光 → +35% 均值)、22→25us、~35→40us。因此 `auto_exposure` 的比例迭代: ① 目标带落在台阶之间时在两个相邻档位间振荡直至 `max_iterations` (有界不收敛, 每次约 17s/20 次修改); ② 目标低于最低档位时正确地在最小曝光边界退出 (1 次修改); ③ 目标带覆盖某档位时正常收敛 (0-4 次修改)。另外该场景 0.02ms 峰值实测 200-220 (2026-10-22 晚, 半小时内漂移 ±5%), 而非早前记录的 140; 曝光 ≤11us 亦有 ~180 counts (旧记录 "<0.1ms 淹没在噪声" 是当时光照条件下的结论, 已不适用)。**帧信息 `MiicamFrameInfoV3.expotime` 在本机不回读** (flag=0x3, 无 `MIICAM_FRAMEINFO_FLAG_EXPOTIME`=0x04, 字段恒 0) —— 设备侧曝光真值只能取驱动缓存 (钳制到 [0.011, 10000] ms 后 **未量化到整数微秒**, `put_ExpoTime` 内部截断; 如请求 0.0117ms 缓存记 0.0117 而设备用 11us, 偏差可达 ~6%)。脏数据排查结论: 帧序号逐帧 +1、时间戳恒 12572us 推进、相邻帧平均绝对差 ~2 counts 无重复帧、交叉切换测试 (16/17/18/19us 多轮交替) 亮度确定性地跟随请求且切换后无滞后 (lag 0-3%) —— **台阶是传感器/SDK 曝光映射的真实属性, 不是读到脏数据**。
- **曝光回读 ≠ 缓存值, 差一个微秒也会骗人** (issue #32, `get_camera_exposure_ms` 回读失真): 两个后端都会把曝光**钳制并量化到整数微秒**才下发给 SDK —— Daheng 是 `ExposureTime.set(int(ms*1000))`, MiiCam 是 `put_ExpoTime(int(ms*1000))` —— 但驱动侧的 `exposure_time_ms` 缓存保留的是**未经量化的原始请求**, 且从不随设备回读刷新。于是「日志里记的曝光」与「传感器实际用的曝光」长期漂移, 一切依赖曝光真值的判据 (饱和保护、δ 扫描、线性度) 都建立在错值上。**契约: 一律经 `get_exposure_time_ms()` 取值** —— Daheng 读 GenICam 特征拿设备真值; MiiCam 因 SDK 无可靠 getter, 以「已钳制+量化」的缓存为权威 (**不可**改回未量化的请求, 那会虚报一个设备并未使用的曝光)。`get_camera_exposure_ms()` 已按此优先级实现, 离线锁定于 `tests/ao_shaping/drivers/ccd/test_camera_registry.py`。
- **MiiCam 位深**: 实测该硬件 `_detect_raw_format` 报 `MONO16` 而非 `MONO14` (test_ccd.py::test_cam_bit_depth_initialization 预期 MONO14 会失败 — 硬件现实, 非代码缺陷)。
- **Daheng 改曝光后, 前 1~2 帧仍是上一档曝光的画面** (2026-10-21 实测, 相机 FJB24112232): 大恒 SDK 在**取流进行中**直接写 GenICam `ExposureTime`, **改动之前**就已经开始积分的帧仍留在 SDK 流水线里, 于是改完曝光后紧接着取到的帧还是旧曝光的画面。实测阶梯 (每次改动后连抓 4 个原始帧, mean/max): `2 -> 20 ms` 的前两帧是 0.343/5、0.320/5 (仍是 2 ms), 第三帧才出现 20 ms 的 0.559/42; `20 -> 2 ms` 的前两帧是 0.596/43、0.596/41 (仍是 20 ms); `1 -> 50`、`10 -> 100 ms` 滞后 1 帧; `100 -> 10 ms` 那次没滞后 ⇒ **间歇发生, 不能靠「发现不一致再补救」, 必须无条件消费**。
  - **实测危害**: 2026-10-20 录下的测光阶梯 (`data/debug/light_level_probe/20261020_092550`) 就这样把 5 ms 档的峰值记成了 2 ms 档的值 (17 == 17), 而相邻两帧之差大到 ≈ 峰值本身 (50 ms 档 delta=156 vs max=158) —— 那一档的读数是旧曝光的画面。**任何「改曝光后立刻测量」的路径都会读到错档**: `common.resolve_exposure_ms` 的曝光二分 (用旧曝光的峰值判断新曝光 → 收敛到错误曝光)、`common.auto_exposure` 比例环、`diagnose_light_level.py` 阶梯扫描。
  - **修复**: `DahengCamera.reset_exposure_time` 在曝光**真的改变**后消费掉 `EXPOSURE_CHANGE_DISCARD_FRAMES`(=3) 帧 (留 1 帧余量; 实测滞后上限 2 帧), 使「改曝光后的第一帧就是新曝光」成为驱动层不变量; 曝光没变则不消费 (避免白花 3 帧采集时间)。代价是 3 帧的采集时间 (曝光 100 ms 时约 300 ms)。离线锁定于 `tests/ao_shaping/drivers/ccd/daheng/test_reset_exposure_time.py` (假相机模拟 SDK 排队: 改曝光时旧帧留在队列里)。
  - **注意**: `get_numpy_image` 的 `skip_first=True` 只丢 1 帧, **不足以**覆盖本坑; 且 MiiCam 走 Stop→设置→重启拉流, 天然没有这个问题 —— 两者不可混为一谈。
  - **复核 (2026-10-22)**: 因上述错档, 2026-10-20 的三份阶梯记录 (`data/debug/light_level_probe/20261020_092526` / `20261020_092550` / `20261020_093523`) **整份作废**, 只保留作危害实例; 引用了它们的**定量**结论必须重验。修复后用 `diagnose_light_level.py` 重录 (相机 FJB24112232, 2026-10-22 09:02, `data/debug/light_level_probe/20261022_090226`): 0.05 ms→2 counts (读出噪声底)、0.2→5、1→24、2→45、5→109、10→213、20→255 (饱和) —— **逐档严格线性, 无错档台阶** ⇒ 本台实验系统当前可用曝光窗口 **1–10 ms**, 20 ms 起饱和; 阶梯不再需要为「错档」留冗余档位。
  - **另一个独立陷阱 (同源不同因)**: 曝光上限估计 (倍增阶梯 + 线性外推) 会被**噪声底**毒化 —— 阶梯最低档画面无光、峰值只有读出噪声 (2 counts), 而估计器原先只跳过「零峰」帧, 于是噪声帧进入外推: `0.02 ms × 0.9 / (2/255) = 2.295 ms`, × headroom 0.9 = **2.0655 ms** (比真值低约 30 倍), 随后的安全检查因此拒掉合法曝光。已修为「峰值须显著高于该帧自身噪声底」(稳健量 1.4826·MAD, k=4; 见 `utils/image/exposure_control.py`); 真机复核: 2.0655 ms → **75.5 ms**。**两个坑会同时出现**: 陈旧帧让读数错档, 噪声底让上限被压低 —— 修一个不够。

## 包入口

`ccd/__init__.py` 按独立 try/except 导出相机后端 (无别名覆盖):

```python
# MIICamera 与 DahengCamera 为独立导出 (无别名覆盖),
# 各自 try/except 优雅降级 (SDK 不可用时置 None)
MIICamera = None
MIICAMError = None
DahengCamera = None
```

详细接口文档见 [`INTERFACE_DOCS.md`](../INTERFACE_DOCS.md) §相机/CCD 接口。
