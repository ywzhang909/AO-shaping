# ccd/ - 相机驱动

相机驱动模块，提供统一的相机接口。

## 结构

```
ccd/
├── base.py              # BaseCamera 抽象基类
├── common.py            # 通用工具
├── daheng/              # 大恒相机 (GigE)
│   ├── driver.py
│   └── __init__.py
├── ffmpeg/              # FFmpeg 相机
│   ├── driver.py
│   └── __init__.py
├── ffmpeg.py            # FFmpegCamera / ImageFolderCamera
├── gxipy/               # 国光像机 SDK 绑定
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
| `FFmpegCamera` | `ffmpeg.py` | FFmpeg 相机 |
| `ImageFolderCamera` | `ffmpeg.py` | 图片文件夹相机 |

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
| `get_camera_exposure_ms(cam)` | 读取当前曝光 (ms) |
| `set_camera_exposure_ms(cam, ms)` | 写曝光, 优先 `reset_exposure_time()` (MiiCam 必需) |
| `get_camera_exposure_range(cam)` | `(min_ms, max_ms)`, 后端未暴露时回退 `0.011–10000` |
| `auto_exposure(cam, target_max, tolerance=5.0, max_iterations=20, n_sample=1)` | 有原生 `auto_exposure` (Daheng/MiiCam) 则委托, 否则通用比例环 (ffmpeg) |

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

具体驱动 (MIICAM/FFmpeg/Daheng) 还提供 `set_auto_exposure_target(target)` 方法 (非抽象, 由各驱动自行实现)。

## 已知坑

- **Daheng gxipy `IntFeature.set` 越界会「静默保留旧窗口」** (SDK gxiapi.py `IntFeature.set`/`range_check`): 请求值不在特征 `[min, max, inc]` 网格 (如 `Height.range=[2, 1138, 2]`) 时, gxipy **不抛异常**, 只 `print()` 噪声 "IntFeature.set: int_value out of bounds, ..." 然后**保留上一次的窗口** — 调用方拿到的仍是旧窗口而无从知晓 (2026-09-23 实测: 开窗设置/关闭时缓冲报错, 窗口被钳到 930×632)。修复: `DahengCamera.reset_window` 在 set 前用 `_clamp_int(value, feature.get_range())` 把宽高/偏移钳到合法网格 (warning 记录钳制), 并 readback (`Width.get()`/`Height.get()`/`OffsetX.get()`) 校验 — 若 SDK 确保留旧值则 warning + 返回实际生效窗口 `((w,h), center)`。调用方必须用返回的生效尺寸/中心, 不要拿请求值当生效值。新代码一律经 `reset_window` 返回的 readback 值; 已在 `tests/ao_shaping/drivers/ccd/daheng/test_reset_window.py` 离线锁定 (钳制/读回/全帧分支)。
- **MiiCam 曝光修改**: 须通过 `reset_exposure_time()` (Stop→put_ExpoTime→重启拉流); 曝光 <0.1ms 信号淹没在噪声 (max≤10), 建议 ≥0.2ms; 峰值可能锁在 ~85, 质量指标优先用 mean/亮区而非 max。
- **MiiCam 位深**: 实测该硬件 `_detect_raw_format` 报 `MONO16` 而非 `MONO14` (test_ccd.py::test_cam_bit_depth_initialization 预期 MONO14 会失败 — 硬件现实, 非代码缺陷)。

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
