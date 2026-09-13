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
├── miicam_driver.py     # MIICAM 相机驱动 (BaseCamera 基类)
└── __init__.py          # 包入口 (MIICamera, DahengCamera, FFmpegCamera 等)
```

## 关键类

| 类 | 文件 | 说明 |
|----|------|------|
| `BaseCamera` | `base.py` | 相机抽象基类 (ABC) |
| `CameraError` | `base.py` | 相机异常基类 |
| `CameraStreamManager` | `miicam/driver.py` | MiiCam 相机流管理器 |
| `MIICAMError` | `miicam/driver.py` | MiiCam 特定异常 |
| `DahengCamManager` | `daheng/driver.py` | 大恒相机管理器 |
| `FFmpegCamera` | `ffmpeg.py` | FFmpeg 相机 |
| `ImageFolderCamera` | `ffmpeg.py` | 图片文件夹相机 |

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

- **MiiCam 曝光修改**: 须通过 `reset_exposure_time()` (Stop→put_ExpoTime→重启拉流); 曝光 <0.1ms 信号淹没在噪声 (max≤10), 建议 ≥0.2ms; 峰值可能锁在 ~85, 质量指标优先用 mean/亮区而非 max。

## 包入口

`ccd/__init__.py` 按优先级自动选择相机后端:

```python
# 若 DahengCamManager 可用则作为 CameraStreamManager,
# 否则使用 MIICamera
if DahengCamera is not None:
    CameraStreamManager = DahengCamera
elif MIICamera is not None:
    CameraStreamManager = MIICamera
```

详细接口文档见 [`INTERFACE_DOCS.md`](../INTERFACE_DOCS.md) §相机/CCD 接口。
