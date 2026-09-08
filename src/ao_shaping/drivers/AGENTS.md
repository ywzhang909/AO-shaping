# drivers/ - Hardware Drivers

Hardware SDK wrappers for SLM, DM, WFS, CCD, TM devices.

## STRUCTURE

```
drivers/
├── device_base.py       # Device base class (Device, DeviceState, DeviceType)
├── device_registry.py    # Device registration and management
├── mock_devices.py      # Mock devices for testing
├── ccd/                 # Cameras
│   ├── base.py          # BaseCamera abstract class
│   ├── daheng.py        # Daheng GigE camera
│   ├── miicam.py        # MiiCam SDK
│   └── miicam_device.py # MIICAMDevice
├── dm/                  # Deformable Mirrors
│   ├── base.py          # DM abstract class
│   ├── NLight.py        # NLight DM driver
│   └── simulateDM.py    # Simulation DM
├── slm/                 # Spatial Light Modulators
│   ├── santec_slm200.py           # Santec SLM-200 SDK (权威驱动)
│   ├── santec_slm200_constants.py # SLM-200 硬件常量 (翻转时间等)
│   ├── slm_calibration.py
│   ├── zernike_slm.py             # Zernike coefficient-driven SLM
│   ├── wavefront_correction.py    # Wavefront error correction (CSV → correction map)
│   └── _slm_win.py                # SDK bindings (internal)
├── wfs/                 # Wavefront Sensors
│   └── ThorlabWFS.py  # Thorlabs WFS
├── tm/                  # Timing Modules
│   └── serial_port_fsm.py
└── sim/                 # Simulation (digital twin)
    ├── base.py          # SimulatedDevice, OpticalDevice
    ├── ccd/             # SimulatedCCD
    ├── laser/           # SimulatedLaser
    ├── optics/          # SimulatedSLM, SimulatedLens
    └── atmos/           # Turbulence screens
```

## KEY CLASSES

| Class | File | Purpose |
|-------|------|---------|
| `Device` | device_base.py | Base class for all devices |
| `DeviceState` | device_base.py | State enum (DISCONNECTED, READY, BUSY, etc.) |
| `DeviceType` | device_base.py | Type enum (CAMERA, SLM, DM, WFS, etc.) |
| `NLightDM` | dm/NLight.py | DM control |
| `BaseDM` | dm/base.py | DM abstract base |
| `ThorlabWFS` | wfs/ThorlabWFS.py | WFS control |
| `WavefrontCorrection` | slm/wavefront_correction.py | SLM wavefront error correction (CSV loading, outlier detection, correction map) |

## REQUIRED INTERFACE

All drivers must implement:
```python
class Device(ABC):
    @abstractmethod
    def open(self) -> None: ...
    @abstractmethod
    def close(self) -> None: ...
    @abstractmethod
    def is_connected(self) -> bool: ...
    @abstractmethod
    def get_hardware_info(self) -> dict: ...
    
    # Context manager support
    def __enter__(self) -> "Device": ...
    def __exit__(self, ...): ...
```

## CONVENTIONS

- Custom exceptions: `*Error` suffix (e.g., `DeviceError`, `CameraError`)
- State tracking via `_set_state(state, error_msg)` 
- SDK imports in `__init__`, handle failure gracefully with try/except
- Use `loguru.logger` for logging
- Parameter registration via `register_parameter()` method

## HARDWARE FACTS (2026-09 实测确认, 见 tools/slm/slm_diagnose.py)

| 设备 | 确认信息 |
|------|---------|
| Santec SLM200 SLM#1 | 序列号 22030108; 1920×1200; 10-bit; 内部内存模式; @1064nm 2π=993 灰度 (设备动态查询) |
| MiiCam 相机 | 序列号 TP2408221418059418FD83E3A448D82; 2688×1520; MONO8; `reset_exposure_time()` 修改曝光必须 Stop→设置→重启拉流 (直接 put_ExpoTime 不生效) |
| 2f Fourier 光路 | SLM 前焦面 125mm → f=125mm 透镜 → CCD 后焦面; CCD 坐标=空间频率; 0 级=帧全局最大 (光轴落点), 非相机几何中心 |

### SLM 故障排查 (2026-09 固化)

`tools/slm/slm_diagnose.py` 三步自检定位"面板不调制光"类故障:

1. **freeze (面板冻结)**: flat/全屏光栅/上下半屏光栅写**轮换内存槽** → 帧必须随图案变化 (≥2/3 帧不同才 PASS)。
2. **modulate (调制能力)**: `set_grayscale` 0→1023 扫描, 0 级桶能量须有 ~993 灰度周期 (相对变化 >15% 才 PASS); 无周期 ⇒ 无振幅耦合 ⇒ 面板不调制。
3. **linearity (到达光强)**: 曝光 ×4、×20, 峰值亮度须增长 (>2× 才 PASS); 恒定峰值 ⇒ 到达相机光强比已知 ~0.02ms 近饱和基线弱 >100×。

### 已知坑 (etched in drivers)

- **DVI 模式 `open()` 会挂起**: `video_mode=1` 可挂 120s/300s, 且挂起后 memory 模式也挂直到**物理断电**。诊断脚本永不自动切 DVI。
- **diff-shaping 硬件闭环挂起 (2026-09-08)**: 日志止于 `成功打开SLM #1`, 之后首次 `get_numpy_image()` 前无限阻塞 (900s 强杀; 分步探针脚本同挂)。`WaitImageV3` 原生等待由 SDK 内部驱动, 不受 Python 超时保护。强杀不会走 finally close → SLM 控制器可能残留异常态; 重跑前确认无残留进程, memory 模式 open() >数秒无日志则物理断电重置 (与 DVI 挂起同一处置)。见 `diff_shaping_runner.py` SLM 连接段注释。
- **`get_displayed_memory_number` 报错码 1 正常**: set_grayscale 模式无内存槽显示, SLM_Ctrl_ReadDS 返回 1 是合理行为, 非故障。
- **同内存槽连续 `display_memory` 是 no-op**: 固件对"正在显示的同一槽位"的 `display_memory(slot)` 不刷新 LCOS 面板 (第二次写入的新相位不会上屏)。驱动已在 `display_memory()` 内置 warning: 检测到连续同槽位调用时 `logger.warning` 提示轮换槽位 (`display_data()` 内部自动轮换 127 槽, 不受影响; 工具类如 `gray_response._display_rotate_slot` 已自带轮换)。
- **MiiCam 曝光修改**: 须 `reset_exposure_time()` (Stop→put_ExpoTime→重启拉流); 曝光 <0.1ms 信号淹没在噪声 (max≤10), 建议 ≥0.2ms; 峰值可能锁在 ~85, 质量指标优先用 mean/亮区而非 max。
