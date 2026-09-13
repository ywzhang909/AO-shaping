# drivers/ - 硬件驱动

AO-Shaping 硬件驱动框架, 为 SLM、DM、WFS、CCD、TM 等设备提供统一接口。

## 目录结构

```
drivers/
├── AGENTS.md                # 本文件 (主文档)
├── INTERFACE_DOCS.md        # 硬件驱动接口文档 (架构/接口/扩展指南)
├── device_base.py           # Device 基类 (Device, DeviceState, DeviceType)
├── device_registry.py       # 设备注册与管理
├── mock_devices.py          # Mock 设备 (简单测试)
├── ccd/                     # 相机驱动 → [ccd/AGENTS.md](ccd/AGENTS.md)
├── dm/                      # 变形镜驱动 → [dm/AGENTS.md](dm/AGENTS.md)
├── slm/                     # SLM 驱动 → [slm/AGENTS.md](slm/AGENTS.md)
├── wfs/                     # 波前传感器驱动 → [wfs/AGENTS.md](wfs/AGENTS.md)
├── tm/                      # 定时模块驱动 → [tm/AGENTS.md](tm/AGENTS.md)
├── sim/                     # 模拟设备/数字孪生 → [sim/AGENTS.md](sim/AGENTS.md)
├── adc/                     # ADC 驱动
│   ├── driver.py
│   └── __init__.py
└── __pycache__/
```

## 各子包文档

| 子包 | 文档 | 主要内容 |
|------|------|----------|
| `ccd/` | [ccd/AGENTS.md](ccd/AGENTS.md) | 相机驱动 (BaseCamera, Daheng, MiiCam, FFmpeg) |
| `dm/` | [dm/AGENTS.md](dm/AGENTS.md) | 变形镜驱动 (NLight, Micro-DM, Hadamard) |
| `slm/` | [slm/AGENTS.md](slm/AGENTS.md) | SLM 驱动 (Santec SLM-200), 标定, 波前校正 |
| `wfs/` | [wfs/AGENTS.md](wfs/AGENTS.md) | 波前传感器 (Thorlab WFS) |
| `tm/` | [tm/AGENTS.md](tm/AGENTS.md) | 定时模块 (串口 FSM) |
| `sim/` | [sim/AGENTS.md](sim/AGENTS.md) | 模拟设备/数字孪生仿真 |

## 核心基类

### Device (`device_base.py`)

所有硬件驱动的抽象基类:

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
    def __enter__(self) -> "Device": ...
    def __exit__(self, ...): ...
```

### 设备类型枚举 (`DeviceType`)

```python
class DeviceType(Enum):
    CAMERA = auto()    # 相机/CCD
    SLM = auto()       # 空间光调制器
    DM = auto()        # 变形镜
    WFS = auto()       # 波前传感器
    STAGE = auto()     # 运动台
    LASER = auto()     # 激光器
    FILTER = auto()    # 滤光轮
    OTHER = auto()     # 其他设备
```

### 设备状态枚举 (`DeviceState`)

```python
class DeviceState(Enum):
    UNKNOWN = auto()
    DISCONNECTED = auto()
    CONNECTING = auto()
    READY = auto()
    BUSY = auto()
    ERROR = auto()
    CALIBRATING = auto()
```

## 约定 (Conventions)

| 约定 | 说明 |
|------|------|
| 自定义异常 | `*Error` 后缀 (如 `DeviceError`, `CameraError`) |
| 状态追踪 | `_set_state(state, error_msg)` |
| SDK 导入 | `try/except` 优雅处理, 不可用时 fallback |
| 日志 | `loguru.logger` |
| 参数注册 | `register_parameter()` 方法 |

详细接口规范见 [`INTERFACE_DOCS.md`](./INTERFACE_DOCS.md)。

## 硬件事实 (2026-09 实测确认)

| 设备 | 确认信息 |
|------|---------|
| Santec SLM200 SLM#1 | 序列号 22030108; 1920×1200; 10-bit; 内部内存模式; @1064nm 2π=993 灰度 (设备动态查询) |
| MiiCam 相机 | 序列号 TP2408221418059418FD83E3A448D82; 2688×1520; MONO8; `reset_exposure_time()` 修改曝光必须 Stop→设置→重启拉流 |
| 2f Fourier 光路 | SLM 前焦面 125mm → f=125mm 透镜 → CCD 后焦面; CCD 坐标=空间频率; 0 级=帧全局最大 (argmax), 非相机几何中心 |

### SLM 故障排查 (详细见 [slm/AGENTS.md](slm/AGENTS.md))

`tools/slm/slm_diagnose.py` 三步自检:

1. **freeze**: flat/全屏光栅/上下半屏光栅写轮换内存槽 → 帧必须随图案变化 (≥2/3 帧不同)
2. **modulate**: `set_grayscale` 扫描, 0 级桶能量须有 ~993 灰度周期 (相对变化 >15%)
3. **linearity**: 曝光 ×4、×20, 峰值亮度须增长 (>2×)

### 跨子包已知坑 (详细见各子包 AGENTS.md)

| 坑 | 影响子包 | 详情 |
|----|----------|------|
| DVI 模式 `open()` 会挂起 | slm | `video_mode=1` 可挂 120s/300s, 需物理断电重置 |
| diff-shaping 硬件闭环挂起 | slm | `WaitImageV3` 不受 Python 超时保护, 需确认无残留进程 |
| `get_displayed_memory_number` 报错码 1 正常 | slm | set_grayscale 模式无内存槽显示 |
| 同内存槽连续 `display_memory` 是 no-op | slm | 固件不刷新正在显示的槽位 |
| MiiCam 曝光修改 | ccd | 须 `reset_exposure_time()` (Stop→put_ExpoTime→重启拉流) |
| 类名 `NLight` 非 `NLightDM` | dm | dm/NLight.py 中类名为 `NLight`, 非 `NLightDM` |

## Mock 设备

`mock_devices.py` 提供简化模拟设备用于无硬件开发测试, 与 `sim/` (数字孪生) 区别在于复杂度较低。

详细使用见 [`INTERFACE_DOCS.md`](./INTERFACE_DOCS.md) §Mock 设备。

## 相关文档

| 文档 | 路径 | 内容 |
|------|------|------|
| 接口文档 | `drivers/INTERFACE_DOCS.md` | 架构概览、接口规范、扩展指南 |
| SLM 标定 | `drivers/slm/README.md` | 闪耀光栅法/零级比值法/干涉法/衍射效率法标定详解 |
| 方形光斑 SPGD | `docs/slm_square_spgd/README.md` | SPGD 整形硬件实测与故障记录 |
