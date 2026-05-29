# AO-Shaping 硬件驱动接口文档

本文档详细描述了 AO-Shaping 项目中各个硬件设备的驱动接口，以及如何实现通用的设备驱动接口。

## 目录

1. [架构概览](#架构概览)
2. [Device 基类接口](#device-基类接口)
3. [设备类型与接口定义](#设备类型与接口定义)
4. [现有驱动实现](#现有驱动实现)
5. [模拟设备 (sim/)](#模拟设备-sim-)
6. [实现新驱动的指南](#实现新驱动的指南)
7. [设备注册与管理](#设备注册与管理)
8. [VISA 通信层](#visa-通信层)
9. [Mock 设备](#mock-设备)

---

## 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                     DeviceRegistry                              │
│                  (设备注册与管理中心)                            │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │  Device  │    │  Device  │    │  Device  │
    │ (Camera) │    │   (SLM)  │    │   (DM)   │
    └──────────┘    └──────────┘    └──────────┘
          │               │               │
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │BaseCamera│    │   SDK    │    │ DM (ABC) │
    │  (ABC)   │    │ 绑定层   │    │  (ABC)   │
    └──────────┘    └──────────┘    └──────────┘
          │                           │
    ┌──────────┐                ┌──────────┐
    │DahengCam │                │ NLightDM │
    │MIICAMDev │                └──────────┘
    └──────────┘

┌─────────────────────────────────────────────────────────────────┐
│               Simulated Devices (sim/ 模块)                      │
│          数字孪生模拟设备 - 集成 sim.digitaltwin                  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
    ┌─────────────────────┼─────────────────────┐
    │                     │                     │
    ▼                     ▼                     ▼
┌──────────┐        ┌──────────┐        ┌──────────┐
│Simulated │        │Simulated │        │Simulated │
│   CCD    │        │   SLM    │        │  Laser   │
└──────────┘        └──────────┘        └──────────┘
    │                     │                     │
    └─────────────────────┼─────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Base Classes        │
              │ (SimulatedDevice)     │
              │ (OpticalDevice)       │
              │ (WavefrontProcessor)  │
              └───────────────────────┘
                          │
    ┌─────────────────────┼─────────────────────┐
    │                     │                     │
    ▼                     ▼                     ▼
┌──────────┐        ┌──────────┐        ┌──────────┐
│Simulated │        │Simulated │        │Simulated │
│  Lens    │        │Aperture  │        │  ATP     │
└──────────┘        └──────────┘        └──────────┘

大气模拟: SimulatedTurbulentScreen, SimulatedThermalScreen
```

---

## Device 基类接口

所有硬件设备驱动都继承自 [`Device`](src/ao_shaping/drivers/device_base.py:129) 基类。该类位于 `device_base.py` 文件中。

### 抽象方法（必须实现）

| 方法 | 说明 | 抛出异常 |
|------|------|----------|
| [`open()`](src/ao_shaping/drivers/device_base.py:201) | 打开设备连接 | `ConnectionError`, `DeviceError` |
| [`close()`](src/ao_shaping/drivers/device_base.py:211) | 关闭设备连接 | - |
| [`is_connected()`](src/ao_shaping/drivers/device_base.py:216) | 检查设备是否已连接 | - |
| [`get_hardware_info()`](src/ao_shaping/drivers/device_base.py:221) | 获取硬件信息 | - |

### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `device_id` | `str` | 设备唯一标识符 |
| `metadata` | `DeviceMetadata` | 设备元数据 |
| `state` | `DeviceState` | 当前设备状态 |

### 状态枚举 ([`DeviceState`](src/ao_shaping/drivers/device_base.py:32))

```python
class DeviceState(Enum):
    UNKNOWN = auto()       # 未知状态
    DISCONNECTED = auto()  # 未连接
    CONNECTING = auto()    # 连接中
    READY = auto()         # 就绪
    BUSY = auto()          # 忙碌中
    ERROR = auto()         # 错误状态
    CALIBRATING = auto()   # 校准中
```

### 类型枚举 ([`DeviceType`](src/ao_shaping/drivers/device_base.py:19))

```python
class DeviceType(Enum):
    CAMERA = auto()   # 相机/CCD
    SLM = auto()      # 空间光调制器
    DM = auto()       # 变形镜
    WFS = auto()      # 波前传感器
    STAGE = auto()    # 运动台
    LASER = auto()    # 激光器
    FILTER = auto()  # 滤光轮
    OTHER = auto()    # 其他设备
```

---

## 设备类型与接口定义

### 1. 相机/CCD 接口 ([`BaseCamera`](src/ao_shaping/drivers/ccd/base.py:15))

相机驱动需继承 `BaseCamera` 抽象类：

```python
class BaseCamera(ABC):
    @abstractmethod
    def initialize(self) -> None:
        """初始化相机设备"""
        pass

    @abstractmethod
    def open(self) -> None:
        """打开相机"""
        pass

    @abstractmethod
    def close(self) -> None:
        """关闭相机"""
        pass

    @abstractmethod
    def reset_exposure_time(self, time_ms: int) -> int:
        """设置曝光时间"""
        pass

    @abstractmethod
    def reset_window(self, center, size) -> tuple:
        """设置ROI窗口"""
        pass

    @abstractmethod
    def get_numpy_image(self, n_sample=1, skip_first=True) -> np.ndarray:
        """获取图像"""
        pass

    @abstractmethod
    def enable_auto_exposure(self, enable=True, mode=1) -> bool:
        """启用/禁用自动曝光"""
        pass

    @abstractmethod
    def set_auto_exposure_target(self, target: int) -> int:
        """设置自动曝光目标"""
        pass

    @abstractmethod
    def get_auto_exposure_state(self) -> dict:
        """获取自动曝光状态"""
        pass

    @abstractmethod
    def set_auto_exposure_range(self, max_time_ms=350, min_time_ms=0, 
                                max_gain=300, min_gain=100) -> bool:
        """设置自动曝光范围"""
        pass

    @staticmethod
    @abstractmethod
    def get_cam_list():
        """获取可用相机列表"""
        pass
```

### 2. 变形镜/DM 接口 ([`DM`](src/ao_shaping/drivers/dm/base.py:4))

DM 驱动需继承 `DM` 抽象类：

```python
class DM(ABC):
    channel: int  # 通道数

    @abstractmethod
    def transform(self, cmd: np.ndarray) -> np.ndarray:
        """将命令转换为 DM 电压值"""
        pass

    @abstractmethod
    def send(self, cmd):
        """发送命令到 DM"""
        pass

    @abstractmethod
    def open(self):
        """打开 DM 连接"""
        pass

    @abstractmethod
    def close(self):
        """关闭 DM 连接"""
        pass

    @abstractmethod
    def get_actuator_positions(self):
        """获取致动器位置"""
        pass
```

### 3. 空间光调制器/SLM 接口

SLM 驱动通常需要实现以下功能（参考 [`SantecSLM200`](src/ao_shaping/drivers/slm/santec_slm200.py:35)）：

| 方法 | 说明 |
|------|------|
| `open()` | 打开 SLM 连接 |
| `close()` | 关闭 SLM 连接 |
| `set_wavelength(wavelength: int)` | 设置工作波长 |
| `write_phase(phase: np.ndarray, memory_number=1)` | 写入相位数据 |
| `display_memory(memory_number)` | 显示内存中的相位图 |
| `display_data(phase: np.ndarray)` | 直接显示相位数据 |
| `set_grayscale(gs: int)` | 设置灰度值 |

### 4. 波前传感器/WFS 接口

WFS 驱动通常需要实现以下功能（参考 [`MockWFS`](src/ao_shaping/drivers/mock_devices.py:693)）：

| 方法 | 说明 |
|------|------|
| `open()` | 打开 WFS 连接 |
| `close()` | 关闭 WFS 连接 |
| `measure_wavefront()` | 测量波前 |
| `fit_zernike(wavefront, n_modes)` | 拟合 Zernike 系数 |
| `get_spot_image()` | 获取斑点图像 |

### 5. 定时模块/TM 接口

定时模块驱动（参考 [`SerialPortFSM`](src/ao_shaping/drivers/tm/serial_port_fsm.py:8)）通常需要实现：

| 方法 | 说明 |
|------|------|
| `open()` | 打开串口连接 |
| `close()` | 关闭连接 |
| `send(x, y)` | 发送位置命令 |
| `send_in_queue(x, y)` | 队列方式发送 |
| `get_rx()` | 获取回读数据 |
| `wait_rx(timeout)` | 等待回读 |

---

## 现有驱动实现

### 1. 相机驱动

| 驱动 | 文件 | 说明 |
|------|------|------|
| [`CameraStreamManager`](src/ao_shaping/drivers/ccd/daheng.py) | `ccd/daheng.py` | 大恒相机 (GigE) |
| [`MiiCamDevice`](src/ao_shaping/drivers/ccd/miicam.py) | `ccd/miicam.py` | Mii相机 SDK |
| [`MIICAMDevice`](src/ao_shaping/drivers/ccd/miicam_device.py:33) | `ccd/miicam_device.py` | MIICAM 相机 (Device基类) |
| [`MockCamera`](src/ao_shaping/drivers/mock_devices.py:25) | `mock_devices.py` | 模拟相机 |

### 2. SLM 驱动

| 驱动 | 文件 | 说明 |
|------|------|------|
| [`SantecSLM200`](src/ao_shaping/drivers/slm/santec_slm200.py:35) | `slm/santec_slm200.py` | Santec SLM-200 SDK |
| [`SantecSLM200Visa`](src/ao_shaping/drivers/slm/santec_slm200_visa.py) | `slm/santec_slm200_visa.py` | Santec SLM-200 VISA |
| [`MockSLM`](src/ao_shaping/drivers/mock_devices.py:250) | `mock_devices.py` | 模拟 SLM |

### 3. DM 驱动

| 驱动 | 文件 | 说明 |
|------|------|------|
| [`NLight`](src/ao_shaping/drivers/dm/NLight.py:17) | `dm/NLight.py` | NLight 变形镜 |
| [`SimulateDM`](src/ao_shaping/drivers/dm/simulateDM.py) | `dm/simulateDM.py` | 模拟 DM |
| [`MockDM`](src/ao_shaping/drivers/mock_devices.py:458) | `mock_devices.py` | 模拟 DM |

### 4. WFS 驱动

| 驱动 | 文件 | 说明 |
|------|------|------|
| [`ThorlabWFS`](src/ao_shaping/drivers/wfs/ThorlabWFS.py) | `wfs/ThorlabWFS.py` | Thorlabs WFS |
| [`MockWFS`](src/ao_shaping/drivers/mock_devices.py:693) | `mock_devices.py` | 模拟 WFS |

### 5. 其他驱动

| 驱动 | 文件 | 说明 |
|------|------|------|
| [`SerialPortFSM`](src/ao_shaping/drivers/tm/serial_port_fsm.py:8) | `tm/serial_port_fsm.py` | 串口定时模块 |
| [`MockLaser`](src/ao_shaping/drivers/mock_devices.py:1114) | `mock_devices.py` | 模拟激光器 |
| [`MockStage`](src/ao_shaping/drivers/mock_devices.py:915) | `mock_devices.py` | 模拟运动台 |
| [`MockFilter`](src/ao_shaping/drivers/mock_devices.py:1341) | `mock_devices.py` | 模拟滤光轮 |

---

## 模拟设备 (sim/)

`sim/` 模块提供高级模拟设备，集成 `sim.digitaltwin` 数值仿真引擎，用于数字孪生和端到端光学系统仿真。

### 与 mock_devices 的区别

| 特性 | mock_devices | sim/ |
|------|-------------|------|
| 用途 | 简单测试/开发 | 数值仿真/数字孪生 |
| 实现 | 简化模拟 | 集成 sim.digitaltwin |
| 复杂度 | 基础 | 高级物理模型 |
| 继承 | BaseCamera, Device | SimulatedDevice |

### 基类接口

#### SimulatedDevice ([`base.py:35`](src/ao_shaping/drivers/sim/base.py:35))

所有模拟设备的基类：

```python
class SimulatedDevice(Device):
    device_type = DeviceType.OTHER
    manufacturer = "Simulation"
    
    def __init__(self, device_id="", enable_noise=True, random_seed=None):
        ...
    
    @abstractmethod
    def compute(self, *args, **kwargs) -> Any:
        """执行仿真计算"""
        pass
    
    def reset(self) -> None:
        """重置仿真状态"""
    
    def set_seed(self, seed: int) -> None:
        """设置随机种子"""
    
    def set_noise(self, enabled: bool) -> None:
        """启用/禁用噪声"""
```

#### OpticalDevice ([`base.py:173`](src/ao_shaping/drivers/sim/base.py:173))

光学仿真设备基类：

```python
class OpticalDevice(SimulatedDevice):
    def __init__(self, device_id="", wavelength=1064.0, ...):
        self.wavelength = wavelength
    
    def set_input(self, wave: Any) -> None:
        """设置输入波前"""
    
    def get_output(self) -> Any:
        """获取输出波前"""
    
    @abstractmethod
    def process(self, wave: Any) -> Any:
        """处理波前"""
        pass
```

#### WavefrontProcessor ([`base.py:247`](src/ao_shaping/drivers/sim/base.py:247))

波前处理器基类（SLM、透镜等）：

```python
class WavefrontProcessor(OpticalDevice):
    def __init__(self, device_id="", wavelength=1064.0, npix=512, dpix=1e-3, ...):
        ...
    
    def set_phase(self, phase: np.ndarray) -> None:
        """设置相位图"""
    
    def get_phase(self) -> np.ndarray:
        """获取当前相位"""
```

### 现有模拟设备实现

#### 1. 相机/CCD

| 驱动 | 文件 | 说明 |
|------|------|------|
| [`SimulatedCCD`](src/ao_shaping/drivers/sim/ccd/simulated_ccd.py:25) | `sim/ccd/simulated_ccd.py` | 模拟 CCD 相机 |

```python
from ao_shaping.drivers.sim.ccd import SimulatedCCD

cam = SimulatedCCD(resolution=(1024, 1024), noise_level=5.0)
with cam:
    img = cam.get_numpy_image()
    print(f"图像形状: {img.shape}")
```

#### 2. 激光器

| 驱动 | 文件 | 说明 |
|------|------|------|
| [`SimulatedLaser`](src/ao_shaping/drivers/sim/laser/simulated_laser.py:25) | `sim/laser/simulated_laser.py` | 模拟激光器 |

```python
from ao_shaping.drivers.sim import SimulatedLaser

laser = SimulatedLaser(power=100, wavelength=1064, aperture=0.2)
with laser:
    wave = laser.generate()
    print(f"波前功率: {laser.power}W")
```

#### 3. 光学元件

| 驱动 | 文件 | 说明 |
|------|------|------|
| [`SimulatedSLM`](src/ao_shaping/drivers/sim/optics/simulated_slm.py:23) | `sim/optics/simulated_slm.py` | 模拟空间光调制器 |
| [`SimulatedLens`](src/ao_shaping/drivers/sim/optics/simulated_slm.py:233) | `sim/optics/simulated_slm.py` | 模拟透镜 |
| [`SimulatedAperture`](src/ao_shaping/drivers/sim/optics/simulated_slm.py:306) | `sim/optics/simulated_slm.py` | 模拟光阑 |

```python
from ao_shaping.drivers.sim import SimulatedSLM, SimulatedLens, SimulatedAperture

# SLM
slm = SimulatedSLM(resolution=(1920, 1080), wavelength=1064)
with slm:
    phase = np.random.rand(1080, 1920) * 2 * np.pi
    slm.set_phase(phase)
    output = slm.process(input_wave)

# 透镜
lens = SimulatedLens(focus_length=0.5, wavelength=1064)
with lens:
    focused = lens.process(input_wave)

# 光阑
aperture = SimulatedAperture(radius=0.05)
with aperture:
    masked = aperture.process(input_wave)
```

#### 4. 大气模拟

| 驱动 | 文件 | 说明 |
|------|------|------|
| [`SimulatedTurbulentScreen`](src/ao_shaping/drivers/sim/atmos/screens.py:18) | `sim/atmos/screens.py` | 湍流相位屏 |
| [`SimulatedThermalScreen`](src/ao_shaping/drivers/sim/atmos/screens.py:190) | `sim/atmos/screens.py` | 热晕相位屏 |
| [`SimulatedATP`](src/ao_shaping/drivers/sim/atmos/screens.py:296) | `sim/atmos/screens.py` | 大气传输仿真 |

```python
from ao_shaping.drivers.sim.atmos import (
    SimulatedTurbulentScreen,
    SimulatedThermalScreen,
    SimulatedATP
)

# 湍流屏
turb = SimulatedTurbulentScreen(Cn2=1e-15, L0=1.0, l0=0.01)
with turb:
    wf_turb = turb.process(input_wave)

# 热晕屏
thermal = SimulatedThermalScreen(absorb=1e-5, wind_x=2.0)
with thermal:
    wf_thermal = thermal.process(input_wave)

# 大气传输
atp = SimulatedATP(prop_dist=3000, layers=10, Cn2=1e-15)
with atp:
    wf_propagated = atp.propagate(input_wave)
```

### 端到端仿真示例

```python
from ao_shaping.drivers.sim import (
    SimulatedLaser,
    SimulatedSLM,
    SimulatedCCD,
)
from ao_shaping.drivers.sim.atmos import SimulatedTurbulentScreen

# 创建仿真链路
laser = SimulatedLaser(power=100, wavelength=1064)
slm = SimulatedSLM(resolution=(1920, 1080))
turb = SimulatedTurbulentScreen(Cn2=1e-14)
ccd = SimulatedCCD(resolution=(512, 512))

# 仿真流程
with laser, slm, turb, ccd:
    # 生成波前
    wave = laser.generate()
    
    # 设置SLM相位
    phase = np.zeros((1080, 1920))
    slm.set_phase(phase)
    
    # 通过SLM调制
    wave = slm.process(wave)
    
    # 通过大气湍流
    wave = turb.process(wave)
    
    # CCD探测
    img = ccd.get_numpy_image()

print(f"探测图像: {img.shape}, 强度: {img.mean():.2f}")
```

---

## 实现新驱动的指南

### 方式一：继承 Device 基类（推荐）

推荐用于需要数字孪生功能的设备：

```python
from ao_shaping.drivers.device_base import (
    Device, DeviceType, DeviceState, DeviceError
)
import numpy as np

class MyCameraError(DeviceError):
    """自定义错误类"""
    pass

class MyCamera(Device):
    """我的相机驱动"""
    
    # 类级别设备标识
    device_type = DeviceType.CAMERA
    manufacturer = "MyCamera"
    model = "MC-100"
    version = "1.0.0"
    
    def __init__(self, device_id: str = ""):
        super().__init__(device_id)
        self._register_parameters()
        self._register_capabilities()
    
    def _register_parameters(self) -> None:
        """注册相机参数"""
        self.register_parameter(
            "exposure_time_ms",
            default_value=20.0,
            min_value=0.1,
            max_value=10000.0,
            unit="ms",
            description="曝光时间"
        )
        self.register_parameter(
            "gain",
            default_value=1.0,
            min_value=1.0,
            max_value=100.0,
            description="模拟增益"
        )
    
    def _register_capabilities(self) -> None:
        """注册相机能力"""
        self.register_capability(
            "capture",
            description="拍摄单张图像",
            return_type=np.ndarray
        )
        self.register_capability(
            "get_resolution",
            description="获取分辨率",
            return_type=tuple
        )
    
    def open(self) -> None:
        """打开相机连接"""
        self._set_state(DeviceState.CONNECTING)
        # 实现连接逻辑...
        self._set_state(DeviceState.READY)
    
    def close(self) -> None:
        """关闭相机连接"""
        self._set_state(DeviceState.DISCONNECTED)
    
    def is_connected(self) -> bool:
        """检查连接状态"""
        return self._state == DeviceState.READY
    
    def get_hardware_info(self) -> dict:
        """获取硬件信息"""
        return {
            "serial_number": "SN12345",
            "firmware_version": "1.0.0",
            "resolution": (1920, 1080)
        }
    
    def capture(self, n_samples: int = 1) -> np.ndarray:
        """拍摄图像"""
        if not self.is_connected():
            raise RuntimeError("相机未连接")
        
        self._set_state(DeviceState.BUSY)
        try:
            # 实现拍摄逻辑...
            return np.zeros((1080, 1920), dtype=np.uint8)
        finally:
            self._set_state(DeviceState.READY)
```

### 方式二：继承特定设备的抽象基类

对于相机和 DM，可以使用特定的抽象基类：

```python
from ao_shaping.drivers.ccd.base import BaseCamera, CameraError

class MyCamera(BaseCamera):
    """继承 BaseCamera 的相机驱动"""
    
    def __init__(self, cam_id: int = 0, exposure_time_ms: int = 20):
        super().__init__(cam_id, exposure_time_ms)
    
    def initialize(self) -> None:
        # 实现初始化
        pass
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    
    def reset_exposure_time(self, time_ms: int) -> int:
        # 实现
        return time_ms
    
    def reset_window(self, center, size):
        # 实现
        return (size, center)
    
    def get_numpy_image(self, n_sample=1, skip_first=True):
        # 实现
        return np.zeros((1080, 1920), dtype=np.uint8)
    
    def enable_auto_exposure(self, enable=True, mode=1):
        return True
    
    def set_auto_exposure_target(self, target: int):
        return target
    
    def get_auto_exposure_state(self):
        return {"enabled": False, "mode": 1, "target": 128}
    
    def set_auto_exposure_range(self, **kwargs):
        return True
    
    @staticmethod
    def get_cam_list():
        return ["Camera1", "Camera2"]
```

### 方式三：独立驱动类（最小实现）

如果只需要基本的设备控制，可以直接实现：

```python
class SimpleSLM:
    """简化的 SLM 驱动"""
    
    def __init__(self, slm_number: int = 1):
        self.slm_number = slm_number
        self.is_open = False
    
    def open(self) -> None:
        """打开 SLM"""
        if self.is_open:
            return
        # 实现连接逻辑
        self.is_open = True
    
    def close(self) -> None:
        """关闭 SLM"""
        self.is_open = False
    
    def write_phase(self, phase: np.ndarray) -> None:
        """写入相位"""
        if not self.is_open:
            raise RuntimeError("SLM 未打开")
        # 实现写入逻辑
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
```

---

## 设备注册与管理

### 使用 DeviceRegistry

```python
from ao_shaping.drivers.device_registry import DeviceRegistry, get_global_registry

# 创建本地注册表
registry = DeviceRegistry()

# 注册设备
registry.register(
    camera,
    alias="main_camera",
    tags=["imaging", "primary"],
    priority=10,
    auto_connect=True
)

# 使用别名访问
camera = registry["main_camera"]

# 按类型查找
cameras = registry.find_by_type(DeviceType.CAMERA)

# 按标签查找
imaging_devices = registry.find_by_tag("imaging")

# 批量操作
registry.connect_all()
registry.disconnect_all()

# 数字孪生
states = registry.get_all_twin_states()
registry.sync_from_twin_states(states)

# 获取全局注册表
global_registry = get_global_registry()
```

### 导出/导入配置

```python
# 导出配置
config = registry.export_config()

# 导入配置（需要重建设备）
for dev_config in config["devices"]:
    device_type = dev_config["device_type"]
    # 根据类型创建设备实例
    ...
```

---

## VISA 通信层

使用 PyVISA 进行仪器控制：

```python
from ao_shaping.drivers.visa_base import (
    VisaResourceManager,
    VisaInstrument,
    VisaInstrumentFactory
)

# 列出可用资源
with VisaResourceManager() as rm:
    resources = rm.list_resources()
    print(resources)

# 直接打开仪器
with VisaInstrument('USB0::0x1234::0x5678::SN001::INSTR') as inst:
    idn = inst.query('*IDN?')
    inst.write('VOLT 12.0')

# 使用工厂批量管理
factory = VisaInstrumentFactory()
factory.register('power_supply', 'GPIB0::12::INSTR')
factory.register('multimeter', 'USB0::...')

with factory.open_all() as instruments:
    ps = instruments['power_supply']
    ...
```

---

## Mock 设备

> **注意**: 有关更高级的数值仿真设备，请参见 [模拟设备 (sim/)](#模拟设备-sim-) 章节。

用于测试和开发：

```python
from ao_shaping.drivers.mock_devices import (
    MockCamera,
    MockSLM,
    MockDM,
    MockWFS,
    MockLaser,
    MockStage,
    MockFilter
)

# 模拟相机
cam = MockCamera(resolution=(1024, 1024), noise_level=5.0)
with cam:
    img = cam.capture()
    print(f"图像形状: {img.shape}")

# 模拟 SLM
slm = MockSLM(resolution=(1920, 1080))
with slm:
    phase = np.random.rand(1080, 1920) * 2 * np.pi
    slm.write_phase(phase)

# 模拟 DM
dm = MockDM(n_actuators=64)
with dm:
    voltages = np.zeros(64)
    dm.apply_voltages(voltages)
    surface = dm.get_surface()

# 模拟 WFS
wfs = MockWFS(n_lenslets=32)
with wfs:
    wf = wfs.measure_wavefront()
    zernike = wfs.fit_zernike(wf, n_modes=15)
```

---

## 错误处理约定

1. **自定义异常**: 每个驱动应有对应的错误类，命名规则为 `{DriverName}Error`
2. **异常继承**: 继承自 `DeviceError` 或对应的基础异常类
3. **状态追踪**: 使用 `self._set_state()` 方法更新设备状态
4. **日志记录**: 使用 `loguru.logger` 记录操作

```python
class MyDeviceError(DeviceError):
    """设备特定错误"""
    pass

# 使用示例
try:
    device.open()
except MyDeviceError as e:
    logger.error(f"设备错误: {e}")
except ConnectionError:
    logger.error("连接失败")
```

---

## 最佳实践

1. **上下文管理器**: 实现 `__enter__` 和 `__exit__` 方法
2. **参数验证**: 使用 `register_parameter()` 注册参数并验证
3. **能力声明**: 使用 `register_capability()` 声明设备能力
4. **数字孪生**: 实现 `get_twin_state()` 和 `sync_from_twin()` 方法
5. **延迟导入**: SDK 导入放在方法内部或使用 try-except
6. **类型注解**: 为所有公开方法添加类型注解
7. **文档字符串**: 为每个类和方法添加完整的文档字符串

---

## PPT 演示文稿介绍

---

# AO-Shaping 硬件驱动框架介绍

## 一、项目概述

AO-Shaping 是一个基于 PyTorch 深度学习的自适应 Optics（AO）系统，用于控制空间光调制器（SLM）、变形镜（DM）、波前传感器（WFS）和 CCD 相机等硬件设备。

**核心特点：**
- 统一的设备驱动接口
- 数字孪生支持
- 灵活的设备注册与管理
- 支持 VISA 通信协议

---

## 二、驱动架构

### 2.1 核心组件

| 组件 | 文件 | 功能 |
|------|------|------|
| Device 基类 | `device_base.py` | 统一设备接口 |
| 设备注册表 | `device_registry.py` | 设备集中管理 |
| VISA 通信 | `visa_base.py` | 仪器控制 |
| Mock 设备 | `mock_devices.py` | 测试模拟 |
| 模拟设备 | `sim/` | 数字孪生仿真 |

### 2.2 设备类型

```
设备类型 (DeviceType)
├── CAMERA   - 相机/CCD
├── SLM      - 空间光调制器  
├── DM       - 变形镜
├── WFS      - 波前传感器
├── STAGE    - 运动台
├── LASER    - 激光器
├── FILTER   - 滤光轮
└── OTHER    - 其他设备 (含模拟设备)
```

**模拟设备继承层次:**
```
SimulatedDevice (Device)
├── OpticalDevice
│   ├── WavefrontProcessor
│   │   ├── SimulatedSLM
│   │   ├── SimulatedLens
│   │   ├── SimulatedAperture
│   │   ├── SimulatedTurbulentScreen
│   │   └── SimulatedThermalScreen
│   └── SimulatedLaser
├── SimulatedCCD (BaseCamera)
└── SimulatedATP
```

---

## 三、接口规范

### 3.1 Device 基类接口

所有设备驱动必须实现以下方法：

```python
class Device(ABC):
    @abstractmethod
    def open(self) -> None:        # 打开设备
    @abstractmethod
    def close(self) -> None:        # 关闭设备
    @abstractmethod
    def is_connected(self) -> bool: # 检查连接
    @abstractmethod
    def get_hardware_info(self) -> dict: # 获取硬件信息
```

### 3.2 设备特定接口

| 设备 | 抽象基类 | 特有方法 |
|------|---------|---------|
| 相机 | BaseCamera | capture(), reset_exposure_time() |
| DM | DM | transform(), send() |
| SLM | - | write_phase(), set_wavelength() |
| WFS | - | measure_wavefront(), fit_zernike() |

### 3.3 模拟设备接口

| 设备 | 抽象基类 | 特有方法 |
|------|---------|---------|
| 模拟设备 | SimulatedDevice | compute(), set_seed(), set_noise() |
| 光学模拟 | OpticalDevice | process(), set_input(), get_output() |
| 波前处理 | WavefrontProcessor | set_phase(), get_phase() |

---

## 四、现有驱动

### 4.1 支持的硬件

| 设备类型 | 驱动实现 | 接口方式 |
|---------|---------|---------|
| SLM | SantecSLM200 | SDK (ctypes) |
| SLM | SantecSLM200Visa | PyVISA |
| DM | NLight | SDK + UDP |
| 相机 | Daheng (大恒) | GigE SDK |
| 相机 | MiiCam | Miic SDK |
| 相机 | MIICAMDevice | USB3.0 SDK |
| WFS | Thorlabs | 专用协议 |
| TM | SerialPortFSM | RS232 串口 |

### 4.2 Mock 设备

用于无硬件环境的开发测试：

- MockCamera - 模拟相机
- MockSLM - 模拟 SLM
- MockDM - 模拟 DM
- MockWFS - 模拟 WFS
- MockLaser - 模拟激光器
- MockStage - 模拟运动台
- MockFilter - 模拟滤光轮

### 4.3 模拟设备 (sim/)

用于数字孪生和端到端数值仿真：

| 类别 | 设备 | 说明 |
|------|------|------|
| 光源 | SimulatedLaser | 模拟激光器 |
| 调制器 | SimulatedSLM | 模拟空间光调制器 |
| 元件 | SimulatedLens, SimulatedAperture | 模拟透镜/光阑 |
| 探测 | SimulatedCCD | 模拟 CCD 相机 |
| 大气 | SimulatedTurbulentScreen | 湍流相位屏 |
| 大气 | SimulatedThermalScreen | 热晕相位屏 |
| 传输 | SimulatedATP | 大气传输仿真 |

---

## 五、使用示例

### 5.1 基本使用

```python
from ao_shaping.drivers import SantecSLM200, NLightDM

# SLM 控制
with SantecSLM200(slm_number=1, wavelength=1064) as slm:
    phase = np.zeros((1080, 1920), dtype=np.uint16)
    slm.write_phase(phase, memory_number=1)
    slm.display_memory(1)

# DM 控制
with NLightDM() as dm:
    voltages = np.zeros(64)
    dm.send_voltages(voltages)
```

### 5.2 设备注册

```python
from ao_shaping.drivers import DeviceRegistry, DeviceType

registry = DeviceRegistry()
registry.register(camera, alias="main_cam", tags=["imaging"])
registry.register(slm, alias="slm1", tags=["modulation"])

# 查找设备
cameras = registry.find_by_type(DeviceType.CAMERA)
```

---

## 六、数字孪生

### 6.1 状态同步

```python
# 获取所有设备状态
states = registry.get_all_twin_states()

# 从数字孪生同步状态
registry.sync_from_twin_states(states)
```

### 6.2 参数管理

```python
# 注册参数
device.register_parameter(
    "exposure_time_ms",
    default_value=20.0,
    min_value=0.1,
    max_value=10000.0,
    unit="ms"
)

# 设置/获取参数
device.set_parameter_value("exposure_time_ms", 50.0)
value = device.get_parameter_value("exposure_time_ms")
```

### 6.3 模拟设备 (SimulatedDevice)

`sim/` 模块中的设备继承自 `SimulatedDevice`，内置数字孪生支持：

```python
from ao_shaping.drivers.sim import SimulatedLaser

laser = SimulatedLaser(power=100, wavelength=1064, random_seed=42)
with laser:
    # 设置随机种子以保证可重复性
    laser.set_seed(123)
    
    # 启用/禁用噪声
    laser.set_noise(True)
    
    # 获取数字孪生状态
    state = laser.get_twin_state()
    print(state)
```

---

## 七、扩展开发

### 7.1 实现新驱动

```python
from ao_shaping.drivers.device_base import Device, DeviceType

class MyDevice(Device):
    device_type = DeviceType.CAMERA
    manufacturer = "MyCompany"
    model = "MC-100"
    
    def open(self): ...
    def close(self): ...
    def is_connected(self): ...
    def get_hardware_info(self): ...
```

### 7.2 实现模拟设备

```python
from ao_shaping.drivers.sim.base import OpticalDevice, WavefrontProcessor
import numpy as np

class MySLM(WavefrontProcessor):
    """自定义模拟 SLM"""
    
    device_type = DeviceType.SLM
    manufacturer = "Simulation"
    model = "My Simulated SLM"
    
    def __init__(self, resolution=(1920, 1080), wavelength=1064, ...):
        super().__init__(wavelength=wavelength, npix=resolution[1])
        self._resolution = resolution
    
    def compute(self, *args, **kwargs):
        """实现仿真计算"""
        return self.process(args[0])
    
    def process(self, wave):
        """处理波前"""
        # 实现光学处理逻辑
        return wave
```

### 7.3 VISA 集成

```python
from ao_shaping.drivers.visa_base import VisaInstrument

with VisaInstrument('GPIB0::1::INSTR') as inst:
    inst.write('COMMAND')
    response = inst.query('QUERY?')
```

---

## 八、总结

- **统一接口**: 所有硬件通过 Device 基类统一管理
- **灵活扩展**: 支持多种驱动实现方式
- **数字孪生**: 内置状态同步支持，SimulatedDevice 集成 sim.digitaltwin
- **开箱即用**: 提供多种主流设备驱动
- **测试友好**: Mock 设备支持无硬件开发
- **仿真能力**: sim/ 模块提供端到端光学系统数值仿真

---

# 谢谢！

---

## 相关文件

- [device_base.py](src/ao_shaping/drivers/device_base.py) - Device 基类
- [device_registry.py](src/ao_shaping/drivers/device_registry.py) - 设备注册表
- [visa_base.py](src/ao_shaping/drivers/visa_base.py) - VISA 通信层
- [mock_devices.py](src/ao_shaping/drivers/mock_devices.py) - Mock 设备
- [sim/](src/ao_shaping/drivers/sim/) - 模拟设备 (数字孪生)
- [ccd/base.py](src/ao_shaping/drivers/ccd/base.py) - 相机抽象基类
- [dm/base.py](src/ao_shaping/drivers/dm/base.py) - DM 抽象基类
- [sim/base.py](src/ao_shaping/drivers/sim/base.py) - 模拟设备基类
