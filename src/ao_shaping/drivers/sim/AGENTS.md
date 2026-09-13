# sim/ - 模拟设备 (数字孪生)

模拟设备模块, 提供数字孪生仿真引擎, 用于端到端光学系统仿真和测试。

## 结构

```
sim/
├── base.py              # SimulatedDevice, OpticalDevice, WavefrontProcessor 基类
├── ccd/                 # 模拟 CCD
│   ├── simulated_ccd.py
│   └── __init__.py
├── laser/               # 模拟激光器
│   ├── simulated_laser.py
│   └── __init__.py
├── optics/              # 模拟光学元件
│   ├── simulated_slm.py     # SimulatedSLM, SimulatedLens, SimulatedAperture
│   └── __init__.py
├── atmos/               # 大气模拟
│   ├── screens.py           # SimulatedTurbulentScreen, SimulatedThermalScreen, SimulatedATP
│   └── __init__.py
├── beam_backend.py      # 光束后端
├── beam_simulation.py   # 光束仿真
├── compat.py            # 兼容层
├── wave.py              # 波前处理
└── __init__.py
```

## 关键类

| 类 | 文件 | 说明 |
|----|------|------|
| `SimulatedDevice` | `base.py` | 模拟设备基类 (继承 `Device`) |
| `OpticalDevice` | `base.py` | 光学设备基类 (继承 `SimulatedDevice`) |
| `WavefrontProcessor` | `base.py` | 波前处理器基类 (继承 `OpticalDevice`) |
| `SimulatedDeviceError` | `base.py` | 模拟设备异常 |
| `SimulatedCCD` | `ccd/simulated_ccd.py` | 模拟 CCD 相机 (继承 `BaseCamera`) |
| `SimulatedLaser` | `laser/simulated_laser.py` | 模拟激光器 (继承 `OpticalDevice`) |
| `SimulatedSLM` | `optics/simulated_slm.py` | 模拟 SLM (继承 `WavefrontProcessor`) |
| `SimulatedLens` | `optics/simulated_slm.py` | 模拟透镜 (继承 `SimulatedDevice`) |
| `SimulatedAperture` | `optics/simulated_slm.py` | 模拟光阑 (继承 `SimulatedDevice`) |
| `SimulatedTurbulentScreen` | `atmos/screens.py` | 湍流相位屏 (继承 `WavefrontProcessor`) |
| `SimulatedThermalScreen` | `atmos/screens.py` | 热晕相位屏 (继承 `WavefrontProcessor`) |
| `SimulatedATP` | `atmos/screens.py` | 大气传输仿真 (继承 `SimulatedDevice`) |
| `SimulatedLaserError` | `laser/simulated_laser.py` | 激光器异常 |
| `SimulatedSLMError` | `optics/simulated_slm.py` | SLM 异常 |
| `SimulatedCCDError` | `ccd/simulated_ccd.py` | CCD 异常 |

## 继承层次

```
SimulatedDevice (Device)
├── OpticalDevice
│   └── SimulatedLaser
├── WavefrontProcessor
│   ├── SimulatedSLM
│   ├── SimulatedTurbulentScreen
│   └── SimulatedThermalScreen
├── SimulatedCCD (BaseCamera)
├── SimulatedATP
├── SimulatedLens
└── SimulatedAperture
```

> 注意: `SimulatedLens` 和 `SimulatedAperture` 直接继承 `SimulatedDevice` (非 WavefrontProcessor), 但实现了 `process()` / `compute()` 方法。`SimulatedTurbulentScreen` 和 `SimulatedThermalScreen` 继承自 `WavefrontProcessor` (非 OpticalDevice)。`SimulatedMicroDM` (dm/ 子包) 继承自 `DM`。

## SimulatedDevice 接口

| 方法 | 说明 |
|------|------|
| `compute(*args, **kwargs)` | 执行仿真计算 (抽象) |
| `reset()` | 重置仿真状态 |
| `set_seed(seed)` | 设置随机种子 |
| `set_noise(enabled)` | 启用/禁用噪声 |
| `get_twin_state()` | 获取数字孪生状态 |
| `sync_from_twin(state)` | 从数字孪生同步状态 |

## OpticalDevice 接口

| 方法 | 说明 |
|------|------|
| `set_input(wave)` | 设置输入波前 |
| `get_output()` | 获取输出波前 |
| `process(wave)` | 处理波前 (抽象) |

## WavefrontProcessor 接口

| 方法 | 说明 |
|------|------|
| `set_phase(phase)` | 设置相位图 |
| `get_phase()` | 获取当前相位 |

## 与 mock_devices 的区别

| 特性 | mock_devices | sim/ |
|------|-------------|------|
| 用途 | 简单测试/开发 | 数值仿真/数字孪生 |
| 实现 | 简化模拟 | 集成 sim.digitaltwin |
| 复杂度 | 基础 | 高级物理模型 |
| 继承 | BaseCamera, Device | SimulatedDevice |

## 使用示例

```python
from ao_shaping.drivers.sim import (
    SimulatedLaser, SimulatedSLM, SimulatedCCD,
)
from ao_shaping.drivers.sim.atmos import SimulatedTurbulentScreen

laser = SimulatedLaser(power=100, wavelength=1064)
slm = SimulatedSLM(resolution=(1920, 1080))
turb = SimulatedTurbulentScreen(Cn2=1e-14)
ccd = SimulatedCCD(resolution=(512, 512))

with laser, slm, turb, ccd:
    wave = laser.generate()
    slm.set_phase(np.zeros((1080, 1920)))
    wave = slm.process(wave)
    wave = turb.process(wave)
    img = ccd.get_numpy_image()
```

详细接口文档见 [`INTERFACE_DOCS.md`](../INTERFACE_DOCS.md) §模拟设备。
