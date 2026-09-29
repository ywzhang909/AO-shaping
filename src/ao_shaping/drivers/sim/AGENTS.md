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
├── beam_backend.py      # 光束后端 (numpy/legacy ↔ OOPAO 路由层)
├── beam_simulation.py   # 光束仿真
├── compat.py            # 兼容层
├── oopao_backend.py     # OOPAO 后端实现 (湍流相位屏 + ASM 传播)
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

## 光束后端路由 (`AO_OOPAO_BACKEND`)

`beam_backend.py` 是 numpy/legacy 与 OOPAO 双后端的**唯一路由层**。OOPAO 实现在
`oopao_backend.py` (OOPAO `Atmosphere` 相位屏 + `Atmosphere.ASM` 传播核)。

| 环境变量 `AO_OOPAO_BACKEND` | 后端 |
|---|---|
| 未设置 / 非 `1`/`true`/`yes`/`on` (**默认**) | numpy/legacy (FFT 谱 Kolmogorov + `beam_simulation.propagation`) |
| `1` / `true` / `yes` / `on` | OOPAO |

**路由范围 (红线 — 只有 2 个函数被路由)**:

| 函数 | OOPAO 分支 | numpy 分支 |
|------|------------|-----------|
| `turbulence_phase()` | `oopao_backend.make_screens()` (单 slab, seed 取自 `rng`) | FFT 谱 Kolmogorov (legacy, 行为不变) |
| `propagate()` | `oopao_backend.propagate_asm()` (ASM 核) | `bs.propagation()` |
| `focal_plane()` | **无** — 始终 `bs.lens_fft_propagation_to_focal()` | 同左 |
| `apply_lens()` | **无** — 始终 `bs.apply_lens()` | 同左 |

因此焦面/Strehl/FWHM/EE 的两后端差异**只来自相位屏**, 不来自传播或取焦; 下游
`scripts/slm_pib_sim.py`、`fouriergsnet_env.py` 等不经 `beam_backend` 的调用点不受影响。

**已知约束**:

1. **后端缓存**: `oopao_backend._get_backend` 是 `@lru_cache(maxsize=8)`。同一进程内切换
   仿真配置 (网格/波长/pixel_size) 后必须调用 `oopao_backend._get_backend.cache_clear()`,
   否则命中旧后端实例得到与配置不符的相位屏。
2. **静默回退**: `_oopao_enabled()` 在 OOPAO 不可导入或不可用时返回 `False` 并**静默退回
   numpy**。开启后务必确认 `oopao_backend._oopao_available()` 为 `True`, 否则会误以为在跑 OOPAO。
3. **两后端不等价 (勿假设同 Cn2 可互换)**: 报告配置下 OOPAO/legacy `phase_std_rad` 比值
   为 `8.72×–8.81×` (中位数 `8.77×`)。这是两条相位屏实现 (含 r0 重标定与内层尺度) 的差异,
   **不是**普适常数 —— 换配置该比值会变 (另一组配置实测 `9.72×`)。跨后端比较绝对 Strehl /
   FWHM 无意义, 报告须显式声明该限制。
4. **安装**: OOPAO 以 editable 方式从 git submodule `libs/OOPAO` 安装 (`uv pip install
   --no-deps -e libs/OOPAO`)。`libs/OOPAO` 为 submodule (canonical
   `https://github.com/cheritier/OOPAO.git`); 换了 OOPAO 版本后重装即可。
5. **API 漂移**: 本地 submodule 比旧 pin `8e12a17f` 领先若干提交, 导入会打印
   `Telescope is no longer the "master" class ...` 警告 — 属预期, 当前用法未触及该 API。

对比报告与图片见 [`docs/oopao_vs_numpy/report.md`](../../../../docs/oopao_vs_numpy/report.md)
(生成器 `scripts/generate_oopao_vs_numpy_report.py`); 后端回归测试
`tests/ao_shaping/drivers/sim/test_oopao_backend.py`。

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
