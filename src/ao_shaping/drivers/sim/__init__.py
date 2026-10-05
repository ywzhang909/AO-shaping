"""模拟设备子包。

本子包提供与设备驱动框架集成的仿真硬件设备。这些仿真设备把 sim.digitaltwin 的
数值仿真代码包装起来, 同时对外呈现与 Device 兼容的接口。

结构:
    sim/
    ├── base.py          # 模拟设备的基类
    ├── wave.py          # 波前生成、传播与度量工具
    ├── ccd/             # 模拟相机
    ├── laser/           # 模拟激光器
    ├── optics/          # 模拟光学元件 (SLM、透镜、光阑)
    └── atmos/           # 模拟大气效应

物理工具 (包装自 sim.digitaltwin):
    create_wave()        # 创建 Wave 对象
    apply_aperture()     # 施加圆形光阑
    apply_focus()        # 施加薄透镜聚焦相位
    propagate()          # 角谱传播
    power_bucket()       # 计算 PIB 指标
    radius_metric()      # 计算能量包含半径

Example:
    >>> from ao_shaping.drivers.sim import (
    ...     SimulatedTurbulentScreen,
    ...     create_wave, apply_aperture, apply_focus,
    ...     propagate, power_bucket, radius_metric,
    ... )
    >>>
    >>> wave = create_wave(256, 0.1e-3, 1550e-9)
    >>> apply_aperture(wave, 0.05)
    >>> apply_focus(wave, 0.5)
    >>> turb = SimulatedTurbulentScreen(Cn2=1e-9)
    >>> turb.process(wave)
    >>> propagate(wave, 0.5)
    >>> pib = power_bucket(wave.intensity, wave.x, wave.y, 'origin', 5e-3)
"""

from ao_shaping.drivers.sim.base import (
    OpticalDevice,
    SimulatedDevice,
    SimulatedDeviceError,
    WavefrontProcessor,
)

from ao_shaping.drivers.sim.ccd import SimulatedCCD

from ao_shaping.drivers.sim.laser import SimulatedLaser

from ao_shaping.drivers.sim.optics import (
    SimulatedAperture,
    SimulatedLens,
    SimulatedSLM,
)

from ao_shaping.drivers.sim.atmos import (
    SimulatedATP,
    SimulatedThermalScreen,
    SimulatedTurbulentScreen,
)

from ao_shaping.drivers.sim.dm import SimMicroDM, SimulateDM

from ao_shaping.drivers.sim.wave import (
    WaveGenerator,
    WavePropagator,
    LensApplier,
    ApertureApplier,
    WaveMetric,
    WaveDeviceError,
    create_wave,
    apply_aperture,
    apply_focus,
    propagate,
    power_bucket,
    radius_metric,
)

# 尝试从 sim.digitaltwin 导入 Environment, 但对缺失的依赖做优雅处理
try:
    from sim.digitaltwin.base import Environment
    _has_environment = True
except ImportError:
    Environment = None
    _has_environment = False

# 动态构造 __all__ 列表
__all__ = [
    # 基类
    "SimulatedDevice",
    "SimulatedDeviceError",
    "OpticalDevice",
    "WavefrontProcessor",
    # 波前工具
    "WaveGenerator",
    "WavePropagator",
    "LensApplier",
    "ApertureApplier",
    "WaveMetric",
    "WaveDeviceError",
    "create_wave",
    "apply_aperture",
    "apply_focus",
    "propagate",
    "power_bucket",
    "radius_metric",
    # DM
    "SimMicroDM",
    "SimulateDM",
    # CCD
    "SimulatedCCD",
    # 激光器
    "SimulatedLaser",
    # 光学元件
    "SimulatedSLM",
    "SimulatedLens",
    "SimulatedAperture",
    # 大气
    "SimulatedTurbulentScreen",
    "SimulatedThermalScreen",
    "SimulatedATP",
]

if _has_environment:
    __all__.append("Environment")

__version__ = "1.0.0"

# 在此处绑定仿真相机, 使 ``--cam_type sim`` 对本子包的**每一个**消费者都可用,
# 而不只是那些恰好自己调用了 ``register_sim_camera()`` 的 runner
# (原始 bug 见 slm_pib_sim)。
from ao_shaping.drivers.sim.slm_pib_sim import register_sim_camera as _register_sim

_register_sim()
del _register_sim
