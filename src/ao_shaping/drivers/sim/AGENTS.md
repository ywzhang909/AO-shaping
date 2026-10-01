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

## 波前干扰: `disturbance.py` (湍流 + 热晕)

| 类 | 文件 | 说明 |
|----|------|------|
| `DisturbanceConfig` | `disturbance.py` | frozen dataclass: `mode` (`none`/`static`/`dynamic`), `cn2`, `distance_m`, `l_max`, `l_min`, `pixel_pitch_m`, `thermal_halo_pv_waves`, `thermal_halo_radius_px`, `halo_noll`, `seed` |
| `SimDisturbance` | `disturbance.py` | 生成 `(h, w)` float64 **raw 弧度** 干扰相位; `.phase()` / `.stats()` (实测 σ) / `.trace()` / `.archive()` / `.to_dict()` |

`SimPibSystem.__init__(..., *, disturbance=None)` 在 `far_field()` 的 **cache-miss 分支**把干扰加到 SLM 命令相位上
(即在 FFT 之前、瞳孔面内 —— 2f 台架的正确位置)。`self._phase` 保持为纯 SLM 命令, 因此干扰不会被重复计入。
缓存语义保证 **每次真实光学评估只推进一次**干扰; `static` 全程复用一张冻结屏, `dynamic` 每次评估重抽一张独立屏
(即完全去相关 / white-in-time 极限, **不是**风场平流模型 —— 真实大气去相关时间 ~10–50 ms 远短于本环路的每评估 ~0.375 s)。

> ⚠️ **`cn2` 与 `distance_m` 是退化旋钮**: 生成器的 `r0 = (0.423·k²·Cn2·L)^(-3/5)` 只依赖二者**乘积**;
> 单层薄屏没有任何传播物理。且 numpy 后端缺少次谐波/低频补偿 (见下方第 4 条), 实测 σ **低于**同 r0 的解析
> von Karman 方差。因此报告一律引用 **实测** σ, 不得用解析公式反推。

> ⚠️ **相位屏只支持方形网格**。`turbulence_phase` 输出 `n×n`; 对 1200×1920 面板的做法是
> `n_grid = 1920`、`aperture_size = 1920 × pixel_pitch` 生成后再**居中裁剪**到 1200 行 —— 各向同性像素、
> 无拉伸、无拼接缝。实测粗网格会系统性丢失约 **41%** 相位幅度 (不得为省时而降网格)。

> ⚠️ **`SimulatedThermalScreen` (热晕) 是一个静默 no-op —— 不可使用**。它的 `process()` 依赖外部包
> `sim.digitaltwin` (`screens.py` L284), 该包**未安装**; `except ImportError` 分支直接 `return wave`
> (L305-307), 既不报错也不产生任何相位。需要热晕相位请用 `disturbance.py` 的
> `SimDisturbance`(负热透镜: Noll 4 离焦 + Noll 11 球差, 经 smoothstep 光晕窗延展到光束半径之外)。

> ⚠️ **扰动会随 `reset_system()` 一起被丢弃**。`slm_pib_runner._maybe_sim_patch` 在 `--cam_type sim`
> 时会调用 `reset_system(seed=42)`, 这会**替换**进程级 system 并清掉已注入的干扰 —— 运行会静默地以
> **无干扰**方式执行, 而 companion 清单却声称有干扰。`scripts/slm_pib_sim_run.py` 因此包装了
> `slm_pib_sim.reset_system`, 使每次调用都重新挂上干扰 (由
> `tests/ao_shaping/scripts/test_slm_pib_sim_run_disturbance.py` 锁定)。

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

因此焦面/Strehl/FWHM/EE 的两后端差异**只来自相位屏**, 不来自传播或取焦。

**端到端实际路由到 OOPAO 的调用点** (实测确认, 2026-09):

| 调用点 | 是否受 `AO_OOPAO_BACKEND` 影响 |
|--------|-------------------------------|
| `drivers/sim/atmos/screens.py` `SimulatedTurbulentScreen._opd()` → `turbulence_phase()` (L192) | **是** |
| `optimizer/rl/envs.py` `SimTurbulenceAOEnv` → `turbulence_phase()` (L882) | **是** |
| `optimizer/wfless/strehl_sim_eval.py` (经 `TraditionalAOSystem` 间接) | **是** |
| `drivers/sim/slm_shaping_bench.py` (PIB bench, `spgd_shape`/`gs_shape`) | **否 — 见下** |

⚠️ **`slm_shaping_bench` 的 `cn2` 是死配置 (实测)**: 该模块 import 了 `turbulence_phase`
但**从不调用**, 其 `forward_intensity()` 只走 `focal_plane()` —— 而 `focal_plane` 永不路由。
实测 `forward_intensity` 在 `cn2=0` 与 `cn2=5e-14` 下输出**字节一致** (max|ΔI| = 0)。
因此**不能**用该 bench 做后端影响测试 (会得到两臂完全相同的假结果); 也不要在该 bench 上
调 `cn2` 后期待看到湍流效果。需真实湍流路由请用 `SimulatedTurbulentScreen` 或
`SimTurbulenceAOEnv`。

**已知约束**:

1. **后端缓存**: `oopao_backend._get_backend` 是 `@lru_cache(maxsize=8)`。同一进程内切换
   仿真配置 (网格/波长/pixel_size) 后必须调用 `oopao_backend._get_backend.cache_clear()`,
   否则命中旧后端实例得到与配置不符的相位屏。
2. **静默回退**: `_oopao_enabled()` 在 OOPAO 不可导入或不可用时返回 `False` 并**静默退回
   numpy**。开启后务必确认 `oopao_backend._oopao_available()` 为 `True`, 否则会误以为在跑 OOPAO。
3. **两后端不等价 (勿假设同 Cn2 可互换)**: 端到端实测 `disturbance_rms` 的
   oopao/numpy 比值在**所有**湍流档位上**恒定** (open 模式 5.428× / closed 模式 13.354×,
   跨 1e-16→5e-14 两个数量级相对离散度 ≤1.2e-09), 指向两个相位屏实现之间的**乘性标定
   偏置**而非统计涨落。绝对 Strehl / FWHM / PIB **不可跨臂直接比较**。要判定哪一臂更接近
   物理真值, 须先把两臂相位屏**标定到同一 r0 / 同一 phase_std** 再重跑。
   ⚠️ 比较时注意 `init_rms` (`compat.py::_phase_rms()` = **截瞳后总波前** RMS, 含像差与 DM)
   与 `disturbance_rms` (`env._disturbance_rms` = **全网格未截瞳**原始相位屏 RMS) 是**两个不同的量**,
   不可互相推断 —— 前者可因像差项跨臂反向。详见 `docs/oopao_impact/report.md` §4.3。
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
