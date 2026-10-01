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
├── dm/                  # 模拟 DM
│   ├── simulated_dm.py      # SimulateDM, SimMicroDM
│   └── __init__.py
├── wfs/                 # 模拟波前传感器
│   ├── simulated_wfs.py     # SimulatedWFS (OOPAO Shack-Hartmann, 瞳面 slope 模型)
│   └── __init__.py
├── beam_backend.py      # 光束后端 (numpy/legacy ↔ OOPAO 路由层)
├── beam_simulation.py   # 光束仿真
├── compat.py            # 兼容层
├── disturbance.py       # 波前干扰: 湍流 + 热晕 (static/dynamic), 供 slm_pib_sim 使用
├── dm_optics.py         # DM 电压 → 瞳孔相位耦合 (可分离高斯影响力函数)
├── oopao_backend.py     # OOPAO 后端实现 (湍流相位屏 + ASM 传播)
├── slm_pib_sim.py       # SLM-PIB 2f-Fourier 数字孪生 (SimPibSystem/SimSLMPib/SimPibCCD)
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
| `SimulateDM` | `dm/simulated_dm.py` | 模拟 DM (注册为 `sim`); `send_voltages` 会把电压推给 `SimDmOptics` |
| `SimDmOptics` | `dm_optics.py` | DM 电压 → 瞳孔相位 (可分离高斯影响力函数, 固定线性算子) |
| `SimulatedWFS` | `wfs/simulated_wfs.py` | 模拟 SH 波前传感器 (继承 `BaseWFS`; 瞳面 slope 模型, 非焦平面) |

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

✅ **DM→相位耦合已接入 (2026-10-01) —— `pib` / `combined` 仿真结果已有物理意义**:

`SimDmOptics` (`sim/dm_optics.py`) 把 DM 电压映射为瞳孔相位, 经
`SimPibSystem.dm_optics` 在 `far_field()` 的 **cache-miss 分支**叠加 (与
`disturbance` 同一约定: 相位贡献者在**求值时**相加, 绝不烘进 `self._phase`)。

| runner | 驱动量 | 仿真下是否有物理意义 |
|--------|--------|---------------------|
| `slm-pib` / `slm-gsnet` / `spgd-square` | SLM 相位 | **有** —— `set_phase_rad` 直接改变远场 |
| `pib` / `combined` | DM 电压 | **有** —— `SimulateDM.send_voltages` → `SimDmOptics` → `far_field()` |

实现要点:
- **可分离高斯影响力函数**: 每个致动器一个高斯凸包, x/y 可分离, 故用一次矩阵乘
  在 DM 包围盒上求值。稠密 influence 矩阵在真实 1920×1200 面板下需 ~1.2 GB, 不可接受。
- **单位链**: `opd_um = stroke_um * v / v_max` → `phase_rad = opd_um * 2π / (λ_nm * 1e-3)`。
  0 V 为平场 (与真实驱动一致)。`V_Min=-300` 与 `V_Max=499` **不对称**, 故负向半程
  行程短于正向 —— 裁剪到负轨**不是**正轨的镜像。
- **相位是 raw 未包裹弧度**, 不做 `mod 2π`: 全项目唯一 wrap 点在 SLM 驱动。
- **cache 失效靠 `dm_optics.version`**: DM 是独立光学状态, 电压变化必须让
  `_far_field` 失效, 否则 camera 继续返回 DM 驱动前的旧图, 看起来"DM 毫无作用"。
- `SimulateDM` 用**函数内延迟 import** 推相位: `drivers/sim/__init__.py` 先 import
  `sim.dm` 再 import `sim.slm_pib_sim`, 顶层 import 会撞上半初始化的包。

⚠️ **相机噪声的地板项会把功率比指标淹没 (2026-10-01 实测并修复)**:
`far_field_noisy()` 曾以 `clip(img, 0, None)` 收尾。`far_field()` 是 `|FFT|²`,
本身非负, 故帧内所有负值**只可能来自读出噪声** —— 在 0 处裁剪会把对称分布**整流**,
凭空造出与**像素数**成正比的 DC 地板:

| 噪声项 | 地板 / 信号 |
|--------|------------|
| shot noise (∝ 信号, 暗区自然消失) | **1.005×** —— 正确 |
| read noise 经 `clip(·, 0, None)` 后 | **4598×** —— 缺陷 |

光斑只占 ~150 px 而画幅有 2.3 M px, 地板因此携带 ~3000× 的光。`pib` 的分母是
整帧总功率, 于是它量的是这块地板 —— 这才是 `pib` / `combined` 停在 0.0145、
`J(+δ)−J(−δ)` 全是噪声的**真正原因** (不是 DM 未接入, 也不是 SPGD 步长不当)。

修复: **不再逐帧裁剪**。真实探测器的本底位于其阈值**之下**, 只在量化时裁剪一次,
raw 帧本就允许略微低于黑电平; 去黑电平靠暗帧扣除, 不是每帧 clip。

修复后实测 `main.py pib --cam_type sim --dm_type sim -e 200`, 目标函数真实上升:

| epoch | 0 | ~100 | ~200 |
|-------|---|------|------|
| `pib` | 0.66 | 2.33 | **2.98** |

(修复前 200 个 epoch 恒为 0.0145。`pib` 单位为 %, 故可 >1。)

⚠️ **诊断陷阱: 不要用整帧 total 判定"噪声是否过大"**。零均值噪声在 2.3 M px 上的
求和标准差约 `σ√N ≈ 758`, 与 152 的信号总量同量级, 因此 total 天然会剧烈波动 ——
即使噪声完全居中。判据应当用**暗区均值** (≈0) 与**桶内信号占比**, 参见
`tests/ao_shaping/drivers/sim/test_sim_noise_model.py`。

## 模拟 WFS: `SimulatedWFS` (OOPAO Shack-Hartmann)

`sim/wfs/simulated_wfs.py`, 实现 `BaseWFS`, 可直接顶替 `ThorlabWFS`。

**必须用瞳面梯度模型, 不能用焦平面传播**。SH 通过微透镜阵列成像瞳孔, 探测器上
光斑位移编码各子孔径的局部 tip/tilt, 因此测量量是**瞳面相位梯度**:
`wfs_measure(src, sh, phase_in=...)` 正是该量。⚠️ 早前计划复用本仓
`wave.py` 的焦平面传播 —— 那产出的是**相机图像**而非 slope, 二者不可互换,
而优化器消费的是 slope。

**slope 数组是行块状布局**: 第 `0:n_subap` 行为 x-slope, 第 `n_subap:` 行为
y-slope (用纯倾斜探针实测确认: 纯 x 倾斜 → 上半 rms 1.29e-4, 下半 0)。

🔴 **SH 测不到 piston**: 子孔径的绝对相位偏移不移动光斑。`list_zernike_modes`
从 Noll 1 = piston 起, 把它放进拟合基会留下一个近零空间列, `pinv` 把它放大成
巨大的伪系数 (实测纯离焦瞳孔下 **+0.50 rad**)。故 piston 必须排除出拟合基
(`_FIT_FIRST_MODE`) —— 这不是 off-by-one, 改回去会静默退化。

**单位** (历史上出过两次真实 bug: 系数放大 1.88×, 相位缩小 2π = 6.28×):
- `get_wavefront()` → **waves** = `phase_rad / 2π`
- `get_zernike()` → **µm** = `phase_rad * λ_nm*1e-3 / 2π`

`um_to_waves()` 是写死 532 nm 的, 故默认波长必须 532 nm, 否则
`um_to_waves()` 再 `×2π` 无法还原弧度。

⚠️ **保真度不足以当波前基准 (实测, 勿高估)**。注入已知模式再读回系数:

| 模式 | 单独注入 | 与其他模式同时注入 |
|------|---------|-------------------|
| noll 2 tilt | 24% | **83%** |
| noll 4 defocus | 1.4% | 0.9% |
| noll 7 coma | — | 17% |
| noll 11 spherical | 32% | 25% |

tilt 单独注入只差 24%, 与其他模式同注入却差到 83% ⇒ 主误差是**模式间串扰**
(cross-talk), 不是逐模式噪声。已排除的解释: **不是**质心量化 (把探测器采样从
npp 8→32、n_subap 6→24 反而更差, 且条件数始终良性 2~6, 排除秩亏); **不是**
标定pass 的加性污染 (平场读数严格为 0)。误差源自 OOPAO 内部 slope 估计器,
在此如实表征而非用宽松容差掩盖。

**结论**: 该类适合跑控制环、验证符号约定、验证 µm/waves 链路; 它**不是**
合格的波前参考 —— 任何由 `get_zernike()` 推出的 RMS 改善率或 Strehl 都必须
标注为**未验证**。要可信数值请像 `zernike-matrix` 那样对硬件标定。

✅ **已可经 CLI 到达 (2026-10-01)**: WFS 现在有注册表 `drivers/wfs/_registry.py`,
镜像 DM 的 `_registry.py` (`register_wfs`/`create_wfs`/`list_wfs_types`/
`resolve_wfs`, 同样的延迟绑定, 因此硬件包不会 import 仿真包)。

`--wfs_type [thorlab|sim]` 定义在共享的 `WfsParams` 上, 8 个已注册命令**既暴露
也真正透传**该 flag: `wf`、`pipeline`、`rms-zernike`、`ga-zernike`、
`greedy-zernike`、`dm-matrix`、`hadamard-matrix`、`zernike-matrix`。
`closed-loop` **刻意没有** —— 它回放已保存的响应矩阵 (`ClosedLoopParams`)。

共改造 **10 处**真实构造点 (此前统计的 13 处里有几处是 docstring 示例, 不是代码):
`optimizer/wf/{rms,rms_by_zernike,ga_zernike,greedy_zernike}.py`、
`runners/slm/rms_zernike_runner.py` (含 `_auto_delta_detect_rms` 辅助函数)、
`runners/{dm_matrix,hadamard_matrix}_runner.py`、
`runners/slm/zernike_matrix_runner.py`。其余优化器/runner 内部已不再直接
`ThorlabWFS(...)`。

**DM→WFS 瞳孔耦合**: `SimulateDM.send_voltages` 现在**同时**推给两处 ——
`SimPibSystem.dm_optics` (远场, 供 `pib`/`combined`) 与当前活跃的
`SimulatedWFS.dm_optics` (瞳孔, 供 `wf`/`rms-zernike` 等)。传感器按自己的
瞳孔网格 (48×48) 持有一份 `SimDmOptics`, 因此**无需**把 1200×1920 重采样。
⚠️ 以前只发布到远场, 于是 `wf` 永远读到平 pupil、无论 DM 怎么动 RMS 都是 0。

⚠️ **`wf --wfs_type sim` 在无像差时最优就是不动 (不是 bug)**: 仿真未注入任何
扰动/像差时, DM 只能**增加**相位, 因此平场 (`wighted_rms ≈ 2.8e-05`) 才是
RMS 最小解, SPGD 正确地保留初始平场命令。要看到真实的校正过程, 必须先注入
像差 —— 已加 `--disturbance-cn2` (经 `SimDisturbance` 注入**瞳孔**, 默认 `0`
= 不注入)。

⚠️ **注入了像差后 SPGD 能收敛, 但比硬件慢约一个数量级 —— 要加 epoch 或提 lr**
(实测 2026-10-01, `wf --wfs_type sim --dm_type sim --disturbance-cn2 2e-13`)。
`schedule_lr_delta` 是按**真实硬件**的电压-相位标度标定的; 对仿真 DM,
`delta=3 V` 时单致动器峰值相位只有 **0.017 waves ≈ 目标量的 2.6%**, 梯度信号偏弱。
**这不是不收敛, 只是慢** —— 逐轮目标量单调下降:

| lr | epochs | 起始 | 最优 | 降幅 |
|----|--------|------|------|------|
| 2 (自动) | 200 | 0.6540 | 0.6439 | 1.5% |
| 2 | 400 | 0.6540 | 0.6172 | 5.6% |
| 2 | 800 | 0.6540 | 0.5360 | 18.0% |
| **8** | **400** | 0.6540 | 0.5301 | **19.0%** |

推荐用法: `--lr 8 -e 400` (或保持默认 lr 把 epoch 提到 800)。
⚠️ `--lr 40` 会**发散** (1.27 → 10.5, 过冲), 故 lr 上限在 8~40 之间, 不要照搬硬件的
大 lr。`--delta` 一般无需改: 默认的 3 V 虽弱但稳定。

> **此前的一次错误结论 (勿重蹈)**: 早期只跑 60~80 epoch 就断言"SPGD 在仿真下不收敛",
> 并据此推测 `lr` 与 `δ` 是对抗旋钮 ("`spgd_gradient` 返回 ΔJ/δ, 故调大 δ 会抵消信号")。
> 该推断是错的 —— 加到 200+ epoch 后目标量**单调下降**, 只调 `lr` 即可见效。
> **教训: 判定"不收敛"前先确认 epoch 足够**; 优化器慢不等于不收敛。

**已知接口约定** (与 `ThorlabWFS` 完全对齐, 缺一个就直接崩):
- `get_wavefront()` 的 stats 键必须是 `min/max/diff/mean/rms/wighted_rms` ——
  runner 会读 `statics["wighted_rms"]` 来调度学习率。
- `build_subaperture_mask()` 返回 **2 元组** `(mask_2d, valid_indices_flat)`;
  三个调用点都按 `mask, _ = ...` 解包。
- 该 mask 必须按 **subaperture 网格** (`num_spots_x * num_spots_y`) 定尺寸,
  **不能**按 flux 定: OOPAO 的 flux 报在 8×8 lenslet 网格 (`n_subap=6` 时 64 项),
  而 slope 跨的是 6×6 subaperture 网格 (36 项, 展开 72)。尺寸错位会让标定报
  `boolean index did not match`。
- runner/优化器直接读取的成员共 12 个: `num_spots_x/num_spots_y`、`mla_index`、
  `serial_num`、`device_name`、`exposure_time`、`high_speed`、`use_custom_ref`、
  `pupil`、`d_x`、`get_mla_name()`、`set_ref_plane()`。
  新增传感器类型时必须一并提供, 详见
  `tests/ao_shaping/drivers/sim/test_simulated_wfs.py::TestRunnerFacingSurface`。


**OOPAO API 坑** (实测): `relay()` 原地修改并返回 `None`; `wfs_measure` 返回
`[slopes, flux, raw]`; CuPy 处于活跃状态, 必须用 `cp.asnumpy`
(OOPAO 自带 `to_numpy` 在当前版本有兼容问题)。

**已知约束**:

1. **后端缓存**: `oopao_backend._get_backend` 是 `@lru_cache(maxsize=8)`。同一进程内切换
   仿真配置 (网格/波长/pixel_size) 后必须调用 `oopao_backend._get_backend.cache_clear()`,
   否则命中旧后端实例得到与配置不符的相位屏。
2. **静默回退**: `_oopao_enabled()` 在 OOPAO 不可导入或不可用时返回 `False` 并**静默退回
   numpy**。开启后务必确认 `oopao_backend._oopao_available()` 为 `True`, 否则会误以为在跑 OOPAO。
3. **两后端不等价 (勿假设同 Cn2 可互换)**: 端到端实测 `disturbance_rms` 的
   oopao/numpy 比值在**所有**湍流档位上**恒定**, 但**随仿真配置变化** (open 模式
   1.068× / closed 模式 2.628×, 跨 1e-16→5e-14 两个数量级相对离散度 ≤1.8e-09), 详见
   `docs/oopao_impact/report.md` §4.3。**不要把某次配置的常数当成普适标定系数。**
   绝对 Strehl / FWHM / PIB **不可跨臂直接比较**。
   ⚠️ 比较时注意 `init_rms` (`compat.py::_phase_rms()` = **截瞳后总波前** RMS, 含像差与 DM)
   与 `disturbance_rms` (`env._disturbance_rms` = **全网格未截瞳**原始相位屏 RMS) 是**两个不同的量**,
   不可互相推断 —— 前者可因像差项跨臂反向。

4. **相位屏标定 (2026-09 修正)**: `_rescale_for(r0_slab) = (_R0_REF_500/r0_slab)**(5/6)`,
   **只此一项**。此前版本额外乘了 `lam/_LAM_REF_500`、并除以经验常数 `_CAL_REF`、再乘
   `sqrt(1.03)`, 三者叠加使后端相位屏 std **完全不含波长依赖** (物理上相位[rad]必须 ∝1/λ),
   且常数项掩盖了真实差异。现已验证: OOPAO 臂 std 随 λ 的变化与 legacy 完全一致
   (比值 0.5000 / 0.6865 = 波长比), 比值收敛为**与波长无关的单一常数**。
   **不要再引入独立的 λ 因子** —— 波长已完全由 `compute_r0()` 里的 `r0_slab` 携带
   (`r0 ∝ λ^(6/5)`, 相位幅度 ∝ `r0**(-5/6) ∝ 1/λ`), 额外因子会重复计入并抵消它。
   OOPAO 自身的次谐波增强使其相位方差**更接近**解析 von-Karman 值 (legacy 纯 FFT 路径
   低阶模欠采样), 故不再除以任何经验归一化常数。

5. **内尺度 `l_min` 无法兑现 (OOPAO 限制)**: `OOPAO/Atmosphere.py` `__init__` **没有 `l0`
   参数**, `generateNewPhaseScreen` 也不传, 故 OOPAO 恒用 `l0=1e-10` —— 内尺度滚降
   位于网格 Nyquist 之上, 实际不存在。仅当 `l_min` 粗到网格可分辨时才有真实影响;
   `oopao_backend.inner_scale_is_resolvable(l_min, dx)` 判定该情形, `beam_backend`
   命中时打 `logger.warning`。默认配置下 `l_min` 为亚像素, 忽略它无害。
6. **安装**: OOPAO 以 editable 方式从 git submodule `libs/OOPAO` 安装 (`uv pip install
   --no-deps -e libs/OOPAO`)。`libs/OOPAO` 为 submodule (canonical
   `https://github.com/cheritier/OOPAO.git`); 换了 OOPAO 版本后重装即可。
7. **API 漂移**: 本地 submodule 比旧 pin `8e12a17f` 领先若干提交, 导入会打印
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
