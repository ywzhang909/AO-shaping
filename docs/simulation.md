# AO 仿真指南（数字孪生）

> 2026-10-13 重写：替换旧的 `sim.digitaltwin` 引擎说明。旧版所述 `sim.digitaltwin`
> 包**已不在仓库中**，`atmos/screens.py` 中的 digitaltwin 路径只是可选回退
> （缺失时 logger.warning 后走内置实现），`SimulatedThermalScreen` 依赖它且回退为
> 静默 no-op，**不可使用**。当前仿真核心是 `beam_backend` + `slm_shaping_bench`。
> 驱动层惰性加载契约见 [src/ao_shaping/drivers/sim/AGENTS.md](../src/ao_shaping/drivers/sim/AGENTS.md)。

## 概览

项目包含完整的数字孪生仿真环境，支持无硬件测试：

```bash
python -c "from ao_shaping.drivers.sim import SimTurbulenceAOEnv; env = SimTurbulenceAOEnv()"
```

仿真模块包括:
- 变形镜仿真 (带迟滞特性) — `dm/simulated_dm.py`: `SimulateDM`, `SimMicroDM`
- 波前传播与光束传播 — `beam_backend.py` (canonical: `turbulence_phase`, Fraunhofer/角谱), `wave.py` (legacy Wave 工具)
- 模拟 WFS (`SimulatedWFS`, OOPAO Shack-Hartmann, 瞳面 slope 模型) — `wfs/simulated_wfs.py`
- DM→瞳孔相位耦合 (`SimDmOptics`, 可分离高斯影响力函数) — `dm_optics.py`
- 2f-Fourier SLM-PIB 数字孪生 (`SimPibSystem`) — `slm_shaping_bench.py` / `slm_pib_sim.py`
- FourierGSNet 仿真环境 — `fouriergsnet_env.py`
- 波前干扰 (湍流 + 热晕) — `disturbance.py`
- 兼容层 (`compat.py`: `AOConfig`, `TraditionalAOSystem`, 供 `sim_spgd.py` 与 RL `envs.py`)

## 2f-Fourier 整形台架的唯一入口

前向模型、目标函数指标与参考优化器**只在** `drivers/sim/slm_shaping_bench.py`。
`optimizer/wfless/slm_shaping_bench.py` 是**向后兼容 re-export shim**。

> ⚠️ **不要再手抄第二份实现**。2026-10-01 的 merge 曾在两个模块各留一份
> `gs_shape`/`differentiable_shape`/`spgd_shape`，而副本用的是
> `beam_backend.focal_plane` 走**未加零填充**的瞳孔，canonical 版则零填充到
> `far_field_size` 再做 Fraunhofer FFT —— 同一个方法**按 import 来源返回不同指标**，
> 且不报任何错。拆分模块时只加 re-export，不要复制函数体。

## DM→相位耦合已接入仿真 (2026-10-01)

`SimDmOptics` (`sim/dm_optics.py`) 把 DM 电压映射为瞳孔相位，经 `SimPibSystem.dm_optics`
在 `far_field()` 的 **cache-miss 分支**叠加 (与 `disturbance` 同一约定: 相位贡献者在**求值时**相加,
绝不烘进 `self._phase`)。此前 `pib` / `combined` 的仿真结果**没有物理意义** (DM 动作对远场无影响)。

| runner | 驱动量 | 仿真下是否有物理意义 |
|--------|--------|---------------------|
| `slm-pib` / `slm-gsnet` / `spgd-square` | SLM 相位 | **有** —— `set_phase_rad` 直接改变远场 |
| `pib` / `combined` | DM 电压 | **有** —— `SimulateDM.send_voltages` → `SimDmOptics` → `far_field()` |
| `wf` / `rms-zernike` 等 | DM 电压 → WFS | **有** —— DM 同时发布给当前活跃的 `SimulatedWFS.dm_optics` |

实现要点:
- **可分离高斯影响力函数**: 每个致动器一个高斯凸包, x/y 可分离, 故用一次矩阵乘求值。稠密 influence 矩阵在真实 1920×1200 面板下需 ~1.2 GB, 不可接受。
- **单位链**: `opd_um = stroke_um * v / v_max` → `phase_rad = opd_um * 2π / (λ_nm * 1e-3)`。0 V 为平场 (与真实驱动一致)。`V_Min=-300` 与 `V_Max=499` **不对称**, 故负向半程行程短于正向 —— 裁剪到负轨**不是**正轨的镜像。
- **相位是 raw 未包裹弧度**, 不做 `mod 2π`: 全项目唯一 wrap 点在 SLM 驱动。
- **cache 失效靠 `dm_optics.version`**: DM 是独立光学状态, 电压变化必须让 `_far_field` 失效, 否则相机继续返回旧图, 看起来"DM 毫无作用"。
- 传感器按自己的瞳孔网格 (48×48) 持有一份 `SimDmOptics`, 因此**无需**把 1200×1920 重采样。
- ⚠️ 以前 DM 只发布到远场, 于是 `wf` 永远读到平 pupil、无论 DM 怎么动 RMS 都是 0。

修复后实测 `main.py pib --cam_type sim --dm_type sim -e 200`, 目标函数真实上升: pib 0.66 → 2.33 → **2.98** (修复前 200 epoch 恒为 0.0145; pib 单位为 %, 故可 >1)。

## 相机噪声模型红线

⚠️ **相机噪声的地板项会把功率比指标淹没**: `far_field_noisy()` 曾以 `clip(img, 0, None)` 收尾。
`far_field()` 是 `|FFT|²` 本身非负, 帧内所有负值**只可能来自读出噪声** —— 在 0 处裁剪会把对称分布
**整流**, 凭空造出与**像素数**成正比的 DC 地板: shot noise 地板/信号 **1.005×** (正确),
read noise 经 clip 后 **4598×** (缺陷)。光斑只占 ~150 px 而画幅 2.3 M px, 地板携带 ~3000× 的光,
`pib` 的分母是整帧总功率, 于是它量的是这块地板。

修复: **不再逐帧裁剪**。真实探测器的本底位于其阈值**之下**, 只在量化时裁剪一次, 去黑电平靠暗帧扣除。
判据用**暗区均值** (≈0) 与**桶内信号占比**, **不要**用整帧 total 判定噪声 (零均值噪声在 2.3 M px 上
求和标准差 ≈ σ√N ≈ 758, 与 152 的信号总量同量级)。参见 `tests/ao_shaping/drivers/sim/test_sim_noise_model.py`。

## 波前干扰 (湍流 + 热晕): 静态 / 动态

`ao_shaping.drivers.sim.disturbance` 为 SLM 类仿真提供**大气湍流 + 热晕** 波前干扰，
`SimPibSystem` 通过关键字参数 `disturbance=` 接入 (`far_field()` 在傅里叶变换前叠加，
且只在 cache-miss 分支推进，故每次真实光学评估只消耗一次；`self._phase` 保持纯 SLM 命令，
不会被重复计入)。两种体制：

| 体制 | 含义 |
|---|---|
| `static` | 全程复用**唯一一张冻结**相位屏 (对应仓库 `closed` 湍流) |
| `dynamic` | **每次光学评估重抽一张独立**相位屏 (对应 `open`/sliding，即完全去相关 "white-in-time" 极限) |

> `dynamic` 不是风场平流模型: 真实大气去相关时间 ~10–50 ms 远短于本环路每评估 ~0.375 s，故完全去相关
> 是此处的合理渐近。

湍流复用 canonical `beam_backend.turbulence_phase` (von Karman，相位屏方阵生成后按面板原生
8 µm 像素居中裁剪，避免粗网格丢失约 41% 相位幅度)；热晕由 canonical
`zernike_utils.generate_zernike_phase` 构造 (Noll 4 离焦 + Noll 11 球差，负号=负热透镜)，
经 smoothstep 光晕窗延展并按 waves 归一化峰谷值。全程 raw 未包裹弧度。

**注意**: `cn2` 与 `distance` 是**退化**旋钮 (`r0` 只依赖二者乘积，单层薄屏无传播物理，本台架是
~0.3 m 实验室 2f 光路) —— 属参数化应力测试而非大气传输仿真；numpy 相位屏缺少次谐波补偿，
实测 σ 低于同 r0 的解析值，故报告一律引用**实测** σ。`SimulatedThermalScreen`(热晕相位屏)依赖未安装的
`sim.digitaltwin` 且回退为静默 no-op，**不可使用**。

配套运行与报告 (完全离线，见 `scripts/README.md`):
```bash
python scripts/slm/slm_pib_sim_run.py --epochs 300 --disturbance static
python scripts/slm/slm_pib_sim_run.py --epochs 300 --disturbance dynamic
python scripts/slm/generate_slm_pib_sim_report.py --max-runs 2   # → report/slm_pib_sim/report.md
```

实测 (300 epochs，各 604 次光学评估): static 用 1 张屏、逐次 RMS 恒定 (σ≈0.566 rad ≈0.090 waves)；
dynamic 用 604 张屏、逐次 RMS 变化 (σ≈0.580 rad ≈0.092 waves)；目标改善 static +0.0286 > dynamic +0.0202
(冻结扰动更易被校正)。

## 仿真 WFS (`SimulatedWFS`) 使用要点

见 [docs/hardware_drivers.md](hardware_drivers.md#波前传感器) 的「仿真 WFS」小节
（piston 不可测、slope 行块布局、单位写死 532 nm、保真度不足以当波前基准等实测红线）。
`--wfs_type sim` 在 8 个 runner 命令上透传。

## Legacy: Wave 工具与 compat 层

`wave.py` 的 `create_wave / apply_aperture / apply_focus / propagate / power_bucket /
radius_metric` 与 `compat.py` 的 `AOConfig / TraditionalAOSystem` 仍供旧优化器
(`sim_spgd.py`, RL `envs.py`) 使用。注意：

- 这些工具历史上包装 `sim.digitaltwin`；该包现已不在仓库中，相关路径为可选回退。
- 新工作一律走 `beam_backend` + `slm_shaping_bench`，不要在 legacy 层上加新功能。
- `SimulatedTurbulentScreen` 若走 digitaltwin 路径，历史 bug：`harmonic=1` 产生 ~100°
  相位伪影，**始终用 `harmonic=0`**。

## 无硬件测试

```bash
uv run pytest tests/ao_shaping/drivers/sim -q
```

关键锁定测试：`test_simulated_wfs.py::TestRunnerFacingSurface`（WFS 接口契约）、
`test_sim_noise_model.py`（噪声地板判据）。
