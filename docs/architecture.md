# 架构：物理-optical 数据类模型与仿真唯一入口

> 本文件从 README.md 拆出（2026-10-13）。驱动分层/惰性加载见 [drivers_architecture.md](drivers_architecture.md) 与 [src/ao_shaping/drivers/AGENTS.md](../src/ao_shaping/drivers/AGENTS.md)。

## 物理-optical 数据类模型 (`ao_shaping.model`)

`ao_shaping.model` 将二维采样值与像素间距、波长及来源信息放在一起，明确区分
相位 (rad)、光程差 (m)、场振幅、强度和电压。数据类已经实现；跨模块调用仍在分阶段迁移，
不要把类型存在理解为所有驱动和优化器都已返回该类型。

| 类 | 单位与用途 |
|---|---|
| `PhaseMap`, `OPDMap` | 未包裹相位 (rad) 与光程差 (m)，用明确的波长相互转换 |
| `AmplitudeMap`, `ComplexField`, `PSFImage` | 场振幅 `|E|`、复电场与焦面强度；强度转振幅使用平方根 |
| `WfsSlopes` | 分开的 `sx`/`sy`，单位为 `arcsec` 或 `rad`，支持 `to_units()` |
| `DmCommands` | 电压数组、致动器数量和允许范围，支持 `clipped()` |
| `ZernikeCoefficients` | 带显式 Noll 索引的 `rad`/`waves` 系数 |
| `CoordinateGrid2D` | 坐标、像素间距、原点和单位 (`um`/`m`/`pixels`) |
| `BeamMetrics`, `TurbulenceParameters`, `WavefrontStatistics` | 光束指标、湍流参数和波前统计 |
| `FieldMetadata` | 时间戳、来源、波长及附加元数据 |

**物理空间重采样**: `PhaseMap`、`OPDMap`、`AmplitudeMap`、`ComplexField` 和
`PSFImage` 都提供 `resample_to(shape, pitch_size_um)` (PSF 参数名为
`pixel_scale_um`)。`model/spatial.py` 将输入和输出网格的几何中心都设为
`(0, 0)`，像素中心坐标为 `(index - (size - 1) / 2) * pitch`；默认双线性插值，
超出输入网格的采样值填零。这是点采样插值，**不保证积分光通量守恒**。
`utils/image/resample.py` 的亮斑居中裁剪用于相机图像对齐，语义不同。

```python
import numpy as np

from ao_shaping.model import AmplitudeMap, ComplexField, PhaseMap

phase = PhaseMap(np.zeros((5, 5)), pitch_size_um=8.0)
opd = phase.to_opd_map(wavelength_m=532e-9)  # metres
phase_16um = opd.resample_to((3, 3), pitch_size_um=16.0).to_phase()

amplitude = AmplitudeMap.from_intensity(np.ones((5, 5)), pitch_size_um=8.0)
field = ComplexField.from_maps(phase, amplitude)  # 检查形状和像素间距
```

**OOPAO**: `PhaseMap`、`AmplitudeMap` 和 `ComplexField` 提供
`from_oopao_source()` / `to_oopao_source()`。实际调用时才通过
`ao_shaping.drivers.sim._oopao_compat` 加载 OOPAO，避免模型包与模拟驱动的循环导入。
可运行以下无硬件测试检查当前环境：

```bash
uv run pytest tests/ao_shaping/model tests/ao_shaping/drivers/sim/test_oopao_backend.py -q
```

**当前接入范围**: DM 基类、`SimMicroDM`、`SimulateDM` 接受 `DmCommands`；
`phase_to_slm_grayscale()` 接受 `PhaseMap`。这些接口仍接受原始数组，原始数组调用
保持原有返回类型。目前没有统一添加弃用警告。硬件驱动覆盖方法、WFS、优化器、GUI
和其余工具尚待迁移。

## 2f-Fourier 整形台架的唯一入口

前向模型、目标函数指标与参考优化器**只在** `drivers/sim/slm_shaping_bench.py`。
`optimizer/wfless/slm_shaping_bench.py` 是**向后兼容 re-export shim**。

> ⚠️ **不要再手抄第二份实现**。2026-10-01 的 merge 曾在两个模块各留一份
> `gs_shape`/`differentiable_shape`/`spgd_shape`，而副本用的是
> `beam_backend.focal_plane` 走**未加零填充**的瞳孔，canonical 版则零填充到
> `far_field_size` 再做 Fraunhofer FFT —— 同一个方法**按 import 来源返回不同指标**，
> 且不报任何错。拆分模块时只加 re-export，不要复制函数体。

同理 `iterative_zernike_shaping._zernike_basis()` 直接返回 `__init__` 里由
canonical `ZernikeGenerator` 预计算的基，**不得**在算法层重新推导径向多项式
(那会构成第二套会漂移的 Zernike 数学)。

## 项目结构（详细）

> 本节从 README.md 迁入（2026-10-13）。

```
AO-shaping/
├── src/
│   ├── ao_shaping/               # 主程序包
│   │   ├── __init__.py
│   │   ├── main.py               # CLI入口点 (Click-based), 注册 20 个命令
│   │   ├── config.py             # 中央配置 (DM_N_ACTUATORS, DEFAULTS, PATHS, ao_config)
│   │   ├── runners/              # 运行器包 (硬件编排层: Click CLI, 设备生命周期)
│   │   │   ├── __init__.py       # Lazy imports + re-exports (via __getattr__)
│   │   │   ├── runner_common.py  # 共享参数 dataclass + with_params click 集成
│   │   │   ├── closed_loop.py    # [向后兼容 shim] AOClosedLoop re-export (canonical: optimizer/wf/)
│   │   │   ├── matrix_runner.py        # DM响应矩阵 (dm-matrix) + Hadamard响应矩阵 (hadamard-matrix)
│   │   │   ├── zernike_search_runner.py  # GA Zernike (ga-zernike) + 贪婪局部搜索 (greedy-zernike)
│   │   │   ├── gsnet_cache.py             # GSNet缓存
│   │   │   ├── gsnet_dataset.py           # GSNet数据集
│   │   │   ├── gsnet_offline.py           # GSNet离线评估
│   │   │   ├── gsnet_train.py             # GSNet训练
│   │   │   ├── nlight_dm/                 # NLight DM 相关 runner
│   │   │   │   ├── wf_runner.py           # Wavefront RMS优化 (wf)
│   │   │   │   ├── axis_beam_runner.py    # PIB优化 (pib)
│   │   │   │   ├── pipeline_runner.py     # 串行WF→PIB (pipeline)
│   │   │   │   └── combined_runner.py     # AdaMOD+SPGD (combined)
│   │   │   ├── micro_drive/               # 微驱 (R50Power) 相关 runner
│   │   │   │   └── voltage_runner.py      # 交替电压 (alt-voltage) + 全量交替电压 (full-voltage)
│   │   │   └── slm/                       # SLM 相关 runner
│   │   │       ├── rms_zernike_runner.py     # SLM Zernike RMS (rms-zernike)
│   │   │       ├── zernike_matrix_runner.py  # Zernike响应矩阵 (zernike-matrix) + closed-loop
│   │   │       ├── slm_shaping_runner.py     # SLM Zernike PIB (slm-pib) + 方形整形 (spgd-square)
│   │   │       ├── gsnet_runner.py           # SLM自由相位方形整形 (slm-gsnet)
│   │   │       ├── gs_refine_runner.py       # GS 预矫正 + SPGD 细化 (slm-gs-refine)
│   │   │       ├── model_in_loop_runner.py   # 正向模型闭环校正 (slm-model-in-loop)
│   │   │       └── offset_runner.py          # SLM偏移校准 (standalone, 未注册)
│   │   ├── algorithm/           # 优化算法 (5 子包: gradient/heuristic/signal_processing/tabu/goal_functions)
│   │   │   ├── gradient/        # 梯度优化 (Adam, SGD, Muno, MuonW, etc.)
│   │   │   ├── heuristic/       # 启发式搜索 (GA, PSO, SA, CEM, DE, HC, RS)
│   │   │   ├── signal_processing/  # 信号处理 (GS, 相位包裹, 可微分光束整形, 控制律)
│   │   │   ├── tabu/            # Tabu 搜索
│   │   │   ├── goal_functions/  # 目标函数 (target_func, image_metrics)
│   │   │   ├── base.py          # Base 梯度优化器基类
│   │   │   └── README.md        # 优化器 class-based 约定
│   │   ├── model/                 # 物理量数据类与物理坐标重采样
│   │   │   ├── __init__.py        # 包级 re-exports
│   │   │   ├── field.py           # PhaseMap, AmplitudeMap, ComplexField + OOPAO 转换
│   │   │   ├── quantities.py      # OPDMap, DmCommands, PSFImage 等物理量
│   │   │   ├── spatial.py         # 以物理坐标为基准的插值
│   │   │   └── metadata.py        # FieldMetadata (timestamp, source, wavelength)
│   │   ├── drivers/              # 硬件驱动 (see drivers/AGENTS.md)
│   │   │   ├── device_base.py    # Device 基类 + _load_sdk()/_ensure_sdk() 惰性契约
│   │   │   ├── device_registry.py
│   │   │   ├── _lazy.py          # PEP 562 install_lazy_attrs (包级惰性)
│   │   │   ├── mock_devices.py
│   │   │   ├── adc/             # NI DAQ ADC 电压采集
│   │   │   ├── ccd/             # 相机 (Daheng, MiiCam, ffmpeg)
│   │   │   ├── dm/              # 变形镜 (NLight, MicroDM, AsyncMicroDM, HadamardDM, ZernikeDM, _registry)
│   │   │   ├── slm/             # 空间光调制器 (Santec, ZernikeSLM)
│   │   │   ├── wfs/             # 波前传感器 (Thorlabs; base/_registry + ThorlabWFS)
│   │   │   ├── tm/              # 定时模块 (Serial/FSM)
│   │   │   └── sim/             # 数字孪生仿真 (atmos/ccd/dm/laser/optics/wfs + dm_optics/disturbance)
│   │   ├── optimizer/           # 高层优化器 (策略实现层)
│   │   │   ├── constants.py, spgd.py, combined_optimizer.py
│   │   │   ├── wf/              # 波前优化 (RMS, Zernike, GA, 响应矩阵, closed_loop)
│   │   │   ├── wfless/          # 无波前优化 (PIB, SPGD, 可微分整形)
│   │   │   └── rl/              # 强化学习 (SAC, LR-WFS)
│   │   ├── utils/               # 工具函数 (io/image/wavefront/slm 子包 + root-level shims)
│   │   │   └── image/
│   │   │       └── target/      # 目标形状: ccd, metrics, objective, patterns, square
│   │   ├── tools/               # 硬件交互工具 (tools/slm/, tools/micro_dm/, slm_train_data_collect.py)
│   │   ├── display/             # 可视化 (窗口, GUI帧, pygame)
│   │   └── gui/                 # GUI组件 (Streamlit), 按设备域分包: r50/dm/slm/zernike/ccd
│   ├── ml/                      # 机器学习独立包
│   │   ├── gsnet/               # GSNet: GS + SPGD 整形训练
│   │   ├── gsnet_debug/         # GSNet 调试/离线评估
│   │   ├── phase/               # U-Net+GAN 相位预测 (dataset, discriminator, trainer, unet)
│   │   ├── hwdataset/           # 硬件相位→相机图像 Dataset (index, records, transforms, cache, dataset, zernike_dataset)
│   │   ├── zernike/             # Zernike forward/inverse model (models, forward_model, train_coeff, train_amp, losses, metrics, augment, dataset, inverse_design, eval_stats)
│   │   ├── train.py             # ML 训练入口
│   │   └── wandb_logger.py      # W&B 集成
│   ├── calculators/               # Cython扩展 (standalone)
│   └── optical_ui/                # [DEPRECATED] Empty package
├── tests/ao_shaping/              # Tests (镜像 src 结构)
├── scripts/                      # 实用脚本 (含报告生成 generate_*_report.py)
├── docs/                         # 设备说明文档 (设备规格/SDK用法/操作手册/图集)
├── report/                       # 实验报告 (结论/测量/基准), 每份带生成脚本溯源块
├── libs/                         # 第三方SDK二进制 (gxipy, Drv_UDPST)
└── AGENTS.md                     # 开发指南
```
