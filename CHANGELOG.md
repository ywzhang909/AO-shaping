# 更新日志

> 本文件从 README.md 拆出（2026-10-13）。README 只保留索引，完整版本历史在此维护。

## v0.19.0 (v0.18.0 末 `e51ed38d` 之后 → 2026-10-19, 387 个提交, `e51ed38d..HEAD`)

**正向模型在环（model-in-loop）闭环成型 + GS→Zernike 阶数扫描立项 + scripts/ 重组 + ruff 硬门禁**

- **model-in-loop 正向模型闭环** (`080e0ac9`, `4c09a5d1`, `89494ea9`, `eb125df6`): 不确定性采样 Step A 探针（R-82/R-86, #134，替代盲加随机探针）；warm-start 复用上一轮已验收 `fitted_coefficients`（#92）；`pupil_oversampling` 透传到正向模型；freeform holdout 探针 + 阶数无关 probe-set loss。实机实验报告：`report/shape_trials/model_in_loop_live_20261013/14/15/16/19.md`（五种目标形状、交错重放、warm/cold 对比、oracle 复核）。
- **model-in-loop 实机验收增强** (`3ddef179`, `07a4a560`, `3a1f8711`): 几何门槛补目检（拦住相关性看不见的离焦失配）；工作精度默认 float32（float64 的精度到不了面板）；报告补 SLM 相位图 + CCD 远场图与 X/Y 截面。
- **GS → Zernike → SPGD 阶数扫描立项** (`9d079cf1`, `b366a634`, `f79a5f84`, `07d5ea23`): Zernike 拟合模块 + 阶数扫描驱动 + 离线报告（仿真：拟合残差 8 阶 0.0719→0.0385 rad 单调不增、`n_max=25` 不触发 `ZERNIKE_CLIP`）；硬件实验计划 [`report/shape_trials/gs_refine_zernike_hardware_plan.md`](report/shape_trials/gs_refine_zernike_hardware_plan.md)（H0~H12，两个 P0 门：GS 对比择优 + 噪声地板 σ）。
- **GS 细化缺陷修复** (`540d51b2`): #140 — `best_phase` 改为保存收敛迭代点而非胜出随机探针（原报告增益 +0.0126 含选择偏差）。
- **scripts/ 按处理对象重组** (`569ea514`, `999b5a08`): 139 个扁平脚本按处理对象归类（`slm/` `ml/` `zernike_bench/` `signal_processing/` 等）；新增离线 model-in-loop 评估器、硬件回放脚本与 focus probe（`cc30224d`）。
- **曝光子系统集成** (`d82903e5`, `aa2ba82e`, `91209176`, `002cbdbb`): 括号式多曝光采集 + Mertens 融合（#141 待实机验收）；Daheng 逐会话曝光上限收敛为单一 helper（#136，取代 8 处重复解析）；曝光选项显式 auto/default 三态。
- **ruff 硬门禁** (`0cec5f5d`): #132 决策 B — `.github/workflows/ruff.yml` 硬门槛（push + PR），pylint 降为参考级；**F-5 切片 1**（`d57647a5`, #56）41 个文件补 `from __future__ import annotations`；F841/F811/F821 全部清零（`a6e41efa`/`10aacef0`/`26fe1e20`）。⚠️ 2026-10-19 起 `.github/` 整体移出 git 跟踪（本机保留），新克隆无 CI 工作流，本地 `ruff check .` 仍须全绿。
- **发送管线叠加层统一** (`660d6bbf`): GUI 与三个闭环优化器（gs-refine / model-in-loop / zernike-pib）共用 `build_send_overlay`（离焦 + 底相位唯一读取点）；`phase_overlay` 代理接入 controller 发送路径（`d3059f2a`）。
- **GUI 文档化与视觉优化** (`5e20967e`, `14cfec70`, `a176ea2c`): 全部 GUI 页面建 SRS + 用户手册并移至 `docs/GUI/`（16 篇）；multi_slm_controller 视觉优化（#131）；SLM 控制器按时效分类重排（`e8f80599`）。
- **去重/重构批次** (`26d01013`, `fdc0696c`, `bea6691b`, `c40d0f6c`, `a0dc66a9`, `d52faf4c`): #87 `fit_exact` 单一定义；#154 composite 分数单源（`shaping_score`）；#153 `predictive_variance` 公共 API；#152 `bench_geometry.json` schema 单入口；#91 P0 相机注入 seam 单一真源；#130 objective 方向单源。另 `531fc1fb` 删除第三套手写 Zernike 实现（#66）、`dd85c0d0` 增加 GS 算法 Cython 版本。
- **论文** (`df90efaa` → `8771e0bc`): v2.1→v4.2 — MIL 理论与可微光学框架综述（§3.5/§5）、TU/e 系统松弛与混合拟合的 sim 复现（§7.7，#149）、T4 不确定性采样定量实验（表16）。
- **实机实验报告体系**：`report/shape_trials/` 持续扩充（model-in-loop 10-13~10-19 五份实机报告、`gs_issue_live_20261015` 完整报告、形状目标整形实验总结）；provenance 基线守卫 gitignore-aware（`069fea94`, #108）；报告图全部换 SLM 相位图 + CCD 远场图 + X/Y 截面（`cdcd3eda`/`ccc571d9`/`afc2300e`）。
- **仓库卫生** (`19239d30` 起): 8 个 machine-local 目录（`.auto_doc_review`/`.claude`/`.codegraph`/`.github`/`.slim`/`.streamlit`/`.vscode`/`_scripts`）移出 git 跟踪并加入 `.gitignore`；本地副本保留，不随仓库分发。

## v0.18.0 (2026-10-07 → 2026-10-12, 107 个提交)

**逆向整形模型、自适应 SPGD、GS 预矫正上硬件、实验系统工具重组**

- **逆向光束整形 PhasorUNet** (`ml/shaping/`, `69950ec`): 新增 `models.py` / `physics.py` / `losses.py` / `dataset.py` / `train.py` / `evaluate.py` / `viz.py` —— 目标远场 → SLM 相位的逆向 U-Net，带物理往返 (round-trip) 一致性损失。语料与结论见 [`report/inverse_shaping_unet/report.md`](report/inverse_shaping_unet/report.md)。
- **slm-gs-refine 上硬件 + 回放** (`ce5df18`, `f4efa0f`, `db38f57`): GS 预矫正新增可选 weighted/HIO 热启动变体；相机 ROI 居中修复 + 完整 debug 记录保留；新增 `scripts/slm/replay_gs_refine_hw.py` 离线相位回放（同一相位序列可复现上硬件）。硬件对比见 [`report/shape_trials/gs_refine_live_20261012.md`](report/shape_trials/gs_refine_live_20261012.md)。
- **自适应 SPGD** (`ab11c19`, `81c0391`, `e9c1f35`): 新增 `adaptive_spgd_core`（stage_weights + 停滞检测器），`spgd-square` 接入 `--adaptive` 闭环；完整方案见 [docs/自适应 SPGD 光束整形闭环：完整方案总结.md](<docs/自适应 SPGD 光束整形闭环：完整方案总结.md>)。
- **整形试验** (`d01a053`, `e2b72f3`, `f402bcf`): 可选空间覆盖目标 (`--coverage`)；PIB 新增实测峰值可行性门限（替代软惩罚）；GS 细化最优结果与复现命令记录进 `report/shape_trials/`。
- **slm-model-in-loop 交互增强** (`5055244`): 共享 `slm-pib` 的 `--show` 实时视图，新增预测远场面板。
- **tools/ 按探测对象重组** (`699f149`): 扁平的 `tools/slm/` 拆为 `tools/kernels/`（共享内核）、`tools/zernike/`、`tools/ccd/`、`tools/wfs/`、`tools/slm/panel/`、`tools/slm/calibration/`。**无兼容 shim**，所有 `python -m` 入口与文档同步改写；`tools/slm/calibration` 经 PEP 562 惰性 re-export 保持"import 不加载原生 SDK"守卫。
- **runner CLI 参数原子化** (`eba0045`, `bf8c9b0`, `2e3354d`): runner 参数重构为原子配置，`with_params` 统一走 `utils.cli.params`。
- **Micro-DM IP 解析收敛** (`8ff9b38`): `resolve_ips` 提取进 `wiring_map` + 控制器采纳逻辑。
- **报告正式化** (`e51ed38`): `report/` 下全部 76 份实验报告补写每轮明确结论（"本轮结论"），provenance 块逐字保留。
- **GS 家族下一步计划** (`0026f58`): slm-gs-refine × FourierGSNet 的下一次尝试计划记录进 [`report/shape_trials/`](report/shape_trials/)。
- **可微仿真基准** (`e9c1f35`): 后端内核性能基准更新（[`report/benchmarks/backend_kernels_performance.md`](report/benchmarks/backend_kernels_performance.md)）。

## v0.17.0 (2026-10-06)

**Zernike 系数→远场前向模型训练** (15 个提交 `0694a0e` → `493ef1e`):

- **物理-optical 前向模型** (`ml/zernike/models.py`): `ZernikeAmpModel` —— 用单个全局 Zernike 系数向量 (raw 弧度, Noll 1 piston 排除) 表示整个台面的波到远场的确定性映射: `phasor(cos,sin) → complex pupil → exp(i·Σc_k·Z_k) → centre-pad + fftshift(fft2(ifftshift)) → |F|² (intensity)`。可观测量 `intensity` (默认) / `amplitude`, 归一化 `peak`/`sum`/`none`。默认 `n_max=15` (135 系数), `grid=64`, `observable="intensity"`, `normalization="peak"`。详见 [`report/zernike_coeff2amp/report.md`](report/zernike_coeff2amp/report.md)。

- **可微前向模型** (`ml/zernike/forward_model.py`): `ZernikeCoeffConvNet` (coefficient projection → conv decoder) + `ZernikeCoeffMLP` (baseline)。两者都输出 raw unbounded 单通道图像, peak-normalize 由 caller 完成。ConvNet 拥有空间局部性优势 (移形/尺度), MLP 作为对照臂。

- **训练器** (`ml/zernike/train_coeff.py`): `CoeffTrainConfig` (冻结 dataclass, `--input-terms`/`--protocol`/`--fold`/`--held-out-path`, W&B 集成, `--w-ellipse`). 两种折叠协议: `"file"` (18 折 leave-one-pickle-out) / `"objective"` (4 折 leave-one-objective-out). R² 选择 checkpoint (PSNR/SSIM/MSE 仅作诊断 —— 它们在 per-frame peak-normalised 目标上是退化的)。`--input-terms` 修剪输入到前 N 项 (语料最长向量 78 项, 136 维中 58 维恒为 0)。

- **物理感知损失** (`ml/zernike/losses.py`): `LossConfig` + `composite_loss`。移植自 `ao_shaping.utils.image.target.metrics` 的 PIB/uniformity/energy-loss (torch 版, ROI mask 复用 canonical builder)。新增:
  - `shape_gap_term` (anchored: pred 与 reference 的 ROI 形状统计绝对误差 —— 解决 unanchored PIB/uniformity 在"拟合"任务上的劣性: val R² +0.78 → -0.86)
  - `spot_moment_gap_term` (锚定的径向二阶矩相对误差, MSE 对核/晕分布盲)
  - `ellipse_parameters`/`ellipse_gap_term` (5 参数椭圆 gap, 捕捉位置/aspect/tilt 而径向矩无法)
  - `speckle_detail_term` (log-intensity 梯度差, 惩罚过度细微结构)
  - `poisson_nll` (FourierGSNet Eq.28)

- **数据增广** (`ml/zernike/augment.py`): `phasor_noise` (感应级 phasor 噪声), `phase_flip`, `phase_shift`, `strong_phase_pairs` (GS 提案 / 大振幅 Zernike / 空间粗糙随机相位 → OOD 合成对, 解决 inverse design 外分布问题)。全部显式 seeded (generator), 模型系数 Init 为 0。

- **配对统计来源** (`src/ml/zernike/eval_stats.py`): `sign_flip_pvalue` (精确 2^n 枚举, 最小 p = 2/2^n), `cohens_dz`, `holm_bonferroni`, `paired_comparison`, `skill_scores`, `seed_agreement`。纯 numpy (无 scipy/sklearn), 唯一来源。scripts/ 只做渲染。⚠️ **fold 与 seed 不能混配**: 同 seed 的 N 折共享一条初始化随机流, sign-flip 检验 p 偏乐观 (实测 `input_terms` 136→78 18 折 p=0.0144, 但固定一折 vary seed 后符号翻转, 差值量级小 30 倍 —— 初始化噪声)。

- **目标形状度量** (`utils/image/target/` 子包):
  - **结构项 `log_gradient_difference_metric`** (`utils/image/target/metrics.py`): `log_gradient_field`/`log_gradient_difference_from_fields`/`log_gradient_difference_metric` + `LOGGRAD_EPS`/`LOGGRAD_SIGMA` 常量。惩罚过度细微结构 (speckle), 能量-in-bucket 目标无法捕捉。
  - **`--w_loggrad` CLI** (`runner_common.py`, `objective.py`, `slm_zernike_pib.py`): 新增 `--w_loggrad` (默认 0), 串接 log-gradient 差异度量进 PIB 目标函数。

- **slm-model-in-loop** (`6b85858` + `efedabe`): `slm_model_in_loop` 闭环优化器 recorder pkl 修复 (写入现在真正生效), 负效应臂移除, README 补充硬件使用指南 (pkl 总是写出, 曝光陷阱, 运行前探针)。

- **CLI 帮助测试** (`493ef1e`): CLI 帮助 golden-file 测试增强, SLM 训练数据采集器 (`slm_train_data_collect.py`) 测试引入。

## v0.16.0 (2026-10-05)
- **`slm-pib` / `spgd-square` 合并进单个 runner**: `runners/slm/pib_runner.py` → `runners/slm/slm_shaping_runner.py`, `runners/slm/square_runner.py` 作为其 `square` 子命令合入。命令行名与全部选项不变; 新增两个等价入口 `main.py slm-pib square` 与 `python -m ...slm_shaping_runner square`。
- 修复: **`--cam_type sim` 在两半都是坏的** —— `spgd-square` 从未注册模拟相机 (所谓"离线试跑"会去开真实 Santec 并以 `SantecError -10002` 收场), `slm-pib` 只换了相机却仍构造真实 `Santec`, 且是在烧掉自动曝光探针之后。现统一走 `runner_common.patch_sim_square_shaping` / `patch_sim_pib_shaping`; 两半都能在 2f-Fourier 数字孪生上跑完 2 epoch。`--slm_type sim` 同时从"声明但无效"变成有效 (缺 `--cam_type sim` 时报 UsageError, 而不是去连硬件)。两个 helper 有一处**故意不同**: pib 那半**不** reset 模拟系统, 因为 `slm_pib_sim_run.py` harness 自己装了带干扰的定种子系统并包装了 `reset_system` 以保住干扰 —— 实测重跑 harness 干扰量不变 (`sigma_total=0.5658 rad`)。
- 修复: `spgd-square --delta` 是**空操作** —— `SlmSquareParams.delta` 硬编码 `0.1`, Click 无法区分"没传"与"传了 0.1", 于是默认 `lr=0` 的自适应调度每轮改写它。现改为 `None` + `resolve_spgd_delta` (省略=自适应, 显式=钉住), 与 `slm-gsnet` / `slm-pib` 一致。
- `spgd-square --debug` 过去不产出任何东西; 现写 `data/debug/slm_square_<ts>/` 的 PNG/PKL/JSON (带自描述 sidecar)。历史上的 CSV / 最优系数 / 最优图输出不变。⚠️ 实测这些记录目前**只有 `--basis freeform` 能进 `ml/hwdataset`** (Zernike 基 0/3, 因 `_c` 存的是活动模式向量而被判 `odd_coefficient_length`), 见 TODO H-23。
- 顺带修复了一个**上一提交遗留的破仓**: `gsnet_runner` 已 import `patch_sim_square_shaping` 但定义没跟着进 `runner_common`, 导致 `import ao_shaping.main` 直接 ImportError (clean checkout 无法跑 CLI)。
- 合并动机不是"文件挪个窝": 两个模块驱动同一对设备, 漂移已经产生了**行为差异**而非仅仅排版差异。详见 commit `278a3bc`; 待上设备复测项见 TODO H-20~H-24。

## v0.15.0 (2026-10-01)

**merge `banckend` (16419e5) 整合 + review 修复**

- **模拟 WFS** (`drivers/sim/wfs/simulated_wfs.py`): 新增 `SimulatedWFS` —— OOPAO Shack-Hartmann, 实现 `BaseWFS`, 可直接顶替 `ThorlabWFS`, 使所有 WFS 类 runner 无需硬件即可执行。测**瞳面相位梯度**(非焦平面传播)，排除 piston (SH 测不到绝对相位偏移，否则 `pinv` 放大成 +0.50 rad 伪系数)；`get_wavefront()` 返回 **waves**，`get_zernike()` 返回 **µm**。⚠️ 保真度不足以当波前基准 (tilt 单独注入只差 24%、与其他模式同注入差 83% ⇒ 主误差是模式间串扰)，任何由 `get_zernike()` 推出的 RMS 改善率或 Strehl **必须标注为「未验证」**。
- **WFS 注册表** (`drivers/wfs/_registry.py`): `register_wfs`/`create_wfs`/`list_wfs_types`/`resolve_wfs`，镜像 DM 注册表，延迟绑定故硬件包不 import 仿真包。`--wfs_type [thorlab|sim]` 定义在共享 `WfsParams` 上，8 个命令**既暴露也真正透传** (`wf`/`pipeline`/`rms-zernike`/`ga-zernike`/`greedy-zernike`/`dm-matrix`/`hadamard-matrix`/`zernike-matrix`)；`closed-loop` 刻意没有 (回放已保存的响应矩阵)。
- **DM→瞳面相位耦合接入仿真** (`sim/dm_optics.py`): `SimDmOptics` 用**可分离高斯影响力函数**把 DM 电压映射为瞳面相位 (稠密矩阵在 1920×1200 下需 ~1.2 GB 不可接受)，`SimulateDM.send_voltages` 同时发布给远场与当前活跃 WFS。此前 `pib`/`combined`/`wf` 的仿真结果**没有物理意义**。
- **驱动惰性加载契约**: `import ao_shaping` 及其任何子包**不得加载原生 SDK、也不得做磁盘 I/O**。包级用 PEP 562 `__getattr__` (`drivers/_lazy.py::install_lazy_attrs`)，构造级用 `Device._load_sdk()`/`_ensure_sdk()` + 只读 `_lib` 属性。`ThorlabWFS` 曾把 `load_dll()` 写在 `__init__` 里使**构造**就 `OSError`; `NLight.py` 曾**类体**执行 `np.loadtxt("data/dm_adj.txt")` 使 `import ao_shaping` 直接失败 (CWD 相对路径) —— 仓库输入资产现经 `drivers/dm/_adjacency.py::load_adjacency()` 锚定包位置。
- **分层方向修正**: `drivers/dm/__init__.py` 曾 import `drivers.sim.dm` (硬件 → 仿真，方向反了) 形成循环，该循环此前靠 `drivers/__init__.py` 的 eager 导入顺序侥幸未暴露 —— 一旦改成惰性就立刻炸。
- **`--disturbance-cn2`** 注入像差 (经 `SimDisturbance` 注入**瞳面**，默认 0 = 不注入)；`wf` 新增 `--lr`/`--delta` 覆盖。
- 修复: `far_field_noisy` 的 `clip(img, 0, None)` **整流**读出噪声、造出与像素数成正比的 DC 地板 (read noise 地板/信号 **4598×**)，使 `pib`/`combined` 停在 0.0145。改为不再逐帧裁剪 (真实探测器的本底在阈值**之下**，只在量化时裁剪一次)。
- 修复 (review): merge 残留的 `pipeline_runner` 引用未定义的 `wfs`(应为 `wfs_params`)；`sim/slm_shaping_bench.py` 的 `differentiable_shape` 在 torch import 被删后残留 19 处 `F821`；`optimizer/wfless/slm_shaping_bench.py` 是一份**手抄副本**且用的是**未加零填充的旧前向模型** —— 现改为对 canonical 实现的 re-export shim (同一方法此前按 import 来源返回不同指标)；`iterative_zernike_shaping` 的 `_zernike_basis` 调用了已被删除的 `_zernike_radial`(3 个测试失败) —— 现直接返回 canonical `ZernikeGenerator` 预计算的基。
- 规模: 93 文件 / +7232 −547。

## v0.14.0 (2026-10-01)
- **SLM-PIB 仿真的波前干扰** (`ao_shaping.drivers.sim.disturbance`): 新增 `SimDisturbance` / `DisturbanceConfig` —— **大气湍流** (复用 canonical `beam_backend.turbulence_phase`, von Karman 相位屏) + **热晕** (canonical `generate_zernike_phase`: Noll 4 离焦 + Noll 11 球差, 负热透镜, smoothstep 光晕窗 + waves 峰谷归一化), 全程 raw 未包裹弧度。提供 **static** (全程冻结屏 ≡ `closed`) 与 **dynamic** (每次光学评估重抽独立屏 ≡ `open`/white-in-time 极限, 非风场模型) 两种体制; `SimPibSystem` 以关键字参数 `disturbance=` 接入, 仅在 `far_field()` 的 cache-miss 分支推进 (每次真实光学评估一次), `self._phase` 保持纯命令不重复计入。`slm_pib_sim_run.py` 新增 `--disturbance/--cn2/--halo-pv-waves/...` 并写出 companion `disturbance.json`+`.npz`; `generate_slm_pib_sim_report.py` 渲染静态/动态对比 (配置派生 tag + 干扰相位屏图 + 逐次 RMS 轨迹图, 证明 static 恒定 / dynamic 变化) 与 8 条口径说明。53 个新测试用例
- 修复: 相位屏粗网格的系统性幅度损失 (方阵生成→居中裁剪至面板原生 8 µm 像素; 粗网格实测丢 ~41% 相位幅度)
- 修复: `slm_pib_runner._maybe_sim_patch` 的 `reset_system(seed=42)` 会丢弃已注入的干扰导致**静默无干扰运行** —— 现由 harness 包装该调用重新挂载干扰 (manifest 不再谎报有干扰)
- 修复: dynamic 模式曾保留全部相位屏 (~18 MB × 604 ≈ 11 GB) —— 改为仅累积标量
- 修复: 报告生成器 config 派生 tag 消除 `run0`/`run1` 图文件名冲突; `disturbance.json` 不再遮蔽 runner 自身 sidecar

## v0.13.0 (2026-09-29)
- **物理-optical 数据类模型** (`ao_shaping.model`): 新增 `model/` 包，包含 `PhaseMap`, `AmplitudeMap`, `ComplexField` 和 `FieldMetadata` 数据类 —— 封装 2-D 相位/振幅/复电场数组 + 像素尺寸 + 元数据(timestamp/source/wavelength)，并通过 `ao_shaping.drivers.sim._oopao_compat` 兼容层提供 OOPAO `Source` 转换 (`from_oopao_source`/`to_oopao_source`/`to_opd`)。71 个测试用例 (69 passed / 4 OOPAO 条件跳过)
- 修复: `scipy.ndimage.unwrap_phase` → 项目自己的 `ao_shaping.utils.wavefront.phase_unwrap.unwrap_phase`
- 修复: `ComplexField.field` 属性名冲突 → 使用 `dc_field` 别名

## v0.12.0 (2026-09-17)
- **共享扫描分析助手** (`tools/slm/slm_scan_analysis.py`): 纯 numpy 提取 `outlier_mask` (Z-score 异常点剔除)、`group_raw_scan` (灰度扫描分批求均值/标准差)、`analyze_linearity` (线性度指标)、`LINEARITY_AMPS`、`latest_match` 等 7 个公共符号; `zernike_matrix_runner` 改用 `outlier_mask` 剔除伪影点; 报告生成脚本 (`generate_zernike_response_matrix_report.py` / `generate_zernike_linearity_report.py`) 委托同一助手, 消除 `calibration.py`/`slm_lut_runner` 中的复刻逻辑
- **共享相机/相位工具** (`utils/image/hardware_utils.py`, `utils/slm_phase.py`): `open_camera()` 统一相机工厂 + `flat_gray`/`capture_frame` 等; 所有相机打开调用点 (slm_lut_runner, phase_capture, slm_diagnose, micro_dm_image_collect) 统一走 `hardware_utils.open_camera`, 消除 `slm_camera.py` 中间层 (注: `slm_camera.py` 模块随后已删除, 相机打开功能统一收敛至 `utils.image.hardware_utils.open_camera`；该模块原先还有一个 `utils/hardware_utils.py` 别名 shim，已于 2026-10-03 删除)
- **slm_slot 助手并入 Santec 驱动**: `utils/slm_slot.py` 删除, `SLOT_MIN`/`SLOT_MAX`、`SlotRotator`、`choose_slot`、`read_current_slot`、`apply_lut_remap` 移至驱动内部 (经 `santec/__init__.py` re-export 保持公共面)
- **calibration.py 拆分**: 离散几何标定 (`SLMCCDCalibrator`, 现行主流程) 与 LUT 标定 (`SLMLUTCalibrator`, `DeprecationWarning` 废弃) 分离; LUT canonical 路径收敛到 `slm_lut_runner` + `utils/slm_lut` → `Santec.load_lut`
- **tools/slm 迁移到 raw-grayscale 契约**: 扫描分析/校准工具统一走 uint16 直接灰度 (不经弧度转换, 2π=993 周期), 消除 `PatternHelper` 遗留 min-max 归一化
- **回归锚点测试 + 硬件自检 pytest 包装** (`a715937`): 纯逻辑回归锚点 (scan/lut/zernike/calibration/cartographer) + `AO_RUN_HARDWARE` 环境变量门控硬件自检用例 (默认 skip, 避免 CI 挂起)
- **文档迁移**: 根目录报告文档 (`beam_shaping_benchmark_metrics.md`, `issues_report.md`) 与 `performance_comparison` 输出迁移至 `docs/`; 重新生成硬件测试报告 (wfs/miicam/slm-200/micro-dm); `scripts/README.md` 补充报告生成架构文档
- **README 同步**: 补全缺失 CLI 文档 (`rms-zernike`, `ga-zernike`, `combined`, `spgd-square`), 修正项目结构树 (`ml/` 独立包位置、新增共享模块), 修复 `combined_runner` 废弃标注矛盾与脚本编号重复

## v0.11.0 (2026-09-16)
- **SLM 相位生成 raw-only 契约**: 所有 SLM 相位生成函数只产生 **raw 未包裹弧度**，不再自行 `mod 2π`——唯一 wrap 点在驱动 `Santec.create_phase_from_array()` 的弧度→灰度转换 (`santec/driver.py` L1382)。涉及 `optimizer/wfless/slm_square_shaping.py: _freeform_phase_radians` 移除末尾 `np.mod`、`_params_to_gray` 与 `slm_zernike_pib._zernike_to_phase` docstring 同步为 raw-only (`_zernike_phase_radians` 保留 wrapped 输出仅作 test-only 参考实现)
- **相位→灰度统一入口**: `utils/slm/phase_display.phase_to_slm_grayscale(phase, max_grayscale=None, slm=None)` — 传入已打开 SLM 时委托 `slm.create_phase_from_array()`（驱动统一管线：弧度→灰度 + 波前矫正 + LUT + 平移, 2π 灰度取设备波长相关 `_max_gray`）；`slm=None`（纯模拟/离线保存/单测）回退内置纯数学转换 (wrap→scale→clip→uint16, 默认 1023)。`gs_hologram_runner` / `diff_beam_runner` 保存路径迁移为 `phase_to_slm_grayscale(phase, slm=slm)`
- **测试修复**: `test_rms_zernike_runner` / `test_rms_by_zernike` 旧函数名 `optimizer_rms` → `optimizer_rms_slm`（对 `rms_by_zernike.py` 既有重命名的同步，5 个预存 ImportError 修复）；`test_gray_csv` roundtrip 断言改为 **mod-2π 相位等价**（`slm._max_gray` 反向换算 roundtrip 相位）；全套 SLM 相关测试通过：drivers/slm+runners 185 passed / wfless+gui/slm 136 passed（各 1 个硬件 skip）/ optimizer-wf 99 passed

## v0.10.0 (2026-09-13)
- **代码整合 (runners/utils 去重)**: `gs_square_runner`/`diff_beam_runner` 复用的质量指标、SLM 相位下发/槽轮换、超时看门狗、自动曝光、帧记录等辅助逻辑统一迁入 `utils/image/beam_metrics.py`、`utils/slm/phase_display.py`、`utils/image/hardware_utils.py` (原 `algorithm/beam_shaping_utils` 保留为兼容 re-export 层)
- **共享相机工厂**: 新增 `utils/image/hardware_utils.open_camera(camera_type, cam_id, exposure_ms, bit_depth)`，消除 `gs_square_runner`/`diff_shaping_runner` 中字节级重复的 daheng/miicam 初始化代码 (驱动延迟导入，保持 utils 叶子层约束)
- **wfless 内部去重**: `slm_zernike_pib` 的 `_zernike_indices` 改为复用 `slm_square_shaping` 同源实现 (字节级一致性校验通过)
- **回归锚点测试**: 新增 4 个 TDD 锚点测试文件 (`tests/ao_shaping/utils/test_{beam_metrics,phase_display,hardware_utils,targets}.py`, 共 153 例)，锁定全部迁移函数行为；`record_frame` 新增 `include_spot` 参数记录真实 0 级光斑 (argmax) 位置
- **代码评审修复**: ruff 清理、异常类型修正 (如 `ConnectionRefusedError`)、`SimDM` 补齐 `open/close` 接口并对齐 DM registry API；全套测试 1506 passed / 9 failed (仅限 Windows-only WFS/DM SDK 环境绑定用例) / 273 skipped

## v0.9.0 (2026-08-27)
- **全量交替电压工具** (`full-voltage`): 新增 `full_voltage_runner.py`，基于 AsyncMicroDM 异步驱动，所有单元电压同时、均匀地在 0V 与指定电压间交替（无逐通道选择），用于老化/寿命测试
- **AsyncMicroDM 延迟优化**:
  - `TCP_NODELAY`: 连接时禁用 Nagle，消除延迟 ACK 引入的每帧几十 ms 缓冲
  - **内联命令字节快路径**: `prebuild_command()` / `build_frame_commands()` 预编码电压帧为命令字节，`send_bytes()` / `send_frame_commands()` 在热循环中零编码、零分配重放（仅 `write`+`drain`）
  - `send_voltages()` 重构为复用预构建+发送快路径（字节级兼容）
- **full-voltage runner 高实时性设计**: deadline 节拍调度（无相位漂移）、进度输出节流 + 平均下发延迟统计、分片睡眠保证 Ctrl+C 响应 ≤50ms

## v0.8.0 (2026-08-27)
- **1300 陶瓷单元查看器** (`ceramic_viewer.py`):
  - 36×36 网格浏览，点击单元格查看详细信息
  - 原始图像 + 差分图像双列显示
  - **Circle 标注模式**: 在原始图像上绘制圆形标注缺陷区域
  - **Transform 模式**: 移动/调整已绘制圆的位置和大小
  - 标注坐标自动映射回原图尺寸 (支持缩放显示)
  - CSV 导出标注数据
- **Micro-DM 数据文档**: 新增 `data/md_test/README.md` 和 `docs/micro deformable mirror/docs/README.md`，详细说明 1300 单元映射关系
- **streamlit-drawable-canvas 集成**: 安装并修补兼容 Streamlit 1.56+ 的绘图组件

## v0.7.0 (2026-08-27)
- **R50 GUI 全面重构**: 单控制器 / 分组控制 / 全部控制 (联合) 三大 Tab 模块化拆分
  - 单控制器 Tab: 支持单次发送、持续保持、正弦/交替/逐序波形下发
  - 联合控制 Tab: 36×36 矩阵全量编辑、单单元格/行列/矩形批量填充、一键归零下发
  - 分组控制 Tab: 按 wiring map 组别选择控制器并批量下发
  - 侧边栏调试面板集成: 仿真状态指示、指令日志、操作日志（含各控制器 IP 详情）
- **单控制器增强**: 支持在「联合控制」连接模式下选择单个控制器 IP 进行独立操作
- **控制器序号自动映射**: 输入 1-26 序号自动设置 IP (192.168.0.101~126) 和端口 (10101~10126)
- **波形下发优化**: 持续保持 / 正弦 / 交替 / 逐序模式统一按钮状态管理，仿真/正式模式均可正常下发
- **矩阵可视化统一**: 36×36 表格全部加粗逻辑统一，下发后（含归零）所有单元格均为粗体
- **Styler 兼容性修复**: 修复 pandas 2.1+ `Styler.applymap` 移除导致的 AttributeError

## v0.6.0 (2026-08-25)
- **AsyncMicroDM 异步驱动**: 新增 `asyn_micro_dm.py`，基于 asyncio 的高性能异步 TCP 驱动
  - LUT-based 预查表电压转换（`VoltageConverter`），零 GC 稳态运行
  - `AsyncR50Controller` 使用 `asyncio.StreamReader/StreamWriter` 非阻塞 TCP
  - 预分配命令缓冲区，避免帧间内存分配
  - 支持同步/异步双模式使用（`open()`/`close()` 同步桥接到异步内部）
  - 并行控制器通信，独立超时控制
  - 复用 `WiringMap` 接线映射系统
  - 注册为 `"asyn_micro"` 类型，支持 `create_dm()` 工厂创建
- **Micro-DM 逐通道响应分析脚本**: 新增 `scripts/dm/md_img_diff_centroid.py`（FFT 去条纹 → signed diff → 阈值去噪 → 主暗斑质心 → jet 伪彩色渲染）、`scripts/dm/md_img_diff_overlay.py`（逐像素最大值合并分析，输出覆盖率统计）、`scripts/dm/md_img_diff_to_gif.py`（按控制器 IP 将 50 通道 diff 图合成为动画 GIF，帧标注通道号与质心坐标）
- **文档完善**: `scripts/README.md` 新增 Micro-DM Diff Analysis Pipeline 章节，详细阐述 diff 计算与合并分析的算法、处理流程和阈值选取方法（经验阈值 15、主暗斑质心替代全图质心的原因、jet 色标 vmax 归一化）

## v0.5.0 (2026-07)
- **SLM波前误差矫正重构**: 引入独立 `WavefrontCorrection` 类，封装CSV加载、异常点检测（Z-score）、中值滤波剔除和矫正映射图计算
- **SLM open() 方法重构**: 拆分为 `_apply_config_params`, `_load_correction`, `_setup_wavelength` 三个子方法，提升可维护性
- **矫正数据加载优先级**: `__init__` 显式指定 > 配置文件 > 默认路径，支持自定义 `calc_fn` 工厂方法

## v0.4.0 (2026-05)
- **根目录访问重构**: 引入 `ROOT_DIR` 常量，统一项目根目录访问，替代多处 `Path(__file__).resolve().parents[N]` 写法
- **WFS配置管理器简化**: 使用 `PROJECT_ROOT` 简化 WFS 配置管理器初始化
- **ZernikeSLM增强**: 添加 `length` 属性，返回 Zernike 多项式数量
- **文档完善**: 更新 Zernike 响应矩阵标定与闭环控制的文档，新增响应矩阵测试用例
- **编码规范更新**: 在 README 中添加详细的编码指南和最佳实践

## v0.3.0 (2026-05)
- **DM 统一接口重构**: 所有 DM 继承自 `base.DM`，提供标准方法 (V_Min/V_Max, DM_NUM, default_dm_unit_mask, check_dm_unit_grad_safe, send_voltages)
- MicroDM 容错连接: `open()` 允许个别控制器连接失败，记录 warning 后继续
- MicroDM 连接状态检查: `get_connection_status()` 返回 ping 可达性和 TCP 连接状态
- MicroDM 单控制器管理: `connect_controller()` / `disconnect_controller()` / `reconnect_controller()`
- MicroDM 排除控制器: 初始化支持 `exclude_ips` 和 `exclude_ids` 参数
- MicroDM 新增 `ControllerStatus` dataclass 和 `_ping_host()` 静态方法
- PIB 优化器多 DM 支持: `optimize_pib` 接受 `dm` 参数，兼容任意 DM 子类
- 轴向光束优化器 `--dm_type` 参数: 支持 nlight/micro/zernike/hadamard，自动检测在线 DM

## v0.2.0 (2026-03)
- 新增SLM (Santec SLM200) 支持
- 重构相机驱动 (DahengCamera, MIICamera)
- 支持14位相机模式
- 新增U-Net+GAN相位预测训练
- 集成WandB实验跟踪
- 重构仿真模块到 `drivers/sim/`
- 新增Zernike多项式计算工具
- 添加AGENTS.md开发文档

## v0.1.0
- 基础波前优化功能
- PIB优化功能
- 串行流水线优化
- SAC强化学习集成
