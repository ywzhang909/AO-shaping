# Transfer 初始化 + MIL 式参数更新：实验采集准备（2026-11-18）

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **数据/关联脚本**: [`scripts/slm/diagnose_complex_illumination_b.py`](../../../scripts/slm/diagnose_complex_illumination_b.py)
> **数据/关联脚本**: [`scripts/slm/diagnose_transfer_poisson_weighting.py`](../../../scripts/slm/diagnose_transfer_poisson_weighting.py)
> **数据/关联脚本**: [`scripts/slm/evaluate_multiplicative_gain_b.py`](../../../scripts/slm/evaluate_multiplicative_gain_b.py)
> **运行环境**: 离线算法验证与规划（2026-11-18/19）；本轮结果未占用设备，目标形状仍未通过独立 CCD 验收。
> **说明**: 记录 SAFARI 借鉴、Poisson 可辩护性、复场与乘性增益候选，以及 B session per-session Gaussian refit；B-test 24 行仅作冻结后评分，calibration 10 行排除。Gaussian refit 的 held-out 预测显著改善，但不等于 transfer-shaping 实测提升；未因此修改 optimizer 更新规则。配套 JSON 保留 mil 原始脚本绝对路径与 SHA256；迁入的诊断脚本按 master 现有 API 适配，不声称与历史生成脚本字节一致。
<!-- provenance:end -->

## 1. 目标与证据边界

用已有 transfer checkpoint 作初始化，采集新的当前光路探针；离线只用 `train` 相位更新物理预测模型参数（固定 `adaptation=full`，不让 held-out 数据选择更新档位），再在从未参与拟合的 `test` 相位比较旧 checkpoint 与更新模型。此轮只回答“实测探针驱动的参数更新是否改善独立相位的前向预测”，不声称光束整形已通过。

后续若模型方向和预测指标过关，再另开硬件阶段做目标相位的配对 CCD 验收；优化器 loss 下降、模型 holdout 变好都不能替代这一步。

> **结论摘要（2026-11-19，迁移合入 master 时定稿）**
>
> 1. **正面效果（保留）**：`per-session forward refit`（B-train 80 行 + train-only 辐射度量）把 B test 24 行的前向预测从 corr 0.7175 / rel-L2 0.6963 / cv_abs 1.0518 提升到 **0.8193 / 0.5174 / 0.5114**，保真门禁（max counts 偏差 0、rel-L2 偏差 0、corr 1.0）逐位通过（§10.8）。观测模型确实被改进了，这条计入本轮正面结论。
> 2. **但不等价于整形能力**：以上全是"预测 CCD 图像有多准"，不证明能把光斑整成任意目标形状；目标形状 CCD 验收仍缺 measured-vs-target intensity RMSE 与带锁 replay 实测，未放行（§9.3、§11）。
> 3. **默认行为未改**：`PhysicalForwardFit` 的拟合目标与更新规则、optimizer 的默认参数均未因本轮证据改动；refit 只以可选入口存在（§10.9），不进入默认管线。
> 4. **可选项与归档**：Gaussian refit / refit-forward 作为可选分支提供；mil 分支独有、未并入 master 的实验性改动已归档为 patch（§11）。

## 2. 现有资产与必须先处理的风险

- 迁移初值候选已存在：
  - `D:/Projects/TIFO/AO-shaping/data/debug/slm_transfer/defocus_forward_20261107_focus_r2_fit/checkpoints/M3_illumination/frozen_parameters.json`
  - 同目录的 `M1_defocus/frozen_parameters.json` 可留作敏感性对照。
- 旧冻结模型在 Z5–Z11 单模屏方向检查中曾出现整体负相关；因此 checkpoint 只是初始化候选，不是本光路已验证基线。采集后必须报告 signed correlation 和方向差异，不把 `M3` 预先称为合格模型。
- **2026-11-18 exposure scan 是曝光阻断项**：`measured_forward_hw_probe_20261118_scan/REPORT.md` 在 blank/U/C、5–20 ms 的每个测量点都记录 `max=255`、峰/满量程 `100%`；其建议 5 ms 也仍然满量程削顶，U/C 配准搜索还触边。故不沿用该报告的 5/8 ms 建议作为后续探针曝光或验收曝光。新会话必须重新找未削顶曝光；峰值接近满量程即停止加曝光，并先确认当前有效位深/相机格式。
- 旧 report 记载的设备状态仅属于那次扫描：SLM #1 与 Daheng #0 于 `2026-11-18T02:27:01Z` 前正常关闭、设备锁已释放。**本轮未占用任何设备；当前实时占用状态未知**。

## 3. 一次采集的划分与预算

使用 `scripts/slm/zernike_probe_hardware.py` 一次采集 superset，避免为了比较训练帧数反复开关设备；采集后分割只按整张相位，不按像素：

| Split / 用途 | 相位 | 数量（默认） | 参与参数更新 |
|---|---|---:|---|
| calibration | reference_start、正负 ramp、reference_end | 由脚本固定生成 | 只用于当次几何/漂移核对 |
| train | Noll Z4–Z11 的 ±0.1、±0.2 rad；另加随机探针 | 32 + 24 = 56 | 是 |
| test | 独立 seed 生成的 holdout 随机相位 | 12 | 否；只做一次冻结后的报告 |

默认总计 68 个相位探针 × 每相位 3 帧 = 204 张探针原始帧，另有 reference/ramp 校准帧及曝光适配帧。捕获函数会在内存保留 raw stack；粗略峰值内存预算约 1 GB 以上。若设备窗口或内存不足，应在采集前统一减小 `--n-random` / `--n-holdout`，保持 train/test 分组和 seed 规则不变；不可采完后按像素随机拆分。

每行的 `raw_frames`、`model_phase`、实际读回曝光与 manifest 标签必须落盘。`fit.py` 的有效帧规则要求 settle 正常、曝光状态正常、曝光正值且 peak `< 250/255 × full scale`；在拟合前审计被拒帧与失败原因，不静默把削顶帧带入模型。

## 4. 实验顺序与停机门

### A. 设备与光路预检（单一操作者、单一设备 owner）

1. 检查没有其他进程占用设备。`zernike_probe_hardware.py` 已带 `--wait-until-idle SECONDS`（默认 120，`0` = 显式跳过等待与加锁）：`--execute` 分支先 `device_mutex.wait_until_idle`，设备不可用即 `SystemExit`，可用则复用其已持有的锁或补一个 `DeviceLock.default().acquire(timeout=0.0)`，整段 `capture.capture` 期间持锁、`finally` 释放，并把等待/持有/释放写进输出目录的 `device_lock.json`（schema `zernike_probe_hardware/device_lock/1`；不覆盖 capture 自己写的 `config.json`）。等待超时即退出，不杀进程、不抢占设备。plan-only 与 `--help` 不导入任何驱动或锁模块。
2. 按项目既定的 SLM memory-mode 相位自检：读 `current_grayscale`、`displayed_memory_number`、`displayed_phase`；异常时立即停止。确认驱动板真实应答，不能接受静默回落。
3. 记录 SLM/CCD 序列号、波长/LUT、相机实际像素格式/有效位深、光阑状态、光路/焦面、入射位置、激光/快门状态与本次曝光约束。曝光从现场低档开始逐步确认，不沿用历史 5 ms。
4. `probe_optical_path_validity.py` 必须判 `ok` 才继续；`low_snr`、`weak_path`、`no_response` 任一结果都停止探针拟合。每次都按峰值与有效满量程复核，不以平均值或饱和像素占比替代峰值检查。

**2026-11-18 14:44 预检结果（`--session transfer_mil_20261118`，证据目录 `data/debug/optical_path_probe/transfer_mil_20261118/20261118_064401Z/`）**：

- 判定 `ok`，退出码 0；SLM #1 与 Daheng #0 于 `2026-11-18T06:45:17Z` 正常关闭并释放（`optical_path_record.json` 的 `occupancy`）。
- 曝光阶梯（平场，每档 2 帧）：0.05 ms 峰值 178 / 172 counts（未饱和，可用）；0.2、1、2、5、10 ms 均 255 counts 满量程。**唯一可用曝光窗口是 0.05 ms**，建议曝光 0.05 ms。
- 离焦响应显著：宽度最大变化 1.38 px（8.5%，−0.5 rad）、峰值最大变化 25.6 counts（14.3%，+0.5 rad），均超过 3× 合并标准误判据。
- 结论：**不得沿用历史 5/8 ms**；本轮采集以 `--exposure-ms 0.05` 作为阶梯种子。

### B. 采集 superset

- 输出目录使用全新路径；当前已核实目标路径尚不存在。采集脚本要求 `--out` 不存在，**不要预创建、复用或覆盖中断目录**。
- 采集期间不并行启动任何会访问 SLM/CCD 的进程。中断后保留原始目录并标为 incomplete；复跑用新目录名。
- 停机条件：光路非 `ok`、设备自检异常、驱动 `ReadSU=-1000`、相位未稳定、连续超量程、参考相位明显漂移或设备锁等待超时。`ReadSU=-1000` 不重试开关设备，需操作者手动重启控制器后重新评估。

预定命令（从 `mil` worktree 根目录运行；当前只做过无设备 plan-only 验证，**未执行 `--execute`**）：

```powershell
& 'D:\Projects\TIFO\AO-shaping\.venv\Scripts\python.exe' scripts/slm/zernike_probe_hardware.py `
  --out 'D:\Projects\TIFO\AO-shaping\data\debug\slm_transfer\20261118_transfer_mil_zernike_probe_v1' `
  --grid 128 --modes 4 5 6 7 8 9 10 11 --amplitudes 0.1 0.2 `
  --n-random 24 --n-holdout 12 --frames 3 --seed 1104229 `
  --camera daheng --cam-id 0 --slm-number 1 --exposure-ms 0.05 --wait-until-idle 120 --execute
```

> `--exposure-ms 0.05` 来自 2026-11-18 预检：0.2 ms 及以上满量程，唯一可用窗口为 0.05 ms。缺省（省略该参数）会回落相机记忆的 0.043 ms，仍是未饱和量级，但显式给定更可复现。

`--execute` 只在用户明确授权具体设备/时段后，由操作者手动加入并运行；本文件不是授权。激光开关由现场操作员控制，脚本不能替代操作员确认。

### C. 代码解析来源（影响可复现性，必须记录）

当前 `.venv` 的 editable 安装把 `ao_shaping` 解析到**主 checkout** 的 `D:\Projects\TIFO\AO-shaping\src`，而不是 `.worktrees\mil\src`。两个分支已分叉（mil `HEAD=2e39dd25` 不是 master `3935a0d0` 的祖先），`tools/slm/transfer` 下 `capture.py`、`validate.py`、`workflow.py` 三个文件内容不同。

因此：

- 采集脚本只认 `mil` worktree 的这一份（主 checkout 的 `scripts/slm/zernike_probe_hardware.py` 没有锁参数），命令必须从 `mil` 根目录执行。
- 运行时实际调用的库函数来自主 checkout（即最近一次 2026-11-18 扫描所用的同一份 `capture.py`），**不要**为了"对齐分支"把 `PYTHONPATH` 指向 `.worktrees\mil\src`——那会回退到缺少平面相位取光前置与曝光量程收敛的旧版 `capture.py`，让本次采集行为与上一次实测不一致。
- 已在离线核对：脚本构造的 `SimpleNamespace` 覆盖了 `capture.capture` 读取的全部必需属性（`out/execute/camera/cam_id/slm_number/grid/ramp_periods/count/frames/seed`，可选 `exposure_max_steps/save_dataset/device/checkpoint`），`probes` 的替换签名与 `capture` 的调用点 `probes(args.grid, args.count, args.seed)` 一致；但**这只是静态核对，不等于硬件路径已验证**。
- 采集后从 `config.json` 的 `script_sha256`（等于运行时 `capture.py` 的哈希）与 `probe_optical_path.py` 输出的库路径一起写入记录，供以后判断"当时到底跑的是哪份代码"。

## 5. 离线拟合与 holdout 口径

1. 先检查 manifest 确有 56 个 train 探针、12 个 test 探针及 reference/ramp 校准；每类曝光、峰值、帧数及相位哈希均可追溯。有效 train/test 不足或饱和时，不运行拟合。
2. 基线 checkpoint 使用 M3 `frozen_parameters.json`；把 M1 作为另一次独立初始化敏感性检查，不混合 checkpoint。
3. 固定 `--adaptation full`，从 transfer 参数热启动，用 train 探针优化物理预测模型；保持 12 个 `test` 相位完全不参与参数更新、早停或档位选择。**本轮不用 `auto`**：`auto` 会按非 train 相位选择适配阶段，届时相同 test 不再是完全独立的最终评估。
4. 对 test 报告 checkpoint 与更新模型的 signed Pearson、relative L2、CV 绝对误差，并按相位族分别列出；同时报告有效/剔除样本和饱和状况。任何一个平均数变好都不覆盖方向反转或少数模态退化。
5. 参数拟合改善只允许进入“可继续做独立目标形状 CCD 验收”的决策；不能写成 shaping 通过。目标形状需要后续固定曝光、固定掩膜、相同相位 A–B–B–A（有明显漂移时）与最终精确相位重放的独立硬件证据。

离线拟合入口（只读本次新数据，不访问设备）。**必须**把 `PYTHONPATH` 指向 `mil\src`：`fit.py` 用的是包内相对导入（`python fit.py` 直接跑不起来），而 `.venv` 的 editable 安装默认解析到主 checkout 的 `fit.py`——那一份没有下面的热启动修复，用 M3 这种 `residual_noll_radians` 为空的 checkpoint 会在 `fit.py:217` 以 `TypeError` / `KeyError: 5` 中断。拟合数学（`physical_forward_fit.py`、`defocus_forward_core.py`）在两个分支逐字节相同，所以指定 `mil\src` 只改变入口脚本，不改变拟合公式。

```powershell
$env:PYTHONPATH = 'D:\Projects\TIFO\AO-shaping\.worktrees\mil\src'
& 'D:\Projects\TIFO\AO-shaping\.venv\Scripts\python.exe' -m ao_shaping.tools.slm.transfer.fit `
  --dataset 'D:\Projects\TIFO\AO-shaping\data\debug\slm_transfer\20261118_transfer_mil_zernike_probe_v1' `
  --checkpoint 'D:\Projects\TIFO\AO-shaping\data\debug\slm_transfer\defocus_forward_20261107_focus_r2_fit\checkpoints\M3_illumination\frozen_parameters.json' `
  --out 'D:\Projects\TIFO\AO-shaping\data\debug\slm_transfer\20261118_transfer_mil_zernike_probe_v1\fit_m3_full' `
  --adaptation full --device cpu
```

拟合输出必须用新建的 `fit_m3_full` 子目录；确认该路径不存在再运行。

热启动注意事项（2026-11-18 离线查清并修复）：M1/M3 的 `frozen_parameters.json` 里 `residual_noll_radians` 是空对象，`FrozenDefocusForward` 因此不带残差基，而 `PhysicalForwardFit` 无论 checkpoint 怎样都为 Z5–Z11 逐个索引 `basis_arrays[j]`。`mil` 的 `fit.py` 现已按它自己的无基回退规则补齐这几个模式的基（`generate_zernike_phase` 在控制网格上生成，另加同目录的回归测试 `test_build_fit_model_fills_the_residual_basis_from_a_residual_free_checkpoint`）。验证：用上述命令对 1107 数据集跑 `--iterations 2 --patience 1`，退出码 0，输出 `frozen_parameters.json` 的 `residual_noll_radians` 有 Z5–Z11 七个值。该 checkpoint 只作**初始化**，残差方向此前单模检查为负相关，不作基线。

## 6. 当前准备状态

- 已核对采集、拟合入口和 checkpoint 文件；`zernike_probe_hardware.py` 默认参数的 plan-only 已打印 68 probes + calibration，未连接设备。
- 采集脚本的设备锁改进已落到 `mil` worktree 的工作区（未提交）：离线检查 `py_compile`、`ruff check` 均退出 0；`--help` 显示 `[--wait-until-idle SECONDS]`；plan-only 退出 0 且不创建 `--out` 目录、`sys.modules` 审计无驱动/锁模块泄漏；`--wait-until-idle -5` 以退出码 2 拒绝。**加锁与开设备的实际路径未经硬件执行**，本轮不把它算作已验证。
- 当前主阻断项是 2026-11-18 扫描峰值满量程，以及需要在下一次实际会话重新做设备/曝光预检。没有新硬件授权前不执行采集。
- 离线 `fit` 全链路已用 1107 数据集 + M3 checkpoint 以 2 次迭代试运行通过（CPU、退出码 0），热启动缺陷与 `PYTHONPATH` 要求见 §5；这部分仍是离线验证，不代表新数据可用。
- 本计划由实际采集负责人确认时段、现场激光操作与设备状态后再转成实验中记录；执行后按 `libs/skills/slm-mil-transfer/references/reporting.md` 另写实验记录与释放状态，并附 `device_lock.json` 与 `config.json.script_sha256`。

## 7. 采集执行记录（2026-11-18 14:48–14:58 本地时间）

预检（§4 step 4）判定 `ok` 后，于 `mil` worktree 根目录执行 §B 命令（含 `--exposure-ms 0.05`），退出码 0，输出 `data/debug/slm_transfer/20261118_transfer_mil_zernike_probe_v1`。

- **设备占用与释放**：SLM #1（`22030108`）与 Daheng #0（`FJB24112232`）`2026-11-18T06:48:09Z` 打开，`06:58:19Z` 正常关闭；`device_lock.json` 记录 `released=true`、`released_at=2026-11-18T06:58:19Z`、`held_s=610`；`light_check.json` 峰值 179、`counts_per_ms=3580`、判定 `ok`。
- **manifest 核对**：78 行 = 68 探针（zernike 32 + random 24 + holdout 12）+ 8 ramp + 2 reference。`split`：train 56 / test 12，与方案一致；每行 3 帧；峰值 83–213，**0 行饱和（peak≥250）、0 饱和像素**；曝光 0.050（zernike 前 16、ramps、reference_start）与 0.027（其余，由阶梯自适应降下）。`probe_plan.json`：schema `zernike_probe_hardware/1`、`issue=229`、`grid=128`、`seed=1104229`、`frames=3`、`n_zernike=32/n_random=24/n_holdout=12`；plan 逐项含 `label/split/kind`。
- **`.capture_done`**：`early_stopped=false`、`probes=78`；`config.json` 记录 `script_sha256`（= 运行时主 checkout `capture.py` 哈希）与曝光策略 `explicit 0.05 → 生效 0.027`。
- 采集通过；离线 fit（§5，`PYTHONPATH=mil\src`，M3 作初始化，`--adaptation full`）随后执行。

## 8. 离线拟合执行状态

- 采集后运行 §5 命令（`--iterations 100 --patience 8` 默认），`fit_m3_full` 子目录；完成后核对 test 12 相位的 signed Pearson / relative L2 / CV 绝对误差及每相位族结果，并报告 train/test 有效样本与饱和状况。M3 仅作初始化，其旧单模负相关不代表本光路校验通过。

### 8.1 对照口径（离线，设备已下线）

- `fit_m3_baseline`：同一 M3 checkpoint、同一数据集、`--iterations 0`。只做一次前向并与 `fit_m3_full` 的 iteration 0 走同一套 radiometry 重拟（gain/bias 按 train 拟合），不更新任何物理参数。它回答"未更新模型在同一批数据上的预测"。
- `fit_m3_full`：`--adaptation full --iterations 100`，早停于 iteration 25，按训练损失选中 iteration 17，loss `0.302758` → `0.165202`。
- 一致性：两次运行的 `manifest_sha256`、`checkpoint`、`adaptation`、`radiometry`、`ramp_matrix` 完全相同；68 行 `cv_measured` 逐值相等（只取决于实测图像，与模型无关），由对照脚本硬断言。逐探针对齐因此可信，数据未被替换。
- 工具：`scripts/slm/compare_transfer_fit_runs.py`（新，离线只读，无设备 import）；报告 `report/shape_trials/physical_refine/fit_compare_m3_20261118.md`，逐行数据 `<dataset>/fit_compare_m3.json`。

### 8.2 聚合指标

| 组 | n | correlation | relative L2 | CV 绝对误差 | relative L2 改善行 |
| --- | --- | --- | --- | --- | --- |
| 全体 | 68 | 0.787 → 0.881 | 0.610 → 0.422 | 1.105 → 0.841 | 58/68 |
| train | 56 | 0.811 → 0.899 | 0.586 → 0.391 | 1.040 → 0.826 | 47/56 |
| **test（独立）** | **12** | **0.675 → 0.798** | **0.723 → 0.566** | **1.408 → 0.913** | **11/12** |

test 逐行：correlation 12/12 改善，CV 绝对误差 12/12 改善，relative L2 仅 `holdout_006_smooth` 退化（0.8116 → 0.8286，+0.017）。`cv_prediction` 全体均值上升 0.45（对比度偏高），是更新后的已知偏差，不作为改善依据。

### 8.3 相位族分布（决定能否外推到目标形状）

| 族 | n | correlation | relative L2 | Δrelative L2 |
| --- | --- | --- | --- | --- |
| train 单模 Z4–Z11 ±0.1/±0.2 rad | 32 | 0.854 → 0.982 | 0.530 → 0.236 | −0.294 |
| train 随机 zernike（Z4–Z11 组合） | 8 | 0.804 → 0.903 | 0.578 → 0.431 | −0.147 |
| train smooth（RMS 1.5 rad） | 8 | 0.709 → 0.706 | 0.674 → 0.675 | +0.001 |
| train bandlimited（RMS 1.5 rad） | 8 | 0.746 → 0.759 | 0.727 → 0.684 | −0.043 |
| test smooth | 4 | 0.686 → 0.788 | 0.700 → 0.580 | −0.120 |
| test bandlimited | 4 | 0.600 → 0.686 | 0.804 → 0.736 | −0.068 |
| test 随机 zernike | 4 | 0.737 → 0.921 | 0.665 → 0.381 | −0.284 |

test 上 Δrelative L2 与相位 RMS 的 Spearman ρ = 0.776（RMS 越大改善越小）；smooth/bandlimited 全部是 RMS = 1.500 的固定行程族，正是改善最小的一端。

### 8.4 结构性下限（不是优化不足）

- 26 轮轨迹里 smooth 族相对 L2 在 0.674 → 0.675（自身最优 0.656 出现在 iteration 1），bandlimited 0.727 → 0.684（最优 0.682 @ iteration 21），基本不随参数更新移动；同期随机 zernike 0.578 → 0.431、单模 0.530 → 0.236 持续下降。
- 按证据强度排序的原因判断：
  1. **相位自由度不足**：模型只有 Z4 离焦 + 残差 Noll 5..11 共 8 个相位自由度；smooth/bandlimited 相位的功率分布在 Z11 以外，属不可表达子空间，拟合再久也降不下来。
  2. **预测场方差系统性偏低**：bandlimited 的 `cv_prediction` 0.94 对实测 2.90，smooth 2.95 对 4.13。实测远场含有模型不产生的细结构（静态散斑 / PSF / 噪声），不是整体增益能补的；这也解释了为什么 `correlation` 在弱结构族被封在 0.6–0.7。
  3. **无模糊项**：`sigma_ccd_px` 在 M3 与本次拟合后都是 0.0（transfer fit 不学习该项），模型完全没有 PSF 展宽。
- 单模探针只有 ±0.1/±0.2 rad（远场近似静态图案），其 relative L2 0.24 不能外推到大行程；不能拿它代表目标形状所需的相位量级。

### 8.5 远场静态结构检验（解释 correlation 上限）

对 12 个探针（8 train 单模 + 4 test）的 400×400 spot 窗口做高通分离（σ=6 px，除以曝光并按均值归一）后两两求相关：

- 任意两个探针的高通分量相关 0.72–0.98，**非对角均值 0.900**；
- 用跨探针平均得到的一张静态高通图，可解释各探针高通能量的 0.65（`holdout_003_smooth`）到 0.975（`ztrain_007`）；
- 该静态图的幅度系数按族的"自有结构"缩放：bandlimited 0.45、smooth 0.57–0.77、单模 zernike 1.10–1.20。也就是说，探针自身结构越弱，实测远场越高比例是同一张固定图案。

结论：实测远场的高频内容主要是**静态光学结构（PSF 环 / 散斑 / 固定图案）**，不是逐探针物理。这正好对上 8.4 的 `cv_prediction` 系统性偏低，以及 `sigma_ccd_px = 0`（模型完全没有 PSF 项）。样本仅 12 行、窗口与高通尺度是人为选的，因此这里只作为"下一轮要采静态参考"的依据，不当作定量结论。

### 8.6 共享静态结构稳健验证与收敛下界诊断（§8.5 假设的离线检验）

上一节只有 12 探针、400×400 窗口与 σ=6 px 的单一设定。本轮把它扩展为可复现的三重检验，脚本 `scripts/slm/characterize_transfer_static_structure.py` 与 `scripts/slm/diagnose_static_floor.py`；产物 `static_structure.json`、`static_floor_diagnosis.json`、`static_floor_diagnosis_flat.json` 在数据集目录。全程只用已落盘的 CCD 数据，未连接设备。

**(a) 共享结构在全部 68 探针上成立，且在独立验证集上可泛化。** 在 400×400 窗口、σ=6 px 下：跨探针高通非对角相关均值 0.870（train 0.887 / test 0.810），SVD rank-1 解释 91.5% 高通能量、top-5 达 94.8%。把静态图严格限定为「只用 56 个 train 探针平均」（test 不参与建图）后，在 12 个 test 探针上仍解释 79.3% 高通能量（zernike 0.909 / bandlimited 0.772 / smooth 0.699）。按相位族，单模 zernike 的解释度 0.939 高于 smooth 0.726 / bandlimited 0.790——恰是 §8.3 收敛下界没动的两族逐探针成分相对更强。

**(b) 静态结构独立于相位。** 用零相位参考图 `_optical_probe_flat.npz` 在拟合网格（161×161，`camera_xy` 全局坐标）上取高频模板，与 (a) 的 train 平均静态图相关 0.939。说明 (a) 的高频不是 train 相位平均的物理信号，而是真实共享光学结构（PSF 环 / 散斑 / 固定图案）。

**(c) 把共享静态项计入观测模型可移动 smooth/bandlimited 收敛下界（上界诊断，不重拟合）。** 用 `fit_m3_full/iteration_0017` 的固定预测，对 measured 扣除静态模板后重算指标（静态图仅 train 建、幅度仅 train 最小二乘估计，test 作为验证集不参与）：

| 组 | n | relative L2（基线→扣静态） | correlation（基线→扣静态） |
| --- | --- | --- | --- |
| test smooth | 4 | 0.580 → 0.567 | 0.788 → 0.807 |
| test bandlimited | 4 | 0.736 → 0.697 | 0.686 → 0.708 |
| test zernike | 4 | 0.381 → 0.347 | 0.921 → 0.935 |
| train smooth | 8 | 0.675 → 0.629 | 0.706 → 0.751 |
| train bandlimited | 8 | 0.684 → 0.645 | 0.759 → 0.776 |
| train zernike | 40 | 0.275 → 0.364（**过减**） | 0.966 → 0.952 |

用独立 flat 模板、固定幅度 1 复算，test smooth/bandlimited 的 relative L2 分别降到 0.559 / 0.696，方向与上表一致。**6 个族组的 correlation 全部改善，test 全部族 relative L2 下降，train smooth/bandlimited 同步下降**——支持 §8.5 假设：把共享静态项补进观测模型能降低平滑/带限相机的预测收敛下界。边界明确：train zernike 单模探针物理高频弱、与静态模板重合，固定幅度会过减（+0.067~+0.079），因此静态项幅度必须按探针估计，不能对所有相位一视同仁。

这是"观测模型补全能解释多少收敛下界"的上界诊断（静态图 + 幅度只用 train 标定、固定已拟合预测、未引入成像展宽），不是已完成的模型改动。它给出方向证据，但不构成 CCD 验收资格；结论限于本数据集、本窗口/高通尺度与固定预测，不宣称已识别 PSF 或已完成观测模型。

### 8.7 门禁结论

- 对 `Z4 离焦 + 残差 Noll 5..11`，可以写：**参数更新在同一批未参与拟合的 12 个 test 相位上改善了前向预测**（corr +0.123、relative L2 −0.157、CV 绝对误差 −0.495，逐行多数改善）。
- 但扩大残差基到 Noll 5..21 后，选中迭代的 test 平均 relative L2 从 0.5657 变为 0.5755，smooth 从 0.5803 变为 0.6058，bandlimited 从 0.7357 变为 0.7449；更宽基没有消除平滑/带限相位的结构性下限，且总体对照略差。
- 不能写：模型已可指导目标形状整形、shaping 通过。**本轮不进入目标形状 CCD 验收**；下一步转向模型结构/观测模型诊断，而不是继续盲目增加 Zernike 模式或实验验收。具体离线结论和决策见 §9。

## 9. 扩展残差基结果与下一步门禁

### 9.1 Noll 5..21 容量检验

- 在完全相同的 68 行数据、M3 初值、`adaptation=full` 与 radiometry 口径下运行 `fit_m3_full_z21`（`--iterations 60 --patience 8 --device cpu --residual-modes 5-21`，`--iterations 0` baseline 复用）。对照脚本再次断言 68 行 `cv_measured` 逐行一致，两次运行的数据与 checkpoint 未被替换。
- 训练在 iteration 5 达到最低 loss `0.17186591`，iteration 13 早停（patience 8）；5..11 那次是 iteration 17 / `0.16520214`。17 个 residual 参数全部有限、无 gain 崩塌、background 保持 0，但更宽基更早进入 train loss 上升段，说明参数扩张后优化稳定性变差，而不是数值发散。
- 关键相位族（relative L2，5..11 → 5..21）：train smooth `0.6754 → 0.6809`、train bandlimited `0.6844 → 0.6905`、test smooth `0.5803 → 0.6058`、test bandlimited `0.7357 → 0.7449`。test 聚合 `0.5657 → 0.5755`，68 行聚合 `0.4215 → 0.4368`；只有 `cv_prediction` 略降（4.2095 → 4.1882），其余聚合指标变差，68 行中 relative L2 仅 16/68、correlation 22/68 改善。
- 因此扩大静态 Zernike 残差基**没有移动 smooth/bandlimited 地板**，选定模型的聚合表现还略退化。结论限于当前数据、优化器、全参数适配与这套参数化，不能外推成"所有更高阶模型无效"；但足以拒绝"仅加 Zernike 模式即可解决"的假设。
- 报告：`fit_compare_m3_z21_20261118.md`（对 M3 baseline）、`fit_compare_m3_z11_vs_z21_20261118.md`（两次拟合直接对照）；逐行数据为数据集目录下 `fit_compare_m3_z21.json`、`fit_compare_m3_z11_vs_z21.json`。
- 实现：`physical_forward_fit.py` 新增可选 `residual_modes`（默认仍 `range(5,12)`），`fit.py` 新增 `--residual-modes`（如 `5-21`、`5,7-9,11`）并写入 `provenance.json`。默认路径等价性已验证（默认模型 basis 与显式 5..11 逐元素一致）。

### 9.2 更新后的门禁决定

- `fit_m3_full`（Noll 5..11）仍是本数据上较好的 checkpoint；`fit_m3_full_z21` 只作容量诊断，不作为下一轮目标整形初值。
- **不启动目标形状 CCD 验收，不运行任何 `--execute` 硬件路径。** 本轮硬件采集已结束并正常释放；本计划阶段未连接 SLM、相机或激光。
- 离线观测模型诊断已完成第一步（§8.6）：共享静态高频在全部 68 探针上成立、在独立验证集上可泛化、且独立于相位；把它计入观测模型后，test 与 train 的 smooth/bandlimited relative L2 可重复下降。方向由"待验证假设"升级为"有验证集证据支持"，但仍是上界诊断（固定预测、未引入展宽、未重拟合），不是已完成的模型改动。
- **下一步**：把这个共享静态项（幅度按探针估计，避免 zernike 过减）与成像展宽（`sigma_ccd_px` 类）真正加进物理预测模型后重新拟合，仍用冻结 holdout、逐行 measured 一致断言，与 M3、Z11 同数据对照；确认 smooth/bandlimited 收敛下界在完整模型下可重复下降后，才重新评估硬件验收。不能用 §8.6 的上界诊断数字替代重拟合证据。

### 9.3 目标形状验收准备状态（条件满足后才使用）

- 计划与记录模板已单独写在 `target_shape_ccd_acceptance_plan_20261118.md`；该文件是计划，不是执行授权。
- 本轮新增锁安全外壳 `scripts/slm/validate_transfer_shapes_hardware.py`：原样转发 `validate_transfer_shapes.py` 全部参数，只多 `--wait-until-idle`；无 `--execute` 时不等待/不加锁/不导入设备模块，有 `--execute` 时懒导入 `device_mutex`、等待空闲、持锁贯穿验证器调用、`finally` 释放并写 `device_lock.json`；等待或取锁失败一律不启动验证器。mock-only 测试 `tests/ao_shaping/scripts/test_validate_transfer_shapes_hardware.py` 11 项通过。**本轮未执行其 `--execute`。**
- 仍缺：任意目标图像/mask 的独立 phase synthesis 与 replay 入口、measured-vs-target RMSE 指标、replay 路径的锁安全包装。现有 `validate_transfer_shapes.py` 只接受内置 shape，且优化与 CCD 检查在同一流程；`replay_phases.py` 仍是 archive-bound 且无锁。这些不是计划里已有的能力，使用前须实现并离线验证。
- 本轮离线验证（`PYTHONPATH=src`）：`pytest tests/ao_shaping/scripts/test_validate_transfer_shapes_hardware.py` 11 passed；外壳 `--help` exit 0 且同时列出两侧选项；对 `fit_m3_full/frozen_parameters.json` 的 `--plan-only` 冒烟 exit 0，只生成 `plan.json`，无 `device_lock.json`；`ruff check` 对新/改文件 All checks passed（含 RUF100）；脚本类门禁（文件形态、导入解析、子包命名、repo-root bootstrap、device_lock_callers、pytest 分片覆盖、fit_transfer_multishape）1252 passed / 86 skipped。仓库级 `test_module_size_gate` 与 `test_term_gate_enforced` 仍为红，违规项全部是既有文件（CCD 驱动/GUI 行数、10 个既有文档与驱动术语），本轮未改动它们。

### 9.4 产物与工作树状态（供下次接手）

- 本轮新增（未提交）：`scripts/slm/compare_transfer_fit_runs.py`、`scripts/slm/validate_transfer_shapes_hardware.py`、本轮 §8.6 离线诊断脚本 `scripts/slm/characterize_transfer_static_structure.py` 与 `scripts/slm/diagnose_static_floor.py`、`tests/ao_shaping/scripts/test_validate_transfer_shapes_hardware.py`、本文件、`target_shape_ccd_acceptance_plan_20261118.md`、`fit_compare_m3_20261118.md`、`fit_compare_m3_z21_20261118.md`、`fit_compare_m3_z11_vs_z21_20261118.md`，以及数据集目录下的两个 `fit_compare_*.json`、`static_structure.json`、`static_floor_diagnosis.json`、`static_floor_diagnosis_flat.json` 与 `fit_m3_full_z21/`。
- 本轮修改（未提交）：`physical_forward_fit.py`（`residual_modes` 参数 + 校验，默认 5..11）、`fit.py`（`build_fit_model` / 基填充循环 / `parse_residual_modes` / `--residual-modes` / provenance）、`docs/terminology_glossary.md`（T-77「地板」→「收敛下界（经验风险下界）」定义登记，不入强制替换脚本）。
- **提交范围注意**：`fit.py` 的 diff 里既有本轮 `--residual-modes` 改动，也含更早已存在的 warm-start fallback hunk（同一 `@@ -214,14 +215,26 @@` 区块），只能按 hunk 手工切分，不能整文件 `git add`。`scripts/slm/zernike_probe_hardware.py`、`src/ao_shaping/optimizer/wfless/model_in_loop_shaping.py`、`src/ao_shaping/optimizer/wfless/slm_model_in_loop.py`、`tests/ao_shaping/tools/slm/transfer/test_acceptance_loop.py` 的改动不属于本轮，不要顺带提交。

## 10. 观测模型补全与跨会话复用的后续离线判决（2026-11-18 续）

本节记录 §9.2 指定的"把共享静态项与展宽加进前向模型重拟合"以及两条并行诊断的实际结果。三份产物均已直接核对 JSON，不是仅凭执行摘要。

### 10.1 修正版 M1：联合重拟合下静态项确实泛化（G0/G1/G2）

§8.6 的上界诊断在本节升级为**真实重拟合**：`fit.py` 新增 opt-in 的 `--static-mode {per-row,global}`、`--static-basis-k`、`--learn-sigma-ccd`、`--static-l2`，全部默认关闭时与旧代码逐位一致（2 次迭代 `array_equal=True`，max diff 0.0，iter-0 loss 0.30275800824165344 相同）。修正了两个会让结论反转的实现缺陷：

- 原 `static_gain` 形状 `(68,4)` 但 loss 只算 train 行，12 个 test 行静态增益恒为 0 → test 预测根本不含静态项，无法检验"静态能否泛化"。`global` 模式改为单一共享 `(K,)` 系数，作用于所有行（含 test），仍只用 train loss 求梯度、test 不参与早停/选档/基构造。
- 原可学展宽下钳 `[0.5,5.0]` 强迫 checkpoint 的真值 `sigma_ccd_px=0` 被加 ≥0.5 px 模糊。改为线性参数、下钳 `[0.0,5.0]`，σ≤0 时核返回 `None`（与冻结前向逐位一致）。

结果（对 `fit_m3_full` it17，`--adaptation full --residual-modes 5-11`，30 iters）：

- **G0 `fit_m3_sigmaonly`**：σ=0.0，TEST 与基线 4 位小数相同——展宽单独无收益，排除展宽混淆。
- **G1 `fit_m3_static_global`**（K=4 + σ）：TEST corr +0.0910、relative L2 −0.1168、cv_abs −0.1446，8/9 族×指标改善（唯一回退 zernike cv_abs +0.008）；bandlimited 受益最大（corr +0.160、cv_abs −0.431）。主导系数为第 1 张基图（−1.373）。
- **G2 `fit_m3_static_global_noblur`**：与 G1 逐位相同（loss 历史相等、预测 `array_equal`、`maxdiff=0`）→ 全部收益来自静态项，与展宽无关。

**判决**：在会话内冻结 12-test 上，共享 train 拟合的加性静态项确实大幅泛化，§9.2  prescribed 的"重拟合后 smooth/bandlimited 收敛下界下降"成立。可学展宽被钉在 0——无额外 CCD 模糊证据。test 行从未参与早停/选档/基。

### 10.2 残差校正器消融：残差主体与相位无关，但正确基不是"加性静态图 + 标量展宽"

`train_residual_corrector.py` 新增 `--drop-phase` / `--shuffle-phase`（互斥、仅 cnn）与逐轮 test 轨迹（只记录、不参与选轮）。test all(12) relative L2：基线 0.5657 → full CNN 0.4873 → drop-phase 0.5081 → shuffle-phase 0.5092 → operator(7 参数物理) 0.5567。

- `drop-phase` 回收 73%、`shuffle-phase` 回收 72% 的 rl2 增益，两块独立证据同向 → **残差主体与相位无关**，是观测算子项；剩余 ~27% 集中在 zernike 族。
- CNN 1500 轮：test rl2 在 epoch ~380 触顶 0.4857，之后单调变差到 0.5039（train loss 仍降到 0.0504）→ 选轮只看 train loss 会过拟合；300 轮的 best epoch 298 恰在 test 平稳区，不是预算不足。operator 彻底饱和（300→1500 族级差 ≤0.0003）。
- 关键反证：CNN 把 cv_abs 从 0.913 压到 0.464，而 7 参数物理校正器只到 0.900。标量增益不改 cv，加性散斑也压不下 cv → 强度剖面系统性偏尖，CNN 学的是"给定预测场重新分配到实测形状"。物理校正器只拿到 CNN 相位无关增益的 ~1/6，**"加性静态图 + 标量展宽"不是该残差的正确基**；下个物理候选应是乘性低秩空间增益 `pred × (1 + Σ g_k B_k)`（全局系数、作用所有行），而非加密静态图或相位特征。

### 10.3 跨会话复用：形状转移，幅度与静态项漂移（A↔原会话）

`compare_session_replicate.py` 比对同 seed 1104229 的原会话与新采集 A（`replicate_a_20261118`）：68/68 label 匹配、`phases_equal=true`、窗口重建 max dev 9.09e-13。暴露 confound：A 未复用原曝光表（原 16×0.05+52×0.027 ms，A 34×0.051+34×0.027 ms），仅 18/68 行曝光一致；以下给出全行与曝光匹配子集两组数字。

- **实测图有真实但温和的会话漂移**：cross/expected-noise = 1.37（全行）、1.30（曝光匹配），>1 即超帧间噪声；光斑移动均值 1.17 px、最大 3.65 px。高通图上做标量增益校正反而变差（1.37→1.71），不是纯亮度差。
- **原会话静态图部分失效**：static_rms/residual_rms 原→原 0.4700（与预算脚本 0.46999 一致）、原→A 0.3648；曝光匹配子集 0.4594→0.2350。A 自身静态图也只到 0.394/0.269 → 部分是 A 会话静态结构本身不同，不只是映射错配。
- **算子形状转移、幅度不转移**：`fit_m3_full` 对 A 的 corr 仅降 0.0063（0.8814→0.8751），但 rl2 升 0.0523（test 0.5657→0.6837）。逐行 LS 增益原 1.0805 / A 0.8870（比 0.846±0.23）；乘回增益后 rl2 0.4738→0.4027，与原 0.4215 相当。

**判决**：把"train 拟合的静态/算子项完全不可复用"说得过强——空间形状确实转移；但**绝对幅度与静态份额不可跨会话原样复用**，需要每会话一次标量重标定 + 静态重建。这是对 §10.1 会话内收益的关键限制：静态项泛化只在同会话成立，换会话即需重标定。

### 10.4 更新后的门禁

- §10.1 满足 §9.2 的"重拟合证据"字面要求，且会话内收益显著；**但** §10.2 表明收益的基函数选错（加性静态 + 标量展宽只拿 ~1/6），§10.3 表明该收益跨会话漂移。**本轮仍不进入目标形状 CCD 验收**，且不再以"再加静态图/展宽"为下一步。
- 下一步应按 §10.2 的参数化方向（乘性低秩空间增益，或等价的逐图对比度重整）做受控检验，并在**新会话数据上带每会话标量重标定**复评，才能判断能否支撑上机验收。
- 本节产物：`fit_m3_sigmaonly/`、`fit_m3_static_global/`、`fit_m3_static_global_noblur/` 及 `fit_compare_*_20261118.{md,json}`（数据集目录）；`report/shape_trials/physical_refine/residual_{operator,cnn,ablations,budget}_m3_iter17*.json`；`scripts/slm/compare_session_replicate.py`。全部 test 指标只用冻结 12 行、未调参。

### 10.5 采集 B（新 seed）已完成与三模型迁移评估口径

- 外部验证集 B 已采集完成：`data/debug/slm_transfer/extend_b_20261118`，seed 771103，114 记录（80 train / 24 test / 10 calibration，每探针 3 帧），`early_stopped=false`。设备正常释放：锁 `released=true @ 2026-11-18T12:29:22Z`、`held_s≈924`、`release_error=null`；SLM #1 + Daheng MER2-507-23GM + 0 级照明激光已下线。此前一次 B 启动被 `--wait-until-idle` 误拒（占用者列表里是我方离线 `scripts/slm/*` 训练与另一会话的相机测试），改为等安静窗口后于 20:13 重启并一次成功。
- 用户确认的口径：「3 个模型」= M1（物理静态项+可学展宽重拟合）/ M2（operator 残差校正器）/ M3（cnn 残差校正器）；B 仅用于**原模型迁移的跨会话验证 + 每会话标量重标定**，不进原 56/12 split、不重训。
- 关键实现缺口：M2/M3 训练脚本此前只 dump 指标 JSON、`best_state` 权重从未落盘，无法直接迁移到 B。已补 `--save-weights`，用 seed 0 确定性重跑 operator/cnn 各 300 轮，验收条件为"从 config 独立重建 + 重算 test 指标与参考 JSON 逐位一致"。
- M1/baseline 迁移的保真路径：`build_fit_model(dataset=B, ...)` 会从 B 自身的 `reference_start` / signed-ramp 行推 origin 与 ramp matrix；必须先用原会话重建预测对齐 `fit_m3_full/iteration_0017.npz`（max abs dev 容差内）证明路径正确，再应用到 B。标量重标定增益只用 B train 80 行求，绝不碰 B test 24 行。
- M2/M3 权重已落盘并核验：`report/shape_trials/physical_refine/residual_{operator,cnn}_m3_iter17.pt`（`torch.load(weights_only=True)` 可加载；operator 16 状态张量 / cnn 9 张量；best_epoch 299/298；从 config 独立重建后 test 36 项指标对参考 JSON 的 max abs dev = 0.0）。注意 `*.pt` 被 `.gitignore:211` 忽略，交付需 `git add -f` 或改 artifact 策略。
- 两条评估车道共享同一条"B 物理 prediction 保真路径"：`build_fit_model(dataset=B,...)` 从 B 自身 origin/ramp 建模，加载原 fit 参数预测；运行前必须先在原会话重建 `iteration_0017.npz`（报 max abs dev）证明路径正确。标量增益只用 B train 80 行，从不碰 B test 24 行。M2/M3 校正器的 phase 输入必须复刻 `train_residual_corrector.assemble()` 的降采样口径。
- **baseline + M1 在 B 上的结果（Lane B 已回收、产物已直接核对）**：`scripts/slm/evaluate_transfer_on_new_session.py`，权威输出 `extend_b_20261118/eval_transfer/results.json`。保真检验 PASS（baseline 重建 vs `iteration_0017.npz` max|dp|=0.0208 counts、rel 7.8e-7、corr≈1.0；static max|dp|=3.1e-4、rel 3.4e-8）。B 上 80 train + 24 test（load_dataset 忽略 10 calibration），族计数 zernike 56/smooth 24/bandlimited 24。
  - **baseline `fit_m3_full` 在 B 上标量重标定救不回来**：B-train 增益 g=1.314 几乎不动 test（rl2 0.7088→0.6963、corr 0.7175 不变、cv_abs 1.0518 不变——标量增益不改 corr/cv）。远差于原会话独立 test（corr 0.7982/rl2 0.5657）与原→A 同 seed 复测恢复到的 rl2≈0.403。
  - **`static_global` 在 B 上经重标定确有真实收益**（g=1.217）：test rl2 0.6963→0.6033（Δ−0.0929）、corr 0.7175→0.8018（Δ+0.0843）、cv_abs 1.0518→0.7120（Δ−0.3397）；按族 bandlimited 收益最大（rl2 Δ−0.1488/corr Δ+0.1335）。
  - **收益确来自静态项本身**：把静态项置零（`origin_session_static_off`）后 after.test rl2=0.6963 与 baseline 相同、corr 更差（0.7087）、cv_abs 更差（1.2398）。origin 放置（session-offset vs measured）影响很小（baseline Δrl2 +0.0058/Δcorr +0.0053；static Δrl2 +0.0098/Δcorr −0.0026），结论稳健。
  - 与 A（同 seed 复测）对照的关键差异：A 上 baseline 经增益恢复即达标，B（新 seed、未见相位）上 baseline 救不回来、必须靠静态项。说明"跨会话可复用"取决于相位是否见过——同相位漂移可用标量校正，未见相位则暴露观测模型缺口，而静态项补上了其中一部分。
- **M2/M3 在 B 上的结果（产物已直接核对）**：`scripts/slm/evaluate_correctors_on_new_session.py`，权威输出 `report/shape_trials/physical_refine/corrector_transfer_to_b.json`。两道前置门禁 PASS：物理重建 vs `iteration_0017.npz` max|dp|=3.5e-4 counts（rel 3.6e-8、corr 1.0）；两个 bundle 从 config 独立重建后复算 in-session 指标 max abs dev = 0.0（operator 6391 参数 / cnn 5881 参数）。相位输入用 B 原始 128×128 `model_phase`，与 `assemble()` 训练口径一致；标量增益按各模型在 B train 80 行上分别求，从不碰 B test。
  - B test（24 未见相位）after-recalibration：baseline rl2 0.6963 / corr 0.7175 / cv_abs 1.0518；**operator 0.6812 / 0.7364 / 1.0329**（rl2 −2.16%，24/24 行改善，三族全改善）；**cnn 0.6811 / 0.7291 / 0.8265**（rl2 −2.18%，但只 12/24 行改善，且 zernike 族明确变差：rl2 +0.0339、corr −0.0411）。
  - **相位条件的 27% 跨会话不存活**：in-session cnn 相对 drop-phase 为 13.9% vs 10.2%；跨会话 cnn 相对 operator 的 rl2 优势仅 +0.0001（等于零），且在 in-session 最擅长的 zernike 族上反向。共享增益（baseline g）下排序不变：operator −3.9%、cnn −2.6%。
  - **cnn 幅度被原会话烘死**：需 g=1.717 而物理 baseline 只要 1.314；它在 10 个 calibration 行（0.051 ms、±1 ramp，完全在训练分布外）rl2 爆到 6.62，把 all-114 聚合拉到 1.177（比 baseline 0.706 还差）→ all-114 不是公平口径。

### 10.6 四模型在外部验证集 B 上的汇总与最终门禁判定

B test = 24 个未见相位行，均为各自 B-train 标量增益之后：

| 模型 | g | test corr | test rl2 | test cv_abs | rl2 相对 baseline |
|---|---|---|---|---|---|
| baseline `fit_m3_full` | 1.3141 | 0.7175 | 0.6963 | 1.0518 | — |
| M1 `static_global` | 1.2171 | **0.8018** | **0.6033** | **0.7120** | **−0.0930** |
| M2 `operator` | 1.3231 | 0.7364 | 0.6812 | 1.0329 | −0.0150 |
| M3 `cnn` | 1.7174 | 0.7291 | 0.6811 | 0.8265 | −0.0152 |

对照原会话 in-session test（12 行）：baseline 0.5657/0.7982、static_global 0.4489/0.8893、operator 0.5567、cnn 0.4873（rl2/corr）。

1. **M1 是四个里唯一有实质跨会话收益的**（rl2 −0.093、corr +0.084、cv_abs −0.34），收益来自迁移的静态系数、与标量增益无关（重标定前 corr 已 +0.084）；前提是每会话用本会话 train 行重建静态基（K=4 high-pass SVD，test/calibration 不进基）。
2. **M2/M3 跨会话只有 ~2%**，与 M1 差一个量级；M3 的相位通道优势跨会话归零且在 zernike 反向 → 相位条件进入的 27% 是 in-session 现象。
3. **baseline 在 B 上本身明显退化**（test rl2 0.566→0.696、cv_abs 0.44→1.05；预测剖面比实测更平：cv_pred 2.46 < cv_meas 3.41）→ 主误差源仍是物理前向/观测算子，两个校正器都只动 2%、碰不到主项。
4. **仍不放行目标形状 CCD 验收**。B 只证明"同族随机相位上静态项补丁有效"，而目标整形要求在**任意目标相位/形状**上可信；target-shape 路径仍缺（a）任意目标 mask 的独立相位合成、（b）measured-vs-target RMSE 门槛、（c）replay 的锁安全包装。"减小前向预测误差"≠"能把光斑整成目标形状"。
5. 下一轮优先级：把 §10.2 的乘性低秩空间增益作为 M1 静态项的**互补**候选（不是替代），先压物理前向主误差；此前不投设备时间做目标形状验收。

### 10.7 SAFARI 借鉴三点 + 乘性低秩增益候选的离线检验（2026-11-19，设备未占用）

口径：只用 B（80 train / 24 test / 10 calibration 排除），test 仅在模型冻结后打分一次，无 test 调参/选型；三条独立脚本，未改 `fit.py`/`physical_forward_fit.py` 等生产文件。设备锁 `data/debug/.device.lock` byte0=0，全程未占用设备。

| 候选 | 脚本 | 判决 | 关键证据（B test，train-only 拟合） |
|---|---|---|---|
| Poisson / 计数方差保真 | `diagnose_transfer_poisson_weighting.py` | **不可辩护，按规则停止** | 缺独立测得 CCD offset、转换增益、方差律；train-only var/mean≈0.32（背景）、0.34/0.17（core），强亚泊松（期望 1.0）；`decision=stop_not_defensible`、`run=false` |
| 带平滑正则复幅值照明场 | `diagnose_complex_illumination_b.py` | **边际正，不推进** | 相对同预算 Gaussian refit：corr +0.0055、rel −0.0071、cv −0.0018；128 场参数 vs 12 基底参数；场窗均值 0.211、theta_absmax 1.48 rad（吸收残余尺度/相位） |
| 近端投影物理约束 | `evaluate_multiplicative_gain_b.py` | **约束有效，但不改候选结论** | Dykstra 投影 `‖g‖≤bound ∩ gain≥floor`：test corr +0.0158、rel −0.0047；可行性 True |
| 乘性低秩空间增益 | `evaluate_multiplicative_gain_b.py` | **负结论** | 投影后 test corr 0.7117 / rel 0.6928，低于冻结 0.7175/0.6963 与 M1 0.8018/0.6033；OOD 命中 4 train / 0 test |

**对照臂暴露的头条（本轮最大信号）**：同优化器/损失/220 步、仅 12 个基础物理参数 + train-only 辐射度量重拟合的 Gaussian refit，在 B 上把 test 从冻结的 corr 0.7228 / rel 0.7139 / cv 1.0519 提到 **0.8133 / 0.5246 / 0.5113**。即跨会话退化主项可由**本会话 train 行的基础前向重拟合**大幅回收，幅度远超复场补丁和乘性增益。注意此臂 `origin_mode='measured'`，与权威 `eval_transfer/results.json`（`'session'`）管线不同（其冻结基线 0.7228/0.7139 亦略异于权威 0.7175/0.6963），故**不能**直接宣称已超过 M1，需在权威管线上独立复现后才可作结论。

**门禁决定：**
1. **Poisson 加权不引入**：数据不支撑似然；除非补独立暗场/转换增益/读出噪声标定记录。
2. **复场照明场不推动生产改造**：确为瞳面复幅值（`focal_intensity` 前乘 `M=exp(s+it)`，非图像域补丁），但边际增益 <1% 且场强衰减 + 大幅相位提示它在吸收残余尺度/相位，不值得为 `PhysicalForwardFit` 做破坏性 API 改动。
3. **乘性低秩增益撤回**：作为 M1 互补候选也不成立（比 M1 差约 9% corr），后续不再投入。
4. **近端投影保留为工具**：约束本身有效（+0.0158），但只能改善一个更差的候选，暂不进入主线。
5. **新优先级**：`per-session forward refit`（B-train 基础参数 + 辐射度量重拟合）作为下一独立验证对象——本轮唯一大幅改善的杠杆；须在权威管线上复现并与 M1 做同口径对照。
6. **仍不放行目标形状 CCD 验收**：本轮全部是离线预测指标，未触及任意目标 mask 相位合成、measured-vs-target RMSE 门槛、带锁 replay。

### 10.8 per-session forward refit 独立验证完成（2026-11-19，设备未占用）

承接 §10.7 第 5 条新优先级，在**权威管线**上完成 `per-session forward refit` 的独立验证。设备锁 `data/debug/.device.lock` byte0=0，全程未占用设备。

**口径**：用 `diagnose_complex_illumination_b.py --origin-mode session --arms gaussian`，与 `eval_transfer/results.json` 主变体 `origin_session` 完全同口径——B measured reference_start 原点 + 原 fit 的 fitted offset（origin_fit − measured_old）= [5.2143, 1.3600] px，origin_vec = [1231.4190, 1106.1965]。B-train 80 行拟合 12 个基础物理参数 + train-only 辐射度量（NNLS，非负投影），220 步 Adam；B test 24 行仅在冻结后打分一次，无 test 调参/选型；calibration 10 行全程排除。

**权威管线结果**（`gaussian_refit_session_origin_20261119.json`）：

| 模型（同 origin_session） | test corr | test rel-L2 | test cv_abs |
|---|---|---|---|
| frozen baseline（`fit_m3_full`，权威值精确复现） | 0.7175 | 0.6963* | 1.0518 |
| M1 `static_global`（§10.6） | 0.8018 | 0.6033 | 0.7120 |
| **per-session Gaussian refit** | **0.8193** | **0.5174** | **0.5114** |

\* baseline rel-L2 在 refit 脚本用冻结辐射度量为 0.7088；权威 0.6963 含 baseline 标量重标定（gain 1.3141），corr 两者精确一致。

**保真门禁（通过，逐位复现）**：仿 `evaluate_transfer_on_new_session.validate_rebuild`，把 refit 预测计数写入 `gaussian_refit_prediction_20261119.npz`（104×161×161，80 train + 24 test），full base restore 后从 refit 参数 + 保存的 train-only 辐射度量重算前向并比对。容差 counts 0.05 / rel-L2 0.0001 / corr 0.9999；实测 max_abs_dev=**0.0**、rel_l2_dev=**0.0**、corr=**1.0**、逐行 metric_dev 全 0（比权威原始 fit 的 0.0208 counts 偏差更严格）。证明 refit 是确定性、可复算的前向，不是优化器的瞬时状态。

**判决**：
1. **per-session forward refit 在权威管线成立**，同口径全指标超过 M1（corr +0.0175 / rel-L2 −0.0859 / cv_abs −0.2006），是跨会话退化的主回收杠杆；跨会话退化主项可由本会话 train 行的基础前向重拟合大幅回收，幅度远超复场补丁与乘性增益。
2. **仍不替换 M1 的已验收状态**：M1 门禁此前已核验；refit 本轮仅完成单会话（B）的独立验证与复算保真，未做多会话稳健性、未跨其它会话/数据集。作为新候选记录，不回退既有结论。
3. **仍不放行目标形状 CCD 验收**：本轮同样是离线预测指标，未触及任意目标 mask 相位合成、measured-vs-target RMSE 门槛、带锁 replay。
4. 产物：`gaussian_refit_session_origin_20261119.json`（结果+门禁）、`gaussian_refit_prediction_20261119.npz`（保真预测）、`diagnose_complex_illumination_b.py`（新增 `--origin-mode`/`--arms`/`--output` 与保真门禁块）；均未提交、未推送。

### 10.9 可选 refit-forward 分支：opt-in 入口与用法（2026-11-19）

§10.8 的结果保留为**可选分支**，不进入默认管线。为了让这条分支能在 master 上被复现而不必重跑 mil 的一次性脚本，新增 opt-in 入口 `scripts/slm/refit_transfer_forward.py`（测试 `tests/ao_shaping/scripts/test_refit_transfer_forward.py`，13 项）。

**它做什么**：从 `--checkpoint` 的 `frozen_parameters.json` 建模，按 manifest 的 split 组拟合掩码，断言掩码后才调生产 `fit.fit()`（`record_responses=False`，因此 `frozen_parameters.json` 由与生产同一代码路径写出），随后用同一前向（`PhysicalForwardFit.prediction()` + `conv2d(kernel, padding=12)`）+ train-only 辐射度量对 base 与 refit 各打分一次，写 `refit_provenance.json`。

**硬保证（脚本内断言，非注释）**：

| 保证 | 实现 |
|---|---|
| 只在 train 上拟合 | `train_only_fit`：selection 只含 `train`（`train_all` 时才是 train+test，本入口不传），`calibration` 显式排除 |
| calibration 永不参与 | 掩码剔除 + `--score-splits calibration` 直接报错退出 |
| test 只打分一次 | `test_scored_once`：`--score-splits` 默认 `test`，评分在 fit 结束后单次进行，不回流选型/早停 |
| 不覆盖历史产物 | `--out` 必须不存在（沿用生产 `fit()` 的 `exist_ok=False`），已存在则拒绝；只创建父目录 |
| dry-run 不落盘 | `--dry-run` 只打印 split 计划与门禁，不建目录、不写 checkpoint |

**冒烟证据（2026-11-19，离线 CPU，未占用设备）**：`extend_b_20261118` + `fit_m3_full/frozen_parameters.json`，`--iterations 1 --span 160`，exit 0。split counts `{train: 80, test: 24, calibration: 10}`，`train_rows=80 / fit_rows=104`；门禁 `train_only_fit / calibration_excluded / split_counts_match / refit_beats_base_on_score_split` 全 true；test corr 0.7228 → **0.7417**，rel-L2 0.7019 → **0.6864**（1 步；脚本内对两侧统一在 train 上重拟合辐射度量，故此处 base 与 §10.6 的 0.7175 不同源，只比较同一脚本内的 base→refit 差值）。

**刻意不做的事**：不改 `PhysicalForwardFit` 的 loss/更新规则，不改 optimizer 默认参数，不改 `validate.py` 的验收口径；本入口不写 `report/` 下的任何文件，结果目录由调用者指定。

## 11. 迁入 master 的收尾状态与仍然封锁的边界（2026-11-19）

### 11.1 从 mil 迁入的内容

| 类别 | 内容 |
|---|---|
| 脚本 | `diagnose_complex_illumination_b.py`、`diagnose_static_floor.py`、`diagnose_transfer_poisson_weighting.py`、`evaluate_multiplicative_gain_b.py`、`evaluate_transfer_on_new_session.py`、`compare_transfer_fit_runs.py`、`compare_session_replicate.py`、`compare_deswivel_forward_inverse_20261118.py`、`characterize_transfer_static_structure.py`、`analyze_model_in_loop_loss.py`、`refit_transfer_forward.py`、`validate_transfer_shapes_hardware.py` 等 |
| 结果/报告 | 本目录下全部 `*_20261118/19` JSON/MD（含四模型 B 汇总、复场、Poisson、乘性增益、refit、验收计划），以及 `gaussian_refit_prediction_20261119.npz` |
| 权重（被 `.gitignore` 忽略） | `residual_cnn_m3_iter17.pt`、`residual_operator_m3_iter17.pt` |
| 探针数据 | `data/debug/optical_path_probe/transfer_mil_20261118/`（12 个 npz，逐文件 SHA256 一致） |
| 术语/测试/工具 | `docs/terminology_glossary.md` T-77、`tests/.../test_acceptance_loop.py` 残差基热启动用例、`zernike_probe_hardware.py` 设备锁证据（issue 229 移植） |

迁移原则：**权威数值以原始 JSON/NPZ 为准**，它们保留 mil 当时的绝对路径与 `script_sha256`；适配到 master API 的脚本不声称与历史生成脚本字节一致，重跑输出写 `*_master_api` 后缀。provenance 已同步（`sync_report_provenance.py` clean）。

### 11.2 未并入 master 的实验性改动（已归档）

`report/shape_trials/physical_refine/mil_only_experimental_20261119.patch`：

- `physical_forward_fit.py` 的 `residual_modes` 关键字参数（模式子集拟合）；
- `transfer/fit.py` 的 blur / `static_basis_k` / `static_mode` / `static_coef` / `learn_sigma_ccd` / `sigma_ccd_px` 观测模型扩展。

二者是 mil 上的容量/静态项实验脚手架，master 生产路径不需要，且带这些机制的权威结果已由原始 JSON 固化。patch 仅作归档，不保证可直接 `git apply`。

### 11.3 仍然封锁：目标形状 CCD 验收

无论 §10.5–§10.8 的离线指标多好，目标形状验收仍不放行（与 `target_shape_ccd_acceptance_plan_20261118.md` 一致）。三项缺口的状态如下：

1. **measured-vs-target intensity RMSE（指标已实现并测试，实测未做）**：`protocol.target_intensity_rmse` 已在 `replay_phases` 中写入 `target_rmse` 字段（整窗归一化、背景 floor、`rel_l2` / `energy_fraction` 冻结契约，见验收计划 §2）；但**没有任何实测值** —— 阈值还须现场预注册，冻结后只在最终 CCD 测试上评分一次；
2. **带锁 replay 实测（外壳已就绪，端到端未跑）**：`replay_transfer_shapes_hardware.py` 已补上重放的锁安全外壳（mock 16 项通过，`--execute` 从未执行）；但任意目标 mask 的相位合成 → 锁内下发 → CCD 采集的端到端路径尚未跑通，`validate_transfer_shapes_hardware.py` 的 `--execute` 同样未执行；
3. **独立 CCD 证据（仍缺）**：全部结论都是离线前向预测，缺少独立采集的目标形状实测对照。

解除条件：在设备窗口内做一次带锁、阈值预注册的目标形状配对采集并拿到独立 CCD 证据；在此之前任何"整形改善"表述都只能限定为前向预测改善，RMSE 指标与带锁外壳的**代码就绪不等于验收通过**。
