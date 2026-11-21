# 物理预测模型保真度阶梯 · 产物占位 (等待下一次设备会话)

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **数据/关联脚本**: [`scripts/slm/_hw_fidelity_ladder_sections.py`](../../../../../scripts/slm/_hw_fidelity_ladder_sections.py)
> **运行环境**: 人工撰写（阶梯目录的导读与运行手册; 数据产物由 ladder_summary.md 注册项覆盖）
> **说明**: 物理预测模型保真度阶梯的产物目录 (占位): 脚本在 SLM #1 上依次上屏多个相位图案, 采集 CCD 远场, 逐图案对比模型预测与实测, 用于判定怀疑的缺失离焦项。整个阶梯只开/关一次 SLM + 相机 (背靠背开关会触发驱动冻结), 设备占用记入 device_log.json。目前仅以 --dry-run 离线演练过 (合成相位 + 占位帧), 尚未执行 --execute, 因此本目录还没有硬件结果, 不伪造数字; 运行手册与判据优先级见正文。主判据为信号区秩相关 spearman_signal, 不使用全域 spearman_coverage 下结论
<!-- provenance:end -->

**状态：占位。** 本目录目前**没有任何硬件结果**，只有这份说明。

生成脚本：[`scripts/slm/hw_forward_model_fidelity_ladder.py`](../../../../../scripts/slm/hw_forward_model_fidelity_ladder.py)
（纯数值分段在 [`scripts/slm/_hw_fidelity_ladder_sections.py`](../../../../../scripts/slm/_hw_fidelity_ladder_sections.py)）。

脚本只在 **`--execute` 真上硬件**时才把关键产物复制进本目录；`--dry-run` 的产物
留在运行目录（默认 `data/debug/differentiable_opt/fidelity_ladder_<timestamp>`），
**不会**进这里。截至本次提交，还没有跑过 `--execute`，所以这里是空的 ——
不伪造结果。

## 这个阶梯要回答什么

物理预测模型 `fftshift(fft2(A·e^{iφ}))` 与真实光路的远场差多少，以及怀疑的
**missing defocus term**（模型少一项离焦/菲涅尔相位）是否成立。

判据（按优先级）：

1. `spearman_signal` — 信号区（覆盖掩膜 ∩ 预测强度 ≥ `--signal-floor` × 预测峰值）
   内的秩相关，**主判据**；
2. `fwhm_ratio` — 实测 / 模型 FWHM，离焦失配会让它系统性偏离 1；
3. `peak_offset_ccd_px` — 峰值偏移，`tilt_x/y` 用它给出轴标定残差作对照；
4. `radial_correlation` — 径向剖面 Pearson。

指纹判读：若 `defocus_z4_pos` / `defocus_z4_neg` 的 `spearman_signal` 显著低于
`flat`，且 `--predict-from runtime`（把焦面补偿项加进预测）能把它们拉回来、
`--predict-from model` 不能 —— 缺失离焦项成立。

> ⚠️ 不要用 `spearman_coverage`（全域 Spearman）下结论：远场是 sinc²/Airy 型，
> 覆盖区里 >99.9% 的格在 CCD 上只是背景+读出噪声，它们的秩是随机的，该列会被
> 噪声底主宰。它保留只为与其它脚本口径可比。取值依据见
> `_hw_fidelity_ladder_sections.DEFAULT_SIGNAL_FLOOR`。

## 下一次设备会话的运行手册

🔴 硬件纪律（`docs/dev/hardware_discipline.md`）：整个阶梯**只开/关一次** SLM + 相机
（背靠背开关 SLM 会触发驱动冻结 `ReadSU=-1000`）；占用情况写进 `device_log.json`
与 `ladder_summary.md` 的设备纪律块。

```powershell
# 0) 先离线演练一遍（不开任何设备；收尾 assert_no_drivers 违约 exit 3）
.venv\Scripts\python.exe scripts\slm\hw_forward_model_fidelity_ladder.py --dry-run

# 1) 基准：裸前向模型（不带焦面补偿）
.venv\Scripts\python.exe scripts\slm\hw_forward_model_fidelity_ladder.py --execute `
    --wait-until-idle 600 --predict-from model `
    --out data\debug\differentiable_opt\fidelity_ladder_model

# 2) 对照：叠加焦面补偿（离焦项进预测）
.venv\Scripts\python.exe scripts\slm\hw_forward_model_fidelity_ladder.py --execute `
    --wait-until-idle 600 --predict-from runtime --focus-compensate `
    --out data\debug\differentiable_opt\fidelity_ladder_runtime
```

两次运行对比 `defocus_z4_*` 的 `spearman_signal` / `fwhm_ratio` 即为结论。
常用旋钮：`--patterns`（默认 `all`）、`--zernike-amp`（默认 1.0 rad 圆内 RMS）、
`--n-pupil`（execute 默认 256；要与 `optimized_final` 取到的历史
`phase_model.npy` 网格口径一致，否则按 `--resample-foreign-phase` 换采样并在
图样 note 里记录）、`--geometry-npz`（默认现场实测；复用存档要显式给）、
`--exposure-ms` / `--exposure-ceiling-ms` / `--frames` / `--discard` / `--settle-s`。

## `--execute` 跑完后本目录会出现

| 文件 | 内容 |
| --- | --- |
| `ladder_summary.md` | 阶梯总表 + 设备纪律块（开/关次数、占用时段）+ 解读指引 |
| `metrics.json` | 逐图样全字段指标（含径向剖面数组） |
| `metrics.csv` | 同上的扁平列，便于表格/脚本二次分析 |
| `plan.json` | 本次运行的全部决定（几何、模型网格、信号门限、设备序号、脚本 sha256） |
| `device_log.json` | SLM/CCD open→ready→display→capture→release 时间线 |
| `figures/fidlad_radial_profiles.png` | 逐图样径向剖面：模型 vs 实测 |
| `figures/fidlad_correlation_bars.png` | 逐图样相关系数横条（含 0.8 参考线） |

完整运行目录（含 `frames/` 原始帧与平均帧、`phase/` 上屏相位块、`pred/` 预测远场、
`source/` 脚本自拷贝）留在 `--out` 指定处，不复制进报告目录。
