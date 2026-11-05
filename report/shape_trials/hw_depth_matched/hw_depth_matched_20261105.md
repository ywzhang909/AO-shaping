# 硬件深度匹配复测与全口径仿真：主动探针样本效率终判（2026-11-05）

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **运行环境**: 硬件：1064 nm Santec SLM #1 + Daheng CCD（run02 采集 2026-11-05 09:33–09:49）；分析离线 CPU
> **说明**: run02_depthmatched 将 ztrain 振幅提至 ±1.0/2.0 rad 消除激励深度混杂；三个变体离线分析复用 _harness_task2 口径（_harness_task2_run02.py）；结论：深度混杂不解释 run01 边际，主动探针样本效率无可测收益，twin NCC 结构失配待诊断。
<!-- provenance:end -->

**一句话结论**：激励深度匹配后（±1.0/2.0 rad vs 随机族 RMS 0.99–2.28 rad），±Zernike 主动变体与随机变体在所有帧预算（n=2–32）下与无适应基线差异仍在 ±2% 内，主动选样无可测样本效率收益；全 harness 口径的 hw2f 仿真同样无区分力，且 twin 与硬件的 NCC 差距（0.97 vs 0.81）证明当前 twin 空间结构不足，不能作为在线回灌/闭环收益的判据。

## 1. 采集（run02_depthmatched，深度匹配）

| 项 | 值 |
|---|---|
| 探针数 | 68 探针 + 10 标定行 = 78 npz |
| ztrain 振幅 | ±1.0 / ±2.0 rad（run01 为 ±0.1/±0.2） |
| 随机族 | 同 run01（zernike/smooth/bandlimited 三族，RMS 0.99–2.28 rad） |
| 振幅 RMS 对齐 | ztrain RMS ≈ 0.89–1.78 rad，与随机族重叠 |
| 光通量 | run01 相同光学状态；exposure ladder 正常，6 次裁剪后恢复 |
| SLM/CCD | 09:33 打开 → 09:49:11 正常关闭（`成功关闭SLM #1`） |

标定行、`model_phase`/`raw_frames` npz 结构与 run01 完全一致，`transfer/fit.load_dataset` 直接可用。

## 2. 三个变体离线对比（同一 harness 口径）

harness：`PhysicalForwardFit`（Z4+残差 Z5–Z11, staged, trust region, Adam 120 it），曝光 radiometry，FFT 256，窗口 161px，σ=8.0px；holdout 12 帧不变。

| n | 基线 NRMSE/NCC | random NRMSE/NCC | zernike NRMSE/NCC |
|---:|---|---|---|
| 2 | 0.0235 / 0.812 | 0.0231 / 0.824 | 0.0230 / 0.824 |
| 4 | — | 0.0230 / 0.824 | 0.0238 / 0.822 |
| 8 | — | 0.0234 / 0.816 | 0.0234 / 0.819 |
| 16 | — | 0.0232 / 0.817 | 0.0231 / 0.824 |
| 24 | — | 0.0232 / 0.817 | 0.0230 / 0.823 |
| 32 | — | n/a | 0.0228 / 0.825 |

全部行与基线差 ≤3%，无单调趋势；run01 的 NCC 配对显著（n=2/8/24）在深度匹配后消失。

## 3. 仿真（hw2f 函几何 + hardware 档 twin，全 harness 口径）

全口径跑完（`--grid 128 --fft-size 256 --frame-budget n --geometry hw2f --twin-profile hardware --no-gate --seeds 229 --sizes 2 4 8 16 24 32 --iterations 120`）：

| 指标 | hw2f 全口径仿真 | 硬件实测（run01/run02） |
|---|---|---|
| 无微调 NRMSE | 0.0154 | 0.0235–0.0243 |
| 无微调 NCC | 0.9665 | 0.810–0.812 |
| 微调后 NRMSE（所有 n） | 0.0152 | 0.0228–0.0238 |
| 微调后 NCC（所有 n） | 0.9679 | 0.816–0.825 |

- 仿真中两个变体在所有 n 完全相同（0.0152/0.9679），说明 twin 太"容易"：模型结构完全覆盖 twin 的像差结构，主动/随机探针没有差别。
- NCC 0.97 vs 0.81 的差距是空间结构不一致，不只是噪声/幅度问题。

## 4. twin 空间结构诊断（已定论）

诊断车道完成；证据落盘于
`data/debug/zernike_probe_hardware/issue229/analysis/twin_gap/`
（`_twin_gap_diag.py` 固化脚本 + `twin_gap_summary.json`）。全部为冷冻
`PhysicalForwardFit`（d=0, residual=0；无 Powell/无优化）在 run01 holdout 上评估，辐射测量每次在训练像素上重拟合。

**受控反事实（决定性）**——同一批 holdout 相位，探针分布/span 完全不变，仅把"设备侧"从真硬件换成 sim twin 渲染：

| 设备侧 | holdout NCC |
|---|---|
| 真硬件 | 0.8101 |
| sim twin 渲染 | 0.7133 |

两者同相位、同 span、同收集；唯一变化量是 `phase→irradiance` 前向模型本身。
**因此 NCC 差距源于前向模型失配，而非探针采样/span**。早前草稿曾把 0.71 读作
"holdout 出 span 导致"的证据，系误读——受控反事实恰好在同 span 内把 0.71 定格为
前向模型问题，这里明确修正。

辅助证据：
- σ 扫描（frozen）：σ=8 为最佳（0.8101），σ 继续加大反而下降（σ=16 → 0.7847），
  说明模型的宽晕分量未过匹配，与"缺更宽的 pedestal"方向一致。
- 径向 profile（measured/model）：中远程比值>1（r=20 约 1.36、r=30 约 1.28），
  实测光斑有更宽的平基底 pedestal，高斯核过早衰减（口径最接近的 probe7 原始观测
  r=10→540×、r=20→2500× 因质心/归一化口径不同，仅作定性参考）。

**twin 修复方向**：在 forward 模型（phase→irradiance，含宽晕/pedestal 结构），
而非探针分布。候选：加未建模宽带底盘（残差散射/stray）、瞳孔边缘效应、更宽 CCD PSF 尾巴。

## 5. 结论与对闭环方案的影响

1. **主动探针样本效率主张撤销**：run01 的配对显著性（NCC n=2/8/24, NRMSE n=8）不构成样本效率优势的证据——run02 深度匹配后消失，且效应量 ≤2%。
2. **闭环主动选样暂停**：在 twin 空间结构改进前，不值得写闭环脚本。
3. **twin 改进优先，且方向锁定为前向模型**：NCC 差距（0.97 vs 0.81）是空间结构失配——受控反事实证明是 `phase→irradiance` 前向模型问题（同 span 同收集下 sim-twin 0.7133 < 真硬件 0.8101），而非探针分布。修复方向候选：未建模宽带底盘（stray/residual scattering）、瞳孔边缘效应、更宽 CCD PSF 尾巴；避开探针/span 侧改动。
4. forward_model_transfer_plan 的 P3–P6 闭环阶段全部推迟，直到 P2 几何对齐 + twin 结构改进后复测通过。
5. run01 结论（ISSUE229_FINAL.md 已加 addendum）：dataset 有效、方法可复跑、PSF/radiometry 管线在生产口径下复现质量——这三点维持成立。
