# Issue #235：冻结模型全幅形态诊断（Q1–Q3，整幅面，离线 2026-11-17）

<!-- provenance:start -->
> **生成脚本**: [`scripts/slm/issue235_model_fill_diagnostic_20261117.py`](../../../scripts/slm/issue235_model_fill_diagnostic_20261117.py)
> **数据/关联脚本**: [`scripts/slm/square_1000_model_diagnostic_20261030.py`](../../../scripts/slm/square_1000_model_diagnostic_20261030.py)
> **运行环境**: 离线（回放 2026-11-17 14:07–14:09 已采集的 14 帧会话；本次分析未占用 SLM/相机/激光）
> **说明**: #235 冻结模型整幅形态诊断（Q1–Q3）：在同一坐标系（整幅原生平面 1944×2592，实测帧与模型预测同公式）量化冻结 model640 前向与实测的形态差。Q1 集中度不可预测（显示相位门禁 0 diff，模型预测满幅平顶、实测点状，整幅 Pearson ≈ −0.010）；Q2 单参数扫描（phase_scale / amplitude_width / sigma_ccd / defocus / 像素尺度）无一能同时复现实测的功率与半高像素数；Q3 反解表明实测光斑隐含有效照明口径 ~363 px，与冻结包络 (352, 133) px 同量级，即问题不在口径量级，而在照明包络→远场的形态映射。数字全部由脚本读取 diagnostic.json 渲染。
<!-- provenance:end -->

数据：`data/debug/square1000_model_diag_20261117_140741`（SLM #1 + Daheng MER2-507-23GM，base 曝光 0.1 ms，探针幅度 0.03 rad，tilt 周期 64 panel px）。
复现：

```bash
python scripts/slm/issue235_model_fill_diagnostic_20261117.py \
  --session data/debug/square1000_model_diag_20261117_140741 \
  --out data/debug/issue235_model_fill_20261117
```

模型：`defocus_forward_20261107_focus_r2_fit/checkpoints/M3_illumination` 同源冻结参数（grid 640 / FFT 2048，`pixel_samples 3`，`origin [1265.67, 941.12]`，`amplitude_center [965.03, 909.92]`，`amplitude_width [352.25, 132.72]`，`sigma_ccd_px 0.0`，`defocus_diopter −0.0244`），0.1 ms 曝光地板 5.713 counts（background 0.2049 + bias/exposure 5.508）。

显示相位门禁：`to_panel(encode(model_phase))` 重建面板灰度 vs 存档 `panel_gray`，14 帧逐像素 mean |diff| 0、max 0（1 LSB = 0.00614 rad）——模型输入确为面板实际写入的相位。

## Q1 — 冻结模型能否预测实测集中度？

| 帧 | 角色 | ±50 px 内功率（实测/预测） | 半高像素数（实测/预测） | FWHM x,y 实测 (px) | FWHM x,y 预测 (px) | Pearson | cosine |
|---|---|---:|---:|---|---|---:|---:|
| 0 | A | 0.2571 / 0.007099 | 194 / 990239 | 18.46, 14.59 | 991.3, 998.6 | −0.0101 | 0.0161 |
| 2 | N4+ | 0.2591 / 0.007105 | 193 / 990614 | 18.43, 14.50 | 991.4, 998.8 | −0.0101 | 0.0158 |
| 3 | N4− | 0.2483 / 0.007092 | 202 / 989842 | 19.67, 13.90 | 991.2, 998.3 | −0.0104 | 0.0168 |

判定：**冻结模型不能预测实测的点状集中**——它把 0.0071 的信号放进 ±50 px、99 万像素高于半高，实测为 0.257、约 194 像素；整幅 Pearson ≈ −0.010。

## Q2 — 单参数扫描能否把预测推到实测形态？

目标（实测锚点 0）：±50 px 功率 0.2571，半高像素 194。

| 参数 | 冻结值 | 复现所需值 | 倍数 | 预测 ±50 px | 预测半高像素 | 比值（功率, 半高） | 复现? | 物理合理性 |
|---|---:|---:|---:|---:|---:|---|---|---|
| phase_scale | 1×命令相位 | 0.9 | 0.9 | 0.00848 | 798830 | 0.033, 4118 | 否 | 重拟合 fitted 0.9992（`fwdmodel_fit_20261117.md`，边际 dPearson 0.0019），所需值落在拟合区间 [0.90, 1.10] 内但方向无改善 |
| amplitude_width_x_px | 352.2 panel px | 2818 | 8 | 0.185 | 161 | 0.720, 0.830 | 否 | 重拟合未调该量（固定为 checkpoint (352.25, 132.72)）；>960 px 已超出面板半幅 |
| sigma_ccd_px | 0 | 16 | n/a | 0.0104 | 990602 | 0.040, 5106 | 否 | 重拟合 blur_ccd_px 2.15；模糊只会铺开功率，不能把 1000 px 方斑压成 16 px 点 |
| defocus_diopter | −0.02439 | −1 | n/a | 0.0179 | 28278 | 0.070, 146 | 否 | checkpoint −0.0244；加离焦只增曲率/模糊 |
| far_field_pixel_scale | 1× | 0.0142 | 0.0142 | 0.919 | 230 | 3.51, 1.04 | 否 | 重拟合 scale_px 2.009 CCD px/sample（2× padding；几何 1.9698），冻结模型满幅为 3.94 CCD px/sample；所需 0.0142 意味着亚 CCD 像素采样 |

判定：**扫描范围内没有任何单参数同时复现实测的功率与半高像素数。**

## Q3 — 实测光斑是否隐含满口径瞳面？

实测光斑 FWHM 18.46 px（0.0406 µm）。单个满口径 DFT 采样 = λf/(Np) = 8.66 µm = 3.94 CCD px；K = λf/(p·p_cam) = 7557 vs 标定 7564（相对误差 0.000937）。

| 候选口径 | 预测 FWHM (px) |
|---|---:|
| 满面板，方孔边 1920 px | 3.487 |
| 满面板，Gaussian 宽 960 px | 2.950 |
| amplitude_width x = 352 px | 8.040 |
| amplitude_width y = 133 px | 21.34 |

反解（由实测 FWHM 18.46 px 反演同一关系）：方孔边 362.7 px（占面板 19%），Gaussian 宽 153.5 px。

判定：**实测光斑不隐含满口径**——满面板方孔聚焦到 3.49 px，比实测窄 5.3 倍；实测光斑反解出的有效孔径 ~363 px，与冻结照明包络 (352, 133) px 同量级。也就是说**照明包络的口径量级本身与实测自洽**，不是量级错误。

## 结论与对 #235 的影响

1. 显示相位门禁 0 diff，说明**输入相位不是变量**——此前 panel_gray 反解得到的同一结论（`model_direction_hw_20261117.md` §panel_gray 反解）在整幅面上被独立复现。
2. 差的是**远场形态**而非口径量级：模型把 1000 px 方相位映射成满幅平顶（99 万像素高于半高），实测是 ~16 px 点；Q3 的反解表明有效照明口径（~363 px）与冻结包络同量级，所以问题不在"远处口径算错一个量级"。
3. Q2 单参数扫描（phase_scale / amplitude_width / sigma_ccd / defocus / 像素尺度）无一能同时复现实测的功率与半高像素数；结合 Q1，模型在整幅面上对形态的预测与实测无关（Pearson ≈ −0.010）。
4. 因此方向门禁失败的**主因收敛到物理预测模型的形态映射环节（照明包络到远场的映射），而非相位输入、窗口或坐标选择**（后者在 `model_direction_hw_20261117.md` 已单独排除）。
5. 该结论**不关闭 #235**：方向门禁（correlation ≥0.85 / cosine ≥0.50）与实验闭环核心 CV ≤0.10 仍未满足；下一步是重拟合照明包络→远场的映射（或直接以实测远场标定前向），而非继续在相位输入上排查。

## 复现与产物

- 同一数据会话的旁证报告：`report/shape_trials/model_direction_hw_20261117/model_direction_hw_20261117.md`（窗口对照、刀口基线、差分刀口、panel_gray 反解、中心回灌）。
- 本报告全部数字为本目录 `diagnostic.json` 的键；`summary.md` 为脚本自动生成的可读摘要。
