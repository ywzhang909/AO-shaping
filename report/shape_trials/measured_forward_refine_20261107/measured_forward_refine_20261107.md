# measured_forward_refine_20261107 规划报告

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **运行环境**: 离线规划文档与离线探针结果（issue #237；未占用设备）
> **说明**: 实测正向精化规划 + 离线探针：2D 配准精化、实测 PSF 标定、增益场、CCD 响应 LUT、杂散光与复振幅；含数据清单、逐项探针计划与验收口径（引 issue 原文）及 §7 离线结果（配准支持、PSF 不稳健、增益场不支持但退化、LUT 数据不足）。人工撰写。
<!-- provenance:end -->

**日期**：2026-11-07  
**设备状态**：离线规划、未占用 SLM/CCD/激光、未采集/操作任何物理设备

---

## 1. 状态概要

本报告为 issue #237 的规划文档与离线探针结果。原规划严格离线、不执行硬件操作；2026-11-07 追加了三个纯离线探针（配准/增益场/LUT、实测 PSF/背景、以及同口径方向评估），未占用任何物理设备。§7 给出逐项结论。

## 2. 目标与验收标准

### 目标
对观测链模型（defocus_forward/physical forward）精细化：
- 引入/替换结构项：亚像素 2D 配准、实测 PSF 标定、增益场、CCD 响应 LUT 扩展、复振幅杂散光，并保持基线逐位无差异、逐项可控切换。
- 全程可离线回放与定向探针评估，绝不依赖新采集。

### 验收判据（引用 issue #237 原文）
1. **2D 配准**：11-03 回放数据上，不做外部窗口对齐时零位移 NCC 恢复，模型估计位移接近实测(-3, +32) px。
2. **PSF**：开启后 d 与 log_width 参数相关降到 0.7 以下（不互补偿）；relative_l2_affine 与 cv_abs_error 均改善。
3. **增益场**：验证集残差的空间结构化模式收窄。
4. **LUT**：多曝光数据上 CCD 响应系统性曲率收窄（低曝光核心 CV 变化小于基线）。
5. **复振幅**：热点残差收窄，且热点形状可被复振幅点源位置/宽度解释。
6. **全部**：默认关时逐位比对完全一致，P0 离线辨识率不降。

## 3. 现有可用数据清单

- `data/debug/slm_transfer/20261107_focus_scan/focus.json` — 离焦/中心/增益/曝光范围标定
- `data/debug/slm_transfer/defocus_forward_20261107_focus_r2/` — 52 帧 {frame, raw_frames, panel_phase} 对应 U/C，2592×1944 原始 CCD、1200×1920 SLM，相位参数齐全
- `data/debug/slm_transfer/defocus_forward_20261107_focus_r2_fit/` — 拟合参数、forward_fit.json，包含 ramp_matrix、illumination_center_estimate、prediction_{M0/M1/M3}_defocus.npz
- `data/debug/slm_transfer/issue234_single_mode_screen_20261107/`, `.../amp05/amp02/` — 单模方向±探针 + 曝光序列，50 帧/组，含多档曝光
- `data/debug/slm_transfer/issue235_parity_20261107/parity_registration_scan.json` — parity 扫描，包括全局 2D 平移扫描
- `data/debug/slm_transfer/loss_compare_20261107_phase_sign_m1/results.json` — phase_sign -1 loss 校验、分数、基线

【npz 形状见附录】

## 4. 离线探针与分析计划（均“未开始实现”）

### 4.1 2D 配准精化
- **数据**：defocus_forward_20261107_focus_r2, parity_registration_scan.json
- **方法**：用 parity_scan 全局最优 dx/dy 作为 init，优化亚像素仿射与 origin 参数。结合 ramp_matrix 评估模型输出与实测位移一致性。增益场/PSF/杂散光均关闭。
- **产出**：估算 origin/dx/dy、比对 NCC 曲线，生成分析图与summary表
- **判据**：NCC 恢复、估计与实测位移相符
- **依赖**：parity_scan
- **状态**：离线探针已完成（2026-11-07），结论“支持进入实现”；上机验证代码已准备，未执行

### 4.2 实测 PSF 标定
- **数据**：focus_scan、defocus_forward_20261107_focus_r2、single_mode_screen
- **方法**：以多曝光/static focus 序列与 panel/CCD 匹配，估测离焦PSF核（高斯混合等），替换模型解析核，与 d 参数可辨识性分析。
- **产出**：PSF参数/核、d vs log_width 相关性矩阵、l2/cv 误差统计
- **判据**：相关降至0.7以下、误差同步改善
- **依赖**：2D 配准结构确定后进行
- **状态**：离线探针已完成（2026-11-07），结论“有信号但不稳健，暂不推进实现”

### 4.3 增益场建模
- **数据**：defocus_forward_20261107_focus_r2、amp05/amp02_screen（可区分空间响应）
- **方法**：以多向探针与拟合结果，低秩/块状方式拟合空间增益，将原标量 gain 替换，核查结构残差/maps
- **产出**：空间增益参数、残差图
- **判据**：空间结构残差收窄
- **依赖**：2D配准/PSF已稳定、LUT/杂散光关闭
- **状态**：离线探针已完成（2026-11-07），结论“不支持”，但拟合退化（14,307/14,641 像素被截断），需先修正则化/掩膜再判

### 4.4 CCD 响应 LUT 扩展
- **数据**：amp02/05/单模 exposure_probe_frames、多曝光 focus.json
- **方法**：把实测曝光对上传导到 LUT，拟合各曝光响应曲率，CV 统计对比开关/参数组
- **产出**：LUT 曲线、响应曲率变化
- **判据**：低曝光 CV 缩减
- **依赖**：2D配准/PSF/增益场前置
- **状态**：离线探针已完成（2026-11-07），结论“数据不足/无法辨识”，需上机同命令 ≥4 曝光档

### 4.5 杂散光（复振幅分量）
- **数据**：defocus_forward_20261107_focus_r2_fit（background）、all主数据
- **方法**：对热点残差做DC聚类与点源叠加拟合/解释。与 bias_counts 协同排障，评估两两相关与可辨识性。
- **产出**：杂散光参数、热点图示
- **判据**：热点结构被复振幅项解释
- **依赖**：基础模型先收敛
- **状态**：离线探针已完成（2026-11-07），结论“加性背景在本数据上不可辨识（拟合恒为 0）”，需上机暗场隔离

## 5. 后续上机验证步骤
- 各模块单独离线评估通过后，**逐项开发/调试/测试需严格设备占用纪律**：明示占用设备类型与编号，调试/采集互斥，实验结束后明确释放，每次串行执行，设备空闲时方可排队上线，不得并行启动同类型设备进程；一次只做一项变更、全开关可逐步回滚

## 6. 风险与未解问题
- #235 phase_sign 修正未过门禁（Pearson +0.481/+0.496 < 0.50）；当前所有结构项识别基线待修正
- phase_sign 作用域、输入一致性（#235/#237 描述）不彻底明晰，存在误配参数与残差解析风险，需与门禁基线联动视作 blocking
- 作用域筛选（2026-11-07 部分结果）：`command` 臂回归护栏精确复现历史 −1（d=−0.002125，Δ 4.7e-10；origin 偏差 0.0000 px）；`aberration` 臂 d=+0.0249、origin 相对偏移 3.94 px。`carrier` 在存档中不可分离（`carrier_available: false`），该臂等同 `aberration`。三组单模屏方向与 `aberration_focus`、导出往返仍待完成。
- 拟合路径存在 HEAD 级差异：`fit_defocus_forward_dataset.py --phase-sign -1 --phase-sign-scope command` 得 d=0.002432，而 Pearson 路径得 −0.002125；+1 侧与存档差 5.7e-4 / origin 0.10 px。HEAD 同样复现，非本次引入；回归护栏以 Pearson 路径为准，脚本路径数值单列记录。
- 2D 配准的 11-03 回放验收数字不在本地存档，**未测**，不能声称通过该条验收
- 3D/PSF 域间不一致（r2 改善、ladder pass2 退化）未解决
- 暗场/多曝光数据缺失，背景与 LUT 只能在线上机采集后判定

## 7. 离线探针结果（2026-11-07）

三个探针脚本全部离线运行（CPU、只读存档、未 import `ao_shaping.drivers`、未占用 SLM #1 / 相机 / 激光），基线逐项复现 `offline_benchmark` 的 axis_A，两次运行确定性一致，`ruff --no-cache` 全绿。产物在 `data/debug/slm_transfer/` 下（被 `.gitignore` 忽略，数字仅存本地）。

> **前提说明**：以下 −1 符号的绝对值受 #235 phase_sign 作用域未定影响；探针只比较“模型 vs 基线”的增量，增量方向不依赖作用域，绝对值需在作用域定案后复核。

### 7.1 全局亚像素配准精化

脚本 `scripts/slm/registration_gain_probe_20261107.py`；训练池 = ladder pass1(42) + r2 train(18)。

| 量 | 值 |
|---|---|
| 整数字段 (dy,dx) | (1, 2) |
| 亚像素 (dy,dx) | (+0.527, +1.762) |
| 训练峰值分数 | 0.7953 |

held-out 增量（模型 − 基线）：

| 评估集 | Δcore Pearson | Δfull rel-L2 |
|---|---:|---:|
| ladder pass2 | +0.0758 | −0.0546 |
| r2 held-out | +0.0207 | −0.0218 |

方向一致性（42 对 ±，sign −1）：基线 median +0.618 / 29 对 ≥0.50 → 配准 median +0.639 / **35 对 ≥0.50**（#235 整数扫描参照 +0.470 / +0.481，17/42、16/42）。

**判定：支持进入 #237 实现。** 只做单一全局位移，不做逐对位移；整数 (1,2) 与亚像素 (+0.527,+1.762) 一致。11-03 回放验收数字（零位移 NCC 0.03、对齐 0.68、实测 (−3,+32) px）不在本地存档，**未测**，未用替代值。

### 7.2 实测 PSF + 加性背景

脚本 `scripts/slm/measured_psf_probe_20261107.py`；PSF/背景只在 ladder pass1(42) + r2 train(18) 上拟合，held-out 无泄漏。

r2 held-out（sign −1）：

| 模型 | core Pearson | core rel-L2 | full Pearson | full rel-L2 |
|---|---:|---:|---:|---:|
| baseline | 0.4986 | 0.7779 | 0.6557 | 0.8182 |
| psf_bg | 0.5479 | 0.6330 | 0.7418 | 0.6736 |
| 增量 | **+0.0492** | **−0.1450** | +0.0860 | −0.1446 |

保留意见：
1. **ladder pass2 的 core Pearson 反而 −0.056**，方向一致性 0.560→0.530（变差）——改善集中在 r2 域，跨域稳健性存疑。
2. 选中 σ=4 ccd px 位于预设网格边界，结论依赖该网格（未事后扩网格）。
3. 背景项（二次 6 项）在所有评估集拟合为 0 → **加性背景/杂散光在本数据上不可辨识**。
4. 宽度未复现历史“实测 14 / 模型 21 px”的对比，改善并非来自宽度匹配。
5. +1 符号的改善幅度远小于 −1（core Pearson +0.007）。

**判定：有信号但不稳健，暂不推进 #237 实现。** 需先解决域间不一致与网格边界问题。

### 7.3 空间增益场

固定 5 基（常数 + 4 个二次/交叉项），超参预注册不调。held-out 增量（sign −1）：ladder pass2 Δcore +0.1798 / Δrel-L2 −0.1072，但 **r2 held-out Δcore −0.0117 / Δrel-L2 +0.0083（恶化）**。

**判定：不支持。** 但该负结果证据很弱——拟合退化，14,307/14,641 像素被截断到 [0.5,1.5]，信号掩膜仅约 198 px，系数 50–80 量级。这是 ill-conditioned 拟合，不等于“平滑增益场无用”。后续应先修掩膜/正则化再判。

### 7.4 CCD 响应 LUT

**判定：未拟合（数据不足）。** 训练划分内无同命令多曝光 bracket（pass1 探针均为单曝光，r2 无曝光探针；仅 pass2 held-out 有 30 组 bracket）。跨次重复曝光跨度约 14%，不足以识别低曝光曲率。需上机在同一会话采集 ≥4 个曝光档。

### 7.5 子项结论汇总

| #237 子项 | 离线结论 | 能否进入实现 |
|---|---|---|
| 2D 全局亚像素配准 | 支持（两集同向改善，方向一致性 35/42） | **是** |
| 实测 PSF | 有信号但不稳健（域间方向相反、网格边界） | 暂否 |
| 空间增益场 | 不支持，但拟合退化 | 暂否（先修正拟合） |
| CCD 响应 LUT | 数据不足 | 否（需上机采集） |
| 复振幅杂散光 | 加性背景不可辨识 | 否（需上机暗场） |

### 7.6 上机代码准备状态

上机采集探针 `scripts/slm/measured_forward_hw_probe_20261107.py` 已准备（默认 dry-run，不带 `--execute` 不 import 驱动、不打开设备）：暗场/杂散光（激光关 + SLM 平场激光开）、同命令 ≥4 曝光 bracket、全局亚像素配准验证。**本次未实际占用任何设备**。

dry-run 已验证（2026-11-08）：`python scripts/slm/measured_forward_hw_probe_20261107.py --print-plan` **exit 0**、日志末尾 `drivers imported = 0`、规划 **30 帧 / 4 阶段**、`ruff check --no-cache` 通过。离线自检 pooled 全局位移整数 (1,2)、亚像素 (+0.762,+1.546)、peak 0.8694；U/C 分别 (+0.410,+1.818)/(+1.961,+1.237) —— 逐对位移不一致，**只应采单一全局值**。

### 7.7 全局亚像素配准实现（2026-11-08）

#237 唯一支持进入实现的子项已落地（默认关闭、关闭时逐位无差异）：
- `src/ao_shaping/algorithm/signal_processing/{defocus_forward_core,differentiable_defocus_uniformity,physical_forward_fit}.py`：新增可微相机采样网格位移（单一全局 `(dy,dx)`，CCD 像素；内容向 +dy/+dx 移动，与 `scipy.ndimage.shift` 同号）；`FrozenDefocusParameters`/`FrozenDefocusForward` 支持配准参数、JSON/checkpoint 往返；`PhysicalForwardFit` 支持固定或可训练位移并导出 `registration_value`。
- `scripts/slm/fit_defocus_forward_dataset.py`：`--registration-shift DY DX`（默认关闭）与 `--learn-registration`。
- `src/ao_shaping/tools/slm/transfer/fit.py`：checkpoint 配准参数传递与导出。
- 测试 `tests/ao_shaping/algorithm/signal_processing/test_defocus_forward_registration.py`（22 项：默认关闭逐位一致、零位移一致、可微性、校验、序列化）。
- 复现脚本 `scripts/slm/registration_eval_20261108.py` + 结果 `data/debug/slm_transfer/registration_eval_20261108/results.json`。

**复现与归档探针存在口径差异**（未硬凑）：最大 Pearson 偏差 0.0210（ladder pass2 core_pearson 0.7486 vs 归档 0.7696）、方向计数 31 vs 35（中位数 current 0.6471 vs 归档 0.6387）。原因是新实现位移**相机采样网格**（在采样前作用于连续场），而归档探针用 `scipy.ndimage.shift` 平移**完成后的预测数组**；两者插值、坐标与边界语义不同，符号约定已核对一致。真正的 `direction_agreement` 门已通过 checkpoint-aware 包装评估器完成离线 A/B 验证，结果见 `data/debug/slm_transfer/registration_eval_20261108/{direction_gate.json,REPORT.md}`。旧评估器基线逐字段完全复现；本次测试的拟合全局位移 `(dy,dx)=(+0.526743,+1.762385)` 开启后，amp 0.10/0.05/0.02 的 median 分别为 0.439027/0.466025/0.423716，均低于 0.50；因此**该全局位移**不能使方向门达标，闭环仍阻塞。该单点结果不排除其他几何变换或位移参数，且逐样本/仿射配准不在本次范围内。全程未占用设备。

## 8. 上机执行单（2026-11-08，**仅准备，未执行**）

> 设备纪律：任何 `--execute` 前先确认 SLM #1 控制器正常（ReadSU 不为 -1000，否则需手动重启）、Daheng 相机可用、激光状态符合阶段要求；单设备独占、串行，结束后在日志与回复中明确报告占用/释放。`--wait-until-idle` 用于等待设备空闲。

### 8.1 采集探针（未执行）

```powershell
# 先确认激光处于所需状态：dark_field 阶段要求激光关，后续阶段要求激光开。
python scripts/slm/measured_forward_hw_probe_20261107.py --execute `
  --wait-until-idle 60 `
  --out data/debug/slm_transfer/measured_forward_hw_probe_20261107
```

dry-run（已执行）exit 0，30 帧 / 4 阶段；产出含暗场曝光 bracket、SLM 平场激光开 bracket、同命令 ≥4 曝光 LUT bracket、全局配准验证。执行后将结果并入 §7（P1-4 LUT / P1-5 杂散光判定）。

### 8.2 方向门验证（未执行，当前需先补评估器适配）

checkpoint-aware 包装评估器 `scripts/slm/registration_direction_gate_20261108.py` 已新增并验证，未修改旧评估器。A 臂关闭配准逐字段完全复现旧基线；B 臂开启 `(dy,dx)=(+0.526743,+1.762385)` 后三组 median 仍低于 0.50：amp 0.10 = 0.439027，amp 0.05 = 0.466025，amp 0.02 = 0.423716。结果见 `data/debug/slm_transfer/registration_eval_20261108/direction_gate.json` 与 `REPORT.md`；方向门未通过，未执行反号臂（没有组下降超过 0.05）。

```powershell
# 已执行过的纯离线方向门评估；不触硬件
python scripts/slm/registration_direction_gate_20261108.py `
  --fit data/debug/slm_transfer/defocus_forward_20261107_focus_r2_fit/forward_fit.json `
  --registration-shift 0.5267430172174841 1.7623859421066448 `
  --origin model --out data/debug/slm_transfer/registration_eval_20261108/direction_gate.json
```

### 8.3 锚定闭环（2026-11-08 owner 决定放行；方向门未达标，显式风险接受）

```powershell
# checkpoint = phase_scope_20261107/checkpoint_command/frozen_parameters.json
#             + 已评估位移 (dy,dx)=(+0.526743,+1.762385)，离线生成（data/ 不入库）：
python -c "import json,pathlib; s=pathlib.Path('data/debug/slm_transfer/phase_scope_20261107/checkpoint_command/frozen_parameters.json'); o=pathlib.Path('data/debug/slm_transfer/registration_eval_20261108/checkpoint_registered/frozen_parameters.json'); o.parent.mkdir(parents=True, exist_ok=True); d=json.loads(s.read_text(encoding='utf-8')); d['registration_shift_ccd_px']=[0.5267430172174841, 1.7623859421066448]; o.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding='utf-8')"
python scripts/slm/shape_transition_timing_20261106.py `
  --mode anchored --execute `
  --checkpoint data/debug/slm_transfer/registration_eval_20261108/checkpoint_registered/frozen_parameters.json `
  --out data/debug/slm_transfer/shape_transition_20261108_anchored_reg `
  --shapes square,circle,triangle,square --side 400 --cv-threshold 0.10 `
  --wait-until-idle 60
```

验收：实测核心 CV ≤ 0.10 且签名方向一致性 ≥ 0.50（§2 判据）。**状态**：方向门未达标（amp 0.10/0.05/0.02 的 median 0.439/0.466/0.424 < 0.50）；2026-11-08 owner 决定在已知该缺口的情况下放行 anchored 闭环，属显式风险接受。运行后按 §2 判据如实记录实测 core CV 与方向一致性，不回溯修改判据；采集探针（§8.1）仍为待设备窗口执行项。


## 附录：部分数据集主要 npz 结构
- defocus_forward_20261107_focus_r2/*.npz： raw_frames [3,1944,2592]、panel_phase [1200,1920]
- defocus_forward_20261107_focus_r2_fit/prediction_M0.npz: predicted/measured [43,121,121]、camera_xy [121,121,2]
- issue234_single_mode_screen_20261107/pass1/*.npz： frame [1944,2592]、panel_phase [1200,1920]、exposure_probe_frames [6,1944,2592]
