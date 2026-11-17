# Issue #247：tilt / 离焦下零级与目标级的分离实测（2026-11-17）

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **数据/关联脚本**: [`scripts/slm/issue247_capture_tilt_20261117.py`](../../../scripts/slm/issue247_capture_tilt_20261117.py)
> **数据/关联脚本**: [`scripts/slm/issue247_zero_order_stability_20261117.py`](../../../scripts/slm/issue247_zero_order_stability_20261117.py)
> **运行环境**: 离线+硬件（Santec SLM-200 #1 + Daheng CCD #0，2026-11-17 真线性 tilt 全幅采集；报告本体不占用设备）
> **说明**: Issue #247 零级/目标级分离实测：13 组全幅 Mono12 真线性 ±tilt/±离焦探针，flat 基线最大漂移 2 px、tilt 下零级稳定 ≤2.3 px、目标级两瓣分离 109–119 px；核心发现是面板轴↔CCD 轴为 +88.54° 的安装旋转（原始 CCD 帧上呈轴交换），由两轴位移矢量实测出，非写死 90°。含 P2 余弦探针 ROI 偏离光束的阴性复核。人工撰写。
<!-- provenance:end -->

**结论先行**：在真线性 ±tilt 与 ±离焦探针下，零级光斑**稳定不动**（7 帧 flat 基线最大漂移 2 px，
tilt 帧里剥离目标分量后零级距 flat 中心 ≤ 2.3 px），而目标级随 tilt 明显偏转（两瓣分离
109–119 px）。关键新发现是**面板轴↔CCD 轴为 +88.54° 的安装旋转**：原始 CCD 帧上 panel-x 斜坡使
目标沿 CCD-y 移动、panel-y 斜坡使目标沿 CCD-x 移动（即轴交换），两轴位移矢量同时给出安装旋转
**+88.54°**（不是恰好 90°，含 1.086 轴间尺度比），逐帧旋转对齐后位移完全落到对应轴上
（跨轴 ≤0.12 px）。这给 #247 前向模型明确了必须匹配的光学映射。
本轮**不启用硬件零级校正**——先验证分离方向，校正留待模型映射对齐后。

- 数据：`data/debug/issue247_tilt_20261117`（13 组全幅 Mono12）、`data/debug/issue247_analysis_20261117`（离线分析 JSON）
- 设备：SLM #1 + Daheng CCD #0，22:04:53 开 CCD / 22:04:54 开 SLM → 22:06:10 全部关闭，锁空；13 组 0 饱和
- 离线分析：`scripts/slm/issue247_zero_order_stability_20261117.py`（物理正确口径，见 §3）
- 本报告不含 MIL 任何代码/文档改动。

## 1. 为什么补采：既有 P2 数据不能回答 #247

先前的 14 帧方向诊断（`square1000_model_diag_20261117b_amp03/amp10`）经离线复核，有两点使它
**不能**作为零级位移证据：

1. 其中的 RX/RY 探针是 **64 px 周期的余弦光栅**，不是线性 tilt；缺少单向斜坡，无法产生
   目标级的定向平移。
2. 标准分析器的采样网格 `camera_xy`（322×310，x∈[1300,2536]）**落在光束之外**——零级光束中心
   约 (x=1227, y=1104)，x=1227 < 1300。分析器测到的是固定背景（~15 counts，与曝光无关）加一个
   淡弱伴斑与漂移，而非探针/光束本身；其"模型与实测不匹配"是采错区域的假象，不是物理结论。
   同一批数据里探针差异的峰值只有 121–339 counts，且其 anchor-to-anchor 自身漂移峰值达
   508–583 counts——**信号小于本底漂移**，符号在 amp03/amp10 两会话间也不一致。

因此新采一批**真线性 ±tilt**、全幅、零级位于采样区内的帧（本节下文），并用物理正确口径分析。

## 2. 采集口径（硬件）

| 项 | 值 |
|---|---|
| 帧 | 1944×2592 全幅 Mono12（full_scale=4095，`effective_full_scale` 与 `bit_depth` 双重确认） |
| 曝光 | 0.05 ms（读回一致），gain 读回；13 组 `saturated_pixels=0`，峰值 1384–2963 counts |
| 组 | flat before×2 / ±tilt-x / ±tilt-y / ±defocus / flat mid×3 / flat after×2 |
| 斜坡 | panel 上真线性 tilt（正负成对），幅度见 manifest `phase_params`；defocus ± |
| 时间戳 | 单调 UTC/ns；`device.json` + `released.json` 记 open/close 与锁释放 |
| 驱动板 | `require_drive_board()` 硬断言、`serial_source=="driveboard"`；`--wait-until-idle`；finally 释放 |
| 释放 | released.json 全 true；holder=None |

## 3. 分析方法（物理正确口径）

把 tilt+ 与 tilt− 两帧当作同一零级 + 一对反对称目标瓣：

- **零级（静止分量）** = `min(rate(tilt+), rate(tilt−))`：目标瓣在一侧时另一侧只剩零级，取逐像素最小
  剥掉移动分量；再对结果做椭圆定心。它应与 flat 中心重合且稳定。
- **目标级（移动分量）** = 反对称差分 `rate(tilt+) − rate(tilt−)`：得到位于 `+d` 与 `−d` 的两瓣，
  其半间距 `d` 给出目标位移幅度与方向；沿/跨 tilt 轴分解揭示光学映射。

旧版脚本把每帧全局峰当零级（tilt 帧里那其实是被偏转的目标瓣），口径错误，已作废；本脚本重写。

## 4. 结果

所有 §4 的坐标都在**旋转配准后的对齐帧** `panel_aligned_rotated`（每帧按 +88.544° 旋转后
裁切/补边到 2641×2010），面板轴与 CCD 轴已被对齐，因而"沿/跨轴"分量可直接读。

### 4.1 零级稳定性

- flat 中心（ellipse）中位数 **(870, 1255)** 对齐帧 px；7 帧 flat 相对中位数的最大漂移 **2 px**。
- tilt 帧剥离目标后零级：panel-x 对 **(869, 1254)**（距中心 1.41 px）、panel-y 对 **(868, 1254)**
  （距中心 2.24 px）——与 flat 漂移同量级，即**零级在 tilt 下不动**。

### 4.2 目标级偏转与旋转配准（+88.54°，非写死 90°）

| 探针轴 | 目标两瓣分离 | 沿该轴分量 | 跨轴分量 | 零级稳定 |
|---|---|---|---|---|
| panel-x tilt | 109.08 px | **+54.54 px（CCD-x）** | 0.12 px | 1.41 px |
| panel-y tilt | 118.57 px | **+59.28 px（CCD-y）** | 0.12 px | 2.24 px |

- 未旋转的原始 CCD 帧上，面板轴与 CCD 轴**交换**：panel-x 斜坡→CCD-y（−54.50 px）、
  panel-y 斜坡→CCD-x（+59.19 px）。两轴实测位移矢量给出安装旋转 **+88.544°**（Kabsch，
  orthogonality error 0.24°、scale ratio 1.086、max residual 2.34 px），不是恰好 90°。
- 逐帧按 +88.544° 旋转对齐后，panel-x 位移落到 **CCD-x**（+54.54 px）、panel-y 落到 **CCD-y**
  （+59.28 px），跨轴串扰 ≤ 0.12 px——轴交换完全由该旋转解释，无残余正交项。
- 这是 #247 前向模型必须内建的映射：`panel (x, y) → CCD` 是旋转 **+88.54°**（含 1.086 的
  轴间尺度比），而非独立的"轴交换"假设。角度与矢量来自 `image_alignment` 元数据。

### 4.3 离焦

defocus+ 中心 (866,1254)、峰值 32536 counts/ms；defocus− 中心 (870,1254)、峰值 49641 counts/ms。
离焦主要改变光斑尺寸/峰高，中心基本不变（这与零级/目标级分离正交）。

## 5. 判读与下一步

1. **#247 可确认**：tilt 下零级稳定、目标级定向偏转，二者可分离；硬件零级校正的物理前提成立。
2. **不可跳过的一步**：前向模型必须匹配 §4.2 的 +88.54° 旋转映射（含 1.086 轴间尺度比），否则任何
   基于模型的零级校正都会把 panel-x 的校正加到 CCD-y 上，越校越偏。
3. **本轮不启用硬件零级校正**：先在模型里对齐映射并用离线帧回验，再进入闭环。
4. 短期上机计划见 `docs/dev/next_hw_plan_transfer_candidates_issue247_20261117.md`；相关方向诊断见
   [`model_direction_hw_20261117`](../model_direction_hw_20261117/model_direction_hw_20261117.md)。
