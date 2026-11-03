# 测量传递层、对齐评分与有界零级：离线结果 2026-11-03

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **数据/关联脚本**: [`scripts/ml/sim2f_fov_hw_refine.py`](../../../scripts/ml/sim2f_fov_hw_refine.py)
> **数据/关联脚本**: [`src/ml/sim2f/calibrate.py`](../../../src/ml/sim2f/calibrate.py)
> **数据/关联脚本**: [`src/ml/sim2f/optics.py`](../../../src/ml/sim2f/optics.py)
> **数据/关联脚本**: [`scripts/ml/sim2f_fov_hw_offline_diagnostics.py`](../../../scripts/ml/sim2f_fov_hw_offline_diagnostics.py)
> **运行环境**: 离线分析（历史硬件采集会话 + 已完成的 ring refine 离线诊断；设备下线，无新增 CCD 采集）
> **说明**: 从已完成的 ring refine 目录与捕获会话中提取的离线诊断；模型喂进的是原始 128-grid 相位而非 SLM 实际显示相位（panel_gray 反解），对齐窗口需 (-3, +32) px 位移，未解决实测增益差异。这些数值来源见 artifacts/offline_diagnostics_20261103.json，离线诊断脚本为 scripts/ml/sim2f_fov_hw_offline_diagnostics.py；不视为物理整形成功或 CCD 验收证据。
<!-- provenance:end -->

## 1. 结论与证据边界

- 从 `data/debug/sim2f_fov_hw_refine_ring_20261103_091657` 与 `data/debug/sim2f_fov_hw_20261103` 提取的离线诊断结果（详见 `artifacts/offline_diagnostics_20261103.json`）。
- 模型喂进的相位是原始 128-grid 相位，非 SLM 实际显示相位（从 `panel_gray` 反解得到的）。使用实际相位后，参考相位模型预测目标函数值 2.06497 与实测 2.06530 近乎一致（差 3×10⁻⁴），表明相位映射问题是导致模型-实测增益差异的根本原因。
- 模型与实测窗口的对齐需要 **(-3, +32) px** 位移。零位移 NCC ≈ 0.03（几乎不相关），对齐后 NCC ≈ 0.68。窗口中心与模型零级不一致是确定的。
- 使用实际相位喂模型后，round_0001 增益 0.159（对比原始相位的 0.957），但仍高于实测 0.0005。方向一致（改善），幅度仍有约 30 倍的差距。
- **未决门禁**：离线诊断已固化为 JSON 可复现，但模型-实测增益差异的完整根因仍需在设备就绪时验证（SLM shift_phase 是否接入、光路中心是否重测等）。

## 2. 关键数值

参见 `artifacts/offline_diagnostics_20261103.json` 中的 JSON 对象。以下摘录关键条目（单位与 JSON 严格一致）：

|ラウンド | measured gain | model_raw_zero_shift | model_effective_aligned | model_effective_zero_shift | alignment shift (px) |
|--------|--------------|----------------------|--------------------------|------------------------------|----------------------|
| reference | 2.06530 | 2.21983 | 2.04294 | 2.06530 | [-3, +32] |
| round_0001 | 0.000509 | 0.956801 | 0.078394 | 0.159100 | [-3, +33] |
| round_0002 | -0.000242 | 0.956801 | 0.075960 | 0.159100 | [-3, +32] |
| round_0003 | -0.001094 | 0.956801 | 0.075960 | 0.159100 | [-3, +32] |
| round_0004 | -0.003353 | 0.956801 | 0.078601 | 0.159100 | [-2, +32] |

## 3. 捕获会话概要（11-03）

参见 `artifacts/offline_diagnostics_20261103.json` 中的 `capture` 条目。关键帧统计：

| 帧 | sum | max | p99.9 |
|-----|------|------|-------|
| flat_0000 | 1437246.0 | 134.0 | 1.0 |
| calib_screen_000 | 1439581.4 | 112.0 | 1.0 |
| shape_ring_nominal_r0 | 1511501.7 | 18.0 | 1.0 |
| shape_cross_nominal_r0 | 1516702.4 | 13.67 | 1.0 |
| shape_frame_nominal_r0 | 1516409.7 | 18.0 | 1.0 |
| shape_two_spot_nominal_r0 | 1484399.7 | 17.0 | 1.0 |

模型对比（reference vs shaped）：
- NCC vs flat: ring 0.3116, cross 0.2661, frame 0.3180, two_spot 0.3043
- pairwise shaped NCC: ~0.25 之间各两两

## 4. CCD 采集帧可视化（11-03 会话）

- 全部 100 帧（平场/标定屏/probe/shape）按采集顺序回放，每帧裁到以记录零级为中心的 256 px 窗口：![采集帧序列](images/ccd_capture_sequence_20261103.gif)
- 10 个 shape 帧（5 类目标 × 2 重复）回放，去背景后按 0–20 count 显示：![形状帧序列](images/ccd_shape_sequence_20261103.gif)
- 左图：每帧窗口峰值 vs 去背景总能量（按类别着色）；右图：质心相对记录零级的偏移（flat/probe/shape）：![帧质量散点](figures/ccd_stats_scatter_20261103.png)
- 逐目标能量落入支持域（r0/r1 两个重复；黑线为支持域面积占比）：![支持域能量](figures/shape_encircled_energy_20261103.png)
- 可视化资产：[采集帧 GIF](images/ccd_capture_sequence_20261103.gif)、[形状帧 GIF](images/ccd_shape_sequence_20261103.gif)、[帧质量散点图](figures/ccd_stats_scatter_20261103.png)、[支持域能量图](figures/shape_encircled_energy_20261103.png)；生成脚本 [`scripts/ml/sim2f_fov_hw_figures.py`](../../../scripts/ml/sim2f_fov_hw_figures.py)。

## 5. 下一次实验准备

- 离线诊断结果已保存为 `artifacts/offline_diagnostics_20261103.json`，可在设备就绪时复现。
- 若要进一步缩小模型-实测增益差距，下一步硬件工作应：
  1. 测量 SLM 实际光束中心（`panel_locate` 或 `find_beam_on_slm`），并将 `--pupil-center-xy` 设为测得的面板 px 位置。
  2. 将 SLM `shift_phase` 参数同时写入模型前向传播，保证模型与硬件看到的相位一致。
  3. 完成光路标定，确认零级相对窗口的精确平移（已知需要 (-3, +32) px 校正）。
- 当前硬件检查点：设备下线，未取得 CCD、SLM 与光路现时可用性的确认；本轮未运行真实 `sequential_ccd`，也未启动硬件采集。
