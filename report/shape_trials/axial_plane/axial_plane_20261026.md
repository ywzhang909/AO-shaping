# SLM 偏离前焦面的物理模型校正（2026-10-26）

<!-- provenance:start -->
> **生成脚本**: [`scripts/slm/generate_axial_plane_report.py`](../../../scripts/slm/generate_axial_plane_report.py)
> **复现命令**: `python scripts/slm/generate_axial_plane_report.py`
> **数据/关联脚本**: [`src/ao_shaping/tools/slm/transfer/axial.py`](../../../src/ao_shaping/tools/slm/transfer/axial.py)
> **数据/关联脚本**: [`src/ao_shaping/tools/slm/transfer/replay_phases.py`](../../../src/ao_shaping/tools/slm/transfer/replay_phases.py)
> **数据/关联脚本**: [`src/ao_shaping/algorithm/signal_processing/defocus_forward_core.py`](../../../src/ao_shaping/algorithm/signal_processing/defocus_forward_core.py)
> **运行环境**: 实验（Santec #1、Daheng Mono12，SLM 输入平面轴向校正）
> **说明**: ABCD 条件距离扫描、相位分组验证与可辨识性边界。
<!-- provenance:end -->

已在冻结物理正向模型增加输入平面轴向偏移校正，并在当前光路重新采集不同相位的 CCD 图像。实验未证明该校正可以显著改善当前预测：条件最优偏移位于扫描边界，验证集改善很小，故没有自动采用该偏移。

## 物理关系与可辨识性

SLM—透镜距离为 d₁，透镜—CCD 距离为 d₂，薄透镜焦距为 f，近轴传播矩阵为 `P(d₂)L(f)P(d₁)`：

```text
A = 1 - d₂/f
B = d₁ + d₂ - d₁*d₂/f
C = -1/f
D = 1 - d₁/f
```

在 CCD 后焦面 d₂=f 时，A=0、B=f，与 d₁ 无关。输入平面移动只改变输出二次相位，强度不变；因此 SLM 不在前焦面本身，不足以否定傅里叶强度模型。这个结论假定近轴传播、薄透镜、口径足够大、无截光。

CCD 离焦时 A≠0。令当前校准的有效输入曲率 q=A/B，已测载波映射正比于 B。输入平面到透镜的传播距离增加 δ 后，矩阵右乘 P(δ)，故：

```text
s = 1 + q*δ
q_new = q/s
ramp_matrix_new = ramp_matrix*s
optical_gain_new = optical_gain/s²
```

传播曲率 q=A/B 应与入射波前的二次相位分开。检查点可用 axial_reference_curvature_diopter 显式给出传播曲率；输入波前的剩余二次相位保持原值。旧检查点没有该字段时，本次条件扫描假设全部 defocus_diopter 都来自传播曲率；这不是已被实验验证的分解，也不是直接测出的 CCD 轴向位移。若已知 CCD 在后焦面，可指定传播曲率为 0，即使入射波前仍有二次相位也不引入虚假的轴向强度变化。输出二次相位模长为一，不影响单平面强度。背景和曝光偏置不乘 s⁻²。实现保留原生 CCD 坐标，未通过缩放 CCD 图像掩盖尺度误差。

如果 q 和载波映射都能自由重新拟合，δ 与这两者退化，单平面强度不能唯一反推出实际 SLM—透镜距离。本次固定标称 q 与映射，仅做条件扫描；任何拟合 δ 都不能直接视为卷尺测量值。有限透镜口径、传播后的截光和透镜局部像差没有包含在此 ABCD 校正中。

## 代码与兼容

共享核心增加 axial_plane_correction；冻结检查点增加 slm_axial_offset_m，缺省 0，旧模型行为保持兼容。effective_parameters 将距离吸收到有效曲率、载波映射和光学增益中；验证入口及再拟合入口先归一化，避免再次拟合几何时重复校正。FrozenDefocusForward 的相位梯度保持可微。

通过 ABCD 直接矩阵乘法、在焦不变性、等效参数预测一致性和相位有限差分梯度检查。新增参数不要求新增神经网络权重。

## 当前实验与数据分组

Santec #1，1064 nm；Daheng Mono12、满量程 4095，原始 uint16 数据完整保留。使用前一轮已知相位，在 50/100 ms 各采 7 组、每组 5 帧，共 70 帧。零级外的目标核心均未饱和，核心排除半径 60 px 的零级区域、边缘内缩 8 px。1200 CCD 像元以 2.2 µm/像元计为 2.64 mm。

按相位数组 SHA256 分组：4 种不同相位，相同相位的不同标签和不同曝光始终属于同一分组，避免跨曝光重复数据进入训练与验证两边。扫描在原有 2 px 相机网格上每 4 点采样一次，即本次拟合空间间隔 8 原生 CCD px；不是像元积分核大小，也不是 SLM 网格间隔。窄散斑因此不一定完整解析，本轮结果仅是此固定采样口径下的条件比较。

每个距离候选均只在训练相位上用 NNLS 重新拟合光学增益、加性背景（杂散光）项和曝光偏置，共享 counts/ms 模型为 gain*optical+background+bias/exposure。零距离基线使用相同拟合程序，所以表中差异不是单纯重新调整曝光响应带来的收益。

| 条件偏移/m | 训练相对 L2 | 验证相关性 | 验证相对 L2 | 验证 CV 绝对误差 |
|---:|---:|---:|---:|---:|
| -0.2 | 0.374387 | 0.118489 | 0.398523 | 0.371607 |
| -0.15 | 0.374481 | 0.116415 | 0.398624 | 0.372267 |
| -0.1 | 0.374585 | 0.114250 | 0.398727 | 0.372995 |
| -0.05 | 0.374697 | 0.111996 | 0.398830 | 0.373786 |
| 0 | 0.374817 | 0.109670 | 0.398934 | 0.374639 |
| 0.05 | 0.374944 | 0.107270 | 0.399038 | 0.375545 |
| 0.1 | 0.375076 | 0.104799 | 0.399141 | 0.376504 |
| 0.15 | 0.375214 | 0.102265 | 0.399243 | 0.377511 |
| 0.2 | 0.375356 | 0.099662 | 0.399345 | 0.378568 |

训练目标选择偏移 -0.2 m；边界解 `True`；验证相对 L2 改善 0.1030%；自动采用 `False`。

候选参数单独保存在 candidate_parameters.json；frozen_parameters.json 在本次未通过采用条件时保留零距离偏移及相同 NNLS 重拟合基线。采用规则要求内点最优、验证相对 L2 改善超过 0.5% 且 CV 误差降低。验证集参与这一选择，不能再当成独立最终测试集。

## 发现与后续

1. 单独加入 SLM 轴向偏移没有解决本轮宽区域预测误差；验证相关性仍很低，预测 CV 仍显著小于 CCD CV，不能把条件扫描的微小改善称为正向模型已经准确。
2. 下午探索组最佳相位在本次重放时不再优于所有旧相位。历史最优依赖当前光路状态，应保存并重测相位库，不应永久锁定一个最佳相位。
3. 下一步首先补充当前载波正负探针和二次相位探针，独立重估 CCD 映射与有效曲率；若仍有位移相关的功率损失，再测透镜有效口径和入射位置，建立 SLM→自由传播→透镜相位/口径→CCD 的双段传播模型。没有已知焦距、口径和距离依据时，不应同时拟合三个未知距离并将其解释为实际位置。
4. 有限口径模型需要检查透镜面场的采样收敛、补零和边界截断，并重新验证相位梯度；不能用过强的图像模糊替代截光。
5. 若只需恢复整形，可采用当前新测几何、有效曲率和 CCD 锚定短块优化；实际距离可保持未知。模型预测和 CCD 验收需分别报告，不因模型拟合损失下降就认定均匀度提高。

## 复现

```powershell
python -m ao_shaping.tools.slm.transfer.axial `
  --checkpoint data/debug/slm_transfer/square1200_20261026_1350/model_wide640/frozen_parameters.json `
  --capture data/debug/slm_transfer/axial_plane_20261026/capture `
  --out data/debug/slm_transfer/axial_plane_20261026/new_scan --device cuda
python scripts/slm/generate_axial_plane_report.py
```

采集数据保存在 data/debug/slm_transfer/axial_plane_20261026/capture；两次扫描都保留，scan_checked 是正式报告来源。每个 offset_*.npz 保存光学预测和经过响应拟合的预测，inputs.npz 保存测量图、坐标、掩膜与分组，iterations.pkl/scan.json 保存每个距离候选的起止主机时间和全部指标。capture 的每个 NPZ 保留完整 CCD 原始帧与加载相位，逐帧曝光和主机时间在 manifest.json。

设备释放 UTC：`2026-10-26T09:42:25.296357+00:00`，状态 `camera and SLM contexts closed`。本次没有把未验证的边界距离用于新的闭环整形。

[全部量化结果与采集记录](artifacts/20261026/evidence.json)。
