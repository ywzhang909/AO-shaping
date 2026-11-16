---
name: slm-mil-transfer
version: 1.1.0
description: Plan, conduct, and analyze SLM beam-shaping experiments using model-in-the-loop (MIL), compact physical forward models, simulation pretraining, and transfer learning. Use for optical-path adaptation, exposure-aware calibration, differentiable phase optimization, or diagnosis of predicted versus measured uniformity. Each experiment phase (planning, in-experiment, post-experiment) produces its own structured report with phase/far-field images following the model_in_loop_live reporting conventions.
---

# SLM MIL 与迁移学习实验

目标：用可验证的物理预测模型适配当前光路，通过可微相位优化获得目标形状，并用 CCD 数据确认效果。经验来自 AO-shaping 的 MIL、离焦、多形状、冷启动与 1200 像元匀化实验；历史配置是实例，不是新光路的默认标定。

按任务所在阶段读取对应参考，不必一次执行全部流程。开始前遵守目标仓库的 AGENTS.md 与术语规范。此 skill 不扩大用户授权；旧会话的设备型号、在线状态、独占权限和曝光限制不能当作新实验事实。

## 1. 规划

读取 [规划指南](references/planning.md)。先将目标写成可测条件：形状、原生 CCD 尺寸、核心掩膜、零级处理、优化优先级、参考相位、曝光比较方式和可用实验时间。

整理当前可信资产，区分标称模型、仿真预训练、当前光路拟合、历史最佳相位。旧模型可作初始化，历史最佳必须重测。不同光路、不同相位族和同相位不同曝光按组划分，不按像素随机拆分。

计划应回答两个不同问题：模型能否预测当前相位族；模型的局部响应能否指导 CCD 改善。内部有限差分通过只回答代码导数问题。

若已有 transfer 数据，可先运行 `scripts/session_review.py review --run RUN`。脚本只读 JSON，默认不加载数组；输出是资料检查，不代表通过设备验收。

## 2. 实验中

读取 [设备使用与在线决策](references/experiment.md)。该阶段可直接使用已有 CLI、GUI 或实验程序，不要求使用配套脚本。优先在设备可用时获取能区分假设的数据，耗时训练和图像渲染留在离线阶段。

基本顺序：低曝光当前参考 → 新测几何/入射位置 → 跨相位与曝光探针 → 少参数适配 → 新相位验证 → 短块可微优化 → 独立 CCD 验收 → 精确相位重放与释放。根据当前证据跳过已有可信环节，避免每次重做全套分析。

不要将“SDK 能枚举”“模型损失降低”“候选被模型接受”写成“光路稳定或 CCD 效果改善”。曝光可在块间自适应；块内参考与候选应同曝光，并向模型传递实际读回曝光。可选 A–B–B–A 用于漂移明显时，不是所有实验的强制配置。

## 3. 实验后分析

读取 [诊断、验收与交接](references/analysis.md)。先重算原始数据，再讨论模型。核对固定掩膜、原生像元尺度、实际曝光、满量程、同相位重复和时间顺序；将训练/选择/最终测试的角色写清楚。

配套脚本：

```powershell
python libs/skills/slm-mil-transfer/scripts/session_review.py review --run RUN
python libs/skills/slm-mil-transfer/scripts/session_review.py review --run RUN --check-raw --audit
python libs/skills/slm-mil-transfer/scripts/session_review.py compare --run REPLAY --baseline-label BASE --candidate-label CAND --check-raw
```

`RUN` 是含 `manifest.json` 的采集目录，不是上层整个实验目录。`--audit` 仅适用于完整 `transfer.validate` 会话；重放目录使用 `--check-raw`。脚本不会加载 pickle、打开设备或修改源数据；仅在显式 `--out NEW.json` 时写新文件，拒绝覆盖已有文件。原始帧检查需 NumPy 和 AO-shaping 包，元数据检查仅需 Python 标准库。实际命令和边界见脚本 `--help`。

最终交付应包含：新光路下可复现的配置、原始相位与 CCD 帧、曝光与时间记录、模型检查点来源、固定掩膜指标、接受/拒绝轨迹、同曝光复测、失败假设、设备释放状态和下一步最有区分力的实验。按本仓库要求登记报告溯源；GIF 只显示真实采集步骤，优化时段与后续重放明确分段。

## 各阶段报告（强制）

规划、实验中、实验后三个阶段各自产出一份独立报告，内容、图片与口径见 [各阶段报告与图片规范](references/reporting.md)。该规范以 `report/shape_trials/model_in_loop/model_in_loop_live_20261014.md` 为样本，但是通用骨架，不绑定某次具体实验。

每篇报告必须包含**当前可信事实**小节：设备型号与在线状态、SLM 波长与 LUT、CCD 像元尺寸与格式、光路是否离焦、几何与入射来源、曝光范围与实际读回满量程、已获授权与时长，并逐项标注本次实测 / 沿用历史 / 待定。**历史数值只作经验来源，不得继承为当前默认标定。** 所有具体数值（设备型号、波长、像元尺寸、曝光、几何、像素格式）均为历史会话实例，新实验必须在报告中显式记录当前实测值。

报告是人工撰写的结构化叙述，不是自动生成文件；`scripts/session_review.py` 产出的 JSON 是报告的输入数据之一。图片遵循规范中的渲染口径（相位 `twilight`、远场 `inferno`、X/Y 过质心剖面、目标轮廓对照），并写明渲染图与原始 CCD 强度数组的口径差异。

## 历史依据

需要确定经验来自何处、哪些结论尚未证明时，读取 [历史证据与适用边界](references/history.md)。不要将其中一次实验的数值或阈值写成所有设备的固定要求。
