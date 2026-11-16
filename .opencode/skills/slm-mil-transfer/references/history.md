# 历史依据：保留结论的适用范围

下列路径相对 AO-shaping 仓库根。skill 是决策指南，报告是具体实验数值的事实来源。读取相关报告时优先查看纠正记录与后续复测。

| 阶段 | 报告 | 影响后续决策的经验 |
|---|---|---|
| MIL 早期离焦拟合 | `report/shape_trials/model_in_loop/physical_forward_defocus_live_20261022.md` | 共享几何、入射振幅、有效曲率与残余像差；相关性通过不等于 CV 预测通过 |
| 传播采样诊断 | `report/shape_trials/model_in_loop/physical_forward_sampling_20261022_r2.md` | 固定参数加密网格/FFT/像元积分；有效 PSF 可吸收部分峰宽，不能任意解释为器件参数 |
| 多形状纠正 | `report/shape_trials/multishape_capability/diagnosis_and_transfer_20261023.md` | 历史 CV=0.1530 来自错选的低信号区域；实际载波主要沿 CCD 负行方向，二维几何比假设轴方向更可靠 |
| 可复现多形状 | `report/shape_trials/multishape_capability/physical_multishape_reproduction_20261023.md` | 当前光路拟合、冻结推理、目标相位合成和 CCD 验收应在同一物理口径下实现 |
| 仿真先验与无 GS 复现 | `report/shape_trials/sim_pretrain_transfer/sim_pretrain_transfer_20261024.md` | 已接受相位起步的逐位复现不是冷启动；仿真先验与历史相位复现实验的检查点不能混为同一个结果 |
| 冷启动 | `report/shape_trials/cold_start/cold_start_20261026.md` | 平面/随机加当前载波；损失降低仍可能损害 CV 预测；初始化来源需实际审计 |
| CV 曝光诊断 | `report/shape_trials/cold_start/cv_diagnosis_20261026.md` | 引入 counts 偏置后整体窗口 CV 误差改善，但四组最终相位核心 CV 误差从 0.122984 变为 0.205925；不能只看整体泛化 |
| 大目标与 Mono12 | `report/shape_trials/square1200/square1200_20261026.md` | 1200 原生像元目标需足够控制带宽和视场；2.2 µm 与 12 bit 为当次设备事实 |
| 非单调探索与重放 | `report/shape_trials/square1200_exploration/exploration_20261026.md` | 50 ms 同曝光重放从旧相位 CV=0.385327 到探索相位 0.309347，但时序混杂不允许将全部收益归因于接受恶化；历史最低运行值不能代替重放 |
| SLM 非前焦面 | `report/shape_trials/axial_plane/axial_plane_20261026.md` | ±0.20 m 条件轴向校正只改善验证相对 L2 0.103%，最优在边界，未采用；单平面强度不能唯一确定实际距离 |

## 从历史数据归纳的模型层次

原 MIL 路径在 `optimizer/wfless/slm_model_in_loop.py` 与 `runners/slm/model_in_loop_runner.py`：Step A 用相位探针拟合 Zernike/入射误差，Step B 冻结模型合成目标相位，逐轮信任域与验收降低两者耦合。后来 transfer 将采集、少参数拟合、仿真先验、相位初始化、可微优化和 CCD 验收拆成有独立产物的阶段，统一入口位于 `tools/slm/transfer/workflow.py`。

两条路径都属于模型在环思路，但参数、初始化和指标口径不能混用。继续旧 MIL 会话时先查其配置与保存格式；使用 transfer 数据工具时确认存在受支持的 manifest/NPZ 字段，不把旧 pickle 强行当作新记录。可复用的经验是阶段间明确冻结与更新哪些量，保留同相位响应对照，以及将模型拟合的成功与 CCD 验收分开。

名义物理传播 → 当前二维几何与振幅 → 少量相位像差 → 曝光感知响应 → 局部 CCD 锚定 → 目标相位附近的响应验证。每次只增加可识别因素，用同一组数据保持对照，并另留最终独立 CCD 测量。

`F(φ)` 绝对预测有偏时，锚定代理可以改善目标，但其 Jacobian 仍可能错误。局部响应检查点不应当成全局强度模型。高质量模型的判据是跨目标相位族可预测，且生成的候选相位能在 CCD 上改善，不是参数数量或训练损失本身。

## 距离误差的特别边界

薄透镜 ABCD：`A=1-d₂/f`、`B=d₁+d₂-d₁*d₂/f`。CCD 在后焦面时 A=0、B=f，SLM 前焦面距离改变只产生输出相位变化；无截光条件下强度不变。CCD 离焦时轴向位置才影响 A/B 与载波尺度。

传播 A/B 与入射二次波前必须分开；不能因为拟合得到二次相位，就直接当成 CCD 离焦距离。若有截光、有限透镜口径或透镜空间像差，ABCD 无口径近似不够，应测相关几何并建立双段传播。边界解与更多自由参数不是证据。
