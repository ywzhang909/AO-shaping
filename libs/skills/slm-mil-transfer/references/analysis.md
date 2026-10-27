# 实验后：先核对数据，再解释优化

## 三层证据分开

1. **数值实现**：拟合与冻结预测一致；梯度有限差分通过；网格与 FFT 加密、CCD 像素积分收敛。
2. **当前相位族预测**：相关性、相对 L2、核心均值/方差/CV、质心、峰宽和能量；按相位族与曝光分别报告。
3. **实际整形**：独立 CCD 候选测量及最终精确相位重放。上一层通过不代替下一层。

## 诊断决策表

| 现象 | 下一步能区分原因的检查 |
|---|---|
| 相关性高，CV 偏差大 | 拆开增益、背景和曝光偏置；分别查看整体窗口与核心 |
| 加密采样后误差不变 | 停止盲目增加 FFT；检查振幅、PSF、相位族与空间背景 |
| 模型内部梯度通过，CCD 方向相反 | 相同相位正负响应；检查物理 Jacobian，不归咎自动微分本身 |
| 载波后低 CV、低 counts | 核对二维映射和目标区域，防止在暗区域测“均匀” |
| 训练改善，最终整形相位预测变差 | 采集目标相位附近局部扰动，新增方向用于验证 |
| 恢复相同相位仍变差 | 检查曝光、功率、入射位置、稳定时间与时间顺序；不是直接证明局部最优 |
| 距离拟合最优在边界 | 不将它解释为测得距离；检查退化、口径和新几何 |
| GIF 与最终图不同 | 核对运行采集/重放、曝光、文件、掩膜、归一化及坐标 |

Pearson 相关性对正增益和常数偏置不敏感；CV 对均值偏置敏感。仅用相关性选择模型会错过曝光误差，也可能掩盖目标核心的外推失败。固定标度下比较预测和 CCD，不能逐图 min-max 归一化后再宣布强度一致。

## 使用配套 Python 工具

`session_review.py review --run DIR` 只读 manifest/config/summary，列出实际曝光、位深、采集数量、指标范围、接受模式、缺文件和时间问题。它不会把缺少 release.json 自动当成设备仍被占用，也不会用 JSON 中的“成功”替代 SDK 释放证据。

`--check-raw` 使用 `np.load(..., allow_pickle=False)` 流式检查记录文件，并复用 `ShapeObjective`、`image_metrics` 重算保存掩膜上的采样 CV。它检查目标/坐标/核心是否匹配、非有限数组、饱和核心、相位哈希和训练/验证重复。此结果是**采样核心 CV**，不冒充原生全分辨率 CV。没有原始 core/target/camera_xy/model_phase 的旧数据无法完成此检查，应保留失败而非猜测。

`--audit` 委托 `transfer.artifacts.audit_run` 检查完整优化会话的原始指标、验收决策与探索最终导出；不是第二套验收实现。不能用于只有重放数据的目录。

`compare --run REPLAY --baseline-label BASE --candidate-label CAND --check-raw` 按相同曝光比较两个标签，要求每标签每曝光恰好一组，拒绝重复模糊匹配与掩膜不一致；记录每档 CV 变化和信号变化。跨曝光不平均后宣称是相位效应。若需将多档曝光用于选择，公开规则并承认这些数据已参与选择。

示例（在 AO-shaping 根目录）：

```powershell
$tool = 'libs/skills/slm-mil-transfer/scripts/session_review.py'
$data = 'data/debug/slm_transfer/square1200_explore_20261026_1435'
python $tool review --run "$data/nonmonotone" --check-raw --audit --out data/debug/mil_skill_review/nonmonotone.json
python $tool compare --run "$data/common_replay" --baseline-label mono12_dense_square_final --candidate-label nonmonotone_square_final --check-raw --out data/debug/mil_skill_review/comparison.json
```

输出已存在时换新路径。默认只打印 JSON；无依赖时先用元数据模式。工具不自动读取历史 pickle；如需加载迭代 pickle，只处理已信任的本项目产物，通过既有 `transfer.progress` 等入口分析。

## 交付与复现

报告记录当前设备与配置来源、原生像元/采样网格、掩膜、曝光与实际满量程。列出训练、选择、独立测试分别用了哪些数据。保存模型参数轨迹与损失、相位变化和 CCD 指标，附已失败假设及其证据。

静态图与 GIF 消费同一份源记录索引；每帧标阶段、标签、曝光和指标。5 帧均值的 CV 与逐帧 CV 不是同一数值，明确统计口径。保留真实 CCD 采集帧，模型迭代之间没有 CCD 数据时不能插值伪造测量过程。重新渲染只写派生资产，不覆盖原始帧或默默重选最佳相位。

交接下一次启动所需的当前检查点、参考相位、已测几何、曝光响应、未验证参数及一条可执行命令；不承诺“任意光路一键保证成功”。当前参考变化较小可快速适配，变化超出验证范围则回到新几何与探针。记录设备释放状态，避免下一位操作者误判仍在占用。
