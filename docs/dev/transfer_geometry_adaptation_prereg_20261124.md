# Transfer 物理预测模型几何残差适配预注册 · 2026-11-24

> **状态**：在本轮新增采集和拟合开始前冻结。采后不得改模型边界、split、门槛或主要指标；若实现限制迫使变更，须先记为偏离并把结果标为探索性。
>
> **范围**：只检验 signed-ramp 全局 2×2 几何标定之后，是否还存在可由小幅全局仿射残差校正的旋转、剪切或各向异性尺度误差。不能代表任意位置相关畸变、镜头畸变、散斑/静态 PSF、运动模糊、相机非线性或目标形状 CCD 验收。

## 1. 依据与边界

- 现有 `fit.geometry()` 根据正负 signed-ramp 对估计全局 `ramp_matrix`；`PhysicalForwardFit` 目前固定其逆矩阵，transfer adaptation 不学习几何。
- 2026-11-18 `20261118_transfer_mil_zernike_probe_v1` 的 56 个 train / 12 个 test phase 上，现存 baseline 的 test 均值为 correlation 0.6747、relative L2 0.7230；full-fit 为 0.7982、0.5657。该数据没有 validation split，且旧比较不能证明新的几何自由度有效，**只用于确定问题背景，不作本轮成功判据**。
- 历史 Zernike 扩基试验未移动平滑/带限预测地板；新增几何参数不能替代成像展宽或静态 PSF 建模。
- `2026-11-23` issue247 波形窗口 verdict 为 `tilt_specific` 且几何 gate 未通过，不把它当作已校准的通用 ramp matrix。

## 2. 采集设计

设备：Santec SLM #1（1064 nm）、Daheng MER2-507-23GM cam 0、1064 nm 激光。遵守 `docs/dev/hardware_discipline.md` 与 `libs/skills/ao-optical-path-calibration/SKILL.md`；设备活动须在完成回复中报告占用、操作、释放时间与关闭结果。

1. 采集前运行光路有效性探测；只有 verdict=`ok`、设备锁可取得、位深回读正确、无饱和且相位调制响应可辨认时才继续。任何门失败均停止并标为 `not assessable`，不靠调曝光或挑帧补救。
2. 训练/验证会话输出目录：`data/debug/slm_transfer/20261124_transfer_geometry_v1/`（必须原先不存在）。signed-ramp 周期固定为 64、128 px；capture 标准正负 x/y ramp 作为 calibration。其余 phase 按采集器既有规则分为 train / validation，fit 仅以 train 优化、validation 仅用于选择/停止。
3. 几何独立探针会话输出目录：`data/debug/slm_transfer/20261124_transfer_geometry_period192_holdout_v1/`（必须原先不存在）。capture CLI 会按固定顺序额外生成 reference 与 6 个普通 phase；冻结判据只使用其中 192 px 的 x/y 正负 ramp 行，其他行不参与几何检验。该会话不得用于拟合参数、选择 checkpoint、调阈值或修改矩阵。离线只在全部参数冻结后比较 train 会话估计出的基础矩阵与适配矩阵，对该会话实测位移的预测误差。若该会话期间设备/光路状态变化或采集门失败，则该检验记为不可评估。
4. 新输出目录不可复用；不覆盖原始或既有 capture。采集故障留下的目录不作为下一次重跑目录。
### 执行事实附录（采集后追加；不改变上方冻结的门槛或指标）

- 前置光路探测 `data/debug/optical_path_probe/transfer_geometry_20261124/20261124_031312Z/optical_path_probe.json` 的 `verdict.state=ok`、退出码 0，读取 `cam_bit_depth=12`；曝光 0.05 ms 峰值 2601/4095 未饱和，离焦宽度变化 19.3% / 峰值 21.2%。SLM #1 与 Daheng cam 0 于约 11:13:13–11:15:29 运行探测并正常关闭；其后确认设备锁未持有。
- 训练/验证会话 `20261124_transfer_geometry_v1` 有 `.capture_done`、22 行（10 calibration、6 train、6 validation）、曝光 0.063 ms、相机回读 `Mono8` / 8 bit，ramp/reference `light_check.json` 为 `ok`；设备锁约 11:16:20 取得，11:20:31 采集结束并释放，SLM #1 与 Daheng cam 0 在脚本退出时正常关闭。
- 独立 held-out 会话 `20261124_transfer_geometry_period192_holdout_v1` 有 `.capture_done`、12 行（6 calibration、6 普通 probe），唯一 192 px x/y ± ramp 均存在，曝光 0.063 ms、`Mono8` / 8 bit，`light_check.json` 为 `ok`；设备锁约 11:25:41 取得，11:27:17 采集结束并释放，SLM #1 与 Daheng cam 0 在脚本退出时正常关闭。结束后再次确认设备锁未持有。
- 前置探测的 12-bit 回读与两个 capture 会话的 Mono8/8-bit 采集设置不同；其绝对亮度不作跨会话对比。拟合与 held-out 门使用各自会话内 ramp 位移与预测值，不以亮度作为几何结论。
- 首次 held-out CLI 尝试 `--count 0` 被参数校验拒绝（要求 `count >= 6`），未创建输出目录或打开设备；锁立即释放。正式运行改用 `--count 6`，输出目录此前仍不存在。此为 CLI 限制，未改变 held-out ramp 判据。
- 最终对照产物目录为 `arm_A/` 与 `arm_B/`。初次 `fit_baseline/` / `fit_geometry/` 运行后，聚焦回归发现 capture 内存 manifest 尚无 `split` 键时被 calibration-only 过滤误拒；仅补充“未声明 split 时按历史 calibration 行为处理”的兼容分支后，以同一数据、checkpoint、预算和命令重跑两个变体。`arm_A` / `arm_B` 的 iteration、loss、逐 phase metrics 和 parameters 与初次两个变体逐项一致，非重新挑选结果；`evaluate_geometry.py` 只读取 `arm_A` / `arm_B`。

## 3. 模型对照与防泄漏

- 对照 A：`learn_geometry=False`，维持 calibration ramp matrix 固定，其他初始化、训练数据、损失、辐射度设置、随机种子、迭代预算与优化器设置与 B 相同。
- 对照 B：`learn_geometry=True`，只增加零初始化、逐元素有界的 2×2 小残差校正；预注册最大偏离为 5%（实现应使每个矩阵元素的校正受该上界限制）。不释放其他额外自由度，不改变 checkpoint schema。
- 基础矩阵只由 train 会话中 `split="calibration"` 的 signed ramps 决定；几何独立探针会话与所有非 calibration 行严禁参与矩阵估计。train phase 只用于拟合；validation phase 可用于模型选择，但不得用来报告最终独立结论。若工具实现无法把选择数据与最终独立评分数据分开，须明确报告限制，不得称为最终泛化验证。
- 记录采集器、拟合器、分析器脚本 SHA256，manifest/checkpoint SHA256、split 数、有效行数、设备位深/曝光、完整 ramp_samples、基础/校正/有效矩阵以及原始逐 phase 指标。
- 采集结束后、开始拟合前追加并冻结的两个变体命令（同一 checkpoint `D:/Projects/TIFO/AO-shaping/data/debug/slm_transfer/defocus_forward_20261107_focus_r2_fit/checkpoints/M3_illumination/frozen_parameters.json`、dataset `20261124_transfer_geometry_v1`）：`python -m ao_shaping.tools.slm.transfer.fit --dataset <dataset> --checkpoint <checkpoint> --out <arm> --span 160 --iterations 100 --adaptation full --radiometry exposure --device cpu`，变体 B 额外加 `--learn-geometry`。选出迭代仍沿用工具的 train-loss 规则；不得用 validation 损失调参或回退到另一 iteration。脚本、manifest 和 checkpoint 哈希在两个变体 provenance 中归档。

## 4. 冻结判据

几何适配仅在以下条件全部满足时判为支持；否则保留默认关闭并报告未通过/不确定：

1. **训练会话 validation**：以 phase 为配对单位比较 A/B。主要指标为未缩放图像 relative L2，B 相对 A 的均值至少改善 5%，且 paired bootstrap 95% CI 不跨 0；辅助指标 correlation 的均值下降不得超过 0.01。固定 seed `20261124`、20,000 次 phase-level bootstrap，不以单帧为独立样本。
2. **独立 192 px ramp 会话**：对 x/y 两列位移分别计算预测误差，B 的合并归一化位移 RMSE 必须低于 A；同时 B 不得使任一轴误差增加超过 10%。若改善方向不一致，判为不确定，不挑轴报告成功。
3. **参数有效性**：所有值有限，校正满足 5% 界；训练与验证均有足够有效帧且无饱和。未满足质量门的样本不进入分母，整个检验需按预注册规则标为 `not assessable`，不事后换样本。
4. 不允许只凭训练损失、样本内图、单一典型 phase 或最佳 checkpoint 的单个指标宣称泛化提升；不得将验证集反复用于参数/门槛设计后仍称其为独立验证。

## 5. 结果解释

通过上述门只支持“该装调状态与本轮图样范围内的小幅全局线性几何残差改善 transfer 预测”的结论，不自动推广到其他日期、设备、目标形状或位置相关误差。目标形状 CCD 验收仍需单独预注册、独立实测。若几何适配未改善而静态 PSF/blur 的历史迹象仍在，应优先调查成像观测模型，不扩大 Zernike 或几何参数自由度。
