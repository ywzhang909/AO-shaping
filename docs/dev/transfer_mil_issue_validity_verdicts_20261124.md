# Transfer shaping / model-in-loop 改进点有效性总账（2026-11-24）

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本。
> **数据/关联脚本**: [`issue_validity_verdict_20261119.md`](issue_validity_verdict_20261119.md)、[`historical_measured_shape_metrics_reanalysis_20261119.md`](historical_measured_shape_metrics_reanalysis_20261119.md)、[`transfer_psf_adaptation_prereg_20261124.md`](transfer_psf_adaptation_prereg_20261124.md)、[`hw_window_20261124_issue247_cross_session_prereg.md`](hw_window_20261124_issue247_cross_session_prereg.md)。
> **运行环境**: 只读 GitHub、仓库文档与冻结数据；本轮仅运行纯离线 pytest/ruff/--help。未打开或占用 Santec SLM #1、Daheng CCD cam 0、激光，未下发相位，未采集新帧。
> **说明**: 汇总 GitHub transfer/MIL 直接相关改进项的三层有效性；最新 issue 评论优先于旧报告。预测指标、仿真指标和工程测试均不等于实测整形或效率收益。
<!-- provenance:end -->

## 1. 判定契约

- **物理/机制成立**：只说明现象、诊断通道或表达能力在指定条件下存在。
- **局部标定有效**：只说明冻结验证/独立验证集上的前向预测或标定改善。
- **实测整形或效率收益有效**：必须由可归因、同起点、同曝光、同预算、交错配对的硬件证据支持；训练损失、单次回放、历史 seed→final 和仿真加速均不能替代。
- 主结论只使用 **已证实有效 / 已证实无效 / 证据不足**。含局部结论时写明适用范围，不把局部结论外推为整个 issue 通过。
- 不事后调整阈值；不修改原始数据；不冻结全局映射；不将 `not_assessable` 或“不可归因”写成 fail/pass。
- GitHub 状态以 2026-11-24 只读盘点为准。本轮没有关闭、重开或评论任何 issue，没有提交或 push。

## 2. 总结论

1. **局部有效已证实**：静态 CCD PSF 适配双冻结门通过，可显式 opt-in；`box centroid` 改善零级定位稳定性；per-session refit 改善单会话前向预测；2f 几何标定、MIL 七因子诊断和若干工程门禁/工具已交付。
2. ** scoped 无效已证实**：K 轴多起点在冻结问题上被严格支配；几何残差适配未过双门；#229 的主动探针样本效率主张被深度匹配复测否定；#244 的双屏逐 ROI 前提不适用于当前单 SLM 2f；#237 原批次五项验收失败。
3. **实测整形/效率收益仍未闭环**：#234/#236 的双门是强度 `not_assessable`、方向“不可归因”，不是有效 fail；#137、#139、#165、#213 L-BFGS 和 per-session refit 都没有完成可归因硬件配对终验。
4. **#247 尚未完成跨会话判定**：冻结主判据是 15 个 pinned base id，5 个 p192 在主门为 `not_assessable`、只进补审。离线检查器已修正，但本机没有真实三会话数据根目录，不能填 `reproduced`。
5. **本轮无需关闭新 issue**：#124/#126 的仿真无正向增量、#139 默认带限机制失败都有明确边界，但不足以证明所有实现或修正后的方案无效；其余 OPEN 项更缺终验数据。

## 3. Transfer / 迁移与观测链

| issue | 主 verdict | 已证实范围与证据边界 | 状态/动作 |
|---|---|---|---|
| #229 | **已证实无效（样本效率主张）** | 深度匹配后主动探针在 n=2–32 与随机族无差异；不否定其作为可归因 Jacobian 探针的用途。 | CLOSED，维持。 |
| #230 | **证据不足** | 在线回灌代码和测试属于工程有效；块间重拟合的端到端效率/质量收益未做同预算物理配对。 | OPEN；随 #234/#254 验收。 |
| #232 | **证据不足** | EHGO 迟滞/耦合/温漂估计仅有 sim 方案，尚无对比择优结果。 | OPEN；先 sim 对照冻结矩阵与重标定 oracle。 |
| #233 | **证据不足** | 单调曝光/CCD 响应有物理动机，独立曝光档标定收益未闭环。 | OPEN；不得由单调性直接推出整形收益。 |
| #234 | **证据不足** | 结构化探针、`zernike_block`、在线回灌已实施；P0 条件数 17425、可辨识模率 0.0，无双门通过 checkpoint。 | OPEN；`light/none` 仍暂停。 |
| #235 | **证据不足（含局部有效与根因否定）** | `box centroid` 将历史双瓣定位 SD 0.394 降至 0.297–0.330 px，仅属局部定位有效；换实际相位后方向仍差，否定“输入相位错是根因”。 | OPEN；剩余 LUT、光轴中心和 shift_phase 未验。 |
| #236 | **证据不足** | 旧消融方向相关 −0.043…+0.070 且无区分度；11-23 起强度门 `not_assessable`、方向门不可归因。 | OPEN；不得再引用“双门有效 fail”，δ=0.20 不放宽。 |
| #237 | **已证实无效（原批次验收）** | 原配准/PSF/增益场/LUT/复振幅整包验收失败；11-24 新静态 PSF 协议是新局部证据，不把原整包改判成功。 | CLOSED，维持；新 PSF 单列。 |
| #238 | **证据不足（fold 方法局部有效）** | fold 三周期 g₂π 962.67/952.60/946.89 自洽，但束功率漂移、缺同曝光暗场，仍不可定标；串扰未判定。 | OPEN；不得发布新 g₂π 或宣称串扰有效。 |
| #239 | **证据不足** | attention 代理指标改善但不稳定，对比择优结论 `inconclusive`；不是“已证实完全无效”。 | CLOSED 不启用，维持；不重开。 |
| #241 | **已证实有效（CI 工程）** | transfer 方向门 CI 守卫修复有定向测试证据。 | CLOSED。 |
| #247 | **证据不足（现象局部成立）** | 零级可分离、面板/CCD 轴交换与局部 A 映射有证据；全局轴映射、统一观测校正、跨会话复现和实测校正收益未通过。 | OPEN；见 §7。 |
| #254 | **证据不足（计划有效）** | 三层 split 与配对计划已成文；没有 T6 2×2 交错物理比较，不能宣称数据复用提效。 | OPEN；按 §8 补数。 |
| #213 | **证据不足（含已证实无效子项）** | K=4/8 多起点无效；L-BFGS 第 5 步达 Adam 115 步终态、约 8.9× 仅是冻结离线问题候选；曝光前馈、`torch.compile`、增量 FFT 不采纳，A1–B1 提前判决无效。 | OPEN；只保留完整 ABBA 周期级序贯早停研究。 |
| #221 | **证据不足** | 总账所依赖的方向、freeform NRMSE、形状 CV/EE 均未同时达门。 | OPEN。 |
| #224 | **证据不足** | 透镜辅助场映射只有 sim 提案，未做同目标同预算比较。 | OPEN。 |
| #41 | **证据不足** | 方形家族换 `fov_px` 后 padding 必须重扫；没有合规单变量硬件扫描结果。 | OPEN。 |
| #252 | **证据不足** | GUI 节拍解耦是工程改进，需设备释放、丢弃计数和节拍终验；明确不能解禁 transfer 双门。 | OPEN。 |

### 3.1 11-24 两个观测链适配结论

- **静态 CCD PSF：已证实局部标定有效**。validation rel-L2 `0.60244→0.43019`；独立 capture 主门三行均值 `0.52979→0.41697`，改善 21.296%，paired bootstrap 95% CI `(0.040656, 0.193446)`，相关均值 `+0.085852`。允许显式 `--learn-psf`，默认 `learn_psf=False`；不证明旋转、倾斜、串扰、暗场杂散、CCD 非线性或目标形状整形有效。
- **几何残差：已证实当前方案无效**。validation 改善仅 0.62% < 5%；独立 192 px 门 x 误差 −25.99%、y +18.96%，未过 ±10%。保持 `learn_geometry=False`；不扩大为“位置相关畸变不存在”。

## 4. MIL / 闭环整形

| issue | 主 verdict | 已证实范围与证据边界 | 状态/动作 |
|---|---|---|---|
| #92 | **已证实有效（热启动工程入口）** | prior-run 加载、几何一致性、验证集守卫已交付；热启动效率收益转 #137，尚未证实。 | CLOSED。 |
| #94 | **已证实有效（离线工具）** | 扰动/回放/曝光/带宽/PSF 分析工具交付。 | CLOSED。 |
| #96 | **已证实有效（报告工程）** | 报告重生成与 provenance 同步完成。 | CLOSED。 |
| #107 | **已证实有效（缺陷修复）** | `dataclasses.fields` 导出修复。 | CLOSED。 |
| #112 | **证据不足** | LUT+串扰接入的 freeform NRMSE 验收未完成，实现由 #238 吸收。 | OPEN；与 #238 同步。 |
| #113 | **证据不足** | 3 写入门哨兵未做“光路未动省帧/真动触发升级”的实验系统标定。 | OPEN。 |
| #114 | **证据不足** | 几何一致性校验缺故意错几何的拦截对照。 | OPEN。 |
| #115 | **证据不足** | 现有五角星低分 accepted 说明需要形状验收；Hausdorff/轮廓阈值和独立终验未完成。 | OPEN。 |
| #116 | **证据不足** | 低/高阶两档与高阶哨兵未做独立运行消融。 | OPEN。 |
| #117 | **证据不足** | 有总体预测相关改善，也有全频随机预测和 CV 变差；未按“独立 freeform 验证集选模型 + 新相位终验”验收。 | OPEN。 |
| #118 | **证据不足** | 带宽受限与残差正则有表达力动机，独立 NRMSE/形状 CV 改善未证。 | OPEN。 |
| #119 | **已证实有效（准入门），当前模型不准入** | 10 次运行部分 corr≥0.8，但峰值偏移 13–14 采样、FWHM 比 1.57–2.69，NRMSE 未优于 Zernike；U-Net 近零相关不能替代。 | OPEN；保持闭环封锁。 |
| #120 | **证据不足** | 串扰/非线性/带宽/PSF 注入消融未形成逐运行 freeform 验收。 | OPEN。 |
| #121 | **证据不足（现 U-Net 前向无效）** | 真实帧相关约 0.030±0.041；扩容/残差/混合架构是否改善未证。 | OPEN；先离线，不进闭环。 |
| #123 | **已证实有效（opt-in 实现与 sim）** | 不确定性探针实现、默认关闭和 sim 验收完成；实验收益由 #139 追踪。 | CLOSED。 |
| #124 | **证据不足（仅反向噪声通道成立）** | σ 0.3→0.5 复合分 0.492→0.360，过注入劣化成立；错模型下正向松弛收益未迁移，无正式硬件 A/B。 | OPEN；不据此宣称鲁棒 CGH 有效。 |
| #125 | **已证实无效（重复条目）** | #124 的重复 POST。 | CLOSED。 |
| #126 | **证据不足（表达力缺口成立，联合增益未成立）** | Zernike-only 不能表达三角/星形；joint 与 per-pixel 数值相同，低阶被 freeform 吸收，独立验证集和硬件增益未证。 | OPEN。 |
| #134 | **已证实有效（opt-in 实现与 sim）** | 在线拒绝轮不确定性采样实现和 sim 验收完成；实验验收由 #139 追踪。 | CLOSED。 |
| #137 | **证据不足** | 单次 warm/cold 回放 cold 更好：score 0.72721/0.74061、CV 1.11416/1.00495；继续闭环细化的局部 CV 改善不能归因于热启动。 | OPEN；需 N≥3 同连接 cold/hot 配对。 |
| #139 | **证据不足（默认带限机制已证实无效）** | 默认 σ=1.5 时机制 0/10、p=0.4219；σ=0 时 sim 8/8 改善，但实测进相机能量比仅 0.534，最新硬件对比择优又因几何相关 0.0840<0.5 中止。 | OPEN；先修窗口约束和几何，不直接设 σ=0。 |
| #149 | **已证实有效（sim 伞任务交付）** | 三组定量仿真揭示边界条件，不证明硬件收益。 | CLOSED。 |
| #156 | **已证实有效（诊断工具），真实归因证据不足** | 七因子、DFT、理论上限及单因子测试已落地；无真实闭环缺口归因、jitter 周期量化或 CLI 采集终验。 | OPEN。 |
| #165 | **证据不足（2f 局部几何标定有效）** | 平场/GS 同帧相关改善可复核，但模型不是绝对光度预测器；FourierGSNet 未过同帧基线和 ≥3 对交错硬件验收。 | OPEN。 |
| #199 | **已证实有效（重构工程）** | MIL runner 重构完成；后续合并/CI 缺陷属 #250/#251，不构成整形收益。 | CLOSED。 |

## 5. 其他直接相关已关闭项

| issue | 主 verdict | 边界 |
|---|---|---|
| #188 | **已证实有效（工具）；原经验在 2f 无效** | 签名哈希与 size sweep 已交付；“10%–60% FOV”经验不适用于本 2f 系统。 |
| #204 | **已证实有效（换算工具）** | 远场/CCD 尺寸换算与黄金测试交付。 |
| #214 | **已证实有效（需求已由后续承接）** | Streamlit 方案被 #215/#242 supersede，不是算法无效。 |
| #216 | **证据不足** | TorchOptics 未采纳为主线，不等于其物理机制被证伪。 |
| #217 | **已证实无效（原复现主张）** | square `dCV=+0.17` 在 55 s 内不可复现。 |
| #218 | **已证实无效（原判读口径）** | 120/120 `contradict` 归因于比较口径/零级污染，不能作为模型方向真值。 |
| #242 | **已证实有效（Qt 工程）** | transfer campaign UI 交付，无算法收益含义。 |
| #243 | **已证实有效（安全门工程）** | G2a/G2b/G3/P4 判定层交付，无算法收益含义。 |
| #244 | **已证实无效（当前系统不适用）** | 单 SLM 2f 没有第二像素栅格，双屏逐 ROI 前提不成立。 |
| #253 | **已证实有效（湍流 sim 工具）** | 动态报告与同种子复现交付，不是硬件整形收益。 |

## 6. 证据口径冲突与不可复算项

1. **历史实测帧只支持相对改善**：s1200 非饱和 seed/final 的 rel-L2 有下降，但绝对 rel-L2 仍约 0.5–1.4，且无预注册阈值、部分档饱和，不能验收。原文摘要“9 对中 8 对下降”与 §3.2“11 对全部下降”口径不一致；在统一排除规则复算前，本总账不引用精确对数作判据。
2. **per-session refit 两组数字不是同一口径**：权威 `origin_session` 为 frozen `0.7175/0.6963/1.0518` → refit `0.8193/0.5174/0.5114`；`0.7228→0.8251` 等来自其他 origin/辐射度量口径。只采用 §10.8 权威管线作局部前向结论，且不跨会话外推。
3. **缺失原始数据的项目不假装复算**：20261021 MIL A/B、部分 20261014 refit JSON、transfer uncertainty 原始目录当前不可见，只引用报告结论并降低证据等级。
4. **高相关不等于绝对保真**：#119/#165 的峰位、FWHM、CV/PIB 和 NRMSE 仍须独立达门。

## 7. #247 跨会话 15+5 判定修正

实现：`scripts/slm/issue247_cross_session_check_20261124.py`；测试：`tests/ao_shaping/scripts/test_issue247_cross_session_check_20261124.py`。

- `overall_pass`/`verdict` 只由 `PINNED_MAIN_BASE_IDS` 的 15 项决定；全部 reproduced 才是主门通过。
- 5 个非 pinned p192 如实记录在 `counts` 和 `non_pinned_patterns`，主门不计通过也不计失败；若出现 `not_reproduced`，只置 `contradiction_outside_gate`，不升级主门。
- pinned 中任一缺会话为 `not_assessable`，方向超门为 `not_reproduced`；不得用补审替代主判据。
- exit 0 仅表示 **pinned 15/15 主门通过且 p192 补审 reproduced**；主门过而补审不过仍 exit 1。输入/覆盖/清单错误沿用 exit 3/4/5。
- 预检必须有可用曝光窗口；s1/s2/s3 各自独立开关设备，每个会话完整执行内部 r1/r2/r3，不是“一会话一轮”。
- 当前没有 `data/debug/issue247_cross_session_20261124` 真实数据根目录，因此只有合成测试证据，没有三会话物理复现结论。

主会话复核：

```text
python -m pytest tests/ao_shaping/scripts/test_issue247_cross_session_check_20261124.py -q
→ exit 0，34 passed in 3.79s
python -m ruff check scripts/slm/issue247_cross_session_check_20261124.py tests/ao_shaping/scripts/test_issue247_cross_session_check_20261124.py
→ exit 0，All checks passed!
```

真实采集完成后只可对新输出目录运行：

```bash
python scripts/slm/issue247_cross_session_check_20261124.py --root <cross-session-root> --out-dir <new-offline-verdict-dir>
```

## 8. 缺数据采集与分析计划

| 对象 | 必补证据 | 冻结执行口径 / 现有计划 |
|---|---|---|
| #247 | 三独立会话 15+5 复现、轴映射冻结、校正前后实测配对 | 每会话独立预检/open/close/release；内部 r1/r2/r3；主门 pinned 15/15、轴差 ≤2°；p192 沿用 35–45 px、`peak-background≥100`、每会话至少 3/4 帧强候选与轴角稳定门。见 `hw_window_20261124_issue247_cross_session_prereg.md`。 |
| #234/#236/#254 | Mono12 可分辨强度/方向门、三层 split、端到端效率配对 | 同会话同曝光同 bit-depth 暗场；train/selection_validation/final_test 隔离；final_test 只读一次；双门达门后做同起点/同 CCD 预算 2×2 交错，适配耗时计入总墙钟，失败/饱和不得增加。见 `transfer_efficiency_and_closed_loop_data_reuse_plan_20261121/plan.md`。 |
| #238 | 可发布 g₂π 与串扰独立验收 | Mono12/4095、全幅、≥4 曝光 bracket、同曝光熄灯暗场、`--wait-until-idle`；先冻结拟合和漂移门，再决定是否更新 LUT。 |
| PSF/refit | 跨会话、跨形状稳健性及实测整形转化 | 保持默认关闭；新鲜验证集不参与选型；按 `closed_loop_refit_abba_plan_20261124.md` 做默认 vs refit 的同曝光 A–B–B–A，只认实测 core CV 主判据。 |
| #137 | 热启动效果和节省时间 | 同一连接 cold/hot 各 N≥3 交错，固定 0.05 ms、同几何同种子；探针预算 4→3→2；报告验证集相关/误差及采集+拟合总时间。 |
| #139 | 窗口约束后的不确定性探针收益 | 先修 U(p) 与相机窗口脱节并通过几何门，再同种子 N≥3 A/B；比较探针数、进窗能量、held-out correlation/CV/EE，不能直接关带限上线。 |
| #124/#126 | 正向增量而非仅机制/反向验证 | 先 sim：错模型独立 freeform 验证集、同探针/步数预算、≥3 种子；#124 保留过噪声劣化反向验证，#126 必须优于 per-pixel-only 才进入硬件配对。 |
| #119/#121/#165 | 多次独立运行的模型准入 | freeform 验证集相关 ≥0.8 且 NRMSE 优于 Zernike，同时满足峰位/FWHM；#165 另需不劣于物理基线和 ≥3 对交错硬件收益。 |
| #213 L-BFGS | 离线加速能否迁移为物理效率 | K=1 Adam 基线 vs L-BFGS 早停，同问题/同质量门/同预算；报告总墙钟、CV、energy、失败率，达离线门后才做硬件 ABBA。 |
| #156 | 七因子真实归因 | 补 phase-stepping 帧栈、jitter 周期量化和闭环内去相关因子；报告各因子对实测增强缺口的置信区间，不把默认值 1.0 当测量。 |

## 9. 可复现入口（本轮未重跑 sim/硬件）

```bash
.venv/Scripts/python.exe report/sim_tue_methods/t1_noise_relaxation.py
.venv/Scripts/python.exe report/sim_tue_methods/t2_mixed_fitting.py
.venv/Scripts/python.exe report/sim_tue_methods/t4_uncertainty_sampling.py
python scripts/slm/analyze_model_hw_fidelity.py --out report/slm/differentiable_opt/artifacts/model_hw_fidelity_20261116
python data/debug/slm_transfer/20261124_transfer_psf_v1/evaluate_psf.py
```

这些入口分别复现 #124/#126/#139 的 sim、#119 的离线保真门和静态 PSF 适配；任何 exit 0 都不是实测整形收益证明。
