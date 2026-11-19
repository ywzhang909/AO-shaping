# 7 个 issue 有效性判定与离线改进结论（2026-11-19）

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **数据/关联脚本**: [`report/shape_trials/physical_refine/transfer_mil_hardware_plan_20261118.md`](../../report/shape_trials/physical_refine/transfer_mil_hardware_plan_20261118.md)
> **数据/关联脚本**: [`report/shape_trials/physical_refine/target_shape_ccd_acceptance_plan_20261118.md`](../../report/shape_trials/physical_refine/target_shape_ccd_acceptance_plan_20261118.md)
> **数据/关联脚本**: [`docs/dev/agent_notes_20261119.md`](agent_notes_20261119.md)
> **运行环境**: 离线分析 + 一次只读设备自检（见 §6）；未开激光、未下发相位、未取数。判定结论不依赖自检数据。
> **说明**: 汇总上下文 7 个 issue 的有效性判定、离线改进证据边界与实验前置条件。逐条判定的权威记录为对应 GitHub issue 的置顶评论；本文件是仓库内可核对的汇总。
<!-- provenance:end -->

## 1. 逐条有效性判定

| issue | 主题 | 判定 | GitHub 状态 | 核心依据 |
|---|---|---|---|---|
| #234 | zernike_block 三档微调 + 结构化探针 + 在线回灌 | **有效、算法收益未证明** | OPEN | 实施代码在 `feat/zernike-block-transfer`（`234f0741`，未入 master）；P0 结构化设计条件数差、可辨识模率 0.0；无双门通过 checkpoint，无实测核心 CV≤0.10。 |
| #235 | 输入相位一致性 + 零级 box centroid | **有效（部分子项已完成），原根因已否定** | OPEN | panel_gray 14 帧 mismatch 0；换实际相位方向 cosine 仍≈0 ⇒「输入相位错是根因」被否；box centroid 已落地（`fc4d3cf3`）；剩可信 LUT / `--pupil-center-xy` / shift_phase 写入前向。 |
| #236 | 拟合损失重构（掩膜 CV 等） | **有效但时机未到** | OPEN | fit.py 仍是旧三项损失；唯一消融无区分度（方向中位 −0.043…+0.070）、基线复现 FAILED；实验复跑需 SLM #1（11-16 探测 −10003）。 |
| #238 | 逐像元 LUT + 3×3/5×5 串扰核 | **有效、未实施，证据偏模型依赖** | OPEN | 核心模块不存在；二值光栅 run1 破 γ–g₂π 简并（g₂π≈964±13，γ≈0.91），但与设备回读 993 未判决；run2 USB `-10032` 中止；串扰仍「不可判定」。 |
| #242 | transfer-desktop campaign 安全门 UI | **有效且已实现** | CLOSED | `a046dc5d` 接入 G2/G2b/G3/P4，`gates.py`/`verdict_lines.py`/`page.py` 全部在 HEAD；`test_transfer_desktop_gates_ui.py` 19 passed（offscreen，`.venv`）。文本 0.80 与后端 0.85 为措辞差，实现遵循后端单一来源 0.85。 |
| #244 | 双屏逐 ROI 对准（10×10 ROI） | **无效（当前系统不适用）** | CLOSED | 单 SLM 2f 只有一个调制器、无第二像素网格，逐 ROI 前提不成立；全局 ramp 拟合 + 瞳面模型已覆盖真实需求。 |
| #247 | 倾斜/离焦探测零级 + 可微前向校正 | **有效、收益未证明** | OPEN | 硬件证实零级可分离与面板↔CCD 轴交换（+88.54°、尺度 1.086）；独立回测 r²=−0.00348（CI 含 0）、方向门 0/3；可微零级模板/统一观测校正全未实现。 |

## 2. 离线改进结论：尚无效整形优化算法

- 全部「改进」数字都是**前向预测指标**（预测 held-out CCD 帧），不是 measured-vs-target 整形结果。最强为 per-session Gaussian forward refit（B test corr 0.7228→0.8251、rel-L2 0.7019→0.5055；A test 亦全改善，全部 gate true，见两会话 `refit_provenance.json`），**仍仅为预测级，不升默认、不合并**。
- M1 static_global 跨会话 B test corr 0.7175→0.8018、rel-L2 0.6963→0.6033；但静态项正确基未定（drop/shuffle-phase 消融回收 73%/72% CNN 增益，加性静态+标量展宽只拿约 1/6），展宽被钉在 0。
- 计划中点名的候选已全部离线检验：Poisson 加权 `stop_not_defensible`；复幅值照明场边际正（+0.5% corr）不推进；乘性低秩空间增益**负结论**撤回；Noll 5..21 残差基未移动 smooth/bandlimited 收敛下界；近端投影仅改善更差候选。**无剩余未检验的离线候选能改变判决。**
- 唯一实测闭环（Anchored M3）实测核心 CV 6.08 / 5.01，远未达 ≤0.10；7.5× 模型-硬件分叉已定位为 8-bit 量化地板 + 过曝 argmax 伪影，非纯模型误差。
- 判决：**继续离线无法认证整形；下一步必须是设备窗口的配对 CCD 目标形状采集，而非再试算法。**
- **2026-11-19 更正**：以上「所有改善数字都是前向预测指标」与「无配对实测」的表述已被历史归档离线重算推翻。配对 measured-vs-target 帧确实存在（325 个目录 / 4222 文件），且 `s1200/common_replay` 的受控 seed→final 重放显示**相对**实测改善（非饱和 11/12 对 rel_l2 下降、efrac 上升 10 对）。但绝对保真度仍 rel_l2≈0.5–1.4、阈值未预注册、历史 replay 用 `floor=0` 不合规、个别帧饱和，**仍不足以宣称算法有效或验收通过**。详见 [`historical_measured_shape_metrics_reanalysis_20261119.md`](historical_measured_shape_metrics_reanalysis_20261119.md)。5 个开放 issue 的逐项判定不变。

## 3. 实验前置：历史数据、脚本与 docs/dev 计划

- **历史数据已优先复查并二次重算**：B（80/24/10）、A（56/12/10）会话、Anchored M3、absolute_fidelity_12bit、单模屏/ladder、issue238 灰度与二值 run1 均已核对。**更正**：历史归档**包含** measured-vs-target 配对帧（`raw_frames` + `target` + `core` + `camera_xy`），此前「缺配对目标形状帧」的说法失实；已用生产函数离线重算，结论见 [`historical_measured_shape_metrics_reanalysis_20261119.md`](historical_measured_shape_metrics_reanalysis_20261119.md)。
- **采集/验收脚本已备且离线验证**：`scripts/slm/validate_transfer_shapes_hardware.py`（11 mock 测试；`--plan-only` 冒烟 exit 0、只生成 plan.json、无锁）、`scripts/slm/replay_transfer_shapes_hardware.py`（16 mock 测试、强制 `--out`/`--exposures`）、`protocol.target_intensity_rmse`（8 单测）、`scripts/slm/refit_transfer_forward.py`（13 测试通过）。两外壳 `--execute` **从未在真实设备执行**。
- **docs/dev 计划已写好**：`report/shape_trials/physical_refine/target_shape_ccd_acceptance_plan_20261118.md`（§4 阈值/坐标/曝光全部待现场预注册）、`transfer_mil_hardware_plan_20261118.md` §9–§11、`docs/dev/issue238_binary_gray_capture_plan_20261119.md` §6.1、`docs/dev/next_hw_plan_transfer_candidates_issue247_20261117.md`。
- **三项实测门禁缺口**（§11.3，**已按本轮重算更正**）：(a) 阈值仍未预注册，历史 RMSE 有值但用 `floor=0` 不合规、不可作验收评分；(b) 带锁 replay 端到端未跑；(c) 历史配对 CCD 目标形状对比**存在**（见重算文档），但非新鲜独立采集、绝对保真度未达标。任何模型 loss/预测指标都不替代独立 CCD 证据。

## 4. 实验 blockers（仅用户可解除）

- SLM #1 二值 run2 USB `-10032`（其他错误）假死需**手动重启控制器**；此前单模屏探测亦出现 `-10003`。健康设备会话是硬前置。
- 新采集须：fresh open + 状态/读写自检、Mono12 / 中段曝光（禁用贴饱和线选点，已改 `PREFERRED_PEAK_FRACTION=0.7`）、`--wait-until-idle` + 锁在 try 内、独立 `--out` 拒复用。
- 阈值预注册（rel-L2 上界 / energy_fraction 下界 / 背景 floor / 零级坐标）与设备时段授权由用户确认。

## 5. 本轮复验命令与结果

- `PYTHONPATH=src python3 -m pytest tests/ao_shaping/scripts/test_refit_transfer_forward.py --import-mode=importlib -q` → **13 passed**（1.57 s）。
- `QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe -m pytest tests/ao_shaping/gui/slm/test_transfer_desktop_gates_ui.py --import-mode=importlib -q` → **19 passed**（13.92 s）。
- `python3 scripts/slm/validate_transfer_shapes_hardware.py --checkpoint <fit_m3_full> --out <tmp> --zero-xy 1226 1104` → exit 0，仅打印 plan、不建目录、不加锁。
- `python3 scripts/sync_report_provenance.py --check` → exit 0（"check: clean"）；`python3 scripts/_common/term_unify_20261016.py` dry-run → exit 0（既有 34 文件差异，本文件无新增违规）。
- 未跑本地全量 pytest（遵循项目纪律）。判定期间除 §6 的一次只读自检外未占用设备。

## 6. 设备占用声明

判定分析全程离线。2026-11-19 13:50 经用户确认后执行过**一次只读设备自检**（脚本 `C:\Users\zhangh\AppData\Local\Temp\opencode\slm_readonly_selfcheck_20261119.py`，结果同目录 `.json`）：

- **占用设备**：SLM #1（1064 nm）单次 `open → check_drive_board → close`；大恒 CCD 仅 SDK 子网枚举（未 open）。**未开激光、未下发相位、未取帧**。
- **结果**：SLM #1 驱动板在线（序列号 `22030108`，`source=driveboard`，`ok=true`），`closed_cleanly=true`，无假死/超时；CCD `sdk_ready=true, online=true, count=1`。
- **结论**：设备健康，历史上的 `-10032`/`-10003` 未复现；具备实验取数条件。**更正**：历史归档已有 measured-vs-target 配对数据并显示相对改善（见重算文档），但阈值未预注册、绝对保真度未达标，7 个 issue 的实测整形验收门槛仍未被满足。

历史 run2 `-10032` 中止时已正常关闭 SLM/CCD 并释放锁（平场归零失败单独记录），无残留占用。
