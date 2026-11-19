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
| #235 | 输入相位一致性 + 零级 box centroid | **有效（部分子项已完成），原根因已否定** | OPEN | panel_gray 14 帧 mismatch 0；换实际相位方向 cosine 仍≈0 ⇒「输入相位错是根因」被否；box centroid 已落地（`fc4d3cf3`），历史双瓣定位 SD 由 argmax 0.394 px 降至 0.297–0.330 px（仅为定位稳定性，不是整形收益）；剩可信 LUT / `--pupil-center-xy` / shift_phase 写入前向。 |
| #236 | 拟合损失重构（掩膜 CV 等） | **有效但时机未到** | OPEN | fit.py 仍是旧三项损失；唯一消融无区分度（方向中位 −0.043…+0.070）、基线复现 FAILED；实验复跑需 SLM #1（11-16 探测 −10003）。 |
| #238 | 逐像元 LUT + 3×3/5×5 串扰核 | **有效、收益未证明，仍不可定标** | OPEN | run4 period32/48 fold `g₂π=952.6 [948,956]` / `946.9 [945,950]`，低于回读 993；但束功率漂移 +31.53%/−13.33%、±1 极小分裂与整曲线残差仍大，bootstrap 仅计帧噪声，不足以更新 LUT；串扰仍不可判定。详见 `issue238_binary_gray_capture_plan_20261119.md` §6.5。 |
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

## 7. 后续离线复核补记（2026-11-19）

- 上述 §4/§6 是当时的状态记录；随后 `issue238_binary_phase_20261119/run4_period_sweep` 的 period32/48 两段已经完成采集并确认 SLM、CCD 与设备锁正常释放。新证据已按离线 fit 复核，详见 `issue238_binary_gray_capture_plan_20261119.md` §6.5。fold 值低于 993 但伴随显著漂移与系统误差，故不改 LUT、不把问题判成已解决。
- 7 个 issue 的逐条状态、#235 定位收益边界以及 #234/#235/#236/#238/#247 的缺失数据与下一步门禁详见 [`open_issue_offline_followup_20261119.md`](open_issue_offline_followup_20261119.md)。本次补记只做离线复核，未新占硬件、未更新 GitHub issue 状态；有效但收益未证明者保持 OPEN，#244 已 CLOSED（无效），#242 已 CLOSED（实现完成）。
- **本窗口 §2.3 主门实测（离线）**：`validate_pairs` 的 `square_seed → square_final` 配对在冻结 floor=`14.9375 counts` 下**未通过且方向为负**——`rel_l2` 1.1575→1.2958（Δ=−0.1383，噪声地板 0.0424）、`energy_fraction` 0.4556→0.3919（Δ=−0.0637，噪声地板 0.0146）；目标核 CV 反降（0.3271→0.1728）。均属实测退化。完整 3 对表、逐 capture、窗口级判定与 replay 边界见 [`hw_window_20261119_preregistration.md`](hw_window_20261119_preregistration.md) §6。此结论只说明端到端相对目标变差，不指向任何单一 issue 子项。

### 7.1 2026-11-20 追加：三份离线结论，判定均未变

- **#238**：§6.6 联合诊断仅有第一阶段（清单核验 + 固定 ROI 描述量）已执行，`scripts/slm/issue238_run4_spatial_diagnostic_20261120.py`，产物只在 Temp。两段 960/960 帧完整、零丢帧；period 与时段/漂移共线，M0/M1 未实现，故永久报告门槛未触发。判定不变：**有效、收益未证明，仍不可定标**。见 `issue238_binary_gray_capture_plan_20261119.md` §6.7。
- **#235**：`shift_phase` 与独立 pupil（瞳面）-center 在离线域已判**零判别力**（归档编码字节一致、前向无该自由度；反演分辨率 Δx ±20 px / Δy ±30–40 px）；物理 LUT 仅剩 `lut_checker_modulation_20261118b` 的棋盘格调制比，属间接代理且全库无消费脚本。判定不变：**OPEN**。
- **#247**：88.544° 可复现（5 次重复 SD **0.088°**），但本质是 **180° 对称族**（全局翻号为 −91.457°），符号分支只由 convention 字符串钉住；`data/slm_geometry_calib.npz` 仍缺 panel/CCD pitch 与焦距放大率。判定不变：**OPEN**，不冻结物理映射、不写 `ramp_matrix`。

### 7.2 2026-11-20 追加：#249 / #250 判定与修复

本轮逐一核验了上下文新发现的两个 code-scan / ci-fix issue，均为**有效**且属离线可修类别；全程未开 SLM/CCD/激光、未新采数据。两 issue 的判定与补丁范围如下。

**#249 — call_graph 双 JSON 函数 span 漂移（12→40→47 三连涨、无 CI 守卫）：判定有效，已修复。**

- 实测复现：`docs/dev/call_graph_wiki/raw/assets/call_graph.json` 与 `call_graph_final.json` **逐字节一致**（234,184 bytes、indent=1、CRLF + 尾换行）；全量 AST 对拍按 `(file, qualname)` 匹配 **checked=190、stale=45**（issue 报 47，差异来自匹配启发式：另有 4 个节点定义已外提/删除，无法对拍）。`runner_common` 的 8 个 span 经基线核对为 **BASE-ALREADY-STALE**，本窗口 #205 外提只是整体平移。
- 修复（对应 issue 推荐档 **A + B**）：脚本按 AST 实测 span **重钉 45 处**，两份 JSON 同步写回，`stale→0`；序列化保持原格式（`ensure_ascii=False, indent=1`、CRLF、尾换行），双文件仍逐字节一致、节点数 215 不变、`git diff` 仅 `lineno`/`end` 两类行（各 89 行）。
- CI 守卫（issue 档 B）：在 `tests/ao_shaping/test_conventions.py` 新增 `test_call_graph_json_function_spans_track_source`，对双 JSON 一致性 + 每个可解析节点的 span 做 **零漂移** 门禁（checked≥180）。4 个外提/删除节点列入具名豁免并注释迁移出处。正向 1 passed；注入人工漂移的负向测试证实能捕获。此前 `test_wiki_source_citations_land_on_definitions` 只读 `.md` front-matter、完全不读 JSON，是三连涨无人发现的原因。

**#250 — master tip 合并损坏（#205 守卫缝坏 + MIL 恢复漏模块 + 本地链无 CI）：判定有效，核心损坏已修复。**

- 三态实测：本地 HEAD `4d0aa1b8` 是损坏合并 `3d0fe7f2`（=`gitub/master`）上的 4 个本地提交；3 个 MIL 模块在 `gitub/master`/`3d0fe7f2`/`HEAD` 全部缺失，仅上游 `00225c3f` 历史存在；#205 守卫文件实测 **73 个常量定义 / 11 个引用未定义**（issue 记 42 行引用 11 常量，吻合）；`workflow.py` 的 SyntaxError 子项已由本地提交 `4d0aa1b8` 修复。
- 修复清单（均离线、无设备占用）：
  1. 从 `00225c3f`（== `af9a55a0^` 的删除前状态，API 与本地 `slm_model_in_loop.py:2970` 的 `AdaptiveTrustRegionConfig(initial_c_l2=…)` 一致）恢复 `adaptive_trust_region.py`、`model_in_loop_summary.py`、`model_in_loop_auto.py`。import 冒烟 + 运行时实例化 OK；`py_compile` exit 0、ruff clean。
  2. #205 守卫补回 11 个上游路径常量（逐值取自 `00225c3f`）；并把已增长到 288 行、越过 280 行 #205 天花板的 `_run_session_rounds` 中 `ccd_mode == "none"` 分支外提为 `_open_loop_stage()`（逻辑逐字搬运，不改 public signature/行为）。
  3. 恢复 `bench_settings.py` 从 `slm_registry` 导入 `MAX_SLMS/connected_slm_handles/connected_slm_numbers`（`__all__` 导出与 `connected_slm_numbers()` 调用此前 F822/F821）；`ccd_analyzer.py` 的两个 `_ccd_render_*` 定义在上游存在、本地被整段删除但其调用点仍在，按上游原样恢复（本地无独有行）。
- 验收：`test_issue205_function_shape.py` **433 passed**（原 collection 即 `NameError`）；8 个相关套件合计 **578 passed**；全树 `pytest --collect-only` **13407 collected，exit 0**；仓库级 ruff 由 278 降到 **4**（剩 `scripts/_common/provenance.py` F601×3 与其他车道 `sweep_stepb_budget.py` F401×1，均非本任务范围）。`test_conventions.py` 全树 11 failed 与 HEAD 基线**逐条相同**（操作前经 detached worktree 对照），本轮未新增门禁失败。
- 未做/不越界：未改 LUT、未碰 `scripts/_common/provenance.py`（其他车道脏）、未调整 `test_conventions.py` 的 print allowlist 与基线数字、未推送。判定：**有效，本轮修复核心合并损坏；#249/#250 的 GitHub 状态维持 OPEN，待 owner 复核后再关闭。**
