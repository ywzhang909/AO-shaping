# 2026-11-24 实验计划更新（基于当日任务完成情况与实验数据）

**性质**：docs/dev 实验计划日更。本文只登记已回写的判定与下一步计划，不新增预注册判据（各判据以对应预注册文档为准）。
**数据基准**：本文所有判定引用均以 **HEAD（04c48220 及此前提交）** 为准；工作树当日被并行会话删改的文件清单见 §5。
**设备声明**：撰写本文为纯离线操作，未占用 SLM / CCD / 激光。

---

## 0. 摘要：当日完成情况一览

| 任务 | 状态 | 关键判定/数字 | 证据（短路径） |
|---|---|---|---|
| #247 同窗口重复性 | 完成 | `overall_pass=true`（限定口径）；reproduced 15 / not_reproduced 0 / not_assessable 5；276 帧 | report/shape_trials/issue247_reproducibility_20261124/ |
| #247 S1b 内容轴 2×2 | 判读回写 | `insufficient_consistency` ⇒ S1 BLOCKED | data/experiments/attribution_20261123/s1/ |
| #247 S1c 最小歧义消解 | 判读回写 | `session_nondeterminism` | 同上 |
| #247 S1d 方向探针 | 判读回写 | `delivery_nondeterministic_same_bytes`，映射 frozen=false | s1d/run_01/axis_mapping_status.json |
| #247 S1e 同字节探针 | **失败（设备）** | `status=error`，0/12 上传；SLM #1 打不开（-10002 USB） | s1e_same_bytes_probe_20261124/run_01/summary.json |
| #247 全阶段证据汇总 + 有效性台账 | 完成 | #247「证据不足（现象局部成立）」；#254「证据不足（计划有效）」 | transfer_mil_issue_validity_verdicts_20261124.md |
| #254 T2 三层防泄漏 | 完成 | splits.py 迁出（bccf258d）+ 封存 final_test（ba50f842） | src/ao_shaping/…/transfer/ |
| transfer 几何残差适配 | 完成，**证伪** | 双门失败（0.62% < 5%；独立 x −25.99%/y +18.96%） | transfer_geometry_adaptation_20261124.md |
| transfer 静态 PSF 适配 | 完成，**通过** | 双门过（−28.59% / −21.30%），sigma_ccd_px=4.33，默认关闭 | transfer_psf_adaptation_20261124.md |
| #254 runner 单会话复用 | 完成 | `--reuse-device-session`（方案 A，71d274f2） | runners/slm/transfer_shapes_runner.py |
| A/B 会话（8 组调度变体） | 完成 | acceptance=sequential_ccd；Jacobian 以 `contradict` 为主；CV 小幅改善 | data/debug/slm_transfer/ab_session_20261124/ |
| #255 CI 门禁 | **仍红** | `_fmt` offender 4 处（今日新增 1） | tests/…/test_common_helpers_not_reintroduced.py |
| 光路复用门（离线部分） | 完成，待试跑 | 离线验证 2c873fe；试跑计划未执行 | next_hw_plan_optical_path_reuse_gate_trial_20261124.md |
| GS 1000px fig8b 离焦响应 | 完成 | ±1 rad 峰 61.2%、宽 29.3%；C 变体脚本就绪未采集 | report/slm/gs_control_1000px/report.md |
| #238 暗场离线回填 | 完成（口径受限） | 12/12 manifest 过；噪声地板 0.8817 counts；无逐帧原始帧 | data/debug/dark_frame_analysis_20261124/ |
| GUI beam_quality 分析页 | 完成 | 自动曝光口径修正 + 热簇鲁棒主成分 + 异步分析（#252 仍未做桌面接入） | gui/ccd/beam_quality/ |

issue 侧动作：#247/#252/#254/#255/#238 已各发当日扫描评论；全部维持 OPEN，无失效或已完成可关闭项（#255 门禁在新增脚本上继续恶化，明确不可关）。

---

## 1. 当前阻断项与解锁条件

| 阻断 | 影响 | 解锁条件 |
|---|---|---|
| SLM #1 USB 打不开（-10002，11-21 同族） | S1e 无法重试；后续所有需 SLM #1 的窗口（复用门试跑、GS C 变体） | 人工恢复 USB/供电/线缆并确认枚举（#254 T0 类任务，需用户/硬件） |
| S1 映射复现性 BLOCKED（S1b/S1c/S1d 三连不过） | 轴映射/全局映射冻结、D bracket、G1/G3/G4 全部暂缓 | ① 完成 §12.16.5 实现门禁（memmove 守卫离线验证，根因候选 `non_contiguous_memmove_transpose`，driver.py `_write_to_memory`）；② S1e 重试取得「同字节 ⇒ 同远场」（主瓣差 ≤1.5 px） |
| #234/#236 双门「不可判定/不可归因」（11-23 新口径） | #254 T6 物理比较不能放行 | 引用新口径重走放行判断；在此之前 T6 仅做离线设计 |
| #255 CI 门禁红（4 处 `_fmt`） | 合入门禁持续红 | 四处收敛到 scripts/_common 六 formatter（纯离线，可立即做） |

---

## 2. 下一实验窗口优先级（P1 > P5）

### P1. S1e 同字节探针重试（#247，最高优先）

前置（缺一不采集）：
1. 人工恢复 SLM #1 USB（用户/硬件）；
2. 完成 hw_window_20261123_attribution_prereg.md §12.16.5 实现门禁：`_write_to_memory()` 增加 stride/连续性守卫（防 F-contiguous 数组被裸 memmove 转置），离线用 H 数组逻辑 SHA 7276d9da… vs 物理 SHA b98be9a6… 的用例回归；
3. 预注册（a82b3ce8）与冻结脚本（94282236）不变，按 E1–E4 分支判读。

验收：24 帧全采齐、判定回写 §12.16；同字节 ⇒ 同远场（主瓣差 ≤1.5 px）才解锁后续轴/K 步骤。任何分支结果下 S1 本轮不升格、不做 D bracket。

### P2. 光路复用门实验窗口试跑

按 next_hw_plan_optical_path_reuse_gate_trial_20261124.md 执行（0dcd0566 已备 harness）。两种结果（过/不过）均可接受，均写入试跑报告；**试跑过关 ≠ 光路有效**，只省六步探测时间。S1e 复用门成功后可与其共窗口。

### P3. #255 门禁清偿（纯离线，随时可做）

4 处 `_fmt`（audit_p3_finite_difference_20261123 / efficiency_match_analyze_20261123 / issue247_waveform_probe_solve_20261122 / issue247_reproducibility_check_20261124）收敛到 _common；收敛后实跑门禁取证并在 #255 回评。

### P4. transfer 线（#254）离线准备 + T6 前置

1. Jacobian `contradict` 复核：对 ab_session_20261124 的 8 组逐 probe 列矛盾清单（哪些 probe、幅度多大），确定 T6 是否需要先修物理预测模型；
2. T6 设计冻结：2×2 交错（continue vs CV）×（square/star），同起点/同预算，`--reuse-device-session`，适配耗时计入端到端效率；PSF 适配按 opt-in 参与比较，几何项不参与（已证伪）；
3. #247 跨会话 15+5 检查器：等三会话真实数据根目录就位后离线跑，当前不虚填。
4. capture 目录纪律：ab_session_20261124 目录已含数据不可复用；中断的采集目录须删除后重建（沿用既有规则）。

### P5. 次优先（可穿插离线做）

- GS 1000px C 变体采集（脚本 4bcb0dac 已就绪，待设备窗口；与 P1/P2 共窗口可省 open/close）；
- 暗场 §6 四项任务项（位深隔离、§0 哈希重签、现场几何标定、checkpoint provenance 核对）——严格逐帧噪声判据须重采并保留逐帧帧；
- 焦点（[1]）与方斑纵向中心散布的 panel-x 束心扫描（GS 1000px 报告遗留）。

---

## 3. 本周实验路线图衔接（对旧计划的覆盖声明）

- next_experiment_plan_issue247_mapping_validation_20261121.md：15 图样部分已由 11-23/11-24 波形与归因窗口执行；其「独立复现」步骤由 S1e 与 15+5 三会话检查器取代，本文 §2.P1/P4.4 为其现行版。
- efficiency_match_experiment_plan_20261121.md 的效率比较：被 #254 T6 取代（三层分组 + 交错配对新口径）。
- hw_plan_consolidated_20261118.md 的 S0/S1 主线：S0 各 run 判定已全部回写（R1–R4 见 HEAD 版 attribution prereg §12）；S1 结论 BLOCKED，后续按本文 §2.P1。

---

## 4. 复用与纪律提醒（当日新证据强化）

1. **S1e/控制器掉线同族**（-10002）：SLM #1 USB 不稳两次（11-21、11-24），建议下次窗口前先做枚举探测（桌面页加载探测只枚举不开相机/设备，已有先例）。
2. **背对背 SLM open/close 假死**既有风险在 11-24 未复现（各窗口单次 open/close 均正常释放），维持「单窗口单 open/close + `--reuse-device-session`」做法。
3. **采集目录不可复用**：ab_session 与 s1e run_01 均为已写目录，重试须新 run 目录。
4. 术语门禁：新增文档前跑 `python scripts/_common/term_unify_20261016.py`（dry-run 0 文件）。

---

## 5. 工作树风险登记（截至 2026-11-24 23:15）

以下文件 **HEAD 完整、工作树被删/截断**（疑为并行会话或合并 stash 恢复缺口；本文分析一律以 HEAD 为准）：

| 路径 | 工作树状态 |
|---|---|
| docs/dev/hw_window_20261123_attribution_prereg.md | 截断至 338 行（HEAD 1365 行，R1–R4/S1e/§12.10 仅在 HEAD） |
| docs/dev/hw_window_20261124_issue247_reproducibility_prereg.md | 删除（HEAD 有） |
| report/shape_trials/issue247_reproducibility_20261124/ | 删除（HEAD 有） |
| report/shape_trials/issue247_evidence_summary_20261124/ | 删除（HEAD 有） |
| report/slm/gs_control_1000px/report.md、report/shape_trials/dark_frame_record_20261119.md | 当日新增段落仅 HEAD |
| scripts/slm/analyze_dark_frame_exposures_20261124.py、analyze_content_axis_2x2_20261124.py | 工作树暂缺（HEAD 可恢复） |

恢复守则（沿用既定规则）：确认无并行会话在飞后，**逐路径** `git checkout HEAD -- <path>` 恢复；严禁 broad checkout / `git checkout -- .`。同目录存在「任务扫描」「Github同步」两个并行会话（23:00–23:08 活跃），恢复动作由主会话在它们结束后统一执行。

---

*判定引用以各预注册/报告文档与 GitHub issue 评论（2026-11-24 各条）为准；本文为计划层汇总。*
