# AO-Shaping 文档总索引

> 本文件是 `docs/` 的**入口索引**：按主题分域，每个文件一句话说明"这里能查到什么"。
> 共 160 个 Markdown 文档（2026-10-23 盘点，2026-10-26 重钉：`spgd-square` 幽灵页删除后 147→146，新增相差光路多方案整形对比实验方案后 146→147，新增 `scripts/` 功能分类两份分析文档后 147→149；2026-10-29 复核：`find docs -name '*.md' | wc -l` = 157；2026-10-30 复核：新增 `multishape_campaign_ui` SRS/用户手册（`0ec5e32d`）+ `streamlit_theme`（`d3fb61c5`）3 份后 157→160）。新文档请在本索引登记后再提交。
>
> 阅读路径建议：
> 1. 第一次接触仓库 → `dev/call_graph_wiki/index.md`（调用图 + 逐函数手册，含全局调用图/类图）
> 2. 要跑某条命令 → 本索引 `dm/` `wfs/` `slm/` 对应命令页（每条命令"1. 这条命令做什么"开头）
> 3. 要改代码 → `dev/coding_conventions.md` + `call_graph_wiki` 相关函数页
> 4. 硬件/驱动细节 → `drivers_architecture.md` + 各品牌目录（`daheng/` `miicam/` `thorlab-wfs/` `slm-200/`）

## 根目录 — 架构与总览

| 文档 | 说明 |
|---|---|
| `architecture.md` | `ao_shaping.model` 光场数据类模型（PhaseMap/ComplexField/OPD 等，像素间距+波长+来源元数据） |
| `drivers_architecture.md` | `drivers/` 硬件抽象层设计：多品牌设备接入 + 虚拟仿真环境 |
| `gui.md` | Streamlit GUI 总览，按设备域分包于 `src/ao_shaping/gui/` |
| `sampling_assist_usage.md` | SLM 整形页「远场尺寸换算 → 采样光路」popover 的用户手册：`S_CCD = S·f_sample/f_main` 换算、五条可达性判据、操作步骤与界面截图（截图由 `scripts/slm/screenshot_sampling_assist.py` 无设备渲染） |
| `hardware_drivers.md` | 硬件驱动清单（Thorlabs WFS / SLM / CCD / DM 各品牌支持矩阵） |
| `issues_report.md` | 问题与故障排查记录 |
| `ml_training.md` | ML 训练指南（`src/ml/` 可选依赖包：训练器/模型/指标，与硬件层解耦） |
| `ml_hwdataset.md` | ML `src/ml/hwdataset/` 模块文档：硬件相位→相机图像 DataLoader 全节（语料实测/模块表/缓存/样本契约/不可回退结论/用法），外迁自 AGENTS.md |
| `simulation.md` | 数字孪生仿真环境总览（无硬件测试） |
| `tabu_search_algorithm.md` | 禁忌搜索算法实现说明（英文，`algorithm/tabu`） |
| `zernike_usage.md` | Zernike 两层入口用法（模式编号/系数换算入口） |
| `terminology_glossary.md` | **术语规范表**（唯一权威）：150+ 条口语→规范对照 + DO-NOT-TOUCH 保留清单 + T-60/T-61 新词登记 |
| `paper_gs_refine_experiment.md` | 实验论文（GS 预矫正 + 随机平行梯度下降两级整形）：v4.8（2026-10-29 修订：补齐复合分数权重与窗口口径、理顺章节编号），含 MIL 物理预测模型 + 可微光学框架综述 |
| `自适应 SPGD 光束整形闭环：完整方案总结.md` | 自适应 SPGD 方案：PPO 可信域思想 + 均匀化复合目标 + 自适应权重 + 历史信息复用 |
| `相差光路多方案整形对比实验方案_v1.0.md` | 相差光路多方案整形对比实验方案 v1.0：6 方案 × 7 档像差，效果（composite/PIB/CV）+ 效率（收敛帧数/墙钟/收益成本比/帕累托）双维度对比 + 每类像差绑定独立工程路线 |
| `scripts_functional_classification.md` | `scripts/` 功能二级分类与复用分析报告（2026-10-26）：在设备域一级分类上补 11 类功能二级分类，识别跨脚本重复功能与可复用函数，给出契约测试驱动、可逆的重构方案（未改动代码）；全量映射见 `scripts_functional_map.json` |
| `scripts_issue_body.md` | `[分析] scripts/ 功能分类报告` 的**可直接粘贴 issue 正文**（`2e8977f7`）：摘要 + 11 类二级分类表 + 重复功能清单，供在 GitHub 立项时直接粘贴 |

## CLI / 用户

| 文档 | 说明 |
|---|---|
| `user/cli-reference.md` | **CLI 正式参考**：全部 flag 拼写/取值域/默认值逐字来自 `--help` 转储 |
| `dev/coding_conventions.md` | 编码约定（Python 3.12+、导入、命名等） |
| `dev/ci_testing.md` | 测试排除/分片策略与 CI 门禁现状（外迁自 AGENTS.md 的 `MISSING INFRASTRUCTURE` 与测试排除叙事） |
| `dev/project_structure.md` | 项目包结构逐文件详解（外迁自根 AGENTS.md 的 `Project Structure`；根文档只留精简树 + 指向本文） |
| `dev/extension_guide.md` | 二次开发与扩展指南：改动分层路由 + 最小模板（何时加新设备/算法/CLI，复用哪些 canonical 入口） |
| `dev/todo_archive.md` | 已决事项档案（每条都是"一次决策的证据"，非变更日志） |

## 开发手册（call_graph_wiki）— 53 篇

`dev/call_graph_wiki/` 是**函数级开发手册**，独立索引见 [`dev/call_graph_wiki/index.md`](dev/call_graph_wiki/index.md)：

| 子集 | 篇数 | 内容 |
|---|---|---|
| `entry/` | 2 | `main.py` 入口、`cli()` 根命令组 |
| `commands/` | 19 | 每条命令：docstring 原文 + 调用链分析（自动）+ 硬件可达性（`spgd-square` 幽灵页 2026-10-26 随 `601b5af9` 删除，20→19） |
| `functions/` | 25 | 骨架函数：签名 + docstring + 入/出边 |
| `diagram/` | 3 | 分层调用图、**全局调用图**（包级聚合）、**全局类图**（继承+协作） |
| `relations/` | 1 | 相似函数关联总览（5 相似对 / 4 簇，含逐维得分） |
| `raw/assets/` | 8 | 机器可读数据：call_graph_final.json（193 节点/290 边）、similar_functions.json、mmd/svg |
| 根 | 3 | `index.md`（总索引）/ `SCHEMA.md`（结构与更新策略）/ `log.md`（操作日志） |

## DM 变形镜（NLight / 微驱）— 6 条命令

| 文档 | 说明 |
|---|---|
| `dm/wf.md` | 波前 RMS 优化（NLight DM + Thorlabs WFS，SPGD） |
| `dm/pib.md` | 轴向光束 PIB 优化（DM + CCD 远场反馈） |
| `dm/pipeline.md` | 串行流水线：先 wf 后 pib，WFS+CCD 双反馈 |
| `dm/combined.md` | AdaMOD + SPGD 混合策略 PIB（DM+CCD 无波前） |
| `dm/dm_matrix.md` | DM 响应矩阵标定（push-pull，顺序 + Hadamard 模式） |
| `dm/hadamard_matrix.md` | Hadamard 并行扰动矩阵标定 |
| `micro_drive/alt_voltage.md` | R50Power 交替电压下发 + NI DAQ ADC 同步采集 |
| `micro_drive/full_voltage.md` | AsyncMicroDM 全量单元 0V↔V 同时交替（老化/寿命验证） |
| `micro_deformable_mirror/docs/refactor.md` | Micro-DM 模块重构记录 |
| `micro_deformable_mirror/docs/README.md` | Micro-DM 模块说明 |

## WFS 波前传感器 — 3 条命令

| 文档 | 说明 |
|---|---|
| `wfs/ga_zernike.md` | 遗传算法搜索 Zernike 系数（WFS RMS 为适应度，无梯度/多峰场景） |
| `wfs/greedy_zernike.md` | 贪婪局部搜索（内部可选 SPGD/GA/PSO/SA/HC/RS/CEM/DE） |
| `wfs/closed_loop.md` | 基于已存 Zernike 响应矩阵的闭环波前优化（回放矩阵，免实时测量） |

## SLM 空间光调制器 — 21 篇

| 文档 | 说明 |
|---|---|
| `slm/slm_pib.md` | SLM Zernike 整形（CCD 远场光斑反馈，PIB） |
| `slm/spgd_square.md` | SLM 方形远场均匀度整形（SPGD，CV/EE/AR 目标） |
| `slm/slm_gsnet.md` | FREEFORM 全频随机相位方形整形 + FourierGSNet 训练 |
| `slm/slm_gs_refine.md` | GS 开环预矫正（0.662→0.812）+ SPGD 细化（→0.849）硬件移植 |
| `slm/slm_model_in_loop.md` | 物理预测模型闭环整形（每轮用实测数据修正模型再合成相位） |
| `slm/model_in_loop_algorithm.md` | 同上命令的**结构与算法**详解（调用关系/计算时序） |
| `slm/model_in_loop_handoff.md` | Model-in-loop 跨光路交接手册（换光路/功率/衰减片/相机/入射光时的交接清单） |
| `slm/model_in_loop_workflow_faq.md` | 跨光路 model-in-loop：探测、物理预测模型迁移学习、可微优化与 FAQ |
| `slm/slm_alignment_probe.md` | 整形前的聚焦、彗差校正与 CCD 中心定位 |
| `slm/zernike_matrix.md` | SLM Zernike 响应矩阵标定 + 离线矫正自检 |
| `slm/rms_zernike.md` | SLM Zernike RMS 优化（SPGD/Adam/AdaMOD） |
| `slm/slm_lut.md` | 灰度-相位 LUT 标定（半屏参考/测试光栅，±1 级衍射效率比值） |
| `slm/闭环光束整形_文献与优化建议_20261014.md` | 闭环整形文献×实验×优化建议（GS/SPGD/物理预测模型/freeform，24 条 DOI 已核实） |
| `slm/slm_diagnose.md` | SLM+相机逐级硬件自检（2026-09 三步证据链固化） |
| `slm/pre_run_characterization.md` | 实验系统开机表征指南（先量实验系统的必要性） |
| `slm/slm_pattern_helper.md` | `PatternHelper` 相位图案生成工具类指南 |
| `slm/slm_gui_manual.md` | `multi_slm_controller.py` Streamlit 前端页面手册 |
| `slm/slm_gui_operation_map.md` | SLM 控制器操作路线图：27 个操作按时效（T0–T4）分类的界面重设计 (v1.0) |
| `slm/align_windowed.md` | SLM+CCD 装配对准（相机 ROI 开窗自动 0级/±1级 对准） |
| `slm/slm_patterns/shaping.md` | SLM 整形图案说明 |
| `slm/slm_transfer_workflow.md` | 三形状（方/三角/星）物理预测模型整形统一入口 `python -m ao_shaping.tools.slm.transfer`：规划、物理迁移、GS 初始化、曝光响应、展宽校准、CCD 锚定可微优化与原始数据验收 |

## GUI 前端（Streamlit）— 21 篇

| 文档 | 说明 |
|---|---|
| `GUI/slm/multi_slm_controller/multi_slm_controller_srs.md` | multi_slm_controller.py 软件需求规格说明书 (SRS)：功能/非功能需求、接口规范、系统类图、设计约束 |
| `GUI/slm/multi_slm_controller/multi_slm_controller_user_manual.md` | multi_slm_controller.py 用户手册：设备连接/相位下发/闭环远场整形操作指南 |
| `GUI/slm/multishape_campaign_ui/multishape_campaign_ui_srs.md` | multishape_campaign_ui.py 软件需求规格说明书 (SRS)：多形状整形 campaign 操作台，「形状 + 尺寸」为唯一必填，其余参数自动派生并逐行标来源（实测/派生/默认/手工） |
| `GUI/slm/multishape_campaign_ui/multishape_campaign_ui_user_manual.md` | multishape_campaign_ui.py 用户手册：多形状整形 campaign 选形状/定尺寸 → 派生参数 → 计划矩阵/时间磁盘预算/检查卡/历史复盘 |
| `GUI/slm/slm_calibration_ui/slm_calibration_ui_srs.md` | SLM LUT 校准 UI (slm_calibration_ui.py) 软件需求规格说明书 (SRS) |
| `GUI/slm/slm_calibration_ui/slm_calibration_ui_user_manual.md` | SLM LUT 校准 UI (slm_calibration_ui.py) 用户手册 |
| `GUI/slm/streamlit_theme.md` | SLM 整形 GUI 主题配置存档（`.streamlit/config.toml` 正本）：白底冷蓝操作台 + 8 档灰阶 + 语义色 + AA 对比度，附复原命令与 `streamlit_theme_config.toml` |
| `GUI/ccd/ccd_analyzer/ccd_analyzer_srs.md` | CCD 实时图像分析 UI (ccd_analyzer.py) 软件需求规格说明书 (SRS) |
| `GUI/ccd/ccd_analyzer/ccd_analyzer_user_manual.md` | CCD 实时图像分析 UI (ccd_analyzer.py) 用户手册 |
| `GUI/r50/r50_controller_ui/r50_controller_ui_srs.md` | R50 控制器控制面板 UI (r50_controller_ui.py) 软件需求规格说明书 (SRS) |
| `GUI/r50/r50_controller_ui/r50_controller_ui_user_manual.md` | R50 控制器控制面板 UI (r50_controller_ui.py) 用户手册 |
| `GUI/r50/ceramic_viewer/ceramic_viewer_srs.md` | 1300 陶瓷单元查看器 UI (ceramic_viewer.py) 软件需求规格说明书 (SRS) |
| `GUI/r50/ceramic_viewer/ceramic_viewer_user_manual.md` | 1300 陶瓷单元查看器 UI (ceramic_viewer.py) 用户手册 |
| `GUI/dm/micro_dm_ui/micro_dm_ui_srs.md` | Micro-DM 微驱动器控制 UI (micro_dm_ui.py) 软件需求规格说明书 (SRS) |
| `GUI/dm/micro_dm_ui/micro_dm_ui_user_manual.md` | Micro-DM 微驱动器控制 UI (micro_dm_ui.py) 用户手册 |
| `GUI/dm/md_img_analysis_ui/md_img_analysis_ui_srs.md` | Micro-DM 差分分析 UI (md_img_analysis_ui.py) 软件需求规格说明书 (SRS) |
| `GUI/dm/md_img_analysis_ui/md_img_analysis_ui_user_manual.md` | Micro-DM 差分分析 UI (md_img_analysis_ui.py) 用户手册 |
| `GUI/zernike/zernike_response_matrix_ui/zernike_response_matrix_ui_srs.md` | Zernike 响应矩阵校准 UI (zernike_response_matrix_ui.py) 软件需求规格说明书 (SRS) |
| `GUI/zernike/zernike_response_matrix_ui/zernike_response_matrix_ui_user_manual.md` | Zernike 响应矩阵校准 UI (zernike_response_matrix_ui.py) 用户手册 |
| `GUI/zernike/zernike_debug_viewer/zernike_debug_viewer_srs.md` | Zernike 响应矩阵调试查看器 UI (zernike_debug_viewer.py) 软件需求规格说明书 (SRS) |
| `GUI/zernike/zernike_debug_viewer/zernike_debug_viewer_user_manual.md` | Zernike 响应矩阵调试查看器 UI (zernike_debug_viewer.py) 用户手册 |

## 硬件品牌目录 — 驱动/SDK 细节

| 目录 | 篇数 | 说明 |
|---|---|---|
| `slm-200/` | 3 | Santec SLM-200：`agent_wiki.md`（驱动编写指南）、`dev.md`、`SLMFuncDLL_Programmer's_Guide(v2.5).md` |
| `miicam/` | 2 | MiiCam 相机：`agent_wiki.md`（核心概念）、`review.md` |
| `thorlab-wfs/` | 1 | Thorlabs WFS Python SDK 驱动编写指南 |
| `daheng/` | 4 | 大恒相机：Python 接口开发说明书、亮度调节、回调采集实战、曝光融合（exposure fusion，文献笔记） |
| `miicam_simulation/` | 1 | `miicam_report.md` MiiCam 仿真报告 |

## 工具（tools/）— 13 篇

| 文档 | 说明 |
|---|---|
| `tools/slm_abba_probe.md` | ABBA 消漂移探针：稠密随机相位可分辨性 + 本底测量（verdict=unusable 时勿测转移矩阵） |
| `tools/slm_bench_fingerprint.md` | 光路指纹：平场 FWHM + 已知倾斜位移 + 质心，判定标定所量光路是否仍是当前光路（防过期标定被静默复用） |
| `tools/micro_dm_image_collect.md` | Micro-DM 逐单元加电压采集远场光斑（响应矩阵/影响函数/系统标定） |
| `tools/slm_beam_extent.md` | 光束范围测量 |
| `tools/slm_diffraction_efficiency.md` | 闪耀光栅衍射效率 / 相位响应标定：一维闪耀光栅测器件相位调制深度与 ±1 级效率（`python -m ao_shaping.tools.slm.calibration.diffraction_efficiency {simulate,measure,analyze}`；measure 需光路标定 `.npz` 前置） |
| `tools/slm_drift_probe.md` | SLM 漂移探测 |
| `tools/slm_exposure_check.md` | 曝光检查 |
| `tools/slm_floor_probe.md` | 本底（floor）探测 |
| `tools/slm_focus_probe.md` | 焦面探测：plain 相位上扫菲涅尔焦距（1/f），逐点重调曝光，找让焦面回到相机的二次相位 |
| `tools/slm_panel_locate.md` | SLM 面板定位 |
| `tools/slm_phase_resolution.md` | 相位分辨率测量 |
| `tools/slm_tilt_probe.md` | 倾斜探测 |
| `tools/slm_zernike_sweep_probe.md` | Zernike 扫描探测 |

## 非文档资产（docs/ 根下）

- `0电压.bmp`、`to_load_V-9635.bmp` — 电压参考图
- `TM/` — 测试矩阵数据目录
