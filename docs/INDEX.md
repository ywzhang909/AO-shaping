# AO-Shaping 文档总索引

> 本文件是 `docs/` 的**入口索引**：按主题分域，每个文件一句话说明"这里能查到什么"。
> 共 120 个 Markdown 文档（2026-10-14 盘点）。新文档请在本索引登记后再提交。
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
| `hardware_drivers.md` | 硬件驱动清单（Thorlabs WFS / SLM / CCD / DM 各品牌支持矩阵） |
| `issues_report.md` | 问题与故障排查记录 |
| `ml_training.md` | ML 训练指南（`src/ml/` 可选依赖包：训练器/模型/指标，与硬件层解耦） |
| `simulation.md` | 数字孪生仿真环境总览（无硬件测试） |
| `tabu_search_algorithm.md` | 禁忌搜索算法实现说明（英文，`algorithm/tabu`） |
| `zernike_usage.md` | Zernike 两层入口用法（模式编号/系数换算入口） |
| `自适应 SPGD 光束整形闭环：完整方案总结.md` | 自适应 SPGD 方案：PPO 可信域思想 + 均匀化复合目标 + 自适应权重 + 历史信息复用 |

## CLI / 用户

| 文档 | 说明 |
|---|---|
| `user/cli-reference.md` | **CLI 正式参考**：全部 flag 拼写/取值域/默认值逐字来自 `--help` 转储 |
| `dev/coding_conventions.md` | 编码约定（Python 3.12+、导入、命名等） |
| `dev/todo_archive.md` | 已决事项档案（每条都是"一次决策的证据"，非变更日志） |

## 开发手册（call_graph_wiki）— 54 篇

`dev/call_graph_wiki/` 是**函数级开发手册**，独立索引见 [`dev/call_graph_wiki/index.md`](dev/call_graph_wiki/index.md)：

| 子集 | 篇数 | 内容 |
|---|---|---|
| `entry/` | 2 | `main.py` 入口、`cli()` 根命令组 |
| `commands/` | 20 | 每条命令：docstring 原文 + 调用链分析（自动）+ 硬件可达性 |
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
| `micro deformable mirror/docs/refactor.md` | Micro-DM 模块重构记录 |
| `micro deformable mirror/docs/README.md` | Micro-DM 模块说明 |

## WFS 波前传感器 — 4 条命令 + 报告

| 文档 | 说明 |
|---|---|
| `wfs/wfs_report.md` | WFS 硬件测试报告（2026-10-12 实测数据） |
| `wfs/ga_zernike.md` | 遗传算法搜索 Zernike 系数（WFS RMS 为适应度，无梯度/多峰场景） |
| `wfs/greedy_zernike.md` | 贪婪局部搜索（内部可选 SPGD/GA/PSO/SA/HC/RS/CEM/DE） |
| `wfs/closed_loop.md` | 基于已存 Zernike 响应矩阵的闭环波前优化（回放矩阵，免实时测量） |

## SLM 空间光调制器 — 15 篇

| 文档 | 说明 |
|---|---|
| `slm/slm_pib.md` | SLM Zernike 整形（CCD 远场光斑反馈，PIB） |
| `slm/spgd_square.md` | SLM 方形远场均匀性整形（SPGD，CV/EE/AR 目标） |
| `slm/slm_gsnet.md` | FREEFORM 自由相位方形整形 + FourierGSNet 训练 |
| `slm/slm_gs_refine.md` | GS 开环预矫正（0.662→0.812）+ SPGD 细化（→0.849）硬件移植 |
| `slm/slm_model_in_loop.md` | 正向模型闭环整形（每轮用实测数据修正模型再合成相位） |
| `slm/model_in_loop_algorithm.md` | 同上命令的**结构与算法**详解（调用关系/计算时序） |
| `slm/zernike_matrix.md` | SLM Zernike 响应矩阵标定 + 离线矫正自检 |
| `slm/rms_zernike.md` | SLM Zernike RMS 优化（SPGD/Adam/AdaMOD） |
| `slm/slm_lut.md` | 灰度-相位 LUT 标定（半屏参考/测试光栅，±1 级衍射效率比值） |
| `slm/闭环光束整形_文献与优化建议_20261014.md` | 闭环整形文献×实验×优化建议（GS/SPGD/正向模型/freeform，24 条 DOI 已核实） |
| `slm/slm_diagnose.md` | SLM+相机逐级硬件自检（2026-09 三步证据链固化） |
| `slm/pre_run_characterization.md` | 台架开机表征指南（先量台架的必要性） |
| `slm/slm_pattern_helper.md` | `PatternHelper` 相位图案生成工具类指南 |
| `slm/slm_gui_manual.md` | `multi_slm_controller.py` Streamlit 前端页面手册 |
| `slm/align_windowed.md` | SLM+CCD 装配对准（相机 ROI 开窗自动 0级/±1级 对准） |
| `slm/slm_patterns/shaping.md` | SLM 整形图案说明 |

## 硬件品牌目录 — 驱动/SDK 细节

| 目录 | 篇数 | 说明 |
|---|---|---|
| `slm-200/` | 3 | Santec SLM-200：`agent_wiki.md`（驱动编写指南）、`dev.md`、`SLMFuncDLL_Programmer's_Guide(v2.5).md` |
| `miicam/` | 2 | MiiCam 相机：`agent_wiki.md`（核心概念）、`review.md` |
| `thorlab-wfs/` | 1 | Thorlabs WFS Python SDK 驱动编写指南 |
| `daheng/` | 3 | 大恒相机：Python 接口开发说明书、亮度调节、回调采集实战 |
| `miicam_simulation/` | 1 | `miicam_report.md` MiiCam 仿真报告 |

## 工具（tools/）— 10 篇

| 文档 | 说明 |
|---|---|
| `tools/slm_abba_probe.md` | ABBA 消漂移探针：稠密随机相位可分辨性 + 本底测量（verdict=unusable 时勿测转移矩阵） |
| `tools/micro_dm_image_collect.md` | Micro-DM 逐单元加电压采集远场光斑（响应矩阵/影响函数/系统标定） |
| `tools/slm_beam_extent.md` | 光束范围测量 |
| `tools/slm_drift_probe.md` | SLM 漂移探测 |
| `tools/slm_exposure_check.md` | 曝光检查 |
| `tools/slm_floor_probe.md` | 本底（floor）探测 |
| `tools/slm_panel_locate.md` | SLM 面板定位 |
| `tools/slm_phase_resolution.md` | 相位分辨率测量 |
| `tools/slm_tilt_probe.md` | 倾斜探测 |
| `tools/slm_zernike_sweep_probe.md` | Zernike 扫描探测 |

## 非文档资产（docs/ 根下）

- `0电压.bmp`、`to_load_V-9635.bmp` — 电压参考图
- `TM/` — 测试矩阵数据目录
