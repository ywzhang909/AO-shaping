---
title: 相似函数（自动）— 函数间相似关联总览
created: 2026-10-13
updated: 2026-10-13
type: diagram
tags: [similar,relations]
source: scripts/call_graph/build_similarity.py
---

## 这是什么

调用图回答「谁调用谁」，本页回答**「哪些函数是彼此的同族/同构实现」**：不共享调用边、但结构或职责高度相近的函数对（near-duplicate / sibling implementations）。这是独立于调用图的**第二条关系层**，用于：

- 改一个函数时，提醒「还有几个同族函数需要同步改」（如两处 `optimize_pib`）；
- 重构时识别可合并/可抽象的重复实现；
- 新增功能时参考「同类功能在仓库里已有的几种做法」。

## 判定方法（自动，可重跑）

由 `scripts/call_graph/build_similarity.py` 对 `call_graph_final.json` 的全部已解析函数两两打分（确定性权重，无随机、无模型）：

| 信号 | 权重 | 含义 |
|---|---|---|
| 被调集合 Jaccard（乘 `min(1, shared/2)` 阻尼） | 0.45 | 两个函数调用结构越像，越可能是同族实现 |
| 名称相似度 | 0.15 | 同名（如两处 `optimize_pib`）计 1；通用 CLI 入口名 `run` 计 0（避免假相似） |
| 同文件 / 同模块 | 0.1 | 同文件 1.0、同模块 0.5 |
| 调用方集合 Jaccard | 0.05 | 服务对象越像，越可能是同族 |
| 共享被调 ≥5 奖励 | 0.05 | 调用结构高度重叠的额外加分 |

得分 ≥ **0.3** 记为**相似对**（写入各函数页「相似函数（自动）」段）；≥ **0.35** 的相似对合并成**相似簇**（下表）。数据文件：`raw/assets/similar_functions.json`（每对含逐维得分与理由）。重跑 `python3 scripts/call_graph/build_similarity.py` 即可刷新。

## 相似簇（≥ 簇阈值，自动）

| 簇 | 成员 |
|---|---|
| [[functions/runner-common|`build_gs_refine_config`]] — [[functions/runner-common|`build_model_in_loop_config`]] |
| [[commands/pib|`run`]] — [[commands/combined|`run`]] |
| [[functions/optimize-pib-combined|`optimize_pib`]] — [[functions/optimize-pib-wfless|`optimize_pib`]] |
| [[functions/slm-lut|`depth_pattern`]] — [[functions/slm-lut|`offset_pattern`]] |

## 全部相似对（≥ 链接阈值，按得分降序）

| 得分 | 函数 A | 函数 B | 判定理由 |
|---|---|---|---|
| 0.438 | [[functions/runner-common|`build_gs_refine_config`]] | [[functions/runner-common|`build_model_in_loop_config`]] | 共享 3 个被调函数；同文件 runner_common.py（同调 0.75 / 共享 3 个被调） |
| 0.396 | [[commands/pib|`run`]] | [[commands/combined|`run`]] | 同名 `run`；共享 10 个被调函数（调用结构高度相似）（同调 0.77 / 共享 10 个被调） |
| 0.384 | [[functions/optimize-pib-wfless|`optimize_pib`]] | [[functions/optimize-pib-combined|`optimize_pib`]] | 同名 `optimize_pib`；共享 9 个被调函数（调用结构高度相似）（同调 0.41 / 共享 9 个被调） |
| 0.375 | [[functions/slm-lut|`depth_pattern`]] | [[functions/slm-lut|`offset_pattern`]] | 同文件 slm_lut.py；调用方高度重叠（同调 1.00 / 共享 1 个被调） |
| 0.336 | [[commands/wf|`run`]] | [[commands/pipeline|`run`]] | 同名 `run`；共享 7 个被调函数（调用结构高度相似）（同调 0.64 / 共享 7 个被调） |
| 0.325 | [[functions/path-dirs|`get_date_dir_name`]] | [[functions/path-dirs|`get_timestamp_str`]] | 同文件 cli_helpers.py（同调 1.00 / 共享 1 个被调） |

## 使用建议

1. **改一个函数前**先看本页与它的「相似函数（自动）」段——同族函数通常需要同步修改，否则会出现行为漂移（仓库里 `optimize_pib` 双实现是典型例子）。
2. 得分 0.30–0.45 的多数是「同一思路在不同设备域/不同算法上的平行实现」（DM 族 vs SLM 族），**不是**复制粘贴——合并前需确认硬件语义是否可抽象。
3. 新代码改动后重跑 `build_call_graph.py` + `build_similarity.py` + `gen_wiki_pages.py`，本页与全部函数页自动刷新。
