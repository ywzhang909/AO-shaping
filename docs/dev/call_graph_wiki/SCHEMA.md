# Wiki Schema — AO-shaping 开发使用手册

## Domain

本 wiki 覆盖 **AO-Shaping 仓库**（`projects/AO-shaping/repo/`）的**开发使用**：
以 `main.py` 为根入口，梳理其 Click CLI 的 20 个命令、命令到 runner / optimizer /
driver 的函数调用图，并为每个骨架函数提供"逐函数"的优化文档（签名、真实 docstring、
入/出边调用关系、调试模式、产物落盘）。

目标读者：要在本仓库上加新功能、改命令、调参、排障的开发者（含 LLM 协作者）。

**范围边界**：本 wiki 只做"调用图 + 逐函数文档 + 使用手册"。它**不**替代仓库自身的
`AGENTS.md`（架构/约定）与 `docs/`（硬件、驱动细节）；遇到硬件/驱动/算法细节时，
页面会指向对应仓库文件路径，不在此重复。

## Conventions

- 文件用小写短横线命名（`commands/wf.md`、`functions/optimizer-rms-dm.md`）。
- 每个 wiki 页首有 YAML frontmatter（`title/created/updated/type/tags/source`）。
  `source` 字段记录该页文档**对应的仓库源码位置**（`path:line`），是"逐函数"的锚点。
- 用 `[[wikilinks]]` 互链，每页至少 2 个出链。
- **数据溯源**：所有调用关系来自 `scripts/call_graph/build_call_graph.py` 的 AST 静态分析
  （产物 `raw/assets/call_graph_final.json`），docstring 逐字摘自源码，**不臆造**。
  每个命令页标注了"调用链分析（自动）"段，硬件/算法/落盘均为自动可达性结果。
- **区分已解析/未解析**：静态分析无法解析到的目标（反射、动态 import、第三方库）
  在页面里显式标为 `*(未解析/外部)*`，不硬连边。
- 命令名 → runner 符号的映射不是纯派生的（`closed-loop`→`zernike_closed_loop_run`、
  `spgd-square`→`slm_square_run`），以 `entry/cli.md` 的 20 行注册表为唯一权威。

## Frontmatter

```yaml
---
title: 页面标题
created: YYYY-MM-DD
updated: YYYY-MM-DD
type: entry | command | function | class | diagram
tags: [见下方分类]
source: src/ao_shaping/.../file.py:line     # 该页对应的源码锚点
---
```

`type` 取值：
- `entry` — 入口（`main.py` / `cli()`）
- `command` — 一个 CLI 命令（一个 `cli.add_command` 注册的 runner）
- `function` / `class` — 被命令调用的骨架函数/类（逐函数文档）
- `diagram` — 调用图总览页

## Tag Taxonomy

- 入口/CLI：`entry`, `cli`, `params`, `debug`
- 命令族：`command`
- 优化目标：`wavefront`, `pib`, `shaping`, `square`, `zernike`, `calibration`, `matrix`
- 算法：`spgd`, `ga`, `pso`, `greedy`, `heuristic`, `gradient`, `adapmod`
- 硬件/驱动：`dm`, `slm`, `ccd`, `wfs`, `driver`, `registry`
- 工具/IO：`io`, `utils`, `time`, `recorder`, `lut`, `image`, `spots`, `wavefront`
- 结构：`module`, `factory`

规则：页面上的 tag 必须在本分类中。需要新 tag 时先在此登记再用。

## Page Thresholds

- **建页**：命令（20 个全建）+ 被 ≥1 个命令直接/间接调用的**骨架**函数（optimizer、
  driver 工厂、CLI 参数装饰器、LUT/矩阵/光斑工具、调试落盘）。
- **不建页**：纯内部辅助（`_*` 私有函数、单命令内部的局部 helper）、第三方库函数、
  仅 1 处且无复用价值的琐碎工具——这些在命令页的"直接调用"列表里点到即可。
- **拆页**：单页 > 200 行时拆成子主题并互链。

## 页面结构

### Entry 页
职责 → 签名 → 调用关系 → 关联页。

### Command 页（逐命令）
一句话 → 入口函数（签名）→ docstring 原文节选 → **调用链分析（自动）**
（硬件 / 调用的优化器算法 / 结果落盘）→ 直接调用（1 级出边）→ 谁调用它（入边）→ 关联页。

### Function 页（逐函数）
签名（含返回类型/装饰器）→ docstring 原文 → 直接调用（出边）→ 谁调用它（入边）→ 关联页。

## Update Policy

- 代码改动后，重跑 `scripts/call_graph/build_call_graph.py` + `gen_wiki_pages.py` 重新生成
  调用图与命令/函数页（幂等，`source` 行号会更新）。
- docstring 变更：wiki 页"docstring 原文"段应同步。若 wiki 与源码 docstring 冲突，
  **以源码为准**，更新 wiki 的 `updated` 日期。
- 新增命令：在 `entry/cli.md` 注册表 + `commands/` 下建页 + `index.md` 登记。
