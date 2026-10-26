# [分析] scripts/ 功能二级分类与复用分析报告及重构方案

> 分析对象：`scripts/` 全部 Python 脚本（288 个 `.py`）。
> 状态：本报告为**分析 + 方案**，未改动任何业务代码。完整版见仓库
> `docs/scripts_functional_classification.md`（本地提交 `290e3a08`），
> 全量映射见 `docs/scripts_functional_map.json`。

## 摘要

`scripts/` 当前只按**设备域**做了一级分类（`slm/` 155、`ml/` 70、`rl/` 11、`dm/` 11…），
但同设备下功能跨度极大，无法表达"脚本到底干什么"，也导致大量功能被重复实现。
本报告在设备域之上补充**功能二级分类**（11 类），并识别跨脚本的重复功能与可复用函数，
给出分阶段、契约测试驱动、可逆的重构方案。

## 1. 二级（功能）分类方案（11 类）

| 代号 | 功能类 | 典型前缀 | 文件数 |
|---|---|---|---:|
| A | 报告生成（Markdown） | `generate_*_report` / `write_report` | 58 |
| B | 图像 / 图表生成 | `render_*` / `*_figures` / `md_img_diff_*` | 6 |
| C | 动画 / GIF 生成 | `*_gif` / `render_*_animation` | 19 |
| D | 数据采集 / 探针 | `capture_*` / `collect_*` / `*_probe` | 20 |
| E | 建模 / 拟合 / 训练 | `fit_*` / `refit_*` / `train*` / `inverse_*` | 22 |
| F | 分析 / 诊断 | `analyze_*` / `diagnose_*` / `evaluate_*` / `compare_*` | 72 |
| G | 验证 / 校验 / 健康检查 | `verify_*` / `validate_*` / `check_*` | 9 |
| H | 调参 / 标定 / 校准 | `tune_*` / `calib*` / `sweep_*` | 3 |
| I | 实验编排 / 运行 | `run_*` / `experiment_*` / `demo_*` / `model_in_loop_*` | 25 |
| J | 计划 / 汇总 / 归档 | `summarize_*` / `archive_*` / `audit_*` | 17 |
| K | 工具 / 基础设施 | `call_graph/*` / `build/*` / `_common/*` | 2 |
| Z | 其他 / 未分类（共享片段等） | — | 23 |

与需求对应：**生成 md 报告 → A**；**生成图片 → B/C**；**辅助调整参数 → H/I**。

## 2. 设备 × 功能 分布矩阵

| 设备域 | A | B | C | D | E | F | G | H | I | J | K | Z | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `slm/` | 34 | 3 | 17 | 15 | 7 | 31 | 8 | 2 | 19 | 9 | 0 | 9 | 154 |
| `ml/` | 10 | 0 | 0 | 2 | 14 | 38 | 1 | 0 | 0 | 4 | 0 | 0 | 69 |
| `dm/` | 2 | 3 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 10 |
| `rl/` | 3 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 2 | 0 | 0 | 3 | 10 |
| `zernike_bench/` | 4 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| `call_graph/` | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 3 | 0 | 3 | 7 |
| `wfs/` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 2 | 3 |
| 其他零散 | 5 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 2 | 0 | 2 | 1 | 12 |
| **合计** | **58** | **6** | **19** | **20** | **22** | **72** | **9** | **3** | **25** | **17** | **2** | **23** | **276** |

## 3. 重复功能与可复用函数发现（核心）

虽已有 `_common/` 抽取了报告生成族的部分能力，但覆盖严重不足：

| 重复单元 | 重定义文件数 | 是否已抽取 | 说明 |
|---|---:|---|---|
| `write_report` | 16 | 否 | Markdown 报告写出 |
| `write_csv` | 11 | 否 | CSV 写出 |
| `build_markdown` | 10 | 否（`_common` 仅有 `markdown_table`） | 报告正文拼接，**签名已漂移 3 种** |
| argparse 脚手架（`build_parser` / `_build_parser` / `parse_args`） | 32 | 否 | 近似的 CLI 参数定义 |
| 运行产物加载（`load_run` / `_load` / `_read_json` 等） | ≈30 | 否 | 重复读取 PKL/JSON 调试产物 |
| 图像渲染（`render_figures` / `setup_cjk_font` 等） | ≈18 | 部分 | `_common` 有 `savefig`/`plot_summary_bars` |
| ML 统计（`folds_of` / `ridge` / `score`） | ≈24 | 否 | 集中在 `ml/` 互相拷贝 |

**反模式**：`_common` 已提供 `plot_summary_bars`，仍有 3 个文件本地重定义。
**覆盖面**：全仓 288 文件仅 53 个引用 `_common`。
**隐患**：`build_markdown` 签名漂移正是 `_common` 当初被创建的同源问题——
历史上两副本 `_fmt` 不一致曾把 `1e-7` 渲染成 `0.0000`，在提交报告中压平测量值。

## 4. 修改方案（分阶段、契约测试驱动、可逆）

- **阶段 0**：生成 `scripts/_common/scripts_index.json` 作检索单一事实源。
- **阶段 1**：扩展 `_common`（`write_markdown_report` / `write_csv` / `build_cli` /
  `load_run_artefacts` / `render_summary_figures`）；**先固化契约 + 契约测试**再替换，
  沿用 `tests/ao_shaping/scripts/test_common_helpers.py` 模式，防签名漂移重演。
- **阶段 2**：推荐"软分类"不搬迁（仅用索引 + 命名前缀约定 + 文档固化），而非物理改目录。
- **阶段 3~5**：渐进迁移带测试、清理本地重定义、每次改动后 `term_unify` 复核。

## 5. 待办 / 讨论点

- [ ] 是否采纳"软分类"方案，还是物理重组目录？
- [ ] 优先抽取哪个公共函数（建议从重复度最高、风险最低的 `write_csv` / `write_report` 入手）？
- [ ] 是否需要为 `build_markdown` 签名漂移做一次性统一契约？

---
*本 issue 正文由分析脚本自动汇总；完整报告见 `docs/scripts_functional_classification.md`。*
