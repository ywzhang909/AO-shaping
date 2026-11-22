# AGENTS.md - AO-Shaping Development Guide

**Generated:** 2026-11-15

## 🔴 必读（所有 AI 工具）

- **硬件设备纪律**：[docs/dev/hardware_discipline.md](docs/dev/hardware_discipline.md)（SLM/CCD/激光占用须在回复中明确报告；在飞保护与 `--wait-until-idle` 见该文件）。
- **上机前光路与设备标定**：[libs/skills/ao-optical-path-calibration/SKILL.md](libs/skills/ao-optical-path-calibration/SKILL.md)（设备在线 / 光路有效 / 曝光不超量程 / SLM 真在调制的六步探测，以及 `optical_path_record.json` 标准记录；含未解冲突清单）。
- **全局编码规范**：[docs/dev/coding_conventions.md](docs/dev/coding_conventions.md)（Python 3.12+、绝对导入、类型注解、日志等）。
- **包/函数放置**：[docs/dev/module_inventory.md](docs/dev/module_inventory.md) §WHERE TO LOOK。
- 术语：[docs/terminology_glossary.md](docs/terminology_glossary.md)（校验 `python3 scripts/_common/term_unify_20261016.py` dry-run `0 个文件`；新词 §3 登记）。
- **SLM 物理 2π 灰度从设备读取**：`Santec.get_wavelength_info()[1]`（1064 nm 典型读数 993；非 radian→gray 编码标度 1023，两者严格区分，见 [决策记录](docs/dev/two_pi_grayscale_source_decision_20261119.md)）；#238 光学拟合简并、相位反演公式纠错（勿用归一化 e0n 的 cos 反演）与串扰不可判定的结论见 [docs/dev/agent_notes_20261119.md](docs/dev/agent_notes_20261119.md) 与 [docs/slm/slm_lut.md](docs/slm/slm_lut.md) §10。
- 工程纪律：[docs/dev/engineering_discipline.md](docs/dev/engineering_discipline.md)。
- **渲染层不把 CLI 校验错误当兜底**：`campaign_matrix()` 对空形状集合抛 `ValueError` 是它的 CLI 入参契约，不是给渲染兜底用的；`campaign_monitor.render_progress_panel` 遇到没有 `plan.shapes` 的旧 journal / 空目录（正常历史态，`data/debug/slm_transfer` 下真实存在）必须先判空、跳过建矩阵并退到读取器合计，且对形状名不被本版识别的 `ValueError` 同样降级（2026-11-20 修，见 `tests/ao_shaping/gui/slm/test_campaign_monitor.py::test_progress_panel_survives_a_legacy_directory_without_a_plan`）。

## Project Overview

AO-Shaping：自适应光学（AO）系统，RL 波前校正与光束整形；集成 WFS 与无波前传感器方法。
逐文件详解 [docs/dev/project_structure.md](docs/dev/project_structure.md)。

## 当日注意事项与历史按日记录

最新：[docs/dev/agent_notes_20261122.md](docs/dev/agent_notes_20261122.md)；历史按日见 `docs/dev/agent_notes_*.md`。

## References

脚本 / CLI / Zernike / 环境 / pytest / 门禁等参考索引见
**[docs/dev/agent_reference_index.md](docs/dev/agent_reference_index.md)**。

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- When graphify-out/graph.json exists, run `graphify query "<question>"` before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Read raw files only for lines the graph or an earlier read already located. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. When the answer is under a Markdown heading, `graphify section "<heading>"` prints that section's text, so the .md need not be opened. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
- A committed graphify-out/graph.json is rewritten whole on every build, so a large diff to it in a pull request is expected. Do not flag it as churn or suggest skipping the graph rebuild or its hooks.
