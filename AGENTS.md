# AGENTS.md - AO-Shaping Development Guide

**Generated:** 2026-11-15

## 🔴 必读（所有 AI 工具）

- **硬件设备纪律**：[docs/dev/hardware_discipline.md](docs/dev/hardware_discipline.md)（SLM/CCD/激光占用须在回复中明确报告；在飞保护与 `--wait-until-idle` 见该文件）。
- **全局编码规范**：[docs/dev/coding_conventions.md](docs/dev/coding_conventions.md)（Python 3.12+、绝对导入、类型注解、日志等）。
- **包/函数放置**：[docs/dev/module_inventory.md](docs/dev/module_inventory.md) §WHERE TO LOOK。
- 术语：[docs/terminology_glossary.md](docs/terminology_glossary.md)（校验 `python3 scripts/_common/term_unify_20261016.py` dry-run `0 个文件`；新词 §3 登记）。
- 工程纪律：[docs/dev/engineering_discipline.md](docs/dev/engineering_discipline.md)。

## Project Overview

AO-Shaping：自适应光学（AO）系统，RL 波前校正与光束整形；集成 WFS 与无波前传感器方法。
逐文件详解 [docs/dev/project_structure.md](docs/dev/project_structure.md)。

## 当日注意事项与历史按日记录

最新：[docs/dev/agent_notes_20261116.md](docs/dev/agent_notes_20261116.md)；历史按日见 `docs/dev/agent_notes_*.md`。

## References

脚本 / CLI / Zernike / 环境 / pytest / 门禁等参考索引见
**[docs/dev/agent_reference_index.md](docs/dev/agent_reference_index.md)**。
