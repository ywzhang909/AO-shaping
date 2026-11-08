# AGENTS.md - AO-Shaping Development Guide

**Generated:** 2026-11-08

## 🔴 必读（所有 AI 工具）

- **硬件设备纪律**：[docs/dev/hardware_discipline.md](docs/dev/hardware_discipline.md)（SLM/CCD/激光占用须在回复中明确报告；在飞保护与 `--wait-until-idle` 见该文件）。
- **全局编码规范**：[docs/dev/coding_conventions.md](docs/dev/coding_conventions.md)（Python 3.12+、绝对导入、类型注解、日志等）。
- **包/函数放置**：[docs/dev/module_inventory.md](docs/dev/module_inventory.md) §WHERE TO LOOK。
- 术语：[docs/terminology_glossary.md](docs/terminology_glossary.md)（校验 `python3 scripts/_common/term_unify_20261016.py` dry-run `0 个文件`；新词 §3 登记）。
- 工程纪律：[docs/dev/engineering_discipline.md](docs/dev/engineering_discipline.md)。

## Project Overview

AO-Shaping：自适应光学（AO）系统，RL 波前校正与光束整形；集成 WFS 与无波前传感器方法。
逐文件详解 [docs/dev/project_structure.md](docs/dev/project_structure.md)。

## References

- 脚本 / CLI / SLM 诊断：[docs/dev/scripts_conventions.md](docs/dev/scripts_conventions.md) · [docs/user/cli-reference.md](docs/user/cli-reference.md) / [docs/dev/entry_points.md](docs/dev/entry_points.md) · `tools/slm/panel/diagnose.py` (slm-diagnose)
- Zernike：[docs/dev/zernike_usage.md](docs/dev/zernike_usage.md)（模块目录树见 [docs/dev/module_inventory.md](docs/dev/module_inventory.md) §Project Structure Tree）
- 环境 / pytest：[docs/environment.md](docs/environment.md) / [docs/dev/ci_testing.md](docs/dev/ci_testing.md)；集中配置 `src/ao_shaping/config.py`；**CUDA 原生库入口**：[docs/dev/cuda_native_libs.md](docs/dev/cuda_native_libs.md)（`import cupy`/OOPAO 前先 `pin_cublas()`；数据见 environment.md §cuBLAS DLL 与 PATH）。
- **门禁（ruff / 模块登记）**：[docs/dev/ci_testing.md](docs/dev/ci_testing.md) §Linting / 模块登记门禁（各测试名与 `>=` 禁令在此，不在本文件重复）。
- 报告规范：[report/AGENTS.md](report/AGENTS.md)
- Agent 注意事项（当天）：[docs/dev/agent_notes_20261108.md](docs/dev/agent_notes_20261108.md)；历史按日见 `docs/dev/agent_notes_*.md`
- CodeGraph：仓库已索引；探索用 `codegraph_explore` / `codegraph explore`；无 `.codegraph/` 时跳过。
