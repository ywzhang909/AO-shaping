# AGENTS.md - AO-Shaping Development Guide

**Generated:** 2026-11-03

## 🔴 必读（所有 AI 工具）

- 术语：[docs/terminology_glossary.md](docs/terminology_glossary.md)（校验 `python3 scripts/_common/term_unify_20261016.py` dry-run `0 个文件`；新词 §3 登记）。
- **全局编码规范**：[docs/dev/coding_conventions.md](docs/dev/coding_conventions.md)（Python 3.12+、绝对导入、类型注解、日志等）。
- **包/函数放置**：[docs/dev/module_inventory.md](docs/dev/module_inventory.md) §WHERE TO LOOK。
- 工程纪律：[docs/dev/engineering_discipline.md](docs/dev/engineering_discipline.md)。

## Project Overview

AO-Shaping：自适应光学（AO）系统，RL 波前校正与光束整形；集成 WFS 与无波前传感器方法。

## Project Structure

逐文件详解 [docs/dev/project_structure.md](docs/dev/project_structure.md)；目录树 [docs/dev/module_inventory.md](docs/dev/module_inventory.md) §Project Structure Tree。

## References

- 脚本约定：[docs/dev/scripts_conventions.md](docs/dev/scripts_conventions.md)
- 任务→位置 / 模块登记 / Zernike：[docs/dev/module_inventory.md](docs/dev/module_inventory.md)；[docs/dev/zernike_usage.md](docs/dev/zernike_usage.md)
- 环境/pytest：[docs/environment.md](docs/environment.md) / [docs/dev/ci_testing.md](docs/dev/ci_testing.md)；集中配置 `src/ao_shaping/config.py`
- **CUDA 原生库入口**：[docs/dev/cuda_native_libs.md](docs/dev/cuda_native_libs.md)（`import cupy`/OOPAO 前先 `pin_cublas()`）；数据见 [docs/environment.md](docs/environment.md) §cuBLAS DLL 与 PATH
- CLI：[docs/user/cli-reference.md](docs/user/cli-reference.md) / [docs/dev/entry_points.md](docs/dev/entry_points.md)
- SLM 诊断：`tools/slm/panel/diagnose.py` (slm-diagnose)
- ruff/模块登记门禁：`tests/ao_shaping/test_conventions.py::test_all_ruff_pins_agree_and_are_exact`、`tests/ao_shaping/ml/test_train_cli_runnable.py::test_ruff_excludes_nothing_under_src`；`tests/ao_shaping/test_runners_doc_paths.py::test_every_runner_module_is_documented`、`tests/ao_shaping/test_tools_doc_coverage.py::test_every_tools_module_is_documented`、`tests/ao_shaping/test_optimizer_doc_coverage.py::test_every_optimizer_module_is_documented`
- 报告规范：[report/AGENTS.md](report/AGENTS.md)
- Agent 注意事项（当天）：[docs/dev/agent_notes_20261103.md](docs/dev/agent_notes_20261103.md)
- CodeGraph：仓库已索引；探索用 `codegraph_explore` / `codegraph explore`；无 `.codegraph/` 时跳过。