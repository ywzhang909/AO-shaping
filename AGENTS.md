# AGENTS.md - AO-Shaping Development Guide

**Generated:** 2026-06-20

## 🔴 必读（所有 AI 工具）

- 术语：[docs/terminology_glossary.md](docs/terminology_glossary.md)（校验 `python3 scripts/_common/term_unify_20261016.py` dry-run `0 个文件`；新词 §3 登记）。
- 全局编码规范：[docs/dev/coding_conventions.md](docs/dev/coding_conventions.md)（Python 3.12+、绝对导入、类型注解、日志等）。
- 包/函数放置：[docs/dev/module_inventory.md](docs/dev/module_inventory.md) §WHERE TO LOOK。
- 工程纪律：[docs/dev/engineering_discipline.md](docs/dev/engineering_discipline.md)。

## Project Overview

AO-Shaping：自适应光学（AO）系统，RL 波前校正与光束整形；集成 WFS 与无波前传感器方法。

## Project Structure

逐文件详解 [docs/dev/project_structure.md](docs/dev/project_structure.md)；目录树 [docs/dev/module_inventory.md](docs/dev/module_inventory.md) §Project Structure Tree。

## References

- 脚本约定：[docs/dev/scripts_conventions.md](docs/dev/scripts_conventions.md)
- 任务→位置 / 模块登记 / Zernike：[docs/dev/module_inventory.md](docs/dev/module_inventory.md)；[docs/dev/zernike_usage.md](docs/dev/zernike_usage.md)
- 环境/pytest：[docs/environment.md](docs/environment.md) / [docs/dev/ci_testing.md](docs/dev/ci_testing.md)；集中配置 `src/ao_shaping/config.py`
- **CUDA 原生库入口硬规则**（`import cupy` / OOPAO 必须先 `pin_cublas()`）：见本文件 §CUDA 原生库入口 与 [docs/environment.md](docs/environment.md) §cuBLAS DLL 与 PATH
- CLI：[docs/user/cli-reference.md](docs/user/cli-reference.md) / [docs/dev/entry_points.md](docs/dev/entry_points.md)
- SLM 诊断：`tools/slm/panel/diagnose.py` (slm-diagnose)
- ruff/模块登记门禁：`tests/ao_shaping/test_conventions.py::test_all_ruff_pins_agree_and_are_exact`、`tests/ao_shaping/ml/test_train_cli_runnable.py::test_ruff_excludes_nothing_under_src`；`tests/ao_shaping/test_runners_doc_paths.py::test_every_runner_module_is_documented`、`tests/ao_shaping/test_tools_doc_coverage.py::test_every_tools_module_is_documented`、`tests/ao_shaping/test_optimizer_doc_coverage.py::test_every_optimizer_module_is_documented`
- 报告规范：[report/AGENTS.md](report/AGENTS.md)
- Agent 注意事项（当天）：[docs/dev/agent_notes_20261103.md](docs/dev/agent_notes_20261103.md)
- CodeGraph：仓库已索引；探索用 `codegraph_explore` / `codegraph explore`；无 `.codegraph/` 时跳过。

## 🔴 CUDA 原生库入口（硬规则 · 2026-11-03 闪退事故后补）

裸名 `cublasLt64_<M>.dll` / `cublas64_<M>.dll` 的绑定**由谁先进进程决定**，代码里看不出
问题，线上表现为**无 traceback 的 `0xC0000005` 闪退**或 `CUBLAS_STATUS_INVALID_VALUE`。

1. **禁止**新增「import 期就绑定 CUDA 原生库」的入口：`src/` 下任何模块级（含模块级
   `try:` / 类体内）的 `import cupy`、`import OOPAO`，其**之前**必须调用
   `ao_shaping._cublas_preload.pin_cublas()`。门禁：
   `tests/ao_shaping/test_native_cuda_import_gate.py`。
2. **禁止**把这些 import 放进 `ao_shaping/__init__.py`（违反 `drivers/AGENTS.md` 的
   「import 不加载原生库、不落盘」契约）。
3. 拿不准就**延迟 import**（放函数内）：CuPy 只是可选加速时，最干净的做法是让它根本不在
   import 期出现（`utils/image/spots_calc.py` 的 4 个 `*_cupy` 函数即此类）。
4. **已实测不成立、禁止再当修法提出**（数据见 `docs/environment.md` §cuBLAS DLL 与 PATH）：
   `.venv` 隔离与 `uv run` **不会**清理 `PATH`；`os.add_dll_directory(torch/lib)` 对 cupy
   的裸名加载**无效**；把 CUDA 目录从 `PATH` 删掉**无效**（scoop 那份仍在）；只钉 torch 的
   `cublasLt`（不钉 `cublas`）会让 cupy **全线报错**。
5. 根治只有一条：**环境里只留一套 CUDA minor**，且它必须是完整集合（含 cusolver/cufft/
   curand）；torch 只自带 cublas/cublasLt/cudart 三个 `*64_13.dll`，永远构不成完整集合。
   uv 侧对照见 `pyproject.toml` 的 extras 注释。