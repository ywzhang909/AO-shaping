# Agent 参考索引（自根 AGENTS.md 迁出）

> 根 [AGENTS.md](../../AGENTS.md) 只保留 🔴 必读、项目概述与当日注意事项；
> 本文件汇总参考入口，按主题分组。**包/函数放置见根 AGENTS.md 必读**（module_inventory §WHERE TO LOOK），此处不重复。

## 脚本 / CLI / SLM 诊断

- 脚本约定：[scripts_conventions.md](scripts_conventions.md)
- CLI 参考：[../user/cli-reference.md](../user/cli-reference.md) · 入口点：[entry_points.md](entry_points.md)
- SLM 诊断工具：`tools/slm/panel/diagnose.py`（`slm-diagnose`）

## Zernike

- [zernike_usage.md](zernike_usage.md)

## 环境 / pytest

- [../environment.md](../environment.md) / [ci_testing.md](ci_testing.md)
- 集中配置：`src/ao_shaping/config.py`
- CUDA 原生库入口：[cuda_native_libs.md](cuda_native_libs.md)（`import cupy`/OOPAO 前先 `pin_cublas()`；数据见 environment.md §cuBLAS DLL 与 PATH）

## 门禁（ruff / 模块登记）

- `tests/ao_shaping/test_conventions.py::test_all_ruff_pins_agree_and_are_exact`
- `tests/ao_shaping/ml/test_train_cli_runnable.py::test_ruff_excludes_nothing_under_src`
- `tests/ao_shaping/test_tools_doc_coverage.py::test_every_tools_module_is_documented`
- `tests/ao_shaping/test_runners_doc_paths.py::test_every_runner_module_is_documented`
- `tests/ao_shaping/test_optimizer_doc_coverage.py::test_every_optimizer_module_is_documented`
- 详见 [ci_testing.md](ci_testing.md) §Linting / 模块登记门禁

## 报告规范

- [report/AGENTS.md](../../report/AGENTS.md)

## CodeGraph

- 仓库已索引；探索用 `codegraph_explore` / `codegraph explore`；无 `.codegraph/` 时跳过。
