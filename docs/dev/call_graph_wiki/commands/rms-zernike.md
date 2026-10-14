---
title: rms-zernike（SLM Zernike RMS 优化（SPGD/启发式））
created: 2026-10-13
updated: 2026-10-13
type: command
tags: [command]
source: src/ao_shaping/runners/slm/rms_zernike_runner.py:109
---

## 一句话

SLM Zernike RMS 优化（SPGD/启发式）

## 入口函数

`runners.slm.rms_zernike_runner.run`（`src/ao_shaping/runners/slm/rms_zernike_runner.py:109`）

```python
run(ctx, params, wfs, slm)
```

## docstring（原文节选）

```
Zernike波前优化器 (基于SLM的RMS最小化)

    使用Zernike多项式通过SLM进行波前校正，最小化WFS测量的波前RMS值。
    ``--algorithm spgd`` 使用梯度/SPGD；其他取值 (ga/pso/sa/hc/rs/cem/de) 使用
    algorithm/heuristic 的启发式搜索。

    调试模式: ``main.py --debug rms-zernike`` / 本命令 ``--debug`` / ``DEBUG=1``。
```

## 调用链分析（自动）

**硬件**：WFS 波前传感器

**调用的优化器/算法**：

- `_get_perturb_weights`
- `ao_shaping.algorithm.heuristic.search.heuristic_algorithm_choices`
- [[functions/run-heuristic-search|`run_heuristic_search`]]
- [[functions/spgd-gradient|`spgd_gradient`]]
- [[functions/optimizer-rms-slm|`optimizer_rms_slm`]]
- `compute_mini_batch_gradient`
- `create_heuristic`
- `early_stopping_check`
- `get_delta_schedule`
- `get_lr_schedule`

**结果/调试落盘**：`_save_data_mode_debug_artifacts`、`build_debug_save_paths`、`save_optimization_debug_artifacts`、`gen_date_dir`

## 直接调用（1 级，按调用次数）

- [[functions/with-params|`with_params`]]
- [[functions/debug-artifacts|`build_debug_save_paths`]]
- `_auto_delta_detect_rms`
- [[functions/optimizer-rms-slm|`optimizer_rms_slm`]]
- `ao_shaping.utils.io.cli_helpers.resolve_debug`
- [[functions/debug-artifacts|`save_optimization_debug_artifacts`]]
- [[functions/matrix-utils|`calc_n_zernike_terms`]]

## 谁调用它

- [[entry/cli|`cli`]]（`cli.add_command` 注册为 `rms-zernike`）

## 相关页面

- [[entry/cli]]
- [[functions/run-heuristic-search|`run_heuristic_search`]]
- [[functions/spgd-gradient|`spgd_gradient`]]
