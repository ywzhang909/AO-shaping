---
title: pib（轴向光束 PIB 优化（DM 电压 + CCD））
created: 2026-10-13
updated: 2026-10-13
type: command
tags: [command]
source: src/ao_shaping/runners/nlight_dm/axis_beam_runner.py:44
---

## 一句话

轴向光束 PIB 优化（DM 电压 + CCD）

## 入口函数

`runners.nlight_dm.axis_beam_runner.run`（`src/ao_shaping/runners/nlight_dm/axis_beam_runner.py:44`）

```python
run(ctx, device, run, optical, algorithm)
```

## docstring（原文节选）

```
轴向光束优化器

    调试模式: ``main.py --debug pib`` / 本命令 ``--debug`` / 环境变量 ``DEBUG=1``
    任一开启即可输出 pkl/json 与汇总图片。
```

## 调用链分析（自动）

**硬件**：CCD 相机、DM 变形镜

**调用的优化器/算法**：

- `_extract_optimizer_momentum`
- `_generate_search_candidates`
- `_objective_to_min`
- `_reset_optimizer_state`
- `_should_trigger_adaptive_search`
- `ao_shaping.optimizer._camera_inject._camera_cls`
- `ao_shaping.optimizer.constants.create_optimizer`
- [[functions/spgd-gradient|`spgd_gradient`]]
- `ao_shaping.optimizer.spid_pib_loop.open_pib_bench`
- `ao_shaping.optimizer.spid_pib_loop.spid_pib_epoch`
- [[functions/optimize-pib-wfless|`optimize_pib`]]
- `learning_schedule`

**结果/调试落盘**：`save_best_voltages`、`gen_date_dir`、`make_date_dir`

## 直接调用（1 级，按调用次数）

- [[functions/with-params|`with_params`]]
- [[functions/resolve-dm|`resolve_dm`]]
- [[functions/optimize-pib-wfless|`optimize_pib`]]
- `ao_shaping.runners.runner_common.load_init_v`
- `ao_shaping.runners.runner_common.plot_pib_debug_grid`
- `ao_shaping.runners.runner_common.save_best_voltages`
- `ao_shaping.utils.cli.exposure.exposure_ms_legacy`
- `ao_shaping.utils.cli.exposure.resolve_exposure`
- `ao_shaping.utils.io.cli_helpers.resolve_debug`
- [[functions/path-dirs|`gen_date_dir`]]
- [[functions/path-dirs|`gen_file_path_uuid`]]
- [[functions/path-dirs|`get_init_V_by_rms`]]

## 谁调用它

- [[entry/cli|`cli`]]（`cli.add_command` 注册为 `pib`）

## 相似函数（自动）

与本页函数结构/职责相近的函数（相似度得分，来自 `build_similarity.py`；含义与阈值见 [[relations/similar-functions|相似函数总览]]）：

- 0.350 — [[commands/combined|`run`]]（同名 `run`；共享 6 个被调函数（调用结构高度相似））


## 相关页面

- [[entry/cli]]
