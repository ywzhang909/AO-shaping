---
title: spgd-square（SLM 方形远场均匀性整形（SPGD））
created: 2026-10-13
updated: 2026-10-13
type: command
tags: [command]
source: src/ao_shaping/runners/slm/slm_shaping_runner.py:804
---

## 一句话

SLM 方形远场均匀性整形（SPGD）

## 入口函数

`runners.slm.slm_shaping_runner.square`（`src/ao_shaping/runners/slm/slm_shaping_runner.py:804`）

```python
square(ctx, params, slm_params)
```

## docstring（原文节选）

```
SLM方形光斑整形优化器 (uniform square far field, camera feedback).

    使用SPGD算法优化SLM上的Zernike系数，通过相机反馈产生均匀方形远场光斑。

    优化目标: 最小化目标方形区域内光强的变异系数(CV = std/mean)，同时可选
    加权环围能量(EE)和宽高比(AR)指标。

    目标方形参数二选一: 边长(px) 或 平均亮度 (--target-side 与
    --target-mean-brightness 互斥)。
```

## 调用链分析（自动）

**硬件**：CCD 相机

**调用的优化器/算法**：

- `_freeform_phase_radians`
- `_locate_square_center`
- `_map_init_to_active_modes`
- `_slm_cls`
- [[functions/run-heuristic-search|`run_heuristic_search`]]
- `ao_shaping.optimizer.constants.create_optimizer`
- `ao_shaping.optimizer.wfless.adaptive_spgd_core.get_stage_weights`
- `ao_shaping.optimizer.wfless.adaptive_spgd_core.should_boost_stagnation`
- [[functions/optimize-slm-square|`optimize_slm_square`]]
- `create_heuristic`
- `learning_schedule`
- `square_aspect_ratio`
- `square_encircled_energy`
- `square_objective_score`
- `square_peak_to_background_ratio`
- `square_uniformity_cost`

**结果/调试落盘**：`_save_square_best_image`、`_save_square_summary`、`save_recorder_debug_artifacts`、`save_optimization_debug_artifacts`

## 直接调用（1 级，按调用次数）

- [[functions/with-params|`with_params`]]
- `_active_zernike_modes`
- `_build_square_config`
- `_echo_square_banner`
- `_parse_zernike_mask`
- `_resolve_run_context`
- `_save_square_best_image`
- `_save_square_summary`
- `_square_json_payload`
- [[functions/optimize-slm-square|`optimize_slm_square`]]
- [[functions/runner-common|`parse_center`]]
- [[functions/runner-common|`patch_sim_square_shaping`]]
- [[functions/debug-artifacts|`save_recorder_debug_artifacts`]]

## 谁调用它

- [[entry/cli|`cli`]]（`cli.add_command` 注册为 `spgd-square`）

## 相关页面

- [[entry/cli]]
