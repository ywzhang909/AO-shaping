---
title: pipeline（串行流水线：先 WF 后 PIB）
created: 2026-10-13
updated: 2026-10-13
type: command
tags: [command]
source: src/ao_shaping/runners/nlight_dm/pipeline_runner.py:35
---

## 一句话

串行流水线：先 WF 后 PIB

## 入口函数

`runners.nlight_dm.pipeline_runner.run`（`src/ao_shaping/runners/nlight_dm/pipeline_runner.py:35`）

```python
run(ctx, device, run, wfs_params)
```

## docstring（原文节选）

```
串行优化器（先波前优化，再轴向光束优化）

    DEBUG环境变量控制调试模式。
```

## 调用链分析（自动）

**硬件**：CCD 相机、DM 变形镜、WFS 波前传感器

**调用的优化器/算法**：

- `_camera_cls`
- `_extract_optimizer_momentum`
- `_generate_search_candidates`
- `_objective_to_min`
- `_reset_optimizer_state`
- `_should_trigger_adaptive_search`
- `ao_shaping.optimizer.constants.create_optimizer`
- [[functions/spgd-gradient|`spgd_gradient`]]
- [[functions/optimizer-rms-dm|`optimizer_rms_dm`]]
- [[functions/optimize-pib-wfless|`optimize_pib`]]
- `learning_schedule`
- `schedule_lr_delta`

**结果/调试落盘**：`gen_date_dir`、`make_date_dir`

## 直接调用（1 级，按调用次数）

- [[functions/with-params|`with_params`]]
- [[functions/resolve-dm|`resolve_dm`]]
- [[functions/optimizer-rms-dm|`optimizer_rms_dm`]]
- [[functions/optimize-pib-wfless|`optimize_pib`]]
- [[functions/path-dirs|`gen_date_dir`]]
- [[functions/path-dirs|`gen_file_path_uuid`]]
- [[functions/path-dirs|`get_date_dir_name`]]
- [[functions/debug-mode|`get_debug_mode`]]

## 谁调用它

- [[entry/cli|`cli`]]（`cli.add_command` 注册为 `pipeline`）

## 相关页面

- [[entry/cli]]
