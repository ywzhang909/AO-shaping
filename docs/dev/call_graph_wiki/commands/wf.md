---
title: wf（波前 RMS 优化（DM 电压 + WFS））
created: 2026-10-13
updated: 2026-10-13
type: command
tags: [command]
source: src/ao_shaping/runners/nlight_dm/wf_runner.py:34
---

## 一句话

波前 RMS 优化（DM 电压 + WFS）

## 入口函数

`runners.nlight_dm.wf_runner.run`（`src/ao_shaping/runners/nlight_dm/wf_runner.py:34`）

```python
run(ctx, dm, run, algorithm, wfs)
```

## docstring（原文节选）

```
波前优化器

    DEBUG环境变量控制调试模式。
```

## 调用链分析（自动）

**硬件**：DM 变形镜、WFS 波前传感器

**调用的优化器/算法**：

- [[functions/spgd-gradient|`spgd_gradient`]]
- [[functions/optimizer-rms-dm|`optimizer_rms_dm`]]
- `schedule_lr_delta`

**结果/调试落盘**：`gen_date_dir`、`make_date_dir`

## 直接调用（1 级，按调用次数）

- [[functions/with-params|`with_params`]]
- [[functions/resolve-dm|`resolve_dm`]]
- [[functions/optimizer-rms-dm|`optimizer_rms_dm`]]
- [[functions/path-dirs|`gen_date_dir`]]
- [[functions/path-dirs|`gen_file_path_uuid`]]
- [[functions/path-dirs|`get_date_dir_name`]]
- [[functions/debug-mode|`get_debug_mode`]]

## 谁调用它

- [[entry/cli|`cli`]]（`cli.add_command` 注册为 `wf`）

## 相关页面

- [[entry/cli]]
- [[functions/spgd-gradient|`spgd_gradient`]]
- [[functions/optimizer-rms-dm|`optimizer_rms_dm`]]
