---
title: combined（AdaMOD 综合 PIB 优化（自适应桶收缩））
created: 2026-10-13
updated: 2026-10-13
type: command
tags: [command]
source: src/ao_shaping/runners/nlight_dm/combined_runner.py:29
---

## 一句话

AdaMOD 综合 PIB 优化（自适应桶收缩）

## 入口函数

`runners.nlight_dm.combined_runner.run`（`src/ao_shaping/runners/nlight_dm/combined_runner.py:29`）

```python
run(ctx, algorithm, run, camera)
```

## docstring（原文节选）

```
AdaMOD 综合PIB优化器

    使用AdaMOD优化器进行桶内功率(PIB)优化，支持自适应桶半径收缩。

    调试模式: ``main.py --debug combined`` / 本命令 ``--debug`` / 环境变量 ``DEBUG=1``
    任一开启即可输出 pkl/json 与汇总图片。
```

## 调用链分析（自动）

**硬件**：CCD 相机、DM 变形镜

**调用的优化器/算法**：

- `_camera_cls`
- [[functions/optimize-pib-combined|`optimize_pib`]]
- [[functions/spgd-gradient|`spgd_gradient`]]
- `learning_schedule`

**结果/调试落盘**：`gen_date_dir`、`make_date_dir`

## 直接调用（1 级，按调用次数）

- [[functions/with-params|`with_params`]]
- [[functions/resolve-dm|`resolve_dm`]]
- [[functions/optimize-pib-combined|`optimize_pib`]]
- [[functions/path-dirs|`get_date_dir_name`]]
- `ao_shaping.utils.io.cli_helpers.resolve_debug`
- [[functions/path-dirs|`gen_date_dir`]]
- [[functions/path-dirs|`gen_file_path_uuid`]]

## 谁调用它

- [[entry/cli|`cli`]]（`cli.add_command` 注册为 `combined`）

## 相关页面

- [[entry/cli]]
- [[functions/optimize-pib-combined|`optimize_pib`]]
- [[functions/spgd-gradient|`spgd_gradient`]]
