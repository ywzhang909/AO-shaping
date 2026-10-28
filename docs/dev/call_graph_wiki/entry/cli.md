---
title: cli（Click 根命令组）
created: 2026-10-13
updated: 2026-10-13
type: entry
tags: [entry,cli]
source: src/ao_shaping/main.py:82
---

## 职责

Click 根命令组 `@click.group()`，负责三件事：

(1) 解析全局选项 `--dir`（数据保存根目录，默认 `data`）与 `--debug`（调试模式旗标）；debug 判定为 `get_debug_mode() or bool(debug)`，即 `--debug` 旗标、环境变量 `DEBUG=1/true` 任一开启即生效（详见 [[functions/debug-mode]]），开启后 loguru 切到 DEBUG 级日志。

(2) 通过 `ctx.ensure_object(dict)` 把 `dir` 与 `debug` 存入 `ctx.obj`，供子命令经 `@click.pass_context` 读取。

(3) 注册 20 个子命令（18 个 runner 符号 + 2 个 tools 命令）。命令注册采用 **懒加载**：`ao_shaping.runners.__init__` 用模块级 `__getattr__` + `_LAZY_RUNNERS` 映射，只有命令真正被调用时才 import 对应 runner 模块（避免 `python -m` 时的 sys.modules 警告，见 R-27）。

## 签名

```python
def cli(ctx: click.Context, dir: str, debug: bool | None)
```

## 注册的 20 个命令

| 命令 | runner 入口函数 | 页面 |
|---|---|---|
| `wf` | `runners.nlight_dm.wf_runner.run` | [[commands/wf]] |
| `pib` | `runners.nlight_dm.axis_beam_runner.run` | [[commands/pib]] |
| `pipeline` | `runners.nlight_dm.pipeline_runner.run` | [[commands/pipeline]] |
| `combined` | `runners.nlight_dm.combined_runner.run` | [[commands/combined]] |
| `dm-matrix` | `runners.matrix_runner.run` | [[commands/dm-matrix]] |
| `hadamard-matrix` | `runners.matrix_runner.hadamard_matrix_run` | [[commands/hadamard-matrix]] |
| `alt-voltage` | `runners.micro_drive.voltage_runner.run` | [[commands/alt-voltage]] |
| `full-voltage` | `runners.micro_drive.voltage_runner.full_voltage_run` | [[commands/full-voltage]] |
| `zernike-matrix` | `runners.slm.zernike_matrix_runner.run` | [[commands/zernike-matrix]] |
| `closed-loop` | `runners.slm.zernike_matrix_runner.closed_loop_run` | [[commands/closed-loop]] |
| `rms-zernike` | `runners.slm.rms_zernike_runner.run` | [[commands/rms-zernike]] |
| `ga-zernike` | `runners.zernike_search_runner.run` | [[commands/ga-zernike]] |
| `greedy-zernike` | `runners.zernike_search_runner.greedy_zernike_run` | [[commands/greedy-zernike]] |
| `slm-pib` | `runners.slm.slm_shaping_runner.run` | [[commands/slm-pib]] |
| `slm-gsnet` | `runners.slm.gsnet_runner.run` | [[commands/slm-gsnet]] |
| `slm-gs-refine` | `runners.slm.gs_refine_runner.run` | [[commands/slm-gs-refine]] |
| `slm-model-in-loop` | `runners.slm.model_in_loop_runner.run` | [[commands/slm-model-in-loop]] |
| `slm-transfer-shapes` | `runners.slm.transfer_shapes_runner.run` | [[commands/slm-transfer-shapes]] |
| `slm-lut` | `tools.slm.calibration.lut_runner.run` | [[commands/slm-lut]] |
| `slm-diagnose` | `tools.slm.panel.diagnose.main` | [[commands/slm-diagnose]] |

## 调用关系

- 被 `main.py` 的 `__main__` 块调用
- 调用 [[functions/debug-mode|`get_debug_mode`]]
- 分发到各 [[commands/wf|命令入口]]（下表）
