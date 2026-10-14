---
title: dm-matrix（DM 响应矩阵标定（push-pull））
created: 2026-10-13
updated: 2026-10-13
type: command
tags: [command]
source: src/ao_shaping/runners/matrix_runner.py:123
---

## 一句话

DM 响应矩阵标定（push-pull）

## 入口函数

`runners.matrix_runner.run`（`src/ao_shaping/runners/matrix_runner.py:123`）

```python
run(ctx, params, wfs, wfs_drv)
```

## docstring（原文节选）

```
获取DM变形镜响应矩阵

    依次对每个DM单元施加正负电压扰动, 记录WFS子孔径斜率偏差,
    构建 DM -> WFS 的响应矩阵。

    支持 N 次正负交替循环测量 + M 次 WFS 读取取平均 +
    方差跟踪 + 逆矩阵计算 + 自动电压优化。

    调试模式 (--debug):
        保存每次测量的原始WFS deviation数据。

    TODO (实机待测试清单 —— sequential/hadamard 双模式尚未上设备实测):
        [ ] sequential 模式真实 DM+WFS n=5 重跑 (n-averages 10) + 自动电压优化:
            矩阵形状/条件数与仿真一致, 平均方差正常 (无死执行器误报)
        [ ] hadamard 模式自动阶数实测: 与 sequential 矩阵一致性 (相关性/条件数
            同量级), 测量时间显著减少 (所有有效单元同时扰动)
        [ ] hadamard 显式阶数: --hadamard-order 指定 ≥有效单元数的最小 2 的幂
            时正常; 非合法值应报清晰错误
        [ ] 矫正闭环验证: 以标定矩阵 (pinv_matrix) 做波前矫正, RMS 明显改善
        [ ] report3: scripts/dm/generate_dm_response_matrix_report.py 以真实 h5
            生成报告 (图/表齐全), 结果写入 docs/slm 日报
```

## 调用链分析（自动）

**硬件**：DM 变形镜

**调用的优化器/算法**：

- `_build_device_config`
- `_decode_hadamard_responses`
- `_hadamard_matrix`
- `_next_hadamard_order`
- `_optimize_perturbation_voltage`
- [[functions/calibrate-dm-matrix|`calibrate_dm_response_matrix`]]
- [[functions/save-dm-matrix|`save_dm_response_matrix`]]
- `measure_actuator_response`

**结果/调试落盘**：`save_dm_response_matrix`、`save_debug_data`

## 直接调用（1 级，按调用次数）

- [[functions/with-params|`with_params`]]
- `_open_wfs`
- `_refresh_wfs_reference`
- [[functions/resolve-dm|`resolve_dm`]]
- [[functions/calibrate-dm-matrix|`calibrate_dm_response_matrix`]]
- [[functions/save-dm-matrix|`save_dm_response_matrix`]]
- [[functions/debug-mode|`get_debug_mode`]]
- [[functions/path-dirs|`get_timestamp_str`]]
- `ao_shaping.utils.wavefront.wfs_utils.make_actuator_debug_callback`

## 谁调用它

- [[entry/cli|`cli`]]（`cli.add_command` 注册为 `dm-matrix`）

## 相关页面

- [[entry/cli]]
