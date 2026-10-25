---
title: zernike-matrix（SLM Zernike 响应矩阵标定 + 离线矫正自检）
created: 2026-10-13
updated: 2026-10-13
type: command
tags: [command]
source: src/ao_shaping/runners/slm/zernike_matrix_runner.py:1057
---

## 一句话

SLM Zernike 响应矩阵标定 + 离线矫正自检

## 入口函数

`runners.slm.zernike_matrix_runner.run`（`src/ao_shaping/runners/slm/zernike_matrix_runner.py:1057`）

```python
run(ctx, params, slm_params, wfs_driver, wfs_params)
```

## docstring（原文节选）

```
获取Zernike响应矩阵

    支持 N 次正负交替循环测量 + M 次 WFS 读取取平均 + 方差跟踪 + 逆矩阵计算。

    多幅度模式 (--n-magnitudes N):
        自动生成N个不同扰动幅度 (2~5 rad)，分别标定并保存。

    调试模式 (--debug):
        保存每次测量的原始数据:
        - SLM相位图 (灰度值, 已应用shift)
        - WFS deviation数据
        - WFS Zernike系数 (averaging前)

    标定内核 (2026-09-16 按已验证的 tools/slm/slm_zernike_response.py 重写):
        - 直控链路: Santec + PatternHelper (与原 ZernikeSLM 链路不同; 旧链路实验原生崩溃)
        - 矩阵单位 **λ/λ**: 列 = Δ(WFS λ) / Δ(SLM 幅度 λ); 幅度 λ = rad/(2π)
        - make_phase **透传 n_max** (不再硬编码 4 → n≥5 模式不再被截断)
        - 测完矩阵后自动执行 offline_correction_test():
            恢复内部参考 → 平场基线 → pinv 反解当前像差 (λ) → ×2π 弧度加载
            → 复测残差 + 模型自检 (‖w0+M·c‖ vs ‖w1‖, 比 0.7~1.4)
            → export_correction_csv 保存施加的矫正相位
              (文件名含 序列号/波长/shift/半径/时间 + sidecar JSON)

    TODO (实验待测试清单 —— 重写后尚未上设备实测):
        [ ] 重标响应矩阵, 检查主导系数: [5](2,0)defocus [9](3,1)coma [13](4,0)spherical
            应与理论形状对应; n≥5 模式不再被截断 (旧 bug: 硬编码 n_max=4)
        [ ] 矩阵量纲: cond_eff 应仍 ≈8 量级; 反解系数幅值与推拉幅度 (λ) 同量级
        [ ] 离线矫正测试: RMS 应有明显改善 (>20%); 模型自检比 ∈ 0.7~1.4
            (偏离即 单位 λ/rad 或 zernike_radius 与标定不一致)
        [ ] export_correction_csv 产物: 数据区为**弧度**; create_phase_from_array 重载
            应还原矫正相位 (勿走 load_gray_from_csv/csv_to_phase 灰度读取管线)
        [ ] closed-loop 用新矩阵: 收敛行为与离线反解一致, 控制系数无 6.28× 缩放
        [ ] --n-magnitudes N: 各幅度结果应线性 (系数 / 幅度 ≈ 常数)
```

## 调用链分析（自动）

**硬件**：WFS 波前传感器

**调用的优化器/算法**：

- [[functions/save-zernike-matrix|`save_zernike_response_matrix`]]

**结果/调试落盘**：`_save_init_state_hdf5`、`_save_wavefront_log_hdf5`、`save_zernike_response_matrix`

## 直接调用（1 级，按调用次数）

- [[functions/with-params|`with_params`]]
- `ao_shaping.tools.zernike.common.effective_cond`
- `_capture_init_state`
- `_compute_calibration_magnitudes`
- `_mode_ids`
- `_normalize_run_options`
- `_print_calibration_summary`
- `_save_init_state_hdf5`
- `_save_wavefront_log_hdf5`
- `_setup_debug_callback`
- `ao_shaping.display.image_display.ZernikeCalibrationDisplay` *(未解析/外部)*
- `ao_shaping.drivers.slm.Santec` *(未解析/外部)*
- [[functions/resolve-wfs|`resolve_wfs`]]
- `ao_shaping.optimizer.wf.zernike_response_matrix.ZernikeResponseMatrixResult` *(未解析/外部)*
- [[functions/save-zernike-matrix|`save_zernike_response_matrix`]]
- `ao_shaping.tools.zernike.common.collect_device_info`
- `ao_shaping.tools.zernike.common.safe_pinv`
- [[functions/path-dirs|`get_timestamp_str`]]
- `ao_shaping.utils.wavefront.pattern_helper.PatternHelper` *(未解析/外部)*
- `calibrate_response_matrix_pushpull`
- `offline_correction_test`

## 谁调用它

- [[entry/cli|`cli`]]（`cli.add_command` 注册为 `zernike-matrix`）

## 相关页面

- [[entry/cli]]
- [[functions/save-zernike-matrix|`save_zernike_response_matrix`]]
