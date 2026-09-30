# SLM PIB 整形台架验收报告 (离线分析)

> **Fully offline.** 本报告只读取 `data/debug` 下已保存的 debug 产物
> (recorder pickle + JSON sidecar), 从不打开相机或 SLM。设备离线期间可
> 随时重跑刷新结论。数据根目录: `data\debug`

## 1. 噪声地板与 SNR

共 3 次扫描。噪声地板 $\sigma_J$ 区间 **1.72e-05 … 3.71e-04**（相差 **22×**）。

> **结论 1：噪声地板不是台架常数。** 同一天、同一光斑位置的三次扫描
> 相差 21 倍, 因此 SNR **必须每次实测**, 不可沿用历史值或写死。

| 扫描时间 | 中心 (x, y) | $\sigma_J$ | SNR@0.0005 | SNR@0.001 | SNR@0.01 |
|---|---|---|---|---|---|
| 20260929_085851 | (665, 1031) | 3.71e-04 | 1.35 | 1.91 | 1.38 |
| 20260929_091112 | (664, 1029) | 8.17e-05 | 2.85 | 3.06 | 2.79 |
| 20260929_220954 | (664, 1029) | 1.72e-05 | 3.25 | 4.93 | 4.06 |

> 表中为**单模式** (defocus) 探测。SPGD 同时扰动全部模式, 信号被
> 稀释 —— 见 §2。

## 2. 门控行为 (为什么小 delta 跑不动)

| 运行 | 标签 | rows | applied | noise | fold | delta | n_eval_frames | best objective |
|---|---|---|---|---|---|---|---|---|
| 20260929_090235 | `smoke_f1` | 31 | 26 | 0 | 4 | 0.0005 | 1 | -0.7825 |
| 20260929_090258 | `smoke_f4` | 17 | 15 | 0 | 1 | 0.0005 | 4 | -0.7778 |
| 20260929_090700 | `smoke_f1` | 31 | 4 | 23 | 3 | 0.0005 | 1 | -0.8788 |
| 20260929_090723 | `smoke_f4` | 17 | 4 | 10 | 2 | 0.0005 | 4 | -0.8371 |
| 20260929_091159 | `smoke_f1` | 31 | 6 | 21 | 3 | 0.0005 | 1 | -0.8430 |
| 20260929_091222 | `smoke_f4` | 17 | 4 | 11 | 1 | 0.0005 | 4 | -0.8565 |
| 20260929_221044 | `smoke_f1` | 31 | 5 | 22 | 3 | 0.0005 | 1 | -0.8160 |
| 20260929_221106 | `smoke_f4` | 17 | 0 | 0 | 16 | 0.0005 | 4 | -0.9065 |

> **结论 2：门控是主要损耗项。** 最差的一次 `20260929_221106` 只有 **0%** 的行真正被采纳, 其余被噪声门或亮度折叠门拦下 —— 搜索几乎没有拿到梯度。

两个独立成因, 都在台架上实测确认:

1. **信噪比被自由度稀释。** SPGD 对所有模式施加随机 ±1 扰动, 而单模式探测只动一个模式。实测 `delta=0.0005` 时单模式 SNR 2.25(可用) 而 54 自由度 SNR 仅 1.34 (不可用) —— **这就是 `delta<0.001` 跑不动的原因, 不是约束本身荒谬**。
2. **噪声是慢漂移, 不是散粒噪声。** 40 帧平均并未降低 $\sigma_J$(2.8e-3 vs 10 帧的 1.7e-3), 所以加帧数无效; 必须用 `+ - - +` 回文序列做漂移对消 (见 §5 工具)。

## 3. 搜索结果 (best vs sustained)

**sustained** 是末帧相对首帧的改善, **best** 是历史最优。两者的差值
是判别「真优化」与「漂移」的关键: 若 best 很大而 sustained 接近 0,
说明所谓改善只是噪声漂移。

`guard` 列是被能量门判为「放弃评估」的行数 (J = 1e3 哨兵值), 这些行不参与统计。

| run | objective | mode | epochs | guard | first | best | final | best % | sustained % | 溯源 |
|---|---|---|---|---|---|---|---|---|---|---|
| slm_pib_pib_20260928_175845 | `shape` | max | 190 | 0 | -1.0811 | -0.4461 | -0.4461 | +58.7 | +58.7 | delta=0.0005 |
| slm_pib_shape_20260926_174011 | `shape` | max | 96 | 5 | -2.6785 | -1.4995 | -1.5803 | +44.0 | +41.0 | sidecar 无关键字段 |
| slm_pib_shape_20260923_175829 | `shape` | max | 43 | 2 | -1.1095 | -0.6476 | -0.7049 | +41.6 | +36.5 | sidecar 无关键字段 |
| slm_pib_shape_20260924_091143 | `shape` | max | 101 | 0 | -1.0424 | -0.6320 | -0.6799 | +39.4 | +34.8 | sidecar 无关键字段 |
| slm_pib_pib_20260928_204741 | `shape` | max | 201 | 0 | -1.2244 | -0.7267 | -0.8061 | +40.6 | +34.2 | delta=0.0005 |
| slm_pib_hw_20260929 | `pearson` | min | 61 | 0 | 0.5316 | 0.3699 | 0.4017 | +30.4 | +24.4 | sidecar 无关键字段 |
| slm_pib_pearson_20260929_161710 | `pearson` | min | 61 | 0 | 0.5316 | 0.3699 | 0.4017 | +30.4 | +24.4 | delta=0.1 |
| slm_pib_shape_20260926_172355 | `shape` | max | 101 | 0 | -1.2017 | -0.7154 | -0.9120 | +40.5 | +24.1 | sidecar 无关键字段 |
| slm_pib_pearson_20260929_161856 | `pearson` | min | 61 | 0 | 0.5322 | 0.3364 | 0.4087 | +36.8 | +23.2 | sidecar 无关键字段 |
| slm_pib_shape_20260929_162049 | `shape` | max | 61 | 0 | -1.3686 | -0.8297 | -1.1501 | +39.4 | +16.0 | delta=0.1 |
| slm_pib_pearson_20260929_163533 | `pearson` | min | 61 | 0 | 0.5330 | 0.3783 | 0.4553 | +29.0 | +14.6 | objective='pearson', cam_type='daheng', cam_id=0, exposure_time_ms=1.2, delta=0.1 |
| slm_zernike_shaping_shape_20260926_170950 | `shape` | max | 101 | 0 | -1.2001 | -0.8425 | -1.0702 | +29.8 | +10.8 | sidecar 无关键字段 |
| slm_pib_shape_20260923_190544 | `shape` | max | 101 | 0 | -1.0840 | -0.9225 | -1.0001 | +14.9 | +7.7 | sidecar 无关键字段 |
| slm_pib_pib_20260928_173303 | `shape` | max | 101 | 0 | -0.9369 | -0.7104 | -0.8984 | +24.2 | +4.1 | delta=0.0005 |
| slm_pib_shape_20260924_090210 | `shape` | max | 101 | 0 | -1.0879 | -0.8379 | -1.0550 | +23.0 | +3.0 | sidecar 无关键字段 |
| slm_pib_pearson_20260929_161417 | `pearson` | min | 201 | 0 | 0.5348 | 0.3800 | 0.5188 | +28.9 | +3.0 | delta=0.01 |
| slm_pib_shape_20260926_172917 | `shape` | max | 101 | 0 | -1.2107 | -0.8410 | -1.1780 | +30.5 | +2.7 | sidecar 无关键字段 |
| slm_pib_rms_pib_20260926_172642 | `rms_pib` | max | 101 | 0 | 0.6195 | 0.6349 | 0.6340 | +2.5 | +2.4 | sidecar 无关键字段 |
| slm_pib_pearson_20260929_151039 | `pearson` | min | 41 | 0 | 0.9911 | 0.9585 | 0.9771 | +3.3 | +1.4 | delta=0.5 |
| slm_pib_shape_20260929_160226 | `shape` | max | 201 | 0 | -1.3732 | -0.9264 | -1.3625 | +32.5 | +0.8 | delta=0.0005 |
| slm_pib_shape_20260929_222639 | `shape` | max | 41 | 0 | -1.4408 | -1.0472 | -1.4324 | +27.3 | +0.6 | objective='shape', cam_type='daheng', cam_id=0, exposure_time_ms=1.2, delta=0.0005 |
| slm_pib_shape_20260923_190206 | `shape` | max | 101 | 0 | -1.0832 | -0.7282 | -1.0807 | +32.8 | +0.2 | sidecar 无关键字段 |
| slm_pib_pearson_20260929_160921 | `pearson` | min | 201 | 0 | 0.5348 | 0.4023 | 0.5337 | +24.8 | +0.2 | delta=0.002 |
| slm_pib_rmse_20260926_174434 | `rmse` | max | 101 | 0 | 0.0000 | 0.0000 | 0.0000 | +1.6 | -0.6 | sidecar 无关键字段 |
| slm_pib_shape_20260929_150531 | `shape` | max | 301 | 0 | -1.8799 | -1.7963 | -1.8914 | +4.5 | -0.6 | delta=0.5 |
| slm_pib_rms_pib_20260928_205958 | `rms_pib` | max | 151 | 0 | 0.6069 | 0.6082 | 0.6007 | +0.2 | -1.0 | delta=0.001 |
| slm_zernike_shaping_rms_pib_20260926_170626 | `rms_pib` | max | 101 | 0 | 0.6199 | 0.6314 | 0.6123 | +1.9 | -1.2 | sidecar 无关键字段 |
| slm_pib_shape_20260926_173633 | `shape` | max | 101 | 0 | -1.1935 | -0.7924 | -1.2107 | +33.6 | -1.4 | sidecar 无关键字段 |
| slm_pib_shape_20260929_222529 | `shape` | max | 41 | 0 | -1.3653 | -0.8334 | -1.3865 | +39.0 | -1.6 | objective='shape', cam_type='daheng', cam_id=0, exposure_time_ms=1.2, delta=0.0005 |
| slm_zernike_shaping_rms_pib_20260926_164154 | `rms_pib` | max | 101 | 0 | 0.6190 | 0.6192 | 0.6089 | +0.0 | -1.6 | sidecar 无关键字段 |
| slm_pib_rms_pib_20260923_201537 | `rms_pib` | max | 101 | 0 | 0.6655 | 0.6693 | 0.6537 | +0.6 | -1.8 | sidecar 无关键字段 |
| slm_pib_shape_20260929_150918 | `shape` | max | 301 | 0 | -1.8799 | -1.8639 | -1.9189 | +0.9 | -2.1 | delta=0.5 |
| slm_pib_pib_20260928_205321 | `shape` | max | 132 | 0 | -1.0720 | -0.6487 | -1.0962 | +39.5 | -2.3 | delta=0.0007 |
| slm_pib_rms_pib_20260928_205645 | `rms_pib` | max | 85 | 0 | 0.6086 | 0.6162 | 0.5940 | +1.3 | -2.4 | delta=0.0007 |
| slm_zernike_shaping_rms_pib_20260926_170021 | `rms_pib` | max | 101 | 0 | 0.6212 | 0.6218 | 0.6062 | +0.1 | -2.4 | sidecar 无关键字段 |
| slm_pib_shape_20260923_201859 | `shape` | max | 101 | 0 | -1.0105 | -0.6248 | -1.0406 | +38.2 | -3.0 | sidecar 无关键字段 |
| slm_zernike_shaping_shape_20260926_170322 | `shape` | max | 101 | 0 | -1.1927 | -1.1151 | -1.2301 | +6.5 | -3.1 | sidecar 无关键字段 |
| slm_zernike_shaping_roi_pib_20260926_171311 | `roi_pib` | max | 101 | 0 | 0.4977 | 0.5007 | 0.4772 | +0.6 | -4.1 | sidecar 无关键字段 |
| slm_zernike_shaping_rms_pib_20260926_162917 | `rms_pib` | max | 101 | 0 | 0.6147 | 0.6237 | 0.5853 | +1.5 | -4.8 | sidecar 无关键字段 |
| slm_pib_shape_20260926_172104 | `shape` | max | 20 | 81 | -1.2018 | -1.1913 | -1.2844 | +0.9 | -6.9 | sidecar 无关键字段 |
| slm_pib_roi_pib_20260926_173223 | `roi_pib` | max | 101 | 0 | 0.4998 | 0.5040 | 0.4632 | +0.8 | -7.3 | sidecar 无关键字段 |
| slm_pib_shape_20260926_175334 | `shape` | max | 64 | 669 | -1.1922 | -1.1868 | -1.3805 | +0.4 | -15.8 | sidecar 无关键字段 |
| slm_pib_rmse_20260928_174041 | `rmse` | max | 61 | 229 | 0.0001 | 0.0001 | 0.0001 | +0.1 | -24.3 | sidecar 无关键字段 |
| slm_pib_shape_20260928_174644 | `shape` | max | 63 | 282 | -1.1098 | -1.1083 | -1.4036 | +0.1 | -26.5 | sidecar 无关键字段 |

- **sustained > 5%** 的运行: 13 / 44
- **best 显著但 sustained ≈ 0** (判为漂移, 非优化): 6 / 44

> **结论 3：能量门频繁触发的运行基本等于没优化。** 6 个运行出现过 guard 惩罚行, 其 sustained 改善中位数 **-11.3%**; 而 38 个无 guard 的运行中位数为 **+0.4%**。被门拦下的迭代不更新系数, 因此门一旦频繁触发, 搜索就原地踏步 —— 排查时先看 guard 计数, 再看 delta。

判为漂移的运行 (仅列名前 8 个):

| run | best % | sustained % |
|---|---|---|
| slm_pib_shape_20260929_222529 | +39.0 | -1.6 |
| slm_pib_shape_20260926_173633 | +33.6 | -1.4 |
| slm_pib_shape_20260923_190206 | +32.8 | +0.2 |
| slm_pib_shape_20260929_160226 | +32.5 | +0.8 |
| slm_pib_shape_20260929_222639 | +27.3 | +0.6 |
| slm_pib_pearson_20260929_160921 | +24.8 | +0.2 |

## 4. 图

![snr_floor_and_snr](figures/snr_floor_and_snr.png)

![smoke_gate_breakdown](figures/smoke_gate_breakdown.png)

![search_best_vs_sustained](figures/search_best_vs_sustained.png)

## 5. 参数测量工具 (设备无关)

本报告引用的 SNR / 噪声地板测量已抽到 **`ao_shaping.tools.slm.slm_snr_probe`**, 设备实例由参数传入, 不构造任何设备:

```python
from ao_shaping.tools.slm.slm_snr_probe import snr_sweep
with create_camera('daheng', cam_id=0, exposure_time_ms=1.2) as cam, \
     Santec(slm_number=1, wavelength=1064) as slm:
    result = snr_sweep(cam, slm, n_max=9, radius=480.0)
    print(result.sigma, result.multi_snrs, result.usable_deltas())
```

| 符号 | 作用 |
|---|---|
| `snr_sweep` | 噪底 + 单模式/多模式逐 delta SNR (设备实例传入) |
| `measure_noise_floor` | 固定相位下 σ (含慢漂移) |
| `abba_signal` | `+ - - +` 回文对消漂移的 ΔJ |
| `snr_verdict` | 阈值 → strong / usable / unusable |
| `SnrSweepResult.usable_deltas` | **按多模式 SNR** 给出的可用 delta |

消费者: `scripts/measure_shape_sensitivity.py` 与硬件门控测试 `tests/ao_shaping/optimizer/wfless/test_slm_zernike_pib_online.py` 均改为委托该工具, 二者不会再各自漂移。离线可测: `tests/ao_shaping/tools/test_slm_snr_probe.py` (纯 fake 设备, 无需硬件)。

> ⚠️ `usable_deltas()` 刻意按**多模式** SNR 过滤: 单模式列会高估 SPGD 实际能分辨的信号, 直接采信会再次把 delta 选到不可用的区间。

## 6. 建议

1. **排查顺序: 先看 guard 计数, 再看 delta。** 6 个运行出现能量门惩罚行, 其 sustained 改善中位数为负 (§3 结论 3)。门拦下的迭代不更新系数, 门频繁触发时搜索原地踏步, 调 delta 无济于事。
2. **按每次实测的多模式 SNR 选 `delta`**, 不要沿用历史值 —— 噪声地板单日波动 21×。用 `snr_sweep(...).usable_deltas()`。
3. **`delta<0.001` 在 `n_max=9` (54 自由度) 下不可用**: 实测多模式 SNR ≈ 1.3, 95% 迭代被门控。若必须满足该约束, 只能显著降低 `n_max`(自由度越少, 稀释越轻)。
4. **`delta≈0.1` 是当前 `n_max=9` 的可用区间**; `0.2` 会触发亮度折叠门(45/60 被拒), 过大反而更差。
5. **加帧数不能降噪**, 只能靠 `+ - - +` 漂移对消; 任何降噪方案先验证是否真的降低了 σ。`n_eval_frames=4` 的 smoke 出现 16/16 全被折断门拒绝, 属于独立失效模式。
6. 目标函数选择请参考 [三目标对比报告](slm_pib_online_hw/objective_comparison.md): `1 - Pearson` 在实机上只优化成功、排序不可靠 (2/4 符号正确), 维持 **RISKY**, 仅作显式可选目标并保留能量门。

---

### 下一步 (设备上线后)

1. 用新工具重跑一次完整 SNR 扫描: `AO_RUN_HARDWARE=1 pytest tests/ao_shaping/optimizer/wfless/test_slm_zernike_pib_online.py -v -s`, 读取 `multi_snrs` 列选 delta。
2. 用选定 delta 跑 `n_max=9` 的方形/PIB 各一次 ≥200 epoch, 确认 sustained > 5%。
3. 硬件验收脚本: `scripts/measure_shape_sensitivity.py` (已改为委托同一工具, 与门控测试结果一致)。

