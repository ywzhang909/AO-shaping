# report → data 引用对照清单

<!-- provenance:start -->
> **生成脚本**: [`scripts/_common/data_reference_inventory.py`](../scripts/_common/data_reference_inventory.py)
> **复现命令**: `python scripts/_common/data_reference_inventory.py`
> **运行环境**: 离线
> **说明**: 静态扫描 report/**/*.{md,json} 对 data/ 的引用并判定本机存在性；只代表报告引用维度，不代表运行时依赖
<!-- provenance:end -->

> 生成日期: 2026-11-01；工具: `scripts/_common/data_reference_inventory.py`。
> 仅静态扫描 `report/**/*.{md,json}` 中的 `data/...` 字符串，存在性按本机文件系统判定；
> 不代表运行时依赖（configs/标定文件等仍可能被主程序读取）。

## 汇总

- 含 `data/` 引用的报告文件: 214 / 扫描总数见上次运行；唯一被引路径 968 条，本机不存在 305 条。

## 逐报告引用明细

### report/data_reference_inventory.md

- ✅ 存在 `data`
- ✅ 存在 `data/1300-5-enriched.csv`
- ❌ 缺失 `data/calib.npz`
- ✅ 存在 `data/calibration`
- ✅ 存在 `data/calibration/WFS_M01219666_MLA150M-5C_3_20260915_232849.ref`
- ✅ 存在 `data/calibration/defocus_radius_diagnostic.json`
- ✅ 存在 `data/calibration/defocus_shift_calib_v3.json`
- ✅ 存在 `data/checkpoint`
- ✅ 存在 `data/debug`
- ❌ 缺失 `data/debug/<脚本名`
- ✅ 存在 `data/debug/_quarantine_20261022_mixed`
- ✅ 存在 `data/debug/_quarantine_20261022_mixed/hw_zernike_sweep_20261022/zernike_order_nmax3/summary.json`
- ❌ 缺失 `data/debug/alignment_probe/20261019_focus_coma`
- ❌ 缺失 `data/debug/alignment_probe/20261019_focus_coma_verify`
- ❌ 缺失 `data/debug/alignment_probe/20261019_realign`
- ✅ 存在 `data/debug/analyze_transfer_direction/anchored_20261023_2002`
- ✅ 存在 `data/debug/analyze_transfer_response/ladder_20261023_1952`
- ❌ 缺失 `data/debug/beam_extent`
- ✅ 存在 `data/debug/bench_settings`
- ❌ 缺失 `data/debug/bench_settings/20261017_ladder_diag.log`
- ✅ 存在 `data/debug/bench_settings/exposure_probe/20261024_110551.json`
- ❌ 缺失 `data/debug/bench_settings/focus_probe`
- ❌ 缺失 `data/debug/bench_settings/focus_probe/slm1_20261020_093932.json`
- ❌ 缺失 `data/debug/bench_stability`
- ❌ 缺失 `data/debug/bench_stability_20261001_164458/bench_stability_`
- ✅ 存在 `data/debug/calibrate_transfer_blur/shapes_20261023_1956`
- ✅ 存在 `data/debug/calibrate_transfer_blur/shapes_20261023_1956/frozen_parameters.json`
- ✅ 存在 `data/debug/capture_transfer_response/ladder_20261023_1952`
- ✅ 存在 `data/debug/defocus_differentiable/20261023_101822`
- ✅ 存在 `data/debug/defocus_differentiable/20261023_101822/manifest_checksums.json`
- ❌ 缺失 `data/debug/defocus_differentiable/<本地时间戳`
- ✅ 存在 `data/debug/differentiable_opt`
- ✅ 存在 `data/debug/differentiable_opt/20261022_213000`
- ✅ 存在 `data/debug/differentiable_opt/20261022_213000/animation`
- ✅ 存在 `data/debug/differentiable_opt/20261022_213000/animation/frames.csv`
- ✅ 存在 `data/debug/differentiable_opt/gs_20261023_092501`
- ❌ 缺失 `data/debug/differentiable_opt/rep_0_20261023_095038`
- ❌ 缺失 `data/debug/differentiable_opt/rep_1_20261023_095038`
- ❌ 缺失 `data/debug/differentiable_opt/rep_2_20261023_095038`
- ✅ 存在 `data/debug/fit_transfer_multishape/current_20261023_1933`
- ✅ 存在 `data/debug/fit_transfer_multishape/current_20261023_1933/frozen_parameters.json`
- ✅ 存在 `data/debug/fit_transfer_multishape/exact_grid_20261023_1935`
- ✅ 存在 `data/debug/fit_transfer_multishape/exact_grid_20261023_1935/iterations.pkl`
- ✅ 存在 `data/debug/fit_transfer_multishape/exact_grid_20261023_1935/result.json`
- ✅ 存在 `data/debug/fit_transfer_multishape/history_20261023_1925`
- ✅ 存在 `data/debug/fit_transfer_multishape/history_20261023_1935_r2`
- ❌ 缺失 `data/debug/focus_gate_check/<UTC戳`
- ❌ 缺失 `data/debug/gs_phase_uniformity/20261015_gradient_probe_1`
- ❌ 缺失 `data/debug/gsnet_hybrid/20261013_092934`
- ❌ 缺失 `data/debug/gsnet_hybrid/20261013_093042`
- ❌ 缺失 `data/debug/gsnet_hybrid/20261013_093916`
- ❌ 缺失 `data/debug/gsnet_hybrid/20261013_100342`
- ❌ 缺失 `data/debug/gsnet_hybrid_camera/20261013_101118_581856`
- ❌ 缺失 `data/debug/gsnet_hybrid_camera/20261013_101248_554998`
- ❌ 缺失 `data/debug/gsnet_hybrid_infer/20261013_094228_032239`
- ❌ 缺失 `data/debug/gsnet_hybrid_infer/20261013_094258_729530`
- ❌ 缺失 `data/debug/gsnet_hybrid_infer/20261013_100454_477105`
- ✅ 存在 `data/debug/hw_groups_20261020`
- ❌ 缺失 `data/debug/hw_groups_20261020/dz_noise_hw_gs`
- ✅ 存在 `data/debug/hw_groups_20261020/dz_scan_hw_gs`
- ✅ 存在 `data/debug/hw_groups_20261020/dz_scan_model_gs`
- ✅ 存在 `data/debug/hw_groups_20261020/groups_manifest.json`
- ❌ 缺失 `data/debug/hw_groups_20261020/n`
- ✅ 存在 `data/debug/hw_groups_20261020/plan_all.json`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax11.npy`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax15.npy`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax20.npy`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax25.npy`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax3.npy`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax4.npy`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax6.npy`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax8.npy`
- ❌ 缺失 `data/debug/hw_groups_20261020/zernike_farfield_nmax11`
- ❌ 缺失 `data/debug/hw_groups_20261020/zernike_farfield_nmax11/n变量组`
- ❌ 缺失 `data/debug/hw_groups_20261020/zernike_farfield_nmax15`
- ❌ 缺失 `data/debug/hw_groups_20261020/zernike_farfield_nmax15/n变量组`
- ❌ 缺失 `data/debug/hw_groups_20261020/zernike_farfield_nmax20`
- ❌ 缺失 `data/debug/hw_groups_20261020/zernike_farfield_nmax20/n变量组`
- ❌ 缺失 `data/debug/hw_groups_20261020/zernike_farfield_nmax25`
- ❌ 缺失 `data/debug/hw_groups_20261020/zernike_farfield_nmax25/n变量组`
- ❌ 缺失 `data/debug/hw_groups_20261020/zernike_farfield_nmax3`
- ❌ 缺失 `data/debug/hw_groups_20261020/zernike_farfield_nmax4`
- ❌ 缺失 `data/debug/hw_groups_20261020/zernike_farfield_nmax6`
- ❌ 缺失 `data/debug/hw_groups_20261020/zernike_farfield_nmax6/n变量组`
- ❌ 缺失 `data/debug/hw_groups_20261020/zernike_farfield_nmax8`
- ❌ 缺失 `data/debug/hw_groups_20261020/zernike_farfield_nmax8/n变量组`
- ✅ 存在 `data/debug/hw_groups_20261021`
- ❌ 缺失 `data/debug/hw_groups_20261021/dz_noise_hw_gs`
- ❌ 缺失 `data/debug/hw_groups_20261021/dz_noise_hw_gs/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261021/dz_noise_hw_gs/n变量组`
- ✅ 存在 `data/debug/hw_groups_20261021/dz_scan_hw_gs`
- ❌ 缺失 `data/debug/hw_groups_20261021/dz_scan_hw_gs/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261021/dz_scan_hw_gs/n变量组`
- ✅ 存在 `data/debug/hw_groups_20261021/dz_scan_model_gs`
- ❌ 缺失 `data/debug/hw_groups_20261021/dz_scan_model_gs/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261021/dz_scan_model_gs/n变量组`
- ✅ 存在 `data/debug/hw_groups_20261021/focus_calibration`
- ✅ 存在 `data/debug/hw_groups_20261021/focus_calibration/slm1_20261021_154749.json`
- ✅ 存在 `data/debug/hw_groups_20261021/groups_manifest.json`
- ❌ 缺失 `data/debug/hw_groups_20261021/manifest.json/n`
- ❌ 缺失 `data/debug/hw_groups_20261021/n`
- ❌ 缺失 `data/debug/hw_groups_20261021/n曝光`
- ✅ 存在 `data/debug/hw_groups_20261021/plan.json`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield/phases/phase_nmax11.npy`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield/phases/phase_nmax15.npy`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield/phases/phase_nmax20.npy`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield/phases/phase_nmax25.npy`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield/phases/phase_nmax3.npy`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield/phases/phase_nmax4.npy`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield/phases/phase_nmax6.npy`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield/phases/phase_nmax8.npy`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax11`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax11/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax11/n变量组`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax15`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax15/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax15/n变量组`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax20`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax20/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax20/n变量组`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax25`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax25/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax25/n变量组`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax3`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax3/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax3/n变量组`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax4`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax4/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax4/n变量组`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax6`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax6/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax6/n变量组`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax8`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax8/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax8/n变量组`
- ✅ 存在 `data/debug/hw_groups_20261022`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_noise_hw_gs`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_noise_hw_gs/frame_000.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_noise_hw_gs/frame_001.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_noise_hw_gs/frame_002.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_noise_hw_gs/frame_003.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_noise_hw_gs/frame_004.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_noise_hw_gs/frame_005.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_noise_hw_gs/frame_006.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_noise_hw_gs/frame_007.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_noise_hw_gs/frame_008.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_noise_hw_gs/frame_009.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_noise_hw_gs/hw_summary.png`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_noise_hw_gs/measurements.csv`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_noise_hw_gs/measurements.json`
- ❌ 缺失 `data/debug/hw_groups_20261022/dz_noise_hw_gs/n分组清单`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_noise_hw_gs/run_config.json`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_000.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_001.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_002.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_003.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_004.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_005.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_006.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_007.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_008.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_009.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_010.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_011.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_012.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_013.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_014.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_015.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_016.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_017.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_018.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_019.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_020.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_021.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_022.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_023.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/hw_summary.png`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/measurements.csv`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/measurements.json`
- ❌ 缺失 `data/debug/hw_groups_20261022/dz_scan_hw_gs/n分组清单`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/run_config.json`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_000.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_001.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_002.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_003.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_004.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_005.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_006.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_007.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_008.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_009.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_010.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_011.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_012.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_013.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_014.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_015.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_016.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_017.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_018.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_019.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_020.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_021.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_022.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_023.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_024.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_025.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_026.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_027.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_028.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_029.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/hw_summary.png`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/measurements.csv`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/measurements.json`
- ❌ 缺失 `data/debug/hw_groups_20261022/dz_scan_model_gs/n分组清单`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/run_config.json`
- ✅ 存在 `data/debug/hw_groups_20261022/focus_calibration`
- ✅ 存在 `data/debug/hw_groups_20261022/focus_calibration/slm1_20261022_105213.json`
- ✅ 存在 `data/debug/hw_groups_20261022/focus_calibration/slm1_20261022_115811.json`
- ✅ 存在 `data/debug/hw_groups_20261022/focus_calibration/slm1_20261022_142207.json`
- ✅ 存在 `data/debug/hw_groups_20261022/groups_manifest.json`
- ✅ 存在 `data/debug/hw_groups_20261022/manifest.json`
- ❌ 缺失 `data/debug/hw_groups_20261022/manifest.json/n`
- ❌ 缺失 `data/debug/hw_groups_20261022/n曝光`
- ✅ 存在 `data/debug/hw_groups_20261022/plan.json`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax11`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax11/frame_000.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax11/frame_001.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax11/hw_summary.png`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax11/measurements.csv`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax11/measurements.json`
- ❌ 缺失 `data/debug/hw_groups_20261022/zernike_farfield_nmax11/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261022/zernike_farfield_nmax11/n变量组`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax11/run_config.json`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax15`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax15/frame_000.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax15/frame_001.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax15/hw_summary.png`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax15/measurements.csv`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax15/measurements.json`
- ❌ 缺失 `data/debug/hw_groups_20261022/zernike_farfield_nmax15/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261022/zernike_farfield_nmax15/n变量组`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax15/run_config.json`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax20`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax20/frame_000.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax20/frame_001.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax20/hw_summary.png`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax20/measurements.csv`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax20/measurements.json`
- ❌ 缺失 `data/debug/hw_groups_20261022/zernike_farfield_nmax20/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261022/zernike_farfield_nmax20/n变量组`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax20/run_config.json`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax25`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax25/frame_000.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax25/frame_001.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax25/hw_summary.png`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax25/measurements.csv`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax25/measurements.json`
- ❌ 缺失 `data/debug/hw_groups_20261022/zernike_farfield_nmax25/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261022/zernike_farfield_nmax25/n变量组`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax25/run_config.json`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax3`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax3/frame_000.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax3/frame_001.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax3/hw_summary.png`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax3/measurements.csv`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax3/measurements.json`
- ❌ 缺失 `data/debug/hw_groups_20261022/zernike_farfield_nmax3/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261022/zernike_farfield_nmax3/n变量组`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax3/run_config.json`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax4`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax4/frame_000.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax4/frame_001.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax4/hw_summary.png`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax4/measurements.csv`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax4/measurements.json`
- ❌ 缺失 `data/debug/hw_groups_20261022/zernike_farfield_nmax4/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261022/zernike_farfield_nmax4/n变量组`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax4/run_config.json`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax6`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax6/frame_000.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax6/frame_001.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax6/hw_summary.png`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax6/measurements.csv`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax6/measurements.json`
- ❌ 缺失 `data/debug/hw_groups_20261022/zernike_farfield_nmax6/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261022/zernike_farfield_nmax6/n变量组`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax6/run_config.json`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax8`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax8/frame_000.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax8/frame_001.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax8/hw_summary.png`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax8/measurements.csv`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax8/measurements.json`
- ❌ 缺失 `data/debug/hw_groups_20261022/zernike_farfield_nmax8/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261022/zernike_farfield_nmax8/n变量组`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax8/run_config.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/blaze/blaze_128/frame_000.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/blaze/blaze_128/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/blaze/blaze_16/frame_000.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/blaze/blaze_16/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/blaze/blaze_24/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/blaze/blaze_32/frame_000.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/blaze/blaze_32/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/blaze/blaze_40/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/blaze/blaze_48/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/blaze/blaze_64/frame_000.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/blaze/blaze_64/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/blaze/blaze_96/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/blaze/measurements.csv`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/blaze/measurements.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/blaze/run_config.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/manifest.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/measurements.csv`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/measurements.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/run_config.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/selected.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_100/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_1000/frame_000.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_1000/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_1100/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_1200/frame_000.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_1200/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_1200/pupil_phase.npy`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_1300/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_1400/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_1500/frame_000.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_1500/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_1600/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_1700/frame_000.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_1700/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_200/frame_000.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_200/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_300/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_400/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_500/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_600/frame_000.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_600/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_700/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_800/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_900/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/measurements.csv`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/measurements.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_10/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_11/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_12/frame_000.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_12/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_13/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_14/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_15/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_16/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_17/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_18/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_19/frame_000.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_19/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_20/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_21/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_22/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_23/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_24/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_25/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_26/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_27/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_28/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_29/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_30/frame_000.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_30/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_7/frame_000.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_7/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_8/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_9/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/run_config.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/blaze/frame_000.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/blaze/frame_002.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/blaze/frame_005.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/blaze/frame_007.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/blaze/measurements.csv`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/blaze/measurements.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/blaze/run_config.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/manifest.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/sides/frame_001.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/sides/frame_005.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/sides/frame_008.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/sides/frame_009.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/sides/frame_011.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/sides/frame_014.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/sides/frame_017.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/sides/measurements.csv`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/sides/measurements.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/sides/run_config.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/zernike/frame_000.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/zernike/frame_011.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/zernike/frame_018.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/zernike/frame_023.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/zernike/measurements.csv`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/zernike/measurements.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/zernike/run_config.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_frozen`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_frozen/blaze/blaze_32/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_frozen/blaze/blaze_40/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_frozen/blaze/blaze_48/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_frozen/blaze/measurements.csv`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_frozen/blaze/measurements.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_frozen/blaze/run_config.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_frozen/manifest.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_frozen/zernike/measurements.csv`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_frozen/zernike/measurements.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_frozen/zernike/nmax_17/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_frozen/zernike/nmax_18/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_frozen/zernike/nmax_19/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_frozen/zernike/run_config.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_power_changed`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_power_changed/blaze/blaze_32/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_power_changed/blaze/blaze_40/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_power_changed/blaze/blaze_48/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_power_changed/blaze/measurements.csv`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_power_changed/blaze/measurements.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_power_changed/manifest.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_power_changed/zernike/measurements.csv`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_power_changed/zernike/measurements.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_power_changed/zernike/nmax_17/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_power_changed/zernike/nmax_18/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_power_changed/zernike/nmax_19/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261030_gs1000`
- ❌ 缺失 `data/debug/hw_gs_square_scans/20261030_gs1000/blaze/metrics_00X.json`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261020b`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261020b/best_phase_nmax4.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/best_phase_nmax4.npy`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_order`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_order/best_phase_nmax4.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax11`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax11/phase_nmax11.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax15`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax15/phase_nmax15.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax20`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax20/phase_nmax20.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax25`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax25/phase_nmax25.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax3`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax3/phase_nmax3.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax4`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax4/best_phase_nmax4.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax4/phase_nmax4.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax6`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax6/phase_nmax6.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax8`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax8/phase_nmax8.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_1`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_1/stage_a.json/n[done`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_1/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_1/summary.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_2`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_2/stage_a.npz`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_2/stage_a.npz/n`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_3`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_3/stage_a.npz`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_3/stage_a.npz/n`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_4`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_4/stage_a.npz`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_4/stage_a.npz/n`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_5`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_5/stage_a.npz`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_5/stage_a.npz/n`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_6`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_6/stage_a.npz`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_6/stage_a.npz/n`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/manifest.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax11`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax11/after_nmax11.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax11/baseline_flat.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax11/baseline_gs_freeform.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax11/before_nmax11.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax11/best_phase_nmax11.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax11/fit_coefficients_nmax11.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax11/phase_nmax11.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax11/spgd_nmax11.csv`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax11/summary.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax15`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax15/after_nmax15.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax15/baseline_flat.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax15/baseline_gs_freeform.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax15/before_nmax15.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax15/best_phase_nmax15.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax15/fit_coefficients_nmax15.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax15/phase_nmax15.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax15/spgd_nmax15.csv`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax15/summary.json`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax15/summary.json/n`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax20`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax20/after_nmax20.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax20/baseline_flat.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax20/baseline_gs_freeform.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax20/before_nmax20.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax20/best_phase_nmax20.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax20/fit_coefficients_nmax20.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax20/phase_nmax20.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax20/spgd_nmax20.csv`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax20/summary.json`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax20/summary.json/n`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax25`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax25/after_nmax25.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax25/baseline_flat.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax25/baseline_gs_freeform.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax25/before_nmax25.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax25/best_phase_nmax25.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax25/fit_coefficients_nmax25.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax25/phase_nmax25.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax25/spgd_nmax25.csv`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax25/summary.json`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax25/summary.json/n`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax3`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax3/after_nmax3.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax3/baseline_flat.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax3/baseline_gs_freeform.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax3/before_nmax3.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax3/best_phase_nmax3.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax3/fit_coefficients_nmax3.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax3/phase_nmax3.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax3/spgd_nmax3.csv`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax3/summary.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4/after_nmax4.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4/baseline_flat.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4/baseline_gs_freeform.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4/before_nmax4.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4/best_phase_nmax4.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4/fit_coefficients_nmax4.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4/phase_nmax4.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4/spgd_nmax4.csv`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4/summary.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax6`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax6/after_nmax6.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax6/baseline_flat.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax6/baseline_gs_freeform.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax6/before_nmax6.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax6/best_phase_nmax6.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax6/fit_coefficients_nmax6.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax6/phase_nmax6.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax6/spgd_nmax6.csv`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax6/summary.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax8`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax8/after_nmax8.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax8/baseline_flat.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax8/baseline_gs_freeform.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax8/before_nmax8.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax8/best_phase_nmax8.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax8/fit_coefficients_nmax8.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax8/phase_nmax8.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax8/spgd_nmax8.csv`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax8/summary.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_1`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_1/stage_a.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_1/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_1/summary.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_2`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_2/stage_a.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_2/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_2/summary.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_3`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_3/stage_a.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_3/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_3/summary.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_4`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_4/stage_a.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_4/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_4/summary.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_5`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_5/stage_a.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_5/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_5/summary.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_6`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_6/stage_a.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_6/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_6/summary.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_7`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_7/stage_a.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_7/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_7/summary.json`
- ✅ 存在 `data/debug/light_level_probe`
- ❌ 缺失 `data/debug/model_in_loop_auto/20261019_camera_reposition`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_164549/baseline_a/stage.log`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_164549/manifest.json`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_170435/baseline_a/stage.log`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_170646/baseline_a/slm_model_in_loop/20261021_170726/panel_scan/summary.json`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_170646/baseline_a/stage.log`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_170646/manifest.json`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_172236`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_172236/baseline_a/slm_model_in_loop/20261021_172309`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_172236/baseline_a/slm_model_in_loop/20261021_172309/bench_geometry.json`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_172236/baseline_a/slm_model_in_loop/20261021_172309/best_phase.npy`
- ❌ 缺失 `data/debug/model_in_loop_hw_ab/20261021_172236/baseline_a/slm_model_in_loop/20261021_172309/failure.json`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_172236/baseline_a/slm_model_in_loop/20261021_172309/fitted_coefficients.npy`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_172236/baseline_a/slm_model_in_loop/20261021_172309/handoff_summary.json`
- ❌ 缺失 `data/debug/model_in_loop_hw_ab/20261021_172236/baseline_a/slm_model_in_loop/20261021_172309/model_error_terms.json`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_172236/baseline_a/slm_model_in_loop/20261021_172309/preflight/flat.npz`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_172236/baseline_a/slm_model_in_loop/20261021_172309/probes/manifest.json`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_172236/baseline_a/slm_model_in_loop/20261021_172309/run_config.json`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_172236/baseline_a/slm_model_in_loop/20261021_172309/source_amplitude.npy`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_172236/large_1000/stage.log`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_172236/manifest.json`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261022_133250`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261022_164859/baseline_a/stage.log`
- ❌ 缺失 `data/debug/model_in_loop_hw_ab/20261022_{133250`
- ❌ 缺失 `data/debug/model_in_loop_hw_ab/<会话`
- ✅ 存在 `data/debug/model_in_loop_hw_collect`
- ❌ 缺失 `data/debug/model_in_loop_hw_collect/20261025`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123031/model_in_loop_hw_collect_20261025_123031.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123037/model_in_loop_hw_collect_20261025_123037.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123042/model_in_loop_hw_collect_20261025_123042.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123048/model_in_loop_hw_collect_20261025_123048.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123116/model_in_loop_hw_collect_20261025_123116.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123151/model_in_loop_hw_collect_20261025_123151.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123157/model_in_loop_hw_collect_20261025_123157.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123202/model_in_loop_hw_collect_20261025_123202.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123208/model_in_loop_hw_collect_20261025_123208.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123738/model_in_loop_hw_collect_20261025_123738.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123745/model_in_loop_hw_collect_20261025_123745.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123750/model_in_loop_hw_collect_20261025_123750.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123756/model_in_loop_hw_collect_20261025_123756.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123826/model_in_loop_hw_collect_20261025_123826.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123902/model_in_loop_hw_collect_20261025_123902.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123908/model_in_loop_hw_collect_20261025_123908.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123913/model_in_loop_hw_collect_20261025_123913.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123919/model_in_loop_hw_collect_20261025_123919.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_150515/model_in_loop_hw_collect_20261025_150515.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_150519/model_in_loop_hw_collect_20261025_150519.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_150523/model_in_loop_hw_collect_20261025_150523.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_150528/model_in_loop_hw_collect_20261025_150528.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_150550/model_in_loop_hw_collect_20261025_150550.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_150621/model_in_loop_hw_collect_20261025_150621.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_150626/model_in_loop_hw_collect_20261025_150626.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_150630/model_in_loop_hw_collect_20261025_150630.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_150635/model_in_loop_hw_collect_20261025_150635.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_151210/model_in_loop_hw_collect_20261025_151210.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_151214/model_in_loop_hw_collect_20261025_151214.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_151218/model_in_loop_hw_collect_20261025_151218.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_151223/model_in_loop_hw_collect_20261025_151223.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_151245/model_in_loop_hw_collect_20261025_151245.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_151316/model_in_loop_hw_collect_20261025_151316.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_151321/model_in_loop_hw_collect_20261025_151321.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_151325/model_in_loop_hw_collect_20261025_151325.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_151329/model_in_loop_hw_collect_20261025_151329.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_171033/model_in_loop_hw_collect_20261025_171033.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_171036/model_in_loop_hw_collect_20261025_171036.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_171040/model_in_loop_hw_collect_20261025_171040.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_171044/model_in_loop_hw_collect_20261025_171044.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_171106/model_in_loop_hw_collect_20261025_171106.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_171135/model_in_loop_hw_collect_20261025_171135.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_171139/model_in_loop_hw_collect_20261025_171139.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_171143/model_in_loop_hw_collect_20261025_171143.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_171148/model_in_loop_hw_collect_20261025_171148.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_172907/model_in_loop_hw_collect_20261025_172907.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_172911/model_in_loop_hw_collect_20261025_172911.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_172915/model_in_loop_hw_collect_20261025_172915.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_172920/model_in_loop_hw_collect_20261025_172920.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_172942/model_in_loop_hw_collect_20261025_172942.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_173012/model_in_loop_hw_collect_20261025_173012.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_173017/model_in_loop_hw_collect_20261025_173017.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_173021/model_in_loop_hw_collect_20261025_173021.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_173026/model_in_loop_hw_collect_20261025_173026.pkl`
- ❌ 缺失 `data/debug/model_in_loop_hw_runbook/20261019_corrected_geometry`
- ✅ 存在 `data/debug/model_in_loop_hw_sweep/20261019_105952`
- ❌ 缺失 `data/debug/model_in_loop_hw_sweep_20260930_171003/20260930_171003/model_in_loop_hw_sweep_20260930_171003_20260930_171003.json`
- ✅ 存在 `data/debug/multishape_forward`
- ❌ 缺失 `data/debug/multishape_forward/<会话时间戳`
- ✅ 存在 `data/debug/multishape_forward/rectangle08_20261023_191444`
- ✅ 存在 `data/debug/multishape_forward/square_20261023_181846`
- ✅ 存在 `data/debug/multishape_forward/square_20261023_182153`
- ✅ 存在 `data/debug/multishape_forward/star08_20261023_191213`
- ✅ 存在 `data/debug/multishape_forward/triangle08_20261023_191105`
- ✅ 存在 `data/debug/optical_path_probe`
- ✅ 存在 `data/debug/optical_path_probe/20261022_084725Z`
- ✅ 存在 `data/debug/optical_path_probe/20261022_084725Z/optical_path_probe.json`
- ✅ 存在 `data/debug/optical_path_probe/20261028_083057Z`
- ✅ 存在 `data/debug/optical_path_probe/gs_control_1000px/20261030_031154Z`
- ✅ 存在 `data/debug/optical_path_probe/gs_control_1000px/20261030_031154Z/optical_path_probe.json`
- ✅ 存在 `data/debug/optical_path_probe/retry_20261029/20261029_013227Z/optical_path_probe.json`
- ✅ 存在 `data/debug/physical_forward_defocus`
- ❌ 缺失 `data/debug/physical_forward_defocus/<YYYYMMDD_HHMMSS`
- ✅ 存在 `data/debug/physical_forward_defocus/alignment_20261022_1812/focus.json`
- ✅ 存在 `data/debug/physical_forward_defocus/capture_after_focus_20261023_145446/seed_probe000.npy`
- ✅ 存在 `data/debug/physical_forward_defocus/sampling_followup_evidence_20261022.json`
- ❌ 缺失 `data/debug/sim_calib_abba_20261001_163743/sim_calib_abba_`
- ✅ 存在 `data/debug/sim_vs_hw_20261022`
- ✅ 存在 `data/debug/sim_vs_hw_20261023_v2`
- ✅ 存在 `data/debug/slm_bench_calibration/20261026_091101/panel_scan`
- ❌ 缺失 `data/debug/slm_disturbance_compare/20261013_165039`
- ❌ 缺失 `data/debug/slm_disturbance_compare/20261013_165039/disturbance.json`
- ❌ 缺失 `data/debug/slm_disturbance_compare/20261013_165039/disturbance_panel_rad.npy`
- ❌ 缺失 `data/debug/slm_exposure_analysis/20261013_1935`
- ❌ 缺失 `data/debug/slm_exposure_analysis/20261013_1935/analysis_pixelwise.json`
- ❌ 缺失 `data/debug/slm_exposure_analysis/20261013_1935/analysis_with_repeat.json`
- ❌ 缺失 `data/debug/slm_flat_stability/20261015_152929`
- ❌ 缺失 `data/debug/slm_flat_stability/20261016_model_in_loop_preflight`
- ✅ 存在 `data/debug/slm_gs_refine`
- ❌ 缺失 `data/debug/slm_gs_refine/20261012_173710`
- ❌ 缺失 `data/debug/slm_gs_refine/20261012_174456`
- ❌ 缺失 `data/debug/slm_gs_refine/20261012_174627`
- ❌ 缺失 `data/debug/slm_gs_refine/20261012_175632`
- ❌ 缺失 `data/debug/slm_gs_refine/20261012_175810`
- ❌ 缺失 `data/debug/slm_gs_refine/20261012_175929`
- ❌ 缺失 `data/debug/slm_gs_refine/20261012_180222`
- ❌ 缺失 `data/debug/slm_gs_refine/20261012_180222/forward_diagnostic.json`
- ✅ 存在 `data/debug/slm_gs_refine/20261014_100441/replay/00_flat.npy`
- ✅ 存在 `data/debug/slm_gs_refine/20261014_100441/replay/02_best.npy`
- ✅ 存在 `data/debug/slm_gs_refine/20261014_143035/replay/00_flat.npy`
- ✅ 存在 `data/debug/slm_gs_refine/20261014_143035/replay/02_best.npy`
- ❌ 缺失 `data/debug/slm_gs_refine/20261014_{100441`
- ✅ 存在 `data/debug/slm_gs_refine/20261015_205654/best_far_field.png`
- ✅ 存在 `data/debug/slm_gs_refine/20261015_205654/slm_gs_refine_20261015_205654.pkl`
- ✅ 存在 `data/debug/slm_gs_refine/20261015_205848/best_far_field.png`
- ✅ 存在 `data/debug/slm_gs_refine/20261015_205848/slm_gs_refine_20261015_205848.pkl`
- ✅ 存在 `data/debug/slm_gs_refine/20261015_210048/best_far_field.png`
- ✅ 存在 `data/debug/slm_gs_refine/20261015_210048/slm_gs_refine_20261015_210048.pkl`
- ✅ 存在 `data/debug/slm_gs_refine/20261015_210155/best_far_field.png`
- ✅ 存在 `data/debug/slm_gs_refine/20261015_210155/slm_gs_refine_20261015_210155.pkl`
- ✅ 存在 `data/debug/slm_gs_refine/20261015_211345/best_far_field.png`
- ✅ 存在 `data/debug/slm_gs_refine/20261015_211345/best_phase.npy`
- ✅ 存在 `data/debug/slm_gs_refine/20261015_211345/slm_gs_refine_20261015_211345.pkl`
- ✅ 存在 `data/debug/slm_gs_refine/20261015_213818`
- ✅ 存在 `data/debug/slm_gs_refine/20261015_213818/replay_uniformity`
- ✅ 存在 `data/debug/slm_gs_refine/20261015_issue_comparison`
- ✅ 存在 `data/debug/slm_gs_refine/20261015_issue_comparison/summary.json`
- ❌ 缺失 `data/debug/slm_gs_refine/<ts`
- ❌ 缺失 `data/debug/slm_gs_refine/<时间戳`
- ❌ 缺失 `data/debug/slm_gs_refine/<运行时间`
- ❌ 缺失 `data/debug/slm_gsnet_square_20261001_160808`
- ❌ 缺失 `data/debug/slm_model_compare/20261013_1943_seed_20261013`
- ❌ 缺失 `data/debug/slm_model_compare/20261013_2015_new_session/phase_bandwidth_paired.csv`
- ❌ 缺失 `data/debug/slm_model_compare/20261013_2015_new_session/psf_paired.csv`
- ❌ 缺失 `data/debug/slm_model_compare/20261013_phase_bandwidth_trial/paired_holdouts.csv`
- ❌ 缺失 `data/debug/slm_model_compare/20261013_psf_trial/paired_observable_holdouts.csv`
- ❌ 缺失 `data/debug/slm_model_compare/20261014_5shape_seed_20261013_verified`
- ✅ 存在 `data/debug/slm_model_in_loop`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_105911/bench_geometry.json`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_110921`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_111258`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_112034`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_113152`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_114650`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_114650/best_far_field.png`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_114650/probes/round_01_shaped_00.npz`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_142415`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_145819/best_far_field.png`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_145819/probes/round_01_shaped_00.npz`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_152341`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_152341/best_far_field.png`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_152341/panel_scan`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_152341/probes/round_01_shaped_00.npz`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_152831`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_173039`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_173230`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_174628`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_174740`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_201100`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_201229`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_202804`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_203426`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_203553`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_203702`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_203905`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_204410`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_210403/preflight`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261014_105041/probes/round_01_shaped_00.npz`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261014_105205/probes/round_01_shaped_00.npz`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261014_105339/probes/round_01_shaped_00.npz`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261014_105438/probes/round_01_shaped_00.npz`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261014_105535/probes/round_01_shaped_00.npz`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261014_110203/probes/round_01_shaped_00.npz`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261014_110305/probes/round_01_shaped_00.npz`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261014_170820/probes/round_01_shaped_00.npz`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261014_171043/probes/round_01_shaped_00.npz`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261014_171219/probes/round_01_shaped_00.npz`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261014_171301/probes/round_01_shaped_00.npz`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261014_171340/probes/round_01_shaped_00.npz`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261014_refit_analysis/paired_holdouts.json`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261016_173002`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261016_174330`
- ✅ 存在 `data/debug/slm_model_in_loop/20261019_094121`
- ✅ 存在 `data/debug/slm_model_in_loop/20261019_094121/panel_scan`
- ✅ 存在 `data/debug/slm_model_in_loop/20261019_100334`
- ✅ 存在 `data/debug/slm_model_in_loop/20261019_100334/preflight/flat.npz`
- ✅ 存在 `data/debug/slm_model_in_loop/20261019_103208`
- ✅ 存在 `data/debug/slm_model_in_loop/20261019_103208/preflight/flat.npz`
- ✅ 存在 `data/debug/slm_model_in_loop/20261019_103259/preflight/flat.npz`
- ✅ 存在 `data/debug/slm_model_in_loop/20261019_103347/preflight/flat.npz`
- ✅ 存在 `data/debug/slm_model_in_loop/20261019_110323`
- ✅ 存在 `data/debug/slm_model_in_loop/20261019_110323/preflight/flat.npz`
- ✅ 存在 `data/debug/slm_model_in_loop/20261019_111359/preflight/flat.npz`
- ✅ 存在 `data/debug/slm_model_in_loop/20261019_111941`
- ✅ 存在 `data/debug/slm_model_in_loop/20261019_111941/preflight/flat.npz`
- ❌ 缺失 `data/debug/slm_model_in_loop/<ts`
- ❌ 缺失 `data/debug/slm_model_in_loop/<时间戳`
- ✅ 存在 `data/debug/slm_model_in_loop/exposure_drift_20261013_1547`
- ✅ 存在 `data/debug/slm_model_in_loop/exposure_flat_20261013_191614`
- ✅ 存在 `data/debug/slm_model_in_loop/panel_scan_20261013_2003`
- ✅ 存在 `data/debug/slm_model_in_loop/panel_scan_fine_20261013_2005`
- ✅ 存在 `data/debug/slm_model_in_loop/panel_scan_vertical_20261013_2007`
- ✅ 存在 `data/debug/slm_model_in_loop/short_exposure_20261013_1553`
- ✅ 存在 `data/debug/slm_model_in_loop/short_exposure_shaped_20261013_1558`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261013_154143`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261013_160630_fixed040`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261013_1733_four_vs_two`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261013_1748_disturbed_four_vs_two`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261013_2013_warm_vs_cold`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261014_1115_zero_vs_refit`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261014_1115_zero_vs_refit/01_A.npy`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261014_1115_zero_vs_refit/02_B.npy`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261014_1115_zero_vs_refit/04_B.npy`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261014_1115_zero_vs_refit/05_A.npy`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261016_1044_uniformity_weights/01_A.npy`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261016_1044_uniformity_weights/02_B.npy`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261016_1044_uniformity_weights/04_B.npy`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261016_1044_uniformity_weights/05_A.npy`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261016_1051_refit_steps/01_A.npy`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261016_1051_refit_steps/02_B.npy`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261016_1051_refit_steps/04_B.npy`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261016_1051_refit_steps/05_A.npy`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261016_1053_refit_steps_reverse/01_A.npy`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261016_1053_refit_steps_reverse/02_B.npy`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261016_1053_refit_steps_reverse/04_B.npy`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261016_1053_refit_steps_reverse/05_A.npy`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261016_1745_soft_edge_abba`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/<ts`
- ❌ 缺失 `data/debug/slm_panel_locate/20261015_152101`
- ❌ 缺失 `data/debug/slm_pib_`
- ❌ 缺失 `data/debug/slm_pib_hw_20260929`
- ❌ 缺失 `data/debug/slm_pib_online`
- ❌ 缺失 `data/debug/slm_pib_online/<ts`
- ❌ 缺失 `data/debug/slm_pib_pearson`
- ❌ 缺失 `data/debug/slm_pib_shape_20260926_175334`
- ❌ 缺失 `data/debug/slm_transfer/$(Get-Date`
- ✅ 存在 `data/debug/slm_transfer/20261026_094555`
- ✅ 存在 `data/debug/slm_transfer/axial_plane_20261026/capture`
- ❌ 缺失 `data/debug/slm_transfer/axial_plane_20261026/new_scan`
- ✅ 存在 `data/debug/slm_transfer/carrier_probe_20261028/carrier_probe.json`
- ✅ 存在 `data/debug/slm_transfer/cold_start_20261026_1004`
- ✅ 存在 `data/debug/slm_transfer/cold_start_20261026_1004/random_no_abba/square_best.npy`
- ✅ 存在 `data/debug/slm_transfer/cold_start_20261026_1004/selected_final/frozen_parameters.json`
- ✅ 存在 `data/debug/slm_transfer/live_no_gs_20261024_0926`
- ✅ 存在 `data/debug/slm_transfer/live_no_gs_20261024_0926/anchored`
- ✅ 存在 `data/debug/slm_transfer/live_no_gs_20261024_0926/fit/frozen_parameters.json`
- ✅ 存在 `data/debug/slm_transfer/live_no_gs_cv_20261024`
- ✅ 存在 `data/debug/slm_transfer/live_no_gs_cv_anchored_20261024`
- ✅ 存在 `data/debug/slm_transfer/live_no_gs_cv_anchored_20261024_analysis`
- ✅ 存在 `data/debug/slm_transfer/multishape_nogs_20261029`
- ✅ 存在 `data/debug/slm_transfer/multishape_nogs_20261029/lanes`
- ✅ 存在 `data/debug/slm_transfer/multishape_nogs_20261029/p20261029`
- ✅ 存在 `data/debug/slm_transfer/multishape_nogs_20261029/p20261029_analytic_init_s600/circle_powerA_20261029`
- ✅ 存在 `data/debug/slm_transfer/multishape_nogs_20261029/p20261029_strict_cold_s200/star`
- ✅ 存在 `data/debug/slm_transfer/multishape_nogs_20261029/p20261029_strict_cold_s200/star/shape/star/mono12_continue/recording/process.csv`
- ✅ 存在 `data/debug/slm_transfer/multishape_nogs_20261029/p20261029_strict_cold_s200/star/shape/star/mono12_cv/recording/process.csv`
- ✅ 存在 `data/debug/slm_transfer/multishape_nogs_20261029/p20261029_strict_cold_s200/star/shape/star/mono12_dense/recording/process.csv`
- ✅ 存在 `data/debug/slm_transfer/multishape_nogs_20261029/p20261029_strict_cold_s200/star/shape/star/transport_cv/recording/process.csv`
- ✅ 存在 `data/debug/slm_transfer/multishape_nogs_20261030`
- ❌ 缺失 `data/debug/slm_transfer/multishape_nogs_20261030/axial_conditional_NEW`
- ❌ 缺失 `data/debug/slm_transfer/multishape_nogs_20261030/square1000_diff_experiment_NEW`
- ❌ 缺失 `data/debug/slm_transfer/multishape_nogs_20261030/square1000_diff_preview_NEW`
- ✅ 存在 `data/debug/slm_transfer/multishape_nogs_20261030/square1000_model_diagnostic_20261030_v7/defect_sim`
- ✅ 存在 `data/debug/slm_transfer/multishape_scale_20261028`
- ✅ 存在 `data/debug/slm_transfer/multishape_scale_20261028/r2_next`
- ✅ 存在 `data/debug/slm_transfer/multishape_scale_20261028/s1200`
- ❌ 缺失 `data/debug/slm_transfer/multishape_scale_20261028/s200`
- ✅ 存在 `data/debug/slm_transfer/multishape_scale_20261028/s600`
- ✅ 存在 `data/debug/slm_transfer/multishape_scale_20261028/zero_ref`
- ❌ 缺失 `data/debug/slm_transfer/new_synthetic_session`
- ✅ 存在 `data/debug/slm_transfer/no_gs_ccd_moved_20261027_2106`
- ✅ 存在 `data/debug/slm_transfer/no_gs_ccd_moved_20261027_2106/local_shape1200`
- ✅ 存在 `data/debug/slm_transfer/no_gs_ccd_moved_20261027_2106/model640/frozen_parameters.json`
- ❌ 缺失 `data/debug/slm_transfer/no_gs_next`
- ❌ 缺失 `data/debug/slm_transfer/pretrain`
- ✅ 存在 `data/debug/slm_transfer/square1200_20261026_1350/model_wide640/frozen_parameters.json`
- ✅ 存在 `data/debug/slm_transfer/square1200_20261026_1350/mono12_continue`
- ✅ 存在 `data/debug/slm_transfer/square1200_20261026_1350/mono12_cv`
- ✅ 存在 `data/debug/slm_transfer/square1200_20261026_1350/mono12_dense`
- ✅ 存在 `data/debug/slm_transfer/square1200_20261026_1350/transport_cv_r2`
- ✅ 存在 `data/debug/slm_transfer/square1200_explore_20261026_1435`
- ❌ 缺失 `data/debug/slm_transfer/square1200_new_session`
- ✅ 存在 `data/debug/slm_transfer/synthetic_20261024_shapes`
- ✅ 存在 `data/debug/slm_transfer/synthetic_20261024_shift`
- ✅ 存在 `data/debug/slm_transfer/transfer_plane_20261027`
- ✅ 存在 `data/debug/slm_transfer/transfer_square1200_20261027_r2`
- ✅ 存在 `data/debug/slm_transfer/transfer_tri_star_20261027`
- ❌ 缺失 `data/debug/slm_zernike_`
- ❌ 缺失 `data/debug/slm_zernike_shaping_rms_pib_20260926_162917/20260926_162917`
- ❌ 缺失 `data/debug/slm_zernike_shaping_rmse_out_20260926_163121/20260926_163121`
- ❌ 缺失 `data/debug/three_shape_reproduction/$tag`
- ✅ 存在 `data/debug/three_shape_reproduction/offline_bootstrap_20261023`
- ✅ 存在 `data/debug/transfer_integration/live_20261023`
- ✅ 存在 `data/debug/transfer_integration/replay_20261023`
- ❌ 缺失 `data/debug/transfer_multishape_experiment/capture_20261023_1905`
- ✅ 存在 `data/debug/transfer_multishape_experiment/capture_20261023_1938`
- ✅ 存在 `data/debug/transfer_repro_no_gs/block0`
- ✅ 存在 `data/debug/transfer_repro_no_gs/block1`
- ❌ 缺失 `data/debug/transfer_repro_no_gs/block{0`
- ❌ 缺失 `data/debug/uncertainty_probes_hw`
- ❌ 缺失 `data/debug/uncertainty_probes_sim`
- ❌ 缺失 `data/debug/uncertainty_selector_sweep`
- ✅ 存在 `data/debug/validate_transfer_shapes/anchored_20261023_2002`
- ✅ 存在 `data/debug/validate_transfer_shapes/anchored_20261023_2002/artifact_audit.json`
- ✅ 存在 `data/debug/validate_transfer_shapes/anchored_20261023_2002/manifest.json`
- ✅ 存在 `data/debug/validate_transfer_shapes/anchored_20261023_2002/summary.json`
- ✅ 存在 `data/debug/validate_transfer_shapes/paired_20261023_1942`
- ✅ 存在 `data/debug/validate_transfer_shapes/paired_20261023_1942/manifest.json`
- ❌ 缺失 `data/debug/worktrees/model-in-loop-20261014`
- ❌ 缺失 `data/debug_dedup_quarantine`
- ❌ 缺失 `data/diff_beam/backprop_square_20260910_212934`
- ❌ 缺失 `data/diff_beam/flat_tiff_20260911/flat_1200us.tiff/.npy`
- ❌ 缺失 `data/diff_beam/run_<ts`
- ❌ 缺失 `data/diff_shaping_circle`
- ❌ 缺失 `data/diff_shaping_hw`
- ❌ 缺失 `data/diff_shaping_spot`
- ✅ 存在 `data/diffraction_efficiency/fine_newoptics_20261023_1415`
- ✅ 存在 `data/diffraction_efficiency/gifscan_20261023_1420`
- ✅ 存在 `data/diffraction_efficiency/gs_ceiling_20261023_151229`
- ❌ 缺失 `data/fouriergsnet_sim/native_w4b`
- ❌ 缺失 `data/fouriergsnet_sim/native_w4b/config.json`
- ❌ 缺失 `data/gs_refine_live_20261012/gs_refine_12_005ms/slm_gs_refine/20261012_163615`
- ❌ 缺失 `data/gs_refine_live_20261012/panel_locate_23020027_005ms`
- ❌ 缺失 `data/gs_refine_live_20261012/replay_23020027_005ms`
- ❌ 缺失 `data/gs_refine_live_20261012/weighted_gs_005ms`
- ❌ 缺失 `data/gsnet_train/run-`
- ❌ 缺失 `data/gsnet_train/run-20260929_225847`
- ❌ 缺失 `data/gsnet_train/run-20260929_225847/comparison.png`
- ❌ 缺失 `data/gsnet_train/run-20260929_225847/fourier_gsnet_best.pt`
- ❌ 缺失 `data/gsnet_train/run-20260929_225847/summary.json`
- ❌ 缺失 `data/gsnet_train/run-20260929_225847/train_history.png`
- ✅ 存在 `data/hw_index_cache.json`
- ❌ 缺失 `data/hw_index_snapshot.json`
- ✅ 存在 `data/md_test.zip`
- ❌ 缺失 `data/md_test/md_img-100v/192.168.0.114/192.168.0.114-000.png`
- ❌ 缺失 `data/md_test/md_img-100v/192.168.0.114/192.168.0.114-001.png`
- ❌ 缺失 `data/md_test/md_img-100v/192.168.0.114/192.168.0.114-002.png`
- ❌ 缺失 `data/md_test/md_img-100v/192.168.0.124/192.168.0.124-002.png`
- ❌ 缺失 `data/model_in_loop_hw/bench_geometry.json`
- ❌ 缺失 `data/model_in_loop_hw/calibration_records.npz`
- ❌ 缺失 `data/model_in_loop_hw/model_vs_measured.png`
- ❌ 缺失 `data/model_in_loop_hw/sweep_records.npz`
- ❌ 缺失 `data/shape_trials/blackbox_path_`
- ✅ 存在 `data/shape_trials/blackbox_path_20261029_194837`
- ✅ 存在 `data/shape_trials/blackbox_path_gs_20261029_201406`
- ✅ 存在 `data/shape_trials/blackbox_replay_20261029_212302`
- ✅ 存在 `data/shape_trials/blackbox_replay_20261029_212830`
- ❌ 缺失 `data/sim_calib/freeform_transfer_hw.npz`
- ❌ 缺失 `data/slm_configs/slm/22030102.json`
- ❌ 缺失 `data/slm_corrections/run_n5_attempt3.log`
- ❌ 缺失 `data/slm_corrections/run_n5_attempt4.log;`
- ❌ 缺失 `data/slm_corrections/wfs_probe.log`
- ❌ 缺失 `data/slm_corrections/wfs_probe.log;`
- ✅ 存在 `data/slm_square/20260910_203509`
- ✅ 存在 `data/slm_square/20260910_204311`
- ❌ 缺失 `data/wfs_configs/wfs/M01219666.json`
- ❌ 缺失 `data/zernike_correction/debug_<ts`
- ❌ 缺失 `data/zernike_correction/raw_scan_`
- ❌ 缺失 `data/zernike_correction/raw_scan_20260915_235549.json`
- ❌ 缺失 `data/zernike_correction/report_20260915_235549.json`
- ❌ 缺失 `data/zernike_response_matrix/zm_recal_532_20260916.h5`
- ❌ 缺失 `data/zernike_response_matrix/zm_recal_532_20260916.json`
- ❌ 缺失 `data/zernike_response_matrix/zm_slm22030102_wfsM01219666_532nm_20260915_235207.h5`
- ❌ 缺失 `data/zernike_response_matrix_n5`

### report/diff_beam/diff_beam_frame_analysis/README.md

- ❌ 缺失 `data/diff_beam/backprop_square_20260910_212934`
- ❌ 缺失 `data/diff_beam/flat_tiff_20260911/flat_1200us.tiff/.npy`
- ❌ 缺失 `data/diff_beam/run_<ts`

### report/fouriergsnet_pipeline/hardware_run_20261001.md

- ❌ 缺失 `data/debug/slm_gsnet_square_20261001_160808`

### report/fouriergsnet_pipeline/offline_training/report.md

- ❌ 缺失 `data/debug/slm_pib_`
- ❌ 缺失 `data/debug/slm_zernike_`
- ❌ 缺失 `data/gsnet_train/run-`
- ❌ 缺失 `data/gsnet_train/run-20260929_225847`
- ❌ 缺失 `data/gsnet_train/run-20260929_225847/comparison.png`
- ❌ 缺失 `data/gsnet_train/run-20260929_225847/fourier_gsnet_best.pt`
- ❌ 缺失 `data/gsnet_train/run-20260929_225847/summary.json`
- ❌ 缺失 `data/gsnet_train/run-20260929_225847/train_history.png`

### report/fouriergsnet_pipeline/TODO.md

- ✅ 存在 `data`

### report/fouriergsnet_sim/fouriergsnet_sim_report/report.md

- ❌ 缺失 `data/fouriergsnet_sim/native_w4b`
- ❌ 缺失 `data/fouriergsnet_sim/native_w4b/config.json`

### report/hwdataset_corpus/hwdataset_corpus_report/report.md

- ✅ 存在 `data/debug`
- ✅ 存在 `data/hw_index_cache.json`

### report/inverse_shaping_unet/inverse_shaping_model_report/model.md

- ✅ 存在 `data/debug`
- ❌ 缺失 `data/hw_index_snapshot.json`

### report/inverse_shaping_unet/inverse_shaping_report/report.md

- ✅ 存在 `data/debug`
- ❌ 缺失 `data/hw_index_snapshot.json`

### report/loss_defects/PROCESS.md

- ✅ 存在 `data/debug`

### report/micro_deformable_mirror/voltage_test.md

- ✅ 存在 `data/1300-5-enriched.csv`
- ❌ 缺失 `data/md_test/md_img-100v/192.168.0.114/192.168.0.114-000.png`
- ❌ 缺失 `data/md_test/md_img-100v/192.168.0.114/192.168.0.114-001.png`
- ❌ 缺失 `data/md_test/md_img-100v/192.168.0.114/192.168.0.114-002.png`
- ❌ 缺失 `data/md_test/md_img-100v/192.168.0.124/192.168.0.124-002.png`

### report/model_free_ao_survey/report.md

- ❌ 缺失 `data/debug/bench_stability`

### report/README.md

- ✅ 存在 `data/debug`
- ✅ 存在 `data/debug/slm_gs_refine`
- ❌ 缺失 `data/debug/slm_gs_refine/20261014_{100441`
- ✅ 存在 `data/debug/slm_gs_refine/20261015_issue_comparison`
- ❌ 缺失 `data/debug/slm_pib_online`
- ❌ 缺失 `data/gsnet_train/run-`
- ❌ 缺失 `data/shape_trials/blackbox_path_`
- ❌ 缺失 `data/zernike_correction/raw_scan_`

### report/shape_trials/axial_plane/axial_plane_20261026.md

- ✅ 存在 `data/debug/slm_transfer/axial_plane_20261026/capture`
- ❌ 缺失 `data/debug/slm_transfer/axial_plane_20261026/new_scan`
- ✅ 存在 `data/debug/slm_transfer/square1200_20261026_1350/model_wide640/frozen_parameters.json`

### report/shape_trials/blackbox_path/analysis_20261029.md

- ❌ 缺失 `data/shape_trials/blackbox_path_`

### report/shape_trials/blackbox_path/artifacts/execution_blackbox_path.json

- ✅ 存在 `data/shape_trials/blackbox_path_20261029_194837`

### report/shape_trials/blackbox_path/artifacts/execution_blackbox_path_gs.json

- ✅ 存在 `data/shape_trials/blackbox_path_gs_20261029_201406`

### report/shape_trials/blackbox_path/artifacts/live_acquisition_20261029.md

- ✅ 存在 `data/shape_trials/blackbox_replay_20261029_212302`
- ✅ 存在 `data/shape_trials/blackbox_replay_20261029_212830`

### report/shape_trials/blackbox_path/live_20261029.md

- ✅ 存在 `data/shape_trials/blackbox_path_20261029_194837`
- ✅ 存在 `data/shape_trials/blackbox_path_gs_20261029_201406`

### report/shape_trials/cold_start/cold_start_20261026.md

- ❌ 缺失 `data/debug/slm_transfer/$(Get-Date`
- ✅ 存在 `data/debug/slm_transfer/cold_start_20261026_1004`
- ✅ 存在 `data/debug/slm_transfer/cold_start_20261026_1004/random_no_abba/square_best.npy`
- ✅ 存在 `data/debug/slm_transfer/cold_start_20261026_1004/selected_final/frozen_parameters.json`

### report/shape_trials/defocus_differentiable/defocus_differentiable_20261023.md

- ✅ 存在 `data/debug/defocus_differentiable/20261023_101822`
- ✅ 存在 `data/debug/defocus_differentiable/20261023_101822/manifest_checksums.json`

### report/shape_trials/defocus_differentiable/defocus_differentiable_20261023_r2.md

- ✅ 存在 `data/debug/physical_forward_defocus/alignment_20261022_1812/focus.json`

### report/shape_trials/defocus_differentiable/multishape_forward_plan_20261023.md

- ❌ 缺失 `data/debug/multishape_forward/<会话时间戳`

### report/shape_trials/defocus_differentiable_test_plan_20261023.md

- ✅ 存在 `data/debug/defocus_differentiable/20261023_101822`
- ❌ 缺失 `data/debug/defocus_differentiable/<本地时间戳`

### report/shape_trials/experiment_summary_20261016.md

- ✅ 存在 `data/debug`
- ❌ 缺失 `data/debug/gs_phase_uniformity/20261015_gradient_probe_1`
- ❌ 缺失 `data/debug/slm_exposure_analysis/20261013_1935`
- ❌ 缺失 `data/debug/slm_flat_stability/20261015_152929`
- ❌ 缺失 `data/debug/slm_gs_refine/<ts`
- ❌ 缺失 `data/debug/slm_model_in_loop/<ts`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/<ts`
- ❌ 缺失 `data/debug/uncertainty_probes_hw`

### report/shape_trials/experiment_summary_20261022.md

- ✅ 存在 `data/debug/light_level_probe`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_172236/baseline_a/slm_model_in_loop/20261021_172309`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261022_133250`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261022_164859/baseline_a/stage.log`
- ❌ 缺失 `data/debug/model_in_loop_hw_ab/20261022_{133250`
- ❌ 缺失 `data/debug/model_in_loop_hw_ab/<会话`
- ✅ 存在 `data/debug/optical_path_probe`
- ✅ 存在 `data/debug/optical_path_probe/20261022_084725Z`
- ✅ 存在 `data/debug/optical_path_probe/20261022_084725Z/optical_path_probe.json`

### report/shape_trials/gs_refine/gs_issue_live_20261015.md

- ✅ 存在 `data/debug/slm_gs_refine/20261015_issue_comparison`
- ✅ 存在 `data/debug/slm_gs_refine/20261015_issue_comparison/summary.json`

### report/shape_trials/gs_refine/gs_refine_live_20261012.md

- ✅ 存在 `data/debug`
- ❌ 缺失 `data/debug/<脚本名`
- ❌ 缺失 `data/debug/slm_gs_refine/20261012_173710`
- ❌ 缺失 `data/debug/slm_gs_refine/20261012_174456`
- ❌ 缺失 `data/debug/slm_gs_refine/20261012_174627`
- ❌ 缺失 `data/debug/slm_gs_refine/20261012_175632`
- ❌ 缺失 `data/debug/slm_gs_refine/20261012_175810`
- ❌ 缺失 `data/debug/slm_gs_refine/20261012_175929`
- ❌ 缺失 `data/debug/slm_gs_refine/20261012_180222`
- ❌ 缺失 `data/debug/slm_gs_refine/<运行时间`
- ❌ 缺失 `data/debug_dedup_quarantine`
- ❌ 缺失 `data/gs_refine_live_20261012/gs_refine_12_005ms/slm_gs_refine/20261012_163615`
- ❌ 缺失 `data/gs_refine_live_20261012/panel_locate_23020027_005ms`
- ❌ 缺失 `data/gs_refine_live_20261012/replay_23020027_005ms`
- ❌ 缺失 `data/gs_refine_live_20261012/weighted_gs_005ms`

### report/shape_trials/gs_refine/gs_refine_zernike_hardware_plan.md

- ✅ 存在 `data/debug`

### report/shape_trials/gs_refine/gs_uniformity_live_20261015.md

- ❌ 缺失 `data/debug/gs_phase_uniformity/20261015_gradient_probe_1`
- ✅ 存在 `data/debug/slm_gs_refine/20261015_211345/best_phase.npy`
- ✅ 存在 `data/debug/slm_gs_refine/20261015_213818`
- ✅ 存在 `data/debug/slm_gs_refine/20261015_213818/replay_uniformity`

### report/shape_trials/gs_refine/gsnet_hybrid_offline_20261013.md

- ✅ 存在 `data/debug`
- ❌ 缺失 `data/debug/gsnet_hybrid/20261013_092934`
- ❌ 缺失 `data/debug/gsnet_hybrid/20261013_093042`
- ❌ 缺失 `data/debug/gsnet_hybrid/20261013_093916`
- ❌ 缺失 `data/debug/gsnet_hybrid/20261013_100342`
- ❌ 缺失 `data/debug/gsnet_hybrid_camera/20261013_101118_581856`
- ❌ 缺失 `data/debug/gsnet_hybrid_camera/20261013_101248_554998`
- ❌ 缺失 `data/debug/gsnet_hybrid_infer/20261013_094228_032239`
- ❌ 缺失 `data/debug/gsnet_hybrid_infer/20261013_094258_729530`
- ❌ 缺失 `data/debug/gsnet_hybrid_infer/20261013_100454_477105`
- ❌ 缺失 `data/debug/slm_gs_refine/20261012_180222`
- ❌ 缺失 `data/debug/slm_gs_refine/20261012_180222/forward_diagnostic.json`
- ❌ 缺失 `data/gsnet_train/run-20260929_225847/fourier_gsnet_best.pt`

### report/shape_trials/gs_refine_fwdmodel/gs_refine_fwdmodel_calibration.md

- ✅ 存在 `data/debug/slm_gs_refine`

### report/shape_trials/gs_refine_zernike/gs_refine_zernike_uniformity.md

- ❌ 缺失 `data/debug/hw_zernike_sweep_20261020b`

### report/shape_trials/gui_closed_loop_shaping/gui_closed_loop_shaping_20261014.md

- ✅ 存在 `data/debug/slm_gs_refine/20261014_100441/replay/00_flat.npy`
- ✅ 存在 `data/debug/slm_gs_refine/20261014_100441/replay/02_best.npy`
- ✅ 存在 `data/debug/slm_gs_refine/20261014_143035/replay/00_flat.npy`
- ✅ 存在 `data/debug/slm_gs_refine/20261014_143035/replay/02_best.npy`
- ❌ 缺失 `data/debug/slm_gs_refine/20261014_{100441`
- ❌ 缺失 `data/debug/slm_gs_refine/<时间戳`

### report/shape_trials/hw_defocus_zernike/artifacts/plan_20261021.json

- ✅ 存在 `data/debug/hw_groups_20261021`
- ✅ 存在 `data/debug/hw_groups_20261021/focus_calibration`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax11/phase_nmax11.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax15/phase_nmax15.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax20/phase_nmax20.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax25/phase_nmax25.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax3/phase_nmax3.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax4/best_phase_nmax4.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax4/phase_nmax4.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax6/phase_nmax6.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax8/phase_nmax8.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_1/stage_a.npz`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_2/stage_a.npz`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_3/stage_a.npz`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_4/stage_a.npz`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_5/stage_a.npz`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_6/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/plan_20261022.json

- ✅ 存在 `data/debug/hw_groups_20261022`
- ✅ 存在 `data/debug/hw_groups_20261022/focus_calibration`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax11/phase_nmax11.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax15/phase_nmax15.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax20/phase_nmax20.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax25/phase_nmax25.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax3/phase_nmax3.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4/best_phase_nmax4.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4/phase_nmax4.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax6/phase_nmax6.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax8/phase_nmax8.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_1/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_2/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_3/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_4/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_5/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_6/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_7/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/README.md

- ✅ 存在 `data/debug`
- ✅ 存在 `data/debug/_quarantine_20261022_mixed`
- ✅ 存在 `data/debug/hw_groups_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022`

### report/shape_trials/hw_defocus_zernike/artifacts/round1_20261020/hw_groups_20261020/dz_noise_hw_gs/run_config.json

- ✅ 存在 `data/debug/hw_groups_20261020`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261020b/best_phase_nmax4.npy`

### report/shape_trials/hw_defocus_zernike/artifacts/round1_20261020/hw_groups_20261020/dz_scan_hw_gs/run_config.json

- ✅ 存在 `data/debug/hw_groups_20261020`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261020b/best_phase_nmax4.npy`

### report/shape_trials/hw_defocus_zernike/artifacts/round1_20261020/hw_groups_20261020/dz_scan_model_gs/run_config.json

- ✅ 存在 `data/debug/hw_groups_20261020`

### report/shape_trials/hw_defocus_zernike/artifacts/round1_20261020/hw_groups_20261020/groups_manifest.json

- ✅ 存在 `data/debug/hw_groups_20261020`
- ❌ 缺失 `data/debug/hw_groups_20261020/dz_noise_hw_gs`
- ✅ 存在 `data/debug/hw_groups_20261020/dz_scan_hw_gs`
- ✅ 存在 `data/debug/hw_groups_20261020/dz_scan_model_gs`
- ❌ 缺失 `data/debug/hw_groups_20261020/n`
- ✅ 存在 `data/debug/hw_groups_20261020/plan_all.json`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax11.npy`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax15.npy`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax20.npy`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax25.npy`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax6.npy`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax8.npy`
- ❌ 缺失 `data/debug/hw_groups_20261020/zernike_farfield_nmax11`
- ❌ 缺失 `data/debug/hw_groups_20261020/zernike_farfield_nmax11/n变量组`
- ❌ 缺失 `data/debug/hw_groups_20261020/zernike_farfield_nmax15`
- ❌ 缺失 `data/debug/hw_groups_20261020/zernike_farfield_nmax15/n变量组`
- ❌ 缺失 `data/debug/hw_groups_20261020/zernike_farfield_nmax20`
- ❌ 缺失 `data/debug/hw_groups_20261020/zernike_farfield_nmax20/n变量组`
- ❌ 缺失 `data/debug/hw_groups_20261020/zernike_farfield_nmax25`
- ❌ 缺失 `data/debug/hw_groups_20261020/zernike_farfield_nmax25/n变量组`
- ❌ 缺失 `data/debug/hw_groups_20261020/zernike_farfield_nmax3`
- ❌ 缺失 `data/debug/hw_groups_20261020/zernike_farfield_nmax4`
- ❌ 缺失 `data/debug/hw_groups_20261020/zernike_farfield_nmax6`
- ❌ 缺失 `data/debug/hw_groups_20261020/zernike_farfield_nmax6/n变量组`
- ❌ 缺失 `data/debug/hw_groups_20261020/zernike_farfield_nmax8`
- ❌ 缺失 `data/debug/hw_groups_20261020/zernike_farfield_nmax8/n变量组`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261020b`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261020b/best_phase_nmax4.npy`

### report/shape_trials/hw_defocus_zernike/artifacts/round1_20261020/hw_groups_20261020/plan.json

- ✅ 存在 `data/debug/hw_groups_20261020`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261020b/best_phase_nmax4.npy`

### report/shape_trials/hw_defocus_zernike/artifacts/round1_20261020/hw_groups_20261020/plan_all.json

- ✅ 存在 `data/debug/hw_groups_20261020`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax11.npy`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax15.npy`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax20.npy`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax25.npy`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax3.npy`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax4.npy`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax6.npy`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax8.npy`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261020b`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261020b/best_phase_nmax4.npy`

### report/shape_trials/hw_defocus_zernike/artifacts/round1_20261020/hw_groups_20261020/plan_zernike_farfield.json

- ✅ 存在 `data/debug/hw_groups_20261020`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax11.npy`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax15.npy`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax20.npy`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax25.npy`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax3.npy`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax4.npy`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax6.npy`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax8.npy`

### report/shape_trials/hw_defocus_zernike/artifacts/round1_20261020/hw_groups_20261020/zernike_farfield_nmax3/run_config.json

- ✅ 存在 `data/debug/hw_groups_20261020`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax3.npy`

### report/shape_trials/hw_defocus_zernike/artifacts/round1_20261020/hw_groups_20261020/zernike_farfield_nmax4/run_config.json

- ✅ 存在 `data/debug/hw_groups_20261020`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax4.npy`

### report/shape_trials/hw_defocus_zernike/artifacts/round1_20261020/hw_zernike_sweep_20261020b/summary.json

- ❌ 缺失 `data/debug/hw_zernike_sweep_20261020b`

### report/shape_trials/hw_defocus_zernike/artifacts/round2_20261021/hw_groups_20261021/dz_noise_hw_gs/run_config.json

- ✅ 存在 `data/debug/hw_groups_20261021`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax4/best_phase_nmax4.npy`

### report/shape_trials/hw_defocus_zernike/artifacts/round2_20261021/hw_groups_20261021/dz_scan_hw_gs/run_config.json

- ✅ 存在 `data/debug/hw_groups_20261021`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax4/best_phase_nmax4.npy`

### report/shape_trials/hw_defocus_zernike/artifacts/round2_20261021/hw_groups_20261021/dz_scan_model_gs/run_config.json

- ✅ 存在 `data/debug/hw_groups_20261021`

### report/shape_trials/hw_defocus_zernike/artifacts/round2_20261021/hw_groups_20261021/groups_manifest.json

- ✅ 存在 `data/debug/hw_groups_20261021`
- ❌ 缺失 `data/debug/hw_groups_20261021/dz_noise_hw_gs`
- ❌ 缺失 `data/debug/hw_groups_20261021/dz_noise_hw_gs/n分组清单`
- ✅ 存在 `data/debug/hw_groups_20261021/dz_scan_hw_gs`
- ❌ 缺失 `data/debug/hw_groups_20261021/dz_scan_hw_gs/n分组清单`
- ✅ 存在 `data/debug/hw_groups_20261021/dz_scan_model_gs`
- ❌ 缺失 `data/debug/hw_groups_20261021/dz_scan_model_gs/n分组清单`
- ✅ 存在 `data/debug/hw_groups_20261021/focus_calibration`
- ❌ 缺失 `data/debug/hw_groups_20261021/manifest.json/n`
- ❌ 缺失 `data/debug/hw_groups_20261021/n曝光`
- ✅ 存在 `data/debug/hw_groups_20261021/plan.json`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax11`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax11/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax11/n变量组`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax15`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax15/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax15/n变量组`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax20`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax20/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax20/n变量组`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax25`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax25/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax25/n变量组`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax3`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax3/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax3/n变量组`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax4`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax4/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax4/n变量组`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax6`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax6/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax6/n变量组`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax8`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax8/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax8/n变量组`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax11`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax11/phase_nmax11.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax15`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax15/phase_nmax15.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax20`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax20/phase_nmax20.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax25`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax25/phase_nmax25.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax3`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax3/phase_nmax3.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax4`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax4/best_phase_nmax4.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax4/phase_nmax4.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax6`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax6/phase_nmax6.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax8`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax8/phase_nmax8.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_1`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_1/stage_a.json/n[done`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_1/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_1/summary.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_2`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_2/stage_a.npz`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_2/stage_a.npz/n`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_3`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_3/stage_a.npz`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_3/stage_a.npz/n`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_4`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_4/stage_a.npz`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_4/stage_a.npz/n`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_5`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_5/stage_a.npz`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_5/stage_a.npz/n`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_6`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_6/stage_a.npz`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_6/stage_a.npz/n`

### report/shape_trials/hw_defocus_zernike/artifacts/round2_20261021/hw_groups_20261021/groups_manifest_failed_round.json

- ✅ 存在 `data/debug/hw_groups_20261021`
- ❌ 缺失 `data/debug/hw_groups_20261021/dz_noise_hw_gs`
- ❌ 缺失 `data/debug/hw_groups_20261021/dz_noise_hw_gs/n变量组`
- ✅ 存在 `data/debug/hw_groups_20261021/dz_scan_hw_gs`
- ❌ 缺失 `data/debug/hw_groups_20261021/dz_scan_hw_gs/n变量组`
- ✅ 存在 `data/debug/hw_groups_20261021/dz_scan_model_gs`
- ❌ 缺失 `data/debug/hw_groups_20261021/dz_scan_model_gs/n变量组`
- ❌ 缺失 `data/debug/hw_groups_20261021/n`
- ✅ 存在 `data/debug/hw_groups_20261021/plan.json`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield/phases/phase_nmax11.npy`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield/phases/phase_nmax15.npy`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield/phases/phase_nmax20.npy`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield/phases/phase_nmax25.npy`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield/phases/phase_nmax3.npy`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield/phases/phase_nmax4.npy`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield/phases/phase_nmax6.npy`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield/phases/phase_nmax8.npy`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax11`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax11/n变量组`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax15`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax15/n变量组`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax20`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax20/n变量组`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax25`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax25/n变量组`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax3`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax3/n变量组`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax4`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax4/n变量组`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax6`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax6/n变量组`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax8`
- ❌ 缺失 `data/debug/hw_groups_20261021/zernike_farfield_nmax8/n变量组`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/best_phase_nmax4.npy`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_order`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_order/best_phase_nmax4.npy`

### report/shape_trials/hw_defocus_zernike/artifacts/round2_20261021/hw_groups_20261021/manifest.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax4/best_phase_nmax4.npy`

### report/shape_trials/hw_defocus_zernike/artifacts/round2_20261021/hw_groups_20261021/plan.json

- ✅ 存在 `data/debug/hw_groups_20261021`
- ✅ 存在 `data/debug/hw_groups_20261021/focus_calibration`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax11/phase_nmax11.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax15/phase_nmax15.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax20/phase_nmax20.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax25/phase_nmax25.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax3/phase_nmax3.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax4/best_phase_nmax4.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax4/phase_nmax4.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax6/phase_nmax6.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax8/phase_nmax8.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_1/stage_a.npz`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_2/stage_a.npz`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_3/stage_a.npz`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_4/stage_a.npz`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_5/stage_a.npz`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_6/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/round2_20261021/hw_groups_20261021/zernike_farfield_nmax11/run_config.json

- ✅ 存在 `data/debug/hw_groups_20261021`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax11/phase_nmax11.npy`

### report/shape_trials/hw_defocus_zernike/artifacts/round2_20261021/hw_groups_20261021/zernike_farfield_nmax15/run_config.json

- ✅ 存在 `data/debug/hw_groups_20261021`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax15/phase_nmax15.npy`

### report/shape_trials/hw_defocus_zernike/artifacts/round2_20261021/hw_groups_20261021/zernike_farfield_nmax20/run_config.json

- ✅ 存在 `data/debug/hw_groups_20261021`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax20/phase_nmax20.npy`

### report/shape_trials/hw_defocus_zernike/artifacts/round2_20261021/hw_groups_20261021/zernike_farfield_nmax25/run_config.json

- ✅ 存在 `data/debug/hw_groups_20261021`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax25/phase_nmax25.npy`

### report/shape_trials/hw_defocus_zernike/artifacts/round2_20261021/hw_groups_20261021/zernike_farfield_nmax3/run_config.json

- ✅ 存在 `data/debug/hw_groups_20261021`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax3/phase_nmax3.npy`

### report/shape_trials/hw_defocus_zernike/artifacts/round2_20261021/hw_groups_20261021/zernike_farfield_nmax4/run_config.json

- ✅ 存在 `data/debug/hw_groups_20261021`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax4/phase_nmax4.npy`

### report/shape_trials/hw_defocus_zernike/artifacts/round2_20261021/hw_groups_20261021/zernike_farfield_nmax6/run_config.json

- ✅ 存在 `data/debug/hw_groups_20261021`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax6/phase_nmax6.npy`

### report/shape_trials/hw_defocus_zernike/artifacts/round2_20261021/hw_groups_20261021/zernike_farfield_nmax8/run_config.json

- ✅ 存在 `data/debug/hw_groups_20261021`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax8/phase_nmax8.npy`

### report/shape_trials/hw_defocus_zernike/artifacts/round2_20261021/hw_zernike_sweep_20261021/zernike_order_nmax11/summary.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261021`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/round2_20261021/hw_zernike_sweep_20261021/zernike_order_nmax15/summary.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261021`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/round2_20261021/hw_zernike_sweep_20261021/zernike_order_nmax20/summary.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261021`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/round2_20261021/hw_zernike_sweep_20261021/zernike_order_nmax25/summary.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261021`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/round2_20261021/hw_zernike_sweep_20261021/zernike_order_nmax3/summary.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261021`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/round2_20261021/hw_zernike_sweep_20261021/zernike_order_nmax4/summary.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261021`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/round2_20261021/hw_zernike_sweep_20261021/zernike_order_nmax6/summary.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261021`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/round2_20261021/hw_zernike_sweep_20261021/zernike_order_nmax8/summary.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261021`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/round2_20261021/hw_zernike_sweep_20261021/zernike_stage_a/stage_a.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261021`

### report/shape_trials/hw_defocus_zernike/artifacts/round2_20261021/hw_zernike_sweep_20261021/zernike_stage_a/summary.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261021`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/round2_20261021/hw_zernike_sweep_20261021/zernike_stage_a_1/stage_a.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261021`

### report/shape_trials/hw_defocus_zernike/artifacts/round2_20261021/hw_zernike_sweep_20261021/zernike_stage_a_1/summary.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261021`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_1/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/archive_manifest.json

- ✅ 存在 `data/debug/hw_groups_20261022`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_noise_hw_gs/frame_000.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_noise_hw_gs/frame_001.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_noise_hw_gs/frame_002.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_noise_hw_gs/frame_003.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_noise_hw_gs/frame_004.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_noise_hw_gs/frame_005.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_noise_hw_gs/frame_006.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_noise_hw_gs/frame_007.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_noise_hw_gs/frame_008.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_noise_hw_gs/frame_009.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_noise_hw_gs/hw_summary.png`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_noise_hw_gs/measurements.csv`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_noise_hw_gs/measurements.json`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_noise_hw_gs/run_config.json`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_000.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_001.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_002.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_003.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_004.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_005.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_006.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_007.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_008.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_009.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_010.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_011.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_012.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_013.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_014.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_015.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_016.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_017.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_018.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_019.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_020.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_021.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_022.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/frame_023.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/hw_summary.png`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/measurements.csv`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/measurements.json`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs/run_config.json`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_000.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_001.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_002.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_003.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_004.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_005.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_006.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_007.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_008.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_009.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_010.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_011.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_012.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_013.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_014.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_015.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_016.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_017.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_018.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_019.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_020.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_021.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_022.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_023.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_024.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_025.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_026.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_027.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_028.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/frame_029.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/hw_summary.png`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/measurements.csv`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/measurements.json`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs/run_config.json`
- ✅ 存在 `data/debug/hw_groups_20261022/focus_calibration`
- ✅ 存在 `data/debug/hw_groups_20261022/focus_calibration/slm1_20261022_105213.json`
- ✅ 存在 `data/debug/hw_groups_20261022/focus_calibration/slm1_20261022_115811.json`
- ✅ 存在 `data/debug/hw_groups_20261022/focus_calibration/slm1_20261022_142207.json`
- ✅ 存在 `data/debug/hw_groups_20261022/groups_manifest.json`
- ✅ 存在 `data/debug/hw_groups_20261022/manifest.json`
- ✅ 存在 `data/debug/hw_groups_20261022/plan.json`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax11/frame_000.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax11/frame_001.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax11/hw_summary.png`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax11/measurements.csv`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax11/measurements.json`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax11/run_config.json`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax15/frame_000.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax15/frame_001.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax15/hw_summary.png`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax15/measurements.csv`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax15/measurements.json`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax15/run_config.json`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax20/frame_000.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax20/frame_001.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax20/hw_summary.png`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax20/measurements.csv`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax20/measurements.json`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax20/run_config.json`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax25/frame_000.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax25/frame_001.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax25/hw_summary.png`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax25/measurements.csv`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax25/measurements.json`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax25/run_config.json`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax3/frame_000.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax3/frame_001.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax3/hw_summary.png`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax3/measurements.csv`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax3/measurements.json`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax3/run_config.json`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax4/frame_000.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax4/frame_001.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax4/hw_summary.png`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax4/measurements.csv`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax4/measurements.json`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax4/run_config.json`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax6/frame_000.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax6/frame_001.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax6/hw_summary.png`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax6/measurements.csv`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax6/measurements.json`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax6/run_config.json`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax8/frame_000.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax8/frame_001.npz`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax8/hw_summary.png`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax8/measurements.csv`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax8/measurements.json`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax8/run_config.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/manifest.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax11/after_nmax11.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax11/baseline_flat.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax11/baseline_gs_freeform.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax11/before_nmax11.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax11/best_phase_nmax11.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax11/fit_coefficients_nmax11.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax11/phase_nmax11.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax11/spgd_nmax11.csv`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax11/summary.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax15/after_nmax15.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax15/baseline_flat.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax15/baseline_gs_freeform.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax15/before_nmax15.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax15/best_phase_nmax15.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax15/fit_coefficients_nmax15.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax15/phase_nmax15.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax15/spgd_nmax15.csv`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax15/summary.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax20/after_nmax20.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax20/baseline_flat.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax20/baseline_gs_freeform.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax20/before_nmax20.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax20/best_phase_nmax20.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax20/fit_coefficients_nmax20.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax20/phase_nmax20.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax20/spgd_nmax20.csv`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax20/summary.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax25/after_nmax25.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax25/baseline_flat.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax25/baseline_gs_freeform.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax25/before_nmax25.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax25/best_phase_nmax25.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax25/fit_coefficients_nmax25.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax25/phase_nmax25.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax25/spgd_nmax25.csv`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax25/summary.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax3/after_nmax3.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax3/baseline_flat.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax3/baseline_gs_freeform.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax3/before_nmax3.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax3/best_phase_nmax3.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax3/fit_coefficients_nmax3.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax3/phase_nmax3.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax3/spgd_nmax3.csv`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax3/summary.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4/after_nmax4.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4/baseline_flat.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4/baseline_gs_freeform.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4/before_nmax4.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4/best_phase_nmax4.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4/fit_coefficients_nmax4.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4/phase_nmax4.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4/spgd_nmax4.csv`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4/summary.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax6/after_nmax6.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax6/baseline_flat.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax6/baseline_gs_freeform.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax6/before_nmax6.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax6/best_phase_nmax6.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax6/fit_coefficients_nmax6.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax6/phase_nmax6.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax6/spgd_nmax6.csv`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax6/summary.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax8/after_nmax8.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax8/baseline_flat.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax8/baseline_gs_freeform.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax8/before_nmax8.png`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax8/best_phase_nmax8.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax8/fit_coefficients_nmax8.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax8/phase_nmax8.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax8/spgd_nmax8.csv`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax8/summary.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_1/stage_a.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_1/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_1/summary.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_2/stage_a.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_2/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_2/summary.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_3/stage_a.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_3/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_3/summary.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_4/stage_a.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_4/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_4/summary.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_5/stage_a.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_5/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_5/summary.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_6/stage_a.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_6/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_6/summary.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_7/stage_a.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_7/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_7/summary.json`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_groups_20261022/dz_noise_hw_gs/run_config.json

- ✅ 存在 `data/debug/hw_groups_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4/best_phase_nmax4.npy`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_groups_20261022/dz_scan_hw_gs/run_config.json

- ✅ 存在 `data/debug/hw_groups_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4/best_phase_nmax4.npy`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_groups_20261022/dz_scan_model_gs/run_config.json

- ✅ 存在 `data/debug/hw_groups_20261022`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_groups_20261022/groups_manifest.json

- ✅ 存在 `data/debug/hw_groups_20261022`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_noise_hw_gs`
- ❌ 缺失 `data/debug/hw_groups_20261022/dz_noise_hw_gs/n分组清单`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs`
- ❌ 缺失 `data/debug/hw_groups_20261022/dz_scan_hw_gs/n分组清单`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs`
- ❌ 缺失 `data/debug/hw_groups_20261022/dz_scan_model_gs/n分组清单`
- ✅ 存在 `data/debug/hw_groups_20261022/focus_calibration`
- ❌ 缺失 `data/debug/hw_groups_20261022/manifest.json/n`
- ❌ 缺失 `data/debug/hw_groups_20261022/n曝光`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax11`
- ❌ 缺失 `data/debug/hw_groups_20261022/zernike_farfield_nmax11/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261022/zernike_farfield_nmax11/n变量组`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax15`
- ❌ 缺失 `data/debug/hw_groups_20261022/zernike_farfield_nmax15/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261022/zernike_farfield_nmax15/n变量组`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax20`
- ❌ 缺失 `data/debug/hw_groups_20261022/zernike_farfield_nmax20/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261022/zernike_farfield_nmax20/n变量组`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax25`
- ❌ 缺失 `data/debug/hw_groups_20261022/zernike_farfield_nmax25/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261022/zernike_farfield_nmax25/n变量组`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax3`
- ❌ 缺失 `data/debug/hw_groups_20261022/zernike_farfield_nmax3/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261022/zernike_farfield_nmax3/n变量组`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax4`
- ❌ 缺失 `data/debug/hw_groups_20261022/zernike_farfield_nmax4/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261022/zernike_farfield_nmax4/n变量组`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax6`
- ❌ 缺失 `data/debug/hw_groups_20261022/zernike_farfield_nmax6/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261022/zernike_farfield_nmax6/n变量组`
- ✅ 存在 `data/debug/hw_groups_20261022/zernike_farfield_nmax8`
- ❌ 缺失 `data/debug/hw_groups_20261022/zernike_farfield_nmax8/n分组清单`
- ❌ 缺失 `data/debug/hw_groups_20261022/zernike_farfield_nmax8/n变量组`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax11`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax11/phase_nmax11.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax15`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax15/phase_nmax15.npy`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax15/summary.json/n`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax20`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax20/phase_nmax20.npy`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax20/summary.json/n`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax25`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax25/phase_nmax25.npy`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax25/summary.json/n`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax3`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax3/phase_nmax3.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4/best_phase_nmax4.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4/phase_nmax4.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax6`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax6/phase_nmax6.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax8`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax8/phase_nmax8.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_1`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_2`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_3`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_4`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_5`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_6`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_7`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_7/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_groups_20261022/manifest.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4/best_phase_nmax4.npy`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_groups_20261022/plan.json

- ✅ 存在 `data/debug/hw_groups_20261022`
- ✅ 存在 `data/debug/hw_groups_20261022/focus_calibration`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax11/phase_nmax11.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax15/phase_nmax15.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax20/phase_nmax20.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax25/phase_nmax25.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax3/phase_nmax3.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4/best_phase_nmax4.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4/phase_nmax4.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax6/phase_nmax6.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax8/phase_nmax8.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_1/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_2/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_3/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_4/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_5/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_6/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_7/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_groups_20261022/zernike_farfield_nmax11/run_config.json

- ✅ 存在 `data/debug/hw_groups_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax11/phase_nmax11.npy`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_groups_20261022/zernike_farfield_nmax15/run_config.json

- ✅ 存在 `data/debug/hw_groups_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax15/phase_nmax15.npy`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_groups_20261022/zernike_farfield_nmax20/run_config.json

- ✅ 存在 `data/debug/hw_groups_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax20/phase_nmax20.npy`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_groups_20261022/zernike_farfield_nmax25/run_config.json

- ✅ 存在 `data/debug/hw_groups_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax25/phase_nmax25.npy`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_groups_20261022/zernike_farfield_nmax3/run_config.json

- ✅ 存在 `data/debug/hw_groups_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax3/phase_nmax3.npy`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_groups_20261022/zernike_farfield_nmax4/run_config.json

- ✅ 存在 `data/debug/hw_groups_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4/phase_nmax4.npy`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_groups_20261022/zernike_farfield_nmax6/run_config.json

- ✅ 存在 `data/debug/hw_groups_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax6/phase_nmax6.npy`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_groups_20261022/zernike_farfield_nmax8/run_config.json

- ✅ 存在 `data/debug/hw_groups_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax8/phase_nmax8.npy`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_zernike_sweep_20261022/zernike_order_nmax11/summary.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_7/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_zernike_sweep_20261022/zernike_order_nmax15/summary.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_7/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_zernike_sweep_20261022/zernike_order_nmax20/summary.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_7/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_zernike_sweep_20261022/zernike_order_nmax25/summary.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_7/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_zernike_sweep_20261022/zernike_order_nmax3/summary.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_7/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_zernike_sweep_20261022/zernike_order_nmax4/summary.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_7/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_zernike_sweep_20261022/zernike_order_nmax6/summary.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_7/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_zernike_sweep_20261022/zernike_order_nmax8/summary.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_7/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_zernike_sweep_20261022/zernike_stage_a_1/stage_a.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261022`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_zernike_sweep_20261022/zernike_stage_a_1/summary.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_1/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_zernike_sweep_20261022/zernike_stage_a_2/stage_a.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_1/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_zernike_sweep_20261022/zernike_stage_a_2/summary.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_1/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_2/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_zernike_sweep_20261022/zernike_stage_a_3/stage_a.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_2/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_zernike_sweep_20261022/zernike_stage_a_3/summary.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_2/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_3/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_zernike_sweep_20261022/zernike_stage_a_4/stage_a.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_3/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_zernike_sweep_20261022/zernike_stage_a_4/summary.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_3/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_4/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_zernike_sweep_20261022/zernike_stage_a_5/stage_a.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_4/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_zernike_sweep_20261022/zernike_stage_a_5/summary.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_4/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_5/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_zernike_sweep_20261022/zernike_stage_a_6/stage_a.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_5/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_zernike_sweep_20261022/zernike_stage_a_6/summary.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_5/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_6/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_zernike_sweep_20261022/zernike_stage_a_7/stage_a.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_6/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/artifacts/round3_20261022/hw_zernike_sweep_20261022/zernike_stage_a_7/summary.json

- ✅ 存在 `data/debug/hw_zernike_sweep_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_6/stage_a.npz`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_7/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/hw_defocus_zernike_20261020.md

- ❌ 缺失 `data/debug/bench_settings/focus_probe/slm1_20261020_093932.json`
- ✅ 存在 `data/debug/hw_groups_20261020`
- ✅ 存在 `data/debug/hw_groups_20261020/dz_scan_hw_gs`
- ✅ 存在 `data/debug/hw_groups_20261020/dz_scan_model_gs`
- ✅ 存在 `data/debug/hw_groups_20261020/groups_manifest.json`
- ✅ 存在 `data/debug/hw_groups_20261020/plan_all.json`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax11.npy`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax15.npy`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax20.npy`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax25.npy`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax6.npy`
- ✅ 存在 `data/debug/hw_groups_20261020/zernike_farfield/phases/phase_nmax8.npy`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261020b`

### report/shape_trials/hw_defocus_zernike/hw_defocus_zernike_20261021.md

- ✅ 存在 `data/debug/hw_groups_20261021`
- ✅ 存在 `data/debug/hw_groups_20261021/dz_scan_hw_gs`
- ✅ 存在 `data/debug/hw_groups_20261021/dz_scan_model_gs`
- ✅ 存在 `data/debug/hw_groups_20261021/focus_calibration`
- ✅ 存在 `data/debug/hw_groups_20261021/focus_calibration/slm1_20261021_154749.json`
- ✅ 存在 `data/debug/hw_groups_20261021/groups_manifest.json`
- ✅ 存在 `data/debug/hw_groups_20261021/plan.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax11/phase_nmax11.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax15/phase_nmax15.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax20/phase_nmax20.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax25/phase_nmax25.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax3/phase_nmax3.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax4/best_phase_nmax4.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax4/phase_nmax4.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax6/phase_nmax6.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_order_nmax8/phase_nmax8.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_1/stage_a.npz`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_2/stage_a.npz`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_3/stage_a.npz`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_4/stage_a.npz`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_5/stage_a.npz`
- ❌ 缺失 `data/debug/hw_zernike_sweep_20261021/zernike_stage_a_6/stage_a.npz`

### report/shape_trials/hw_defocus_zernike/hw_defocus_zernike_20261022.md

- ✅ 存在 `data/debug/_quarantine_20261022_mixed/hw_zernike_sweep_20261022/zernike_order_nmax3/summary.json`
- ✅ 存在 `data/debug/hw_groups_20261022`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_hw_gs`
- ✅ 存在 `data/debug/hw_groups_20261022/dz_scan_model_gs`
- ✅ 存在 `data/debug/hw_groups_20261022/focus_calibration`
- ✅ 存在 `data/debug/hw_groups_20261022/focus_calibration/slm1_20261022_142207.json`
- ✅ 存在 `data/debug/hw_groups_20261022/groups_manifest.json`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax11/phase_nmax11.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax15/phase_nmax15.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax20/phase_nmax20.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax25/phase_nmax25.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax3/phase_nmax3.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4/best_phase_nmax4.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4/phase_nmax4.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax6/phase_nmax6.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax8/phase_nmax8.npy`
- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_stage_a_7/stage_a.npz`

### report/shape_trials/hw_gs_square_scans/artifacts/20261023_atomic_full/frames_selection.json

- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/blaze/blaze_128/frame_000.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/blaze/blaze_16/frame_000.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/blaze/blaze_32/frame_000.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/blaze/blaze_64/frame_000.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_1000/frame_000.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_1200/frame_000.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_1500/frame_000.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_1700/frame_000.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_200/frame_000.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_600/frame_000.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_12/frame_000.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_19/frame_000.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_30/frame_000.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_7/frame_000.npz`

### report/shape_trials/hw_gs_square_scans/artifacts/20261023_atomic_full/PROVENANCE.json

- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/blaze/blaze_128/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/blaze/blaze_16/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/blaze/blaze_24/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/blaze/blaze_32/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/blaze/blaze_40/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/blaze/blaze_48/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/blaze/blaze_64/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/blaze/blaze_96/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/blaze/measurements.csv`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/blaze/measurements.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/blaze/run_config.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/manifest.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/measurements.csv`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/measurements.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/run_config.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/selected.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_100/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_1000/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_1100/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_1200/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_1300/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_1400/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_1500/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_1600/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_1700/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_200/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_300/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_400/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_500/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_600/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_700/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_800/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_900/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/measurements.csv`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/measurements.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_10/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_11/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_12/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_13/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_14/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_15/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_16/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_17/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_18/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_19/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_20/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_21/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_22/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_23/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_24/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_25/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_26/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_27/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_28/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_29/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_30/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_7/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_8/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/nmax_9/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/zernike/run_config.json`

### report/shape_trials/hw_gs_square_scans/artifacts/20261023_gs_square/blaze/run_config.json

- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square`

### report/shape_trials/hw_gs_square_scans/artifacts/20261023_gs_square/frames_selection.json

- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/blaze/frame_000.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/blaze/frame_002.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/blaze/frame_005.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/blaze/frame_007.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/sides/frame_001.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/sides/frame_005.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/sides/frame_008.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/sides/frame_009.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/sides/frame_011.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/sides/frame_014.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/sides/frame_017.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/zernike/frame_000.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/zernike/frame_011.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/zernike/frame_018.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/zernike/frame_023.npz`

### report/shape_trials/hw_gs_square_scans/artifacts/20261023_gs_square/PROVENANCE.json

- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/blaze/measurements.csv`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/blaze/measurements.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/blaze/run_config.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/manifest.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/sides/measurements.csv`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/sides/measurements.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/sides/run_config.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/zernike/measurements.csv`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/zernike/measurements.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/zernike/run_config.json`

### report/shape_trials/hw_gs_square_scans/artifacts/20261023_gs_square/sides/run_config.json

- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square`

### report/shape_trials/hw_gs_square_scans/artifacts/20261023_gs_square/zernike/run_config.json

- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square`

### report/shape_trials/hw_gs_square_scans/artifacts/20261023_retest_frozen/PROVENANCE.json

- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_frozen`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_frozen/blaze/blaze_32/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_frozen/blaze/blaze_40/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_frozen/blaze/blaze_48/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_frozen/blaze/measurements.csv`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_frozen/blaze/measurements.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_frozen/blaze/run_config.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_frozen/manifest.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_frozen/zernike/measurements.csv`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_frozen/zernike/measurements.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_frozen/zernike/nmax_17/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_frozen/zernike/nmax_18/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_frozen/zernike/nmax_19/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_frozen/zernike/run_config.json`

### report/shape_trials/hw_gs_square_scans/artifacts/20261023_retest_power_changed/PROVENANCE.json

- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_power_changed`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_power_changed/blaze/blaze_32/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_power_changed/blaze/blaze_40/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_power_changed/blaze/blaze_48/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_power_changed/blaze/measurements.csv`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_power_changed/blaze/measurements.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_power_changed/manifest.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_power_changed/zernike/measurements.csv`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_power_changed/zernike/measurements.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_power_changed/zernike/nmax_17/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_power_changed/zernike/nmax_18/metrics.json`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_power_changed/zernike/nmax_19/metrics.json`

### report/shape_trials/hw_gs_square_scans/hw_gs_square_scans_20261023.md

- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_atomic_full/sides/side_1200/pupil_phase.npy`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/blaze/frame_000.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/blaze/frame_002.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/blaze/frame_005.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/blaze/frame_007.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/sides/frame_001.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/sides/frame_005.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/sides/frame_008.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/sides/frame_009.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/sides/frame_011.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/sides/frame_014.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/sides/frame_017.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/zernike/frame_000.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/zernike/frame_011.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/zernike/frame_018.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_gs_square/zernike/frame_023.npz`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_frozen`
- ✅ 存在 `data/debug/hw_gs_square_scans/20261023_retest_power_changed`

### report/shape_trials/model_in_loop/model_in_loop_hw_ab_20261021.md

- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_164549/baseline_a/stage.log`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_164549/manifest.json`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_170435/baseline_a/stage.log`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_170646/baseline_a/slm_model_in_loop/20261021_170726/panel_scan/summary.json`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_170646/baseline_a/stage.log`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_170646/manifest.json`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_172236`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_172236/large_1000/stage.log`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_172236/manifest.json`

### report/shape_trials/model_in_loop/model_in_loop_live_20261013.md

- ✅ 存在 `data/debug`
- ❌ 缺失 `data/debug/slm_disturbance_compare/20261013_165039`
- ❌ 缺失 `data/debug/slm_disturbance_compare/20261013_165039/disturbance.json`
- ❌ 缺失 `data/debug/slm_exposure_analysis/20261013_1935/analysis_pixelwise.json`
- ❌ 缺失 `data/debug/slm_exposure_analysis/20261013_1935/analysis_with_repeat.json`
- ❌ 缺失 `data/debug/slm_model_compare/20261013_1943_seed_20261013`
- ❌ 缺失 `data/debug/slm_model_compare/20261013_2015_new_session/phase_bandwidth_paired.csv`
- ❌ 缺失 `data/debug/slm_model_compare/20261013_2015_new_session/psf_paired.csv`
- ❌ 缺失 `data/debug/slm_model_compare/20261013_phase_bandwidth_trial/paired_holdouts.csv`
- ❌ 缺失 `data/debug/slm_model_compare/20261013_psf_trial/paired_observable_holdouts.csv`
- ✅ 存在 `data/debug/slm_model_in_loop`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_105911/bench_geometry.json`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_110921`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_111258`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_112034`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_113152`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_114650`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_142415`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_152341`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_152341/panel_scan`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_152831`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_173039`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_173230`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_174628`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_174740`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_201100`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_201229`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_202804`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_203426`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_203553`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_203702`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_203905`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_204410`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_210403/preflight`
- ❌ 缺失 `data/debug/slm_model_in_loop/<时间戳`
- ✅ 存在 `data/debug/slm_model_in_loop/exposure_drift_20261013_1547`
- ✅ 存在 `data/debug/slm_model_in_loop/exposure_flat_20261013_191614`
- ✅ 存在 `data/debug/slm_model_in_loop/panel_scan_20261013_2003`
- ✅ 存在 `data/debug/slm_model_in_loop/panel_scan_fine_20261013_2005`
- ✅ 存在 `data/debug/slm_model_in_loop/panel_scan_vertical_20261013_2007`
- ✅ 存在 `data/debug/slm_model_in_loop/short_exposure_20261013_1553`
- ✅ 存在 `data/debug/slm_model_in_loop/short_exposure_shaped_20261013_1558`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261013_154143`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261013_160630_fixed040`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261013_1733_four_vs_two`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261013_1748_disturbed_four_vs_two`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261013_2013_warm_vs_cold`

### report/shape_trials/model_in_loop/model_in_loop_live_20261014.md

- ✅ 存在 `data/debug`
- ❌ 缺失 `data/debug/slm_disturbance_compare/20261013_165039/disturbance_panel_rad.npy`
- ❌ 缺失 `data/debug/slm_model_compare/20261014_5shape_seed_20261013_verified`
- ✅ 存在 `data/debug/slm_model_in_loop`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261014_refit_analysis/paired_holdouts.json`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261014_1115_zero_vs_refit`
- ❌ 缺失 `data/debug/worktrees/model-in-loop-20261014`

### report/shape_trials/model_in_loop/model_in_loop_live_20261015.md

- ✅ 存在 `data/debug`
- ❌ 缺失 `data/debug/slm_flat_stability/20261015_152929`
- ❌ 缺失 `data/debug/slm_panel_locate/20261015_152101`

### report/shape_trials/model_in_loop/model_in_loop_live_20261016.md

- ❌ 缺失 `data/debug/slm_flat_stability/20261016_model_in_loop_preflight`
- ✅ 存在 `data/debug/slm_model_in_loop`

### report/shape_trials/model_in_loop/model_in_loop_live_20261019.md

- ❌ 缺失 `data/debug/alignment_probe/20261019_focus_coma`
- ❌ 缺失 `data/debug/alignment_probe/20261019_focus_coma_verify`
- ❌ 缺失 `data/debug/alignment_probe/20261019_realign`
- ❌ 缺失 `data/debug/model_in_loop_auto/20261019_camera_reposition`
- ❌ 缺失 `data/debug/model_in_loop_hw_runbook/20261019_corrected_geometry`
- ✅ 存在 `data/debug/model_in_loop_hw_sweep/20261019_105952`
- ✅ 存在 `data/debug/slm_model_in_loop`
- ✅ 存在 `data/debug/slm_model_in_loop/20261019_094121`
- ✅ 存在 `data/debug/slm_model_in_loop/20261019_094121/panel_scan`
- ✅ 存在 `data/debug/slm_model_in_loop/20261019_100334`
- ✅ 存在 `data/debug/slm_model_in_loop/20261019_103208`
- ✅ 存在 `data/debug/slm_model_in_loop/20261019_110323`
- ✅ 存在 `data/debug/slm_model_in_loop/20261019_111941`

### report/shape_trials/model_in_loop/model_in_loop_next_steps_20261013.md

- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_114650`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_210403/preflight`
- ❌ 缺失 `data/debug/slm_model_in_loop/<时间戳`

### report/shape_trials/model_in_loop/model_in_loop_pause_handoff_20261013.md

- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_114650`
- ❌ 缺失 `data/debug/slm_model_in_loop/<时间戳`

### report/shape_trials/model_in_loop/model_in_loop_record_20261021_172309.md

- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_172236/baseline_a/slm_model_in_loop/20261021_172309`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_172236/baseline_a/slm_model_in_loop/20261021_172309/bench_geometry.json`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_172236/baseline_a/slm_model_in_loop/20261021_172309/best_phase.npy`
- ❌ 缺失 `data/debug/model_in_loop_hw_ab/20261021_172236/baseline_a/slm_model_in_loop/20261021_172309/failure.json`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_172236/baseline_a/slm_model_in_loop/20261021_172309/fitted_coefficients.npy`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_172236/baseline_a/slm_model_in_loop/20261021_172309/handoff_summary.json`
- ❌ 缺失 `data/debug/model_in_loop_hw_ab/20261021_172236/baseline_a/slm_model_in_loop/20261021_172309/model_error_terms.json`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_172236/baseline_a/slm_model_in_loop/20261021_172309/preflight/flat.npz`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_172236/baseline_a/slm_model_in_loop/20261021_172309/probes/manifest.json`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_172236/baseline_a/slm_model_in_loop/20261021_172309/run_config.json`
- ✅ 存在 `data/debug/model_in_loop_hw_ab/20261021_172236/baseline_a/slm_model_in_loop/20261021_172309/source_amplitude.npy`

### report/shape_trials/model_in_loop/model_in_loop_uniformity_live_20261016.md

- ❌ 缺失 `data/debug/beam_extent`
- ✅ 存在 `data/debug/bench_settings`
- ❌ 缺失 `data/debug/bench_settings/20261017_ladder_diag.log`
- ✅ 存在 `data/debug/bench_settings/exposure_probe/20261024_110551.json`
- ✅ 存在 `data/debug/slm_bench_calibration/20261026_091101/panel_scan`
- ✅ 存在 `data/debug/slm_model_in_loop`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261016_173002`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261016_174330`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261016_1745_soft_edge_abba`

### report/shape_trials/model_in_loop/physical_forward_defocus_live_20261022.md

- ✅ 存在 `data/debug/physical_forward_defocus`

### report/shape_trials/model_in_loop/physical_forward_sampling_20261022_r2.md

- ✅ 存在 `data/debug/physical_forward_defocus`
- ✅ 存在 `data/debug/physical_forward_defocus/sampling_followup_evidence_20261022.json`

### report/shape_trials/model_in_loop/uncertainty_probe_hw_20261016.md

- ❌ 缺失 `data/debug/uncertainty_probes_hw`
- ❌ 缺失 `data/debug/uncertainty_probes_sim`
- ❌ 缺失 `data/debug/uncertainty_selector_sweep`

### report/shape_trials/multishape_capability/artifacts/no_gs_20261024/anchored_cv_config.json

- ✅ 存在 `data/debug/slm_transfer/live_no_gs_20261024_0926/anchored`

### report/shape_trials/multishape_capability/artifacts/no_gs_20261024/anchored_cv_direction.json

- ✅ 存在 `data/debug/slm_transfer/live_no_gs_cv_anchored_20261024`

### report/shape_trials/multishape_capability/artifacts/no_gs_20261024/combined.json

- ✅ 存在 `data/debug/slm_transfer/live_no_gs_20261024_0926/anchored`
- ✅ 存在 `data/debug/slm_transfer/live_no_gs_cv_20261024`
- ✅ 存在 `data/debug/slm_transfer/live_no_gs_cv_anchored_20261024`

### report/shape_trials/multishape_capability/artifacts/no_gs_20261024/initial_direction_recovered.json

- ✅ 存在 `data/debug/slm_transfer/live_no_gs_20261024_0926/anchored`

### report/shape_trials/multishape_capability/artifacts/reproduction_20261023/figure_data.json

- ✅ 存在 `data/debug/calibrate_transfer_blur/shapes_20261023_1956/frozen_parameters.json`
- ✅ 存在 `data/debug/fit_transfer_multishape/exact_grid_20261023_1935/iterations.pkl`
- ✅ 存在 `data/debug/fit_transfer_multishape/exact_grid_20261023_1935/result.json`
- ✅ 存在 `data/debug/validate_transfer_shapes/anchored_20261023_2002/manifest.json`
- ✅ 存在 `data/debug/validate_transfer_shapes/anchored_20261023_2002/summary.json`

### report/shape_trials/multishape_capability/capability_rectangle_20261023_191444.md

- ✅ 存在 `data/debug/multishape_forward/rectangle08_20261023_191444`

### report/shape_trials/multishape_capability/capability_square_20261023_182153.md

- ✅ 存在 `data/debug/multishape_forward/square_20261023_182153`

### report/shape_trials/multishape_capability/capability_star_20261023_191213.md

- ✅ 存在 `data/debug/multishape_forward/star08_20261023_191213`

### report/shape_trials/multishape_capability/capability_triangle_20261023_191105.md

- ✅ 存在 `data/debug/multishape_forward/triangle08_20261023_191105`

### report/shape_trials/multishape_capability/diagnosis_and_transfer_20261023.md

- ✅ 存在 `data/debug/analyze_transfer_direction/anchored_20261023_2002`
- ✅ 存在 `data/debug/analyze_transfer_response/ladder_20261023_1952`
- ✅ 存在 `data/debug/calibrate_transfer_blur/shapes_20261023_1956`
- ✅ 存在 `data/debug/capture_transfer_response/ladder_20261023_1952`
- ✅ 存在 `data/debug/fit_transfer_multishape/current_20261023_1933`
- ✅ 存在 `data/debug/fit_transfer_multishape/exact_grid_20261023_1935`
- ✅ 存在 `data/debug/fit_transfer_multishape/history_20261023_1925`
- ✅ 存在 `data/debug/fit_transfer_multishape/history_20261023_1935_r2`
- ✅ 存在 `data/debug/multishape_forward/square_20261023_181846`
- ❌ 缺失 `data/debug/transfer_multishape_experiment/capture_20261023_1905`
- ✅ 存在 `data/debug/transfer_multishape_experiment/capture_20261023_1938`
- ✅ 存在 `data/debug/validate_transfer_shapes/anchored_20261023_2002`
- ✅ 存在 `data/debug/validate_transfer_shapes/anchored_20261023_2002/artifact_audit.json`
- ✅ 存在 `data/debug/validate_transfer_shapes/paired_20261023_1942`

### report/shape_trials/multishape_capability/physical_multishape_reproduction_20261023.md

- ✅ 存在 `data/debug`
- ✅ 存在 `data/debug/analyze_transfer_direction/anchored_20261023_2002`
- ✅ 存在 `data/debug/analyze_transfer_response/ladder_20261023_1952`
- ✅ 存在 `data/debug/calibrate_transfer_blur/shapes_20261023_1956`
- ✅ 存在 `data/debug/calibrate_transfer_blur/shapes_20261023_1956/frozen_parameters.json`
- ✅ 存在 `data/debug/capture_transfer_response/ladder_20261023_1952`
- ✅ 存在 `data/debug/fit_transfer_multishape/current_20261023_1933/frozen_parameters.json`
- ✅ 存在 `data/debug/fit_transfer_multishape/exact_grid_20261023_1935`
- ❌ 缺失 `data/debug/slm_transfer/$(Get-Date`
- ✅ 存在 `data/debug/slm_transfer/20261026_094555`
- ✅ 存在 `data/debug/slm_transfer/live_no_gs_20261024_0926`
- ✅ 存在 `data/debug/slm_transfer/live_no_gs_20261024_0926/anchored`
- ✅ 存在 `data/debug/slm_transfer/live_no_gs_20261024_0926/fit/frozen_parameters.json`
- ✅ 存在 `data/debug/slm_transfer/live_no_gs_cv_20261024`
- ✅ 存在 `data/debug/slm_transfer/live_no_gs_cv_anchored_20261024`
- ✅ 存在 `data/debug/slm_transfer/live_no_gs_cv_anchored_20261024_analysis`
- ❌ 缺失 `data/debug/slm_transfer/new_synthetic_session`
- ✅ 存在 `data/debug/slm_transfer/synthetic_20261024_shapes`
- ✅ 存在 `data/debug/slm_transfer/synthetic_20261024_shift`
- ❌ 缺失 `data/debug/three_shape_reproduction/$tag`
- ✅ 存在 `data/debug/three_shape_reproduction/offline_bootstrap_20261023`
- ✅ 存在 `data/debug/transfer_integration/live_20261023`
- ✅ 存在 `data/debug/transfer_integration/replay_20261023`
- ✅ 存在 `data/debug/transfer_multishape_experiment/capture_20261023_1938`
- ✅ 存在 `data/debug/validate_transfer_shapes/anchored_20261023_2002`
- ✅ 存在 `data/debug/validate_transfer_shapes/paired_20261023_1942`

### report/shape_trials/multishape_capability/shape_comparison_20261023.md

- ✅ 存在 `data/debug/multishape_forward`
- ✅ 存在 `data/debug/multishape_forward/rectangle08_20261023_191444`
- ✅ 存在 `data/debug/multishape_forward/square_20261023_182153`
- ✅ 存在 `data/debug/multishape_forward/star08_20261023_191213`
- ✅ 存在 `data/debug/multishape_forward/triangle08_20261023_191105`
- ✅ 存在 `data/debug/physical_forward_defocus/capture_after_focus_20261023_145446/seed_probe000.npy`

### report/shape_trials/multishape_scale/artifacts/plan_20261028.json

- ✅ 存在 `data/debug/slm_transfer/carrier_probe_20261028/carrier_probe.json`
- ✅ 存在 `data/debug/slm_transfer/multishape_scale_20261028`
- ✅ 存在 `data/debug/slm_transfer/multishape_scale_20261028/s1200`
- ❌ 缺失 `data/debug/slm_transfer/multishape_scale_20261028/s200`
- ✅ 存在 `data/debug/slm_transfer/multishape_scale_20261028/s600`
- ✅ 存在 `data/debug/slm_transfer/multishape_scale_20261028/zero_ref`
- ✅ 存在 `data/debug/slm_transfer/no_gs_ccd_moved_20261027_2106/model640/frozen_parameters.json`

### report/shape_trials/multishape_scale/artifacts/star_stage_traces_20261029.json

- ✅ 存在 `data/debug/slm_transfer/multishape_nogs_20261029/p20261029_strict_cold_s200/star`
- ✅ 存在 `data/debug/slm_transfer/multishape_nogs_20261029/p20261029_strict_cold_s200/star/shape/star/mono12_continue/recording/process.csv`
- ✅ 存在 `data/debug/slm_transfer/multishape_nogs_20261029/p20261029_strict_cold_s200/star/shape/star/mono12_cv/recording/process.csv`
- ✅ 存在 `data/debug/slm_transfer/multishape_nogs_20261029/p20261029_strict_cold_s200/star/shape/star/mono12_dense/recording/process.csv`
- ✅ 存在 `data/debug/slm_transfer/multishape_nogs_20261029/p20261029_strict_cold_s200/star/shape/star/transport_cv/recording/process.csv`

### report/shape_trials/multishape_scale/live_20261029_device_session.md

- ✅ 存在 `data/debug/slm_transfer/multishape_nogs_20261029`
- ✅ 存在 `data/debug/slm_transfer/multishape_nogs_20261029/lanes`
- ✅ 存在 `data/debug/slm_transfer/multishape_nogs_20261029/p20261029_analytic_init_s600/circle_powerA_20261029`

### report/shape_trials/multishape_scale/multishape_scale_analysis_20261028.md

- ✅ 存在 `data/debug/slm_transfer/multishape_scale_20261028/s1200`

### report/shape_trials/multishape_scale/multishape_scale_live_20261028.md

- ✅ 存在 `data/debug/optical_path_probe/retry_20261029/20261029_013227Z/optical_path_probe.json`
- ✅ 存在 `data/debug/slm_transfer/multishape_nogs_20261029`
- ✅ 存在 `data/debug/slm_transfer/multishape_nogs_20261029/p20261029`
- ✅ 存在 `data/debug/slm_transfer/multishape_scale_20261028/s1200`

### report/shape_trials/multishape_scale/multishape_scale_planning_20261028.md

- ✅ 存在 `data/debug/slm_transfer/multishape_scale_20261028`
- ✅ 存在 `data/debug/slm_transfer/multishape_scale_20261028/s1200`
- ❌ 缺失 `data/debug/slm_transfer/multishape_scale_20261028/s200`
- ✅ 存在 `data/debug/slm_transfer/multishape_scale_20261028/s600`

### report/shape_trials/multishape_scale/multishape_scale_wallclock_20261028.md

- ✅ 存在 `data/debug/optical_path_probe/20261028_083057Z`
- ✅ 存在 `data/debug/slm_transfer/multishape_scale_20261028/r2_next`

### report/shape_trials/multishape_scale/square_1000_diff_abba_20261030.md

- ✅ 存在 `data/debug`
- ✅ 存在 `data/debug/slm_transfer/multishape_nogs_20261030`
- ❌ 缺失 `data/debug/slm_transfer/multishape_nogs_20261030/axial_conditional_NEW`
- ❌ 缺失 `data/debug/slm_transfer/multishape_nogs_20261030/square1000_diff_experiment_NEW`
- ❌ 缺失 `data/debug/slm_transfer/multishape_nogs_20261030/square1000_diff_preview_NEW`
- ✅ 存在 `data/debug/slm_transfer/multishape_nogs_20261030/square1000_model_diagnostic_20261030_v7/defect_sim`

### report/shape_trials/no_gs_ccd_moved/analysis_20261027.md

- ✅ 存在 `data/debug`
- ✅ 存在 `data/debug/slm_transfer/no_gs_ccd_moved_20261027_2106`
- ✅ 存在 `data/debug/slm_transfer/no_gs_ccd_moved_20261027_2106/local_shape1200`

### report/shape_trials/no_gs_ccd_moved/live_20261027.md

- ✅ 存在 `data/debug/slm_transfer/no_gs_ccd_moved_20261027_2106`

### report/shape_trials/physical_forward_defocus_plan_20261022.md

- ❌ 缺失 `data/debug/physical_forward_defocus/<YYYYMMDD_HHMMSS`

### report/shape_trials/physical_forward_no_gs_plan_20261027.md

- ❌ 缺失 `data/debug/slm_transfer/no_gs_next`

### report/shape_trials/sim_pretrain_transfer/sim_pretrain_transfer_20261024.md

- ✅ 存在 `data/debug/calibrate_transfer_blur/shapes_20261023_1956/frozen_parameters.json`
- ❌ 缺失 `data/debug/slm_transfer/pretrain`
- ✅ 存在 `data/debug/transfer_repro_no_gs/block0`
- ✅ 存在 `data/debug/transfer_repro_no_gs/block1`
- ❌ 缺失 `data/debug/transfer_repro_no_gs/block{0`
- ✅ 存在 `data/debug/validate_transfer_shapes/anchored_20261023_2002`
- ✅ 存在 `data/debug/validate_transfer_shapes/paired_20261023_1942/manifest.json`

### report/shape_trials/sim_vs_hw/artifacts/run_config_20261023_015046.json

- ❌ 缺失 `data/calib.npz`
- ❌ 缺失 `data/debug/differentiable_opt/rep_1_20261023_095038`

### report/shape_trials/sim_vs_hw/artifacts/run_config_20261023_015046b.json

- ❌ 缺失 `data/calib.npz`
- ❌ 缺失 `data/debug/differentiable_opt/rep_0_20261023_095038`

### report/shape_trials/sim_vs_hw/artifacts/run_config_20261023_020125.json

- ❌ 缺失 `data/calib.npz`
- ❌ 缺失 `data/debug/differentiable_opt/rep_2_20261023_095038`

### report/shape_trials/sim_vs_hw/artifacts/sim_vs_hw_20261022.md

- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4/best_phase_nmax4.npy`
- ✅ 存在 `data/debug/sim_vs_hw_20261022`

### report/shape_trials/sim_vs_hw/artifacts/sim_vs_hw_20261023.md

- ✅ 存在 `data/debug/hw_zernike_sweep_20261022/zernike_order_nmax4/best_phase_nmax4.npy`
- ✅ 存在 `data/debug/sim_vs_hw_20261023_v2`

### report/shape_trials/sim_vs_hw/compare_v1_v2_20261023.md

- ✅ 存在 `data/debug/sim_vs_hw_20261023_v2`

### report/shape_trials/sim_vs_hw/sim_vs_hw_20261022.md

- ✅ 存在 `data/debug/sim_vs_hw_20261022`

### report/shape_trials/sim_vs_hw/sim_vs_hw_20261023.md

- ✅ 存在 `data/debug/sim_vs_hw_20261023_v2`

### report/shape_trials/slides_ppt_20261023.json

- ✅ 存在 `data/debug/differentiable_opt`

### report/shape_trials/square1200/artifacts/20261026/results.json

- ✅ 存在 `data/debug/slm_transfer/square1200_20261026_1350/mono12_continue`
- ✅ 存在 `data/debug/slm_transfer/square1200_20261026_1350/mono12_cv`
- ✅ 存在 `data/debug/slm_transfer/square1200_20261026_1350/transport_cv_r2`

### report/shape_trials/square1200/square1200_20261026.md

- ❌ 缺失 `data/debug/slm_transfer/square1200_new_session`

### report/shape_trials/square1200_exploration/artifacts/20261026/evidence.json

- ✅ 存在 `data/debug/slm_transfer/square1200_20261026_1350/mono12_dense`

### report/shape_trials/square1200_exploration/exploration_20261026.md

- ✅ 存在 `data/debug/slm_transfer/square1200_explore_20261026_1435`

### report/shape_trials/transfer_square1200/transfer_square1200_20261027.md

- ✅ 存在 `data/debug/slm_transfer/transfer_square1200_20261027_r2`

### report/shape_trials/tri_plane_20261027/tri_plane_20261027.md

- ✅ 存在 `data/debug/slm_transfer/transfer_plane_20261027`
- ✅ 存在 `data/debug/slm_transfer/transfer_tri_star_20261027`

### report/shape_trials/xy_profile_manifest.json

- ✅ 存在 `data/debug/slm_gs_refine/20261014_100441/replay/00_flat.npy`
- ✅ 存在 `data/debug/slm_gs_refine/20261014_100441/replay/02_best.npy`
- ✅ 存在 `data/debug/slm_gs_refine/20261014_143035/replay/00_flat.npy`
- ✅ 存在 `data/debug/slm_gs_refine/20261014_143035/replay/02_best.npy`
- ✅ 存在 `data/debug/slm_gs_refine/20261015_205654/best_far_field.png`
- ✅ 存在 `data/debug/slm_gs_refine/20261015_205654/slm_gs_refine_20261015_205654.pkl`
- ✅ 存在 `data/debug/slm_gs_refine/20261015_205848/best_far_field.png`
- ✅ 存在 `data/debug/slm_gs_refine/20261015_205848/slm_gs_refine_20261015_205848.pkl`
- ✅ 存在 `data/debug/slm_gs_refine/20261015_210048/best_far_field.png`
- ✅ 存在 `data/debug/slm_gs_refine/20261015_210048/slm_gs_refine_20261015_210048.pkl`
- ✅ 存在 `data/debug/slm_gs_refine/20261015_210155/best_far_field.png`
- ✅ 存在 `data/debug/slm_gs_refine/20261015_210155/slm_gs_refine_20261015_210155.pkl`
- ✅ 存在 `data/debug/slm_gs_refine/20261015_211345/best_far_field.png`
- ✅ 存在 `data/debug/slm_gs_refine/20261015_211345/slm_gs_refine_20261015_211345.pkl`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_114650/best_far_field.png`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_114650/probes/round_01_shaped_00.npz`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_145819/best_far_field.png`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_145819/probes/round_01_shaped_00.npz`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_152341/best_far_field.png`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261013_152341/probes/round_01_shaped_00.npz`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261014_105041/probes/round_01_shaped_00.npz`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261014_105205/probes/round_01_shaped_00.npz`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261014_105339/probes/round_01_shaped_00.npz`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261014_105438/probes/round_01_shaped_00.npz`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261014_105535/probes/round_01_shaped_00.npz`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261014_110203/probes/round_01_shaped_00.npz`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261014_110305/probes/round_01_shaped_00.npz`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261014_170820/probes/round_01_shaped_00.npz`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261014_171043/probes/round_01_shaped_00.npz`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261014_171219/probes/round_01_shaped_00.npz`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261014_171301/probes/round_01_shaped_00.npz`
- ❌ 缺失 `data/debug/slm_model_in_loop/20261014_171340/probes/round_01_shaped_00.npz`
- ✅ 存在 `data/debug/slm_model_in_loop/20261019_100334/preflight/flat.npz`
- ✅ 存在 `data/debug/slm_model_in_loop/20261019_103208/preflight/flat.npz`
- ✅ 存在 `data/debug/slm_model_in_loop/20261019_103259/preflight/flat.npz`
- ✅ 存在 `data/debug/slm_model_in_loop/20261019_103347/preflight/flat.npz`
- ✅ 存在 `data/debug/slm_model_in_loop/20261019_110323/preflight/flat.npz`
- ✅ 存在 `data/debug/slm_model_in_loop/20261019_111359/preflight/flat.npz`
- ✅ 存在 `data/debug/slm_model_in_loop/20261019_111941/preflight/flat.npz`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261014_1115_zero_vs_refit/01_A.npy`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261014_1115_zero_vs_refit/02_B.npy`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261014_1115_zero_vs_refit/04_B.npy`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261014_1115_zero_vs_refit/05_A.npy`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261016_1044_uniformity_weights/01_A.npy`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261016_1044_uniformity_weights/02_B.npy`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261016_1044_uniformity_weights/04_B.npy`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261016_1044_uniformity_weights/05_A.npy`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261016_1051_refit_steps/01_A.npy`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261016_1051_refit_steps/02_B.npy`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261016_1051_refit_steps/04_B.npy`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261016_1051_refit_steps/05_A.npy`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261016_1053_refit_steps_reverse/01_A.npy`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261016_1053_refit_steps_reverse/02_B.npy`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261016_1053_refit_steps_reverse/04_B.npy`
- ❌ 缺失 `data/debug/slm_model_in_loop_replay/20261016_1053_refit_steps_reverse/05_A.npy`

### report/sim2f_bench/alignment_20261027/report.md

- ✅ 存在 `data/debug/model_in_loop_hw_collect`

### report/sim2f_bench/optical_scale_20261027/artifacts/metrics.json

- ✅ 存在 `data/debug/model_in_loop_hw_collect`
- ❌ 缺失 `data/debug/model_in_loop_hw_collect/20261025`

### report/sim2f_bench/optical_scale_20261027/report.md

- ✅ 存在 `data/debug`
- ❌ 缺失 `data/debug/model_in_loop_hw_collect/20261025`

### report/sim2f_bench/twin_vs_hw_corpus/20261027/artifacts/metrics.json

- ✅ 存在 `data/debug/model_in_loop_hw_collect`
- ❌ 缺失 `data/debug/model_in_loop_hw_collect/20261025`

### report/sim2f_bench/twin_vs_hw_corpus/20261027/artifacts/per_record.json

- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123031/model_in_loop_hw_collect_20261025_123031.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123037/model_in_loop_hw_collect_20261025_123037.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123042/model_in_loop_hw_collect_20261025_123042.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123048/model_in_loop_hw_collect_20261025_123048.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123116/model_in_loop_hw_collect_20261025_123116.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123151/model_in_loop_hw_collect_20261025_123151.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123157/model_in_loop_hw_collect_20261025_123157.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123202/model_in_loop_hw_collect_20261025_123202.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123208/model_in_loop_hw_collect_20261025_123208.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123738/model_in_loop_hw_collect_20261025_123738.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123745/model_in_loop_hw_collect_20261025_123745.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123750/model_in_loop_hw_collect_20261025_123750.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123756/model_in_loop_hw_collect_20261025_123756.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123826/model_in_loop_hw_collect_20261025_123826.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123902/model_in_loop_hw_collect_20261025_123902.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123908/model_in_loop_hw_collect_20261025_123908.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123913/model_in_loop_hw_collect_20261025_123913.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123919/model_in_loop_hw_collect_20261025_123919.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_150515/model_in_loop_hw_collect_20261025_150515.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_150519/model_in_loop_hw_collect_20261025_150519.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_150523/model_in_loop_hw_collect_20261025_150523.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_150528/model_in_loop_hw_collect_20261025_150528.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_150550/model_in_loop_hw_collect_20261025_150550.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_150621/model_in_loop_hw_collect_20261025_150621.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_150626/model_in_loop_hw_collect_20261025_150626.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_150630/model_in_loop_hw_collect_20261025_150630.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_150635/model_in_loop_hw_collect_20261025_150635.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_151210/model_in_loop_hw_collect_20261025_151210.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_151214/model_in_loop_hw_collect_20261025_151214.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_151218/model_in_loop_hw_collect_20261025_151218.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_151223/model_in_loop_hw_collect_20261025_151223.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_151245/model_in_loop_hw_collect_20261025_151245.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_151316/model_in_loop_hw_collect_20261025_151316.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_151321/model_in_loop_hw_collect_20261025_151321.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_151325/model_in_loop_hw_collect_20261025_151325.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_151329/model_in_loop_hw_collect_20261025_151329.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_171033/model_in_loop_hw_collect_20261025_171033.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_171036/model_in_loop_hw_collect_20261025_171036.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_171040/model_in_loop_hw_collect_20261025_171040.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_171044/model_in_loop_hw_collect_20261025_171044.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_171106/model_in_loop_hw_collect_20261025_171106.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_171135/model_in_loop_hw_collect_20261025_171135.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_171139/model_in_loop_hw_collect_20261025_171139.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_171143/model_in_loop_hw_collect_20261025_171143.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_171148/model_in_loop_hw_collect_20261025_171148.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_172907/model_in_loop_hw_collect_20261025_172907.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_172911/model_in_loop_hw_collect_20261025_172911.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_172915/model_in_loop_hw_collect_20261025_172915.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_172920/model_in_loop_hw_collect_20261025_172920.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_172942/model_in_loop_hw_collect_20261025_172942.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_173012/model_in_loop_hw_collect_20261025_173012.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_173017/model_in_loop_hw_collect_20261025_173017.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_173021/model_in_loop_hw_collect_20261025_173021.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_173026/model_in_loop_hw_collect_20261025_173026.pkl`

### report/sim2f_bench/twin_vs_hw_corpus/20261027/report.md

- ✅ 存在 `data/debug`
- ❌ 缺失 `data/debug/model_in_loop_hw_collect/20261025`

### report/simulation/bench_twin/bench_twin_fidelity_20261021.md

- ✅ 存在 `data/debug/slm_model_in_loop/20261019_103208`

### report/simulation/sim_credibility/artifacts/direction_corpus_20261030.json

- ✅ 存在 `data/debug/model_in_loop_hw_collect`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123031/model_in_loop_hw_collect_20261025_123031.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123151/model_in_loop_hw_collect_20261025_123151.pkl`
- ✅ 存在 `data/debug/model_in_loop_hw_collect/20261025_123208/model_in_loop_hw_collect_20261025_123208.pkl`

### report/simulation/sim_credibility/sim_credibility_20261030.md

- ✅ 存在 `data/debug`
- ✅ 存在 `data/debug/model_in_loop_hw_collect`

### report/slm/bench/bench_calibration_20261001.md

- ✅ 存在 `data/debug`
- ❌ 缺失 `data/debug/bench_stability_20261001_164458/bench_stability_`
- ❌ 缺失 `data/debug/sim_calib_abba_20261001_163743/sim_calib_abba_`
- ❌ 缺失 `data/sim_calib/freeform_transfer_hw.npz`

### report/slm/bench_probe/report.md

- ❌ 缺失 `data/debug/model_in_loop_hw_sweep_20260930_171003/20260930_171003/model_in_loop_hw_sweep_20260930_171003_20260930_171003.json`
- ❌ 缺失 `data/model_in_loop_hw/bench_geometry.json`
- ❌ 缺失 `data/model_in_loop_hw/sweep_records.npz`

### report/slm/daily/daily_2026-09-15.md

- ✅ 存在 `data/calibration`
- ✅ 存在 `data/calibration/WFS_M01219666_MLA150M-5C_3_20260915_232849.ref`
- ✅ 存在 `data/calibration/defocus_radius_diagnostic.json`
- ✅ 存在 `data/calibration/defocus_shift_calib_v3.json`
- ❌ 缺失 `data/slm_configs/slm/22030102.json`
- ❌ 缺失 `data/zernike_correction/raw_scan_20260915_235549.json`
- ❌ 缺失 `data/zernike_correction/report_20260915_235549.json`
- ❌ 缺失 `data/zernike_response_matrix/zm_slm22030102_wfsM01219666_532nm_20260915_235207.h5`

### report/slm/daily/daily_2026-09-16.md

- ❌ 缺失 `data/slm_configs/slm/22030102.json`
- ❌ 缺失 `data/slm_corrections/run_n5_attempt3.log`
- ❌ 缺失 `data/slm_corrections/run_n5_attempt4.log;`
- ❌ 缺失 `data/slm_corrections/wfs_probe.log`
- ❌ 缺失 `data/slm_corrections/wfs_probe.log;`
- ❌ 缺失 `data/wfs_configs/wfs/M01219666.json`
- ❌ 缺失 `data/zernike_response_matrix_n5`

### report/slm/differentiable_opt/artifacts/20261022_091111/run_config.json

- ❌ 缺失 `data/calib.npz`

### report/slm/differentiable_opt/artifacts/20261022_132915/run_config.json

- ❌ 缺失 `data/calib.npz`
- ✅ 存在 `data/debug/differentiable_opt/20261022_213000`

### report/slm/differentiable_opt/artifacts/20261023_012505/run_config.json

- ❌ 缺失 `data/calib.npz`
- ✅ 存在 `data/debug/differentiable_opt/gs_20261023_092501`

### report/slm/differentiable_opt/differentiable_opt_animation_20261022.md

- ✅ 存在 `data/debug/differentiable_opt/20261022_213000`
- ✅ 存在 `data/debug/differentiable_opt/20261022_213000/animation`
- ✅ 存在 `data/debug/differentiable_opt/20261022_213000/animation/frames.csv`

### report/slm/differentiable_opt/differentiable_opt_report.md

- ✅ 存在 `data/debug/differentiable_opt/gs_20261023_092501`

### report/slm/differentiable_opt/focus_gate_verification_20261022.md

- ❌ 缺失 `data/debug/focus_gate_check/<UTC戳`

### report/slm/diffraction_efficiency/hw_blaze_20261023/final_numbers.json

- ✅ 存在 `data/diffraction_efficiency/gs_ceiling_20261023_151229`

### report/slm/diffraction_efficiency/hw_blaze_20261023/report.md

- ✅ 存在 `data/diffraction_efficiency/fine_newoptics_20261023_1415`
- ✅ 存在 `data/diffraction_efficiency/gifscan_20261023_1420`

### report/slm/focus_plane_report/focus_plane_20261019.md

- ❌ 缺失 `data/debug/bench_settings/focus_probe`

### report/slm/gs_control_1000px/report.md

- ✅ 存在 `data/debug/hw_gs_square_scans/20261030_gs1000`
- ❌ 缺失 `data/debug/hw_gs_square_scans/20261030_gs1000/blaze/metrics_00X.json`
- ✅ 存在 `data/debug/optical_path_probe/gs_control_1000px/20261030_031154Z`
- ✅ 存在 `data/debug/optical_path_probe/gs_control_1000px/20261030_031154Z/optical_path_probe.json`

### report/slm/model_in_loop_hw_runbook/model_in_loop_bench_calibration.md

- ❌ 缺失 `data/debug/slm_pib_shape_20260926_175334`
- ❌ 缺失 `data/model_in_loop_hw/calibration_records.npz`
- ❌ 缺失 `data/model_in_loop_hw/model_vs_measured.png`

### report/slm/report2.md

- ❌ 缺失 `data/zernike_correction/debug_<ts`

### report/slm/slm_shaping_diff/readme.md

- ❌ 缺失 `data/diff_shaping_circle`
- ❌ 缺失 `data/diff_shaping_hw`
- ❌ 缺失 `data/diff_shaping_spot`

### report/slm/slm_square_spgd/README.md

- ✅ 存在 `data/slm_square/20260910_203509`
- ✅ 存在 `data/slm_square/20260910_204311`

### report/slm/zernike_linearity/linearity.md

- ❌ 缺失 `data/zernike_correction/raw_scan_`

### report/slm/zernike_response_matrix_report/report3.md

- ❌ 缺失 `data/zernike_response_matrix/zm_recal_532_20260916.h5`
- ❌ 缺失 `data/zernike_response_matrix/zm_recal_532_20260916.json`

### report/slm_pib_bench/explore_delta/delta_scan.md

- ❌ 缺失 `data/debug/slm_pib_pearson`

### report/slm_pib_bench/explore_delta/delta_scan_large.md

- ❌ 缺失 `data/debug/slm_pib_pearson`

### report/slm_pib_bench/explore_delta/delta_scan_small.md

- ❌ 缺失 `data/debug/slm_pib_pearson`

### report/slm_pib_bench/pib_bench_report/EXPERIMENT_REPORT.md

- ❌ 缺失 `data/debug/slm_pib_`
- ❌ 缺失 `data/debug/slm_pib_hw_20260929`
- ❌ 缺失 `data/debug/slm_pib_online`
- ❌ 缺失 `data/debug/slm_pib_online/<ts`

### report/slm_pib_bench/pib_bench_report/report.md

- ✅ 存在 `data/debug`

### report/slm_pib_online/slm_pib_online_report/report.md

- ❌ 缺失 `data/debug/slm_pib_online`

### report/slm_zernike_shaping/slm_zernike_shaping_report/report.md

- ❌ 缺失 `data/debug/slm_zernike_shaping_rms_pib_20260926_162917/20260926_162917`
- ❌ 缺失 `data/debug/slm_zernike_shaping_rmse_out_20260926_163121/20260926_163121`

## 零报告引用的 data 顶层条目（潜在可清理候选）

- `ccd_configs`
- `dm_adj.txt`
- `dm_configs`
- `layouts`
- `micro_dm_configs`
- `slm_geometry_calib.npz`

> 仅“报告引用”维度；清理/移动前需 grep `src/`、`scripts/` 确认运行时依赖。
