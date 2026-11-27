# S1e arm P D5 同序位跨变体对比

<!-- provenance:start -->
> **生成脚本**: [`scripts/slm/s1e_arm_p_d5_judge_20261126.py`](../../../scripts/slm/s1e_arm_p_d5_judge_20261126.py)
> **复现命令**: `python scripts/slm/s1e_arm_p_d5_judge_20261126.py`
> **运行环境**: 离线 CPU 判读（读两个变体冻结 JSON, 不打开设备）
> **说明**: s1e arm-p vs d5 对比冻结表格 (缺省读 s1e_same_bytes_probe_20261126.json 与 s1e_arm_p_20261126.json)
<!-- provenance:end -->

- 判定：**no_power_cycle_effect**
- 冻结阈值：`FROZEN_CRITERIA['intra_upload_max_dist_px']` = 1.5 px
- 口径：dr/dc relative to each variant's own 0th (frozen §12.16.3)
- 同序位最大相对差：**1.389 px**（s1e_a1）
- 绝对坐标差最大：2.613 px（仅佐证）
- 轴型翻转：无；E4 触发：**否**

> 说明：
同序位主瓣相对各自 0th 的差最大 1.389 px（s1e_a1），未超冻结阈值 1.5 px；跨变体轴型全一致、无 col/row 翻转；绝对坐标差最大 2.613 px 仅作佐证（含各变体 0th 质心抖动）。

| pattern | 数组 | dr_A | dr_P | dc_A | dc_P | 相对差 px | 绝对差 px | 轴型 A | 轴型 P | 翻转 |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| `s1e_a1` | H | -1.634 | -2.418 | -93.587 | -94.734 | 1.389 | 2.362 | col-type | col-type | 否 |
| `s1e_a2` | H | -1.85 | -1.902 | -94.185 | -94.673 | 0.491 | 1.431 | col-type | col-type | 否 |
| `s1e_b1` | B | -0.842 | -1.989 | -59.197 | -58.884 | 1.189 | 2.487 | other | other | 否 |
| `s1e_a3` | H | -2.13 | -2.904 | -93.318 | -93.763 | 0.893 | 2.121 | col-type | col-type | 否 |
| `s1e_b2` | B | -1.092 | -2.33 | -58.56 | -59.186 | 1.387 | 2.613 | other | other | 否 |
| `s1e_a4` | H | -1.963 | -2.731 | -94.306 | -94.058 | 0.807 | 2.104 | col-type | col-type | 否 |
| `s1e_a5` | H | -3.503 | -3.466 | -95.142 | -94.416 | 0.727 | 1.499 | col-type | col-type | 否 |
| `s1e_b3` | B | -3.454 | -3.412 | -58.899 | -58.946 | 0.063 | 1.271 | other | other | 否 |
| `s1e_a6` | H | -2.401 | -2.624 | -95.115 | -93.81 | 1.324 | 2.056 | col-type | col-type | 否 |
