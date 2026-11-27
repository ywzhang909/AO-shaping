# S1f 随机抖动归因判读 (issue247-s1f-jitter-attribution-judgment/v1)

<!-- provenance:start -->
> **生成脚本**: [`scripts/slm/s1f_jitter_attribution_judge_20261127.py`](../../../scripts/slm/s1f_jitter_attribution_judge_20261127.py)
> **复现命令**: `python scripts/slm/s1f_jitter_attribution_judge_20261127.py --capture-dir data/experiments/attribution_20261123/s1/s1f_jitter_attribution_20261127/run_02 --out report/shape_trials/s1f_jitter_attribution_20261127`
> **运行环境**: 离线判读（只读 run_02 的 frames 与 capture_journal.jsonl）
> **说明**: run_02 三变体逐帧统计表（A/B/C 与 ratio/drift），按冻结 §3.4 口径生成；仅佐证，不改判据。
<!-- provenance:end -->

## 采集

| 项 | 值 |
|---|---|
| capture_dir | `data\experiments\attribution_20261123\s1\s1f_jitter_attribution_20261127\run_02` |
| generated_at_utc | 2026-11-27T05:33:01.734340+00:00 |
| 帧数合计 | 96 |
| 帧形状 | (1944, 2592) |
| 各段帧数 | A=24, B=24, flat_head=16, flat_mid=16, flat_tail=16 |

## 0 级参考（flat 组逐帧 max-peak 质心，逐轴中位）

| 参考 | 来源 | 帧数 | row (px) | col (px) |
|---|---|---|---|---|
| zr_head | flat_head | 16 | 1105.281 | 1228.530 |
| zr_mid | flat_mid | 16 | 1105.012 | 1229.353 |
| zr_tail | flat_tail | 16 | 1105.039 | 1228.452 |

## 抖动变体

| 变体 | 含义 | 参考 | 帧数 | rms_px | pairwise_max_px | std_row | std_col |
|---|---|---|---|---|---|---|---|
| A | 同写屏多帧（读取/提取地板） | zr_head | 24 | 0.820 | 2.254 | 0.555 | 0.603 |
| B | 同字节重写（写屏随机性） | zr_mid | 24 | 0.955 | 2.868 | 0.744 | 0.600 |

## 漂移（flat 头 -> 尾）

| 量 | 值 |
|---|---|
| head_to_tail_px | 0.254 |
| dr (row) | -0.242 |
| dc (col) | -0.078 |
| drift_material (>= 1.0 px) | False |

## 归因

| 量 | 值 |
|---|---|
| readout_floor_rms | 0.820 px |
| write_screen_rms | 0.136 px |
| drift_px | 0.254 px |
| ratio = B.rms_px / A.rms_px | 1.165 |
| 分量占比 | 读取/提取 85.8% / 写屏 14.2% |
| 裁定 | **V1** 读取/提取主导 (readout/extraction dominant) |

## 交叉核对（仅佐证，非门禁）

| 量 | 值 |
|---|---|
| S1e run_05 intra_upload_dist | 1.68-1.85 px |
| A.rms_px | 0.820 px |
| B.rms_px | 0.955 px |

> evidence only: compare with S1e run_05_postfix intra_upload_dist 1.68-1.85 px
