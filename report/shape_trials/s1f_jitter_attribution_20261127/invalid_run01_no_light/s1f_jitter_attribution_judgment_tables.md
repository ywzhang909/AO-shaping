# S1f 随机抖动归因判读 (issue247-s1f-jitter-attribution-judgment/v1)

## 采集

| 项 | 值 |
|---|---|
| capture_dir | `data\experiments\attribution_20261123\s1\s1f_jitter_attribution_20261127\run_01` |
| generated_at_utc | 2026-11-27T05:18:23.158935+00:00 |
| 帧数合计 | 96 |
| 帧形状 | (1944, 2592) |
| 各段帧数 | A=24, B=24, flat_head=16, flat_mid=16, flat_tail=16 |

## 0 级参考（flat 组逐帧 max-peak 质心，逐轴中位）

| 参考 | 来源 | 帧数 | row (px) | col (px) |
|---|---|---|---|---|
| zr_head | flat_head | 16 | 4.462 | 460.013 |
| zr_mid | flat_mid | 16 | 4.459 | 460.012 |
| zr_tail | flat_tail | 16 | 4.467 | 459.992 |

## 抖动变体

| 变体 | 含义 | 参考 | 帧数 | rms_px | pairwise_max_px | std_row | std_col |
|---|---|---|---|---|---|---|---|
| A | 同写屏多帧（读取/提取地板） | zr_head | 24 | 977.828 | 2675.663 | 580.359 | 786.976 |
| B | 同字节重写（写屏随机性） | zr_mid | 24 | 1046.848 | 2675.702 | 626.774 | 838.479 |

## 漂移（flat 头 -> 尾）

| 量 | 值 |
|---|---|
| head_to_tail_px | 0.021 |
| dr (row) | +0.004 |
| dc (col) | -0.020 |
| drift_material (>= 1.0 px) | False |

## 归因

| 量 | 值 |
|---|---|
| readout_floor_rms | 977.828 px |
| write_screen_rms | 69.020 px |
| drift_px | 0.021 px |
| ratio = B.rms_px / A.rms_px | 1.071 |
| 分量占比 | 读取/提取 93.4% / 写屏 6.6% |
| 裁定 | **V1** 读取/提取主导 (readout/extraction dominant) |

## 交叉核对（仅佐证，非门禁）

| 量 | 值 |
|---|---|
| S1e run_05 intra_upload_dist | 1.68-1.85 px |
| A.rms_px | 977.828 px |
| B.rms_px | 1046.848 px |

> evidence only: compare with S1e run_05_postfix intra_upload_dist 1.68-1.85 px
