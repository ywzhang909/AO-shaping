# SPGD delta 扫描 (实机)

<!-- provenance:start -->
> **生成脚本**: [`scripts/explore_delta.py`](../../scripts/explore_delta.py)
> **复现命令**: `python scripts/explore_delta.py`
> **运行环境**: 硬件
> **说明**: SPGD delta 扫描 (实机)
<!-- provenance:end -->

- 目标 `pearson`, n_max=9, epochs=200, lr=0.5, 相机 `daheng`/0, 开窗 320px, 曝光 1.2ms
- 每个候选都带 `--debug`, recorder/sidecar/PNG 落在 `data/debug/slm_pib_pearson/`

判据: `dec` = 下降步占比 (0.5 为随机游走), `late` = 前 1/3 与后 1/3 均值之差。
**不使用 final-vs-first** —— 实机上它会被端点噪声骗到 (见 report/slm_pib_bench/EXPERIMENT_REPORT.md §3.2)。

| delta | verdict | dec | late % | span % | guard % | accepted | first | final |
|---|---|---|---|---|---|---|---|---|
| 0.0005 | random-walk | 0.52 | -1.4 | 22.3 | 0.0 | - | 0.5308 | 0.5308 |
| 0.001 | random-walk | 0.52 | +0.1 | 38.2 | 0.0 | - | 0.5293 | 0.5209 |
| 0.02 | not-converging | 0.56 | +8.5 | 29.8 | 0.0 | - | 0.5293 | 0.4019 |
| 0.05 | random-walk | 0.53 | +21.5 | 34.0 | 0.0 | - | 0.5301 | 0.4218 |
| 0.1 | not-converging | 0.43 | -16.7 | 46.0 | 31.4 | - | 0.5295 | 0.6027 |
| 0.2 | random-walk | 0.52 | +8.9 | 42.9 | 0.0 | - | 0.5287 | 0.3983 |
| 0.3 | random-walk | 0.51 | +1.7 | 41.4 | 0.7 | - | 0.5315 | 0.4513 |
| 0.5 | not-converging | 0.43 | -1.7 | 9.2 | 84.1 | - | 0.5292 | 0.5692 |

> **没有候选真正收敛。** 所有 delta 的 `dec` 都在 0.5 附近 = 随机游走。
> 这不是 delta 选错, 而是每步 SPGD 信噪比不足; 优先排查慢漂移对梯度估计的污染
> (把 `+ - - +` 回文采样接入主循环), 再回来扫 delta。
