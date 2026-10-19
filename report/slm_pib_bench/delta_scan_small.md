# SPGD delta 扫描 (实验)

<!-- provenance:start -->
> **生成脚本**: [`scripts/slm/explore_delta.py`](../../scripts/slm/explore_delta.py)
> **复现命令**: `python scripts/slm/explore_delta.py`
> **运行环境**: 硬件
> **说明**: delta 小范围扫描 (实验)
<!-- provenance:end -->

**本轮结论**：小 delta 档（0.0005/0.001）两档均判为随机游走（dec=0.52），late 增益 −1.4%/+0.1% 均在噪声内。小 delta 下信号被噪声地板淹没（多模式 SNR≈1.3），与理论预期一致：delta 越小，单步差分信号 ∝δ 而噪声不变，SNR 线性恶化。结论与全扫描一致——**不是 delta 选错，而是每步 SPGD 信噪比不足**；优先排查慢漂移对梯度估计的污染，再谈参数选择。

- 目标 `pearson`, n_max=9, epochs=200, lr=0.5, 相机 `daheng`/0, 开窗 320px, 曝光 1.2ms
- 每个候选都带 `--debug`, recorder/sidecar/PNG 落在 `data/debug/slm_pib_pearson/`

判据: `dec` = 下降步占比 (0.5 为随机游走), `late` = 前 1/3 与后 1/3 均值之差。
**不使用 final-vs-first** —— 实验上它会被端点噪声骗到 (见 report/slm_pib_bench/EXPERIMENT_REPORT.md §3.2)。

| delta | verdict | dec | late % | span % | guard % | accepted | first | final |
|---|---|---|---|---|---|---|---|---|
| 0.0005 | random-walk | 0.52 | -1.4 | 22.3 | 0.0 | - | 0.5308 | 0.5308 |
| 0.001 | random-walk | 0.52 | +0.1 | 38.2 | 0.0 | - | 0.5293 | 0.5209 |

> **没有候选真正收敛。** 所有 delta 的 `dec` 都在 0.5 附近 = 随机游走。
> 这不是 delta 选错, 而是每步 SPGD 信噪比不足; 优先排查慢漂移对梯度估计的污染
> (把 `+ - - +` 回文采样接入主循环), 再回来扫 delta。
