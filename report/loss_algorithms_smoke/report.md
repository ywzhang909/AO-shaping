# Loss x algorithm comparison (offline sim bench)

<!-- provenance:start -->
> **生成脚本**: [`scripts/ml/compare_loss_algorithms.py`](../../scripts/ml/compare_loss_algorithms.py)
> **复现命令**: `python scripts/ml/compare_loss_algorithms.py --quick --out report/loss_algorithms_smoke`
> **运行环境**: 离线
> **说明**: 测试用例跑 --quick 生成的冒烟版本
<!-- provenance:end -->

**本轮结论**（loss × 算法冒烟基准）：本轮为 `--quick` 冒烟档（epochs=2、单种子），仅验证两条评测轨道的管线可执行性，**不构成性能结论**：Track A（ML 目标 × torch 优化器）mse/adam 与 physical/adam 的 val R² 分别为 +0.616/+0.585，差距在 2 epoch 下无判别力；Track B（AO 目标 × 搜索算法）SPGD-adamod 的 m_pib=0.765 显著优于 GA 的 0.031，但 n=1 且 GA 系数范数（7.01）远大于 SPGD（0.0001），提示 GA 在 2 epoch 预算内尚未进入有效搜索。正式对比需增加种子数与 epochs 后重跑。

- epochs per cell: 2
- paired seeds: 1 (same list for every cell)
- grid 32, n_max 4, cam_size 128

## How to read this

Rank on **R2** and the physical ROI terms, never on MSE/PSNR/SSIM:
`normalization="sum"` once reported PSNR 72 dB / SSIM 0.9996 while its
R2 was *worse* than a constant predictor. Single-seed deltas are inside
the noise, which is why seeds are paired and spread is reported.

Track A optimises a torch loss through the forward model. Track B is a
measurement-driven search and never sees that loss -- they are reported
separately on purpose.

## Track A -- ML objective x torch optimizer

| loss | optimizer | val R2 | val PSNR (dB) | best epoch | max abs coeff |
|---|---|---|---|---|---|
| mse | adam | +0.6163 | +18.4814 | 1 | +0.0747 |
| physical | adam | +0.5854 | +18.1678 | 0 | +0.0500 |

## Track B -- AO objective x search algorithm (sim bench)

| objective | algorithm | spgd optimizer | seed | final J | m_pib | m_shape | m_ee | m_rmse | coeff norm |
|---|---|---|---|---|---|---|---|---|---|
| pib | spgd | adamod | 0 | +2650.3972 | +0.7651 | -1.2868 | +1.0000 | +0.0005 | +0.0001 |
| pib | ga | - | 0 | +1552.2816 | +0.0307 | -1.0993 | +1.0000 | +0.0001 | +7.0067 |

### Track B aggregated over seeds (mean +/- spread)

| objective | algorithm | spgd optimizer | n | m_pib mean | m_shape mean | m_ee mean |
|---|---|---|---|---|---|---|
| pib | ga | - | 1 | 0.0307 +/-0.0000 | -1.0993 +/-0.0000 | 1.0000 +/-0.0000 |
| pib | spgd | adamod | 1 | 0.7651 +/-0.0000 | -1.2868 +/-0.0000 | 1.0000 +/-0.0000 |
